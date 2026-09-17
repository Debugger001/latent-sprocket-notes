"""Hugging Face runtime for the answer-only GRPO baselines.

The two baselines deliberately share one runtime.  They differ only in the
fixed credit prepared by :mod:`answer_rl` and in the corresponding pure loss:

* GRPO assigns one group-normalized sequence advantage to every completion
  token and uses token-level importance ratios.
* Rank-GRPO assigns independently normalized rank credit to JSON-list item
  actions and uses each item's geometric-mean importance ratio.

Exactly four completions are sampled per query.  Unlike MaskPO, there are no
counterfactual generations.  Gradient accumulation is query-balanced: each
query-level loss has equal weight even when completions have different token
or item counts.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .answer_rl import (
    AnswerRLAlgorithm,
    AnswerRLGroupCredit,
    prepare_answer_rl_group,
)
from .losses import AnswerRLLossOutput, rank_grpo_loss, sequence_grpo_loss
from .trainer import (
    GeneratedCompletion,
    HuggingFacePolicyBackend,
    LogProbBatch,
    PolicyBackend,
    SamplingConfig,
    TrainingExample,
    _load_effective_peft_config,
    _resolve_dtype,
    _torch,
)


Tensor = Any
RankMetric = Literal["ndcg", "dcg"]


@dataclass(frozen=True)
class AnswerRLPassMetrics:
    """Equal-query diagnostics for one completed optimizer pass."""

    ppo_pass: int
    loss: float
    policy_loss: float
    kl: float
    clip_fraction: float
    completion_count: int
    action_count: int
    active_token_count: int
    query_count: int


@dataclass(frozen=True)
class AnswerTrainStepResult:
    """Outputs and diagnostics for one answer-only query group."""

    example_id: str
    algorithm: AnswerRLAlgorithm
    credit: AnswerRLGroupCredit
    originals: tuple[str, ...]
    loss: float
    policy_loss: float
    kl: float
    clip_fraction: float
    completion_count: int
    action_count: int
    active_token_count: int
    optimizer_stepped: bool
    optimizer_step: int
    optimizer_steps_applied: int
    rollout_step_completed: bool
    rollout_step: int
    ppo_pass_metrics: tuple[AnswerRLPassMetrics, ...]


class AnswerRLTrainer:
    """Train either answer-only GRPO or answer-only Rank-GRPO.

    ``gradient_accumulation_steps`` counts *queries*.  Each query first forms
    its complete group of four sibling completions and its own BNPO mean.  The
    accumulated gradients are divided by the actual number of queries only
    when the optimizer step is taken, including for a partial final flush.
    This prevents longer JSON lists from silently receiving more query weight.
    """

    group_size = 4

    def __init__(
        self,
        *,
        actor: PolicyBackend,
        reference: PolicyBackend,
        optimizer: Any,
        algorithm: AnswerRLAlgorithm,
        sampling_config: SamplingConfig = SamplingConfig(),
        rank_metric: RankMetric = "ndcg",
        clip_low: float | None = None,
        clip_high: float | None = None,
        reference_kl_coefficient: float = 0.001,
        normalization_epsilon: float = 1e-4,
        normalization_ddof: int = 1,
        underflow_penalty: float = -0.1,
        overflow_penalty: float = -0.1,
        gradient_accumulation_steps: int = 1,
        ppo_passes: int = 1,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if algorithm not in {"grpo", "rank_grpo"}:
            raise ValueError("algorithm must be 'grpo' or 'rank_grpo'")
        if rank_metric not in {"ndcg", "dcg"}:
            raise ValueError("rank_metric must be 'ndcg' or 'dcg'")
        if type(ppo_passes) is not int or ppo_passes != 1:
            raise ValueError("answer-only baselines require ppo_passes=1")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if reference_kl_coefficient < 0:
            raise ValueError("reference_kl_coefficient must be non-negative")
        if normalization_epsilon < 0:
            raise ValueError("normalization_epsilon must be non-negative")
        if type(normalization_ddof) is not int or normalization_ddof < 0:
            raise ValueError("normalization_ddof must be a non-negative integer")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")

        default_low, default_high = (
            (0.2, 0.26) if algorithm == "grpo" else (0.06, 0.08)
        )
        self.clip_low = default_low if clip_low is None else clip_low
        self.clip_high = default_high if clip_high is None else clip_high
        if not 0 <= self.clip_low < 1:
            raise ValueError("clip_low must be in [0, 1)")
        if not 0 <= self.clip_high < 1:
            raise ValueError("clip_high must be in [0, 1)")

        self.actor = actor
        self.reference = reference
        self.optimizer = optimizer
        self.algorithm = algorithm
        self.sampling_config = sampling_config
        self.rank_metric = rank_metric
        self.reference_kl_coefficient = reference_kl_coefficient
        self.normalization_epsilon = normalization_epsilon
        self.normalization_ddof = normalization_ddof
        self.underflow_penalty = underflow_penalty
        self.overflow_penalty = overflow_penalty
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ppo_passes = ppo_passes
        self.max_grad_norm = max_grad_norm

        self._pending_query_metrics: list[AnswerRLPassMetrics] = []
        self._optimizer_steps = 0
        self._rollout_steps = 0
        self._last_ppo_pass_metrics: tuple[AnswerRLPassMetrics, ...] = ()
        self._poisoned = False

    @property
    def optimizer_steps(self) -> int:
        return self._optimizer_steps

    @property
    def rollout_steps(self) -> int:
        return self._rollout_steps

    @property
    def last_ppo_pass_metrics(self) -> tuple[AnswerRLPassMetrics, ...]:
        return self._last_ppo_pass_metrics

    @property
    def pending_micro_steps(self) -> int:
        return len(self._pending_query_metrics)

    @property
    def pending_active_tokens(self) -> int:
        return sum(item.active_token_count for item in self._pending_query_metrics)

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _ensure_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "AnswerRLTrainer is poisoned after a failed backward or optimizer "
                "pass; restart from the last completed-rollout checkpoint"
            )

    def _poison(self) -> None:
        self._poisoned = True
        try:
            self.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        self._pending_query_metrics.clear()

    def _require_clean_rollout_boundary(self) -> None:
        self._ensure_healthy()
        if self._pending_query_metrics:
            raise RuntimeError(
                "training state is only available at a clean rollout boundary"
            )
        if self._optimizer_steps != self._rollout_steps:
            raise RuntimeError(
                "trainer counters are inconsistent: optimizer_steps must equal "
                "rollout_steps when ppo_passes=1"
            )

    def training_state_dict(self) -> dict[str, int]:
        """Return counters compatible with the MaskPO checkpoint schema."""

        self._require_clean_rollout_boundary()
        return {
            "version": 1,
            "rollout_steps": self._rollout_steps,
            "optimizer_steps": self._optimizer_steps,
            "ppo_passes": self.ppo_passes,
        }

    def load_training_state_dict(self, state: Mapping[str, object]) -> None:
        self._require_clean_rollout_boundary()
        if not isinstance(state, Mapping):
            raise TypeError("training state must be a mapping")
        expected = {"version", "rollout_steps", "optimizer_steps", "ppo_passes"}
        if set(state) != expected:
            raise ValueError(
                "training state must contain exactly version, rollout_steps, "
                "optimizer_steps, and ppo_passes"
            )

        def nonnegative_int(key: str) -> int:
            value = state[key]
            if type(value) is not int or value < 0:
                raise ValueError(f"training state {key} must be a non-negative integer")
            return value

        version = nonnegative_int("version")
        rollout_steps = nonnegative_int("rollout_steps")
        optimizer_steps = nonnegative_int("optimizer_steps")
        state_ppo_passes = nonnegative_int("ppo_passes")
        if version != 1:
            raise ValueError(f"unsupported training state version: {version}")
        if state_ppo_passes != 1:
            raise ValueError("answer-only training state must have ppo_passes=1")
        if optimizer_steps != rollout_steps:
            raise ValueError(
                "training state optimizer_steps must equal rollout_steps when "
                "ppo_passes=1"
            )
        self._rollout_steps = rollout_steps
        self._optimizer_steps = optimizer_steps
        self._last_ppo_pass_metrics = ()

    def _generate_batched(
        self, prefixes: Sequence[str]
    ) -> tuple[GeneratedCompletion, ...]:
        generated: list[GeneratedCompletion] = []
        batch_size = self.sampling_config.original_batch_size
        for start in range(0, len(prefixes), batch_size):
            batch = prefixes[start : start + batch_size]
            result = tuple(self.actor.generate(batch, self.sampling_config))
            if len(result) != len(batch):
                raise RuntimeError(
                    "policy backend returned a different number of generations "
                    "than prefixes"
                )
            generated.extend(result)
        return tuple(generated)

    def _sample_group(
        self, example: TrainingExample
    ) -> tuple[GeneratedCompletion, ...]:
        model_prompt = self.actor.render_user_prompt(example.prompt)
        originals = self._generate_batched([model_prompt] * self.group_size)
        if len(originals) != self.group_size:
            raise RuntimeError("answer-only training requires exactly four originals")
        return originals

    @staticmethod
    def _validate_logprob_batches(
        current: LogProbBatch,
        old: LogProbBatch,
        reference: LogProbBatch,
    ) -> None:
        torch = _torch()
        expected = tuple(current.logps.shape)
        if tuple(old.logps.shape) != expected or tuple(reference.logps.shape) != expected:
            raise ValueError("current, old, and reference log probabilities must align")
        for name, batch in (
            ("current", current),
            ("old", old),
            ("reference", reference),
        ):
            if tuple(batch.completion_mask.shape) != expected:
                raise ValueError(f"{name} completion mask has the wrong shape")
        current_mask = current.completion_mask.detach().to("cpu").bool()
        for name, batch in (("old", old), ("reference", reference)):
            if not torch.equal(current_mask, batch.completion_mask.detach().to("cpu").bool()):
                raise ValueError(f"{name} completion mask differs from current mask")

    @staticmethod
    def _padded_credit_tensor(
        rows: Sequence[Sequence[float]], like: Tensor
    ) -> Tensor:
        torch = _torch()
        if like.ndim != 2 or len(rows) != like.shape[0]:
            raise ValueError("credit rows must align with a [batch, tokens] tensor")
        values = torch.zeros_like(like)
        for row_index, row in enumerate(rows):
            if len(row) > like.shape[1]:
                raise ValueError("credit row is wider than the sampled-token batch")
            if row:
                values[row_index, : len(row)] = torch.as_tensor(
                    row, dtype=like.dtype, device=like.device
                )
        return values

    @staticmethod
    def _padded_item_ids(
        rows: Sequence[Sequence[int]], like: Tensor
    ) -> Tensor:
        torch = _torch()
        if like.ndim != 2 or len(rows) != like.shape[0]:
            raise ValueError("item-id rows must align with a [batch, tokens] tensor")
        values = torch.full_like(like, -1, dtype=torch.long)
        for row_index, row in enumerate(rows):
            if len(row) > like.shape[1]:
                raise ValueError("item-id row is wider than the sampled-token batch")
            if row:
                values[row_index, : len(row)] = torch.as_tensor(
                    row, dtype=torch.long, device=like.device
                )
        return values

    @staticmethod
    def _padded_loss_mask(
        rows: Sequence[Sequence[bool]], like: Tensor
    ) -> Tensor:
        torch = _torch()
        if like.ndim != 2 or len(rows) != like.shape[0]:
            raise ValueError("loss-mask rows must align with a [batch, tokens] tensor")
        values = torch.zeros_like(like, dtype=torch.bool)
        for row_index, row in enumerate(rows):
            if len(row) > like.shape[1]:
                raise ValueError("loss-mask row is wider than the sampled-token batch")
            if row:
                values[row_index, : len(row)] = torch.as_tensor(
                    row, dtype=torch.bool, device=like.device
                )
        return values

    def _compute_loss(
        self,
        credit: AnswerRLGroupCredit,
        current: LogProbBatch,
        old: LogProbBatch,
        reference: LogProbBatch,
    ) -> AnswerRLLossOutput:
        if self.algorithm == "grpo":
            sequence_advantages = [
                item.sequence_advantage for item in credit.completions
            ]
            return sequence_grpo_loss(
                current.logps,
                old.logps,
                reference.logps,
                sequence_advantages,
                current.completion_mask,
                clip_epsilon_low=self.clip_low,
                clip_epsilon_high=self.clip_high,
                beta=self.reference_kl_coefficient,
            )

        token_advantages = self._padded_credit_tensor(
            [item.token_advantages for item in credit.completions], current.logps
        )
        item_ids = self._padded_item_ids(
            [item.item_ids for item in credit.completions], current.logps
        )
        semantic_mask = self._padded_loss_mask(
            [item.loss_mask for item in credit.completions], current.logps
        )
        semantic_mask &= current.completion_mask.to(
            device=semantic_mask.device, dtype=_torch().bool
        )
        return rank_grpo_loss(
            current.logps,
            old.logps,
            reference.logps,
            token_advantages,
            item_ids,
            semantic_mask,
            expected_ranks=credit.slate_k,
            clip_epsilon_low=self.clip_low,
            clip_epsilon_high=self.clip_high,
            beta=self.reference_kl_coefficient,
        )

    @staticmethod
    def _validate_finite_loss_output(loss_output: AnswerRLLossOutput) -> None:
        torch = _torch()
        for name in ("loss", "policy_loss", "kl", "clip_fraction"):
            value = torch.as_tensor(getattr(loss_output, name)).detach()
            if value.numel() != 1:
                raise ValueError(f"answer-only {name} diagnostic must be scalar")
            if not bool(torch.isfinite(value).to(device="cpu").item()):
                raise FloatingPointError(f"non-finite answer-only {name} diagnostic")

    @staticmethod
    def _scalar(value: Tensor) -> float:
        torch = _torch()
        return float(value.detach().to(dtype=torch.float32, device="cpu").item())

    def _query_metrics(
        self, loss_output: AnswerRLLossOutput
    ) -> AnswerRLPassMetrics:
        return AnswerRLPassMetrics(
            ppo_pass=1,
            loss=self._scalar(loss_output.loss),
            policy_loss=self._scalar(loss_output.policy_loss),
            kl=self._scalar(loss_output.kl),
            clip_fraction=self._scalar(loss_output.clip_fraction),
            completion_count=self.group_size,
            action_count=int(loss_output.action_count.detach().to("cpu").item()),
            active_token_count=int(loss_output.token_count.detach().to("cpu").item()),
            query_count=1,
        )

    @staticmethod
    def _aggregate_query_metrics(
        metrics: Sequence[AnswerRLPassMetrics],
    ) -> AnswerRLPassMetrics:
        if not metrics:
            raise RuntimeError("cannot aggregate an empty optimizer pass")
        count = len(metrics)

        def mean(name: str) -> float:
            return sum(float(getattr(item, name)) for item in metrics) / count

        return AnswerRLPassMetrics(
            ppo_pass=1,
            loss=mean("loss"),
            policy_loss=mean("policy_loss"),
            kl=mean("kl"),
            clip_fraction=mean("clip_fraction"),
            completion_count=sum(item.completion_count for item in metrics),
            action_count=sum(item.action_count for item in metrics),
            active_token_count=sum(item.active_token_count for item in metrics),
            query_count=count,
        )

    def _take_optimizer_step(self, query_count: int) -> None:
        torch = _torch()
        parameters = tuple(self.actor.trainable_parameters())
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(query_count)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                parameters, self.max_grad_norm, error_if_nonfinite=True
            )
        else:
            for parameter in parameters:
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all().to(device="cpu").item()
                ):
                    raise FloatingPointError(
                        "non-finite trainable gradient before optimizer step"
                    )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._optimizer_steps += 1

    def _finish_rollout_step(self) -> tuple[AnswerRLPassMetrics, ...]:
        if not self._pending_query_metrics:
            raise RuntimeError("cannot optimize an empty rollout batch")
        metrics = self._aggregate_query_metrics(self._pending_query_metrics)
        try:
            self._take_optimizer_step(len(self._pending_query_metrics))
        except BaseException:
            self._poison()
            raise
        self._pending_query_metrics.clear()
        self._rollout_steps += 1
        self._last_ppo_pass_metrics = (metrics,)
        return self._last_ppo_pass_metrics

    def flush_gradients(self) -> bool:
        self._ensure_healthy()
        if not self._pending_query_metrics:
            return False
        self._finish_rollout_step()
        return True

    def train_query(self, example: TrainingExample) -> AnswerTrainStepResult:
        """Sample one four-way group, score it, and accumulate one query mean."""

        self._ensure_healthy()
        originals = self._sample_group(example)
        original_texts = tuple(sample.text for sample in originals)
        credit = prepare_answer_rl_group(
            original_texts,
            [sample.token_offsets for sample in originals],
            positives=example.positives,
            slate_k=example.slate_k,
            algorithm=self.algorithm,
            expected_group_size=self.group_size,
            metric=self.rank_metric,
            eps=self.normalization_epsilon,
            normalization_ddof=self.normalization_ddof,
            underflow_penalty=self.underflow_penalty,
            overflow_penalty=self.overflow_penalty,
        )

        old = self.actor.token_logps(originals, requires_grad=False)
        reference = self.reference.token_logps(originals, requires_grad=False)
        current = self.actor.token_logps(originals, requires_grad=True)
        self._validate_logprob_batches(current, old, reference)

        try:
            loss_output = self._compute_loss(credit, current, old, reference)
            self._validate_finite_loss_output(loss_output)
            if not loss_output.loss.requires_grad:
                raise RuntimeError("actor log probabilities do not carry gradients")
            query_metrics = self._query_metrics(loss_output)
            loss_output.loss.backward()
            self._pending_query_metrics.append(query_metrics)
            ppo_metrics = ()
            if len(self._pending_query_metrics) >= self.gradient_accumulation_steps:
                ppo_metrics = self._finish_rollout_step()
        except BaseException:
            self._poison()
            raise

        rollout_completed = bool(ppo_metrics)
        return AnswerTrainStepResult(
            example_id=example.example_id,
            algorithm=self.algorithm,
            credit=credit,
            originals=original_texts,
            loss=query_metrics.loss,
            policy_loss=query_metrics.policy_loss,
            kl=query_metrics.kl,
            clip_fraction=query_metrics.clip_fraction,
            completion_count=query_metrics.completion_count,
            action_count=query_metrics.action_count,
            active_token_count=query_metrics.active_token_count,
            optimizer_stepped=rollout_completed,
            optimizer_step=self._optimizer_steps,
            optimizer_steps_applied=int(rollout_completed),
            rollout_step_completed=rollout_completed,
            rollout_step=self._rollout_steps,
            ppo_pass_metrics=ppo_metrics,
        )


def load_huggingface_answer_rl_trainer(
    *,
    model_name_or_path: str,
    adapter_path: str,
    algorithm: AnswerRLAlgorithm,
    actor_adapter_path: str | None = None,
    revision: str | None = None,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    actor_device_map: Any = "auto",
    reference_device_map: Any | None = None,
    gradient_checkpointing: bool = True,
    sampling_config: SamplingConfig = SamplingConfig(),
    rank_metric: RankMetric = "ndcg",
    clip_low: float | None = None,
    clip_high: float | None = None,
    beta: float = 0.001,
    normalization_epsilon: float = 1e-4,
    normalization_ddof: int = 1,
    underflow_penalty: float = -0.1,
    overflow_penalty: float = -0.1,
    gradient_accumulation_steps: int = 1,
    ppo_passes: int = 1,
    max_grad_norm: float | None = 1.0,
) -> AnswerRLTrainer:
    """Load trainable and frozen copies of an answer-only SFT adapter.

    ``adapter_path`` always initializes the frozen reference.  A resumed RL
    checkpoint may be supplied through ``actor_adapter_path`` without changing
    that reference.  Actor and reference device maps are independent so the
    frozen model can live on a separate GPU class.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    try:
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional training install
        raise ImportError(
            "Hugging Face answer-only RL training requires transformers, peft, "
            "and accelerate; install the project's training dependencies"
        ) from exc

    torch = _torch()
    torch_dtype = _resolve_dtype(dtype)
    common_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if revision is not None:
        common_kwargs["revision"] = revision

    effective_actor_adapter = actor_adapter_path or adapter_path
    tokenizer = AutoTokenizer.from_pretrained(
        effective_actor_adapter, trust_remote_code=trust_remote_code
    )
    actor_peft_config = _load_effective_peft_config(
        effective_actor_adapter, PeftConfig
    )
    reference_peft_config = (
        copy.deepcopy(actor_peft_config)
        if effective_actor_adapter == adapter_path
        else _load_effective_peft_config(adapter_path, PeftConfig)
    )

    actor_kwargs = dict(common_kwargs)
    if actor_device_map is not None:
        actor_kwargs["device_map"] = actor_device_map
    actor_base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **actor_kwargs
    )
    actor_model = PeftModel.from_pretrained(
        actor_base,
        effective_actor_adapter,
        config=copy.deepcopy(actor_peft_config),
        is_trainable=True,
    )
    if gradient_checkpointing:
        actor_model.gradient_checkpointing_enable()
        if hasattr(actor_model, "enable_input_require_grads"):
            actor_model.enable_input_require_grads()
    if hasattr(actor_model, "config"):
        actor_model.config.use_cache = False

    reference_kwargs = dict(common_kwargs)
    if reference_device_map is not None:
        reference_kwargs["device_map"] = reference_device_map
    elif actor_device_map is not None:
        reference_kwargs["device_map"] = actor_device_map
    reference_base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, **reference_kwargs
    )
    reference_model = PeftModel.from_pretrained(
        reference_base,
        adapter_path,
        config=copy.deepcopy(reference_peft_config),
        is_trainable=False,
    )
    reference_model.requires_grad_(False)
    reference_model.eval()
    if hasattr(reference_model, "config"):
        reference_model.config.use_cache = False

    # Qwen3's default assistant prefix permits a ``<think>`` trace.  The
    # archived answer-only task instead trains a bare JSON-style list, so use
    # the tokenizer's explicit non-thinking prefix.  Without it the base
    # model can spend the entire 2,048-token budget reasoning and never emit
    # a gradeable list.
    actor = HuggingFacePolicyBackend(
        actor_model,
        tokenizer,
        enable_thinking=False,
    )
    reference = HuggingFacePolicyBackend(
        reference_model,
        tokenizer,
        enable_thinking=False,
    )
    trainable = tuple(actor.trainable_parameters())
    if not trainable:
        raise RuntimeError("the actor adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    optimizer.zero_grad(set_to_none=True)
    return AnswerRLTrainer(
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        algorithm=algorithm,
        sampling_config=sampling_config,
        rank_metric=rank_metric,
        clip_low=clip_low,
        clip_high=clip_high,
        reference_kl_coefficient=beta,
        normalization_epsilon=normalization_epsilon,
        normalization_ddof=normalization_ddof,
        underflow_penalty=underflow_penalty,
        overflow_penalty=overflow_penalty,
        gradient_accumulation_steps=gradient_accumulation_steps,
        ppo_passes=ppo_passes,
        max_grad_norm=max_grad_norm,
    )
