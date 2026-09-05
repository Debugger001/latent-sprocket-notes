"""Single-machine Hugging Face runtime for the latest MaskPO algorithm.

The pure credit-assignment implementation lives in :mod:`pipeline` and
:mod:`routing`.  This module is deliberately a thin runtime around it:

1. sample four original completions from the unchanged actor;
2. construct and sample up to sixteen scoring-only counterfactual probes;
3. score and route advantages;
4. compute old-policy and frozen-initial-SFT reference log probabilities; and
5. update the actor with tokenwise clipped BNPO on originals only.

Heavy dependencies are imported only by the Hugging Face loader or tensor
operations.  Importing the evaluation package therefore does not require
Transformers, PEFT, Accelerate, or even PyTorch.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .advantages import MaskPOConfig
from .losses import BNPOLossOutput, tokenwise_bnpo_loss
from .masking import parse_counterfactual
from .pipeline import QueryCredit, plan_counterfactual_probes, score_maskpo_group
from .routing import RoutingResult, route_token_advantages

if TYPE_CHECKING:
    import torch

    Tensor = torch.Tensor
else:
    Tensor = Any


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "MaskPO training requires PyTorch; install the training dependencies"
        ) from exc
    return torch


@dataclass(frozen=True)
class TrainingExample:
    """One fixed-slate query presented to a MaskPO training step."""

    prompt: str
    positives: frozenset[int]
    slate_k: int
    example_id: str = ""

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("prompt must not be empty")
        if self.slate_k < 1:
            raise ValueError("slate_k must be positive")
        if any(type(item) is not int for item in self.positives):
            raise TypeError("positives must contain integer candidate indices")
        if any(item < 1 or item > self.slate_k for item in self.positives):
            raise ValueError("positives must be candidate indices in 1..slate_k")


@dataclass(frozen=True)
class SamplingConfig:
    """Generation settings shared by original and probe sampling."""

    do_sample: bool = True
    temperature: float = 0.6
    top_k: int = 20
    top_p: float = 0.95
    max_new_tokens: int = 2048
    counterfactual_max_new_tokens: int | None = None
    original_batch_size: int = 4
    counterfactual_batch_size: int = 4

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if (
            self.counterfactual_max_new_tokens is not None
            and self.counterfactual_max_new_tokens <= 0
        ):
            raise ValueError("counterfactual_max_new_tokens must be positive")
        if self.original_batch_size <= 0 or self.counterfactual_batch_size <= 0:
            raise ValueError("generation batch sizes must be positive")

    def for_counterfactual(self) -> "SamplingConfig":
        """Return the suffix-generation variant without changing sampling law."""

        return SamplingConfig(
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=(
                self.counterfactual_max_new_tokens or self.max_new_tokens
            ),
            counterfactual_max_new_tokens=self.counterfactual_max_new_tokens,
            original_batch_size=self.original_batch_size,
            counterfactual_batch_size=self.counterfactual_batch_size,
        )

    def generation_kwargs(self) -> dict[str, object]:
        values: dict[str, object] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": True,
        }
        if self.do_sample:
            values.update(
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        return values


@dataclass(frozen=True)
class GeneratedCompletion:
    """A decoded suffix together with the exact sampled token IDs.

    ``prompt_token_ids`` represent the model-ready prefix used during
    generation (chat control tokens included).  ``token_offsets`` are
    half-open character offsets into ``text`` for each completion token.
    Special stop tokens have the zero-width offset ``(0, 0)`` and retain
    sequence credit during routing.
    """

    text: str
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    token_offsets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids must not be empty")
        if len(self.token_ids) != len(self.token_offsets):
            raise ValueError("token_ids and token_offsets must have equal length")


@dataclass(frozen=True)
class LogProbBatch:
    """Padded sampled-token log probabilities and their active-token mask."""

    logps: Tensor
    completion_mask: Tensor


class PolicyBackend(Protocol):
    """Small injectable boundary used by the trainer and offline tests."""

    def render_user_prompt(self, prompt: str) -> str:
        """Render user content into the model's assistant-generation prefix."""

    def generate(
        self,
        model_prefixes: Sequence[str],
        config: SamplingConfig,
    ) -> Sequence[GeneratedCompletion]:
        """Generate only the suffix following each supplied model prefix."""

    def token_logps(
        self,
        samples: Sequence[GeneratedCompletion],
        *,
        requires_grad: bool,
    ) -> LogProbBatch:
        """Return log probabilities for sampled completion tokens only."""

    def trainable_parameters(self) -> Iterable[Tensor]:
        """Return actor parameters eligible for optimization."""


@dataclass(frozen=True)
class TrainStepResult:
    """Outputs and diagnostics for one query-level MaskPO micro-step."""

    example_id: str
    query_credit: QueryCredit
    routes: tuple[RoutingResult, ...]
    originals: tuple[str, ...]
    counterfactuals: tuple[tuple[str | None, ...], ...]
    loss: float
    policy_loss: float
    kl: float
    clip_fraction: float
    active_token_count: int
    optimizer_stepped: bool
    optimizer_step: int

    @property
    def routing_fallback_count(self) -> int:
        return sum(route.used_sequence_fallback for route in self.routes)


class MaskPOTrainer:
    """Coordinate frozen-policy probing and an originals-only BNPO update."""

    def __init__(
        self,
        *,
        actor: PolicyBackend,
        reference: PolicyBackend,
        optimizer: Any,
        maskpo_config: MaskPOConfig = MaskPOConfig(),
        sampling_config: SamplingConfig = SamplingConfig(),
        ppo_clip: float = 0.2,
        reference_kl_coefficient: float = 0.001,
        normalization_epsilon: float = 1e-8,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        if maskpo_config.num_siblings != 4:
            raise ValueError("the configured MaskPO run requires exactly four sibling rollouts")
        if maskpo_config.num_rubrics != 4:
            raise ValueError("latest MaskPO requires exactly four rubrics")
        if maskpo_config.metric != "ndcg":
            raise ValueError("latest MIND MaskPO uses lenient nDCG")
        if not 0 <= ppo_clip < 1:
            raise ValueError("ppo_clip must be in [0, 1)")
        if reference_kl_coefficient < 0:
            raise ValueError("reference_kl_coefficient must be non-negative")
        if normalization_epsilon < 0:
            raise ValueError("normalization_epsilon must be non-negative")
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")

        self.actor = actor
        self.reference = reference
        self.optimizer = optimizer
        self.maskpo_config = maskpo_config
        self.sampling_config = sampling_config
        self.ppo_clip = ppo_clip
        self.reference_kl_coefficient = reference_kl_coefficient
        self.normalization_epsilon = normalization_epsilon
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self._pending_micro_steps = 0
        self._pending_active_tokens = 0
        self._optimizer_steps = 0

    @property
    def optimizer_steps(self) -> int:
        return self._optimizer_steps

    @property
    def pending_micro_steps(self) -> int:
        return self._pending_micro_steps

    @property
    def pending_active_tokens(self) -> int:
        return self._pending_active_tokens

    def _generate_batched(
        self,
        prefixes: Sequence[str],
        *,
        config: SamplingConfig,
        batch_size: int,
    ) -> tuple[GeneratedCompletion, ...]:
        generated: list[GeneratedCompletion] = []
        for start in range(0, len(prefixes), batch_size):
            batch = prefixes[start : start + batch_size]
            result = tuple(self.actor.generate(batch, config))
            if len(result) != len(batch):
                raise RuntimeError(
                    "policy backend returned a different number of generations "
                    "than prefixes"
                )
            generated.extend(result)
        return tuple(generated)

    def _sample_frozen_group(
        self,
        example: TrainingExample,
    ) -> tuple[
        tuple[GeneratedCompletion, ...],
        tuple[tuple[str | None, ...], ...],
    ]:
        """Sample originals and all probes before any optimization work."""

        model_prompt = self.actor.render_user_prompt(example.prompt)
        original_prefixes = [model_prompt] * self.maskpo_config.num_siblings
        originals = self._generate_batched(
            original_prefixes,
            config=self.sampling_config,
            batch_size=self.sampling_config.original_batch_size,
        )

        requests = plan_counterfactual_probes([sample.text for sample in originals])
        counterfactuals: list[list[str | None]] = [
            [None] * self.maskpo_config.num_rubrics
            for _ in range(self.maskpo_config.num_siblings)
        ]
        if not requests:
            return originals, tuple(tuple(row) for row in counterfactuals)

        probe_model_prefixes = [
            model_prompt + request.prefix.text for request in requests
        ]
        probe_config = self.sampling_config.for_counterfactual()
        generated_suffixes = self._generate_batched(
            probe_model_prefixes,
            config=probe_config,
            batch_size=self.sampling_config.counterfactual_batch_size,
        )
        for request, generated in zip(requests, generated_suffixes, strict=True):
            parsed = parse_counterfactual(request.prefix, generated.text)
            counterfactuals[request.rollout_index][request.rubric_index] = (
                parsed.completion
            )
        return originals, tuple(tuple(row) for row in counterfactuals)

    def _route_group(
        self,
        originals: Sequence[GeneratedCompletion],
        credit: QueryCredit,
    ) -> tuple[RoutingResult, ...]:
        routes: list[RoutingResult] = []
        for sample, rollout in zip(originals, credit.rollouts, strict=True):
            route = route_token_advantages(
                sample.text,
                sample.token_offsets,
                sequence_advantage=rollout.sequence_advantage,
                rubric_advantages=rollout.rubric_advantages,
                rank_advantages=rollout.rank_advantages,
                mask_advantages_by_item=rollout.mask_advantages_by_item,
                config=self.maskpo_config,
            )
            if len(route.advantages) != len(sample.token_ids):
                raise RuntimeError("routed advantages do not align to sampled tokens")
            routes.append(route)
        return tuple(routes)

    @staticmethod
    def _advantage_tensor(
        routes: Sequence[RoutingResult],
        like: Tensor,
    ) -> Tensor:
        torch = _torch()
        if like.ndim != 2:
            raise ValueError("log probabilities must have shape [batch, tokens]")
        if len(routes) != like.shape[0]:
            raise ValueError("route count must match log-probability batch size")
        values = torch.zeros_like(like)
        for row_index, route in enumerate(routes):
            width = len(route.advantages)
            if width > like.shape[1]:
                raise ValueError("route is wider than the log-probability batch")
            if width:
                values[row_index, :width] = torch.as_tensor(
                    route.advantages,
                    dtype=like.dtype,
                    device=like.device,
                )
        return values

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
        if tuple(current.completion_mask.shape) != expected:
            raise ValueError("current completion mask does not align with log probabilities")
        for name, batch in (("old", old), ("reference", reference)):
            if tuple(batch.completion_mask.shape) != expected:
                raise ValueError(f"{name} completion mask has the wrong shape")
            if not torch.equal(
                current.completion_mask.detach().to("cpu").bool(),
                batch.completion_mask.detach().to("cpu").bool(),
            ):
                raise ValueError(f"{name} completion mask differs from current mask")

    def _apply_gradients_if_ready(self, active_token_count: int) -> bool:
        self._pending_micro_steps += 1
        self._pending_active_tokens += active_token_count
        if self._pending_micro_steps < self.gradient_accumulation_steps:
            return False
        self._finish_optimizer_step()
        return True

    def _finish_optimizer_step(self) -> None:
        torch = _torch()
        parameters = tuple(self.actor.trainable_parameters())
        # Micro-queries backpropagate token-loss sums.  Normalize once over all
        # active tokens in the complete (or flushed partial) accumulation window
        # so variable-length query groups receive true batch-normalized weight.
        denominator = max(self._pending_active_tokens, 1)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.div_(denominator)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self._pending_micro_steps = 0
        self._pending_active_tokens = 0
        self._optimizer_steps += 1

    def flush_gradients(self) -> bool:
        """Apply a final partial accumulation window, if one exists."""

        if self._pending_micro_steps == 0:
            return False
        self._finish_optimizer_step()
        return True

    def train_query(self, example: TrainingExample) -> TrainStepResult:
        """Run one complete query group and possibly one optimizer update.

        Counterfactuals are only passed to :func:`score_maskpo_group`.  The
        three log-probability calls receive ``originals`` exclusively, making
        it structurally impossible for probe tokens to enter the BNPO loss.
        """

        torch = _torch()

        # This is the complete frozen-policy sampling phase.  In particular,
        # zero_grad/backward/step cannot occur while originals or probes are
        # being generated.
        originals, counterfactuals = self._sample_frozen_group(example)
        original_texts = tuple(sample.text for sample in originals)
        credit = score_maskpo_group(
            original_texts,
            counterfactuals,
            positives=example.positives,
            slate_k=example.slate_k,
            config=self.maskpo_config,
            normalization_eps=self.normalization_epsilon,
        )
        routes = self._route_group(originals, credit)

        # All fixed targets are evaluated after the entire probe matrix exists
        # but before any backward pass or update.
        old = self.actor.token_logps(originals, requires_grad=False)
        reference = self.reference.token_logps(originals, requires_grad=False)
        current = self.actor.token_logps(originals, requires_grad=True)
        self._validate_logprob_batches(current, old, reference)
        advantages = self._advantage_tensor(routes, current.logps)

        loss_output: BNPOLossOutput = tokenwise_bnpo_loss(
            current.logps,
            old.logps,
            reference.logps,
            advantages,
            current.completion_mask,
            clip_epsilon=self.ppo_clip,
            beta=self.reference_kl_coefficient,
        )
        active_token_count = int(loss_output.token_count.detach().to("cpu").item())
        summed_loss = loss_output.loss * active_token_count
        if not summed_loss.requires_grad:
            raise RuntimeError("actor log probabilities do not carry gradients")
        summed_loss.backward()
        optimizer_stepped = self._apply_gradients_if_ready(active_token_count)

        def scalar(value: Tensor) -> float:
            return float(value.detach().to(dtype=torch.float32, device="cpu").item())

        return TrainStepResult(
            example_id=example.example_id,
            query_credit=credit,
            routes=routes,
            originals=original_texts,
            counterfactuals=counterfactuals,
            loss=scalar(loss_output.loss),
            policy_loss=scalar(loss_output.policy_loss),
            kl=scalar(loss_output.kl),
            clip_fraction=scalar(loss_output.clip_fraction),
            active_token_count=active_token_count,
            optimizer_stepped=optimizer_stepped,
            optimizer_step=self._optimizer_steps,
        )


class HuggingFacePolicyBackend:
    """Adapter for a decoder-only Transformers/PEFT policy."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a pad token or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def render_user_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            # Some compatible tokenizers predate the Qwen ``enable_thinking``
            # keyword but still expose a valid assistant-generation template.
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(rendered, str) or not rendered:
            raise RuntimeError("chat template did not produce a text prompt")
        return rendered

    def _input_device(self):
        try:
            return self.model.get_input_embeddings().weight.device
        except (AttributeError, StopIteration):
            return next(self.model.parameters()).device

    def _eos_ids(self) -> set[int]:
        raw = getattr(self.model.generation_config, "eos_token_id", None)
        if raw is None:
            raw = self.tokenizer.eos_token_id
        if raw is None:
            return set()
        if isinstance(raw, int):
            return {raw}
        return {int(item) for item in raw}

    def _trim_generation(self, token_ids: Sequence[int]) -> tuple[int, ...]:
        eos_ids = self._eos_ids()
        for index, token_id in enumerate(token_ids):
            if token_id in eos_ids:
                return tuple(int(item) for item in token_ids[: index + 1])
        # Padding without EOS should not normally occur, but trimming it is
        # safer than assigning policy credit to batch-shape artifacts.
        values = list(int(item) for item in token_ids)
        while values and values[-1] == self.tokenizer.pad_token_id:
            values.pop()
        return tuple(values)

    def _decode_with_offsets(
        self,
        token_ids: Sequence[int],
    ) -> tuple[str, tuple[tuple[int, int], ...]]:
        special_ids = set(getattr(self.tokenizer, "all_special_ids", ()))
        visible_ids = [int(item) for item in token_ids if item not in special_ids]
        text = self.tokenizer.decode(
            visible_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        offsets = [(0, 0)] * len(token_ids)
        if not visible_ids:
            return text, tuple(offsets)
        try:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            reencoded = encoded["input_ids"]
            visible_offsets = encoded["offset_mapping"]
            if list(reencoded) != visible_ids or len(visible_offsets) != len(visible_ids):
                return text, tuple(offsets)
            visible_index = 0
            for index, token_id in enumerate(token_ids):
                if token_id in special_ids:
                    continue
                start, end = visible_offsets[visible_index]
                offsets[index] = (int(start), int(end))
                visible_index += 1
        except (KeyError, TypeError, ValueError, NotImplementedError):
            # A slow tokenizer has no exact character offsets.  Zero-width
            # offsets deliberately trigger the documented all-A_seq fallback.
            pass
        return text, tuple(offsets)

    def generate(
        self,
        model_prefixes: Sequence[str],
        config: SamplingConfig,
    ) -> Sequence[GeneratedCompletion]:
        if not model_prefixes:
            return ()
        torch = _torch()
        encoded = self.tokenizer(
            list(model_prefixes),
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        prompt_width = int(input_ids.shape[1])
        prompt_token_ids = tuple(
            tuple(int(item) for item in row[mask.bool()].tolist())
            for row, mask in zip(input_ids, attention_mask, strict=True)
        )
        device = self._input_device()
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **config.generation_kwargs(),
                )
        finally:
            self.model.train(was_training)
        sequences = output.sequences if hasattr(output, "sequences") else output
        if len(sequences) != len(model_prefixes):
            raise RuntimeError("model.generate returned an unexpected batch size")

        samples: list[GeneratedCompletion] = []
        for row, prompt_ids in zip(sequences, prompt_token_ids, strict=True):
            sampled_ids = self._trim_generation(row[prompt_width:].tolist())
            text, offsets = self._decode_with_offsets(sampled_ids)
            samples.append(
                GeneratedCompletion(
                    text=text,
                    prompt_token_ids=prompt_ids,
                    token_ids=sampled_ids,
                    token_offsets=offsets,
                )
            )
        return tuple(samples)

    def token_logps(
        self,
        samples: Sequence[GeneratedCompletion],
        *,
        requires_grad: bool,
    ) -> LogProbBatch:
        if not samples:
            raise ValueError("at least one generated completion is required")
        if any(not sample.token_ids for sample in samples):
            raise ValueError("cannot train on an empty generated completion")

        torch = _torch()
        device = self._input_device()
        was_training = self.model.training
        # Eval mode makes old/current probabilities comparable even for models
        # with dropout.  Autograd remains fully enabled for the current pass.
        self.model.eval()
        rows: list[Tensor] = []
        context = nullcontext() if requires_grad else torch.no_grad()
        try:
            with context:
                for sample in samples:
                    prompt_length = len(sample.prompt_token_ids)
                    ids = (*sample.prompt_token_ids, *sample.token_ids)
                    input_ids = torch.tensor(
                        [ids], dtype=torch.long, device=device
                    )
                    attention_mask = torch.ones_like(input_ids)
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    logits = output.logits[
                        0, prompt_length - 1 : len(ids) - 1, :
                    ].float()
                    targets = input_ids[0, prompt_length:].to(logits.device)
                    selected = logits.gather(1, targets[:, None]).squeeze(1)
                    rows.append(selected - torch.logsumexp(logits, dim=-1))
        finally:
            self.model.train(was_training)

        width = max(len(sample.token_ids) for sample in samples)
        padded_rows: list[Tensor] = []
        masks: list[Tensor] = []
        for row, sample in zip(rows, samples, strict=True):
            padding = width - len(sample.token_ids)
            padded_rows.append(torch.nn.functional.pad(row, (0, padding)))
            masks.append(
                torch.arange(width, device=row.device) < len(sample.token_ids)
            )
        return LogProbBatch(
            logps=torch.stack(padded_rows),
            completion_mask=torch.stack(masks),
        )

    def trainable_parameters(self) -> Iterable[Tensor]:
        return (parameter for parameter in self.model.parameters() if parameter.requires_grad)

    def save_pretrained(self, output_dir: str | Path) -> None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(destination)
        self.tokenizer.save_pretrained(destination)


def _resolve_dtype(name: str):
    torch = _torch()
    normalized = name.lower().replace("torch.", "")
    choices = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return choices[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported model dtype: {name!r}") from exc


def load_huggingface_maskpo_trainer(
    *,
    model_name_or_path: str,
    adapter_path: str,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    device_map: Any = "auto",
    reference_device_map: Any | None = None,
    gradient_checkpointing: bool = True,
    maskpo_config: MaskPOConfig = MaskPOConfig(),
    sampling_config: SamplingConfig = SamplingConfig(),
    ppo_clip: float = 0.2,
    reference_kl_coefficient: float = 0.001,
    normalization_epsilon: float = 1e-8,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = 1.0,
) -> MaskPOTrainer:
    """Load trainable and reference copies of the Phase-2 SFT policy.

    ``adapter_path`` must point to ``p2_rubric_reasoning_sft`` (or an explicit
    compatible reproduction).  The adapter is loaded twice: once trainable,
    and once frozen as the fixed reference used by the KL penalty.  Loading a
    raw base model as reference would implement a different algorithm.
    """

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Hugging Face MaskPO training requires transformers, peft, and "
            "accelerate; install the project's training dependencies"
        ) from exc

    torch = _torch()
    torch_dtype = _resolve_dtype(dtype)
    common_model_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if device_map is not None:
        common_model_kwargs["device_map"] = device_map

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=trust_remote_code,
    )
    actor_base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **common_model_kwargs,
    )
    actor_model = PeftModel.from_pretrained(
        actor_base,
        adapter_path,
        is_trainable=True,
    )
    if gradient_checkpointing:
        actor_model.gradient_checkpointing_enable()
        if hasattr(actor_model, "enable_input_require_grads"):
            actor_model.enable_input_require_grads()
    if hasattr(actor_model, "config"):
        actor_model.config.use_cache = False

    reference_kwargs = dict(common_model_kwargs)
    if reference_device_map is not None:
        reference_kwargs["device_map"] = reference_device_map
    reference_base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **reference_kwargs,
    )
    reference_model = PeftModel.from_pretrained(
        reference_base,
        adapter_path,
        is_trainable=False,
    )
    reference_model.requires_grad_(False)
    reference_model.eval()
    if hasattr(reference_model, "config"):
        reference_model.config.use_cache = False

    actor = HuggingFacePolicyBackend(actor_model, tokenizer)
    reference = HuggingFacePolicyBackend(reference_model, tokenizer)
    trainable = tuple(actor.trainable_parameters())
    if not trainable:
        raise RuntimeError("the actor adapter has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimizer.zero_grad(set_to_none=True)
    return MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        maskpo_config=maskpo_config,
        sampling_config=sampling_config,
        ppo_clip=ppo_clip,
        reference_kl_coefficient=reference_kl_coefficient,
        normalization_epsilon=normalization_epsilon,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
    )
