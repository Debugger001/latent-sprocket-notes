from __future__ import annotations

import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from openbench_rerank_rl.losses import BNPOLossOutput
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.trainer import (
    GeneratedCompletion,
    HuggingFacePolicyBackend,
    LogProbBatch,
    MaskPOTrainer,
    PPOPassMetrics,
    SamplingConfig,
    TrainingExample,
    _targets_with_serialized_lora_weights,
    load_huggingface_maskpo_trainer,
)


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 7
    padding_side = "right"


class _NonCanonicalTokenizer(_TinyTokenizer):
    all_special_ids = []

    def decode(self, token_ids, **_kwargs):
        return "".join({1: "a", 2: "b", 3: "ab"}[token_id] for token_id in token_ids)

    def __call__(self, text, **_kwargs):
        assert text == "ab"
        return {"input_ids": [3], "offset_mapping": [(0, 2)]}


class _SelectiveLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.projection = torch.nn.Linear(3, 8, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=7)
        self.forward_training: list[bool] = []
        self.kept_positions: list[tuple[int, ...]] = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        *,
        input_ids,
        attention_mask,
        use_cache,
        logits_to_keep=0,
    ):
        del attention_mask, use_cache
        self.forward_training.append(self.training)
        hidden = self.embedding(input_ids)
        if isinstance(logits_to_keep, int):
            indices = slice(-logits_to_keep, None)
        else:
            self.kept_positions.append(tuple(int(item) for item in logits_to_keep))
            indices = logits_to_keep
        return SimpleNamespace(logits=self.projection(hidden[:, indices, :]))


class _FullLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        self.projection = torch.nn.Linear(3, 8, bias=False)
        self.generation_config = SimpleNamespace(eos_token_id=7)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, input_ids, attention_mask, use_cache):
        del attention_mask, use_cache
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def _completion(order: str) -> str:
    bodies = "\n\n".join(
        f"{header} evidence-{index}"
        for index, header in enumerate(DEFAULT_RUBRIC_HEADERS)
    )
    return (
        f"<think>\n{bodies}\n\n**Synthesis:** combined\n</think>\n"
        f"<answer>\n{order}\n</answer>"
    )


def _sample(text: str, prompt_marker: int = 101) -> GeneratedCompletion:
    return GeneratedCompletion(
        text=text,
        prompt_token_ids=(prompt_marker,),
        token_ids=tuple((ord(character) % 251) + 1 for character in text),
        token_offsets=tuple((index, index + 1) for index in range(len(text))),
    )


class FakeBackend:
    def __init__(self, name: str, events: list[tuple], *, trainable: bool) -> None:
        self.name = name
        self.events = events
        self.parameter = torch.nn.Parameter(
            torch.tensor(0.0), requires_grad=trainable
        )
        self.generated_sample_count = 0
        self.logprob_sample_counts: list[int] = []
        self._original_index = 0
        self._probe_index = 0

    def render_user_prompt(self, prompt: str) -> str:
        return f"CHAT:{prompt}\nASSISTANT:"

    def generate(
        self,
        model_prefixes: Sequence[str],
        config: SamplingConfig,
    ) -> Sequence[GeneratedCompletion]:
        assert self.name == "actor"
        assert all(self.parameter.item() == 0.0 for _ in model_prefixes)
        is_probe = bool(model_prefixes and model_prefixes[0].endswith("**Synthesis:**"))
        self.events.append(("generate", is_probe, len(model_prefixes)))
        self.generated_sample_count += len(model_prefixes)
        samples = []
        original_orders = (
            "[1, 2, 3]",
            "[2, 1, 3]",
            "[3, 2, 1]",
            "[1, 3, 2]",
        )
        for prefix in model_prefixes:
            if is_probe:
                # Exactly one of the 16 suffixes is unparseable.  It remains a
                # scoring probe but is excluded from the delta/RMS matrix.
                text = (
                    " no parseable answer"
                    if self._probe_index == 0
                    else " regenerated\n</think>\n<answer>[1, 2, 3]</answer>"
                )
                self._probe_index += 1
            else:
                text = _completion(original_orders[self._original_index % 4])
                self._original_index += 1
            samples.append(_sample(text, prompt_marker=(len(prefix) % 200) + 1))
        return samples

    def token_logps(
        self,
        samples: Sequence[GeneratedCompletion],
        *,
        requires_grad: bool,
    ) -> LogProbBatch:
        self.events.append(("logps", self.name, requires_grad, len(samples)))
        self.logprob_sample_counts.append(len(samples))
        width = max(len(sample.token_ids) for sample in samples)
        base = self.parameter.expand(len(samples), width)
        logps = base if requires_grad else base.detach().clone()
        mask = torch.zeros((len(samples), width), dtype=torch.bool)
        for index, sample in enumerate(samples):
            mask[index, : len(sample.token_ids)] = True
        return LogProbBatch(logps=logps, completion_mask=mask)

    def trainable_parameters(self):
        return (self.parameter,) if self.parameter.requires_grad else ()


class RecordingSGD(torch.optim.SGD):
    def __init__(self, params, events):
        super().__init__(params, lr=0.01)
        self.events = events

    def step(self, closure=None):
        self.events.append(("optimizer_step",))
        return super().step(closure)


def _controlled_mean_loss(specs):
    remaining = iter(specs)

    def loss(current_logps, *_args, **_kwargs):
        token_count, mean_gradient = next(remaining)
        mean = current_logps[0, 0] * mean_gradient
        zero = mean.detach() * 0.0
        return BNPOLossOutput(
            loss=mean,
            policy_loss=mean,
            kl=zero,
            clip_fraction=zero,
            token_count=torch.tensor(token_count, device=current_logps.device),
        )

    return loss


def test_training_step_samples_all_probes_before_originals_only_loss():
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    optimizer = RecordingSGD(actor.trainable_parameters(), events)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        sampling_config=SamplingConfig(
            max_new_tokens=50,
            counterfactual_max_new_tokens=25,
            original_batch_size=4,
            counterfactual_batch_size=4,
        ),
        ppo_passes=1,
    )

    result = trainer.train_query(
        TrainingExample(
            prompt="rank these articles",
            positives=frozenset({1}),
            slate_k=3,
            example_id="synthetic-1",
        )
    )

    assert actor.generated_sample_count == 20  # four originals plus 16 probes
    assert reference.generated_sample_count == 0
    assert actor.logprob_sample_counts == [4, 4]
    assert reference.logprob_sample_counts == [4]
    assert result.query_credit.valid_probe_count == 15
    assert result.active_token_count == sum(len(text) for text in result.originals)
    assert result.optimizer_stepped
    assert trainer.optimizer_steps == 1
    first_logps = next(index for index, event in enumerate(events) if event[0] == "logps")
    assert all(event[0] == "generate" for event in events[:first_logps])
    assert sum(event[2] for event in events if event[0] == "generate") == 20
    assert events[-1][0] == "optimizer_step"
    assert all(not route.used_sequence_fallback for route in result.routes)


def test_gradient_accumulation_flushes_partial_window():
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=1,
        max_grad_norm=None,
    )
    result = trainer.train_query(
        TrainingExample("prompt", frozenset({1}), 3)
    )
    assert not result.optimizer_stepped
    assert trainer.pending_micro_steps == 1
    assert trainer.flush_gradients()
    assert trainer.optimizer_steps == 1
    assert trainer.rollout_steps == 1
    assert not trainer.flush_gradients()


def test_gradient_accumulation_normalizes_once_over_all_active_tokens(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=1,
        max_grad_norm=None,
    )
    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss",
        _controlled_mean_loss([(2, 1.0), (8, 3.0)]),
    )

    first = trainer.train_query(TrainingExample("prompt-1", frozenset({1}), 3))
    second = trainer.train_query(TrainingExample("prompt-2", frozenset({1}), 3))

    assert not first.optimizer_stepped
    assert second.optimizer_stepped
    assert trainer.pending_active_tokens == 0
    # The accumulated gradient is (2 * 1 + 8 * 3) / (2 + 8) = 2.6.
    assert actor.parameter.item() == pytest.approx(-0.026)


def test_partial_accumulation_uses_its_actual_active_token_count(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=4,
        ppo_passes=1,
        max_grad_norm=None,
    )
    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss",
        _controlled_mean_loss([(7, 2.5)]),
    )

    result = trainer.train_query(TrainingExample("prompt", frozenset({1}), 3))
    assert not result.optimizer_stepped
    assert trainer.pending_active_tokens == 7
    assert trainer.flush_gradients()
    assert trainer.pending_active_tokens == 0
    assert actor.parameter.item() == pytest.approx(-0.025)


def test_single_micro_step_preserves_mean_loss_gradient(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=1,
        ppo_passes=1,
        max_grad_norm=None,
    )
    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss",
        _controlled_mean_loss([(11, 1.75)]),
    )

    result = trainer.train_query(TrainingExample("prompt", frozenset({1}), 3))

    assert result.optimizer_stepped
    assert actor.parameter.item() == pytest.approx(-0.0175)


def test_two_ppo_passes_replay_one_rollout_step_against_fixed_old_policy(
    monkeypatch,
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=2,
        max_grad_norm=None,
    )
    seen_logps: list[tuple[float, float]] = []

    def recording_loss(current_logps, old_logps, *_args, **_kwargs):
        seen_logps.append(
            (float(current_logps[0, 0].detach()), float(old_logps[0, 0]))
        )
        mean = current_logps[0, 0]
        zero = mean.detach() * 0.0
        return BNPOLossOutput(
            loss=mean,
            policy_loss=mean,
            kl=zero,
            clip_fraction=zero,
            token_count=torch.tensor(1, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss", recording_loss
    )

    first = trainer.train_query(TrainingExample("prompt-1", frozenset({1}), 3))
    second = trainer.train_query(TrainingExample("prompt-2", frozenset({1}), 3))

    assert not first.rollout_step_completed
    assert first.optimizer_steps_applied == 0
    assert second.rollout_step_completed
    assert second.optimizer_steps_applied == 2
    assert second.rollout_step == 1
    assert second.optimizer_step == 2
    assert trainer.rollout_steps == 1
    assert trainer.optimizer_steps == 2
    assert actor.generated_sample_count == 40
    assert reference.logprob_sample_counts == [4, 4]
    assert actor.logprob_sample_counts == [4, 4, 4, 4, 4, 4]

    # Both prompt groups were sampled before the first update. The replayed
    # pass sees the updated actor while retaining log-probabilities from the
    # unchanged behavior policy.
    first_optimizer_event = events.index(("optimizer_step",))
    assert sum(
        event[2] for event in events[:first_optimizer_event] if event[0] == "generate"
    ) == 40
    assert seen_logps[:2] == [(0.0, 0.0), (0.0, 0.0)]
    assert [old for _current, old in seen_logps[2:]] == [0.0, 0.0]
    assert [current for current, _old in seen_logps[2:]] == pytest.approx(
        [-0.01, -0.01]
    )
    assert actor.parameter.item() == pytest.approx(-0.02)


def test_completed_rollout_exposes_token_weighted_metrics_for_every_pass(
    monkeypatch,
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=2,
        max_grad_norm=None,
    )
    specs = iter(
        [
            (2, 1.0, 2.0, 3.0, 0.1),
            (8, 11.0, 12.0, 13.0, 0.6),
            (2, 21.0, 22.0, 23.0, 0.2),
            (8, 31.0, 32.0, 33.0, 0.7),
        ]
    )

    def metric_loss(current_logps, *_args, **_kwargs):
        token_count, loss_value, policy_loss, kl, clip_fraction = next(specs)
        differentiable = current_logps[0, 0]
        loss = differentiable + (loss_value - float(differentiable.detach()))
        return BNPOLossOutput(
            loss=loss,
            policy_loss=differentiable.detach().new_tensor(policy_loss),
            kl=differentiable.detach().new_tensor(kl),
            clip_fraction=differentiable.detach().new_tensor(clip_fraction),
            token_count=torch.tensor(token_count, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss", metric_loss
    )

    first = trainer.train_query(TrainingExample("prompt-1", frozenset({1}), 3))
    completed = trainer.train_query(
        TrainingExample("prompt-2", frozenset({1}), 3)
    )

    assert first.ppo_pass_metrics == ()
    assert completed.loss == pytest.approx(11.0)  # Backwards-compatible query metric.
    assert len(completed.ppo_pass_metrics) == 2
    pass_one, pass_two = completed.ppo_pass_metrics
    assert isinstance(pass_one, PPOPassMetrics)
    assert pass_one.ppo_pass == 1
    assert pass_one.active_token_count == 10
    assert pass_one.loss == pytest.approx(9.0)
    assert pass_one.policy_loss == pytest.approx(10.0)
    assert pass_one.kl == pytest.approx(11.0)
    assert pass_one.clip_fraction == pytest.approx(0.5)
    assert pass_two.ppo_pass == 2
    assert pass_two.active_token_count == 10
    assert pass_two.loss == pytest.approx(29.0)
    assert pass_two.policy_loss == pytest.approx(30.0)
    assert pass_two.kl == pytest.approx(31.0)
    assert pass_two.clip_fraction == pytest.approx(0.6)
    assert trainer.last_ppo_pass_metrics == completed.ppo_pass_metrics


def test_pending_replay_targets_are_cpu_resident_and_graph_free(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=2,
    )
    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss",
        _controlled_mean_loss([(3, 1.0)]),
    )

    trainer.train_query(TrainingExample("prompt", frozenset({1}), 3))
    pending = trainer._pending_replay_queries[0]

    for batch in (pending.old, pending.reference):
        assert batch.logps.device.type == "cpu"
        assert not batch.logps.requires_grad
        assert batch.logps.grad_fn is None
        assert batch.completion_mask.device.type == "cpu"
        assert not batch.completion_mask.requires_grad
        assert batch.completion_mask.grad_fn is None
    with pytest.raises(RuntimeError, match="clean rollout boundary"):
        trainer.training_state_dict()
    with pytest.raises(RuntimeError, match="clean rollout boundary"):
        trainer.load_training_state_dict(
            {"version": 1, "rollout_steps": 0, "optimizer_steps": 0, "ppo_passes": 2}
        )


def test_replay_failure_poisons_trainer_and_discards_partial_state(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=2,
        max_grad_norm=None,
    )
    calls = 0

    def failing_loss(current_logps, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("replay exploded")
        mean = current_logps[0, 0]
        zero = mean.detach() * 0.0
        return BNPOLossOutput(
            loss=mean,
            policy_loss=mean,
            kl=zero,
            clip_fraction=zero,
            token_count=torch.tensor(1, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss", failing_loss
    )
    trainer.train_query(TrainingExample("prompt-1", frozenset({1}), 3))

    with pytest.raises(RuntimeError, match="replay exploded"):
        trainer.train_query(TrainingExample("prompt-2", frozenset({1}), 3))

    assert trainer.poisoned
    assert trainer.optimizer_steps == 1
    assert trainer.rollout_steps == 0
    assert trainer.pending_micro_steps == 0
    assert trainer.pending_active_tokens == 0
    assert trainer._pending_replay_queries == []
    assert actor.parameter.grad is None
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.train_query(TrainingExample("prompt-3", frozenset({1}), 3))
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.flush_gradients()
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.training_state_dict()


@pytest.mark.parametrize("nonfinite_field", ["loss", "policy_loss", "kl", "clip_fraction"])
def test_nonfinite_first_pass_diagnostic_poisons_pending_rollout_before_step(
    monkeypatch,
    nonfinite_field,
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=2,
        ppo_passes=1,
        max_grad_norm=None,
    )
    calls = 0

    def nonfinite_loss(current_logps, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        differentiable = current_logps[0, 0]
        values = {
            "loss": differentiable,
            "policy_loss": differentiable.detach(),
            "kl": differentiable.detach(),
            "clip_fraction": differentiable.detach(),
        }
        if calls == 2:
            replacement = differentiable.detach().new_tensor(float("nan"))
            if nonfinite_field == "loss":
                replacement = differentiable * 0.0 + replacement
            values[nonfinite_field] = replacement
        return BNPOLossOutput(
            **values,
            token_count=torch.tensor(1, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss", nonfinite_loss
    )
    trainer.train_query(TrainingExample("prompt-1", frozenset({1}), 3))

    with pytest.raises(FloatingPointError, match=f"non-finite MaskPO {nonfinite_field}"):
        trainer.train_query(TrainingExample("prompt-2", frozenset({1}), 3))

    assert trainer.poisoned
    assert trainer.optimizer_steps == 0
    assert trainer.rollout_steps == 0
    assert actor.parameter.item() == pytest.approx(0.0)
    assert actor.parameter.grad is None
    assert ("optimizer_step",) not in events


def test_nonfinite_replay_diagnostic_prevents_second_optimizer_step(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=1,
        ppo_passes=2,
        max_grad_norm=None,
    )
    calls = 0

    def nonfinite_replay_loss(current_logps, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        loss = current_logps[0, 0]
        zero = loss.detach() * 0.0
        clip_fraction = zero if calls == 1 else zero.new_tensor(float("nan"))
        return BNPOLossOutput(
            loss=loss,
            policy_loss=loss,
            kl=zero,
            clip_fraction=clip_fraction,
            token_count=torch.tensor(1, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss", nonfinite_replay_loss
    )

    with pytest.raises(FloatingPointError, match="non-finite MaskPO clip_fraction"):
        trainer.train_query(TrainingExample("prompt", frozenset({1}), 3))

    assert trainer.poisoned
    assert trainer.optimizer_steps == 1
    assert trainer.rollout_steps == 0
    assert actor.parameter.item() == pytest.approx(-0.01)
    assert torch.isfinite(actor.parameter)
    assert actor.parameter.grad is None
    assert events.count(("optimizer_step",)) == 1


@pytest.mark.parametrize(
    ("max_grad_norm", "error_type"),
    [(1.0, RuntimeError), (None, FloatingPointError)],
)
def test_nonfinite_gradient_is_rejected_before_optimizer_step(
    monkeypatch,
    max_grad_norm,
    error_type,
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=1,
        ppo_passes=1,
        max_grad_norm=max_grad_norm,
    )

    def nonfinite_gradient_loss(current_logps, *_args, **_kwargs):
        loss = current_logps[0, 0]
        loss.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        zero = loss.detach() * 0.0
        return BNPOLossOutput(
            loss=loss,
            policy_loss=loss,
            kl=zero,
            clip_fraction=zero,
            token_count=torch.tensor(1, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss",
        nonfinite_gradient_loss,
    )

    with pytest.raises(error_type, match="non-finite"):
        trainer.train_query(TrainingExample("prompt", frozenset({1}), 3))

    assert trainer.poisoned
    assert trainer.optimizer_steps == 0
    assert trainer.rollout_steps == 0
    assert actor.parameter.item() == pytest.approx(0.0)
    assert torch.isfinite(actor.parameter)
    assert actor.parameter.grad is None
    assert ("optimizer_step",) not in events


def test_canonical_eight_prompt_rollout_counts_two_optimizer_steps(monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        gradient_accumulation_steps=8,
        ppo_passes=2,
        max_grad_norm=None,
    )

    def zero_gradient_loss(current_logps, *_args, **_kwargs):
        loss = current_logps[0, 0] * 0.0
        zero = loss.detach()
        return BNPOLossOutput(
            loss=loss,
            policy_loss=loss,
            kl=zero,
            clip_fraction=zero,
            token_count=torch.tensor(1, device=current_logps.device),
        )

    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss", zero_gradient_loss
    )

    results = [
        trainer.train_query(
            TrainingExample(f"prompt-{index}", frozenset({1}), 3)
        )
        for index in range(8)
    ]

    assert all(not result.rollout_step_completed for result in results[:7])
    assert results[-1].rollout_step_completed
    assert results[-1].rollout_step == 1
    assert results[-1].optimizer_steps_applied == 2
    assert results[-1].optimizer_step == 2
    assert trainer.rollout_steps == 1
    assert trainer.optimizer_steps == 2
    assert actor.generated_sample_count == 8 * 20


def test_training_state_counters_round_trip_at_a_clean_boundary():
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        ppo_passes=2,
    )
    state = {
        "version": 1,
        "rollout_steps": 300,
        "optimizer_steps": 600,
        "ppo_passes": 2,
    }

    trainer.load_training_state_dict(state)

    assert trainer.rollout_steps == 300
    assert trainer.optimizer_steps == 600
    assert trainer.training_state_dict() == state
    assert trainer.last_ppo_pass_metrics == ()


@pytest.mark.parametrize(
    ("state", "message"),
    [
        (
            {"version": 1, "rollout_steps": 3, "optimizer_steps": 6, "ppo_passes": 1},
            "ppo_passes does not match",
        ),
        (
            {"version": 1, "rollout_steps": 3, "optimizer_steps": 5, "ppo_passes": 2},
            r"optimizer_steps must equal rollout_steps \* ppo_passes",
        ),
        (
            {"version": 2, "rollout_steps": 3, "optimizer_steps": 6, "ppo_passes": 2},
            "unsupported training state version",
        ),
        (
            {"version": 1, "rollout_steps": True, "optimizer_steps": 0, "ppo_passes": 2},
            "rollout_steps must be a non-negative integer",
        ),
    ],
)
def test_training_state_rejects_incompatible_or_invalid_counters(state, message):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = MaskPOTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        ppo_passes=2,
    )

    with pytest.raises(ValueError, match=message):
        trainer.load_training_state_dict(state)
    assert trainer.training_state_dict() == {
        "version": 1,
        "rollout_steps": 0,
        "optimizer_steps": 0,
        "ppo_passes": 2,
    }


@pytest.mark.parametrize("ppo_passes", [0, -1, 1.5, True])
def test_ppo_passes_requires_a_positive_integer(ppo_passes):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    with pytest.raises(ValueError, match="ppo_passes"):
        MaskPOTrainer(
            actor=actor,
            reference=reference,
            optimizer=RecordingSGD(actor.trainable_parameters(), events),
            ppo_passes=ppo_passes,
        )


def test_training_example_rejects_out_of_range_positive():
    with pytest.raises(ValueError, match="1..slate_k"):
        TrainingExample("prompt", frozenset({4}), 3)


def test_sampling_config_keeps_exact_generation_defaults():
    config = SamplingConfig()
    assert config.generation_kwargs() == {
        "do_sample": True,
        "max_new_tokens": 2048,
        "use_cache": True,
        "temperature": 0.6,
        "top_k": 20,
        "top_p": 0.95,
    }


def test_huggingface_logps_keep_only_prediction_positions_and_train_for_gradients():
    torch.manual_seed(0)
    model = _SelectiveLogitModel()
    model.eval()
    backend = HuggingFacePolicyBackend(model, _TinyTokenizer())
    sample = GeneratedCompletion(
        text="two tokens",
        prompt_token_ids=(1, 2),
        token_ids=(3, 4),
        token_offsets=((0, 3), (4, 10)),
    )

    batch = backend.token_logps([sample], requires_grad=True)

    assert model.forward_training == [True]
    assert not model.training  # restore the caller's original mode
    assert model.kept_positions == [(1, 2)]
    assert batch.logps.shape == (1, 2)
    assert batch.logps.requires_grad
    batch.logps.sum().backward()
    assert model.projection.weight.grad is not None


def test_decode_offsets_preserve_noncanonical_sampled_bpe_path():
    backend = HuggingFacePolicyBackend(_FullLogitModel(), _NonCanonicalTokenizer())
    text, offsets = backend._decode_with_offsets([1, 2])
    assert text == "ab"
    assert offsets == ((0, 1), (1, 2))


def test_huggingface_logps_eval_no_grad_and_full_logit_fallback_are_exact():
    torch.manual_seed(0)
    model = _FullLogitModel()
    model.train()
    backend = HuggingFacePolicyBackend(model, _TinyTokenizer())
    sample = GeneratedCompletion(
        text="two tokens",
        prompt_token_ids=(1, 2),
        token_ids=(3, 4),
        token_offsets=((0, 3), (4, 10)),
    )

    batch = backend.token_logps([sample], requires_grad=False)

    assert model.training  # restore the caller's original mode
    assert not batch.logps.requires_grad
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        full_logits = model.projection(model.embedding(ids))[0, 1:3]
        expected = -torch.nn.functional.cross_entropy(
            full_logits.float(),
            torch.tensor([3, 4]),
            reduction="none",
        )
    assert torch.allclose(batch.logps[0], expected)


def test_adapter_targets_are_limited_to_complete_serialized_lora_pairs():
    retained, missing = _targets_with_serialized_lora_weights(
        ["q_proj", "k_proj", "lm_head"],
        [
            "base.model.layers.0.q_proj.lora_A.weight",
            "base.model.layers.0.q_proj.lora_B.weight",
            "base.model.layers.0.k_proj.lora_A.weight",
            "base.model.lm_head.lora_A.weight",
        ],
    )
    assert retained == ("q_proj",)
    assert missing == ("k_proj", "lm_head")


def test_huggingface_loader_can_resume_actor_without_changing_phase2_reference(
    monkeypatch,
):
    tokenizer_calls: list[str] = []
    config_calls: list[str] = []
    peft_calls: list[tuple[str, bool, str]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            tokenizer_calls.append(str(path))
            return _TinyTokenizer()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            model_calls.append((str(path), kwargs))
            return torch.nn.Module()

    class FakePeftConfig:
        @classmethod
        def from_pretrained(cls, path):
            config_calls.append(str(path))
            return SimpleNamespace(target_modules=None, source=str(path))

    class FakePeftModel(torch.nn.Module):
        def __init__(self, *, trainable: bool) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor(0.0), requires_grad=trainable
            )
            self.config = SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, _base, path, *, config, is_trainable):
            peft_calls.append((str(path), bool(is_trainable), config.source))
            return cls(trainable=bool(is_trainable))

    peft_module = ModuleType("peft")
    peft_module.PeftConfig = FakePeftConfig
    peft_module.PeftModel = FakePeftModel
    transformers_module = ModuleType("transformers")
    transformers_module.AutoModelForCausalLM = FakeAutoModel
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    resumed = load_huggingface_maskpo_trainer(
        model_name_or_path="base-model",
        adapter_path="phase2-reference",
        actor_adapter_path="rollout-300-adapter",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        dtype="float32",
        device_map=None,
        gradient_checkpointing=False,
    )

    assert tokenizer_calls == ["rollout-300-adapter"]
    assert config_calls == ["rollout-300-adapter", "phase2-reference"]
    assert peft_calls == [
        ("rollout-300-adapter", True, "rollout-300-adapter"),
        ("phase2-reference", False, "phase2-reference"),
    ]
    assert [path for path, _kwargs in model_calls] == ["base-model", "base-model"]
    assert [kwargs["revision"] for _path, kwargs in model_calls] == [
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    ]
    assert next(resumed.actor.trainable_parameters()).requires_grad
    assert not next(resumed.reference.model.parameters()).requires_grad

    tokenizer_calls.clear()
    config_calls.clear()
    peft_calls.clear()
    model_calls.clear()
    load_huggingface_maskpo_trainer(
        model_name_or_path="base-model",
        adapter_path="phase2-reference",
        dtype="float32",
        device_map=None,
        gradient_checkpointing=False,
    )
    assert tokenizer_calls == ["phase2-reference"]
    assert config_calls == ["phase2-reference"]
    assert peft_calls == [
        ("phase2-reference", True, "phase2-reference"),
        ("phase2-reference", False, "phase2-reference"),
    ]
    assert all("revision" not in kwargs for _path, kwargs in model_calls)
