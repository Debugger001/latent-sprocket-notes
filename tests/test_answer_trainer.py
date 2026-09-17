from __future__ import annotations

import sys
from collections.abc import Sequence
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from openbench_rerank_rl.trainer import (
    GeneratedCompletion,
    LogProbBatch,
    SamplingConfig,
    TrainingExample,
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
        self._generation_index = 0

    def render_user_prompt(self, prompt: str) -> str:
        return f"CHAT:{prompt}\nASSISTANT:"

    def generate(
        self,
        model_prefixes: Sequence[str],
        config: SamplingConfig,
    ) -> Sequence[GeneratedCompletion]:
        del config
        assert self.name == "actor"
        self.events.append(("generate", len(model_prefixes)))
        self.generated_sample_count += len(model_prefixes)
        orders = ("[1, 2, 3]", "[2, 1, 3]", "[3, 2, 1]", "[1, 3, 2]")
        samples = []
        for prefix in model_prefixes:
            text = orders[self._generation_index % len(orders)]
            self._generation_index += 1
            samples.append(_sample(text, prompt_marker=len(prefix)))
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
        values = self.parameter.expand(len(samples), width)
        logps = values if requires_grad else values.detach().clone()
        mask = torch.zeros((len(samples), width), dtype=torch.bool)
        for row, sample in enumerate(samples):
            mask[row, : len(sample.token_ids)] = True
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


def _loss_output(loss, *, tokens: int, actions: int):
    zero = loss.detach() * 0.0
    return SimpleNamespace(
        loss=loss,
        policy_loss=loss,
        kl=zero,
        clip_fraction=zero,
        token_count=torch.tensor(tokens, device=loss.device),
        action_count=torch.tensor(actions, device=loss.device),
    )


@pytest.fixture
def answer_trainer_module():
    # Core and runtime are developed as separate, import-light modules.  A
    # fixture keeps the optional torch dependency local to these tests.
    import openbench_rerank_rl.answer_trainer as module

    return module


def test_grpo_samples_exactly_four_originals_and_no_probes(answer_trainer_module):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = answer_trainer_module.AnswerRLTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        algorithm="grpo",
        ppo_passes=1,
    )

    result = trainer.train_query(
        TrainingExample("rank the slate", frozenset({1}), 3, "row-1")
    )

    assert actor.generated_sample_count == 4
    assert reference.generated_sample_count == 0
    assert [event for event in events if event[0] == "generate"] == [("generate", 4)]
    assert actor.logprob_sample_counts == [4, 4]
    assert reference.logprob_sample_counts == [4]
    assert result.originals == ("[1, 2, 3]", "[2, 1, 3]", "[3, 2, 1]", "[1, 3, 2]")
    assert result.completion_count == 4
    assert result.action_count == 4
    assert result.active_token_count == sum(map(len, result.originals))
    assert result.optimizer_stepped
    assert result.rollout_step == 1
    assert trainer.optimizer_steps == 1
    assert events[-1] == ("optimizer_step",)


def test_gradient_accumulation_gives_queries_equal_weight(
    answer_trainer_module, monkeypatch
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = answer_trainer_module.AnswerRLTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        algorithm="grpo",
        gradient_accumulation_steps=2,
        max_grad_norm=None,
    )
    specs = iter(((3, 1, 1.0), (300, 4, 3.0)))

    def controlled(current_logps, *_args, **_kwargs):
        tokens, actions, gradient = next(specs)
        return _loss_output(
            current_logps[0, 0] * gradient,
            tokens=tokens,
            actions=actions,
        )

    monkeypatch.setattr(answer_trainer_module, "sequence_grpo_loss", controlled)
    first = trainer.train_query(TrainingExample("one", frozenset({1}), 3))
    second = trainer.train_query(TrainingExample("two", frozenset({1}), 3))

    assert not first.optimizer_stepped
    assert second.optimizer_stepped
    # Equal query weighting: (1 + 3) / 2 = 2.  Token weighting would be ~2.98.
    assert actor.parameter.item() == pytest.approx(-0.02)
    metrics = second.ppo_pass_metrics[0]
    assert metrics.loss == pytest.approx(0.0)
    assert metrics.query_count == 2
    assert metrics.completion_count == 8
    assert metrics.action_count == 5
    assert metrics.active_token_count == 303


def test_flush_uses_actual_partial_query_count(answer_trainer_module, monkeypatch):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = answer_trainer_module.AnswerRLTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        algorithm="grpo",
        gradient_accumulation_steps=8,
        max_grad_norm=None,
    )

    def controlled(current_logps, *_args, **_kwargs):
        return _loss_output(current_logps[0, 0] * 2.5, tokens=9, actions=4)

    monkeypatch.setattr(answer_trainer_module, "sequence_grpo_loss", controlled)
    result = trainer.train_query(TrainingExample("one", frozenset({1}), 3))
    assert not result.optimizer_stepped
    assert trainer.pending_micro_steps == 1
    assert trainer.pending_active_tokens == 9
    assert trainer.flush_gradients()
    assert actor.parameter.item() == pytest.approx(-0.025)
    assert trainer.rollout_steps == 1
    assert not trainer.flush_gradients()


def test_rank_grpo_passes_only_segmented_item_tokens(
    answer_trainer_module, monkeypatch
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    trainer = answer_trainer_module.AnswerRLTrainer(
        actor=actor,
        reference=reference,
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        algorithm="rank_grpo",
        max_grad_norm=None,
    )
    observed = {}

    def controlled(
        current_logps,
        _old_logps,
        _ref_logps,
        token_advantages,
        item_ids,
        completion_mask,
        **_kwargs,
    ):
        observed["item_ids"] = item_ids.detach().clone()
        observed["mask"] = completion_mask.detach().clone()
        observed["advantages"] = token_advantages.detach().clone()
        return _loss_output(
            current_logps[completion_mask].mean(),
            tokens=int(completion_mask.sum()),
            actions=sum(
                len(set(row[row >= 0].tolist())) for row in item_ids
            ),
        )

    monkeypatch.setattr(answer_trainer_module, "rank_grpo_loss", controlled)
    result = trainer.train_query(TrainingExample("rank", frozenset({1}), 3))

    assert result.action_count == 12
    assert observed["mask"].shape == observed["item_ids"].shape
    assert torch.all(observed["item_ids"][~observed["mask"]] == -1)
    assert observed["mask"].any()
    assert torch.isfinite(observed["advantages"]).all()


def test_ppo_passes_state_boundary_and_nonfinite_poisoning(
    answer_trainer_module, monkeypatch
):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    reference = FakeBackend("reference", events, trainable=False)
    optimizer = RecordingSGD(actor.trainable_parameters(), events)
    with pytest.raises(ValueError, match="ppo_passes=1"):
        answer_trainer_module.AnswerRLTrainer(
            actor=actor,
            reference=reference,
            optimizer=optimizer,
            algorithm="grpo",
            ppo_passes=2,
        )

    trainer = answer_trainer_module.AnswerRLTrainer(
        actor=actor,
        reference=reference,
        optimizer=optimizer,
        algorithm="grpo",
        gradient_accumulation_steps=2,
    )

    def nonfinite(current_logps, *_args, **_kwargs):
        return _loss_output(current_logps[0, 0] * float("nan"), tokens=1, actions=1)

    monkeypatch.setattr(answer_trainer_module, "sequence_grpo_loss", nonfinite)
    with pytest.raises(FloatingPointError, match="non-finite"):
        trainer.train_query(TrainingExample("bad", frozenset({1}), 3))
    assert trainer.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        trainer.training_state_dict()


def test_training_state_schema_matches_maskpo(answer_trainer_module):
    events: list[tuple] = []
    actor = FakeBackend("actor", events, trainable=True)
    trainer = answer_trainer_module.AnswerRLTrainer(
        actor=actor,
        reference=FakeBackend("reference", events, trainable=False),
        optimizer=RecordingSGD(actor.trainable_parameters(), events),
        algorithm="grpo",
    )
    expected = {
        "version": 1,
        "rollout_steps": 0,
        "optimizer_steps": 0,
        "ppo_passes": 1,
    }
    assert trainer.training_state_dict() == expected
    trainer.load_training_state_dict(
        {"version": 1, "rollout_steps": 7, "optimizer_steps": 7, "ppo_passes": 1}
    )
    assert trainer.training_state_dict()["rollout_steps"] == 7


def test_loader_resumes_only_actor_and_honors_separate_device_maps(
    answer_trainer_module, monkeypatch
):
    tokenizer_calls: list[str] = []
    config_calls: list[str] = []
    peft_calls: list[tuple[str, bool, str]] = []
    model_calls: list[tuple[str, object]] = []

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, path, **_kwargs):
            tokenizer_calls.append(str(path))
            return TinyTokenizer()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            model_calls.append((str(path), kwargs.get("device_map")))
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

    trainer = answer_trainer_module.load_huggingface_answer_rl_trainer(
        model_name_or_path="base-model",
        adapter_path="answer-sft-reference",
        actor_adapter_path="rollout-150-adapter",
        algorithm="rank_grpo",
        dtype="float32",
        actor_device_map="actor-map",
        reference_device_map="reference-map",
        gradient_checkpointing=False,
    )

    assert tokenizer_calls == ["rollout-150-adapter"]
    assert config_calls == ["rollout-150-adapter", "answer-sft-reference"]
    assert peft_calls == [
        ("rollout-150-adapter", True, "rollout-150-adapter"),
        ("answer-sft-reference", False, "answer-sft-reference"),
    ]
    assert model_calls == [
        ("base-model", "actor-map"),
        ("base-model", "reference-map"),
    ]
    assert next(trainer.actor.trainable_parameters()).requires_grad
    assert not next(trainer.reference.model.parameters()).requires_grad
    assert trainer.actor.enable_thinking is False
    assert trainer.reference.enable_thinking is False
