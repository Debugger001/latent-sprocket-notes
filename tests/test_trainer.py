from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from openbench_rerank_rl.losses import BNPOLossOutput
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.trainer import (
    GeneratedCompletion,
    HuggingFacePolicyBackend,
    LogProbBatch,
    MaskPOTrainer,
    SamplingConfig,
    TrainingExample,
    _targets_with_serialized_lora_weights,
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
        max_grad_norm=None,
    )
    result = trainer.train_query(
        TrainingExample("prompt", frozenset({1}), 3)
    )
    assert not result.optimizer_stepped
    assert trainer.pending_micro_steps == 1
    assert trainer.flush_gradients()
    assert trainer.optimizer_steps == 1
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
        max_grad_norm=None,
    )
    monkeypatch.setattr(
        "openbench_rerank_rl.trainer.tokenwise_bnpo_loss",
        _controlled_mean_loss([(11, 1.75)]),
    )

    result = trainer.train_query(TrainingExample("prompt", frozenset({1}), 3))

    assert result.optimizer_stepped
    assert actor.parameter.item() == pytest.approx(-0.0175)


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
