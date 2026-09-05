from __future__ import annotations

from collections.abc import Sequence

import pytest

torch = pytest.importorskip("torch")

from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.trainer import (
    GeneratedCompletion,
    LogProbBatch,
    MaskPOTrainer,
    SamplingConfig,
    TrainingExample,
)


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
