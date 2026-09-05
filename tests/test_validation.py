from __future__ import annotations

from collections.abc import Sequence

import pytest

from openbench_rerank_rl.evaluation import aggregate_evaluations, evaluate_prediction
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.rewards import evaluate_format
from openbench_rerank_rl.trainer import (
    GeneratedCompletion,
    SamplingConfig,
    TrainingExample,
)
from openbench_rerank_rl.validation import (
    fixed_validation_examples,
    run_greedy_validation,
    validation_fingerprint,
)


def _completion(order: str) -> str:
    bodies = "\n\n".join(f"{header} evidence" for header in DEFAULT_RUBRIC_HEADERS)
    return (
        f"<think>\n{bodies}\n\n**Synthesis:** combined\n</think>\n"
        f"<answer>\n{order}\n</answer>"
    )


def _example(index: int, *, k: int = 2) -> TrainingExample:
    return TrainingExample(
        prompt=f"prompt-{index}",
        positives=frozenset({1}),
        slate_k=k,
        example_id=f"held-out-{index}",
    )


class ValidationBackend:
    def __init__(self, texts: Sequence[str]) -> None:
        self.texts = iter(texts)
        self.batch_sizes: list[int] = []
        self.sampling_configs: list[SamplingConfig] = []

    def render_user_prompt(self, prompt: str) -> str:
        return f"CHAT:{prompt}"

    def generate(self, model_prefixes, config):
        self.batch_sizes.append(len(model_prefixes))
        self.sampling_configs.append(config)
        return [
            GeneratedCompletion(
                text=text,
                prompt_token_ids=(1,),
                token_ids=(2,),
                token_offsets=((0, len(text)),),
            )
            for text in (next(self.texts) for _ in model_prefixes)
        ]

    def token_logps(self, samples, *, requires_grad):  # pragma: no cover - protocol
        raise AssertionError("validation must not compute token log probabilities")

    def trainable_parameters(self):  # pragma: no cover - protocol
        return ()


def test_fixed_validation_examples_materializes_exact_stable_file():
    examples = [_example(index) for index in range(3)]

    selected = fixed_validation_examples(examples, expected_rows=3)

    assert selected == tuple(examples)
    assert validation_fingerprint(selected) == validation_fingerprint(selected)
    assert validation_fingerprint(selected) != validation_fingerprint(
        tuple(reversed(selected))
    )


def test_fixed_validation_examples_rejects_short_or_large_slate_sets():
    with pytest.raises(ValueError, match="requires exactly 3 rows"):
        fixed_validation_examples([_example(0)], expected_rows=3)
    with pytest.raises(ValueError, match="requires exactly 3 rows, found 4"):
        fixed_validation_examples([_example(index) for index in range(4)], expected_rows=3)
    with pytest.raises(ValueError, match=r"K=21.*K<=20"):
        fixed_validation_examples([_example(0, k=21)], expected_rows=1)


def test_greedy_validation_generates_once_per_prompt_and_reuses_shared_grader():
    examples = tuple(_example(index) for index in range(3))
    completions = (_completion("[1, 2]"), "malformed [2, 1]", "no integer list")
    actor = ValidationBackend(completions)

    result = run_greedy_validation(
        actor,
        examples,
        max_new_tokens=123,
        generation_batch_size=2,
    )

    expected = aggregate_evaluations(
        evaluate_prediction(text, positives={1}, slate_k=2) for text in completions
    )
    assert actor.batch_sizes == [2, 1]
    assert all(not config.do_sample for config in actor.sampling_configs)
    assert all(config.max_new_tokens == 123 for config in actor.sampling_configs)
    assert result.rows == 3
    assert result.ndcg == pytest.approx(expected["ndcg_at_k"])
    assert result.format_reward == pytest.approx(expected["format_reward"])
    assert result.format_rate == pytest.approx(expected["format_reward"] / 0.1)
    checks = [evaluate_format(text, 2) for text in completions]
    assert result.envelope_rate == pytest.approx(
        sum(check.envelope for check in checks) / 3
    )
    assert result.rubric_header_rates == pytest.approx(
        tuple(
            sum(check.rubric_headers[index] for check in checks) / 3
            for index in range(4)
        )
    )
    assert result.synthesis_rate == pytest.approx(
        sum(check.synthesis for check in checks) / 3
    )
    assert result.parse_rate == pytest.approx(expected["parse_rate"])
    assert result.valid_unique_ids_rate == pytest.approx(
        sum(check.valid_unique_ids for check in checks) / 3
    )
    assert result.exact_permutation_rate == pytest.approx(
        expected["exact_permutation_rate"]
    )
