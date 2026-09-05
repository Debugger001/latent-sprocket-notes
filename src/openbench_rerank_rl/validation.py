"""Deterministic online validation for MaskPO training runs.

Validation deliberately reuses the same lenient parser, ranking reward, and
nine-check format grader as training.  It generates one greedy completion for
each member of a fixed held-out set; no counterfactual probes are generated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .evaluation import (
    PredictionEvaluation,
    aggregate_evaluations,
    evaluate_prediction,
)
from .rewards import evaluate_format
from .trainer import PolicyBackend, SamplingConfig, TrainingExample


@dataclass(frozen=True)
class ValidationResult:
    """Aggregate metrics for one pass over a fixed validation set."""

    rows: int
    dataset_fingerprint: str
    ndcg: float
    format_reward: float
    format_rate: float
    envelope_rate: float
    rubric_header_rates: tuple[float, float, float, float]
    synthesis_rate: float
    parse_rate: float
    valid_unique_ids_rate: float
    exact_permutation_rate: float

    def as_dict(self) -> dict[str, float | int | str]:
        values: dict[str, float | int | str] = {
            "rows": self.rows,
            "dataset_fingerprint": self.dataset_fingerprint,
            "ndcg": self.ndcg,
            "format_reward": self.format_reward,
            "format_rate": self.format_rate,
            "format_envelope_rate": self.envelope_rate,
            "format_synthesis_rate": self.synthesis_rate,
            "parse_rate": self.parse_rate,
            "format_valid_unique_ids_rate": self.valid_unique_ids_rate,
            "exact_permutation_rate": self.exact_permutation_rate,
        }
        for index, rate in enumerate(self.rubric_header_rates, start=1):
            values[f"format_rubric_header_{index}_rate"] = rate
        return values


def fixed_validation_examples(
    examples: Iterable[TrainingExample],
    *,
    expected_rows: int = 200,
    max_slate_size: int = 20,
) -> tuple[TrainingExample, ...]:
    """Materialize an exactly ``expected_rows`` held-out file.

    Call this once before training and reuse the returned tuple for every
    validation pass.  The input file should already be a deterministic,
    held-out materialization; this function never resamples it.
    """

    if expected_rows <= 0:
        raise ValueError("expected_rows must be positive")
    if max_slate_size <= 0:
        raise ValueError("max_slate_size must be positive")

    selected: list[TrainingExample] = []
    for example in examples:
        if example.slate_k > max_slate_size:
            raise ValueError(
                f"validation example {example.example_id!r} has K={example.slate_k}; "
                f"expected K<={max_slate_size}"
            )
        selected.append(example)
    if len(selected) != expected_rows:
        raise ValueError(
            f"validation requires exactly {expected_rows} rows, found {len(selected)}"
        )
    return tuple(selected)


def validation_fingerprint(examples: Sequence[TrainingExample]) -> str:
    """Hash validation inputs and labels without exposing them to telemetry."""

    digest = hashlib.sha256()
    for example in examples:
        material = json.dumps(
            {
                "id": example.example_id,
                "prompt": example.prompt,
                "positive_indices": sorted(example.positives),
                "k": example.slate_k,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(material.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_greedy_validation(
    actor: PolicyBackend,
    examples: Sequence[TrainingExample],
    *,
    max_new_tokens: int = 2048,
    generation_batch_size: int = 4,
) -> ValidationResult:
    """Generate and grade one deterministic completion per validation prompt."""

    if not examples:
        raise ValueError("validation examples must not be empty")
    if generation_batch_size <= 0:
        raise ValueError("generation_batch_size must be positive")

    sampling = SamplingConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        original_batch_size=generation_batch_size,
        counterfactual_batch_size=generation_batch_size,
    )
    evaluations: list[PredictionEvaluation] = []
    format_checks = []
    for start in range(0, len(examples), generation_batch_size):
        batch = examples[start : start + generation_batch_size]
        prefixes = [actor.render_user_prompt(example.prompt) for example in batch]
        completions = tuple(actor.generate(prefixes, sampling))
        if len(completions) != len(batch):
            raise RuntimeError(
                "policy backend returned a different number of validation "
                "completions than prompts"
            )
        for example, completion in zip(batch, completions, strict=True):
            evaluations.append(
                evaluate_prediction(
                    completion.text,
                    positives=example.positives,
                    slate_k=example.slate_k,
                )
            )
            format_checks.append(evaluate_format(completion.text, example.slate_k))

    aggregate = aggregate_evaluations(evaluations)
    mean_format_reward = float(aggregate["format_reward"])
    row_count = len(format_checks)

    def mean_check(values: Iterable[bool]) -> float:
        return sum(values) / row_count

    return ValidationResult(
        rows=len(examples),
        dataset_fingerprint=validation_fingerprint(examples),
        ndcg=float(aggregate["ndcg_at_k"]),
        format_reward=mean_format_reward,
        # The canonical reward is 0.1 times the mean of nine binary checks.
        # Dividing by 0.1 exposes the more intuitive [0, 1] check-pass rate.
        format_rate=mean_format_reward / 0.1,
        envelope_rate=mean_check(check.envelope for check in format_checks),
        rubric_header_rates=tuple(
            mean_check(check.rubric_headers[index] for check in format_checks)
            for index in range(4)
        ),  # type: ignore[arg-type]
        synthesis_rate=mean_check(check.synthesis for check in format_checks),
        parse_rate=float(aggregate["parse_rate"]),
        valid_unique_ids_rate=mean_check(
            check.valid_unique_ids for check in format_checks
        ),
        exact_permutation_rate=float(aggregate["exact_permutation_rate"]),
    )
