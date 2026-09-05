"""Pure, model-independent assembly of one MaskPO query group."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .advantages import (
    MaskPOConfig,
    aggregate_mask_advantages,
    combine_answer_advantages,
    grpo_group_advantages,
    rank_grpo_position_advantages,
    rank_shift_mask_residuals,
    rubric_delta_advantages,
)
from .masking import CounterfactualPrefix, build_counterfactual_prefixes
from .parsers import parse_answer, parse_reasoning_structure
from .rewards import FormatCheckResult, evaluate_format, format_reward, ranking_reward


@dataclass(frozen=True)
class ProbeRequest:
    """One suffix-generation request produced from an original rollout."""

    rollout_index: int
    rubric_index: int
    prefix: CounterfactualPrefix


@dataclass(frozen=True)
class ProbeCredit:
    """Ranking-only evaluation of one counterfactual probe."""

    completion: str | None
    order: tuple[int, ...] | None
    rank_reward: float | None
    delta: float | None
    residuals: Mapping[int, float]

    @property
    def valid(self) -> bool:
        return self.order is not None


@dataclass(frozen=True)
class RolloutCredit:
    """All scalar, section, and answer-item credit for one original."""

    completion: str
    order: tuple[int, ...]
    answer_parseable: bool
    rank_reward: float
    format_checks: FormatCheckResult
    format_reward: float
    sequence_reward: float
    sequence_advantage: float
    rubric_advantages: tuple[float | None, ...]
    rank_advantages: tuple[float, ...]
    mask_advantages_by_item: Mapping[int, float]
    answer_advantages: tuple[float, ...]
    probes: tuple[ProbeCredit, ...]


@dataclass(frozen=True)
class QueryCredit:
    """Credit-assignment result for one six-sibling query group."""

    positives: frozenset[int]
    slate_k: int
    rollouts: tuple[RolloutCredit, ...]
    rubric_rms_scale: float
    valid_probe_count: int


def plan_counterfactual_probes(
    original_completions: Sequence[str],
) -> tuple[ProbeRequest, ...]:
    """Create four probes for every semantically parseable original.

    Malformed originals remain eligible for sequence-level training but cannot
    be safely masked, so they simply contribute no probe requests.
    """

    requests: list[ProbeRequest] = []
    for rollout_index, completion in enumerate(original_completions):
        try:
            prefixes = build_counterfactual_prefixes(completion)
        except ValueError:
            continue
        requests.extend(
            ProbeRequest(
                rollout_index=rollout_index,
                rubric_index=prefix.rubric_index,
                prefix=prefix,
            )
            for prefix in prefixes
        )
    return tuple(requests)


def _rubric_rms_scale(deltas: Sequence[Sequence[float | None]]) -> float:
    values = [value for row in deltas for value in row if value is not None]
    if not values:
        return 0.0
    return (sum(value * value for value in values) / len(values)) ** 0.5


def score_maskpo_group(
    original_completions: Sequence[str],
    counterfactual_completions: Sequence[Sequence[str | None]],
    *,
    positives: set[int] | frozenset[int],
    slate_k: int,
    config: MaskPOConfig = MaskPOConfig(),
    normalization_eps: float = 1e-8,
) -> QueryCredit:
    """Compute latest MaskPO credit for one frozen-policy sample group.

    Both originals and probes use :func:`parse_answer`.  Any parsed integer
    list receives the same lenient nDCG scorer; duplicates and invalid IDs keep
    their emitted positions, while only a positive's first occurrence earns
    credit.  Probe format never enters the counterfactual delta.
    """

    if config.metric != "ndcg":
        raise ValueError("the latest MIND MaskPO pipeline uses nDCG ranking reward")
    if slate_k < 1:
        raise ValueError("slate_k must be positive")
    if len(original_completions) != config.num_siblings:
        raise ValueError(
            f"expected {config.num_siblings} original rollouts, "
            f"received {len(original_completions)}"
        )
    if len(counterfactual_completions) != len(original_completions):
        raise ValueError("counterfactual rows must match original rollouts")
    if any(len(row) != config.num_rubrics for row in counterfactual_completions):
        raise ValueError(
            f"each rollout requires {config.num_rubrics} counterfactual slots"
        )

    for rollout_index, (original, probes) in enumerate(
        zip(original_completions, counterfactual_completions, strict=True)
    ):
        try:
            parse_reasoning_structure(original)
        except ValueError:
            if any(probe is not None for probe in probes):
                raise ValueError(
                    f"rollout {rollout_index} is not safely maskable, so all of "
                    "its counterfactual slots must be None"
                ) from None

    positive_set = set(positives)
    parsed_originals = [parse_answer(text) for text in original_completions]
    orders = [tuple(order) if order is not None else () for order in parsed_originals]
    rank_rewards = [
        ranking_reward(order, positive_set, slate_k) for order in orders
    ]
    format_checks = [
        evaluate_format(completion, slate_k) for completion in original_completions
    ]
    format_rewards = [
        format_reward(completion, slate_k) for completion in original_completions
    ]
    sequence_rewards = [
        rank_reward + dense_format
        for rank_reward, dense_format in zip(rank_rewards, format_rewards, strict=True)
    ]
    group_ids = ["query"] * len(orders)
    sequence_advantages = grpo_group_advantages(
        sequence_rewards,
        group_ids,
        expected_group_size=config.num_siblings,
        eps=normalization_eps,
    )
    rank_advantages = rank_grpo_position_advantages(
        orders,
        [positive_set] * len(orders),
        group_ids,
        metric=config.metric,
        slate_k=slate_k,
        expected_group_size=config.num_siblings,
        eps=normalization_eps,
    )

    delta_matrix: list[list[float | None]] = []
    probe_rows: list[list[ProbeCredit]] = []
    residual_rows: list[list[Mapping[int, float]]] = []
    for original_order, original_rank_reward, probes in zip(
        orders, rank_rewards, counterfactual_completions, strict=True
    ):
        row_deltas: list[float | None] = []
        row_credits: list[ProbeCredit] = []
        row_residuals: list[Mapping[int, float]] = []
        for completion in probes:
            parsed = parse_answer(completion) if completion is not None else None
            if parsed is None:
                row_deltas.append(None)
                row_credits.append(
                    ProbeCredit(
                        completion=completion,
                        order=None,
                        rank_reward=None,
                        delta=None,
                        residuals={},
                    )
                )
                continue

            masked_order = tuple(parsed)
            masked_rank_reward = ranking_reward(
                masked_order, positive_set, slate_k
            )
            residual_delta, residuals = rank_shift_mask_residuals(
                original_order,
                masked_order,
                positive_set,
                slate_k,
                metric=config.metric,
            )
            delta = original_rank_reward - masked_rank_reward
            if not math.isclose(delta, residual_delta, rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError(
                    "rank-shift residual decomposition disagrees with the shared "
                    f"ranking grader: {residual_delta} != {delta}"
                )
            row_deltas.append(delta)
            row_residuals.append(residuals)
            row_credits.append(
                ProbeCredit(
                    completion=completion,
                    order=masked_order,
                    rank_reward=masked_rank_reward,
                    delta=delta,
                    residuals=residuals,
                )
            )
        delta_matrix.append(row_deltas)
        probe_rows.append(row_credits)
        residual_rows.append(row_residuals)

    normalized_rubrics = rubric_delta_advantages(
        delta_matrix,
        expected_rollouts=config.num_siblings,
        expected_rubrics=config.num_rubrics,
        eps=normalization_eps,
    )
    mask_advantages = [
        aggregate_mask_advantages(
            residuals,
            tau_mask=config.tau_mask,
            mask_clip=config.mask_clip,
        )
        for residuals in residual_rows
    ]
    answer_advantages = [
        combine_answer_advantages(order, rank_advantage, mask_advantage, config=config)
        for order, rank_advantage, mask_advantage in zip(
            orders, rank_advantages, mask_advantages, strict=True
        )
    ]

    rollout_credits = tuple(
        RolloutCredit(
            completion=completion,
            order=order,
            answer_parseable=parsed is not None,
            rank_reward=rank_reward,
            format_checks=checks,
            format_reward=dense_format,
            sequence_reward=sequence_reward,
            sequence_advantage=sequence_advantage,
            rubric_advantages=tuple(rubric_advantage),
            rank_advantages=tuple(rank_advantage),
            mask_advantages_by_item=mask_advantage,
            answer_advantages=tuple(answer_advantage),
            probes=tuple(probes),
        )
        for (
            completion,
            order,
            parsed,
            rank_reward,
            checks,
            dense_format,
            sequence_reward,
            sequence_advantage,
            rubric_advantage,
            rank_advantage,
            mask_advantage,
            answer_advantage,
            probes,
        ) in zip(
            original_completions,
            orders,
            parsed_originals,
            rank_rewards,
            format_checks,
            format_rewards,
            sequence_rewards,
            sequence_advantages,
            normalized_rubrics,
            rank_advantages,
            mask_advantages,
            answer_advantages,
            probe_rows,
            strict=True,
        )
    )
    return QueryCredit(
        positives=frozenset(positive_set),
        slate_k=slate_k,
        rollouts=rollout_credits,
        rubric_rms_scale=_rubric_rms_scale(delta_matrix),
        valid_probe_count=sum(
            value is not None for row in delta_matrix for value in row
        ),
    )
