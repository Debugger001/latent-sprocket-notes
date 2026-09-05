"""Sequence, position, rubric, and answer-item advantages for MaskPO."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .metrics import ideal_dcg


DEFAULT_NUM_SIBLINGS = 4
DEFAULT_NUM_RUBRICS = 4


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    """Population standard deviation, matching within-group reward scaling."""

    if len(values) <= 1:
        return 0.0
    mu = _mean(values)
    var = sum((value - mu) ** 2 for value in values) / len(values)
    return math.sqrt(var)


def _z_normalize(values: Sequence[float], eps: float = 1e-8) -> list[float]:
    if eps < 0:
        raise ValueError("eps must be non-negative")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cannot normalize non-finite values")
    mu = _mean(values)
    sigma = _std(values)
    if sigma <= eps:
        return [0.0 for _ in values]
    return [(value - mu) / (sigma + eps) for value in values]


def _group_indices(
    group_ids: Sequence[str], expected_group_size: int | None
) -> dict[str, list[int]]:
    if expected_group_size is not None and expected_group_size <= 0:
        raise ValueError("expected_group_size must be positive")
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        grouped[group_id].append(idx)
    if expected_group_size is not None:
        bad = {group_id: len(indices) for group_id, indices in grouped.items() if len(indices) != expected_group_size}
        if bad:
            details = ", ".join(f"{group_id!r}: {size}" for group_id, size in bad.items())
            raise ValueError(
                f"each prompt must have {expected_group_size} sibling rollouts; got {details}"
            )
    return grouped


def grpo_group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[str],
    *,
    expected_group_size: int | None = None,
    eps: float = 1e-8,
) -> list[float]:
    """Z-normalize sequence rewards within each prompt's sibling rollouts.

    The current MaskPO run uses four siblings; callers can set
    ``expected_group_size=4`` to enforce that invariant.  The optional check is
    disabled by default to keep this low-level utility useful for small tests
    and ablations.
    """

    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")
    grouped = _group_indices(group_ids, expected_group_size)

    out = [0.0] * len(rewards)
    for indices in grouped.values():
        normalized = _z_normalize([rewards[idx] for idx in indices], eps=eps)
        for local_idx, global_idx in enumerate(indices):
            out[global_idx] = normalized[local_idx]
    return out


def _discount(rank: int, *, metric: str, idcg: float = 1.0) -> float:
    if rank <= 0:
        return 0.0
    if metric in {"rr", "mrr"}:
        return 1.0 / rank
    if metric == "dcg":
        return 1.0 / math.log2(rank + 1)
    if metric == "ndcg":
        return (1.0 / math.log2(rank + 1)) / idcg if idcg > 0 else 0.0
    raise ValueError(f"unsupported metric: {metric}")


def _position_contributions(
    order: Sequence[int],
    positives: set[int],
    *,
    metric: str,
    slate_k: int | None,
) -> list[float]:
    """Return the lenient ranking contribution of every emitted position."""

    cutoff = len(order) if slate_k is None else slate_k
    if cutoff < 0:
        raise ValueError("slate_k must be non-negative")
    positive_ids = {item for item in positives if item > 0}
    idcg = ideal_dcg(len(positive_ids), cutoff)
    seen_positives: set[int] = set()
    rr_credited = False
    scores: list[float] = []
    for zero_pos, item in enumerate(order):
        rank = zero_pos + 1
        score = 0.0
        if (
            rank <= cutoff
            and item in positive_ids
            and item not in seen_positives
        ):
            seen_positives.add(item)
            if metric not in {"rr", "mrr"} or not rr_credited:
                score = _discount(rank, metric=metric, idcg=idcg)
                rr_credited = True
        scores.append(score)
    return scores


def rank_grpo_position_advantages(
    orders: Sequence[Sequence[int]],
    positives: Sequence[set[int]],
    group_ids: Sequence[str],
    *,
    metric: str = "ndcg",
    slate_k: int | None = None,
    expected_group_size: int | None = None,
    eps: float = 1e-8,
) -> list[list[float]]:
    """Position-normalized Rank-GRPO credit for answer integers.

    Local ranking contributions are z-normalized across sibling rollouts for
    the same query and emitted answer position.  Like the sequence scorer,
    malformed but parseable lists are not repaired: duplicates occupy slots,
    and only the first appearance of a positive id can earn credit.  Preserving
    the archived Rank-GRPO rule, a sibling that emits no integer at position
    ``p`` is absent from that position's normalization bucket rather than
    imputed as a zero.
    """

    if not (len(orders) == len(positives) == len(group_ids)):
        raise ValueError("orders, positives, and group_ids must have the same length")
    _group_indices(group_ids, expected_group_size)

    scores = [
        _position_contributions(order, pos, metric=metric, slate_k=slate_k)
        for order, pos in zip(orders, positives, strict=True)
    ]

    buckets: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
    for rollout_idx, row_scores in enumerate(scores):
        for pos_idx, score in enumerate(row_scores):
            buckets[(group_ids[rollout_idx], pos_idx)].append((rollout_idx, score))

    advantages = [[0.0 for _ in row_scores] for row_scores in scores]
    for (_, pos_idx), values in buckets.items():
        normalized = _z_normalize([score for _, score in values], eps=eps)
        for local_idx, (rollout_idx, _) in enumerate(values):
            advantages[rollout_idx][pos_idx] = normalized[local_idx]
    return advantages


def rms_normalize_rubric_deltas(
    deltas: Sequence[Sequence[float | None]],
    valid_mask: Sequence[Sequence[bool]] | None = None,
    *,
    expected_rollouts: int | None = None,
    expected_rubrics: int | None = None,
    eps: float = 1e-8,
) -> list[list[float | None]]:
    """RMS-normalize one query's signed rubric deltas without centering.

    ``None`` denotes an invalid/unparseable counterfactual and is excluded from
    the RMS denominator.  An explicit valid mask may additionally exclude
    entries.  Valid zero deltas remain in the denominator and receive exactly
    zero advantage.  If all valid deltas are zero, all valid outputs are zero;
    invalid outputs remain ``None``.
    """

    if eps < 0:
        raise ValueError("eps must be non-negative")
    if expected_rollouts is not None and len(deltas) != expected_rollouts:
        raise ValueError(
            f"expected {expected_rollouts} rollouts, received {len(deltas)}"
        )
    if not deltas:
        return []

    width = len(deltas[0])
    if any(len(row) != width for row in deltas):
        raise ValueError("rubric delta matrix must be rectangular")
    if expected_rubrics is not None and width != expected_rubrics:
        raise ValueError(f"expected {expected_rubrics} rubrics, received {width}")
    if valid_mask is not None:
        if len(valid_mask) != len(deltas) or any(
            len(mask_row) != width for mask_row in valid_mask
        ):
            raise ValueError("valid_mask must have the same shape as deltas")

    is_valid = [[False] * width for _ in deltas]
    valid_values: list[float] = []
    for row_idx, row in enumerate(deltas):
        for rubric_idx, value in enumerate(row):
            explicitly_valid = (
                valid_mask is None or bool(valid_mask[row_idx][rubric_idx])
            )
            if not explicitly_valid or value is None:
                if valid_mask is not None and explicitly_valid and value is None:
                    raise ValueError("a valid rubric delta cannot be None")
                continue
            if not math.isfinite(value):
                raise ValueError("a valid rubric delta must be finite")
            is_valid[row_idx][rubric_idx] = True
            valid_values.append(value)

    out: list[list[float | None]] = [
        [None for _ in range(width)] for _ in deltas
    ]
    if not valid_values:
        return out

    scale = math.sqrt(sum(value * value for value in valid_values) / len(valid_values))
    for row_idx, row in enumerate(deltas):
        for rubric_idx, value in enumerate(row):
            if is_valid[row_idx][rubric_idx]:
                assert value is not None
                out[row_idx][rubric_idx] = 0.0 if scale <= eps else value / scale
    return out


def rubric_delta_advantages(
    deltas: Sequence[Sequence[float | None]],
    valid_mask: Sequence[Sequence[bool]] | None = None,
    *,
    expected_rollouts: int = DEFAULT_NUM_SIBLINGS,
    expected_rubrics: int = DEFAULT_NUM_RUBRICS,
    eps: float = 1e-8,
) -> list[list[float | None]]:
    """MaskPO query-level normalization for the valid group-by-rubric matrix."""

    return rms_normalize_rubric_deltas(
        deltas,
        valid_mask,
        expected_rollouts=expected_rollouts,
        expected_rubrics=expected_rubrics,
        eps=eps,
    )


@dataclass(frozen=True)
class MaskPOConfig:
    """Latest answer-token Rank-MaskPO configuration."""

    metric: str = "ndcg"
    num_siblings: int = DEFAULT_NUM_SIBLINGS
    num_rubrics: int = DEFAULT_NUM_RUBRICS
    tau_mask: float = 0.05
    mask_clip: float = 2.0
    lambda_rank: float = 1.0
    lambda_mask: float = 0.5

    def __post_init__(self) -> None:
        if self.metric not in {"rr", "mrr", "dcg", "ndcg"}:
            raise ValueError(f"unsupported metric: {self.metric}")
        if self.num_siblings <= 0 or self.num_rubrics <= 0:
            raise ValueError("num_siblings and num_rubrics must be positive")
        if self.tau_mask <= 0:
            raise ValueError("tau_mask must be positive")
        if self.mask_clip < 0:
            raise ValueError("mask_clip must be non-negative")


def _rank_map(order: Sequence[int], slate_k: int) -> dict[int, int]:
    """Map each valid candidate id to its first *emitted* rank.

    Unlike deduplication followed by enumeration, this keeps the positional cost
    of preceding duplicates and invalid ids, matching lenient nDCG.
    """

    if slate_k < 0:
        raise ValueError("slate_k must be non-negative")
    ranks: dict[int, int] = {}
    for zero_pos, item in enumerate(order[:slate_k]):
        if 1 <= item <= slate_k and item not in ranks:
            ranks[item] = zero_pos + 1
    return ranks


def _metric_contributions_by_positive(
    order: Sequence[int],
    positives: set[int],
    slate_k: int,
    *,
    metric: str,
) -> tuple[dict[int, int], dict[int, float]]:
    ranks = _rank_map(order, slate_k)
    positive_ids = {item for item in positives if item > 0}
    idcg = ideal_dcg(len(positive_ids), slate_k)
    contributions: dict[int, float] = {item: 0.0 for item in positive_ids}

    if metric in {"rr", "mrr"}:
        present = [(rank, item) for item, rank in ranks.items() if item in positive_ids]
        if present:
            rank, item = min(present)
            contributions[item] = _discount(rank, metric=metric, idcg=idcg)
        return ranks, contributions

    for item in positive_ids:
        rank = ranks.get(item)
        if rank is not None:
            contributions[item] = _discount(rank, metric=metric, idcg=idcg)
    return ranks, contributions


def rank_shift_mask_residuals(
    original_order: Sequence[int],
    masked_order: Sequence[int],
    positives: set[int],
    slate_k: int,
    *,
    metric: str = "ndcg",
) -> tuple[float, dict[int, float]]:
    """Return ranking delta and per-original-item rank-shift residuals.

    The scalar delta is ``R_rank(original) - R_rank(masked)`` under the same
    lenient grading rule as the rollout reward.  If masking hurts a positive,
    its original answer item is reinforced.  If masking improves a positive,
    the original items in the slots it crossed receive the negative residual.
    Residuals for multiple positives add, and their sum conserves the scalar
    delta whenever at least one affected original answer token exists.
    """

    if metric not in {"rr", "mrr", "dcg", "ndcg"}:
        raise ValueError(f"unsupported metric: {metric}")
    if slate_k < 0:
        raise ValueError("slate_k must be non-negative")
    if not positives:
        return 0.0, {}

    original_ranks, original_contributions = _metric_contributions_by_positive(
        original_order, positives, slate_k, metric=metric
    )
    masked_ranks, masked_contributions = _metric_contributions_by_positive(
        masked_order, positives, slate_k, metric=metric
    )
    idcg = ideal_dcg(len({item for item in positives if item > 0}), slate_k)
    missing_rank = slate_k + 1

    residuals: dict[int, float] = defaultdict(float)
    total_delta = 0.0
    for target in sorted(original_contributions):
        component = original_contributions[target] - masked_contributions[target]
        total_delta += component
        if component == 0.0:
            continue

        if component > 0.0:
            # A positive's useful original contribution disappeared or shrank.
            # It necessarily has an original answer token to reinforce.
            residuals[target] += component
            continue

        original_rank = original_ranks.get(target, missing_rank)
        masked_rank = masked_ranks.get(target, missing_rank)
        crossed: list[tuple[int, float]] = []
        if masked_rank < original_rank:
            last_crossed = min(original_rank - 1, slate_k, len(original_order))
            for rank in range(max(masked_rank, 1), last_crossed + 1):
                item = original_order[rank - 1]
                weight = max(
                    _discount(rank, metric=metric, idcg=idcg)
                    - _discount(rank + 1, metric=metric, idcg=idcg),
                    0.0,
                )
                if weight > 0.0:
                    crossed.append((item, weight))

        weight_sum = sum(weight for _, weight in crossed)
        if weight_sum > 0.0:
            for item, weight in crossed:
                residuals[item] += component * weight / weight_sum
        elif target in original_ranks:
            # This branch is mainly relevant to RR when which positive is first
            # changes without a simple monotone crossing for this target.
            residuals[target] += component

    return total_delta, dict(residuals)


def aggregate_mask_advantages(
    residuals_by_mask: Sequence[Mapping[int, float]],
    *,
    tau_mask: float = 0.05,
    mask_clip: float = 2.0,
) -> dict[int, float]:
    """Aggregate only each item's existing residuals, then fixed-scale clip.

    Missing keys are excluded rather than silently imputed as zeros.  Explicit
    zero residuals, when supplied, do participate in the item's mean.
    """

    if tau_mask <= 0:
        raise ValueError("tau_mask must be positive")
    if mask_clip < 0:
        raise ValueError("mask_clip must be non-negative")

    buckets: dict[int, list[float]] = defaultdict(list)
    for residuals in residuals_by_mask:
        for item, value in residuals.items():
            if not math.isfinite(value):
                raise ValueError("mask residuals must be finite")
            buckets[item].append(value)

    out: dict[int, float] = {}
    for item, values in buckets.items():
        scaled = _mean(values) / tau_mask
        out[item] = max(-mask_clip, min(mask_clip, scaled))
    return out


def combine_answer_advantages(
    order: Sequence[int],
    rank_advantages: Sequence[float],
    mask_advantages_by_item: Mapping[int, float],
    *,
    config: MaskPOConfig = MaskPOConfig(),
) -> list[float]:
    """Compute ``A_rank(i,p) + 0.5 A_mask(i,c[i,p])`` by default."""

    if len(order) != len(rank_advantages):
        raise ValueError("order and rank_advantages must have the same length")
    return [
        config.lambda_rank * rank_advantage
        + config.lambda_mask * mask_advantages_by_item.get(item, 0.0)
        for item, rank_advantage in zip(order, rank_advantages, strict=True)
    ]
