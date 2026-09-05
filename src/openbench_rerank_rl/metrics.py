"""Ranking metrics for fixed-slate recommendation reranking.

Items are represented by 1-based candidate indices. Positives may contain one
or multiple clicked items. Unparseable outputs should be handled by the caller;
these metric functions assume they receive a list of emitted candidate indices.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def dedupe_order(order: Iterable[int], *, k: int | None = None) -> list[int]:
    """Keep the first occurrence of each positive integer candidate id."""

    seen: set[int] = set()
    out: list[int] = []
    for item in order:
        if item <= 0 or item in seen:
            continue
        if k is not None and item > k:
            continue
        seen.add(item)
        out.append(item)
    return out


def complete_permutation(order: Iterable[int], k: int) -> list[int]:
    """Dedupe an emitted order and append omitted candidates in slate order."""

    out = dedupe_order(order, k=k)
    present = set(out)
    out.extend(i for i in range(1, k + 1) if i not in present)
    return out


def _positive_set(positives: Iterable[int]) -> set[int]:
    return {p for p in positives if p > 0}


def dcg_at_k(order: Sequence[int], positives: Iterable[int], k: int | None = None) -> float:
    """Lenient binary-relevance DCG for the emitted order.

    The emitted list is deliberately *not* repaired before scoring.  Invalid
    candidate ids and duplicates therefore still consume a rank position.  A
    positive candidate contributes at most once, at its first appearance.  This
    is the grading rule used for both policy rollouts and parseable
    counterfactual suffixes.
    """

    pos = _positive_set(positives)
    if not pos:
        return 0.0
    limit = len(order) if k is None else min(k, len(order))
    total = 0.0
    credited: set[int] = set()
    for zero_pos, item in enumerate(order[:limit]):
        if item in pos and item not in credited:
            credited.add(item)
            rank = zero_pos + 1
            total += 1.0 / math.log2(rank + 1)
    return total


def ideal_dcg(num_positives: int, k: int) -> float:
    """Best possible binary DCG for a row with ``num_positives`` positives."""

    return sum(1.0 / math.log2(rank + 1) for rank in range(1, min(num_positives, k) + 1))


def ndcg_at_k(order: Sequence[int], positives: Iterable[int], k: int | None = None) -> float:
    """Lenient binary nDCG over the emitted order.

    Only a positive id's first appearance is relevant; the list need not be a
    complete permutation.  See :func:`dcg_at_k` for the treatment of malformed
    ids and duplicates.
    """

    pos = _positive_set(positives)
    if not pos:
        return 0.0
    cutoff = len(order) if k is None else k
    denom = ideal_dcg(len(pos), cutoff)
    if denom <= 0:
        return 0.0
    return dcg_at_k(order, pos, cutoff) / denom


def recall_at_k(order: Sequence[int], positives: Iterable[int], k: int | None = None) -> float:
    """Fraction of positives that appear in the top-k emitted order."""

    pos = _positive_set(positives)
    if not pos:
        return 0.0
    cutoff = len(order) if k is None else k
    found = sum(1 for item in dedupe_order(order[:cutoff]) if item in pos)
    return found / len(pos)


def mrr_at_k(order: Sequence[int], positives: Iterable[int], k: int | None = None) -> float:
    """Reciprocal rank of the first positive item, not a sum over positives."""

    pos = _positive_set(positives)
    if not pos:
        return 0.0
    cutoff = len(order) if k is None else min(k, len(order))
    for zero_pos, item in enumerate(order[:cutoff]):
        if item in pos:
            return 1.0 / (zero_pos + 1)
    return 0.0


def slate_auc(order: Sequence[int], positives: Iterable[int], slate_k: int) -> float:
    """Pairwise AUC within a fixed slate.

    Every positive-negative candidate pair receives:

    - 1.0 if the positive is ranked above the negative;
    - 0.5 if both are omitted from the emitted list;
    - 0.0 otherwise.

    Rows with no positives or no negatives return 0.0 because they have no
    pairwise ordering signal.
    """

    pos = _positive_set(positives)
    candidates = set(range(1, slate_k + 1))
    neg = candidates - pos
    if not pos or not neg:
        return 0.0

    clean = dedupe_order(order, k=slate_k)
    rank = {item: i + 1 for i, item in enumerate(clean)}
    omitted_rank = slate_k + 1
    score = 0.0
    count = 0
    for p in pos:
        rp = rank.get(p, omitted_rank)
        for n in neg:
            rn = rank.get(n, omitted_rank)
            if rp < rn:
                score += 1.0
            elif rp == rn:
                score += 0.5
            count += 1
    return score / count
