"""Offline scoring for generated MIND reranking completions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .metrics import mrr_at_k, ndcg_at_k, recall_at_k, slate_auc
from .parsers import parse_answer
from .rewards import evaluate_format, ranking_reward


@dataclass(frozen=True)
class PredictionEvaluation:
    parseable: bool
    parsed_order: tuple[int, ...]
    rank_reward: float
    format_reward: float
    exact_permutation: bool
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_k: float
    recall_at_5: float
    recall_at_10: float
    recall_at_k: float
    mrr_at_k: float
    auc: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_prediction(
    completion: str,
    *,
    positives: set[int] | frozenset[int],
    slate_k: int,
) -> PredictionEvaluation:
    """Score one completion with the same lenient parser used in training."""

    if slate_k < 1:
        raise ValueError("slate_k must be positive")
    parsed = parse_answer(completion)
    order = tuple(parsed) if parsed is not None else ()
    positive_set = set(positives)
    checks = evaluate_format(completion, slate_k)
    cutoff_5 = min(5, slate_k)
    cutoff_10 = min(10, slate_k)
    return PredictionEvaluation(
        parseable=parsed is not None,
        parsed_order=order,
        rank_reward=ranking_reward(order, positive_set, slate_k),
        format_reward=checks.reward(),
        exact_permutation=checks.exact_permutation,
        ndcg_at_5=ndcg_at_k(order, positive_set, cutoff_5),
        ndcg_at_10=ndcg_at_k(order, positive_set, cutoff_10),
        ndcg_at_k=ndcg_at_k(order, positive_set, slate_k),
        recall_at_5=recall_at_k(order, positive_set, cutoff_5),
        recall_at_10=recall_at_k(order, positive_set, cutoff_10),
        recall_at_k=recall_at_k(order, positive_set, slate_k),
        mrr_at_k=mrr_at_k(order, positive_set, slate_k),
        auc=slate_auc(order, positive_set, slate_k),
    )


def aggregate_evaluations(
    evaluations: Iterable[PredictionEvaluation],
) -> dict[str, float | int]:
    """Macro-average a sequence of row-level evaluations."""

    rows = list(evaluations)
    if not rows:
        return {"rows": 0}
    numeric_fields = (
        "rank_reward",
        "format_reward",
        "ndcg_at_5",
        "ndcg_at_10",
        "ndcg_at_k",
        "recall_at_5",
        "recall_at_10",
        "recall_at_k",
        "mrr_at_k",
        "auc",
    )
    result: dict[str, float | int] = {
        "rows": len(rows),
        "parse_rate": sum(row.parseable for row in rows) / len(rows),
        "exact_permutation_rate": sum(row.exact_permutation for row in rows)
        / len(rows),
    }
    for field in numeric_fields:
        result[field] = sum(float(getattr(row, field)) for row in rows) / len(rows)
    return result
