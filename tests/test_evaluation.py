import pytest

from openbench_rerank_rl.evaluation import (
    aggregate_evaluations,
    evaluate_prediction,
)
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS


def completion(order: str) -> str:
    bodies = "\n".join(f"{header} evidence" for header in DEFAULT_RUBRIC_HEADERS)
    return (
        f"<think>\n{bodies}\n**Synthesis:** combine\n</think>\n"
        f"<answer>{order}</answer>"
    )


def test_offline_evaluation_uses_training_parser_and_first_positive_credit():
    result = evaluate_prediction(
        completion("[1, 1, 2]"),
        positives={1, 2},
        slate_k=3,
    )
    assert result.parseable
    assert not result.exact_permutation
    assert 0.0 < result.ndcg_at_k <= 1.0
    assert result.rank_reward == pytest.approx(result.ndcg_at_k)


def test_offline_aggregate_counts_unparseable_rows_as_zero():
    good = evaluate_prediction(completion("[1, 2]"), positives={1}, slate_k=2)
    bad = evaluate_prediction("no list", positives={1}, slate_k=2)
    aggregate = aggregate_evaluations([good, bad])
    assert aggregate["rows"] == 2
    assert aggregate["parse_rate"] == pytest.approx(0.5)
    assert aggregate["ndcg_at_k"] == pytest.approx(0.5)
