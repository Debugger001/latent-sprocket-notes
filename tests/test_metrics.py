import math

import pytest

from openbench_rerank_rl.metrics import (
    complete_permutation,
    dcg_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    slate_auc,
)


def test_complete_permutation_dedupes_and_appends_missing():
    assert complete_permutation([3, 2, 2, 99, 1], 4) == [3, 2, 1, 4]


def test_lenient_dcg_credits_only_first_positive_appearance():
    assert dcg_at_k([1, 1, 2], {1, 2}, 3) == pytest.approx(1.0 + 0.5)
    assert dcg_at_k([2, 2, 2], {2}, 3) == 1.0


def test_lenient_ndcg_does_not_compress_duplicate_or_invalid_slots():
    idcg_two = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k([1, 1, 2], {1, 2}, 3) == pytest.approx(1.5 / idcg_two)
    assert ndcg_at_k([999, 1], {1}, 2) == pytest.approx(1.0 / math.log2(3))


def test_lenient_ndcg_multi_positive_is_bounded_by_one_with_duplicates():
    score = ndcg_at_k([2, 2, 1, 3, 3], {1, 2, 3}, 5)
    assert 0.0 < score <= 1.0
    assert score < ndcg_at_k([2, 1, 3], {1, 2, 3}, 5)


def test_ndcg_honors_cutoff_before_a_late_positive():
    assert ndcg_at_k([2, 2, 1], {1}, 2) == 0.0
    assert ndcg_at_k([2, 2, 1], {1}, 3) == 0.5


def test_mrr_uses_first_positive_only():
    assert mrr_at_k([4, 2, 1, 3], {1, 3}, 4) == 1 / 3


def test_recall_dedupes_positive_ids():
    assert recall_at_k([1, 1, 2], {1, 2}, 3) == 1.0


def test_metrics_are_zero_without_positive_labels():
    assert ndcg_at_k([1, 2], set(), 2) == 0.0
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert mrr_at_k([1, 2], set(), 2) == 0.0


def test_slate_auc_counts_omitted_pairs_as_ties():
    assert slate_auc([], {1}, 3) == 0.5
    assert slate_auc([1], {1}, 3) == 1.0
    assert slate_auc([2, 3], {1}, 3) == 0.0
