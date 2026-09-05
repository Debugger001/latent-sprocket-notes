import math
import random

import pytest

from openbench_rerank_rl.advantages import (
    MaskPOConfig,
    aggregate_mask_advantages,
    combine_answer_advantages,
    grpo_group_advantages,
    rank_grpo_position_advantages,
    rank_shift_mask_residuals,
    rms_normalize_rubric_deltas,
    rubric_delta_advantages,
)
from openbench_rerank_rl.metrics import ndcg_at_k


def test_sequence_advantages_z_normalize_four_siblings():
    advantages = grpo_group_advantages(
        [0.0, 1.0, 2.0, 3.0],
        ["q"] * 4,
        expected_group_size=4,
    )
    assert sum(advantages) == pytest.approx(0.0, abs=1e-12)
    assert sum(value * value for value in advantages) / 4 == pytest.approx(1.0)
    assert advantages == sorted(advantages)


def test_sequence_advantages_are_query_local_and_zero_safe():
    advantages = grpo_group_advantages(
        [1.0] * 4 + [0.0, 1.0, 2.0, 3.0],
        ["flat"] * 4 + ["varied"] * 4,
        expected_group_size=4,
    )
    assert advantages[:4] == [0.0] * 4
    assert sum(advantages[4:]) == pytest.approx(0.0, abs=1e-12)


def test_sequence_advantages_can_enforce_four_siblings():
    with pytest.raises(ValueError, match="four|4|sibling"):
        grpo_group_advantages([1.0, 2.0], ["q", "q"], expected_group_size=4)


def test_rank_grpo_normalizes_each_answer_position_across_siblings():
    orders = [
        [1, 2, 3],
        [2, 1, 3],
        [2, 3, 1],
        [3, 2, 1],
    ]
    advantages = rank_grpo_position_advantages(
        orders,
        [{1}] * 4,
        ["q"] * 4,
        slate_k=3,
        expected_group_size=4,
    )
    assert advantages[0][0] == pytest.approx(math.sqrt(3))
    assert all(row[0] == pytest.approx(-1 / math.sqrt(3)) for row in advantages[1:])
    assert advantages[1][1] > 0.0
    assert advantages[0][1] < 0.0


def test_rank_grpo_duplicate_positive_does_not_get_second_credit():
    orders = [
        [1, 1, 2],
        [2, 1, 3],
        [2, 3, 1],
        [3, 2, 1],
    ]
    advantages = rank_grpo_position_advantages(
        orders,
        [{1}] * 4,
        ["q"] * 4,
        slate_k=3,
        expected_group_size=4,
    )
    # Rollout zero's repeated 1 has zero local utility at position two; only
    # rollout one's first appearance of 1 is positive in that bucket.
    assert advantages[0][1] < 0.0
    assert advantages[1][1] > 0.0


def test_rank_grpo_preserves_archived_short_list_bucket_behavior():
    orders = [[2]] + [[2, 1] for _ in range(3)]
    advantages = rank_grpo_position_advantages(
        orders,
        [{1}] * 4,
        ["q"] * 4,
        slate_k=2,
        expected_group_size=4,
    )
    assert len(advantages[0]) == 1
    # The three emitted position-two contributions are identical and therefore
    # normalize to zero; the short sibling is not inserted as a zero sample.
    assert all(row[1] == 0.0 for row in advantages[1:])


def test_rubric_deltas_use_one_signed_query_level_rms_without_centering():
    deltas = [
        [3.0, -4.0, None, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    advantages = rubric_delta_advantages(deltas)
    scale = math.sqrt(25.0 / 15.0)
    assert advantages[0][0] == pytest.approx(3.0 / scale)
    assert advantages[0][1] == pytest.approx(-4.0 / scale)
    assert advantages[0][2] is None
    assert advantages[0][3] == 0.0
    # No mean centering: signed advantages need not sum to zero.
    assert sum(value for row in advantages for value in row if value is not None) != pytest.approx(0.0)


def test_valid_zeros_participate_in_rubric_rms_and_invalids_do_not():
    deltas = [[None, 0.1, 0.0, 0.0]] + [[0.0] * 4 for _ in range(3)]
    advantages = rubric_delta_advantages(deltas)
    assert advantages[0][0] is None
    assert advantages[0][1] == pytest.approx(math.sqrt(15.0))
    assert advantages[0][2] == 0.0


def test_rubric_rms_explicit_validity_mask_excludes_values():
    deltas = [[100.0, 1.0, 0.0, 0.0]] + [[0.0] * 4 for _ in range(3)]
    valid = [[False, True, True, True]] + [[True] * 4 for _ in range(3)]
    advantages = rms_normalize_rubric_deltas(deltas, valid)
    assert advantages[0][0] is None
    assert advantages[0][1] == pytest.approx(math.sqrt(15.0))


def test_rubric_rms_is_zero_safe_and_preserves_invalid_entries():
    deltas = [[None, 0.0, 0.0, 0.0]] + [[0.0] * 4 for _ in range(3)]
    advantages = rubric_delta_advantages(deltas)
    assert advantages[0][0] is None
    assert all(
        value == 0.0
        for row in advantages
        for value in row
        if value is not None
    )


def test_rank_shift_rewards_original_positive_when_masking_hurts():
    delta, residuals = rank_shift_mask_residuals(
        [1, 2, 3], [2, 3, 1], {1}, 3, metric="ndcg"
    )
    assert delta == pytest.approx(0.5)
    assert residuals == {1: pytest.approx(0.5)}


def test_rank_shift_penalizes_crossed_original_items_when_masking_helps():
    delta, residuals = rank_shift_mask_residuals(
        [2, 3, 1], [1, 2, 3], {1}, 3, metric="ndcg"
    )
    assert delta == pytest.approx(-0.5)
    assert residuals[2] < 0.0
    assert residuals[3] < 0.0
    assert sum(residuals.values()) == pytest.approx(delta)


def test_rank_shift_matches_lenient_ndcg_with_duplicates_and_multiple_positives():
    original = [1, 1, 2]
    masked = [2, 1, 3]
    positives = {1, 2}
    delta, residuals = rank_shift_mask_residuals(
        original, masked, positives, 3, metric="ndcg"
    )
    expected = ndcg_at_k(original, positives, 3) - ndcg_at_k(masked, positives, 3)
    assert delta == pytest.approx(expected)
    assert sum(residuals.values()) == pytest.approx(delta)


def test_rank_shift_preserves_positions_consumed_by_invalid_ids():
    delta, residuals = rank_shift_mask_residuals(
        [999, 1, 2], [1, 999, 2], {1}, 3, metric="ndcg"
    )
    assert delta == pytest.approx(1 / math.log2(3) - 1.0)
    assert residuals[999] == pytest.approx(delta)


def test_rank_shift_scalar_matches_lenient_grader_on_malformed_lists():
    rng = random.Random(20260904)
    for _ in range(500):
        slate_k = rng.randint(1, 12)
        positives = set(
            rng.sample(
                range(1, slate_k + 1),
                k=rng.randint(1, min(3, slate_k)),
            )
        )
        original = [
            rng.randint(-2, slate_k + 3)
            for _ in range(rng.randint(0, slate_k + 4))
        ]
        masked = [
            rng.randint(-2, slate_k + 3)
            for _ in range(rng.randint(0, slate_k + 4))
        ]
        delta, _ = rank_shift_mask_residuals(
            original, masked, positives, slate_k, metric="ndcg"
        )
        expected = ndcg_at_k(original, positives, slate_k) - ndcg_at_k(
            masked, positives, slate_k
        )
        assert delta == pytest.approx(expected)


def test_multi_positive_rank_shift_adds_item_residuals():
    original = [1, 3, 2, 4]
    masked = [3, 2, 4, 1]
    positives = {1, 2}
    delta, residuals = rank_shift_mask_residuals(
        original, masked, positives, 4, metric="ndcg"
    )
    assert delta == pytest.approx(
        ndcg_at_k(original, positives, 4) - ndcg_at_k(masked, positives, 4)
    )
    assert residuals[1] > 0.0
    assert residuals[3] < 0.0
    assert sum(residuals.values()) == pytest.approx(delta)


def test_mask_aggregation_uses_only_existing_residuals_and_clips():
    advantages = aggregate_mask_advantages(
        [{1: 0.1, 2: 0.1}, {2: -0.1}], tau_mask=0.05, mask_clip=2.0
    )
    assert advantages == {1: 2.0, 2: 0.0}


def test_explicit_zero_residual_participates_in_item_mean():
    advantages = aggregate_mask_advantages(
        [{1: 0.1}, {1: 0.0}], tau_mask=0.05, mask_clip=2.0
    )
    assert advantages[1] == 1.0


def test_default_hybrid_uses_half_weight_mask_credit():
    config = MaskPOConfig()
    assert config.tau_mask == 0.05
    assert config.mask_clip == 2.0
    assert config.lambda_mask == 0.5
    combined = combine_answer_advantages([2], [1.0], {2: 2.0}, config=config)
    assert combined == [2.0]


def test_pure_maskpo_sets_rank_lambda_to_zero():
    mask_advantages = aggregate_mask_advantages(
        [{2: 0.1}], tau_mask=0.05, mask_clip=2.0
    )
    combined = combine_answer_advantages(
        [2],
        [100.0],
        mask_advantages,
        config=MaskPOConfig(lambda_rank=0.0, lambda_mask=1.0),
    )
    assert combined == [2.0]


def test_answer_advantage_rejects_position_mismatch():
    with pytest.raises(ValueError, match="same length"):
        combine_answer_advantages([1, 2], [0.0], {})
