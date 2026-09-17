import math

import pytest

from openbench_rerank_rl.answer_rl import (
    normalize_rank_reward_matrix,
    prepare_answer_rl_group,
    rank_grpo_exp_inf_rewards,
    segment_json_answer_tokens,
)


def character_offsets(text: str, *, eos: bool = False):
    offsets = [(index, index + 1) for index in range(len(text))]
    if eos:
        offsets.append((0, 0))
    return offsets


def test_exp_inf_is_fixed_width_and_handles_missing_invalid_and_duplicates():
    rewards = rank_grpo_exp_inf_rewards(
        [
            [1],
            [2, 1],
            [2, 3, 1],
            [99, 1, 1, 4],
        ],
        positives={1},
        slate_k=3,
    )

    assert rewards == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    )


def test_rank_normalization_uses_upstream_sample_std_and_epsilon():
    advantages = normalize_rank_reward_matrix(
        [(1.0,), (0.0,), (0.0,), (0.0,)]
    )
    expected_positive = 0.75 / (0.5 + 1e-4)
    expected_negative = -0.25 / (0.5 + 1e-4)
    assert advantages[0][0] == pytest.approx(expected_positive)
    assert all(row[0] == pytest.approx(expected_negative) for row in advantages[1:])
    assert sum(row[0] for row in advantages) == pytest.approx(0.0)


def test_rank_normalization_is_zero_safe_for_flat_columns():
    assert normalize_rank_reward_matrix([(0.0,)] * 4) == ((0.0,),) * 4


def test_rank_normalization_keeps_nonzero_signal_below_epsilon():
    advantages = normalize_rank_reward_matrix(
        [(0.0,), (1e-6,), (0.0,), (0.0,)]
    )
    sample_std = 5e-7
    assert advantages[1][0] == pytest.approx(0.75e-6 / (sample_std + 1e-4))
    assert advantages[0][0] == pytest.approx(-0.25e-6 / (sample_std + 1e-4))


def test_json_segmentation_owns_delimiters_by_preceding_item_and_eos_by_last():
    text = "[1, 2]"
    segmented = segment_json_answer_tokens(
        text, character_offsets(text, eos=True), slate_k=2
    )

    # `[` and `, ` are item zero; `]` and EOS are item one.
    assert segmented.item_ids == (0, 0, 0, 0, 1, 1, 1)
    assert segmented.loss_mask == (True,) * 7
    assert not segmented.underflow
    assert segmented.action_count == 2


def test_token_containing_delimiter_and_next_integer_belongs_to_next_item():
    text = "[1, 2]"
    # The third token is the tokenizer-merged substring `, 2`.
    segmented = segment_json_answer_tokens(
        text,
        [(0, 1), (1, 2), (2, 5), (5, 6)],
        slate_k=2,
    )
    assert segmented.item_ids == (0, 0, 1, 1)


def test_parseable_extra_text_stays_in_item_objective_and_reference_kl():
    text = "oops [1, 2] trailing"
    segmented = segment_json_answer_tokens(
        text, character_offsets(text, eos=True), slate_k=2
    )
    assert set(segmented.item_ids[: text.index("1")]) == {0}
    assert set(segmented.item_ids[text.index("]") :]) == {1}


def test_segmentation_routes_the_same_explicit_answer_list_as_the_grader():
    text = "incidental [9]\n<answer>[1, 2]</answer>"
    segmented = segment_json_answer_tokens(
        text, character_offsets(text, eos=True), slate_k=2
    )
    assert segmented.order == (1, 2)
    # Prefix prose, including its incidental list, attaches to the first
    # graded answer item instead of becoming a separate rank action.
    assert set(segmented.item_ids[: text.index("1", text.index("<answer>"))]) == {0}
    assert segmented.action_count == 2
    assert all(segmented.loss_mask)


def test_short_json_list_has_a_separate_penalized_termination_action():
    text = "[1]"
    segmented = segment_json_answer_tokens(
        text, character_offsets(text, eos=True), slate_k=3
    )

    assert segmented.item_ids == (0, 0, 1, 1)
    assert segmented.underflow_token_mask == (False, False, True, True)
    assert segmented.termination_item_id == 1
    assert segmented.underflow
    assert segmented.action_count == 2


def test_overflow_items_and_eos_are_segmented_as_natural_actions():
    text = "[1, 2, 3]"
    segmented = segment_json_answer_tokens(
        text, character_offsets(text, eos=True), slate_k=2
    )

    three = text.index("3")
    close = text.index("]")
    assert segmented.item_ids[three] == 2
    assert segmented.item_ids[close] == 2
    assert segmented.item_ids[-1] == 2
    assert segmented.overflow_token_mask[three]
    assert segmented.overflow_token_mask[close]
    assert segmented.overflow_token_mask[-1]
    assert segmented.overflow_count == 1


def test_unparseable_completion_retains_zero_reward_action_and_kl_mask():
    text = "not a list"
    segmented = segment_json_answer_tokens(
        text, character_offsets(text, eos=True), slate_k=3
    )

    assert segmented.order is None
    assert segmented.reason is not None
    assert set(segmented.item_ids[:-1]) == {0}
    assert segmented.item_ids[-1] == 1
    assert all(segmented.loss_mask)
    assert segmented.underflow_token_mask[-1]


def test_prepare_grpo_group_scores_lenient_ndcg_and_uses_four_siblings():
    texts = ("[1]", "[2, 1]", "[2, 3, 1]", "[99, 1, 1, 4]")
    group = prepare_answer_rl_group(
        texts,
        [character_offsets(text, eos=True) for text in texts],
        positives={1},
        slate_k=3,
        algorithm="grpo",
    )

    assert group.sequence_rewards == pytest.approx(
        (1.0, 1 / math.log2(3), 0.5, 1 / math.log2(3))
    )
    assert sum(row.sequence_advantage for row in group.completions) == pytest.approx(0.0)
    assert all(all(row.loss_mask) for row in group.completions)
    assert group.parseable_count == 4
    assert group.action_count == 4

    with pytest.raises(ValueError, match="4 sibling"):
        prepare_answer_rl_group(
            texts[:2],
            [character_offsets(text) for text in texts[:2]],
            positives={1},
            slate_k=3,
            algorithm="grpo",
        )


def test_prepare_rank_grpo_routes_rank_and_length_credit():
    texts = ("[1]", "[2, 1]", "[2, 3, 1]", "[99, 1, 1, 4]")
    group = prepare_answer_rl_group(
        texts,
        [character_offsets(text, eos=True) for text in texts],
        positives={1},
        slate_k=3,
        algorithm="rank_grpo",
    )

    first = group.completions[0]
    # Rank zero is positive for only this sibling; the close bracket/EOS are
    # the separate early-termination action with fixed -0.1 credit.
    assert first.token_advantages[texts[0].index("1")] > 0.0
    assert first.token_advantages[texts[0].index("]")] == pytest.approx(-0.1)
    assert first.token_advantages[-1] == pytest.approx(-0.1)

    last = group.completions[-1]
    overflow_integer = texts[-1].rindex("4")
    assert last.token_advantages[overflow_integer] == pytest.approx(-0.1)
    assert group.underflow_count == 2
    assert group.overflow_count == 1


def test_prepare_unparseable_rank_output_keeps_kl_and_penalizes_stop_only():
    texts = ("bad", "[1, 2]", "[2, 1]", "[2, 3]")
    group = prepare_answer_rl_group(
        texts,
        [character_offsets(text, eos=True) for text in texts],
        positives={1},
        slate_k=2,
        algorithm="rank_grpo",
    )
    bad = group.completions[0]
    assert bad.order is None
    assert bad.token_advantages[:-1] == (0.0,) * len(texts[0])
    assert bad.token_advantages[-1] == pytest.approx(-0.1)
    assert all(bad.loss_mask)
