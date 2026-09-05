import pytest

from openbench_rerank_rl.rewards import (
    DEFAULT_RUBRIC_HEADERS,
    FormatCheckResult,
    RewardWeights,
    evaluate_format,
    format_reward,
    format_reward_from_checks,
    ranking_reward,
    rollout_reward,
)


def _completion(answer: str = "[2, 1, 3]") -> str:
    bodies = "\n".join(f"{header} evidence" for header in DEFAULT_RUBRIC_HEADERS)
    return (
        f"<think>\n{bodies}\n**Synthesis:** combine evidence\n</think>\n"
        f"<answer>{answer}</answer>"
    )


def test_complete_response_passes_all_nine_format_checks():
    checks = evaluate_format(_completion(), 3)
    assert isinstance(checks, FormatCheckResult)
    assert checks.checks == (True,) * 9
    assert checks.passed == 9
    assert checks.fraction == 1.0
    assert checks.reward() == pytest.approx(0.1)
    assert format_reward(_completion(), 3) == pytest.approx(0.1)


def test_format_checks_are_independent_for_parseable_duplicate_answer():
    checks = evaluate_format(_completion("[2, 2, 3]"), 3)
    assert checks.envelope
    assert checks.rubric_headers == (True, True, True, True)
    assert checks.synthesis
    assert checks.parseable
    assert not checks.valid_unique_ids
    assert not checks.exact_permutation
    assert checks.reward() == pytest.approx(0.1 * 7 / 9)


def test_out_of_range_unique_ids_fail_validity_and_exactness():
    checks = evaluate_format(_completion("[2, 4, 3]"), 3)
    assert checks.parseable
    assert not checks.valid_unique_ids
    assert not checks.exact_permutation


def test_missing_header_loses_only_its_binary_check():
    completion = _completion().replace(DEFAULT_RUBRIC_HEADERS[2], "Candidate angle")
    checks = evaluate_format(completion, 3)
    assert checks.rubric_headers == (True, True, False, True)
    assert checks.passed == 8


def test_headers_outside_think_or_in_wrong_order_do_not_get_full_credit():
    headers_in_answer = (
        "<think>nothing</think>\n<answer>"
        + "".join(DEFAULT_RUBRIC_HEADERS)
        + "**Synthesis:** [2, 1, 3]</answer>"
    )
    outside = evaluate_format(headers_in_answer, 3)
    assert outside.envelope
    assert outside.rubric_headers == (False, False, False, False)
    assert not outside.synthesis
    assert outside.reward() < 0.1

    reversed_headers = _completion().replace(
        DEFAULT_RUBRIC_HEADERS[0] + " evidence\n" + DEFAULT_RUBRIC_HEADERS[1],
        DEFAULT_RUBRIC_HEADERS[1] + " evidence\n" + DEFAULT_RUBRIC_HEADERS[0],
    )
    reversed_checks = evaluate_format(reversed_headers, 3)
    assert not all(reversed_checks.rubric_headers)
    assert reversed_checks.reward() < 0.1


def test_malformed_envelope_can_still_have_parseable_ranking():
    completion = _completion() + " trailing text"
    checks = evaluate_format(completion, 3)
    assert not checks.envelope
    assert checks.parseable
    assert checks.exact_permutation
    assert checks.passed == 8


def test_format_reward_accepts_exactly_nine_precomputed_booleans():
    assert format_reward_from_checks([True] * 8 + [False]) == pytest.approx(0.1 * 8 / 9)
    assert format_reward_from_checks({str(i): True for i in range(9)}) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="nine"):
        format_reward_from_checks([True] * 8)


def test_ranking_reward_uses_lenient_first_positive_ndcg():
    reward = ranking_reward([1, 1, 2], {1, 2}, 3)
    assert 0.0 < reward <= 1.0


def test_sequence_reward_adds_bounded_format_reward_by_default():
    rank = ranking_reward([1, 2, 3], {1}, 3)
    sequence = rollout_reward([1, 2, 3], {1}, 3, format_score=0.1)
    assert sequence == pytest.approx(rank + 0.1)


def test_masking_delta_can_use_ranking_reward_without_format_term():
    weights = RewardWeights()
    original = ranking_reward([1, 2, 3], {1}, 3, weights=weights)
    masked = ranking_reward([2, 3, 1], {1}, 3, weights=weights)
    assert original - masked == pytest.approx(0.5)
