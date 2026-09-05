import pytest

from openbench_rerank_rl.pipeline import (
    plan_counterfactual_probes,
    score_maskpo_group,
)
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS


def completion(order: str, *, synthesis: str = "combined") -> str:
    sections = "\n".join(
        f"{header} body-{index}"
        for index, header in enumerate(DEFAULT_RUBRIC_HEADERS, start=1)
    )
    return (
        f"<think>\n{sections}\n**Synthesis:** {synthesis}\n</think>\n"
        f"<answer>\n{order}\n</answer>"
    )


def test_probe_plan_is_four_per_strict_original_and_skips_malformed():
    originals = [completion("[1, 2, 3]"), "just [2, 1, 3]"]
    requests = plan_counterfactual_probes(originals)
    assert len(requests) == 4
    assert {(item.rollout_index, item.rubric_index) for item in requests} == {
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
    }
    assert all(item.prefix.text.endswith("**Synthesis:**") for item in requests)


def test_probe_plan_keeps_well_formed_reasoning_with_bad_answer():
    text = completion("not parseable")
    requests = plan_counterfactual_probes([text])
    assert len(requests) == 4


def test_group_scoring_uses_ranking_only_for_probe_delta_and_excludes_invalid():
    originals = [
        completion("[1, 2, 3]"),
        completion("[2, 1, 3]"),
        completion("[1, 1, 99]"),
        "malformed response [3, 1, 2]",
    ]
    probes: list[list[str | None]] = []
    for original in originals:
        # A malformed but parseable probe is still valid.  Equal orders have a
        # zero ranking delta even though their nine-check format reward differs.
        from openbench_rerank_rl.parsers import parse_answer

        order = parse_answer(original)
        rendered = str(order) if order is not None else "[3, 1, 2]"
        probes.append(
            [
                f"no envelope {rendered}",
                completion("[3, 2, 1]"),
                "not parseable",
                completion("[1, 2, 3]"),
            ]
            if original.startswith("<think>")
            else [None, None, None, None]
        )

    result = score_maskpo_group(
        originals,
        probes,
        positives={1},
        slate_k=3,
    )

    assert len(result.rollouts) == 4
    assert result.valid_probe_count == 9
    assert result.rollouts[0].probes[0].valid
    assert result.rollouts[0].probes[0].delta == pytest.approx(0.0)
    assert result.rollouts[0].probes[2].delta is None
    assert result.rollouts[0].rubric_advantages[2] is None
    assert result.rollouts[0].format_reward == pytest.approx(0.1)
    assert result.rollouts[3].answer_parseable
    assert result.rollouts[3].format_reward < 0.1
    assert sum(item.sequence_advantage for item in result.rollouts) == pytest.approx(
        0.0, abs=1e-7
    )


def test_group_scoring_rejects_probe_for_unmaskable_original():
    originals = [completion("[1, 2]")] * 3 + ["plain answer [1, 2]"]
    probes = [[completion("[1, 2]")] * 4 for _ in range(4)]
    with pytest.raises(ValueError, match="not safely maskable"):
        score_maskpo_group(originals, probes, positives={1}, slate_k=2)


def test_valid_zero_probe_participates_and_gets_zero_rubric_advantage():
    originals = [completion("[1, 2]")] * 4
    probes = [[completion("[1, 2]")] * 4 for _ in range(4)]
    result = score_maskpo_group(
        originals,
        probes,
        positives={1},
        slate_k=2,
    )
    assert result.valid_probe_count == 16
    assert result.rubric_rms_scale == 0.0
    assert all(
        advantage == 0.0
        for rollout in result.rollouts
        for advantage in rollout.rubric_advantages
    )
