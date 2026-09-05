import pytest

from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.routing import route_token_advantages


def structured_completion(answer: str = "[12, 1]") -> str:
    sections = "\n".join(
        f"{header} body-{index}"
        for index, header in enumerate(DEFAULT_RUBRIC_HEADERS, start=1)
    )
    return (
        f"<think>\n{sections}\n**Synthesis:** combined\n</think>\n"
        f"<answer>\n{answer}\n</answer>"
    )


def character_offsets(text: str) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(len(text))]


def test_routes_body_and_integer_credit_without_touching_format_tokens():
    text = structured_completion()
    routed = route_token_advantages(
        text,
        character_offsets(text),
        sequence_advantage=0.25,
        rubric_advantages=(1.0, -1.0, 2.0, 0.0),
        rank_advantages=(0.3, -0.4),
        mask_advantages_by_item={12: 2.0, 1: -2.0},
    )

    assert not routed.used_sequence_fallback
    for rubric_index, expected in enumerate((1.0, -1.0, 2.0, 0.0), start=1):
        body_start = text.index(f"body-{rubric_index}")
        assert routed.advantages[body_start] == pytest.approx(expected)

    first_number = text.index("12", text.index("<answer>"))
    assert routed.advantages[first_number] == pytest.approx(1.3)
    assert routed.advantages[first_number + 1] == pytest.approx(1.3)
    second_number = text.index("1", first_number + 2)
    assert routed.advantages[second_number] == pytest.approx(-1.4)

    header_start = text.index(DEFAULT_RUBRIC_HEADERS[0])
    comma = text.index(",", first_number)
    delimiter_newline = text.index("\n", text.index("body-1"))
    assert routed.advantages[header_start] == pytest.approx(0.25)
    assert routed.advantages[comma] == pytest.approx(0.25)
    assert routed.advantages[delimiter_newline] == pytest.approx(0.25)


def test_missing_probe_uses_sequence_credit_only_for_that_rubric_body():
    text = structured_completion()
    routed = route_token_advantages(
        text,
        character_offsets(text),
        sequence_advantage=0.5,
        rubric_advantages=(1.0, None, -1.0, 0.0),
        rank_advantages=(0.0, 0.0),
        mask_advantages_by_item={},
    )
    second_body = text.index("body-2")
    third_body = text.index("body-3")
    assert not routed.used_sequence_fallback
    assert routed.unavailable_rubrics == (1,)
    assert routed.advantages[second_body] == pytest.approx(0.5)
    assert routed.advantages[third_body] == pytest.approx(-1.0)


def test_semantic_parse_failure_falls_back_everywhere():
    text = structured_completion().replace("**Synthesis:**", "Synthesis:")
    routed = route_token_advantages(
        text,
        character_offsets(text),
        sequence_advantage=-0.75,
        rubric_advantages=(1.0, 1.0, 1.0, 1.0),
        rank_advantages=(0.0, 0.0),
        mask_advantages_by_item={},
    )
    assert routed.used_sequence_fallback
    assert set(routed.advantages) == {-0.75}


def test_failed_answer_token_alignment_falls_back_everywhere():
    text = structured_completion()
    answer_start = text.index("<answer>")
    offsets = [offset for offset in character_offsets(text) if offset[0] < answer_start]
    routed = route_token_advantages(
        text,
        offsets,
        sequence_advantage=0.2,
        rubric_advantages=(1.0, 1.0, 1.0, 1.0),
        rank_advantages=(0.0, 0.0),
        mask_advantages_by_item={},
    )
    assert routed.used_sequence_fallback
    assert set(routed.advantages) == {0.2}


def test_failed_rubric_token_alignment_falls_back_everywhere():
    text = structured_completion()
    answer_start = text.index("<answer>")
    offsets = [
        offset for offset in character_offsets(text) if offset[0] >= answer_start
    ]
    routed = route_token_advantages(
        text,
        offsets,
        sequence_advantage=0.2,
        rubric_advantages=(1.0, 1.0, 1.0, 1.0),
        rank_advantages=(0.0, 0.0),
        mask_advantages_by_item={},
    )
    assert routed.used_sequence_fallback
    assert "rubric body" in (routed.fallback_reason or "")
    assert set(routed.advantages) == {0.2}
