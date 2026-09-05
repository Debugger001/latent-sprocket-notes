import pytest

from openbench_rerank_rl.parsers import (
    DEFAULT_RUBRIC_HEADERS,
    parse_answer,
    parse_completion_structure,
    parse_index_list,
    parse_reasoning_structure,
)


def completion(answer: str = "[2, 1, 3]") -> str:
    bodies = [" one", " two", " three", " four"]
    think = "\n".join(
        ["<think>"]
        + [f"{header}{body}" for header, body in zip(DEFAULT_RUBRIC_HEADERS, bodies, strict=True)]
        + ["**Synthesis:** combined", "</think>"]
    )
    return f"{think}\n<answer>\n{answer}\n</answer>"


def test_lenient_answer_prefers_answer_block_but_falls_back():
    assert parse_answer("reasoning [9]\n<answer>[2, 1]</answer>") == [2, 1]
    assert parse_answer("malformed envelope then [3, 1]") == [3, 1]
    assert parse_answer("reasoning [9]\n<answer>not a list</answer>") is None


def test_parser_rejects_booleans_as_candidate_ids():
    assert parse_index_list("[1, true]") is None


def test_completion_structure_locates_rubric_bodies_and_answer_numbers():
    text = completion()
    structure = parse_completion_structure(text)
    assert [text[s.start : s.end].strip() for s in structure.rubric_body_spans] == [
        "one",
        "two",
        "three",
        "four",
    ]
    assert structure.answer_list.values == (2, 1, 3)
    assert [text[s.start : s.end] for s in structure.answer_list.value_spans] == ["2", "1", "3"]


def test_completion_structure_rejects_ambiguous_headers():
    text = completion().replace(DEFAULT_RUBRIC_HEADERS[0], DEFAULT_RUBRIC_HEADERS[0] * 2)
    with pytest.raises(ValueError, match="exactly one"):
        parse_completion_structure(text)


def test_reasoning_structure_does_not_require_a_parseable_answer():
    text = completion("not an integer list")
    reasoning = parse_reasoning_structure(text)
    assert len(reasoning.rubric_body_spans) == 4
    with pytest.raises(ValueError, match="integer list"):
        parse_completion_structure(text)


@pytest.mark.parametrize(
    "text",
    [
        "</think>" + completion(),
        completion().replace(" one", " <answer> one", 1),
        completion() + "<answer>[1]</answer>",
    ],
)
def test_semantic_structure_rejects_stray_envelope_tags(text):
    with pytest.raises(ValueError):
        parse_completion_structure(text)
