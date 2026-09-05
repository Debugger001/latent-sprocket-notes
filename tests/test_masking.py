import pytest

from openbench_rerank_rl.masking import (
    MASKED_RUBRIC_CONTENT,
    build_counterfactual_prefix,
    build_counterfactual_prefixes,
    parse_counterfactual,
)
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS


def structured_completion() -> str:
    sections = "\n".join(
        f"{header} body-{index}"
        for index, header in enumerate(DEFAULT_RUBRIC_HEADERS, start=1)
    )
    return (
        f"<think>\n{sections}\n**Synthesis:** original synthesis\n</think>\n"
        "<answer>\n[2, 1, 3]\n</answer>"
    )


def test_counterfactual_masks_one_body_and_stops_at_synthesis_header():
    original = structured_completion()
    probe = build_counterfactual_prefix(original, 1)

    assert probe.text.endswith("**Synthesis:**")
    assert probe.text.count(MASKED_RUBRIC_CONTENT) == 1
    assert "body-1" in probe.text
    assert "body-2" not in probe.text
    assert "body-3" in probe.text
    assert "body-4" in probe.text
    assert "original synthesis" not in probe.text
    assert "<answer>" not in probe.text


def test_counterfactual_prefixes_mask_each_rubric_once():
    probes = build_counterfactual_prefixes(structured_completion())
    assert len(probes) == 4
    assert [probe.rubric_index for probe in probes] == [0, 1, 2, 3]
    for masked_index, probe in enumerate(probes, start=1):
        assert f"body-{masked_index}" not in probe.text
        assert all(
            f"body-{other}" in probe.text
            for other in range(1, 5)
            if other != masked_index
        )


def test_probe_validity_is_lenient_and_uses_shared_answer_parser():
    prefix = build_counterfactual_prefix(structured_completion(), 0)
    parsed = parse_counterfactual(
        prefix,
        " regenerated synthesis\n</think>\n<answer>[1, 1, 99]</answer>",
    )
    assert parsed.valid
    assert parsed.order == (1, 1, 99)

    missing_envelope = parse_counterfactual(prefix, " prose then [3, 1]")
    assert missing_envelope.valid
    assert missing_envelope.order == (3, 1)

    invalid = parse_counterfactual(prefix, " no integer list")
    assert not invalid.valid
    assert invalid.order is None


def test_counterfactual_rejects_out_of_range_rubric():
    with pytest.raises(IndexError):
        build_counterfactual_prefix(structured_completion(), 4)


def test_unparseable_original_answer_does_not_prevent_reasoning_mask():
    text = structured_completion().replace("[2, 1, 3]", "not parseable")
    probe = build_counterfactual_prefix(text, 2)
    assert probe.text.endswith("**Synthesis:**")
    assert probe.text.count(MASKED_RUBRIC_CONTENT) == 1
