"""Optional integration check against an imported Qwen3 adapter tokenizer."""

from __future__ import annotations

import os

import pytest

from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.routing import route_token_advantages
from openbench_rerank_rl.trainer import HuggingFacePolicyBackend


ADAPTER_PATH = os.environ.get("MASKPO_ADAPTER_PATH")
pytestmark = pytest.mark.skipif(
    not ADAPTER_PATH,
    reason="set MASKPO_ADAPTER_PATH to run the real-tokenizer integration test",
)


def test_qwen_fast_tokenizer_roundtrip_preserves_semantic_offsets():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(ADAPTER_PATH)

    class UnusedModel:
        training = False

    backend = HuggingFacePolicyBackend(UnusedModel(), tokenizer)
    text = (
        "<think>\n"
        + "\n\n".join(
            f"{header} evidence for candidate 12"
            for header in DEFAULT_RUBRIC_HEADERS
        )
        + "\n\n**Synthesis:** rank 12 first\n</think>\n"
        "<answer>\n[12, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]\n</answer>"
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded, offsets = backend._decode_with_offsets(token_ids)
    assert decoded == text

    routed = route_token_advantages(
        decoded,
        offsets,
        sequence_advantage=0.2,
        rubric_advantages=(1.0, 2.0, 3.0, 4.0),
        rank_advantages=(0.0,) * 12,
        mask_advantages_by_item={12: 2.0},
    )
    assert not routed.used_sequence_fallback
    assert set(routed.regions) >= {
        "rubric_0",
        "rubric_1",
        "rubric_2",
        "rubric_3",
        "answer_0",
        "sequence",
    }
    assert {
        advantage
        for advantage, region in zip(
            routed.advantages, routed.regions, strict=True
        )
        if region == "answer_0"
    } == {1.0}
