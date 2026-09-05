"""Route MaskPO advantages to semantic completion-token regions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .advantages import MaskPOConfig, combine_answer_advantages
from .parsers import DEFAULT_RUBRIC_HEADERS, TextSpan, parse_completion_structure


@dataclass(frozen=True)
class RoutingResult:
    """Per-token advantages and diagnostics for one original rollout."""

    advantages: tuple[float, ...]
    regions: tuple[str, ...]
    used_sequence_fallback: bool
    fallback_reason: str | None = None
    unavailable_rubrics: tuple[int, ...] = ()


def _sequence_fallback(
    token_count: int,
    sequence_advantage: float,
    reason: str,
) -> RoutingResult:
    return RoutingResult(
        advantages=(sequence_advantage,) * token_count,
        regions=("sequence_fallback",) * token_count,
        used_sequence_fallback=True,
        fallback_reason=reason,
    )


def route_token_advantages(
    completion: str,
    token_offsets: Sequence[tuple[int, int]],
    *,
    sequence_advantage: float,
    rubric_advantages: Sequence[float | None],
    rank_advantages: Sequence[float],
    mask_advantages_by_item: Mapping[int, float],
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
    config: MaskPOConfig = MaskPOConfig(),
) -> RoutingResult:
    """Assign latest MaskPO credit to completion tokens.

    ``token_offsets`` are half-open character offsets into ``completion`` (as
    returned by a fast tokenizer when the completion is tokenized alone).
    Formatting tokens, headers, delimiters, synthesis prose, and punctuation
    retain ``A_seq``.  A rubric body receives only its counterfactual rubric
    advantage.  Each integer in the answer receives its position-normalized
    rank credit plus ``lambda_mask`` times that candidate's mask residual.

    A missing/unparseable counterfactual has no rubric advantage; only that
    body's tokens conservatively retain ``A_seq``.  If semantic parsing or
    integer-to-token alignment fails, the documented safe fallback assigns
    ``A_seq`` to the entire completion.
    """

    token_count = len(token_offsets)
    if not math.isfinite(sequence_advantage):
        raise ValueError("sequence_advantage must be finite")
    if len(rubric_advantages) != len(rubric_headers):
        return _sequence_fallback(
            token_count,
            sequence_advantage,
            "rubric advantage count does not match rubric header count",
        )
    if any(value is not None and not math.isfinite(value) for value in rubric_advantages):
        raise ValueError("rubric advantages must be finite or None")
    if any(not math.isfinite(value) for value in rank_advantages):
        raise ValueError("rank advantages must be finite")
    if any(not math.isfinite(value) for value in mask_advantages_by_item.values()):
        raise ValueError("mask advantages must be finite")

    offsets: list[TextSpan] = []
    for start, end in token_offsets:
        if start < 0 or end < start or end > len(completion):
            return _sequence_fallback(
                token_count,
                sequence_advantage,
                f"invalid token offset [{start}, {end})",
            )
        offsets.append(TextSpan(start, end))

    try:
        structure = parse_completion_structure(
            completion, rubric_headers=rubric_headers
        )
    except ValueError as exc:
        return _sequence_fallback(token_count, sequence_advantage, str(exc))

    order = structure.answer_list.values
    if len(rank_advantages) != len(order):
        return _sequence_fallback(
            token_count,
            sequence_advantage,
            "rank advantage count does not match parsed answer length",
        )
    try:
        answer_advantages = combine_answer_advantages(
            order,
            rank_advantages,
            mask_advantages_by_item,
            config=config,
        )
    except ValueError as exc:
        return _sequence_fallback(token_count, sequence_advantage, str(exc))

    advantages = [sequence_advantage] * token_count
    regions = ["sequence"] * token_count
    unavailable_rubrics: list[int] = []

    for rubric_index, (body_span, rubric_advantage) in enumerate(
        zip(structure.rubric_body_spans, rubric_advantages, strict=True)
    ):
        if rubric_advantage is None:
            unavailable_rubrics.append(rubric_index)
            continue
        routed_body_tokens = 0
        for token_index, token_span in enumerate(offsets):
            token_text = completion[token_span.start : token_span.end]
            if (
                token_span.start != token_span.end
                and token_text.strip()
                and body_span.contains(token_span)
            ):
                advantages[token_index] = rubric_advantage
                regions[token_index] = f"rubric_{rubric_index}"
                routed_body_tokens += 1
        body_text = completion[body_span.start : body_span.end]
        if body_text.strip() and routed_body_tokens == 0:
            return _sequence_fallback(
                token_count,
                sequence_advantage,
                f"rubric body {rubric_index} has no aligned token",
            )

    # Integers can span multiple subword tokens; every overlapping token gets
    # the integer's credit.  Alignment is rejected if an integer has no token
    # or a token overlaps two integer spans.
    matched_tokens: set[int] = set()
    for position, (value_span, answer_advantage) in enumerate(
        zip(structure.answer_list.value_spans, answer_advantages, strict=True)
    ):
        position_tokens = [
            token_index
            for token_index, token_span in enumerate(offsets)
            if token_span.start != token_span.end and token_span.overlaps(value_span)
        ]
        if not position_tokens:
            return _sequence_fallback(
                token_count,
                sequence_advantage,
                f"answer integer at position {position} has no aligned token",
            )
        if any(token_index in matched_tokens for token_index in position_tokens):
            return _sequence_fallback(
                token_count,
                sequence_advantage,
                "one token overlaps multiple answer integers",
            )
        for token_index in position_tokens:
            matched_tokens.add(token_index)
            advantages[token_index] = answer_advantage
            regions[token_index] = f"answer_{position}"

    return RoutingResult(
        advantages=tuple(advantages),
        regions=tuple(regions),
        used_sequence_fallback=False,
        unavailable_rubrics=tuple(unavailable_rubrics),
    )
