"""Parsing helpers for rubric-structured reranking completions.

The ranking reward deliberately uses a lenient parser: if an integer list can
be recovered from the answer block (or, when the envelope is malformed, from
the whole completion), it can be scored.  Semantic token routing is stricter;
it only activates when all expected sections can be located unambiguously.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
SYNTHESIS_HEADER = "**Synthesis:**"
DEFAULT_RUBRIC_HEADERS: tuple[str, ...] = (
    "**Section And Topic Affinity:**",
    "**Entity And Storyline Continuity:**",
    "**Candidate Angle And Marginal Novelty:**",
    "**Temporal And Session Intent:**",
)

_LIST_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)
_INTEGER_RE = re.compile(r"(?<![\w.])-?\d+(?![\w.])")


@dataclass(frozen=True, order=True)
class TextSpan:
    """Half-open character span in a completion."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid text span: [{self.start}, {self.end})")

    def overlaps(self, other: "TextSpan") -> bool:
        return self.start < other.end and other.start < self.end

    def contains(self, other: "TextSpan") -> bool:
        return self.start <= other.start and other.end <= self.end


@dataclass(frozen=True)
class ParsedIndexList:
    """A parsed integer list and the source span of each integer."""

    values: tuple[int, ...]
    list_span: TextSpan
    value_spans: tuple[TextSpan, ...]


@dataclass(frozen=True)
class ReasoningStructure:
    """Strict four-rubric reasoning structure, independent of the answer."""

    text: str
    think_span: TextSpan
    rubric_header_spans: tuple[TextSpan, ...]
    rubric_body_spans: tuple[TextSpan, ...]
    synthesis_header_span: TextSpan
    synthesis_body_span: TextSpan


@dataclass(frozen=True)
class CompletionStructure:
    """Strict reasoning plus a parseable answer, used for token routing."""

    text: str
    think_span: TextSpan
    answer_span: TextSpan
    rubric_header_spans: tuple[TextSpan, ...]
    rubric_body_spans: tuple[TextSpan, ...]
    synthesis_header_span: TextSpan
    synthesis_body_span: TextSpan
    answer_list: ParsedIndexList


def _decode_integer_list(raw: str) -> list[int] | None:
    parsed: object
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return None
    # bool is an int subclass, but candidate identifiers must not accept it.
    if not isinstance(parsed, list) or not all(type(x) is int for x in parsed):
        return None
    return list(parsed)


def find_index_list(text: str, *, offset: int = 0) -> ParsedIndexList | None:
    """Parse the first JSON/Python-style integer list and preserve its spans."""

    for match in _LIST_RE.finditer(text):
        values = _decode_integer_list(match.group(0))
        if values is None:
            continue
        integer_matches = list(_INTEGER_RE.finditer(match.group(0)))
        if len(integer_matches) != len(values):
            # This should only happen for exotic Python syntax.  Refuse to
            # route tokens rather than silently attaching credit incorrectly.
            continue
        base = offset + match.start()
        spans = tuple(TextSpan(base + m.start(), base + m.end()) for m in integer_matches)
        return ParsedIndexList(
            values=tuple(values),
            list_span=TextSpan(offset + match.start(), offset + match.end()),
            value_spans=spans,
        )
    return None


def parse_index_list(text: str) -> list[int] | None:
    """Parse the first JSON/Python-style integer list in ``text``."""

    parsed = find_index_list(text)
    return list(parsed.values) if parsed is not None else None


def extract_answer_block(text: str) -> str:
    """Return the first answer body when present, otherwise return ``text``."""

    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def parse_answer(text: str) -> list[int] | None:
    """Leniently recover an answer list using identical original/probe logic."""

    # Preserve the archived grader's behavior: use the answer block when a
    # complete one exists, but tolerate a missing envelope by scanning the
    # generated text.  Do not rescue a malformed explicit answer from an
    # incidental list in its reasoning.
    return parse_index_list(extract_answer_block(text))


def strict_permutation(order: Iterable[int], k: int) -> bool:
    """Return true iff ``order`` is exactly a permutation of ``1..K``."""

    values = list(order)
    return len(values) == k and set(values) == set(range(1, k + 1))


def _unique_match(text: str, needle: str, *, start: int, end: int) -> TextSpan:
    positions: list[int] = []
    cursor = start
    while True:
        position = text.find(needle, cursor, end)
        if position < 0:
            break
        positions.append(position)
        cursor = position + len(needle)
    if len(positions) != 1:
        raise ValueError(f"expected exactly one {needle!r}; found {len(positions)}")
    return TextSpan(positions[0], positions[0] + len(needle))


def parse_reasoning_structure(
    text: str,
    *,
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
) -> ReasoningStructure:
    """Parse rubric and synthesis regions without requiring a valid answer.

    This function intentionally raises ``ValueError`` for missing, repeated, or
    out-of-order markers.  Separating it from answer parsing lets MaskPO probe a
    sound reasoning trace even when the original final list is unparseable.
    """

    if len(rubric_headers) != 4:
        raise ValueError("MaskPO requires exactly four rubric headers")

    think_open = _unique_match(text, THINK_OPEN, start=0, end=len(text))
    think_close = _unique_match(text, THINK_CLOSE, start=0, end=len(text))
    if think_open.end > think_close.start:
        raise ValueError("think envelope markers are out of order")
    think_contents = TextSpan(think_open.end, think_close.start)
    if (
        text.find(ANSWER_OPEN, think_contents.start, think_contents.end) >= 0
        or text.find(ANSWER_CLOSE, think_contents.start, think_contents.end) >= 0
    ):
        raise ValueError("answer envelope marker appears inside reasoning")

    header_spans = tuple(
        _unique_match(text, header, start=think_open.end, end=think_close.start)
        for header in rubric_headers
    )
    synthesis_header = _unique_match(
        text, SYNTHESIS_HEADER, start=think_open.end, end=think_close.start
    )
    ordered = (*header_spans, synthesis_header)
    if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
        raise ValueError("rubric and synthesis headers are out of order")

    body_spans = tuple(
        TextSpan(header.end, next_header.start)
        for header, next_header in zip(header_spans, ordered[1:], strict=True)
    )

    return ReasoningStructure(
        text=text,
        think_span=TextSpan(think_open.start, think_close.end),
        rubric_header_spans=header_spans,
        rubric_body_spans=body_spans,
        synthesis_header_span=synthesis_header,
        synthesis_body_span=TextSpan(synthesis_header.end, think_close.start),
    )


def parse_completion_structure(
    text: str,
    *,
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
) -> CompletionStructure:
    """Parse the exact semantic regions needed for advantage routing.

    Unlike :func:`parse_reasoning_structure`, this stricter parser also
    requires an answer envelope with a parseable integer list.  Routing callers
    should catch ``ValueError`` and apply the documented sequence fallback.
    """

    reasoning = parse_reasoning_structure(text, rubric_headers=rubric_headers)
    answer_open = _unique_match(
        text, ANSWER_OPEN, start=0, end=len(text)
    )
    answer_close = _unique_match(
        text, ANSWER_CLOSE, start=0, end=len(text)
    )
    if not (
        reasoning.think_span.end <= answer_open.start <= answer_close.start
    ):
        raise ValueError("think/answer envelope markers are out of order")

    answer_body = TextSpan(answer_open.end, answer_close.start)
    parsed_answer = find_index_list(
        text[answer_body.start : answer_body.end], offset=answer_body.start
    )
    if parsed_answer is None:
        raise ValueError("answer block does not contain an integer list")

    return CompletionStructure(
        text=text,
        think_span=reasoning.think_span,
        answer_span=TextSpan(answer_open.start, answer_close.end),
        rubric_header_spans=reasoning.rubric_header_spans,
        rubric_body_spans=reasoning.rubric_body_spans,
        synthesis_header_span=reasoning.synthesis_header_span,
        synthesis_body_span=reasoning.synthesis_body_span,
        answer_list=parsed_answer,
    )
