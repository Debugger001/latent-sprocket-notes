"""Counterfactual rubric masking for MaskPO probes.

Each probe changes exactly one rubric body in an already sampled completion.
The remaining three rubric bodies are held fixed, and generation resumes after
the ``**Synthesis:**`` header.  Probe suffixes are scoring-only samples and
must never be included in the policy-gradient batch.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .parsers import (
    DEFAULT_RUBRIC_HEADERS,
    parse_answer,
    parse_reasoning_structure,
)


MASKED_RUBRIC_CONTENT = "[MASKED_RUBRIC_CONTENT]"


@dataclass(frozen=True)
class CounterfactualPrefix:
    """Assistant prefix used to sample one counterfactual suffix."""

    rubric_index: int
    rubric_header: str
    text: str


@dataclass(frozen=True)
class ParsedCounterfactual:
    """Lenient parse result for a generated counterfactual suffix."""

    completion: str
    order: tuple[int, ...] | None

    @property
    def valid(self) -> bool:
        """A probe is valid whenever the shared ranking parser succeeds."""

        return self.order is not None


def _replace_body_preserving_whitespace(body: str, replacement: str) -> str:
    """Replace body prose while preserving its surrounding whitespace."""

    leading_end = len(body) - len(body.lstrip())
    trailing_start = len(body.rstrip())
    if leading_end > trailing_start:  # the original body was all whitespace
        separator = "\n" if "\n" in body else " "
        return f"{separator}{replacement}{separator}"
    return f"{body[:leading_end]}{replacement}{body[trailing_start:]}"


def build_counterfactual_prefix(
    completion: str,
    rubric_index: int,
    *,
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
    placeholder: str = MASKED_RUBRIC_CONTENT,
) -> CounterfactualPrefix:
    """Mask one rubric body and retain the prefix through ``**Synthesis:**``.

    The completion must have an unambiguous four-rubric structure.  This is a
    deliberate strict boundary: if the semantic regions cannot be located, no
    counterfactual is generated for that rollout and token routing later falls
    back to sequence credit.
    """

    if not 0 <= rubric_index < len(rubric_headers):
        raise IndexError(
            f"rubric_index must be in [0, {len(rubric_headers) - 1}], "
            f"got {rubric_index}"
        )
    if not placeholder:
        raise ValueError("placeholder must not be empty")

    structure = parse_reasoning_structure(completion, rubric_headers=rubric_headers)
    body_span = structure.rubric_body_spans[rubric_index]
    masked_body = _replace_body_preserving_whitespace(
        completion[body_span.start : body_span.end], placeholder
    )
    prefix = (
        completion[: body_span.start]
        + masked_body
        + completion[body_span.end : structure.synthesis_header_span.end]
    )
    return CounterfactualPrefix(
        rubric_index=rubric_index,
        rubric_header=rubric_headers[rubric_index],
        text=prefix,
    )


def build_counterfactual_prefixes(
    completion: str,
    *,
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
    placeholder: str = MASKED_RUBRIC_CONTENT,
) -> tuple[CounterfactualPrefix, ...]:
    """Build the four one-body-at-a-time probes for a completion."""

    return tuple(
        build_counterfactual_prefix(
            completion,
            rubric_index,
            rubric_headers=rubric_headers,
            placeholder=placeholder,
        )
        for rubric_index in range(len(rubric_headers))
    )


def parse_counterfactual(
    prefix: CounterfactualPrefix | str,
    generated_suffix: str,
) -> ParsedCounterfactual:
    """Join and score a probe with the same lenient parser as originals."""

    prefix_text = prefix.text if isinstance(prefix, CounterfactualPrefix) else prefix
    completion = prefix_text + generated_suffix
    parsed = parse_answer(completion)
    return ParsedCounterfactual(
        completion=completion,
        order=tuple(parsed) if parsed is not None else None,
    )
