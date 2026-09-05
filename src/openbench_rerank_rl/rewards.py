"""Ranking and dense-format rewards for rubric reranking rollouts."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .metrics import mrr_at_k, ndcg_at_k, recall_at_k, slate_auc
from .parsers import parse_answer, strict_permutation


DEFAULT_RUBRIC_HEADERS: tuple[str, str, str, str] = (
    "**Section And Topic Affinity:**",
    "**Entity And Storyline Continuity:**",
    "**Candidate Angle And Marginal Novelty:**",
    "**Temporal And Session Intent:**",
)
DEFAULT_SYNTHESIS_HEADER = "**Synthesis:**"


@dataclass(frozen=True)
class RewardWeights:
    """Weights in the sequence-level reward.

    The latest MaskPO setup uses ``nDCG + format_reward``.  The format reward
    is already bounded by 0.1, so its multiplier is one by default.  The other
    ranking metrics and the legacy instruction-failure penalty remain available
    for ablations.
    """

    ndcg: float = 1.0
    auc: float = 0.0
    recall: float = 0.0
    mrr: float = 0.0
    dense_format: float = 1.0
    if_penalty: float = 0.0


@dataclass(frozen=True)
class FormatCheckResult:
    """The nine independent binary checks that make up format reward."""

    envelope: bool
    rubric_headers: tuple[bool, bool, bool, bool]
    synthesis: bool
    parseable: bool
    valid_unique_ids: bool
    exact_permutation: bool

    def __post_init__(self) -> None:
        if len(self.rubric_headers) != 4:
            raise ValueError("format scoring requires exactly four rubric-header checks")

    @property
    def checks(self) -> tuple[bool, ...]:
        """Return the checks in their canonical nine-check order."""

        return (
            self.envelope,
            *self.rubric_headers,
            self.synthesis,
            self.parseable,
            self.valid_unique_ids,
            self.exact_permutation,
        )

    @property
    def passed(self) -> int:
        return sum(self.checks)

    @property
    def fraction(self) -> float:
        return self.passed / 9.0

    def reward(self, max_reward: float = 0.1) -> float:
        """Scale the mean of the checks into ``[0, max_reward]``."""

        if max_reward < 0:
            raise ValueError("max_reward must be non-negative")
        return max_reward * self.fraction


_ENVELOPE_RE = re.compile(
    r"\A<think>.*</think>\n<answer>.*</answer>\Z",
    flags=re.DOTALL,
)


def _canonical_header(header: str) -> str:
    """Accept either a bare rubric name or its exact bold marker."""

    stripped = header.strip()
    if stripped.startswith("**") and stripped.endswith("**"):
        return stripped
    return f"**{stripped.rstrip(':')}:**"


def evaluate_format(
    completion: str,
    k: int,
    *,
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
    synthesis_header: str = DEFAULT_SYNTHESIS_HEADER,
) -> FormatCheckResult:
    """Evaluate the nine binary format checks for one completion.

    Checks are intentionally independent.  For example, an answer list can be
    parseable even if the outer ``<think>/<answer>`` envelope is malformed.
    This supplies a smooth format signal while the ranking reward remains
    lenient whenever an integer list can be parsed.
    """

    if k < 0:
        raise ValueError("k must be non-negative")
    if len(rubric_headers) != 4:
        raise ValueError("format scoring requires exactly four rubric headers")

    headers = tuple(_canonical_header(header) for header in rubric_headers)
    synthesis = _canonical_header(synthesis_header)
    think_open = completion.find("<think>")
    think_close = completion.find("</think>", think_open + len("<think>"))
    has_think_region = (
        think_open >= 0
        and think_close >= think_open + len("<think>")
        and completion.count("<think>") == 1
        and completion.count("</think>") == 1
    )
    think_start = think_open + len("<think>")
    marker_positions: list[int | None] = []
    for header in headers:
        position = completion.find(header)
        marker_positions.append(
            position
            if has_think_region
            and completion.count(header) == 1
            and think_start <= position < think_close
            else None
        )

    # A rubric earns its check only when it is in the reasoning block and in
    # its prescribed relative position.  Missing markers do not make an
    # otherwise correctly placed header fail its own independent check.
    rubric_checks_list: list[bool] = []
    for index, position in enumerate(marker_positions):
        earlier = [value for value in marker_positions[:index] if value is not None]
        later = [value for value in marker_positions[index + 1 :] if value is not None]
        rubric_checks_list.append(
            position is not None
            and all(value < position for value in earlier)
            and all(position < value for value in later)
        )
    rubric_checks = tuple(rubric_checks_list)

    synthesis_position = completion.find(synthesis)
    synthesis_ok = (
        has_think_region
        and completion.count(synthesis) == 1
        and think_start <= synthesis_position < think_close
        and all(
            position < synthesis_position
            for position in marker_positions
            if position is not None
        )
    )

    order = parse_answer(completion)
    parseable = order is not None
    valid_unique_ids = bool(order) and len(order) == len(set(order)) and all(
        1 <= item <= k for item in order
    )
    exact_permutation = order is not None and strict_permutation(order, k)

    return FormatCheckResult(
        envelope=_ENVELOPE_RE.fullmatch(completion) is not None
        and completion.count("<think>") == 1
        and completion.count("</think>") == 1
        and completion.count("<answer>") == 1
        and completion.count("</answer>") == 1,
        rubric_headers=rubric_checks,  # type: ignore[arg-type]
        synthesis=synthesis_ok,
        parseable=parseable,
        valid_unique_ids=valid_unique_ids,
        exact_permutation=exact_permutation,
    )


def format_reward_from_checks(
    checks: FormatCheckResult | Sequence[bool] | Mapping[str, bool],
    *,
    max_reward: float = 0.1,
) -> float:
    """Compute dense format reward from exactly nine binary checks.

    ``FormatCheckResult`` is preferred.  A sequence or mapping is accepted for
    trainers that already compute the nine checks while tokenizing.
    """

    if isinstance(checks, FormatCheckResult):
        return checks.reward(max_reward)
    values = tuple(checks.values()) if isinstance(checks, Mapping) else tuple(checks)
    if len(values) != 9:
        raise ValueError("format reward requires exactly nine checks")
    if max_reward < 0:
        raise ValueError("max_reward must be non-negative")
    return max_reward * sum(bool(value) for value in values) / 9.0


def format_reward(
    completion: str,
    k: int,
    *,
    rubric_headers: Sequence[str] = DEFAULT_RUBRIC_HEADERS,
    synthesis_header: str = DEFAULT_SYNTHESIS_HEADER,
    max_reward: float = 0.1,
) -> float:
    """Return the dense nine-check format reward in ``[0, 0.1]`` by default."""

    return evaluate_format(
        completion,
        k,
        rubric_headers=rubric_headers,
        synthesis_header=synthesis_header,
    ).reward(max_reward)


def ranking_reward(
    order: Sequence[int],
    positives: set[int],
    slate_k: int,
    *,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """Weighted ranking-only reward, shared by originals and mask probes."""

    return (
        weights.ndcg * ndcg_at_k(order, positives, slate_k)
        + weights.auc * slate_auc(order, positives, slate_k)
        + weights.recall * recall_at_k(order, positives, slate_k)
        + weights.mrr * mrr_at_k(order, positives, slate_k)
    )


def rollout_reward(
    order: Sequence[int],
    positives: set[int],
    slate_k: int,
    *,
    format_score: float = 0.0,
    instruction_penalty: float = 0.0,
    weights: RewardWeights = RewardWeights(),
) -> float:
    """Sequence reward ``R_seq = R_rank + R_format`` used for group z-scoring.

    Counterfactual rubric deltas must call :func:`ranking_reward` directly: the
    format term is deliberately not causal rubric credit.
    """

    return (
        ranking_reward(order, positives, slate_k, weights=weights)
        + weights.dense_format * format_score
        - weights.if_penalty * instruction_penalty
    )
