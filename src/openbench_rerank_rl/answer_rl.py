"""Pure answer-only GRPO and Rank-GRPO credit assignment.

This module deliberately has no trainer or model dependencies.  It prepares
the fixed targets consumed by the answer-only baselines:

* GRPO z-normalizes one lenient ranking reward across sibling completions.
* Rank-GRPO (``exp_inf``) z-normalizes immediate binary relevance separately
  at every output position.
* JSON-list tokens are grouped into item actions for Rank-GRPO's geometric
  item-level importance ratio.

The item tokenizer follows the separator convention in the reference
Rank-GRPO implementation: punctuation *following* an item (a comma and any
space before the next integer) belongs to the preceding item.  The opening
bracket belongs to the first item and, for an exact/overlong list, the closing
  bracket and stop token belong to the final item.  For a short list, the
  closing suffix/stop token is a separate termination action carrying the
  underflow penalty.  A token that
contains an integer always belongs to that integer's item; this semantic rule
also handles tokenizers that merge a comma/space with the next integer.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from .metrics import dcg_at_k, ndcg_at_k
from .parsers import TextSpan, find_answer_index_list


AnswerRLAlgorithm = Literal["grpo", "rank_grpo"]


def _group_zscore(
    values: Sequence[float],
    *,
    eps: float,
    ddof: int,
) -> tuple[float, ...]:
    """Return group-standardized values, with a zero-safe flat group.

    The Rank-GRPO reference uses ``torch.std``'s sample standard deviation and
    adds ``1e-4`` to the denominator.  ``ddof`` remains explicit so controlled
    population-standardized ablations can use ``ddof=0``.
    """

    if eps < 0:
        raise ValueError("eps must be non-negative")
    if not values:
        return ()
    if type(ddof) is not int or ddof < 0:
        raise ValueError("ddof must be a non-negative integer")
    if len(values) <= ddof:
        return (0.0,) * len(values)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cannot normalize non-finite values")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - ddof)
    std = math.sqrt(variance)
    # Match the released Rank-GRPO normalization exactly: only a truly flat
    # group has no relative signal. Small-but-nonzero variation is still
    # divided by ``std + eps`` rather than being discarded.
    if std == 0.0:
        return (0.0,) * len(values)
    return tuple((value - mean) / (std + eps) for value in values)


def _validate_offsets(
    completion: str,
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[TextSpan, ...]:
    spans: list[TextSpan] = []
    previous_end = 0
    for token_index, offset in enumerate(token_offsets):
        if len(offset) != 2:
            raise ValueError(f"token offset {token_index} must contain two integers")
        start, end = offset
        if type(start) is not int or type(end) is not int:
            raise TypeError("token offsets must contain integers")
        if start < 0 or end < start or end > len(completion):
            raise ValueError(
                f"invalid token offset {token_index}: [{start}, {end}) for "
                f"completion of length {len(completion)}"
            )
        # Generated special tokens use the conventional zero-width (0, 0)
        # offset.  Ignore them for monotonicity and assign them explicitly
        # below when they express early termination.
        if start != end:
            if start < previous_end:
                raise ValueError("non-empty token offsets must be ordered and non-overlapping")
            previous_end = end
        spans.append(TextSpan(start, end))
    return tuple(spans)


@dataclass(frozen=True)
class AnswerTokenSegmentation:
    """Item-action ownership for one generated completion.

    ``item_ids`` has one entry per completion token.  ``-1`` means the token
    is outside the Rank-GRPO loss.  Emitted items use their natural zero-based
    output position.  If the list is short, one synthetic action with id
    ``len(order)`` owns the closing suffix/stop token and receives the
    underflow penalty.  Emitted positions ``>= slate_k`` are overflow actions.
    """

    order: tuple[int, ...] | None
    item_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    underflow_token_mask: tuple[bool, ...]
    overflow_token_mask: tuple[bool, ...]
    underflow: bool
    overflow_count: int
    termination_item_id: int | None = None
    reason: str | None = None

    @property
    def parseable(self) -> bool:
        return self.order is not None

    @property
    def action_count(self) -> int:
        return len({item_id for item_id in self.item_ids if item_id >= 0})


def segment_json_answer_tokens(
    completion: str,
    token_offsets: Sequence[tuple[int, int]],
    *,
    slate_k: int,
) -> AnswerTokenSegmentation:
    """Segment a leniently parsed JSON/Python integer list into item actions.

    The parser accepts the first integer list, matching the project's lenient
    ranking grader.  Although the requested format is a bare list, unexpected
    leading/trailing text remains in the objective: leading tokens attach to
    the first item, and trailing tokens attach to the final/termination item.
    This mirrors the reference implementation's whole-completion newline
    segmentation and keeps reference KL active on format mistakes.

    Alignment is conservative: if one tokenizer token overlaps two answer
    integers, segmentation is rejected rather than silently assigning the
    wrong rank credit.
    """

    if type(slate_k) is not int or slate_k <= 0:
        raise ValueError("slate_k must be a positive integer")
    offsets = _validate_offsets(completion, token_offsets)
    parsed = find_answer_index_list(completion)
    empty_ids = (-1,) * len(offsets)
    empty_mask = (False,) * len(offsets)

    def fallback(
        *,
        order: tuple[int, ...] | None,
        underflow: bool,
        overflow_count: int,
        reason: str,
    ) -> AnswerTokenSegmentation:
        """Keep malformed output in the KL/objective as one zero-reward action.

        The upstream newline segmenter never drops a generated completion just
        because its recommendation parser fails.  Reproduce that property by
        assigning visible tokens to one fallback action.  If a trailing
        zero-width stop token exists and the output underflows, it becomes a
        separate termination action so it can carry ``-0.1``.
        """

        visible = [index for index, span in enumerate(offsets) if span.start != span.end]
        special = [index for index, span in enumerate(offsets) if span.start == span.end]
        fallback_ids = [-1] * len(offsets)
        for index in visible:
            fallback_ids[index] = 0
        termination_item_id: int | None = None
        if special:
            if underflow:
                termination_item_id = 1 if visible else 0
                for index in special:
                    fallback_ids[index] = termination_item_id
            else:
                # Exact/long malformed alignments retain reference KL on the
                # stop token as part of the fallback action.
                for index in special:
                    fallback_ids[index] = 0
        fallback_underflow = [
            item_id >= 0 and item_id == termination_item_id
            for item_id in fallback_ids
        ]
        return AnswerTokenSegmentation(
            order=order,
            item_ids=tuple(fallback_ids),
            loss_mask=tuple(item_id >= 0 for item_id in fallback_ids),
            underflow_token_mask=tuple(fallback_underflow),
            overflow_token_mask=empty_mask,
            underflow=underflow,
            overflow_count=overflow_count,
            termination_item_id=termination_item_id,
            reason=reason,
        )

    if parsed is None:
        return fallback(
            order=None,
            underflow=True,
            overflow_count=0,
            reason="completion does not contain a parseable integer list",
        )

    order = parsed.values
    emitted = len(order)
    underflow = emitted < slate_k
    overflow_count = max(0, emitted - slate_k)
    termination_id = emitted
    item_ids = [-1] * len(offsets)
    underflow_mask = [False] * len(offsets)
    overflow_mask = [False] * len(offsets)

    for token_index, token_span in enumerate(offsets):
        if token_span.start == token_span.end:
            # A stop/control token has no character location.  On underflow it
            # represents the decision to terminate; otherwise it is excluded.
            if underflow:
                item_ids[token_index] = termination_id
                underflow_mask[token_index] = True
            elif emitted:
                # Match the upstream item segmenter: EOS is part of the final
                # emitted item when the requested list length was reached.
                item_ids[token_index] = emitted - 1
                if emitted - 1 >= slate_k:
                    overflow_mask[token_index] = True
            continue

        overlapping_values = [
            position
            for position, value_span in enumerate(parsed.value_spans)
            if token_span.overlaps(value_span)
        ]
        if len(overlapping_values) > 1:
            return fallback(
                order=order,
                underflow=underflow,
                overflow_count=overflow_count,
                reason="one token overlaps multiple answer integers",
            )

        owner: int | None = None
        if overlapping_values:
            # Semantic content wins over delimiter ownership when a tokenizer
            # merges e.g. `, 12` into one token.
            owner = overlapping_values[0]
        elif parsed.list_span.overlaps(token_span) or parsed.list_span.contains(token_span):
            if emitted == 0:
                owner = termination_id
            elif token_span.end <= parsed.value_spans[0].start:
                owner = 0
            elif underflow and token_span.start >= parsed.value_spans[-1].end:
                # The close bracket/suffix is the observable early-stop action.
                owner = termination_id
            else:
                # Delimiters belong to the preceding item.  A token that starts
                # before a following integer but overlaps it was already caught
                # by the semantic-content branch above.
                preceding = [
                    position
                    for position, value_span in enumerate(parsed.value_spans)
                    if value_span.end <= token_span.start
                ]
                owner = preceding[-1] if preceding else 0
        elif token_span.end <= parsed.list_span.start:
            owner = 0 if emitted else termination_id
        elif token_span.start >= parsed.list_span.end:
            owner = termination_id if underflow else emitted - 1

        if owner is None:
            continue
        item_ids[token_index] = owner
        if underflow and owner == termination_id:
            underflow_mask[token_index] = True
        if owner >= slate_k and owner < emitted:
            overflow_mask[token_index] = True

    # Every emitted integer must own at least one token.  Otherwise geometric
    # item probabilities cannot be computed faithfully.
    owned = set(item_ids)
    missing = [position for position in range(emitted) if position not in owned]
    if missing:
        return fallback(
            order=order,
            underflow=underflow,
            overflow_count=overflow_count,
            reason=f"answer integers have no aligned token at positions {missing}",
        )

    return AnswerTokenSegmentation(
        order=order,
        item_ids=tuple(item_ids),
        loss_mask=tuple(item_id >= 0 for item_id in item_ids),
        underflow_token_mask=tuple(underflow_mask),
        overflow_token_mask=tuple(overflow_mask),
        underflow=underflow,
        overflow_count=overflow_count,
        termination_item_id=termination_id if underflow else None,
    )


def rank_grpo_exp_inf_rewards(
    orders: Sequence[Sequence[int] | None],
    *,
    positives: set[int] | frozenset[int],
    slate_k: int,
) -> tuple[tuple[float, ...], ...]:
    """Return the paper's fixed-width immediate-relevance reward matrix.

    Every row has exactly ``slate_k`` entries.  Missing positions, invalid
    candidate ids, and unparseable completions are zeros.  A positive earns
    one only at its first emitted occurrence; duplicates continue to consume a
    rank position but cannot receive relevance again.
    """

    if type(slate_k) is not int or slate_k <= 0:
        raise ValueError("slate_k must be a positive integer")
    positive_ids = {
        item for item in positives if type(item) is int and 1 <= item <= slate_k
    }
    rows: list[tuple[float, ...]] = []
    for order in orders:
        row = [0.0] * slate_k
        credited: set[int] = set()
        if order is not None:
            for position, item in enumerate(order[:slate_k]):
                if (
                    type(item) is int
                    and 1 <= item <= slate_k
                    and item in positive_ids
                    and item not in credited
                ):
                    row[position] = 1.0
                    credited.add(item)
        rows.append(tuple(row))
    return tuple(rows)


def normalize_rank_reward_matrix(
    rewards: Sequence[Sequence[float]],
    *,
    eps: float = 1e-4,
    ddof: int = 1,
) -> tuple[tuple[float, ...], ...]:
    """Z-normalize a rectangular ``G x K`` reward matrix by rank position."""

    if not rewards:
        return ()
    width = len(rewards[0])
    if any(len(row) != width for row in rewards):
        raise ValueError("rank reward matrix must be rectangular")
    columns = [
        _group_zscore(
            [float(row[position]) for row in rewards], eps=eps, ddof=ddof
        )
        for position in range(width)
    ]
    return tuple(
        tuple(columns[position][row] for position in range(width))
        for row in range(len(rewards))
    )


@dataclass(frozen=True)
class AnswerRLCompletionCredit:
    """All fixed reward/advantage targets for one sibling completion."""

    completion: str
    order: tuple[int, ...] | None
    sequence_reward: float
    sequence_advantage: float
    rank_rewards: tuple[float, ...]
    rank_advantages: tuple[float, ...]
    token_advantages: tuple[float, ...]
    loss_mask: tuple[bool, ...]
    item_ids: tuple[int, ...]
    underflow: bool
    overflow_count: int
    segmentation_reason: str | None

    @property
    def parseable(self) -> bool:
        return self.order is not None

    @property
    def active_token_count(self) -> int:
        return sum(self.loss_mask)

    @property
    def action_count(self) -> int:
        return len({item_id for item_id in self.item_ids if item_id >= 0})


@dataclass(frozen=True)
class AnswerRLGroupCredit:
    """Prepared pure-GRPO targets and query-level diagnostics."""

    algorithm: AnswerRLAlgorithm
    slate_k: int
    positives: frozenset[int]
    completions: tuple[AnswerRLCompletionCredit, ...]

    @property
    def orders(self) -> tuple[tuple[int, ...] | None, ...]:
        return tuple(completion.order for completion in self.completions)

    @property
    def sequence_rewards(self) -> tuple[float, ...]:
        return tuple(completion.sequence_reward for completion in self.completions)

    @property
    def mean_reward(self) -> float:
        rewards = self.sequence_rewards
        return sum(rewards) / len(rewards) if rewards else 0.0

    @property
    def parseable_count(self) -> int:
        return sum(completion.parseable for completion in self.completions)

    @property
    def underflow_count(self) -> int:
        return sum(completion.underflow for completion in self.completions)

    @property
    def overflow_count(self) -> int:
        return sum(completion.overflow_count for completion in self.completions)

    @property
    def active_token_count(self) -> int:
        return sum(completion.active_token_count for completion in self.completions)

    @property
    def action_count(self) -> int:
        if self.algorithm == "grpo":
            return sum(completion.active_token_count > 0 for completion in self.completions)
        return sum(completion.action_count for completion in self.completions)


def prepare_answer_rl_group(
    completions: Sequence[str],
    token_offsets: Sequence[Sequence[tuple[int, int]]],
    *,
    positives: set[int] | frozenset[int],
    slate_k: int,
    algorithm: AnswerRLAlgorithm,
    expected_group_size: int = 4,
    metric: Literal["ndcg", "dcg"] = "ndcg",
    eps: float = 1e-4,
    normalization_ddof: int = 1,
    underflow_penalty: float = -0.1,
    overflow_penalty: float = -0.1,
) -> AnswerRLGroupCredit:
    """Parse, score, normalize, segment, and route one sibling group.

    GRPO activates every generated token and gives it the query-local scalar
    advantage.  Rank-GRPO activates item actions only: the first ``K`` items
    receive their position-normalized immediate relevance, a short list's
    termination action receives ``underflow_penalty``, and every emitted item
    beyond ``K`` receives ``overflow_penalty``.
    """

    if algorithm not in {"grpo", "rank_grpo"}:
        raise ValueError("algorithm must be 'grpo' or 'rank_grpo'")
    if metric not in {"ndcg", "dcg"}:
        raise ValueError("metric must be 'ndcg' or 'dcg'")
    if expected_group_size <= 0:
        raise ValueError("expected_group_size must be positive")
    if len(completions) != expected_group_size:
        raise ValueError(
            f"expected {expected_group_size} sibling completions, got {len(completions)}"
        )
    if len(token_offsets) != len(completions):
        raise ValueError("completions and token_offsets must have the same length")
    if not math.isfinite(underflow_penalty) or not math.isfinite(overflow_penalty):
        raise ValueError("length penalties must be finite")

    segmentations = tuple(
        segment_json_answer_tokens(completion, offsets, slate_k=slate_k)
        for completion, offsets in zip(completions, token_offsets, strict=True)
    )
    orders = tuple(segmentation.order for segmentation in segmentations)
    score = ndcg_at_k if metric == "ndcg" else dcg_at_k
    sequence_rewards = tuple(
        score(order or (), positives, slate_k) for order in orders
    )
    sequence_advantages = _group_zscore(
        sequence_rewards, eps=eps, ddof=normalization_ddof
    )
    rank_rewards = rank_grpo_exp_inf_rewards(
        orders, positives=positives, slate_k=slate_k
    )
    rank_advantages = normalize_rank_reward_matrix(
        rank_rewards, eps=eps, ddof=normalization_ddof
    )

    prepared: list[AnswerRLCompletionCredit] = []
    for completion, offsets, segmentation, sequence_reward, sequence_advantage, row_rewards, row_advantages in zip(
        completions,
        token_offsets,
        segmentations,
        sequence_rewards,
        sequence_advantages,
        rank_rewards,
        rank_advantages,
        strict=True,
    ):
        if algorithm == "grpo":
            # Completion masks used by the runtime include sampled special
            # stop tokens, represented here by zero-width offsets.
            loss_mask = (True,) * len(offsets)
            token_advantages = (sequence_advantage,) * len(offsets)
            item_ids = (-1,) * len(offsets)
        else:
            loss_mask = segmentation.loss_mask
            item_ids = segmentation.item_ids
            routed: list[float] = []
            for item_id in item_ids:
                if item_id < 0:
                    routed.append(0.0)
                elif (
                    segmentation.underflow
                    and item_id == segmentation.termination_item_id
                ):
                    routed.append(underflow_penalty)
                elif segmentation.reason is not None:
                    # A parser/alignment fallback remains active for reference
                    # KL but has no potentially misrouted policy reward.
                    routed.append(0.0)
                elif item_id >= slate_k:
                    routed.append(overflow_penalty)
                else:
                    routed.append(row_advantages[item_id])
            token_advantages = tuple(routed)

        prepared.append(
            AnswerRLCompletionCredit(
                completion=completion,
                order=segmentation.order,
                sequence_reward=sequence_reward,
                sequence_advantage=sequence_advantage,
                rank_rewards=row_rewards,
                rank_advantages=row_advantages,
                token_advantages=token_advantages,
                loss_mask=loss_mask,
                item_ids=item_ids,
                underflow=segmentation.underflow,
                overflow_count=segmentation.overflow_count,
                segmentation_reason=segmentation.reason,
            )
        )

    return AnswerRLGroupCredit(
        algorithm=algorithm,
        slate_k=slate_k,
        positives=frozenset(positives),
        completions=tuple(prepared),
    )
