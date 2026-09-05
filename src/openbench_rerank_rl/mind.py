"""Strict, dependency-free readers for the public MIND news dataset.

MIND distributes two tab-separated files.  ``news.tsv`` has eight columns and
``behaviors.tsv`` has five; keeping those contracts explicit catches the most
common silent data-corruption error (tabs or columns being dropped while the
dataset is moved between machines).

The readers retain the logged candidate order.  Click labels are represented
separately as one-based positive indices so they can never accidentally become
candidate features in a model prompt.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_HISTORY = 50


class MindFormatError(ValueError):
    """Raised when a MIND TSV row does not match the public file schema."""


@dataclass(frozen=True, slots=True)
class MindNews:
    """One eight-field row from ``news.tsv``."""

    news_id: str
    category: str
    subcategory: str
    title: str
    abstract: str
    url: str
    title_entities: str
    abstract_entities: str

    def prompt_dict(self) -> dict[str, str]:
        """Return only article attributes permitted in a reranking prompt."""

        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "title": self.title,
            "abstract": self.abstract,
        }


@dataclass(frozen=True, slots=True)
class MindBehavior:
    """One five-field row from ``behaviors.tsv``.

    ``candidate_news_ids`` stays in logged impression order.  Positive indices
    are one-based to match the candidate indices emitted by the reranker.
    """

    impression_id: str
    user_id: str
    impression_time: str
    history_news_ids: tuple[str, ...]
    candidate_news_ids: tuple[str, ...]
    positive_indices: tuple[int, ...]
    positive_news_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MindExample:
    """A behavior row joined to its history and candidate articles."""

    impression_id: str
    user_id: str
    impression_time: str
    history: tuple[MindNews, ...]
    candidates: tuple[MindNews, ...]
    positive_indices: tuple[int, ...]
    positive_news_ids: tuple[str, ...]

    @property
    def k(self) -> int:
        return len(self.candidates)


def _split_exact_fields(
    line: str,
    *,
    expected: int,
    path: str | Path,
    line_number: int,
) -> list[str]:
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != expected:
        raise MindFormatError(
            f"{path}:{line_number}: expected exactly {expected} tab-separated "
            f"fields, found {len(fields)}"
        )
    return fields


def parse_news_line(
    line: str,
    *,
    path: str | Path = "news.tsv",
    line_number: int = 1,
) -> MindNews:
    """Parse one MIND ``news.tsv`` row, requiring all eight fields."""

    fields = _split_exact_fields(
        line, expected=8, path=path, line_number=line_number
    )
    if not fields[0]:
        raise MindFormatError(f"{path}:{line_number}: news ID must not be empty")
    return MindNews(*fields)


def iter_news_tsv(path: str | Path) -> Iterator[MindNews]:
    """Yield rows from a UTF-8 MIND ``news.tsv`` file."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            yield parse_news_line(line, path=source, line_number=line_number)


def load_news_tsv(path: str | Path) -> dict[str, MindNews]:
    """Load ``news.tsv`` keyed by news ID, rejecting duplicate IDs."""

    result: dict[str, MindNews] = {}
    for article in iter_news_tsv(path):
        if article.news_id in result:
            raise MindFormatError(f"{path}: duplicate news ID {article.news_id!r}")
        result[article.news_id] = article
    return result


def parse_behavior_line(
    line: str,
    *,
    path: str | Path = "behaviors.tsv",
    line_number: int = 1,
) -> MindBehavior:
    """Parse one labeled MIND ``behaviors.tsv`` row.

    Impression labels must be the public train/dev representation, ``0`` or
    ``1`` after the final hyphen.  The function deliberately does not reorder
    candidates or collapse duplicate candidate IDs.
    """

    impression_id, user_id, impression_time, raw_history, raw_impressions = (
        _split_exact_fields(
            line, expected=5, path=path, line_number=line_number
        )
    )
    if not impression_id:
        raise MindFormatError(
            f"{path}:{line_number}: impression ID must not be empty"
        )

    history = tuple(raw_history.split()) if raw_history else ()
    candidate_ids: list[str] = []
    positive_indices: list[int] = []
    positive_ids: list[str] = []

    impression_tokens = raw_impressions.split()
    if not impression_tokens:
        raise MindFormatError(
            f"{path}:{line_number}: impression must contain at least one candidate"
        )

    for one_based_index, token in enumerate(impression_tokens, start=1):
        try:
            news_id, label = token.rsplit("-", 1)
        except ValueError as error:
            raise MindFormatError(
                f"{path}:{line_number}: malformed impression token {token!r}"
            ) from error
        if not news_id or label not in {"0", "1"}:
            raise MindFormatError(
                f"{path}:{line_number}: impression token {token!r} must end in -0 or -1"
            )
        candidate_ids.append(news_id)
        if label == "1":
            positive_indices.append(one_based_index)
            positive_ids.append(news_id)

    return MindBehavior(
        impression_id=impression_id,
        user_id=user_id,
        impression_time=impression_time,
        history_news_ids=history,
        candidate_news_ids=tuple(candidate_ids),
        positive_indices=tuple(positive_indices),
        positive_news_ids=tuple(positive_ids),
    )


def iter_behaviors_tsv(path: str | Path) -> Iterator[MindBehavior]:
    """Yield rows from a UTF-8 MIND ``behaviors.tsv`` file."""

    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            yield parse_behavior_line(line, path=source, line_number=line_number)


def _look_up_articles(
    news_by_id: Mapping[str, MindNews],
    ids: Sequence[str],
    *,
    impression_id: str,
    role: str,
) -> tuple[MindNews, ...]:
    articles: list[MindNews] = []
    for news_id in ids:
        try:
            articles.append(news_by_id[news_id])
        except KeyError as error:
            raise MindFormatError(
                f"impression {impression_id!r}: {role} references missing news ID "
                f"{news_id!r}"
            ) from error
    return tuple(articles)


def build_example(
    behavior: MindBehavior,
    news_by_id: Mapping[str, MindNews],
    *,
    max_history: int | None = DEFAULT_MAX_HISTORY,
) -> MindExample:
    """Join a behavior row to news content.

    MIND history is already ordered oldest to newest.  Truncation therefore
    takes the final ``max_history`` IDs while preserving their order.
    ``None`` disables truncation; ``0`` deliberately produces an empty history.
    """

    if max_history is not None and max_history < 0:
        raise ValueError("max_history must be non-negative or None")

    history_ids = behavior.history_news_ids
    if max_history is not None:
        history_ids = history_ids[-max_history:] if max_history else ()

    history = _look_up_articles(
        news_by_id,
        history_ids,
        impression_id=behavior.impression_id,
        role="history",
    )
    candidates = _look_up_articles(
        news_by_id,
        behavior.candidate_news_ids,
        impression_id=behavior.impression_id,
        role="candidate slate",
    )
    return MindExample(
        impression_id=behavior.impression_id,
        user_id=behavior.user_id,
        impression_time=behavior.impression_time,
        history=history,
        candidates=candidates,
        positive_indices=behavior.positive_indices,
        positive_news_ids=behavior.positive_news_ids,
    )


def load_examples(
    news_path: str | Path,
    behaviors_path: str | Path,
    *,
    max_history: int | None = DEFAULT_MAX_HISTORY,
) -> list[MindExample]:
    """Load and join public MIND news and behavior TSV files."""

    news_by_id = load_news_tsv(news_path)
    return [
        build_example(behavior, news_by_id, max_history=max_history)
        for behavior in iter_behaviors_tsv(behaviors_path)
    ]


def filter_examples(
    examples: Iterable[MindExample],
    *,
    min_history: int = 0,
    min_candidates: int = 1,
    max_candidates: int | None = None,
    require_positive: bool = True,
) -> list[MindExample]:
    """Apply deterministic, order-preserving preparation filters."""

    if min_history < 0:
        raise ValueError("min_history must be non-negative")
    if min_candidates < 1:
        raise ValueError("min_candidates must be at least one")
    if max_candidates is not None and max_candidates < min_candidates:
        raise ValueError("max_candidates must be >= min_candidates")

    return [
        example
        for example in examples
        if len(example.history) >= min_history
        and len(example.candidates) >= min_candidates
        and (
            max_candidates is None
            or len(example.candidates) <= max_candidates
        )
        and (not require_positive or bool(example.positive_indices))
    ]


def _stable_score(example: MindExample, *, seed: int, purpose: str) -> bytes:
    material = (
        f"{purpose}\0{seed}\0{example.impression_id}\0{example.user_id}\0"
        f"{example.impression_time}"
    ).encode("utf-8")
    return hashlib.sha256(material).digest()


def deterministic_sample(
    examples: Sequence[MindExample],
    sample_size: int | None,
    *,
    seed: int = 0,
) -> list[MindExample]:
    """Choose a stable exact-size sample and retain source-file order."""

    values = list(examples)
    if sample_size is None:
        return values
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative or None")
    if sample_size >= len(values):
        return values

    ranked_indices = sorted(
        range(len(values)),
        key=lambda index: (
            _stable_score(values[index], seed=seed, purpose="sample"),
            index,
        ),
    )
    selected = set(ranked_indices[:sample_size])
    return [value for index, value in enumerate(values) if index in selected]


def deterministic_split(
    examples: Sequence[MindExample],
    *,
    validation_fraction: float,
    seed: int = 0,
) -> tuple[list[MindExample], list[MindExample]]:
    """Return an exact-size, stable train/validation split in source order."""

    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    values = list(examples)
    validation_size = int(len(values) * validation_fraction + 0.5)
    ranked_indices = sorted(
        range(len(values)),
        key=lambda index: (
            _stable_score(values[index], seed=seed, purpose="split"),
            index,
        ),
    )
    validation_indices = set(ranked_indices[:validation_size])
    training = [
        value for index, value in enumerate(values) if index not in validation_indices
    ]
    validation = [
        value for index, value in enumerate(values) if index in validation_indices
    ]
    return training, validation
