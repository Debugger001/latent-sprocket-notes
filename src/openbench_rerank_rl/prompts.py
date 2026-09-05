"""Prompt templates for public MIND Rank-MaskPO experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .mind import MindExample, MindNews


SYNTHESIS_HEADER = "**Synthesis:**"


@dataclass(frozen=True, slots=True)
class Rubric:
    key: str
    header: str
    description: str


# These are the complete four descriptions used by the archived rubric teacher
# prompt.  Keep them centralized: the same headers delimit semantic token spans
# and define which body is replaced for each counterfactual probe.
RUBRICS: tuple[Rubric, ...] = (
    Rubric(
        key="section_topic_affinity",
        header="**Section And Topic Affinity:**",
        description=(
            "infer stable category/subcategory and broad subject preferences from "
            "the clicked-news history, such as sports, politics, finance, "
            "entertainment, lifestyle, health, food, or autos. Keep this to broad "
            "taxonomy preference, not specific people, events, or headline angle."
        ),
    ),
    Rubric(
        key="entity_storyline_continuity",
        header="**Entity And Storyline Continuity:**",
        description=(
            "compare candidates to recurring people, teams, organizations, places, "
            "franchises, public figures, events, and developing stories inferred "
            "from titles, abstracts, and world knowledge. Keep this to specific "
            "semantic continuity, independent of broad topic preference."
        ),
    ),
    Rubric(
        key="candidate_angle_marginal_novelty",
        header="**Candidate Angle And Marginal Novelty:**",
        description=(
            "judge whether each candidate adds a new, useful, or clickable angle "
            "relative to the user's history and the candidate slate, such as update "
            "versus repeat, explainer versus shallow duplicate, practical advice "
            "versus generic mention, or specific headline promise versus vague "
            "article. Keep this to marginal article value, not whether the "
            "topic/entity is familiar."
        ),
    ),
    Rubric(
        key="temporal_session_intent",
        header="**Temporal And Session Intent:**",
        description=(
            "use clicked-history order, recent clicks, impression time, and "
            "time-sensitive cues in titles/abstracts to infer short-term intent and "
            "freshness needs. Do not infer exact article age unless it is exposed by "
            'text like "live", "today", or "latest", or by the impression context.'
        ),
    ),
)


def _display_text(value: str) -> str:
    """Keep one article on one line without changing its words."""

    return " ".join(value.split())


def _article_line(article: MindNews, prefix: str) -> str:
    category = _display_text(article.category)
    subcategory = _display_text(article.subcategory)
    title = _display_text(article.title)
    abstract = _display_text(article.abstract)
    return f"{prefix} [{category} / {subcategory}] {title} -- {abstract}"


def build_reranking_prompt(example: MindExample) -> str:
    """Build the latest four-rubric MIND reranking prompt.

    Candidate click labels and news IDs are intentionally unreachable here: only
    category, subcategory, title, and abstract are rendered.  The format example
    is a literal schema placeholder rather than an arbitrary ranking that could
    bias the model toward the demonstrated order.
    """

    k = example.k
    if k < 1:
        raise ValueError("a reranking prompt requires at least one candidate")

    rubric_instructions = "\n".join(
        f"- {rubric.header} {rubric.description}" for rubric in RUBRICS
    )
    rubric_example = "\n".join(f"{rubric.header} ..." for rubric in RUBRICS)
    if example.history:
        history_lines = "\n".join(
            _article_line(article, f"H{index}.")
            for index, article in enumerate(example.history, start=1)
        )
    else:
        history_lines = "(none)"
    candidate_lines = "\n".join(
        _article_line(article, f"{index}.")
        for index, article in enumerate(example.candidates, start=1)
    )

    return (
        "You are ranking a logged news impression slate for a user.\n"
        "Given the user's clicked-news history, the impression timestamp, and the "
        "candidate news articles, rank candidates by how likely the user is to "
        "click/read them in this impression.\n"
        "Use only the candidate indices shown below; do not invent news items.\n\n"
        "Reason using exactly the rubric headers below inside <think>...</think>, "
        "then return the final ranking inside <answer>...</answer>. The answer must "
        "be one JSON-style list of candidate indices. Return a JSON-style list "
        "containing every candidate index from 1 to K exactly once, ordered from "
        "most likely clicked/read to least likely. Do not omit, duplicate, or invent "
        f"indices. For this row, K={k}, so return exactly {k} indices. Use exactly "
        "one newline between </think> and <answer>, and stop immediately after "
        "</answer>.\n"
        "Use these independent, maskable rubric sections before synthesis. Each "
        "section should focus on its own evidence source, avoid restating the other "
        "rubrics, and leave final cross-rubric tradeoffs for **Synthesis:**.\n"
        f"{rubric_instructions}\n\n"
        "Put final cross-rubric tradeoffs, candidate ordering, and confidence only "
        "in **Synthesis:**.\n\n"
        "Example format:\n"
        "<think>\n"
        f"{rubric_example}\n"
        "**Synthesis:** ...\n"
        "</think>\n"
        "<answer>\n"
        "[permutation of 1 through K]\n"
        "</answer>\n\n"
        f"Impression time: {_display_text(example.impression_time)}\n\n"
        "Clicked-news history, oldest to newest:\n"
        f"{history_lines}\n\n"
        "Candidate news articles:\n"
        f"{candidate_lines}"
    )
