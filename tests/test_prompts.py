from openbench_rerank_rl.mind import MindExample, MindNews
from openbench_rerank_rl.parsers import DEFAULT_RUBRIC_HEADERS
from openbench_rerank_rl.prompts import (
    RUBRICS,
    SYNTHESIS_HEADER,
    build_reranking_prompt,
)


def _news(
    news_id: str,
    category: str,
    subcategory: str,
    title: str,
    abstract: str,
) -> MindNews:
    return MindNews(
        news_id=news_id,
        category=category,
        subcategory=subcategory,
        title=title,
        abstract=abstract,
        url=f"SECRET_URL_{news_id}",
        title_entities=f"SECRET_TITLE_ENTITIES_{news_id}",
        abstract_entities=f"SECRET_ABSTRACT_ENTITIES_{news_id}",
    )


def _example() -> MindExample:
    return MindExample(
        impression_id="I-secret",
        user_id="U-secret",
        impression_time="11/13/2019 1:16:51 PM",
        history=(
            _news("NH", "news", "politics", "History title", "History abstract"),
        ),
        candidates=(
            _news("N1", "food", "recipes", "Candidate one", "Abstract one"),
            _news("N2", "sports", "nba", "Candidate two", "Abstract two"),
            _news("N3", "weather", "forecast", "Candidate three", "Abstract three"),
        ),
        positive_indices=(2,),
        positive_news_ids=("N2",),
    )


def test_prompt_has_complete_archived_rubrics_and_latest_schema():
    prompt = build_reranking_prompt(_example())

    assert "K=3, so return exactly 3 indices" in prompt
    assert "[permutation of 1 through K]" in prompt
    assert "[3, 7, 1, 2, 4, 5, 6, 8, 9, 10]" not in prompt
    assert "from 1 to K exactly once" in prompt
    assert "<think>" in prompt and "</think>\n<answer>" in prompt
    assert SYNTHESIS_HEADER in prompt
    for rubric in RUBRICS:
        assert prompt.count(rubric.header) == 2  # instruction plus format example
        assert rubric.description in prompt
    assert tuple(rubric.header for rubric in RUBRICS) == DEFAULT_RUBRIC_HEADERS


def test_prompt_only_exposes_allowed_article_fields_and_never_labels():
    prompt = build_reranking_prompt(_example())

    for expected in (
        "[news / politics] History title -- History abstract",
        "1. [food / recipes] Candidate one -- Abstract one",
        "2. [sports / nba] Candidate two -- Abstract two",
    ):
        assert expected in prompt
    for forbidden in (
        "SECRET_URL",
        "SECRET_TITLE_ENTITIES",
        "SECRET_ABSTRACT_ENTITIES",
        "I-secret",
        "U-secret",
        "positive_indices",
        "N2-1",
    ):
        assert forbidden not in prompt
