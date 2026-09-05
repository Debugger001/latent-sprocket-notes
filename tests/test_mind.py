import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from openbench_rerank_rl.mind import (
    MindFormatError,
    build_example,
    deterministic_sample,
    deterministic_split,
    deterministic_training_order,
    filter_examples,
    iter_behaviors_tsv,
    load_examples,
    load_news_tsv,
    parse_behavior_line,
    parse_news_line,
)


FIXTURES = Path(__file__).parent / "fixtures" / "mind"


def test_news_requires_exactly_eight_fields():
    article = parse_news_line("N1\tcat\tsub\ttitle\tabstract\turl\t[]\t[]\n")
    assert article.news_id == "N1"
    assert article.prompt_dict() == {
        "category": "cat",
        "subcategory": "sub",
        "title": "title",
        "abstract": "abstract",
    }
    with pytest.raises(MindFormatError, match="exactly 8"):
        parse_news_line("N1\tcat\tsub\ttitle\tabstract\turl\t[]\n")
    with pytest.raises(MindFormatError, match="exactly 8"):
        parse_news_line("N1\tcat\tsub\ttitle\tabstract\turl\t[]\t[]\textra\n")


def test_behavior_requires_five_fields_and_labeled_candidates():
    behavior = parse_behavior_line(
        "I1\tU1\t11/13/2019 1:16:51 PM\tN1 N2\tN3-1 N4-0\n"
    )
    assert behavior.history_news_ids == ("N1", "N2")
    assert behavior.candidate_news_ids == ("N3", "N4")
    assert behavior.positive_indices == (1,)
    with pytest.raises(MindFormatError, match="exactly 5"):
        parse_behavior_line("I1\tU1\ttime\tN1\n")
    with pytest.raises(MindFormatError, match="must end in -0 or -1"):
        parse_behavior_line("I1\tU1\ttime\tN1\tN2-x\n")


def test_join_preserves_candidate_order_all_positives_and_recent_history():
    news = load_news_tsv(FIXTURES / "news.tsv.fixture")
    behavior = next(iter_behaviors_tsv(FIXTURES / "behaviors.tsv.fixture"))
    example = build_example(behavior, news, max_history=2)

    assert [article.news_id for article in example.history] == ["N3", "N4"]
    assert [article.news_id for article in example.candidates] == [
        "N3",
        "N1",
        "N5",
        "N2",
    ]
    assert example.positive_indices == (2, 4)
    assert example.positive_news_ids == ("N1", "N2")


def test_filters_sampling_and_split_are_stable_and_order_preserving():
    examples = load_examples(
        FIXTURES / "news.tsv.fixture",
        FIXTURES / "behaviors.tsv.fixture",
        max_history=50,
    )
    filtered = filter_examples(examples, min_history=1, require_positive=True)
    assert [example.impression_id for example in filtered] == ["I1", "I2"]

    sample_a = deterministic_sample(examples, 2, seed=17)
    sample_b = deterministic_sample(examples, 2, seed=17)
    assert sample_a == sample_b
    original_positions = {example.impression_id: i for i, example in enumerate(examples)}
    assert [original_positions[x.impression_id] for x in sample_a] == sorted(
        original_positions[x.impression_id] for x in sample_a
    )

    train_a, validation_a = deterministic_split(
        examples, validation_fraction=1 / 3, seed=29
    )
    train_b, validation_b = deterministic_split(
        examples, validation_fraction=1 / 3, seed=29
    )
    assert (train_a, validation_a) == (train_b, validation_b)
    assert len(train_a) == 2
    assert len(validation_a) == 1
    assert {x.impression_id for x in train_a}.isdisjoint(
        {x.impression_id for x in validation_a}
    )


def test_training_order_is_repeatable_seed_sensitive_and_membership_preserving():
    template = load_examples(
        FIXTURES / "news.tsv.fixture",
        FIXTURES / "behaviors.tsv.fixture",
    )[0]
    examples = [
        replace(
            template,
            impression_id=f"I{index:02d}",
            user_id=f"U{index % 7}",
            impression_time=f"11/{index + 1}/2019 1:00:00 PM",
        )
        for index in range(32)
    ]
    train, validation = deterministic_split(
        examples,
        validation_fraction=0.25,
        seed=29,
    )

    ordered_a = deterministic_training_order(train, seed=123)
    ordered_b = deterministic_training_order(train, seed=123)
    ordered_other_seed = deterministic_training_order(train, seed=124)

    assert ordered_a == ordered_b
    assert ordered_a != ordered_other_seed
    assert {row.impression_id for row in ordered_a} == {
        row.impression_id for row in train
    }
    assert len(ordered_a) == len(train)
    # Ordering happens after the split and neither mutates nor reorders validation.
    assert validation == deterministic_split(
        examples,
        validation_fraction=0.25,
        seed=29,
    )[1]


def test_prepare_cli_is_reproducible_and_does_not_copy_raw_files(tmp_path):
    root = Path(__file__).parents[1]
    script = root / "scripts" / "prepare_mind.py"
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    base_command = [
        sys.executable,
        str(script),
        "--news",
        str(FIXTURES / "news.tsv.fixture"),
        "--behaviors",
        str(FIXTURES / "behaviors.tsv.fixture"),
        "--validation-fraction",
        "0.5",
        "--seed",
        "123",
        "--include-prompts",
    ]
    environment = os.environ.copy()
    source_path = str(root / "src")
    environment["PYTHONPATH"] = source_path + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.run(
        base_command + ["--output-dir", str(output_a)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        base_command + ["--output-dir", str(output_b)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    for filename in ("train.jsonl", "validation.jsonl", "summary.json"):
        assert (output_a / filename).read_bytes() == (output_b / filename).read_bytes()
    assert not (output_a / "news.tsv").exists()
    assert not (output_a / "behaviors.tsv").exists()
    rows = [
        json.loads(line)
        for path in (output_a / "train.jsonl", output_a / "validation.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2  # no-positive I3 is filtered by default
    assert all("prompt" in row for row in rows)


def test_prepare_cli_shuffle_only_changes_training_order(tmp_path):
    root = Path(__file__).parents[1]
    script = root / "scripts" / "prepare_mind.py"
    behaviors = tmp_path / "behaviors.tsv"
    behaviors.write_text(
        "".join(
            f"I{index:02d}\tU{index % 7}\t11/{index + 1}/2019 1:00:00 PM"
            "\tN1 N2\tN2-1 N3-0\n"
            for index in range(32)
        ),
        encoding="utf-8",
    )
    output_source_order = tmp_path / "source-order"
    output_shuffled = tmp_path / "shuffled"
    base_command = [
        sys.executable,
        str(script),
        "--news",
        str(FIXTURES / "news.tsv.fixture"),
        "--behaviors",
        str(behaviors),
        "--validation-fraction",
        "0.25",
        "--seed",
        "123",
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.run(
        base_command + ["--output-dir", str(output_source_order)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        base_command
        + ["--shuffle-training", "--output-dir", str(output_shuffled)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    source_train = [
        json.loads(line)["id"]
        for line in (output_source_order / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    shuffled_train = [
        json.loads(line)["id"]
        for line in (output_shuffled / "train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert shuffled_train != source_train
    assert set(shuffled_train) == set(source_train)
    assert (output_shuffled / "validation.jsonl").read_bytes() == (
        output_source_order / "validation.jsonl"
    ).read_bytes()
    assert "shuffle_training" not in json.loads(
        (output_source_order / "summary.json").read_text(encoding="utf-8")
    )
    assert json.loads(
        (output_shuffled / "summary.json").read_text(encoding="utf-8")
    )["shuffle_training"] is True
