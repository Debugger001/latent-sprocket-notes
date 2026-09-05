#!/usr/bin/env python3
"""Prepare deterministic JSONL splits from a local public MIND download.

This script reads the original TSVs in place and writes joined, filtered
examples.  It never copies the raw MIND files into the repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openbench_rerank_rl.mind import (
    DEFAULT_MAX_HISTORY,
    MindExample,
    MindNews,
    deterministic_sample,
    deterministic_split,
    filter_examples,
    load_examples,
)
from openbench_rerank_rl.prompts import build_reranking_prompt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news", type=Path, required=True, help="path to MIND news.tsv")
    parser.add_argument(
        "--behaviors", type=Path, required=True, help="path to MIND behaviors.tsv"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-history", type=int, default=DEFAULT_MAX_HISTORY)
    parser.add_argument("--min-history", type=int, default=0)
    parser.add_argument("--min-candidates", type=int, default=1)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument(
        "--allow-no-positive",
        action="store_true",
        help="retain impressions with no clicked candidate",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="take an exact-size stable sample after filtering",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="stable held-out fraction (default: 0, preserving official splits)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-prompts",
        action="store_true",
        help="include the latest four-rubric prompt in each output row",
    )
    return parser


def _article_dict(article: MindNews) -> dict[str, str]:
    # ``MindNews.prompt_dict`` is intentionally the allowlist here.  URLs and
    # entity annotations are unnecessary for this experiment and are omitted.
    return article.prompt_dict()


def _example_dict(example: MindExample, *, include_prompt: bool) -> dict[str, object]:
    row: dict[str, object] = {
        "id": example.impression_id,
        "dataset": "mind",
        "user_id": example.user_id,
        "impression_time": example.impression_time,
        "history": [_article_dict(article) for article in example.history],
        "candidates": [_article_dict(article) for article in example.candidates],
        "positive_indices": list(example.positive_indices),
        "positive_news_ids": list(example.positive_news_ids),
        "k": example.k,
    }
    if include_prompt:
        row["prompt"] = build_reranking_prompt(example)
    return row


def _write_jsonl(
    path: Path,
    examples: list[MindExample],
    *,
    include_prompts: bool,
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            row = _example_dict(example, include_prompt=include_prompts)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    args = _parser().parse_args()
    if args.max_history < 0:
        raise SystemExit("--max-history must be non-negative")
    if not 0.0 <= args.validation_fraction <= 1.0:
        raise SystemExit("--validation-fraction must be between 0 and 1")

    examples = load_examples(
        args.news,
        args.behaviors,
        max_history=args.max_history,
    )
    filtered = filter_examples(
        examples,
        min_history=args.min_history,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
        require_positive=not args.allow_no_positive,
    )
    sampled = deterministic_sample(filtered, args.sample_size, seed=args.seed)
    train, validation = deterministic_split(
        sampled,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.output_dir / "train.jsonl",
        train,
        include_prompts=args.include_prompts,
    )
    _write_jsonl(
        args.output_dir / "validation.jsonl",
        validation,
        include_prompts=args.include_prompts,
    )
    summary = {
        "source_rows": len(examples),
        "filtered_rows": len(filtered),
        "sampled_rows": len(sampled),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "seed": args.seed,
        "max_history": args.max_history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
