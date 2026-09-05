#!/usr/bin/env python3
"""Score JSONL completions produced for prepared MIND examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openbench_rerank_rl.evaluation import aggregate_evaluations, evaluate_prediction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional row-level scored JSONL path",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    evaluations = []
    scored_rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            completion = row.get("completion", row.get("prediction"))
            if not isinstance(completion, str):
                raise ValueError(
                    f"{args.input}:{line_number}: expected string completion or prediction"
                )
            positives = row.get("positive_indices")
            slate_k = row.get("k", row.get("candidate_count"))
            if not isinstance(positives, list) or not all(
                type(item) is int for item in positives
            ):
                raise ValueError(
                    f"{args.input}:{line_number}: positive_indices must be an integer list"
                )
            if type(slate_k) is not int:
                raise ValueError(f"{args.input}:{line_number}: k must be an integer")
            evaluation = evaluate_prediction(
                completion,
                positives=set(positives),
                slate_k=slate_k,
            )
            evaluations.append(evaluation)
            scored_rows.append({**row, "evaluation": evaluation.as_dict()})

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            for row in scored_rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
    print(json.dumps(aggregate_evaluations(evaluations), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
