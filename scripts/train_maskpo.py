#!/usr/bin/env python3
"""Train Qwen3-1.7B from the archived Phase-2 rubric SFT adapter."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from openbench_rerank_rl.advantages import MaskPOConfig
from openbench_rerank_rl.trainer import (
    HuggingFacePolicyBackend,
    SamplingConfig,
    TrainingExample,
    load_huggingface_maskpo_trainer,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/maskpo_qwen3_1p7b.yaml"),
    )
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--device-map",
        help="Transformers device map for the actor (default: config or auto)",
    )
    parser.add_argument(
        "--reference-device-map",
        help="optional separate Transformers device map for the frozen reference",
    )
    return parser


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section {name!r} must be a mapping")
    return value


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Training configuration requires PyYAML; install `.[training]`"
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("top-level training configuration must be a mapping")
    for required in ("model", "data", "generation", "maskpo", "optimization", "output"):
        _mapping(value.get(required), required)
    return value


def iter_training_examples(path: Path) -> Iterator[TrainingExample]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            prompt = row.get("prompt")
            positives = row.get("positive_indices")
            slate_k = row.get("k")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"{path}:{line_number}: missing prompt; rerun prepare_mind.py "
                    "with --include-prompts"
                )
            if not isinstance(positives, list) or not all(
                type(item) is int for item in positives
            ):
                raise ValueError(
                    f"{path}:{line_number}: positive_indices must be an integer list"
                )
            if type(slate_k) is not int:
                raise ValueError(f"{path}:{line_number}: k must be an integer")
            yield TrainingExample(
                prompt=prompt,
                positives=frozenset(positives),
                slate_k=slate_k,
                example_id=str(row.get("id", line_number)),
            )


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_canonical(
    section: Mapping[str, Any], key: str, expected: object
) -> None:
    value = section.get(key, expected)
    if value != expected:
        raise ValueError(
            f"the latest MaskPO runtime requires {key}={expected!r}; got {value!r}"
        )


def _save_actor(trainer: Any, destination: Path) -> None:
    actor = trainer.actor
    if not isinstance(actor, HuggingFacePolicyBackend):
        raise TypeError("the command-line trainer requires a HuggingFacePolicyBackend")
    actor.save_pretrained(destination)


def _diagnostic_row(step: int, result: Any) -> dict[str, object]:
    region_counts = Counter(
        region for route in result.routes for region in route.regions
    )
    return {
        "step": step,
        "example_id": result.example_id,
        "loss": result.loss,
        "policy_loss": result.policy_loss,
        "kl": result.kl,
        "clip_fraction": result.clip_fraction,
        "active_original_tokens": result.active_token_count,
        "optimizer_step": result.optimizer_step,
        "optimizer_stepped": result.optimizer_stepped,
        "valid_counterfactuals": result.query_credit.valid_probe_count,
        "counterfactual_slots": sum(
            len(rollout.probes) for rollout in result.query_credit.rollouts
        ),
        "rubric_rms_scale": result.query_credit.rubric_rms_scale,
        "routing_fallbacks": result.routing_fallback_count,
        "unavailable_rubric_bodies": sum(
            len(route.unavailable_rubrics) for route in result.routes
        ),
        "routed_token_regions": dict(sorted(region_counts.items())),
        "sequence_rewards": [
            rollout.sequence_reward for rollout in result.query_credit.rollouts
        ],
        "ranking_rewards": [
            rollout.rank_reward for rollout in result.query_credit.rollouts
        ],
        "format_rewards": [
            rollout.format_reward for rollout in result.query_credit.rollouts
        ],
    }


def main() -> None:
    args = _parser().parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    generation = _mapping(config["generation"], "generation")
    maskpo = _mapping(config["maskpo"], "maskpo")
    optimization = _mapping(config["optimization"], "optimization")
    output = _mapping(config["output"], "output")

    # These settings describe fixed code paths rather than tunable
    # hyperparameters.  Reject a configuration that would otherwise claim to
    # run a variant the runtime does not implement.
    for key, expected in (
        ("masked_placeholder", "[MASKED_RUBRIC_CONTENT]"),
        ("counterfactual_boundary", "synthesis_suffix_only"),
        ("lenient_integer_list_grading", True),
        ("positive_first_occurrence_only", True),
        ("format_max_reward", 0.1),
        ("rubric_delta_normalization", "rms_no_centering"),
        ("invalid_probe_body_fallback", "sequence"),
    ):
        _require_canonical(maskpo, key, expected)
    _require_canonical(optimization, "objective", "tokenwise_bnpo")

    train_file = Path(args.train_file or str(data["train_file"])).expanduser().resolve()
    output_dir = Path(args.output_dir or str(output["directory"])).expanduser().resolve()
    max_steps = args.max_steps or _positive_int(optimization["max_steps"], "max_steps")
    seed = int(optimization.get("seed", 42))

    random.seed(seed)
    try:
        import torch
        from transformers import set_seed
    except ImportError as exc:
        raise SystemExit("Install the training dependencies with `.[training]`") from exc
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    algorithm = MaskPOConfig(
        metric=str(maskpo.get("rank_metric", "ndcg")),
        num_siblings=int(generation.get("num_siblings", 4)),
        num_rubrics=int(maskpo.get("num_rubrics", 4)),
        tau_mask=float(maskpo.get("tau_mask", 0.05)),
        mask_clip=float(maskpo.get("mask_clip", 2.0)),
        lambda_rank=float(maskpo.get("lambda_rank", 1.0)),
        lambda_mask=float(maskpo.get("lambda_mask", 0.5)),
    )
    sampling = SamplingConfig(
        do_sample=bool(generation.get("do_sample", True)),
        temperature=float(generation.get("temperature", 0.6)),
        top_k=int(generation.get("top_k", 20)),
        top_p=float(generation.get("top_p", 0.95)),
        max_new_tokens=int(generation.get("max_new_tokens", 2048)),
        counterfactual_max_new_tokens=int(
            generation.get("counterfactual_max_new_tokens", 2048)
        ),
        original_batch_size=int(generation.get("original_batch_size", 4)),
        counterfactual_batch_size=int(
            generation.get("counterfactual_batch_size", 4)
        ),
    )

    device_map: object = args.device_map or model.get("device_map", "auto")
    reference_device_map: object | None = (
        args.reference_device_map
        if args.reference_device_map is not None
        else model.get("reference_device_map")
    )
    accumulation_steps = int(optimization.get("gradient_accumulation_steps", 2))
    effective_batch_size = int(
        optimization.get(
            "effective_original_batch_size",
            algorithm.num_siblings * accumulation_steps,
        )
    )
    if effective_batch_size != algorithm.num_siblings * accumulation_steps:
        raise ValueError(
            "effective_original_batch_size must equal num_siblings * "
            "gradient_accumulation_steps"
        )

    trainer = load_huggingface_maskpo_trainer(
        model_name_or_path=str(model["name_or_path"]),
        adapter_path=str(Path(str(model["adapter_path"])).expanduser().resolve()),
        learning_rate=float(optimization.get("learning_rate", 1e-5)),
        weight_decay=float(optimization.get("weight_decay", 0.0)),
        dtype=str(model.get("dtype", "bfloat16")),
        trust_remote_code=bool(model.get("trust_remote_code", False)),
        device_map=device_map,
        reference_device_map=reference_device_map,
        gradient_checkpointing=bool(model.get("gradient_checkpointing", True)),
        maskpo_config=algorithm,
        sampling_config=sampling,
        ppo_clip=float(optimization.get("ppo_clip", 0.2)),
        reference_kl_coefficient=float(
            optimization.get("reference_kl_coefficient", 0.001)
        ),
        normalization_epsilon=float(maskpo.get("normalization_eps", 1e-8)),
        gradient_accumulation_steps=accumulation_steps,
        max_grad_norm=(
            None
            if optimization.get("max_grad_norm") is None
            else float(optimization["max_grad_norm"])
        ),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "run_config.yaml")
    metadata = {
        "config": str(config_path),
        "train_file": str(train_file),
        "seed": seed,
        "requested_steps": max_steps,
        "num_siblings": algorithm.num_siblings,
        "effective_original_batch_size": effective_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "torch": torch.__version__,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    log_steps = _positive_int(output.get("log_steps", 1), "log_steps")
    save_steps = _positive_int(output.get("save_steps", 25), "save_steps")
    completed = 0
    log_path = output_dir / "train_log.jsonl"
    with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
        for completed, example in enumerate(iter_training_examples(train_file), start=1):
            if completed > max_steps:
                completed -= 1
                break
            result = trainer.train_query(example)
            diagnostic = _diagnostic_row(completed, result)
            log_handle.write(
                json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            log_handle.flush()
            if completed % log_steps == 0:
                print(json.dumps(diagnostic, sort_keys=True))
            if completed % save_steps == 0:
                _save_actor(trainer, output_dir / f"checkpoint-step-{completed}")

    trainer.flush_gradients()
    _save_actor(trainer, output_dir / "final-adapter")
    final_state = {
        **metadata,
        "completed_steps": completed,
        "optimizer_steps": trainer.optimizer_steps,
    }
    (output_dir / "final_state.json").write_text(
        json.dumps(final_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if completed < max_steps:
        print(
            f"Training input ended after {completed} rows; requested {max_steps}.",
            flush=True,
        )
    print(f"Saved final adapter to {output_dir / 'final-adapter'}")


if __name__ == "__main__":
    main()
