#!/usr/bin/env python3
"""Train Qwen3-1.7B from the archived Phase-2 rubric SFT adapter."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from openbench_rerank_rl.advantages import MaskPOConfig
from openbench_rerank_rl.checkpointing import (
    CHECKPOINT_VERSION,
    ValidatedCheckpoint,
    capture_rng_state,
    durable_log_offsets,
    find_latest_checkpoint,
    preserve_rng_state,
    restore_rng_state,
    sha256_file,
    sha256_python_tree,
    sha256_tree,
    trainable_parameter_schema,
    truncate_scoped_jsonl_logs,
    validate_checkpoint,
    write_checkpoint,
)
from openbench_rerank_rl.trainer import (
    HuggingFacePolicyBackend,
    SamplingConfig,
    TrainingExample,
    load_huggingface_maskpo_trainer,
)
from openbench_rerank_rl.tracking import (
    ExperimentTracker,
    MetricValue,
    init_wandb_tracker,
    parse_wandb_settings,
)
from openbench_rerank_rl.validation import (
    ValidationResult,
    fixed_validation_examples,
    run_greedy_validation,
    validation_fingerprint,
)


MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/maskpo_qwen3_1p7b.yaml"),
    )
    parser.add_argument("--train-file", type=Path)
    parser.add_argument(
        "--max-rollout-steps",
        "--max-steps",
        dest="max_rollout_steps",
        type=int,
        help="number of fresh rollout batches (the --max-steps alias has the same meaning)",
    )
    parser.add_argument(
        "--max-query-steps",
        type=int,
        help="optional prompt-group cap, primarily for a partial-window smoke test",
    )
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="skip held-out generation for a short smoke test",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("disabled", "offline", "online"),
        help="override the YAML W&B mode; use disabled for a local smoke test",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--resume-from-checkpoint",
        metavar="PATH|latest",
        help=(
            "resume a matching complete checkpoint; 'latest' selects the newest "
            "checkpoint under the configured output directory"
        ),
    )
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


def iter_training_examples(
    path: Path,
    *,
    max_slate_size: int | None = None,
) -> Iterator[TrainingExample]:
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
            if max_slate_size is not None and slate_k > max_slate_size:
                raise ValueError(
                    f"{path}:{line_number}: k={slate_k} exceeds the configured "
                    f"RL boundary K<={max_slate_size}; filter the prepared data"
                )
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


def _device_runtime_fingerprints(
    actor_device_map: object,
    reference_device_map: object | None,
) -> dict[str, str]:
    """Record device routing that can affect exact CUDA-RNG continuation."""

    effective_reference_map = (
        actor_device_map if reference_device_map is None else reference_device_map
    )

    def stable_json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    return {
        "actor_device_map_json": stable_json(actor_device_map),
        "reference_device_map_json": stable_json(effective_reference_map),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }


def _save_actor(trainer: Any, destination: Path) -> None:
    actor = trainer.actor
    if not isinstance(actor, HuggingFacePolicyBackend):
        raise TypeError("the command-line trainer requires a HuggingFacePolicyBackend")
    actor.save_pretrained(destination)


def _diagnostic_row(query_step: int, result: Any) -> dict[str, object]:
    region_counts = Counter(
        region for route in result.routes for region in route.regions
    )
    fallback_reasons = Counter(
        route.fallback_reason
        for route in result.routes
        if route.used_sequence_fallback and route.fallback_reason is not None
    )
    return {
        "query_step": query_step,
        "example_id": result.example_id,
        "loss": result.loss,
        "policy_loss": result.policy_loss,
        "kl": result.kl,
        "clip_fraction": result.clip_fraction,
        "active_original_tokens": result.active_token_count,
        "optimizer_step": result.optimizer_step,
        "optimizer_stepped": result.optimizer_stepped,
        "optimizer_steps_applied": result.optimizer_steps_applied,
        "rollout_step": result.rollout_step,
        "rollout_step_completed": result.rollout_step_completed,
        "ppo_pass_metrics": [
            {
                "ppo_pass": item.ppo_pass,
                "loss": item.loss,
                "policy_loss": item.policy_loss,
                "kl": item.kl,
                "clip_fraction": item.clip_fraction,
                "active_token_count": item.active_token_count,
            }
            for item in result.ppo_pass_metrics
        ],
        "valid_counterfactuals": result.query_credit.valid_probe_count,
        "counterfactual_slots": sum(
            len(rollout.probes) for rollout in result.query_credit.rollouts
        ),
        "rubric_rms_scale": result.query_credit.rubric_rms_scale,
        "routing_fallbacks": result.routing_fallback_count,
        "routing_fallback_reasons": dict(sorted(fallback_reasons.items())),
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


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _rollout_tracking_metrics(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    ppo_pass_metrics: Sequence[Any],
    completed_queries: int,
    num_siblings: int,
    optimizer_step: int,
    ppo_passes: int,
) -> dict[str, MetricValue]:
    """Reduce one fresh rollout window to scalar, non-sensitive metrics."""

    if not diagnostics:
        raise ValueError("cannot aggregate an empty rollout window")
    if not ppo_pass_metrics:
        raise ValueError("completed rollout window has no PPO-pass metrics")

    def scalar_mean(key: str) -> float:
        return _mean([float(row[key]) for row in diagnostics])

    def flattened_mean(key: str) -> float:
        values = [
            float(value)
            for row in diagnostics
            for value in row[key]  # type: ignore[union-attr]
        ]
        return _mean(values)

    valid_probes = sum(int(row["valid_counterfactuals"]) for row in diagnostics)
    probe_slots = sum(int(row["counterfactual_slots"]) for row in diagnostics)
    active_tokens = int(ppo_pass_metrics[0].active_token_count)
    final_pass = ppo_pass_metrics[-1]
    metrics: dict[str, MetricValue] = {
        "train/loss": float(final_pass.loss),
        "train/policy_loss": float(final_pass.policy_loss),
        "train/kl": float(final_pass.kl),
        "train/clip_fraction": float(final_pass.clip_fraction),
        "train/query_pass_1_loss_mean": scalar_mean("loss"),
        "train/active_original_tokens": active_tokens,
        "train/valid_counterfactual_rate": (
            valid_probes / probe_slots if probe_slots else 0.0
        ),
        "train/rubric_rms_scale": scalar_mean("rubric_rms_scale"),
        "train/routing_fallbacks": sum(
            int(row["routing_fallbacks"]) for row in diagnostics
        ),
        "train/unavailable_rubric_bodies": sum(
            int(row["unavailable_rubric_bodies"]) for row in diagnostics
        ),
        "train/sequence_reward": flattened_mean("sequence_rewards"),
        "train/ranking_reward": flattened_mean("ranking_rewards"),
        "train/format_reward": flattened_mean("format_rewards"),
        "progress/query_steps": completed_queries,
        "progress/original_rollouts": completed_queries * num_siblings,
        "progress/optimizer_step": optimizer_step,
        "progress/ppo_passes": ppo_passes,
    }
    for item in ppo_pass_metrics:
        prefix = f"train/ppo_pass_{item.ppo_pass}"
        metrics[f"{prefix}_loss"] = float(item.loss)
        metrics[f"{prefix}_policy_loss"] = float(item.policy_loss)
        metrics[f"{prefix}_kl"] = float(item.kl)
        metrics[f"{prefix}_clip_fraction"] = float(item.clip_fraction)
    return metrics


def _validation_tracking_metrics(result: ValidationResult) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        "validation/rows": result.rows,
        "validation/ndcg": result.ndcg,
        "validation/format_reward": result.format_reward,
        "validation/format_rate": result.format_rate,
        "validation/format_envelope_rate": result.envelope_rate,
        "validation/format_synthesis_rate": result.synthesis_rate,
        "validation/parse_rate": result.parse_rate,
        "validation/format_valid_unique_ids_rate": result.valid_unique_ids_rate,
        "validation/exact_permutation_rate": result.exact_permutation_rate,
    }
    for index, rate in enumerate(result.rubric_header_rates, start=1):
        metrics[f"validation/format_rubric_header_{index}_rate"] = rate
    return metrics


def _log_validation_result(
    handle: Any,
    *,
    rollout_step: int,
    optimizer_step: int,
    result: ValidationResult,
) -> None:
    row = {
        "rollout_step": rollout_step,
        "optimizer_step": optimizer_step,
        **result.as_dict(),
    }
    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()


def _safe_tracking_config(
    *,
    model_name: str,
    algorithm: MaskPOConfig,
    sampling: SamplingConfig,
    optimization: Mapping[str, Any],
    max_rollout_steps: int,
    prompt_batch_size: int,
    original_batch_size: int,
    ppo_passes: int,
    max_slate_size: int,
    validation_rows: int,
    validation_interval: int,
    validation_batch_size: int,
    validation_max_new_tokens: int,
    validation_dataset_fingerprint: str | None,
    normalization_epsilon: float,
    checkpoint_rollout_steps: int,
    keep_checkpoints: int,
    actor_device_map: object,
    reference_device_map: object | None,
    input_fingerprints: Mapping[str, str],
    model_revision: str,
    resume_rollout_step: int | None,
    resume_manifest_sha256: str | None,
) -> dict[str, object]:
    """Whitelist public run metadata; never forward raw YAML or file paths."""

    return {
        "model": Path(model_name).name,
        "model_revision": model_revision,
        "starting_adapter": "p2_rubric_reasoning_sft",
        "learning_rate": float(optimization.get("learning_rate", 1e-5)),
        "reference_kl_coefficient": float(
            optimization.get("reference_kl_coefficient", 0.001)
        ),
        "weight_decay": float(optimization.get("weight_decay", 0.0)),
        "max_grad_norm": optimization.get("max_grad_norm"),
        "seed": int(optimization.get("seed", 42)),
        "ppo_clip": float(optimization.get("ppo_clip", 0.2)),
        "ppo_passes": ppo_passes,
        "max_rollout_steps": max_rollout_steps,
        "prompts_per_rollout_step": prompt_batch_size,
        "originals_per_prompt": algorithm.num_siblings,
        "originals_per_rollout_step": original_batch_size,
        "gradient_accumulation_steps": prompt_batch_size,
        "num_rubrics": algorithm.num_rubrics,
        "tau_mask": algorithm.tau_mask,
        "mask_clip": algorithm.mask_clip,
        "lambda_rank": algorithm.lambda_rank,
        "lambda_mask": algorithm.lambda_mask,
        "normalization_epsilon": normalization_epsilon,
        "do_sample": sampling.do_sample,
        "temperature": sampling.temperature,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
        "max_new_tokens": sampling.max_new_tokens,
        "counterfactual_max_new_tokens": (
            sampling.counterfactual_max_new_tokens or sampling.max_new_tokens
        ),
        "original_generation_batch_size": sampling.original_batch_size,
        "counterfactual_generation_batch_size": sampling.counterfactual_batch_size,
        "max_slate_size": max_slate_size,
        "validation_rows": validation_rows,
        "validation_interval_rollout_steps": validation_interval,
        "validation_generation_batch_size": validation_batch_size,
        "validation_max_new_tokens": validation_max_new_tokens,
        "validation_dataset_sha256": validation_dataset_fingerprint,
        "train_dataset_sha256": input_fingerprints["train_sha256"],
        "p2_adapter_sha256": input_fingerprints["original_p2_sha256"],
        "training_source_sha256": input_fingerprints["package_source_sha256"],
        "actor_device_map": json.dumps(actor_device_map, sort_keys=True, default=str),
        "reference_device_map": json.dumps(
            actor_device_map if reference_device_map is None else reference_device_map,
            sort_keys=True,
            default=str,
        ),
        "checkpoint_interval_rollout_steps": checkpoint_rollout_steps,
        "keep_checkpoints": keep_checkpoints,
        "resume_rollout_step": resume_rollout_step or 0,
        "resume_checkpoint_manifest_sha256": resume_manifest_sha256 or "",
    }


def _prepare_output_directory(output_dir: Path, *, resume: bool) -> None:
    """Require an empty fresh output, or an existing regular resume output."""

    if output_dir.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {output_dir}")
    if resume:
        if not output_dir.is_dir():
            raise ValueError("resume requires an existing output directory")
        return
    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise ValueError(
                "output directory is not empty; choose a new directory or pass "
                "--resume-from-checkpoint"
            )
    else:
        output_dir.mkdir(parents=True)


def _resolve_resume_checkpoint(
    value: str | None,
    *,
    output_dir: Path,
) -> ValidatedCheckpoint | None:
    if value is None:
        return None
    latest = find_latest_checkpoint(output_dir)
    if value == "latest":
        return latest
    selected = validate_checkpoint(
        Path(value).expanduser(),
        output_dir=output_dir,
    )
    if selected.path != latest.path:
        raise ValueError(
            "resume checkpoint must be the newest complete checkpoint in the "
            "output directory"
        )
    return selected


def _validated_resume_state(
    checkpoint: ValidatedCheckpoint,
    *,
    fingerprints: Mapping[str, str],
    accumulation_steps: int,
) -> dict[str, object]:
    expected = {
        "version",
        "rollout_step",
        "completed_queries",
        "last_example_id",
        "trainer_state",
        "fingerprints",
        "trainable_parameters",
        "log_offsets",
    }
    state = dict(checkpoint.state)
    if set(state) != expected:
        raise ValueError("checkpoint state has missing or unexpected fields")
    rollout_step = state["rollout_step"]
    completed_queries = state["completed_queries"]
    if state["version"] != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint state version")
    if type(rollout_step) is not int or rollout_step <= 0:
        raise ValueError("checkpoint rollout_step must be a positive integer")
    if rollout_step != checkpoint.rollout_step:
        raise ValueError("checkpoint rollout counters disagree")
    if type(completed_queries) is not int or completed_queries <= 0:
        raise ValueError("checkpoint completed_queries must be a positive integer")
    if completed_queries != rollout_step * accumulation_steps:
        raise ValueError(
            "checkpoint is not at a complete prompt-accumulation boundary"
        )
    if not isinstance(state["last_example_id"], str) or not state["last_example_id"]:
        raise ValueError("checkpoint last_example_id must be a non-empty string")
    if state["fingerprints"] != dict(fingerprints):
        raise ValueError("checkpoint inputs do not match this run")
    if not isinstance(state["trainer_state"], Mapping):
        raise ValueError("checkpoint trainer_state must be a mapping")
    if not isinstance(state["trainable_parameters"], list):
        raise ValueError("checkpoint trainable parameter schema must be a list")
    offsets = state["log_offsets"]
    if not isinstance(offsets, Mapping) or set(offsets) != {
        "train_log.jsonl",
        "validation_log.jsonl",
    }:
        raise ValueError("checkpoint JSONL offsets are missing or unexpected")
    return state


def _skip_completed_examples(
    examples: Iterator[TrainingExample],
    *,
    completed_queries: int,
    expected_last_example_id: str,
) -> None:
    last_example: TrainingExample | None = None
    for _ in range(completed_queries):
        try:
            last_example = next(examples)
        except StopIteration as exc:
            raise ValueError(
                "training input ends before the checkpoint's completed query count"
            ) from exc
    if last_example is None or last_example.example_id != expected_last_example_id:
        actual = None if last_example is None else last_example.example_id
        raise ValueError(
            "training data order differs at the checkpoint boundary: "
            f"{actual!r} != {expected_last_example_id!r}"
        )


def _write_run_info(
    output_dir: Path,
    tracker: ExperimentTracker,
    *,
    resume_rollout_step: int | None,
) -> None:
    run_info = tracker.run_info
    if run_info is None:
        return
    suffix = (
        ""
        if resume_rollout_step is None
        else f"-resume-from-{resume_rollout_step:08d}"
    )
    payload = json.dumps(dict(run_info), indent=2, sort_keys=True) + "\n"
    (output_dir / "wandb_run.json").write_text(payload, encoding="utf-8")
    if suffix:
        (output_dir / f"wandb_run{suffix}.json").write_text(
            payload,
            encoding="utf-8",
        )
    url = run_info.get("url")
    if url:
        print(f"W&B run: {url}", flush=True)


def main() -> None:
    args = _parser().parse_args()
    if args.max_rollout_steps is not None and args.max_rollout_steps <= 0:
        raise SystemExit("--max-rollout-steps must be positive")
    if args.max_query_steps is not None and args.max_query_steps <= 0:
        raise SystemExit("--max-query-steps must be positive")

    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    model = _mapping(config["model"], "model")
    data = _mapping(config["data"], "data")
    generation = _mapping(config["generation"], "generation")
    maskpo = _mapping(config["maskpo"], "maskpo")
    optimization = _mapping(config["optimization"], "optimization")
    output = _mapping(config["output"], "output")
    validation = _mapping(config.get("validation", {}), "validation")
    tracking = _mapping(config.get("tracking", {}), "tracking")
    wandb_values = _mapping(tracking.get("wandb", {}), "tracking.wandb")

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
    _require_canonical(data, "max_slate_size", 20)
    _require_canonical(validation, "rows", 200)
    _require_canonical(validation, "interval_rollout_steps", 10)
    _require_canonical(model, "revision", MODEL_REVISION)

    train_file = Path(args.train_file or str(data["train_file"])).expanduser().resolve()
    output_dir = Path(args.output_dir or str(output["directory"])).expanduser().resolve()
    original_adapter_path = Path(str(model["adapter_path"])).expanduser().resolve()
    model_revision = str(model["revision"])
    configured_rollout_steps = optimization.get(
        "max_rollout_steps", optimization.get("max_steps")
    )
    max_rollout_steps = args.max_rollout_steps or _positive_int(
        configured_rollout_steps, "max_rollout_steps"
    )
    max_slate_size = _positive_int(data.get("max_slate_size", 20), "max_slate_size")
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
    accumulation_steps = _positive_int(
        optimization.get("gradient_accumulation_steps", 8),
        "gradient_accumulation_steps",
    )
    effective_prompt_batch_size = int(
        optimization.get("effective_prompt_batch_size", accumulation_steps)
    )
    if effective_prompt_batch_size != accumulation_steps:
        raise ValueError(
            "effective_prompt_batch_size must equal gradient_accumulation_steps "
            "because each microstep contains one prompt"
        )
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
    ppo_passes = _positive_int(optimization.get("ppo_passes", 2), "ppo_passes")

    validation_enabled = bool(validation.get("enabled", True)) and not args.skip_validation
    validation_rows = _positive_int(validation.get("rows", 200), "validation.rows")
    validation_interval = _positive_int(
        validation.get("interval_rollout_steps", 10),
        "validation.interval_rollout_steps",
    )
    validation_batch_size = _positive_int(
        validation.get("generation_batch_size", 4),
        "validation.generation_batch_size",
    )
    validation_max_new_tokens = _positive_int(
        validation.get("max_new_tokens", sampling.max_new_tokens),
        "validation.max_new_tokens",
    )
    log_query_steps = _positive_int(
        output.get("log_query_steps", output.get("log_steps", 1)),
        "log_query_steps",
    )
    checkpoint_rollout_steps = _positive_int(
        output.get("checkpoint_rollout_steps", 5),
        "checkpoint_rollout_steps",
    )
    keep_checkpoints = _positive_int(
        output.get("keep_checkpoints", 2),
        "keep_checkpoints",
    )
    validation_file_value = args.validation_file or data.get("validation_file")
    validation_file = (
        Path(str(validation_file_value)).expanduser().resolve()
        if validation_file_value is not None
        else None
    )
    validation_examples: tuple[TrainingExample, ...] = ()
    validation_dataset_fingerprint: str | None = None
    if validation_enabled:
        if validation_file is None:
            raise ValueError(
                "validation is enabled but data.validation_file is not configured"
            )
        if validation_file == train_file:
            raise ValueError("training and validation files must be different")
        validation_examples = fixed_validation_examples(
            iter_training_examples(
                validation_file,
                max_slate_size=max_slate_size,
            ),
            expected_rows=validation_rows,
            max_slate_size=max_slate_size,
        )
        validation_dataset_fingerprint = validation_fingerprint(validation_examples)

    resume_requested = args.resume_from_checkpoint is not None
    _prepare_output_directory(output_dir, resume=resume_requested)
    fingerprints = {
        "config_sha256": sha256_file(config_path),
        "train_sha256": sha256_file(train_file),
        "validation_sha256": (
            sha256_file(validation_file)
            if validation_enabled and validation_file is not None
            else "validation-disabled"
        ),
        "original_p2_sha256": sha256_tree(original_adapter_path),
        "model_revision": model_revision,
        "train_script_sha256": sha256_file(Path(__file__).resolve()),
        "package_source_sha256": sha256_python_tree(
            Path(__file__).parents[1] / "src" / "openbench_rerank_rl"
        ),
        **_device_runtime_fingerprints(device_map, reference_device_map),
    }
    resume_checkpoint = _resolve_resume_checkpoint(
        args.resume_from_checkpoint,
        output_dir=output_dir,
    )
    resume_state = (
        _validated_resume_state(
            resume_checkpoint,
            fingerprints=fingerprints,
            accumulation_steps=accumulation_steps,
        )
        if resume_checkpoint is not None
        else None
    )

    wandb_settings = parse_wandb_settings(
        wandb_values,
        mode_override=args.wandb_mode,
    )

    if resume_checkpoint is None:
        shutil.copy2(config_path, output_dir / "run_config.yaml")
    safe_tracking_config = _safe_tracking_config(
        model_name=str(model["name_or_path"]),
        algorithm=algorithm,
        sampling=sampling,
        optimization=optimization,
        max_rollout_steps=max_rollout_steps,
        prompt_batch_size=effective_prompt_batch_size,
        original_batch_size=effective_batch_size,
        ppo_passes=ppo_passes,
        max_slate_size=max_slate_size,
        validation_rows=validation_rows,
        validation_interval=validation_interval,
        validation_batch_size=validation_batch_size,
        validation_max_new_tokens=validation_max_new_tokens,
        validation_dataset_fingerprint=validation_dataset_fingerprint,
        normalization_epsilon=float(maskpo.get("normalization_eps", 1e-8)),
        checkpoint_rollout_steps=checkpoint_rollout_steps,
        keep_checkpoints=keep_checkpoints,
        actor_device_map=device_map,
        reference_device_map=reference_device_map,
        input_fingerprints=fingerprints,
        model_revision=model_revision,
        resume_rollout_step=(
            resume_checkpoint.rollout_step if resume_checkpoint is not None else None
        ),
        resume_manifest_sha256=(
            resume_checkpoint.manifest_sha256
            if resume_checkpoint is not None
            else None
        ),
    )
    try:
        with preserve_rng_state(torch):
            tracker: ExperimentTracker = init_wandb_tracker(
                wandb_settings,
                safe_config=safe_tracking_config,
                output_dir=output_dir,
            )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc
    _write_run_info(
        output_dir,
        tracker,
        resume_rollout_step=(
            resume_checkpoint.rollout_step if resume_checkpoint is not None else None
        ),
    )

    exit_code = 1
    try:
        trainer = load_huggingface_maskpo_trainer(
            model_name_or_path=str(model["name_or_path"]),
            adapter_path=str(original_adapter_path),
            actor_adapter_path=(
                str(resume_checkpoint.actor_path)
                if resume_checkpoint is not None
                else None
            ),
            revision=model_revision,
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
            ppo_passes=ppo_passes,
            max_grad_norm=(
                None
                if optimization.get("max_grad_norm") is None
                else float(optimization["max_grad_norm"])
            ),
        )

        parameter_schema = trainable_parameter_schema(trainer.actor)
        resume_rng_state: Mapping[str, object] | None = None
        if resume_checkpoint is not None and resume_state is not None:
            if parameter_schema != resume_state["trainable_parameters"]:
                raise ValueError(
                    "checkpoint trainable parameter names/shapes do not match the actor"
                )
            optimizer_state = torch.load(
                resume_checkpoint.optimizer_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("checkpoint optimizer payload must be a mapping")
            trainer.optimizer.load_state_dict(optimizer_state)
            trainer.load_training_state_dict(resume_state["trainer_state"])
            if (
                trainer.rollout_steps != resume_checkpoint.rollout_step
                or trainer.optimizer_steps != resume_checkpoint.rollout_step * ppo_passes
            ):
                raise ValueError("restored trainer counters disagree with the checkpoint")
            loaded_rng_state = torch.load(
                resume_checkpoint.rng_path,
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(loaded_rng_state, Mapping):
                raise ValueError("checkpoint RNG payload must be a mapping")
            resume_rng_state = loaded_rng_state

        metadata = {
            "config": str(config_path),
            "train_file": str(train_file),
            "validation_file": str(validation_file) if validation_file else None,
            "eval_file": str(data.get("eval_file")) if data.get("eval_file") else None,
            "validation_dataset_sha256": validation_dataset_fingerprint,
            "seed": seed,
            "max_slate_size": max_slate_size,
            "requested_rollout_steps": max_rollout_steps,
            "expected_optimizer_steps": max_rollout_steps * ppo_passes,
            "ppo_passes": ppo_passes,
            "num_siblings": algorithm.num_siblings,
            "effective_prompt_batch_size": effective_prompt_batch_size,
            "effective_original_batch_size": effective_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "validation_rows": validation_rows if validation_enabled else 0,
            "validation_interval_rollout_steps": validation_interval,
            "wandb_mode": wandb_settings.mode,
            "model_revision": model_revision,
            "fingerprints": fingerprints,
            "resumed_from_checkpoint": (
                str(resume_checkpoint.path) if resume_checkpoint is not None else None
            ),
            "resumed_from_manifest_sha256": (
                resume_checkpoint.manifest_sha256
                if resume_checkpoint is not None
                else None
            ),
            "torch": torch.__version__,
        }
        metadata_suffix = (
            ""
            if resume_checkpoint is None
            else f"-resume-from-{resume_checkpoint.rollout_step:08d}"
        )
        (output_dir / f"run_metadata{metadata_suffix}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if validation_enabled and resume_checkpoint is None:
            (output_dir / "validation_manifest.json").write_text(
                json.dumps(
                    {
                        "rows": validation_rows,
                        "max_slate_size": max_slate_size,
                        "dataset_sha256": validation_dataset_fingerprint,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        completed_queries = (
            int(resume_state["completed_queries"]) if resume_state is not None else 0
        )
        last_example_id = (
            str(resume_state["last_example_id"])
            if resume_state is not None
            else None
        )
        rollout_diagnostics: list[dict[str, object]] = []
        log_path = output_dir / "train_log.jsonl"
        validation_log_path = output_dir / "validation_log.jsonl"
        training_examples = iter_training_examples(
            train_file,
            max_slate_size=max_slate_size,
        )
        if resume_state is not None:
            _skip_completed_examples(
                training_examples,
                completed_queries=completed_queries,
                expected_last_example_id=str(resume_state["last_example_id"]),
            )
            truncate_scoped_jsonl_logs(
                output_dir,
                resume_state["log_offsets"],  # type: ignore[arg-type]
            )

        with (
            log_path.open("a", encoding="utf-8", newline="\n") as log_handle,
            validation_log_path.open(
                "a", encoding="utf-8", newline="\n"
            ) as validation_log_handle,
        ):

            def validate_and_add_metrics(
                metrics: dict[str, MetricValue], *, rollout_step: int
            ) -> None:
                if not validation_enabled:
                    return
                with preserve_rng_state(torch):
                    result = run_greedy_validation(
                        trainer.actor,
                        validation_examples,
                        max_new_tokens=validation_max_new_tokens,
                        generation_batch_size=validation_batch_size,
                    )
                _log_validation_result(
                    validation_log_handle,
                    rollout_step=rollout_step,
                    optimizer_step=trainer.optimizer_steps,
                    result=result,
                )
                metrics.update(_validation_tracking_metrics(result))

            def record_completed_rollout(
                ppo_pass_metrics: Sequence[Any],
                *,
                full_rollout: bool,
            ) -> None:
                rollout_step = trainer.rollout_steps
                if full_rollout and len(rollout_diagnostics) != accumulation_steps:
                    raise RuntimeError(
                        "a full rollout checkpoint boundary must contain exactly "
                        f"{accumulation_steps} prompts"
                    )
                metrics = _rollout_tracking_metrics(
                    rollout_diagnostics,
                    ppo_pass_metrics=ppo_pass_metrics,
                    completed_queries=completed_queries,
                    num_siblings=algorithm.num_siblings,
                    optimizer_step=trainer.optimizer_steps,
                    ppo_passes=ppo_passes,
                )
                if validation_enabled and rollout_step % validation_interval == 0:
                    validate_and_add_metrics(metrics, rollout_step=rollout_step)
                checkpoint_saved = (
                    full_rollout
                    and rollout_step % checkpoint_rollout_steps == 0
                )
                if checkpoint_saved:
                    if last_example_id is None:
                        raise RuntimeError("cannot checkpoint without a last example ID")
                    log_offsets = durable_log_offsets(
                        {
                            "train_log.jsonl": log_handle,
                            "validation_log.jsonl": validation_log_handle,
                        }
                    )
                    checkpoint_state = {
                        "version": CHECKPOINT_VERSION,
                        "rollout_step": rollout_step,
                        "completed_queries": completed_queries,
                        "last_example_id": last_example_id,
                        "trainer_state": trainer.training_state_dict(),
                        "fingerprints": fingerprints,
                        "trainable_parameters": parameter_schema,
                        "log_offsets": log_offsets,
                    }
                    saved_rng_state = capture_rng_state(torch)
                    with preserve_rng_state(torch):
                        write_checkpoint(
                            output_dir,
                            rollout_step=rollout_step,
                            state=checkpoint_state,
                            optimizer_state=trainer.optimizer.state_dict(),
                            rng_state=saved_rng_state,
                            save_actor=lambda destination: _save_actor(
                                trainer, destination
                            ),
                            torch_module=torch,
                            keep=keep_checkpoints,
                        )
                metrics["checkpoint/saved"] = checkpoint_saved
                metrics["checkpoint/optimizer_step"] = (
                    trainer.optimizer_steps if checkpoint_saved else 0
                )
                with preserve_rng_state(torch):
                    tracker.log(metrics, rollout_step=rollout_step)
                rollout_diagnostics.clear()

            if validation_enabled and resume_checkpoint is None:
                baseline_metrics: dict[str, MetricValue] = {
                    "progress/query_steps": 0,
                    "progress/original_rollouts": 0,
                    "progress/optimizer_step": 0,
                    "progress/ppo_passes": ppo_passes,
                    "checkpoint/saved": False,
                    "checkpoint/optimizer_step": 0,
                }
                validate_and_add_metrics(baseline_metrics, rollout_step=0)
                with preserve_rng_state(torch):
                    tracker.log(baseline_metrics, rollout_step=0)

            # Restoration comes last: model/optimizer loading, W&B setup, data
            # skipping, log repair, and baseline validation cannot perturb the
            # exact next sampling decision.
            if resume_rng_state is not None:
                restore_rng_state(resume_rng_state, torch)

            for example in training_examples:
                if trainer.rollout_steps >= max_rollout_steps:
                    break
                if (
                    args.max_query_steps is not None
                    and completed_queries >= args.max_query_steps
                ):
                    break
                result = trainer.train_query(example)
                completed_queries += 1
                last_example_id = example.example_id
                diagnostic = _diagnostic_row(completed_queries, result)
                rollout_diagnostics.append(diagnostic)
                log_handle.write(
                    json.dumps(diagnostic, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                log_handle.flush()
                if completed_queries % log_query_steps == 0:
                    print(json.dumps(diagnostic, sort_keys=True))
                if result.rollout_step_completed:
                    record_completed_rollout(
                        result.ppo_pass_metrics,
                        full_rollout=True,
                    )

            if trainer.flush_gradients():
                record_completed_rollout(
                    trainer.last_ppo_pass_metrics,
                    full_rollout=False,
                )

        _save_actor(trainer, output_dir / "final-adapter")
        final_state = {
            **metadata,
            "completed_query_steps": completed_queries,
            "rollout_steps": trainer.rollout_steps,
            "optimizer_steps": trainer.optimizer_steps,
            "stop_reason": (
                "rollout_target"
                if trainer.rollout_steps >= max_rollout_steps
                else "query_limit"
                if args.max_query_steps is not None
                and completed_queries >= args.max_query_steps
                else "input_exhausted"
            ),
        }
        (output_dir / "final_state.json").write_text(
            json.dumps(final_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if trainer.rollout_steps < max_rollout_steps and args.max_query_steps is None:
            print(
                "Training input ended after "
                f"{completed_queries} prompts / {trainer.rollout_steps} rollout "
                f"steps / {trainer.optimizer_steps} optimizer updates; requested "
                f"{max_rollout_steps} rollout steps.",
                flush=True,
            )
        print(f"Saved final adapter to {output_dir / 'final-adapter'}")
        exit_code = 0
    finally:
        tracker.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
