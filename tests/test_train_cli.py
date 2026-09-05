from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "maskpo_train_script", ROOT / "scripts" / "train_maskpo.py"
)
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)


def test_checked_in_training_config_matches_canonical_constants():
    config = TRAIN_SCRIPT.load_config(ROOT / "configs" / "maskpo_qwen3_1p7b.yaml")
    assert config["generation"]["num_siblings"] == 4
    assert config["generation"]["temperature"] == 0.6
    assert config["generation"]["top_k"] == 20
    assert config["generation"]["top_p"] == 0.95
    assert config["generation"]["max_new_tokens"] == 2048
    assert config["generation"]["counterfactual_max_new_tokens"] == 2048
    assert config["generation"]["counterfactual_batch_size"] == 16
    assert config["model"]["revision"] == TRAIN_SCRIPT.MODEL_REVISION
    assert config["data"]["max_slate_size"] == 20
    assert config["maskpo"]["num_rubrics"] == 4
    assert config["maskpo"]["tau_mask"] == 0.05
    assert config["maskpo"]["mask_clip"] == 2.0
    assert config["maskpo"]["lambda_mask"] == 0.5
    assert config["optimization"]["ppo_clip"] == 0.2
    assert config["optimization"]["reference_kl_coefficient"] == 0.001
    assert config["optimization"]["learning_rate"] == 1e-5
    assert config["optimization"]["effective_prompt_batch_size"] == 8
    assert config["optimization"]["effective_original_batch_size"] == 32
    assert config["optimization"]["gradient_accumulation_steps"] == 8
    assert config["optimization"]["ppo_passes"] == 2
    assert config["optimization"]["max_rollout_steps"] == 3000
    assert config["validation"]["rows"] == 200
    assert config["validation"]["interval_rollout_steps"] == 10
    assert config["validation"]["generation_batch_size"] == 16
    assert config["tracking"]["wandb"]["mode"] == "online"
    assert config["tracking"]["wandb"]["project"] == "maskpo-mind"
    assert config["tracking"]["wandb"]["entity"] == "franknlp"
    assert config["output"]["directory"] == "outputs/maskpo-qwen3-1.7b"
    assert config["output"]["checkpoint_rollout_steps"] == 5
    assert config["output"]["keep_checkpoints"] == 2
    assert "24k-shuffled" in config["data"]["train_file"]


def test_step_flag_counts_rollout_batches_and_query_cap_is_separate():
    args = TRAIN_SCRIPT._parser().parse_args(
        ["--max-steps", "3", "--max-query-steps", "7"]
    )
    assert args.max_rollout_steps == 3
    assert args.max_query_steps == 7


def test_explicit_resume_checkpoint_must_be_latest(monkeypatch, tmp_path):
    old = TRAIN_SCRIPT.ValidatedCheckpoint(
        path=tmp_path / "checkpoints" / "checkpoint-v1-rollout-00000005",
        rollout_step=5,
        state={},
        manifest_sha256="old",
    )
    newest = TRAIN_SCRIPT.ValidatedCheckpoint(
        path=tmp_path / "checkpoints" / "checkpoint-v1-rollout-00000010",
        rollout_step=10,
        state={},
        manifest_sha256="new",
    )
    monkeypatch.setattr(TRAIN_SCRIPT, "find_latest_checkpoint", lambda _path: newest)
    monkeypatch.setattr(
        TRAIN_SCRIPT,
        "validate_checkpoint",
        lambda _path, *, output_dir: old,
    )
    with pytest.raises(ValueError, match="newest complete checkpoint"):
        TRAIN_SCRIPT._resolve_resume_checkpoint(str(old.path), output_dir=tmp_path)


def test_resume_state_requires_full_boundary_and_matching_fingerprints(tmp_path):
    fingerprints = {"train_sha256": "train", "model_revision": "revision"}
    state = {
        "version": 1,
        "rollout_step": 5,
        "completed_queries": 40,
        "last_example_id": "row-39",
        "trainer_state": {
            "version": 1,
            "rollout_steps": 5,
            "optimizer_steps": 10,
            "ppo_passes": 2,
        },
        "fingerprints": fingerprints,
        "trainable_parameters": [{"name": "lora", "shape": [2, 2]}],
        "log_offsets": {"train_log.jsonl": 100, "validation_log.jsonl": 20},
    }
    checkpoint = TRAIN_SCRIPT.ValidatedCheckpoint(
        path=tmp_path / "checkpoint-v1-rollout-00000005",
        rollout_step=5,
        state=state,
        manifest_sha256="manifest",
    )
    assert TRAIN_SCRIPT._validated_resume_state(
        checkpoint,
        fingerprints=fingerprints,
        accumulation_steps=8,
    ) == state

    with pytest.raises(ValueError, match="inputs do not match"):
        TRAIN_SCRIPT._validated_resume_state(
            checkpoint,
            fingerprints={"train_sha256": "changed"},
            accumulation_steps=8,
        )
    state["completed_queries"] = 39
    with pytest.raises(ValueError, match="complete prompt-accumulation boundary"):
        TRAIN_SCRIPT._validated_resume_state(
            checkpoint,
            fingerprints=fingerprints,
            accumulation_steps=8,
        )


def test_resume_skips_exact_rows_and_checks_last_id():
    examples = iter(
        TRAIN_SCRIPT.TrainingExample(
            prompt=f"prompt-{index}",
            positives=frozenset({1}),
            slate_k=2,
            example_id=f"row-{index}",
        )
        for index in range(4)
    )
    TRAIN_SCRIPT._skip_completed_examples(
        examples,
        completed_queries=2,
        expected_last_example_id="row-1",
    )
    assert next(examples).example_id == "row-2"

    with pytest.raises(ValueError, match="data order differs"):
        TRAIN_SCRIPT._skip_completed_examples(
            iter(
                [
                    TRAIN_SCRIPT.TrainingExample(
                        prompt="prompt",
                        positives=frozenset({1}),
                        slate_k=2,
                        example_id="different",
                    )
                ]
            ),
            completed_queries=1,
            expected_last_example_id="expected",
        )


def test_fresh_output_must_be_empty_but_resume_reuses_existing(tmp_path):
    output = tmp_path / "output"
    TRAIN_SCRIPT._prepare_output_directory(output, resume=False)
    (output / "existing").write_text("run", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        TRAIN_SCRIPT._prepare_output_directory(output, resume=False)
    TRAIN_SCRIPT._prepare_output_directory(output, resume=True)


def test_training_jsonl_reader_requires_prompt_and_preserves_labels(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "I1",
                "prompt": "rank",
                "positive_indices": [2, 4],
                "k": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = list(TRAIN_SCRIPT.iter_training_examples(path))
    assert len(examples) == 1
    assert examples[0].example_id == "I1"
    assert examples[0].positives == frozenset({2, 4})

    path.write_text(
        json.dumps({"positive_indices": [1], "k": 1}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="--include-prompts"):
        list(TRAIN_SCRIPT.iter_training_examples(path))


def test_training_jsonl_reader_enforces_rl_slate_boundary(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "too-large",
                "prompt": "rank",
                "positive_indices": [1],
                "k": 21,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"K<=20"):
        list(TRAIN_SCRIPT.iter_training_examples(path, max_slate_size=20))


def test_rollout_tracking_uses_token_weighted_ppo_pass_metrics_without_ids():
    diagnostics = [
        {
            "loss": 1.0,
            "policy_loss": 0.9,
            "kl": 0.1,
            "clip_fraction": 0.0,
            "active_original_tokens": 10,
            "valid_counterfactuals": 12,
            "counterfactual_slots": 16,
            "rubric_rms_scale": 0.4,
            "routing_fallbacks": 1,
            "unavailable_rubric_bodies": 2,
            "sequence_rewards": [0.2, 0.4],
            "ranking_rewards": [0.1, 0.3],
            "format_rewards": [0.1, 0.1],
            "example_id": "must-not-leak",
        },
        {
            "loss": 3.0,
            "policy_loss": 2.8,
            "kl": 0.2,
            "clip_fraction": 0.5,
            "active_original_tokens": 30,
            "valid_counterfactuals": 8,
            "counterfactual_slots": 16,
            "rubric_rms_scale": 0.2,
            "routing_fallbacks": 0,
            "unavailable_rubric_bodies": 1,
            "sequence_rewards": [0.6, 0.8],
            "ranking_rewards": [0.5, 0.7],
            "format_rewards": [0.1, 0.1],
            "example_id": "also-private",
        },
    ]
    pass_metrics = [
        SimpleNamespace(
            ppo_pass=1,
            loss=2.5,
            policy_loss=2.4,
            kl=0.1,
            clip_fraction=0.0,
            active_token_count=40,
        ),
        SimpleNamespace(
            ppo_pass=2,
            loss=2.0,
            policy_loss=1.8,
            kl=0.2,
            clip_fraction=0.25,
            active_token_count=40,
        ),
    ]

    metrics = TRAIN_SCRIPT._rollout_tracking_metrics(
        diagnostics,
        ppo_pass_metrics=pass_metrics,
        completed_queries=8,
        num_siblings=4,
        optimizer_step=2,
        ppo_passes=2,
    )

    assert metrics["train/loss"] == pytest.approx(2.0)
    assert metrics["train/query_pass_1_loss_mean"] == pytest.approx(2.0)
    assert metrics["train/active_original_tokens"] == 40
    assert metrics["train/valid_counterfactual_rate"] == pytest.approx(20 / 32)
    assert metrics["train/ppo_pass_1_loss"] == pytest.approx(2.5)
    assert metrics["train/ppo_pass_2_clip_fraction"] == pytest.approx(0.25)
    assert metrics["progress/original_rollouts"] == 32
    assert not any("example" in key or "prompt" in key for key in metrics)


def test_safe_tracking_config_uses_model_basename_and_never_paths():
    config = TRAIN_SCRIPT._safe_tracking_config(
        model_name="/private/models/Qwen3-1.7B",
        algorithm=TRAIN_SCRIPT.MaskPOConfig(num_siblings=4),
        sampling=TRAIN_SCRIPT.SamplingConfig(counterfactual_batch_size=16),
        optimization={"learning_rate": 1e-5},
        max_rollout_steps=3000,
        prompt_batch_size=8,
        original_batch_size=32,
        ppo_passes=2,
        max_slate_size=20,
        validation_rows=200,
        validation_interval=10,
        validation_batch_size=16,
        validation_max_new_tokens=2048,
        validation_dataset_fingerprint="abc123",
        normalization_epsilon=1e-8,
        checkpoint_rollout_steps=5,
        keep_checkpoints=2,
        actor_device_map="cuda:0",
        reference_device_map="cuda:1",
        input_fingerprints={
            "train_sha256": "train-hash",
            "original_p2_sha256": "adapter-hash",
            "package_source_sha256": "source-hash",
        },
        model_revision=TRAIN_SCRIPT.MODEL_REVISION,
        resume_rollout_step=None,
        resume_manifest_sha256=None,
    )

    assert config["model"] == "Qwen3-1.7B"
    assert config["counterfactual_generation_batch_size"] == 16
    assert config["validation_generation_batch_size"] == 16
    assert config["seed"] == 42
    assert config["tau_mask"] == 0.05
    assert config["actor_device_map"] == '"cuda:0"'
    assert config["reference_device_map"] == '"cuda:1"'
    assert "/private" not in json.dumps(config)
    assert not any("path" in key for key in config)


def test_main_commits_one_tracking_row_per_rollout_and_validates_at_zero_and_ten(
    monkeypatch, tmp_path
):
    class FakeTracker:
        def __init__(self):
            self.logged = []
            self.finished = []

        def log(self, metrics, *, rollout_step):
            self.logged.append((rollout_step, dict(metrics)))

        @property
        def run_info(self):
            return None

        def finish(self, *, exit_code=0):
            self.finished.append(exit_code)

    class FakeOptimizer:
        def state_dict(self):
            return {"state": {}, "param_groups": []}

    class FakeTrainer:
        def __init__(self):
            self.actor = object()
            self.rollout_steps = 0
            self.optimizer_steps = 0
            self.queries = 0
            self.last_ppo_pass_metrics = ()
            self.optimizer = FakeOptimizer()

        def train_query(self, example):
            del example
            self.queries += 1
            completed = self.queries % 8 == 0
            pass_metrics = ()
            if completed:
                self.rollout_steps += 1
                self.optimizer_steps += 2
                pass_metrics = tuple(
                    SimpleNamespace(
                        ppo_pass=index,
                        loss=0.1 * index,
                        policy_loss=0.1 * index,
                        kl=0.01 * index,
                        clip_fraction=0.0,
                        active_token_count=80,
                    )
                    for index in (1, 2)
                )
                self.last_ppo_pass_metrics = pass_metrics
            return SimpleNamespace(
                rollout_step_completed=completed,
                ppo_pass_metrics=pass_metrics,
            )

        def flush_gradients(self):
            return False

        def training_state_dict(self):
            return {
                "version": 1,
                "rollout_steps": self.rollout_steps,
                "optimizer_steps": self.optimizer_steps,
                "ppo_passes": 2,
            }

    tracker = FakeTracker()
    trainer = FakeTrainer()
    validation_steps = []
    checkpoint_steps = []

    def examples(path, *, max_slate_size=None):
        del max_slate_size
        count = 200 if path.name == "validation.jsonl" else 80
        return iter(
            TRAIN_SCRIPT.TrainingExample(
                prompt=f"prompt-{index}",
                positives=frozenset({1}),
                slate_k=2,
                example_id=f"row-{index}",
            )
            for index in range(count)
        )

    def diagnostic(query_step, result):
        del result
        return {
            "query_step": query_step,
            "loss": 0.1,
            "policy_loss": 0.1,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "active_original_tokens": 10,
            "valid_counterfactuals": 16,
            "counterfactual_slots": 16,
            "rubric_rms_scale": 1.0,
            "routing_fallbacks": 0,
            "unavailable_rubric_bodies": 0,
            "sequence_rewards": [0.5] * 4,
            "ranking_rewards": [0.4] * 4,
            "format_rewards": [0.1] * 4,
        }

    def validate(actor, held_out, **kwargs):
        del actor, held_out, kwargs
        validation_steps.append(trainer.rollout_steps)
        return TRAIN_SCRIPT.ValidationResult(
            rows=200,
            dataset_fingerprint="validation-hash",
                ndcg=0.5,
                format_reward=0.08,
                format_rate=0.8,
                envelope_rate=0.75,
                rubric_header_rates=(0.8, 0.81, 0.82, 0.83),
                synthesis_rate=0.84,
                parse_rate=0.9,
                valid_unique_ids_rate=0.85,
                exact_permutation_rate=0.7,
        )

    monkeypatch.setattr(TRAIN_SCRIPT, "iter_training_examples", examples)
    monkeypatch.setattr(TRAIN_SCRIPT, "_diagnostic_row", diagnostic)
    monkeypatch.setattr(TRAIN_SCRIPT, "run_greedy_validation", validate)
    monkeypatch.setattr(TRAIN_SCRIPT, "init_wandb_tracker", lambda *a, **k: tracker)
    monkeypatch.setattr(
        TRAIN_SCRIPT, "load_huggingface_maskpo_trainer", lambda **kwargs: trainer
    )
    monkeypatch.setattr(TRAIN_SCRIPT, "_save_actor", lambda *args: None)
    monkeypatch.setattr(TRAIN_SCRIPT, "sha256_file", lambda path: f"hash:{path.name}")
    monkeypatch.setattr(TRAIN_SCRIPT, "sha256_tree", lambda path: "adapter-hash")
    monkeypatch.setattr(
        TRAIN_SCRIPT, "sha256_python_tree", lambda path: "package-hash"
    )
    monkeypatch.setattr(
        TRAIN_SCRIPT,
        "trainable_parameter_schema",
        lambda actor: [{"name": "p", "shape": [1]}],
    )

    def save_checkpoint(output_dir, **kwargs):
        del output_dir
        checkpoint_steps.append(kwargs["rollout_step"])

    monkeypatch.setattr(TRAIN_SCRIPT, "write_checkpoint", save_checkpoint)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(set_seed=lambda seed: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_maskpo.py",
            "--config",
            str(ROOT / "configs" / "maskpo_qwen3_1p7b.yaml"),
            "--train-file",
            str(tmp_path / "train.jsonl"),
            "--validation-file",
            str(tmp_path / "validation.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-rollout-steps",
            "10",
        ],
    )

    TRAIN_SCRIPT.main()

    assert validation_steps == [0, 10]
    assert [step for step, _metrics in tracker.logged] == list(range(11))
    assert len({step for step, _metrics in tracker.logged}) == 11
    assert "validation/ndcg" in tracker.logged[0][1]
    assert "validation/ndcg" not in tracker.logged[1][1]
    assert "validation/ndcg" in tracker.logged[10][1]
    assert tracker.logged[10][1]["progress/optimizer_step"] == 20
    assert checkpoint_steps == [5, 10]
    assert tracker.finished == [0]


def test_main_resumes_actor_optimizer_counters_rows_and_logs_without_baseline(
    monkeypatch, tmp_path
):
    import torch

    class FakeTracker:
        def __init__(self):
            self.logged = []
            self.finished = []

        @property
        def run_info(self):
            return None

        def log(self, metrics, *, rollout_step):
            self.logged.append((rollout_step, dict(metrics)))

        def finish(self, *, exit_code=0):
            self.finished.append(exit_code)

    class FakeOptimizer:
        def __init__(self):
            self.loaded = None

        def load_state_dict(self, state):
            self.loaded = dict(state)

        def state_dict(self):
            return {"optimizer": "current"}

    class FakeTrainer:
        def __init__(self):
            self.actor = object()
            self.optimizer = FakeOptimizer()
            self.rollout_steps = 0
            self.optimizer_steps = 0
            self.queries = 0
            self.last_ppo_pass_metrics = ()
            self.loaded_training_state = None

        def load_training_state_dict(self, state):
            self.loaded_training_state = dict(state)
            self.rollout_steps = int(state["rollout_steps"])
            self.optimizer_steps = int(state["optimizer_steps"])

        def train_query(self, example):
            del example
            self.queries += 1
            completed = self.queries == 8
            pass_metrics = ()
            if completed:
                self.rollout_steps += 1
                self.optimizer_steps += 2
                pass_metrics = tuple(
                    SimpleNamespace(
                        ppo_pass=index,
                        loss=0.1,
                        policy_loss=0.1,
                        kl=0.0,
                        clip_fraction=0.0,
                        active_token_count=80,
                    )
                    for index in (1, 2)
                )
                self.last_ppo_pass_metrics = pass_metrics
            return SimpleNamespace(
                rollout_step_completed=completed,
                ppo_pass_metrics=pass_metrics,
            )

        def flush_gradients(self):
            return False

    def examples(path, *, max_slate_size=None):
        del max_slate_size
        count = 200 if path.name == "validation.jsonl" else 48
        return iter(
            TRAIN_SCRIPT.TrainingExample(
                prompt=f"prompt-{index}",
                positives=frozenset({1}),
                slate_k=2,
                example_id=f"row-{index}",
            )
            for index in range(count)
        )

    def diagnostic(query_step, result):
        del result
        return {
            "query_step": query_step,
            "loss": 0.1,
            "policy_loss": 0.1,
            "kl": 0.0,
            "clip_fraction": 0.0,
            "active_original_tokens": 10,
            "valid_counterfactuals": 16,
            "counterfactual_slots": 16,
            "rubric_rms_scale": 1.0,
            "routing_fallbacks": 0,
            "unavailable_rubric_bodies": 0,
            "sequence_rewards": [0.5] * 4,
            "ranking_rewards": [0.4] * 4,
            "format_rewards": [0.1] * 4,
        }

    config_path = ROOT / "configs" / "maskpo_qwen3_1p7b.yaml"
    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    (output / "run_config.yaml").write_text("prior\n", encoding="utf-8")
    train_log = output / "train_log.jsonl"
    validation_log = output / "validation_log.jsonl"
    with (
        train_log.open("a", encoding="utf-8") as train_handle,
        validation_log.open("a", encoding="utf-8") as validation_handle,
    ):
        train_handle.write('{"kept":true}\n')
        offsets = TRAIN_SCRIPT.durable_log_offsets(
            {
                "train_log.jsonl": train_handle,
                "validation_log.jsonl": validation_handle,
            }
        )

    fingerprints = {
        "config_sha256": f"hash:{config_path.name}",
        "train_sha256": f"hash:{train_path.name}",
        "validation_sha256": f"hash:{validation_path.name}",
        "original_p2_sha256": "adapter-hash",
        "model_revision": TRAIN_SCRIPT.MODEL_REVISION,
        "train_script_sha256": "hash:train_maskpo.py",
        "package_source_sha256": "package-hash",
        **TRAIN_SCRIPT._device_runtime_fingerprints("auto", None),
    }
    checkpoint_state = {
        "version": 1,
        "rollout_step": 5,
        "completed_queries": 40,
        "last_example_id": "row-39",
        "trainer_state": {
            "version": 1,
            "rollout_steps": 5,
            "optimizer_steps": 10,
            "ppo_passes": 2,
        },
        "fingerprints": fingerprints,
        "trainable_parameters": [{"name": "p", "shape": [1]}],
        "log_offsets": offsets,
    }

    def save_checkpoint_actor(destination):
        destination.mkdir()
        (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    TRAIN_SCRIPT.write_checkpoint(
        output,
        rollout_step=5,
        state=checkpoint_state,
        optimizer_state={"optimizer": "saved"},
        rng_state=TRAIN_SCRIPT.capture_rng_state(torch),
        save_actor=save_checkpoint_actor,
        torch_module=torch,
    )
    with train_log.open("a", encoding="utf-8") as handle:
        handle.write('{"tail":true}\n')
    with validation_log.open("a", encoding="utf-8") as handle:
        handle.write('{"tail":true}\n')

    tracker = FakeTracker()
    trainer = FakeTrainer()
    loader_kwargs = {}
    monkeypatch.setattr(TRAIN_SCRIPT, "iter_training_examples", examples)
    monkeypatch.setattr(TRAIN_SCRIPT, "_diagnostic_row", diagnostic)
    monkeypatch.setattr(
        TRAIN_SCRIPT,
        "run_greedy_validation",
        lambda *args, **kwargs: pytest.fail("resume repeated baseline validation"),
    )
    monkeypatch.setattr(TRAIN_SCRIPT, "init_wandb_tracker", lambda *a, **k: tracker)

    def load_trainer(**kwargs):
        loader_kwargs.update(kwargs)
        return trainer

    monkeypatch.setattr(TRAIN_SCRIPT, "load_huggingface_maskpo_trainer", load_trainer)
    monkeypatch.setattr(TRAIN_SCRIPT, "_save_actor", lambda *args: None)
    monkeypatch.setattr(TRAIN_SCRIPT, "sha256_file", lambda path: f"hash:{path.name}")
    monkeypatch.setattr(TRAIN_SCRIPT, "sha256_tree", lambda path: "adapter-hash")
    monkeypatch.setattr(
        TRAIN_SCRIPT, "sha256_python_tree", lambda path: "package-hash"
    )
    monkeypatch.setattr(
        TRAIN_SCRIPT,
        "trainable_parameter_schema",
        lambda actor: [{"name": "p", "shape": [1]}],
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(set_seed=lambda seed: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_maskpo.py",
            "--config",
            str(config_path),
            "--train-file",
            str(train_path),
            "--validation-file",
            str(validation_path),
            "--output-dir",
            str(output),
            "--max-rollout-steps",
            "6",
            "--resume-from-checkpoint",
            "latest",
        ],
    )

    TRAIN_SCRIPT.main()

    assert loader_kwargs["actor_adapter_path"].endswith(
        "/checkpoints/checkpoint-v1-rollout-00000005/actor"
    )
    assert loader_kwargs["adapter_path"].endswith("p2_rubric_reasoning_sft")
    assert loader_kwargs["revision"] == TRAIN_SCRIPT.MODEL_REVISION
    assert trainer.optimizer.loaded == {"optimizer": "saved"}
    assert trainer.loaded_training_state == checkpoint_state["trainer_state"]
    assert trainer.queries == 8
    assert [step for step, _metrics in tracker.logged] == [6]
    assert '"tail":true' not in train_log.read_text(encoding="utf-8")
    assert validation_log.read_text(encoding="utf-8") == ""
    assert json.loads((output / "final_state.json").read_text())["completed_query_steps"] == 48
    assert tracker.finished == [0]
