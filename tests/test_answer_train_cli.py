from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "answer_train_script", ROOT / "scripts" / "train_answer_rl.py"
)
assert SPEC is not None and SPEC.loader is not None
TRAIN_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN_SCRIPT)


@pytest.mark.parametrize(
    ("filename", "algorithm", "clip_low", "clip_high"),
    [
        ("grpo_answer_only_qwen3_1p7b_a100_stable.yaml", "grpo", 0.2, 0.26),
        (
            "rank_grpo_answer_only_qwen3_1p7b_a100_stable.yaml",
            "rank_grpo",
            0.06,
            0.08,
        ),
    ],
)
def test_checked_in_configs_match_shared_run_and_method_specific_clips(
    filename, algorithm, clip_low, clip_high
):
    config = TRAIN_SCRIPT.load_config(ROOT / "configs" / filename)
    assert config["model"]["revision"] == TRAIN_SCRIPT.MODEL_REVISION
    assert config["data"]["prompt_style"] == "answer-only-permutation-schema"
    assert config["data"]["max_slate_size"] == 20
    assert config["generation"]["num_siblings"] == 4
    assert config["generation"]["temperature"] == 0.6
    assert config["generation"]["top_k"] == 20
    assert config["generation"]["top_p"] == 0.95
    assert config["generation"]["max_new_tokens"] == 2048
    assert config["algorithm"]["name"] == algorithm
    assert config["algorithm"]["clip_low"] == clip_low
    assert config["algorithm"]["clip_high"] == clip_high
    assert config["optimization"]["learning_rate"] == 5e-6
    assert config["optimization"]["reference_kl_coefficient"] == 0.01
    assert config["optimization"]["effective_prompt_batch_size"] == 8
    assert config["optimization"]["effective_original_batch_size"] == 32
    assert config["optimization"]["gradient_accumulation_steps"] == 8
    assert config["optimization"]["ppo_passes"] == 1
    assert config["optimization"]["max_rollout_steps"] == 3000
    assert config["validation"]["rows"] == 200
    assert config["validation"]["interval_rollout_steps"] == 10
    assert config["output"]["checkpoint_rollout_steps"] == 50
    assert config["output"]["keep_checkpoints"] == 5

    name = TRAIN_SCRIPT._validate_canonical_config(
        model=config["model"],
        data=config["data"],
        generation=config["generation"],
        algorithm=config["algorithm"],
        optimization=config["optimization"],
        validation=config["validation"],
        output=config["output"],
    )
    assert name == algorithm


def test_canonical_config_rejects_smaller_update_and_checkpoint_windows():
    config = TRAIN_SCRIPT.load_config(
        ROOT / "configs" / "grpo_answer_only_qwen3_1p7b_a100_stable.yaml"
    )
    config["optimization"]["gradient_accumulation_steps"] = 2
    with pytest.raises(ValueError, match="gradient_accumulation_steps=8"):
        TRAIN_SCRIPT._validate_canonical_config(
            model=config["model"],
            data=config["data"],
            generation=config["generation"],
            algorithm=config["algorithm"],
            optimization=config["optimization"],
            validation=config["validation"],
            output=config["output"],
        )

    config["optimization"]["gradient_accumulation_steps"] = 8
    config["output"]["checkpoint_rollout_steps"] = 5
    with pytest.raises(ValueError, match="checkpoint_rollout_steps=50"):
        TRAIN_SCRIPT._validate_canonical_config(
            model=config["model"],
            data=config["data"],
            generation=config["generation"],
            algorithm=config["algorithm"],
            optimization=config["optimization"],
            validation=config["validation"],
            output=config["output"],
        )


def test_jsonl_reader_requires_permutation_schema_and_k20(tmp_path):
    prompt = TRAIN_SCRIPT._expected_answer_only_header(4) + "2019-11-15\n"
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "row-private",
                "prompt": prompt,
                "positive_indices": [2, 4],
                "k": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    examples = list(TRAIN_SCRIPT.iter_training_examples(path, max_slate_size=20))
    assert len(examples) == 1
    assert examples[0].example_id == "row-private"
    assert examples[0].positives == frozenset({2, 4})

    changed = prompt.replace(
        "[permutation of 1 through K]",
        "[3, 7, 1, 2, 4, 5, 6, 8, 9, 10]",
    )
    path.write_text(
        json.dumps(
            {
                "prompt": changed,
                "positive_indices": [2],
                "k": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordering-neutral 1-through-K schema"):
        list(TRAIN_SCRIPT.iter_training_examples(path, max_slate_size=20))

    path.write_text(
        json.dumps(
            {
                "prompt": TRAIN_SCRIPT._expected_answer_only_header(21) + "date",
                "positive_indices": [1],
                "k": 21,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"K<=20"):
        list(TRAIN_SCRIPT.iter_training_examples(path, max_slate_size=20))


def test_rollout_tracking_is_scalar_and_does_not_forward_ids_or_prompts():
    diagnostics = [
        {
            "example_id": "private-row-a",
            "completion_count": 4,
            "active_actions": 15,
            "active_tokens": 40,
            "parseable_completions": 3,
            "underflow_completions": 1,
            "overflow_items": 2,
            "segmentation_failures": 1,
            "ranking_rewards": [0.2, 0.4, 0.6, 0.8],
            "loss": 0.4,
        },
        {
            "example_id": "private-row-b",
            "completion_count": 4,
            "active_actions": 16,
            "active_tokens": 50,
            "parseable_completions": 4,
            "underflow_completions": 0,
            "overflow_items": 0,
            "segmentation_failures": 0,
            "ranking_rewards": [0.1, 0.3, 0.5, 0.7],
            "loss": 0.2,
        },
    ]
    pass_metrics = [
        SimpleNamespace(
            ppo_pass=1,
            loss=0.25,
            policy_loss=0.2,
            kl=0.05,
            clip_fraction=0.125,
            active_token_count=90,
            action_count=31,
            completion_count=8,
        )
    ]
    metrics = TRAIN_SCRIPT._rollout_tracking_metrics(
        diagnostics,
        ppo_pass_metrics=pass_metrics,
        completed_queries=8,
        optimizer_step=1,
    )
    assert metrics["train/loss"] == pytest.approx(0.25)
    assert metrics["train/ranking_reward"] == pytest.approx(0.45)
    assert metrics["train/parse_rate"] == pytest.approx(7 / 8)
    assert metrics["train/underflow_rate"] == pytest.approx(1 / 8)
    assert metrics["train/overflow_items_per_completion"] == pytest.approx(2 / 8)
    assert metrics["train/active_tokens"] == 90
    assert metrics["train/active_actions"] == 31
    assert metrics["progress/original_rollouts"] == 32
    assert all(isinstance(value, (bool, float, int)) for value in metrics.values())
    assert not any("example" in key or "prompt" in key or "id" in key for key in metrics)


def test_safe_tracking_config_contains_hashes_but_no_paths_or_row_content():
    values = TRAIN_SCRIPT._safe_tracking_config(
        model_name="/private/models/Qwen3-1.7B",
        algorithm_name="rank_grpo",
        algorithm={
            "return": "exp_inf",
            "clip_low": 0.06,
            "clip_high": 0.08,
            "normalization_eps": 1e-4,
            "underflow_penalty": -0.1,
            "overflow_penalty": -0.1,
        },
        sampling=TRAIN_SCRIPT.SamplingConfig(original_batch_size=4),
        optimization={"learning_rate": 5e-6, "reference_kl_coefficient": 0.01},
        max_rollout_steps=3000,
        validation_rows=200,
        validation_interval=10,
        validation_batch_size=16,
        validation_max_new_tokens=2048,
        validation_dataset_fingerprint="validation-hash",
        checkpoint_rollout_steps=50,
        keep_checkpoints=5,
        actor_device_map="cuda:0",
        reference_device_map="cuda:1",
        input_fingerprints={
            "train_sha256": "train-hash",
            "starting_adapter_sha256": "adapter-hash",
            "package_source_sha256": "source-hash",
        },
        model_revision=TRAIN_SCRIPT.MODEL_REVISION,
        resume_rollout_step=None,
        resume_manifest_sha256=None,
    )
    encoded = json.dumps(values)
    assert values["model"] == "Qwen3-1.7B"
    assert values["checkpoint_interval_rollout_steps"] == 50
    assert values["keep_checkpoints"] == 5
    assert "/private" not in encoded
    assert "clicked-news history" not in encoded
    assert "private-row" not in encoded
    assert not any("path" in key or key.endswith("_id") for key in values)


def test_main_validates_at_zero_and_ten_and_checkpoints_at_fifty(
    monkeypatch, tmp_path
):
    class FakeTracker:
        def __init__(self):
            self.logged = []
            self.finished = []

        @property
        def run_info(self):
            return None

        def log(self, metrics, *, rollout_step):
            assert all(isinstance(value, (bool, float, int)) for value in metrics.values())
            self.logged.append((rollout_step, dict(metrics)))

        def finish(self, *, exit_code=0):
            self.finished.append(exit_code)

    class FakeOptimizer:
        def state_dict(self):
            return {"state": {}, "param_groups": []}

    class FakeTrainer:
        def __init__(self):
            self.actor = object()
            self.optimizer = FakeOptimizer()
            self.rollout_steps = 0
            self.optimizer_steps = 0
            self.queries = 0
            self.last_ppo_pass_metrics = ()

        def train_query(self, example):
            del example
            self.queries += 1
            completed = self.queries % 8 == 0
            metrics = ()
            if completed:
                self.rollout_steps += 1
                self.optimizer_steps += 1
                metrics = (
                    SimpleNamespace(
                        ppo_pass=1,
                        loss=0.1,
                        policy_loss=0.09,
                        kl=0.01,
                        clip_fraction=0.0,
                        active_token_count=80,
                        action_count=32,
                        completion_count=32,
                    ),
                )
                self.last_ppo_pass_metrics = metrics
            return SimpleNamespace(
                rollout_step_completed=completed,
                ppo_pass_metrics=metrics,
            )

        def flush_gradients(self):
            return False

        def training_state_dict(self):
            return {
                "version": 1,
                "algorithm": "grpo",
                "rollout_steps": self.rollout_steps,
                "optimizer_steps": self.optimizer_steps,
                "pending_micro_steps": 0,
                "ppo_passes": 1,
            }

    tracker = FakeTracker()
    trainer = FakeTrainer()
    validation_steps = []
    checkpoint_calls = []

    def examples(path, *, max_slate_size=None):
        del max_slate_size
        count = 200 if path.name == "validation.jsonl" else 400
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
            "example_id": f"local-row-{query_step}",
            "loss": 0.1,
            "completion_count": 4,
            "active_tokens": 10,
            "active_actions": 4,
            "parseable_completions": 4,
            "underflow_completions": 0,
            "overflow_items": 0,
            "segmentation_failures": 0,
            "ranking_rewards": [0.5] * 4,
        }

    def validate(actor, held_out, **kwargs):
        del actor, held_out, kwargs
        validation_steps.append(trainer.rollout_steps)
        return TRAIN_SCRIPT.AnswerOnlyValidationResult(
            rows=200,
            dataset_fingerprint="validation-hash",
            ndcg=0.6,
            parse_rate=0.99,
            bare_list_rate=0.98,
            valid_unique_ids_rate=0.97,
            exact_permutation_rate=0.96,
        )

    monkeypatch.setattr(TRAIN_SCRIPT, "iter_training_examples", examples)
    monkeypatch.setattr(TRAIN_SCRIPT, "_diagnostic_row", diagnostic)
    monkeypatch.setattr(TRAIN_SCRIPT, "run_answer_only_validation", validate)
    monkeypatch.setattr(TRAIN_SCRIPT, "init_wandb_tracker", lambda *a, **k: tracker)
    monkeypatch.setattr(
        TRAIN_SCRIPT, "_load_huggingface_answer_rl_trainer", lambda **kwargs: trainer
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
        checkpoint_calls.append(
            (kwargs["rollout_step"], kwargs["keep"], kwargs["state"])
        )

    monkeypatch.setattr(TRAIN_SCRIPT, "write_checkpoint", save_checkpoint)
    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(set_seed=lambda seed: None)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_answer_rl.py",
            "--config",
            str(ROOT / "configs" / "grpo_answer_only_qwen3_1p7b_a100_stable.yaml"),
            "--train-file",
            str(tmp_path / "train.jsonl"),
            "--validation-file",
            str(tmp_path / "validation.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--max-rollout-steps",
            "50",
        ],
    )

    TRAIN_SCRIPT.main()

    assert validation_steps == [0, 10, 20, 30, 40, 50]
    assert [step for step, _ in tracker.logged] == list(range(51))
    assert "validation/ndcg" in tracker.logged[0][1]
    assert "validation/ndcg" not in tracker.logged[1][1]
    assert "validation/ndcg" in tracker.logged[50][1]
    assert [(step, keep) for step, keep, _ in checkpoint_calls] == [(50, 5)]
    checkpoint_state = checkpoint_calls[0][2]
    assert checkpoint_state["completed_queries"] == 400
    assert checkpoint_state["last_example_id"] == "row-399"
    assert tracker.finished == [0]
