from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
    assert config["maskpo"]["num_rubrics"] == 4
    assert config["maskpo"]["tau_mask"] == 0.05
    assert config["maskpo"]["mask_clip"] == 2.0
    assert config["maskpo"]["lambda_mask"] == 0.5
    assert config["optimization"]["ppo_clip"] == 0.2
    assert config["optimization"]["reference_kl_coefficient"] == 0.001
    assert config["optimization"]["learning_rate"] == 1e-5
    assert config["optimization"]["effective_original_batch_size"] == 8
    assert config["optimization"]["gradient_accumulation_steps"] == 2
    assert config["output"]["directory"] == "outputs/maskpo-qwen3-1.7b"


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
