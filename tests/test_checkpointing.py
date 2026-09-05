from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from openbench_rerank_rl.checkpointing import (
    capture_rng_state,
    durable_log_offsets,
    find_latest_checkpoint,
    preserve_rng_state,
    restore_rng_state,
    sha256_python_tree,
    sha256_tree,
    trainable_parameter_schema,
    truncate_scoped_jsonl_logs,
    validate_checkpoint,
    write_checkpoint,
)


torch = pytest.importorskip("torch")


def _save_actor(destination: Path) -> None:
    destination.mkdir()
    (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (destination / "adapter_model.safetensors").write_bytes(b"weights")


def _state(step: int) -> dict[str, object]:
    return {
        "version": 1,
        "rollout_step": step,
        "completed_queries": step * 8,
    }


def test_checkpoint_is_manifested_atomic_and_retains_newest_two(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for step in (5, 10, 15):
        checkpoint = write_checkpoint(
            output,
            rollout_step=step,
            state=_state(step),
            optimizer_state={"step": step},
            rng_state=capture_rng_state(torch),
            save_actor=_save_actor,
            torch_module=torch,
            keep=2,
        )
        assert checkpoint.rollout_step == step
        assert (checkpoint.path / "COMPLETE").is_file()
        assert (checkpoint.path / "SHA256SUMS.json").is_file()
        assert validate_checkpoint(checkpoint.path, output_dir=output) == checkpoint

    checkpoint_root = output / "checkpoints"
    assert not (checkpoint_root / "checkpoint-v1-rollout-00000005").exists()
    assert (checkpoint_root / "checkpoint-v1-rollout-00000010").is_dir()
    assert find_latest_checkpoint(output).rollout_step == 15
    assert not any(path.name.startswith(".checkpoint-v1-tmp-") for path in checkpoint_root.iterdir())


def test_checkpoint_validation_detects_payload_tampering_and_symlinks(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = write_checkpoint(
        output,
        rollout_step=5,
        state=_state(5),
        optimizer_state={},
        rng_state=capture_rng_state(torch),
        save_actor=_save_actor,
        torch_module=torch,
    )
    weights = checkpoint.actor_path / "adapter_model.safetensors"
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="(size|hash) mismatch"):
        validate_checkpoint(checkpoint.path, output_dir=output)

    other = tmp_path / "other"
    other.mkdir()
    (other / "x.py").write_text("x = 1\n", encoding="utf-8")
    link = tmp_path / "source-link"
    link.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        sha256_tree(link)


def test_rng_round_trip_and_preservation():
    random.seed(19)
    torch.manual_seed(19)
    state = capture_rng_state(torch)
    expected_python = random.random()
    expected_torch = torch.rand(3)

    restore_rng_state(state, torch)
    with preserve_rng_state(torch):
        random.random()
        torch.rand(10)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)


def test_durable_offsets_truncate_only_named_top_level_jsonl(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    train = output / "train_log.jsonl"
    validation = output / "validation_log.jsonl"
    with (
        train.open("a", encoding="utf-8") as train_handle,
        validation.open("a", encoding="utf-8") as validation_handle,
    ):
        train_handle.write('{"kept":1}\n')
        validation_handle.write('{"kept":2}\n')
        offsets = durable_log_offsets(
            {
                "train_log.jsonl": train_handle,
                "validation_log.jsonl": validation_handle,
            }
        )
        train_handle.write('{"tail":1}\n')
        validation_handle.write('{"tail":2}\n')

    truncate_scoped_jsonl_logs(output, offsets)
    assert train.read_text(encoding="utf-8") == '{"kept":1}\n'
    assert validation.read_text(encoding="utf-8") == '{"kept":2}\n'
    with pytest.raises(ValueError, match="unsafe"):
        truncate_scoped_jsonl_logs(output, {"../other.jsonl": 0})


def test_source_fingerprint_ignores_generated_caches_and_schema_is_ordered(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "a.py").write_text("A = 1\n", encoding="utf-8")
    original = sha256_python_tree(package)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython.pyc").write_bytes(b"generated")
    assert sha256_python_tree(package) == original
    (package / "a.py").write_text("A = 2\n", encoding="utf-8")
    assert sha256_python_tree(package) != original

    actor = SimpleNamespace(
        named_parameters=lambda: iter(
            (
                ("frozen", SimpleNamespace(shape=(2,), requires_grad=False)),
                ("lora.a", SimpleNamespace(shape=(3, 4), requires_grad=True)),
                ("lora.b", SimpleNamespace(shape=(4, 3), requires_grad=True)),
            )
        )
    )
    assert trainable_parameter_schema(actor) == [
        {"name": "lora.a", "shape": [3, 4]},
        {"name": "lora.b", "shape": [4, 3]},
    ]
