"""Crash-safe, scoped rolling checkpoints for MaskPO training."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_VERSION = 1
_CHECKPOINT_RE = re.compile(r"checkpoint-v1-rollout-(\d{8})\Z")
_MANIFEST_NAME = "SHA256SUMS.json"
_COMPLETE_NAME = "COMPLETE"
_REQUIRED_PAYLOADS = frozenset({"optimizer.pt", "rng.pt", "state.json"})


@dataclass(frozen=True)
class ValidatedCheckpoint:
    path: Path
    rollout_step: int
    state: Mapping[str, object]
    manifest_sha256: str

    @property
    def actor_path(self) -> Path:
        return self.path / "actor"

    @property
    def optimizer_path(self) -> Path:
        return self.path / "optimizer.pt"

    @property
    def rng_path(self) -> Path:
        return self.path / "rng.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash regular-file names, sizes, and contents in a non-symlink tree."""

    if path.is_symlink():
        raise ValueError(f"fingerprint target must not be a symlink: {path}")
    root = path.resolve(strict=True)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise ValueError(f"fingerprint target is not a file or directory: {path}")
    entries: list[dict[str, object]] = []
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            raise ValueError(f"fingerprint tree contains a symlink: {candidate}")
        if candidate.is_file():
            relative = candidate.relative_to(root).as_posix()
            entries.append(
                {
                    "path": relative,
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_python_tree(path: Path) -> str:
    """Hash Python source names and bytes while ignoring generated caches."""

    if path.is_symlink():
        raise ValueError(f"source tree must not be a symlink: {path}")
    root = path.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"source tree is not a directory: {path}")
    entries: list[dict[str, object]] = []
    for candidate in sorted(root.rglob("*.py")):
        if candidate.is_symlink():
            raise ValueError(f"source tree contains a symlink: {candidate}")
        if candidate.is_file():
            entries.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    if not entries:
        raise ValueError(f"source tree contains no Python files: {path}")
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def trainable_parameter_schema(actor: Any) -> list[dict[str, object]]:
    """Return ordered trainable parameter names and shapes for optimizer safety."""

    model = getattr(actor, "model", actor)
    named_parameters = getattr(model, "named_parameters", None)
    if not callable(named_parameters):
        raise TypeError("actor does not expose named_parameters")
    schema = [
        {"name": name, "shape": list(parameter.shape)}
        for name, parameter in named_parameters()
        if parameter.requires_grad
    ]
    if not schema:
        raise ValueError("actor has no named trainable parameters")
    return schema


def capture_rng_state(torch_module: Any) -> dict[str, object]:
    cuda_available = bool(torch_module.cuda.is_available())
    return {
        "python": random.getstate(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": torch_module.cuda.get_rng_state_all() if cuda_available else [],
        "cuda_device_count": torch_module.cuda.device_count() if cuda_available else 0,
    }


def restore_rng_state(state: Mapping[str, object], torch_module: Any) -> None:
    expected = {"python", "torch_cpu", "torch_cuda", "cuda_device_count"}
    if set(state) != expected:
        raise ValueError("RNG state has missing or unexpected fields")
    saved_cuda_count = state["cuda_device_count"]
    if type(saved_cuda_count) is not int or saved_cuda_count < 0:
        raise ValueError("invalid saved CUDA device count")
    current_cuda_count = (
        torch_module.cuda.device_count() if torch_module.cuda.is_available() else 0
    )
    if saved_cuda_count != current_cuda_count:
        raise ValueError(
            "CUDA device count differs from checkpoint: "
            f"{current_cuda_count} != {saved_cuda_count}"
        )
    random.setstate(state["python"])
    torch_module.set_rng_state(state["torch_cpu"])
    if saved_cuda_count:
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])


@contextmanager
def preserve_rng_state(torch_module: Any):
    """Prevent validation or diagnostics from perturbing training randomness."""

    state = capture_rng_state(torch_module)
    try:
        yield
    finally:
        restore_rng_state(state, torch_module)


def durable_log_offsets(handles: Mapping[str, Any]) -> dict[str, int]:
    """Flush/fsync open JSONL handles and return durable byte offsets."""

    offsets: dict[str, int] = {}
    for name, handle in handles.items():
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError(f"unsafe JSONL log name: {name!r}")
        handle.flush()
        os.fsync(handle.fileno())
        offsets[name] = os.fstat(handle.fileno()).st_size
    return offsets


def _write_json(path: Path, value: object) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_text(path: Path, value: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _payload_manifest(staging: Path) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"checkpoint payload contains a symlink: {path}")
        if path.is_file() and path not in {
            staging / _MANIFEST_NAME,
            staging / _COMPLETE_NAME,
        }:
            files.append(
                {
                    "path": path.relative_to(staging).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {"version": CHECKPOINT_VERSION, "files": files}


def write_checkpoint(
    output_dir: Path,
    *,
    rollout_step: int,
    state: Mapping[str, object],
    optimizer_state: Mapping[str, object],
    rng_state: Mapping[str, object],
    save_actor: Callable[[Path], None],
    torch_module: Any,
    keep: int = 2,
) -> ValidatedCheckpoint:
    """Atomically publish one complete checkpoint, then retain the newest N."""

    if rollout_step <= 0:
        raise ValueError("rollout_step must be positive")
    if keep <= 0:
        raise ValueError("keep must be positive")
    checkpoint_root = output_dir.resolve() / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    target = checkpoint_root / f"checkpoint-v1-rollout-{rollout_step:08d}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"checkpoint already exists: {target}")

    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-v1-tmp-", dir=checkpoint_root))
    try:
        actor_dir = staging / "actor"
        save_actor(actor_dir)
        _write_json(staging / "state.json", dict(state))
        torch_module.save(dict(optimizer_state), staging / "optimizer.pt")
        _fsync_file(staging / "optimizer.pt")
        torch_module.save(dict(rng_state), staging / "rng.pt")
        _fsync_file(staging / "rng.pt")
        for path in sorted(actor_dir.rglob("*")):
            if path.is_file() and not path.is_symlink():
                _fsync_file(path)

        manifest = _payload_manifest(staging)
        _write_json(staging / _MANIFEST_NAME, manifest)
        manifest_sha256 = sha256_file(staging / _MANIFEST_NAME)
        _write_text(staging / _COMPLETE_NAME, manifest_sha256 + "\n")
        for directory in sorted(
            (path for path in actor_dir.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(actor_dir)
        _fsync_directory(staging)
        os.replace(staging, target)
        _fsync_directory(checkpoint_root)
    except BaseException:
        if staging.exists() and not staging.is_symlink():
            shutil.rmtree(staging)
        raise

    checkpoint = validate_checkpoint(target, output_dir=output_dir)
    retain_newest_checkpoints(output_dir, keep=keep)
    return checkpoint


def _scoped_checkpoint_path(path: Path, output_dir: Path) -> tuple[Path, int]:
    checkpoint_root = output_dir.resolve() / "checkpoints"
    candidate = path.absolute()
    if candidate.is_symlink():
        raise ValueError(f"checkpoint must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved.parent != checkpoint_root:
        raise ValueError("checkpoint must be an immediate child of output/checkpoints")
    match = _CHECKPOINT_RE.fullmatch(resolved.name)
    if match is None or not resolved.is_dir():
        raise ValueError(f"invalid checkpoint directory name: {resolved.name}")
    return resolved, int(match.group(1))


def validate_checkpoint(path: Path, *, output_dir: Path) -> ValidatedCheckpoint:
    checkpoint, name_rollout_step = _scoped_checkpoint_path(path, output_dir)
    for item in checkpoint.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"checkpoint contains a symlink: {item}")
    manifest_path = checkpoint / _MANIFEST_NAME
    complete_path = checkpoint / _COMPLETE_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("checkpoint has no regular SHA256 manifest")
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError("checkpoint is not marked COMPLETE")
    manifest_sha256 = sha256_file(manifest_path)
    if complete_path.read_text(encoding="utf-8").strip() != manifest_sha256:
        raise ValueError("checkpoint COMPLETE marker does not match its manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint manifest")
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("checkpoint manifest files must be a list")

    declared: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("malformed checkpoint manifest entry")
        relative = Path(str(entry["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint manifest contains an unsafe path")
        relative_text = relative.as_posix()
        if relative_text in declared:
            raise ValueError("checkpoint manifest contains a duplicate path")
        declared.add(relative_text)
        payload = checkpoint / relative
        if payload.is_symlink() or not payload.is_file():
            raise ValueError(f"checkpoint payload is missing or unsafe: {relative_text}")
        if type(entry["bytes"]) is not int or entry["bytes"] < 0:
            raise ValueError(f"invalid checkpoint payload size: {relative_text}")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise ValueError(f"invalid checkpoint payload hash: {relative_text}")
        if payload.stat().st_size != entry["bytes"]:
            raise ValueError(f"checkpoint payload size mismatch: {relative_text}")
        if sha256_file(payload) != entry["sha256"]:
            raise ValueError(f"checkpoint payload hash mismatch: {relative_text}")

    actual = {
        item.relative_to(checkpoint).as_posix()
        for item in checkpoint.rglob("*")
        if item.is_file()
        and item not in {checkpoint / _MANIFEST_NAME, checkpoint / _COMPLETE_NAME}
    }
    if actual != declared:
        raise ValueError("checkpoint manifest does not exactly cover payload files")
    if not _REQUIRED_PAYLOADS.issubset(actual) or not any(
        name.startswith("actor/") for name in actual
    ):
        raise ValueError("checkpoint is missing a required payload")

    state = json.loads((checkpoint / "state.json").read_text(encoding="utf-8"))
    if not isinstance(state, dict) or state.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported checkpoint state")
    if state.get("rollout_step") != name_rollout_step:
        raise ValueError("checkpoint directory and state rollout steps differ")
    return ValidatedCheckpoint(
        path=checkpoint,
        rollout_step=name_rollout_step,
        state=state,
        manifest_sha256=manifest_sha256,
    )


def find_latest_checkpoint(output_dir: Path) -> ValidatedCheckpoint:
    checkpoint_root = output_dir.resolve() / "checkpoints"
    if not checkpoint_root.is_dir():
        raise FileNotFoundError("no checkpoint directory exists under the output")
    candidates: list[tuple[int, Path]] = []
    for entry in checkpoint_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        match = _CHECKPOINT_RE.fullmatch(entry.name)
        if match is not None:
            candidates.append((int(match.group(1)), entry))
    if not candidates:
        raise FileNotFoundError("no complete versioned checkpoints found")
    return validate_checkpoint(max(candidates)[1], output_dir=output_dir)


def retain_newest_checkpoints(output_dir: Path, *, keep: int = 2) -> None:
    if keep <= 0:
        raise ValueError("keep must be positive")
    checkpoint_root = output_dir.resolve() / "checkpoints"
    if not checkpoint_root.is_dir():
        return
    validated: list[ValidatedCheckpoint] = []
    for entry in checkpoint_root.iterdir():
        if entry.is_symlink() or not entry.is_dir():
            continue
        if _CHECKPOINT_RE.fullmatch(entry.name) is not None:
            validated.append(validate_checkpoint(entry, output_dir=output_dir))
    validated.sort(key=lambda item: item.rollout_step)
    for checkpoint in validated[:-keep]:
        shutil.rmtree(checkpoint.path)
    if validated[:-keep]:
        _fsync_directory(checkpoint_root)


def truncate_scoped_jsonl_logs(
    output_dir: Path,
    offsets: Mapping[str, object],
) -> None:
    """Discard only uncheckpointed tails of top-level output JSONL logs."""

    root = output_dir.resolve(strict=True)
    for name, raw_offset in offsets.items():
        if Path(name).name != name or not name.endswith(".jsonl"):
            raise ValueError(f"unsafe checkpoint log name: {name!r}")
        if type(raw_offset) is not int or raw_offset < 0:
            raise ValueError(f"invalid checkpoint offset for {name!r}")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint log is missing or unsafe: {name!r}")
        if path.stat().st_size < raw_offset:
            raise ValueError(f"checkpoint offset exceeds current {name!r} size")
        with path.open("r+b") as handle:
            handle.truncate(raw_offset)
            handle.flush()
            os.fsync(handle.fileno())
