#!/usr/bin/env python3
"""Import the audited Phase-2 rubric SFT adapter from the handoff archive.

The archive is intentionally not unpacked wholesale.  Only the files beneath
``models/mind/qwen3-1.7b/p2_rubric_reasoning_sft`` are copied, and the resulting
directory remains gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


ADAPTER_SUFFIX = PurePosixPath(
    "models/mind/qwen3-1.7b/p2_rubric_reasoning_sft"
)
REQUIRED_FILES = frozenset(
    {
        "adapter_config.json",
        "adapter_model.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    }
)
# Canonical checksums from the audited public handoff.  Verifying every copied
# file makes the starting point reproducible without hashing unrelated 4 GB of
# material in the outer archive.
EXPECTED_SHA256 = {
    "adapter_config.json": "8430f527999143b4c01b91ab71e089fd57f31702e201ede589229ee662a4ed90",
    "adapter_model.safetensors": "2f9cf9f00d430d914efe6fae21f81378a8567de4932640c114d01f2d74747818",
    "added_tokens.json": "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
    "chat_template.jinja": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "config.json": "8765bfdb0d5e7fbc73263401e44a30f9be8f4889cda3536c27d4124c9b3b3d9b",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "443bfa629eb16387a12edbf92a76f6a6f10b2af3b53d87ba1550adfcf45f7fa0",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="path to the audit-copy ZIP")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/adapters/p2_rubric_reasoning_sft"),
        help="new destination directory (default: %(default)s)",
    )
    return parser


def _member_map(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Locate exactly one flat adapter directory inside ``archive``."""

    suffix_parts = ADAPTER_SUFFIX.parts
    candidates: dict[str, dict[str, zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        parts = path.parts
        parent_suffix = tuple(parts[-len(suffix_parts) - 1 : -1])
        if len(parts) <= len(suffix_parts) or parent_suffix != suffix_parts:
            continue
        filename = parts[-1]
        if not filename or filename in {".", ".."} or "/" in filename:
            continue
        prefix = "/".join(parts[: -len(suffix_parts) - 1])
        candidates.setdefault(prefix, {})[filename] = info

    matching = [files for files in candidates.values() if REQUIRED_FILES <= files.keys()]
    if len(matching) != 1:
        raise SystemExit(
            "expected exactly one p2_rubric_reasoning_sft directory containing "
            f"{sorted(REQUIRED_FILES)}; found {len(matching)}"
        )
    return matching[0]


def _copy_and_hash(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> str:
    digest = hashlib.sha256()
    with archive.open(info) as source, destination.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parser().parse_args()
    archive_path = args.archive.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not archive_path.is_file():
        raise SystemExit(f"archive does not exist: {archive_path}")
    if output_dir.exists():
        raise SystemExit(
            f"destination already exists: {output_dir}\n"
            "Choose a new --output-dir or move the existing directory first."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = _member_map(archive)
        with tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}.", dir=output_dir.parent
        ) as temporary:
            temporary_dir = Path(temporary)
            manifest: dict[str, dict[str, int | str]] = {}
            for filename, info in sorted(members.items()):
                destination = temporary_dir / filename
                sha256 = _copy_and_hash(archive, info, destination)
                expected = EXPECTED_SHA256.get(filename)
                if expected is None or sha256 != expected:
                    raise SystemExit(
                        f"checksum mismatch for {filename}: expected {expected!r}, "
                        f"received {sha256}"
                    )
                manifest[filename] = {"bytes": info.file_size, "sha256": sha256}

            (temporary_dir / "IMPORT_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "source_archive_name": archive_path.name,
                        "adapter": str(ADAPTER_SUFFIX),
                        "files": manifest,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.move(str(temporary_dir), output_dir)

    print(f"Imported {len(members)} adapter files into {output_dir}")
    print(f"Checksums: {output_dir / 'IMPORT_MANIFEST.json'}")


if __name__ == "__main__":
    main()
