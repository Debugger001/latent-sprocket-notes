#!/usr/bin/env python3
"""Download official MIND archives after the user accepts the dataset terms.

MIND access is currently gated on Hugging Face.  First request access to
``yjw1029/MIND`` and authenticate with ``hf auth login``.  This helper does not
grant access or accept the upstream license on the user's behalf.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


REPO_ID = "yjw1029/MIND"
AVAILABLE_SPLITS = {
    "small": ("train", "dev"),
    "large": ("train", "dev", "test"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(AVAILABLE_SPLITS), default="small")
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        action="append",
        dest="splits",
        help="split to download; repeat as needed (default: train and dev)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/mind"),
        help="gitignored destination (default: %(default)s)",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="also extract each downloaded ZIP into a same-named directory",
    )
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="confirm that you reviewed and accepted the official MIND terms",
    )
    return parser


def _safe_extract(archive_path: Path, destination: Path) -> None:
    """Extract a ZIP without allowing members to escape ``destination``."""

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if target != root and root not in target.parents:
                raise SystemExit(f"unsafe path in {archive_path.name}: {info.filename}")
        archive.extractall(destination)


def main() -> None:
    args = _parser().parse_args()
    if not args.accept_license:
        raise SystemExit(
            "MIND is distributed under its own terms. Review the license linked "
            "from https://msnews.github.io/, request dataset access, then rerun "
            "with --accept-license."
        )

    splits = tuple(args.splits or ("train", "dev"))
    unsupported = set(splits) - set(AVAILABLE_SPLITS[args.size])
    if unsupported:
        raise SystemExit(
            f"MIND{args.size} does not publish split(s): {sorted(unsupported)}"
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Install the data helper first: python -m pip install -e '.[data]'"
        ) from exc

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in dict.fromkeys(splits):
        filename = f"MIND{args.size}_{split}.zip"
        try:
            cached_path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                )
            )
        except Exception as exc:  # provider exceptions vary by hub version
            raise SystemExit(
                f"Could not download {filename}. Confirm that access to {REPO_ID} "
                "has been approved and that `hf auth login` succeeds.\n"
                f"Hugging Face reported: {exc}"
            ) from exc

        local_archive = output_dir / filename
        if cached_path.resolve() != local_archive.resolve():
            shutil.copy2(cached_path, local_archive)
        print(f"Downloaded {local_archive}")
        if args.extract:
            extracted = output_dir / local_archive.stem
            _safe_extract(local_archive, extracted)
            print(f"Extracted {extracted}")


if __name__ == "__main__":
    main()
