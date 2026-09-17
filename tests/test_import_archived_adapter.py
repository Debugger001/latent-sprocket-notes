from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "import_archived_adapter",
    ROOT / "scripts" / "import_archived_adapter.py",
)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)


def test_member_map_selects_requested_adapter_without_crossing_directories(tmp_path):
    archive_path = tmp_path / "handoff.zip"
    prefix = "outer/openbench-rerank-rl/models/mind/qwen3-1.7b"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for adapter in IMPORTER.ADAPTERS:
            for filename in IMPORTER.REQUIRED_FILES:
                archive.writestr(f"{prefix}/{adapter}/{filename}", adapter)

    with zipfile.ZipFile(archive_path) as archive:
        members = IMPORTER._member_map(
            archive,
            PurePosixPath(
                "models/mind/qwen3-1.7b/p2_teacher_answer_only_sft"
            ),
        )

    assert set(members) == set(IMPORTER.REQUIRED_FILES)
    assert all("p2_teacher_answer_only_sft" in info.filename for info in members.values())


def test_audited_phase2_adapters_have_distinct_pinned_weight_hashes():
    assert set(IMPORTER.EXPECTED_MODEL_SHA256) == set(IMPORTER.ADAPTERS)
    assert len(set(IMPORTER.EXPECTED_MODEL_SHA256.values())) == len(IMPORTER.ADAPTERS)
    assert all(len(value) == 64 for value in IMPORTER.EXPECTED_MODEL_SHA256.values())
