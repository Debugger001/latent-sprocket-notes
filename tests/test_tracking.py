from __future__ import annotations

import importlib

import pytest

from openbench_rerank_rl.tracking import (
    DisabledTracker,
    WandbSettings,
    WandbTracker,
    init_wandb_tracker,
    parse_wandb_settings,
)


class FakeRun:
    def __init__(self) -> None:
        self.id = "run-123"
        self.name = "canonical"
        self.url = "https://wandb.ai/research/maskpo/runs/run-123"
        self.project = "maskpo"
        self.entity = "research"
        self.defined: list[tuple[tuple, dict]] = []
        self.logged: list[tuple[dict, int]] = []
        self.exit_codes: list[int] = []

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, values, *, step):
        self.logged.append((values, step))

    def finish(self, *, exit_code):
        self.exit_codes.append(exit_code)


class FakeWandb:
    def __init__(self) -> None:
        self.run = FakeRun()
        self.init_kwargs = None

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        return self.run


def test_disabled_tracking_does_not_import_wandb(monkeypatch, tmp_path):
    def unexpected_import(name):  # pragma: no cover - called only on regression
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", unexpected_import)
    tracker = init_wandb_tracker(
        WandbSettings(mode="disabled"),
        safe_config={"learning_rate": 1e-5},
        output_dir=tmp_path,
    )
    assert isinstance(tracker, DisabledTracker)
    tracker.log({"train/loss": 1.0}, rollout_step=1)
    tracker.finish()


def test_online_tracking_only_sends_explicit_config_and_scalar_metrics(
    monkeypatch, tmp_path
):
    fake = FakeWandb()
    monkeypatch.setattr(importlib, "import_module", lambda name: fake)
    settings = WandbSettings(
        mode="online",
        project="maskpo",
        entity="research",
        name="canonical",
        group="qwen",
        tags=("mind", "maskpo"),
    )

    tracker = init_wandb_tracker(
        settings,
        safe_config={"learning_rate": 1e-5, "validation_rows": 200},
        output_dir=tmp_path,
    )
    tracker.log(
        {"train/loss": 0.4, "progress/optimizer_step": 20},
        rollout_step=10,
    )
    tracker.finish(exit_code=0)

    assert fake.init_kwargs["mode"] == "online"
    assert fake.init_kwargs["save_code"] is False
    assert fake.init_kwargs["settings"] == {
        "console": "off",
        "disable_code": True,
        "disable_git": True,
        "x_disable_meta": True,
    }
    assert fake.init_kwargs["config"] == {
        "learning_rate": 1e-5,
        "validation_rows": 200,
    }
    assert tracker.run_info == {
        "id": "run-123",
        "name": "canonical",
        "url": "https://wandb.ai/research/maskpo/runs/run-123",
        "project": "maskpo",
        "entity": "research",
    }
    assert fake.run.logged == [
        (
            {
                "rollout_step": 10,
                "train/loss": 0.4,
                "progress/optimizer_step": 20,
            },
            10,
        )
    ]
    assert fake.run.exit_codes == [0]


def test_tracking_rejects_secret_like_config_and_nonscalar_metrics(tmp_path):
    with pytest.raises(ValueError, match="secret-like"):
        init_wandb_tracker(
            WandbSettings(mode="disabled"),
            safe_config={"api_token": "never-log-this"},
            output_dir=tmp_path,
        )

    tracker = WandbTracker(FakeRun())
    with pytest.raises(TypeError, match="must be a scalar"):
        tracker.log({"unsafe/list": [1, 2]}, rollout_step=1)  # type: ignore[dict-item]


def test_parse_wandb_settings_allows_explicit_disable_override():
    settings = parse_wandb_settings(
        {
            "mode": "online",
            "project": "maskpo",
            "tags": ["canonical", "mind"],
        },
        mode_override="disabled",
    )
    assert settings.mode == "disabled"
    assert settings.tags == ("canonical", "mind")
