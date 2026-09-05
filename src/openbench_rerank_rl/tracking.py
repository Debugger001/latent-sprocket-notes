"""Small, optional W&B boundary for MaskPO experiment metrics."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


MetricValue = bool | float | int
_SECRET_KEY_FRAGMENTS = (
    "access_token",
    "api_key",
    "api_token",
    "auth_token",
    "credential",
    "password",
    "secret",
)


class ExperimentTracker(Protocol):
    @property
    def run_info(self) -> Mapping[str, str] | None:
        """Return non-secret run identity fields for a local monitoring link."""

    def log(self, metrics: Mapping[str, MetricValue], *, rollout_step: int) -> None:
        """Log one atomic set of scalar metrics at a rollout-batch step."""

    def finish(self, *, exit_code: int = 0) -> None:
        """Finish the tracking run."""


@dataclass(frozen=True)
class WandbSettings:
    """Non-secret W&B run settings.

    Authentication is intentionally absent.  The SDK obtains credentials from
    its normal environment or local login state; callers must never put API
    keys in the experiment YAML.
    """

    mode: str = "disabled"
    project: str = "maskpo-mind"
    entity: str | None = None
    name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "offline", "online"}:
            raise ValueError("W&B mode must be disabled, offline, or online")
        if self.mode != "disabled" and not self.project:
            raise ValueError("W&B project must not be empty when tracking is enabled")


class DisabledTracker:
    """No-op tracker used without importing the optional W&B dependency."""

    @property
    def run_info(self) -> Mapping[str, str] | None:
        return None

    def log(self, metrics: Mapping[str, MetricValue], *, rollout_step: int) -> None:
        del metrics, rollout_step

    def finish(self, *, exit_code: int = 0) -> None:
        del exit_code


class WandbTracker:
    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def run_info(self) -> Mapping[str, str] | None:
        values = {
            key: str(value)
            for key, value in (
                ("id", getattr(self._run, "id", None)),
                ("name", getattr(self._run, "name", None)),
                ("url", getattr(self._run, "url", None)),
                ("project", getattr(self._run, "project", None)),
                ("entity", getattr(self._run, "entity", None)),
            )
            if value
        }
        return values or None

    def log(self, metrics: Mapping[str, MetricValue], *, rollout_step: int) -> None:
        if rollout_step < 0:
            raise ValueError("rollout_step must be non-negative")
        values: dict[str, MetricValue] = {"rollout_step": rollout_step}
        for key, value in metrics.items():
            if not isinstance(value, (bool, float, int)):
                raise TypeError(f"W&B metric {key!r} must be a scalar")
            values[str(key)] = value
        self._run.log(values, step=rollout_step)

    def finish(self, *, exit_code: int = 0) -> None:
        self._run.finish(exit_code=exit_code)


def _check_safe_config(config: Mapping[str, object]) -> None:
    for key in config:
        normalized = str(key).lower()
        if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
            raise ValueError(f"refusing to send secret-like config key {key!r} to W&B")


def init_wandb_tracker(
    settings: WandbSettings,
    *,
    safe_config: Mapping[str, object],
    output_dir: Path,
) -> ExperimentTracker:
    """Initialize W&B lazily, or return a no-op tracker when disabled."""

    _check_safe_config(safe_config)
    if settings.mode == "disabled":
        return DisabledTracker()
    try:
        wandb = importlib.import_module("wandb")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "W&B tracking is enabled; install the `tracking` project extra"
        ) from exc

    init_kwargs: dict[str, object] = {
        "project": settings.project,
        "mode": settings.mode,
        "config": dict(safe_config),
        "dir": str(output_dir),
        "tags": list(settings.tags),
        "job_type": "train",
        # Avoid uploading source or capturing console rows that contain local
        # example identifiers.  Only the explicit whitelist reaches W&B.
        "save_code": False,
        "settings": {
            "console": "off",
            "disable_code": True,
            "disable_git": True,
            "x_disable_meta": True,
        },
    }
    for key, value in (
        ("entity", settings.entity),
        ("name", settings.name),
        ("group", settings.group),
    ):
        if value is not None:
            init_kwargs[key] = value
    run = wandb.init(**init_kwargs)
    if run is None:  # pragma: no cover - defensive SDK boundary
        raise RuntimeError("wandb.init did not return a run")
    run.define_metric("rollout_step")
    for namespace in ("checkpoint/*", "progress/*", "train/*", "validation/*"):
        run.define_metric(namespace, step_metric="rollout_step")
    return WandbTracker(run)


def parse_wandb_settings(
    values: Mapping[str, object],
    *,
    mode_override: str | None = None,
) -> WandbSettings:
    """Parse the public, non-secret W&B subsection of the run YAML."""

    raw_tags = values.get("tags", ())
    if isinstance(raw_tags, str) or not isinstance(raw_tags, Sequence):
        raise ValueError("W&B tags must be a sequence of strings")
    tags = tuple(str(tag) for tag in raw_tags)

    def optional_string(key: str) -> str | None:
        value = values.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"W&B {key} must be a string or null")
        return value

    mode = mode_override or str(values.get("mode", "disabled"))
    return WandbSettings(
        mode=mode,
        project=str(values.get("project", "maskpo-mind")),
        entity=optional_string("entity"),
        name=optional_string("name"),
        group=optional_string("group"),
        tags=tags,
    )
