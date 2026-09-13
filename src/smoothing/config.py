"""Load :class:`SmoothingConfig` from ``configs/smoothing.yaml``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.smoothing.exponential import SmoothingConfig


def smoothing_config_from_dict(raw: dict[str, Any]) -> SmoothingConfig:
    method = str(raw.get("method", "exponential"))
    if method != "exponential":
        raise ValueError(f"unsupported smoothing method {method!r}")
    if raw.get("alpha") is not None:
        return SmoothingConfig(alpha=float(raw["alpha"]))
    return SmoothingConfig.from_half_life(
        half_life_s=float(raw["half_life_s"]),
        sample_interval_s=float(raw.get("sample_interval_s", 1.0)),
    )


def load_smoothing_config(path: Path | str) -> SmoothingConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return smoothing_config_from_dict(raw)
