"""Load :class:`ControllerConfig` from ``configs/controller.yaml``."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.controller.allocation import ControllerConfig


def controller_config_from_dict(raw: dict[str, Any]) -> ControllerConfig:
    cfg = ControllerConfig(
        cycle_s=float(raw["cycle_s"]),
        yellow_s=float(raw.get("yellow_s", 3.0)),
        all_red_s=float(raw.get("all_red_s", 0.0)),
        min_green_s=float(raw.get("min_green_s", 0.0)),
        max_green_s=None if raw.get("max_green_s") is None else float(raw["max_green_s"]),
        approaches=tuple(str(a) for a in raw.get("approaches", ControllerConfig.approaches)),
    )
    cfg.validate()
    return cfg


def load_controller_config(path: Path | str) -> ControllerConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return controller_config_from_dict(raw)
