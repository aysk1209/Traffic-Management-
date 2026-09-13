"""Typed views over the YAML configs in ``configs/``.

Configs are loaded once (``load_network_config`` / ``load_demand_config``) and passed
explicitly to the generators — nothing here reads global state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SIDES = ("north", "east", "south", "west")


@dataclass(frozen=True)
class NetworkConfig:
    name: str
    rows: int
    cols: int
    block_length_m: float
    approach_length_m: float
    speed_kmh: float
    street_lanes: int
    avenue_lanes: int
    avenue_rows: tuple[int, ...]
    avenue_cols: tuple[int, ...]
    no_turnarounds: bool
    signal_scheme: str
    green_s: float
    yellow_s: float
    all_red_s: float

    @property
    def speed_ms(self) -> float:
        return self.speed_kmh / 3.6

    def lanes_for_row(self, row: int) -> int:
        """Lanes per direction on the E-W road with index ``row`` (0 = south)."""
        return self.avenue_lanes if row in self.avenue_rows else self.street_lanes

    def lanes_for_col(self, col: int) -> int:
        """Lanes per direction on the N-S road with index ``col`` (0 = west)."""
        return self.avenue_lanes if col in self.avenue_cols else self.street_lanes


@dataclass(frozen=True)
class Peak:
    centre_h: float
    sigma_h: float
    extra_veh_h_lane: float


@dataclass(frozen=True)
class DemandCurve:
    night_veh_h_lane: float
    day_veh_h_lane: float
    day_start_h: float
    day_end_h: float
    ramp_width_h: float
    peaks: tuple[Peak, ...]


@dataclass(frozen=True)
class DemandConfig:
    name: str
    seed: int
    day_seconds: float
    time_scale: float
    step_length_s: float
    curve: DemandCurve
    side_weights: dict[str, float]
    arrivals: str
    vehicle_mix: dict[str, float]

    @property
    def sim_seconds(self) -> float:
        """Length of the simulation after applying ``time_scale``."""
        return self.day_seconds / self.time_scale


def _read_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at top level")
    return data


def network_config_from_dict(raw: dict[str, Any]) -> NetworkConfig:
    grid, roads, signals = raw["grid"], raw["roads"], raw["signals"]
    cfg = NetworkConfig(
        name=str(raw["name"]),
        rows=int(grid["rows"]),
        cols=int(grid["cols"]),
        block_length_m=float(grid["block_length_m"]),
        approach_length_m=float(grid["approach_length_m"]),
        speed_kmh=float(roads["speed_kmh"]),
        street_lanes=int(roads["street_lanes"]),
        avenue_lanes=int(roads["avenue_lanes"]),
        avenue_rows=tuple(int(r) for r in roads.get("avenue_rows", [])),
        avenue_cols=tuple(int(c) for c in roads.get("avenue_cols", [])),
        no_turnarounds=bool(roads.get("no_turnarounds", True)),
        signal_scheme=str(signals.get("scheme", "per_approach")),
        green_s=float(signals["green_s"]),
        yellow_s=float(signals["yellow_s"]),
        all_red_s=float(signals.get("all_red_s", 0.0)),
    )
    if cfg.rows < 1 or cfg.cols < 1:
        raise ValueError("grid.rows and grid.cols must be >= 1")
    if any(r >= cfg.rows for r in cfg.avenue_rows) or any(c >= cfg.cols for c in cfg.avenue_cols):
        raise ValueError("avenue_rows/avenue_cols index outside the grid")
    if cfg.signal_scheme != "per_approach":
        raise ValueError(f"unsupported signal scheme {cfg.signal_scheme!r}")
    return cfg


def demand_config_from_dict(raw: dict[str, Any]) -> DemandConfig:
    time, curve = raw["time"], raw["curve"]
    peaks = tuple(
        Peak(float(p["centre_h"]), float(p["sigma_h"]), float(p["extra_veh_h_lane"]))
        for p in curve.get("peaks", [])
    )
    side_weights = {s: float(raw.get("side_weights", {}).get(s, 1.0)) for s in SIDES}
    mix = {str(k): float(v) for k, v in raw["vehicle_mix"].items()}
    if abs(sum(mix.values()) - 1.0) > 1e-6:
        raise ValueError(f"vehicle_mix must sum to 1, got {sum(mix.values())}")
    cfg = DemandConfig(
        name=str(raw["name"]),
        seed=int(raw.get("seed", 0)),
        day_seconds=float(time["day_seconds"]),
        time_scale=float(time.get("time_scale", 1.0)),
        step_length_s=float(time.get("step_length_s", 1.0)),
        curve=DemandCurve(
            night_veh_h_lane=float(curve["night_veh_h_lane"]),
            day_veh_h_lane=float(curve["day_veh_h_lane"]),
            day_start_h=float(curve["day_start_h"]),
            day_end_h=float(curve["day_end_h"]),
            ramp_width_h=float(curve["ramp_width_h"]),
            peaks=peaks,
        ),
        side_weights=side_weights,
        arrivals=str(raw.get("arrivals", "poisson")),
        vehicle_mix=mix,
    )
    if cfg.time_scale <= 0:
        raise ValueError("time.time_scale must be > 0")
    if cfg.arrivals != "poisson":
        raise ValueError(f"unsupported arrivals model {cfg.arrivals!r}")
    return cfg


def load_network_config(path: Path | str) -> NetworkConfig:
    return network_config_from_dict(_read_yaml(path))


def load_demand_config(path: Path | str) -> DemandConfig:
    return demand_config_from_dict(_read_yaml(path))
