"""Time-varying demand generator: diurnal rate curve -> Poisson arrivals -> SUMO trips.

The curve (:func:`rate_veh_h_lane`) is the deliberate design choice documented in
``configs/demand_*.yaml``: night floor, logistic ramps to a daytime level, and Gaussian
rush-hour bumps. Arrivals per perimeter entry edge follow a non-homogeneous Poisson
process with rate ``curve(t) * lanes * side_weight``, sampled per simulation second.

Trips are written perimeter-entry -> perimeter-exit; SUMO routes them through the grid
at insertion time, so no separate ``duarouter`` step is needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from src.citygen.config import DemandConfig, DemandCurve
from src.citygen.network import GridPlan, side_of_ext_node

# vType definitions: (length m, max speed m/s, accel, decel, guiShape, colour)
VEHICLE_TYPES: dict[str, dict[str, str]] = {
    "car": dict(vClass="passenger", length="4.5", maxSpeed="40", accel="2.6", decel="4.5",
                sigma="0.5", guiShape="passenger", color="0.85,0.85,0.9"),
    "van": dict(vClass="delivery", length="6.0", maxSpeed="33", accel="2.0", decel="4.0",
                sigma="0.5", guiShape="delivery", color="0.95,0.75,0.3"),
    "truck": dict(vClass="truck", length="9.0", maxSpeed="25", accel="1.3", decel="3.5",
                  sigma="0.5", guiShape="truck", color="0.4,0.6,0.9"),
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rate_veh_h_lane(hour: np.ndarray | float, curve: DemandCurve) -> np.ndarray:
    """Arrival rate (veh/h per entry lane) at clock time ``hour`` (0-24, may be an array)."""
    h = np.asarray(hour, dtype=float)
    w = curve.ramp_width_h
    daytime = _sigmoid((h - curve.day_start_h) / w) - _sigmoid((h - curve.day_end_h) / w)
    rate = curve.night_veh_h_lane + (curve.day_veh_h_lane - curve.night_veh_h_lane) * daytime
    for p in curve.peaks:
        rate = rate + p.extra_veh_h_lane * np.exp(-0.5 * ((h - p.centre_h) / p.sigma_h) ** 2)
    return rate


def sim_time_to_hour(t_sim: np.ndarray | float, cfg: DemandConfig) -> np.ndarray:
    """Map simulation seconds to clock hours, honouring ``time_scale``."""
    return np.asarray(t_sim, dtype=float) * cfg.time_scale / 3600.0


@dataclass(frozen=True)
class Trip:
    id: str
    depart_s: float
    from_edge: str
    to_edge: str
    vtype: str


def sample_poisson_departures(rate_per_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Departure times (s) for a non-homogeneous Poisson process with piecewise-constant
    rate ``rate_per_s[t]`` on each 1-second bin ``[t, t+1)``."""
    counts = rng.poisson(rate_per_s)
    seconds = np.repeat(np.arange(len(rate_per_s)), counts)
    return seconds + rng.uniform(0.0, 1.0, size=len(seconds))


def generate_trips(plan: GridPlan, cfg: DemandConfig, rng: np.random.Generator | None = None) -> list[Trip]:
    """Sample a full day of trips for every perimeter entry of ``plan``."""
    rng = rng or np.random.default_rng(cfg.seed)
    n_seconds = int(round(cfg.sim_seconds))
    t = np.arange(n_seconds)
    base_rate_per_s = rate_veh_h_lane(sim_time_to_hour(t, cfg), cfg.curve) / 3600.0

    exits = plan.exit_edges
    vtypes = list(cfg.vehicle_mix)
    vtype_p = np.array([cfg.vehicle_mix[v] for v in vtypes])

    trips: list[Trip] = []
    for entry in plan.entry_edges:
        side = side_of_ext_node(entry.from_node)
        rate = base_rate_per_s * entry.lanes * cfg.side_weights[side]
        departs = sample_poisson_departures(rate, rng)
        # any exit except the one returning to this entry's own stub (no immediate U-turn)
        candidates = [e for e in exits if e.to_node != entry.from_node]
        dest_idx = rng.integers(0, len(candidates), size=len(departs))
        types = rng.choice(len(vtypes), size=len(departs), p=vtype_p)
        for k, (d, j, v) in enumerate(zip(departs, dest_idx, types)):
            trips.append(Trip(f"{entry.from_node}_{k}", float(d), entry.id, candidates[j].id, vtypes[v]))
    trips.sort(key=lambda tr: tr.depart_s)  # SUMO requires departures in order
    return trips


def routes_xml(trips: list[Trip], cfg: DemandConfig) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<routes>"]
    for name in cfg.vehicle_mix:
        attrs = " ".join(f'{k}="{v}"' for k, v in VEHICLE_TYPES[name].items())
        lines.append(f'    <vType id="{name}" {attrs}/>')
    for tr in trips:
        lines.append(
            f'    <trip id={quoteattr(tr.id)} type="{tr.vtype}" depart="{tr.depart_s:.2f}" '
            f'from={quoteattr(tr.from_edge)} to={quoteattr(tr.to_edge)} '
            'departLane="best" departSpeed="max"/>'
        )
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


def write_routes(trips: list[Trip], cfg: DemandConfig, out_path: Path) -> Path:
    out_path.write_text(routes_xml(trips, cfg), encoding="utf-8")
    return out_path
