"""Tests for src/controller: proportional-with-floor green allocation on a fixed cycle."""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.controller import ControllerConfig, allocate_green, load_controller_config, phase_plan
from src.controller.config import controller_config_from_dict

ROOT = Path(__file__).resolve().parents[1]
CFG = ControllerConfig(cycle_s=96, yellow_s=3, all_red_s=1, min_green_s=5, max_green_s=None)
CFG_CAP = ControllerConfig(cycle_s=96, yellow_s=3, all_red_s=1, min_green_s=5, max_green_s=45)


def _sum(g: dict[str, float]) -> float:
    return sum(g.values())


# ----------------------------------------------------------------------------- config

def test_budget_and_lost_time() -> None:
    assert CFG.lost_time_s == 16 and CFG.green_budget_s == 80


def test_shipped_controller_config_loads_and_matches_network_baseline() -> None:
    cfg = load_controller_config(ROOT / "configs" / "controller.yaml")
    assert cfg.approaches == ("N", "E", "S", "W")
    # the baseline program in citygen is 4 x (20 + 3 + 1) = 96 s; equal split must reproduce it
    equal = allocate_green({}, cfg)
    assert all(v == pytest.approx(20.0) for v in equal.values())


@pytest.mark.parametrize("bad", [
    dict(cycle_s=30, min_green_s=5),                   # 4 x 5 + 16 lost > 30
    dict(cycle_s=96, min_green_s=5, max_green_s=4),    # cap below floor
    dict(cycle_s=96, min_green_s=5, max_green_s=15),   # 4 x 15 = 60 < 80 budget
    dict(cycle_s=96, approaches=["N", "N", "S", "W"]),  # duplicate names
    dict(cycle_s=-1),
])
def test_invalid_config_rejected(bad: dict) -> None:
    raw = dict(cycle_s=96, yellow_s=3, all_red_s=1, min_green_s=5)
    raw.update(bad)
    with pytest.raises(ValueError):
        controller_config_from_dict(raw)


# ----------------------------------------------------------------------------- allocation

def test_balanced_demand_gives_equal_split() -> None:
    g = allocate_green({"N": 7, "E": 7, "S": 7, "W": 7}, CFG)
    assert all(v == pytest.approx(20.0) for v in g.values())
    assert list(g) == ["N", "E", "S", "W"]


def test_all_zero_demand_gives_equal_split() -> None:
    g = allocate_green({"N": 0, "E": 0, "S": 0, "W": 0}, CFG)
    assert all(v == pytest.approx(20.0) for v in g.values())
    assert _sum(g) == pytest.approx(CFG.green_budget_s)


def test_heavily_skewed_demand_is_proportional_above_floor() -> None:
    g = allocate_green({"N": 30, "E": 5, "S": 5, "W": 0}, CFG)
    spare = CFG.green_budget_s - 4 * CFG.min_green_s  # 60
    assert g["N"] == pytest.approx(5 + spare * 30 / 40)
    assert g["E"] == pytest.approx(5 + spare * 5 / 40) == pytest.approx(g["S"])
    assert g["W"] == pytest.approx(5.0)
    assert _sum(g) == pytest.approx(CFG.green_budget_s)


def test_near_zero_approach_still_gets_floor_not_starved() -> None:
    g = allocate_green({"N": 100, "E": 100, "S": 100, "W": 0.001}, CFG)
    assert g["W"] >= CFG.min_green_s
    assert g["W"] == pytest.approx(CFG.min_green_s, abs=0.01)
    assert _sum(g) == pytest.approx(CFG.green_budget_s)


def test_missing_approaches_count_as_zero_and_negative_is_clamped() -> None:
    g = allocate_green({"N": 10, "S": -4}, CFG)
    assert g["S"] == pytest.approx(5.0) and g["E"] == pytest.approx(5.0) and g["W"] == pytest.approx(5.0)
    assert g["N"] == pytest.approx(CFG.green_budget_s - 15)


def test_unknown_approach_and_nan_rejected() -> None:
    with pytest.raises(KeyError):
        allocate_green({"NE": 1}, CFG)
    with pytest.raises(ValueError):
        allocate_green({"N": math.nan}, CFG)


def test_monotonic_more_density_never_less_green() -> None:
    base = {"N": 10, "E": 10, "S": 10, "W": 10}
    prev = allocate_green(base, CFG)["N"]
    for n in (12, 20, 50, 200, 1000):
        cur = allocate_green({**base, "N": n}, CFG)["N"]
        assert cur >= prev
        prev = cur
    assert prev <= CFG.green_budget_s - 3 * CFG.min_green_s + 1e-9


def test_scale_invariance() -> None:
    a = allocate_green({"N": 1, "E": 2, "S": 3, "W": 4}, CFG)
    b = allocate_green({"N": 10, "E": 20, "S": 30, "W": 40}, CFG)
    for k in a:
        assert a[k] == pytest.approx(b[k])


# ----------------------------------------------------------------------------- cap

def test_cap_pins_dominant_approach_and_redistributes_proportionally() -> None:
    g = allocate_green({"N": 10, "E": 2, "S": 0, "W": 40}, CFG_CAP)
    assert g["W"] == pytest.approx(45.0)
    assert g["S"] == pytest.approx(5.0)
    # remaining 80 - 45 - 5 = 30 split floor + proportional between N (10) and E (2)
    assert g["N"] == pytest.approx(5 + 20 * 10 / 12)
    assert g["E"] == pytest.approx(5 + 20 * 2 / 12)
    assert _sum(g) == pytest.approx(CFG_CAP.green_budget_s)


def test_cap_cascades_when_two_approaches_overflow() -> None:
    cfg = ControllerConfig(cycle_s=96, yellow_s=3, all_red_s=1, min_green_s=5, max_green_s=30)
    g = allocate_green({"N": 100, "E": 100, "S": 1, "W": 0}, cfg)
    assert g["N"] == pytest.approx(30) and g["E"] == pytest.approx(30)
    assert g["W"] == pytest.approx(5)
    assert g["S"] == pytest.approx(15)  # 80 - 60 - 5
    assert _sum(g) == pytest.approx(cfg.green_budget_s)
    assert max(g.values()) <= cfg.max_green_s + 1e-9


def test_cap_inactive_when_nobody_exceeds_it() -> None:
    d = {"N": 3, "E": 2, "S": 2, "W": 1}
    assert allocate_green(d, CFG_CAP) == pytest.approx(allocate_green(d, CFG))


# ----------------------------------------------------------------------------- phase plan

def test_phase_plan_matches_citygen_program_layout_and_cycle() -> None:
    g = allocate_green({"N": 3, "E": 1, "S": 1, "W": 1}, CFG)
    plan = phase_plan(g, CFG)
    assert [name for name, _ in plan] == [
        f"{a}_{k}" for a in ("N", "E", "S", "W") for k in ("green", "yellow", "allred")
    ]
    assert sum(d for _, d in plan) == pytest.approx(CFG.cycle_s)
    assert dict(plan)["N_green"] == pytest.approx(g["N"])


def test_phase_plan_omits_all_red_when_zero() -> None:
    cfg = ControllerConfig(cycle_s=92, yellow_s=3, all_red_s=0, min_green_s=5)
    plan = phase_plan(allocate_green({}, cfg), cfg)
    assert len(plan) == 8 and not any(n.endswith("_allred") for n, _ in plan)
    assert sum(d for _, d in plan) == pytest.approx(cfg.cycle_s)
