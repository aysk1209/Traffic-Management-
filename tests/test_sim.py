"""Tests for src/sim: the traci-free per-junction cycle logic, and (with SUMO) a short
end-to-end adaptive run on the single intersection."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.controller import ControllerConfig
from src.sim.junction import JunctionController, approach_of_phase
from src.smoothing import SmoothingConfig

PHASES = [f"{a}_{k}" for a in ("N", "E", "S", "W") for k in ("green", "yellow", "allred")]
CTRL = ControllerConfig(cycle_s=96, yellow_s=3, all_red_s=1, min_green_s=5, max_green_s=45)


def _jc(alpha: float = 1.0, aggregate: str = "cycle_max") -> JunctionController:
    return JunctionController(
        tls_id="j", phase_names=PHASES,
        approach_lanes={a: [f"{a}_0"] for a in "NESW"},
        controller_cfg=CTRL, smoothing_cfg=SmoothingConfig(alpha), aggregate=aggregate,
    )


def test_approach_of_phase() -> None:
    assert approach_of_phase("W_green") == "W"
    assert approach_of_phase("W_yellow") is None and approach_of_phase("W_allred") is None


def test_program_must_match_controller_approaches() -> None:
    with pytest.raises(ValueError):
        JunctionController("j", ["N_green", "N_yellow", "S_green", "S_yellow"], {}, CTRL, SmoothingConfig(1.0))
    with pytest.raises(ValueError):
        _jc(aggregate="median")


def test_first_step_plans_and_sets_first_green() -> None:
    jc = _jc()
    act = jc.step(0.0, 0, {"N": 0, "E": 0, "S": 0, "W": 0})
    assert act.new_cycle is not None
    assert act.set_phase_duration == pytest.approx(20.0)      # equal split
    assert act.new_cycle.green == pytest.approx({"N": 20, "E": 20, "S": 20, "W": 20})


def test_durations_only_on_phase_change_and_only_for_green_phases() -> None:
    jc = _jc()
    jc.step(0.0, 0, {"N": 5, "E": 0, "S": 0, "W": 0})
    assert jc.step(1.0, 0, {"N": 5}).set_phase_duration is None      # same phase: nothing
    assert jc.step(20.0, 1, {"N": 5}).set_phase_duration is None     # N_yellow
    assert jc.step(23.0, 2, {"N": 5}).set_phase_duration is None     # N_allred
    act = jc.step(24.0, 3, {"N": 5})                                 # E_green
    assert act.set_phase_duration == pytest.approx(jc.plan["E"]) and act.new_cycle is None


def test_plan_is_frozen_within_cycle_and_recomputed_at_cycle_start() -> None:
    jc = _jc(aggregate="instant")
    jc.step(0.0, 0, {"N": 10, "E": 0, "S": 0, "W": 0})
    plan1 = dict(jc.plan)
    # W fills up mid-cycle, but the plan does not change until phase 0 comes round again
    jc.step(50.0, 6, {"N": 0, "E": 0, "S": 0, "W": 30})
    assert jc.plan == plan1
    act = jc.step(96.0, 0, {"N": 0, "E": 0, "S": 0, "W": 30})
    assert act.new_cycle is not None and jc.plan["W"] > plan1["W"] and jc.plan["N"] < plan1["N"]


def test_cycle_max_uses_peak_queue_not_value_at_decision_time() -> None:
    inst, cmax = _jc(aggregate="instant"), _jc(aggregate="cycle_max")
    for jc in (inst, cmax):
        jc.step(0.0, 0, {"N": 0, "E": 0, "S": 0, "W": 0})
        for t in range(1, 90):                                     # W queues up during its red...
            jc.step(float(t), 1, {"N": 0, "E": 0, "S": 0, "W": 12})
        jc.step(95.0, 11, {"N": 0, "E": 0, "S": 0, "W": 0})          # ...and has just been discharged
        jc.step(96.0, 0, {"N": 2, "E": 2, "S": 2, "W": 0})
    assert inst.plan["W"] == pytest.approx(5.0)                    # instant: W looks empty -> floor
    assert cmax.plan["W"] > cmax.plan["N"] > 5.0                   # cycle_max: remembers the queue
    assert sum(cmax.plan.values()) == pytest.approx(CTRL.green_budget_s)


def test_cycle_mean_averages_over_cycle() -> None:
    jc = _jc(aggregate="cycle_mean")
    jc.step(0.0, 0, {"N": 0, "E": 0, "S": 0, "W": 0})
    for t in range(1, 5):
        jc.step(float(t), 1, {"N": 4, "E": 0, "S": 0, "W": 0})
    act = jc.step(5.0, 0, {"N": 0, "E": 0, "S": 0, "W": 0})
    assert act.new_cycle.smoothed["N"] == pytest.approx(16 / 5)


def test_smoothing_is_applied_before_aggregation() -> None:
    jc = _jc(alpha=0.5, aggregate="instant")
    jc.step(0.0, 0, {"N": 0, "E": 0, "S": 0, "W": 0})
    jc.step(20.0, 1, {"N": 0, "E": 0, "S": 0, "W": 0})
    act = jc.step(96.0, 0, {"N": 10, "E": 0, "S": 0, "W": 0})
    assert act.new_cycle.raw["N"] == 10 and act.new_cycle.smoothed["N"] == pytest.approx(5.0)


# ----------------------------------------------------------------------------- SUMO

def _sumo_available() -> bool:
    try:
        from src.sumo_env import sumo_binary
        sumo_binary("sumo")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _sumo_available(), reason="SUMO not installed")
def test_adaptive_run_on_single_intersection(tmp_path: Path) -> None:
    from src.citygen.config import demand_config_from_dict, load_network_config
    from src.citygen.scenario import build_scenario
    from src.sim.bridge import RunConfig, run
    from src.smoothing import load_smoothing_config

    root = Path(__file__).resolve().parents[1]
    net = load_network_config(root / "configs" / "network_single.yaml")
    dem = demand_config_from_dict(dict(
        name="t", seed=3,
        time=dict(day_seconds=1500, time_scale=1.0, step_length_s=1.0),
        curve=dict(night_veh_h_lane=250, day_veh_h_lane=250, day_start_h=6.5, day_end_h=21.5,
                   ramp_width_h=0.75, peaks=[]),
        side_weights=dict(north=0.3, east=0.3, south=0.3, west=2.5),   # west-heavy
        arrivals="poisson", vehicle_mix=dict(car=1.0),
    ))
    sc = build_scenario(net, dem, tmp_path)
    ctrl = ControllerConfig(cycle_s=96, yellow_s=3, all_red_s=1, min_green_s=5, max_green_s=45)
    smooth = load_smoothing_config(root / "configs" / "smoothing.yaml")

    results = {}
    for mode in ("fixed", "adaptive"):
        stats = run(RunConfig(sumocfg=sc.cfg_path, control=mode, out_dir=tmp_path / "out"), ctrl, smooth,
                    progress_every_s=None)
        results[mode] = stats
        assert stats.vehicles_arrived > 100 and stats.cycles_logged >= 10
        assert (tmp_path / "out" / f"{stats.run_name}_stats.json").exists()

    with open(tmp_path / "out" / "single_intersection__t__adaptive_cycles.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["approach"] for r in rows} == {"N", "E", "S", "W"}
    late = [r for r in rows if float(r["time_s"]) > 600]
    mean_green = {a: sum(float(r["green_s"]) for r in late if r["approach"] == a) / max(1, sum(r["approach"] == a for r in late)) for a in "NESW"}
    assert mean_green["W"] > max(mean_green["N"], mean_green["E"], mean_green["S"])   # heavy approach gets more green
    assert all(g >= 5.0 - 1e-6 for g in mean_green.values())                          # floor honoured
    assert results["adaptive"].mean_waiting_time_s < results["fixed"].mean_waiting_time_s
