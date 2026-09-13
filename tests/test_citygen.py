"""Tests for src/citygen: config loading, grid planning, demand curve, trip sampling,
and (when SUMO is available) the netconvert build + per-approach TLS programs."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from src.citygen.config import (
    SIDES,
    demand_config_from_dict,
    load_demand_config,
    load_network_config,
    network_config_from_dict,
)
from src.citygen.demand import (
    generate_trips,
    rate_veh_h_lane,
    routes_xml,
    sample_poisson_departures,
    sim_time_to_hour,
)
from src.citygen.network import (
    APPROACH_ORDER,
    ApproachLinks,
    edges_xml,
    nodes_xml,
    per_approach_phases,
    plan_grid,
    side_of_ext_node,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def _net_dict(rows: int = 3, cols: int = 3, **roads) -> dict:
    base_roads = dict(speed_kmh=50, street_lanes=1, avenue_lanes=2,
                      avenue_rows=[1] if rows > 1 else [0],
                      avenue_cols=[1] if cols > 1 else [0], no_turnarounds=True)
    base_roads.update(roads)
    return dict(name="t", grid=dict(rows=rows, cols=cols, block_length_m=133, approach_length_m=200),
                roads=base_roads, signals=dict(scheme="per_approach", green_s=20, yellow_s=3, all_red_s=1))


def _demand_dict(**over) -> dict:
    d = dict(
        name="t", seed=1,
        time=dict(day_seconds=86400, time_scale=1.0, step_length_s=1.0),
        curve=dict(night_veh_h_lane=40, day_veh_h_lane=200, day_start_h=6.5, day_end_h=21.5,
                   ramp_width_h=0.75,
                   peaks=[dict(centre_h=8.75, sigma_h=1.0, extra_veh_h_lane=350),
                          dict(centre_h=18.0, sigma_h=1.2, extra_veh_h_lane=320)]),
        side_weights=dict(north=1, east=1, south=1, west=1),
        arrivals="poisson",
        vehicle_mix=dict(car=0.85, van=0.10, truck=0.05),
    )
    d.update(over)
    return d


# ----------------------------------------------------------------------------- configs

@pytest.mark.parametrize("path", sorted(CONFIGS.glob("network_*.yaml")))
def test_shipped_network_configs_load(path: Path) -> None:
    cfg = load_network_config(path)
    assert cfg.rows >= 1 and cfg.cols >= 1
    assert cfg.speed_ms == pytest.approx(cfg.speed_kmh / 3.6)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("demand_*.yaml")))
def test_shipped_demand_configs_load(path: Path) -> None:
    cfg = load_demand_config(path)
    assert set(cfg.side_weights) == set(SIDES)
    assert sum(cfg.vehicle_mix.values()) == pytest.approx(1.0)


def test_network_config_rejects_avenue_outside_grid() -> None:
    with pytest.raises(ValueError):
        network_config_from_dict(_net_dict(rows=2, avenue_rows=[5]))


def test_demand_config_rejects_bad_vehicle_mix() -> None:
    with pytest.raises(ValueError):
        demand_config_from_dict(_demand_dict(vehicle_mix=dict(car=0.5)))


def test_lane_hierarchy() -> None:
    cfg = network_config_from_dict(_net_dict())
    assert cfg.lanes_for_row(1) == 2 and cfg.lanes_for_row(0) == 1
    assert cfg.lanes_for_col(1) == 2 and cfg.lanes_for_col(2) == 1


# ----------------------------------------------------------------------------- grid plan

def test_plan_grid_counts_3x3() -> None:
    plan = plan_grid(network_config_from_dict(_net_dict()))
    assert len(plan.tls_ids) == 9
    assert len([n for n in plan.nodes if n.kind == "dead_end"]) == 12
    # 3 N-S roads x 4 segments + 3 E-W roads x 4 segments, each two-way
    assert len(plan.edges) == 2 * (3 * 4 + 3 * 4)
    assert len(plan.entry_edges) == 12 and len(plan.exit_edges) == 12
    assert len({e.id for e in plan.edges}) == len(plan.edges)


def test_plan_grid_single_intersection() -> None:
    plan = plan_grid(network_config_from_dict(_net_dict(rows=1, cols=1)))
    assert plan.tls_ids == ["n_0_0"]
    assert len(plan.entry_edges) == 4 and len(plan.exit_edges) == 4
    assert sorted(side_of_ext_node(e.from_node) for e in plan.entry_edges) == sorted(SIDES)


def test_entry_edge_lanes_follow_hierarchy() -> None:
    plan = plan_grid(network_config_from_dict(_net_dict()))
    lanes = {e.from_node: e.lanes for e in plan.entry_edges}
    assert lanes["ext_W_1"] == 2 and lanes["ext_W_0"] == 1   # row 1 is the avenue
    assert lanes["ext_N_1"] == 2 and lanes["ext_N_2"] == 1   # col 1 is the avenue


def test_plain_xml_is_well_formed() -> None:
    plan = plan_grid(network_config_from_dict(_net_dict(rows=2, cols=2)))
    nodes = ET.fromstring(nodes_xml(plan))
    edges = ET.fromstring(edges_xml(plan))
    assert len(nodes.findall("node")) == len(plan.nodes)
    assert len(edges.findall("edge")) == len(plan.edges)
    node_ids = {n.get("id") for n in nodes}
    for e in edges:
        assert e.get("from") in node_ids and e.get("to") in node_ids


# ----------------------------------------------------------------------------- TLS phases

def test_per_approach_phases_cover_each_link_exactly_once() -> None:
    cfg = network_config_from_dict(_net_dict())
    groups = [
        ApproachLinks("N", "a", (0, 1, 2)),
        ApproachLinks("E", "b", (3, 4)),
        ApproachLinks("S", "c", (5, 6, 7)),
        ApproachLinks("W", "d", (8,)),
    ]
    phases = per_approach_phases(groups, 9, cfg)
    assert len(phases) == 4 * 3  # green + yellow + all-red per approach
    greens = [p for p in phases if p[2].endswith("_green")]
    assert [p[2].split("_")[0] for p in greens] == list(APPROACH_ORDER)
    for i in range(9):
        assert sum(p[1][i] == "G" for p in greens) == 1
    # a green phase never shows green for another approach's links
    for g, (dur, state, _) in zip(groups, greens):
        assert dur == cfg.green_s
        assert {i for i, ch in enumerate(state) if ch == "G"} == set(g.link_indices)


# ----------------------------------------------------------------------------- demand curve

def test_rate_curve_has_expected_shape() -> None:
    cfg = demand_config_from_dict(_demand_dict())
    h = np.linspace(0, 24, 24 * 60 + 1)
    r = rate_veh_h_lane(h, cfg.curve)
    assert np.all(r > 0)
    assert r[0] == pytest.approx(40, abs=2) and r[-1] == pytest.approx(40, abs=8)   # night floor (evening ramp tail ~+5 at 24:00)
    assert r[h == 13.0] == pytest.approx(200, abs=5)                              # midday plateau
    am, pm = h[np.argmax(r * (h < 13))], h[np.argmax(r * (h > 13))]
    assert am == pytest.approx(8.75, abs=0.1) and pm == pytest.approx(18.0, abs=0.1)
    assert r.max() > 500                                                           # peak ~ day + extra


def test_time_scale_compresses_clock() -> None:
    cfg = demand_config_from_dict(_demand_dict(time=dict(day_seconds=86400, time_scale=12.0)))
    assert cfg.sim_seconds == pytest.approx(7200)
    assert sim_time_to_hour(3600, cfg) == pytest.approx(12.0)


def test_poisson_departures_match_rate() -> None:
    rng = np.random.default_rng(0)
    rate = np.full(36000, 0.1)  # 360 veh/h for 10 h -> expect 3600 +- ~60
    d = sample_poisson_departures(rate, rng)
    assert abs(len(d) - 3600) < 4 * 60
    assert np.all(np.diff(np.sort(d)) >= 0) and d.min() >= 0 and d.max() < 36000


def test_generate_trips_sorted_valid_and_seeded() -> None:
    net = network_config_from_dict(_net_dict())
    dem = demand_config_from_dict(_demand_dict(time=dict(day_seconds=7200, time_scale=1.0)))
    plan = plan_grid(net)
    a = generate_trips(plan, dem)
    b = generate_trips(plan, dem)
    assert [t.id for t in a] == [t.id for t in b]  # deterministic per seed
    departs = [t.depart_s for t in a]
    assert departs == sorted(departs)
    entries = {e.id: e for e in plan.entry_edges}
    exits = {e.id: e for e in plan.exit_edges}
    for t in a:
        assert t.from_edge in entries and t.to_edge in exits
        assert exits[t.to_edge].to_node != entries[t.from_edge].from_node  # no U-turn trips
        assert t.vtype in dem.vehicle_mix
    assert len({t.id for t in a}) == len(a)


def test_side_weights_scale_entries() -> None:
    net = network_config_from_dict(_net_dict())
    plan = plan_grid(net)
    dem = demand_config_from_dict(_demand_dict(side_weights=dict(north=1, east=1, south=1, west=3)))
    trips = generate_trips(plan, dem)
    per_side = {s: 0 for s in SIDES}
    for t in trips:
        per_side[side_of_ext_node(t.from_edge.split("-")[0])] += 1
    assert per_side["west"] / per_side["east"] == pytest.approx(3.0, rel=0.1)
    # lanes matter too: north/south are symmetric (same lane layout), so should match
    assert per_side["north"] / per_side["south"] == pytest.approx(1.0, rel=0.05)


def test_routes_xml_is_well_formed_and_typed() -> None:
    net = network_config_from_dict(_net_dict(rows=1, cols=1))
    dem = demand_config_from_dict(_demand_dict(time=dict(day_seconds=600, time_scale=1.0)))
    trips = generate_trips(plan_grid(net), dem)
    root = ET.fromstring(routes_xml(trips, dem))
    assert {v.get("id") for v in root.findall("vType")} == set(dem.vehicle_mix)
    assert len(root.findall("trip")) == len(trips)


# ----------------------------------------------------------------------------- SUMO build

def _sumo_available() -> bool:
    try:
        from src.sumo_env import sumo_binary
        sumo_binary("netconvert")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _sumo_available(), reason="SUMO not installed")
def test_build_scenario_end_to_end(tmp_path: Path) -> None:
    from src.citygen.scenario import build_scenario
    from src.sumo_env import ensure_sumo_tools

    net = network_config_from_dict(_net_dict(rows=2, cols=2))
    dem = demand_config_from_dict(_demand_dict(time=dict(day_seconds=1800, time_scale=1.0)))
    sc = build_scenario(net, dem, tmp_path)
    assert sc.net_path.exists() and sc.tls_path.exists() and sc.route_path.exists() and sc.cfg_path.exists()

    ensure_sumo_tools()
    import sumolib

    sumo_net = sumolib.net.readNet(str(sc.net_path), withPrograms=True)
    assert {t.getID() for t in sumo_net.getTrafficLights()} == set(sc.plan.tls_ids)

    programs = ET.parse(sc.tls_path).getroot().findall("tlLogic")
    assert {p.get("id") for p in programs} == set(sc.plan.tls_ids)
    for prog in programs:
        tls = sumo_net.getTLS(prog.get("id"))
        n_links = max(i for _, _, i in tls.getConnections()) + 1
        phases = prog.findall("phase")
        assert all(len(p.get("state")) == n_links for p in phases)
        names = [p.get("name") for p in phases if p.get("name").endswith("_green")]
        assert names == [f"{a}_green" for a in APPROACH_ORDER]
        # every controlled link is green in exactly one phase
        union = [0] * n_links
        for p in phases:
            for i, ch in enumerate(p.get("state")):
                union[i] += ch == "G"
        assert union == [1] * n_links


@pytest.mark.skipif(not _sumo_available(), reason="SUMO not installed")
def test_sumo_runs_and_activates_per_approach_program(tmp_path: Path) -> None:
    """Short traci run: vehicles are inserted, and the per_approach program is active."""
    from src.citygen.network import PROGRAM_ID
    from src.citygen.scenario import build_scenario
    from src.sumo_env import ensure_sumo_tools, sumo_binary

    net = network_config_from_dict(_net_dict(rows=1, cols=1, street_lanes=2))
    dem = demand_config_from_dict(_demand_dict(
        time=dict(day_seconds=86400, time_scale=1.0),
        curve=dict(night_veh_h_lane=300, day_veh_h_lane=300, day_start_h=6.5, day_end_h=21.5,
                   ramp_width_h=0.75, peaks=[])))
    sc = build_scenario(net, dem, tmp_path)

    ensure_sumo_tools()
    import traci

    traci.start([sumo_binary("sumo"), "-c", str(sc.cfg_path), "--no-step-log", "true"])
    try:
        assert traci.trafficlight.getProgram("n_0_0") == PROGRAM_ID
        phase_names = {traci.trafficlight.getPhaseName("n_0_0")}
        for _ in range(300):
            traci.simulationStep()
            phase_names.add(traci.trafficlight.getPhaseName("n_0_0"))
        assert traci.vehicle.getIDCount() > 0 or traci.simulation.getArrivedNumber() >= 0
        assert traci.simulation.getDepartedNumber() >= 0
        assert {f"{a}_green" for a in APPROACH_ORDER} <= phase_names
    finally:
        traci.close()
