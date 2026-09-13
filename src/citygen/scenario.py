"""Assemble a runnable SUMO scenario (net + routes + TLS programs + .sumocfg).

Usage (CLI)::

    python -m src.citygen --network configs/network_eixample.yaml \
                          --demand configs/demand_balanced.yaml [--out sumo_scenarios]

writes ``sumo_scenarios/<network>__<demand>.sumocfg`` plus its inputs. Network files are
shared between demand scenarios (they only depend on the network config).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.citygen.config import DemandConfig, NetworkConfig
from src.citygen.demand import Trip, generate_trips, write_routes
from src.citygen.network import GridPlan, build_network


@dataclass(frozen=True)
class Scenario:
    name: str
    net_path: Path
    tls_path: Path
    route_path: Path
    cfg_path: Path
    plan: GridPlan
    trips: list[Trip]


def sumocfg_xml(net: Path, routes: Path, tls: Path, demand: DemandConfig, base: Path) -> str:
    rel = lambda p: p.relative_to(base).as_posix()  # noqa: E731
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="{rel(net)}"/>
        <route-files value="{rel(routes)}"/>
        <additional-files value="{rel(tls)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{demand.sim_seconds:g}"/>
        <step-length value="{demand.step_length_s:g}"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
        <ignore-route-errors value="true"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <duration-log.statistics value="true"/>
    </report>
</configuration>
"""


def build_scenario(net_cfg: NetworkConfig, demand_cfg: DemandConfig, out_dir: Path) -> Scenario:
    out_dir = Path(out_dir)
    net_path, tls_path, plan = build_network(net_cfg, out_dir)
    name = f"{net_cfg.name}__{demand_cfg.name}"
    trips = generate_trips(plan, demand_cfg, np.random.default_rng(demand_cfg.seed))
    route_path = write_routes(trips, demand_cfg, out_dir / f"{name}.rou.xml")
    cfg_path = out_dir / f"{name}.sumocfg"
    cfg_path.write_text(sumocfg_xml(net_path, route_path, tls_path, demand_cfg, out_dir), encoding="utf-8")
    return Scenario(name, net_path, tls_path, route_path, cfg_path, plan, trips)
