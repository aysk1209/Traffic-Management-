"""CLI: build a SUMO scenario from a network config and a demand config."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.citygen.config import load_demand_config, load_network_config
from src.citygen.scenario import build_scenario


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.citygen", description=__doc__)
    ap.add_argument("--network", required=True, help="network YAML, e.g. configs/network_eixample.yaml")
    ap.add_argument("--demand", required=True, help="demand YAML, e.g. configs/demand_balanced.yaml")
    ap.add_argument("--out", default="sumo_scenarios", help="output directory (default: sumo_scenarios)")
    args = ap.parse_args(argv)

    net_cfg = load_network_config(args.network)
    demand_cfg = load_demand_config(args.demand)
    sc = build_scenario(net_cfg, demand_cfg, Path(args.out))
    print(f"scenario   : {sc.name}")
    print(f"network    : {sc.net_path}  ({len(sc.plan.tls_ids)} signalized junctions, "
          f"{len(sc.plan.entry_edges)} perimeter entries)")
    print(f"routes     : {sc.route_path}  ({len(sc.trips)} trips over {demand_cfg.sim_seconds/3600:.1f} sim-hours)")
    print(f"tls        : {sc.tls_path}")
    print(f"run        : sumo-gui -c {sc.cfg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
