"""One-command demo.

    python scripts/demo.py watch     # sumo-gui: 3x3 grid, imbalanced demand, 08:00-09:30, adaptive control
    python scripts/demo.py compare   # headless: fixed vs adaptive on the same window, prints a table (~3 min)
    python scripts/demo.py watch --control fixed   # the baseline, for contrast
    python scripts/demo.py watch --network single  # one intersection instead of the grid

Scenarios are built on first use (a few seconds). Everything else is the normal
pipeline: src.citygen -> src.sim; this script only chooses sensible arguments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citygen.config import load_demand_config, load_network_config  # noqa: E402
from src.citygen.scenario import build_scenario  # noqa: E402
from src.controller import load_controller_config  # noqa: E402
from src.sim.bridge import RunConfig, run  # noqa: E402
from src.smoothing import load_smoothing_config  # noqa: E402

NETWORKS = {"grid": "configs/network_eixample.yaml", "single": "configs/network_single.yaml"}
WINDOW = (8 * 3600, 9 * 3600 + 1800)  # 08:00 - 09:30, the morning peak


def scenario_path(network: str, demand: str) -> Path:
    net_cfg = load_network_config(NETWORKS[network])
    dem_cfg = load_demand_config(f"configs/demand_{demand}.yaml")
    cfg_path = Path("sumo_scenarios") / f"{net_cfg.name}__{dem_cfg.name}.sumocfg"
    if not cfg_path.exists():
        print(f"building scenario {cfg_path.stem} ...", flush=True)
        build_scenario(net_cfg, dem_cfg, Path("sumo_scenarios"))
    return cfg_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["watch", "compare"])
    ap.add_argument("--network", choices=list(NETWORKS), default="grid")
    ap.add_argument("--demand", default="imbalanced", choices=["light", "balanced", "imbalanced", "heavy"])
    ap.add_argument("--control", choices=["fixed", "adaptive"], default="adaptive", help="watch mode only")
    args = ap.parse_args(argv)

    sumocfg = scenario_path(args.network, args.demand)
    controller_cfg = load_controller_config("configs/controller.yaml")
    smoothing_cfg = load_smoothing_config("configs/smoothing.yaml")
    begin, end = WINDOW
    out = Path("sumo_scenarios/output/demo")

    if args.mode == "watch":
        print(f"opening sumo-gui: {sumocfg.stem}, {args.control} control, 08:00-09:30 "
              f"(zoom in on a junction; close the window to stop)", flush=True)
        stats = run(RunConfig(sumocfg=sumocfg, control=args.control, gui=True, begin_s=begin, end_s=end,
                              out_dir=out, label=f"demo_{sumocfg.stem}__{args.control}"),
                    controller_cfg, smoothing_cfg, progress_every_s=900)
        print(f"{args.control}: mean wait {stats.mean_waiting_time_s:.0f} s, "
              f"mean travel {stats.mean_travel_time_s:.0f} s, teleports {stats.teleports}")
        return 0

    results = {}
    for control in ("fixed", "adaptive"):
        print(f"running {control} ...", flush=True)
        results[control] = run(RunConfig(sumocfg=sumocfg, control=control, begin_s=begin, end_s=end,
                                         out_dir=out, label=f"demo_{sumocfg.stem}__{control}"),
                               controller_cfg, smoothing_cfg, progress_every_s=None)
    f, a = results["fixed"], results["adaptive"]
    print(f"\n{sumocfg.stem}, 08:00-09:30")
    print(f"{'':10s} {'mean wait':>10s} {'mean travel':>12s} {'arrived':>8s} {'teleports':>10s} {'max queue':>10s}")
    for name, s in (("fixed", f), ("adaptive", a)):
        print(f"{name:10s} {s.mean_waiting_time_s:9.0f}s {s.mean_travel_time_s:11.0f}s {s.vehicles_arrived:8d} "
              f"{s.teleports:10d} {s.max_halting:10d}")
    print(f"\nadaptive vs fixed: wait {100 * (a.mean_waiting_time_s / f.mean_waiting_time_s - 1):+.0f} %, "
          f"travel {100 * (a.mean_travel_time_s / f.mean_travel_time_s - 1):+.0f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
