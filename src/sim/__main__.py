"""CLI: run a scenario under fixed or adaptive signal control.

    python -m src.sim --scenario sumo_scenarios/eixample_3x3__balanced.sumocfg --control adaptive
    python -m src.sim --scenario ... --control adaptive --no-smoothing   # raw counts into the controller
    python -m src.sim --scenario ... --control fixed                     # baseline, same logging
    python -m src.sim --scenario ... --gui --begin 25200 --end 39600     # watch 07:00-11:00
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.controller import load_controller_config
from src.sim.bridge import CONTROL_MODES, RunConfig, run
from src.sim.junction import AGGREGATES
from src.smoothing import SmoothingConfig, load_smoothing_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, help=".sumocfg built by src.citygen")
    ap.add_argument("--control", choices=CONTROL_MODES, default="adaptive")
    ap.add_argument("--no-smoothing", action="store_true", help="feed raw counts to the controller (alpha=1)")
    ap.add_argument("--controller", default="configs/controller.yaml")
    ap.add_argument("--smoothing", default="configs/smoothing.yaml")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--begin", type=float, default=None, help="sim start time (s); earlier departures are dropped")
    ap.add_argument("--end", type=float, default=None, help="sim end time (s)")
    ap.add_argument("--out", default="sumo_scenarios/output")
    ap.add_argument("--label", default=None, help="output file stem (default: <scenario>__<control>[_nosmooth])")
    ap.add_argument("--seed", type=int, default=None, help="SUMO random seed")
    ap.add_argument("--aggregate", choices=AGGREGATES, default="cycle_max",
                    help="how smoothed estimates are condensed per cycle (default: cycle_max)")
    args = ap.parse_args(argv)

    controller_cfg = load_controller_config(args.controller)
    smoothing_cfg = SmoothingConfig.passthrough() if args.no_smoothing else load_smoothing_config(args.smoothing)
    label = args.label or (f"{Path(args.scenario).stem}__{args.control}" + ("_nosmooth" if args.no_smoothing else ""))
    cfg = RunConfig(sumocfg=Path(args.scenario), control=args.control, gui=args.gui,
                    begin_s=args.begin, end_s=args.end, out_dir=Path(args.out), label=label, seed=args.seed,
                    aggregate=args.aggregate)
    print(f"run {cfg.run_name}: control={cfg.control} alpha={smoothing_cfg.alpha:.3f} "
          f"cycle={controller_cfg.cycle_s:g}s floor={controller_cfg.min_green_s:g}s cap={controller_cfg.max_green_s}")
    stats = run(cfg, controller_cfg, smoothing_cfg)
    print(f"arrived={stats.vehicles_arrived}  mean travel={stats.mean_travel_time_s:.1f}s  "
          f"mean wait={stats.mean_waiting_time_s:.1f}s  mean time loss={stats.mean_time_loss_s:.1f}s  "
          f"teleports={stats.teleports}  max halting={stats.max_halting}  cycles={stats.cycles_logged}")
    print(f"outputs: {cfg.out_dir / (cfg.run_name + '_{cycles.csv,stats.json,tripinfo.xml,summary.xml}')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
