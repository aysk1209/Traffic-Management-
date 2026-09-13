"""Run the validation matrix (demand profiles x control modes) and tabulate results.

    python scripts/run_experiments.py --network configs/network_eixample.yaml \
        --demands light balanced imbalanced heavy \
        --modes fixed adaptive adaptive_nosmooth \
        [--begin 21600 --end 46800]   # e.g. 06:00-13:00 only

Each (profile, mode) run writes its own outputs to sumo_scenarios/output/ via
src.sim; this script builds the scenarios first, runs everything sequentially and
writes a summary table to sumo_scenarios/output/experiments_<network>.md (and .csv).
Runs whose stats.json already exists are skipped unless --force is given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citygen.config import load_demand_config, load_network_config  # noqa: E402
from src.citygen.scenario import build_scenario  # noqa: E402
from src.controller import load_controller_config  # noqa: E402
from src.sim.bridge import RunConfig, RunStats, run  # noqa: E402
from src.smoothing import SmoothingConfig, load_smoothing_config  # noqa: E402

MODES = {
    "fixed": dict(control="fixed", smoothing=True),
    "adaptive": dict(control="adaptive", smoothing=True),
    "adaptive_nosmooth": dict(control="adaptive", smoothing=False),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--network", default="configs/network_eixample.yaml")
    ap.add_argument("--demands", nargs="+", default=["light", "balanced", "imbalanced", "heavy"])
    ap.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    ap.add_argument("--begin", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--out", default="sumo_scenarios/output")
    ap.add_argument("--scenarios", default="sumo_scenarios")
    ap.add_argument("--force", action="store_true", help="re-run even if stats.json exists")
    args = ap.parse_args(argv)

    net_cfg = load_network_config(args.network)
    controller_cfg = load_controller_config("configs/controller.yaml")
    smoothing_cfg = load_smoothing_config("configs/smoothing.yaml")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for demand in args.demands:
        dem_cfg = load_demand_config(f"configs/demand_{demand}.yaml")
        sc = build_scenario(net_cfg, dem_cfg, Path(args.scenarios))
        for mode in args.modes:
            spec = MODES[mode]
            label = f"{sc.name}__{mode}"
            stats_path = out_dir / f"{label}_stats.json"
            if stats_path.exists() and not args.force:
                stats = RunStats(**json.loads(stats_path.read_text(encoding="utf-8")))
                print(f"[skip] {label} (cached)")
            else:
                print(f"[run ] {label}", flush=True)
                t0 = time.time()
                stats = run(
                    RunConfig(sumocfg=sc.cfg_path, control=spec["control"], begin_s=args.begin,
                              end_s=args.end, out_dir=out_dir, label=label),
                    controller_cfg,
                    smoothing_cfg if spec["smoothing"] else SmoothingConfig.passthrough(),
                )
                print(f"       done in {(time.time() - t0) / 60:.1f} min: wait={stats.mean_waiting_time_s:.1f}s "
                      f"travel={stats.mean_travel_time_s:.1f}s teleports={stats.teleports}", flush=True)
            rows.append(dict(demand=demand, mode=mode, **stats.as_dict()))

    _write_table(rows, out_dir / f"experiments_{net_cfg.name}")
    return 0


def _write_table(rows: list[dict], stem: Path) -> None:
    cols = ["demand", "mode", "vehicles_arrived", "mean_travel_time_s", "mean_waiting_time_s",
            "mean_time_loss_s", "teleports", "max_halting"]
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = [f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {stem.with_suffix('.md')} / .csv")


if __name__ == "__main__":
    raise SystemExit(main())
