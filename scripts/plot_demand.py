"""Plot the analytic diurnal demand curve against the trips actually generated.

    python scripts/plot_demand.py --network configs/network_eixample.yaml \
        --demand configs/demand_balanced.yaml [--out sumo_scenarios/demand_balanced.png]

Top panel: rate per entry lane (veh/h) as defined by the config.
Bottom panel: total sampled departures across the whole network, binned per 15 min,
split by entry side — this is what SUMO will actually see.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citygen.config import SIDES, load_demand_config, load_network_config  # noqa: E402
from src.citygen.demand import generate_trips, rate_veh_h_lane, sim_time_to_hour  # noqa: E402
from src.citygen.network import plan_grid, side_of_ext_node  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--network", required=True)
    ap.add_argument("--demand", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    net_cfg = load_network_config(args.network)
    dem_cfg = load_demand_config(args.demand)
    plan = plan_grid(net_cfg)
    trips = generate_trips(plan, dem_cfg)
    out = Path(args.out) if args.out else Path("sumo_scenarios") / f"demand_{net_cfg.name}__{dem_cfg.name}.png"

    hours = np.linspace(0, dem_cfg.day_seconds / 3600.0, 24 * 60)
    curve = rate_veh_h_lane(hours, dem_cfg.curve)

    entry_side = {e.id: side_of_ext_node(e.from_node) for e in plan.entry_edges}
    depart_h = np.array([sim_time_to_hour(t.depart_s, dem_cfg) for t in trips])
    sides = np.array([entry_side[t.from_edge] for t in trips])
    bin_h = 0.25
    bins = np.arange(0, dem_cfg.day_seconds / 3600.0 + bin_h, bin_h)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(hours, curve, color="#1f4e79", lw=2)
    for p in dem_cfg.curve.peaks:
        ax1.axvline(p.centre_h, color="#c0504d", ls="--", lw=1)
    ax1.set_ylabel("veh / h per entry lane")
    ax1.set_title(f"Demand profile '{dem_cfg.name}' on network '{net_cfg.name}'")
    ax1.grid(alpha=0.3)

    stack = [np.histogram(depart_h[sides == s], bins=bins)[0] / bin_h for s in SIDES]
    ax2.stackplot(bins[:-1] + bin_h / 2, stack, labels=SIDES, alpha=0.85)
    ax2.set_ylabel("sampled departures (veh / h, whole network)")
    ax2.set_xlabel("clock hour")
    ax2.set_xlim(0, dem_cfg.day_seconds / 3600.0)
    ax2.set_xticks(range(0, 25, 2))
    ax2.legend(loc="upper left", ncol=4)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)

    total = len(trips)
    peak_bin = bins[np.argmax(sum(stack))]
    print(f"{total} trips; busiest 15-min bin starts at {peak_bin:.2f} h "
          f"({max(sum(stack)):.0f} veh/h network-wide); per-side totals: "
          + ", ".join(f"{s}={int((sides == s).sum())}" for s in SIDES))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
