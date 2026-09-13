"""Render the project's result figures from the simulation outputs into docs/figures/.

    python scripts/plot_results.py [--full-day sumo_scenarios/output/full_day]
        [--noise sumo_scenarios/output/noise_window] [--out docs/figures]

Figures:
  control_comparison.png     fixed vs adaptive mean wait / travel time per demand profile (full day)
  day_timeline_heavy.png     vehicles running and halting over the day, fixed vs adaptive, heavy profile
  green_allocation.png       green time per approach over the day at the west-entry junction (imbalanced)
  smoothing_experiment.png   mean wait and green oscillation for the noise / smoothing / aggregation conditions
"""
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette, fixed slot order (see dataviz skill / references/palette.md)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
PROFILES = ["light", "balanced", "imbalanced", "heavy"]
APPROACHES = ["N", "E", "S", "W"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def _stats(d: Path, profile: str, mode: str) -> dict:
    return json.loads((d / f"eixample_3x3__{profile}__{mode}_stats.json").read_text(encoding="utf-8"))


def control_comparison(full_day: Path, out: Path) -> None:
    fixed = [_stats(full_day, p, "fixed") for p in PROFILES]
    adapt = [_stats(full_day, p, "adaptive") for p in PROFILES]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, ["mean_waiting_time_s", "mean_travel_time_s"],
                              ["Mean waiting time per vehicle (s)", "Mean travel time per vehicle (s)"]):
        x = np.arange(len(PROFILES))
        w = 0.36
        fv = [s[key] for s in fixed]
        av = [s[key] for s in adapt]
        b1 = ax.bar(x - w / 2 - 0.01, fv, w, color=SERIES[0], label="fixed 20 s timer")
        b2 = ax.bar(x + w / 2 + 0.01, av, w, color=SERIES[1], label="adaptive (this project)")
        ax.set_xticks(x, PROFILES)
        ax.set_title(title, loc="left", color=INK)
        # gridlocked runs are an order of magnitude off; clip them so the rest stays readable
        gridlocked = [s["teleports"] > 100 for s in fixed]
        in_scale = [v for v, g in zip(fv, gridlocked) if not g] + av
        top = max(in_scale) * 1.3
        ax.set_ylim(0, top)
        for bars, vals, flags in ((b1, fv, gridlocked), (b2, av, [False] * len(av))):
            for bar, v, g in zip(bars, vals, flags):
                if g:
                    bar.set_height(top * 0.8)
                    bar.set_hatch("///")
                    bar.set_edgecolor(SURFACE)
                    ax.annotate(f"{v:.0f}\noff scale, gridlock", (bar.get_x() + bar.get_width() / 2, top * 0.8),
                                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                                fontsize=8, color=INK2)
                else:
                    ax.annotate(f"{v:.0f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 3),
                                textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("3x3 grid, full simulated day (24 h): fixed timing vs. queue-proportional control",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "control_comparison.png", dpi=130)
    plt.close(fig)


def _summary_series(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t, running, halting = [], [], []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "step":
            t.append(float(el.get("time")))
            running.append(int(el.get("running")))
            halting.append(int(el.get("halting")))
            el.clear()
    return np.array(t) / 3600.0, np.array(running), np.array(halting)


def day_timeline(full_day: Path, out: Path, profile: str = "heavy") -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for mode, colour in (("fixed", SERIES[0]), ("adaptive", SERIES[1])):
        h, running, halting = _summary_series(full_day / f"eixample_3x3__{profile}__{mode}_summary.xml")
        k = 300  # 5-minute moving average for readability
        kern = np.ones(k) / k
        for ax, y in zip(axes, (running, halting)):
            ax.plot(h[k - 1:], np.convolve(y, kern, mode="valid"), color=colour, lw=2, label=f"{mode}")
    axes[0].set_title("Vehicles in the network", loc="left", color=INK)
    axes[1].set_title("Vehicles halting (speed < 0.1 m/s)", loc="left", color=INK)
    axes[1].set_xlabel("clock hour")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 2))
    axes[0].legend(frameon=False, loc="upper left")
    for ax in axes:
        ax.axvspan(7.75, 9.75, color=GRID, alpha=0.6, lw=0)
        ax.axvspan(16.8, 19.2, color=GRID, alpha=0.6, lw=0)
    axes[0].annotate("rush hours shaded", (9.75, axes[0].get_ylim()[1] * 0.92), fontsize=8.5, color=INK2)
    fig.suptitle(f"'{profile}' demand over one day: fixed timer collapses into gridlock, adaptive control does not",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"day_timeline_{profile}.png", dpi=130)
    plt.close(fig)


def green_allocation(full_day: Path, out: Path, profile: str = "imbalanced", tls: str = "n_1_0") -> None:
    rows = []
    with open(full_day / f"eixample_3x3__{profile}__adaptive_cycles.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["tls"] == tls:
                rows.append(r)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for a, colour in zip(APPROACHES, SERIES):
        t = np.array([float(r["time_s"]) for r in rows if r["approach"] == a]) / 3600.0
        g = np.array([float(r["green_s"]) for r in rows if r["approach"] == a])
        q = np.array([float(r["smoothed"]) for r in rows if r["approach"] == a])
        k = 15  # ~24 min moving average over cycles
        kern = np.ones(k) / k
        axes[0].plot(t[k - 1:], np.convolve(q, kern, mode="valid"), color=colour, lw=2, label=f"from {a}")
        axes[1].plot(t[k - 1:], np.convolve(g, kern, mode="valid"), color=colour, lw=2, label=f"from {a}")
    axes[0].set_title(f"Smoothed queue estimate per approach at junction {tls} (vehicles)", loc="left", color=INK)
    axes[1].set_title("Green time allocated per approach (s of a 96 s cycle)", loc="left", color=INK)
    axes[1].axhline(20, color=INK2, lw=1, ls="--")
    axes[1].annotate("fixed timer: 20 s each", (23.9, 20.6), fontsize=8.5, color=INK2, ha="right")
    axes[1].axhline(5, color=GRID, lw=1)
    axes[1].axhline(45, color=GRID, lw=1)
    axes[1].annotate("cap 45 s", (23.9, 45.6), fontsize=8.5, color=INK2, ha="right")
    axes[1].annotate("floor 5 s", (23.9, 5.6), fontsize=8.5, color=INK2, ha="right")
    axes[1].set_ylim(0, 50)
    axes[1].set_xlabel("clock hour")
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks(range(0, 25, 2))
    axes[0].legend(frameon=False, loc="upper left", ncol=4)
    fig.suptitle(f"'{profile}' demand (west side x2.6): the busiest approach is given the most green all day",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "green_allocation.png", dpi=130)
    plt.close(fig)


def smoothing_experiment(noise: Path, out: Path) -> None:
    conditions = [
        ("fixed timer", "eixample_3x3__imbalanced__fixed"),
        ("instant · clean · raw", "inst_clean_raw"),
        ("instant · noisy · raw", "inst_noisy_raw"),
        ("instant · noisy · EMA", "inst_noisy_smooth"),
        ("cycle-peak · clean · raw", "eixample_3x3__imbalanced__adaptive_nosmooth"),
        ("cycle-peak · noisy · raw", "eixample_3x3__imbalanced__adaptive_nosmooth_noisy"),
        ("cycle-peak · noisy · EMA", "eixample_3x3__imbalanced__adaptive_noisy"),
    ]
    stats = [json.loads((noise / f"{stem}_stats.json").read_text(encoding="utf-8")) for _, stem in conditions]
    labels = [c[0] for c in conditions]
    colours = [INK2] + [SERIES[0]] * 3 + [SERIES[1]] * 3
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    y = np.arange(len(conditions))[::-1]
    for ax, key, title in zip(axes, ["mean_waiting_time_s", "green_oscillation_s"],
                              ["Mean waiting time (s)", "Green-time oscillation (s per cycle)"]):
        vals = [s[key] for s in stats]
        bars = ax.barh(y, vals, color=colours, height=0.62)
        ax.set_yticks(y, labels)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlim(0, max(vals) * 1.22)
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}", (v, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    axes[0].annotate("blue: single-frame decisions · orange: previous-cycle peak",
                     (0, -0.16), xycoords="axes fraction", fontsize=8.5, color=INK2)
    fig.suptitle("Imbalanced demand, 07:00-11:00: filtering the count signal (EMA and/or per-cycle aggregation)",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "smoothing_experiment.png", dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full-day", default="sumo_scenarios/output/full_day")
    ap.add_argument("--noise", default="sumo_scenarios/output/noise_window")
    ap.add_argument("--out", default="docs/figures")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    control_comparison(Path(args.full_day), out)
    day_timeline(Path(args.full_day), out)
    green_allocation(Path(args.full_day), out)
    smoothing_experiment(Path(args.noise), out)
    print(f"figures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
