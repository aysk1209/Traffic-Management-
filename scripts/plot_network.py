"""Render a built SUMO network to PNG for a quick visual check of the layout.

    python scripts/plot_network.py sumo_scenarios/eixample_3x3.net.xml [--out file.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sumo_env import ensure_sumo_tools  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("net")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    ensure_sumo_tools()
    import sumolib

    net = sumolib.net.readNet(args.net)
    out = Path(args.out) if args.out else Path(args.net).with_suffix(".png")

    fig, ax = plt.subplots(figsize=(9, 9))
    for edge in net.getEdges():
        for lane in edge.getLanes():
            xs, ys = zip(*lane.getShape())
            ax.plot(xs, ys, color="#555", lw=1.2, solid_capstyle="round")
    for node in net.getNodes():
        x, y = node.getCoord()
        is_tls = node.getType() == "traffic_light"
        ax.plot(x, y, "o", ms=9 if is_tls else 5, color="#c0504d" if is_tls else "#4f81bd", zorder=3)
        ax.annotate(node.getID(), (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_aspect("equal")
    ax.set_title(f"{Path(args.net).name}: {len(net.getTrafficLights())} signalized junctions, "
                 f"{len(net.getEdges())} edges (red = signalized, blue = perimeter)")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
