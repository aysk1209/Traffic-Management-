"""Capture an auto-labelled detection dataset from sumo-gui and lay it out for YOLOv6.

    python scripts/build_detection_dataset.py [--out sumo_scenarios/output/detection/dataset]
        [--frames-per-job 40] [--val-fraction 0.2]

Each job below opens a scenario in sumo-gui at a given junction, time and zoom and
captures frames every few seconds; every visible vehicle gets a YOLO box computed
from traci state (src/detection/labels.py). Output layout (YOLOv6 convention):

    <out>/images/{train,val}/*.png
    <out>/labels/{train,val}/*.txt
    <out>/dataset.yaml
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.citygen.config import load_demand_config, load_network_config  # noqa: E402
from src.citygen.scenario import build_scenario  # noqa: E402
from src.detection.capture import CaptureConfig, capture  # noqa: E402


@dataclass(frozen=True)
class Job:
    network: str
    demand: str
    tls_id: str
    begin_s: float
    view_width_m: float
    interval_s: float = 4.0


# junction x time-of-day x zoom variety; times chosen across the diurnal curve
JOBS = [
    Job("network_single.yaml", "demand_imbalanced.yaml", "n_0_0", 31000, 160),   # 08:36 peak, skewed
    Job("network_single.yaml", "demand_balanced.yaml", "n_0_0", 46800, 200),     # 13:00 midday
    Job("network_single.yaml", "demand_light.yaml", "n_0_0", 79200, 160),        # 22:00 sparse
    Job("network_eixample.yaml", "demand_balanced.yaml", "n_1_1", 31500, 160),   # grid centre, avenue x avenue
    Job("network_eixample.yaml", "demand_heavy.yaml", "n_0_0", 32400, 220),      # grid corner, dense queues
    Job("network_eixample.yaml", "demand_balanced.yaml", "n_2_1", 64800, 180),   # 18:00 evening peak
    Job("network_eixample.yaml", "demand_light.yaml", "n_1_2", 27000, 200),      # 07:30 ramp-up
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sumo_scenarios/output/detection/dataset")
    ap.add_argument("--frames-per-job", type=int, default=40)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    out = Path(args.out)
    raw = out / "raw"
    pairs: list[tuple[Path, Path]] = []
    for i, job in enumerate(JOBS):
        net_cfg = load_network_config(Path("configs") / job.network)
        dem_cfg = load_demand_config(Path("configs") / job.demand)
        sc = build_scenario(net_cfg, dem_cfg, Path("sumo_scenarios"))
        job_dir = raw / f"job{i:02d}_{sc.name}_{job.tls_id}_{int(job.begin_s)}"
        if not (job_dir / "frames.csv").exists():
            print(f"[capture] {job_dir.name}: {args.frames_per_job} frames", flush=True)
            capture(CaptureConfig(sumocfg=sc.cfg_path, tls_id=job.tls_id, out_dir=job_dir, begin_s=job.begin_s,
                                  n_frames=args.frames_per_job, interval_s=job.interval_s,
                                  view_width_m=job.view_width_m, write_labels=True))
        else:
            print(f"[cached ] {job_dir.name}")
        for png in sorted((job_dir / "frames").glob("*.png")):
            label = job_dir / "labels" / f"{png.stem}.txt"
            if label.exists():
                pairs.append((png, label))

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_fraction))
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}
    for split, items in splits.items():
        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for png, label in items:
            name = f"{png.parent.parent.name}_{png.stem}"
            shutil.copyfile(png, img_dir / f"{name}.png")
            shutil.copyfile(label, lbl_dir / f"{name}.txt")
    (out / "dataset.yaml").write_text(
        f"train: {(out / 'images' / 'train').resolve().as_posix()}\n"
        f"val: {(out / 'images' / 'val').resolve().as_posix()}\n"
        f"test: {(out / 'images' / 'val').resolve().as_posix()}\n"
        "is_coco: False\nnc: 1\nnames: ['vehicle']\n", encoding="utf-8")
    n_boxes = sum(1 for _, l in pairs for line in open(l, encoding="utf-8") if line.strip())
    print(f"dataset: {len(splits['train'])} train / {len(splits['val'])} val frames, {n_boxes} boxes -> {out / 'dataset.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
