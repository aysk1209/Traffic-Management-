"""Detection proof-of-concept: capture frames, detect vehicles, compare per-approach
counts with traci ground truth.

    python -m src.detection capture --scenario sumo_scenarios/single_intersection__imbalanced.sumocfg \
        --tls n_0_0 --begin 30600 --frames 30 --out sumo_scenarios/output/detection/poc
    python -m src.detection evaluate --capture sumo_scenarios/output/detection/poc \
        [--weights <fine-tuned .pt>] [--conf 0.25] [--annotate]

``evaluate`` writes ``<capture>/detections.csv`` (frame, approach, truth, detected) and
``<capture>/evaluation.json`` (MAE, bias, correlation per approach and overall), and with
``--annotate`` saves frames with detections (green) and ROIs (red) drawn.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _capture(args: argparse.Namespace) -> int:
    from src.detection.capture import CaptureConfig, capture

    out = capture(CaptureConfig(sumocfg=Path(args.scenario), tls_id=args.tls, out_dir=Path(args.out),
                                begin_s=args.begin, n_frames=args.frames, interval_s=args.interval,
                                view_width_m=args.view_width, write_labels=True))
    print(f"captured {args.frames} frames -> {out}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    import cv2

    from src.detection.count import count_per_roi
    from src.detection.roi import Box
    from src.detection.yolo import DEFAULT_WEIGHTS, YoloV6Detector

    cap = Path(args.capture)
    rois_json = json.loads((cap / "rois.json").read_text(encoding="utf-8"))
    rois = {a: Box(*v) for a, v in rois_json["rois_px"].items()}
    weights = Path(args.weights) if args.weights else DEFAULT_WEIGHTS
    det = YoloV6Detector(weights=weights, conf_thres=args.conf, img_size=args.img_size,
                         class_names={0: "vehicle"} if args.weights else None)
    print(f"weights: {weights}  conf>={args.conf}  img={args.img_size}")

    with open(cap / "frames.csv", newline="", encoding="utf-8") as fh:
        frames = list(csv.DictReader(fh))
    approaches = [a for a in frames[0] if a not in ("frame", "time_s")]
    rows: list[dict] = []
    ann_dir = cap / "annotated"
    if args.annotate:
        ann_dir.mkdir(exist_ok=True)
    for rec in frames:
        img = cv2.imread(str(cap / "frames" / rec["frame"]))
        dets = det.detect(img)
        counts = count_per_roi(dets, rois, min_score=args.conf, labels=frozenset(det.names.values()))
        for a in approaches:
            rows.append(dict(frame=rec["frame"], time_s=rec["time_s"], approach=a,
                             truth=int(rec[a]), detected=int(counts.get(a, 0))))
        if args.annotate:
            for d in dets:
                cv2.rectangle(img, (int(d.box.x0), int(d.box.y0)), (int(d.box.x1), int(d.box.y1)), (0, 255, 0), 2)
            for a, r in rois.items():
                cv2.rectangle(img, (int(r.x0), int(r.y0)), (int(r.x1), int(r.y1)), (0, 0, 255), 2)
                cv2.putText(img, f"{a}: det {counts.get(a, 0)} / gt {rec[a]}", (int(r.x0) + 6, int(r.y0) + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(str(ann_dir / rec["frame"]), img)

    with open(cap / "detections.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "time_s", "approach", "truth", "detected"])
        w.writeheader()
        w.writerows(rows)

    summary = {"weights": str(weights), "conf": args.conf, "frames": len(frames), "per_approach": {}}
    all_t, all_d = [], []
    for a in approaches:
        t = np.array([r["truth"] for r in rows if r["approach"] == a], dtype=float)
        d = np.array([r["detected"] for r in rows if r["approach"] == a], dtype=float)
        all_t.extend(t); all_d.extend(d)
        summary["per_approach"][a] = _stats(t, d)
    summary["overall"] = _stats(np.array(all_t), np.array(all_d))
    (cap / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{'approach':9s} {'mean gt':>8s} {'mean det':>9s} {'MAE':>6s} {'bias':>6s} {'corr':>6s}")
    for a, s in list(summary["per_approach"].items()) + [("overall", summary["overall"])]:
        print(f"{a:9s} {s['mean_truth']:8.2f} {s['mean_detected']:9.2f} {s['mae']:6.2f} {s['bias']:+6.2f} {s['corr']:6.2f}")
    print(f"saved {cap / 'evaluation.json'}")
    return 0


def _stats(t: np.ndarray, d: np.ndarray) -> dict:
    corr = float(np.corrcoef(t, d)[0, 1]) if len(t) > 1 and t.std() > 0 and d.std() > 0 else float("nan")
    return dict(mean_truth=float(t.mean()) if len(t) else 0.0, mean_detected=float(d.mean()) if len(d) else 0.0,
                mae=float(np.abs(t - d).mean()) if len(t) else 0.0,
                bias=float((d - t).mean()) if len(t) else 0.0, corr=corr,
                recall_proxy=float(d.sum() / t.sum()) if t.sum() > 0 else float("nan"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.detection", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--scenario", required=True)
    c.add_argument("--tls", default="n_0_0")
    c.add_argument("--begin", type=float, default=30600)
    c.add_argument("--frames", type=int, default=30)
    c.add_argument("--interval", type=float, default=5.0)
    c.add_argument("--view-width", type=float, default=160.0)
    c.add_argument("--out", default="sumo_scenarios/output/detection/poc")
    c.set_defaults(func=_capture)
    e = sub.add_parser("evaluate")
    e.add_argument("--capture", default="sumo_scenarios/output/detection/poc")
    e.add_argument("--weights", default=None, help="fine-tuned checkpoint; default = COCO yolov6n")
    e.add_argument("--conf", type=float, default=0.25)
    e.add_argument("--img-size", type=int, default=640)
    e.add_argument("--annotate", action="store_true")
    e.set_defaults(func=_evaluate)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
