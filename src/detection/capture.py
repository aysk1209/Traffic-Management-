"""Capture rendered frames from sumo-gui together with ground-truth counts per approach.

Runs a scenario in sumo-gui via traci, centres the view on one junction, and every
``interval_s`` schedules a screenshot; after the following step it records, for each
approach, the number of vehicles whose position lies inside that approach's ROI (the
same "is the centre inside the box" rule the detector counts are scored with). Output:

    <out>/frames/frame_<k>.png
    <out>/frames.csv        frame, time_s, <approach> ... ground-truth counts
    <out>/rois.json         pixel ROIs + view transform, for the detector stage

The captured view uses the "real world" colour scheme (grass, asphalt, vehicle
sprites) which is the closest sumo-gui gets to camera footage.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.detection.labels import VehicleState, vehicle_pixel_box, write_yolo_label
from src.detection.roi import Box, ViewTransform, approach_box, pixel_rois
from src.sim.bridge import PROGRAM_ID, discover_approach_lanes
from src.sumo_env import ensure_sumo_tools, sumo_binary


@dataclass(frozen=True)
class CaptureConfig:
    sumocfg: Path
    tls_id: str
    out_dir: Path
    begin_s: float
    n_frames: int
    interval_s: float = 5.0
    view_width_m: float = 160.0     # metres spanned horizontally by the frame
    width_px: int = 1280
    height_px: int = 960
    schema: str = "real world"
    write_labels: bool = False      # also write YOLO-format boxes for every visible vehicle
    settle_s: float = 2.0           # let the GUI window appear and paint before the first request
    pre_shot_s: float = 0.3         # let the GUI catch up before each screenshot request


def wait_for_png(path: Path, timeout_s: float = 10.0) -> bool:
    """sumo-gui writes the screenshot asynchronously; block until it decodes cleanly."""
    from PIL import Image

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with Image.open(path) as im:
                im.verify()
            return True
        except Exception:
            time.sleep(0.05)
    return False


def _fit_zoom(view: str, target_height_m: float) -> None:
    """Adjust the view zoom until the visible boundary spans ~target_height_m vertically.

    Screenshots are rendered at the on-screen vertical scale (see
    :meth:`ViewTransform.from_screenshot`), so the vertical extent is what fixes the
    metres-per-pixel of the captured frame.
    """
    import traci

    for _ in range(6):
        (_xmin, ymin), (_xmax, ymax) = traci.gui.getBoundary(view)  # ((xmin, ymin), (xmax, ymax))
        height = ymax - ymin
        if abs(height - target_height_m) / target_height_m < 0.01:
            break
        traci.gui.setZoom(view, traci.gui.getZoom(view) * height / target_height_m)
        traci.simulationStep()


def capture(cfg: CaptureConfig) -> Path:
    ensure_sumo_tools()
    import traci

    frames_dir = cfg.out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = cfg.out_dir / "labels"
    if cfg.write_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sumo_binary("sumo-gui"), "-c", str(cfg.sumocfg), "--begin", str(cfg.begin_s),
           "--start", "true", "--quit-on-end", "true", "--no-warnings", "true",
           "--window-size", f"{cfg.width_px},{cfg.height_px}", "--no-step-log", "true"]
    traci.start(cmd)
    try:
        view = traci.gui.getIDList()[0]
        time.sleep(cfg.settle_s)  # a screenshot requested before the first paint blocks forever
        traci.gui.setSchema(view, cfg.schema)
        jx, jy = traci.junction.getPosition(cfg.tls_id)
        traci.gui.setOffset(view, jx, jy)
        traci.simulationStep()
        _fit_zoom(view, cfg.view_width_m * cfg.height_px / cfg.width_px)
        (xmin, ymin), (xmax, ymax) = traci.gui.getBoundary(view)
        transform = ViewTransform.from_screenshot(Box(xmin, ymin, xmax, ymax), cfg.width_px, cfg.height_px)

        if traci.trafficlight.getProgram(cfg.tls_id) != PROGRAM_ID:
            raise RuntimeError(f"{cfg.tls_id}: program {PROGRAM_ID!r} not active")
        logic = next(l for l in traci.trafficlight.getAllProgramLogics(cfg.tls_id) if l.programID == PROGRAM_ID)
        approach_lanes = discover_approach_lanes(cfg.tls_id, logic.phases)
        world_boxes = {
            a: approach_box([traci.lane.getShape(l) for l in lanes], [traci.lane.getWidth(l) for l in lanes])
            for a, lanes in approach_lanes.items()
        }
        world_boxes = {a: b.clip(transform.world) for a, b in world_boxes.items()}
        world_boxes = {a: b for a, b in world_boxes.items() if b is not None}
        rois = pixel_rois(world_boxes, transform)

        (cfg.out_dir / "rois.json").write_text(json.dumps({
            "tls_id": cfg.tls_id,
            "view": {"world": [xmin, ymin, xmax, ymax], "width_px": cfg.width_px, "height_px": cfg.height_px,
                     "metres_per_pixel": transform.metres_per_pixel},
            "approach_lanes": approach_lanes,
            "rois_px": {a: [b.x0, b.y0, b.x1, b.y1] for a, b in rois.items()},
            "rois_world": {a: [b.x0, b.y0, b.x1, b.y1] for a, b in world_boxes.items()},
        }, indent=2), encoding="utf-8")

        approaches = list(rois)
        with open(cfg.out_dir / "frames.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "time_s", *approaches])
            steps_between = max(1, int(round(cfg.interval_s)))
            for k in range(cfg.n_frames):
                for _ in range(steps_between - 1):
                    traci.simulationStep()
                path = frames_dir / f"frame_{k:04d}.png"
                time.sleep(cfg.pre_shot_s)
                traci.gui.screenshot(view, str(path), cfg.width_px, cfg.height_px)
                traci.simulationStep()  # rendered during this step, showing the post-step state
                truth = _ground_truth(approach_lanes, world_boxes)
                w.writerow([path.name, f"{traci.simulation.getTime():.0f}", *[truth[a] for a in approaches]])
                if cfg.write_labels:
                    boxes = [(0, b) for b in _visible_vehicle_boxes(transform)]
                    write_yolo_label(labels_dir / f"{path.stem}.txt", boxes, cfg.width_px, cfg.height_px)
                if not wait_for_png(path):
                    raise RuntimeError(f"screenshot {path} was not written")
    finally:
        try:
            traci.close(wait=False)
        except Exception:
            pass
    return cfg.out_dir


def _visible_vehicle_boxes(view: ViewTransform) -> list[Box]:
    """Pixel boxes of every vehicle at least partly inside the frame (all classes -> one label)."""
    import traci

    boxes: list[Box] = []
    for vid in traci.vehicle.getIDList():
        state = VehicleState(front=traci.vehicle.getPosition(vid), angle_deg=traci.vehicle.getAngle(vid),
                             length_m=traci.vehicle.getLength(vid), width_m=traci.vehicle.getWidth(vid))
        b = vehicle_pixel_box(state, view)
        if b is not None:
            boxes.append(b)
    return boxes


def _ground_truth(approach_lanes: dict[str, list[str]], world_boxes: dict[str, Box]) -> dict[str, int]:
    """Vehicles on each approach's lanes whose position lies inside the approach ROI."""
    import traci

    counts: dict[str, int] = {}
    for a, box in world_boxes.items():
        n = 0
        for lane in approach_lanes[a]:
            for vid in traci.lane.getLastStepVehicleIDs(lane):
                if box.contains(traci.vehicle.getPosition(vid)):
                    n += 1
        counts[a] = n
    return counts
