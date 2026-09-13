"""Auto-labelling of captured frames from simulator state (pure geometry).

A SUMO vehicle is described by its front-bumper position, heading (degrees, 0 = north,
clockwise), length and width. :func:`vehicle_pixel_box` turns that into the axis-aligned
pixel box of the rendered sprite, which is exactly what a detector is trained to output.
Labels are written in YOLO format (``class cx cy w h``, normalised to image size).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.detection.roi import Box, ViewTransform, bbox_of


@dataclass(frozen=True)
class VehicleState:
    front: tuple[float, float]   # world coords of the front-bumper centre (traci getPosition)
    angle_deg: float             # traci getAngle: 0 = north, clockwise
    length_m: float
    width_m: float
    vclass: str = "vehicle"


def vehicle_corners(v: VehicleState) -> list[tuple[float, float]]:
    """Four world-coordinate corners of the vehicle footprint."""
    a = math.radians(v.angle_deg)
    dx, dy = math.sin(a), math.cos(a)          # unit heading vector (x east, y north)
    px, py = dy, -dx                            # unit vector to the right of travel
    fx, fy = v.front
    rx, ry = fx - dx * v.length_m, fy - dy * v.length_m
    hw = v.width_m / 2
    return [(fx + px * hw, fy + py * hw), (fx - px * hw, fy - py * hw),
            (rx - px * hw, ry - py * hw), (rx + px * hw, ry + py * hw)]


def vehicle_pixel_box(v: VehicleState, view: ViewTransform, min_visible: float = 0.3) -> Box | None:
    """Axis-aligned pixel box of the vehicle, clipped to the image; ``None`` if less than
    ``min_visible`` of its area is inside the frame."""
    world_box = bbox_of(vehicle_corners(v))
    full = view.box_to_pixel(world_box)
    image = Box(0, 0, view.width, view.height)
    clipped = full.clip(image)
    if clipped is None:
        return None
    if clipped.width * clipped.height < min_visible * full.width * full.height:
        return None
    return clipped


def yolo_label_lines(boxes: Iterable[tuple[int, Box]], width: int, height: int) -> list[str]:
    lines = []
    for cls, b in boxes:
        cx, cy = b.centre
        lines.append(f"{cls} {cx / width:.6f} {cy / height:.6f} {b.width / width:.6f} {b.height / height:.6f}")
    return lines


def write_yolo_label(path: Path, boxes: Iterable[tuple[int, Box]], width: int, height: int) -> None:
    path.write_text("\n".join(yolo_label_lines(boxes, width, height)) + ("\n" if boxes else ""), encoding="utf-8")
