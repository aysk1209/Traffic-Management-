"""Turn detector boxes into a per-approach vehicle count (pure, testable)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from src.detection.roi import Box


@dataclass(frozen=True)
class Detection:
    box: Box            # pixel coordinates
    score: float
    label: str          # detector class name, e.g. "car", "truck"


VEHICLE_LABELS = frozenset({"car", "truck", "bus", "motorcycle", "van"})


def count_per_roi(detections: Iterable[Detection], rois: Mapping[str, Box],
                  min_score: float = 0.25, labels: frozenset[str] = VEHICLE_LABELS) -> dict[str, int]:
    """Count detections whose box centre lies inside each ROI. A detection is assigned to
    at most one ROI (the first match in ``rois`` order); ROIs from the grid never overlap."""
    counts = {a: 0 for a in rois}
    for d in detections:
        if d.score < min_score or d.label not in labels:
            continue
        c = d.box.centre
        for a, roi in rois.items():
            if roi.contains(c):
                counts[a] += 1
                break
    return counts


def agreement(detected: Mapping[str, int], truth: Mapping[str, int]) -> dict[str, float]:
    """Per-approach absolute error plus totals, for one frame."""
    out: dict[str, float] = {}
    for a in truth:
        out[a] = abs(detected.get(a, 0) - truth[a])
    return out
