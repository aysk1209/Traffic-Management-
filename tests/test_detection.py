"""Tests for src/detection: ROI geometry, view transform, auto-labelling and ROI counting.
The YOLOv6 wrapper itself is exercised only when the vendored repo + weights are present."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.detection.count import Detection, count_per_roi
from src.detection.labels import VehicleState, vehicle_corners, vehicle_pixel_box, yolo_label_lines
from src.detection.roi import Box, ViewTransform, approach_box, bbox_of, lane_box, pixel_rois


# ----------------------------------------------------------------------------- Box / ROI

def test_box_basics_clip_and_contains() -> None:
    b = Box(0, 0, 10, 5)
    assert b.width == 10 and b.height == 5 and b.centre == (5, 2.5)
    assert b.contains((5, 2.5)) and not b.contains((11, 1))
    assert b.clip(Box(8, 3, 20, 20)) == Box(8, 3, 10, 5)
    assert b.clip(Box(20, 20, 30, 30)) is None
    assert bbox_of([(1, 2), (3, -1)]) == Box(1, -1, 3, 2)


def test_lane_box_widens_perpendicular_to_travel() -> None:
    horizontal = lane_box([(0.0, 10.0), (100.0, 10.0)], 3.2)
    assert horizontal == Box(0.0, 8.4, 100.0, 11.6)
    vertical = lane_box([(5.0, 0.0), (5.0, 50.0)], 3.2)
    assert vertical == Box(3.4, 0.0, 6.6, 50.0)


def test_approach_box_is_union_of_lanes() -> None:
    b = approach_box([[(0.0, 10.0), (100.0, 10.0)], [(0.0, 13.2), (100.0, 13.2)]], [3.2, 3.2])
    assert (b.x0, b.y0, b.x1, b.y1) == pytest.approx((0.0, 8.4, 100.0, 14.8))


# ----------------------------------------------------------------------------- transform

def test_view_transform_round_trip_and_y_flip() -> None:
    vt = ViewTransform(Box(100, 200, 260, 320), 1280, 960)   # 0.125 m/px both axes
    assert vt.to_pixel((100, 320)) == (0.0, 0.0)              # top-left of the world box
    assert vt.to_pixel((260, 200)) == (1280.0, 960.0)         # bottom-right
    p = (137.5, 250.0)
    assert vt.to_world(vt.to_pixel(p)) == pytest.approx(p)
    pb = vt.box_to_pixel(Box(100, 300, 120, 320))
    assert pb == Box(0.0, 0.0, 160.0, 160.0)


def test_from_screenshot_uses_vertical_scale_and_centre() -> None:
    # on-screen boundary 198 x 120 m (canvas aspect), screenshot 1280 x 960 -> 0.125 m/px, 160 m wide
    vt = ViewTransform.from_screenshot(Box(151, 190, 349, 310), 1280, 960)
    assert vt.metres_per_pixel == pytest.approx(0.125)
    assert vt.world == Box(pytest.approx(170), 190, pytest.approx(330), 310)


def test_pixel_rois_clip_and_drop_outside() -> None:
    vt = ViewTransform(Box(0, 0, 100, 100), 100, 100)
    rois = pixel_rois({"W": Box(-50, 40, 30, 50), "far": Box(500, 500, 600, 600)}, vt)
    assert set(rois) == {"W"}
    assert rois["W"] == Box(0, 50, 30, 60)


# ----------------------------------------------------------------------------- labels

@pytest.mark.parametrize("angle,expected_rear", [(0, (0, -5)), (90, (-5, 0)), (180, (0, 5)), (270, (5, 0))])
def test_vehicle_corners_rear_is_behind_front(angle: float, expected_rear: tuple) -> None:
    v = VehicleState(front=(0.0, 0.0), angle_deg=angle, length_m=5.0, width_m=2.0)
    c = vehicle_corners(v)
    rear_mid = ((c[2][0] + c[3][0]) / 2, (c[2][1] + c[3][1]) / 2)
    assert rear_mid == pytest.approx(expected_rear, abs=1e-9)
    b = bbox_of(c)
    if angle in (0, 180):
        assert (b.width, b.height) == pytest.approx((2.0, 5.0))
    else:
        assert (b.width, b.height) == pytest.approx((5.0, 2.0))


def test_vehicle_pixel_box_and_visibility_threshold() -> None:
    vt = ViewTransform(Box(0, 0, 100, 100), 1000, 1000)     # 0.1 m/px
    inside = VehicleState((50.0, 50.0), 0.0, 4.0, 2.0)
    b = vehicle_pixel_box(inside, vt)
    assert b == Box(pytest.approx(490), pytest.approx(500), pytest.approx(510), pytest.approx(540))
    mostly_out = VehicleState((0.5, 50.0), 90.0, 4.0, 2.0)    # heading east, front just inside
    assert vehicle_pixel_box(mostly_out, vt) is None
    partly = VehicleState((2.5, 50.0), 90.0, 4.0, 2.0)         # 2.5 of 4 m inside
    assert vehicle_pixel_box(partly, vt) is not None


def test_yolo_label_lines_normalised() -> None:
    lines = yolo_label_lines([(0, Box(100, 200, 140, 260))], 1000, 1000)
    assert lines == ["0 0.120000 0.230000 0.040000 0.060000"]


# ----------------------------------------------------------------------------- counting

def test_count_per_roi_by_centre_score_and_label() -> None:
    rois = {"N": Box(0, 0, 100, 100), "S": Box(0, 200, 100, 300)}
    dets = [
        Detection(Box(10, 10, 30, 30), 0.9, "car"),
        Detection(Box(40, 40, 60, 60), 0.1, "car"),        # below threshold
        Detection(Box(10, 210, 30, 230), 0.8, "truck"),
        Detection(Box(10, 150, 30, 170), 0.8, "car"),      # between ROIs
        Detection(Box(50, 250, 70, 270), 0.8, "person"),   # not a vehicle
    ]
    assert count_per_roi(dets, rois, min_score=0.25) == {"N": 1, "S": 1}


# ----------------------------------------------------------------------------- YOLOv6

def test_yolov6_wrapper_runs_if_available() -> None:
    from src.detection.yolo import yolov6_available
    if not yolov6_available():
        pytest.skip("YOLOv6 / torch / weights not available")
    from src.detection.yolo import YoloV6Detector
    det = YoloV6Detector()
    img = np.full((480, 640, 3), 60, dtype=np.uint8)
    assert det.detect(img) == []   # blank image: no vehicles, no crash
