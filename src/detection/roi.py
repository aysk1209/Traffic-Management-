"""Lane ROIs for the detection proof-of-concept, derived from the network geometry.

Nothing is hand-drawn: each approach's ROI is the axis-aligned box around its
incoming lanes (centreline shape widened by the lane width), clipped to the visible
view and converted to pixels with the view boundary reported by sumo-gui. This is
PROJECT_CONTEXT.md open decision 7 resolved as "computed from the config", so the
same code serves the single intersection and any junction of the grid.

Pure functions; the only inputs are geometry numbers, so this is unit-testable
without SUMO or torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

Point = tuple[float, float]


@dataclass(frozen=True)
class Box:
    """Axis-aligned box, any coordinate system. ``x0 <= x1``, ``y0 <= y1``."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def centre(self) -> Point:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def contains(self, p: Point) -> bool:
        return self.x0 <= p[0] <= self.x1 and self.y0 <= p[1] <= self.y1

    def clip(self, other: "Box") -> "Box | None":
        b = Box(max(self.x0, other.x0), max(self.y0, other.y0), min(self.x1, other.x1), min(self.y1, other.y1))
        return b if b.x1 > b.x0 and b.y1 > b.y0 else None

    def expand(self, dx: float, dy: float) -> "Box":
        return Box(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)


def bbox_of(points: Iterable[Point]) -> Box:
    xs, ys = zip(*points)
    return Box(min(xs), min(ys), max(xs), max(ys))


def lane_box(shape: Sequence[Point], width: float) -> Box:
    """Box around a straight (axis-aligned) lane centreline widened by its lane width."""
    b = bbox_of(shape)
    horizontal = b.width >= b.height
    return b.expand(0.0, width / 2) if horizontal else b.expand(width / 2, 0.0)


def approach_box(lane_shapes: Sequence[Sequence[Point]], lane_widths: Sequence[float]) -> Box:
    """Union box over an approach's lanes."""
    boxes = [lane_box(s, w) for s, w in zip(lane_shapes, lane_widths)]
    return Box(min(b.x0 for b in boxes), min(b.y0 for b in boxes),
               max(b.x1 for b in boxes), max(b.y1 for b in boxes))


@dataclass(frozen=True)
class ViewTransform:
    """Map SUMO world coordinates to pixels of a screenshot taken from a view whose
    visible boundary is ``world`` (xmin, ymin, xmax, ymax) rendered at ``width`` x ``height``.
    SUMO's y axis points up; image rows point down."""

    world: Box
    width: int
    height: int

    def to_pixel(self, p: Point) -> Point:
        sx = self.width / self.world.width
        sy = self.height / self.world.height
        return ((p[0] - self.world.x0) * sx, (self.world.y1 - p[1]) * sy)

    def to_world(self, px: Point) -> Point:
        sx = self.world.width / self.width
        sy = self.world.height / self.height
        return (self.world.x0 + px[0] * sx, self.world.y1 - px[1] * sy)

    def box_to_pixel(self, b: Box) -> Box:
        (x0, y1), (x1, y0) = self.to_pixel((b.x0, b.y0)), self.to_pixel((b.x1, b.y1))
        return Box(x0, y0, x1, y1)

    @property
    def metres_per_pixel(self) -> float:
        return self.world.width / self.width

    @classmethod
    def from_screenshot(cls, boundary: Box, width: int, height: int) -> "ViewTransform":
        """Transform for a sumo-gui screenshot of ``width`` x ``height`` pixels.

        sumo-gui renders screenshots isotropically at the on-screen *vertical* scale
        (``boundary.height / height``), centred on the view centre; the horizontal extent is
        therefore ``width * m_per_px``, not the on-screen boundary's width (the canvas has a
        different aspect ratio than the requested screenshot).
        """
        mpp = boundary.height / height
        cx, cy = boundary.centre
        half_w = width * mpp / 2
        return cls(Box(cx - half_w, boundary.y0, cx + half_w, boundary.y1), width, height)


def pixel_rois(approach_boxes: dict[str, Box], view: ViewTransform) -> dict[str, Box]:
    """Clip each approach's world box to the view and convert to pixel boxes; approaches
    entirely outside the view are dropped."""
    out: dict[str, Box] = {}
    for approach, b in approach_boxes.items():
        clipped = b.clip(view.world)
        if clipped is not None:
            out[approach] = view.box_to_pixel(clipped)
    return out
