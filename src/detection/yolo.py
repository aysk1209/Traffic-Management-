"""Thin wrapper around the vendored YOLOv6 (external/YOLOv6) for single-image inference.

Nothing in external/ is modified: the repo is put on ``sys.path`` and its own
``DetectBackend`` / ``letterbox`` / ``non_max_suppression`` are used. Two shims are
needed for a modern PyTorch:

* YOLOv6 checkpoints pickle the whole model object, which ``torch.load``'s default
  ``weights_only=True`` (PyTorch >= 2.6) refuses; the loader is wrapped to pass
  ``weights_only=False`` for the trusted upstream weights only.
* RepVGG blocks are switched to deploy mode after loading, as YOLOv6's own Inferer does.

Weights: ``external/weights/yolov6n.pt`` (COCO-pretrained nano model, ~10 MB, from the
YOLOv6 0.4.0 release). COCO class ids for vehicles: car=2, motorcycle=3, bus=5, truck=7.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.detection.count import Detection
from src.detection.roi import Box

ROOT = Path(__file__).resolve().parents[2]
YOLOV6_DIR = ROOT / "external" / "YOLOv6"
DEFAULT_WEIGHTS = ROOT / "external" / "weights" / "yolov6n.pt"

COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def yolov6_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return (YOLOV6_DIR / "yolov6" / "__init__.py").exists() and DEFAULT_WEIGHTS.exists()


@dataclass
class YoloV6Detector:
    weights: Path = DEFAULT_WEIGHTS
    img_size: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = "cpu"
    class_names: dict[int, str] | None = None   # default: COCO vehicle classes only

    def __post_init__(self) -> None:
        if str(YOLOV6_DIR) not in sys.path:
            sys.path.insert(0, str(YOLOV6_DIR))
        import torch
        from yolov6.layers.common import DetectBackend, RepVGGBlock

        original_load = torch.load

        def _load_full_pickle(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_load(*args, **kwargs)

        torch.load = _load_full_pickle
        try:
            self._backend = DetectBackend(str(self.weights), device=torch.device(self.device))
        finally:
            torch.load = original_load
        for layer in self._backend.model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, "recompute_scale_factor"):
                layer.recompute_scale_factor = None
        self._backend.model.eval()
        self.stride = int(self._backend.stride)
        self.names = self.class_names or dict(COCO_VEHICLES)
        self._torch = torch

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        """Run the model on one BGR image (H, W, 3 uint8); return detections in pixel coords."""
        from yolov6.data.data_augment import letterbox
        from yolov6.utils.nms import non_max_suppression

        torch = self._torch
        img = letterbox(image_bgr, self.img_size, stride=self.stride)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC BGR -> CHW RGB
        tensor = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self._backend(tensor)
            det = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                      classes=list(self.names), agnostic=False, max_det=1000)[0]
        out: list[Detection] = []
        if det is None or len(det) == 0:
            return out
        det[:, :4] = _rescale(tensor.shape[2:], det[:, :4], image_bgr.shape[:2])
        for x0, y0, x1, y1, conf, cls in det.cpu().numpy():
            out.append(Detection(Box(float(x0), float(y0), float(x1), float(y1)), float(conf),
                                 self.names.get(int(cls), str(int(cls)))))
        return out


def _rescale(from_shape, boxes, to_shape):
    """Undo the letterbox: map boxes from the padded model input back to the original image."""
    ratio = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
    pad_x = (from_shape[1] - to_shape[1] * ratio) / 2
    pad_y = (from_shape[0] - to_shape[0] * ratio) / 2
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= ratio
    boxes[:, 0].clamp_(0, to_shape[1])
    boxes[:, 1].clamp_(0, to_shape[0])
    boxes[:, 2].clamp_(0, to_shape[1])
    boxes[:, 3].clamp_(0, to_shape[0])
    return boxes
