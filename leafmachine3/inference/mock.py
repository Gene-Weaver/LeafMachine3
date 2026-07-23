"""Deterministic synthetic backends for ``compute.mock: true`` runs.

These mirror the real backends' return shapes exactly but need no weights, GPU, or heavy
dependency — only ``numpy`` (and ``cv2`` solely to read image dimensions from a path).
Outputs are fully deterministic functions of the input image size and configured classes,
so tests and dry-runs are reproducible.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from leafmachine3.core.records import Detection
from leafmachine3.inference.leaf_segmenter import Instance

log = logging.getLogger("leafmachine3.inference.mock")

# Fractional (x1, y1, x2, y2) box footprints used by the mock detector, in priority order.
_MOCK_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (0.10, 0.10, 0.42, 0.44),
    (0.55, 0.18, 0.90, 0.62),
    (0.28, 0.60, 0.72, 0.94),
)

# Segmentation class ids used by the mock segmenter (mirrors the seg palette order).
_SEG_CLASS_IDS = {"Leaf": 0, "Petiole": 1, "Hole": 2}


def _image_hw(image) -> tuple[int, int]:
    """Return (height, width) for a path or an ``np.ndarray`` (BGR/gray)."""
    if isinstance(image, np.ndarray):
        return int(image.shape[0]), int(image.shape[1])
    if isinstance(image, str):
        import cv2

        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"mock backend cannot read image: {image}")
        return int(img.shape[0]), int(img.shape[1])
    # PIL image
    w, h = image.size
    return int(h), int(w)


class MockDetector:
    """Emit 1–3 deterministic boxes drawn from the configured class list."""

    def __init__(self, class_names: Optional[Sequence[str]] = None) -> None:
        self.class_names = [str(c) for c in (class_names or ["Object"])]

    def predict(self, image, **_kwargs) -> list[Detection]:
        h, w = _image_hw(image)
        n = min(len(self.class_names), len(_MOCK_BOXES))
        out: list[Detection] = []
        for i in range(n):
            fx1, fy1, fx2, fy2 = _MOCK_BOXES[i]
            out.append(
                Detection(
                    cls_id=i,
                    cls_name=self.class_names[i],
                    conf=round(0.90 - 0.10 * i, 4),
                    xyxy=(fx1 * w, fy1 * h, fx2 * w, fy2 * h),
                )
            )
        return out


class MockSegmenter:
    """Emit one ~60%-of-crop 'Leaf' polygon plus a small 'Hole' inside it."""

    def __init__(self, leaf_class: str = "Leaf", hole_class: str = "Hole") -> None:
        self.leaf_class = leaf_class
        self.hole_class = hole_class

    @staticmethod
    def _octagon(cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
        """A deterministic 8-gon centred at (cx, cy) with radii (rx, ry)."""
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False) + np.pi / 8.0
        xs = cx + rx * np.cos(angles)
        ys = cy + ry * np.sin(angles)
        return np.stack([xs, ys], axis=1).astype(float)

    def predict(self, image, **_kwargs) -> list[Instance]:
        h, w = _image_hw(image)
        cx, cy = w / 2.0, h / 2.0
        leaf = self._octagon(cx, cy, 0.30 * w, 0.30 * h)  # ~60% of each dimension
        out = [
            Instance(
                cls_id=_SEG_CLASS_IDS.get(self.leaf_class, 0),
                cls_name=self.leaf_class,
                conf=0.95,
                polygon=leaf,
            )
        ]
        if min(h, w) >= 40:  # only carve a hole when the crop is large enough
            hole = self._octagon(cx, cy, 0.06 * w, 0.06 * h)
            out.append(
                Instance(
                    cls_id=_SEG_CLASS_IDS.get(self.hole_class, 2),
                    cls_name=self.hole_class,
                    conf=0.80,
                    polygon=hole,
                )
            )
        return out


class MockEnsemble:
    """Return a fixed ruler unit-type verdict for every crop."""

    def __init__(self, unit_type: str = "METRIC_MM") -> None:
        self.unit_type = unit_type

    def predict(self, _image) -> dict:
        return {
            "ensemble": self.unit_type,
            "x224": self.unit_type,
            "n224": self.unit_type,
            "dinov2mlp": self.unit_type,
        }
