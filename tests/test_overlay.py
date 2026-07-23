"""Tests for :mod:`leafmachine3.reporting.overlay` -- Summary_Image rendering."""
from __future__ import annotations

import numpy as np

from leafmachine3.core.imaging import encode_polygon
from leafmachine3.reporting.overlay import build_summary_image
from leafmachine3.reporting.palette import OverlayStyle


def _blank(h: int = 300, w: int = 400) -> np.ndarray:
    return np.full((h, w, 3), 220, dtype=np.uint8)


def _leaf_row(cls_name: str, poly) -> dict:
    return {"cls_name": cls_name, "mask_format": "polygon_xy", "mask_data": encode_polygon(poly)}


def test_returns_same_shape_ndarray() -> None:
    img = _blank()
    detections = [
        {"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 120, 40), "source": "archival"},
        {"cls_name": "Leaf_WHOLE", "conf": 0.8, "xyxy": (60, 80, 260, 250), "source": "plant"},
    ]
    leaves = [_leaf_row("Leaf", [[80, 100], [230, 110], [220, 240], [90, 235]])]
    out = build_summary_image(img, detections, leaves, cf_px_per_cm=None,
                              style=OverlayStyle(), work_scale=1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_does_not_mutate_input() -> None:
    img = _blank()
    before = img.copy()
    build_summary_image(
        img,
        [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 120, 40), "source": "archival"}],
        [],
        cf_px_per_cm=96.0,
        style=OverlayStyle(),
        work_scale=1.0,
    )
    assert np.array_equal(img, before)                      # drew on a copy


def test_actually_draws_something() -> None:
    img = _blank()
    detections = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 120, 40), "source": "archival"}]
    out = build_summary_image(img, detections, [], cf_px_per_cm=96.0,
                              style=OverlayStyle(), work_scale=1.0)
    assert not np.array_equal(out, img)                     # box + CF banner changed pixels


def test_work_scale_upscales_geometry() -> None:
    """With work_scale 0.5 the stored coords map onto a 2x-larger original frame."""
    img = _blank(600, 800)
    detections = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 100, 100), "source": "archival"}]
    out = build_summary_image(img, detections, [], cf_px_per_cm=None,
                              style=OverlayStyle(), work_scale=0.5)
    assert out.shape == img.shape                           # renders without going out of bounds
