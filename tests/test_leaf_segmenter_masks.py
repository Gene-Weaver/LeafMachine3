"""Regression tests for YoloSegmenter raster mask -> Instance reduction.

Guards the fix for the ``masks.xy`` stringing artifact: masks are built from the raster
``masks.data`` with per-class reduction (Leaf/Petiole = largest component, Hole = keep
all). These exercise ``build_instances`` directly with synthetic masks -- no GPU, no
Ultralytics -- so the polygon logic is covered without the model.
"""
from __future__ import annotations

import cv2
import numpy as np

from leafmachine3.inference.leaf_segmenter import build_instances

NAMES = {0: "Leaf", 1: "Petiole", 2: "Hole"}


def _raster(poly: np.ndarray, hw) -> np.ndarray:
    m = np.zeros(hw, np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m


def test_leaf_uses_largest_component_and_never_strings() -> None:
    """A Leaf mask with a disconnected island keeps only the main blob, with no bridge."""
    h, w = 120, 120
    leaf = np.zeros((h, w), bool)
    leaf[20:60, 20:60] = True       # main blob, 40x40 = 1600 px
    leaf[95:110, 95:110] = True     # disconnected island, 15x15 = 225 px
    insts = build_instances(leaf[None].astype("float32"), [0], [0.9], NAMES, (h, w))

    assert len(insts) == 1 and insts[0].cls_name == "Leaf"
    m = _raster(insts[0].polygon, (h, w))
    assert m[20:60, 20:60].sum() > 0           # main blob kept
    assert m[95:110, 95:110].sum() == 0        # island dropped
    # If the polygon bridged to the island (the old masks.xy behaviour) the filled area
    # would span most of the frame; the largest-component polygon stays ~the main blob.
    assert m.sum() < 1.3 * 1600


def test_holes_keep_all_components_drop_speckle() -> None:
    """Every real hole survives; sub-min_area speckle is removed."""
    h, w = 120, 120
    holes = np.zeros((h, w), bool)
    holes[10:25, 10:25] = True      # 225 px
    holes[60:80, 60:80] = True      # 400 px
    holes[100, 100] = True          # 1 px speckle -> dropped
    insts = build_instances(holes[None].astype("float32"), [2], [0.7], NAMES, (h, w), min_area_px=64)

    assert len(insts) == 2
    assert {i.cls_name for i in insts} == {"Hole"}


def test_multiple_leaf_detections_reduced_to_single_largest() -> None:
    """Two disjoint Leaf detections collapse to the largest, carrying the max conf."""
    h, w = 120, 120
    big = np.zeros((h, w), bool)
    big[10:50, 10:50] = True        # 1600 px
    small = np.zeros((h, w), bool)
    small[90:100, 90:100] = True    # 100 px
    data = np.stack([big, small]).astype("float32")
    insts = build_instances(data, [0, 0], [0.8, 0.9], NAMES, (h, w))

    assert len(insts) == 1
    m = _raster(insts[0].polygon, (h, w))
    assert m[10:50, 10:50].sum() > 0 and m[90:100, 90:100].sum() == 0
    assert insts[0].conf == 0.9     # max conf across the class's detections


def test_instances_ordered_leaf_first() -> None:
    """Leaf is emitted before Petiole/Hole so the stage can attach parts to it."""
    h, w = 100, 100
    leaf = np.zeros((h, w), bool); leaf[10:40, 10:40] = True
    pet = np.zeros((h, w), bool); pet[10:40, 45:55] = True
    hole = np.zeros((h, w), bool); hole[15:25, 15:25] = True
    data = np.stack([pet, hole, leaf]).astype("float32")   # scrambled input order
    insts = build_instances(data, [1, 2, 0], [0.5, 0.6, 0.9], NAMES, (h, w))

    names = [i.cls_name for i in insts]
    assert names[0] == "Leaf"
    assert names.index("Petiole") < names.index("Hole")


def test_empty_result_returns_no_instances() -> None:
    insts = build_instances(np.zeros((0, 50, 50), "float32"), [], [], NAMES, (50, 50))
    assert insts == []
