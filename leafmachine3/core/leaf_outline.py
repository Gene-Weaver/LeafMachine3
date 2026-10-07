"""LeafMachine2-style leaf-outline extraction + Douglas-Peucker simplification (LM3 port).

Ports LM2's outline conventions (leafmachine2/segmentation/detectron2/segment_leaves.py and
leafmachine2/ect_methods/leaf_ect.py): the ordered boundary contour of a binary leaf mask (largest
external contour, every boundary pixel), and the Shapely Douglas-Peucker simplification LM2 uses for
its ``Simple_Labels_Douglas_Peucker`` outlines.

The unit-circle NORMALIZATION is deliberately NOT done here -- it is performed natively by the
modern ``ect`` package (EmbeddedGraph.center_coordinates + scale_coordinates) in
:mod:`leafmachine3.core.ect_compute`, which is the same package LM2's own ECT path delegated to.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def extract_contour(mask) -> Optional[np.ndarray]:
    """Largest external boundary of a binary mask as ordered ``(N, 2)`` float px coords ``(x, y)``.

    ``cv2.RETR_EXTERNAL`` + ``cv2.CHAIN_APPROX_NONE`` (keeps every boundary pixel, like LM2) and the
    largest contour by area. Returns ``None`` for an empty / degenerate mask (< 3 points).
    """
    import cv2

    m = (np.asarray(mask) > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = np.asarray(max(cnts, key=cv2.contourArea)).squeeze()
    c = c.astype(float).reshape(-1, 2)
    return c if len(c) >= 3 else None


def douglas_peucker(coords, tolerance: float, cutoff: int = 0) -> np.ndarray:
    """Shapely Douglas-Peucker simplification (LM2's ``LineString.simplify``, preserve_topology).

    Outlines with ``<= cutoff`` points are returned unchanged (LM2's cutoff gate). ``tolerance`` is
    in the coordinate units passed in (LM2 applies it to the normalized ~unit-scale outline).
    """
    coords = np.asarray(coords, dtype=float).reshape(-1, 2)
    if cutoff and len(coords) <= int(cutoff):
        return coords
    from shapely.geometry import LineString

    simp = LineString(coords).simplify(float(tolerance), preserve_topology=True)
    out = np.asarray(simp.coords, dtype=float).reshape(-1, 2)
    return out if len(out) >= 3 else coords
