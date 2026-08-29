"""Leaf orientation from keypoints -- the rotation that stands a leaf tip-up / base-down.

Primary method: the ``lamina_tip`` -> ``lamina_base`` axis. When either endpoint is missing but at
least ``min_midvein`` midvein (lamina-trace) points are present, fall back to a PCA principal axis of
the midvein and pick which end is the tip using (in priority order) the petiole trace, the base
points, then the apex points. If none of that is possible the leaf is left unoriented.

``angle_cw`` is the **clockwise** rotation in degrees that brings the base->tip direction to point
straight up (12 o'clock). Distances/angles are translation-invariant, so working- or crop-frame
coordinates give the same angle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from leafmachine3.core.landmarks import MIDVEIN_N, PETIOLE_N

Point = tuple[float, float]

_EPS = 1e-6


@dataclass
class Orientation:
    success: bool
    angle_cw: Optional[float] = None      # clockwise degrees to rotate the leaf upright
    method: str = "none"                  # "tip_base" | "pca" | "none"


def _angle_cw_to_up(dx: float, dy: float) -> float:
    """Clockwise degrees to rotate the image so vector ``(dx, dy)`` (image coords, y-down) points up.

    The vector currently sits ``atan2(dx, -dy)`` clockwise from straight up; rotating the image by
    the negative of that (mod 360) carries it to vertical.
    """
    return (-math.degrees(math.atan2(dx, -dy))) % 360.0


def _centroid(points: dict[str, Point], names) -> Optional[np.ndarray]:
    pts = [points[n] for n in names if n in points]
    return np.asarray(pts, dtype=float).mean(axis=0) if pts else None


def _pca_tip_base(points: dict[str, Point], mv: list[Point]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(tip_end, base_end)`` extreme midvein points along the PCA axis, or ``None`` if the
    tip/base ends cannot be disambiguated (no petiole/base/apex reference present)."""
    a = np.asarray(mv, dtype=float)
    centered = a - a.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]                                        # principal (long) axis, unit vector
    proj = centered @ axis
    end_lo, end_hi = a[int(np.argmin(proj))], a[int(np.argmax(proj))]

    petiole = _centroid(points, [f"petiole_{i}" for i in range(PETIOLE_N)] + ["petiole_tip"])
    base_ref = _centroid(points, ["base_left", "base_center", "base_right"])
    apex_ref = _centroid(points, ["apex_left", "apex_center", "apex_right"])

    def _split(ref: np.ndarray, ref_is_base: bool) -> tuple[np.ndarray, np.ndarray]:
        near_lo = np.linalg.norm(end_lo - ref) < np.linalg.norm(end_hi - ref)
        near, far = (end_lo, end_hi) if near_lo else (end_hi, end_lo)
        return (far, near) if ref_is_base else (near, far)   # (tip, base)

    if petiole is not None:                             # petiole marks the BASE end
        return _split(petiole, ref_is_base=True)
    if base_ref is not None:
        return _split(base_ref, ref_is_base=True)
    if apex_ref is not None:                            # apex marks the TIP end
        return _split(apex_ref, ref_is_base=False)
    return None


def compute_orientation(points: dict[str, Point], *, min_midvein: int = 5) -> Orientation:
    """Determine the upright rotation for a leaf from its present keypoints (name -> (x, y))."""
    tip, base = points.get("lamina_tip"), points.get("lamina_base")
    if tip is not None and base is not None:
        dx, dy = tip[0] - base[0], tip[1] - base[1]
        if math.hypot(dx, dy) > _EPS:
            return Orientation(True, _angle_cw_to_up(dx, dy), "tip_base")

    mv = [points[f"midvein_{i}"] for i in range(MIDVEIN_N) if f"midvein_{i}" in points]
    if len(mv) >= max(2, int(min_midvein)):
        ends = _pca_tip_base(points, mv)
        if ends is not None:
            tip_end, base_end = ends
            dx, dy = float(tip_end[0] - base_end[0]), float(tip_end[1] - base_end[1])
            if math.hypot(dx, dy) > _EPS:
                return Orientation(True, _angle_cw_to_up(dx, dy), "pca")

    return Orientation(False, None, "none")


def length_width_from_box(corners, angle_cw: Optional[float]) -> tuple[Optional[float], Optional[float]]:
    """Split a rotated bbox's two side lengths into (length along tip->base, perpendicular width).

    ``rotated_bbox_dim_max`` / ``dim_min`` are the GEOMETRIC long/short sides -- for a leaf that is
    wider than it is long, ``dim_max`` is the width. Length vs width is an ORIENTATION call, so this
    rotates both side vectors by the leaf's upright ``angle_cw`` and keeps the one that ends up more
    vertical as the length: after that rotation the tip->base axis points straight up by definition.

    Only |y| magnitudes are compared, so the rotation's sign convention cannot flip the result. The
    two returned values are the box's own side lengths (no new geometry, nothing re-measured).
    ``corners`` is the 4-corner ``[[x, y], ...]`` polygon; returns ``(None, None)`` if unusable.
    """
    if angle_cw is None or corners is None:
        return None, None
    pts = np.asarray(corners, dtype=float)
    if pts.shape != (4, 2):
        return None, None
    v1, v2 = pts[1] - pts[0], pts[3] - pts[0]        # two adjacent sides of the rectangle
    n1, n2 = float(np.hypot(*v1)), float(np.hypot(*v2))
    if n1 <= _EPS or n2 <= _EPS:
        return None, None
    theta = math.radians(float(angle_cw))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    def rotated_dy(v):                                # y-component after applying the upright rotation
        return abs(-v[0] * sin_t + v[1] * cos_t)

    # the side whose direction survives as the more vertical one runs tip->base
    if rotated_dy(v1) / n1 >= rotated_dy(v2) / n2:
        return n1, n2
    return n2, n1
