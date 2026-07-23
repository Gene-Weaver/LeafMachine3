"""Unit tests for the ported LeafMachine2 morphometrics (core.morphometrics)."""
from __future__ import annotations

import math

import numpy as np

from leafmachine3.core.morphometrics import polygon_morphology, rotate_polygon_by_angle


def _rectangle(w: float, h: float, cx: float = 500.0, cy: float = 500.0, angle: float = 0.0) -> np.ndarray:
    """An axis-aligned w x h rectangle centered at (cx, cy), optionally rotated by `angle` deg."""
    corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
    rot = rotate_polygon_by_angle([(x + cx, y + cy) for x, y in corners], angle, cx, cy)
    return np.asarray(rot, dtype=float)


def test_area_perimeter_of_axis_aligned_rectangle():
    m = polygon_morphology(_rectangle(300, 100), find_min_bbox=True)
    assert m is not None
    assert m.area_px == 300 * 100                       # exact for an integer rectangle
    assert abs(m.perimeter_px - 2 * (300 + 100)) < 2.0
    assert m.n_vertices >= 4


def _ellipse(a: float, b: float, cx: float = 500.0, cy: float = 500.0, angle: float = 0.0, n: int = 240):
    """A leaf-like ellipse with semi-axes (a, b), optionally rotated by `angle` deg."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = [(a * math.cos(u) + cx, b * math.sin(u) + cy) for u in t]
    return np.asarray(rotate_polygon_by_angle(pts, angle, cx, cy), dtype=float)


def test_lm2_rotated_bbox_dims_are_length_and_width():
    # LM2's procedure ties dim_max to the min-enclosing-circle diameter; for a leaf-like
    # ellipse (major 400, minor 120) that recovers the length (400) and width (120).
    m = polygon_morphology(_ellipse(200, 60, angle=30), method="lm2", find_min_bbox=True)
    assert m is not None
    assert m.dim_max >= m.dim_min > 0
    assert abs(m.dim_max - 400) < 50, m.dim_max         # major axis -> leaf length
    assert abs(m.dim_min - 120) < 70, m.dim_min         # minor axis -> leaf width
    assert m.aspect_ratio > 2.0                          # clearly elongated (length/width)


def test_feret_method_is_tight():
    m = polygon_morphology(_ellipse(200, 60, angle=25), method="feret")
    assert m is not None
    assert m.dim_max >= m.dim_min > 0
    assert abs(m.dim_max - 400) < 25, m.dim_max
    assert abs(m.dim_min - 120) < 25, m.dim_min
    assert m.dim_max * m.dim_min >= m.area_px - 1        # a bounding box must cover the mask
    assert len(m.rotated_bbox) == 4


def test_pca_is_the_default():
    # default method is now "pca": robust axis; the default call must equal an explicit pca call.
    default = polygon_morphology(_ellipse(200, 60, angle=25))
    explicit = polygon_morphology(_ellipse(200, 60, angle=25), method="pca")
    assert default is not None and explicit is not None
    assert abs(default.dim_max - 400) < 40 and abs(default.dim_min - 120) < 40
    assert default.dim_max * default.dim_min >= default.area_px - 1        # box covers the mask
    assert (round(default.dim_max), round(default.dim_min)) == (round(explicit.dim_max), round(explicit.dim_min))


def test_circularity_of_circle_near_one():
    t = np.linspace(0, 2 * math.pi, 200, endpoint=False)
    circle = np.column_stack([500 + 150 * np.cos(t), 500 + 150 * np.sin(t)])
    m = polygon_morphology(circle, find_min_bbox=True)
    assert m is not None
    assert 0.9 < m.circularity <= 1.05                   # a circle -> circularity ~ 1
    assert m.convexity > 0.95                            # convex shape


def test_minarearect_recovers_tight_dims():
    # cv2.minAreaRect is exact: a 400x120 ellipse -> dim_max ~ 400, dim_min ~ 120, any rotation
    m = polygon_morphology(_ellipse(200, 60, angle=37), method="minarearect")
    assert m is not None
    assert m.dim_max >= m.dim_min > 0
    assert abs(m.dim_max - 400) < 25, m.dim_max
    assert abs(m.dim_min - 120) < 25, m.dim_min
    assert len(m.rotated_bbox) == 4


def test_degenerate_polygon_returns_none():
    assert polygon_morphology(np.array([[0, 0], [1, 1]]), find_min_bbox=True) is None
    assert polygon_morphology(np.array([[0, 0], [1, 1]]), method="minarearect") is None
