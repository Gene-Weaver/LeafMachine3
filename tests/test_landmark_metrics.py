"""Unit tests for :mod:`leafmachine3.core.landmark_metrics`.

Covers the two things that matter most: (1) every metric degrades to ``None`` when the keypoints
it needs are missing/occluded (nothing raises, nothing is fabricated), and (2) the apex/base angle
classification matches the confirmed convention in ``modules/experiments/angle_checks.html``
(acute / obtuse away from the centroid, reflex toward it).
"""
from __future__ import annotations

import pytest

from leafmachine3.core.landmark_metrics import (
    _classify_angle,
    _interior_angle,
    compute_measurements,
)

# Image coords (y increases downward); centroid sits below the apex / above the base.
_CENTROID = (150.0, 160.0)


# -- angle classification (mirrors the angle_checks.html cases) ---------------------
def test_acute_apex_points_away_from_centroid():
    ang, typ = _classify_angle((150, 30), (130, 72), (170, 72), _CENTROID)
    assert typ == "acute" and ang < 90


def test_obtuse_apex_points_away_from_centroid():
    ang, typ = _classify_angle((150, 30), (86, 64), (214, 64), _CENTROID)
    assert typ == "obtuse" and 90 <= ang < 180


def test_reflex_apex_points_toward_centroid():
    # emarginate/notched apex: vertex pushed inward, both arms point toward the leaf body
    ang, typ = _classify_angle((150, 96), (118, 40), (182, 40), _CENTROID)
    assert typ == "reflex" and ang > 180


def test_reflex_base_cordate():
    # heart-shaped base: base_center pushed up into the blade, lobes hang below
    ang, typ = _classify_angle((150, 224), (118, 282), (182, 282), _CENTROID)
    assert typ == "reflex" and ang > 180


def test_reflex_is_360_minus_interior():
    center, left, right = (150, 96), (118, 40), (182, 40)
    interior = _interior_angle(center, left, right)
    ang, typ = _classify_angle(center, left, right, _CENTROID)
    assert typ == "reflex"
    assert ang == pytest.approx(360.0 - interior, abs=1e-6)


def test_degenerate_arm_returns_none():
    assert _interior_angle((10, 10), (10, 10), (20, 20)) is None


# -- full-leaf measurement + robustness --------------------------------------------
def _full_leaf() -> dict:
    pts: dict[str, tuple[float, float]] = {"lamina_tip": (150, 20), "lamina_base": (150, 300)}
    for i in range(15):
        pts[f"midvein_{i}"] = (150.0, 20 + (300 - 20) * i / 14.0)
    pts["width_left"] = (100, 160)
    pts["width_right"] = (210, 160)
    pts["apex_center"] = (150, 20)
    pts["apex_left"] = (130, 55)
    pts["apex_right"] = (170, 55)
    pts["base_center"] = (150, 300)
    pts["base_left"] = (120, 265)
    pts["base_right"] = (180, 265)
    for i in range(5):
        pts[f"petiole_{i}"] = (150.0, 305 + 20 * i / 4.0)
    pts["petiole_tip"] = (150, 330)
    return pts


def test_full_leaf_all_metrics_sane():
    m = compute_measurements(_full_leaf())
    assert m.n_present == 31
    assert m.lamina_extent == pytest.approx(280, abs=1)
    assert m.leaf_width == pytest.approx(110, abs=1)
    assert m.lamina_trace_length >= m.lamina_extent - 1e-6      # arc >= chord
    assert m.lamina_curvature == pytest.approx(1.0, abs=1e-3)   # straight midrib
    assert m.apex_angle_type in {"acute", "obtuse"}             # convex apex
    assert m.base_angle_type in {"acute", "obtuse"}
    # lamina trace = the 15 midvein pts (150,20)->(150,300); petiole trace = the 5 petiole pts
    # (150,305)->(150,325); anchors (tip/base/petiole_tip) are NOT summed into the traces.
    assert m.lamina_trace_length == pytest.approx(280, abs=1)
    assert m.petiole_trace_length == pytest.approx(20, abs=1)
    assert m.lamina_centroid is not None


def test_trace_length_sums_only_trace_points_not_anchors():
    # midvein/petiole points are close together; the tip/base/petiole_tip anchors are FAR away.
    # lamina_trace_length / petiole_trace_length must sum ONLY the trace points (anchors excluded),
    # while lamina_extent DOES use the far tip/base.
    pts = {
        "lamina_tip": (0, -1000),
        "midvein_0": (0, 0), "midvein_1": (0, 10), "midvein_2": (0, 20),
        "lamina_base": (0, 1000),
        "petiole_0": (0, 0), "petiole_1": (0, 5), "petiole_2": (0, 15),
        "petiole_tip": (0, 5000),
    }
    m = compute_measurements(pts)
    assert m.lamina_trace_length == pytest.approx(20)    # 10 + 10, anchors excluded
    assert m.petiole_trace_length == pytest.approx(15)   # 5 + 10, anchors excluded
    assert m.lamina_extent == pytest.approx(2000)        # extent uses the far tip/base


def test_curved_midrib_has_curvature_above_one():
    pts = {"lamina_tip": (100, 0), "lamina_base": (100, 100)}
    # midrib bows out to x=140 in the middle -> arc longer than the 100px chord
    for i in range(15):
        t = i / 14.0
        pts[f"midvein_{i}"] = (100 + 40 * (1 - abs(2 * t - 1)), 100 * t)
    m = compute_measurements(pts)
    assert m.lamina_extent == pytest.approx(100, abs=1)
    assert m.lamina_curvature > 1.05


def test_missing_points_degrade_to_none():
    m = compute_measurements({"width_left": (0, 0), "width_right": (10, 0)})
    assert m.leaf_width == pytest.approx(10)
    assert m.lamina_extent is None
    assert m.lamina_trace_length is None
    assert m.lamina_curvature is None
    assert m.apex_angle is None and m.apex_angle_type is None
    assert m.base_angle is None and m.base_angle_type is None
    assert m.petiole_trace_length is None


def test_empty_input_is_all_none():
    m = compute_measurements({})
    assert m.n_present == 0
    assert all(v is None for v in (
        m.lamina_trace_length, m.lamina_extent, m.leaf_width, m.apex_angle,
        m.base_angle, m.petiole_trace_length, m.lamina_curvature, m.lamina_centroid,
    ))


def test_angle_without_centroid_gives_interior_but_no_type():
    # apex triple present but no lamina points -> centroid unknown -> can't classify reflex
    m = compute_measurements({"apex_left": (130, 72), "apex_center": (150, 30), "apex_right": (170, 72)})
    assert m.apex_angle is not None
    assert m.apex_angle_type is None
