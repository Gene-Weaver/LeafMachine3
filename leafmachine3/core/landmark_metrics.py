"""Derived leaf measurements from predicted keypoints (the ``landmark_measurements`` stage).

Turns the raw 31-keypoint pose (``core.landmarks`` / the ``leaf_landmark`` table) into the
biological measurements Will asked for:

    * ``lamina_trace_length``  -- summed distance along the midvein trace points (midvein_0..14)
    * ``lamina_extent``        -- straight chord between the FIRST and LAST midvein trace points
    * ``lamina_tip_base_length`` -- straight lamina_tip -> lamina_base distance (separate anchors)
    * ``leaf_width``           -- width_left -> width_right distance
    * ``apex_angle`` / type    -- angle at apex_center (acute / obtuse / reflex)
    * ``base_angle`` / type    -- angle at base_center (acute / obtuse / reflex)
    * ``petiole_trace_length`` -- summed distance along the petiole trace points (petiole_0..4)
    * ``lamina_curvature``     -- lamina_trace_length / lamina_extent (>= 1; both use the SAME
                                  midvein endpoints, so it is a true arc/chord ratio)

``lamina_extent`` deliberately spans the same first/last midvein points as ``lamina_trace_length``
(NOT lamina_tip -> lamina_base), so the curvature ratio compares an arc and its chord over one
identical point set and is always >= 1. The distinct tip->base distance is ``lamina_tip_base_length``.

**Robust to occlusion.** The pose model always emits 31 keypoints, but occluded/uncertain ones
come back with low confidence; the caller drops those below ``min_kpt_conf`` before building the
``points`` mapping, so any landmark can simply be absent here. Every metric returns ``None`` when
the specific points it needs are missing -- nothing raises, nothing is fabricated.

Coordinates are plain ``(x, y)`` pixel tuples in a single frame (the working/parent frame). All
metrics are translation-invariant, so crop-frame coordinates give identical lengths and angles.

**Angle-type convention (matches LeafMachine2 ``determine_reflex``, confirmed via the
``modules/experiments/angle_checks.html`` schematic).** For the apex, form the two arm vectors
that point *into* the vertex -- ``v_left = apex_center - apex_left`` and
``v_right = apex_center - apex_right`` (i.e. "from apex_left to apex_center" and "from apex_right
to apex_center"). Let ``lamina_centroid`` be the mean of the lamina-trace points. A normal
(convex) apex has both arms pointing AWAY from the centroid -> ``acute`` (<90) or ``obtuse``
(>=90), reported as the interior angle. A notched/emarginate apex has both arms pointing TOWARD
the centroid -> ``reflex``, reported as ``360 - interior``. The base uses the identical rule with
its own three points (a cordate/heart-shaped base is the reflex case).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from leafmachine3.core.landmarks import MIDVEIN_N, PETIOLE_N

Point = tuple[float, float]

# Below this the vertex is treated as degenerate (points coincide) and the angle is skipped.
_EPS = 1e-9


@dataclass
class LandmarkMeasurements:
    """All derived per-leaf measurements. Every field is ``None`` when its inputs are missing."""

    lamina_trace_length: Optional[float] = None
    lamina_extent: Optional[float] = None
    lamina_tip_base_length: Optional[float] = None
    leaf_width: Optional[float] = None
    apex_angle: Optional[float] = None
    apex_angle_type: Optional[str] = None
    base_angle: Optional[float] = None
    base_angle_type: Optional[str] = None
    petiole_trace_length: Optional[float] = None
    lamina_curvature: Optional[float] = None
    lamina_centroid: Optional[Point] = None
    n_present: int = 0


# -- primitives --------------------------------------------------------------------
def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points: list[Point]) -> Optional[float]:
    """Summed length of the ordered polyline; ``None`` if fewer than two points."""
    if len(points) < 2:
        return None
    return float(sum(_dist(points[i], points[i + 1]) for i in range(len(points) - 1)))


def _mean(points: list[Point]) -> Optional[Point]:
    if not points:
        return None
    n = len(points)
    return (sum(p[0] for p in points) / n, sum(p[1] for p in points) / n)


# -- ordered traces ----------------------------------------------------------------
def _lamina_trace_points(pts: dict[str, Point]) -> list[Point]:
    """The lamina (midvein) trace: the present ``midvein_0..N-1`` points, index-ordered tip->base.

    ``lamina_trace_length`` is the sum of consecutive distances along THESE points only --
    ``midvein_0->midvein_1 + midvein_1->midvein_2 + ... + midvein_{N-2}->midvein_{N-1}``. The
    ``lamina_tip`` / ``lamina_base`` anchors are NOT part of the trace (they define
    ``lamina_extent``). Filtering to the present points keeps the walk correctly ordered.
    """
    return [pts[f"midvein_{i}"] for i in range(MIDVEIN_N) if f"midvein_{i}" in pts]


def _petiole_trace_points(pts: dict[str, Point]) -> list[Point]:
    """The petiole trace: the present ``petiole_0..N-1`` points, index-ordered. ``petiole_trace_length``
    sums consecutive distances along these points only (``petiole_0->petiole_1 + ...``); the
    ``lamina_base`` / ``petiole_tip`` anchors are NOT part of the trace."""
    return [pts[f"petiole_{i}"] for i in range(PETIOLE_N) if f"petiole_{i}" in pts]


# -- angles ------------------------------------------------------------------------
def _interior_angle(center: Point, left: Point, right: Point) -> Optional[float]:
    """Interior angle in degrees at ``center`` between the arms to ``left`` and ``right`` (0..180).

    Uses ``atan2(|cross|, dot)`` (numerically stable near 0 and 180). ``None`` if either arm is
    degenerate (an endpoint coincides with the center).
    """
    v1 = (left[0] - center[0], left[1] - center[1])
    v2 = (right[0] - center[0], right[1] - center[1])
    if math.hypot(*v1) < _EPS or math.hypot(*v2) < _EPS:
        return None
    cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return math.degrees(math.atan2(cross, dot))


def _classify_angle(
    center: Point, left: Point, right: Point, centroid: Optional[Point]
) -> tuple[Optional[float], Optional[str]]:
    """Return ``(angle_degrees, type)`` at ``center``; type in acute/obtuse/reflex.

    Reflex (angle > 180) is detected exactly as LM2's ``determine_reflex``: both arms that point
    *into* the vertex (``center - left`` and ``center - right``) also point toward the lamina
    centroid. When they do, the real angle wraps to ``360 - interior``. Without a centroid the
    reflex test can't run, so the interior angle is returned with ``type = None``.
    """
    interior = _interior_angle(center, left, right)
    if interior is None:
        return None, None
    if centroid is None:
        return interior, None

    v_left_in = (center[0] - left[0], center[1] - left[1])     # "from apex_left to apex_center"
    v_right_in = (center[0] - right[0], center[1] - right[1])   # "from apex_right to apex_center"
    to_centroid = (centroid[0] - center[0], centroid[1] - center[1])
    left_toward = (v_left_in[0] * to_centroid[0] + v_left_in[1] * to_centroid[1]) > 0
    right_toward = (v_right_in[0] * to_centroid[0] + v_right_in[1] * to_centroid[1]) > 0

    if left_toward and right_toward:                            # wedge opens toward the leaf body
        return 360.0 - interior, "reflex"
    return interior, ("acute" if interior < 90.0 else "obtuse")


# -- entry point -------------------------------------------------------------------
def compute_measurements(points: dict[str, Point]) -> LandmarkMeasurements:
    """Compute all derived measurements from the present (name -> ``(x, y)``) keypoints.

    ``points`` should already be filtered to confident keypoints; absent names => ``None`` metrics.
    """
    m = LandmarkMeasurements(n_present=len(points))

    lamina_pts = _lamina_trace_points(points)          # present midvein points, tip->base order
    m.lamina_centroid = _mean(lamina_pts)
    m.lamina_trace_length = _polyline_length(lamina_pts)

    # lamina_extent is the straight chord of the SAME midvein points the trace runs along (its first
    # and last present point), so lamina_curvature = trace / extent is a true arc/chord ratio (>= 1).
    # The lamina_tip -> lamina_base distance is kept separately as lamina_tip_base_length.
    if len(lamina_pts) >= 2:
        m.lamina_extent = _dist(lamina_pts[0], lamina_pts[-1])
    if "lamina_tip" in points and "lamina_base" in points:
        m.lamina_tip_base_length = _dist(points["lamina_tip"], points["lamina_base"])

    if m.lamina_trace_length is not None and m.lamina_extent and m.lamina_extent > _EPS:
        m.lamina_curvature = m.lamina_trace_length / m.lamina_extent

    if "width_left" in points and "width_right" in points:
        m.leaf_width = _dist(points["width_left"], points["width_right"])

    if all(k in points for k in ("apex_left", "apex_center", "apex_right")):
        m.apex_angle, m.apex_angle_type = _classify_angle(
            points["apex_center"], points["apex_left"], points["apex_right"], m.lamina_centroid
        )

    if all(k in points for k in ("base_left", "base_center", "base_right")):
        m.base_angle, m.base_angle_type = _classify_angle(
            points["base_center"], points["base_left"], points["base_right"], m.lamina_centroid
        )

    m.petiole_trace_length = _polyline_length(_petiole_trace_points(points))

    return m
