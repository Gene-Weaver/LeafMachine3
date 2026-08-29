"""Petiole width measurement from the Petiole mask + the landmark petiole centerline.

We already predict the petiole as an ordered landmark trace (``lamina_base`` -> ``petiole_0..4`` ->
``petiole_tip``), so instead of skeletonising the mask we use that trace as the centerline and
measure the **perpendicular** thickness of the Petiole mask at several points near the blade
junction (the ``lamina_base`` end), reporting the **median** (LM2 sampled a single width -> noisy).
A short thin petiole is very sensitive to leftover background at its edges, so this is intended to
run on the **edge-refined** petiole mask once ``LM3_Specimen_Segmentation`` lands (TODO #5); until
then it uses the raw Petiole mask. (If the landmark centerline is ever unavailable, a future
fallback is to skeletonise the mask + BFS the centerline.)

All geometry is in ONE frame (the caller passes crop-frame masks + a crop-frame centerline); the
returned width/sample segments are in that same frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 is always present at runtime
    cv2 = None

Point = tuple[float, float]
Segment = list                        # [[x1, y1], [x2, y2]]

# Sample the width at these fractions of the petiole length, measured from the blade junction
# (lamina_base end) -- the near-base region below the junction flare. Median of the valid samples.
DEFAULT_FRACTIONS: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30)
_FALLBACK_FRACTIONS: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
_EPS = 1e-9


@dataclass
class PetioleWidth:
    width_px: Optional[float] = None
    length_px: Optional[float] = None
    n_samples: int = 0
    touches_leaf: bool = False
    measure_location: str = "none"                 # "near_base" | "none"
    width_segment: Optional[Segment] = None        # the reported (median) width segment
    sample_segments: list[Segment] = field(default_factory=list)   # every valid sample segment


# -- centerline helpers ------------------------------------------------------------
def _seg_lengths(pts: list[Point]) -> list[float]:
    return [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)]


def _point_and_tangent(pts: list[Point], seglens: list[float], target: float) -> tuple[Point, Point]:
    """Point at arc-length ``target`` along the polyline and the local unit tangent there."""
    acc = 0.0
    for i, L in enumerate(seglens):
        if L < _EPS:
            continue
        if acc + L >= target or i == len(seglens) - 1:
            t = min(1.0, max(0.0, (target - acc) / L))
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            p = (ax + t * (bx - ax), ay + t * (by - ay))
            d = ((bx - ax) / L, (by - ay) / L)
            return p, d
        acc += L
    return pts[-1], (1.0, 0.0)


def _perpendicular_width(mask: np.ndarray, p: Point, d: Point, max_r: float):
    """Thickness of ``mask`` across the line through ``p`` perpendicular to tangent ``d``.

    Returns ``(width, [edge_minus, edge_plus])`` for the mask run nearest ``p``, or ``None`` if there
    is no run or the run reaches the probe end (direction ~parallel to the petiole => discard).
    """
    h, w = mask.shape[:2]
    n = (-d[1], d[0])                                # unit perpendicular (d is already unit length)
    ts = np.arange(-max_r, max_r + 1.0, 1.0)
    inside = np.zeros(ts.shape, dtype=bool)
    for k, t in enumerate(ts):
        x = int(round(p[0] + t * n[0]))
        y = int(round(p[1] + t * n[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            inside[k] = True
    if not inside.any():
        return None
    i0 = len(ts) // 2                                # index nearest t = 0 (the centerline point)
    if not inside[i0]:
        trues = np.where(inside)[0]
        i0 = int(trues[np.argmin(np.abs(trues - i0))])
    lo = hi = i0
    while lo - 1 >= 0 and inside[lo - 1]:
        lo -= 1
    while hi + 1 < len(inside) and inside[hi + 1]:
        hi += 1
    if lo == 0 or hi == len(inside) - 1:             # run hit the probe boundary -> bad perpendicular
        return None
    t_lo, t_hi = ts[lo], ts[hi]
    e_minus = (p[0] + t_lo * n[0], p[1] + t_lo * n[1])
    e_plus = (p[0] + t_hi * n[0], p[1] + t_hi * n[1])
    return math.hypot(e_plus[0] - e_minus[0], e_plus[1] - e_minus[1]), [list(e_minus), list(e_plus)]


def _touches_leaf(petiole_mask: np.ndarray, leaf_mask: Optional[np.ndarray], touch_dist: int) -> bool:
    if leaf_mask is None or cv2 is None or not petiole_mask.any() or not leaf_mask.any():
        return False
    k = 2 * int(touch_dist) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    grown = cv2.dilate(leaf_mask.astype(np.uint8), kernel) > 0
    return bool((grown & petiole_mask).any())


def measure_petiole(
    petiole_mask: np.ndarray,
    leaf_mask: Optional[np.ndarray],
    centerline: list[Point],
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    touch_dist: int = 20,
) -> PetioleWidth:
    """Measure petiole width (median of perpendicular samples near the blade junction).

    ``centerline`` is the ordered petiole trace (lamina_base -> ... -> petiole_tip) in the mask
    frame; ``petiole_mask`` / ``leaf_mask`` are boolean arrays in that same frame.
    """
    out = PetioleWidth()
    pm = np.asarray(petiole_mask, dtype=bool)
    if pm.ndim != 2 or not pm.any():
        return out
    out.touches_leaf = _touches_leaf(pm, None if leaf_mask is None else np.asarray(leaf_mask, bool), touch_dist)
    if len(centerline) < 2:                            # no usable landmark centerline -> width unknown
        return out

    seglens = _seg_lengths(centerline)
    length = float(sum(seglens))
    out.length_px = length
    if length < _EPS:
        return out
    max_r = float(max(pm.shape))                      # generous probe; the mask run is short

    def _collect(fracs):
        widths, segs = [], []
        for f in fracs:
            p, d = _point_and_tangent(centerline, seglens, f * length)
            res = _perpendicular_width(pm, p, d, max_r)
            if res is not None:
                widths.append(res[0])
                segs.append(res[1])
        return widths, segs

    widths, segs = _collect(fractions)
    if not widths:                                    # near-base samples all missed -> sweep the whole petiole
        widths, segs = _collect(_FALLBACK_FRACTIONS)
    if not widths:
        return out

    med = float(median(widths))
    best = min(range(len(widths)), key=lambda i: abs(widths[i] - med))   # sample closest to the median
    out.width_px = med
    out.n_samples = len(widths)
    out.sample_segments = segs
    out.width_segment = segs[best]
    out.measure_location = "near_base"
    return out
