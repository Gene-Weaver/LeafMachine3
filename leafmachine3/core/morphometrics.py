"""Leaf-mask morphometrics, ported from LeafMachine2's ``measure_leaf_segmentation``.

Pure functions over a polygon (Nx2 array in the working/parent pixel frame). Scalar
metrics (area, perimeter, centroid, convex hull, convexity/concavity, circularity,
aspect ratio, vertex count) plus the rotated (minimum) bounding box.

Four rotated-bbox methods are available via ``polygon_morphology(..., method=...)``:

* ``"feret"`` (DEFAULT) — orient by the maximum-Feret axis (the longest chord across the convex
  hull), then measure the hull extents along and perpendicular to it. Tracks the leaf's true
  long axis (tip->base) and stays tight; the recommended default.
* ``"lm2"`` — LM2's ``fit_min_bbox``: rotate the simplified polygon in 1-degree steps until the
  axis-aligned box's long side matches the min-enclosing-circle diameter.
* ``"minarearect"`` — OpenCV ``cv2.minAreaRect``: the true minimum-AREA rotated rectangle.
  Geometrically tightest, but minimizes area rather than orientation, so it can mis-align.
* ``"pca"`` — orient by the mask's area-weighted principal (second-moment) axis, then measure
  the hull extents; robust for near-symmetric shapes.

All four return the same tuple ``(angle, dim_max, dim_min, corners, circle)`` and are drop-in.
Elliptic Fourier descriptors are not ported yet — the module is structured so an ``efds``
field can be added later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:  # pragma: no cover - cv2 is always present at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class Morphometrics:
    """All scalar morphology for one mask (working-frame pixels)."""
    area_px: float
    perimeter_px: float
    centroid: tuple[float, float]           # (x, y) working coords (mean of vertices, LM2 style)
    convex_hull_area: float
    convexity: float
    concavity: float
    circularity: float
    aspect_ratio: float                     # dim_max / dim_min (rotated), i.e. length / width
    n_vertices: int
    bbox: tuple[float, float, float, float]  # axis-aligned (x1, y1, x2, y2), working coords
    rotate_angle: float                     # degrees
    dim_max: float                          # rotated bbox LONG side  = leaf length
    dim_min: float                          # rotated bbox SHORT side = leaf width
    rotated_bbox: list[list[int]]           # 4 corners [[x, y], ...], working coords
    circle: tuple[float, float, float]      # min enclosing circle (cx, cy, radius)


def rotate_polygon_by_angle(polygon, angle_deg: float, cx: float, cy: float) -> list[tuple[int, int]]:
    """Rotate ``polygon`` (iterable of (x, y)) by ``angle_deg`` about ``(cx, cy)`` (LM2 port)."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    out: list[tuple[int, int]] = []
    for x, y in polygon:
        tx, ty = x - cx, y - cy
        out.append((int(tx * cos_a - ty * sin_a + cx), int(tx * sin_a + ty * cos_a + cy)))
    return out


def _rect_edges(rect) -> list[float]:
    return [math.hypot(rect[i][0] - rect[i + 1][0], rect[i][1] - rect[i + 1][1]) for i in range(3)]


def fit_min_bbox(polygon, polygon_approx, cx: int, cy: int, *, find_min: bool = True, tolerance: int = 2):
    """LM2 ``fit_min_bbox`` port.

    Returns ``(angle, dim_max, dim_min, rotated_bbox_corners, (cir_x, cir_y, radius))`` where
    ``dim_max``/``dim_min`` are the long/short sides of the rotated minimum bounding box.
    """
    poly_i = np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
    x, y, w, h = cv2.boundingRect(poly_i)
    rect_orig = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    edges = _rect_edges(rect_orig)
    max_len, min_len = max(edges), min(edges)
    (cir_x, cir_y), radius = cv2.minEnclosingCircle(poly_i.astype(np.float32))
    circle = (float(cir_x), float(cir_y), float(radius))

    if not find_min:
        return 0.0, float(2.0 * radius), float(min_len), rect_orig, circle
    if abs(max_len - 2 * radius) < tolerance:
        return 0.0, float(max_len), float(min_len), rect_orig, circle

    angle = 0
    tol = tolerance
    while abs(max_len - 2 * radius) >= tol:
        if angle > 180:                       # widen tolerance and restart the sweep (LM2)
            tol += 2
            angle = 0
        angle += 1
        rot = rotate_polygon_by_angle(polygon_approx, angle, cx, cy)
        rx, ry, rw, rh = cv2.boundingRect(np.asarray(rot, dtype=np.int32))
        rect = [(rx, ry), (rx + rw, ry), (rx + rw, ry + rh), (rx, ry + rh)]
        edges = _rect_edges(rect)
        max_len, min_len = max(edges), min(edges)
        if tol > 4:                           # give up: fall back to the axis-aligned box
            return 0.0, float(max_len), float(min_len), rect_orig, circle

    bbox_min = rotate_polygon_by_angle(rect, -angle, cx, cy)   # rotate the tight rect back
    return float(angle), float(max_len), float(min_len), bbox_min, circle


_MIN_AREA_RECT_ALIASES = {"minarearect", "min_area_rect", "cv2", "cv2.minarearect"}
_LM2_ALIASES = {"lm2", "fit_min_bbox", "circle"}
_PCA_ALIASES = {"pca", "principal_axis", "moments"}
_FERET_ALIASES = {"feret", "max_feret", "tier1"}


def min_area_rect_bbox(polygon):
    """OpenCV minimum-AREA rotated rectangle.

    Returns ``(angle, dim_max, dim_min, corners, (cir_x, cir_y, radius))`` matching
    :func:`fit_min_bbox`'s signature so the methods are drop-in interchangeable.
    """
    poly_f = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    (cx, cy), (w, h), angle = cv2.minAreaRect(poly_f)
    box = cv2.boxPoints(((cx, cy), (w, h), angle))
    (cir_x, cir_y), radius = cv2.minEnclosingCircle(poly_f)
    dim_max, dim_min = (max(w, h), min(w, h))
    return (float(angle), float(dim_max), float(dim_min),
            [[int(x), int(y)] for x, y in box], (float(cir_x), float(cir_y), float(radius)))


def _oriented_from_axis(polygon, theta_deg):
    """Bounding box of the convex hull measured in the frame rotated to ``theta_deg``.

    Shared by ``feret`` and ``pca``: given a long-axis angle, measure the hull extents along
    and perpendicular to it. Returns the standard ``(angle, dim_max, dim_min, corners, circle)``.
    """
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(float)
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    xr = hull[:, 0] * c + hull[:, 1] * s            # hull rotated by -theta (axis -> x)
    yr = -hull[:, 0] * s + hull[:, 1] * c
    mnx, mxx, mny, mxy = float(xr.min()), float(xr.max()), float(yr.min()), float(yr.max())
    corners_r = np.array([[mnx, mny], [mxx, mny], [mxx, mxy], [mnx, mxy]])
    corners = np.column_stack([corners_r[:, 0] * c - corners_r[:, 1] * s,     # rotate back by +theta
                               corners_r[:, 0] * s + corners_r[:, 1] * c])
    dx, dy = mxx - mnx, mxy - mny
    dim_max, dim_min = (dx, dy) if dx >= dy else (dy, dx)
    long_deg = theta_deg if dx >= dy else theta_deg + 90.0
    angle = ((long_deg + 90.0) % 180.0) - 90.0       # long-axis tilt in [-90, 90)
    (cir_x, cir_y), radius = cv2.minEnclosingCircle(pts)
    return (float(angle), float(dim_max), float(dim_min),
            [[int(x), int(y)] for x, y in corners], (float(cir_x), float(cir_y), float(radius)))


def feret_bbox(polygon):
    """Tier 1: orient by the max-Feret axis (longest hull chord), measure hull extents."""
    hull = cv2.convexHull(np.asarray(polygon, dtype=np.float32).reshape(-1, 2)).reshape(-1, 2).astype(float)
    best, pa, pb = -1.0, hull[0], hull[0]
    for i in range(len(hull)):
        d = ((hull - hull[i]) ** 2).sum(1)
        j = int(d.argmax())
        if d[j] > best:
            best, pa, pb = d[j], hull[i], hull[j]
    theta = math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
    return _oriented_from_axis(polygon, theta)


def pca_bbox(polygon):
    """Tier 2: orient by the area-weighted principal (second-moment) axis, measure hull extents."""
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    m = cv2.moments(pts.reshape(-1, 1, 2).astype(np.int32))
    if m["m00"] != 0:
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        theta = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
    else:
        theta = 0.0
    return _oriented_from_axis(polygon, theta)


def polygon_morphology(polygon, *, find_min_bbox: bool = True, method: str = "feret") -> Morphometrics | None:
    """Compute all scalar morphology + the rotated bbox for one polygon (working coords).

    ``method`` selects the rotated-bbox algorithm: ``"feret"`` (default), ``"lm2"``,
    ``"minarearect"`` (cv2), or ``"pca"``.
    """
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return None
    closed = np.vstack([poly, poly[:1]])
    contour = closed.astype(np.int32).reshape(-1, 1, 2)

    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(closed.astype(np.float32)))
    cx, cy = float(closed[:, 0].mean()), float(closed[:, 1].mean())   # LM2: mean of vertices
    hull = cv2.convexHull(closed.astype(np.float32))
    convex_hull_area = float(cv2.contourArea(hull))
    convexity = area / convex_hull_area if convex_hull_area > 0 else -1.0
    concavity = (1.0 - convexity) if convexity != -1.0 else -1.0
    circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else -1.0
    bx, by, bw, bh = cv2.boundingRect(closed.astype(np.int32))
    bbox = (float(bx), float(by), float(bx + bw), float(by + bh))

    method = str(method).lower()
    if method in _MIN_AREA_RECT_ALIASES:
        angle, dim_max, dim_min, rotated_bbox, circle = min_area_rect_bbox(closed)
    elif method in _PCA_ALIASES:
        angle, dim_max, dim_min, rotated_bbox, circle = pca_bbox(closed)
    elif method in _LM2_ALIASES:
        approx = cv2.approxPolyDP(contour, 0.010 * perimeter, True).reshape(-1, 2)   # LM2 1% simplify
        if len(approx) < 3:
            approx = closed
        angle, dim_max, dim_min, rotated_bbox, circle = fit_min_bbox(
            closed, approx, int(round(cx)), int(round(cy)), find_min=find_min_bbox
        )
    else:  # default: "feret" (Tier 1)
        angle, dim_max, dim_min, rotated_bbox, circle = feret_bbox(closed)
    aspect_ratio = (dim_max / dim_min) if dim_min else 0.0

    return Morphometrics(
        area_px=area, perimeter_px=perimeter, centroid=(cx, cy),
        convex_hull_area=convex_hull_area, convexity=convexity, concavity=concavity,
        circularity=circularity, aspect_ratio=aspect_ratio, n_vertices=int(len(closed)),
        bbox=bbox, rotate_angle=angle, dim_max=dim_max, dim_min=dim_min,
        rotated_bbox=[[int(px), int(py)] for px, py in rotated_bbox], circle=circle,
    )
