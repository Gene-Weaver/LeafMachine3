"""Leaf midline axes and the curvilinear ``(s, u)`` frame every symmetry metric is measured in.

Two axes are built for each leaf so the SAME metric can be computed both ways:

* ``chord``   -- the straight ``lamina_base`` -> ``lamina_tip`` line. This is what Shi et al. (2018)
  and Wang et al. (2018) use: they have only a silhouette, so the reference axis must be a chord.
* ``midvein`` -- the traced midvein polyline (``lamina_tip``, ``midvein_0..14``, ``lamina_base``),
  smoothed and resampled to equal arclength. LM3 has this from the pose model, so the lamina can be
  measured perpendicular to the LOCAL midvein direction instead of one global straight line.

The difference between the two, per metric, isolates how much apparent left/right disagreement is
really just a curved midvein being treated as straight.

Frame: image coords, y DOWN, tip at min y. ``s`` runs 0 (tip) -> 1 (base) along the axis; ``u`` is
the SIGNED perpendicular offset in pixels, **positive to the viewer's LEFT**. With a tip->base
tangent of roughly ``(0, +1)`` in y-down coords, ``cross(t, d) = t.x*d.y - t.y*d.x`` is positive for
``d = (-1, 0)`` -- a point at smaller x, i.e. screen-left. Because every mask is tip-up oriented,
"left" means the same physical side for every leaf in the cohort.

Equal-arclength perpendicular strips along a CURVED axis would overlap on the inside of a bend and
leave gaps on the outside. That is avoided here by assigning each pixel to its NEAREST point on the
axis (a Voronoi partition of the polyline), which tiles the lamina exactly once with no overlap and
no gaps -- so the per-bin areas always sum back to the total lamina area.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# axis resampling / binning defaults
N_AXIS = 512          # resampled axis vertices (the Voronoi sites)
N_BINS = 200          # arclength bins the profiles are reported on
SPLINE_SMOOTH = 3.0   # smoothing factor per point for the midvein spline fit


@dataclass
class AxisFrame:
    """A leaf midline plus the curvilinear coordinates it induces on the mask pixels."""

    kind: str                    # "chord" | "midvein"
    path: np.ndarray             # (N_AXIS, 2) equal-arclength axis points, tip -> base
    tangent: np.ndarray          # (N_AXIS, 2) unit tangents (tip -> base)
    length: float                # axis arclength, px
    chord_length: float          # straight tip->base distance, px
    xy: np.ndarray               # (P, 2) mask pixel coords (x, y)
    s: np.ndarray                # (P,) normalized arclength 0..1 of each mask pixel
    u: np.ndarray                # (P,) signed perpendicular offset, px (+ = LEFT, 0 = ON the axis)
    shape: tuple[int, int]       # (H, W) of the mask this frame was built on

    @property
    def sinuosity(self) -> float:
        """Axis arclength / straight tip-base distance. 1.0 for a perfectly straight midvein."""
        return float(self.length / self.chord_length) if self.chord_length > 0 else float("nan")

    def max_chord_deviation(self) -> float:
        """Largest perpendicular departure of the axis from its own tip->base chord, px."""
        a, b = self.path[0], self.path[-1]
        d = b - a
        n = float(np.hypot(*d))
        if n <= 0:
            return float("nan")
        d = d / n
        rel = self.path - a
        return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())

    def integrated_curvature(self) -> float:
        """Total absolute turning of the axis in radians (0 for a straight axis)."""
        t = self.tangent
        ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        return float(np.abs(np.diff(ang)).sum())


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` points at equal arclength."""
    p = np.asarray(pts, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])])


def _smooth_midvein(pts: np.ndarray, n: int) -> np.ndarray:
    """Equal-arclength resample of a smoothing-spline fit through the midvein keypoints.

    The pose model's midvein points jitter by a few pixels; without smoothing that jitter becomes
    spurious local curvature and rotates the perpendicular, which would show up as fake asymmetry.
    Falls back to a plain polyline resample if the spline cannot be fitted (too few points, or
    duplicate/collinear input).
    """
    p = np.asarray(pts, float)
    if len(p) < 4:
        return _resample_polyline(p, n)
    try:
        from scipy.interpolate import splev, splprep

        # k=3 needs >=4 points; s scales with point count (splprep's smoothing is a sum of squares)
        tck, _ = splprep([p[:, 0], p[:, 1]], s=SPLINE_SMOOTH * len(p), k=min(3, len(p) - 1))
        dense = np.column_stack(splev(np.linspace(0.0, 1.0, max(n, 4 * len(p))), tck))
        if not np.isfinite(dense).all():
            return _resample_polyline(p, n)
        return _resample_polyline(dense, n)
    except Exception:
        return _resample_polyline(p, n)


def _project(path: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point projection of pixels onto the axis -> ``(s_index_frac, signed_offset)``.

    Each pixel is assigned to its nearest axis VERTEX (a Voronoi partition of the polyline, so the
    lamina is tiled exactly once), then refined to a sub-vertex arclength position and an exact
    signed perpendicular offset using the local tangent.
    """
    from scipy.spatial import cKDTree

    n = len(path)
    _d, j = cKDTree(path).query(xy, k=1)
    j = np.clip(np.asarray(j, int), 0, n - 1)

    # local tangent at the matched vertex (central difference, unit length)
    t = np.gradient(path, axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn <= 0] = 1.0
    t = t / tn[:, None]

    d = xy - path[j]
    tj = t[j]
    along = d[:, 0] * tj[:, 0] + d[:, 1] * tj[:, 1]          # tangential residual, px
    cross = tj[:, 0] * d[:, 1] - tj[:, 1] * d[:, 0]          # signed perpendicular, + = screen-left

    # vertex spacing in px -> convert the tangential residual into a fractional vertex offset, so s
    # is continuous across vertices instead of quantized to 1/N_AXIS.
    step = float(np.hypot(*np.diff(path, axis=0).T).mean()) if n > 1 else 1.0
    s_idx = j + (along / step if step > 0 else 0.0)
    return np.clip(s_idx / max(1, n - 1), 0.0, 1.0), cross


def build_axis(mask: np.ndarray, kind: str, *, midvein: Optional[np.ndarray] = None,
               tip: Optional[np.ndarray] = None, base: Optional[np.ndarray] = None,
               n_axis: int = N_AXIS) -> Optional[AxisFrame]:
    """Build a ``chord`` or ``midvein`` frame for ``mask`` (bool, the shape being measured)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return None

    if kind == "chord":
        if tip is None or base is None:
            return None
        path = _resample_polyline(np.vstack([np.asarray(tip, float), np.asarray(base, float)]), n_axis)
    elif kind == "midvein":
        if midvein is None or len(midvein) < 3:
            return None
        path = _smooth_midvein(np.asarray(midvein, float), n_axis)
    else:
        raise ValueError(f"unknown axis kind: {kind!r}")

    seg = np.hypot(*np.diff(path, axis=0).T)
    length = float(seg.sum())
    if length <= 0:
        return None
    chord_len = float(np.hypot(*(path[-1] - path[0])))

    t = np.gradient(path, axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn <= 0] = 1.0
    t = t / tn[:, None]

    ys, xs = np.nonzero(m)
    xy = np.column_stack([xs, ys]).astype(float)
    s, u = _project(path, xy)
    return AxisFrame(kind=kind, path=path, tangent=t, length=length, chord_length=chord_len,
                     xy=xy, s=s, u=u, shape=m.shape)


@dataclass
class Profiles:
    """Binned left/right profiles along the axis -- the input to every scalar metric."""

    s: np.ndarray            # (n_bins,) bin centers, 0 (tip) .. 1 (base)
    area_l: np.ndarray       # (n_bins,) lamina pixel count left of the axis in each bin
    area_r: np.ndarray
    w_l: np.ndarray          # (n_bins,) margin envelope: max perpendicular offset, left, px
    w_r: np.ndarray
    ds_px: float             # bin width in px of arclength
    n_used: np.ndarray       # (n_bins,) total pixels in the bin (area_l + area_r)


def profiles(frame: AxisFrame, n_bins: int = N_BINS) -> Profiles:
    """Bin a frame's pixels into equal-arclength strips and split each strip left/right.

    Areas are exact pixel counts (the Voronoi partition tiles the lamina once), so
    ``area_l.sum() + area_r.sum()`` is the lamina's total pixel area. Widths are the margin
    ENVELOPE (max |u| on that side), which is what a caliper would measure and is not simply
    ``area / ds`` for a lobed leaf with a sinus.

    ON-AXIS PIXELS. ``u`` is exactly 0 for any pixel the axis passes through, which happens for a
    whole pixel column whenever the axis is axis-aligned on integer coordinates -- the case of an
    exactly mirror-symmetric synthetic mask. Lumping them with one side (they used to fall to the
    right, since ``u > 0`` is false at 0) made such a mask report a nonzero ``A_star``. They are
    split HALF to each side instead, which keeps both properties that matter: the strip areas still
    sum EXACTLY to the lamina area, and mirroring the mask (which negates ``u`` and leaves the
    on-axis set alone) still negates every signed measure exactly. They are left out of the margin
    envelopes entirely -- a pixel ON the axis lies on neither margin.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, frame.s, side="right") - 1, 0, n_bins - 1)
    left = frame.u > 0
    right = frame.u < 0
    on_axis = ~(left | right)

    half = 0.5 * np.bincount(idx[on_axis], minlength=n_bins).astype(float)
    area_l = np.bincount(idx[left], minlength=n_bins).astype(float) + half
    area_r = np.bincount(idx[right], minlength=n_bins).astype(float) + half

    w_l = np.zeros(n_bins)
    w_r = np.zeros(n_bins)
    np.maximum.at(w_l, idx[left], frame.u[left])
    np.maximum.at(w_r, idx[right], -frame.u[right])

    centers = 0.5 * (edges[:-1] + edges[1:])
    return Profiles(s=centers, area_l=area_l, area_r=area_r, w_l=w_l, w_r=w_r,
                    ds_px=float(frame.length / n_bins), n_used=area_l + area_r)


def straighten(frame: AxisFrame, *, n_s: int = 256, n_u: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the two half-laminae into a common ``(s, u)`` grid, right half MIRRORED.

    Returns ``(left, right)`` boolean grids of shape ``(n_s, n_u)``: row = arclength bin, column =
    |perpendicular offset| bin, both scaled by the SAME ``u`` range. Reflecting in ``(s, u)`` rather
    than across one Euclidean line is what makes the overlap measures meaningful for a leaf whose
    midvein is curved -- the two halves are compared after the leaf is straightened.

    Each grid CELL is sampled by mapping its ``(s, |u|)`` back to a sheet point
    ``path[s] +/- |u| * normal[s]`` and testing the mask there. Scattering the mask's pixels
    FORWARD into the grid instead would be resolution-dependent and badly wrong for small leaves: a
    20k-px lamina has fewer pixels per half than the grid has cells, so most cells stay empty, the
    two halves' holes do not coincide, and Dice collapses toward 0 no matter how symmetric the leaf
    actually is (measured: several leaves scored a literal 0.000 that way). Sampling backward fills
    every cell and makes the result independent of both leaf size and grid resolution.

    The backward map is the exact inverse of the forward nearest-point projection only where the
    axis's radius of curvature exceeds ``|u|``; on a strongly bowed midvein the normals converge on
    the concave side. Both halves are sampled identically, so the comparison stays fair, and leaf
    midveins are far too gently curved (sinuosity ~1.01) for the crossing to arise in practice.
    """
    umax = float(np.abs(frame.u).max()) if frame.u.size else 0.0
    empty = (np.zeros((n_s, n_u), bool), np.zeros((n_s, n_u), bool))
    if umax <= 0:
        return empty

    h, w = frame.shape
    mask = np.zeros((h, w), bool)
    ij = np.rint(frame.xy).astype(int)
    np.clip(ij[:, 0], 0, w - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, h - 1, out=ij[:, 1])
    mask[ij[:, 1], ij[:, 0]] = True

    # cell centers: s in (0,1), |u| in (0, umax]
    s_c = (np.arange(n_s) + 0.5) / n_s
    u_c = (np.arange(n_u) + 0.5) / n_u * umax
    idx = np.clip(np.rint(s_c * (len(frame.path) - 1)).astype(int), 0, len(frame.path) - 1)
    base = frame.path[idx]                       # (n_s, 2)
    t = frame.tangent[idx]
    n_left = np.column_stack([-t[:, 1], t[:, 0]])   # cross(t, n_left) = +1 -> the viewer's LEFT

    out = []
    for sign in (+1.0, -1.0):
        p = base[:, None, :] + sign * u_c[None, :, None] * n_left[:, None, :]
        xi = np.rint(p[..., 0]).astype(int)
        yi = np.rint(p[..., 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        g = np.zeros((n_s, n_u), bool)
        g[ok] = mask[yi[ok], xi[ok]]
        out.append(g)
    return out[0], out[1]
