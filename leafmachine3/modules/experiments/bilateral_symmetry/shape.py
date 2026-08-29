"""Whole-shape symmetry of the two half-laminae, compared as SHAPES rather than strip by strip.

Everything here is measured in the curvilinear ``(s, u)`` frame of ``axes``, so "reflect the right
half onto the left" means ``u -> -u`` about the LOCAL midvein direction, not a mirror across one
global Euclidean line. That distinction is the experiment: on a leaf with a curved midvein a
Euclidean mirror charges the bend itself as asymmetry, while the curvilinear mirror charges only
genuine differences between the two laminae.

Five families of measure, all flattened onto one record:

1. OVERLAP of the two straightened halves rasterized on a common grid -- IoU, Dice, and the
   normalized symmetric difference. Dice and SD are algebraically complementary
   (``SD == 1 - Dice``, because ``|L ^ R| == |L| + |R| - 2|L & R|``); both are kept because both are
   quoted in the literature and readers look for the one they know.
2. MARGIN CURVE distance between the envelopes ``w_L(s)`` and ``w_R(s)`` -- mean and max absolute
   gap, plus a Hausdorff distance that treats each margin as a point SET in an ISOTROPIC
   ``(s_px, u)`` plane. Scaling ``s`` back to pixels matters: without it the two axes would carry
   different units and the "distance" would depend on the bin count.
3. CENTROIDS and second moments of each half in ``(s_px, |u|)`` -- where each half's mass sits and
   how differently it is spread. Insensitive to the fine margin, so it complements (2).
4. MAX-WIDTH position and magnitude per side -- the single most-quoted shape landmark.
5. MARGIN CURVATURE and LOBING -- curvature of each margin treated as a 1-D graph, and a
   prominence-gated lobe count with the positional shift of matched lobes.

Units. Pixel-valued: ``mad_boundary``, ``d_max``, ``hausdorff``, ``d_s``, ``d_u``, ``d_c``,
``d_w_max``, and the centroid coordinates. Normalized arclength 0 (tip) .. 1 (base): ``s_at_d_max``,
``s_l_max``, ``s_r_max``, ``d_s_max``, ``mean_lobe_shift``. Dimensionless: ``iou``, ``dice``, ``sd``,
``hausdorff_norm``, ``d_c_norm``, ``cov_dissimilarity``, ``d_w_max_rel``. ``rmse_kappa`` is 1/px.

``rmse_kappa`` IS UNRELIABLE BELOW ROUGHLY 800 px OF AXIS LENGTH -- treat it as a raster-noise
readout, not a curvature one. Curvature is a second derivative, so it should scale as 1/L and
``rmse_kappa * L`` should be constant for a fixed shape. Measured on a rasterized ellipse with
unequal half-widths (b_l/b_r = 70/55), against the analytic margin curves of the same shape fed
through this exact code:

===========  ==============  ================  =============
axis length  rmse_kappa      analytic          measured / true
===========  ==============  ================  =============
200 px       9.29e-02        2.50e-03          37.2x
400 px       4.95e-03        1.25e-03          4.0x
800 px       1.32e-03        6.24e-04          2.1x
1600 px      3.97e-04        3.12e-04          1.3x
===========  ==============  ================  =============

``rmse_kappa * L`` is 18.6, 1.98, 1.06, 0.64 while the analytic value holds 0.499 throughout: the
1/L scaling only starts to appear above ~800 px. The margin of a small raster leaf is a staircase,
and :func:`_smooth` at ``SMOOTH_SIGMA`` bins is deliberately gentle enough to leave real lobes
intact, so it cannot also flatten single-pixel steps. Smoothing hard enough to fix this would
destroy the lobe detection that shares the same smoothed profile, so the field is kept as-is and
documented rather than silently "improved". On the 105-leaf test cohort the median axis length is
424 px and 84% of leaves are under 800 px, so cohort-level ``rmse_kappa`` there is dominated by
rasterization, and any comparison between leaves of different sizes is meaningless.

Degenerate input (an empty side, too few bins, a zero denominator) yields ``nan`` -- never an
exception, and never a 0.0 that would read as "perfectly symmetric". That includes a mask with no
lamina off the axis at all, which has no margin and therefore no ``mad_boundary``, ``d_max``,
``hausdorff``, ``d_w_max`` or ``rmse_kappa``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from leafmachine3.modules.experiments.bilateral_symmetry.axes import (
    AxisFrame,
    Profiles,
    straighten,
)

SMOOTH_SIGMA = 2.0      # gaussian_filter1d sigma, in BINS, applied to the margin before derivatives
LOBE_PROMINENCE = 0.05  # peak prominence gate, as a fraction of that side's max width
N_S, N_U = 256, 128     # straightened-grid resolution for the overlap measures

_NAN = float("nan")


@dataclass(frozen=True)
class ShapeSymmetry:
    """One leaf-axis pair's whole-shape symmetry measures. All fields are scalars."""

    # (1) overlap of the straightened halves
    iou: float
    dice: float
    sd: float
    # (2) margin curve distance
    mad_boundary: float
    d_max: float
    s_at_d_max: float
    hausdorff: float
    hausdorff_norm: float
    # (3) centroids and second moments in (s_px, |u|)
    sbar_l: float
    ubar_l: float
    sbar_r: float
    ubar_r: float
    d_s: float
    d_u: float
    d_c: float
    d_c_norm: float
    cov_dissimilarity: float
    # (4) max-width location
    s_l_max: float
    s_r_max: float
    d_s_max: float
    d_w_max: float
    d_w_max_rel: float
    # (5) margin curvature and lobing
    rmse_kappa: float          # raster-noise dominated below ~800 px axis length -- see module doc
    n_lobes_l: int
    n_lobes_r: int
    lobe_count_diff: int
    n_unmatched_lobes: int
    mean_lobe_shift: float

    def as_dict(self) -> dict:
        return asdict(self)


def _span(prof: Profiles) -> slice:
    """First .. last non-empty bin.

    Bins beyond the lamina hold ``w == 0`` on BOTH sides, so keeping them would dilute the mean
    margin gap with pairs of points that lie on no margin at all. A contiguous span is used instead
    of a boolean mask because the curvature and peak finding assume uniformly spaced bins -- which
    means empty bins INSIDE the span survive this slice. They are dropped separately in
    :func:`_margin`, the one place where averaging them in would bias a number.
    """
    used = np.nonzero(np.asarray(prof.n_used) > 0)[0]
    if used.size == 0:
        return slice(0, 0)
    return slice(int(used[0]), int(used[-1]) + 1)


def _overlap(frame: AxisFrame, n_s: int, n_u: int) -> tuple[float, float, float]:
    """IoU, Dice and normalized symmetric difference of the two straightened halves."""
    left, right = straighten(frame, n_s=n_s, n_u=n_u)
    inter = float(np.count_nonzero(left & right))
    union = float(np.count_nonzero(left | right))
    total = float(np.count_nonzero(left) + np.count_nonzero(right))
    iou = inter / union if union > 0 else _NAN
    dice = 2.0 * inter / total if total > 0 else _NAN
    sd = float(np.count_nonzero(left ^ right)) / total if total > 0 else _NAN
    return iou, dice, sd


def _margin(frame: AxisFrame, prof: Profiles, span: slice) -> tuple[float, float, float, float, float]:
    """Mean/max margin gap, the ``s`` of the max, and the two-sided Hausdorff distance.

    Empty bins INSIDE the span are dropped here rather than in :func:`_span`: they carry ``w == 0``
    on both sides, so they are not points on either margin, and averaging them in would pull
    ``mad_boundary`` toward 0 and hand the Hausdorff search free exact matches. ``_span`` itself has
    to stay contiguous for the curvature and peak finding.
    """
    s = np.asarray(prof.s[span], float)
    w_l = np.asarray(prof.w_l[span], float)
    w_r = np.asarray(prof.w_r[span], float)
    occ = np.asarray(prof.n_used, float)[span] > 0
    if s.size == 0 or not occ.any():
        return _NAN, _NAN, _NAN, _NAN, _NAN
    s, w_l, w_r = s[occ], w_l[occ], w_r[occ]

    # No lamina anywhere off the axis means there is no margin to compare, which is not the same
    # thing as two margins that agree perfectly -- report nan, not the 0.0 that reads as symmetric.
    if not (np.any(w_l > 0) or np.any(w_r > 0)):
        return _NAN, _NAN, _NAN, _NAN, _NAN

    gap = np.abs(w_l - w_r)
    finite = np.isfinite(gap)
    mad = float(gap.mean())
    d_max = float(gap.max()) if finite.all() else _NAN
    # np.argmax on an all-zero gap returns 0 and would name the TIP as the site of the largest
    # mismatch; with no mismatch anywhere there is no such site.
    s_at = float(s[int(np.argmax(gap))]) if (finite.all() and gap.max() > 0) else _NAN

    length = float(frame.length)
    # A nan anywhere in either margin makes the point sets meaningless; scipy would still return a
    # finite Hausdorff distance from the remaining rows, so bail out the way mad/d_max do.
    if not np.isfinite(length) or length <= 0 or not (np.isfinite(w_l).all() and np.isfinite(w_r).all()):
        return mad, d_max, s_at, _NAN, _NAN

    # Both curves live in the same isotropic plane: arclength in px against offset in px.
    s_px = s * length
    p_l = np.column_stack([s_px, w_l])
    p_r = np.column_stack([s_px, w_r])
    try:
        from scipy.spatial.distance import directed_hausdorff

        h = max(float(directed_hausdorff(p_l, p_r)[0]), float(directed_hausdorff(p_r, p_l)[0]))
    except Exception:
        h = _NAN
    return mad, d_max, s_at, h, (h / length)


def _half_moments(s_px: np.ndarray, abs_u: np.ndarray, side: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Centroid and 2x2 covariance of one half's pixels in ``(s_px, |u|)``."""
    n = int(np.count_nonzero(side))
    if n == 0:
        return _NAN, _NAN, np.full((2, 2), _NAN)
    x, y = s_px[side], abs_u[side]
    cov = np.cov(x, y) if n >= 2 else np.full((2, 2), _NAN)
    return float(x.mean()), float(y.mean()), np.asarray(cov, float).reshape(2, 2)


def _centroids(frame: AxisFrame) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Per-half centroids in ``(s_px, |u|)`` (right half reflected) and a covariance dissimilarity."""
    length = float(frame.length)
    u = np.asarray(frame.u, float)
    s_px = np.asarray(frame.s, float) * length
    abs_u = np.abs(u)
    # A pixel exactly ON the axis (u == 0, a whole column of them when the axis is axis-aligned)
    # belongs to neither half, so it is excluded rather than swept into the right side by `u > 0`
    # -- otherwise a mirror-symmetric mask reports a nonzero d_u/d_c. `profiles` splits its AREA
    # half and half, which it can because area is additive; a centroid is not.
    left = u > 0
    right = u < 0

    sbar_l, ubar_l, cov_l = _half_moments(s_px, abs_u, left)
    sbar_r, ubar_r, cov_r = _half_moments(s_px, abs_u, right)

    d_s = sbar_l - sbar_r
    d_u = ubar_l - ubar_r
    d_c = float(np.hypot(d_s, d_u))
    d_c_norm = d_c / length if np.isfinite(length) and length > 0 else _NAN

    # Normalizing by the mean trace turns an area-scaled quantity into a shape one, so leaves of
    # different sizes stay comparable.
    trace = 0.5 * (float(np.trace(cov_l)) + float(np.trace(cov_r)))
    if np.isfinite(trace) and trace > 0:
        cov_d = float(np.linalg.norm(cov_l - cov_r, ord="fro") / trace)
    else:
        cov_d = _NAN
    return sbar_l, ubar_l, sbar_r, ubar_r, d_s, d_u, d_c, d_c_norm, cov_d


def _max_width(prof: Profiles, span: slice) -> tuple[float, float, float, float, float]:
    """Position and magnitude of each side's widest point, and their differences."""
    s = np.asarray(prof.s[span], float)
    w_l = np.asarray(prof.w_l[span], float)
    w_r = np.asarray(prof.w_r[span], float)
    if s.size == 0:
        return _NAN, _NAN, _NAN, _NAN, _NAN

    m_l, m_r = float(w_l.max()), float(w_r.max())
    # neither side has any width: no widest point exists, so d_w_max is nan rather than a 0.0 that
    # would read as "the two halves are equally wide"
    if not (m_l > 0 or m_r > 0):
        return _NAN, _NAN, _NAN, _NAN, _NAN
    s_l = float(s[int(np.argmax(w_l))]) if m_l > 0 else _NAN
    s_r = float(s[int(np.argmax(w_r))]) if m_r > 0 else _NAN
    d_w = m_l - m_r
    total = m_l + m_r
    return s_l, s_r, s_l - s_r, d_w, (d_w / total if total > 0 else _NAN)


def _smooth(w: np.ndarray, sigma: float) -> np.ndarray:
    """Light 1-D smoothing of a margin profile.

    Derivatives of a raster margin amplify single-pixel staircase noise into huge fake curvature, so
    the profile is smoothed first; sigma is in bins and stays small enough to leave real lobes.
    """
    y = np.asarray(w, float)
    if sigma <= 0 or y.size < 3:
        return y
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(y, float(sigma), mode="nearest")


def _kappa(y: np.ndarray, ds_px: float) -> np.ndarray:
    """Curvature of the margin taken as a graph ``u = w(s_px)``: ``f'' / (1 + f'^2)^1.5``.

    The formula is fine; the INPUT is the problem on small leaves. See the ``rmse_kappa`` warning in
    the module docstring before using anything derived from this.
    """
    if y.size < 3 or not np.isfinite(ds_px) or ds_px <= 0:
        return np.full(y.shape, _NAN)
    d1 = np.gradient(y, ds_px)
    d2 = np.gradient(d1, ds_px)
    return d2 / np.power(1.0 + d1 * d1, 1.5)


def _peaks(y: np.ndarray, s: np.ndarray, prominence_frac: float) -> np.ndarray:
    """``s`` positions of prominent local maxima of one side's width profile.

    Prominence is scaled by that side's own max width so the gate means the same thing on a small
    leaf as on a large one. Endpoint maxima are deliberately not counted -- a margin that is simply
    widest at the base is not a lobe.
    """
    if y.size < 3:
        return np.empty(0, float)
    peak = float(y.max())
    if not np.isfinite(peak) or peak <= 0:
        return np.empty(0, float)
    from scipy.signal import find_peaks

    idx, _props = find_peaks(y, prominence=max(float(prominence_frac) * peak, 1e-12))
    return np.asarray(s, float)[idx]


def _match_peaks(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Greedy nearest-first pairing of two lobe position sets -> (mean shift, unmatched count).

    Greedy is used rather than an optimal assignment because lobes are ordered along ``s``: the
    closest pair is almost always the correct pair, and greedy degrades gracefully when the two
    sides disagree on how many lobes there are.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return _NAN, int(a.size + b.size)

    dist = np.abs(a[:, None] - b[None, :])
    used_a = np.zeros(a.size, bool)
    used_b = np.zeros(b.size, bool)
    shifts: list[float] = []
    for flat in np.argsort(dist, axis=None):
        i, j = divmod(int(flat), b.size)
        if used_a[i] or used_b[j]:
            continue
        used_a[i] = used_b[j] = True
        shifts.append(float(dist[i, j]))
        if len(shifts) == min(a.size, b.size):
            break
    unmatched = int(np.count_nonzero(~used_a) + np.count_nonzero(~used_b))
    return (float(np.mean(shifts)) if shifts else _NAN), unmatched


def _lobing(prof: Profiles, span: slice, sigma: float,
            prominence: float) -> tuple[float, int, int, int, int, float]:
    """Margin curvature RMSE plus the lobe counts, count difference and matched-lobe shift."""
    s = np.asarray(prof.s[span], float)
    if s.size == 0:
        return _NAN, 0, 0, 0, 0, _NAN

    y_l = _smooth(prof.w_l[span], sigma)
    y_r = _smooth(prof.w_r[span], sigma)
    # no margin at all (no lamina off the axis) -> no curvature to compare, so nan not 0.0
    if not (np.any(y_l > 0) or np.any(y_r > 0)):
        return _NAN, 0, 0, 0, 0, _NAN

    k_l = _kappa(y_l, float(prof.ds_px))
    k_r = _kappa(y_r, float(prof.ds_px))
    d = k_l - k_r
    ok = np.isfinite(d)
    rmse = float(np.sqrt(np.mean(np.square(d[ok])))) if np.any(ok) else _NAN

    p_l = _peaks(y_l, s, prominence)
    p_r = _peaks(y_r, s, prominence)
    shift, unmatched = _match_peaks(p_l, p_r)
    return rmse, int(p_l.size), int(p_r.size), int(abs(p_l.size - p_r.size)), unmatched, shift


def compute_shape(frame: AxisFrame, prof: Profiles, *, smooth_sigma: float = SMOOTH_SIGMA,
                  lobe_prominence: float = LOBE_PROMINENCE, n_s: int = N_S,
                  n_u: int = N_U) -> ShapeSymmetry:
    """All whole-shape symmetry measures for one leaf under one axis frame."""
    span = _span(prof)
    iou, dice, sd = _overlap(frame, n_s, n_u)
    mad, d_max, s_at_d_max, hausdorff, hausdorff_norm = _margin(frame, prof, span)
    sbar_l, ubar_l, sbar_r, ubar_r, d_s, d_u, d_c, d_c_norm, cov_d = _centroids(frame)
    s_l_max, s_r_max, d_s_max, d_w_max, d_w_max_rel = _max_width(prof, span)
    rmse_kappa, n_l, n_r, n_diff, n_unmatched, shift = _lobing(
        prof, span, smooth_sigma, lobe_prominence)

    return ShapeSymmetry(
        iou=iou, dice=dice, sd=sd,
        mad_boundary=mad, d_max=d_max, s_at_d_max=s_at_d_max,
        hausdorff=hausdorff, hausdorff_norm=hausdorff_norm,
        sbar_l=sbar_l, ubar_l=ubar_l, sbar_r=sbar_r, ubar_r=ubar_r,
        d_s=d_s, d_u=d_u, d_c=d_c, d_c_norm=d_c_norm, cov_dissimilarity=cov_d,
        s_l_max=s_l_max, s_r_max=s_r_max, d_s_max=d_s_max,
        d_w_max=d_w_max, d_w_max_rel=d_w_max_rel,
        rmse_kappa=rmse_kappa, n_lobes_l=n_l, n_lobes_r=n_r,
        lobe_count_diff=n_diff, n_unmatched_lobes=n_unmatched, mean_lobe_shift=shift,
    )
