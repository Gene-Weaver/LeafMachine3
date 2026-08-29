"""Scalar bilateral-symmetry metrics computed from a :class:`Profiles` in an :class:`AxisFrame`.

Three families, deliberately kept side by side because they answer different questions and the
literature does not agree on one number:

* **Area, Shi et al. (2018) Symmetry 10:118** -- ``AR`` (total left/right area ratio), ``RMSE_A``
  (unstandardized, so it scales with leaf size and cannot be pooled across a cohort) and ``SI_A``,
  the paper's headline standardized index.
* **Signed / bounded** -- ``SI_A`` throws away direction, so a leaf that is fat on the left near the
  tip and fat on the right near the base scores the same as a uniformly lopsided one. ``A_star``,
  ``DSI_A``, ``WSI_A`` and ``WDSI_A`` keep the sign and/or weight bins by how much lamina they
  actually contain. ``WSI_A`` is a LOWER BOUND on the symmetric-difference area over the total area:
  it sums ``|A_L - A_R|`` per strip, which collapses the ``u`` distribution inside each strip, so it
  equals the true symmetric difference only when the narrower half of every strip nests entirely
  inside the wider one. Two halves of equal area but different shape within a strip contribute 0
  here and a positive amount to the true symmetric difference. Measuring that exactly needs the
  two-dimensional overlap of the straightened halves, which is what ``shape.sd`` (``1 - Dice``) does.
* **Width, Wang et al. (2018) Forests 9:500** -- margin envelope agreement, plus the Taylor
  power-law inputs (``M_D``, ``V_D``) that are fitted at the COHORT level, not here.

DENOMINATOR GUARD. Bins at the very tip and very base hold a handful of pixels, so the standardized
ratios ``(A_L - A_R) / (A_L + A_R)`` there are one-pixel noise amplified to +/-1 and would dominate
any mean. Two guards drop such bins, one per profile, because area and margin width are not on the
same scale:

* AREA, ``EPS_FRAC`` of the summed bin area -- covers ``SI_A``, ``DSI_A``, the ``aA_*`` summaries and
  ``M_d``/``V_d``; survivors counted in ``n_bins_used`` / ``n_bins_dropped``.
* WIDTH, ``EPS_W_FRAC`` of the WIDEST bin's total width -- covers ``NRMSE_w`` and the ``aw_*``
  summaries; survivors counted in ``n_width_bins_used`` / ``n_width_bins_dropped``. Widths are
  envelopes (a max, not a sum), so a fraction of the summed width would not mean anything; the
  constant is set so the two guards drop a comparable share of bins on a real cohort.

The totals -- ``AR``, ``A_star``, ``WSI_A``, ``WDSI_A``, ``RMSE_A``, the cumulative curve, ``M_D`` /
``V_D`` -- have no per-bin denominator and use ALL bins, which is also why they are the trustworthy
numbers to compare across leaves. ``MAE_w``, ``RMSE_w`` and ``r_w`` are unstandardized for the same
reason and use every OCCUPIED bin, so their sample is not ``n_width_bins_used``.

Orientation: ``s = 0`` is the TIP and ``s = 1`` the BASE (see axes.py), so the apex third is
``s in [0, 1/3]`` and the base third is ``s in [2/3, 1]``. Positive values mean the viewer's LEFT
side is larger.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

from .axes import AxisFrame, Profiles

# a bin whose two sides together hold less than this fraction of the lamina is too small for its
# ratio to mean anything (see module docstring)
EPS_FRAC = 1e-3

# the same idea for the margin envelope: a bin spanning less than this fraction of the WIDEST bin's
# total width is a tip/base sliver a couple of pixels tall. Calibrated against EPS_FRAC on the
# 105-leaf test cohort: the area guard drops 3.9% of occupied bins, this one 2.6%.
EPS_W_FRAC = 0.1

# |asymmetry| above which a bin counts as "visibly" asymmetric, for the frac_gt_10pct summaries
ASYM_THRESH = 0.1

_NAN = float("nan")


@dataclass(frozen=True)
class SymmetryMetrics:
    """Every scalar this experiment reports for ONE leaf measured against ONE axis."""

    # --- area, Shi et al. -------------------------------------------------------------------
    AR: float          # sum(A_L) / sum(A_R); 1.0 = balanced, unbounded above; nan if a side is empty
    RMSE_A: float      # px^2, unstandardized -- comparable only between leaves of equal size
    SI_A: float        # mean per-bin |imbalance|, 0 = perfect, 1 = every bin one-sided

    # --- signed / bounded ------------------------------------------------------------------
    A_star: float      # signed total imbalance in [-1, 1], + = left larger
    DSI_A: float       # mean signed per-bin imbalance (bins weighted equally)
    WSI_A: float       # area-weighted unsigned; LOWER BOUND on symmetric-difference area / total
    WDSI_A: float      # area-weighted signed; algebraically identical to A_star

    # --- width, Wang et al. ----------------------------------------------------------------
    MAE_w: float
    RMSE_w: float
    NRMSE_w: float
    r_w: float         # Pearson r of the two margin envelopes

    # --- area asymmetry profile a_A(s) summaries -------------------------------------------
    aA_mean_abs: float
    aA_rms: float
    aA_max_abs: float
    aA_s_at_max: float
    aA_median_abs: float
    aA_q90_abs: float
    aA_frac_gt_10pct: float
    aA_SI_apex: float  # s in [0, 1/3]
    aA_SI_mid: float   # s in [1/3, 2/3]
    aA_SI_base: float  # s in [2/3, 1]

    # --- width asymmetry profile a_w(s) summaries ------------------------------------------
    aw_mean_abs: float
    aw_rms: float
    aw_max_abs: float
    aw_s_at_max: float
    aw_median_abs: float
    aw_q90_abs: float
    aw_frac_gt_10pct: float
    aw_SI_apex: float
    aw_SI_mid: float
    aw_SI_base: float

    # --- cumulative imbalance C(s) ---------------------------------------------------------
    C_final: float     # == A_star
    C_max: float       # largest running imbalance anywhere along the axis
    I_C: float         # integral of |C(s)| ds -- persistent drift scores high, a crossover low

    # --- Taylor power-law inputs (fitted across a cohort, not here) -------------------------
    M_D: float         # mean per-bin |area difference|, px^2
    V_D: float         # its sample variance (ddof=1), px^4
    M_d: float         # size-normalized variant, dimensionless
    V_d: float

    # --- axis geometry ---------------------------------------------------------------------
    sinuosity: float
    max_chord_deviation: float
    integrated_curvature: float

    # --- scale and bookkeeping -------------------------------------------------------------
    lamina_area_px: float
    axis_length_px: float
    n_bins: int
    n_bins_used: int
    n_bins_dropped: int
    n_width_bins_used: int      # survivors of the WIDTH guard, not the sample of MAE_w/RMSE_w/r_w
    n_width_bins_dropped: int

    def as_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _div(num: float, den: float) -> float:
    """Division that yields nan instead of raising or returning a meaningless 0.0."""
    return float(num) / float(den) if den else _NAN


def area_asymmetry_profile(prof: Profiles, *, eps_frac: float = EPS_FRAC) -> np.ndarray:
    """``a_A(s) = (A_Li - A_Ri) / (A_Li + A_Ri)``, nan in bins below the area guard."""
    al = np.asarray(prof.area_l, float)
    ar = np.asarray(prof.area_r, float)
    tot = al + ar
    keep = tot >= eps_frac * float(tot.sum())
    out = np.full(al.shape, _NAN)
    np.divide(al - ar, tot, out=out, where=keep & (tot > 0))
    return out


def _width_guard(prof: Profiles, eps_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """``(total width per bin, keep mask)`` for the margin-width denominator guard.

    Widths are envelopes, so unlike areas they do not sum to anything meaningful along the axis; the
    guard is therefore relative to the WIDEST bin. Without it the tip and base slivers, where
    ``w_L + w_R`` is a couple of pixels, drive ``|a_w|`` to ~0.6 while a full-width bin sits near
    0.14, and they dominate every ``aw_*`` summary and ``NRMSE_w``.
    """
    tot = np.asarray(prof.w_l, float) + np.asarray(prof.w_r, float)
    mx = float(tot.max()) if tot.size else 0.0
    return tot, (tot > 0) & (tot >= eps_frac * mx)


def width_asymmetry_profile(prof: Profiles, *, eps_frac: float = EPS_W_FRAC) -> np.ndarray:
    """``a_w(s) = (w_L - w_R) / (w_L + w_R)``, nan in bins below the width guard."""
    wl = np.asarray(prof.w_l, float)
    wr = np.asarray(prof.w_r, float)
    tot, keep = _width_guard(prof, eps_frac)
    out = np.full(wl.shape, _NAN)
    np.divide(wl - wr, tot, out=out, where=keep)
    return out


def cumulative_imbalance(prof: Profiles) -> np.ndarray:
    """Signed running imbalance ``C(s_k) = cumsum(A_L - A_R)[k] / (A_L + A_R)``.

    Reading the curve tip -> base separates a leaf that is lopsided the same way along its whole
    length (monotone drift) from one that trades sides (a crossover), which the scalar ``A_star``
    cannot tell apart.

    A cumulative sum is the value at each bin's RIGHT EDGE, not at its center: ``C[k]`` is the
    imbalance accumulated over everything from the tip through the end of bin ``k``. Use
    :func:`cumulative_s` for the matching abscissa -- ``prof.s`` (bin centers) is half a bin too
    small for every point.
    """
    al = np.asarray(prof.area_l, float)
    ar = np.asarray(prof.area_r, float)
    total = float(al.sum() + ar.sum())
    if total <= 0:
        return np.full(al.shape, _NAN)
    return np.cumsum(al - ar) / total


def cumulative_s(prof: Profiles) -> np.ndarray:
    """Bin RIGHT EDGES: the abscissa :func:`cumulative_imbalance` is actually sampled at.

    The last edge is exactly ``s = 1``, which is why ``C_final`` equals ``A_star`` on the nose.
    """
    s = np.asarray(prof.s, float)
    half = 0.5 / s.size if s.size else 0.0
    return s + half


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, nan for fewer than 3 points or a constant input."""
    if len(x) < 3:
        return _NAN
    xs, ys = x - x.mean(), y - y.mean()
    den = float(np.sqrt((xs * xs).sum() * (ys * ys).sum()))
    return _div(float((xs * ys).sum()), den)


def _profile_summary(a: np.ndarray, s: np.ndarray, prefix: str) -> dict[str, float]:
    """Shape summaries of an asymmetry profile, ignoring guarded-out (nan) bins."""
    keys = ("mean_abs", "rms", "max_abs", "s_at_max", "median_abs", "q90_abs",
            "frac_gt_10pct", "SI_apex", "SI_mid", "SI_base")
    out = {f"{prefix}_{k}": _NAN for k in keys}
    valid = np.isfinite(a)
    if not valid.any():
        return out

    aa = np.abs(a[valid])
    ss = np.asarray(s, float)[valid]
    k = int(np.argmax(aa))
    out[f"{prefix}_mean_abs"] = float(aa.mean())
    out[f"{prefix}_rms"] = float(np.sqrt(np.mean(aa * aa)))
    out[f"{prefix}_max_abs"] = float(aa[k])
    out[f"{prefix}_s_at_max"] = float(ss[k])
    out[f"{prefix}_median_abs"] = float(np.median(aa))
    out[f"{prefix}_q90_abs"] = float(np.quantile(aa, 0.9))
    out[f"{prefix}_frac_gt_10pct"] = float(np.mean(aa > ASYM_THRESH))

    # s = 0 is the TIP, so the FIRST third is the apex and the LAST third the base
    for key, lo, hi in (("SI_apex", 0.0, 1.0 / 3.0), ("SI_mid", 1.0 / 3.0, 2.0 / 3.0),
                        ("SI_base", 2.0 / 3.0, 1.0)):
        upper = ss <= hi if hi >= 1.0 else ss < hi   # last third keeps s == 1
        sel = (ss >= lo) & upper
        if sel.any():
            out[f"{prefix}_{key}"] = float(aa[sel].mean())
    return out


def _integrate_abs(c: np.ndarray, s: np.ndarray) -> float:
    """Integral of |C(s)| over the FULL s in [0, 1].

    ``s`` must be the bin RIGHT EDGES the cumulative sum is really sampled at (see
    :func:`cumulative_s`); the last of them already sits at s = 1, so only the tip end has to be
    pinned, at C = 0 because nothing has accumulated there yet. Integrating against bin CENTERS
    instead shifts the whole curve half a bin toward the tip and inflates the result -- on a
    constant-drift profile with 200 bins it returns 0.251247 for an exact 0.25.
    """
    ok = np.isfinite(c)
    if not ok.any():
        return _NAN
    cc, ss = np.abs(c[ok]), np.asarray(s, float)[ok]
    x = np.concatenate([[0.0], ss])
    y = np.concatenate([[0.0], cc])
    return float(np.trapezoid(y, x))


def compute_metrics(frame: AxisFrame, prof: Profiles) -> SymmetryMetrics:
    """All scalars for one leaf/axis pair. Degenerate input yields nan fields, never an exception."""
    al = np.asarray(prof.area_l, float)
    ar = np.asarray(prof.area_r, float)
    wl = np.asarray(prof.w_l, float)
    wr = np.asarray(prof.w_r, float)
    n_bins = int(al.size)

    d_area = al - ar
    t_area = al + ar
    sum_l, sum_r = float(al.sum()), float(ar.sum())
    total = sum_l + sum_r

    a_prof = area_asymmetry_profile(prof)
    valid_a = np.isfinite(a_prof)
    n_used = int(valid_a.sum())

    # --- area ------------------------------------------------------------------------------
    # AR is nan whenever EITHER side is empty, not just the denominator: area_l=[5,9]/area_r=[0,0]
    # and its exact mirror must not report nan one way round and 0.0 the other.
    AR = _div(sum_l, sum_r) if sum_l > 0 else _NAN
    # a profile with no lamina at all has nothing to be symmetric about, so nan rather than the 0.0
    # that would read as "perfectly symmetric" (matches SI_A / AR on the same input)
    RMSE_A = float(np.sqrt(np.mean(d_area * d_area))) if (n_bins and total > 0) else _NAN
    SI_A = float(np.abs(a_prof[valid_a]).mean()) if n_used else _NAN
    A_star = _div(sum_l - sum_r, total)
    DSI_A = float(a_prof[valid_a].mean()) if n_used else _NAN
    WSI_A = _div(float(np.abs(d_area).sum()), float(t_area.sum()))
    WDSI_A = _div(float(d_area.sum()), float(t_area.sum()))

    # --- width -----------------------------------------------------------------------------
    # MAE_w / RMSE_w / r_w carry no per-bin denominator, so like RMSE_A they use every OCCUPIED bin;
    # only the standardized NRMSE_w (and the a_w profile) needs the guard.
    t_width, keep_w = _width_guard(prof, EPS_W_FRAC)
    occ_w = t_width > 0
    n_w = int(keep_w.sum())
    if occ_w.any():
        dw = wl[occ_w] - wr[occ_w]
        MAE_w = float(np.abs(dw).mean())
        RMSE_w = float(np.sqrt(np.mean(dw * dw)))
        r_w = _pearson(wl[occ_w], wr[occ_w])
    else:
        MAE_w = RMSE_w = r_w = _NAN
    if n_w:
        dwg = wl[keep_w] - wr[keep_w]
        NRMSE_w = float(np.sqrt(np.mean((dwg / t_width[keep_w]) ** 2)))
    else:
        NRMSE_w = _NAN

    # --- cumulative ------------------------------------------------------------------------
    c = cumulative_imbalance(prof)
    finite_c = c[np.isfinite(c)]
    C_final = float(finite_c[-1]) if finite_c.size else _NAN
    C_max = float(np.abs(finite_c).max()) if finite_c.size else _NAN
    I_C = _integrate_abs(c, cumulative_s(prof))

    # --- Taylor ----------------------------------------------------------------------------
    D = np.abs(d_area)
    M_D = float(D.mean()) if (n_bins and total > 0) else _NAN
    V_D = float(D.var(ddof=1)) if (n_bins > 1 and total > 0) else _NAN
    d_norm = np.abs(a_prof[valid_a])
    M_d = float(d_norm.mean()) if n_used else _NAN
    V_d = float(d_norm.var(ddof=1)) if n_used > 1 else _NAN

    summaries = {**_profile_summary(a_prof, prof.s, "aA"),
                 **_profile_summary(width_asymmetry_profile(prof), prof.s, "aw")}

    return SymmetryMetrics(
        AR=AR, RMSE_A=RMSE_A, SI_A=SI_A,
        A_star=A_star, DSI_A=DSI_A, WSI_A=WSI_A, WDSI_A=WDSI_A,
        MAE_w=MAE_w, RMSE_w=RMSE_w, NRMSE_w=NRMSE_w, r_w=r_w,
        **summaries,
        C_final=C_final, C_max=C_max, I_C=I_C,
        M_D=M_D, V_D=V_D, M_d=M_d, V_d=V_d,
        sinuosity=float(frame.sinuosity),
        max_chord_deviation=float(frame.max_chord_deviation()),
        integrated_curvature=float(frame.integrated_curvature()),
        lamina_area_px=float(total),
        axis_length_px=float(frame.length),
        n_bins=n_bins, n_bins_used=n_used, n_bins_dropped=n_bins - n_used,
        n_width_bins_used=n_w, n_width_bins_dropped=n_bins - n_w,
    )
