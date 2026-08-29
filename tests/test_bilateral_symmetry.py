"""Unit tests for the bilateral-symmetry experiment's axis frame and metrics.

The headline case is the one that used to fail: an EXACTLY mirror-symmetric mask whose axis runs
down a pixel column, so a whole column of pixels lands on ``u == 0``. Those pixels belong to
neither half; assigning them all to one side made a perfectly symmetric leaf score as lopsided.
The rest of the file pins the ``nan``-not-``0.0`` rule for degenerate input, which is what keeps
"no data" from being read as "perfectly symmetric".
"""
from __future__ import annotations

import numpy as np

from leafmachine3.modules.experiments.bilateral_symmetry.axes import (
    Profiles,
    build_axis,
    profiles,
)
from leafmachine3.modules.experiments.bilateral_symmetry.metrics import compute_metrics
from leafmachine3.modules.experiments.bilateral_symmetry.shape import compute_shape

# A tall ellipse on an odd-width canvas: the tip/base column is an exact integer, so the chord axis
# lies ON pixel column CX and every pixel there gets u == 0 exactly.
_H, _W, _CY, _CX, _A, _B = 260, 201, 130, 100, 120, 70


def _ellipse(bl: float, br: float) -> np.ndarray:
    """Half-ellipse pair sharing one vertical axis: left semi-width ``bl``, right ``br``."""
    yy, xx = np.mgrid[0:_H, 0:_W]
    dy = (yy - _CY) / _A
    dx = np.where(xx <= _CX, (xx - _CX) / bl, (xx - _CX) / br)
    return (dy * dy + dx * dx) <= 1.0


def _frame(mask: np.ndarray, cx: float = _CX):
    return build_axis(mask, "chord",
                      tip=np.array([cx, _CY - _A], float),
                      base=np.array([cx, _CY + _A], float))


def _flat(al, ar, wl=None, wr=None) -> Profiles:
    al = np.asarray(al, float)
    ar = np.asarray(ar, float)
    n = al.size
    wl = np.zeros(n) if wl is None else np.asarray(wl, float)
    wr = np.zeros(n) if wr is None else np.asarray(wr, float)
    return Profiles(s=(np.arange(n) + 0.5) / n, area_l=al, area_r=ar, w_l=wl, w_r=wr,
                    ds_px=1.0, n_used=al + ar)


class _NullFrame:
    """Axis frame with no pixels, for exercising the profile-only metrics."""

    length = 100.0
    sinuosity = 1.0
    u = np.zeros(0)
    s = np.zeros(0)
    xy = np.zeros((0, 2))
    shape = (10, 10)

    def max_chord_deviation(self) -> float:
        return 0.0

    def integrated_curvature(self) -> float:
        return 0.0


# -- on-axis pixels -----------------------------------------------------------------
def test_mirror_symmetric_mask_is_exactly_symmetric():
    """A_star and d_c must be EXACTLY zero, not merely small, on a mirror-symmetric mask."""
    mask = _ellipse(_B, _B)
    frame = _frame(mask)
    prof = profiles(frame)

    assert np.count_nonzero(frame.u == 0) > 0, "test is pointless unless the axis hits pixels"
    assert compute_metrics(frame, prof).A_star == 0.0
    assert compute_shape(frame, prof).d_c == 0.0


def test_on_axis_pixels_preserve_total_area():
    """Splitting on-axis pixels half and half must not lose or invent a single pixel."""
    prof = profiles(_frame(_ellipse(_B, 55.0)))
    assert prof.area_l.sum() + prof.area_r.sum() == float(_ellipse(_B, 55.0).sum())


def test_mirroring_the_mask_negates_the_signed_measures_exactly():
    mask = _ellipse(_B, 55.0)
    m = compute_metrics(_frame(mask), profiles(_frame(mask)))
    fm = _frame(mask[:, ::-1], cx=_W - 1 - _CX)
    mm = compute_metrics(fm, profiles(fm))
    assert m.A_star == -mm.A_star
    assert compute_shape(_frame(mask), profiles(_frame(mask))).d_u == -compute_shape(fm, profiles(fm)).d_u


# -- "no data" is nan, never 0.0 ----------------------------------------------------
def test_empty_profile_yields_nan_not_zero():
    m = compute_metrics(_NullFrame(), _flat(np.zeros(8), np.zeros(8)))
    assert np.isnan(m.RMSE_A) and np.isnan(m.M_D) and np.isnan(m.V_D)


def test_area_ratio_is_mirror_consistent():
    left_only = compute_metrics(_NullFrame(), _flat([5.0, 9.0], [0.0, 0.0])).AR
    right_only = compute_metrics(_NullFrame(), _flat([0.0, 0.0], [5.0, 9.0])).AR
    assert np.isnan(left_only) and np.isnan(right_only)


def test_no_lamina_off_the_axis_yields_nan_margins():
    """A 1 px wide mask has no margin at all -- that is not perfect symmetry."""
    line = np.zeros((200, 61), bool)
    line[20:180, 30] = True
    frame = build_axis(line, "chord", tip=np.array([30.0, 20.0]), base=np.array([30.0, 179.0]))
    sh = compute_shape(frame, profiles(frame))
    for name in ("mad_boundary", "d_max", "hausdorff", "hausdorff_norm", "d_w_max", "rmse_kappa"):
        assert np.isnan(getattr(sh, name)), name


def test_s_at_d_max_is_nan_when_there_is_no_mismatch():
    n = 200
    same = compute_shape(_NullFrame(), _flat(np.ones(n), np.ones(n), np.full(n, 5.0), np.full(n, 5.0)))
    assert np.isnan(same.s_at_d_max)
    allnan = compute_shape(_NullFrame(), _flat(np.ones(n), np.ones(n),
                                               np.full(n, np.nan), np.full(n, np.nan)))
    assert np.isnan(allnan.s_at_d_max)


def test_hausdorff_is_nan_when_the_margin_is_nan_contaminated():
    n = 200
    wl = np.full(n, 10.0)
    wl[100] = np.nan
    sh = compute_shape(_NullFrame(), _flat(np.ones(n), np.ones(n), wl, np.full(n, 8.0)))
    assert np.isnan(sh.hausdorff) and np.isnan(sh.hausdorff_norm)


# -- guards and alignment -----------------------------------------------------------
def test_width_guard_drops_the_tip_slivers():
    """A symmetric taper with 1 px of tip noise must read as symmetric once the guard applies."""
    n = 200
    taper = 60.0 * np.sin(np.pi * (np.arange(n) + 0.5) / n)
    wl = taper.copy()
    wl[:3] += 1.0
    m = compute_metrics(_NullFrame(), _flat(np.ones(n), np.ones(n), wl, taper))
    assert m.n_width_bins_dropped > 0
    assert m.NRMSE_w == 0.0 and m.aw_mean_abs == 0.0


def test_cumulative_integral_has_no_half_bin_bias():
    """Constant drift gives C(s) = A_star * s, so I_C is exactly A_star / 2."""
    n = 200
    m = compute_metrics(_NullFrame(), _flat(np.full(n, 3.0), np.full(n, 1.0)))
    assert m.C_final == m.A_star
    assert abs(m.I_C - 0.25) < 1e-12


def test_interior_empty_bins_do_not_dilute_mad_boundary():
    n = 200
    al = np.full(n, 10.0)
    wl = np.full(n, 10.0)
    wr = np.full(n, 10.0)
    wr[80:120] = 4.0                      # a real 6 px gap over 40 bins
    ar = al.copy()
    al[40:60] = ar[40:60] = 0.0           # 20 INTERIOR empty bins
    wl[40:60] = wr[40:60] = 0.0
    sh = compute_shape(_NullFrame(), _flat(al, ar, wl, wr))
    assert abs(sh.mad_boundary - 6.0 * 40 / 180) < 1e-12
