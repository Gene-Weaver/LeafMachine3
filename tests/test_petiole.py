"""Unit tests for :mod:`leafmachine3.core.petiole` (petiole width from mask + centerline)."""
from __future__ import annotations

import numpy as np
import pytest

from leafmachine3.core.petiole import measure_petiole


def test_vertical_petiole_width_and_length():
    # a vertical bar 10 px wide (x 45..55), 100 px tall (y 10..110); a "blade" touches its top
    mask = np.zeros((120, 100), dtype=bool)
    mask[10:110, 45:55] = True
    leaf = np.zeros_like(mask)
    leaf[0:20, 40:60] = True
    centerline = [(50, 12), (50, 40), (50, 70), (50, 100)]   # junction (top) -> free end (bottom)
    pw = measure_petiole(mask, leaf, centerline)
    assert pw.width_px == pytest.approx(10, abs=2)           # perpendicular thickness of the bar
    assert pw.length_px == pytest.approx(88, abs=1)          # 12 -> 100 along the centerline
    assert pw.n_samples >= 1
    assert pw.touches_leaf is True
    assert pw.width_segment is not None and len(pw.sample_segments) == pw.n_samples
    assert pw.measure_location == "near_base"


def test_width_is_median_of_perpendicular_samples():
    # bar that steps from 8 px to 20 px wide partway down; near-base samples see the ~8 px part
    mask = np.zeros((140, 120), dtype=bool)
    mask[10:120, 56:64] = True          # 8 px wide near the junction
    mask[80:120, 40:80] = True          # widens to 40 px lower down (outside the near-base window)
    centerline = [(60, 12), (60, 60), (60, 110)]
    pw = measure_petiole(mask, None, centerline)
    assert pw.width_px == pytest.approx(8, abs=2)            # near-base sampling avoids the wide part


def test_no_petiole_mask_returns_none():
    pw = measure_petiole(np.zeros((50, 50), dtype=bool), None, [(10, 10), (10, 40)])
    assert pw.width_px is None and pw.length_px is None and pw.n_samples == 0


def test_no_centerline_records_touch_but_no_width():
    mask = np.zeros((60, 60), dtype=bool)
    mask[10:50, 28:32] = True
    leaf = np.zeros_like(mask)
    leaf[0:12, 20:40] = True
    pw = measure_petiole(mask, leaf, [(30, 20)])             # only one point -> no direction
    assert pw.touches_leaf is True
    assert pw.width_px is None and pw.n_samples == 0


# --------------------------------------------------------------------------- #
# Leaf mass per area (Royer petiole-width scaling)
#   log10(LMA) = 3.070 + 0.382 * log10(PW^2 / A),  LMA in g/m^2
# --------------------------------------------------------------------------- #
import math

from leafmachine3.modules.petiole_width import LMA_A, LMA_B, _area_by_leaf, _lma


def test_lma_matches_the_published_formula():
    pw, area = 7.5, 70312.5
    assert _lma(pw, area) == pytest.approx(10 ** (LMA_A + LMA_B * math.log10(pw * pw / area)))


def test_lma_is_scale_invariant_so_no_ruler_cf_is_needed():
    """PW^2/A is a ratio of areas, so px, mm and cm must all give the same LMA."""
    cf = 37.5                                     # px per cm
    px = _lma(7.5, 70312.5)
    cm = _lma(7.5 / cf, 70312.5 / cf ** 2)
    mm = _lma(7.5 / cf * 10, 70312.5 / cf ** 2 * 100)
    assert px == pytest.approx(cm) == pytest.approx(mm)


def test_lma_lands_in_a_biologically_plausible_range():
    """A ~2 mm petiole on a ~50 cm^2 lamina is a real leaf; guards against a log-base slip.

    Natural logs here would yield ~1.4 g/m^2 instead of ~77, which is why the base matters.
    """
    lma = _lma(2.0, 5000.0)                       # mm and mm^2
    assert 20.0 < lma < 200.0


def test_lma_grows_with_petiole_width_and_falls_with_leaf_area():
    assert _lma(4.0, 5000.0) > _lma(2.0, 5000.0)
    assert _lma(2.0, 9000.0) < _lma(2.0, 5000.0)


@pytest.mark.parametrize("pw,area", [(None, 5000.0), (2.0, None), (0.0, 5000.0),
                                     (2.0, 0.0), (-1.0, 5000.0), (2.0, -5.0)])
def test_lma_is_none_when_not_computable(pw, area):
    """A missing petiole or a degenerate mask must yield NULL, never a crash or a nan."""
    assert _lma(pw, area) is None


def test_area_by_leaf_skips_rows_without_a_usable_area():
    rows = [{"leaf_id": 1, "area_px": 1200.0}, {"leaf_id": 2, "area_px": None},
            {"leaf_id": 3, "area_px": 0.0}, {"area_px": 500.0}]
    assert _area_by_leaf(rows) == {1: 1200.0}
    assert _area_by_leaf([]) == {} and _area_by_leaf(None) == {}
