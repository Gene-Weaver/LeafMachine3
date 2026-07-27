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
