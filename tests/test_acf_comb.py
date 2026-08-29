"""Regression: recovering a tick fundamental that sits BELOW the ACF search floor.

``A_1989713254`` is a plain metric ruler -- 1 mm ticks, longer 1 cm ticks -- that the engine read at
94.82 px/cm when the truth is ~59.3. The cause is structural, not statistical:

    lattice.MIN_PERIOD = 7.0        # the ACF is only searched from lag 7 px upward

One millimetre on that scan is **5.93 px**, below the floor. The fundamental is not missed, it is
outside the search window, so no amount of sub-harmonic descent can return it. What the ACF does
show is a clean comb of its harmonics -- 12, 17, 23, 29, 35, ... -- spaced by ~5.9 px. The
fundamental is the SPACING, and never appears as a peak.

These tests pin that, using the crop's real ACF peak lags as a fixture so the case cannot silently
regress if the peak-picking or the floor is ever retuned.

The estimators under test live in ``leafmachine3.modules.experiments.ACF_comb`` and are NOT wired
into the pipeline. Measured on 896 crops of the 549 human-measured sheets they take mean error
2.05% -> 0.78% and eliminate all 8 catastrophic (>25%) readings with zero regressions from
"already correct" to "wrong", but that is an argument for adopting them, not evidence that they
are adopted.
"""
from __future__ import annotations

import numpy as np
import pytest

from leafmachine3.inference.ruler_lattice.lattice import MIN_PERIOD
from leafmachine3.modules.experiments.ACF_comb.comb_eval import (
    acf_peaks, fundamental_comb, fundamental_ladder, name_period,
)

#: The ACF peak lags actually produced for A_1989713254 det7 (METRIC_MM, 994 px deskewed strip).
A1989_PEAKS = [12, 17, 23, 29, 35, 40, 46, 52, 58, 63, 69, 75, 81, 87, 93, 98,
               104, 110, 116, 122, 128, 134, 140, 146, 152, 157, 163, 169, 175]
A1989_ANCHOR = 60.86          # working-frame sqrt MP anchor for that sheet
A1989_TRUTH = 59.26           # px/cm, from the ruler's own printed graduations
A1989_SHIPPED = 94.82         # what the engine published before -- 60% high


def test_the_fundamental_is_below_the_search_floor() -> None:
    """The premise. If this stops being true the whole case study is obsolete."""
    assert A1989_TRUTH / 10.0 < MIN_PERIOD, (
        "1 mm is no longer below MIN_PERIOD; this regression no longer reproduces"
    )


def test_comb_recovers_a_sub_floor_fundamental() -> None:
    """The peak SPACING carries the 1 mm period even though no peak sits at it."""
    P = fundamental_comb(A1989_PEAKS)
    assert P is not None
    assert P < MIN_PERIOD, "the recovered period must be below the floor, else nothing was fixed"
    assert P == pytest.approx(A1989_TRUTH / 10.0, rel=0.05)

    cf, unit = name_period(P, "METRIC_MM", A1989_ANCHOR)
    assert unit == "metric__MM"                       # 1 mm, which is what the ruler prints
    assert abs(cf / A1989_TRUTH - 1) < 0.05           # within 5% of the printed graduations
    # and it is a decisive improvement on what shipped
    assert abs(cf - A1989_TRUTH) < abs(A1989_SHIPPED - A1989_TRUTH) / 10


def test_ladder_declines_rather_than_guessing_on_this_crop() -> None:
    """"First credible peak is the finest graduation" CANNOT work when the fundamental is
    unobservable: peak 12 is the k=2 harmonic, so it does not divide peak 17 (k=3). Returning
    nothing is the correct behavior -- an answer here would be a fabricated one."""
    assert fundamental_ladder(A1989_PEAKS) is None


def test_ladder_does_find_a_fundamental_that_is_actually_present() -> None:
    """The same rule works when the fundamental IS above the floor and visible."""
    P = 11.0
    peaks = [int(round(k * P)) for k in range(1, 12)]
    assert fundamental_ladder(peaks) == pytest.approx(P, abs=1.0)


def test_comb_rejects_a_ragged_peak_set() -> None:
    """Unevenly spaced maxima are not a comb, and must not be fitted as one."""
    assert fundamental_comb([9, 14, 31, 33, 70, 96, 101]) is None
    assert fundamental_comb([12, 17]) is None                       # too few to be a comb


def test_peaks_are_found_on_a_synthetic_sub_floor_comb() -> None:
    """End to end from a profile: 5.5 px ticks (below the floor) are still recovered."""
    P = 5.5
    x = np.arange(1200)
    prof = np.zeros_like(x, dtype=float)
    for k in range(1, int(len(x) / P)):
        c = int(round(k * P))
        if 0 <= c < len(prof):
            prof[c] = 1.0
        if k % 10 == 0 and c < len(prof):                            # taller cm ticks
            prof[c] = 2.0
    peaks, _ = acf_peaks(prof)
    assert len(peaks) >= 4
    got = fundamental_comb(peaks)
    assert got is not None and got == pytest.approx(P, rel=0.05)
