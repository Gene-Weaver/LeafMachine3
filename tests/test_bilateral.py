"""core.bilateral -- the invariants that make the symmetry numbers mean anything.

Each of these corresponds to a failure that was demonstrated with running code: getting any of them
wrong yields plausible numbers rather than an exception, which is why they are pinned here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from leafmachine3.core import bilateral as B


def _sym_mask(h=240, w=121, rx=42.0, ry=110.0):
    yy, xx = np.mgrid[0:h, 0:w]
    cx = (w - 1) / 2.0
    return (((xx - cx) / rx) ** 2 + ((yy - h / 2) / ry) ** 2) <= 1.0, cx


def _midvein(mask, cx, n=17):
    ys = mask.nonzero()[0]
    return np.column_stack([np.full(n, cx), np.linspace(ys.min(), ys.max(), n)])


def _frame(mask, mv):
    return B.build_axis(mask, "midvein", midvein=mv, tip=mv[0], base=mv[-1])


def _si_astar(frame, n_bins=200):
    p = B.profiles(frame, n_bins=n_bins)
    a = B.area_asymmetry_profile(p)
    v = np.isfinite(a)
    si = float(np.abs(a[v]).mean()) if v.sum() else float("nan")
    return si, B._div(p.area_l.sum() - p.area_r.sum(), p.area_l.sum() + p.area_r.sum()), p


def test_voronoi_partition_tiles_the_lamina_exactly_once():
    """area_l + area_r must equal the mask area EXACTLY -- perpendicular strips would not."""
    mask, cx = _sym_mask()
    _si, _a, p = _si_astar(_frame(mask, _midvein(mask, cx)))
    assert p.area_l.sum() + p.area_r.sum() == float(mask.sum())


def test_mirror_symmetric_mask_has_exactly_zero_signed_imbalance():
    """a_star must be EXACTLY 0.0, not merely small.

    This is what the on-axis tolerance buys. The midvein path comes out of a smoothing spline that
    leaves a ~5e-14 px wobble; testing ``u == 0`` exactly sent geometrically on-axis pixels left or
    right by the sign of that noise and produced a_star = -0.0025 on a perfectly symmetric mask.
    """
    mask, cx = _sym_mask()
    _si, a_star, _p = _si_astar(_frame(mask, _midvein(mask, cx)))
    assert a_star == 0.0


def test_mirroring_negates_a_star_and_preserves_si_a():
    mask, cx = _sym_mask()
    asym = mask.copy()
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    narrow = (((xx - cx) / 34.0) ** 2 + ((yy - mask.shape[0] / 2) / 110.0) ** 2) <= 1.0
    asym[:, : int(cx)] &= narrow[:, : int(cx)]
    mv = _midvein(mask, cx)
    si1, a1, _ = _si_astar(_frame(asym, mv))
    si2, a2, _ = _si_astar(_frame(asym[:, ::-1], mv))
    assert a1 != 0.0                                   # the fixture must actually be asymmetric
    # a_star is EXACT under mirroring: it is a total, so it does not care which bin a pixel fell in.
    assert a1 == pytest.approx(-a2, abs=1e-9)
    # si_a is a per-BIN average, so it is only invariant to within a bin reassignment: the midvein
    # spline is not perfectly symmetric (~5e-14 px), which can move one pixel across a bin edge.
    # Measured drift is ~2e-4; anything larger would mean the halves are genuinely being swapped.
    assert si1 == pytest.approx(si2, abs=1e-3)


def test_on_axis_pixels_split_half_to_each_side():
    """A pixel the axis runs through belongs to neither half; splitting it keeps the sum exact."""
    mask, cx = _sym_mask()
    frame = _frame(mask, _midvein(mask, cx))
    p = B.profiles(frame, n_bins=200)
    assert p.area_l.sum() == pytest.approx(p.area_r.sum())
    assert (p.area_l % 1 != 0).any(), "a half-pixel share should appear somewhere"


def test_degenerate_input_is_nan_never_zero():
    """A 1-px mask must not read as 'perfectly symmetric'."""
    tiny = np.zeros((5, 5), bool)
    tiny[2, 2] = True
    mv = np.array([[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]])
    assert B.build_axis(tiny, "midvein", midvein=mv, tip=mv[0], base=mv[-1]) is None
    assert B.build_axis(np.zeros((5, 5), bool), "midvein", midvein=mv, tip=mv[0], base=mv[-1]) is None


def test_missing_symmetry_never_outranks_a_measured_leaf():
    """archetype_score once returned 1.0 for a leaf with no symmetry measurement at all."""
    q = B.QualityDiagnostics(
        specimen_id=1, leaf_id=1, detection_id=1, n_components=1, largest_frac=1.0, solidity=0.97,
        border_contact_frac=0.01, flush_edge_frac=0.05, truncated=False, perimeter_ratio=1.02,
        boundary_roughness=0.01, hole_frac=0.0, kpt_conf_mean=0.9, kpt_conf_min=0.8,
        n_midvein_kpts=17, area_px=70000.0, aspect=0.4, axis_aspect=0.4)
    score, _reasons = B.archetype_score({}, q)
    assert math.isnan(score)


def test_gates_veto_a_truncated_or_fragmented_leaf():
    base = dict(specimen_id=1, leaf_id=1, detection_id=1, n_components=1, largest_frac=1.0,
                solidity=0.97, border_contact_frac=0.01, flush_edge_frac=0.05, truncated=False,
                perimeter_ratio=1.02, boundary_roughness=0.01, hole_frac=0.0, kpt_conf_mean=0.9,
                kpt_conf_min=0.8, n_midvein_kpts=17, area_px=70000.0, aspect=0.4, axis_aspect=0.4)
    assert B.gates_pass(B.QualityDiagnostics(**base))[0] is True
    assert B.gates_pass(B.QualityDiagnostics(**{**base, "truncated": True}))[0] is False
    assert B.gates_pass(B.QualityDiagnostics(**{**base, "largest_frac": 0.90}))[0] is False
    assert B.gates_pass(B.QualityDiagnostics(**{**base, "n_midvein_kpts": 4}))[0] is False


def test_bins_adapt_to_leaf_size():
    """A fixed bin count made si_a a leaf-SIZE proxy (Spearman +0.84 against 1/sqrt(area))."""
    assert B.bins_for(5_000) < B.bins_for(50_000) < B.bins_for(200_000)
    assert B.bins_for(200_000, 200) == 200                  # n_bins is an upper bound
    assert B.bins_for(0) == B.MIN_BINS and B.bins_for(float("nan")) == B.MIN_BINS
    for area in (2_000, 20_000, 90_000):
        assert area / B.bins_for(area) >= B.MIN_PX_PER_BIN or B.bins_for(area) == B.MIN_BINS


def test_qc_filename_round_trips_and_carries_no_index():
    """Crops are identified by their canonical filename, never by a '#N' display key."""
    from leafmachine3.core.config import Config
    from leafmachine3.core.imaging import parse_crop_filename
    from leafmachine3.modules.bilateral_symmetry import _qc_name, QC_ALL

    cfg = Config({"naming": {"bilateral_prefix": "BSYM", "friendly_names": {"Leaf": "leaf"}}})
    leaf = type("L", (), {"stem": "ASC_1_Fabaceae_Prosopis", "det_box": (10, 20, 300, 400)})()
    m = type("M", (), {"gates_pass": True, "is_archetypal": True})()
    name = _qc_name(cfg, leaf, m, QC_ALL)
    assert name.endswith(".jpg") and "#" not in name
    parsed = parse_crop_filename(name)
    assert parsed["stem"] == "ASC_1_Fabaceae_Prosopis" and parsed["prefix"] == "BSYM"
    assert parsed["xyxy"] == (10, 20, 300, 400)
