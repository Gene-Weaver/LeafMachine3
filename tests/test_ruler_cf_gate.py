"""The ruler-CF gate's dissent rule: what may and may not withhold a sheet's CF.

The veto used to be unconditional -- ANY well-supported non-winning crop set confidence to "low".
Measured against 549 human-measured sheets, that was blocking a good CF on 19 sheets and a bad one
on 1. The anchor now overrules it. These tests pin both halves of that, because the failure mode is
silent in either direction: too strict and correct CFs vanish, too loose and wrong ones ship.
"""
from __future__ import annotations

import math

from leafmachine3.inference.ruler_lattice.sheet_cf import DISAGREEMENT_FLOOR, reconcile_parent
from leafmachine3.inference.ruler_lattice.units import RUNG_HALF_LOG


def _crop(key, cf, weight=40):
    return {"key": key, "cf": cf, "weight": weight, "ruler_class": "cm", "skipped": False}


def test_anchor_overrules_a_dissenting_crop() -> None:
    """Two crops agree and the anchor confirms them; a third disagrees. The CF still publishes."""
    crops = [_crop("a", 100.0), _crop("b", 101.0), _crop("c", 143.0)]
    out = reconcile_parent(crops, anchor=100.5, frame_width_px=3000)
    assert out["n_dissenting"] >= 1                       # the disagreement is still REPORTED
    assert out["confidence"] == "high" and out["cf_px_per_cm"] is not None
    assert any("OVERRULED by the anchor" in r for r in out["confidence_reasons"])


def test_dissent_still_vetoes_when_the_anchor_does_not_confirm() -> None:
    """Without anchor support the veto is unchanged -- two readings conflict and nothing arbitrates."""
    crops = [_crop("a", 100.0), _crop("b", 101.0), _crop("c", 143.0)]
    anchor = 100.0 * math.exp(2 * RUNG_HALF_LOG)          # winner is >half a rung from the anchor
    out = reconcile_parent(crops, anchor=anchor, frame_width_px=3000)
    assert out["n_dissenting"] >= 1
    assert out["cf_px_per_cm"] is None                    # withheld
    assert out["confidence"] in ("low", "medium")


def test_a_weak_outlier_was_never_dissent() -> None:
    """A crop under the evidence floor is noise, not a second opinion -- unchanged behavior."""
    crops = [_crop("a", 100.0), _crop("c", 143.0, weight=DISAGREEMENT_FLOOR - 1)]
    out = reconcile_parent(crops, anchor=100.5, frame_width_px=3000)
    assert out["n_dissenting"] == 0 and out["cf_px_per_cm"] is not None


def test_the_anchor_check_itself_is_not_weakened() -> None:
    """The DBG failure the module docstring warns about: two crops agreeing with each other and both
    2x wrong. The ANCHOR is what catches it, and loosening dissent must not touch that."""
    crops = [_crop("a", 200.0), _crop("b", 201.0)]        # agree to 0.5%, both 2x the truth
    out = reconcile_parent(crops, anchor=100.0, frame_width_px=3000)
    assert out["cf_px_per_cm"] is None                    # still withheld
    assert out["corroborated_by_peers"] and not out["anchor_supported"]


# --------------------------------------------------------------------------- #
# length-backed acceptance
# --------------------------------------------------------------------------- #
from leafmachine3.inference.ruler_lattice.sheet_cf import MIN_TRUSTED_RULER_CM


def _crop_len(key, cf, length, weight=40):
    return {"key": key, "cf": cf, "weight": weight, "ruler_class": "cm",
            "skipped": False, "implied_len_cm": length}


def test_a_long_ruler_outvotes_a_disagreeing_anchor() -> None:
    """The anchor is off by more than half a rung but the reading has 11 cm of ruler behind it.

    This is the SMF/MnhnL case: the anchor was wrong by 10-21% and the reading was within 2.5% of
    the human measurement, and the old gate withheld the sheet because the two disagreed.
    """
    anchor = 68.0
    cf = anchor * math.exp(1.6 * RUNG_HALF_LOG)          # beyond half a rung, inside one full rung
    out = reconcile_parent([_crop_len("a", cf, 11.0)], anchor=anchor, frame_width_px=3000)
    assert out["confidence"] == "high" and out["cf_px_per_cm"] is not None
    assert out["length_backed"] is True
    assert any("cm of ruler" in r for r in out["confidence_reasons"])


def test_a_short_ruler_does_not() -> None:
    """Same disagreement, 6 cm of ruler: defer to the anchor.

    The three genuine failures in the reference corpus were all rulers of 5.3-5.8 cm whose readings
    were 17-20% wrong. A short ruler read CORRECTLY agrees with the anchor and never gets here.
    """
    anchor = 68.0
    cf = anchor * math.exp(1.6 * RUNG_HALF_LOG)
    out = reconcile_parent([_crop_len("a", cf, 6.0)], anchor=anchor, frame_width_px=3000)
    assert out["cf_px_per_cm"] is None
    assert not out["length_backed"]


def test_length_never_admits_a_reading_beyond_one_rung() -> None:
    """A long ruler is not a licence to ignore the anchor entirely -- two rungs stays withheld,
    which is what keeps a clean 2x misnaming out however much ruler supports it."""
    anchor = 68.0
    out = reconcile_parent([_crop_len("a", anchor * 2.0, 25.0)], anchor=anchor, frame_width_px=3000)
    assert out["cf_px_per_cm"] is None and not out["length_backed"]


def test_length_backing_is_additive_only() -> None:
    """A sheet the anchor already agrees with is unaffected -- same verdict, whatever the length."""
    for length in (2.0, 30.0):
        out = reconcile_parent([_crop_len("a", 100.0, length)], anchor=100.5, frame_width_px=3000)
        assert out["confidence"] == "high" and out["cf_px_per_cm"] is not None


def test_a_dissenting_crop_still_vetoes_a_long_ruler() -> None:
    """Length backing is weaker evidence than anchor support, so it does not clear dissent."""
    anchor = 68.0
    cf = anchor * math.exp(1.6 * RUNG_HALF_LOG)
    crops = [_crop_len("a", cf, 11.0), _crop_len("b", cf * 1.4, 11.0)]
    out = reconcile_parent(crops, anchor=anchor, frame_width_px=3000)
    assert out["n_dissenting"] >= 1 and out["cf_px_per_cm"] is None


def test_threshold_is_the_documented_value() -> None:
    assert MIN_TRUSTED_RULER_CM == 9.0
