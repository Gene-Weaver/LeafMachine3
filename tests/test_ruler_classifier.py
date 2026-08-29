"""Tests for the Ruler Classifier: squarify preprocessing, per-specimen consensus, DB round-trip."""
from __future__ import annotations

import numpy as np

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import RulerClassRow, SpecimenRecord
from leafmachine3.inference.ruler_squarify import RulerSquarifier
from leafmachine3.modules.ruler_classifier import _specimen_ruler_class


def _row(unit_type: str) -> RulerClassRow:
    return RulerClassRow(detection_id=0, unit_type=unit_type, votes={}, conf=None)


def test_specimen_ruler_class_consensus() -> None:
    assert _specimen_ruler_class([_row("METRIC_MM"), _row("METRIC_MM"), _row("STD_IN8")]) == "METRIC_MM"
    assert _specimen_ruler_class([_row("UNKNOWN"), _row("METRIC_CM")]) == "METRIC_CM"   # UNKNOWN ignored
    assert _specimen_ruler_class([_row("UNKNOWN"), _row("UNKNOWN")]) is None
    assert _specimen_ruler_class([]) is None


def test_set_specimen_ruler_class_roundtrip(tmp_path) -> None:
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    sid = db.upsert_specimen(SpecimenRecord(
        image_name="a.jpg", image_stem="a", original_path="/o/a.jpg", working_path="/w/a.jpg"))
    db.set_specimen_ruler_class(sid, "METRIC_MM")
    assert db.get_specimen(sid)["ruler_class_type"] == "METRIC_MM"
    db.set_specimen_ruler_class(sid, None)
    assert db.get_specimen(sid)["ruler_class_type"] is None


def test_squarify_tile_four_shape_determinism_grayscale() -> None:
    """tile_four @ sz=720 -> 1440x1440; deterministic (augment=False); channels grayscale-equal
    (color input is normalized to grayscale, matching the grayscale crops the members trained on)."""
    sq = RulerSquarifier(sz=720, method="tile_four", augment=False)
    strip = np.zeros((80, 800, 3), np.uint8)         # a wide COLOR ruler strip (w/h = 10 > 4 -> stacks)
    strip[:, ::20] = (0, 0, 255)                     # red "ticks"
    out = sq.transform(strip)
    assert out.shape == (1440, 1440, 3)              # 2*sz square
    assert np.array_equal(out, sq.transform(strip))  # deterministic
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_squarify_makes_portrait_horizontal() -> None:
    """A PORTRAIT strip is rotated to landscape internally, so the collage is square regardless."""
    sq = RulerSquarifier(sz=256, method="tile_four", augment=False)
    out = sq.transform(np.full((800, 80, 3), 127, np.uint8))
    assert out.shape == (512, 512, 3)                # 2*sz, orientation-agnostic
