"""Tests for the SpecimenSegmenter stage: paperclean, mock backend, DB round-trip, overlay."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.paper_removal import paperclean, remove_paper
from leafmachine3.core.records import SpecimenMaskResult, SpecimenRecord
from leafmachine3.inference.mock import MockSpecimenSegmenter
from leafmachine3.reporting.overlay import build_specimen_overlay
from leafmachine3.reporting.palette import SpecimenStyle


@pytest.fixture
def db(tmp_path: Path) -> ProjectDB:
    return ProjectDB.open_or_create(tmp_path / "proj.sqlite")


def _sheet_with_overshooting_mask():
    """A pale-paper sheet with a green blob; the mask overshoots the blob into the paper."""
    sheet = np.full((200, 200, 3), 245, np.uint8)
    cv2.circle(sheet, (100, 100), 55, (40, 120, 40), -1)   # green plant blob
    mask = np.zeros((200, 200), np.uint8)
    cv2.circle(mask, (100, 100), 72, 1, -1)                # mask spills onto paper
    return sheet, mask


def test_paperclean_removes_paper_and_returns_centers() -> None:
    sheet, mask = _sheet_with_overshooting_mask()
    final, removed, centers = paperclean(sheet, mask)
    assert final.shape == mask.shape and final.dtype == np.uint8
    assert int(final.sum()) < int(mask.sum())              # paper trimmed away
    assert int(removed.sum()) > 0                          # something was refined
    # removed == mask AND NOT final (the red region), disjoint from the final mask
    assert not np.any((removed > 0) & (final > 0))
    assert len(centers) >= 3                               # blue sampling boxes captured
    # remove_paper is the mask-only convenience wrapper and agrees with paperclean's final
    assert np.array_equal(remove_paper(sheet, mask), final)


def test_paperclean_full_frame_mask_is_noop() -> None:
    sheet = np.full((80, 80, 3), 245, np.uint8)
    mask = np.ones((80, 80), np.uint8)                     # nothing outside -> cannot sample paper
    final, removed, centers = paperclean(sheet, mask)
    assert np.array_equal(final, mask)
    assert int(removed.sum()) == 0 and centers == []


def test_mock_backend_shape_and_determinism() -> None:
    img = np.full((400, 300, 3), 200, np.uint8)
    a = MockSpecimenSegmenter().predict(img)
    b = MockSpecimenSegmenter().predict(img)
    assert isinstance(a, SpecimenMaskResult)
    assert a.frame_width == 300 and a.frame_height == 400
    assert 0.0 < a.area_frac < 1.0 and len(a.centers) == 4
    assert a.final_png == b.final_png                      # deterministic


def test_record_specimen_mask_round_trip(db, tmp_path) -> None:
    sid = db.upsert_specimen(SpecimenRecord(
        image_name="a.jpg", image_stem="a", original_path="/orig/a.jpg",
        working_path="/work/a.jpg", width=300, height=400,
    ))
    res = MockSpecimenSegmenter().predict(np.full((400, 300, 3), 200, np.uint8))
    dirs = SimpleNamespace(masks=tmp_path / "_specimen_masks")
    db.record_specimen_mask(sid, "a", res, dirs)

    row = db.specimen_mask(sid)
    assert row is not None
    assert row["frame_width"] == 300 and row["frame_height"] == 400
    assert row["model_name"] == "mock"
    # both PNGs were written to disk and are readable binary masks
    final = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    assert final is not None and final.shape == (400, 300)
    import json
    assert len(json.loads(row["sample_centers_json"])) == 4
    # upsert: a second write for the same specimen replaces (not duplicates) the row
    db.record_specimen_mask(sid, "a", res, dirs)
    assert len(db._query("SELECT 1 FROM specimen_mask WHERE specimen_id = ?", (sid,))) == 1


def test_config_drift_reruns_dependents(mock_config_path, tmp_path) -> None:
    """Swapping the specimen model (config drift) must reset the drifted stage AND its transitive
    dependents (e.g. the Reporter that renders the specimen overlay), not just the stage itself."""
    from leafmachine3.core.config import Config
    from leafmachine3.pipeline import _sync_settings_hash, build_pipeline

    cfg = Config.load(mock_config_path)
    db = ProjectDB.open_or_create(tmp_path / "drift.sqlite")
    stages = build_pipeline(cfg)
    for key in ("specimen_segmenter", "reporter"):        # both complete + in sync
        db.mark_stage_done(key)
        db.set_stage_settings_hash(key, cfg.stage_settings_hash(key))
    db.set_stage_settings_hash("specimen_segmenter", "STALE")   # simulate a model/settings change

    ss = next(s for s in stages if s.key == "specimen_segmenter")
    _sync_settings_hash(SimpleNamespace(db=db), ss, cfg, stages)

    assert db.stage_state("specimen_segmenter") == "pending"    # the drifted stage reset...
    assert db.stage_state("reporter") == "pending"              # ...and its dependent cascaded


def test_build_specimen_overlay_two_panel_semantics() -> None:
    img = np.full((120, 160, 3), 210, np.uint8)
    final = np.zeros((120, 160), np.uint8)
    cv2.rectangle(final, (40, 30), (120, 90), 255, -1)     # central mask
    removed = np.zeros_like(final)
    style = SpecimenStyle(display_max_dim=0)               # full-res, no fit
    out = build_specimen_overlay(img, final, removed, [(20, 60), (140, 60)], style)

    assert out.shape == (120, 320, 3)                      # two panels side by side
    left, right = out[:, :160], out[:, 160:]
    bg = np.array([210, 210, 210], np.uint8)
    # LEFT: interior of the mask is tinted (green fill) -> differs from the flat background
    assert not np.array_equal(left[60, 80], bg)
    # RIGHT: cutout keeps pixels inside the mask, black outside
    assert np.array_equal(right[60, 80], img[60, 80])      # inside mask -> original pixel
    assert np.array_equal(right[5, 5], np.array([0, 0, 0], np.uint8))   # outside -> black
