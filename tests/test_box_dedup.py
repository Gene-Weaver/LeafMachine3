"""Tests for same-class duplicate-box suppression (core.box_dedup) + its DB/stage wiring."""
from __future__ import annotations

from dataclasses import dataclass

from leafmachine3.core.box_dedup import box_overlap, suppress_duplicate_boxes
from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import DetRow, SpecimenRecord


@dataclass
class _Box:
    cls_id: int
    conf: float
    xyxy: tuple


def test_box_overlap_metrics() -> None:
    big = (0, 0, 100, 100)
    small = (10, 10, 90, 90)                 # fully nested (area 6400) inside big (area 10000)
    assert box_overlap(small, big, "min") == 1.0            # smaller box 100% covered
    assert abs(box_overlap(small, big, "iou") - 0.64) < 1e-6  # 6400/10000 -- far below 0.95
    assert box_overlap((0, 0, 10, 10), (50, 50, 60, 60), "min") == 0.0   # disjoint


def test_suppresses_lower_conf_same_class_duplicate() -> None:
    dets = [_Box(0, 0.9, (10, 10, 50, 50)), _Box(0, 0.6, (11, 11, 49, 49))]   # ~identical, same class
    d = suppress_duplicate_boxes(dets, overlap=0.95, metric="min")
    assert not d[0].suppressed                              # higher-conf kept
    assert d[1].suppressed and d[1].keeper_index == 0 and d[1].overlap >= 0.95


def test_nested_duplicate_needs_min_metric() -> None:
    dets = [_Box(0, 0.9, (0, 0, 100, 100)), _Box(0, 0.5, (10, 10, 90, 90))]   # small nested in big
    assert suppress_duplicate_boxes(dets, metric="min")[1].suppressed          # containment catches it
    assert not suppress_duplicate_boxes(dets, metric="iou")[1].suppressed      # IoU 0.64 < 0.95 -> kept


def test_different_classes_not_suppressed() -> None:
    dets = [_Box(0, 0.9, (0, 0, 100, 100)), _Box(1, 0.5, (0, 0, 100, 100))]   # identical box, diff class
    assert not any(x.suppressed for x in suppress_duplicate_boxes(dets))


def test_below_threshold_kept_and_multi_dup() -> None:
    # A keeper + two duplicates of it, plus one distinct box that only half-overlaps.
    dets = [
        _Box(0, 0.9, (0, 0, 100, 100)),
        _Box(0, 0.8, (2, 2, 98, 98)),        # dup of #0
        _Box(0, 0.7, (1, 1, 99, 99)),        # dup of #0
        _Box(0, 0.6, (60, 0, 160, 100)),     # overlaps #0 by ~40% only -> kept
    ]
    d = suppress_duplicate_boxes(dets, overlap=0.95, metric="min")
    assert [x.suppressed for x in d] == [False, True, True, False]
    assert d[1].keeper_index == 0 and d[2].keeper_index == 0


def test_record_detections_suppression_roundtrip(tmp_path) -> None:
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    sid = db.upsert_specimen(SpecimenRecord(
        image_name="a.jpg", image_stem="a", original_path="/o/a.jpg", working_path="/w/a.jpg",
        width=200, height=200))
    rows = [
        DetRow(0, "Ruler", 0.9, (10, 10, 50, 50), "BBOX-ruler", crop_path="/c/a.jpg"),   # keeper (idx 0)
        DetRow(0, "Ruler", 0.7, (11, 11, 49, 49), "BBOX-ruler", crop_path=None,
               suppressed=True, suppressed_by_index=0, suppress_overlap=0.98),
    ]
    db.record_detections("archival_detection", sid, rows)

    kept = db.detections("archival_detection", sid)
    assert len(kept) == 1 and kept[0]["conf"] == 0.9                    # only the keeper by default
    allrows = db.detections("archival_detection", sid, include_suppressed=True)
    assert len(allrows) == 2
    sup = next(r for r in allrows if r["suppressed"])
    assert sup["suppressed_by"] == kept[0]["detection_id"]              # index -> keeper detection_id
    assert abs(sup["suppress_overlap"] - 0.98) < 1e-6
    # every downstream read excludes the suppressed box
    assert list(db.detection_boxes(sid, "archival_detection")) == [kept[0]["detection_id"]]
    assert len(db.overlay_detections(sid)) == 1
    assert db.specimens_with_crops("archival_detection") == [sid]      # keeper has a crop


def test_run_detection_dedupes_and_skips_crop(mock_config_path, tmp_path) -> None:
    import cv2
    import numpy as np

    from leafmachine3.core.config import Config
    from leafmachine3.core.records import Detection, Unit
    from leafmachine3.modules.detector_common import run_detection

    cfg = Config.load(mock_config_path)
    img_p = tmp_path / "work.png"
    cv2.imwrite(str(img_p), np.full((200, 200, 3), 255, np.uint8))
    unit = Unit(specimen_id=1, stem="s", working_path=str(img_p), original_path=str(img_p),
                crops_dir=str(tmp_path / "crops"))

    class _FakeModel:
        def predict(self, _img):
            return [
                Detection(0, "Ruler", 0.9, (20, 20, 120, 120)),
                Detection(0, "Ruler", 0.6, (22, 22, 118, 118)),   # nested dup of the 0.9 Ruler
                Detection(1, "Label", 0.8, (20, 20, 120, 120)),   # same box, different class -> kept
            ]

    rows = run_detection(cfg, "plant_detector", unit, _FakeModel())
    assert len(rows) == 3                                            # all boxes recorded (for provenance)
    kept = [r for r in rows if not r.suppressed]
    sup = [r for r in rows if r.suppressed]
    assert len(kept) == 2 and len(sup) == 1
    assert sup[0].cls_name == "Ruler" and sup[0].conf == 0.6
    assert sup[0].crop_path is None and sup[0].suppressed_by_index == 0   # no crop; keyed to the keeper
    assert all(r.crop_path for r in kept)                           # keepers got crops saved
