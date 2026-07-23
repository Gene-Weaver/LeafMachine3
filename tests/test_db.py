"""Tests for :mod:`leafmachine3.core.db` -- schema, CRUD, and the resume ledger."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import DetRow, PhenologyResult, SpecimenRecord


@dataclass
class _StageStub:
    """Minimal stand-in for a PipelineStage that ``reset_stages`` can introspect."""

    key: str
    owns_tables: tuple[str, ...]


@pytest.fixture
def db(tmp_path: Path) -> ProjectDB:
    return ProjectDB.open_or_create(tmp_path / "proj.sqlite")


def _specimen(stem: str, path: str = "/orig") -> SpecimenRecord:
    return SpecimenRecord(
        image_name=f"{stem}.jpg",
        image_stem=stem,
        original_path=f"{path}/{stem}.jpg",
        working_path=f"/work/{stem}.jpg",
        width=100,
        height=200,
        original_width=100,
        original_height=200,
        work_scale=1.0,
        orig_size_bytes=123,
        orig_mtime=1.0,
    )


def test_schema_seeds_project_status(db: ProjectDB) -> None:
    rows = db._query("SELECT stage_key, stage_order, state FROM project_status ORDER BY stage_order")
    keys = [r["stage_key"] for r in rows]
    assert keys == [
        "archival_detector", "plant_detector", "phenology_detector", "ruler_classifier",
        "ruler_cf", "leaf_segmenter", "morphology", "metric_grounding", "reporter",
    ]
    assert all(r["state"] == "pending" for r in rows)


def test_upsert_specimen_is_idempotent(db: ProjectDB) -> None:
    sid1 = db.upsert_specimen(_specimen("a"))
    sid2 = db.upsert_specimen(_specimen("a", path="/moved"))  # same stem -> update, same id
    assert sid1 == sid2
    assert len(db.iter_specimens()) == 1
    assert db.get_specimen(sid1)["original_path"] == "/moved/a.jpg"


def test_record_detections_and_crops(db: ProjectDB) -> None:
    sid = db.upsert_specimen(_specimen("a"))
    rows = [
        DetRow(0, "Ruler", 0.9, (1.0, 2.0, 3.0, 4.0), tag="R", crop_path="/crops/a__R__.jpg"),
        DetRow(1, "Label", 0.8, (5.0, 6.0, 7.0, 8.0), tag="L", crop_path="/crops/a__L__.jpg"),
    ]
    with db.transaction():
        db.record_detections("archival_detection", sid, rows)

    assert len(db.detections("archival_detection", sid)) == 2
    ruler_crops = db.crops("archival_detection", sid, cls_name="Ruler")
    assert len(ruler_crops) == 1
    assert ruler_crops[0].cls_name == "Ruler"
    assert db.specimens_with_crops("archival_detection", cls_name="Ruler") == [sid]


def test_record_detections_delete_then_insert(db: ProjectDB) -> None:
    """A re-run replaces a specimen's rows rather than appending them."""
    sid = db.upsert_specimen(_specimen("a"))
    with db.transaction():
        db.record_detections("archival_detection", sid, [DetRow(0, "Ruler", 0.9, (1, 2, 3, 4))])
    with db.transaction():
        db.record_detections("archival_detection", sid, [DetRow(0, "Ruler", 0.5, (1, 2, 3, 4))])
    got = db.detections("archival_detection", sid)
    assert len(got) == 1
    assert got[0]["conf"] == 0.5


def test_phenology_mirrors_flags_onto_specimen(db: ProjectDB) -> None:
    sid = db.upsert_specimen(_specimen("a"))
    with db.transaction():
        db.record_phenology(sid, PhenologyResult(leaves=(True, 3), flowers=(False, 0), fruits=(True, 1)))
    spec = db.get_specimen(sid)
    assert spec["has_leaves"] == 1 and spec["has_flowers"] == 0 and spec["has_fruits"] == 1


def test_mark_image_done_shrinks_pending(db: ProjectDB) -> None:
    a = db.upsert_specimen(_specimen("a"))
    b = db.upsert_specimen(_specimen("b"))
    stage = "archival_detector"

    assert set(db.pending_specimens(stage)) == {a, b}
    db.mark_image_done(a, stage)
    assert db.pending_specimens(stage) == [b]
    assert db.done_ids(stage) == {a}
    db.mark_image_done(b, stage)
    assert db.pending_specimens(stage) == []


def test_eligible_specimens_gated_by_depends_on(db: ProjectDB) -> None:
    a = db.upsert_specimen(_specimen("a"))
    b = db.upsert_specimen(_specimen("b"))
    # phenology depends on plant_detector: only specimens done there are eligible.
    assert db.eligible_specimens("phenology_detector", depends_on=("plant_detector",)) == []
    db.mark_image_done(a, "plant_detector")
    assert db.eligible_specimens("phenology_detector", depends_on=("plant_detector",)) == [a]
    db.mark_image_done(b, "plant_detector")
    assert set(db.eligible_specimens("phenology_detector", depends_on=("plant_detector",))) == {a, b}


def test_mark_stage_complete_no_work_satisfies_downstream(db: ProjectDB) -> None:
    a = db.upsert_specimen(_specimen("a"))
    db.mark_stage_complete_no_work("ruler_cf")
    assert db.stage_state("ruler_cf") == "done"
    # every specimen counts as done for the disabled stage
    assert db.done_ids("ruler_cf") == {a}


def test_reset_stages_purges_rows_and_ledger(db: ProjectDB) -> None:
    sid = db.upsert_specimen(_specimen("a"))
    with db.transaction():
        db.record_detections("archival_detection", sid, [DetRow(0, "Ruler", 0.9, (1, 2, 3, 4))])
    db.mark_image_done(sid, "archival_detector")
    db.mark_stage_done("archival_detector")
    db.set_stage_settings_hash("archival_detector", "deadbeef")

    stub = _StageStub(key="archival_detector", owns_tables=("archival_detection",))
    db.reset_stages(["archival_detector"], {"archival_detector": stub})

    assert db.detections("archival_detection", sid) == []          # method rows gone
    assert db.done_ids("archival_detector") == set()               # ledger cleared
    assert db.stage_state("archival_detector") == "pending"        # status reset
    assert db.stage_settings_hash("archival_detector") is None
    assert db.pending_specimens("archival_detector") == [sid]      # work is pending again


def test_reclaim_running_reverts_to_pending(db: ProjectDB) -> None:
    db.mark_stage_running("plant_detector", n_total=5)
    assert db.stage_state("plant_detector") == "running"
    db.reclaim_running()
    assert db.stage_state("plant_detector") == "pending"


def test_settings_hash_roundtrip(db: ProjectDB) -> None:
    assert db.stage_settings_hash("reporter") is None
    db.set_stage_settings_hash("reporter", "abc123")
    assert db.stage_settings_hash("reporter") == "abc123"
