"""modules.ruler_cf.use_CF_predicted_by_MP -- the megapixel CF as a fallback, and its provenance.

Off (the default), a sheet with no ruler or a lattice that did not pass keeps a NULL CF. On, that
sheet gets the MP prediction written INTO ``specimen.cf_px_per_cm``. Because the value alone cannot
say whether it was measured or predicted, ``specimen.cf_source`` must always say so -- these tests
pin the gate (only non-published sheets, only when on), the frame (the WORKING-frame anchor, never
the raw stored prediction), and the provenance column through write, reset and migration.
"""
from __future__ import annotations

import types

import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import CF_SOURCE_MP, CF_SOURCE_RULER, SpecimenRecord
from leafmachine3.inference.ruler_lattice import RulerCFLattice
from leafmachine3.modules.ruler_conversion_factor import RulerConversionFactor, _writeback


def _record(status, cf=None, anchor=120.0, crops=()):
    return {"image": {"status": status, "cf_px_per_cm": cf, "mp_anchor_working": anchor},
            "crops": list(crops)}


def _stage(use_mp: bool) -> RulerConversionFactor:
    cfg = {"ruler_cf": {"use_CF_predicted_by_MP": use_mp}}
    return RulerConversionFactor(types.SimpleNamespace(modules={}, stage=lambda k: cfg.get(k, {})))


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
def test_published_sheet_is_measured_whatever_the_option() -> None:
    crops = [{"verdict": "used", "ruler_class": "METRIC_MM"}]
    for on in (False, True):
        wb = _writeback(_record("published", cf=101.5, crops=crops), use_mp_fallback=on)
        assert wb == {"cf_px_per_cm": 101.5, "unit_type": "METRIC_MM", "source": CF_SOURCE_RULER}


@pytest.mark.parametrize("status", ["withheld", "no_reading", "no_ruler"])
def test_option_off_leaves_every_other_sheet_without_a_cf(status) -> None:
    assert _writeback(_record(status), use_mp_fallback=False) is None


@pytest.mark.parametrize("status", ["withheld", "no_reading", "no_ruler"])
def test_option_on_substitutes_the_working_frame_anchor(status) -> None:
    wb = _writeback(_record(status, anchor=88.25), use_mp_fallback=True)
    assert wb == {"cf_px_per_cm": 88.25, "unit_type": None, "source": CF_SOURCE_MP}


def test_no_anchor_means_no_fallback() -> None:
    """mp_conversion_factor disabled -> nothing to fall back to; the sheet stays NULL."""
    assert _writeback(_record("no_ruler", anchor=None), use_mp_fallback=True) is None


def test_fallback_is_the_working_frame_value_for_an_original_frame_model() -> None:
    """A LINEAR MP model predicts in the ORIGINAL frame; the CF column is WORKING frame. The
    fallback must be the engine's rescaled anchor, not the raw stored prediction."""
    engine = RulerCFLattice(artifact_dir="", write_qc=False, write_rasters=False)
    rec = engine.process_specimen({"specimen_id": 1, "image_name": "a.jpg", "work_scale": 0.5,
                                   "working_width": 1500, "working_height": 2000,
                                   "cf_px_per_cm_predicted_by_mp": 120.0,
                                   "anchor_frame": "original"}, [])
    assert rec["image"]["status"] == "no_ruler"
    assert _writeback(rec, use_mp_fallback=True)["cf_px_per_cm"] == pytest.approx(60.0)


# --------------------------------------------------------------------------- #
# collect -> infer -> persist on a real project DB
# --------------------------------------------------------------------------- #
@pytest.fixture()
def project(tmp_path):
    """Two sheets, both through the ruler_cf dependencies: sheet 1 has a Ruler crop, sheet 2 none."""
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    for stem in ("withRuler", "noRuler"):
        db.upsert_specimen(SpecimenRecord(
            image_name=f"{stem}.jpg", image_stem=stem,
            original_path=f"/orig/{stem}.jpg", working_path=f"/work/{stem}.jpg",
            width=1000, height=2000, original_width=1000, original_height=2000, work_scale=1.0))
    db.conn.execute("UPDATE specimen SET cf_px_per_cm_predicted_by_mp = 97.5")
    db.conn.execute(
        "INSERT INTO archival_detection (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2, "
        " crop_path) VALUES (1, 0, 'Ruler', 0.9, 10, 10, 400, 60, '/crops/r.jpg')")
    for sid in (1, 2):
        for dep in ("archival_detector", "ruler_classifier"):
            db.mark_image_done(sid, dep)
    return types.SimpleNamespace(db=db)


def test_option_off_collects_only_sheets_with_rulers(project) -> None:
    assert [it.specimen_id for it in _stage(False).collect_items(project)] == [1]


def test_option_on_also_collects_rulerless_sheets(project) -> None:
    items = {it.specimen_id: it for it in _stage(True).collect_items(project)}
    assert sorted(items) == [1, 2]
    _spec, crops = items[2].payload
    assert crops == []


def test_persist_writes_cf_source_on_the_specimen_and_the_audit_row(project) -> None:
    stage = _stage(True)
    item = next(it for it in stage.collect_items(project) if it.specimen_id == 2)
    engine = RulerCFLattice(artifact_dir="", write_qc=False, write_rasters=False)
    stage.persist(project, item, stage.infer(item, engine))

    s = project.db.get_specimen(2)
    assert s["cf_px_per_cm"] == pytest.approx(97.5)          # the sqrt model is working-frame
    assert s["cf_source"] == CF_SOURCE_MP
    assert s["ruler_unit_type"] is None
    audit = project.db.ruler_cf_lattice_record(2)["image"]
    assert audit["status"] == "no_ruler"
    assert audit["cf_source"] == CF_SOURCE_MP                 # the QC panel reads "applied" here
    assert audit["cf_px_per_cm"] is None                      # the lattice itself produced nothing


def test_reset_nulls_cf_source_with_the_cf(project) -> None:
    project.db.set_specimen_cf(2, 97.5, source=CF_SOURCE_MP)
    project.db.reset_stages(["ruler_cf"], {"ruler_cf": RulerConversionFactor})
    s = project.db.get_specimen(2)
    assert s["cf_px_per_cm"] is None and s["cf_source"] is None


def test_no_cf_means_no_source(project) -> None:
    project.db.set_specimen_cf(2, None, source=CF_SOURCE_MP)
    assert project.db.get_specimen(2)["cf_source"] is None


def test_migration_backfills_existing_ruler_cfs(tmp_path) -> None:
    """Before cf_source existed, only a published ruler CF was ever written to cf_px_per_cm."""
    path = tmp_path / "old.sqlite"
    db = ProjectDB.open_or_create(path)
    for stem in ("a", "b"):
        db.upsert_specimen(SpecimenRecord(image_name=f"{stem}.jpg", image_stem=stem,
                                          original_path=f"/o/{stem}", working_path=f"/w/{stem}",
                                          width=10, height=10, original_width=10,
                                          original_height=10, work_scale=1.0))
    db.conn.execute("UPDATE specimen SET cf_px_per_cm = 100.0 WHERE image_stem = 'a'")
    db.conn.execute("ALTER TABLE specimen DROP COLUMN cf_source")
    db.close()

    db = ProjectDB.open_or_create(path)
    got = {r["image_stem"]: r["cf_source"] for r in db.conn.execute("SELECT * FROM specimen")}
    assert got == {"a": CF_SOURCE_RULER, "b": None}


# --------------------------------------------------------------------------- #
# QC panel says whether the fallback was APPLIED
# --------------------------------------------------------------------------- #
def _texts(monkeypatch, fn, *a, **kw) -> str:
    from PIL import ImageDraw

    seen: list[str] = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        seen.append(str(text))
        return real(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    fn(*a, **kw)
    return "\n".join(seen)


def test_cf_summary_names_the_cf_in_use(monkeypatch) -> None:
    from leafmachine3.inference.ruler_lattice import qc

    applied = _texts(monkeypatch, qc.build_cf_summary_section, None, 97.5, "k*sqrt(MP)",
                     fallback_applied=True)
    assert "CF used for this sheet: 97.50 px/cm -- PREDICTED from megapixels" in applied
    off = _texts(monkeypatch, qc.build_cf_summary_section, None, 97.5, "k*sqrt(MP)")
    assert "CF used for this sheet: none" in off
    measured = _texts(monkeypatch, qc.build_cf_summary_section, 101.0, 97.5, "k*sqrt(MP)")
    assert "CF used for this sheet: 101.00 px/cm -- measured from the ruler" in measured


def test_settings_default_is_off() -> None:
    from leafmachine3.core.config import builtin_defaults

    assert builtin_defaults()["modules"]["ruler_cf"]["use_CF_predicted_by_MP"] is False
