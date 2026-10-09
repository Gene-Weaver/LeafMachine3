"""FieldPrism (FP) inside the ruler_cf engine: anchor, reconciliation, gating, persistence.

Every image is SYNTHETIC and written to tmp_path: an FPfit crop of a catalog sheet (the outermost
square centers 15 mm from the edges) with the markers drawn exactly where fieldprism_sheets.json
puts them, optionally with a mm tick ruler pasted in the middle. The engine reads the working
image itself (``working_path``) and the tick-ruler crop from ``crop_path``, as in production.

The headline rules pinned here:
  * a sheet with FP crops is anchored on the FieldPrism geometry -- the MP prediction is recorded
    but never used, not even as the use_CF_predicted_by_MP fallback;
  * the MIN_FRAME_CM frame-width drop (an MP-model guard) is not applied to that anchor;
  * a sheet WITHOUT FP crops behaves exactly as before (anchor_source 'megapixels').
"""
from __future__ import annotations

import json
import sqlite3
import types

import cv2
import numpy as np
import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import (CF_SOURCE_FP, CF_SOURCE_MP, CF_SOURCE_RULER,
                                       SpecimenRecord)
from leafmachine3.inference.ruler_lattice import (RulerCFLattice, engine as E, fieldprism as fp,
                                                  fp_marker_columns, fp_sheet_columns,
                                                  image_columns)
from leafmachine3.inference.ruler_lattice.analysis import analyse
from leafmachine3.inference.ruler_lattice.sheet_cf import MIN_FRAME_CM
from leafmachine3.inference.ruler_lattice.units import is_fieldprism, is_skipped
from leafmachine3.modules.ruler_conversion_factor import RulerConversionFactor, _writeback

CAT = fp.load_sheet_catalog()
CORNERS = fp.CORNERS
PPMM = 8.0                  # 80 px/cm (the synthetic mm ruler aliases at half-pixel tick pitches)
WILD_MP = 20.0              # an MP anchor that is wildly wrong for these sheets
RULER_DET = 50              # detection_id of the pasted tick ruler (FP markers are 100..103)


# ============================================================================================
# synthetic sheets
# ============================================================================================
def _tick_ruler(ppcm, length_cm=12.0, h=90, seed=0):
    """A horizontal mm ruler: 1 mm minors, 5 mm mids, 1 cm majors (grayscale)."""
    ppmm = ppcm / 10.0
    img = np.full((h, int(round((length_cm + 1.0) * ppcm))), 235, np.float32)
    for k in range(int(length_cm * 10) + 1):
        x = 0.5 * ppcm + k * ppmm
        L = h * (0.6 if k % 10 == 0 else 0.42 if k % 5 == 0 else 0.28)
        x0, x1 = x - 0.12 * ppmm, x + 0.12 * ppmm
        poly = np.array([[x0, 4], [x1, 4], [x1, 4 + L], [x0, 4 + L]])
        cv2.fillPoly(img, [np.round(poly * 256).astype(np.int32)], 20, cv2.LINE_AA, shift=8)
    img += np.random.default_rng(seed).normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_sheet(tmp_path, *, sheet="Letter", ppmm=PPMM, drawn=CORNERS, boxed=None,
               marker_scale=None, ruler_ppcm=None, mp=WILD_MP, name="sheet", sid=1):
    """Render an FPfit sheet to tmp_path -> (specimen payload, crops), as the stage builds them.

    drawn = corners whose marker is printed; boxed = corners that get an FP detector box
    (default: the drawn ones). marker_scale scales one marker about its C square (a marker
    printed at the wrong size disagrees with its peers)."""
    boxed = drawn if boxed is None else boxed
    marker_scale = marker_scale or {}
    sc = CAT["sheets"][sheet]["square_centers_mm"]
    pts = np.array([sc[c][r] for c in CORNERS for r in fp.ROLES], float)
    lo, hi = pts.min(0) - 15.0, pts.max(0) + 15.0
    W, H = int(round((hi[0] - lo[0]) * ppmm)), int(round((hi[1] - lo[1]) * ppmm))
    img = np.full((H, W), 232, np.uint8)

    def px(p):
        return ((p[0] - lo[0]) * ppmm, (p[1] - lo[1]) * ppmm)

    crops = []
    for i, c in enumerate(CORNERS):
        k = marker_scale.get(c, 1.0)
        cen = np.array(sc[c]["C"], float)
        if c in drawn:
            for r in fp.ROLES:
                p = cen + (np.array(sc[c][r]) - cen) * k
                a, b = px(p - 5 * k), px(p + 5 * k)
                cv2.rectangle(img, (int(round(a[0])), int(round(a[1]))),
                              (int(round(b[0])) - 1, int(round(b[1])) - 1), 25, -1)
        if c in boxed:
            a, b = px(cen - 15 * k - 1.0), px(cen + 15 * k + 1.0)
            crops.append(dict(detection_id=100 + i, det_conf=0.9 - 0.01 * i,
                              crop_path=str(tmp_path / f"{name}_fp{i}.jpg"), ruler_class="FP",
                              cls_conf=None, tile_four_path=None,
                              x1=a[0], y1=a[1], x2=b[0], y2=b[1]))
    if ruler_ppcm is not None:
        r = _tick_ruler(ruler_ppcm)
        y0, x0 = H // 2 - r.shape[0] // 2, (W - r.shape[1]) // 2
        assert x0 > 0, "ruler wider than the sheet"
        img[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
        cp = tmp_path / f"{name}_ruler.png"
        cv2.imwrite(str(cp), r)
        crops.append(dict(detection_id=RULER_DET, det_conf=0.95, crop_path=str(cp),
                          ruler_class="METRIC_MM", cls_conf=None, tile_four_path=None,
                          x1=float(x0), y1=float(y0), x2=float(x0 + r.shape[1]),
                          y2=float(y0 + r.shape[0])))
    wp = tmp_path / f"{name}.png"
    cv2.imwrite(str(wp), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    specimen = dict(specimen_id=sid, image_name=f"{name}.png", work_scale=1.0,
                    working_width=W, working_height=H, original_mp=W * H / 1e6,
                    cf_px_per_cm_predicted_by_mp=mp, anchor_frame="working",
                    anchor_formula="k*sqrt(MP)", anchor_formula_symbolic="k*sqrt(MP)",
                    working_path=str(wp))
    return specimen, crops


def _engine(tmp_path, **kw):
    kw.setdefault("write_qc", False)
    return RulerCFLattice(artifact_dir=tmp_path / "art", write_rasters=False, **kw)


def _crop(rec, det):
    return next(r for r in rec["crops"] if r["detection_id"] == det)


def _reasons(rec):
    return json.loads(rec["image"]["confidence_reasons_json"])


# ============================================================================================
# class routing
# ============================================================================================
def test_fp_class_is_routed_to_fieldprism_and_kept_out_of_the_lattice():
    assert is_fieldprism("FP") and not is_fieldprism("METRIC_MM")
    assert "FieldPrism" in is_skipped("FP")               # the lattice guard still holds
    res = analyse(np.zeros((40, 200), np.uint8), "FP")
    assert res["skipped"] and "FieldPrism" in res["skip_reason"]


# ============================================================================================
# FP-only sheets
# ============================================================================================
def test_fp_only_sheet_publishes_from_fieldprism_and_ignores_a_wild_mp_anchor(tmp_path):
    spec, crops = make_sheet(tmp_path)
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img = rec["image"]
    assert img["status"] == "published" and img["confidence"] == "high"
    assert img["cf_px_per_cm"] == pytest.approx(10 * PPMM, rel=0.01)
    assert img["cf_source"] == CF_SOURCE_FP
    assert img["anchor_source"] == "fieldprism" and img["fp_detected"] == 1
    assert img["anchor_cf_working"] == pytest.approx(10 * PPMM, rel=0.005)
    assert img["mp_anchor_working"] == WILD_MP              # recorded for audit only
    assert img["pct_vs_anchor"] == pytest.approx(0.0, abs=1.0)
    assert img["fallback"] == "" and img["n_used"] == 4 and img["n_skipped"] == 0
    assert "FieldPrism geometry is authoritative" in " ".join(_reasons(rec))
    assert "MP anchor" not in " ".join(_reasons(rec))

    sheet = rec["fp_sheet"]
    assert sheet["sheet_status"] == "identified" and sheet["sheet_type"] == "Letter"
    assert sheet["cf_source_detail"] == "sheet_fit" and sheet["fp_confidence"] == "high"
    assert sheet["catalog_version"] == CAT["catalog_version"]
    assert {m["detection_id"]: (m["sheet_corner"], m["verdict"]) for m in rec["fp_markers"]} == \
        {100 + i: (c, "used") for i, c in enumerate(CORNERS)}
    assert [m["crop_index"] for m in rec["fp_markers"]] == [0, 1, 2, 3]
    for r in rec["crops"]:
        assert r["class_layout"] == "fieldprism" and r["status"] == "measured"
        assert r["n_kept"] is None and r["verdict"] == "used"

    wb = _writeback(rec, use_mp_fallback=True)
    assert wb == {"cf_px_per_cm": img["cf_px_per_cm"], "unit_type": "FP", "source": CF_SOURCE_FP}


def test_min_frame_drop_is_not_applied_to_the_fieldprism_anchor(tmp_path):
    """A Letter FPfit crop is ~19.6 cm wide: the MP-model frame guard would drop ANY anchor."""
    spec, crops = make_sheet(tmp_path, mp=10 * PPMM)
    assert spec["working_width"] / (10 * PPMM) < MIN_FRAME_CM
    img = _engine(tmp_path).process_specimen(spec, crops)["image"]
    assert img["anchor_dropped"] == 0
    assert "DROPPED" not in (img["method"] or "")
    assert img["anchor_supported"] == 1 and img["status"] == "published"


def test_three_markers_infer_the_fourth(tmp_path):
    spec, crops = make_sheet(tmp_path, drawn=("TR", "BL", "BR"))
    rec = _engine(tmp_path).process_specimen(spec, crops)
    assert rec["image"]["status"] == "published"
    sheet = rec["fp_sheet"]
    assert sheet["sheet_type"] == "Letter" and sheet["n_fp_inferred"] == 1
    assert json.loads(sheet["corners_json"])["TL"]["observed"] is False


def test_disagreeing_fp_markers_are_withheld_and_never_fall_back_to_mp(tmp_path):
    spec, crops = make_sheet(tmp_path, drawn=("TL", "BR"), marker_scale={"BR": 1.08})
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img = rec["image"]
    assert rec["fp_sheet"]["fp_confidence"] == "low"
    assert img["status"] == "withheld" and img["cf_px_per_cm"] is None
    assert img["confidence"] in ("medium", "low") and img["cf_source"] is None
    assert img["fallback"] == "none"
    assert _writeback(rec, use_mp_fallback=True) is None


@pytest.mark.parametrize("allow,status,conf", [(True, "published", "high"),
                                               (False, "withheld", "medium")])
def test_single_valid_marker_obeys_fp_allow_single_marker(tmp_path, allow, status, conf):
    spec, crops = make_sheet(tmp_path, drawn=("TL",))
    rec = _engine(tmp_path, fp_allow_single_marker=allow).process_specimen(spec, crops)
    img = rec["image"]
    assert img["status"] == status and img["confidence"] == conf
    assert img["anchor_source"] == "fieldprism"
    assert rec["fp_sheet"]["sheet_status"] == "undetermined"
    assert rec["fp_sheet"]["cf_source_detail"] == "marker_mean"
    if not allow:
        assert any("CF withheld" in r for r in _reasons(rec))
        assert _writeback(rec, use_mp_fallback=True) is None


def test_all_fp_markers_failed_means_no_anchor_and_no_mp_fallback(tmp_path):
    spec, crops = make_sheet(tmp_path, drawn=(), boxed=CORNERS)     # boxes on blank paper
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img = rec["image"]
    assert img["anchor_source"] == "none" and img["anchor_cf_working"] is None
    assert img["fp_detected"] == 1 and img["status"] == "no_reading"
    assert img["fallback"] == "none" and img["n_failed"] == 4
    assert all(r["status"] == "failed" and r["status_reason"] for r in rec["crops"])
    assert rec["fp_sheet"]["fp_confidence"] is None
    assert all(m["verdict"] == "skipped" for m in rec["fp_markers"])
    assert _writeback(rec, use_mp_fallback=True) is None


def test_unreadable_working_image_is_recorded_not_raised(tmp_path):
    spec, crops = make_sheet(tmp_path)
    spec["working_path"] = str(tmp_path / "missing.png")
    rec = _engine(tmp_path).process_specimen(spec, crops)
    assert rec["image"]["anchor_source"] == "none" and rec["image"]["status"] == "no_reading"
    assert {r["status"] for r in rec["crops"]} == {"unreadable"}
    assert {m["status"] for m in rec["fp_markers"]} == {"unreadable"}
    assert "could not be read" in json.loads(rec["fp_sheet"]["fp_reasons_json"])[0]
    assert _writeback(rec, use_mp_fallback=True) is None


# ============================================================================================
# FP + tick rulers on one sheet
# ============================================================================================
def test_agreeing_tick_ruler_is_used_alongside_fp(tmp_path):
    spec, crops = make_sheet(tmp_path, ruler_ppcm=10 * PPMM)
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img, ruler = rec["image"], _crop(rec, RULER_DET)
    assert ruler["status"] == "measured" and ruler["pxcm"] == pytest.approx(10 * PPMM, rel=0.01)
    assert ruler["verdict"] == "used" and ruler["class_layout"] != "fieldprism"
    assert ruler["pct_vs_anchor"] == pytest.approx(0.0, abs=1.0)    # vs the FP anchor, not MP
    assert img["status"] == "published" and img["cf_source"] == CF_SOURCE_FP
    assert img["n_used"] == 5
    assert img["cf_px_per_cm"] == pytest.approx(10 * PPMM, rel=0.01)


def test_published_cf_is_the_fieldprism_anchor_not_a_weighted_mean(tmp_path):
    """FieldPrism is authoritative for the VALUE: the published CF is the sheet fit, exactly. A
    corroborating tick ruler inside the tolerance (+2.5% here, with far more ticks than a marker has
    squares) is used as a witness but must not drag the published number toward itself."""
    spec, crops = make_sheet(tmp_path, ruler_ppcm=1.025 * 10 * PPMM)
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img, sheet, ruler = rec["image"], rec["fp_sheet"], _crop(rec, RULER_DET)
    assert ruler["verdict"] == "used"
    assert ruler["pct_vs_parent"] == pytest.approx(2.5, abs=1.0)
    assert sheet["cf_source_detail"] == "sheet_fit"
    assert img["status"] == "published"
    assert img["cf_px_per_cm"] == sheet["cf_px_per_cm_fp"] == img["anchor_cf_working"]
    assert img["cf_px_per_cm_measured"] == img["cf_px_per_cm"]
    assert img["method"] == "fieldprism_sheet_fit"
    assert any("corroborate the FieldPrism CF" in r for r in _reasons(rec))


def test_agreeing_markers_are_never_split_by_a_second_clustering(tmp_path):
    """Markers that FieldPrism's median-based peer step keeps (all within 3% of the median, so up
    to ~5% apart) must stay 'used' everywhere: a second greedy 3% clustering used to split them
    2+2 and stamp two of the markers that DEFINE the published CF as 'rejected'."""
    spec, crops = make_sheet(tmp_path, marker_scale={"TL": 0.975, "TR": 0.99, "BL": 1.01,
                                                     "BR": 1.025})
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img, sheet = rec["image"], rec["fp_sheet"]
    assert sheet["fp_confidence"] == "high"
    assert [m["verdict"] for m in rec["fp_markers"]] == ["used"] * 4
    assert [r["verdict"] for r in rec["crops"]] == ["used"] * 4
    assert img["n_used"] == 4 and img["n_rejected"] == 0 and img["n_dissenting"] == 0
    assert sheet["n_fp_used"] == 4 and sheet["n_fp_rejected"] == 0
    assert img["status"] == "published" and img["cf_px_per_cm"] == sheet["cf_px_per_cm_fp"]


def test_marker_rejected_by_fieldprism_stays_rejected_and_counters_agree(tmp_path):
    spec, crops = make_sheet(tmp_path, marker_scale={"BR": 1.06})
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img, sheet = rec["image"], rec["fp_sheet"]
    verdicts = {m["sheet_corner"] or m["detection_id"]: m["verdict"] for m in rec["fp_markers"]}
    assert sorted(verdicts.values()) == ["rejected", "used", "used", "used"]
    crop_v = sorted(r["verdict"] for r in rec["crops"])
    assert crop_v == ["rejected", "used", "used", "used"]
    assert sheet["n_fp_used"] == img["n_used"] == 3
    assert sheet["n_fp_rejected"] == img["n_rejected"] == 1
    assert img["status"] == "published"


def test_disagreeing_markers_give_no_anchor_so_a_tick_ruler_cannot_borrow_one(tmp_path):
    """Two FP markers 8% apart are low confidence: their mean is no reference at all. A tick ruler
    reading near that mean must NOT be certified 'high' by it (nor 'overrule' the markers)."""
    spec, crops = make_sheet(tmp_path, drawn=("TL", "BR"), marker_scale={"BR": 1.08},
                             ruler_ppcm=1.04 * 10 * PPMM)
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img = rec["image"]
    assert rec["fp_sheet"]["fp_confidence"] == "low"
    assert img["anchor_source"] == "none" and img["anchor_cf_working"] is None
    assert img["status"] != "published" and img["cf_px_per_cm"] is None
    assert "OVERRULED" not in " ".join(_reasons(rec))
    assert _writeback(rec, use_mp_fallback=True) is None


def test_fieldprism_cf_writes_back_unit_type_fp_even_with_a_tick_witness(tmp_path):
    spec, crops = make_sheet(tmp_path, drawn=("TL",), ruler_ppcm=10 * PPMM)
    rec = _engine(tmp_path).process_specimen(spec, crops)
    assert rec["image"]["cf_source"] == CF_SOURCE_FP
    assert _crop(rec, RULER_DET)["verdict"] == "used"
    assert _writeback(rec)["unit_type"] == "FP"


def test_disagreeing_tick_ruler_is_rejected_and_fp_stays_authoritative(tmp_path):
    spec, crops = make_sheet(tmp_path, ruler_ppcm=100.0)   # printed 25% off the sheet's scale
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img, ruler = rec["image"], _crop(rec, RULER_DET)
    assert ruler["status"] == "measured" and abs(ruler["pxcm"] / (10 * PPMM) - 1) > 0.05
    assert ruler["verdict"] == "rejected"
    assert img["status"] == "published" and img["confidence"] == "high"
    assert img["cf_source"] == CF_SOURCE_FP
    assert img["cf_px_per_cm"] == pytest.approx(10 * PPMM, rel=0.01)
    assert all(m["verdict"] == "used" for m in rec["fp_markers"])


# ============================================================================================
# sheets WITHOUT FP crops are unchanged
# ============================================================================================
def _strip(rec):
    img = {k: v for k, v in rec["image"].items()
           if k not in ("created_at", "runtime_ms", "engine_params_json")}
    return json.loads(E._json(dict(image=img, crops=rec["crops"])))


def test_non_fp_sheet_keeps_the_megapixel_anchor(tmp_path):
    spec, crops = make_sheet(tmp_path, drawn=(), ruler_ppcm=10 * PPMM, mp=10 * PPMM)
    crops = [c for c in crops if c["ruler_class"] != "FP"]
    rec = _engine(tmp_path).process_specimen(spec, crops)
    img = rec["image"]
    assert img["anchor_source"] == "megapixels" and img["fp_detected"] == 0
    assert img["anchor_cf_working"] == img["mp_anchor_working"] == 10 * PPMM
    assert rec["fp_sheet"] is None and rec["fp_markers"] == []
    # the MP-model frame guard still applies to the MP anchor (this frame is < 20 cm wide)
    assert img["anchor_dropped"] == 1 and img["cf_source"] in (None, CF_SOURCE_RULER)
    # ...and FP settings cannot touch a sheet without FP crops
    other = _engine(tmp_path, fp_enabled=False, fp_anchor_tol=0.5,
                    fp_allow_single_marker=False).process_specimen(spec, crops)
    assert _strip(other) == _strip(rec)

    no_anchor = _engine(tmp_path).process_specimen(dict(spec, cf_px_per_cm_predicted_by_mp=None),
                                                   crops)["image"]
    assert no_anchor["anchor_source"] == "none" and no_anchor["fallback"] == "none"


def test_fp_disabled_skips_fp_crops_and_keeps_the_mp_fallback(tmp_path):
    spec, crops = make_sheet(tmp_path)
    rec = _engine(tmp_path, fp_enabled=False).process_specimen(spec, crops)
    img = rec["image"]
    assert img["anchor_source"] == "megapixels" and img["fp_detected"] == 0
    assert img["status"] == "no_reading" and img["n_skipped"] == 4
    assert all(r["status"] == "skipped_class" and "disabled" in r["status_reason"]
               for r in rec["crops"])
    assert rec["fp_sheet"] is None and rec["fp_markers"] == []
    assert _writeback(rec, use_mp_fallback=True)["source"] == CF_SOURCE_MP


def test_engine_version_and_params_record_fieldprism(tmp_path):
    p = _engine(tmp_path, fp_peer_tol=0.02).params()
    assert p["engine_version"] == E.ENGINE_VERSION == "lattice-2026.10.09-fp"
    assert p["fieldprism"]["peer_tol"] == 0.02
    assert p["fieldprism"]["catalog_version"] == CAT["catalog_version"]
    assert p["fieldprism"]["reconcile_weight"] == E.FP_RECONCILE_WEIGHT == 40


# ============================================================================================
# QC wiring
# ============================================================================================
def test_qc_panel_renders_and_rebuilds_identically_from_the_db(tmp_path):
    spec, crops = make_sheet(tmp_path, ruler_ppcm=10 * PPMM)
    for c in crops:                      # FP crop files, so the marker panels show real pixels
        if c["ruler_class"] == "FP":
            wimg = cv2.imread(spec["working_path"])
            x1, y1, x2, y2 = (int(round(c[k])) for k in ("x1", "y1", "x2", "y2"))
            cv2.imwrite(c["crop_path"], wimg[max(0, y1):y2, max(0, x1):x2])
    eng = RulerCFLattice(artifact_dir=tmp_path / "art", write_qc=True, write_rasters=True)
    rec = eng.process_specimen(spec, crops)
    assert rec["qc_image"] is not None and rec["image"]["qc_image_path"]

    db = _db(tmp_path, spec, crops)
    with db.transaction():
        db.record_ruler_cf_lattice(rec)
    back = db.ruler_cf_lattice_record(spec["specimen_id"])
    rebuilt = eng.render_qc(back)
    assert np.array_equal(np.asarray(eng.render_qc(rec)), np.asarray(rebuilt))


# ============================================================================================
# persistence
# ============================================================================================
def _db(tmp_path, spec, crops, name="p.sqlite"):
    db = ProjectDB.open_or_create(tmp_path / name)
    sid = db.upsert_specimen(SpecimenRecord(
        image_name=spec["image_name"], image_stem="sheet", original_path=spec["working_path"],
        working_path=spec["working_path"], width=spec["working_width"],
        height=spec["working_height"], original_width=spec["working_width"],
        original_height=spec["working_height"], work_scale=1.0))
    assert sid == spec["specimen_id"]
    for c in crops:
        db.conn.execute(
            "INSERT INTO archival_detection (detection_id, specimen_id, cls_id, cls_name, conf, "
            "x1, y1, x2, y2, crop_path) VALUES (?, ?, 0, 'Ruler', ?, ?, ?, ?, ?, ?)",
            (c["detection_id"], sid, c["det_conf"], c["x1"], c["y1"], c["x2"], c["y2"],
             c["crop_path"]))
        db.conn.execute("INSERT INTO ruler_classification (specimen_id, detection_id, unit_type, "
                        "squarify_path) VALUES (?, ?, ?, ?)",
                        (sid, c["detection_id"], c["ruler_class"], c["tile_four_path"]))
    return db


def test_write_db_read_record_round_trip_including_the_fp_tables(tmp_path):
    spec, crops = make_sheet(tmp_path, drawn=("TR", "BL", "BR"), boxed=CORNERS)
    rec = _engine(tmp_path).process_specimen(spec, crops)
    db = _db(tmp_path, spec, crops)
    for _ in range(2):                                    # idempotent: no duplicate rows
        with db.transaction():
            db.record_ruler_cf_lattice(rec)
    back = db.ruler_cf_lattice_record(spec["specimen_id"])

    assert back["fp_sheet"] == {c: E._num(rec["fp_sheet"].get(c)) for c in fp_sheet_columns()}
    assert len(back["fp_markers"]) == 4
    for got, want in zip(back["fp_markers"], rec["fp_markers"]):
        assert {c: got[c] for c in fp_marker_columns()} == \
            {c: E._num(want.get(c)) for c in fp_marker_columns()}
    for c in ("anchor_source", "anchor_cf_working", "fp_detected"):
        assert back["image"][c] == rec["image"][c]
    assert set(image_columns()) <= set(back["image"])
    assert back["fp_sheet"]["n_fp_inferred"] == 1
    failed = next(m for m in back["fp_markers"] if m["detection_id"] == 100)
    assert failed["status"] == "failed" and failed["verdict"] == "skipped"

    # what the Reporter / exporter read
    dets = db.overlay_detections(spec["specimen_id"])
    assert sorted(d["detection_id"] for d in dets) == [100, 101, 102, 103]
    row = dict(db.export_specimen_rows()[0])
    assert row["fp_sheet_type"] == "Letter" and row["fp_sheet_status"] == "identified"
    assert row["ruler_cf_anchor_source"] == "fieldprism" and row["fp_n_markers_inferred"] == 1
    assert row["fp_n_markers_detected"] == 4 and row["fp_n_markers_used"] == 3
    assert row["fp_cf_px_per_cm"] == pytest.approx(10 * PPMM, rel=0.01)
    assert row["fp_confidence"] == "high" and row["fp_orientation_deg"] == 0
    assert [r["crop_index"] for r in db.export_table("ruler_FP_marker")] == [0, 1, 2, 3]
    assert len(db.export_table("ruler_FP_sheet")) == 1
    assert "sheet_corner" in db.export_table_columns("ruler_FP_marker")

    # re-running the sheet without FieldPrism leaves no stale FP rows
    with db.transaction():
        db.record_ruler_cf_lattice(_engine(tmp_path, fp_enabled=False).process_specimen(spec, crops))
    again = db.ruler_cf_lattice_record(spec["specimen_id"])
    assert again["fp_sheet"] is None and again["fp_markers"] == []


def test_reset_deletes_fp_rows_but_never_foreign_files(tmp_path):
    spec, crops = make_sheet(tmp_path)
    for c in crops:                                       # foreign artifacts the rows reference
        c["tile_four_path"] = str(tmp_path / f"tile{c['detection_id']}.jpg")
        for p in (c["crop_path"], c["tile_four_path"]):
            open(p, "wb").write(b"x")
    rec = _engine(tmp_path).process_specimen(spec, crops)
    db = _db(tmp_path, spec, crops)
    with db.transaction():
        db.record_ruler_cf_lattice(rec)
    assert db.conn.execute("SELECT COUNT(*) FROM ruler_FP_marker").fetchone()[0] == 4

    assert RulerConversionFactor.owns_tables == (
        "ruler_FP_marker", "ruler_FP_sheet", "ruler_CF_lattice_crop", "ruler_CF_lattice")
    db.reset_stages(["ruler_cf"], {"ruler_cf": RulerConversionFactor})
    for t in RulerConversionFactor.owns_tables:
        assert db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
    for c in crops:
        assert open(c["crop_path"], "rb").read() == b"x"
        assert open(c["tile_four_path"], "rb").read() == b"x"
    assert open(spec["working_path"], "rb").read(4)       # the working image too


def test_generic_migration_adds_new_engine_columns_to_an_old_db(tmp_path):
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)       # an old DB: lattice tables from before FieldPrism
    con.executescript("""
        CREATE TABLE ruler_CF_lattice (specimen_id INTEGER PRIMARY KEY, image_name TEXT,
            engine_version TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL,
            cf_px_per_cm REAL, confidence TEXT NOT NULL, qc_image_path TEXT);
        CREATE TABLE ruler_FP_marker (fp_marker_id INTEGER PRIMARY KEY,
            specimen_id INTEGER NOT NULL, detection_id INTEGER NOT NULL, status TEXT NOT NULL);
        INSERT INTO ruler_CF_lattice VALUES (1, 'a', 'v', 't', 'published', 50.0, 'high', NULL);
    """)
    con.commit()
    con.close()

    db = ProjectDB.open_or_create(path)
    cols = {r[1]: r[2] for r in db.conn.execute("PRAGMA table_info(ruler_CF_lattice)")}
    assert {"anchor_source", "anchor_cf_working", "fp_detected", "n_rejected",
            "mp_anchor_working"} <= set(cols)
    assert cols["anchor_cf_working"] == "REAL" and cols["fp_detected"] == "INTEGER"
    mcols = {r[1] for r in db.conn.execute("PRAGMA table_info(ruler_FP_marker)")}
    assert set(fp_marker_columns()) <= mcols
    assert tuple(db.conn.execute("SELECT cf_px_per_cm, anchor_source FROM ruler_CF_lattice")
                 .fetchone()) == (50.0, None)
    db.close()
    ProjectDB.open_or_create(path).close()                 # idempotent on the second open


# ============================================================================================
# the stage
# ============================================================================================
def test_stage_reads_fp_settings_and_ships_the_working_path(tmp_path):
    def stage(cfg):
        return RulerConversionFactor(types.SimpleNamespace(
            modules={}, stage=lambda k: cfg.get(k, {})))

    d = stage({})._settings()
    assert (d["fp_enabled"], d["fp_peer_tol"], d["fp_anchor_tol"], d["fp_allow_single_marker"]) \
        == (True, fp.FP_PEER_TOL, fp.FP_ANCHOR_TOL, True)
    st = stage({"ruler_cf": {"fieldprism": {"enabled": False, "peer_tol": 0.05, "anchor_tol": 0.04,
                                            "allow_single_marker": False}}})
    eng = st._build_engine(tmp_path / "art")
    assert (eng.fp_enabled, eng.fp_peer_tol, eng.fp_anchor_tol, eng.fp_allow_single_marker) \
        == (False, 0.05, 0.04, False)
    # a partial block fills the rest from the defaults; the ruler-lattice anchor_tol is NOT the
    # FieldPrism one, and an explicit null block means "all defaults", not a crash
    d = stage({"ruler_cf": {"anchor_tol": 0.5, "fieldprism": {"peer_tol": 0.05}}})._settings()
    assert (d["fp_enabled"], d["fp_peer_tol"], d["fp_anchor_tol"], d["fp_allow_single_marker"]) \
        == (True, 0.05, fp.FP_ANCHOR_TOL, True)
    assert stage({"ruler_cf": {"fieldprism": None}})._settings()["fp_enabled"] is True

    spec, crops = make_sheet(tmp_path)
    db = _db(tmp_path, spec, crops)
    for dep in ("archival_detector", "ruler_classifier"):
        db.mark_image_done(spec["specimen_id"], dep)
    (item,) = stage({}).collect_items(types.SimpleNamespace(db=db))
    payload_spec, payload_crops = item.payload
    assert payload_spec["working_path"] == spec["working_path"]
    assert {c["ruler_class"] for c in payload_crops} == {"FP"}

    st = stage({"ruler_cf": {"use_CF_predicted_by_MP": True}})
    record, wb = st.infer(item, st._build_engine(tmp_path / "art"))
    assert wb["source"] == CF_SOURCE_FP
    st.persist(types.SimpleNamespace(db=db), item, (record, wb))
    s = db.get_specimen(spec["specimen_id"])
    assert s["cf_source"] == CF_SOURCE_FP and s["ruler_unit_type"] == "FP"
    assert s["cf_px_per_cm"] == pytest.approx(10 * PPMM, rel=0.01)


def test_writeback_published_sources():
    crops = [{"verdict": "used", "ruler_class": "METRIC_MM"}]
    rec = {"image": {"status": "published", "cf_px_per_cm": 80.0, "cf_source": CF_SOURCE_RULER},
           "crops": crops}
    assert _writeback(rec)["source"] == CF_SOURCE_RULER
    rec["image"]["cf_source"] = CF_SOURCE_FP
    assert _writeback(rec)["source"] == CF_SOURCE_FP
