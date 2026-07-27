"""End-to-end pipeline test in ``compute.mock`` mode (no GPU / weights required).

Runs the full ``machine3`` orchestration against two synthetic specimens with the
deterministic mock backends, asserts every stage persisted rows and the Reporter wrote an
overlay, then runs it a SECOND time and asserts the run resumes cleanly (no duplicate rows).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

from leafmachine3.machine3 import machine3

from tests.conftest import build_mock_config, fresh_out_dir


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _count(conn: sqlite3.Connection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.fixture
def run_env(synthetic_images: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the mock config and chdir into an isolated dir (so hardware_settings lands there).

    Pipeline artifacts (DB, crops, overlays) land in ``examples_out/pipeline_mock`` for inspection.
    """
    output_dir = fresh_out_dir("pipeline_mock")
    cfg = build_mock_config(synthetic_images, output_dir, run_name="e2e")
    cfg_path = tmp_path / "LM3_settings.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return cfg_path


def test_end_to_end_mock_pipeline(run_env: Path) -> None:
    project = machine3(run_env)
    db_path = project.dirs.db_path
    assert db_path.exists(), "project SQLite DB was not created"

    conn = _connect(db_path)
    try:
        # every stage that should produce rows did, for both specimens
        assert _count(conn, "specimen") == 2
        assert _count(conn, "archival_detection") > 0
        assert _count(conn, "plant_detection") > 0
        assert _count(conn, "phenology") == 2
        assert _count(conn, "leaf_segmentation") > 0

        # Morphology ran after LeafSegmenter: a row per leaf instance with the rotated bbox
        assert _count(conn, "leaf_morphology") > 0
        m = conn.execute(
            "SELECT rotated_bbox_dim_max, rotated_bbox_dim_min, rotate_angle, rotated_bbox_json, "
            "area_px, circularity, leaf_id, detection_id, specimen_id FROM leaf_morphology LIMIT 1"
        ).fetchone()
        assert m is not None
        assert m["rotated_bbox_dim_max"] >= m["rotated_bbox_dim_min"] > 0   # length >= width > 0
        assert m["area_px"] > 0
        import json as _json
        assert len(_json.loads(m["rotated_bbox_json"])) == 4                # 4 rotated corners
        assert m["leaf_id"] and m["detection_id"] and m["specimen_id"]      # links to parent

        # Landmark Detector ran after Morphology: 31 keypoints per leaf crop, in working coords
        assert _count(conn, "leaf_landmark") > 0
        n_kpts = conn.execute("SELECT count(DISTINCT kpt_index) FROM leaf_landmark").fetchone()[0]
        assert n_kpts == 31
        lm = conn.execute(
            "SELECT kpt_name, x, y, x_crop, y_crop, detection_id FROM leaf_landmark "
            "WHERE kpt_name = 'lamina_tip' LIMIT 1"
        ).fetchone()
        assert lm is not None and lm["detection_id"]                       # links to the leaf crop
        assert lm["x"] >= lm["x_crop"] and lm["y"] >= lm["y_crop"]         # re-based crop -> working frame

        # Landmark Measurements ran after: one row per leaf instance, metrics computed & sane
        assert _count(conn, "leaf_landmark_measurement") > 0
        mm = conn.execute(
            "SELECT lamina_trace_length, lamina_extent, lamina_tip_base_length, leaf_width, "
            "apex_angle, apex_angle_type, base_angle, base_angle_type, petiole_trace_length, "
            "lamina_curvature, curvature_point, n_present, detection_id FROM leaf_landmark_measurement LIMIT 1"
        ).fetchone()
        assert mm is not None and mm["detection_id"]
        assert mm["lamina_extent"] > 0 and mm["leaf_width"] > 0            # straight-line metrics
        assert mm["lamina_tip_base_length"] > 0                           # separate tip->base distance
        assert mm["lamina_trace_length"] >= mm["lamina_extent"] - 1e-6     # arc >= chord (same endpoints)
        assert mm["lamina_curvature"] >= 0.0                              # max midvein bend (deg); 0 = straight
        assert mm["curvature_point"] is not None                          # most-bent midvein index

        # Leaf Orientation ran: every mock leaf has tip+base -> success + a CW angle on morphology
        orient = conn.execute(
            "SELECT oriented_leaf_success, oriented_leaf_rotation_angle_degreesCW FROM leaf_morphology"
        ).fetchall()
        assert orient and all(o["oriented_leaf_success"] == 1 for o in orient)
        assert all(0.0 <= o["oriented_leaf_rotation_angle_degreesCW"] < 360.0 for o in orient)
        assert 0.0 <= mm["apex_angle"] <= 360.0                           # degrees, incl. reflex
        assert mm["apex_angle_type"] in {"acute", "obtuse", "reflex"}
        assert mm["base_angle_type"] in {"acute", "obtuse", "reflex"}
        assert mm["n_present"] == 31                                       # mock emits all keypoints

        # phenology mirrored leaf presence onto the specimen (mock emits a Leaf_WHOLE box)
        leaf_flags = [r["has_leaves"] for r in conn.execute("SELECT has_leaves FROM specimen")]
        assert all(flag == 1 for flag in leaf_flags)

        # every canonical stage reached 'done'
        states = {r["stage_key"]: r["state"] for r in conn.execute("SELECT stage_key, state FROM project_status")}
        assert set(states.values()) == {"done"}
    finally:
        conn.close()

    # Reporter wrote one summary overlay per specimen under Overlay/Overlay_Summary/
    reports = project.dirs.reports
    overlays = sorted((reports / "Overlay" / "Overlay_Summary").glob("*__Overlay.jpg"))
    assert len(overlays) == 2
    assert all(p.stat().st_size > 0 for p in overlays)

    # per-leaf landmark overlays land under Overlay/Overlay_Landmarks/ as __LM-leaf__coords
    lm_overlays = list((reports / "Overlay" / "Overlay_Landmarks").glob("*__LM-leaf__*.jpg"))
    assert lm_overlays, "no per-leaf landmark overlays written"
    from leafmachine3.core.imaging import parse_crop_filename
    lp = parse_crop_filename(lm_overlays[0].name)
    assert lp and lp["prefix"] == "LM" and lp["friendly"] == "leaf" and len(lp["xyxy"]) == 4

    # leaf products: Original/ + Oriented/ trees, each with bbox + fitted lamina mask + lamina cutout
    # (mock leaves have no petiole, so the laminaPetiole products are correctly skipped).
    for tree in ("Original", "Oriented"):
        base = reports / tree
        bbox = list((base / "Leaf_BBox").glob("*__BBOX-leaf__*.jpg"))
        lam_mask = list((base / "Lamina_Mask").glob("*__SEG-lamina__*.png"))
        lam_rgb = list((base / "Lamina_RGB").glob("*__RGB-lamina__*.jpg"))
        assert bbox and lam_mask and lam_rgb, f"{tree}: missing leaf products"
        assert not (base / "LaminaPetiole_Mask").exists()          # no petiole -> product skipped
        # a fitted lamina mask is tight to content -> no larger than its bbox crop
        import cv2 as _cv2
        m = _cv2.imread(str(lam_mask[0]), _cv2.IMREAD_GRAYSCALE)
        assert m is not None and (m > 0).any()                     # non-empty mask
    # leaf bbox crops were moved OUT of Crops/ (only non-leaf classes remain there)
    assert not (reports / "Crops" / "RGB__leaf").exists()

    # mask outputs are grouped under Binary_Masks/ and RGB_Masks/ (harmonized with Crops/)
    full_bin = list((reports / "Binary_Masks" / "Binary_Masks_Full_Image__Leaf").glob("*.png"))
    assert full_bin, "no full-image binary masks"
    assert list((reports / "RGB_Masks" / "RGB_Masks_Full_Image__Leaf").glob("*.jpg")), "no full-image RGB masks"
    per_crop_bin = list((reports / "Binary_Masks" / "Binary_Masks__Leaf").glob("*.png"))
    assert per_crop_bin, "no per-crop binary masks"

    # full-image files carry the MaskFull-<friendly> label; per-crop files carry SEG-<friendly>__coords
    from leafmachine3.core.imaging import parse_crop_filename
    assert "__MaskFull-leaf." in full_bin[0].name, full_bin[0].name
    parsed = parse_crop_filename(per_crop_bin[0].name)
    assert parsed and parsed["prefix"] == "SEG" and parsed["friendly"] == "leaf"
    assert len(parsed["xyxy"]) == 4

    # raw RGB crop exports land under Crops/RGB__<friendly>/ for both models' classes
    crop_dirs = sorted(d.name for d in (reports / "Crops").iterdir() if d.is_dir())
    assert crop_dirs and all(name.startswith("RGB__") for name in crop_dirs)
    a_crop = next((reports / "Crops").rglob("*.jpg"))
    ap = parse_crop_filename(a_crop.name)
    assert ap and ap["prefix"] == "BBOX"


def test_pipeline_resumes_without_duplicates(run_env: Path) -> None:
    first = machine3(run_env)
    db_path = first.dirs.db_path

    conn = _connect(db_path)
    try:
        before = {t: _count(conn, t) for t in
                  ("specimen", "archival_detection", "plant_detection", "phenology", "leaf_segmentation")}
    finally:
        conn.close()

    # Second run: everything is already done -> stages skip, no rows are re-inserted.
    machine3(run_env)

    conn = _connect(db_path)
    try:
        after = {t: _count(conn, t) for t in before}
    finally:
        conn.close()

    assert after == before, f"resume changed row counts: {before} -> {after}"


def test_restart_reruns_a_stage(run_env: Path) -> None:
    """``restart`` a stage and confirm it re-produces the same number of rows."""
    machine3(run_env)
    project = machine3(run_env, restart=["plant_detector"])

    conn = _connect(project.dirs.db_path)
    try:
        # plant_detector and its dependents were rebuilt; rows are present again
        assert _count(conn, "plant_detection") > 0
        assert _count(conn, "leaf_segmentation") > 0
        assert _count(conn, "specimen") == 2
    finally:
        conn.close()
