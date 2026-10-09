"""End-to-end pipeline test in ``compute.mock`` mode (no GPU / weights required).

Runs the full ``machine3`` orchestration against two synthetic specimens with the
deterministic mock backends, asserts every stage persisted rows and the Reporter wrote an
overlay, then runs it a SECOND time and asserts the run resumes cleanly (no duplicate rows).
"""
from __future__ import annotations

import csv as _csv
import json
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
    cfg = build_mock_config(synthetic_images, output_dir, run_name="e2e", ruler_classifier_enabled=True)
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
            "area_px, lamina_area_incl_holes_px, lamina_area_excl_holes_px, lamina_hole_area_px, "
            "n_holes, circularity, leaf_id, detection_id, specimen_id FROM leaf_morphology LIMIT 1"
        ).fetchone()
        assert m is not None
        assert m["rotated_bbox_dim_max"] >= m["rotated_bbox_dim_min"] > 0   # length >= width > 0
        assert m["area_px"] > 0
        # hole-aware areas: incl == outer silhouette (area_px); the mock carves one hole per leaf
        assert m["lamina_area_incl_holes_px"] == m["area_px"]
        assert m["n_holes"] >= 1 and m["lamina_hole_area_px"] > 0
        assert m["lamina_area_excl_holes_px"] < m["lamina_area_incl_holes_px"]   # tissue < silhouette
        assert m["lamina_area_excl_holes_px"] == pytest.approx(
            m["lamina_area_incl_holes_px"] - m["lamina_hole_area_px"], abs=1.0)
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

        # Petiole Width ran: the mock petiole yields a measured width, touches the leaf, has samples
        pet = conn.execute(
            "SELECT width_px, length_px, n_samples, touches_leaf, width_segment_json FROM leaf_petiole"
        ).fetchall()
        assert pet, "no leaf_petiole rows"
        assert all(p["width_px"] is not None and p["width_px"] > 0 for p in pet)
        assert all(p["touches_leaf"] == 1 for p in pet)
        assert all(p["n_samples"] >= 1 and p["width_segment_json"] for p in pet)
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

        # mp_conversion_factor (runs first) populated megapixels + the resolution-based CF everywhere
        mp_rows = list(conn.execute("SELECT original_mp, cf_px_per_cm_predicted_by_mp FROM specimen"))
        assert mp_rows and all(r["original_mp"] is not None and r["original_mp"] > 0 for r in mp_rows)
        assert all(r["cf_px_per_cm_predicted_by_mp"] is not None and r["cf_px_per_cm_predicted_by_mp"] > 0
                   for r in mp_rows)

        # ruler_classifier wrote the per-specimen consensus class onto the specimen (mock -> METRIC_MM)
        ruler_classes = [r["ruler_class_type"] for r in conn.execute("SELECT ruler_class_type FROM specimen")]
        assert ruler_classes and all(v == "METRIC_MM" for v in ruler_classes)
        # ...and pre-made a four-tile squarify collage per Ruler crop (reused by the lattice CF)
        sq = list(conn.execute("SELECT squarify_path FROM ruler_classification"))
        assert sq and all(r["squarify_path"] and Path(r["squarify_path"]).exists() for r in sq)

        # ruler_cf lattice stage wrote one image row + per-crop rows per sheet with rulers
        lat = list(conn.execute("SELECT specimen_id, status, n_ruler_crops FROM ruler_CF_lattice"))
        assert lat and all(r["status"] in ("published", "withheld", "no_reading", "no_ruler") for r in lat)
        crop_rows = conn.execute("SELECT COUNT(*) FROM ruler_CF_lattice_crop").fetchone()[0]
        assert crop_rows >= len(lat)                     # >= one crop row per sheet
        # gate honoured: cf_px_per_cm is set iff the sheet published (NULL otherwise -> MP fallback)
        for r in conn.execute("SELECT s.cf_px_per_cm, l.status FROM specimen s "
                              "JOIN ruler_CF_lattice l USING (specimen_id)"):
            assert (r["cf_px_per_cm"] is not None) == (r["status"] == "published")

        # ECT stage: at least one oriented leaf got an ECT (loaded from the Reporter's Leaf_Oriented masks)
        ect_rows = list(conn.execute("SELECT h5_path, mask_includes, num_dirs FROM leaf_ect"))
        assert ect_rows and all(r["mask_includes"] == "lamina" for r in ect_rows)

        # Momocs stage: every exported leaf has its JPG on disk, listed in the run-level files
        mom_rows = list(conn.execute("SELECT mask_path, json_path, tree, mask_includes FROM leaf_momocs"))
        assert mom_rows and all(r["tree"] == "Leaf_Oriented" and r["mask_includes"] == "lamina" for r in mom_rows)
        assert all(Path(r["mask_path"]).is_file() and Path(r["json_path"]).is_file() for r in mom_rows)
        mom_dir = Path(mom_rows[0]["mask_path"]).parent
        fac = (mom_dir / "momocs_fac.csv").read_text().splitlines()
        assert len(fac) == len(mom_rows) + 1                  # header + one row per image
        run_json = json.loads((mom_dir / "momocs_outlines.json").read_text())
        assert run_json["metadata"]["n_rows"] == len(mom_rows)
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

    # per-leaf petiole overlays land under Overlay/Overlay_Petiole/ as __PET-leaf__coords
    pet_overlays = list((reports / "Overlay" / "Overlay_Petiole").glob("*__PET-leaf__*.jpg"))
    assert pet_overlays, "no per-leaf petiole overlays"
    pp = parse_crop_filename(pet_overlays[0].name)
    assert pp and pp["prefix"] == "PET" and pp["friendly"] == "leaf"

    # per-specimen segmentation overlay lands under Overlay/Overlay_Specimen_Segmentation/
    spec_overlays = sorted((reports / "Overlay" / "Overlay_Specimen_Segmentation").glob("*__SpecimenSeg.jpg"))
    assert len(spec_overlays) == 2 and all(p.stat().st_size > 0 for p in spec_overlays)

    # per-sheet lattice ruler-CF QC panels (rebuilt from the stored record) under Overlay/Overlay_Ruler_Lattice/
    ruler_lat = sorted((reports / "Overlay" / "Overlay_Ruler_Lattice").glob("*__RulerLattice.png"))
    assert ruler_lat and all(p.stat().st_size > 0 for p in ruler_lat), "no ruler lattice QC overlays"

    # whole-specimen masks land under Specimen_Masks/ (one per specimen); every mask tree --
    # binary and RGB, full-image, per-crop and whole-specimen -- shares that single parent.
    import cv2 as _cv2
    import numpy as _np
    masks_root = reports / "Specimen_Masks"
    spec_bin = sorted((masks_root / "Binary_Masks_Specimen").glob("*__MaskFull-specimen.png"))
    spec_rgb = sorted((masks_root / "RGB_Masks_Specimen").glob("*__MaskRGBFull-specimen.jpg"))
    assert len(spec_bin) == 2 and len(spec_rgb) == 2
    _sm = _cv2.imread(str(spec_bin[0]), _cv2.IMREAD_GRAYSCALE)
    assert _sm is not None and (_sm > 0).any()                     # non-empty specimen mask

    # ...and its complement. The binary inverse must partition the sheet with the mask above:
    # every pixel is in exactly one of them, which is the whole claim the folder name makes.
    inv_bin = sorted((masks_root / "Binary_Masks_Specimen_Inverse").glob("*__MaskFull-specimenInverse.png"))
    inv_rgb = sorted((masks_root / "RGB_Masks_Specimen_Inverse").glob("*__MaskRGBFull-specimenInverse.jpg"))
    assert len(inv_bin) == 2 and len(inv_rgb) == 2, "not-specimen exports missing"
    _im = _cv2.imread(str(inv_bin[0]), _cv2.IMREAD_GRAYSCALE)
    assert _im is not None and _im.shape == _sm.shape
    assert _np.array_equal(_im > 0, _sm == 0), "inverse mask is not the complement of the mask"
    assert (_im > 0).any() and (_im == 0).any()                    # a real partition, not all-or-nothing

    # the RGB inverse paints the plant out with report.masks.inverse_fill ([255, 0, 0] RGB in the
    # mock config), so the plant region must come back as RED -- not as the black `background`,
    # which is the bug this guards (the two colors are separate settings and easy to cross).
    # Compared as a MEDIAN, not for equality: the file is JPEG, and chroma subsampling rings hard
    # along a saturated-red silhouette edge (deviations up to 121 there, ~1 in the interior).
    _ir = _cv2.imread(str(inv_rgb[0]), _cv2.IMREAD_COLOR)          # BGR
    assert _ir is not None and _ir.shape[:2] == _sm.shape
    _plant = _sm > 0
    _med = _np.median(_ir[_plant], axis=0)
    assert _np.all(_np.abs(_med - _np.array([0, 0, 255])) <= 3), \
        f"inverse fill color was not applied (median BGR {_med}, wanted [0, 0, 255])"
    # ...and the sheet OUTSIDE the plant is untouched, or this would be a solid red rectangle
    assert not _np.all(_np.median(_ir[~_plant], axis=0) == _med)

    # every export is in the DB manifest under its own kind -- that is what --restart deletes by,
    # so a file on disk with no manifest row would survive a reset and go stale in place.
    _mc = _connect(db_path)
    try:
        for _kind in ("Binary_Masks_Specimen_Inverse", "RGB_Masks_Specimen_Inverse"):
            _rows = _mc.execute("SELECT path FROM report_manifest WHERE kind = ?",
                                (f"Specimen_Masks/{_kind}",)).fetchall()
            assert len(_rows) == 2, f"{_kind}: {len(_rows)} manifest rows, expected 2"
            assert all(Path(r["path"]).is_file() for r in _rows), f"{_kind}: manifest path missing"
    finally:
        _mc.close()

    # leaf products: Leaf_Original/ + Leaf_Oriented/ trees, each with bbox + fitted lamina mask + cutout
    # (mock leaves have no petiole, so the laminaPetiole products are correctly skipped).
    # Each product name carries its tree tag (og-/or-), which is what keeps the two trees' files
    # distinct -- they are otherwise the same stem, prefix and box.
    for tree, tag in (("Leaf_Original", "og"), ("Leaf_Oriented", "or")):
        base = reports / tree
        bbox = list((base / "Leaf_BBox").glob(f"*__{tag}-BBOX-leaf__*.jpg"))
        lam_mask = list((base / "Lamina_Mask").glob(f"*__{tag}-SEG-lamina__*.png"))
        lam_rgb = list((base / "Lamina_RGB").glob(f"*__{tag}-RGB-lamina__*.jpg"))
        holes_mask = list((base / "Lamina_Holes_Mask").glob(f"*__{tag}-SEG-laminaHoles__*.png"))
        holes_rgb = list((base / "Lamina_Holes_RGB").glob(f"*__{tag}-RGB-laminaHoles__*.jpg"))
        assert bbox and lam_mask and lam_rgb, f"{tree}: missing lamina products"
        assert holes_mask and holes_rgb, f"{tree}: missing laminaHoles products"
        # the mock now emits a petiole, so the laminaPetiole products are present
        assert list((base / "LaminaPetiole_Mask").glob(f"*__{tag}-SEG-laminaPetiole__*.png"))
        assert list((base / "LaminaPetiole_RGB").glob(f"*__{tag}-RGB-laminaPetiole__*.jpg"))
        m = _cv2.imread(str(lam_mask[0]), _cv2.IMREAD_GRAYSCALE)
        assert m is not None and (m > 0).any()                     # non-empty mask
        # the holes RGB paints holes (10,10,10) so they can be color-thresholded back out
        hr = _cv2.imread(str(holes_rgb[0]), _cv2.IMREAD_COLOR)      # BGR; (10,10,10) is symmetric
        assert hr is not None and bool((_np.all(hr == (10, 10, 10), axis=2)).any())
    # leaf bbox crops were moved OUT of Crops/ (only non-leaf classes remain there)
    assert not (reports / "Crops" / "RGB__leaf").exists()

    # every mask subfolder now sits under the single Specimen_Masks/ parent; the old sibling
    # Binary_Masks/ and RGB_Masks/ trees must be gone, or consumers would read a stale layout.
    assert not (reports / "Binary_Masks").exists() and not (reports / "RGB_Masks").exists()
    full_bin = list((masks_root / "Binary_Masks_Specimen__Leaf").glob("*.png"))
    assert full_bin, "no full-image binary masks"
    assert list((masks_root / "RGB_Masks_Specimen__Leaf").glob("*.jpg")), "no full-image RGB masks"
    per_crop_bin = list((masks_root / "Binary_Masks__Leaf").glob("*.png"))
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

    # Data export: the Reporter's last step turns the whole database into reports/Data/*.csv.
    # Checked end-to-end here because the export is the one output assembled from EVERY specimen's
    # rows at once -- a per-file unit test cannot catch a stage that failed to commit.
    data_dir = reports / "Data"
    leaf_csv = data_dir / "leaf_measurements.csv"
    assert leaf_csv.is_file(), "the Reporter did not write reports/Data/leaf_measurements.csv"
    with leaf_csv.open(newline="", encoding="utf-8") as fh:
        leaf_rows = list(_csv.DictReader(fh))
    conn = _connect(db_path)
    try:
        n_leaves = conn.execute(
            "SELECT COUNT(*) FROM leaf_segmentation ls JOIN plant_detection pd USING (detection_id) "
            "WHERE ls.cls_name = 'Leaf' AND pd.suppressed = 0").fetchone()[0]
    finally:
        conn.close()
    assert len(leaf_rows) == n_leaves, "one CSV row per segmented leaf"
    # Each row points back at real files by the token the images are named with. Rebuilt with the
    # same parser the naming contract uses, so this breaks if either side changes its format.
    tokens = set()
    for p in (reports / "Leaf_Original" / "Leaf_BBox").glob("*.jpg"):
        parsed = parse_crop_filename(p.name)
        assert parsed, f"unparseable leaf product name: {p.name}"
        tokens.add(parsed["stem"] + "__" + "_".join(str(v) for v in parsed["xyxy"]))
    assert tokens, "no leaf products to cross-reference"
    for row in leaf_rows:
        assert row["leaf_uid"].startswith(row["crop_file_token"] + "__i")
        assert row["crop_file_token"] in tokens, (
            f"crop_file_token {row['crop_file_token']!r} matches no exported leaf product")
    # and the sheet-level roll-up agrees with the per-leaf file it was computed from
    with (data_dir / "specimen_summary.csv").open(newline="", encoding="utf-8") as fh:
        summary = list(_csv.DictReader(fh))
    assert sum(int(r["n_leaf_instances"]) for r in summary) == len(leaf_rows)

    # THE naming contract: every derived file in the run is uniquely named WITHOUT its extension,
    # so a user can pour the whole reports tree into one directory and lose nothing. Extensions are
    # stripped before comparing because .png/.jpg twins (a binary mask and its RGB cutout) are
    # exactly the pairs that used to collide -- they now differ by prefix (SEG/SEGRGB, MaskFull/
    # MaskRGBFull) rather than by suffix. Failures print the offenders, since the useful question
    # is always "which two outputs share a token", not "how many".
    # The one sanctioned exception: Leaf_Data/Coordinates/<...>__ECT__<box>.h5 shares the Cartesian
    # ECT image's token on purpose (it is coordinate data, not a picture). So the contract is checked
    # over IMAGES -- if a future .h5-like data export starts colliding with a picture, it shows up here.
    _IMG = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    by_base: dict[str, list[str]] = {}
    for f in reports.rglob("*"):
        if f.is_file() and f.suffix.lower() in _IMG:
            by_base.setdefault(f.stem, []).append(str(f.relative_to(reports)))
    clashes = {b: sorted(v) for b, v in by_base.items() if len(v) > 1}
    assert not clashes, "report filenames collide once flattened:\n" + "\n".join(
        f"  {b}: {v}" for b, v in sorted(clashes.items())[:20])


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
