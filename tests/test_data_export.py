"""reports/Data -- the CSV export of the project database.

The export is a PROJECTION: it must show what the pipeline actually stored, and must not invent a
value the pipeline never produced. So the tests here mostly guard against three specific ways a
projection goes wrong:

  * the exporter and the SQL drift apart, and a declared column silently exports as empty;
  * a missing measurement (occluded keypoint, withheld conversion factor) is written as 0 or "nan"
    instead of as an absence;
  * the folder accumulates files from an earlier, differently-configured run.
"""
from __future__ import annotations

import csv
import types

import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import SpecimenRecord
from leafmachine3.reporting import data_export
from leafmachine3.reporting.data_export import FILES, export_data_csvs


# --------------------------------------------------------------------------- #
# fixtures: a small but REAL project database
# --------------------------------------------------------------------------- #
@pytest.fixture()
def project(tmp_path):
    """A project with two sheets: one fully measured and grounded, one bare.

    The second sheet is the important one -- it has no CF, no landmarks and no petiole, which is
    exactly the row shape that tempts an exporter into writing zeros.
    """
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    for i, stem in enumerate(("sheetA", "sheetB"), start=1):
        db.upsert_specimen(SpecimenRecord(
            image_name=f"{stem}.jpg", image_stem=stem,
            original_path=f"/orig/{stem}.jpg", working_path=f"/work/{stem}.jpg",
            width=1000, height=2000, original_width=1000, original_height=2000,
            work_scale=1.0,
        ))
    conn = db.conn
    # sheetA gets a published CF; sheetB deliberately does not.
    conn.execute("UPDATE specimen SET cf_px_per_cm = 10.0, cf_source = 'measured_from_ruler', "
                 "cf_px_per_cm_predicted_by_mp = 9.0 WHERE image_stem = 'sheetA'")
    conn.execute("UPDATE specimen SET cf_px_per_cm_predicted_by_mp = 9.5 WHERE image_stem = 'sheetB'")

    for sid, (x1, y1, x2, y2) in ((1, (10, 20, 110, 220)), (2, (30, 40, 130, 240))):
        conn.execute(
            "INSERT INTO plant_detection (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2) "
            "VALUES (?, 0, 'Leaf_WHOLE', 0.9, ?, ?, ?, ?)", (sid, x1, y1, x2, y2))
        conn.execute(
            "INSERT INTO leaf_segmentation (specimen_id, detection_id, instance_index, cls_name, "
            " cls_id, mask_format, mask_data, frame_width, frame_height, area_px, perimeter_px) "
            "VALUES (?, ?, 0, 'Leaf', 0, 'polygon_xy', '[[0,0],[1,0],[1,1]]', 1000, 2000, 400.0, 80.0)",
            (sid, sid))
        conn.execute(
            "INSERT INTO leaf_morphology (leaf_id, specimen_id, detection_id, instance_index, "
            " cls_name, area_px, perimeter_px, lamina_area_incl_holes_px, "
            " lamina_area_excl_holes_px, lamina_hole_area_px, n_holes, bbox_x1, bbox_y1, "
            " bbox_x2, bbox_y2, oriented_leaf_success) "
            "VALUES (?, ?, ?, 0, 'Leaf', 400.0, 80.0, 400.0, 380.0, 20.0, 1, 0, 0, 20, 40, 1)",
            (sid, sid, sid))
    # only sheetA is measured + grounded
    conn.execute(
        "INSERT INTO leaf_landmark_measurement (specimen_id, detection_id, instance_index, "
        " lamina_trace_length, lamina_extent, leaf_width, n_present, lamina_trace_length_cm) "
        "VALUES (1, 1, 0, 300.0, 280.0, NULL, 25, 30.0)")
    conn.execute("UPDATE leaf_segmentation SET area_cm2 = 4.0, perimeter_cm = 8.0 WHERE specimen_id = 1")
    conn.execute(
        "INSERT INTO leaf_petiole (leaf_id, specimen_id, detection_id, instance_index, width_px, "
        " length_px, width_cm) VALUES (1, 1, 1, 0, 20.0, 100.0, 2.0)")
    return types.SimpleNamespace(db=db, dirs=types.SimpleNamespace(reports=tmp_path / "reports"))


def _cfg(**data):
    return types.SimpleNamespace(report={"data": {"enabled": True, **data}})


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _data_dir(project, folder="Data"):
    return project.dirs.reports / folder


# --------------------------------------------------------------------------- #
# the bundle
# --------------------------------------------------------------------------- #
def test_writes_every_enabled_file_with_a_header(project):
    written = export_data_csvs(project, _cfg())
    assert {p.name for p in written} == {f"{f.stem}.csv" for f in FILES}
    for path in written:
        assert path.read_text(encoding="utf-8").splitlines(), f"{path.name} is empty"


def test_empty_table_still_gets_its_header(project):
    """A zero-row CSV with columns says "exported, nothing matched"; a zero-byte one says nothing."""
    export_data_csvs(project, _cfg())
    rows = (_data_dir(project) / "stage_errors.csv").read_text().splitlines()
    assert rows[0].startswith("image_stem,specimen_id,stage_key")
    assert len(rows) == 1


def test_disabled_export_writes_nothing(project):
    assert export_data_csvs(project, types.SimpleNamespace(
        report={"data": {"enabled": False}})) == []
    assert not _data_dir(project).exists()


def test_tsv_format_and_custom_folder(project):
    written = export_data_csvs(project, _cfg(format="tsv", folder="Tables"))
    leaf = _data_dir(project, "Tables") / "leaf_measurements.tsv"
    assert leaf in written
    assert "\t" in leaf.read_text().splitlines()[0]


def test_rerun_removes_files_a_toggle_turned_off(project):
    """The folder must describe the CURRENT database, not accumulate old exports."""
    export_data_csvs(project, _cfg())
    stale = _data_dir(project) / "landmarks.csv"
    assert stale.is_file()

    export_data_csvs(project, _cfg(files={"landmarks": False}))
    assert not stale.exists()
    assert (_data_dir(project) / "leaf_measurements.csv").is_file()


def test_switching_format_does_not_leave_the_old_files_behind(project):
    export_data_csvs(project, _cfg())
    assert (_data_dir(project) / "leaf_measurements.csv").is_file()
    export_data_csvs(project, _cfg(format="tsv"))
    assert not (_data_dir(project) / "leaf_measurements.csv").exists()
    assert (_data_dir(project) / "leaf_measurements.tsv").is_file()


# --------------------------------------------------------------------------- #
# leaf_measurements.csv -- the master file
# --------------------------------------------------------------------------- #
def test_one_row_per_leaf_with_identity_and_measurements(project):
    export_data_csvs(project, _cfg())
    rows = _read(_data_dir(project) / "leaf_measurements.csv")
    assert len(rows) == 2

    a = next(r for r in rows if r["image_stem"] == "sheetA")
    assert a["leaf_uid"] == "sheetA__10_20_110_220__i0"
    assert a["crop_file_token"] == "sheetA__10_20_110_220"
    assert a["lamina_area_incl_holes_px"] == "400"
    assert a["lamina_area_excl_holes_px"] == "380"
    assert a["n_holes"] == "1"
    assert a["lamina_area_incl_holes_cm2"] == "4"
    assert a["petiole_width_cm"] == "2"
    assert a["lamina_trace_length_px"] == "300"
    assert a["lamina_trace_length_cm"] == "30"


def test_leaf_uid_is_rebuilt_from_the_image_and_box_not_the_row_id(project):
    """leaf_id is a project-local autoincrement; leaf_uid must survive a re-run, so it is derived
    from the stem and the detection box -- the same token the exported image files carry."""
    export_data_csvs(project, _cfg())
    for r in _read(_data_dir(project) / "leaf_measurements.csv"):
        assert r["leaf_uid"] == f"{r['crop_file_token']}__i{r['instance_index']}"
        assert r["leaf_id"] not in r["leaf_uid"].split("__")


def test_missing_measurements_are_empty_never_zero(project):
    """sheetB has no CF, no landmarks and no petiole. Those columns must be blank."""
    export_data_csvs(project, _cfg())
    b = next(r for r in _read(_data_dir(project) / "leaf_measurements.csv")
             if r["image_stem"] == "sheetB")
    for col in ("lamina_area_incl_holes_cm2", "lamina_perimeter_cm", "cf_px_per_cm",
                "lamina_trace_length_px", "lamina_trace_length_cm", "petiole_width_px",
                "petiole_width_cm"):
        assert b[col] == "", f"{col} should be empty, got {b[col]!r}"
    # ...but the pixel measurements it DOES have are present
    assert b["lamina_area_incl_holes_px"] == "400"


def test_a_present_metric_with_one_missing_component_stays_partial(project):
    """sheetA measured a trace but not a width: the width must stay blank, not inherit the trace."""
    export_data_csvs(project, _cfg())
    a = next(r for r in _read(_data_dir(project) / "leaf_measurements.csv")
             if r["image_stem"] == "sheetA")
    assert a["lamina_extent_px"] == "280"
    assert a["leaf_width_px"] == ""


def test_cf_source_names_what_produced_the_cm_columns(project):
    """cf_source is the STORED specimen.cf_source: a measured sheet says so, and a sheet with no CF
    says 'none' -- its MP prediction is carried alongside but is not its grounding CF."""
    export_data_csvs(project, _cfg())
    rows = {r["image_stem"]: r for r in _read(_data_dir(project) / "leaf_measurements.csv")}
    assert rows["sheetA"]["cf_source"] == "measured_from_ruler"
    assert rows["sheetA"]["cf_px_per_cm"] == "10"
    assert rows["sheetB"]["cf_source"] == "none"
    # sheetB HAS an MP prediction, and it must not be mistaken for a grounding CF
    assert rows["sheetB"]["cf_px_per_cm_predicted_by_mp"] == "9.5"
    assert rows["sheetB"]["cf_px_per_cm"] == ""
    assert rows["sheetB"]["lamina_area_incl_holes_cm2"] == ""
    spec = {r["image_stem"]: r for r in _read(_data_dir(project) / "specimen_summary.csv")}
    assert spec["sheetA"]["cf_source"] == "measured_from_ruler"
    assert spec["sheetB"]["cf_source"] == "none"


def test_cf_source_marks_a_megapixel_prediction(project):
    """With use_CF_predicted_by_MP the stage writes the MP value INTO cf_px_per_cm. The export must
    say it was predicted, not measured -- from the stored column, since the value alone can't tell."""
    project.db.set_specimen_cf(2, 9.5, source="predicted_from_megapixels")
    export_data_csvs(project, _cfg())
    for name in ("leaf_measurements.csv", "specimen_summary.csv"):
        b = next(r for r in _read(_data_dir(project) / name) if r["image_stem"] == "sheetB")
        assert b["cf_source"] == "predicted_from_megapixels"
        assert b["cf_px_per_cm"] == "9.5"
        assert b["ruler_unit_type"] == ""


def test_no_export_column_still_claims_to_be_the_ruler_cf():
    """cf_px_per_cm can hold a PREDICTED value now, so the old cf_px_per_cm_ruler name would lie."""
    names = {c.name for f in FILES for c in (f.columns or ())}
    assert "cf_px_per_cm_ruler" not in names
    assert {"cf_px_per_cm", "cf_source"} <= names


def test_no_dead_cm_columns_from_leaf_morphology(project):
    """leaf_morphology's four never-written cm columns were dropped on purpose; if they come back,
    the export would ship all-empty columns that read as a bug."""
    export_data_csvs(project, _cfg())
    header = _read(_data_dir(project) / "leaf_measurements.csv")[0].keys()
    assert "lamina_area_incl_holes_cm2" in header      # the real one, from leaf_segmentation
    assert "length_cm" not in header and "width_cm" not in header
    assert "area_cm2" not in header and "perimeter_cm" not in header


def test_ect_columns_are_not_exported(project):
    """ECT runs AFTER the Reporter (it consumes the Reporter's oriented masks), so at export time
    its rows are either absent or left over from a previous run. Exporting them would ship a column
    that is stale exactly when it looks freshest."""
    header = set(_read_header(project))
    ect_cols = {"h5_path", "radial_png", "ect_png", "overlay_png", "mask_includes",
                "n_outline_points", "num_dirs"}
    assert not (header & ect_cols)


def _read_header(project):
    export_data_csvs(project, _cfg())
    with open(_data_dir(project) / "leaf_measurements.csv", newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


# --------------------------------------------------------------------------- #
# the other files
# --------------------------------------------------------------------------- #
def test_specimen_summary_rollups_agree_with_the_leaf_file(project):
    """The counts are computed from the same rows the leaf file is written from, so they cannot
    disagree -- this test is what keeps that true."""
    export_data_csvs(project, _cfg())
    leaves = _read(_data_dir(project) / "leaf_measurements.csv")
    for s in _read(_data_dir(project) / "specimen_summary.csv"):
        mine = [x for x in leaves if x["specimen_id"] == s["specimen_id"]]
        assert int(s["n_leaf_instances"]) == len(mine)
        assert int(s["n_leaves_grounded_cm"]) == sum(
            1 for x in mine if x["lamina_area_incl_holes_cm2"])


def test_specimen_summary_median_is_blank_when_nothing_was_grounded(project):
    export_data_csvs(project, _cfg())
    b = next(r for r in _read(_data_dir(project) / "specimen_summary.csv")
             if r["image_stem"] == "sheetB")
    assert b["median_lamina_area_incl_holes_cm2"] == ""
    assert b["median_lamina_area_incl_holes_px"] == "400"


# --------------------------------------------------------------------------- #
# the bundle is settable: code defaults, builtin_defaults and the shipped YAML agree
# --------------------------------------------------------------------------- #
# The settings form renders a row only for a leaf of the MERGED config tree, so a knob that
# exists solely as a Python fallback is invisible in the GUI and unsettable from the file.
# report.data was in that state: ten documented CSV toggles that rendered nowhere. Now the same
# defaults live in three places, and these tests are what stop them drifting apart.
def test_builtin_defaults_expose_every_export_file_as_a_toggle(project):
    from leafmachine3.core.config import builtin_defaults
    files = builtin_defaults()["report"]["data"]["files"]
    assert set(files) == {f.key for f in FILES}, (
        "builtin_defaults and data_export.FILES disagree about which files exist, so a file is "
        "either unsettable or has a toggle that controls nothing"
    )
    for f in FILES:
        assert files[f.key] == f.default, f"{f.key}: default disagrees with ExportFile.default"


def test_shipped_settings_yaml_exposes_every_export_file_as_a_toggle():
    """The GUI walks the merged tree, but the FILE is what a user edits and what a run reads."""
    import pathlib
    import yaml
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in ("LM3_settings.yaml",):
        cfg = yaml.safe_load((root / name).read_text())
        data = cfg.get("report", {}).get("data")
        assert data is not None, f"{name} has no report.data block, so the CSV bundle is unsettable"
        assert set(data["files"]) == {f.key for f in FILES}, f"{name}: report.data.files is stale"
        for f in FILES:
            assert data["files"][f.key] == f.default, f"{name}: {f.key} disagrees with the code default"


# --------------------------------------------------------------------------- #
# phenology.csv -- the LeafMachine2-compatible file
# --------------------------------------------------------------------------- #
#: LM2's phenology.csv header, transcribed from utils_detect_phenology.py's fieldnames. The whole
#: point of the file is that this list keeps reading it, so it is pinned here as a literal rather
#: than derived from anything LM3 owns -- otherwise a rename on our side would silently "fix" the
#: test while breaking every LM2 script the file exists to serve.
_LM2_HEADER = [
    "file_name", "leaf_whole", "leaf_partial", "leaflet", "seed_fruit_one", "seed_fruit_many",
    "flower_one", "flower_many", "bud", "specimen", "roots", "wood", "has_leaves", "is_fertile",
]


def _phenology(project):
    with (_data_dir(project) / "phenology.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], {r[0]: dict(zip(rows[0], r)) for r in rows[1:]}


def test_phenology_starts_with_lm2s_header_in_lm2s_order(project):
    export_data_csvs(project, _cfg())
    header, _ = _phenology(project)
    assert header[:len(_LM2_HEADER)] == _LM2_HEADER


def test_phenology_counts_kept_boxes_per_class(project):
    """One row per sheet, counting the plant boxes -- LM2 counted lines in the label file."""
    project.db.conn.execute(
        "INSERT INTO plant_detection (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2) "
        "VALUES (1, 6, 'Bud', 0.7, 5, 5, 15, 15)")
    export_data_csvs(project, _cfg())
    _, rows = _phenology(project)
    assert rows["sheetA.jpg"]["leaf_whole"] == "1"
    assert rows["sheetA.jpg"]["bud"] == "1"
    assert rows["sheetA.jpg"]["flower_one"] == "0"
    assert rows["sheetB.jpg"]["bud"] == "0"


def test_phenology_excludes_suppressed_duplicates(project):
    """A box suppressed as a same-class duplicate is not a second organ, so it must not be counted
    -- otherwise every duplicate inflates the phenological signal this file exists to report."""
    project.db.conn.execute(
        "INSERT INTO plant_detection (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2, "
        " suppressed, suppressed_by) VALUES (1, 0, 'Leaf_WHOLE', 0.4, 11, 21, 111, 221, 1, 1)")
    export_data_csvs(project, _cfg())
    _, rows = _phenology(project)
    assert rows["sheetA.jpg"]["leaf_whole"] == "1"
    assert rows["sheetA.jpg"]["n_plant_detections"] == "1"


def test_phenology_classes_lm3_cannot_detect_are_blank_never_zero(project):
    """`leaflet` and `specimen` are not classes of LM3's detector. A 0 would assert we looked and
    found none; the truth is the detector cannot emit them at all, so they must read as absent."""
    export_data_csvs(project, _cfg())
    _, rows = _phenology(project)
    for stem in ("sheetA.jpg", "sheetB.jpg"):
        assert rows[stem]["leaflet"] == ""
        assert rows[stem]["specimen"] == ""


def test_phenology_is_fertile_is_flowers_or_fruits(project):
    from leafmachine3.core.records import PhenologyResult
    project.db.record_phenology(1, PhenologyResult(leaves=(True, 1), flowers=(False, 0),
                                                   fruits=(True, 2)))
    project.db.record_phenology(2, PhenologyResult(leaves=(True, 1), flowers=(False, 0),
                                                   fruits=(False, 0)))
    export_data_csvs(project, _cfg())
    _, rows = _phenology(project)
    assert rows["sheetA.jpg"]["is_fertile"] == "1"
    assert rows["sheetB.jpg"]["is_fertile"] == "0"


def test_phenology_flags_are_blank_when_the_stage_never_ran(project):
    """No phenology row means unknown. Writing 0 would report every sheet of a run that skipped the
    stage as leafless and sterile -- a real botanical claim LM3 never made."""
    export_data_csvs(project, _cfg())
    _, rows = _phenology(project)
    assert rows["sheetA.jpg"]["has_leaves"] == ""
    assert rows["sheetA.jpg"]["is_fertile"] == ""
    # the counts are still real: they come from the detections, not from the stage
    assert rows["sheetA.jpg"]["leaf_whole"] == "1"


def test_phenology_keeps_sheets_with_no_plant_detections(project):
    """LM2 had no row for such a sheet (no boxes -> no label file). Dropping it would shrink the
    denominator of any rate computed from the file."""
    project.db.conn.execute("DELETE FROM plant_detection WHERE specimen_id = 2")
    export_data_csvs(project, _cfg())
    _, rows = _phenology(project)
    assert rows["sheetB.jpg"]["leaf_whole"] == "0"
    assert rows["sheetB.jpg"]["n_plant_detections"] == "0"


def test_detections_include_suppressed_boxes_with_their_flag(project):
    """Every consuming read hides suppressed duplicates; this file is the record of what the
    detectors proposed, so it keeps them and says so."""
    project.db.conn.execute(
        "INSERT INTO plant_detection (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2, "
        " suppressed, suppressed_by) VALUES (1, 0, 'Leaf_WHOLE', 0.4, 11, 21, 111, 221, 1, 1)")
    export_data_csvs(project, _cfg())
    rows = _read(_data_dir(project) / "detections.csv")
    assert sum(1 for r in rows if r["suppressed"] == "1") == 1
    # ...and the suppressed box must NOT reach the leaf file
    leaves = _read(_data_dir(project) / "leaf_measurements.csv")
    assert len(leaves) == 2


def test_detection_geometry_is_derived(project):
    export_data_csvs(project, _cfg())
    r = _read(_data_dir(project) / "detections.csv")[0]
    assert float(r["box_w_px"]) == float(r["x2"]) - float(r["x1"])
    assert float(r["box_area_px"]) == float(r["box_w_px"]) * float(r["box_h_px"])


# --------------------------------------------------------------------------- #
# the data dictionary is the contract
# --------------------------------------------------------------------------- #
def test_data_dictionary_documents_every_column_of_every_file(project):
    written = export_data_csvs(project, _cfg())
    documented: dict[str, set[str]] = {}
    for row in _read(_data_dir(project) / "data_dictionary.csv"):
        if row["column"]:
            documented.setdefault(row["file"], set()).add(row["column"])

    for path in written:
        if path.name == "data_dictionary.csv":
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            header = set(next(csv.reader(fh)))
        assert documented.get(path.name) == header, (
            f"{path.name}: dictionary and header disagree; "
            f"undocumented={sorted(header - documented.get(path.name, set()))} "
            f"stale={sorted(documented.get(path.name, set()) - header)}"
        )


@pytest.mark.parametrize("spec", [f for f in FILES if f.columns], ids=lambda f: f.key)
def test_declared_columns_all_come_from_the_query(project, spec):
    """The exporter names columns the SQL in db.py aliases. If the two drift, the column exports as
    silently empty -- which is indistinguishable from a real missing measurement.
    """
    export_data_csvs(project, _cfg())
    if spec.key == "data_dictionary":
        return
    rows = _read(_data_dir(project) / f"{spec.stem}.csv")
    if not rows:
        return
    declared = {c.name for c in spec.columns}
    assert declared == set(rows[0].keys())


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    (None, ""),
    (float("nan"), ""),
    (float("inf"), ""),
    (0.1 + 0.2, 0.3),          # rounded, so 0.30000000000000004 never reaches a data file
    (4.0, 4),                  # a whole float writes as an int, not "4.0"
    (0, 0),                    # zero is a VALUE and must survive
    ("", ""),
])
def test_value_formatting(value, expected):
    assert data_export._fmt(value, "", 6) == expected


def test_na_rep_is_configurable(project):
    export_data_csvs(project, _cfg(na_rep="NA"))
    b = next(r for r in _read(_data_dir(project) / "leaf_measurements.csv")
             if r["image_stem"] == "sheetB")
    assert b["lamina_area_incl_holes_cm2"] == "NA"


def test_zero_is_not_written_as_missing(project):
    """n_holes = 0 is a measurement. Confusing it with 'not measured' would be a data error."""
    project.db.conn.execute("UPDATE leaf_morphology SET n_holes = 0, lamina_hole_area_px = 0.0")
    export_data_csvs(project, _cfg(na_rep="NA"))
    for r in _read(_data_dir(project) / "leaf_measurements.csv"):
        assert r["n_holes"] == "0"
        assert r["lamina_hole_area_px"] == "0"


def test_export_failure_does_not_raise(project, monkeypatch):
    """A broken table must cost that one file, not the run -- every measurement is already durable."""
    monkeypatch.setattr(ProjectDB, "export_landmark_rows",
                        lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    written = export_data_csvs(project, _cfg())
    names = {p.name for p in written}
    assert "landmarks.csv" not in names
    assert "leaf_measurements.csv" in names


# --------------------------------------------------------------------------- #
# db-layer guards
# --------------------------------------------------------------------------- #
def test_passthrough_table_whitelist_is_enforced(project):
    """The table name is interpolated into SQL, so it must never accept an arbitrary string."""
    with pytest.raises(ValueError):
        project.db.export_table("specimen; DROP TABLE specimen")
    with pytest.raises(ValueError):
        project.db.export_table_columns("specimen")
