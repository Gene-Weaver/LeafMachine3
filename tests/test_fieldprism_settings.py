"""FieldPrism (FP) settings and export declarations -- the SETTINGS side of FP ruler support.

A FieldPrism knob has to exist in four places that nothing but these tests keeps in step: the
built-in defaults (so the settings form renders it even without the YAML), the shipped YAML and
the shipped preset (what a user edits and what a run reads), and settings_meta.json (label, help
and which rail group it lands in). The export side has to name the exact column aliases db.py
selects, or a declared column silently exports as empty.
"""
from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from leafmachine3.core.config import Config, builtin_defaults
from leafmachine3.reporting import data_export

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
META = json.loads((REPO_ROOT / "leafmachine3" / "server" / "ui" / "settings_meta.json").read_text())
SHIPPED_YAMLS = ("LM3_settings.yaml", "presets/default.yaml")

#: The modules.ruler_cf.fieldprism block and its shipped defaults.
FP_RULER_CF = {"enabled": True, "peer_tol": 0.03, "anchor_tol": 0.03, "allow_single_marker": True}

#: The flat names the knobs had before they were nested (never in a release, so not retired).
OLD_FLAT_KEYS = ("fp_enabled", "fp_peer_tol", "fp_anchor_tol", "fp_allow_single_marker")

#: The specimen-level aliases db.export_specimen_rows / export_leaf_rows select (contract section 4),
#: with the unit each must be declared in.
FP_SUMMARY_ALIASES = {
    "ruler_cf_anchor_source": "",
    "fp_sheet_type": "",
    "fp_sheet_status": "",
    "fp_orientation_deg": "degrees",
    "fp_n_markers_detected": "count",
    "fp_n_markers_used": "count",
    "fp_n_markers_inferred": "count",
    "fp_cf_px_per_cm": "px/cm",
    "fp_confidence": "",
}

#: ruler_FP_marker / ruler_FP_sheet columns, transcribed from the contract's DDL. Pinned as literals
#: so the data dictionary's per-column docs cannot quietly fall behind the tables.
FP_MARKER_TABLE_COLUMNS = [
    "fp_marker_id", "specimen_id", "detection_id", "crop_index", "det_conf", "x1", "y1", "x2", "y2",
    "roi_x0", "roi_y0", "roi_x1", "roi_y1", "status", "status_reason", "valid", "validation_json",
    "verdict", "verdict_note", "n_peaks", "holes_filled", "peak_area_ratio", "tl_x", "tl_y", "tr_x",
    "tr_y", "c_x", "c_y", "bl_x", "bl_y", "br_x", "br_y", "pitch_h_px", "pitch_v_px", "pxcm",
    "pxcm_original", "pct_vs_fp", "orientation_deg", "sheet_corner",
]
FP_SHEET_TABLE_COLUMNS = [
    "specimen_id", "catalog_version", "n_fp_detected", "n_fp_measured", "n_fp_valid", "n_fp_used",
    "n_fp_rejected", "n_fp_inferred", "sheet_status", "sheet_type", "sheet_label",
    "corners_ambiguous", "sheet_candidates_json", "orientation_deg", "fit_rotation_deg",
    "fit_scale_px_per_mm", "fit_tx", "fit_ty", "fit_rms_mm", "fit_max_mm", "fit_scale_dev_mm",
    "fit_cost_mm", "cf_px_per_cm_sheet_fit", "cf_px_per_cm_marker_mean", "cf_px_per_cm_fp",
    "cf_source_detail", "fp_peer_spread_pct", "fp_confidence", "fp_reasons_json", "corners_json",
    "page_corners_json", "fpfit_margins_json",
]


def _shipped(name: str) -> dict:
    return yaml.safe_load((REPO_ROOT / name).read_text())


# --------------------------------------------------------------------------- #
# modules.ruler_cf.fieldprism.*
# --------------------------------------------------------------------------- #
def test_builtin_defaults_carry_the_fieldprism_ruler_cf_keys():
    ruler_cf = builtin_defaults()["modules"]["ruler_cf"]
    assert ruler_cf["fieldprism"] == FP_RULER_CF
    assert not set(OLD_FLAT_KEYS) & set(ruler_cf), "the flat fp_* keys came back"


@pytest.mark.parametrize("name", SHIPPED_YAMLS)
def test_shipped_yaml_carries_the_fieldprism_ruler_cf_keys(name):
    ruler_cf = _shipped(name)["modules"]["ruler_cf"]
    assert ruler_cf.get("fieldprism") == FP_RULER_CF, f"{name}: modules.ruler_cf.fieldprism"
    assert not set(OLD_FLAT_KEYS) & set(ruler_cf), f"{name}: the flat fp_* keys came back"


def test_a_settings_file_without_the_fp_keys_still_gets_them(tmp_path):
    """An older settings file predates FieldPrism; the merge must fill the knobs in, not drop them."""
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({"modules": {"ruler_cf": {"enabled": True, "anchor_tol": 0.25}}}))
    rcf = Config.load(path).modules.ruler_cf
    assert rcf.fieldprism.enabled is True and rcf.fieldprism.allow_single_marker is True
    assert rcf.fieldprism.peer_tol == pytest.approx(0.03)
    assert rcf.fieldprism.anchor_tol == pytest.approx(0.03)
    assert rcf.anchor_tol == pytest.approx(0.25)


def test_default_tolerances_match_the_fieldprism_module():
    """The settings default and the engine's own fallback must be the same number."""
    from leafmachine3.inference.ruler_lattice import fieldprism
    fp = builtin_defaults()["modules"]["ruler_cf"]["fieldprism"]
    assert fp["peer_tol"] == fieldprism.FP_PEER_TOL
    assert fp["anchor_tol"] == fieldprism.FP_ANCHOR_TOL


# --------------------------------------------------------------------------- #
# report.overlay.draw_fieldprism + report.overlay_fieldprism
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", SHIPPED_YAMLS)
def test_shipped_yaml_carries_the_fieldprism_overlay_toggles(name):
    report = _shipped(name)["report"]
    assert report["overlay"]["draw_fieldprism"] is True
    assert report["overlay_fieldprism"] == {"enabled": True}
    # palette.FieldPrismStyle reads an OPTIONAL report.overlay.fieldprism block; it is not shipped,
    # so the app colors stay the code defaults.
    assert "fieldprism" not in report["overlay"]


# --------------------------------------------------------------------------- #
# settings_meta.json
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path,type_,group", [
    ("modules.ruler_cf.fieldprism.enabled", "bool", "FieldPrism"),
    ("modules.ruler_cf.fieldprism.peer_tol", "float", "FieldPrism"),
    ("modules.ruler_cf.fieldprism.anchor_tol", "float", "FieldPrism"),
    ("modules.ruler_cf.fieldprism.allow_single_marker", "bool", "FieldPrism"),
    ("report.overlay.draw_fieldprism", "bool", "Summary overlay"),
    ("report.overlay_fieldprism.enabled", "bool", "FieldPrism overlay"),
    ("report.data.files.fieldprism_markers", "bool", "Data Export (CSV)"),
    ("report.data.files.fieldprism_sheets", "bool", "Data Export (CSV)"),
])
def test_fieldprism_settings_are_documented(path, type_, group):
    entry = META.get(path)
    assert entry, f"{path} has no settings_meta entry"
    assert entry["type"] == type_ and entry["group"] == group
    assert entry["label"] and len(entry["help"]) > 40


@pytest.mark.parametrize("path", ["modules.ruler_cf.fieldprism.peer_tol",
                                  "modules.ruler_cf.fieldprism.anchor_tol"])
def test_fieldprism_tolerance_bounds_admit_the_default(path):
    entry, default = META[path], FP_RULER_CF[path.rsplit(".", 1)[1]]
    assert entry["min"] <= default <= entry["max"]


def test_no_metadata_left_under_the_old_flat_names():
    stale = [f"modules.ruler_cf.{k}" for k in OLD_FLAT_KEYS if f"modules.ruler_cf.{k}" in META]
    assert not stale, f"settings_meta.json still documents the flat names: {stale}"


# --------------------------------------------------------------------------- #
# the settings rail: Scale > Ruler Conversion Factor > FieldPrism
# --------------------------------------------------------------------------- #
#: Every FieldPrism knob that belongs on the ruler_cf rail entry (the two CSV switches stay with
#: the other CSV files under Reporter > Data export).
FP_RAIL_LEAVES = (
    "modules.ruler_cf.fieldprism.enabled", "modules.ruler_cf.fieldprism.peer_tol",
    "modules.ruler_cf.fieldprism.anchor_tol", "modules.ruler_cf.fieldprism.allow_single_marker",
    "report.overlay_fieldprism.enabled", "report.overlay.draw_fieldprism",
)


def _section(stage: str) -> dict:
    return next(s for s in META["_sections"] if s.get("stage") == stage)


def _rail(sections, leaf_paths, section):
    """(declared sub-section label -> the leaf paths declaredRailGroups gives it), mirroring
    settings.js: own:true takes a group's direct leaves, otherwise the whole subtree."""
    from tests.test_settings_ui import _rendered_group_paths, _section_for
    owned = [p for p in leaf_paths if _section_for(p, sections) is section]
    rendered = _rendered_group_paths(section, sections, leaf_paths)
    out = {}
    for sub in section.get("subsections", []):
        got = []
        for base in sub["paths"]:
            assert base in rendered, f"{sub['label']!r}: {base!r} is not a rendered rail group"
            for p in owned:
                if not p.startswith(base + "."):
                    continue
                if sub.get("own") and "." in p[len(base) + 1:]:
                    continue
                got.append(p)
        out[sub["label"]] = got
    return owned, out


@pytest.fixture(scope="module")
def leaf_paths():
    from tests.test_settings_ui import SETTINGS_YAML
    from leafmachine3.server.settings_api import read_settings

    def walk(node, parts, out):
        if isinstance(node, dict) and node:
            for key, value in node.items():
                walk(value, parts + [key], out)
            return out
        if parts:
            out.append(".".join(parts))
        return out

    return walk(read_settings(SETTINGS_YAML, with_text=False)["effective"], [], [])


def test_fieldprism_settings_route_to_the_ruler_cf_section(leaf_paths):
    from tests.test_settings_ui import _section_for
    sections = META["_sections"]
    ruler_cf = _section("ruler_cf")
    assert ruler_cf["phase"] == "scale"
    for path in FP_RAIL_LEAVES:
        assert path in leaf_paths, f"{path} is not rendered at all"
        assert _section_for(path, sections) is ruler_cf, f"{path} does not route to ruler_cf"
    for key in ("report.data.files.fieldprism_markers", "report.data.files.fieldprism_sheets"):
        assert _section_for(key, sections) is _section("reporter"), f"{key} left Data export"


def test_ruler_cf_rail_has_a_fieldprism_entry_holding_every_fp_knob(leaf_paths):
    sections = META["_sections"]
    ruler_cf = _section("ruler_cf")
    owned, rail = _rail(sections, leaf_paths, ruler_cf)
    assert list(rail) == ["Ruler lattice", "FieldPrism"]
    assert sorted(rail["FieldPrism"]) == sorted(FP_RAIL_LEAVES)
    assert not any("fieldprism" in p for p in rail["Ruler lattice"])
    # nothing falls into the "Other settings" catch-all, and no leaf is listed twice
    listed = [p for paths in rail.values() for p in paths]
    assert sorted(listed) == sorted(owned) and len(listed) == len(set(listed))


def test_reporter_no_longer_lists_the_fieldprism_overlay(leaf_paths):
    """report.overlay_fieldprism routes to ruler_cf now; a Reporter sub-section naming it would
    resolve to nothing (the path renders no group in the Reporter's pane)."""
    reporter = _section("reporter")
    declared = {p for sub in reporter["subsections"] for p in sub["paths"]}
    assert "report.overlay_fieldprism" not in declared


# --------------------------------------------------------------------------- #
# reports/Data declarations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["leaf_measurements", "specimen_summary"])
def test_fieldprism_summary_columns_use_the_db_aliases(key):
    spec = next(f for f in data_export.FILES if f.key == key)
    declared = {c.name: c for c in spec.columns}
    for alias, units in FP_SUMMARY_ALIASES.items():
        assert alias in declared, f"{key}: {alias} not declared"
        assert declared[alias].units == units, f"{key}: {alias} units"
        assert declared[alias].desc
    names = [c.name for c in spec.columns]
    assert len(names) == len(set(names)), f"{key}: duplicate column names"


@pytest.mark.parametrize("key", ["leaf_measurements", "specimen_summary"])
def test_cf_source_documents_the_fieldprism_value(key):
    spec = next(f for f in data_export.FILES if f.key == key)
    desc = next(c.desc for c in spec.columns if c.name == "cf_source")
    assert "measured_from_fieldprism" in desc


@pytest.mark.parametrize("key,table", [("fieldprism_markers", "ruler_FP_marker"),
                                       ("fieldprism_sheets", "ruler_FP_sheet")])
def test_fieldprism_files_are_passthrough_dumps_on_by_default(key, table):
    spec = next(f for f in data_export.FILES if f.key == key)
    assert spec.columns is None and spec.source_table == table
    assert spec.stem == key and spec.default is True
    assert builtin_defaults()["report"]["data"]["files"][key] is True


@pytest.mark.parametrize("key,columns", [("fieldprism_markers", FP_MARKER_TABLE_COLUMNS),
                                         ("fieldprism_sheets", FP_SHEET_TABLE_COLUMNS)])
def test_fieldprism_column_docs_cover_the_table(key, columns):
    spec = next(f for f in data_export.FILES if f.key == key)
    assert [c.name for c in spec.column_docs] == columns


def test_dictionary_uses_column_docs_and_falls_back_for_unknown_columns():
    spec = next(f for f in data_export.FILES if f.key == "fieldprism_markers")
    rows = data_export._dictionary_rows(
        [spec], {"fieldprism_markers": ["pxcm", "br_x", "validation_json", "added_later"]}, "csv")
    by_col = {r["column"]: r for r in rows if r["column"]}
    assert rows[0]["column"] == "" and rows[0]["description"] == spec.blurb
    assert by_col["pxcm"]["units"] == "px/cm"
    assert by_col["br_x"]["units"] == "px" and "PREDICTED" in by_col["br_x"]["description"]
    assert by_col["validation_json"]["units"] == "JSON"
    assert by_col["added_later"]["units"] == ""
    assert by_col["added_later"]["description"].startswith("Verbatim ruler_FP_marker.added_later")


def test_other_passthrough_files_keep_the_generic_dictionary_line():
    spec = next(f for f in data_export.FILES if f.key == "ruler_crops")
    rows = data_export._dictionary_rows([spec], {"ruler_crops": ["pxcm"]}, "csv")
    assert rows[1]["units"] == ""
    assert rows[1]["description"].startswith("Verbatim ruler_CF_lattice_crop.pxcm")
