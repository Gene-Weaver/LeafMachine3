"""Green Index stage: the formulas, the white-peak check, the leaf measurement, and its outputs.

The formula values are not computed here and compared with themselves: six of the seven are pinned to
the table printed in Signorelli et al. (bioRxiv 10.1101/2023.08.23.554481, Fig. 1B), which scores a
24-color card with every one of them. TREx is not in that table, so its values are worked by hand.
"""
from __future__ import annotations

import csv
import importlib
import json
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.green_index import METHODS
from leafmachine3.core.green_index import pixel_exclusion as px_ex
from leafmachine3.core.green_index.Excess_Green_Thrash_2022 import (
    calculate_excess_green,
    calculate_vegetation_index,
    excess_green,
)
from leafmachine3.core.green_index.Green_Chromatic_Coordinate_Woebbecke_1995 import (
    calculate_green_chromatic_coordinate,
    calculate_greenness_index,
)
from leafmachine3.core.green_index.Green_Index_Signorelli_2025 import calculate_green_index
from leafmachine3.core.green_index.Green_Leaf_Index_Louhaichi_2001 import calculate_green_leaf_index
from leafmachine3.core.green_index.Greenness_Lazarevic_2022 import calculate_greenness
from leafmachine3.core.green_index.Two_Fold_Red_Excess_Agarwal_2025 import calculate_two_fold_red_excess
from leafmachine3.core.green_index.Vegetation_Index_2_Kawashima_1998 import calculate_vegetation_index_2
from leafmachine3.core.green_index._common import _safe_divide
from leafmachine3.core.records import SpecimenRecord
from leafmachine3.core.stage import WorkItem
from leafmachine3.modules.green_index import GreenIndex, enforce_report_deps, measure_leaf
from leafmachine3.reporting.data_export import STAGE_FILES, export_data_csvs, export_stage_csv
from leafmachine3.reporting.green_index_viz import paint_excluded, render_summary_panel

BROWN = (120, 90, 60)
TAPE = (232, 230, 226)


# --------------------------------------------------------------------------- #
# the formulas
# --------------------------------------------------------------------------- #
# Printed in Fig. 1B: color -> (R, G, B), GI, Veg. I (ExG), Veg. I 2, Greenness I. (GCC), GLI, Greenness
CARD = {
    "Green": ((51, 141, 60), 0.98, 171, -0.08, 0.56, 0.44, 0.56),
    "Dark green": ((45, 86, 34), 0.63, 93, 0.14, 0.52, 0.37, 0.44),
    "Lime": ((133, 170, 45), 0.51, 162, 0.49, 0.49, 0.31, 0.34),
    "Orange": ((200, 99, 21), 0.18, -23, 0.81, 0.31, -0.05, -0.04),
    "Black": ((17, 19, 18), 0.26, 3, -0.03, 0.35, 0.04, 0.04),
}


@pytest.mark.parametrize("color", CARD)
def test_formulas_reproduce_the_published_color_card(color: str) -> None:
    rgb, gi, exg, vi2, gcc, gli, gr = CARD[color]
    # 0.006, not 0.005: the table prints two decimals and 0.625 prints as 0.63.
    assert calculate_green_index(*rgb) == pytest.approx(gi, abs=0.006)
    assert calculate_excess_green(*rgb) == pytest.approx(exg, abs=0.5)
    assert calculate_vegetation_index_2(*rgb) == pytest.approx(vi2, abs=0.006)
    assert calculate_green_chromatic_coordinate(*rgb) == pytest.approx(gcc, abs=0.006)
    assert calculate_green_leaf_index(*rgb) == pytest.approx(gli, abs=0.006)
    assert calculate_greenness(*rgb) == pytest.approx(gr, abs=0.006)


def test_two_fold_red_excess_by_hand() -> None:
    assert calculate_two_fold_red_excess(200, 100, 50) == 250       # (200-100) + (200-50)
    assert calculate_two_fold_red_excess(50, 100, 50) == -50        # green leaf: negative


def test_signorelli_is_not_capped_at_one() -> None:
    """An 8-bit pixel next to the reference green scores 764/12, and R = B = 37.5 cannot occur on one."""
    assert calculate_green_index(38, 165, 38) == pytest.approx(764 / 12)
    assert np.isnan(calculate_green_index(37.5, 165, 37.5))         # 0/0, reachable only by a non-integer


def test_integer_pixels_do_not_overflow() -> None:
    """uint8 arithmetic would wrap 2G - R - B around 256 without any error."""
    r, g, b = (np.array([v], dtype=np.uint8) for v in (10, 200, 10))
    assert calculate_excess_green(r, g, b)[0] == 380


def test_black_pixels_give_nan_not_a_crash() -> None:
    zeros = np.zeros(3)
    assert np.isnan(calculate_green_chromatic_coordinate(zeros, zeros, zeros)).all()
    assert np.isnan(calculate_vegetation_index_2(zeros, zeros, zeros)).all()
    assert np.isnan(calculate_green_leaf_index(zeros, zeros, zeros)).all()
    assert np.isnan(calculate_greenness(zeros, zeros, zeros)).all()
    assert np.isnan(_safe_divide(1.0, 0.0)) and isinstance(_safe_divide(1.0, 2.0), float)


def test_aliases_are_the_canonical_formulas() -> None:
    assert calculate_greenness_index(51, 141, 60) == calculate_green_chromatic_coordinate(51, 141, 60)
    assert calculate_vegetation_index(51, 141, 60) == calculate_excess_green(51, 141, 60)


def test_thrash_window_counts_only_29_to_202() -> None:
    # ExG values: 28 (out), 29 (in), 202 (in), 203 (out)
    r = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.zeros(4)
    g = np.array([14.0, 14.5, 101.0, 101.5])
    exg, mask, n, mean = excess_green(r, g, b)
    assert mask.tolist() == [False, True, True, False] and n == 2 and mean == pytest.approx((29 + 202) / 2)
    assert np.isnan(excess_green(np.array([100.0]), np.array([100.0]), np.array([100.0]))[3])


def test_every_method_is_named_for_its_file() -> None:
    """Provenance: a column leads to a file of the same name, which holds that citation and formula."""
    assert len(METHODS) == 7
    for m in METHODS:
        module = importlib.import_module(f"leafmachine3.core.green_index.{m.name}")
        assert module.METHOD is m
        assert m.column == m.name.lower()
        assert m.calculate.__module__ == module.__name__
        assert m.citation and m.url.startswith("https://") and m.formula
        assert m.axis[0] < m.axis[1]


# --------------------------------------------------------------------------- #
# the white-peak check
# --------------------------------------------------------------------------- #
def _noisy(color, n, spread=12, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(color, spread, (n, 3)), 0, 255).astype(np.uint8)


def _leaf_px(n_leaf=5000, n_tape=500, leaf=BROWN, tape=TAPE, leaf_spread=12, tape_spread=4) -> np.ndarray:
    return np.concatenate([_noisy(leaf, n_leaf, leaf_spread, 1), _noisy(tape, n_tape, tape_spread, 2)])


def test_tape_is_cut_in_the_valley() -> None:
    res = px_ex.find_white_cut(_leaf_px())
    assert res.status == px_ex.STATUS_CUT
    assert res.leaf_peak < res.cut < res.white_peak
    w = px_ex.whiteness(_leaf_px())
    assert (w > res.cut).sum() == pytest.approx(500, abs=5)


def test_a_tape_spike_taller_than_the_leaf_is_still_tape() -> None:
    """A narrow tape spike is often the tallest peak. Treating the tallest peak as the leaf misses it."""
    px = _leaf_px(n_leaf=4000, n_tape=1500, leaf_spread=25, tape_spread=1)
    assert px_ex.find_white_cut(px).status == px_ex.STATUS_CUT


def test_tape_may_outnumber_the_leaf() -> None:
    """Tape over a curved or broken leaf can cover most of what is left, and must still come out."""
    res = px_ex.find_white_cut(_leaf_px(n_leaf=1800, n_tape=3200))
    assert res.status == px_ex.STATUS_CUT
    assert (px_ex.whiteness(_leaf_px(n_leaf=1800, n_tape=3200)) > res.cut).mean() > 0.6


def test_pale_leaf_with_a_small_darker_part_is_left_alone() -> None:
    """Over the 80% limit: a pale, woolly leaf with one dark fold must not turn entirely red."""
    px = np.concatenate([_noisy((215, 215, 210), 9500, 4, 1), _noisy(BROWN, 500, 8, 2)])
    assert px_ex.find_white_cut(px).status == "too_large"


def test_majority_white_needs_a_colored_leaf_side() -> None:
    """A pale gray leaf whose darker side is gray too, with no color to set it apart from tape."""
    px = np.concatenate([_noisy((225, 225, 225), 3000, 3, 1), _noisy((100, 100, 100), 2000, 6, 2)])
    assert px_ex.find_white_cut(px).status == "pale_leaf_not_tape"


def test_majority_white_must_be_near_neutral() -> None:
    """Creamy tissue outnumbering a brown part is a pale leaf, not tape."""
    px = np.concatenate([_noisy((230, 225, 180), 3000, 3, 1), _noisy(BROWN, 2000, 6, 2)])
    assert px_ex.find_white_cut(px).status == "pale_leaf_not_tape"


def test_a_whole_pale_leaf_has_nothing_to_cut() -> None:
    assert px_ex.find_white_cut(_noisy((215, 215, 210), 5000, 4)).status == "no_leaf_peak"


def test_a_bright_yellow_leaf_is_not_white() -> None:
    """Whiteness is min(R, G, B), so a bright leaf with a low blue channel never reads as white."""
    px = np.concatenate([_noisy((230, 220, 90), 4000, 6, 1), _noisy((150, 140, 60), 1000, 6, 2)])
    assert px_ex.find_white_cut(px).status == "no_white_peak"


def test_colored_white_peak_is_refused() -> None:
    px = np.concatenate([_noisy((90, 70, 50), 4000, 8, 1), _noisy((255, 240, 172), 500, 2, 2)])
    assert px_ex.find_white_cut(px).status == "colored_peak"


def test_tape_clipped_at_255_is_still_a_peak() -> None:
    px = np.concatenate([_noisy(BROWN, 4000, 12, 1), np.full((800, 3), 255, np.uint8)])
    res = px_ex.find_white_cut(px)
    assert res.status == px_ex.STATUS_CUT and res.white_peak >= 250


def test_too_few_pixels_and_too_little_tape() -> None:
    assert px_ex.find_white_cut(_leaf_px(n_leaf=200, n_tape=50)).status == "too_few_px"
    assert px_ex.find_white_cut(_leaf_px(n_leaf=10000, n_tape=10)).status in ("too_small", "no_white_peak")


def test_a_shallow_valley_is_not_a_separate_peak() -> None:
    """A leaf whose pale tail runs right up into a small bump has no trough to cut at.

    Built from exact counts rather than random pixels, so the histogram has only these two humps.
    """
    levels = np.arange(256)
    counts = (5000 * np.exp(-0.5 * ((levels - 155) / 16) ** 2) / (16 * np.sqrt(2 * np.pi))
              + 250 * np.exp(-0.5 * ((levels - 190) / 3) ** 2) / (3 * np.sqrt(2 * np.pi)))
    px = np.repeat(levels.astype(np.uint8), counts.round().astype(int))
    res = px_ex.find_white_cut(np.stack([px, px, px], axis=1))
    assert res.status == "shallow_valley" and res.cut is None and res.white_peak >= 185


# --------------------------------------------------------------------------- #
# one leaf
# --------------------------------------------------------------------------- #
def _flat_brown_leaf() -> tuple[np.ndarray, np.ndarray]:
    """ExG is exactly 0 everywhere, so nothing falls in the Thrash 29-202 window."""
    return np.full((40, 40, 3), BROWN, np.uint8), np.ones((40, 40), bool)


def _taped_leaf(h=200, w=120, tape_rows=(90, 120), leaf=BROWN) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    rgb = np.clip(rng.normal(leaf, 10, (h, w, 3)), 0, 255).astype(np.uint8)
    rgb[tape_rows[0]:tape_rows[1]] = np.clip(rng.normal(TAPE, 3, (tape_rows[1] - tape_rows[0], w, 3)), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    lamina = mask > 0
    rgb[~lamina] = 0
    return rgb, lamina


def test_excluded_pixels_are_left_out_of_every_value() -> None:
    rgb, lamina = _taped_leaf()
    with_check = measure_leaf(rgb, lamina).fields
    without = measure_leaf(rgb, lamina, white_peak_check=False).fields
    assert with_check["white_peak_status"] == "cut" and with_check["n_excluded_white_peak_px"] > 0
    assert with_check["n_used_px"] + with_check["n_excluded_white_peak_px"] == with_check["n_lamina_px"]
    assert with_check["mean_r"] == pytest.approx(BROWN[0], abs=2)     # the tape is gone from the color
    assert without["mean_r"] > with_check["mean_r"] + 5
    assert without["white_peak_status"] == "off" and without["white_peak_cut"] is None
    assert without["n_excluded_white_peak_px"] == 0 and without["used_frac"] == 1.0


def test_values_are_the_mean_of_per_pixel_values_on_the_used_pixels() -> None:
    rgb, lamina = _taped_leaf()
    m = measure_leaf(rgb, lamina)
    px = rgb[m.used].astype(float)
    for method in METHODS:
        expected = float(np.nanmean(method.calculate(px[:, 0], px[:, 1], px[:, 2])))
        assert m.fields[method.column] == pytest.approx(expected)


def test_channel_order_is_rgb() -> None:
    """cv2 reads BGR. A swapped cutout would give a plausible, wrong number for every index."""
    red_leaf = np.zeros((40, 40, 3), np.uint8)
    red_leaf[:] = (200, 40, 40)
    fields = measure_leaf(red_leaf, np.ones((40, 40), bool), white_peak_check=False).fields
    assert fields["mean_r"] == 200 and fields["two_fold_red_excess_agarwal_2025"] == 320


def test_a_leaf_outside_the_thrash_window_has_an_empty_window_mean() -> None:
    f = measure_leaf(*_flat_brown_leaf()).fields
    assert f["excess_green_thrash_2022"] is not None
    assert f["excess_green_thrash_2022_window_mean"] is None
    assert f["excess_green_thrash_2022_window_n_px"] == 0 and f["excess_green_thrash_2022_window_frac"] == 0


def test_an_empty_mask_gives_no_row() -> None:
    assert measure_leaf(np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), bool)) is None


def test_summary_image_paints_excluded_pixels_red_at_60_percent() -> None:
    rgb = np.full((2, 1, 3), 100, np.uint8)
    out = paint_excluded(rgb, np.array([[True], [False]]))
    assert out[0, 0].tolist() == [193, 40, 40]        # 0.4*100 + 0.6*255, 0.4*100, 0.4*100
    assert out[1, 0].tolist() == [100, 100, 100] and rgb[0, 0].tolist() == [100, 100, 100]


def test_summary_image_renders_for_a_taped_and_an_untaped_leaf() -> None:
    for tape_rows in ((90, 120), (0, 0)):
        rgb, lamina = _taped_leaf(tape_rows=tape_rows)
        m = measure_leaf(rgb, lamina)
        img = render_summary_panel(rgb, lamina, m.used, m.tiles, m.white_peak, title="leaf")
        assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8 and img.std() > 1


def test_summary_image_survives_a_leaf_with_no_finite_value() -> None:
    black = np.zeros((30, 30, 3), np.uint8)
    lamina = np.ones((30, 30), bool)
    m = measure_leaf(black, lamina, white_peak_check=False)
    assert m.fields["green_chromatic_coordinate_woebbecke_1995"] is None
    assert render_summary_panel(black, lamina, m.used, m.tiles, m.white_peak, title="x").shape[2] == 3


# --------------------------------------------------------------------------- #
# the stage's files and rows
# --------------------------------------------------------------------------- #
def _stage(tmp_path: Path, **settings) -> tuple[GreenIndex, WorkItem]:
    rgb, lamina = _taped_leaf()
    rgb_path, mask_path = tmp_path / "leaf.jpg", tmp_path / "leaf.png"
    cv2.imwrite(str(rgb_path), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(mask_path), lamina.astype(np.uint8) * 255)
    cfg = types.SimpleNamespace(stage=lambda key: settings, report={})
    item = WorkItem(3, {"stem": "sheet", "det_box": (1, 2, 3, 4), "reports_dir": str(tmp_path / "reports"), "leaf_id": 9,
                        "detection_id": 7, "instance_index": 0, "rgb_name": rgb_path.name, "rgb_path": str(rgb_path),
                        "mask_path": str(mask_path)})
    return GreenIndex(cfg), item


def test_overlay_switch_only_controls_the_image(tmp_path: Path) -> None:
    stage, item = _stage(tmp_path, write_overlay=True)
    with_image = stage.infer(item, None)
    path = Path(with_image["overlay_png"])
    assert path.is_file() and path.parent.parts[-2:] == ("Overlay", "Overlay_Green_Index")
    assert path.name == "sheet__or-GI-lamina__1_2_3_4.jpg"

    stage, item = _stage(tmp_path, write_overlay=False)
    without = stage.infer(item, None)
    assert without["overlay_png"] is None
    assert {k: v for k, v in without.items() if k != "overlay_png"} == \
           {k: v for k, v in with_image.items() if k != "overlay_png"}


def test_white_peak_switch(tmp_path: Path) -> None:
    stage, item = _stage(tmp_path, write_overlay=False, white_peak_check=False)
    assert stage.infer(item, None)["white_peak_status"] == "off"


def test_a_missing_cutout_is_skipped_not_fatal(tmp_path: Path) -> None:
    stage, item = _stage(tmp_path, write_overlay=False)
    Path(item.payload["rgb_path"]).unlink()
    assert stage.infer(item, None) is None


@pytest.fixture()
def project(tmp_path):
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    db.upsert_specimen(SpecimenRecord(
        image_name="sheetA.jpg", image_stem="sheetA", original_path="/orig/sheetA.jpg", working_path="/work/sheetA.jpg",
        width=1000, height=2000, original_width=2000, original_height=4000, work_scale=0.5))
    conn = db.conn
    conn.execute("INSERT INTO plant_detection (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2) "
                 "VALUES (1, 0, 'Leaf_WHOLE', 0.9, 10, 20, 110, 220)")
    conn.execute("INSERT INTO leaf_segmentation (specimen_id, detection_id, instance_index, cls_name, cls_id, mask_format, "
                 " mask_data, frame_width, frame_height, area_px, perimeter_px) "
                 "VALUES (1, 1, 0, 'Leaf', 0, 'polygon_xy', '[[0,0],[1,0],[1,1]]', 1000, 2000, 400.0, 80.0)")
    conn.execute("INSERT INTO leaf_morphology (leaf_id, specimen_id, detection_id, instance_index, cls_name, area_px, "
                 " perimeter_px, bbox_x1, bbox_y1, bbox_x2, bbox_y2, oriented_leaf_success, "
                 " oriented_leaf_rotation_angle_degreesCW) VALUES (1, 1, 1, 0, 'Leaf', 400, 80, 0, 0, 20, 40, 1, 33.5)")
    return types.SimpleNamespace(db=db, dirs=types.SimpleNamespace(reports=tmp_path / "reports"))


def _record(project, tmp_path: Path) -> dict:
    cutout = tmp_path / "cutout.jpg"
    overlay = tmp_path / "overlay.jpg"
    cutout.write_bytes(b"x")
    overlay.write_bytes(b"x")
    row = {"leaf_id": 1, "specimen_id": 1, "detection_id": 1, "instance_index": 0, "leaf_image_name": cutout.name,
           "leaf_image_path": str(cutout), "overlay_png": str(overlay), **measure_leaf(*_flat_brown_leaf()).fields}
    project.db.record_leaf_green_index_one(row)
    return row


def _cfg(module_on=True, **data):
    return types.SimpleNamespace(report={"data": {"enabled": True, **data}}, is_enabled=lambda key: module_on)


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


APPROVED_COLUMNS = [
    "leaf_image_name", "leaf_image_path", "overlay_png",
    "image_name", "image_stem", "original_path",
    "leaf_uid", "crop_file_token", "specimen_id", "leaf_id", "detection_id", "instance_index",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2", "work_scale", "oriented_leaf_rotation_angle_degreesCW",
    "n_lamina_px", "n_excluded_white_peak_px", "n_used_px", "used_frac", "white_peak_status", "white_peak_cut",
    "mean_r", "mean_g", "mean_b",
    "green_index_signorelli_2025", "green_chromatic_coordinate_woebbecke_1995", "excess_green_thrash_2022",
    "two_fold_red_excess_agarwal_2025", "vegetation_index_2_kawashima_1998", "green_leaf_index_louhaichi_2001",
    "greenness_lazarevic_2022",
    "excess_green_thrash_2022_window_mean", "excess_green_thrash_2022_window_n_px",
    "excess_green_thrash_2022_window_frac",
]


def test_csv_has_the_approved_columns_and_traces_back_to_the_parent(project, tmp_path) -> None:
    row = _record(project, tmp_path)
    path = export_stage_csv(project, _cfg(), "green_index")
    assert path == project.dirs.reports / "Data" / "green_index.csv"
    assert path.read_text().splitlines()[0].split(",") == APPROVED_COLUMNS
    rows = _read(path)
    assert len(rows) == 1
    r = rows[0]
    assert r["image_name"] == "sheetA.jpg" and r["original_path"] == "/orig/sheetA.jpg"
    assert r["leaf_uid"] == "sheetA__10_20_110_220__i0" and r["crop_file_token"] == "sheetA__10_20_110_220"
    assert (r["crop_x1"], r["crop_y2"]) == ("10", "220") and r["work_scale"] == "0.5"
    assert r["oriented_leaf_rotation_angle_degreesCW"] == "33.5"
    assert r["leaf_image_name"] == row["leaf_image_name"]
    assert r["excess_green_thrash_2022_window_mean"] == ""             # an absence, not 0 or "nan"


def test_csv_follows_report_data_settings(project, tmp_path) -> None:
    _record(project, tmp_path)
    path = export_stage_csv(project, _cfg(format="tsv", folder="Tables"), "green_index")
    assert path == project.dirs.reports / "Tables" / "green_index.tsv"
    assert "\t" in path.read_text().splitlines()[0]
    assert export_stage_csv(project, _cfg(enabled=False), "green_index") is None


def test_reporter_export_documents_the_file_but_never_writes_stale_rows(project, tmp_path) -> None:
    _record(project, tmp_path)
    written = export_data_csvs(project, _cfg())
    data = project.dirs.reports / "Data"
    assert data / "green_index.csv" not in written and not (data / "green_index.csv").exists()
    documented = {r["column"] for r in _read(data / "data_dictionary.csv") if r["file"] == "green_index.csv"}
    assert set(APPROVED_COLUMNS) <= documented


def test_an_older_project_gets_the_file_added_to_its_dictionary(project, tmp_path) -> None:
    """Its Reporter wrote the dictionary before this stage existed and has no reason to run again."""
    _record(project, tmp_path)
    export_data_csvs(project, _cfg(module_on=False))
    dictionary = project.dirs.reports / "Data" / "data_dictionary.csv"
    assert "green_index.csv" not in {r["file"] for r in _read(dictionary)}
    export_stage_csv(project, _cfg(), "green_index")
    export_stage_csv(project, _cfg(), "green_index")                   # a second run must not add it twice
    documented = [r["column"] for r in _read(dictionary) if r["file"] == "green_index.csv"]
    assert documented == ["", *APPROVED_COLUMNS]


def test_turning_the_module_off_cleans_up_its_csv(project, tmp_path) -> None:
    _record(project, tmp_path)
    path = export_stage_csv(project, _cfg(), "green_index")
    export_data_csvs(project, _cfg())
    assert path.is_file()                                              # the Reporter leaves it while enabled
    export_data_csvs(project, _cfg(module_on=False))
    assert not path.exists()
    assert "green_index.csv" not in {r["file"] for r in _read(path.parent / "data_dictionary.csv")}


def test_the_declared_columns_are_the_approved_ones() -> None:
    assert [c.name for c in STAGE_FILES[0].columns] == APPROVED_COLUMNS


def test_a_reset_deletes_the_summary_image_but_never_the_reporters_cutout(project, tmp_path) -> None:
    row = _record(project, tmp_path)
    stage = types.SimpleNamespace(owns_tables=("leaf_green_index",))
    project.db.reset_stages(["green_index"], {"green_index": stage})
    assert not Path(row["overlay_png"]).exists()
    assert Path(row["leaf_image_path"]).exists()
    assert project.db.leaf_green_index(1) == []


def test_an_existing_project_renumbers_the_stages_after_the_new_one(tmp_path) -> None:
    """An older DB has ect at 17. Opening it must not leave green_index and ect sharing a position."""
    path = tmp_path / "old.sqlite"
    db = ProjectDB.open_or_create(path)
    db._exec("DELETE FROM project_status WHERE stage_key = 'green_index'")
    db._exec("UPDATE project_status SET stage_order = 17, state = 'done' WHERE stage_key = 'ect'")
    db._exec("UPDATE project_status SET stage_order = 18 WHERE stage_key = 'momocs'")
    db.conn.commit()
    db.close()
    db = ProjectDB.open_or_create(path)
    order = {r["stage_key"]: r["stage_order"] for r in db._query("SELECT stage_key, stage_order FROM project_status")}
    assert order["reporter"] < order["green_index"] < order["ect"] < order["momocs"]
    assert db.stage_state("ect") == "done"                             # renumbering keeps the ledger state
    assert db.stage_state("green_index") == "pending"


# --------------------------------------------------------------------------- #
# report-dependency enforcement
# --------------------------------------------------------------------------- #
def test_enabling_the_module_does_not_change_the_reporters_settings_hash(mock_config_path) -> None:
    """Otherwise every existing run re-renders its Reporter, ECT and Momocs output when opened."""
    from leafmachine3.core.config import Config

    cfg = Config.load(mock_config_path)
    cfg.modules["green_index"] = {"enabled": False}
    before = cfg.stage_settings_hash("reporter")
    cfg.modules["green_index"] = {"enabled": True, "write_overlay": True, "white_peak_check": True}
    enforce_report_deps(cfg)
    assert cfg.stage_settings_hash("reporter") == before


def test_products_switched_off_are_forced_back_on(mock_config_path) -> None:
    from leafmachine3.core.config import Config

    cfg = Config.load(mock_config_path)
    cfg.modules["green_index"] = {"enabled": True}
    cfg.report["leaf_products"] = {"enabled": False, "oriented": False,
                                   "products": {"lamina_rgb": False, "lamina_mask": False, "bbox": False}}
    enforce_report_deps(cfg)
    lp = cfg.report["leaf_products"]
    assert lp["enabled"] and lp["oriented"]
    assert lp["products"]["lamina_rgb"] and lp["products"]["lamina_mask"]
    assert lp["products"]["bbox"] is False                             # nothing else is touched


def test_a_disabled_module_forces_nothing(mock_config_path) -> None:
    from leafmachine3.core.config import Config

    cfg = Config.load(mock_config_path)
    cfg.modules["green_index"] = {"enabled": False}
    cfg.report["leaf_products"] = {"products": {"lamina_rgb": False}}
    before = json.dumps(cfg.report["leaf_products"], sort_keys=True, default=str)
    enforce_report_deps(cfg)
    assert json.dumps(cfg.report["leaf_products"], sort_keys=True, default=str) == before
