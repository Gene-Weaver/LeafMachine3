"""Tests for the generate_leaf_collage postprocessing tool."""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from leafmachine3.postprocessing.generate_leaf_collage import (  # noqa: E402
    _VARIANTS,
    generate_collage,
    leaf_scores,
    run,
    select_leaves,
    _white_feret, _leaf_piece, _layout_puzzle, _puzzle_leaf_px, _fit_rotate,
)

_SCHEMA = """
CREATE TABLE specimen (specimen_id INTEGER PRIMARY KEY, image_stem TEXT);
CREATE TABLE plant_detection (detection_id INTEGER PRIMARY KEY, specimen_id INTEGER,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL, suppressed INTEGER DEFAULT 0);
CREATE TABLE leaf_segmentation (leaf_id INTEGER PRIMARY KEY, specimen_id INTEGER,
    detection_id INTEGER, instance_index INTEGER, cls_name TEXT);
CREATE TABLE bilateral_symmetry (bsym_id INTEGER PRIMARY KEY, leaf_id INTEGER,
    archetype_score REAL, gates_pass INTEGER);
"""


def _leaf_png(w=60, h=90, val=255):
    """A blob that fills most of its frame, like a content-fitted Lamina_Mask."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, int(val), -1)
    return m


def _make_run(tmp_path, leaves, *, name="demo", tree="Leaf_Oriented", variant="lamina_mask",
              write_files=True, with_rgb=False, table=True):
    """Build a synthetic LM3 run: <run>/<run>.sqlite + reports/<tree>/<folder>/<crop names>.

    ``leaves`` is a list of ``(stem, box, score, gates_pass)`` -- one bilateral_symmetry row each,
    several of which may share a ``(stem, box)`` to exercise the multi-instance collision.
    """
    root = tmp_path / name
    folder, label, rgb_folder, rgb_label = _VARIANTS[variant]
    mask_dir = root / "reports" / tree / folder
    mask_dir.mkdir(parents=True)
    if with_rgb:
        (root / "reports" / tree / rgb_folder).mkdir(parents=True)

    conn = sqlite3.connect(root / f"{name}.sqlite")
    if table:
        conn.executescript(_SCHEMA)
    else:                                            # a run whose bilateral stage never ran
        conn.executescript(_SCHEMA.replace(
            "CREATE TABLE bilateral_symmetry (bsym_id INTEGER PRIMARY KEY, leaf_id INTEGER,\n"
            "    archetype_score REAL, gates_pass INTEGER);", ""))
    spec_ids, det_ids = {}, {}
    for i, (stem, box, score, gates) in enumerate(leaves, start=1):
        sid = spec_ids.setdefault(stem, len(spec_ids) + 1)
        if sid not in [r[0] for r in conn.execute("SELECT specimen_id FROM specimen")]:
            conn.execute("INSERT INTO specimen VALUES (?, ?)", (sid, stem))
        did = det_ids.setdefault((stem, box), len(det_ids) + 1)
        if did not in [r[0] for r in conn.execute("SELECT detection_id FROM plant_detection")]:
            conn.execute("INSERT INTO plant_detection VALUES (?, ?, ?, ?, ?, ?, 0)",
                         (did, sid, *box))
        conn.execute("INSERT INTO leaf_segmentation VALUES (?, ?, ?, ?, 'Leaf')", (i, sid, did, i))
        if table:
            conn.execute("INSERT INTO bilateral_symmetry VALUES (?, ?, ?, ?)", (i, i, score, gates))
        if write_files:
            # name exactly as core/imaging.crop_filename does -- int(round(v)), half-to-EVEN
            bx = tuple(int(round(v)) for v in box)
            base = f"{stem}__{label}__{bx[0]}_{bx[1]}_{bx[2]}_{bx[3]}"
            cv2.imwrite(str(mask_dir / f"{base}.png"), _leaf_png())
            if with_rgb:
                rgb = cv2.cvtColor(_leaf_png(), cv2.COLOR_GRAY2BGR)
                rgb[:, :, 1] = (rgb[:, :, 1] * 0.5).astype(np.uint8)
                cv2.imwrite(str(root / "reports" / tree / rgb_folder
                                / f"{stem}__{rgb_label}__{bx[0]}_{bx[1]}_{bx[2]}_{bx[3]}.jpg"), rgb)
    conn.commit()
    conn.close()
    return root


def _primary(tmp_path, w=300, h=400):
    """A big white ellipse standing in for a leaf silhouette."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2 - 10, h // 2 - 10), 0, 0, 360, 255, -1)
    p = tmp_path / "primary.png"
    cv2.imwrite(str(p), m)
    return p


def _grid_of(n, score_from=0.99):
    return [(f"HERB_{i:03d}_Fake_species", (10 * i, 20, 10 * i + 100, 140),
             round(score_from - i * 0.001, 4), 1) for i in range(n)]


# -- selection ---------------------------------------------------------------------
def test_vetoed_leaves_are_excluded_however_they_scored(tmp_path) -> None:
    """gates_pass = 0 disqualifies a leaf even with a perfect score -- the veto is not the score."""
    root = _make_run(tmp_path, [
        ("A_1_X_y", (0, 0, 50, 50), 1.00, 0),        # perfect score, but vetoed
        ("B_2_X_y", (0, 0, 50, 50), 0.85, 1),
    ])
    leaves, stats = select_leaves(root, min_archetype_score=0.8)
    assert [lf["stem"] for lf in leaves] == ["B_2_X_y"]
    assert stats["n_passing"] == 1


def test_threshold_is_strictly_greater_than(tmp_path) -> None:
    """A leaf exactly ON the threshold is excluded, matching `archetype_score > ?`."""
    root = _make_run(tmp_path, [("A_1_X_y", (0, 0, 50, 50), 0.80, 1),
                                ("B_2_X_y", (0, 0, 50, 50), 0.801, 1)])
    leaves, _ = select_leaves(root, min_archetype_score=0.8)
    assert [lf["stem"] for lf in leaves] == ["B_2_X_y"]


def test_null_scores_drop_out(tmp_path) -> None:
    """A NULL archetype_score means unmeasurable, not zero, and never passes the filter."""
    root = _make_run(tmp_path, [("A_1_X_y", (0, 0, 50, 50), None, 1),
                                ("B_2_X_y", (0, 0, 50, 50), 0.9, 1)])
    leaves, _ = select_leaves(root, min_archetype_score=0.0)
    assert [lf["stem"] for lf in leaves] == ["B_2_X_y"]


def test_leaves_come_back_best_first_and_the_cap_keeps_the_best(tmp_path) -> None:
    """max_leaves takes the N highest scores, not the first N rows."""
    root = _make_run(tmp_path, _grid_of(10))
    leaves, stats = select_leaves(root, min_archetype_score=0.5, max_leaves=3)
    scores = [lf["score"] for lf in leaves]
    assert scores == sorted(scores, reverse=True) and len(scores) == 3
    assert stats["n_passing"] == 10 and stats["n_used"] == 3


def test_multi_instance_detection_collapses_to_one_tile(tmp_path) -> None:
    """Leaf products are named PER DETECTION: N leaves on one box share ONE merged raster file.

    Without de-duplication the same merged two-leaf blob would be tiled once per leaf_id.
    """
    box = (100, 100, 200, 240)
    root = _make_run(tmp_path, [("SHARED_1_X_y", box, 0.95, 1),      # same stem + same box ...
                                ("SHARED_1_X_y", box, 0.93, 1),      # ... two bilateral rows
                                ("OTHER_2_X_y", (0, 0, 50, 50), 0.9, 1)])
    leaves, stats = select_leaves(root, min_archetype_score=0.8)
    assert stats["n_merged"] == 1
    assert len(leaves) == 2
    assert len({lf["mask_path"] for lf in leaves}) == 2               # no path used twice


def test_missing_oriented_file_is_skipped_not_fatal(tmp_path) -> None:
    """Leaf_Oriented exists only where orientation succeeded; a gap skips that leaf and is counted."""
    root = _make_run(tmp_path, _grid_of(4))
    folder = _VARIANTS["lamina_mask"][0]
    victim = sorted((root / "reports" / "Leaf_Oriented" / folder).iterdir())[0]
    victim.unlink()
    leaves, stats = select_leaves(root, min_archetype_score=0.5)
    assert len(leaves) == 3 and stats["n_missing_mask"] == 1


def test_no_bilateral_table_is_an_actionable_error(tmp_path) -> None:
    """A run whose bilateral stage never ran must say so, not raise sqlite's own message."""
    root = _make_run(tmp_path, _grid_of(2), table=False)
    with pytest.raises(ValueError, match="bilateral_symmetry"):
        select_leaves(root, min_archetype_score=0.5)


def test_rgb_style_requires_a_cutout_sibling(tmp_path) -> None:
    """lamina_petiole_holes_mask has no RGB product, so style='rgb' must refuse it up front."""
    root = _make_run(tmp_path, _grid_of(2))
    with pytest.raises(ValueError, match="no RGB cutout sibling"):
        select_leaves(root, mask_variant="lamina_petiole_holes_mask", style="rgb")


def test_leaf_scores_is_keyed_by_stem_and_box(tmp_path) -> None:
    """The picker's index keys one leaf across EVERY product folder, and omits vetoed leaves."""
    root = _make_run(tmp_path, [("A_1_X_y", (1, 2, 3, 4), 0.95, 1),
                                ("B_2_X_y", (5, 6, 7, 8), 0.99, 0)])
    idx = leaf_scores(root, min_score=0.8)
    assert idx == {("A_1_X_y", (1, 2, 3, 4)): 0.95}


def test_run_dir_comes_from_the_database_not_the_cwd(tmp_path, monkeypatch) -> None:
    """Stored paths may be relative to another CWD, so masks resolve from the DB file's own parent."""
    root = _make_run(tmp_path, _grid_of(3))
    monkeypatch.chdir(tmp_path.parent)
    leaves, _ = select_leaves(root, min_archetype_score=0.5)
    assert len(leaves) == 3
    assert all(Path(lf["mask_path"]).is_file() for lf in leaves)


# -- collage rendering -------------------------------------------------------------
@pytest.mark.parametrize("layout", ["grid", "mosaic", "organic", "puzzle"])
def test_every_layout_fills_the_silhouette_and_stays_inside_the_canvas(tmp_path, layout) -> None:
    """Each layout draws leaves, and the drawn pixels sit inside the primary mask's bounding shape."""
    root = _make_run(tmp_path, _grid_of(40))
    out = tmp_path / "out" / f"{layout}.png"
    res = generate_collage(root, _primary(tmp_path), out, min_archetype_score=0.5,
                           layout=layout, max_dim_px=400, layout_px=256)
    assert out.exists() and res["n_placed"] > 0
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert im.shape[2] == 4 and max(im.shape[:2]) == 400
    alpha = im[:, :, 3]
    assert alpha.max() == 255                        # something was actually drawn
    # the silhouette is an inset ellipse, so the canvas corners must stay empty
    assert alpha[:12, :12].max() == 0 and alpha[-12:, -12:].max() == 0


def test_mask_style_paints_the_requested_color(tmp_path) -> None:
    """style='mask' recolors the binary mask; the opaque pixels are exactly that color."""
    root = _make_run(tmp_path, _grid_of(12))
    out = tmp_path / "c.png"
    generate_collage(root, _primary(tmp_path), out, min_archetype_score=0.5,
                     color=[255, 0, 0], max_dim_px=300, layout_px=200)   # RGB red
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    solid = im[im[:, :, 3] == 255]
    assert len(solid) > 0
    assert np.all(solid[:, 0] == 0) and np.all(solid[:, 1] == 0) and np.all(solid[:, 2] == 255)  # BGR


def test_background_transparent_versus_solid(tmp_path) -> None:
    """'transparent' leaves alpha 0 outside the leaves; a color fills the whole canvas opaque."""
    root = _make_run(tmp_path, _grid_of(12))
    clear_png, solid_png = tmp_path / "clear.png", tmp_path / "solid.png"
    generate_collage(root, _primary(tmp_path), clear_png, min_archetype_score=0.5,
                     background="transparent", max_dim_px=200, layout_px=150)
    generate_collage(root, _primary(tmp_path), solid_png, min_archetype_score=0.5,
                     background=[10, 20, 30], max_dim_px=200, layout_px=150)
    assert cv2.imread(str(clear_png), cv2.IMREAD_UNCHANGED)[:, :, 3].min() == 0
    solid = cv2.imread(str(solid_png), cv2.IMREAD_UNCHANGED)
    assert solid[:, :, 3].min() == 255
    assert tuple(solid[0, 0, :3]) == (30, 20, 10)    # [R,G,B] in -> BGR out


def test_rgb_style_uses_the_cutout_pixels(tmp_path) -> None:
    """style='rgb' composites the RGB sibling, so the output is not one flat color."""
    root = _make_run(tmp_path, _grid_of(12), with_rgb=True)
    out = tmp_path / "rgb.png"
    res = generate_collage(root, _primary(tmp_path), out, min_archetype_score=0.5, style="rgb",
                           max_dim_px=300, layout_px=200)
    assert res["n_placed"] > 0
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    solid = im[im[:, :, 3] == 255]
    assert len(np.unique(solid[:, 1])) > 1 or not np.array_equal(solid[:, 0], solid[:, 1])


def test_center_ranking_puts_the_best_leaves_deepest_inside(tmp_path) -> None:
    """ranking='center' must place higher scores nearer the silhouette's core than its margin."""
    root = _make_run(tmp_path, _grid_of(60))
    out = tmp_path / "c.png"
    res = generate_collage(root, _primary(tmp_path), out, min_archetype_score=0.5,
                           ranking="center", layout="grid", max_dim_px=400, layout_px=256,
                           write_manifest=True)
    placed = json.load(open(res["manifest"]))["leaves"]
    cx = sum(p["cx"] for p in placed) / len(placed)
    cy = sum(p["cy"] for p in placed) / len(placed)
    rad = [((p["cx"] - cx) ** 2 + (p["cy"] - cy) ** 2) ** 0.5 for p in placed]
    best = sorted(range(len(placed)), key=lambda i: -placed[i]["score"])
    top, bottom = best[: len(best) // 4], best[-len(best) // 4:]
    assert sum(rad[i] for i in top) / len(top) < sum(rad[i] for i in bottom) / len(bottom)


def test_the_same_seed_reproduces_the_collage(tmp_path) -> None:
    """Organic packing is random but seeded: one seed -> identical pixels, another -> different."""
    root = _make_run(tmp_path, _grid_of(30))
    prim = _primary(tmp_path)
    a, b, c = (tmp_path / f"{n}.png" for n in "abc")
    for path, seed in ((a, 7), (b, 7), (c, 8)):
        generate_collage(root, prim, path, min_archetype_score=0.5, layout="organic",
                         random_seed=seed, max_dim_px=300, layout_px=200)
    read = lambda p: cv2.imread(str(p), cv2.IMREAD_UNCHANGED)  # noqa: E731
    assert np.array_equal(read(a), read(b))
    assert not np.array_equal(read(a), read(c))


def test_shuffle_top_reorders_only_the_head(tmp_path) -> None:
    """shuffle_top varies WHICH archetype gets the prime tile without admitting a weaker leaf."""
    root = _make_run(tmp_path, _grid_of(40))
    prim = _primary(tmp_path)
    plain = generate_collage(root, prim, tmp_path / "p.png", min_archetype_score=0.5,
                             max_dim_px=300, layout_px=200)
    mixed = generate_collage(root, prim, tmp_path / "m.png", min_archetype_score=0.5,
                             shuffle_top=10, random_seed=3, max_dim_px=300, layout_px=200)
    a = [lf["leaf_id"] for lf in json.load(open(plain["manifest"]))["leaves"]][:10]
    b = [lf["leaf_id"] for lf in json.load(open(mixed["manifest"]))["leaves"]][:10]
    assert a != b and set(a) == set(b)               # same ten leaves, different tiles


def test_manifest_records_every_placed_leaf(tmp_path) -> None:
    """The sidecar json is the audit trail: one entry per drawn leaf, with its score and position."""
    root = _make_run(tmp_path, _grid_of(15))
    res = generate_collage(root, _primary(tmp_path), tmp_path / "c.png", min_archetype_score=0.5,
                           max_dim_px=300, layout_px=200)
    doc = json.load(open(res["manifest"]))
    assert len(doc["leaves"]) == res["n_placed"]
    assert all({"leaf_id", "stem", "score", "cx", "cy", "w", "h"} <= set(lf) for lf in doc["leaves"])


def test_max_dim_px_sets_the_longest_side(tmp_path) -> None:
    """The canvas keeps the primary mask's aspect ratio with its longest side at max_dim_px."""
    root = _make_run(tmp_path, _grid_of(10))
    res = generate_collage(root, _primary(tmp_path, w=300, h=400), tmp_path / "c.png",
                           min_archetype_score=0.5, max_dim_px=800, layout_px=256)
    w, h = res["size_px"]
    assert h == 800 and abs(w / h - 300 / 400) < 0.02


def test_primary_mask_with_no_matching_color_is_rejected(tmp_path) -> None:
    """A silhouette that selects nothing must say which color it looked for."""
    root = _make_run(tmp_path, _grid_of(4))
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.zeros((100, 100), np.uint8))
    with pytest.raises(ValueError, match="no pixels matched"):
        generate_collage(root, blank, tmp_path / "c.png", min_archetype_score=0.5, max_dim_px=200)


def test_transparent_edges_are_straight_alpha_not_premultiplied(tmp_path) -> None:
    """Anti-aliased edge pixels must keep the leaf's COLOR, with transparency only in the alpha.

    Storing `color * alpha` in a PNG (which is straight-alpha by definition) darkens every soft edge
    toward black, so a white collage grows a gray hairline once it is dropped on a light background.
    """
    root = _make_run(tmp_path, _grid_of(30))
    out = tmp_path / "c.png"
    generate_collage(root, _primary(tmp_path), out, min_archetype_score=0.5,
                     color=[255, 255, 255], background="transparent",
                     max_dim_px=400, layout_px=256)
    im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    soft = im[(im[:, :, 3] > 0) & (im[:, :, 3] < 255)]
    assert len(soft) > 50                            # the downscale really did produce soft edges
    # premultiplied storage would make RGB track alpha; straight alpha keeps it at the leaf color
    assert soft[:, :3].min() >= 250, f"edge pixels darkened: min channel {soft[:, :3].min()}"


def test_no_leaf_is_cut_by_the_canvas_edge(tmp_path) -> None:
    """The docstring promises leaves spill past the OUTLINE but are never cut. A tile whose cell
    hangs off the canvas would be sliced by a straight border line, so tiles get nudged inside."""
    root = _make_run(tmp_path, _grid_of(50))
    for layout in ("grid", "mosaic", "organic", "puzzle"):
        out = tmp_path / f"{layout}.png"
        res = generate_collage(root, _primary(tmp_path, w=300, h=400), out, min_archetype_score=0.5,
                               layout=layout, max_dim_px=500, layout_px=256, write_manifest=True)
        w, h = res["size_px"]
        for lf in json.load(open(res["manifest"]))["leaves"]:
            assert lf["cx"] - lf["w"] / 2 >= -1 and lf["cx"] + lf["w"] / 2 <= w + 1, (layout, lf)
            assert lf["cy"] - lf["h"] / 2 >= -1 and lf["cy"] + lf["h"] / 2 <= h + 1, (layout, lf)


def test_half_pixel_boxes_still_resolve(tmp_path) -> None:
    """Box edges landing exactly on .5 must round the way the Reporter did.

    SQLite ROUND() is half-away-from-zero while ``int(round(v))`` in core/imaging is half-to-EVEN,
    so rounding in SQL silently loses every leaf whose detection box sits on a half pixel.
    """
    leaves = [(f"HALF_{i}_X_y", (10.5, 20.5, 110.5, 140.5), 0.9, 1) for i in range(1)]
    root = _make_run(tmp_path, leaves)
    got, stats = select_leaves(root, min_archetype_score=0.5)
    assert stats["n_missing_mask"] == 0 and len(got) == 1
    assert Path(got[0]["mask_path"]).name.endswith("__10_20_110_140.png")   # half-to-even


def test_reported_denominator_is_the_scored_count_not_the_passing_count(tmp_path) -> None:
    """'N pass out of M scored' must have a real M, or the ratio is always 100%."""
    root = _make_run(tmp_path, [("A_1_X_y", (0, 0, 50, 50), 0.95, 1),
                                ("B_2_X_y", (0, 0, 50, 50), 0.60, 1),    # scored, below threshold
                                ("C_3_X_y", (0, 0, 50, 50), 0.99, 0)])   # scored, vetoed
    _, stats = select_leaves(root, min_archetype_score=0.8)
    assert stats["n_scored"] == 3 and stats["n_passing"] == 1


def test_color_spellings_all_reach_the_canvas(tmp_path) -> None:
    """'white', '#rrggbb', 'R,G,B' and [R,G,B] are all advertised, so all four must work on the CLI."""
    root = _make_run(tmp_path, _grid_of(10))
    prim = _primary(tmp_path)
    for spec, want_bgr in [("#ff0000", (0, 0, 255)), ("0,255,0", (0, 255, 0)),
                           ([0, 0, 255], (255, 0, 0)), ("white", (255, 255, 255))]:
        out = tmp_path / "c.png"
        generate_collage(root, prim, out, min_archetype_score=0.5, color=spec,
                         max_dim_px=250, layout_px=180)
        im = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
        solid = im[im[:, :, 3] == 255]
        assert tuple(solid[0][:3]) == want_bgr, (spec, tuple(solid[0][:3]))


def test_null_color_and_background_fall_back_to_defaults(tmp_path) -> None:
    """A cleared form field arrives as None; it must not abort AFTER the PNG is already written."""
    root = _make_run(tmp_path, _grid_of(10))
    res = generate_collage(root, _primary(tmp_path), tmp_path / "c.png", min_archetype_score=0.5,
                           color=None, background=None, max_dim_px=250, layout_px=180)
    assert res["color"] == [255, 255, 255] and res["background"] == "transparent"
    assert cv2.imread(res["collage"], cv2.IMREAD_UNCHANGED)[:, :, 3].min() == 0


def test_manifest_paths_are_absolute_even_from_a_relative_run_dir(tmp_path, monkeypatch) -> None:
    """The manifest is the traceability artifact; a CWD-relative path in it is not traceable."""
    root = _make_run(tmp_path, _grid_of(10))
    monkeypatch.chdir(tmp_path)
    res = generate_collage("demo", _primary(tmp_path).name, tmp_path / "c.png",
                           min_archetype_score=0.5, max_dim_px=250, layout_px=180)
    doc = json.load(open(res["manifest"]))
    assert Path(doc["run_dir"]).is_absolute() and Path(doc["primary_mask"]).is_absolute()
    assert all(Path(lf["mask"]).is_absolute() for lf in doc["leaves"])


def test_a_read_only_run_dir_still_produces_a_collage(tmp_path) -> None:
    """Staging falls back to the output folder when the run cannot be written to."""
    root = _make_run(tmp_path, _grid_of(10))
    outdir = tmp_path / "elsewhere"
    outdir.mkdir()
    mode = root.stat().st_mode
    root.chmod(0o555)
    try:
        res = generate_collage(root, _primary(tmp_path), outdir / "c.png", min_archetype_score=0.5,
                               max_dim_px=250, layout_px=180, write_manifest=False)
        assert Path(res["collage"]).is_file()
        assert not (root / "_leaf_collage").exists()
    finally:
        root.chmod(mode)


def test_run_writes_to_reports_collage_by_default(tmp_path) -> None:
    """With no output_dir the collage lands in the run's own reports/Collage folder."""
    root = _make_run(tmp_path, _grid_of(12))
    res = run({"min_archetype_score": 0.5, "max_dim_px": 200, "layout_px": 150},
              run_dir=str(root), primary_mask=str(_primary(tmp_path)))
    out = res[0]["collage"]
    assert out.endswith(".png") and "/reports/Collage/" in out
    assert (root / "reports" / "Collage").is_dir()


# -- puzzle layout -----------------------------------------------------------------
def test_feret_is_the_rotation_invariant_long_dimension() -> None:
    """The size rule promises "longest dimension of the white region", which must not change when
    the leaf is turned. The axis-aligned bounding box does change, which is why it is not the rule."""
    m = np.zeros((80, 40), np.uint8)
    cv2.ellipse(m, (20, 40), (14, 36), 0, 0, 360, 255, -1)
    f0 = _white_feret(m)
    for ang in (0.0, 17.0, 45.0, 73.0, 90.0):
        a, _ = _leaf_piece(m, None, 60.0, ang)
        assert a is not None
        assert abs(_white_feret(a) - 60.0) <= 1.5, f"angle {ang}: got {_white_feret(a)}"
    assert f0 > 0


def test_fit_rotate_cannot_deliver_the_size_rule() -> None:
    """Why _leaf_piece exists. _fit_rotate rotates and then fits the result into a BOX, so the
    drawn long dimension tracks the box: a rotated leaf's bounding box grows, the fit shrinks it to
    compensate, and the leaf itself ends up a different length at every angle. Asking both helpers
    for the same 60 px long dimension makes the difference plain."""
    m = np.zeros((110, 24), np.uint8)                # elongated, where the box effect is largest
    cv2.ellipse(m, (12, 55), (9, 53), 0, 0, 360, 255, -1)
    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    boxed = [_white_feret(_fit_rotate(m, 60.0, 60.0, a, 1.0)) for a in angles]
    exact = [_white_feret(_leaf_piece(m, None, 60.0, a)[0]) for a in angles]
    assert max(exact) / min(exact) < 1.05, f"_leaf_piece should hold the size: {exact}"
    assert all(abs(v - 60.0) <= 2.0 for v in exact), f"_leaf_piece missed the target: {exact}"
    worst = max(abs(v - 60.0) for v in boxed) / 60.0
    assert worst > 0.10, f"_fit_rotate should NOT hold the size, worst miss {worst:.1%}: {boxed}"


def test_puzzle_never_overlaps_two_leaves() -> None:
    """The erosion admission test is a feasibility PROOF, not a penalty: stamped ink must equal the
    sum of the pieces' ink exactly, or two leaves are sharing pixels."""
    dom = np.zeros((300, 300), np.uint8)
    cv2.circle(dom, (150, 150), 140, 1, -1)
    leaf = np.zeros((40, 22), np.uint8)
    cv2.ellipse(leaf, (11, 20), (9, 19), 0, 0, 360, 255, -1)
    pieces = [(leaf, _white_feret(leaf), float((leaf > 127).sum()) / _white_feret(leaf) ** 2)] * 40
    cells = _layout_puzzle(dom, pieces, leaf_px=40.0, gap_px=2.0, angles=6, coarse=3, refine=2,
                           backfill_ratio=0.0, min_leaf_px=6.0, max_shrink=0,
                           rng=np.random.default_rng(0))
    assert cells, "the puzzle placed nothing in a large open circle"
    canvas = np.zeros(dom.shape, np.int32)
    for c in cells:
        k, _ = _leaf_piece(leaf, None, c["long_px"], c["angle"])
        k = (k > 127).astype(np.int32)
        y = int(round(c["cy"] - k.shape[0] / 2.0)); x = int(round(c["cx"] - k.shape[1] / 2.0))
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(dom.shape[0], y + k.shape[0]), min(dom.shape[1], x + k.shape[1])
        canvas[y0:y1, x0:x1] += k[y0 - y:y1 - y, x0 - x:x1 - x]
    assert canvas.max() <= 1, f"{int((canvas > 1).sum())} px covered by two leaves at once"


def test_puzzle_keeps_every_leaf_at_one_size_when_backfill_is_off() -> None:
    """backfill_ratio = 0 is the strict reading of the size rule: no leaf may be shrunk."""
    dom = np.zeros((240, 240), np.uint8)
    cv2.circle(dom, (120, 120), 110, 1, -1)
    leaf = np.zeros((40, 22), np.uint8)
    cv2.ellipse(leaf, (11, 20), (9, 19), 0, 0, 360, 255, -1)
    pieces = [(leaf, _white_feret(leaf), float((leaf > 127).sum()) / _white_feret(leaf) ** 2)] * 30
    cells = _layout_puzzle(dom, pieces, leaf_px=36.0, gap_px=2.0, angles=6, coarse=3, refine=2,
                           backfill_ratio=0.0, min_leaf_px=6.0, max_shrink=3,
                           rng=np.random.default_rng(0))
    assert cells
    assert {c["long_px"] for c in cells} == {36.0}
    assert all(c["shrink"] == 0 for c in cells)


def test_puzzle_stays_inside_the_shape() -> None:
    """Containment comes from erode(borderValue=0); a leaf outside the domain means that flag was
    dropped, which is silent and would only show as leaves floating off the outline."""
    dom = np.zeros((200, 200), np.uint8)
    cv2.circle(dom, (100, 100), 80, 1, -1)
    leaf = np.zeros((30, 16), np.uint8)
    cv2.ellipse(leaf, (8, 15), (6, 14), 0, 0, 360, 255, -1)
    pieces = [(leaf, _white_feret(leaf), float((leaf > 127).sum()) / _white_feret(leaf) ** 2)] * 25
    cells = _layout_puzzle(dom, pieces, leaf_px=28.0, gap_px=2.0, angles=4, coarse=2, refine=2,
                           backfill_ratio=0.0, min_leaf_px=6.0, max_shrink=0,
                           rng=np.random.default_rng(0))
    assert cells
    for c in cells:
        assert 0 <= c["cx"] <= dom.shape[1] and 0 <= c["cy"] <= dom.shape[0]


def test_puzzle_does_not_hang_on_a_leaf_bigger_than_the_shape() -> None:
    """A leaf size a GUI slider can reach must fail fast, not turn into an O(size^4) erosion."""
    import time
    dom = np.zeros((200, 200), np.uint8)
    cv2.circle(dom, (100, 100), 60, 1, -1)
    leaf = np.zeros((30, 16), np.uint8)
    cv2.ellipse(leaf, (8, 15), (6, 14), 0, 0, 360, 255, -1)
    pieces = [(leaf, _white_feret(leaf), float((leaf > 127).sum()) / _white_feret(leaf) ** 2)] * 5
    t0 = time.perf_counter()
    _layout_puzzle(dom, pieces, leaf_px=4000.0, gap_px=2.0, angles=4, coarse=4, refine=2,
                   backfill_ratio=0.0, min_leaf_px=6.0, max_shrink=0,
                   rng=np.random.default_rng(0))
    assert time.perf_counter() - t0 < 5.0


def test_puzzle_does_not_mutate_the_silhouette_it_is_given() -> None:
    """_layout_view hands back the SAME array when no downscale is needed, so an in-place write
    here would corrupt the silhouette every later stage reads."""
    dom = np.zeros((160, 160), np.uint8)
    cv2.circle(dom, (80, 80), 70, 1, -1)
    before = dom.copy()
    leaf = np.zeros((30, 16), np.uint8)
    cv2.ellipse(leaf, (8, 15), (6, 14), 0, 0, 360, 255, -1)
    pieces = [(leaf, _white_feret(leaf), float((leaf > 127).sum()) / _white_feret(leaf) ** 2)] * 10
    _layout_puzzle(dom, pieces, leaf_px=26.0, gap_px=2.0, angles=4, coarse=2, refine=2,
                   backfill_ratio=0.0, min_leaf_px=6.0, max_shrink=0,
                   rng=np.random.default_rng(0))
    assert np.array_equal(dom, before)
    assert _layout_puzzle(dom.astype(bool), pieces, leaf_px=26.0, gap_px=2.0, angles=4, coarse=2,
                          refine=2, backfill_ratio=0.0, min_leaf_px=6.0, max_shrink=0,
                          rng=np.random.default_rng(0)) is not None


def test_puzzle_size_solves_to_the_requested_coverage() -> None:
    """The closed form is the whole reason there is no binary search: total leaf ink = fill * area."""
    ratios = [0.30, 0.25, 0.28]
    px = _puzzle_leaf_px(10000.0, ratios, 0.5)
    assert abs(sum(r * px * px for r in ratios) - 5000.0) < 1e-6


def test_puzzle_reports_its_density_and_size_discipline(tmp_path) -> None:
    """The summary must expose the tradeoff -- coverage, and how many leaves kept the asked-for
    size -- because neither can be recovered from the PNG afterwards."""
    root = _make_run(tmp_path, _grid_of(40))
    res = generate_collage(root, _primary(tmp_path), tmp_path / "p.png", min_archetype_score=0.5,
                           layout="puzzle", max_dim_px=400, layout_px=256, puzzle_nest_px=256)
    assert 0.0 < res["puzzle_coverage"] <= 1.0
    at, tot = res["puzzle_at_full_size"]
    assert 0 < at <= tot == res["n_placed"]
    assert res["puzzle_leaf_px"] > 0


def test_leaf_order_random_breaks_up_the_incoming_order(tmp_path) -> None:
    """Placement order is a ranking in every layout, so a caller that hands over leaves grouped by
    class gets visible bands. leaf_order="random" must scramble that, stay seeded, and reach all
    four layouts -- not just the two that re-sort their cells."""
    root = _make_run(tmp_path, _grid_of(40))
    for layout in ("grid", "mosaic", "organic", "puzzle"):
        seen = {}
        for order, seed in (("score", 0), ("random", 0), ("random", 0), ("random", 7)):
            res = generate_collage(root, _primary(tmp_path), tmp_path / f"{layout}{order}{seed}.png",
                                   min_archetype_score=0.5, layout=layout, max_dim_px=400,
                                   layout_px=256, puzzle_nest_px=256, leaf_order=order,
                                   random_seed=seed, write_manifest=True)
            ids = [lf["leaf_id"] for lf in json.load(open(res["manifest"]))["leaves"]]
            seen.setdefault((order, seed), []).append(ids)
            assert res["leaf_order"] == order
        assert seen[("random", 0)][0] == seen[("random", 0)][1], f"{layout}: not reproducible"
        assert seen[("random", 0)][0] != seen[("score", 0)][0], f"{layout}: order unchanged"
        assert seen[("random", 7)][0] != seen[("random", 0)][0], f"{layout}: seed ignored"


# -- section 2.8: the standalone CLI uses the same guard as the HTTP API ------------
# Plan section 2.8: a postprocessor "is **refused** when its target is the currently active
# pipeline's run directory", two ``read_write`` tools on one completed run "are serialized by a
# per-run advisory lock", and -- last bullet -- "Standalone CLI tools use the same guard as the
# HTTP API, not a parallel one." These tests are about the CLI actually REACHING that guard;
# the guard's own behavior is pinned in tests/test_progress_registry.py.

_FLAG = "LM3_RUNTIME_V2"

_FAKE_RESULT = {"collage": "/tmp/x/collage.png", "size_px": (10, 10), "n_placed": 1,
                "n_passing": 1, "layout": "grid", "score_range": (0.9, 0.9)}


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A private deployment: its own runtime dir (where the advisory locks live), its own roots.

    The flag goes on explicitly rather than being inherited -- a developer with ``LM3_RUNTIME_V2``
    exported would otherwise run these in the opposite mode from CI. Individual tests turn it off
    again to assert the flag-off path.
    """
    import yaml

    from leafmachine3.server import postprocess_api

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("LM3_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("LM3_DEPLOYMENT_ID", "test-collage-guard")
    monkeypatch.setenv("LM3_POSTPROCESS_ROOTS", str(tmp_path))
    settings = tmp_path / "LM3_settings.yaml"
    settings.write_text(
        yaml.safe_dump({"project": {"run_name": "", "output": {"dir": str(tmp_path / "runs")}}}),
        encoding="utf-8")
    monkeypatch.setenv("LM3_SETTINGS", str(settings))
    monkeypatch.setenv(_FLAG, "1")
    postprocess_api._LOCAL_LOCKS.clear()
    yield
    postprocess_api._LOCAL_LOCKS.clear()


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """``main(argv)`` with the real runner replaced, plus the calls it received.

    Whether the collage is actually built is beside the point here: a refusal must happen BEFORE
    the runner is reached, and an allowed invocation must still reach it.
    """
    from leafmachine3.postprocessing import generate_leaf_collage as mod

    calls: list = []

    def fake_run(settings=None, run_dir=None, primary_mask=None, output_dir=None, **kw):
        calls.append({"run_dir": run_dir, "primary_mask": primary_mask, "output_dir": output_dir})
        return [dict(_FAKE_RESULT)]

    monkeypatch.setattr(mod, "run", fake_run)
    cfg = tmp_path / "postprocessing.yaml"
    cfg.write_text("generate_leaf_collage: {}\n", encoding="utf-8")

    def invoke(*args):
        return mod.main(["--config", str(cfg), *[str(a) for a in args]])

    invoke.calls = calls                     # a handle for the assertions, not an API
    return invoke


def _run_dir(root: Path, name: str = "demo") -> Path:
    """A directory ``postprocess_api._is_run_dir`` recognizes -- ledger and reports tree."""
    d = root / name
    (d / "reports").mkdir(parents=True)
    sqlite3.connect(d / f"{name}.sqlite").close()
    return d


@contextlib.contextmanager
def _live_pipeline(run_dir: Path):
    """Publish ``run_dir`` as the active ``pipeline`` record while genuinely holding the lease.

    Nothing here is stubbed. ``active_run_target`` reads ``active.json`` off disk and classifies it
    off the OS lock (section 2.9), so a record written WITHOUT a lease reads ``abandoned`` and
    refuses nothing -- which would make every assertion below pass for the wrong reason.
    """
    from leafmachine3.core import paths as core_paths
    from leafmachine3.core.runtime import (
        Activity, ActivityRole, ArchiveMode, ArchiveStatus, ConfigRef, Launcher, ProjectBlock,
        RunState, RuntimeRecord,
    )
    from leafmachine3.core.runtime import lease as lease_mod
    from leafmachine3.core.runtime import records as records_mod

    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    config = run_dir / "used_at_launch.yaml"
    config.write_text("version: 3\n", encoding="utf-8")
    db = run_dir / f"{run_dir.name}.sqlite"
    now = records_mod.utc_now()
    record = RuntimeRecord(
        run_id="rid-live", activity=Activity.PIPELINE, activity_role=ActivityRole.ROOT,
        state=RunState.RUNNING, launcher=Launcher.CLI, pid=os.getpid(),
        process_started_at=records_mod.process_start_time(), started_at=now, updated_at=now,
        deployment=records_mod.build_deployment_info(),
        config=ConfigRef(path=str(config), sha256="0" * 64),
        project=ProjectBlock(
            run_name=run_dir.name, input_dirs=(str(run_dir.parent),), artifact_dir=str(run_dir),
            active_state_dir=str(run_dir), active_db_path=str(db),
            archive_mode=ArchiveMode.IN_PLACE, archive_status=ArchiveStatus.NOT_APPLICABLE,
            archive_pointer_path=None, archived_db_path=str(db), run_dir=str(run_dir),
            log_path=str(run_dir / "logs" / "lm3.log"),
        ),
    )
    lease = lease_mod.RuntimeLease()
    lease.acquire()
    store = records_mod.RecordStore(
        core_paths.deployment_runtime_dir(create=True, check_filesystem=False), run_id=record.run_id)
    try:
        store.write_active(record)
        yield record
    finally:
        with contextlib.suppress(Exception):
            lease.release()


def test_the_cli_refuses_the_run_a_pipeline_is_writing(tmp_path, capsys, deployment, cli) -> None:
    """Section 2.8, bullet 2, reached from the CLI rather than from the HTTP route."""
    live = _run_dir(tmp_path / "runs", "live")
    with _live_pipeline(live):
        code = cli("--run-dir", live, "--primary-mask", _primary(tmp_path))

    assert code == 3 and code != 0, "a refused CLI invocation must not exit 0"
    assert not cli.calls, "the runner ran anyway -- the guard came too late to matter"
    err = capsys.readouterr().err
    assert "refused" in err and "live" in err


def test_a_different_completed_run_is_still_allowed(tmp_path, deployment, cli) -> None:
    """Section 2.8, bullet 3: a live pipeline does not put every other run off limits."""
    done = _run_dir(tmp_path / "runs", "finished")
    with _live_pipeline(_run_dir(tmp_path / "runs", "live")):
        assert cli("--run-dir", done, "--primary-mask", _primary(tmp_path)) == 0
    assert len(cli.calls) == 1


def test_a_symlinked_spelling_of_the_live_run_is_refused_too(tmp_path, deployment, cli) -> None:
    """The reason the guard is fed through ``validate_params`` and not raw argv.

    ``active_run_target`` realpaths the record's root, so a target that was never resolved would
    simply never meet it -- and an output root reached through a symlink is the normal case on a
    cluster, not an exotic one.
    """
    real = tmp_path / "real_runs"
    live = _run_dir(real, "live")
    link = tmp_path / "runs_link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):          # pragma: no cover - unprivileged Windows
        pytest.skip("this filesystem/user cannot create symlinks")
    with _live_pipeline(live):
        assert cli("--run-dir", link / "live", "--primary-mask", _primary(tmp_path)) == 3
    assert not cli.calls


def test_the_advisory_lock_is_held_across_the_whole_run(tmp_path, monkeypatch, deployment,
                                                        cli) -> None:
    """Section 2.8, bullet 5. A lock taken after argparse and dropped before ``run()`` serializes
    nothing, so the assertion is made from INSIDE the runner, and again after ``main`` returns."""
    from leafmachine3.postprocessing import generate_leaf_collage as mod
    from leafmachine3.server import postprocess_api as pp

    done = _run_dir(tmp_path / "runs", "finished")
    tool = pp.get_tool(mod.TOOL_ID)
    seen: list = []

    def fake_run(settings=None, run_dir=None, primary_mask=None, output_dir=None, **kw):
        with pytest.raises(pp.TargetLocked):
            pp._acquire_artifact_locks(tool, [done])
        seen.append(True)
        return [dict(_FAKE_RESULT)]

    monkeypatch.setattr(mod, "run", fake_run)

    assert cli("--run-dir", done, "--primary-mask", _primary(tmp_path)) == 0
    assert seen, "the runner never ran, so nothing was proved about the lock"
    after = pp._acquire_artifact_locks(tool, [done])
    assert after, "the lock outlived main() -- this run dir is now wedged"
    for lock in after:
        lock.release()


def test_the_lock_is_released_even_when_the_run_raises(tmp_path, monkeypatch, deployment,
                                                       cli) -> None:
    from leafmachine3.postprocessing import generate_leaf_collage as mod
    from leafmachine3.server import postprocess_api as pp

    done = _run_dir(tmp_path / "runs", "finished")
    monkeypatch.setattr(mod, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        cli("--run-dir", done, "--primary-mask", _primary(tmp_path))
    held = pp._acquire_artifact_locks(pp.get_tool(mod.TOOL_ID), [done])
    assert held, "a crashing runner leaked the advisory lock"
    for lock in held:
        lock.release()


def test_a_second_writer_on_one_completed_run_is_refused(tmp_path, deployment, cli) -> None:
    """The other half of bullet 5: the CLI is the one that loses when the lock is already held."""
    from leafmachine3.postprocessing import generate_leaf_collage as mod
    from leafmachine3.server import postprocess_api as pp

    done = _run_dir(tmp_path / "runs", "finished")
    held = pp._acquire_artifact_locks(pp.get_tool(mod.TOOL_ID), [done])
    try:
        assert cli("--run-dir", done, "--primary-mask", _primary(tmp_path)) == 3
        assert not cli.calls
    finally:
        for lock in held:
            lock.release()


def test_the_yaml_supplied_run_dir_is_guarded_too(tmp_path, monkeypatch, deployment) -> None:
    """``--run-dir`` is optional: the run just as often comes from the settings block, and a guard
    that only looked at argv would wave that straight through."""
    from leafmachine3.postprocessing import generate_leaf_collage as mod

    live = _run_dir(tmp_path / "runs", "live")
    calls: list = []
    monkeypatch.setattr(mod, "run", lambda *a, **k: calls.append(1) or [dict(_FAKE_RESULT)])
    cfg = tmp_path / "postprocessing.yaml"
    cfg.write_text(
        f"generate_leaf_collage:\n  run_dir: {live}\n  primary_mask: {_primary(tmp_path)}\n",
        encoding="utf-8")

    with _live_pipeline(live):
        assert mod.main(["--config", str(cfg)]) == 3
    assert not calls


def test_with_the_flag_off_nothing_is_guarded(tmp_path, monkeypatch, deployment, cli) -> None:
    """The Step 3 contract: with ``LM3_RUNTIME_V2`` off the CLI behaves exactly as it shipped --
    same target, same exit code, no refusal, no server package in the way."""
    live = _run_dir(tmp_path / "runs", "live")
    with _live_pipeline(live):
        monkeypatch.setenv(_FLAG, "0")   # only AFTER the record is published and leased
        assert cli("--run-dir", live, "--primary-mask", _primary(tmp_path)) == 0
    assert len(cli.calls) == 1 and cli.calls[0]["run_dir"] == str(live)


def test_a_missing_run_dir_still_gets_the_tools_own_error(tmp_path, monkeypatch, deployment,
                                                          cli) -> None:
    """The guard must not turn "you forgot --run-dir" into a param-validation refusal: ``run()``
    already refuses, with the message that says which key to set."""
    from leafmachine3.postprocessing import generate_leaf_collage as mod

    def boom(settings=None, run_dir=None, primary_mask=None, output_dir=None, **kw):
        raise ValueError("no run: set generate_leaf_collage.run_dir in the yaml or pass --run-dir")

    monkeypatch.setattr(mod, "run", boom)
    with pytest.raises(ValueError, match="no run:"):
        cli("--primary-mask", _primary(tmp_path))


def test_a_public_guard_wrapper_is_preferred_when_postprocess_api_grows_one(
    tmp_path, monkeypatch, deployment, cli
) -> None:
    """Section 2.8's last bullet is about there being ONE guard. The composition above is a
    stand-in for the single public wrapper that belongs in ``postprocess_api``; the moment that
    wrapper exists this CLI must route through it instead, or the two start to drift."""
    from leafmachine3.server import postprocess_api as pp

    seen: list = []

    @contextlib.contextmanager
    def fake_guard(tool_id, params):
        seen.append((tool_id, dict(params)))
        yield []

    monkeypatch.setattr(pp, "guard_cli", fake_guard, raising=False)
    done = _run_dir(tmp_path / "runs", "finished")

    assert cli("--run-dir", done, "--primary-mask", _primary(tmp_path)) == 0
    assert seen and seen[0][0] == "generate_leaf_collage"
    assert seen[0][1]["run_dir"] == str(done)
