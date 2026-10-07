"""generate_leaf_collage -- tile a run's best leaf masks into the shape of one PRIMARY mask.

Reads a finished LM3 run's project database, keeps the leaves that cleared every structural veto
(``bilateral_symmetry.gates_pass = 1``) AND scored above ``min_archetype_score``, then arranges
their per-leaf mask PNGs inside the silhouette of a user-supplied primary mask -- a leaf built out
of leaves. Nothing is segmented or re-measured here: this tool only READS existing Reporter output.

Three arrangements, all of which place the highest-scoring leaves deepest inside the silhouette:

    grid      one square cell per leaf; the cell size is solved so the number of cells whose
              CENTER falls inside the silhouette matches the number of qualifying leaves
    mosaic    a quadtree -- large tiles in the interior, recursively subdivided toward the
              boundary, so the outline stays crisp and the tile sizes vary
    organic   greedy distance-transform packing -- each leaf drops into the largest remaining
              empty pocket, at that pocket's scale and (optionally) a random rotation

A cell is kept when its center is inside the silhouette, so leaves may spill slightly past the
outline; no leaf is ever cut. Leaves render either as their binary mask recolored to ``color``, or
as the matching RGB cutout, over a transparent or solid background.

Standalone postprocessing tool -- NOT part of the pipeline. Configure in ``postprocessing_settings.yaml``:

    .venv_LM3/bin/python -m leafmachine3.postprocessing.generate_leaf_collage \
        --config postprocessing_settings.yaml --run-dir <run> --primary-mask <mask.png> \
        --layout grid --style mask --color 255 255 255 --max-dim-px 10000

The CLI runs under plan section 2.8's postprocessing guard (:func:`cli_target_guard`): it refuses
a run a pipeline is writing right now, and serializes itself against any other read/write tool on
the same finished run. Both are inert until ``LM3_RUNTIME_V2`` is on.

Heavy deps (cv2, numpy) are imported lazily so the module imports without them.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger("leafmachine3.postprocessing.generate_leaf_collage")

# Defaults mirror postprocessing_settings.yaml (used when a key is absent).
_DEFAULTS: dict[str, Any] = {
    "run_dir": None, "primary_mask": None, "output_dir": None, "name": None,
    "min_archetype_score": 0.8, "max_leaves": 0,
    "tree": "Leaf_Oriented", "mask_variant": "lamina_mask", "style": "mask",
    "layout": "grid", "ranking": "center", "max_dim_px": 10000,
    "color": [255, 255, 255], "background": "transparent",
    "primary_colors": ["white"], "primary_color_tolerance": 0, "primary_fill_holes": True,
    "tile_scale": 0.98, "random_seed": 0, "shuffle_top": 0, "leaf_order": "score",
    "mosaic_min_cell_px": 24,
    "organic_rotate": True, "organic_fill": 0.9, "organic_gap_px": 2, "organic_min_tile_px": 12,
    "organic_max_scale": 3.0,
    # puzzle: true shape nesting. puzzle_fill is the TARGET ink coverage -- it sets the leaf size
    # through a closed form, so 0.6 means "aim to ink 60% of the shape". Measured on real leaves,
    # 0.60 places every leaf with ~96% of them at the requested size; pushing higher buys a few
    # points of ink by shrinking a growing minority through the backfill queue.
    "puzzle_leaf_px": 0, "puzzle_fill": 0.6, "puzzle_gap_px": 2, "puzzle_angles": 8,
    "puzzle_coarse": 6, "puzzle_refine": 3, "puzzle_nest_px": 3072, "puzzle_overhang": 0.0,
    "puzzle_backfill_ratio": 0.72, "puzzle_min_leaf_px": 12, "puzzle_max_shrink": 3,
    "layout_px": 2048, "tmp_dir": None, "write_manifest": True,
}

# The layout is solved on a DOWNSCALED copy of the silhouette and the resulting cell geometry is
# scaled up to the canvas. Inside/outside tests and distance transforms cost O(pixels), and a
# 10000 px canvas would make the quadtree's integral image alone 400 MB for no added fidelity.
_LAYOUT_PX_CAP = 8192

# One entry per leaf-mask product the Reporter writes (leafmachine3/reporting/leaf_products.py).
# (mask folder, mask filename label, RGB folder, RGB filename label). A None RGB pair means the
# variant has no cutout sibling, so ``style: rgb`` is not available for it.
# The labels here are UNTAGGED: the Reporter prefixes them with the tree tag (og-/or-), which is
# folded in at lookup time by ``_labels_for_tree`` -- and which older runs predate.
_VARIANTS: dict[str, tuple[str, str, Optional[str], Optional[str]]] = {
    "lamina_mask": ("Lamina_Mask", "SEG-lamina", "Lamina_RGB", "RGB-lamina"),
    "lamina_holes_mask": ("Lamina_Holes_Mask", "SEG-laminaHoles", "Lamina_Holes_RGB", "RGB-laminaHoles"),
    "lamina_petiole_mask": ("LaminaPetiole_Mask", "SEG-laminaPetiole", "LaminaPetiole_RGB", "RGB-laminaPetiole"),
    "lamina_petiole_holes_mask": ("LaminaPetiole_Holes_Mask", "SEG-laminaPetioleHoles", None, None),
}
_TREES = ("Leaf_Oriented", "Leaf_Original")
# Runs made before the tree tag existed hold the untagged names, and this tool reads finished runs
# on disk rather than re-running them -- so every lookup tries the tagged name first, then the old
# one. Dropping the fallback would silently make old runs look like they contain no leaves at all.
_TREE_TAGS = {"Leaf_Original": "og", "Leaf_Oriented": "or"}
_LAYOUTS = ("grid", "mosaic", "organic", "puzzle")
_RANKINGS = ("center", "reading", "random")
_LEAF_ORDERS = ("score", "random")
_STYLES = ("mask", "rgb")

# The Reporter's extension is configurable (report.formats.mask_ext / image_ext), and this tool is
# decoupled from LM3_settings.yaml -- so the file is found by trying the plausible suffixes.
_MASK_EXTS = (".png", ".tif", ".tiff", ".bmp")
_RGB_EXTS = (".jpg", ".jpeg", ".png", ".webp")


# -- database ----------------------------------------------------------------------
def _db_path(run_dir) -> Path:
    """The project ledger inside a run directory: ``<run>/<run>.sqlite`` (``run.sqlite`` is legacy)."""
    root = Path(run_dir)
    for cand in (root / f"{root.name}.sqlite", root / "run.sqlite"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"no project database in {root} -- expected {root.name}.sqlite (is this an LM3 run dir?)"
    )


def _open_db_readonly(db_path):
    """Open the project DB strictly read-only, over WAL, without ever writing DDL.

    ``ProjectDB.open_or_create`` applies the schema and seeds rows -- it must never be used to read
    a finished run. This mirrors ``leafmachine3.server.results_api._readonly_connect``: try
    ``mode=ro``, fall back to ``immutable=1`` for a run directory that is not writable, and force a
    real read so a bad handle fails here rather than at the first query.
    """
    import sqlite3

    last: Optional[Exception] = None
    for uri in (f"file:{db_path}?mode=ro", f"file:{db_path}?mode=ro&immutable=1"):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            return conn
        except sqlite3.Error as exc:
            last = exc
    raise RuntimeError(f"cannot open {db_path} read-only: {last}")


_SELECT_SQL = """
SELECT s.image_stem                              AS stem,
       ls.leaf_id, ls.detection_id, ls.instance_index,
       pd.x1, pd.y1, pd.x2, pd.y2,
       bs.archetype_score, bs.gates_pass
  FROM leaf_segmentation  ls
  JOIN specimen           s  ON s.specimen_id   = ls.specimen_id
  JOIN plant_detection    pd ON pd.detection_id = ls.detection_id
  JOIN bilateral_symmetry bs ON bs.leaf_id      = ls.leaf_id
 WHERE ls.cls_name = 'Leaf'
   AND pd.suppressed = 0
   AND bs.gates_pass = 1
   AND bs.archetype_score > ?
 ORDER BY bs.archetype_score DESC, ls.leaf_id ASC
"""


def _require_bilateral(conn, db_path) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='bilateral_symmetry'"
    ).fetchone()
    if row is None:
        raise ValueError(
            f"{db_path} has no bilateral_symmetry table -- there are no archetype scores to rank "
            f"by. Re-run the project with modules.bilateral_symmetry.enabled: true."
        )


def _box_of(row) -> tuple[int, int, int, int]:
    """The detection box as the Reporter spelled it into the filename.

    Rounding happens HERE, in Python, not in SQL: ``core/imaging.crop_filename`` uses ``int(round(v))``
    (banker's rounding, half-to-EVEN) while SQLite's ROUND() is half-AWAY-from-zero, so a box edge
    landing exactly on .5 would reconstruct a filename that does not exist and silently drop the leaf.
    """
    return (int(round(row["x1"])), int(round(row["y1"])),
            int(round(row["x2"])), int(round(row["y2"])))


def _count_scored(conn) -> int:
    """Every scored, non-suppressed Leaf row -- the DENOMINATOR the pass count is reported against."""
    return int(conn.execute(
        "SELECT COUNT(*) FROM leaf_segmentation ls "
        "JOIN plant_detection pd ON pd.detection_id = ls.detection_id "
        "JOIN bilateral_symmetry bs ON bs.leaf_id = ls.leaf_id "
        "WHERE ls.cls_name = 'Leaf' AND pd.suppressed = 0"
    ).fetchone()[0])


def _find_file(folder: Path, base: str, exts) -> Optional[Path]:
    """The first ``<folder>/<base><ext>`` that exists (the Reporter's extension is configurable)."""
    for ext in exts:
        p = folder / f"{base}{ext}"
        if p.is_file():
            return p
    return None


def _labels_for_tree(tree: str, *labels: Optional[str]) -> list[tuple[str, ...]]:
    """Each untagged product label -> the names to try, current first: ``("or-SEG-lamina",
    "SEG-lamina")``. The second is the pre-tag layout still on disk in older runs."""
    tag = _TREE_TAGS[tree]
    return [() if lab is None else (f"{tag}-{lab}", lab) for lab in labels]


def _find_leaf_file(folder: Optional[Path], stem: str, labels: tuple[str, ...], box: str,
                    exts) -> tuple[Optional[Path], Optional[str]]:
    """Locate one leaf product by trying each candidate label. Returns ``(path, base)`` so the
    caller can dedupe on the base name it actually matched."""
    if folder is None:
        return None, None
    for lab in labels:
        base = f"{stem}__{lab}__{box}"
        found = _find_file(folder, base, exts)
        if found is not None:
            return found, base
    return None, None


def leaf_scores(run_dir, *, min_score: float = 0.0) -> dict:
    """``{(image_stem, (x1, y1, x2, y2)): archetype_score}`` for every leaf that passes.

    Keyed by stem + detection box rather than by path, because ONE key identifies the same leaf in
    every leaf-product folder (``Lamina_Mask``, ``Lamina_RGB``, ...). The GUI mask picker uses this
    to show only non-vetoed, high-scoring masks. Vetoed leaves (``gates_pass = 0``) never appear.
    """
    db = _db_path(run_dir)
    conn = _open_db_readonly(db)
    try:
        _require_bilateral(conn, db)
        rows = conn.execute(_SELECT_SQL, (float(min_score),)).fetchall()
    finally:
        conn.close()
    out: dict = {}
    for r in rows:
        key = (str(r["stem"]), _box_of(r))
        score = float(r["archetype_score"])
        if score > out.get(key, -1.0):        # a multi-instance detection shares one file: keep the best
            out[key] = score
    return out


def select_leaves(
    run_dir,
    *,
    min_archetype_score: float = 0.8,
    max_leaves: int = 0,
    tree: str = "Leaf_Oriented",
    mask_variant: str = "lamina_mask",
    style: str = "mask",
) -> tuple[list[dict], dict]:
    """Pick the qualifying leaves and resolve each one to its mask (and RGB) file on disk.

    Returns ``(leaves, stats)``, best score first. Every leaf dict carries ``leaf_id``, ``stem``,
    ``score``, ``mask_path`` and (in ``rgb`` style) ``rgb_path``.

    Two facts drive the bookkeeping here. (1) Paths are NOT stored per leaf -- they are rebuilt from
    ``<run>/reports/<tree>/<folder>/<stem>__<LABEL>__x1_y1_x2_y2.<ext>``, and ``run_dir`` must come
    from the DB file's own parent because a stored path may be relative to a different CWD.
    (2) The Reporter names leaf products PER DETECTION, so when one detection holds several Leaf
    instances they all map to a single merged raster -- those rows are de-duplicated to one tile and
    counted in ``stats["n_merged"]`` rather than tiled once per leaf_id.
    """
    if mask_variant not in _VARIANTS:
        raise ValueError(f"mask_variant must be one of {list(_VARIANTS)}, got {mask_variant!r}")
    if tree not in _TREES:
        raise ValueError(f"tree must be one of {list(_TREES)}, got {tree!r}")
    if style not in _STYLES:
        raise ValueError(f"style must be one of {list(_STYLES)}, got {style!r}")

    mask_folder, mask_label, rgb_folder, rgb_label = _VARIANTS[mask_variant]
    mask_labels, rgb_labels = _labels_for_tree(tree, mask_label, rgb_label)
    if style == "rgb" and rgb_folder is None:
        raise ValueError(
            f"mask_variant {mask_variant!r} has no RGB cutout sibling -- use style 'mask', or pick "
            f"a variant among {[k for k, v in _VARIANTS.items() if v[2]]}"
        )

    root = Path(run_dir)
    db = _db_path(root)
    root = Path(os.path.realpath(db)).parent           # authoritative run root (never a stored path)
    conn = _open_db_readonly(db)
    try:
        _require_bilateral(conn, db)
        rows = conn.execute(_SELECT_SQL, (float(min_archetype_score),)).fetchall()
        n_scored = _count_scored(conn)
    finally:
        conn.close()

    mask_dir = root / "reports" / tree / mask_folder
    rgb_dir = (root / "reports" / tree / rgb_folder) if rgb_folder else None

    leaves: list[dict] = []
    seen: dict[str, int] = {}
    n_merged = n_missing_mask = n_missing_rgb = 0
    for r in rows:
        box = "{}_{}_{}_{}".format(*_box_of(r))
        # Dedupe on stem+box, not on the matched filename: several Leaf instances share one raster,
        # and which of the tagged/untagged names matched must not change what counts as a duplicate.
        key = f"{r['stem']}__{box}"
        if key in seen:
            n_merged += 1
            continue
        mask_path, _ = _find_leaf_file(mask_dir, str(r["stem"]), mask_labels, box, _MASK_EXTS)
        if mask_path is None:
            n_missing_mask += 1
            continue
        rgb_path = None
        if style == "rgb":
            rgb_path, _ = _find_leaf_file(rgb_dir, str(r["stem"]), rgb_labels, box, _RGB_EXTS)
            if rgb_path is None:
                n_missing_rgb += 1
                continue
        seen[key] = 1
        leaves.append({
            "leaf_id": int(r["leaf_id"]),
            "stem": str(r["stem"]),
            "score": float(r["archetype_score"]),
            "mask_path": str(mask_path),
            "rgb_path": str(rgb_path) if rgb_path else None,
        })

    n_passing = len(leaves)
    if max_leaves and int(max_leaves) > 0:
        leaves = leaves[: int(max_leaves)]              # already sorted best-first
    stats = {
        "n_scored": n_scored, "n_over_threshold": len(rows),
        "n_passing": n_passing, "n_used": len(leaves),
        "n_merged": n_merged, "n_missing_mask": n_missing_mask, "n_missing_rgb": n_missing_rgb,
        "mask_dir": str(mask_dir), "rgb_dir": str(rgb_dir) if rgb_dir else None,
    }
    if n_merged:
        log.warning("%d leaf row(s) share a mask file with another leaf (multi-instance detections) "
                    "and were collapsed to one tile each", n_merged)
    if n_missing_mask:
        # Two ordinary causes: Leaf_Oriented is written only where the orientation stage succeeded,
        # and the petiole variants are skipped entirely for a leaf with no petiole mask.
        log.warning("%d qualifying leaf/leaves have no %s file in %s -- %s", n_missing_mask,
                    mask_folder, mask_dir,
                    "Leaf_Oriented is only written when a leaf's orientation succeeded"
                    if tree == "Leaf_Oriented" else
                    "the petiole variants are skipped for leaves with no petiole"
                    if "Petiole" in mask_folder else "the Reporter did not write it")
    if n_missing_rgb:
        log.warning("%d qualifying leaf/leaves have no %s cutout in %s", n_missing_rgb, rgb_folder, rgb_dir)
    if not leaves:
        raise ValueError(
            f"no usable leaves: {len(rows)} of {n_scored} scored row(s) cleared the vetoes and "
            f"scored above {min_archetype_score}, but none resolved to a file under {mask_dir}. "
            f"Lower min_archetype_score, pick another tree/mask_variant, or re-run the Reporter."
        )
    return leaves, stats


# -- silhouette --------------------------------------------------------------------
def _load_silhouette(primary_mask, colors, tol: int, fill_holes: bool, max_dim: int):
    """Read the primary mask, trim it to its content, and scale its longest side to ``max_dim``."""
    import cv2
    import numpy as np

    # Same color-selection semantics as the STL builder, deliberately shared so "white" and a
    # tolerance mean exactly the same thing in both tools.
    from leafmachine3.postprocessing.generate_stl_from_mask import _fill_holes, _select_mask

    mask = _select_mask(primary_mask, colors, tol)
    if not mask.any():
        raise ValueError(
            f"no pixels matched colors={colors} (tolerance={tol}) in {primary_mask} -- LM3 binary "
            f"masks are white on black, so 'white' is normally right"
        )
    if fill_holes:
        mask = _fill_holes(mask)
    ys, xs = np.where(mask)
    mask = mask[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = mask.shape
    scale = float(max_dim) / float(max(h, w))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    big = cv2.resize(mask.astype(np.uint8) * 255, (new_w, new_h), interpolation=interp)
    return big > 127


def _layout_view(sil, layout_px: int):
    """A downscaled copy of the silhouette for solving the layout, plus its canvas scale factor."""
    import cv2
    import numpy as np

    H, W = sil.shape
    cap = max(64, min(int(layout_px), _LAYOUT_PX_CAP))
    if max(H, W) <= cap:
        return sil, 1.0
    scale = cap / float(max(H, W))
    small = cv2.resize(sil.astype(np.uint8) * 255,
                       (max(1, int(round(W * scale))), max(1, int(round(H * scale)))),
                       interpolation=cv2.INTER_AREA) > 127
    if not small.any():                                # a hairline silhouette can vanish -- don't
        return sil, 1.0
    return small, float(W) / float(small.shape[1])


def _depth_map(sil):
    """Distance (px) from every foreground pixel to the nearest background pixel."""
    import cv2
    import numpy as np

    return cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


# -- layout: grid ------------------------------------------------------------------
def _grid_probe(sil, s: float):
    """Cell geometry for square-ish cells of side ~``s``, plus which cell centers are inside."""
    import numpy as np

    H, W = sil.shape
    n_cols = max(1, int(round(W / max(s, 1e-6))))
    n_rows = max(1, int(round(H / max(s, 1e-6))))
    cw, ch = W / n_cols, H / n_rows
    cx = (np.arange(n_cols) + 0.5) * cw
    cy = (np.arange(n_rows) + 0.5) * ch
    ix = np.clip(cx.astype(np.int64), 0, W - 1)
    iy = np.clip(cy.astype(np.int64), 0, H - 1)
    return n_rows, n_cols, cw, ch, cx, cy, sil[np.ix_(iy, ix)]


def _layout_grid(sil, n: int):
    """Solve for the LARGEST square cell whose in-silhouette count still reaches ``n`` leaves.

    The count falls monotonically as the cell grows, so the size is bisected around the analytic
    estimate ``sqrt(area / n)`` -- starting from an estimate keeps the search off the pathological
    end where a one-pixel cell would mean a hundred million probes.
    """
    area = float(sil.sum())
    est = max(2.0, math.sqrt(area / max(1, n)))
    lo, hi = max(1.5, est / 4.0), est * 4.0
    limit = float(max(sil.shape))
    while hi < limit and _grid_probe(sil, hi)[6].sum() >= n:
        hi = min(limit, hi * 2.0)                      # the estimate under-shot; open the bracket
    if _grid_probe(sil, lo)[6].sum() < n:              # even the finest grid cannot hold them all
        hi = lo
    else:
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            if _grid_probe(sil, mid)[6].sum() >= n:
                lo = mid
            else:
                hi = mid
    n_rows, n_cols, cw, ch, cx, cy, inside = _grid_probe(sil, lo)
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            if inside[r, c]:
                cells.append({"cx": float(cx[c]), "cy": float(cy[r]), "w": float(cw), "h": float(ch),
                              "row": r, "col": c})
    return cells


# -- layout: mosaic (quadtree) -----------------------------------------------------
# How much finer than the "typical" tile (the size the leaf count implies) the boundary may go.
# The floor has to be RELATIVE, not a pixel count: with an absolute floor a 10000 px canvas lets the
# edge subdivide two levels deeper than a 2000 px one, the whole tile budget drains into edge crumbs,
# and the same settings produce a different mosaic at a different output size.
_MOSAIC_EDGE_RATIO = 4.0


def _layout_mosaic(sil, n: int, min_cell_px: float):
    """Quadtree: subdivide boundary cells first (crisp outline), then the largest cells (capacity).

    Cells that straddle the silhouette edge are split before interior ones, so the collage's
    outline sharpens as fast as the tile budget allows; interior cells stay large, which is what
    gives the mosaic its size variety.
    """
    import heapq

    import cv2
    import numpy as np

    H, W = sil.shape
    integral = cv2.integral(sil.astype(np.uint8))       # (H+1, W+1) int32 prefix sums

    def coverage(x, y, s):
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(math.ceil(x + s))), min(H, int(math.ceil(y + s)))
        if x1 <= x0 or y1 <= y0:
            return 0, 0
        total = (int(integral[y1, x1]) - int(integral[y0, x1])
                 - int(integral[y1, x0]) + int(integral[y0, x0]))
        return total, (x1 - x0) * (y1 - y0)

    def inside(x, y, s):
        cx, cy = int(x + s / 2.0), int(y + s / 2.0)
        return 0 <= cx < W and 0 <= cy < H and bool(sil[cy, cx])

    # Floor the tile size against what the leaf count implies, so a bigger canvas cannot drain the
    # whole tile budget into edge crumbs; the user's pixel floor only ever makes it coarser.
    floor = max(float(min_cell_px), math.sqrt(float(sil.sum()) / max(1, n)) / _MOSAIC_EDGE_RATIO)
    s0 = max(W, H) / 3.0
    heap: list = []
    counter = 0
    kept = 0
    cells: dict[int, tuple] = {}                       # id -> (x, y, s)
    for j in range(int(math.ceil(H / s0))):
        for i in range(int(math.ceil(W / s0))):
            x, y = i * s0, j * s0
            fg, area = coverage(x, y, s0)
            if area == 0 or fg == 0:
                continue
            cells[counter] = (x, y, s0)
            kept += 1 if inside(x, y, s0) else 0
            heapq.heappush(heap, (0 if fg < area else 1, -s0, counter))
            counter += 1

    while kept < n and heap:
        _, _neg_s, cid = heapq.heappop(heap)
        if cid not in cells:
            continue
        x, y, s = cells[cid]
        half = s / 2.0
        if half < floor:
            continue                                   # too small to split -- but it STAYS a tile
        del cells[cid]
        kept -= 1 if inside(x, y, s) else 0
        for dx, dy in ((0.0, 0.0), (half, 0.0), (0.0, half), (half, half)):
            nx, ny = x + dx, y + dy
            fg, area = coverage(nx, ny, half)
            if area == 0 or fg == 0:
                continue
            cells[counter] = (nx, ny, half)
            kept += 1 if inside(nx, ny, half) else 0
            heapq.heappush(heap, (0 if fg < area else 1, -half, counter))
            counter += 1

    # The seed grid deliberately OVER-covers a non-square canvas, so the last row/column of cells
    # hangs off the edge. Emitting those unclamped would center a tile outside the canvas and the
    # compositor would slice the leaf off with a straight line -- clamp to the visible rectangle.
    out = []
    for x, y, s in cells.values():
        if not inside(x, y, s):
            continue
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(W), x + s), min(float(H), y + s)
        if x1 <= x0 or y1 <= y0:
            continue
        out.append({"cx": 0.5 * (x0 + x1), "cy": 0.5 * (y0 + y1), "w": x1 - x0, "h": y1 - y0})
    return out


# -- layout: organic packing -------------------------------------------------------
def _layout_organic(sil, proxies, *, fill: float, gap_px: float, min_tile_px: float,
                    max_tile_px: float, rotate: bool, rng, on_step=None):
    """Drop each leaf into the largest empty pocket left inside the silhouette.

    The distance transform of the still-free area gives, at its maximum, the center and radius of
    the biggest inscribed circle; the leaf is scaled to that circle and its ACTUAL footprint (not
    its bounding box) is subtracted from the free area, so later leaves nest into the real gaps.

    ``max_tile_px`` is what keeps this from degenerating: the FIRST pocket is the whole silhouette,
    so an uncapped packer hands leaf #1 an inscribed circle spanning most of the canvas and the
    collage becomes two enormous leaves ringed by crumbs. Capping the tile relative to the size the
    leaf count implies keeps the scale varied but bounded.
    """
    import cv2
    import numpy as np

    free = sil.astype(np.uint8).copy()
    H, W = free.shape
    placements: list[dict] = []
    for i, proxy in enumerate(proxies):
        dt = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        r = float(dt.max())
        side = 2.0 * r * float(fill)
        if side < float(min_tile_px):
            break
        side = min(side, float(max_tile_px))
        # Among the near-best pockets, pick one at random: a pure argmax always resolves ties the
        # same way (top-left first), which reads as a diagonal drift across the collage.
        ys, xs = np.where(dt >= r * 0.995)
        k = int(rng.integers(len(xs))) if len(xs) > 1 else 0
        cx, cy = float(xs[k]), float(ys[k])
        angle = float(rng.uniform(0.0, 360.0)) if rotate else 0.0

        stamp = _fit_rotate(proxy, side, side, angle, 1.0)
        if stamp is None:
            continue
        sh, sw = stamp.shape[:2]
        x0, y0 = int(round(cx - sw / 2.0)), int(round(cy - sh / 2.0))
        sx0, sy0 = max(0, x0), max(0, y0)
        sx1, sy1 = min(W, x0 + sw), min(H, y0 + sh)
        if sx1 > sx0 and sy1 > sy0:
            sub = stamp[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0] > 127
            if gap_px and float(gap_px) > 0:
                ksz = max(1, int(round(float(gap_px))) * 2 + 1)
                sub = cv2.dilate(sub.astype(np.uint8), np.ones((ksz, ksz), np.uint8)).astype(bool)
            free[sy0:sy1, sx0:sx1][sub] = 0
        placements.append({"cx": cx, "cy": cy, "w": side, "h": side, "angle": angle, "index": i})
        if on_step is not None and (i % 25 == 0):
            on_step(len(placements))
    return placements


# -- layout: puzzle ----------------------------------------------------------------
# True irregular-shape nesting. Two identities carry the whole mode:
#
#   feasibility is an EROSION -- cv2.erode(free, K, anchor=(0,0), BORDER_CONSTANT, 0) is 1 at
#     exactly those top-left offsets where every set pixel of the stamp K lands on free space.
#     borderValue=0 is MANDATORY: cv2's erode border defaults to +inf, which reports positions
#     hanging off the array as feasible.
#   contact is a CORRELATION -- filter2D(blocked, ring) with ring = dilate(K) - K counts how much
#     of the stamp's outline touches already-blocked pixels.
#
# Maximizing contact is the exact inverse of the organic packer, which maximizes clearance. That
# sign flip is what makes leaves nest into each other's concavities instead of sitting in pockets.
#
# The search is coarse-to-fine: a min-pooled free map finds candidate spots cheaply (pooling is
# conservative -- coarse-feasible implies fine-feasible), then the best few are re-scored exactly
# inside a small ROI. Leaves are placed largest-area first; ones that no longer fit are re-queued
# at ``backfill_ratio`` of their size so the leftover gaps close.


def _white_feret(mask) -> float:
    """The Feret (max caliper) diameter of a mask's white region, in px.

    The only reading of "the longest dimension of the shape" that survives rotation: rotating a
    point set changes no pairwise distance, so the Feret diameter is rotation-invariant, and
    scaling by s multiplies it by s. The axis-aligned bounding box and cv2.minAreaRect are both
    rotation-dependent -- minAreaRect's long side measured as little as 0.79 of the Feret on real
    leaves, which would undersize those leaves by a fifth.
    """
    import cv2
    import numpy as np

    cnts, _ = cv2.findContours((np.asarray(mask) > 127).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    hull = cv2.convexHull(np.vstack(cnts).reshape(-1, 2).astype(np.float32)).reshape(-1, 2)
    return float(np.sqrt(((hull[:, None] - hull[None]) ** 2).sum(-1)).max()) + 1.0


def _leaf_piece(mask, rgb, long_px: float, angle: float, feret: Optional[float] = None):
    """Scale a leaf so its white region's Feret diameter is ``long_px``, THEN rotate it.

    Returns ``(alpha, bgr)`` cropped tight to the rotated white region (``bgr`` is None when
    ``rgb`` is). This is why :func:`_fit_rotate` cannot be reused for the puzzle layout: that
    helper rotates first and then fits the result into a BOX, so the drawn long side ends up
    equal to the box rather than to the leaf -- measured across real leaves and angles, its
    achieved Feret spans a 1.6x range, while this helper holds it to within 1.5%.
    """
    import cv2
    import numpy as np

    mask = np.asarray(mask)
    f = float(feret) if feret else _white_feret(mask)
    if f <= 1.0 or long_px <= 0:
        return None, None
    s = float(long_px) / f
    h, w = mask.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle), s)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    m[0, 2] += nw / 2.0 - w / 2.0
    m[1, 2] += nh / 2.0 - h / 2.0
    stack = mask if rgb is None else np.dstack([np.asarray(rgb), mask])
    out = cv2.warpAffine(stack, m, (max(1, nw), max(1, nh)),
                         flags=cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR, borderValue=0)
    alpha = out if out.ndim == 2 else out[:, :, 3]
    ys, xs = np.nonzero(alpha > 127)
    if not len(ys):
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    a = np.ascontiguousarray(alpha[y0:y1, x0:x1])
    b = None if out.ndim == 2 else np.ascontiguousarray(out[y0:y1, x0:x1, :3])
    return a, b


def _minpool(a, cf: int):
    """Downsample by taking the MIN of each cf x cf block -- a cell is free only if all of it is."""
    h, w = a.shape
    h2, w2 = h // cf, w // cf
    return a[:h2 * cf, :w2 * cf].reshape(h2, cf, w2, cf).min(axis=(1, 3))


def _maxpool(a, cf: int):
    """Downsample by taking the MAX of each cf x cf block -- a stamp cell is set if any of it is.

    Min-pooling the free space and max-pooling the stamp together make the coarse test
    CONSERVATIVE: anything the coarse pass calls feasible really is feasible at full resolution,
    so the fine pass never has to undo a coarse decision.
    """
    import numpy as np

    h, w = a.shape
    h2, w2 = (h + cf - 1) // cf, (w + cf - 1) // cf
    p = np.zeros((h2 * cf, w2 * cf), a.dtype)
    p[:h, :w] = a
    return p.reshape(h2, cf, w2, cf).max(axis=(1, 3))


def _puzzle_leaf_px(domain_area: float, ratios, fill: float) -> float:
    """The leaf size that inks ``fill`` of the domain, in closed form.

    A leaf whose white area is ``a`` and Feret ``F`` covers exactly ``long^2 * (a / F^2)`` when
    scaled to Feret ``long``. Summing that over the selection and setting it equal to
    ``fill * domain_area`` solves for ``long`` directly -- no packing iterations, and no sampling
    error from estimating a mean shape constant.
    """
    import math

    tot = float(sum(ratios))
    if tot <= 0.0:
        return 0.0
    return math.sqrt(max(0.0, float(domain_area) * float(fill)) / tot)


def _layout_puzzle(domain, pieces, *, leaf_px: float, gap_px: float, angles: int, coarse: int,
                   refine: int, backfill_ratio: float, min_leaf_px: float, max_shrink: int,
                   rng, on_step=None):
    """Nest real leaf silhouettes into ``domain``. Returns cells in the domain's pixel space.

    ``pieces[i]`` is ``(mask, feret, ratio)`` -- a small binary stand-in, its Feret diameter, and
    its white-area / Feret^2 (the scale-free shape constant). Cells carry ``long_px`` so the
    renderer can reproduce the exact stamp geometry the nester collided with.
    """
    import cv2
    import numpy as np

    cf = max(1, int(coarse))
    g = max(2, int(round(float(gap_px))))
    dom = (np.asarray(domain) > 0).astype(np.uint8)      # never alias or mutate the caller's array
    h0, w0 = dom.shape
    # Clamp the leaf to something the shape can actually hold. Without this a huge puzzle_leaf_px
    # makes the padded raster O(leaf_px^2) and the coarse erode O(leaf_px^4 / cf^4) -- minutes of
    # work for a value a GUI slider can reach.
    cap = 2.0 * float(cv2.distanceTransform(dom, cv2.DIST_L2, 5).max())
    leaf_px = max(1.0, min(float(leaf_px), max(cap, 2.0) * 4.0))
    pad = int(min(leaf_px, max(h0, w0))) + g + 8

    free = np.zeros((h0 + 2 * pad, w0 + 2 * pad), np.uint8)
    free[pad:pad + h0, pad:pad + w0] = dom
    blocked = (1 - free).astype(np.uint8)
    free_c = _minpool(free, cf)
    blk_c = (1 - free_c).astype(np.uint8)

    box = max(3, int(round(leaf_px / cf)))
    def _crowd_all():
        # anchor=(0,0) matches the erode anchor: both maps are indexed by the stamp's TOP-LEFT.
        # A centered anchor here would bias every pick by half a leaf toward the blockers.
        return cv2.boxFilter(blk_c, cv2.CV_32F, (box, box), anchor=(0, 0), normalize=False,
                             borderType=cv2.BORDER_CONSTANT)

    crowd = _crowd_all()

    def _crowd_patch(y0, y1, x0, x1):
        r = box // 2 + 1
        yy0, yy1 = max(0, y0 - r), min(blk_c.shape[0], y1 + r)
        xx0, xx1 = max(0, x0 - r), min(blk_c.shape[1], x1 + r)
        py0, py1 = max(0, yy0 - box), min(blk_c.shape[0], yy1 + box)
        px0, px1 = max(0, xx0 - box), min(blk_c.shape[1], xx1 + box)
        sub = cv2.boxFilter(blk_c[py0:py1, px0:px1], cv2.CV_32F, (box, box), anchor=(0, 0),
                            normalize=False, borderType=cv2.BORDER_CONSTANT)
        crowd[yy0:yy1, xx0:xx1] = sub[yy0 - py0:yy1 - py0, xx0 - px0:xx1 - px0]

    # Every leaf gets its own random starting orientation, drawn once and reused if it is
    # re-queued by the backfill. Without it the candidate angles are the same fixed set for every
    # leaf -- 0, 360/A, 2*360/A ... -- and since the Leaf_Oriented tree hands them all over
    # standing tip-up, whole neighbourhoods come out pointing the same way.
    base = rng.random(len(pieces)) * 360.0

    # Largest first. Nesting is order-sensitive: big pieces placed late have nowhere to go, and
    # the leftovers are exactly what the backfill queue is for.
    order = sorted(range(len(pieces)),
                   key=lambda i: -(pieces[i][2] if pieces[i] is not None else 0.0))
    queue = [(i, float(leaf_px), 0) for i in order]

    ring_ker = np.ones((2 * g + 3, 2 * g + 3), np.uint8)
    dil_ker = np.ones((2 * g + 1, 2 * g + 1), np.uint8)
    cells: list[dict] = []
    qi = 0
    while qi < len(queue):
        i, side, level = queue[qi]
        qi += 1
        pc = pieces[i]
        if pc is None:
            continue
        mask, feret, _ = pc

        def _requeue():
            if backfill_ratio > 0.0 and level + 1 <= int(max_shrink) \
                    and side * backfill_ratio >= float(min_leaf_px):
                queue.append((i, side * backfill_ratio, level + 1))

        stamps = []
        na = max(1, int(angles))
        for j in range(na):
            ang = (base[i] + 360.0 * j / na) % 360.0
            k, _ = _leaf_piece(mask, None, side, ang, feret=feret)
            if k is None:
                continue
            k = (k > 127).astype(np.uint8)
            # An empty kernel is a no-op that would make EVERY position feasible -- a phantom
            # placement, not an exception. Drop it rather than trust it.
            if k.sum() == 0 or k.shape[0] >= free_c.shape[0] * cf or k.shape[1] >= free_c.shape[1] * cf:
                continue
            stamps.append((ang, k))

        cands = []
        for ang, k in stamps:
            kc = _maxpool(k, cf)
            if kc.shape[0] >= free_c.shape[0] or kc.shape[1] >= free_c.shape[1]:
                continue
            feas_c = cv2.erode(free_c, kc, anchor=(0, 0),
                               borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if not feas_c.any():
                continue
            _, sc, _, loc = cv2.minMaxLoc(crowd, feas_c)
            cands.append((sc, loc, ang, k))
        if not cands:
            _requeue()
            continue

        cands.sort(key=lambda c: -c[0])
        best = None
        for _sc, loc, ang, k in cands[:max(1, int(refine))]:
            kh, kw = k.shape
            win = cf + g + 2
            fx, fy = loc[0] * cf, loc[1] * cf
            x0, y0 = max(0, fx - win), max(0, fy - win)
            x1 = min(free.shape[1], fx + kw + win)
            y1 = min(free.shape[0], fy + kh + win)
            sub_free = np.ascontiguousarray(free[y0:y1, x0:x1])
            if sub_free.shape[0] < kh or sub_free.shape[1] < kw:
                continue
            feas = cv2.erode(sub_free, k, anchor=(0, 0),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if not feas.any():
                continue
            ring = (cv2.dilate(k, ring_ker) - k).astype(np.float32)
            con = cv2.filter2D(np.ascontiguousarray(blocked[y0:y1, x0:x1]), cv2.CV_32F, ring,
                               anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
            _, s2, _, l2 = cv2.minMaxLoc(con, feas)
            if best is None or s2 > best[0]:
                best = (s2, x0 + l2[0], y0 + l2[1], k, ang)
        if best is None:
            _requeue()
            continue

        _, x, y, k, ang = best
        kh, kw = k.shape
        blocked[y:y + kh, x:x + kw] |= k
        free[y:y + kh, x:x + kw][cv2.dilate(k, dil_ker) > 0] = 0
        cy0, cy1 = y // cf, min((y + kh) // cf + 1, free_c.shape[0])
        cx0, cx1 = x // cf, min((x + kw) // cf + 1, free_c.shape[1])
        if cy1 > cy0 and cx1 > cx0:
            free_c[cy0:cy1, cx0:cx1] = _minpool(free[cy0 * cf:cy1 * cf, cx0 * cf:cx1 * cf], cf)
            blk_c[cy0:cy1, cx0:cx1] = 1 - free_c[cy0:cy1, cx0:cx1]
            _crowd_patch(cy0, cy1, cx0, cx1)
        cells.append({"cx": x + kw / 2.0 - pad, "cy": y + kh / 2.0 - pad,
                      "w": float(kw), "h": float(kh), "angle": ang, "index": i,
                      "long_px": float(side), "shrink": int(level), "ink": int(k.sum())})
        if on_step is not None and (len(cells) % 25 == 0):
            on_step(len(cells))
    return cells


# -- image helpers -----------------------------------------------------------------
def _fit_rotate(img, box_w: float, box_h: float, angle: float, scale: float):
    """Rotate ``img`` about its center (canvas expanded to fit), then fit it inside the box.

    Rotation happens BEFORE fitting so the rotated leaf still lands inside its cell -- fitting
    first and rotating after would push the corners out by up to sqrt(2).
    """
    import cv2
    import numpy as np

    if img is None or img.size == 0:
        return None
    out = img
    if angle:
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
        m[0, 2] += nw / 2.0 - w / 2.0
        m[1, 2] += nh / 2.0 - h / 2.0
        out = cv2.warpAffine(out, m, (max(1, nw), max(1, nh)), flags=cv2.INTER_LINEAR,
                             borderValue=0)
    h, w = out.shape[:2]
    f = min(float(box_w) / w, float(box_h) / h) * float(scale)
    tw, th = max(1, int(round(w * f))), max(1, int(round(h * f)))
    interp = cv2.INTER_AREA if f < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(out, (tw, th), interpolation=interp)


def _read_mask(path):
    """A leaf mask PNG as a single-channel uint8 alpha (0..255)."""
    import cv2

    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 3:
        im = im[:, :, 3] if im.shape[2] == 4 else cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    return im


def _composite(canvas, tile_bgr, alpha, cx: float, cy: float) -> bool:
    """Alpha-over one tile centered on ``(cx, cy)``. Returns False when it lands off-canvas.

    STRAIGHT (unassociated) alpha, which is what PNG stores. The naive ``src*a + dst*(1-a)`` is the
    "over" formula for an OPAQUE destination; run against a transparent canvas it leaves
    PREMULTIPLIED color -- every anti-aliased edge pixel darkened toward black in proportion to its
    own transparency, which reads as a gray hairline around every leaf once the PNG is placed on a
    light background. Coverage likewise has to ACCUMULATE (``a + da*(1-a)``); ``max(a, da)`` would
    leave see-through seams where two soft edges meet.

    Only the clipped sub-rectangle is touched, which is what keeps a 10000 px canvas affordable.
    """
    import numpy as np

    H, W = canvas.shape[:2]
    th, tw = alpha.shape[:2]
    x0, y0 = int(round(cx - tw / 2.0)), int(round(cy - th / 2.0))
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + tw), min(H, y0 + th)
    if sx1 <= sx0 or sy1 <= sy0:
        return False
    a = alpha[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0].astype(np.float32) / 255.0
    src = tile_bgr[sy0 - y0: sy1 - y0, sx0 - x0: sx1 - x0].astype(np.float32)
    dst = canvas[sy0:sy1, sx0:sx1]
    da = dst[:, :, 3].astype(np.float32) / 255.0
    keep = da * (1.0 - a)                              # what the destination still contributes
    out_a = a + keep
    inv = np.where(out_a > 1e-6, 1.0 / np.maximum(out_a, 1e-6), 0.0)
    for c in range(3):
        blended = (src[:, :, c] * a + dst[:, :, c].astype(np.float32) * keep) * inv
        dst[:, :, c] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    dst[:, :, 3] = np.clip(out_a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return True


def _rgb_of(color) -> tuple[int, int, int]:
    """``"white"`` / ``"#rrggbb"`` / ``[R, G, B]`` / ``"R,G,B"`` -> an RGB triple.

    All four spellings are advertised in ``postprocessing_settings.yaml``, but the STL builder's
    palette knows only the two NAMES: the hex and comma forms are folded to a list by the server
    before they reach a tool, so on the CLI path they have to be parsed here.
    """
    if isinstance(color, str):
        s = color.strip()
        if s.startswith("#"):
            h = s[1:]
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            try:
                if len(h) not in (6, 8):
                    raise ValueError
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            except ValueError:
                raise ValueError(f"{color!r} is not a #rrggbb color") from None
        parts = [p for p in re.split(r"[,\s]+", s) if p]
        if len(parts) in (3, 4):
            try:
                rgb = [int(round(float(p))) for p in parts[:3]]
            except ValueError:
                raise ValueError(f"{color!r} is not a color name or R,G,B triple") from None
            return (max(0, min(255, rgb[0])), max(0, min(255, rgb[1])), max(0, min(255, rgb[2])))

    from leafmachine3.postprocessing.generate_stl_from_mask import _parse_colors

    r, g, b = _parse_colors(color)[0]
    return (int(r), int(g), int(b))


def _bgr_of(color) -> tuple[int, int, int]:
    """The same, as an OpenCV BGR triple."""
    r, g, b = _rgb_of(color)
    return (b, g, r)


def _is_transparent(background) -> bool:
    if background is None:
        return True                                    # an unset background means "no background"
    return isinstance(background, str) and background.strip().lower() in ("transparent", "none", "")


# -- the collage -------------------------------------------------------------------
def generate_collage(
    run_dir,
    primary_mask,
    out_path,
    *,
    min_archetype_score: float = 0.8,
    max_leaves: int = 0,
    tree: str = "Leaf_Oriented",
    mask_variant: str = "lamina_mask",
    style: str = "mask",
    layout: str = "grid",
    ranking: str = "center",
    max_dim_px: int = 10000,
    color=(255, 255, 255),
    background="transparent",
    primary_colors=("white",),
    primary_color_tolerance: int = 0,
    primary_fill_holes: bool = True,
    tile_scale: float = 0.98,
    random_seed: int = 0,
    shuffle_top: int = 0,
    leaf_order: str = "score",
    mosaic_min_cell_px: float = 24.0,
    organic_rotate: bool = True,
    organic_fill: float = 0.9,
    organic_gap_px: float = 2.0,
    organic_min_tile_px: float = 12.0,
    organic_max_scale: float = 3.0,
    puzzle_leaf_px: float = 0.0,
    puzzle_fill: float = 0.6,
    puzzle_gap_px: float = 2.0,
    puzzle_angles: int = 8,
    puzzle_coarse: int = 6,
    puzzle_refine: int = 3,
    puzzle_nest_px: int = 3072,
    puzzle_overhang: float = 0.0,
    puzzle_backfill_ratio: float = 0.72,
    puzzle_min_leaf_px: float = 12.0,
    puzzle_max_shrink: int = 3,
    layout_px: int = 2048,
    tmp_dir=None,
    write_manifest: bool = True,
    on_progress: Optional[Callable[[float, str, int, int], None]] = None,
) -> dict:
    """Build one collage PNG. Returns a summary dict of JSON-safe values."""
    import cv2
    import numpy as np

    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {list(_LAYOUTS)}, got {layout!r}")
    if ranking not in _RANKINGS:
        raise ValueError(f"ranking must be one of {list(_RANKINGS)}, got {ranking!r}")
    if leaf_order not in _LEAF_ORDERS:
        raise ValueError(f"leaf_order must be one of {list(_LEAF_ORDERS)}, got {leaf_order!r}")
    if int(max_dim_px) < 64:
        raise ValueError(f"max_dim_px must be at least 64, got {max_dim_px}")
    # An emptied form field arrives as None. Fall back to the documented default rather than
    # failing -- and do the color parsing NOW, so a bad color cannot abort after the PNG is written.
    if color is None:
        color = _DEFAULTS["color"]
    if background is None:
        background = _DEFAULTS["background"]
    fg = _bgr_of(color)
    bg_bgr = None if _is_transparent(background) else _bgr_of(background)

    def tick(frac, msg, done=0, total=0):
        if on_progress is not None:
            on_progress(float(frac), str(msg), int(done), int(total))

    tick(0.02, "reading the project database")
    leaves, stats = select_leaves(
        run_dir, min_archetype_score=min_archetype_score, max_leaves=max_leaves,
        tree=tree, mask_variant=mask_variant, style=style,
    )
    rng = np.random.default_rng(int(random_seed))      # always seeded: same seed -> same collage
    if str(leaf_order) == "random":
        # The position of a leaf in this list IS its priority in every layout -- grid and mosaic
        # hand the front of it the most central cells, organic the biggest pockets, puzzle the
        # first pick of the free space. So any structure in the incoming order becomes structure
        # in the picture: score order puts the best leaves in the middle, and a caller that
        # assembles the list class by class gets visible concentric bands of margin type. A full
        # permutation removes both. Seeded, so the same seed still reproduces the same collage.
        order = rng.permutation(len(leaves))
        leaves = [leaves[i] for i in order]
    elif int(shuffle_top) > 1:
        # Shuffling the top of the ranking varies WHICH archetype gets the visual emphasis without
        # letting a mediocre leaf into a prime tile -- the same seed reproduces the same collage.
        head = leaves[: int(shuffle_top)]
        order = rng.permutation(len(head))
        leaves = [head[i] for i in order] + leaves[int(shuffle_top):]
    n = len(leaves)
    log.info("%d leaf/leaves pass (score > %s, gates_pass = 1) out of %d scored leaf row(s)",
             stats["n_passing"], min_archetype_score, stats["n_scored"])

    tick(0.06, "loading the primary mask")
    sil = _load_silhouette(primary_mask, list(primary_colors), int(primary_color_tolerance),
                           bool(primary_fill_holes), int(max_dim_px))
    H, W = sil.shape
    small, up = _layout_view(sil, int(layout_px))

    tick(0.10, f"solving the {layout} layout for {n} leaves", 0, n)
    proxies = None
    if layout == "grid":
        cells = _layout_grid(small, n)
    elif layout == "mosaic":
        cells = _layout_mosaic(small, n, float(mosaic_min_cell_px) / max(up, 1e-6))
    elif layout == "puzzle":
        # The puzzle nests at its OWN resolution -- the packing is only as fine as the raster it
        # runs on, and layout_px is tuned for the other three layouts. Rebinding small/up here
        # keeps every downstream cell -> canvas conversion consistent.
        small, up = _layout_view(sil, int(puzzle_nest_px))
        domain = (small > 0).astype(np.uint8)
        over = int(round(float(puzzle_overhang) * max(1.0, float(puzzle_leaf_px) or 1.0)))
        pieces = []
        for lf in leaves:
            m = _read_mask(lf["mask_path"])
            if m is None:
                pieces.append(None)
                continue
            f = _white_feret(m)
            area = float((m > 127).sum())
            if f <= 1.0 or area <= 0.0:
                pieces.append(None)
                continue
            ratio = area / (f * f)                     # scale-free: area of the leaf at Feret 1
            k = min(1.0, 256.0 / max(m.shape[:2]))     # the nester only needs the SHAPE, not detail
            if k < 1.0:
                m = cv2.resize(m, (max(1, int(m.shape[1] * k)), max(1, int(m.shape[0] * k))),
                               interpolation=cv2.INTER_AREA)
                f = _white_feret(m)
            pieces.append((m, f, ratio))
        leaf_px = float(puzzle_leaf_px) / max(up, 1e-6)
        if leaf_px <= 0:
            leaf_px = _puzzle_leaf_px(float(domain.sum()),
                                      [p[2] for p in pieces if p is not None], float(puzzle_fill))
        if over > 0:                                   # overhang widens the DOMAIN, so density holds
            domain = cv2.dilate(domain, np.ones((2 * over + 1, 2 * over + 1), np.uint8))
        cells = _layout_puzzle(
            domain, pieces, leaf_px=leaf_px,
            gap_px=float(puzzle_gap_px) / max(up, 1e-6), angles=int(puzzle_angles),
            coarse=int(puzzle_coarse), refine=int(puzzle_refine),
            backfill_ratio=float(puzzle_backfill_ratio),
            min_leaf_px=float(puzzle_min_leaf_px) / max(up, 1e-6),
            max_shrink=int(puzzle_max_shrink), rng=rng,
            on_step=lambda k: tick(0.10 + 0.30 * (k / max(1, n)), f"nesting {k} / {n}", k, n),
        )
    else:
        proxies = []
        for lf in leaves:                              # small stand-ins: packing only needs shape
            m = _read_mask(lf["mask_path"])
            proxies.append(None if m is None else _fit_rotate(m, 128, 128, 0.0, 1.0))
        # The cap is relative to the tile size the leaf count implies (the same sqrt(area / n) the
        # grid solves for), so it scales with the collage instead of being a fixed pixel number.
        typical = math.sqrt(float(small.sum()) / max(1, n))
        cells = _layout_organic(
            small, proxies, fill=float(organic_fill),
            gap_px=float(organic_gap_px) / max(up, 1e-6),
            min_tile_px=float(organic_min_tile_px) / max(up, 1e-6),
            max_tile_px=max(2.0, float(organic_max_scale) * typical),
            rotate=bool(organic_rotate), rng=rng,
            on_step=lambda k: tick(0.10 + 0.30 * (k / max(1, n)), f"packing {k} / {n}", k, n),
        )

    if not cells:
        raise ValueError(
            f"the {layout} layout produced no tiles inside {primary_mask} -- the silhouette may be "
            f"a sliver at max_dim_px={max_dim_px}, or every cell fell outside it"
        )

    # Order the tiles so the best leaves land where the chosen ranking says they should. The organic
    # packer already emits its cells in placement order (biggest pocket first), which IS the ranking.
    if layout not in ("organic", "puzzle"):
        depth = _depth_map(small)
        cy0, cx0 = float(small.shape[0]) / 2.0, float(small.shape[1]) / 2.0
        for c in cells:
            iy = min(small.shape[0] - 1, max(0, int(c["cy"])))
            ix = min(small.shape[1] - 1, max(0, int(c["cx"])))
            c["depth"] = float(depth[iy, ix])
            c["radial"] = math.hypot(c["cx"] - cx0, c["cy"] - cy0)
        if ranking == "center":
            # Deepest inside first; in the mosaic the tile AREA leads, so "the best leaf gets the
            # most space AND the most central spot" falls out of one sort rather than two.
            key = ((lambda c: (-c["w"] * c["h"], -c["depth"], c["radial"])) if layout == "mosaic"
                   else (lambda c: (-c["depth"], c["radial"])))
            cells.sort(key=key)
        elif ranking == "reading":
            cells.sort(key=lambda c: (c["cy"], c["cx"]))
        else:
            cells = [cells[i] for i in rng.permutation(len(cells))]

    tick(0.42, "rendering", 0, n)
    canvas = np.zeros((H, W, 4), np.uint8)
    if bg_bgr is not None:
        canvas[:, :, :3] = np.asarray(bg_bgr, np.uint8)
        canvas[:, :, 3] = 255

    placed: list[dict] = []
    n_slots = min(len(cells), n)
    if n_slots < n and layout == "puzzle":
        log.warning("the shape saturated at %d of %d leaves -- lower puzzle_fill, raise "
                    "max_dim_px, or raise puzzle_backfill_ratio to close the remaining gaps",
                    n_slots, n)
    elif n_slots < n:
        log.warning("the silhouette holds %d tile(s) but %d leaves qualify -- the %d "
                    "lowest-scoring leaves were dropped", n_slots, n, n - n_slots)
    for i in range(n_slots):
        cell = cells[i]
        leaf = leaves[cell["index"]] if "index" in cell else leaves[i]
        alpha = _read_mask(leaf["mask_path"])
        if alpha is None:
            log.warning("unreadable mask, skipped: %s", leaf["mask_path"])
            continue
        angle = float(cell.get("angle", 0.0))
        bw, bh = cell["w"] * up, cell["h"] * up
        if layout == "puzzle":
            # Scale-then-rotate on the Feret, the same rule the nester used, so the drawn leaf is
            # the stamp that was collided with. tile_scale is deliberately NOT applied: shrinking
            # a nested piece would open seams the nester did not plan for.
            rgb = None
            if style == "rgb":
                rgb = cv2.imread(str(leaf["rgb_path"]), cv2.IMREAD_COLOR)
                if rgb is None:
                    log.warning("unreadable cutout, skipped: %s", leaf["rgb_path"])
                    continue
                if rgb.shape[:2] != alpha.shape[:2]:
                    rgb = cv2.resize(rgb, (alpha.shape[1], alpha.shape[0]),
                                     interpolation=cv2.INTER_AREA)
            a, tile = _leaf_piece(alpha, rgb, float(cell["long_px"]) * up, angle)
            if a is None or a.size == 0:
                continue
            if style != "rgb":
                tile = np.empty((a.shape[0], a.shape[1], 3), np.uint8)
                tile[:, :] = np.asarray(fg, np.uint8)
            bw, bh = float(a.shape[1]), float(a.shape[0])
        else:
            a = _fit_rotate(alpha, bw, bh, angle, float(tile_scale))
            if a is None or a.size == 0:
                continue
            if style == "rgb":
                rgb = cv2.imread(str(leaf["rgb_path"]), cv2.IMREAD_COLOR)
                if rgb is None:
                    log.warning("unreadable cutout, skipped: %s", leaf["rgb_path"])
                    continue
                if rgb.shape[:2] != alpha.shape[:2]:
                    rgb = cv2.resize(rgb, (alpha.shape[1], alpha.shape[0]),
                                     interpolation=cv2.INTER_AREA)
                tile = _fit_rotate(rgb, bw, bh, angle, float(tile_scale))
                if tile is None or tile.shape[:2] != a.shape[:2]:
                    continue
            else:
                tile = np.empty((a.shape[0], a.shape[1], 3), np.uint8)
                tile[:, :] = np.asarray(fg, np.uint8)
        # Nudge a tile that overhangs back inside. Leaves are meant to spill past the SILHOUETTE,
        # never past the CANVAS -- the compositor would clip them with a straight edge, which is the
        # one way a leaf can end up visibly cut.
        px, py = cell["cx"] * up, cell["cy"] * up
        th, tw = a.shape[:2]
        if tw <= W:
            px = min(max(px, tw / 2.0), W - tw / 2.0)
        if th <= H:
            py = min(max(py, th / 2.0), H - th / 2.0)
        if _composite(canvas, tile, a, px, py):
            # w/h are the DRAWN tile, not the cell it was fitted into -- the manifest should say
            # what is actually on the canvas.
            placed.append({"leaf_id": leaf["leaf_id"], "stem": leaf["stem"], "score": leaf["score"],
                           "mask": leaf["mask_path"],
                           "cx": round(px, 1), "cy": round(py, 1),
                           "w": int(tw), "h": int(th),
                           "cell_w": round(bw, 1), "cell_h": round(bh, 1),
                           "angle": round(angle, 2),
                           **({"long_px": round(float(cell["long_px"]) * up, 1),
                               "shrink": int(cell.get("shrink", 0))} if layout == "puzzle" else {})})
        if (i % 25 == 0) or i == n_slots - 1:
            tick(0.42 + 0.50 * ((i + 1) / max(1, n_slots)), f"{i + 1} / {n_slots} leaves",
                 i + 1, n_slots)

    if not placed:
        raise ValueError("no leaf could be drawn -- every tile fell outside the canvas")

    tick(0.94, "writing the collage")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Encode to scratch and swap in, so reports/Collage never holds a half-written PNG. The scratch
    # dir sits beside the run by default; a read-only run directory falls back to the output folder,
    # which is by definition writable. os.replace is atomic but same-device only, so a scratch dir on
    # another filesystem falls back to a plain move.
    staging = _tmp_root(run_dir, tmp_dir)
    try:
        staging.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        log.info("cannot use %s for staging (%s); staging in the output folder instead", staging, exc)
        staging = out_path.parent
    tmp_png = staging / f".{out_path.stem}.partial.png"
    try:
        ok = cv2.imwrite(str(tmp_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    except Exception:  # noqa: BLE001 - cv2 raises rather than returning False on some failures
        ok = False
    if not ok and staging != out_path.parent:
        staging = out_path.parent
        tmp_png = staging / f".{out_path.stem}.partial.png"
        ok = cv2.imwrite(str(tmp_png), canvas, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError(f"cv2 could not encode the collage to {tmp_png}")
    try:
        os.replace(tmp_png, out_path)
    except OSError:
        import shutil

        shutil.move(str(tmp_png), str(out_path))

    result = {
        "collage": str(out_path),
        "run_dir": str(Path(os.path.realpath(_db_path(run_dir))).parent),
        "primary_mask": os.path.realpath(primary_mask),
        "layout": layout, "ranking": ranking, "style": style, "tree": tree,
        "mask_variant": mask_variant,
        "size_px": [int(W), int(H)],
        "n_placed": len(placed), "n_cells": len(cells),
        "n_passing": stats["n_passing"], "n_scored_rows": stats["n_scored"],
        "n_merged_detections": stats["n_merged"], "n_missing_files": stats["n_missing_mask"] + stats["n_missing_rgb"],
        "min_archetype_score": float(min_archetype_score),
        "score_range": [round(min(p["score"] for p in placed), 4),
                        round(max(p["score"] for p in placed), 4)],
        "color": color if isinstance(color, str) else list(color),
        "background": background if isinstance(background, str) else list(background),
        "random_seed": int(random_seed), "leaf_order": str(leaf_order),
    }
    if layout == "puzzle":
        # The one number that says whether the nest worked, and the one that says whether the
        # size rule held. Both are cheap here and impossible to recover from the PNG alone.
        ink = float(sum(int(c.get("ink", 0)) for c in cells[:n_slots]))
        at_size = sum(1 for c in cells[:n_slots] if int(c.get("shrink", 0)) == 0)
        result["puzzle_leaf_px"] = round(float(cells[0]["long_px"]) * up, 1) if cells else 0.0
        result["puzzle_coverage"] = round(ink / max(1.0, float(small.sum())), 4)
        result["puzzle_at_full_size"] = [at_size, len(placed)]
    if write_manifest:
        import json

        manifest = out_path.with_suffix(".json")
        with manifest.open("w", encoding="utf-8") as fh:
            json.dump({**result, "leaves": placed}, fh, indent=2)
        result["manifest"] = str(manifest)
    tick(1.0, f"{len(placed)} leaves placed", len(placed), len(placed))
    return result


# -- driver + CLI ------------------------------------------------------------------
def _tmp_root(run_dir, tmp_dir) -> Path:
    """The scratch dir: ``<tmp_dir>/_leaf_collage``, else ``<run>/_leaf_collage``.

    Defaults beside the run's own ``_tmp_original`` rather than reading LM3_settings.yaml, which a
    postprocessing tool deliberately has no coupling to. Pass ``tmp_dir`` to point it elsewhere.
    """
    base = Path(tmp_dir) if tmp_dir else Path(_db_path(run_dir)).parent
    return base / "_leaf_collage"


def _out_path(run_dir, primary_mask, output_dir=None, name=None, layout="grid") -> Path:
    """``<output_dir or <run>/reports/Collage>/<name or collage__<layout>__<primary stem>>.png``."""
    folder = Path(output_dir) if output_dir else (Path(_db_path(run_dir)).parent / "reports" / "Collage")
    stem = str(name).strip() if name and str(name).strip() else \
        f"collage__{layout}__{Path(primary_mask).stem}"
    return folder / f"{Path(stem).name}.png"


def run(settings: Optional[dict] = None, run_dir=None, primary_mask=None, output_dir=None,
        on_progress: Optional[Callable[[float, str, int, int], None]] = None) -> list[dict]:
    """Run generate_leaf_collage once. ``run_dir``/``primary_mask``/``output_dir`` override settings."""
    s = {**_DEFAULTS, **(settings or {})}
    run_dir = run_dir if run_dir is not None else s.get("run_dir")
    primary_mask = primary_mask if primary_mask is not None else s.get("primary_mask")
    outdir = output_dir if output_dir is not None else s.get("output_dir")
    if not run_dir:
        raise ValueError("no run: set generate_leaf_collage.run_dir in the yaml or pass --run-dir")
    if not primary_mask:
        raise ValueError("no primary mask: set generate_leaf_collage.primary_mask or pass --primary-mask")
    if not Path(primary_mask).is_file():
        raise FileNotFoundError(f"cannot read primary mask image: {primary_mask}")

    out = _out_path(run_dir, primary_mask, outdir, s.get("name"), str(s["layout"]))
    return [generate_collage(
        run_dir, primary_mask, out,
        min_archetype_score=float(s["min_archetype_score"]), max_leaves=int(s["max_leaves"] or 0),
        tree=str(s["tree"]), mask_variant=str(s["mask_variant"]), style=str(s["style"]),
        layout=str(s["layout"]), ranking=str(s["ranking"]), max_dim_px=int(s["max_dim_px"]),
        color=s["color"], background=s["background"],
        primary_colors=s["primary_colors"], primary_color_tolerance=int(s["primary_color_tolerance"]),
        primary_fill_holes=bool(s["primary_fill_holes"]), tile_scale=float(s["tile_scale"]),
        random_seed=int(s["random_seed"] or 0), shuffle_top=int(s["shuffle_top"] or 0),
        leaf_order=str(s["leaf_order"]),
        mosaic_min_cell_px=float(s["mosaic_min_cell_px"]),
        organic_rotate=bool(s["organic_rotate"]), organic_fill=float(s["organic_fill"]),
        organic_gap_px=float(s["organic_gap_px"]), organic_min_tile_px=float(s["organic_min_tile_px"]),
        organic_max_scale=float(s["organic_max_scale"]),
        puzzle_leaf_px=float(s["puzzle_leaf_px"] or 0), puzzle_fill=float(s["puzzle_fill"]),
        puzzle_gap_px=float(s["puzzle_gap_px"]), puzzle_angles=int(s["puzzle_angles"]),
        puzzle_coarse=int(s["puzzle_coarse"]), puzzle_refine=int(s["puzzle_refine"]),
        puzzle_nest_px=int(s["puzzle_nest_px"]), puzzle_overhang=float(s["puzzle_overhang"]),
        puzzle_backfill_ratio=float(s["puzzle_backfill_ratio"]),
        puzzle_min_leaf_px=float(s["puzzle_min_leaf_px"]), puzzle_max_shrink=int(s["puzzle_max_shrink"]),
        layout_px=int(s["layout_px"]), tmp_dir=s.get("tmp_dir"),
        write_manifest=bool(s["write_manifest"]), on_progress=on_progress,
    )]


def _color_from_cli(tokens):
    """`--color white` -> 'white'; `--color 255 0 0` -> [255, 0, 0]; `--color transparent` -> the word."""
    try:
        return [int(t) for t in tokens]
    except ValueError:
        return tokens[0] if len(tokens) == 1 else list(tokens)


# -- section 2.8: the postprocessing concurrency guard -------------------------------
#: This tool's id in the postprocessing registry -- the key both the HTTP layer and this CLI
#: identify themselves by, so both reach the SAME ``Tool`` entry (its ``target_keys``, its
#: ``access="read_write"``) rather than two descriptions of one tool that can drift apart.
TOOL_ID = "generate_leaf_collage"

#: Exit code for a section 2.8 refusal. Deliberately NOT 75: section 2.3 reserves 75 for "the root
#: lease is held, retry later", and a supervisor that reads 75 may legitimately re-run -- which is
#: exactly the wrong response to "you aimed a writer at a live run". Not 2 either, which argparse
#: already owns for a usage error.
EXIT_TARGET_REFUSED = 3


def _refusal_types() -> tuple:
    """The exception types a section 2.8 refusal arrives as, or ``()`` when there is no server pkg.

    Evaluated lazily -- an ``except`` clause's expression only runs when an exception is actually
    propagating -- so the flag-off CLI never imports the server package at all. An empty tuple
    matches nothing, which is the right answer when there was no guard to refuse anything.

    ``ParamError`` is in the list because under the flag the guard resolves this CLI's targets
    through the same ``validate_params`` the HTTP route uses, and that is where a path outside
    :func:`~leafmachine3.server.postprocess_api.allowed_roots` is rejected. That rejection is a
    refusal of the same kind and deserves the same one-line message, not a traceback.
    """
    try:
        from leafmachine3.server.postprocess_api import ParamError, TargetActive, TargetLocked
    except Exception:                       # noqa: BLE001 - no server package, so no refusal
        return ()
    return (ParamError, TargetActive, TargetLocked)


@contextmanager
def cli_target_guard(run_dir=None, primary_mask=None, output_dir=None):
    """Hold section 2.8's guard around one CLI invocation; yields the resolved target run dirs.

    Section 2.8's last bullet is "Standalone CLI tools use the same guard as the HTTP API, not a
    parallel one", so this composes the SHIPPED functions in
    :mod:`leafmachine3.server.postprocess_api` and re-implements none of the policy:

    * ``validate_params`` -- which is what applies ``resolve_path``/``allowed_roots``, so the CLI's
      targets are resolved (and realpath'd) exactly the way a request's are. That matters for the
      comparison itself: ``active_run_target`` realpaths the record's root, and an unresolved
      target reached through a symlinked output root would never meet it.
    * ``check_target_allowed`` -- rule 1: refuse when the target IS the live run (or contains it,
      or is contained by it).
    * ``_acquire_artifact_locks`` -- rule 2: serialize two ``read_write`` tools on one completed
      run through the per-run advisory lock in the local deployment runtime registry.

    The lock WRAPS the run instead of being taken and dropped after argument parsing: bullet 5
    serializes two *writers*, and a lock released before ``run()`` starts serializes nothing.

    Inert unless ``LM3_RUNTIME_V2`` is on (``runtime_v2_enabled`` is the one reader of that flag),
    so today's shipped behavior is byte-identical -- including path handling, since the server's
    ``allowed_roots`` sandbox only begins to apply to this CLI under the flag.
    """
    from leafmachine3.core.runtime.execution import runtime_v2_enabled

    # Without the flag there is no guard and no import of the server package. Without BOTH a run
    # and a primary mask there is nothing to guard either: ``run()`` refuses on the missing one
    # before it reads or writes anything, and its message is the more actionable of the two.
    if not runtime_v2_enabled() or not run_dir or not primary_mask:
        yield []
        return

    from leafmachine3.server import postprocess_api as pp

    params = {"run_dir": str(run_dir), "primary_mask": str(primary_mask)}
    if output_dir:
        params["output_dir"] = str(output_dir)

    # If ``postprocess_api`` ever exports the single public wrapper this composition stands in for,
    # use it -- that is the shape section 2.8 asks for, and delegating keeps the two from drifting.
    guard = getattr(pp, "guard_cli", None)
    if guard is not None:
        with guard(TOOL_ID, params) as targets:
            yield targets
        return

    tool = pp.get_tool(TOOL_ID)
    clean = pp.validate_params(tool, params)
    targets = pp.check_target_allowed(tool, clean)
    locks = pp._acquire_artifact_locks(tool, targets)
    try:
        yield targets
    finally:
        # Every exit path, the crash and the KeyboardInterrupt included: a leaked advisory lock
        # would wedge this run directory against every later writer for the life of the process.
        for lock in locks:
            lock.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Arrange a run's best leaf masks into the shape of a primary mask.")
    ap.add_argument(
        "--config", default=None,
        help="postprocessing settings YAML (default: the deployment's canonical "
             "postprocessing.yaml -- plan section 3.1 row 3, never the working directory)")
    ap.add_argument("--run-dir", default=None, help="a finished LM3 run directory")
    ap.add_argument("--primary-mask", default=None, help="PNG whose foreground is the collage outline")
    ap.add_argument("--output-dir", default=None, help="output dir (default: <run>/reports/Collage)")
    ap.add_argument("--name", default=None, help="output file stem (without .png)")
    ap.add_argument("--min-archetype-score", type=float, default=None)
    ap.add_argument("--max-leaves", type=int, default=None, help="0 = every leaf that passes")
    ap.add_argument("--tree", choices=list(_TREES), default=None)
    ap.add_argument("--mask-variant", choices=list(_VARIANTS), default=None)
    ap.add_argument("--style", choices=list(_STYLES), default=None)
    ap.add_argument("--layout", choices=list(_LAYOUTS), default=None)
    ap.add_argument("--ranking", choices=list(_RANKINGS), default=None)
    ap.add_argument("--max-dim-px", type=int, default=None)
    ap.add_argument("--color", nargs="+", default=None, help="e.g. 'white', or '255 0 0'")
    ap.add_argument("--background", nargs="+", default=None, help="'transparent', or 'R G B'")
    ap.add_argument("--primary-colors", nargs="+", default=None)
    ap.add_argument("--primary-color-tolerance", type=int, default=None)
    ap.add_argument("--primary-fill-holes", dest="primary_fill_holes", action="store_true", default=None)
    ap.add_argument("--no-primary-fill-holes", dest="primary_fill_holes", action="store_false")
    ap.add_argument("--tile-scale", type=float, default=None)
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--shuffle-top", type=int, default=None)
    ap.add_argument("--leaf-order", choices=list(_LEAF_ORDERS), default=None,
                    help="score keeps the best leaves in the prime spots; random removes any "
                         "systematic order from the placement")
    ap.add_argument("--mosaic-min-cell-px", type=float, default=None)
    ap.add_argument("--organic-rotate", dest="organic_rotate", action="store_true", default=None)
    ap.add_argument("--no-organic-rotate", dest="organic_rotate", action="store_false")
    ap.add_argument("--organic-fill", type=float, default=None)
    ap.add_argument("--organic-gap-px", type=float, default=None)
    ap.add_argument("--organic-min-tile-px", type=float, default=None)
    ap.add_argument("--organic-max-scale", type=float, default=None)
    ap.add_argument("--puzzle-leaf-px", type=float, default=None,
                    help="white-region Feret diameter per leaf, in canvas px (0 = solve it)")
    ap.add_argument("--puzzle-fill", type=float, default=None,
                    help="target ink coverage of the shape; sets the leaf size when --puzzle-leaf-px is 0")
    ap.add_argument("--puzzle-gap-px", type=float, default=None)
    ap.add_argument("--puzzle-angles", type=int, default=None)
    ap.add_argument("--puzzle-coarse", type=int, default=None)
    ap.add_argument("--puzzle-refine", type=int, default=None)
    ap.add_argument("--puzzle-nest-px", type=int, default=None)
    ap.add_argument("--puzzle-overhang", type=float, default=None)
    ap.add_argument("--puzzle-backfill-ratio", type=float, default=None,
                    help="re-try a leaf that no longer fits at this fraction of its size (0 = off)")
    ap.add_argument("--puzzle-min-leaf-px", type=float, default=None)
    ap.add_argument("--puzzle-max-shrink", type=int, default=None)
    ap.add_argument("--layout-px", type=int, default=None)
    ap.add_argument("--tmp-dir", default=None)
    ap.add_argument("--no-manifest", dest="write_manifest", action="store_false", default=None)
    args = ap.parse_args(argv)

    from leafmachine3.postprocessing.config import load_settings, module_settings

    # A CLI main() is a controlled entry point -- once per process, user-initiated -- so the
    # one-release adopt of a checkout-level postprocessing_settings.yaml belongs here and not
    # in load_settings(), which must stay a pure read.
    if args.config is None:
        from leafmachine3.core.paths import PathsError, migrate_legacy_postprocessing_settings
        try:
            migrate_legacy_postprocessing_settings()
        except PathsError:
            pass                      # row 3 on-miss is "packaged defaults", never a crash

    s = module_settings(load_settings(args.config), "generate_leaf_collage")
    overrides = {
        "name": args.name, "min_archetype_score": args.min_archetype_score,
        "max_leaves": args.max_leaves, "tree": args.tree, "mask_variant": args.mask_variant,
        "style": args.style, "layout": args.layout, "ranking": args.ranking,
        "max_dim_px": args.max_dim_px, "primary_color_tolerance": args.primary_color_tolerance,
        "primary_fill_holes": args.primary_fill_holes, "tile_scale": args.tile_scale,
        "random_seed": args.random_seed, "shuffle_top": args.shuffle_top,
        "leaf_order": args.leaf_order,
        "mosaic_min_cell_px": args.mosaic_min_cell_px, "organic_rotate": args.organic_rotate,
        "organic_fill": args.organic_fill, "organic_gap_px": args.organic_gap_px,
        "organic_min_tile_px": args.organic_min_tile_px,
        "organic_max_scale": args.organic_max_scale,
        "puzzle_leaf_px": args.puzzle_leaf_px, "puzzle_fill": args.puzzle_fill,
        "puzzle_gap_px": args.puzzle_gap_px, "puzzle_angles": args.puzzle_angles,
        "puzzle_coarse": args.puzzle_coarse, "puzzle_refine": args.puzzle_refine,
        "puzzle_nest_px": args.puzzle_nest_px, "puzzle_overhang": args.puzzle_overhang,
        "puzzle_backfill_ratio": args.puzzle_backfill_ratio,
        "puzzle_min_leaf_px": args.puzzle_min_leaf_px, "puzzle_max_shrink": args.puzzle_max_shrink,
        "layout_px": args.layout_px,
        "tmp_dir": args.tmp_dir, "write_manifest": args.write_manifest,
    }
    for k, v in overrides.items():
        if v is not None:
            s[k] = v
    if args.color is not None:
        s["color"] = _color_from_cli(args.color)
    if args.background is not None:
        s["background"] = _color_from_cli(args.background)
    if args.primary_colors is not None:
        s["primary_colors"] = ([[int(t) for t in args.primary_colors]]
                               if all(t.lstrip("-").isdigit() for t in args.primary_colors)
                               else list(args.primary_colors))

    # The guard needs the EFFECTIVE targets -- what run() will actually use -- because the yaml
    # supplies them just as often as the flags do, and a guard that only saw --run-dir would wave
    # a settings-file run straight through.
    run_dir = args.run_dir if args.run_dir is not None else s.get("run_dir")
    primary_mask = args.primary_mask if args.primary_mask is not None else s.get("primary_mask")
    output_dir = args.output_dir if args.output_dir is not None else s.get("output_dir")
    try:
        with cli_target_guard(run_dir, primary_mask, output_dir):
            results = run(s, run_dir=args.run_dir, primary_mask=args.primary_mask,
                          output_dir=args.output_dir)
    except _refusal_types() as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_TARGET_REFUSED
    for r in results:
        print(f"  {Path(r['collage']).name}  {r['size_px'][0]}x{r['size_px'][1]}px  "
              f"leaves={r['n_placed']}/{r['n_passing']}  layout={r['layout']}  "
              f"score={r['score_range'][0]}-{r['score_range'][1]}")
        print(f"  -> {r['collage']}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
