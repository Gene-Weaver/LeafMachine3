"""Green Index stage -- several published color indices for every oriented leaf.

Runs AFTER the Reporter (and before ECT), because it CONSUMES the Reporter's oriented leaf products
rather than re-cutting leaves. Per leaf it reads the RGB cutout and its lossless mask, decides which
pixels count, runs every index in :data:`leafmachine3.core.green_index.METHODS`, and writes:

    leaf_green_index                                              one row per cutout
    reports/Data/green_index.csv                                  the same rows + parent-image identity
    reports/Overlay/Overlay_Green_Index/<stem>__or-GI-lamina__x_y_x_y.jpg   summary image

``write_overlay`` switches the images; the calculations and the CSV do not depend on them.

WHICH PIXELS. ``Lamina_Mask`` selects the leaf, not a color test on the cutout: JPEG ringing leaves
about a third of a cutout's "black" background not quite black, and near-black tissue is real. Mounting
tape sits inside the mask, so :mod:`leafmachine3.core.green_index.pixel_exclusion` looks for it
(``white_peak_check``). Excluded pixels are left out of every value, including the mean color.

WHAT A LEAF'S VALUE IS. The mean of that index's per-pixel values over the used pixels. Applying a
formula to the mean color instead differs by under 2% of the across-leaf range on real leaves; the
mean color is stored too, so that version can be recomputed.

ONE ROW PER CUTOUT. The Reporter writes one Lamina_RGB per detection box and only when orientation
succeeded, so this stage takes the lowest oriented instance of each box and nothing else.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from leafmachine3.core.green_index import METHODS
from leafmachine3.core.green_index.Excess_Green_Thrash_2022 import THRASH_WINDOW, excess_green
from leafmachine3.core.green_index.pixel_exclusion import STATUS_OFF, WhitePeakResult, find_white_cut, whiteness
from leafmachine3.core.imaging import crop_filename, save_image
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.reporting.data_export import export_stage_csv
from leafmachine3.reporting.green_index_viz import IndexTile, render_summary_panel
from leafmachine3.reporting.leaf_products import leaf_product_label

log = logging.getLogger("leafmachine3.green_index")

_LEAF_SEG = "Leaf"              # leaf_morphology.cls_name for a whole leaf instance
SOURCE_TREE = "Leaf_Oriented"
RGB_FOLDER, MASK_FOLDER = "Lamina_RGB", "Lamina_Mask"
OVERLAY_DIR = ("Overlay", "Overlay_Green_Index")
OVERLAY_JPG_QUALITY = 90        # a rendered figure, not data


@dataclass(frozen=True)
class LeafMeasure:
    """Everything worked out for one leaf: the numbers to store and what the summary image draws."""
    fields: dict
    used: np.ndarray
    tiles: tuple[IndexTile, ...]
    white_peak: WhitePeakResult


def measure_leaf(rgb: np.ndarray, lamina: np.ndarray, *, white_peak_check: bool = True) -> Optional[LeafMeasure]:
    """Measure one leaf from its RGB cutout and boolean lamina mask, or None when the mask is empty.

    Pure: no files, no database. ``rgb`` must be RGB order; a BGR array swaps red and blue, which
    changes the results without any error.
    """
    if not lamina.any():
        return None
    lamina_px = rgb[lamina]
    white_peak = find_white_cut(lamina_px) if white_peak_check else WhitePeakResult(STATUS_OFF)
    kept = np.ones(len(lamina_px), bool) if white_peak.cut is None else whiteness(lamina_px) <= white_peak.cut
    used = np.zeros(lamina.shape, bool)
    rows, cols = np.nonzero(lamina)
    used[rows[kept], cols[kept]] = True

    px = lamina_px[kept].astype(float)
    n_lamina, n_used = len(lamina_px), len(px)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    fields: dict[str, Any] = {
        "n_lamina_px": n_lamina,
        "n_excluded_white_peak_px": n_lamina - n_used,
        "n_used_px": n_used,
        "used_frac": n_used / n_lamina,
        "white_peak_status": white_peak.status,
        "white_peak_cut": white_peak.cut,
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
    }
    _, in_window, n_window, window_mean = excess_green(r, g, b)
    window_frac = n_window / n_used
    fields["excess_green_thrash_2022_window_mean"] = _finite(window_mean)
    fields["excess_green_thrash_2022_window_n_px"] = int(n_window)
    fields["excess_green_thrash_2022_window_frac"] = window_frac

    tiles = []
    for method in METHODS:
        values = np.asarray(method.calculate(r, g, b), dtype=float)
        final = _finite_mean(values)
        fields[method.column] = final
        note = ""
        if method.window is not None:
            note = f"{100 * window_frac:.1f}% of pixels in shaded Thrash {THRASH_WINDOW[0]}-{THRASH_WINDOW[1]} window"
        tiles.append(IndexTile(method, values, final, note))
    return LeafMeasure(fields, used, tuple(tiles), white_peak)


class GreenIndex(PipelineStage):
    """Compute + store the green indices of each oriented leaf (reads the Reporter's cutouts)."""

    key: str = "green_index"
    name: str = "Green Index"
    device_kind: str = "cpu"
    cpu_parallel: str = "process"   # matplotlib rendering is GIL-bound -> processes, like ECT
    fanout: bool = True             # one WorkItem PER LEAF -> the pool load-balances across workers
    est_item_seconds: float = 0.3   # the summary image; without it a leaf takes milliseconds
    depends_on: tuple[str, ...] = ("reporter",)   # consumes the Reporter's Leaf_Oriented products
    owns_tables: tuple[str, ...] = ("leaf_green_index",)

    def _settings(self) -> dict:
        c = self.cfg.stage(self.key)
        g = c.get if hasattr(c, "get") else (lambda k, d: d)
        return {
            "write_overlay": bool(g("write_overlay", True)),
            "white_peak_check": bool(g("white_peak_check", True)),
        }

    def collect_items(self, project) -> list[WorkItem]:
        """ONE WorkItem PER CUTOUT: the worker reads, measures and renders, returning a small row."""
        reports = Path(project.dirs.reports)
        rgb_ext = str(_get(self.cfg.report, "formats", "image_ext", default="jpg")).lstrip(".")
        mask_ext = str(_get(self.cfg.report, "formats", "mask_ext", default="png")).lstrip(".")
        # Must match the Reporter's leaf-product filenames exactly, tree tag included.
        rgb_label = leaf_product_label(SOURCE_TREE, "RGB", "lamina")
        mask_label = leaf_product_label(SOURCE_TREE, "SEG", "lamina")
        tree = reports / SOURCE_TREE

        items: list[WorkItem] = []
        for sid in project.db.specimens_with_rows("leaf_segmentation"):
            spec = project.db.get_specimen(sid)
            stem = str(spec["image_stem"] if spec is not None else sid)
            crop_boxes = project.db.detection_boxes(sid, "plant_detection")
            seen = set()
            morph = sorted(project.db.leaf_morphology(sid),          # lowest instance first per box
                           key=lambda m: (int(_g(m, "detection_id", -1)), int(_g(m, "instance_index", 0))))
            for m in morph:
                if str(_g(m, "cls_name", "")) != _LEAF_SEG or not int(_g(m, "oriented_leaf_success", 0) or 0):
                    continue
                did = int(_g(m, "detection_id", -1))
                box = crop_boxes.get(did)
                if not box or did in seen:
                    continue
                seen.add(did)
                det_box = tuple(int(round(v)) for v in box)
                rgb_name = crop_filename(stem, rgb_label, det_box, rgb_ext)
                items.append(WorkItem(sid, {
                    "stem": stem, "det_box": det_box, "reports_dir": str(reports),
                    "leaf_id": int(_g(m, "leaf_id", -1)), "detection_id": did,
                    "instance_index": int(_g(m, "instance_index", 0)),
                    "rgb_name": rgb_name, "rgb_path": str(tree / RGB_FOLDER / rgb_name),
                    "mask_path": str(tree / MASK_FOLDER / crop_filename(stem, mask_label, det_box, mask_ext)),
                }))
        return items

    def infer(self, item: WorkItem, model: Any):
        """Measure ONE leaf and write its summary image in the worker, returning only the small row."""
        import cv2

        s = self._settings()
        p = item.payload
        bgr = cv2.imread(p["rgb_path"], cv2.IMREAD_COLOR)
        mask = cv2.imread(p["mask_path"], cv2.IMREAD_GRAYSCALE)
        if bgr is None or mask is None:
            log.warning("Green Index: cutout or mask not found, skipping leaf: %s", p["rgb_path"])
            return None
        if bgr.shape[:2] != mask.shape:
            log.warning("Green Index: cutout and mask differ in size, skipping leaf: %s", p["rgb_path"])
            return None
        rgb, lamina = bgr[:, :, ::-1], mask > 127
        measured = measure_leaf(rgb, lamina, white_peak_check=s["white_peak_check"])
        if measured is None:
            log.warning("Green Index: empty lamina mask, skipping leaf: %s", p["mask_path"])
            return None

        overlay_path = None
        if s["write_overlay"]:
            label = leaf_product_label(SOURCE_TREE, "GI", "lamina")
            overlay_path = Path(p["reports_dir"]).joinpath(*OVERLAY_DIR) / crop_filename(p["stem"], label, p["det_box"], "jpg")
            panel = render_summary_panel(rgb, lamina, measured.used, measured.tiles, measured.white_peak,
                                         title=Path(p["rgb_name"]).stem)
            save_image(panel[:, :, ::-1], overlay_path, quality=OVERLAY_JPG_QUALITY)
        return {
            "leaf_id": p["leaf_id"], "specimen_id": item.specimen_id, "detection_id": p["detection_id"],
            "instance_index": p["instance_index"], "leaf_image_name": p["rgb_name"],
            "leaf_image_path": p["rgb_path"], "overlay_png": str(overlay_path) if overlay_path else None,
            **measured.fields,
        }

    def persist(self, project, item: WorkItem, row) -> None:
        """Record ONE leaf's row (upsert on leaf_id). A leaf that produced nothing returns None."""
        if row is not None:
            project.db.record_leaf_green_index_one(row)

    def run(self, project, ctx) -> None:
        """Per-leaf work, then green_index.csv. The Reporter's CSV export runs before this stage, so the
        stage writes its own file, from every row in the table so a resumed run still lists them all."""
        super().run(project, ctx)
        export_stage_csv(project, self.cfg, self.key)


# -- report-dependency enforcement -------------------------------------------------
def enforce_report_deps(cfg) -> None:
    """When Green Index is enabled, make sure the Reporter exports the two products it reads.

    Only a setting that is actually off gets changed. The Reporter's settings hash covers the whole
    ``report`` block, so writing even an already-effective ``True`` into it would look like a config
    change and re-render the Reporter, ECT and Momocs on every existing run.
    """
    if not cfg.is_enabled("green_index"):
        return
    rep = cfg.report
    lp = rep.get("leaf_products")
    if lp is None:
        return                                    # unset: leaf products default on, these two included
    if not isinstance(lp, dict):
        lp = {}
        rep["leaf_products"] = lp
    forced = []
    for key in ("enabled", "oriented"):
        if not lp.get(key, True):
            lp[key] = True
            forced.append(key)
    products = lp.get("products")
    if isinstance(products, dict):                # a partial dict: each product keeps its own default
        for key in ("lamina_rgb", "lamina_mask"):
            if not products.get(key, True):
                products[key] = True
                forced.append(key)
    if forced:
        log.info("Green Index enabled -> forcing Reporter leaf products on: %s", forced)


# -- tolerant accessors ------------------------------------------------------------
def _finite(v) -> Optional[float]:
    v = float(v)
    return v if np.isfinite(v) else None


def _finite_mean(values: np.ndarray) -> Optional[float]:
    ok = np.isfinite(values)
    return float(values[ok].mean()) if ok.any() else None


def _g(row, key, default=None):
    if row is None:
        return default
    if hasattr(row, key):
        v = getattr(row, key)
        return default if v is None else v
    try:
        v = row[key]
    except Exception:
        return default
    return default if v is None else v


def _get(obj, *path, default=None):
    cur = obj
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p, None) if hasattr(cur, "get") else getattr(cur, p, None)
    return default if cur is None else cur
