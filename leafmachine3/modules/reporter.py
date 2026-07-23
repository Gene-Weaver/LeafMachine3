"""Reporter — stage 8: render user-selected outputs from stored records + originals.

The Reporter never re-runs inference. It reconstructs every requested view on demand
from the per-project database and the IMMUTABLE original image:

* ``report.overlay`` -> a Summary_Image with detection boxes, leaf masks and a CF banner
  (see :mod:`leafmachine3.reporting.overlay`);
* ``report.export_binary_masks`` -> one combined binary PNG per specimen (holes subtracted);
* ``report.export_rgb_on_black`` / ``report.export_rgb_on_white`` -> per-leaf RGB composites.

Stored geometry is in the WORKING frame, so it is scaled by ``1/work_scale`` before being
composited onto the full-resolution original. Written paths are recorded in
``report_manifest`` so ``--restart reporter`` can delete exactly what was produced.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

from leafmachine3.core.imaging import (
    composite,
    decode_polygon,
    polygon_mask,
    read_image,
    save_image,
    scale_polygon,
)
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.reporting.overlay import build_summary_image
from leafmachine3.reporting.palette import OverlayStyle

log = logging.getLogger("leafmachine3.reporter")

# Instance classes that contribute positive vs. negative area for binary/composite masks.
_FOREGROUND = ("Leaf", "Petiole")
_HOLE = "Hole"


class Reporter(PipelineStage):
    """Render Summary_Images and mask/composite exports from stored records."""

    key: str = "reporter"
    name: str = "Reporter"
    device_kind: str = "cpu"
    depends_on: tuple[str, ...] = (
        "archival_detector", "plant_detector", "phenology_detector",
        "ruler_cf", "leaf_segmenter", "metric_grounding",
    )
    owns_tables: tuple[str, ...] = ()

    # ---- pipeline hooks --------------------------------------------------- #
    def collect_items(self, project) -> list[WorkItem]:
        return [WorkItem(sid, project.report_bundle(sid)) for sid in project.db.iter_specimen_ids()]

    def infer(self, item: WorkItem, model: Any) -> list[tuple[str, str]]:
        r = self.cfg.report
        bundle = item.payload
        img = read_image(bundle.original_path)                 # immutable original
        scale = 1.0 / float(bundle.work_scale or 1.0)          # working -> original pixels
        reports = Path(str(bundle.reports_dir))
        stem = str(bundle.stem)
        quality = int(_get(r, "formats", "jpg_quality", default=100))
        mask_ext = str(_get(r, "formats", "mask_ext", default="png")).lstrip(".")
        img_ext = str(_get(r, "formats", "image_ext", default="jpg")).lstrip(".")
        written: list[tuple[str, str]] = []

        overlay_cfg = r.get("overlay", None) if hasattr(r, "get") else None
        overlay_on = bool(_getk(overlay_cfg, "enabled", True))
        if overlay_on:
            style = OverlayStyle.from_config(self.cfg)
            summary = build_summary_image(
                img, bundle.detections, bundle.leaves, bundle.cf_px_per_cm, style, bundle.work_scale,
            )
            path = reports / "overlay" / f"{stem}.{img_ext}"
            save_image(summary, path, quality=quality)
            written.append((str(path), "overlay"))

        # Per-specimen scaled foreground/hole polygons (shared by all mask exports).
        fg, holes = _scaled_polygons(bundle.leaves, scale)

        if bool(r.get("export_binary_masks", False)) and fg:
            binary = _binary_mask(img.shape, fg, holes)
            path = reports / "masks" / f"{stem}_mask.{mask_ext}"
            save_image((binary.astype(np.uint8) * 255), path)
            written.append((str(path), "binary_mask"))

        for flag, subdir, bg, kind in (
            ("export_rgb_on_black", "rgb_on_black", 0, "rgb_on_black"),
            ("export_rgb_on_white", "rgb_on_white", 255, "rgb_on_white"),
        ):
            if not bool(r.get(flag, False)):
                continue
            for idx, poly in enumerate(fg):
                mask = polygon_mask(poly, img.shape)
                _subtract_holes(mask, holes)
                comp = composite(img, mask, bg=bg)
                path = reports / subdir / f"{stem}__leaf{idx:02d}.{img_ext}"
                save_image(comp, path, quality=quality)
                written.append((str(path), kind))

        return written

    def persist(self, project, item: WorkItem, payload: list[tuple[str, str]]) -> None:
        project.db.record_report_manifest(item.specimen_id, payload)


# -- geometry helpers --------------------------------------------------------------
def _scaled_polygons(leaves, scale: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split leaf rows into scaled foreground (Leaf/Petiole) and hole polygons."""
    fg: list[np.ndarray] = []
    holes: list[np.ndarray] = []
    for row in leaves:
        cls_name = str(_row_get(row, "cls_name", ""))
        if str(_row_get(row, "mask_format", "polygon_xy")) != "polygon_xy":
            continue
        data = _row_get(row, "mask_data")
        if not data:
            continue
        try:
            poly = scale_polygon(decode_polygon(str(data)), scale)
        except Exception:
            log.warning("skipping unreadable mask polygon for cls=%s", cls_name)
            continue
        if len(poly) < 3:
            continue
        if cls_name in _FOREGROUND:
            fg.append(poly)
        elif cls_name == _HOLE:
            holes.append(poly)
    return fg, holes


def _binary_mask(shape_hw, fg: list[np.ndarray], holes: list[np.ndarray]) -> np.ndarray:
    """Union of the foreground polygons with hole polygons subtracted (boolean)."""
    out = np.zeros(shape_hw[:2], dtype=bool)
    for poly in fg:
        out |= polygon_mask(poly, shape_hw)
    _subtract_holes(out, holes)
    return out


def _subtract_holes(mask: np.ndarray, holes: list[np.ndarray]) -> None:
    """Zero out ``mask`` pixels covered by any hole polygon (in place)."""
    for hole in holes:
        mask[polygon_mask(hole, mask.shape)] = False


# -- tolerant config / row accessors -----------------------------------------------
def _row_get(row: Any, key: str, default: Any = None) -> Any:
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


def _getk(obj: Any, key: str, default: Any) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        try:
            v = obj.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    v = getattr(obj, key, default)
    return default if v is None else v


def _get(obj: Any, *path: str, default: Optional[Any] = None) -> Any:
    cur = obj
    for p in path:
        cur = _getk(cur, p, None)
        if cur is None:
            return default
    return cur
