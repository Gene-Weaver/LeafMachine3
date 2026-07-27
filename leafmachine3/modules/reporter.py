"""Reporter — stage 8: render user-selected outputs from stored records + originals.

The Reporter never re-runs inference. It reconstructs every requested view on demand from
the per-project database and the images. Each output is a folder (named exactly as below)
and can be toggled independently in ``report`` of ``LM3_settings.yaml``. The layout mirrors
``Crops`` — a category folder with per-output subfolders::

    Overlay/Overlay_Summary/<stem>__Overlay.<ext>          (masks + boxes + landmarks on top)
    Overlay/Overlay_Landmarks/<stem>__LM-leaf__x_y_x_y.<ext>  (one per leaf: keypoints + measures)
    Overlay/Overlay_Petiole/<stem>__PET-leaf__x_y_x_y.<ext>   (one per leaf: petiole width band + panel)
    Crops/RGB__<friendly>/<stem>__BBOX-<friendly>__x_y_x_y.<ext>   (NON-leaf detector classes)
    {Original,Oriented}/<product>/<stem>__<PREFIX>-<friendly>__x_y_x_y.<ext>  (7 leaf products;
        friendly = leaf/lamina/laminaPetiole/laminaHoles; holes in laminaHoles RGB = (10,10,10))
    Binary_Masks/Binary_Masks_Full_Image__<Cls>/<stem>__MaskFull-<friendly>.<ext>
    Binary_Masks/Binary_Masks__<Cls>/<stem>__SEG-<friendly>__x_y_x_y.<ext>
    RGB_Masks/RGB_Masks_Full_Image__<Cls>/<stem>__MaskFull-<friendly>.<ext>
    RGB_Masks/RGB_Masks__<Cls>/<stem>__SEG-<friendly>__x_y_x_y.<ext>

``<Cls>`` (masks) is each segmentation class in ``report.masks.classes`` (default ``Leaf``);
``Crops`` covers both detectors (``report.crops``; ``classes: all`` or a list). Files are named
so they can be reinserted into the parent by filename. Full-image mask outputs render on the
immutable original (geometry scaled by ``1/work_scale``); per-crop outputs use the working-copy
pixels the crop was cut from.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from leafmachine3.core.imaging import (
    composite,
    crop_filename,
    decode_polygon,
    offset_polygon,
    polygon_mask,
    read_image,
    save_crop,
    save_image,
    scale_polygon,
)
from leafmachine3.core.naming import crop_label, friendly_name
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.reporting.leaf_products import PRODUCTS, PRODUCT_KEYS, render_leaf_products
from leafmachine3.reporting.overlay import (
    build_leaf_landmark_overlay,
    build_leaf_petiole_overlay,
    build_summary_image,
)
from leafmachine3.reporting.palette import LandmarkStyle, OverlayStyle, PetioleStyle

log = logging.getLogger("leafmachine3.reporter")

_HOLE = "Hole"
_LEAF = "Leaf"
_PETIOLE = "Petiole"
_LEAF_DET_CLASSES = ("Leaf_WHOLE", "Leaf_PARTIAL")   # leaf bbox crops come from the Leaf_Products tree
_MASK_OUTPUTS = ("Binary_Masks_Full_Image", "RGB_Masks_Full_Image", "Binary_Masks", "RGB_Masks")


class Reporter(PipelineStage):
    """Render the Overlay, mask exports, and raw crop exports from stored records."""

    key: str = "reporter"
    name: str = "Reporter"
    device_kind: str = "cpu"
    depends_on: tuple[str, ...] = (
        "archival_detector", "plant_detector", "phenology_detector",
        "ruler_cf", "leaf_segmenter", "morphology",
        "landmark_detector", "landmark_measurements", "leaf_orientation",
        "petiole_width", "metric_grounding",
    )
    owns_tables: tuple[str, ...] = ()

    # ---- pipeline hooks --------------------------------------------------- #
    def collect_items(self, project) -> list[WorkItem]:
        return [WorkItem(sid, project.report_bundle(sid)) for sid in project.db.iter_specimen_ids()]

    def infer(self, item: WorkItem, model: Any) -> list[tuple[str, str]]:
        r = self.cfg.report
        b = item.payload
        reports = Path(str(b.reports_dir))
        stem = str(b.stem)
        img_ext = str(_get(r, "formats", "image_ext", default="jpg")).lstrip(".")
        mask_ext = str(_get(r, "formats", "mask_ext", default="png")).lstrip(".")
        quality = int(_get(r, "formats", "jpg_quality", default=100))
        written: list[tuple[str, str]] = []

        # cache decoded images so a large original is read at most once per specimen
        cache: dict[str, np.ndarray] = {}

        def read(path) -> np.ndarray:
            key = str(path)
            if key not in cache:
                cache[key] = read_image(key)
            return cache[key]

        # ---- Overlay_Summary (masks + boxes + landmarks on top) ----------
        overlay_cfg = _sub(r, "overlay")
        lm_style = LandmarkStyle.from_config(self.cfg)
        pet_style = PetioleStyle.from_config(self.cfg)
        if _flag(overlay_cfg, "enabled", True):
            style = OverlayStyle.from_config(self.cfg)
            summary = build_summary_image(
                read(b.original_path), b.detections, b.leaves, b.cf_px_per_cm, style, b.work_scale,
                morphology=b.morphology, landmarks=b.landmarks, landmark_style=lm_style,
                landmark_measurements=b.landmark_measurements,
                petioles=b.petioles, petiole_style=pet_style,
            )
            path = reports / "Overlay" / "Overlay_Summary" / f"{stem}__Overlay.{img_ext}"
            save_image(summary, path, quality=quality)
            written.append((str(path), "Overlay/Overlay_Summary"))

        # ---- Overlay_Landmarks (one per leaf: keypoints + measurement panel)
        ovlm_cfg = _sub(r, "overlay_landmarks")
        if _flag(ovlm_cfg, "enabled", True) and b.landmarks:
            written += self._export_landmark_overlays(b, reports, stem, img_ext, quality, read, lm_style)

        # ---- Overlay_Petiole (one per leaf with a petiole: masks + width band + panel)
        ovpet_cfg = _sub(r, "overlay_petiole")
        style_overlay = OverlayStyle.from_config(self.cfg)
        if _flag(ovpet_cfg, "enabled", True) and b.petioles:
            written += self._export_petiole_overlays(b, reports, stem, img_ext, quality, read, pet_style, style_overlay)

        # ---- mask exports ------------------------------------------------
        masks_cfg = _sub(r, "masks")
        enabled = {name: _flag(masks_cfg, name, False) for name in _MASK_OUTPUTS}
        if any(enabled.values()) and b.leaves:
            written += self._export_masks(b, masks_cfg, enabled, reports, stem, img_ext, mask_ext, quality, read)

        # ---- raw RGB crop exports (non-leaf classes; leaf crops -> Leaf_Products) ----
        crops_cfg = _sub(r, "crops")
        if _flag(crops_cfg, "enabled", False):
            written += self._export_crops(b, crops_cfg, reports, stem, img_ext, quality, read)

        # ---- leaf products: Original/ + Oriented/ trees (5 products each) ----
        lp_cfg = _sub(r, "leaf_products")
        if _flag(lp_cfg, "enabled", True) and b.leaves:
            written += self._export_leaf_products(b, lp_cfg, reports, stem, img_ext, mask_ext, quality, read)

        return written

    def persist(self, project, item: WorkItem, payload: list[tuple[str, str]]) -> None:
        project.db.record_report_manifest(item.specimen_id, payload)

    # ---- mask exports ---------------------------------------------------- #
    def _export_masks(self, b, masks_cfg, enabled, reports, stem, img_ext, mask_ext, quality, read):
        classes = _list(_get(masks_cfg, "classes", default=[_LEAF])) or [_LEAF]
        bg = 0 if str(_get(masks_cfg, "background", default="black")).lower() == "black" else 255
        subtract_holes = bool(_get(masks_cfg, "subtract_holes", default=True))
        scale = 1.0 / float(b.work_scale or 1.0)

        need_full = enabled["Binary_Masks_Full_Image"] or enabled["RGB_Masks_Full_Image"]
        need_crop = enabled["Binary_Masks"] or enabled["RGB_Masks"]
        original = read(b.original_path) if need_full else None
        working = read(b.working_path) if (need_crop and b.working_path) else None

        parsed = _parse_leaves(b.leaves)                       # (detection_id, cls_name, poly) working coords
        written: list[tuple[str, str]] = []

        for cls in classes:
            holes_wanted = subtract_holes and cls == _LEAF
            seg_label = crop_label(self.cfg, "seg", cls)         # SEG-leaf   (per-crop files)
            full_label = crop_label(self.cfg, "mask_full", cls)   # MaskFull-leaf (full-image files)

            # -- FULL-IMAGE (one file per specimen, original frame) --------
            #    reports/{Binary,RGB}_Masks/<Binary,RGB>_Masks_Full_Image__<Cls>/<stem>__MaskFull-<friendly>.<ext>
            if need_full and original is not None:
                fg = [scale_polygon(p, scale) for (_d, cn, p) in parsed if cn == cls]
                holes = [scale_polygon(p, scale) for (_d, cn, p) in parsed if cn == _HOLE] if holes_wanted else []
                if fg:
                    binary = _binary(original.shape, fg, holes)
                    if enabled["Binary_Masks_Full_Image"]:
                        sub = f"Binary_Masks_Full_Image__{cls}"
                        p = reports / "Binary_Masks" / sub / f"{stem}__{full_label}.{mask_ext}"
                        save_image((binary.astype(np.uint8) * 255), p)
                        written.append((str(p), f"Binary_Masks/{sub}"))
                    if enabled["RGB_Masks_Full_Image"]:
                        sub = f"RGB_Masks_Full_Image__{cls}"
                        p = reports / "RGB_Masks" / sub / f"{stem}__{full_label}.{img_ext}"
                        save_image(composite(original, binary, bg=bg), p, quality=quality)
                        written.append((str(p), f"RGB_Masks/{sub}"))

            # -- PER-CROP (one file per leaf detection crop, crop frame) ----
            #    reports/{Binary,RGB}_Masks/<Binary,RGB>_Masks__<Cls>/<stem>__SEG-<friendly>__x_y_x_y.<ext>
            if need_crop:
                by_det = _group_by_detection(parsed, cls, holes_wanted)
                for did, grp in by_det.items():
                    box = b.crop_boxes.get(did)
                    if not box or not grp["fg"]:
                        continue
                    x1, y1, x2, y2 = (int(round(v)) for v in box)
                    ch, cw = max(1, y2 - y1), max(1, x2 - x1)
                    fg = [offset_polygon(p, -x1, -y1) for p in grp["fg"]]
                    holes = [offset_polygon(p, -x1, -y1) for p in grp["holes"]] if holes_wanted else []
                    binary = _binary((ch, cw), fg, holes)
                    if enabled["Binary_Masks"]:
                        sub = f"Binary_Masks__{cls}"
                        name = crop_filename(stem, seg_label, (x1, y1, x2, y2), mask_ext)
                        p = reports / "Binary_Masks" / sub / name
                        save_image((binary.astype(np.uint8) * 255), p)
                        written.append((str(p), f"Binary_Masks/{sub}"))
                    if enabled["RGB_Masks"] and working is not None:
                        crop = working[max(0, y1):y2, max(0, x1):x2]
                        mh, mw = crop.shape[:2]
                        if mh and mw:
                            comp = composite(crop, binary[:mh, :mw], bg=bg)
                            sub = f"RGB_Masks__{cls}"
                            name = crop_filename(stem, seg_label, (x1, y1, x2, y2), img_ext)
                            p = reports / "RGB_Masks" / sub / name
                            save_image(comp, p, quality=quality)
                            written.append((str(p), f"RGB_Masks/{sub}"))
        return written

    # ---- raw RGB crop exports (both detectors) --------------------------- #
    def _export_crops(self, b, crops_cfg, reports, stem, img_ext, quality, read):
        """Export the raw RGB bbox crops into ``Crops/RGB__<friendly>/`` for selected classes.

        ``classes`` is ``"all"`` (every archival + plant class) or a list of real class names.
        ``source`` chooses which pixels to cut: ``working`` (matches the detector crops, no
        scaling) or ``original`` (full-res; boxes scaled by ``1/work_scale``).
        """
        classes = _get(crops_cfg, "classes", default="all")
        want_all = isinstance(classes, str) and classes.lower() == "all"
        wanted = None if want_all else set(_list(classes))
        source = str(_get(crops_cfg, "source", default="working")).lower()
        img = read(b.working_path if (source == "working" and b.working_path) else b.original_path)
        box_scale = 1.0 if source == "working" else (1.0 / float(b.work_scale or 1.0))
        crops_root = reports / "Crops"

        written: list[tuple[str, str]] = []
        for det in b.detections or []:
            cls = str(_row_get(det, "cls_name", ""))
            if cls in _LEAF_DET_CLASSES:
                continue                                        # leaf crops live in the Leaf_Products tree
            if wanted is not None and cls not in wanted:
                continue
            xyxy = _row_get(det, "xyxy")
            if not xyxy:
                continue
            box = tuple(float(v) * box_scale for v in xyxy)
            friendly = friendly_name(self.cfg, cls)
            label = crop_label(self.cfg, "bbox", cls)          # BBOX-<friendly>
            folder = crops_root / f"RGB__{friendly}"
            path = save_crop(img, box, stem, label, folder, ext=img_ext, quality=quality)
            written.append((path, f"Crops/RGB__{friendly}"))
        return written

    # ---- per-leaf landmark overlays -------------------------------------- #
    def _export_landmark_overlays(self, b, reports, stem, img_ext, quality, read, lm_style):
        """Write ``Overlay/Overlay_Landmarks/<stem>__LM-leaf__x_y_x_y.<ext>`` -- one per leaf crop,
        with the keypoints/skeleton drawn on the crop plus a panel of the derived measurements.

        The crop is re-cut from the WORKING image with the same clamping the detector used, so the
        stored crop-frame keypoints (``x_crop``/``y_crop``) line up exactly.
        """
        if not b.working_path:
            return []
        working = read(b.working_path)
        h, w = working.shape[:2]
        by_det = _group_landmark_rows(b.landmarks)
        meas = {
            (int(_row_get(m, "detection_id", -1)), int(_row_get(m, "instance_index", 0))): m
            for m in (b.landmark_measurements or [])
        }
        folder = reports / "Overlay" / "Overlay_Landmarks"
        label = crop_label(self.cfg, "landmark", _LEAF)      # LM-leaf
        written: list[tuple[str, str]] = []
        for (did, inst), rows in by_det.items():
            box = b.crop_boxes.get(did)
            if not box:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            crop = working[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            mrow = meas.get((did, inst))
            lines = _landmark_measure_lines(mrow)
            cp = _row_get(mrow, "curvature_point", None) if mrow is not None else None
            img = build_leaf_landmark_overlay(
                crop, rows, lines, lm_style, curvature_idx=int(cp) if cp is not None else None)
            # Usually one leaf per crop (inst 0). If the pose model emits >1 instance for a crop
            # they share the detection box, so fold the instance into the stem (parse-safe) to keep
            # each file distinct instead of overwriting.
            file_stem = stem if inst == 0 else f"{stem}__i{inst}"
            name = crop_filename(file_stem, label, (x1, y1, x2, y2), img_ext)
            path = folder / name
            save_image(img, path, quality=quality)
            written.append((str(path), "Overlay/Overlay_Landmarks"))
        return written

    # ---- per-leaf petiole overlays --------------------------------------- #
    def _export_petiole_overlays(self, b, reports, stem, img_ext, quality, read, pet_style, style):
        """Write ``Overlay/Overlay_Petiole/<stem>__PET-leaf__x_y_x_y.<ext>`` -- the leaf crop with the
        Leaf + Petiole masks filled (no outline), the width sample probes + reported width band, and a
        panel of the lamina areas + petiole width. One per leaf that has a petiole."""
        if not b.working_path:
            return []
        working = read(b.working_path)
        h, w = working.shape[:2]
        by_det = _group_seg_by_detection(_parse_leaves(b.leaves))
        morph_by_det = {int(_row_get(m, "detection_id", -1)): m for m in (b.morphology or [])}
        seg_colors = {"Leaf": style.color_for(_LEAF), "Petiole": style.color_for(_PETIOLE)}
        folder = reports / "Overlay" / "Overlay_Petiole"
        label = crop_label(self.cfg, "petiole", _LEAF)          # PET-leaf
        written: list[tuple[str, str]] = []
        for pr in b.petioles:
            did = int(_row_get(pr, "detection_id", -1))
            box = b.crop_boxes.get(did)
            if not box:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            crop = working[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                continue
            groups = by_det.get(did, {})
            class_polys = {cls: [offset_polygon(p, -x1, -y1) for p in groups.get(cls, [])]
                           for cls in (_LEAF, _PETIOLE)}
            wseg = _load_seg(_row_get(pr, "width_segment_json", None), x1, y1)        # working -> crop
            sseg = _load_segs(_row_get(pr, "sample_segments_json", None), x1, y1)
            lines = _petiole_measure_lines(morph_by_det.get(did), pr)
            img = build_leaf_petiole_overlay(crop, class_polys, sseg, wseg, lines, seg_colors, pet_style)
            path = folder / crop_filename(stem, label, (x1, y1, x2, y2), img_ext)
            save_image(img, path, quality=quality)
            written.append((str(path), "Overlay/Overlay_Petiole"))
        return written

    # ---- leaf products (Original + Oriented trees) ----------------------- #
    def _export_leaf_products(self, b, lp_cfg, reports, stem, img_ext, mask_ext, quality, read):
        """Render the 5 leaf products under ``Original/`` and (when orientation succeeded)
        ``Oriented/``. Masks/cutouts are content-fitted; the bbox crop is not. Uses the working
        image + segmentation polygons + the per-leaf rotation stored on ``leaf_morphology``.
        """
        if not b.working_path:
            return []
        want = _leaf_products_wanted(lp_cfg)
        if not want:
            return []
        do_original = _flag(lp_cfg, "original", True)
        do_oriented = _flag(lp_cfg, "oriented", True)
        bg = 0 if str(_get(lp_cfg, "background", default="black")).lower() == "black" else 255
        pad = int(_get(lp_cfg, "fit_pad", default=0))
        hole_color = tuple(int(c) for c in (_get(lp_cfg, "hole_rgb_color", default=(10, 10, 10))))
        working = read(b.working_path)
        H, W = working.shape[:2]

        # per-leaf-crop orientation (success + CW angle) from the morphology rows
        orient: dict[int, tuple[bool, Optional[float]]] = {}
        for m in b.morphology or []:
            did = int(_row_get(m, "detection_id", -1))
            orient[did] = (bool(_row_get(m, "oriented_leaf_success", 0)),
                           _row_get(m, "oriented_leaf_rotation_angle_degreesCW", None))

        by_det = _group_seg_by_detection(_parse_leaves(b.leaves))
        written: list[tuple[str, str]] = []
        for did, groups in by_det.items():
            lam_polys = groups.get(_LEAF, [])
            box = b.crop_boxes.get(did)
            if not lam_polys or not box:
                continue
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            crop = working[cy1:cy2, cx1:cx2]
            ch, cw = crop.shape[:2]
            if ch == 0 or cw == 0:
                continue

            def raster(polys):
                out = np.zeros((ch, cw), dtype=bool)
                for p in polys:
                    out |= polygon_mask(offset_polygon(p, -cx1, -cy1), (ch, cw))
                return out

            silhouette = raster(lam_polys)                     # Leaf outline with holes FILLED in
            hole = raster(groups.get(_HOLE, []))
            lamina = silhouette & ~hole                        # tissue only (holes removed)
            pet_polys = groups.get(_PETIOLE, [])
            laminapet = ((silhouette | raster(pet_polys)) & ~hole) if pet_polys else None
            masks = {"lamina": lamina, "laminapet": laminapet, "silhouette": silhouette, "hole": hole}
            det_box = (x1, y1, x2, y2)

            if do_original:
                products = render_leaf_products(crop, masks, angle_cw=None, bg=bg, want=want, pad=pad, hole_color=hole_color)
                written += self._write_leaf_products(products, "Original", reports, stem, det_box, img_ext, mask_ext, quality)

            success, angle = orient.get(did, (False, None))
            if do_oriented and success and angle is not None:
                products = render_leaf_products(crop, masks, angle_cw=float(angle), bg=bg, want=want, pad=pad, hole_color=hole_color)
                written += self._write_leaf_products(products, "Oriented", reports, stem, det_box, img_ext, mask_ext, quality)
        return written

    def _write_leaf_products(self, products, tree, reports, stem, det_box, img_ext, mask_ext, quality):
        """Save each rendered product to ``reports/<tree>/<folder>/<stem>__<PREFIX>-<friendly>__x_y_x_y.<ext>``."""
        written: list[tuple[str, str]] = []
        for p in PRODUCTS:
            img = products.get(p.key)
            if img is None:
                continue
            ext = mask_ext if p.is_mask else img_ext
            name = crop_filename(stem, f"{p.prefix}-{p.friendly}", det_box, ext)
            path = reports / tree / p.folder / name
            save_image(img, path, quality=quality)
            written.append((str(path), f"{tree}/{p.folder}"))
        return written


# -- geometry helpers --------------------------------------------------------------
def _group_seg_by_detection(parsed) -> dict:
    """Group parsed leaf polygons ``(detection_id, cls_name, poly)`` -> ``{det: {cls: [poly]}}``."""
    by_det: dict[int, dict[str, list]] = {}
    for did, cls, poly in parsed:
        by_det.setdefault(did, {}).setdefault(cls, []).append(poly)
    return by_det


def _leaf_products_wanted(lp_cfg) -> set:
    """Set of enabled product keys from ``report.leaf_products.products`` (default: all)."""
    pcfg = _getk(lp_cfg, "products", None)
    if pcfg is None:
        return set(PRODUCT_KEYS)
    return {k for k in PRODUCT_KEYS if _flag(pcfg, k, True)}


def _group_landmark_rows(landmarks) -> dict:
    """Group leaf_landmark rows by ``(detection_id, instance_index)`` (one leaf per group)."""
    groups: dict = {}
    for r in landmarks or []:
        key = (int(_row_get(r, "detection_id", -1)), int(_row_get(r, "instance_index", 0)))
        groups.setdefault(key, []).append(r)
    return groups


def _load_seg(raw, x1: int, y1: int):
    """A JSON width segment (working coords) -> crop-frame ``[[x,y],[x,y]]``, or None."""
    if not raw:
        return None
    try:
        s = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return None
    if not s or len(s) < 2:
        return None
    return [[p[0] - x1, p[1] - y1] for p in s]


def _load_segs(raw, x1: int, y1: int) -> list:
    """A JSON list of segments (working coords) -> crop-frame segments."""
    if not raw:
        return []
    try:
        ss = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    return [[[p[0] - x1, p[1] - y1] for p in s] for s in ss if s and len(s) >= 2]


def _petiole_measure_lines(morph, pr) -> list[str]:
    """Panel lines for the petiole overlay: all lamina-area versions + the petiole width/length."""
    def gi(row, key: str) -> str:
        v = _row_get(row, key, None)
        return "n/a" if v is None else str(int(round(float(v))))

    return [
        f"lamina_incl: {gi(morph, 'lamina_area_incl_holes_px')} px",
        f"lamina_excl: {gi(morph, 'lamina_area_excl_holes_px')} px",
        f"lamina_hole: {gi(morph, 'lamina_hole_area_px')} px",
        f"n_holes: {gi(morph, 'n_holes')}",
        f"petiole_w: {gi(pr, 'width_px')} px",
        f"petiole_len: {gi(pr, 'length_px')} px",
    ]


def _landmark_measure_lines(m) -> list[str]:
    """Format a leaf_landmark_measurement row into overlay text lines (values rounded to int;
    curvature to 2dp). Missing metrics render as ``n/a`` (no degree glyph -- Hershey font safe)."""
    if m is None:
        return []

    def as_int(key: str) -> str:
        v = _row_get(m, key, None)
        return "n/a" if v is None else str(int(round(float(v))))

    def angle(key: str, type_key: str) -> str:
        v = _row_get(m, key, None)
        if v is None:
            return "n/a"
        t = _row_get(m, type_key, None)
        base = f"{int(round(float(v)))}deg"
        return f"{base} {t}" if t else base

    cv = _row_get(m, "lamina_curvature", None)
    return [
        f"lamina_trace: {as_int('lamina_trace_length')} px",
        f"lamina_extent: {as_int('lamina_extent')} px",
        f"tip_base: {as_int('lamina_tip_base_length')} px",
        f"leaf_width: {as_int('leaf_width')} px",
        f"apex: {angle('apex_angle', 'apex_angle_type')}",
        f"base: {angle('base_angle', 'base_angle_type')}",
        f"petiole_trace: {as_int('petiole_trace_length')} px",
        f"curvature: {'n/a' if cv is None else f'{int(round(float(cv)))}deg'}",
    ]
def _parse_leaves(leaves) -> list[tuple[int, str, np.ndarray]]:
    out: list[tuple[int, str, np.ndarray]] = []
    for row in leaves or []:
        if str(_row_get(row, "mask_format", "polygon_xy")) != "polygon_xy":
            continue
        data = _row_get(row, "mask_data")
        if not data:
            continue
        try:
            poly = decode_polygon(str(data))
        except Exception:
            continue
        if len(poly) < 3:
            continue
        out.append((int(_row_get(row, "detection_id", -1)), str(_row_get(row, "cls_name", "")), poly))
    return out


def _group_by_detection(parsed, cls: str, holes_wanted: bool) -> dict:
    by_det: dict[int, dict] = {}
    for did, cn, poly in parsed:
        if cn == cls:
            by_det.setdefault(did, {"fg": [], "holes": []})["fg"].append(poly)
        elif cn == _HOLE and holes_wanted:
            by_det.setdefault(did, {"fg": [], "holes": []})["holes"].append(poly)
    return by_det


def _binary(shape_hw, fg: list[np.ndarray], holes: list[np.ndarray]) -> np.ndarray:
    out = np.zeros(shape_hw[:2], dtype=bool)
    for poly in fg:
        out |= polygon_mask(poly, shape_hw)
    for hole in holes:
        out[polygon_mask(hole, shape_hw)] = False
    return out


# -- tolerant config / row accessors -----------------------------------------------
def _list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def _sub(obj, key):
    return _getk(obj, key, None)


def _flag(obj, key, default) -> bool:
    return bool(_getk(obj, key, default))


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
