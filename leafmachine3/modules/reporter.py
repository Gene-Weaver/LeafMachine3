"""Reporter — stage 8: render user-selected outputs from stored records + working images.

The Reporter never re-runs inference. It reconstructs every requested view on demand from
the per-project database and the images. Each output is a folder (named exactly as below)
and can be toggled independently in ``report`` of ``LM3_settings.yaml``. The layout mirrors
``Crops`` — a category folder with per-output subfolders::

    Overlay/Overlay_Summary/<stem>__Overlay.<ext>          (masks + boxes + landmarks on top)
        ``report.overlay.insert_cf_exterior`` appends a 1 cm checkerboard margin to this one's
        top and left edges, so it is the single output whose size is not the original's.
    Overlay/Overlay_Landmarks/<stem>__LM-leaf__x_y_x_y.<ext>  (one per leaf: keypoints + measures)
    Overlay/Overlay_Petiole/<stem>__PET-leaf__x_y_x_y.<ext>   (one per leaf: petiole width band + panel)
    Overlay/Overlay_Specimen_Segmentation/<stem>__SpecimenSeg.<ext>  (2-panel: annotated sheet | cutout)
    Crops/RGB__<friendly>/<stem>__BBOX-<friendly>__x_y_x_y.<ext>   (NON-leaf detector classes)
    Leaf_Original/<product>/<stem>__og-<PREFIX>-<friendly>__x_y_x_y.<ext>  (7 leaf products;
    Leaf_Oriented/<product>/<stem>__or-<PREFIX>-<friendly>__x_y_x_y.<ext>   friendly =
        leaf/lamina/laminaPetiole/laminaHoles; holes in laminaHoles RGB = (10,10,10))
    Specimen_Masks/Binary_Masks_Specimen__<Cls>/<stem>__MaskFull-<friendly>.<ext>
    Specimen_Masks/Binary_Masks__<Cls>/<stem>__SEG-<friendly>__x_y_x_y.<ext>
    Specimen_Masks/Binary_Masks_Specimen/<stem>__MaskFull-specimen.<ext>  (whole-specimen mask)
    Specimen_Masks/RGB_Masks_Specimen__<Cls>/<stem>__MaskRGBFull-<friendly>.<ext>
    Specimen_Masks/RGB_Masks__<Cls>/<stem>__SEGRGB-<friendly>__x_y_x_y.<ext>
    Specimen_Masks/RGB_Masks_Specimen/<stem>__MaskRGBFull-specimen.<ext>  (whole-specimen cutout)
    Specimen_Masks/Binary_Masks_Specimen_Inverse/<stem>__MaskFull-specimenInverse.<ext>
    Specimen_Masks/RGB_Masks_Specimen_Inverse/<stem>__MaskRGBFull-specimenInverse.<ext>
    Data/*.csv                                             (the numbers behind all of the above)

``Data/`` is the LAST thing the Reporter writes and the only output that is not per-specimen: one
pass over the whole database emitting ``leaf_measurements.csv`` (one row per leaf, every
measurement) and its companions. See ``leafmachine3.reporting.data_export``.

The ``*_Specimen_Inverse`` pair is the complement of the whole-specimen pair: the binary mask of
everything that is NOT plant, and the sheet with the plant lifted off it and replaced by
``report.masks.inverse_fill`` (white by default). Both are optional and default OFF.

Every ID token above (the ``__<id>__`` middle part) is unique across the whole tree WITHOUT its
extension, so a user can flatten an entire run into one directory and lose nothing. That is why
each RGB output carries its own prefix rather than sharing the binary one (``SEGRGB``/``SEG``,
``MaskRGBFull``/``MaskFull``) and why the leaf products are tagged ``og-``/``or-`` per tree.

``<Cls>`` (masks) is each segmentation class in ``report.masks.classes`` (default ``Leaf``);
``Crops`` covers both detectors (``report.crops``; ``classes: all`` or a list). Files are named
so they can be reinserted into the parent by filename. EVERY output is rendered in the WORKING frame -- the resized copy every stage actually analyzed
(see ``ingest.max_working_dim``). The Reporter never opens an original: an overlay upscaled by
``1/work_scale`` would carry 4x the pixels of the evidence behind it, invent mask detail that was
never measured, and make the originals a permanent hard dependency of every re-report. The
originals' *dimensions* stay on the specimen row -- the MP->CF regression is fit on them -- but
their *pixels* are read exactly once, by ingest.
"""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

import numpy as np

_SUB_LOCK = threading.Lock()   # guards the Reporter's cross-thread component-timing accumulator

from leafmachine3.core.imaging import (
    composite,
    crop_filename,
    decode_polygon,
    offset_polygon,
    polygon_mask,
    read_image,
    save_crop,
    save_image,
)
from leafmachine3.core.naming import crop_label, friendly_name
from leafmachine3.core.records import CF_SOURCE_MP
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.reporting.data_export import export_data_csvs
from leafmachine3.reporting.leaf_products import (
    PRODUCTS,
    PRODUCT_KEYS,
    leaf_product_label,
    render_leaf_products,
)
from leafmachine3.reporting.overlay import (
    build_leaf_landmark_overlay,
    build_leaf_petiole_overlay,
    build_specimen_overlay,
    build_summary_image,
    cf_fallback_reason,
)
from leafmachine3.reporting.palette import (
    CFScalebarStyle,
    LandmarkStyle,
    OverlayStyle,
    PetioleStyle,
    SpecimenStyle,
    parse_fill_color,
)

log = logging.getLogger("leafmachine3.reporter")

_HOLE = "Hole"
_LEAF = "Leaf"
_PETIOLE = "Petiole"
_LEAF_DET_CLASSES = ("Leaf_WHOLE", "Leaf_PARTIAL")   # leaf bbox crops come from the Leaf_Products tree
_MASK_OUTPUTS = ("Binary_Masks_Full_Image", "RGB_Masks_Full_Image", "Binary_Masks", "RGB_Masks")
# Single parent for every mask export. The per-class subfolder names are unchanged
# (Binary_Masks_Specimen, RGB_Masks_Specimen__Leaf, ...); they simply all sit here now
# instead of being split across two sibling Binary_Masks/ and RGB_Masks/ trees.
_MASKS_ROOT = "Specimen_Masks"
# Stored geometry IS working-frame, and every output is now rendered there, so the Reporter's
# scale factor is identically 1. Named rather than inlined so the frame contract stays greppable.
_WORKING = 1.0


class _SpecimenExport(NamedTuple):
    """One whole-specimen file the Reporter can write from the SpecimenSegmenter's mask."""

    key: str          # report.masks toggle
    folder: str       # subfolder under Specimen_Masks/
    cls_name: str     # naming.friendly_names key -> the MaskFull-<friendly> filename label
    rgb: bool         # RGB cutout (True) or {0,255} binary mask (False)
    invert: bool      # the NOT-specimen side of the mask
    default: bool     # inline default, used when the config predates the key


# The whole-specimen exports, in settings-file order. The two INVERSE outputs default OFF so a
# config written before they existed keeps producing exactly the files it always produced; the
# shipped LM3_settings.yaml turns them on.
_SPECIMEN_EXPORTS: tuple[_SpecimenExport, ...] = (
    _SpecimenExport("Binary_Masks__Specimen", "Binary_Masks_Specimen",
                    "Specimen", rgb=False, invert=False, default=True),
    _SpecimenExport("RGB_Masks__Specimen", "RGB_Masks_Specimen",
                    "Specimen", rgb=True, invert=False, default=True),
    _SpecimenExport("Binary_Masks__Specimen_Inverse", "Binary_Masks_Specimen_Inverse",
                    "Specimen_Inverse", rgb=False, invert=True, default=False),
    _SpecimenExport("RGB_Masks__Specimen_Inverse", "RGB_Masks_Specimen_Inverse",
                    "Specimen_Inverse", rgb=True, invert=True, default=False),
)


class Reporter(PipelineStage):
    """Render the Overlay, mask exports, and raw crop exports from stored records."""

    key: str = "reporter"
    name: str = "Reporter"
    device_kind: str = "cpu"
    io_bound: bool = True              # disk-write-bound: sized to the disk-write knee, not cpu_cores-2
    depends_on: tuple[str, ...] = (
        "archival_detector", "plant_detector", "specimen_segmenter", "phenology_detector",
        "ruler_classifier", "ruler_cf", "leaf_segmenter", "morphology",
        "landmark_detector", "landmark_measurements", "leaf_orientation",
        "petiole_width", "metric_grounding",
    )
    owns_tables: tuple[str, ...] = ()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._subtimes: dict[str, float] = {}                # per-component seconds (timing profiler)
        self._timing_on = bool(cfg.timing.get("enabled", False))

    @contextlib.contextmanager
    def _phase(self, name: str):
        """Accumulate a component's wall time across the Reporter's worker threads (no-op unless the
        timing profiler is on). The Reporter is thread-pooled, so a module lock guards the dict."""
        if not self._timing_on:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with _SUB_LOCK:
                self._subtimes[name] = self._subtimes.get(name, 0.0) + dt

    def run(self, project, ctx) -> None:
        super().run(project, ctx)
        # LAST step, and the only one that is not per-specimen: one pass over the whole database
        # writing reports/Data/*.csv. It runs here rather than in `infer` because it is a
        # project-level product -- 19 specimens produce ONE leaf_measurements.csv, not 19 of them --
        # and because it must see every specimen's rows already committed.
        with self._phase("data_export"):
            try:
                export_data_csvs(project, self.cfg)
            except Exception:
                # Never fail a completed run over the CSVs: every image has been processed and every
                # measurement is already durable in the DB, which is what the export reads FROM.
                log.exception("data export failed")
        if self._subtimes:                                   # hand the component breakdown to the timer
            ctx.timer.add_subsections(self.key, dict(self._subtimes))

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
          with self._phase("overlay_summary"):
            style = OverlayStyle.from_config(self.cfg)
            summary = build_summary_image(
                read(b.working_path), b.detections, b.leaves, b.cf_px_per_cm, style, _WORKING,
                morphology=b.morphology, landmarks=b.landmarks, landmark_style=lm_style,
                landmark_measurements=b.landmark_measurements,
                petioles=b.petioles, petiole_style=pet_style,
                cf_style=CFScalebarStyle.from_config(self.cfg), cf_source=b.cf_source,
                cf_note=(cf_fallback_reason((b.ruler_lattice or {}).get("image"))
                         if b.cf_source == CF_SOURCE_MP else None),
            )
            path = reports / "Overlay" / "Overlay_Summary" / f"{stem}__Overlay.{img_ext}"
            save_image(summary, path, quality=quality)
            written.append((str(path), "Overlay/Overlay_Summary"))

        # ---- Overlay_Landmarks (one per leaf: keypoints + measurement panel)
        ovlm_cfg = _sub(r, "overlay_landmarks")
        if _flag(ovlm_cfg, "enabled", True) and b.landmarks:
            with self._phase("overlay_landmarks"):
                written += self._export_landmark_overlays(b, reports, stem, img_ext, quality, read, lm_style)

        # ---- Overlay_Petiole (one per leaf with a petiole: masks + width band + panel)
        ovpet_cfg = _sub(r, "overlay_petiole")
        style_overlay = OverlayStyle.from_config(self.cfg)
        if _flag(ovpet_cfg, "enabled", True) and b.petioles:
            with self._phase("overlay_petiole"):
                written += self._export_petiole_overlays(b, reports, stem, img_ext, quality, read, pet_style, style_overlay)

        # ---- Overlay_Specimen_Segmentation (2-panel: annotated sheet | masked cutout) ----
        ovspec_cfg = _sub(r, "overlay_specimen")
        if _flag(ovspec_cfg, "enabled", True) and b.specimen_mask is not None:
          with self._phase("overlay_specimen"):
            loaded = _load_specimen_masks(b.specimen_mask)
            if loaded is not None:
                final_m, removed_m, centers = loaded
                img = build_specimen_overlay(read(b.working_path), final_m, removed_m, centers,
                                             SpecimenStyle.from_config(self.cfg))
                path = reports / "Overlay" / "Overlay_Specimen_Segmentation" / f"{stem}__SpecimenSeg.{img_ext}"
                save_image(img, path, quality=quality)
                written.append((str(path), "Overlay/Overlay_Specimen_Segmentation"))

        # ---- Overlay_Ruler_Lattice (lattice ruler-CF QC panel, redrawn from the stored record) ----
        ovrl_cfg = _sub(r, "overlay_ruler_lattice")
        if _flag(ovrl_cfg, "enabled", True) and b.ruler_lattice:
            with self._phase("overlay_ruler_lattice"):
                written += self._export_ruler_lattice(b, reports)

        # ---- mask exports ------------------------------------------------
        masks_cfg = _sub(r, "masks")
        enabled = {name: _flag(masks_cfg, name, False) for name in _MASK_OUTPUTS}
        if any(enabled.values()) and b.leaves:
            with self._phase("masks_leaf"):
                written += self._export_masks(b, masks_cfg, enabled, reports, stem, img_ext, mask_ext, quality, read)

        # ---- whole-specimen mask exports (SpecimenSegmenter) -> Specimen_Masks/{Binary,RGB}_Masks_Specimen[_Inverse]
        spec_wanted = [e for e in _SPECIMEN_EXPORTS if _flag(masks_cfg, e.key, e.default)]
        if spec_wanted and b.specimen_mask is not None:
            with self._phase("masks_specimen"):
                written += self._export_specimen_masks(b, masks_cfg, spec_wanted, reports, stem,
                                                       img_ext, mask_ext, quality, read)

        # ---- raw RGB crop exports (non-leaf classes; leaf crops -> Leaf_Products) ----
        crops_cfg = _sub(r, "crops")
        if _flag(crops_cfg, "enabled", False):
            with self._phase("crops"):
                written += self._export_crops(b, crops_cfg, reports, stem, img_ext, quality, read)

        # ---- leaf products: Leaf_Original/ + Leaf_Oriented/ trees (5 products each) ----
        lp_cfg = _sub(r, "leaf_products")
        if _flag(lp_cfg, "enabled", True) and b.leaves:
            with self._phase("leaf_products"):
                written += self._export_leaf_products(b, lp_cfg, reports, stem, img_ext, mask_ext, quality, read)

        return written

    def persist(self, project, item: WorkItem, payload: list[tuple[str, str]]) -> None:
        project.db.record_report_manifest(item.specimen_id, payload)

    # ---- mask exports ---------------------------------------------------- #
    def _export_masks(self, b, masks_cfg, enabled, reports, stem, img_ext, mask_ext, quality, read):
        classes = _list(_get(masks_cfg, "classes", default=[_LEAF])) or [_LEAF]
        bg = 0 if str(_get(masks_cfg, "background", default="black")).lower() == "black" else 255
        subtract_holes = bool(_get(masks_cfg, "subtract_holes", default=True))

        need_full = enabled["Binary_Masks_Full_Image"] or enabled["RGB_Masks_Full_Image"]
        need_crop = enabled["Binary_Masks"] or enabled["RGB_Masks"]
        # ONE image for both: full-image and per-crop outputs are now the same frame as the
        # polygons, so the whole-sheet raster needs no scale_polygon pass at all.
        working = read(b.working_path) if (need_full or need_crop) and b.working_path else None

        parsed = _parse_leaves(b.leaves)                       # (detection_id, cls_name, poly) working coords
        written: list[tuple[str, str]] = []

        for cls in classes:
            holes_wanted = subtract_holes and cls == _LEAF
            # Each binary/RGB pair below writes the same class, box and pixels-worth of geometry
            # into two files that differ ONLY in extension -- so each half gets its own prefix.
            seg_label = crop_label(self.cfg, "seg", cls)              # SEG-leaf      (per-crop binary)
            seg_rgb_label = crop_label(self.cfg, "seg_rgb", cls)      # SEGRGB-leaf   (per-crop cutout)
            full_label = crop_label(self.cfg, "mask_full", cls)       # MaskFull-leaf    (full-image binary)
            full_rgb_label = crop_label(self.cfg, "mask_rgb_full", cls)  # MaskRGBFull-leaf (full-image cutout)

            # -- FULL-IMAGE (one file per specimen, working frame) ---------
            #    reports/{Binary,RGB}_Masks/<Binary,RGB>_Masks_Full_Image__<Cls>/<stem>__MaskFull-<friendly>.<ext>
            if need_full and working is not None:
                fg = [p for (_d, cn, p) in parsed if cn == cls]
                holes = [p for (_d, cn, p) in parsed if cn == _HOLE] if holes_wanted else []
                if fg:
                    binary = _binary(working.shape, fg, holes)
                    if enabled["Binary_Masks_Full_Image"]:
                        sub = f"Binary_Masks_Specimen__{cls}"
                        p = reports / _MASKS_ROOT / sub / f"{stem}__{full_label}.{mask_ext}"
                        save_image((binary.astype(np.uint8) * 255), p)
                        written.append((str(p), f"{_MASKS_ROOT}/{sub}"))
                    if enabled["RGB_Masks_Full_Image"]:
                        sub = f"RGB_Masks_Specimen__{cls}"
                        p = reports / _MASKS_ROOT / sub / f"{stem}__{full_rgb_label}.{img_ext}"
                        save_image(composite(working, binary, bg=bg), p, quality=quality)
                        written.append((str(p), f"{_MASKS_ROOT}/{sub}"))

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
                        p = reports / _MASKS_ROOT / sub / name
                        save_image((binary.astype(np.uint8) * 255), p)
                        written.append((str(p), f"{_MASKS_ROOT}/{sub}"))
                    if enabled["RGB_Masks"] and working is not None:
                        crop = working[max(0, y1):y2, max(0, x1):x2]
                        mh, mw = crop.shape[:2]
                        if mh and mw:
                            comp = composite(crop, binary[:mh, :mw], bg=bg)
                            sub = f"RGB_Masks__{cls}"
                            name = crop_filename(stem, seg_rgb_label, (x1, y1, x2, y2), img_ext)
                            p = reports / _MASKS_ROOT / sub / name
                            save_image(comp, p, quality=quality)
                            written.append((str(p), f"{_MASKS_ROOT}/{sub}"))
        return written

    # ---- whole-specimen mask exports (SpecimenSegmenter) ----------------- #
    def _export_specimen_masks(self, b, masks_cfg, wanted, reports, stem, img_ext, mask_ext, quality, read):
        """Export the whole-specimen (plant-vs-background) mask on the WORKING frame, one file per
        specimen per entry in ``wanted`` (a subset of :data:`_SPECIMEN_EXPORTS`).

        Four files are possible, and each ``_Inverse`` one is the pixelwise complement of its
        positive twin -- a binary mask of the sheet minus the plant, and the sheet with the plant
        lifted off it and replaced by ``masks.inverse_fill``. The complement uses its OWN fill
        rather than ``masks.background`` because the two answer different questions: ``background``
        is what surrounds a cutout (black, so it thresholds away), while the inverse fill marks
        where the plant WAS, and wants to be visible against the sheet -- hence white by default.

        The segmenter already stores this mask at working res, which is now also the output frame, so
        it is used verbatim -- the old nearest-neighbour upscale to the original is gone, and with it
        the blocky edges it invented. The complement is still taken after any shape fix-ups, so the
        two masks partition the sheet exactly. An empty final mask (segmentation found no plant)
        writes nothing at all: a full-sheet "inverse" would be technically true and completely
        misleading."""
        import cv2

        loaded = _load_specimen_masks(b.specimen_mask)
        if loaded is None:
            return []
        working = read(b.working_path)
        H, W = working.shape[:2]
        final = loaded[0] > 0
        if final.shape != (H, W):
            # Defensive only: the segmenter writes this mask at working res. A stale PNG from an
            # older run with a different max_working_dim would otherwise broadcast-error here.
            final = cv2.resize(loaded[0].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        if not final.any():
            return []
        bg = 0 if str(_get(masks_cfg, "background", default="black")).lower() == "black" else 255
        # config colors are RGB (like report.overlay's); the images here are OpenCV BGR
        fill_bgr = tuple(reversed(parse_fill_color(_getk(masks_cfg, "inverse_fill", "white"))))

        written: list[tuple[str, str]] = []
        for exp in wanted:
            mask = ~final if exp.invert else final
            kind = "mask_rgb_full" if exp.rgb else "mask_full"
            label = crop_label(self.cfg, kind, exp.cls_name)   # Mask[RGB]Full-specimen[Inverse]
            if exp.rgb:
                p = reports / _MASKS_ROOT / exp.folder / f"{stem}__{label}.{img_ext}"
                save_image(composite(working, mask, bg=(fill_bgr if exp.invert else bg)),
                           p, quality=quality)
            else:
                p = reports / _MASKS_ROOT / exp.folder / f"{stem}__{label}.{mask_ext}"
                save_image((mask.astype(np.uint8) * 255), p)
            written.append((str(p), f"{_MASKS_ROOT}/{exp.folder}"))
        return written

    # ---- lattice ruler-CF QC panel (rebuilt from the stored record) ------ #
    def _export_ruler_lattice(self, b, reports):
        """Render the sheet's lattice ruler-CF QC panel from the STORED ``ruler_CF_lattice`` record
        alone (the engine guarantees the DB + its rasters hold enough to redraw it) into
        ``Overlay/Overlay_Ruler_Lattice/``. One PNG per sheet; the four-tile squarify tile, the
        deskewed strip + tick overlays, the per-unit combs, and the multi-ruler reconciliation are
        all reconstructed by ``RulerCFLattice.render_qc`` -- no live engine output is consulted."""
        from leafmachine3.inference.ruler_lattice import RulerCFLattice

        record = b.ruler_lattice
        if not record or not (record.get("crops") if hasattr(record, "get") else None):
            return []
        try:
            panel = RulerCFLattice(artifact_dir="", write_qc=False).render_qc(record)
        except Exception as exc:                             # a QC failure must never fail the report
            log.warning("ruler lattice QC render failed for specimen %s (%s)", b.specimen_id, exc)
            return []
        if panel is None:
            return []
        p = reports / "Overlay" / "Overlay_Ruler_Lattice" / f"{b.stem}__RulerLattice.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        panel.save(str(p))                                   # render_qc returns a PIL RGB image
        return [(str(p), "Overlay/Overlay_Ruler_Lattice")]

    # ---- raw RGB crop exports (both detectors) --------------------------- #
    def _export_crops(self, b, crops_cfg, reports, stem, img_ext, quality, read):
        """Export the raw RGB bbox crops into ``Crops/RGB__<friendly>/`` for selected classes.

        ``classes`` is ``"all"`` (every archival + plant class) or a list of real class names.

        Pixels are always cut from the WORKING image, so a crop is exactly the pixels the detector
        saw. The old ``source: original`` option is gone: it re-cut the box from the full-res
        original, which produced a crop that no stage had ever looked at and kept the originals a
        hard dependency. A full-res crop is still recoverable outside LM3 -- the box is stored in
        working coords and ``specimen.work_scale`` converts it -- it is just not something the
        Reporter does.
        """
        classes = _get(crops_cfg, "classes", default="all")
        want_all = isinstance(classes, str) and classes.lower() == "all"
        wanted = None if want_all else set(_list(classes))
        # report.crops.source is retired; machine3 warns once per run (core.config.RETIRED_SETTINGS).
        img = read(b.working_path)
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
            box = tuple(float(v) for v in xyxy)          # working coords, working pixels
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
        """Render the 5 leaf products under ``Leaf_Original/`` and (when orientation succeeded)
        ``Leaf_Oriented/``. Masks/cutouts are content-fitted; the bbox crop is not. Uses the working
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
            pet = raster(pet_polys) if pet_polys else None
            laminapet = ((silhouette | pet) & ~hole) if pet is not None else None
            silhouettepet = (silhouette | pet) if pet is not None else None   # holes FILLED, + petiole
            masks = {"lamina": lamina, "laminapet": laminapet, "silhouette": silhouette,
                     "silhouettepet": silhouettepet, "hole": hole}
            det_box = (x1, y1, x2, y2)

            if do_original:
                products = render_leaf_products(crop, masks, angle_cw=None, bg=bg, want=want, pad=pad, hole_color=hole_color)
                written += self._write_leaf_products(products, "Leaf_Original", reports, stem, det_box, img_ext, mask_ext, quality)

            success, angle = orient.get(did, (False, None))
            if do_oriented and success and angle is not None:
                products = render_leaf_products(crop, masks, angle_cw=float(angle), bg=bg, want=want, pad=pad, hole_color=hole_color)
                written += self._write_leaf_products(products, "Leaf_Oriented", reports, stem, det_box, img_ext, mask_ext, quality)
        return written

    def _write_leaf_products(self, products, tree, reports, stem, det_box, img_ext, mask_ext, quality):
        """Save each rendered product to ``reports/<tree>/<folder>/<stem>__<og|or>-<PREFIX>-<friendly>__x_y_x_y.<ext>``.

        The ``og-``/``or-`` tree tag is not decoration: Leaf_Original and Leaf_Oriented render the
        same leaf, at the same detection box, into the same product folders, so the tag is the only
        thing distinguishing the two files once they leave their directories.
        """
        written: list[tuple[str, str]] = []
        for p in PRODUCTS:
            img = products.get(p.key)
            if img is None:
                continue
            ext = mask_ext if p.is_mask else img_ext
            name = crop_filename(stem, leaf_product_label(tree, p.prefix, p.friendly), det_box, ext)
            path = reports / tree / p.folder / name
            save_image(img, path, quality=quality)
            written.append((str(path), f"{tree}/{p.folder}"))
        return written


def _load_specimen_masks(row):
    """Load ``(final_mask, removed_mask, centers)`` from a specimen_mask row, or None if the final
    mask PNG is missing/unreadable. Masks come back as ``uint8`` grayscale (working frame)."""
    import cv2

    mp = _row_get(row, "mask_path", None)
    if not mp:
        return None
    final = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    if final is None:
        return None
    rp = _row_get(row, "refined_path", None)
    removed = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE) if rp else None
    if removed is None or removed.shape != final.shape:
        removed = np.zeros_like(final)
    try:
        centers = json.loads(_row_get(row, "sample_centers_json", None) or "[]")
    except Exception:
        centers = []
    return final, removed, centers


# -- geometry helpers --------------------------------------------------------------
def _group_seg_by_detection(parsed) -> dict:
    """Group parsed leaf polygons ``(detection_id, cls_name, poly)`` -> ``{det: {cls: [poly]}}``."""
    by_det: dict[int, dict[str, list]] = {}
    for did, cls, poly in parsed:
        by_det.setdefault(did, {}).setdefault(cls, []).append(poly)
    return by_det


def _leaf_products_wanted(lp_cfg) -> set:
    """Set of enabled product keys from ``report.leaf_products.products``. Unset -> the ``default_on``
    products; a partial dict -> per-product (each defaults to its own ``default_on``)."""
    pcfg = _getk(lp_cfg, "products", None)
    if pcfg is None:
        return {p.key for p in PRODUCTS if p.default_on}
    return {p.key for p in PRODUCTS if _flag(pcfg, p.key, p.default_on)}


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
