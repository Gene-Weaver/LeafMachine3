"""Shared detection-inference glue for the ArchivalDetector and PlantDetector stages.

Both detectors run the identical pipeline: read the working image, ``model.predict``, suppress
same-class duplicate boxes (:mod:`leafmachine3.core.box_dedup`, because YOLO26 is NMS-free), save a
crop for every KEEPER, and build one :class:`DetRow` per box (keepers + suppressed, so the DB keeps
the suppression as provenance). Suppressed boxes get no crop and are filtered out of all downstream
reads by the DB layer. Kept here so the dedup wiring lives in exactly one place.
"""
from __future__ import annotations

from leafmachine3.core.box_dedup import suppress_duplicate_boxes
from leafmachine3.core.imaging import read_image, save_crop
from leafmachine3.core.naming import crop_label
from leafmachine3.core.records import DetRow


def _dedup_settings(cfg, stage_key) -> tuple[bool, float, str]:
    """Read ``modules.<stage_key>.dedup {enabled, overlap, metric}`` (defaults: on / 0.95 / 'min')."""
    blk = cfg.stage(stage_key).get("dedup")
    get = blk.get if hasattr(blk, "get") else (lambda k, d: d)
    return bool(get("enabled", True)), float(get("overlap", 0.95)), str(get("metric", "min"))


def run_detection(cfg, stage_key: str, unit, model) -> list[DetRow]:
    """Detect on one specimen's working image, dedupe same-class boxes, and return rows to persist."""
    img = read_image(unit.working_path)
    dets = model.predict(img)
    enabled, overlap, metric = _dedup_settings(cfg, stage_key)
    decisions = suppress_duplicate_boxes(dets, overlap=overlap, metric=metric) if enabled else None

    rows: list[DetRow] = []
    for i, det in enumerate(dets):
        label = crop_label(cfg, "bbox", det.cls_name)          # e.g. BBOX-ruler
        dec = decisions[i] if decisions is not None else None
        if dec is not None and dec.suppressed:
            rows.append(DetRow(det.cls_id, det.cls_name, det.conf, det.xyxy, label, crop_path=None,
                               suppressed=True, suppressed_by_index=dec.keeper_index,
                               suppress_overlap=dec.overlap))
        else:
            crop_path = save_crop(img, det.xyxy, unit.stem, label, unit.crops_dir)
            rows.append(DetRow(det.cls_id, det.cls_name, det.conf, det.xyxy, label, crop_path))
    return rows
