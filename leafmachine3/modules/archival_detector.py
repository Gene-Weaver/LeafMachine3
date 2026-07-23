"""Stage 1 — Archival Detector.

Detect archival sheet elements (Ruler, Barcode, Colorcard, Label, ...) on each specimen's
working image with the exported YOLO26 archival model. Consumes specimen working images and
writes ``archival_detection`` rows, saving one ``__TAG__`` crop per detection.
"""
from __future__ import annotations

import logging

from leafmachine3.core.imaging import read_image, save_crop
from leafmachine3.core.naming import crop_label
from leafmachine3.core.records import DetRow
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_detector

log = logging.getLogger("leafmachine3.archival_detector")


class ArchivalDetector(PipelineStage):
    """Detect archival sheet elements and persist their boxes and crops."""

    key: str = "archival_detector"
    name: str = "Archival Detector"
    depends_on: tuple[str, ...] = ()
    owns_tables: tuple[str, ...] = ("archival_detection",)
    device_kind: str = "cuda"

    def build_model(self, device):
        """Warm-load the archival detector backend once per worker."""
        return load_detector(self.cfg, self.key, device)

    def collect_items(self, project) -> list[WorkItem]:
        """One work item per specimen, carrying a picklable :class:`Unit` payload."""
        return [
            WorkItem(int(row["specimen_id"]), project.unit(row))
            for row in project.db.iter_specimens()
        ]

    def infer(self, item: WorkItem, model) -> list[DetRow]:
        """Run detection and save one crop per box; returns the rows to persist."""
        unit = item.payload
        img = read_image(unit.working_path)
        rows: list[DetRow] = []
        for det in model.predict(img):
            label = crop_label(self.cfg, "bbox", det.cls_name)   # e.g. BBOX-ruler
            crop_path = save_crop(img, det.xyxy, unit.stem, label, unit.crops_dir)
            rows.append(DetRow(det.cls_id, det.cls_name, det.conf, det.xyxy, label, crop_path))
        return rows

    def persist(self, project, item: WorkItem, rows: list[DetRow]) -> None:
        """Delete-then-insert the specimen's archival detection rows."""
        project.db.record_detections("archival_detection", item.specimen_id, rows)
