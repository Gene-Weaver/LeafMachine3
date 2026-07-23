"""Stage 2 — Plant Detector.

Detect plant organs (Leaf_WHOLE, Leaf_PARTIAL, Flower, Bud, ...) on each specimen's working
image with the exported YOLO26 plant model. Shares the detector interface with the archival
stage; the ``__LW__`` / ``__LP__`` leaf crops feed the Leaf Segmenter downstream. Consumes
specimen working images and writes ``plant_detection`` rows.
"""
from __future__ import annotations

import logging

from leafmachine3.core.imaging import read_image, save_crop
from leafmachine3.core.naming import crop_label
from leafmachine3.core.records import DetRow
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_detector

log = logging.getLogger("leafmachine3.plant_detector")


class PlantDetector(PipelineStage):
    """Detect plant organs and persist their boxes and crops."""

    key: str = "plant_detector"
    name: str = "Plant Detector"
    depends_on: tuple[str, ...] = ()
    owns_tables: tuple[str, ...] = ("plant_detection",)
    device_kind: str = "cuda"

    def build_model(self, device):
        """Warm-load the plant detector backend once per worker."""
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
            label = crop_label(self.cfg, "bbox", det.cls_name)   # e.g. BBOX-leaf
            crop_path = save_crop(img, det.xyxy, unit.stem, label, unit.crops_dir)
            rows.append(DetRow(det.cls_id, det.cls_name, det.conf, det.xyxy, label, crop_path))
        return rows

    def persist(self, project, item: WorkItem, rows: list[DetRow]) -> None:
        """Delete-then-insert the specimen's plant detection rows."""
        project.db.record_detections("plant_detection", item.specimen_id, rows)
