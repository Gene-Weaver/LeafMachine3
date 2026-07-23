"""Stage 3 — Phenology Detector (CPU).

Roll the plant detections up to per-specimen presence of leaves / flowers / fruits. This is
a pure aggregation over already-stored ``plant_detection`` rows, so it runs on the CPU with
no model. Writes a ``phenology`` row and mirrors the ``has_*`` flags onto the specimen.
"""
from __future__ import annotations

import logging

from leafmachine3.core.records import PhenologyResult
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.phenology_detector")


class PhenologyDetector(PipelineStage):
    """Aggregate plant-organ detections into leaf / flower / fruit presence."""

    key: str = "phenology_detector"
    name: str = "Phenology Detector"
    depends_on: tuple[str, ...] = ("plant_detector",)
    owns_tables: tuple[str, ...] = ("phenology",)
    device_kind: str = "cpu"

    #: which plant classes count toward each phenological organ group.
    ROLLUP = {
        "leaves": ("Leaf_WHOLE", "Leaf_PARTIAL"),
        "flowers": ("Flower_ONE", "Flower_MANY", "Bud"),
        "fruits": ("Seed_Fruit_ONE", "Seed_Fruit_MANY"),
    }

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen that has plant detections; payload = those detection rows."""
        return [
            WorkItem(sid, project.db.detections("plant_detection", sid))
            for sid in project.db.specimens_with_rows("plant_detection")
        ]

    def infer(self, item: WorkItem, model) -> PhenologyResult:
        """Threshold each organ group by min confidence and min count."""
        targets = self.cfg.stage(self.key).targets
        result: dict[str, tuple[bool, int]] = {}
        for organ, classes in self.ROLLUP.items():
            spec = targets[organ]
            min_conf = float(spec["min_conf"])
            min_count = int(spec["min_count"])
            hits = [
                d for d in item.payload
                if d["cls_name"] in classes and float(d["conf"]) >= min_conf
            ]
            result[organ] = (len(hits) >= min_count, len(hits))
        return PhenologyResult(
            leaves=result["leaves"],
            flowers=result["flowers"],
            fruits=result["fruits"],
        )

    def persist(self, project, item: WorkItem, result: PhenologyResult) -> None:
        """Write the phenology row and the specimen ``has_*`` flags."""
        project.db.record_phenology(item.specimen_id, result)
