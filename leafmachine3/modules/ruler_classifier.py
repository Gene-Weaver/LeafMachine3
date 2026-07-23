"""Stage 4 — Ruler Classifier.

Classify every ``__R__`` Ruler crop to a measurement unit-type with the 3-model ONNX
majority-vote ensemble. Consumes the ``Ruler`` crops produced by the archival detector and
writes ``ruler_classification`` rows (one per Ruler crop, keyed on its detection id).
"""
from __future__ import annotations

import logging

from leafmachine3.core.records import RulerClassRow
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_ruler_ensemble

log = logging.getLogger("leafmachine3.ruler_classifier")


class RulerClassifier(PipelineStage):
    """Assign a unit-type to each Ruler crop via the classifier ensemble."""

    key: str = "ruler_classifier"
    name: str = "Ruler Classifier"
    depends_on: tuple[str, ...] = ("archival_detector",)
    owns_tables: tuple[str, ...] = ("ruler_classification",)
    device_kind: str = "cuda"

    def build_model(self, device):
        """Warm-load the ruler unit-type ensemble once per worker."""
        return load_ruler_ensemble(self.cfg, device)

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen that owns Ruler crops; payload = those crop refs."""
        return [
            WorkItem(sid, project.db.crops("archival_detection", sid, cls_name="Ruler"))
            for sid in project.db.specimens_with_crops("archival_detection", cls_name="Ruler")
        ]

    def infer(self, item: WorkItem, model) -> list[RulerClassRow]:
        """Classify each Ruler crop; store the raw vote dict for provenance."""
        rows: list[RulerClassRow] = []
        for crop in item.payload:
            out = model.predict(crop.crop_path)
            conf = out.get("conf") if hasattr(out, "get") else None
            rows.append(
                RulerClassRow(
                    detection_id=crop.detection_id,
                    unit_type=out["ensemble"],
                    votes=out,
                    conf=float(conf) if conf is not None else None,
                )
            )
        return rows

    def persist(self, project, item: WorkItem, rows: list[RulerClassRow]) -> None:
        """Upsert the specimen's ruler classifications (keyed on detection id)."""
        project.db.record_ruler_classifications(item.specimen_id, rows)
