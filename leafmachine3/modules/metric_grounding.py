"""Stage 7 — Metric Grounding (CPU).

Apply each specimen's conversion factor (px per cm) to the stored pixel measurements:
``area_px -> cm^2`` and ``perimeter_px -> cm``. Both the CF and the pixel measurements live
in the working frame, so ``area_px / cf^2`` is frame-consistent with no rescaling. Consumes
``specimen.cf_px_per_cm`` + ``leaf_segmentation`` and writes the ``*_cm`` columns. The CF is
always ``None`` until the ruler-CF stage is ported, so this is currently a harmless no-op.
"""
from __future__ import annotations

import logging

from leafmachine3.core.records import Grounded
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.metric_grounding")


class MetricGrounding(PipelineStage):
    """Convert per-leaf pixel measurements to physical units using the specimen CF."""

    key: str = "metric_grounding"
    name: str = "Metric Grounding"
    depends_on: tuple[str, ...] = ("ruler_cf", "leaf_segmenter")
    owns_tables: tuple[str, ...] = ()
    device_kind: str = "cpu"

    def collect_items(self, project) -> list[WorkItem]:
        """One item per segmented specimen; payload = (specimen CF, its leaf instance rows)."""
        return [
            WorkItem(sid, (project.db.specimen_cf(sid), project.db.leaf_instances(sid)))
            for sid in project.db.specimens_with_rows("leaf_segmentation")
        ]

    def infer(self, item: WorkItem, model) -> list[Grounded]:
        """Ground each leaf's pixel metrics to cm; empty when no CF is available."""
        cf, leaves = item.payload
        if cf is None:
            return []
        cf = float(cf)
        grounded: list[Grounded] = []
        for leaf in leaves:
            area_px = leaf["area_px"]
            perimeter_px = leaf["perimeter_px"]
            bbox_x1, bbox_y1 = leaf["bbox_x1"], leaf["bbox_y1"]
            bbox_x2, bbox_y2 = leaf["bbox_x2"], leaf["bbox_y2"]
            grounded.append(
                Grounded(
                    leaf_id=int(leaf["leaf_id"]),
                    area_cm2=None if area_px is None else float(area_px) / (cf * cf),
                    perimeter_cm=None if perimeter_px is None else float(perimeter_px) / cf,
                    bbox_w_cm=(
                        None if bbox_x1 is None or bbox_x2 is None
                        else (float(bbox_x2) - float(bbox_x1)) / cf
                    ),
                    bbox_h_cm=(
                        None if bbox_y1 is None or bbox_y2 is None
                        else (float(bbox_y2) - float(bbox_y1)) / cf
                    ),
                )
            )
        return grounded

    def persist(self, project, item: WorkItem, grounded: list[Grounded]) -> None:
        """Write the ``*_cm`` columns for the grounded leaf instances."""
        project.db.set_leaf_metrics_cm(grounded)
