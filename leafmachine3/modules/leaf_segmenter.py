"""Stage 6 — Leaf Segmenter.

Segment each leaf-box crop into instances (Leaf / Petiole / Hole) with the exported
YOLO26-seg model. Polygons come back in crop coordinates and are re-based to the parent
(working) frame so the Reporter can reconstruct any view. Hole / Petiole instances carry a
``parent_instance_index`` pointing at their owning Leaf instance. Consumes the plant
detector's leaf crops and writes ``leaf_segmentation`` rows.
"""
from __future__ import annotations

import logging

import numpy as np

from leafmachine3.core.imaging import (
    encode_polygon,
    offset_polygon,
    polygon_area,
    polygon_bbox,
    polygon_centroid,
    polygon_perimeter,
)
from leafmachine3.core.records import LeafRow
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_segmenter

log = logging.getLogger("leafmachine3.leaf_segmenter")


def _nearest_leaf(poly: np.ndarray, leaves: list[tuple[int, np.ndarray]]) -> int | None:
    """Return the instance index of the Leaf whose centroid is closest to ``poly``.

    Used to attach a Hole / Petiole to its owning Leaf. Returns ``None`` when no Leaf
    instance has been seen yet in the crop.
    """
    if not leaves:
        return None
    cx, cy = polygon_centroid(poly)
    best_index, best_dist = leaves[0][0], float("inf")
    for index, leaf_poly in leaves:
        lx, ly = polygon_centroid(leaf_poly)
        dist = (lx - cx) ** 2 + (ly - cy) ** 2
        if dist < best_dist:
            best_index, best_dist = index, dist
    return best_index


class LeafSegmenter(PipelineStage):
    """Segment leaf crops into instances re-based to the parent working frame."""

    key: str = "leaf_segmenter"
    name: str = "Leaf Segmenter"
    depends_on: tuple[str, ...] = ("plant_detector",)
    owns_tables: tuple[str, ...] = ("leaf_segmentation",)
    device_kind: str = "cuda"

    def build_model(self, device):
        """Warm-load the leaf instance segmenter once per worker."""
        return load_segmenter(self.cfg, device)

    def _classes(self) -> tuple[str, ...]:
        """Leaf detection classes to segment (optionally including partial leaves)."""
        if bool(self.cfg.stage(self.key).get("include_partial", False)):
            return ("Leaf_WHOLE", "Leaf_PARTIAL")
        return ("Leaf_WHOLE",)

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen with leaf crops; payload = those crop refs."""
        cls_in = self._classes()
        return [
            WorkItem(sid, project.db.crops("plant_detection", sid, cls_in=cls_in))
            for sid in project.db.specimens_with_crops("plant_detection", cls_in=cls_in)
        ]

    def infer(self, item: WorkItem, model) -> list[LeafRow]:
        """Segment each crop and re-base every polygon to parent (working) coords."""
        rows: list[LeafRow] = []
        for crop in item.payload:
            leaves: list[tuple[int, np.ndarray]] = []
            for i, inst in enumerate(model.predict(crop.crop_path)):
                poly = offset_polygon(inst.polygon, crop.x1, crop.y1)
                if inst.cls_name == "Leaf":
                    owner = i
                    leaves.append((i, poly))
                else:
                    owner = _nearest_leaf(poly, leaves)
                rows.append(
                    LeafRow(
                        detection_id=crop.detection_id,
                        instance_index=i,
                        cls_id=inst.cls_id,
                        cls_name=inst.cls_name,
                        conf=inst.conf,
                        parent_instance_index=owner,
                        mask_format="polygon_xy",
                        mask_data=encode_polygon(poly),
                        frame_width=crop.frame_width,
                        frame_height=crop.frame_height,
                        bbox=polygon_bbox(poly),
                        num_parts=1,
                        area_px=polygon_area(poly),
                        perimeter_px=polygon_perimeter(poly),
                    )
                )
        return rows

    def persist(self, project, item: WorkItem, rows: list[LeafRow]) -> None:
        """Upsert the specimen's leaf instances (keyed on detection id + instance index)."""
        project.db.record_leaf_instances(item.specimen_id, rows)
