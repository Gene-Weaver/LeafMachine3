"""Morphology — runs right after LeafSegmenter.

Computes LeafMachine2-style shape metrics for each leaf-instance mask (area, perimeter,
centroid, convex hull, convexity/concavity, circularity, aspect ratio, vertex count) plus
the rotated (minimum) bounding box (rotation angle + long/short = leaf length/width, per
LM2's ``fit_min_bbox``). Results go to the ``leaf_morphology`` table, one row per instance,
with the specimen / detection / leaf ids and the crop box so every row links back to its
parent image.

Which classes are measured is controlled by ``modules.morphology.classes`` (default
``[Leaf]``); the code already handles Petiole/Hole when they are added to that list.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from leafmachine3.core.imaging import decode_polygon
from leafmachine3.core.morphometrics import polygon_morphology
from leafmachine3.core.records import MorphRow
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.morphology")


class Morphology(PipelineStage):
    """Measure per-leaf morphology + the rotated bounding box from stored masks."""

    key: str = "morphology"
    name: str = "Morphology"
    device_kind: str = "cpu"
    depends_on: tuple[str, ...] = ("leaf_segmenter",)
    owns_tables: tuple[str, ...] = ("leaf_morphology",)

    def _classes(self) -> set[str]:
        cfg = self.cfg.stage(self.key)
        want = _get(cfg, "classes", default=["Leaf"]) or ["Leaf"]
        return {str(c) for c in want}

    def _find_min_bbox(self) -> bool:
        return bool(_get(self.cfg.stage(self.key), "find_minimum_bounding_box", default=True))

    def collect_items(self, project) -> list[WorkItem]:
        return [
            WorkItem(sid, (project.db.leaf_instances(sid),
                           project.db.detection_boxes(sid, "plant_detection")))
            for sid in project.db.specimens_with_rows("leaf_segmentation")
        ]

    def infer(self, item: WorkItem, model: Any) -> list[MorphRow]:
        leaves, crop_boxes = item.payload
        classes = self._classes()
        find_min = self._find_min_bbox()
        rows: list[MorphRow] = []
        for r in leaves:
            cls_name = str(_row_get(r, "cls_name", ""))
            if cls_name not in classes:
                continue
            if str(_row_get(r, "mask_format", "polygon_xy")) != "polygon_xy":
                continue
            data = _row_get(r, "mask_data")
            if not data:
                continue
            try:
                poly = decode_polygon(str(data))
            except Exception:
                continue
            m = polygon_morphology(poly, find_min_bbox=find_min)
            if m is None:
                continue
            did = int(_row_get(r, "detection_id", -1))
            rows.append(MorphRow(
                leaf_id=int(_row_get(r, "leaf_id", -1)),
                detection_id=did,
                instance_index=int(_row_get(r, "instance_index", 0)),
                cls_name=cls_name,
                crop_box=tuple(crop_boxes.get(did, (0.0, 0.0, 0.0, 0.0))),
                area_px=m.area_px, perimeter_px=m.perimeter_px, centroid=m.centroid,
                convex_hull_area=m.convex_hull_area, convexity=m.convexity,
                concavity=m.concavity, circularity=m.circularity,
                aspect_ratio=m.aspect_ratio, n_vertices=m.n_vertices, bbox=m.bbox,
                rotate_angle=m.rotate_angle, dim_max=m.dim_max, dim_min=m.dim_min,
                rotated_bbox_json=json.dumps(m.rotated_bbox), circle=m.circle,
            ))
        return rows

    def persist(self, project, item: WorkItem, rows: list[MorphRow]) -> None:
        project.db.record_leaf_morphology(item.specimen_id, rows)


# -- tolerant accessors ------------------------------------------------------------
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


def _get(obj: Any, key: str, default: Any = None) -> Any:
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
