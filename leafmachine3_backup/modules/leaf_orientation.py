"""Stage 10 -- Leaf Orientation (CPU post-process).

Determine, per leaf, the clockwise rotation that stands it tip-up / base-down, from the predicted
keypoints (``core.orientation``): primary = the lamina_tip->lamina_base axis; fallback = PCA of the
midvein points (>= ``min_midvein``) with the tip end picked from the petiole / base / apex points.
The angle + a success flag are written onto that leaf's ``leaf_morphology`` row
(``oriented_leaf_rotation_angle_degreesCW`` / ``oriented_leaf_success``); the Reporter uses them to
emit the ``Oriented`` leaf-product tree. Leaves with no usable landmarks get success = 0 and no
oriented output.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from leafmachine3.core.orientation import compute_orientation, length_width_from_box
from leafmachine3.core.records import OrientationRow
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.leaf_orientation")

_DEFAULT_MIN_KPT_CONF = 0.25
_DEFAULT_MIN_MIDVEIN = 5


class LeafOrientation(PipelineStage):
    """Compute each leaf's upright rotation and store it on the morphology row."""

    key: str = "leaf_orientation"
    name: str = "Leaf Orientation"
    device_kind: str = "cpu"
    depends_on: tuple[str, ...] = ("morphology", "landmark_detector")
    owns_tables: tuple[str, ...] = ()          # updates leaf_morphology cols (see _OWNED_MORPH_COLS)

    def _min_kpt_conf(self) -> float:
        return float(_get(self.cfg.stage(self.key), "min_kpt_conf", default=_DEFAULT_MIN_KPT_CONF))

    def _min_midvein(self) -> int:
        return int(_get(self.cfg.stage(self.key), "min_midvein", default=_DEFAULT_MIN_MIDVEIN))

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen with morphology; payload = (leaf_id/detection pairs, points-by-det)."""
        thr = self._min_kpt_conf()
        items: list[WorkItem] = []
        for sid in project.db.specimens_with_rows("leaf_segmentation"):
            morph = project.db.leaf_morphology(sid)
            if not morph:
                continue
            points_by_det: dict[int, dict[str, tuple[float, float]]] = {}
            for r in project.db.leaf_landmarks(sid):
                if int(r["instance_index"]) != 0:          # one leaf per crop; use the primary instance
                    continue
                conf = r["conf"]
                if conf is not None and float(conf) >= thr and r["x"] is not None and r["y"] is not None:
                    points_by_det.setdefault(int(r["detection_id"]), {})[str(r["kpt_name"])] = (
                        float(r["x"]), float(r["y"]),
                    )
            # carry each leaf's rotated-bbox corners so the same angle can split its two side
            # lengths into tip->base length vs perpendicular width (see length_width_from_box)
            morph_ids = [(int(m["leaf_id"]), int(m["detection_id"]), m["rotated_bbox_json"])
                         for m in morph]
            items.append(WorkItem(sid, (morph_ids, points_by_det)))
        return items

    def infer(self, item: WorkItem, model: Any) -> list[OrientationRow]:
        morph_ids, points_by_det = item.payload
        min_mv = self._min_midvein()
        rows: list[OrientationRow] = []
        for leaf_id, det_id, bbox_json in morph_ids:
            o = compute_orientation(points_by_det.get(det_id, {}), min_midvein=min_mv)
            length = width = None
            if o.success:
                try:
                    corners = json.loads(bbox_json) if bbox_json else None
                except (TypeError, ValueError):
                    corners = None
                length, width = length_width_from_box(corners, o.angle_cw)
            rows.append(OrientationRow(leaf_id=leaf_id, success=o.success, angle_cw=o.angle_cw,
                                       bbox_length=length, bbox_width=width))
        return rows

    def persist(self, project, item: WorkItem, rows: list[OrientationRow]) -> None:
        project.db.set_leaf_orientation(rows)


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
