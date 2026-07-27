"""Stage 9 -- Landmark Measurements.

A pure post-process (no model, CPU) that turns the predicted keypoints in ``leaf_landmark`` into
the derived per-leaf measurements Will asked for: lamina trace length, lamina extent, leaf width,
apex / base angles (+ acute/obtuse/reflex type), petiole trace length, and lamina curvature.
Results go to ``leaf_landmark_measurement`` (one row per leaf instance), linked to the leaf crop.

Occlusion-robust by construction: keypoints below ``min_kpt_conf`` are dropped before measuring,
and :func:`core.landmark_metrics.compute_measurements` returns ``None`` for any metric whose inputs
are absent -- nothing is fabricated. See ``core/landmark_metrics.py`` (and the confirmed
``modules/experiments/angle_checks.html`` schematic) for the exact definitions.
"""
from __future__ import annotations

import logging
from typing import Any

from leafmachine3.core.landmark_metrics import compute_measurements
from leafmachine3.core.records import LandmarkMeasureRow
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.landmark_measurements")

_DEFAULT_MIN_KPT_CONF = 0.25


class LandmarkMeasurements(PipelineStage):
    """Compute derived measurements from stored keypoints; robust to occluded/missing points."""

    key: str = "landmark_measurements"
    name: str = "Landmark Measurements"
    device_kind: str = "cpu"
    depends_on: tuple[str, ...] = ("landmark_detector",)
    owns_tables: tuple[str, ...] = ("leaf_landmark_measurement",)

    def _min_kpt_conf(self) -> float:
        return float(_get(self.cfg.stage(self.key), "min_kpt_conf", default=_DEFAULT_MIN_KPT_CONF))

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen with keypoints; payload = confident points per leaf instance.

        Payload: ``{(detection_id, instance_index): {kpt_name: (x, y)}}`` in working coords, already
        filtered to keypoints with ``conf >= min_kpt_conf`` (the "present" landmarks).
        """
        thr = self._min_kpt_conf()
        items: list[WorkItem] = []
        for sid in project.db.specimens_with_landmarks():
            groups: dict[tuple[int, int], dict[str, tuple[float, float]]] = {}
            for r in project.db.leaf_landmarks(sid):
                key = (int(r["detection_id"]), int(r["instance_index"]))
                pts = groups.setdefault(key, {})          # keep the instance even if all pts are weak
                conf = r["conf"]
                if conf is not None and float(conf) >= thr and r["x"] is not None and r["y"] is not None:
                    pts[str(r["kpt_name"])] = (float(r["x"]), float(r["y"]))
            items.append(WorkItem(sid, groups))
        return items

    def infer(self, item: WorkItem, model: Any) -> list[LandmarkMeasureRow]:
        rows: list[LandmarkMeasureRow] = []
        for (det_id, inst_index), points in item.payload.items():
            m = compute_measurements(points)
            cx, cy = (m.lamina_centroid if m.lamina_centroid is not None else (None, None))
            rows.append(LandmarkMeasureRow(
                detection_id=det_id,
                instance_index=inst_index,
                lamina_trace_length=m.lamina_trace_length,
                lamina_extent=m.lamina_extent,
                leaf_width=m.leaf_width,
                apex_angle=m.apex_angle,
                apex_angle_type=m.apex_angle_type,
                base_angle=m.base_angle,
                base_angle_type=m.base_angle_type,
                petiole_trace_length=m.petiole_trace_length,
                lamina_curvature=m.lamina_curvature,
                lamina_centroid_x=cx,
                lamina_centroid_y=cy,
                n_present=m.n_present,
            ))
        return rows

    def persist(self, project, item: WorkItem, rows: list[LandmarkMeasureRow]) -> None:
        project.db.record_leaf_landmark_measurements(item.specimen_id, rows)


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
