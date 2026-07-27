"""Stage 8 — Landmark Detector.

Run the yolo26x-pose model on each leaf crop to predict the 31-keypoint mid15_pet5 skeleton
(lamina tip/base, apex/base triples, midvein × 15, petiole × 5 + tip, width × 2). The model is
trained on white-padded crops; the inference wrapper re-adds that border and maps keypoints back
to the crop frame (as if never padded). Each keypoint is then re-based to the parent (working)
frame and written to ``leaf_landmark``. Consumes the plant detector's leaf crops.

Keypoint positions feed the future ``landmark_measurements`` step (traces, lengths, apex/base
angles) and the leaf-orientation code (which will assign the rotated-bbox length/width — see
TODO #1a). Structure/relationships live in ``core.landmarks`` and are seeded into the DB.
"""
from __future__ import annotations

import logging

from leafmachine3.core.landmarks import KPT_INDEX
from leafmachine3.core.records import LandmarkRow
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_landmark_pose

log = logging.getLogger("leafmachine3.landmark_detector")


class LandmarkDetector(PipelineStage):
    """Predict per-leaf keypoints and re-base them to the parent working frame."""

    key: str = "landmark_detector"
    name: str = "Landmark Detector"
    depends_on: tuple[str, ...] = ("plant_detector",)
    owns_tables: tuple[str, ...] = ("leaf_landmark",)
    device_kind: str = "cuda"

    def build_model(self, device):
        """Warm-load the pose model once per worker."""
        return load_landmark_pose(self.cfg, device)

    def _classes(self) -> tuple[str, ...]:
        """Leaf detection classes to landmark (optionally including partial leaves)."""
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

    def infer(self, item: WorkItem, model) -> list[LandmarkRow]:
        """Predict keypoints per crop; re-base each to parent (working) coords."""
        rows: list[LandmarkRow] = []
        for crop in item.payload:
            for inst_index, leaf in enumerate(model.predict(crop.crop_path)):
                for name, (xc, yc, conf) in leaf.items():
                    idx = KPT_INDEX.get(name)
                    if idx is None:
                        continue
                    rows.append(
                        LandmarkRow(
                            detection_id=crop.detection_id,
                            instance_index=inst_index,
                            kpt_index=idx,
                            kpt_name=name,
                            x=float(xc) + crop.x1,        # crop frame -> working (parent) frame
                            y=float(yc) + crop.y1,
                            x_crop=float(xc),
                            y_crop=float(yc),
                            conf=float(conf),
                        )
                    )
        return rows

    def persist(self, project, item: WorkItem, rows: list[LandmarkRow]) -> None:
        """Delete-then-insert the specimen's landmark rows."""
        project.db.record_leaf_landmarks(item.specimen_id, rows)
