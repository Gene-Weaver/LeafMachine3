"""Project — a thin run context bundling the config, dirs, and ProjectDB.

Stages access persistence via ``project.db`` (the single writer). A few convenience
helpers build the picklable per-specimen payloads and the Reporter bundle from DB reads.
"""
from __future__ import annotations

from leafmachine3.core.records import ReportBundle, Unit


class Project:
    def __init__(self, cfg, dirs, db) -> None:
        self.cfg = cfg
        self.dirs = dirs
        self.db = db

    def unit(self, row) -> Unit:
        """Build the detector-stage payload (a picklable Unit) from a specimen row."""
        return Unit(
            specimen_id=int(_g(row, "specimen_id")),
            stem=str(_g(row, "image_stem")),
            working_path=str(_g(row, "working_path")),
            original_path=str(_g(row, "original_path")),
            crops_dir=str(self.dirs.crops),
            width=int(_g(row, "width", 0) or 0),
            height=int(_g(row, "height", 0) or 0),
        )

    def working_image(self, specimen_id: int) -> str:
        return str(_g(self.db.get_specimen(specimen_id), "working_path"))

    def report_bundle(self, specimen_id: int) -> ReportBundle:
        s = self.db.get_specimen(specimen_id)
        return ReportBundle(
            specimen_id=specimen_id,
            stem=str(_g(s, "image_stem")),
            original_path=str(_g(s, "original_path")),
            work_scale=float(_g(s, "work_scale", 1.0) or 1.0),
            cf_px_per_cm=_g(s, "cf_px_per_cm", None),
            detections=self.db.overlay_detections(specimen_id),
            leaves=self.db.leaf_instances(specimen_id),
            reports_dir=str(self.dirs.reports),
            working_path=str(_g(s, "working_path")),
            crop_boxes=self.db.detection_boxes(specimen_id, "plant_detection"),
            morphology=self.db.leaf_morphology(specimen_id),
        )


def _g(row, key, default=None):
    """Read a column/attr from a sqlite Row, dataclass, or dict (tolerant)."""
    if row is None:
        return default
    if hasattr(row, key):
        v = getattr(row, key)
        return default if v is None else v
    try:
        v = row[key]
        return default if v is None else v
    except Exception:
        return default
