"""Stage 7 — Metric Grounding (CPU).

Apply each specimen's conversion factor (px per cm) to the stored pixel measurements:

* ``leaf_segmentation`` -- ``area_px -> cm^2``, ``perimeter_px -> cm``, bbox sides -> cm. This is
  the ONE home for a leaf's cm area/perimeter; ``leaf_morphology`` deliberately has no cm columns.
* ``leaf_petiole``      -- ``width_px`` / ``length_px -> cm``.
* ``leaf_landmark_measurement`` -- the five LENGTH metrics (lamina trace, lamina extent, tip-base,
  leaf width, petiole trace) -> cm. Angles and curvature are already unit-free degrees, so they get
  no cm twin; a NULL px metric (occluded keypoint) stays NULL rather than becoming a fabricated 0.

Both the CF and the pixel measurements live in the working frame, so ``area_px / cf^2`` is
frame-consistent with no rescaling.

The CF comes from ``specimen.cf_px_per_cm``, written by the ruler-CF stage: the lattice CF for a
high-confidence sheet (``cf_source = 'measured_from_ruler'``), or -- only when
``modules.ruler_cf.use_CF_predicted_by_MP`` is on -- the megapixel prediction for a sheet with no
ruler or a lattice that did not pass (``cf_source = 'predicted_from_megapixels'``). This stage
grounds against whichever is there; a specimen with no CF is simply skipped (its ``*_cm`` columns
stay NULL). ``specimen.cf_source`` is what tells a consumer which kind of cm value it holds.
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
    depends_on: tuple[str, ...] = ("ruler_cf", "leaf_segmenter", "petiole_width",
                                   "landmark_measurements")
    owns_tables: tuple[str, ...] = ()
    device_kind: str = "cpu"

    def collect_items(self, project) -> list[WorkItem]:
        """One item per segmented specimen; payload = (CF, leaf rows, petiole rows, landmark rows)."""
        return [
            WorkItem(sid, (project.db.specimen_cf(sid), project.db.leaf_instances(sid),
                           project.db.leaf_petioles(sid),
                           project.db.leaf_landmark_measurements(sid)))
            for sid in project.db.specimens_with_rows("leaf_segmentation")
        ]

    def infer(self, item: WorkItem, model) -> tuple[list[Grounded], list[tuple], list[tuple]]:
        """Ground each leaf's, petiole's and landmark row's pixel metrics to cm.

        Returns three empty lists when the sheet has no CF -- every ``*_cm`` column stays NULL
        (see the module docstring for when the MP prediction is used instead).
        """
        cf, leaves, petioles, landmarks = item.payload
        if cf is None:
            return [], [], []
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

        # petiole width/length are plain lengths in the same working frame -> divide by the CF
        pet: list[tuple] = []
        for p in petioles:
            width_px, length_px = p["width_px"], p["length_px"]
            pet.append((
                int(p["leaf_id"]),
                None if width_px is None else float(width_px) / cf,
                None if length_px is None else float(length_px) / cf,
            ))
        # landmark LENGTHS are plain working-frame lengths too -> divide by the CF. A NULL px value
        # (an occluded keypoint) must stay NULL: `_cm(None)` is None, never 0.0.
        def _cm(v):
            return None if v is None else float(v) / cf

        lm: list[tuple] = []
        for m in landmarks:
            lm.append((
                int(m["measure_id"]),
                _cm(m["lamina_trace_length"]),
                _cm(m["lamina_extent"]),
                _cm(m["lamina_tip_base_length"]),
                _cm(m["leaf_width"]),
                _cm(m["petiole_trace_length"]),
            ))
        return grounded, pet, lm

    def persist(self, project, item: WorkItem, payload) -> None:
        """Write the ``*_cm`` columns for the grounded leaves, petioles and landmark measurements."""
        grounded, pet, lm = payload
        project.db.set_leaf_metrics_cm(grounded)
        if pet:
            project.db.set_petiole_metrics_cm(pet)
        if lm:
            project.db.set_landmark_metrics_cm(lm)
