"""Stage 0 — Megapixel Conversion Factor (runs FIRST, before the detectors).

Predicts a pixel->metric conversion factor (pixels per cm) for each specimen from nothing but the
ORIGINAL image resolution, via a one-feature linear model
``cf_px_per_cm = slope * megapixels + intercept`` (``megapixels = original_width*original_height/1e6``).
The fitted coefficients ship in ``models/mp_conversion_factor/model.json`` (see
``models/mp_conversion_factor/fit_mp_cf.py`` + ``fit_data.csv``).

This is the FIRST entry in ``STAGE_ORDER`` so ``specimen.cf_px_per_cm_predicted_by_mp`` is populated
before any other pipeline component runs. It needs only the specimen dimensions (available straight
after ingest), so it has no ``depends_on``. CPU-only; no GPU, no image decode.
"""
from __future__ import annotations

import logging

from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_mp_conversion_factor

log = logging.getLogger("leafmachine3.mp_conversion_factor")


class MPConversionFactor(PipelineStage):
    """Resolution-based CF predictor -> ``specimen.cf_px_per_cm_predicted_by_mp`` (float, 2 dp)."""

    key: str = "mp_conversion_factor"
    name: str = "MP Conversion Factor"
    depends_on: tuple[str, ...] = ()
    owns_tables: tuple[str, ...] = ()          # owns specimen.cf_px_per_cm_predicted_by_mp (nulled on reset)
    device_kind: str = "cpu"

    def build_model(self, device):
        """Load the fitted linear coefficients (model.json) once per worker."""
        return load_mp_conversion_factor(self.cfg, device)

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen; payload = the (width, height) THIS MODEL FORM is defined on.

        The linear form was fit on original resolutions and must be evaluated there. The sqrt form
        is homogeneous in the resize factor, so evaluating it on the WORKING dims yields the
        working-frame answer directly -- which is the frame every measurement in LM3 is made in,
        and it removes the last reason the pipeline needs the original image's dimensions.
        """
        model = self.build_model(None)
        working = model.frame == "working"
        rows = []
        for r in project.db.iter_specimens():
            if working:
                w, h = int(r["width"] or 0), int(r["height"] or 0)
            else:
                w, h = int(r["original_width"] or 0), int(r["original_height"] or 0)
            rows.append(WorkItem(int(r["specimen_id"]), (w, h)))
        return rows

    def infer(self, item: WorkItem, model):
        """Return ``(megapixels, predicted CF)`` for the image ((None, None) for a degenerate size)."""
        width, height = item.payload
        return (model.megapixels(width, height), model.predict_cf(width, height))

    def persist(self, project, item: WorkItem, payload) -> None:
        """Write the megapixels + predicted CF onto the specimen row."""
        mp, cf = payload
        project.db.set_specimen_mp_cf(item.specimen_id, cf, mp)
