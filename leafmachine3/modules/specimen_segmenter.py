"""Stage 2b — Specimen Segmenter.

Segment the whole sheet into a plant-vs-background (specimen) binary mask with the exported
UNet++ ONNX model, then apply the informed HSV **paperclean** follow-up step. Consumes each
specimen's working image and writes one ``specimen_mask`` row (final mask PNG + the paperclean
removed-region PNG + the paper-sampling box centers) per specimen. The mask is stored in the
WORKING frame; the Reporter renders it on the original (scaled by ``1/work_scale``) as the
``Overlay_Specimen_Segmentation`` view.

Independent of the plant boxes (a boxless full-sheet segmenter), so it has no ``depends_on``;
it is sequenced right after the Plant Detector purely by ``STAGE_ORDER`` position.
"""
from __future__ import annotations

import logging

from leafmachine3.core.imaging import read_image
from leafmachine3.core.records import SpecimenMaskResult
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_specimen_segmenter

log = logging.getLogger("leafmachine3.specimen_segmenter")


class SpecimenSegmenter(PipelineStage):
    """Whole-specimen segmentation (UNet++ ONNX + paperclean) -> one mask per specimen."""

    key: str = "specimen_segmenter"
    name: str = "Specimen Segmenter"
    depends_on: tuple[str, ...] = ()
    owns_tables: tuple[str, ...] = ("specimen_mask",)
    device_kind: str = "cuda"

    def build_model(self, device):
        """Warm-load the specimen segmenter backend once per worker."""
        return load_specimen_segmenter(self.cfg, device)

    def collect_items(self, project) -> list[WorkItem]:
        """One work item per specimen, carrying a picklable :class:`Unit` payload."""
        return [
            WorkItem(int(row["specimen_id"]), project.unit(row))
            for row in project.db.iter_specimens()
        ]

    def infer(self, item: WorkItem, model) -> SpecimenMaskResult:
        """Run the segmenter + paperclean on the working image; returns the mask payload."""
        img = read_image(item.payload.working_path)
        return model.predict(img)

    def persist(self, project, item: WorkItem, result: SpecimenMaskResult) -> None:
        """Write the final + removed mask PNGs and upsert the specimen_mask row."""
        project.db.record_specimen_mask(item.specimen_id, item.payload.stem, result, project.dirs)
