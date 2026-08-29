"""Stage 4 — Ruler Classifier.

Classify every ``__R__`` Ruler crop to a measurement unit-type with the 3-model ONNX
majority-vote ensemble. Consumes the ``Ruler`` crops produced by the archival detector and
writes ``ruler_classification`` rows (one per Ruler crop, keyed on its detection id).
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from leafmachine3.core.records import RulerClassRow
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.inference import load_ruler_ensemble

log = logging.getLogger("leafmachine3.ruler_classifier")

# QC-thumbnail size/quality of the pre-made four-tile collage. MUST match the RulerCFLattice
# engine defaults (tile_store_px / tile_quality), which get recorded in ruler_CF_lattice
# engine_params_json for provenance -- keep the two in sync if you change either.
_TILE_STORE_PX = 512
_TILE_QUALITY = 80


def _save_squarify_tile(model, crop, squarify_dir: Optional[str]) -> Optional[str]:
    """Persist the four-tile collage the ensemble saw to ``_ruler_squarify`` as a 512px q80
    JPEG, so the lattice CF stage + its QC panel reuse it instead of re-squarifying. Returns
    the path, or None on any failure (never breaks classification)."""
    if not squarify_dir or not hasattr(model, "squarify_tile"):
        return None
    try:
        import cv2
        tile = model.squarify_tile(crop.crop_path)     # BGR (1440x1440 at sz=720)
        if tile is None:
            return None
        if max(tile.shape[:2]) > _TILE_STORE_PX:
            tile = cv2.resize(tile, (_TILE_STORE_PX, _TILE_STORE_PX), interpolation=cv2.INTER_AREA)
        p = Path(squarify_dir) / f"det{int(crop.detection_id)}__tile_four.jpg"
        if not cv2.imwrite(str(p), tile, [cv2.IMWRITE_JPEG_QUALITY, _TILE_QUALITY]):
            return None
        return str(p)
    except Exception as exc:  # noqa: BLE001 - a missing QC tile must never fail the stage
        log.warning("squarify-tile save failed for det %s (%s)", crop.detection_id, exc)
        return None


def _specimen_ruler_class(rows: Sequence[RulerClassRow]) -> Optional[str]:
    """Per-specimen consensus ruler unit-type: the most common ensemble class across the
    specimen's Ruler crops, ignoring ``UNKNOWN``. ``None`` when there are no rulers or all
    classified as UNKNOWN. Ties resolve to the first-seen (crop-order) class."""
    votes = [r.unit_type for r in rows if r.unit_type and r.unit_type != "UNKNOWN"]
    return Counter(votes).most_common(1)[0][0] if votes else None


class RulerClassifier(PipelineStage):
    """Assign a unit-type to each Ruler crop via the classifier ensemble."""

    key: str = "ruler_classifier"
    name: str = "Ruler Classifier"
    depends_on: tuple[str, ...] = ("archival_detector",)
    owns_tables: tuple[str, ...] = ("ruler_classification",)
    device_kind: str = "cuda"
    # STATIC worker policy -- deliberately unlike every other GPU stage, which is sized purely
    # by free VRAM. The ensemble is small enough that ~16 workers FIT on a 48 GB card, but the
    # stage is far too cheap to repay them: each extra worker costs spawn + model warm-load, and
    # measured on 120 sheets (176 crops) 16 workers took 6.6s against 4.0s for four. At 33 crops
    # even ONE worker beat four. So cap by batch size instead of by what memory allows. These
    # are upper bounds -- `_apply_worker_tiers` runs after the VRAM planner, so a machine that
    # cannot host the tier gets the most it can hold.
    worker_tiers: tuple[tuple[int | None, int], ...] = ((100, 2), (1000, 4), (None, 16))

    def build_model(self, device):
        """Warm-load the ruler unit-type ensemble once per worker."""
        return load_ruler_ensemble(self.cfg, device)

    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen that owns Ruler crops; payload = (crop refs, _ruler_squarify dir)."""
        squarify_dir = str(project.dirs.ruler_squarify)
        return [
            WorkItem(sid, (project.db.crops("archival_detection", sid, cls_name="Ruler"), squarify_dir))
            for sid in project.db.specimens_with_crops("archival_detection", cls_name="Ruler")
        ]

    def infer(self, item: WorkItem, model) -> list[RulerClassRow]:
        """Classify each Ruler crop; store the raw vote dict + pre-made squarify tile path."""
        crops, squarify_dir = item.payload
        rows: list[RulerClassRow] = []
        for crop in crops:
            out = model.predict(crop.crop_path)
            conf = out.get("conf") if hasattr(out, "get") else None
            rows.append(
                RulerClassRow(
                    detection_id=crop.detection_id,
                    unit_type=out["ensemble"],
                    votes=out,
                    conf=float(conf) if conf is not None else None,
                    squarify_path=_save_squarify_tile(model, crop, squarify_dir),
                )
            )
        return rows

    def persist(self, project, item: WorkItem, rows: list[RulerClassRow]) -> None:
        """Upsert the per-crop ruler classifications AND the specimen-level consensus class."""
        project.db.record_ruler_classifications(item.specimen_id, rows)
        project.db.set_specimen_ruler_class(item.specimen_id, _specimen_ruler_class(rows))
