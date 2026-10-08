"""Stage 5 — Ruler Conversion Factor (lattice method).

Determine one pixel->cm conversion factor per sheet by measuring the tick lattice of every
Ruler crop and fusing them, using the MP-predicted CF as an anchor. Ported from the
LM3_Ruler_Segmentation research pipeline (``inference/ruler_lattice``): pure cv2/numpy tick
analysis, no models. The heavy lifting lives in :class:`RulerCFLattice`; this stage only wires
its ``process_specimen -> write_db`` contract into LM3.

The engine holds CONFIGURATION ONLY (no per-image state), so the single instance built in
``collect_items`` is shared across the CPU worker threads safely -- every per-sheet value is a
local inside ``process_specimen``.

CF PUBLISHING IS GATED. A sheet's CF is written to ``specimen.cf_px_per_cm`` (working frame,
``cf_source = 'measured_from_ruler'``) only when the engine certifies it 'high' confidence
('published'). By default a withheld / no-reading / no-ruler sheet leaves ``cf_px_per_cm`` NULL --
a visible absence rather than an unknown unit-misnaming error.

``modules.ruler_cf.use_CF_predicted_by_MP`` (off by default) changes only those sheets: they get
the megapixel prediction instead, in the WORKING frame (the engine's ``mp_anchor_working``, so a
linear original-frame model is rescaled exactly as the lattice anchor is), with ``cf_source =
'predicted_from_megapixels'`` and no ``ruler_unit_type``. With the option on, sheets that have NO
Ruler crop are also run through the engine (it records them as ``no_ruler``), so they get both
the fallback CF and an audit row. The full audit trail (including the withheld reading) always
lands in ``ruler_CF_lattice`` + its per-crop table. The QC panel is NOT drawn here; it is deferred to the Reporter, which rebuilds it from
the stored record alone (``Overlay/Overlay_Ruler_Lattice``).
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from leafmachine3.core.records import CF_SOURCE_MP, CF_SOURCE_RULER
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.ruler_cf")


class RulerConversionFactor(PipelineStage):
    """Lattice ruler conversion-factor stage: measure every Ruler, fuse to one gated sheet CF."""

    key: str = "ruler_cf"
    name: str = "Ruler Conversion Factor"
    depends_on: tuple[str, ...] = ("archival_detector", "ruler_classifier")
    owns_tables: tuple[str, ...] = ("ruler_CF_lattice_crop", "ruler_CF_lattice")
    device_kind: str = "cpu"
    cpu_parallel: str = "process"      # the tick-lattice measurement is GIL-bound; processes scale ~linearly
    est_item_seconds: float = 2.0      # ~per-sheet lattice measurement cost (analyse runs 2x per ruler crop)

    # ---- config ------------------------------------------------------------
    def _settings(self) -> dict:
        c = self.cfg.stage(self.key)
        g = c.get if hasattr(c, "get") else (lambda k, d: d)
        # squarify config comes from the RulerClassifier (it made the pre-made tiles); recorded
        # only for provenance in engine_params_json -- the engine never re-squarifies.
        sq = self.cfg.stage("ruler_classifier")
        sqg = sq.get if hasattr(sq, "get") else (lambda k, d: d)
        squarify = sqg("squarify", {}) or {}
        return {
            "anchor_tol": float(g("anchor_tol", 0.25)),
            "min_frame_cm": (None if g("min_frame_cm", None) is None else float(g("min_frame_cm", None))),
            "squarify_sz": int((squarify.get("sz") if hasattr(squarify, "get") else None) or 720),
            "squarify_method": str((squarify.get("method") if hasattr(squarify, "get") else None) or "tile_four"),
            "use_mp_fallback": bool(g("use_CF_predicted_by_MP", False)),
        }

    def build_model(self, device):
        """Warm-build the (config-only, model-less) lattice engine once per process worker.

        Returns the engine so it is created ONCE per spawn worker rather than pickled per item.
        The rot/tick raster dir is resolved from cfg (idempotent) since build_model has no dirs."""
        from leafmachine3.core.dirs import build_dirs
        return self._build_engine(build_dirs(self.cfg).ruler_cf_lattice)

    def _build_engine(self, artifact_dir):
        from leafmachine3.inference.ruler_lattice import RulerCFLattice
        s = self._settings()
        # tile_store_px/tile_quality are left at the engine defaults (512 / q80), which MUST match
        # the RulerClassifier constants that actually write the pre-made tile (ruler_classifier.py
        # _TILE_STORE_PX / _TILE_QUALITY) -- they are recorded in engine_params_json for provenance,
        # so there is no ruler_cf knob that could disagree with the tile on disk.
        return RulerCFLattice(
            artifact_dir=artifact_dir,
            write_qc=False,          # QC panel is deferred to the Reporter (rebuilt from the DB record)
            write_rasters=True,      # rot/tick rasters are what the Reporter redraws from
            squarify_sz=s["squarify_sz"], squarify_method=s["squarify_method"],
            anchor_tol=s["anchor_tol"], min_frame_cm=s["min_frame_cm"],
        )

    # ---- pipeline hooks ----------------------------------------------------
    def collect_items(self, project) -> list[WorkItem]:
        """One item per specimen owning Ruler crops; payload = (specimen dict, crops). The engine
        is built per worker in build_model, so nothing heavy is pickled into the process pool.

        With ``use_CF_predicted_by_MP`` on, every eligible specimen WITHOUT a Ruler crop gets an
        item too (empty crop list -> the engine records 'no_ruler' and its working-frame MP anchor,
        which becomes the fallback CF). With it off those specimens get no item, as before, and the
        executor marks them done(no_work)."""
        from leafmachine3.inference import load_mp_conversion_factor
        mp_model = load_mp_conversion_factor(self.cfg, None)
        anchor_frame = mp_model.frame

        sids = list(project.db.specimens_with_crops("archival_detection", cls_name="Ruler"))
        if self._settings()["use_mp_fallback"]:
            with_rulers = set(sids)
            sids += [sid for sid in project.db.eligible_specimens(self.key, self.depends_on)
                     if sid not in with_rulers]

        items: list[WorkItem] = []
        for sid in sids:
            s = project.db.get_specimen(sid)
            specimen = {
                "specimen_id": sid,
                "image_name": (s["image_stem"] if s is not None else str(sid)),
                "work_scale": (s["work_scale"] if s is not None else 1.0),
                "working_width": (s["width"] if s is not None else None),
                "working_height": (s["height"] if s is not None else None),
                "original_mp": (s["original_mp"] if s is not None else None),
                "cf_px_per_cm_predicted_by_mp": (s["cf_px_per_cm_predicted_by_mp"] if s is not None else None),
                # Which frame the stored anchor is expressed in. Carried explicitly rather than
                # assumed, because it depends on the MP model's FORM: the sqrt form is evaluated on
                # the working dims and is already working-frame, while the linear form is evaluated
                # on the originals and still needs the * work_scale step in the engine.
                "anchor_frame": anchor_frame,
                # The anchor's own equation with this image's megapixels substituted in. Carried so
                # a QC panel that has to fall back to the anchor can SHOW the user where the number
                # came from, instead of printing a bare px/cm with no provenance.
                "anchor_formula": mp_model.formula_text(
                    s["original_mp"] if s is not None else None),
                "anchor_formula_symbolic": mp_model.formula_symbolic(),
            }
            items.append(WorkItem(sid, (specimen, project.db.ruler_lattice_crops(sid))))
        return items

    def infer(self, item: WorkItem, model) -> tuple:
        """Measure + fuse this sheet's rulers (no DB, no QC). -> (record, write-back or None).

        ``model`` is the per-worker RulerCFLattice engine from build_model. The settings are read
        here (not baked into the engine) because the fallback is a publishing policy, not a
        measurement parameter -- the engine's record is identical with the option on or off."""
        specimen, crops = item.payload
        record = model.process_specimen(specimen, crops)
        return record, _writeback(record, use_mp_fallback=self._settings()["use_mp_fallback"])

    def persist(self, project, item: WorkItem, payload: tuple) -> None:
        """Store the full lattice trail; write the sheet CF + its source when there is one."""
        record, writeback = payload
        if writeback is not None and writeback["source"] == CF_SOURCE_MP:
            # Stamp the audit row too, so the QC panel and ruler_conversion_factor.csv say the
            # fallback was APPLIED rather than merely available.
            record = {**record, "image": {**record["image"], "cf_source": CF_SOURCE_MP}}
        project.db.record_ruler_cf_lattice(record)
        if writeback is not None:
            project.db.set_specimen_cf(item.specimen_id, writeback["cf_px_per_cm"],
                                       unit_type=writeback["unit_type"],
                                       source=writeback["source"])


def _writeback(record: dict, *, use_mp_fallback: bool = False) -> Optional[dict]:
    """The specimen CF write-back for this sheet, or None to leave ``cf_px_per_cm`` NULL.

    A PUBLISHED (high-confidence) sheet writes the lattice CF, source 'measured_from_ruler'.
    unit_type is the dominant classifier ruler_class among the crops that were actually USED to
    produce the CF (the reconciliation verdict), so the stored unit-type matches the reading.

    Any other sheet (withheld / no_reading / no_ruler) writes the WORKING-frame MP anchor, source
    'predicted_from_megapixels', only when ``use_mp_fallback`` is on and an anchor exists. It
    carries no unit_type: no ruler stands behind it."""
    img = record.get("image") or {}
    if img.get("status") == "published" and img.get("cf_px_per_cm") is not None:
        used = [c.get("ruler_class") for c in record.get("crops", [])
                if c.get("verdict") == "used" and c.get("ruler_class")]
        unit = Counter(used).most_common(1)[0][0] if used else None
        return {"cf_px_per_cm": float(img["cf_px_per_cm"]), "unit_type": unit,
                "source": CF_SOURCE_RULER}
    anchor = img.get("mp_anchor_working")
    if use_mp_fallback and anchor:
        return {"cf_px_per_cm": float(anchor), "unit_type": None, "source": CF_SOURCE_MP}
    return None
