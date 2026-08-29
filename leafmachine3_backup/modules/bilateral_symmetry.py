"""Stage 12 -- Bilateral Symmetry (CPU post-process).

Measure how well each oriented leaf mirrors about its TRACED MIDVEIN, and roll that up with three
mask/trace quality terms into a single ``archetype_score``. The two columns worth querying are
``archetype_score`` and ``gates_pass``: together they answer "which leaf of this species is the one
to put in a figure?".

Read the score as an EXEMPLAR-QUALITY ranking, not as a symmetry result and not as a mask-fault
detector. On already-clean masks the composite tracks ``si_a`` at Spearman -0.97 because the other
three terms saturate, so symmetry IS the ranking.

A low score has several honest causes and the score cannot tell them apart:

* a **folded leaf** -- the midvein legitimately sits along one side of the silhouette;
* a genuinely **asymmetric leaf** -- oblique bases are normal in many taxa;
* a **mis-traced midvein** from the pose model.

All three are reasons not to put that leaf in a figure, so the ranking is doing its job either way.
None of them means the mask is broken -- that question is answered by the structural diagnostics
(``largest_frac``, ``truncated``, ``solidity``) and by looking at the QC panel. See
``core/bilateral.py`` for the geometry and every caveat.

Results go to ``bilateral_symmetry`` (one row per oriented Leaf_WHOLE leaf).

**The QC panels are rendered HERE, in this stage's process pool -- not in the Reporter.** That is
ECT's precedent, and for the same measured reason: the panel is matplotlib, and matplotlib does not
parallelize under threads. Rendering 24 real panels in the Reporter's thread pool measured 2.76s on
1 thread and 2.80s on 8 -- a 0.99x "speedup", i.e. the GIL serializes it completely. This stage
already runs on a spawn PROCESS pool, and the worker still holds the oriented frame it just
measured, so rendering here is both genuinely parallel and free of the frame rebuild the Reporter
would have had to do. The row still carries the full frame geometry so the panel can be regenerated
later without re-deriving anything from landmarks.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any, Optional

from pathlib import Path

from leafmachine3.core.bilateral import build_leaf, leaf_inputs, measure_leaf
from leafmachine3.core.imaging import crop_filename, save_image
from leafmachine3.core.naming import BILATERAL, crop_label
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.bilateral_symmetry")

#: ``qc_images`` values. ``flagged`` renders only the leaves someone would actually open.
QC_NONE, QC_FLAGGED, QC_ALL = "none", "flagged", "all"


class BilateralSymmetry(PipelineStage):
    """Per-leaf bilateral-symmetry metrics + the archetype score, about the traced midvein."""

    key: str = "bilateral_symmetry"
    name: str = "Bilateral Symmetry"
    device_kind: str = "cpu"
    # cKDTree, the smoothing spline and the mask rasterization are all GIL-bound numpy/scipy, so a
    # thread pool would serialize them -- same reasoning as ECT and ruler_cf.
    cpu_parallel: str = "process"
    # One WorkItem PER LEAF. A sheet can carry 1 or 60 leaves, so per-specimen items would leave the
    # pool waiting on whichever worker drew the densest sheet.
    fanout: bool = True
    # measure ~0.07s + the QC panel ~0.115s when one is drawn. In the default `flagged` mode only
    # ~25% of leaves get a panel, so the average lands near 0.10; the pool sizer only needs the
    # right order of magnitude to decide the batch is worth a process pool.
    est_item_seconds: float = 0.10
    # NOT ("reporter",) -- unlike ECT this stage does not read the Reporter's PNGs. It rebuilds the
    # oriented frame from the stored polygons + orientation angle, which matches the saved
    # Lamina_Holes_Mask at IoU 0.999999, so it can run BEFORE the Reporter and hand it the geometry.
    depends_on: tuple[str, ...] = ("leaf_segmenter", "landmark_detector", "morphology",
                                   "leaf_orientation")
    owns_tables: tuple[str, ...] = ("bilateral_symmetry",)

    def _settings(self) -> dict:
        """Inline defaults are the single source of truth -- a missing YAML key is never an error."""
        c = self.cfg.stage(self.key)
        g = c.get if hasattr(c, "get") else (lambda k, d: d)
        qc = str(g("qc_images", QC_FLAGGED)).strip().lower()
        return {
            "min_kpt_conf": float(g("min_kpt_conf", 0.25)),
            "n_bins": int(g("n_bins", 200)),          # UPPER bound; core.bilateral.bins_for adapts
            "min_score": float(g("min_score", 0.50)),
            "qc_images": qc if qc in (QC_NONE, QC_FLAGGED, QC_ALL) else QC_FLAGGED,
        }

    def collect_items(self, project) -> list[WorkItem]:
        """ONE WorkItem PER LEAF (fanout), carrying only the small ingredients.

        The DB read happens here in the parent because workers are pure; the rasterize + rotate +
        measure all happen in the worker, so the parent never becomes the bottleneck.
        """
        s = self._settings()
        reports = str(project.dirs.reports)      # workers write their own panel, so they need this
        items: list[WorkItem] = []
        for sid in project.db.specimens_with_rows("leaf_segmentation"):
            for inp in leaf_inputs(project.db, sid, min_kpt_conf=s["min_kpt_conf"]):
                items.append(WorkItem(int(sid), {"leaf": inp, "settings": s, "reports_dir": reports}))
        return items

    def infer(self, item: WorkItem, model: Any) -> Optional[dict]:
        """Measure one leaf. ``None`` when it cannot be measured (no midvein, empty mask)."""
        p = item.payload
        s = p["settings"]
        leaf = build_leaf(p["leaf"])
        if leaf is None:
            return None
        m = measure_leaf(leaf, n_bins=s["n_bins"], min_score=s["min_score"])
        if m is None:
            return None

        row = {
            "leaf_id": leaf.leaf_id, "specimen_id": leaf.specimen_id,
            "detection_id": leaf.detection_id, "instance_index": 0,
            "si_a": _f(m.si_a), "a_star": _f(m.a_star), "dice": _f(m.dice),
            "sinuosity": _f(m.sinuosity), "archetype_score": _f(m.archetype_score),
            "term_symmetry": _f(m.term_symmetry), "term_integrity": _f(m.term_integrity),
            "term_completeness": _f(m.term_completeness), "term_trace": _f(m.term_trace),
            "gates_pass": bool(m.gates_pass), "is_archetypal": bool(m.is_archetypal),
            "reasons_json": json.dumps(m.reasons or []),
            "largest_frac": _f(m.largest_frac), "solidity": _f(m.solidity),
            "perimeter_ratio": _f(m.perimeter_ratio), "hole_frac": _f(m.hole_frac),
            "kpt_conf_mean": _f(m.kpt_conf_mean), "kpt_conf_min": _f(m.kpt_conf_min),
            "n_midvein_kpts": int(m.n_midvein_kpts), "truncated": bool(m.truncated),
            "angle_cw": _f(m.angle_cw), "crop_w": m.crop_w, "crop_h": m.crop_h,
            "mask_w": m.mask_w, "mask_h": m.mask_h,
            "tip_x": _f(m.tip_x), "tip_y": _f(m.tip_y),
            "base_x": _f(m.base_x), "base_y": _f(m.base_y),
            "midvein_json": json.dumps([[round(x, 2), round(y, 2)] for x, y in (m.midvein or [])]),
            "n_bins": int(m.n_bins),
        }
        # Render the QC panel HERE (see the module docstring): this worker is a process, so the
        # matplotlib cost actually parallelizes, and the frame is already built.
        name = _qc_name(self.cfg, leaf, m, s["qc_images"])
        row["qc_png"] = _render_qc(leaf, row, name, Path(p["reports_dir"])) if name else None
        return row

    def persist(self, project, item: WorkItem, row: Optional[dict]) -> None:
        if row:
            project.db.record_bilateral_one(row)


def _f(v: Any) -> Optional[float]:
    """SQLite has no NaN: store an unmeasurable quantity as NULL, never as 0.0."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _render_qc(leaf, row: dict, name: str, reports: Path) -> Optional[str]:
    """Draw + write one QC panel, returning the filename stored on the row (or ``None``).

    Imported lazily so a run with ``qc_images: none`` never pays matplotlib's import cost, and so a
    rendering failure costs one leaf's picture rather than its measurements -- the numbers are the
    product here, the panel is an aid.
    """
    try:
        from leafmachine3.reporting.bilateral_viz import render_qc_panel

        img = render_qc_panel(leaf.silhouette, row)
        if img is None:
            return None
        # 85, not the report's jpg_quality of 100: this is a rendered chart of flat fills and text,
        # where 100 costs ~2.4x the bytes for no visible gain.
        save_image(img[:, :, ::-1], reports / "Leaf_Data" / "Bilateral_Symmetry" / name, quality=85)
        return name
    except Exception:  # noqa: BLE001 - never lose a measured leaf over its illustration
        log.exception("bilateral QC panel failed for leaf_id=%s", row.get("leaf_id"))
        return None


def _qc_name(cfg, leaf, m, mode: str) -> Optional[str]:
    """Filename the Reporter should write for this leaf's QC panel, or ``None`` for no panel.

    Identified by the canonical crop filename -- ``<stem>__BSYM-leaf__x1_y1_x2_y2.jpg`` -- never by
    an index. JPEG rather than PNG: the panel is a rendered figure at ~1500 px, where JPEG is several
    times smaller, and it is a QC aid rather than data (the numbers all live in the table).
    """
    if mode == QC_NONE:
        return None
    if mode != QC_ALL and (m.gates_pass and bool(m.is_archetypal)):
        return None                       # `flagged`: only leaves someone would actually open
    inst = 0
    stem = leaf.stem if inst == 0 else f"{leaf.stem}__i{inst}"
    return crop_filename(stem, crop_label(cfg, BILATERAL, "Leaf"), leaf.det_box, "jpg")
