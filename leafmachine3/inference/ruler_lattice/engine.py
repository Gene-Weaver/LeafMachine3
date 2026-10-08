"""Production packaging of the lattice ruler-CF method: one class, one record, one QC panel.

This is the hand-off artifact for the LM3 integration. It does not modify LM3; it
wraps the experimental modules in this folder behind a single class whose contract
is what a LeafMachine3 stage needs:

    engine = RulerCFLattice(artifact_dir=<run>/ruler_cf_lattice)
    record = engine.process_specimen(specimen, crops)     # ONE parent image
    engine.write_db(con, record)                          # idempotent upsert

`process_specimen` returns a plain, JSON-safe dict whose keys are exactly the
column names of the two tables in SCHEMA_SQL, so ingestion is a dict-to-INSERT
with no translation layer. Nothing is written to the DB by the engine itself --
the caller owns the transaction, as every other LM3 stage does.

WHY TWO TABLES. The request was for a table `ruler_CF_lattice` holding an image-
level section plus each candidate ruler crop's info. In SQLite that is naturally a
parent row plus N child rows, mirroring how LM3 already models specimen ->
archival_detection, so the crop section lives in `ruler_CF_lattice_crop` keyed by
detection_id. `v_ruler_CF_lattice` joins them back into the single flat view the
request describes. Nothing is lost and the crop section stays queryable.

THE AUDIT GUARANTEE. The QC panel is rendered ONLY from the record plus the
rasters the record points at -- `render_qc(record)` never sees the live engine
output. So if the panel can be drawn, the DB provably contains enough to redraw
it later, and that is enforced by construction rather than by a promise: the live
path and the reconstruction path are the same code path. `--selftest` re-renders
every panel from a round-trip through SQLite and compares pixels.

STATE. The class holds CONFIGURATION ONLY. Every per-image value lives in a local
inside `process_specimen`, so one parent image can never contaminate the next.
`--selftest` proves it: it processes A, then B, then A again, and asserts the two
A records are byte-identical, and that the instance `__dict__` is unchanged.

Run it exactly as production would, over a prepared LM3 run DB:

    ../.venv_sync_labelbox/bin/python ruler_CF_with_lattice_detection.py \
        --run-name herbcode2 --qc-dir QC_images/test_real_pipeline_1imgperherbcode
    ../.venv_sync_labelbox/bin/python ruler_CF_with_lattice_detection.py --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from leafmachine3.core.records import CF_SOURCE_MP, CF_SOURCE_RULER

from .analysis import analyse, build_masks, summarise
from .sheet_cf import reconcile_parent
from .units import is_skipped, spec_of, system_of
from . import qc as QC
from .qc import mask_overlay, comb_overlay, ruler_bars

HERE = Path(__file__).resolve().parent

# Bump when a change alters numeric output, so a DB row always says which engine
# produced it and a mixed-version project is detectable with one GROUP BY.
ENGINE_VERSION = "lattice-2026.07.30"

CM_PER_INCH = 2.54


# --------------------------------------------------------------------- schema --
SCHEMA_SQL = """
-- ruler_CF_lattice : ONE row per parent image (specimen). The lattice ruler-CF
-- stage's complete verdict for that sheet -- what it published, what it measured,
-- how confident it is, why, and where the QC raster lives.
--
-- cf_px_per_cm is the PUBLISHED conversion factor and is NULL unless the sheet
-- cleared the confidence gate. That is deliberate: a wrong CF silently corrupts
-- every downstream measurement on the sheet, whereas a NULL is a visible absence
-- that falls back to specimen.cf_px_per_cm_predicted_by_mp (a regression with a
-- known ~4.4 px/cm rmse) instead of an unknown unit-misnaming error.
-- cf_px_per_cm_measured keeps the withheld reading for audit. NEVER consume it
-- as the sheet's CF.
CREATE TABLE IF NOT EXISTS ruler_CF_lattice (
    specimen_id           INTEGER PRIMARY KEY REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    image_name            TEXT,
    engine_version        TEXT NOT NULL,
    engine_params_json    TEXT,               -- every tunable, so a row is reproducible
    created_at            TEXT NOT NULL,      -- UTC ISO-8601
    runtime_ms            INTEGER,

    -- frame. Everything measured here is in the WORKING frame; original-frame
    -- values are derived by dividing by work_scale and are stored explicitly so
    -- no consumer has to remember which frame it is looking at.
    work_scale            REAL,
    working_width         INTEGER,
    working_height        INTEGER,

    -- outcome
    status                TEXT NOT NULL,      -- published | withheld | no_reading | no_ruler
    cf_px_per_cm          REAL,               -- PUBLISHED (working frame); NULL unless confidence='high'
    cf_px_per_inch        REAL,
    cf_px_per_cm_original REAL,               -- same CF expressed in the ORIGINAL frame
    cf_px_per_cm_measured REAL,               -- what the lattice read, even when withheld
    cf_source             TEXT,               -- 'measured_from_ruler' when published; 'predicted_from_megapixels'
                                              -- when the ruler_cf stage APPLIED the MP fallback
                                              -- (modules.ruler_cf.use_CF_predicted_by_MP); else NULL
    fallback              TEXT,               -- what a consumer should use instead: 'mp_anchor' | 'none'

    -- confidence gate
    confidence            TEXT NOT NULL,      -- high | medium | low
    confidence_reasons_json TEXT,             -- ordered human-readable reasons
    anchor_supported      INTEGER,
    anchor_log_dist       REAL,               -- |log(cf/anchor)|; compare with RUNG_HALF_LOG
    -- Length-backed acceptance: the CF disagreed with the anchor by more than half a rung but was
    -- published anyway because it rests on enough ruler to be trusted (sheet_cf.MIN_TRUSTED_RULER_CM).
    -- Stored so the decision is queryable instead of only greppable in confidence_reasons_json.
    length_backed         INTEGER,
    win_ruler_len_cm      REAL,               -- implied physical length of the winning cluster's ruler
    corroborated_by_peers INTEGER,
    n_dissenting          INTEGER,            -- well-supported crops that DISAGREED

    -- MP anchor. original_mp is the megapixel count the regression was evaluated
    -- at; outside its fitted 14.6-36.2 range the anchor is an extrapolation, so an
    -- auditor needs the input, not only the output.
    original_mp           REAL,
    mp_anchor_original    REAL,
    mp_anchor_working     REAL,
    -- Which frame the MP model predicted in. "working" (sqrt form, evaluated on the working dims)
    -- needs no rescaling; "original" (linear form) is multiplied by work_scale to get here. Stored
    -- so a rebuilt QC panel states the truth instead of assuming the old conversion happened.
    anchor_frame          TEXT,
    anchor_formula        TEXT,             -- the anchor equation with this image's MP filled in
    anchor_formula_symbolic TEXT,           -- the same equation with MP left symbolic
    anchor_dropped        INTEGER,            -- 1 = implied an impossible sheet width
    pct_vs_anchor         REAL,

    -- multi-ruler reconciliation
    method                TEXT,
    n_clusters            INTEGER,
    peer_spread_pct       REAL,
    peers_agree           INTEGER,
    n_ruler_crops         INTEGER,
    n_measured            INTEGER,
    n_skipped             INTEGER,            -- class not supported (FP / messy)
    n_failed              INTEGER,            -- no lattice recovered / unreadable
    n_used                INTEGER,
    n_rejected            INTEGER,

    qc_image_path         TEXT                -- stacked QC panel for the whole sheet
);
CREATE INDEX IF NOT EXISTS ix_rcfl_status ON ruler_CF_lattice (status, confidence);

-- ruler_CF_lattice_crop : ONE row per candidate ruler crop on the sheet, INCLUDING
-- the ones that were skipped or failed -- "this box was considered and rejected"
-- is part of the audit trail, so absence never has to be interpreted.
--
-- The three *_path columns are what make the QC panel reconstructible without the
-- engine. Rasters are files referenced by path, following the leaf_segmentation /
-- specimen mask convention in this schema.
--   tile_four_path  a 512x512 JPEG q80 THUMBNAIL of the four-tile collage. It shows
--                   the composition the RulerClassifier was given, but it is NOT
--                   that image: the classifier receives the full-resolution
--                   1440x1440 collage. Downsampled and lossy on purpose (~40x
--                   smaller); never feed it to a model as if it were the original.
--                   Size and quality are recorded in engine_params_json.
--   rot_path        the deskewed grayscale strip every overlay is drawn on, exact.
--   tick_mask_path  per-level tick label raster, uint8, same size as rot, exact.
CREATE TABLE IF NOT EXISTS ruler_CF_lattice_crop (
    crop_row_id      INTEGER PRIMARY KEY,
    specimen_id      INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    detection_id     INTEGER NOT NULL REFERENCES archival_detection(detection_id) ON DELETE CASCADE,
    crop_index       INTEGER NOT NULL,        -- stable order within the sheet's panel

    -- provenance of the box
    crop_path        TEXT,
    det_conf         REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    crop_w           INTEGER,
    crop_h           INTEGER,

    -- what the classifier said
    ruler_class      TEXT,
    cls_conf         REAL,
    class_systems    TEXT,                    -- '+'-joined, from CLASS_SPEC
    class_layout     TEXT,
    class_units      TEXT,                    -- printed units the class declares
    admissible_units TEXT,                    -- what naming was allowed to choose from

    -- outcome for THIS crop
    status           TEXT NOT NULL,           -- measured | skipped_class | failed | unreadable
    status_reason    TEXT,
    verdict          TEXT,                    -- used | rejected | skipped  (set by reconciliation)
    verdict_note     TEXT,
    pxcm             REAL,                    -- this crop's own CF, working frame
    pxcm_original    REAL,
    pxcm_no_anchor   REAL,                    -- the same read with the MP anchor withheld
    anchor_changed_read INTEGER,              -- 1 = the anchor flipped the harmonic
    pct_vs_parent    REAL,
    pct_vs_anchor    REAL,

    -- how the lattice was recovered (the measurement audit trail)
    rotation_deg     REAL,
    band_y0          INTEGER,
    band_y1          INTEGER,
    band_kind        TEXT,                    -- adaptive | fixed
    band_periodicity REAL,
    P_acf_px         REAL,                    -- the autocorrelation's chosen lag
    P_base_px        REAL,                    -- the base period after descent
    period_read_as   TEXT,                    -- the UNIT that period was named
    ladder           TEXT,                    -- level multipliers found on the row
    first_tick_x     REAL,
    n_ticks          INTEGER,
    n_kept           INTEGER,
    keep_ratio       REAL,
    occupancy        REAL,
    salvaged         INTEGER,
    lattice_fallback INTEGER,

    -- how the units were named and fused
    layout           TEXT,
    relation         TEXT,                    -- single unit | multi-scale | unresolved
    systems          TEXT,
    n_units          INTEGER,
    units            TEXT,
    unit_estimates   TEXT,
    n_corroborating  INTEGER,
    rec_agree        INTEGER,   -- did the per-unit estimates agree
    cross_unit_spread_pct REAL,
    cf_contributions TEXT,                    -- unit:est:n_ticks:source:used/rejected
    cf_n_used        INTEGER,
    cf_n_rejected    INTEGER,
    cf_spread_pct    REAL,
    cf_fallback      INTEGER,
    implied_len_cm   REAL,                    -- physical plausibility of this read

    -- band reconciliation within the crop
    bands_evaluated  INTEGER,
    n_band_clusters  INTEGER,
    n_bands_agree    INTEGER,
    band_spread_pct  REAL,
    band_cf          REAL,
    bands_thin       INTEGER,
    shadows          TEXT,                    -- harmonic shadows that were rejected
    stacked_pair     TEXT,

    -- Transition rulers (mm printed on one part of the strip, cm on the rest).
    -- The per-side periods and their named units are stored, not just the ratio:
    -- the panel prints them, and the 10:1 ratio check is the single most reliable
    -- signal in the pipeline, so an auditor needs to see both sides of it.
    trans_ratio      REAL,
    trans_ratio_err_pct REAL,
    trans_expect     REAL,
    trans_split_x    INTEGER,
    trans_order      TEXT,
    trans_used       INTEGER,
    trans_P_fine     REAL,
    trans_P_coarse   REAL,
    trans_unit_fine  TEXT,
    trans_unit_coarse TEXT,
    trans_pxcm_fine  REAL,
    trans_pxcm_coarse REAL,

    -- rasters: everything render_qc needs to redraw this crop's sections
    tile_four_path   TEXT,                    -- 1440x1440 collage the classifier saw
    rot_path         TEXT,                    -- deskewed grayscale strip
    tick_mask_path   TEXT,                    -- uint8 per-level tick labels, same size as rot
    groups_json      TEXT,                    -- named unit groups -> the predicted combs
    -- The engine's OWN output, verbatim. The flattened columns above exist to be
    -- queried; these two exist so the panel can be redrawn without a single value
    -- being inferred. Nothing in the reconstruction path is allowed to invent a
    -- default: if a field is absent here it is NULL, and NULL renders as absent.
    summary_json     TEXT,                    -- ruler_analysis.summarise(res), complete
    rec_json         TEXT,                    -- res['rec'], the unit reconciliation
    px_per_cm_bar    INTEGER,                 -- drawn 1 cm bar length, px
    px_per_inch_bar  INTEGER,

    UNIQUE (detection_id)                     -- idempotent re-run
);
CREATE INDEX IF NOT EXISTS ix_rcfl_crop_spec ON ruler_CF_lattice_crop (specimen_id, crop_index);

-- The flat image+crops view the audit trail is meant to be read through.
CREATE VIEW IF NOT EXISTS v_ruler_CF_lattice AS
SELECT p.*,
       c.detection_id, c.crop_index, c.ruler_class, c.status AS crop_status,
       c.verdict, c.pxcm, c.period_read_as, c.P_acf_px, c.P_base_px,
       c.rotation_deg, c.n_kept, c.implied_len_cm,
       c.tile_four_path, c.rot_path, c.tick_mask_path, c.crop_path
FROM ruler_CF_lattice p
LEFT JOIN ruler_CF_lattice_crop c USING (specimen_id);
"""

_IMAGE_COLS = None      # filled lazily from SCHEMA_SQL so code and DDL cannot drift
_CROP_COLS = None


def _cols(table: str) -> list[str]:
    """Column names parsed out of SCHEMA_SQL -- the DDL is the single source."""
    body = SCHEMA_SQL.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1]
    depth, out = 1, []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith(")"):
            break
        if not s or s.startswith("--"):
            continue
        s = s.split("--")[0].strip().rstrip(",")
        if not s or s.upper().startswith(("UNIQUE", "PRIMARY KEY", "FOREIGN KEY",
                                          "CHECK", "CONSTRAINT")):
            continue
        for decl in s.split(","):
            decl = decl.strip()
            if decl:
                out.append(decl.split()[0])
    return out


def image_columns() -> list[str]:
    global _IMAGE_COLS
    if _IMAGE_COLS is None:
        _IMAGE_COLS = _cols("ruler_CF_lattice")
    return _IMAGE_COLS


def crop_columns() -> list[str]:
    global _CROP_COLS
    if _CROP_COLS is None:
        _CROP_COLS = [c for c in _cols("ruler_CF_lattice_crop") if c != "crop_row_id"]
    return _CROP_COLS


# ---------------------------------------------------------------- small utils --
def _num(v):
    """SQLite-safe scalar: numpy -> python, NaN/inf -> None."""
    if v is None or isinstance(v, (str, bytes)):
        return v
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, int):
        return v
    return v


def _json(v):
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            f = float(o)
            return None if (math.isnan(f) or math.isinf(f)) else f
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)
    return json.dumps(v, default=default, sort_keys=True)


def _pct(a, b):
    return None if not (a and b) else 100.0 * (float(a) / float(b) - 1.0)


# ------------------------------------------------------------------- the class --
class RulerCFLattice:
    """Run the whole lattice ruler-CF method for one parent image.

    The instance carries CONFIGURATION ONLY. `process_specimen` builds its result
    entirely in locals and returns it, so no value from one sheet can reach the
    next. See `--selftest`.
    """

    ENGINE_VERSION = ENGINE_VERSION
    SCHEMA_SQL = SCHEMA_SQL

    # A ruler crop taller than it is wide is the same ruler photographed sideways;
    # the lattice engine measures along x, so it is rotated upright first. This is
    # the engine's own convention, restated here because the stored rot raster and
    # every x-coordinate in the crop row are in the ROTATED frame.
    def __init__(self, artifact_dir, *, write_qc=True, write_rasters=True,
                 squarify_sz=720, squarify_method="tile_four",
                 min_frame_cm=None, anchor_tol=0.25, qc_dir=None,
                 tile_format="jpg", tile_quality=80, tile_store_px=512):
        self.artifact_dir = Path(artifact_dir)
        self.qc_dir = Path(qc_dir) if qc_dir else self.artifact_dir / "qc"
        self.write_qc = bool(write_qc)
        self.write_rasters = bool(write_rasters)
        self.squarify_sz = int(squarify_sz)
        self.squarify_method = str(squarify_method)
        self.anchor_tol = float(anchor_tol)
        self.min_frame_cm = min_frame_cm          # None = ruler_sheet_cf's own default
        # WHAT IS STORED IS A QC THUMBNAIL, NOT THE CLASSIFIER'S INPUT.
        # RulerClassifier receives the full-resolution tile_four collage (1440x1440
        # at sz=720) and that is unchanged. What lands on disk here is a 512x512
        # JPEG q80 copy, ~40x smaller, kept so a human can see the composition the
        # classifier was shown. It is downsampled and lossy by design, so it must
        # never be fed back to a model as if it were the original. The panel renders
        # it at 260 px, well under 512, so nothing visible is lost.
        self.tile_format = str(tile_format).lower().lstrip(".")
        self.tile_quality = int(tile_quality)
        self.tile_store_px = int(tile_store_px)
        # No squarifier here: the four-tile collage is PRE-MADE by the LM3
        # RulerClassifier (in _ruler_squarify/) and its path is handed to
        # process_specimen per crop as ``tile_four_path`` -- the engine never
        # rebuilds it, so there is no model and no per-image state to carry.

    # ---- configuration surface, recorded on every row so a result is reproducible
    def params(self) -> dict:
        from . import sheet_cf as _p
        from . import lattice as _l
        from . import analysis as _a
        from . import units as _u
        return dict(
            engine_version=self.ENGINE_VERSION,
            squarify=dict(sz=self.squarify_sz, method=self.squarify_method),
            tile_format=self.tile_format, tile_quality=self.tile_quality,
            tile_store_px=self.tile_store_px,
            anchor_tol=self.anchor_tol,
            min_frame_cm=(self.min_frame_cm if self.min_frame_cm is not None
                          else _p.MIN_FRAME_CM),
            distance_weight_floor=_p.DISTANCE_WEIGHT_FLOOR,
            peer_tol=_p.PEER_TOL,
            rung_half_log=getattr(_u, "RUNG_HALF_LOG", None),
            min_period=_l.MIN_PERIOD, max_levels=_l.MAX_LEVELS,
            fund_ratio=getattr(_l, "FUND_RATIO", None),
            band_topk=_a.BAND_TOPK, band_min_periodicity=_a.BAND_MIN_PERIODICITY,
            band_peer_tol=_a.BAND_PEER_TOL, unsharp=_a.UNSHARP,
        )

    # ---- schema ------------------------------------------------------------
    def ensure_schema(self, con: sqlite3.Connection) -> None:
        con.executescript(self.SCHEMA_SQL)

    # ---- the one entry point ----------------------------------------------
    def process_specimen(self, specimen: dict, crops: list) -> dict:
        """Measure every ruler crop of ONE parent image and fuse them into one CF.

        specimen : {specimen_id, image_name, work_scale, working_width,
                    working_height, cf_px_per_cm_predicted_by_mp}
        crops    : [{detection_id, crop_path, ruler_class, cls_conf, det_conf,
                     x1, y1, x2, y2}]  -- every Ruler box on the sheet, in any order

        Returns {"image": {...}, "crops": [{...}], "qc_image": PIL.Image|None}.
        The two dict sections use exactly the column names in SCHEMA_SQL.
        """
        t0 = time.time()
        spec_id = specimen.get("specimen_id")
        image_name = specimen.get("image_name")
        work_scale = float(specimen.get("work_scale") or 1.0)
        # Crops are cut from the resized WORKING copy, so the anchor has to reach this comparison
        # in the working frame. Getting it wrong makes the error look like 42% instead of 1%.
        # Which conversion is needed depends on the MP model's form, which the caller declares:
        #   "original" (linear fit)  -- predicted on the original dims, so scale by work_scale
        #   "working"  (sqrt fit)    -- already working-frame; scaling again would double-apply it
        anchor_stored = specimen.get("cf_px_per_cm_predicted_by_mp")
        anchor_frame = str(specimen.get("anchor_frame") or "original")
        anchor_formula = specimen.get("anchor_formula")
        anchor_formula_symbolic = specimen.get("anchor_formula_symbolic")
        if not anchor_stored:
            anchor_work = None
        elif anchor_frame == "working":
            anchor_work = float(anchor_stored)
        else:
            anchor_work = float(anchor_stored) * work_scale
        anchor_orig = anchor_stored
        frame_w = specimen.get("working_width")

        # deterministic order so crop_index and the stacked panel are stable across
        # runs regardless of what order the caller happened to hand us the boxes
        ordered = sorted(crops, key=lambda c: (int(c.get("detection_id") or 0)))

        crop_rows, live = [], []
        for i, c in enumerate(ordered):
            row, live_entry = self._measure_crop(i, c, specimen, anchor_work)
            crop_rows.append(row)
            live.append(live_entry)

        # ---- fuse the sheet's crops into ONE CF ----------------------------
        kw = dict(anchor=anchor_work, anchor_tol=self.anchor_tol,
                  frame_width_px=frame_w)
        pr = reconcile_parent(
            [dict(key=f"det{r['detection_id']}", cf=r["pxcm"], weight=r["n_kept"] or 0,
                  ruler_class=r["ruler_class"],
                  implied_len_cm=r.get("implied_len_cm"),
                  skipped=(r["status"] != "measured"),
                  skip_reason=r["status_reason"]) for r in crop_rows], **kw)

        by_key = {c.get("key"): c for c in pr.get("per_crop", [])}
        for r in crop_rows:
            v = by_key.get(f"det{r['detection_id']}") or {}
            r["verdict"] = v.get("verdict")
            r["verdict_note"] = v.get("note")
            r["pct_vs_parent"] = _num(v.get("pct_vs_parent"))

        cf = pr.get("cf_px_per_cm")
        meas = pr.get("cf_px_per_cm_measured")
        n_meas = sum(1 for r in crop_rows if r["status"] == "measured")
        if not crop_rows:
            status = "no_ruler"
        elif cf is not None:
            status = "published"
        elif meas is not None:
            status = "withheld"
        else:
            status = "no_reading"

        image_row = dict(
            specimen_id=spec_id, image_name=image_name,
            engine_version=self.ENGINE_VERSION,
            engine_params_json=_json(self.params()),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            runtime_ms=int(round(1000 * (time.time() - t0))),
            work_scale=_num(work_scale),
            working_width=_num(specimen.get("working_width")),
            working_height=_num(specimen.get("working_height")),
            status=status,
            cf_px_per_cm=_num(cf),
            cf_px_per_inch=_num(None if cf is None else cf * CM_PER_INCH),
            cf_px_per_cm_original=_num(None if cf is None else cf / (work_scale or 1.0)),
            cf_px_per_cm_measured=_num(meas),
            cf_source=(CF_SOURCE_RULER if cf is not None else None),   # the stage stamps CF_SOURCE_MP
            fallback=("" if cf is not None else
                      ("mp_anchor" if anchor_work else "none")),
            confidence=pr.get("confidence") or "low",
            confidence_reasons_json=_json(pr.get("confidence_reasons") or []),
            anchor_supported=_num(pr.get("anchor_supported")),
            anchor_log_dist=_num(pr.get("anchor_log_dist")),
            length_backed=_num(pr.get("length_backed")),
            win_ruler_len_cm=_num(pr.get("win_ruler_len_cm")),
            corroborated_by_peers=_num(pr.get("corroborated_by_peers")),
            n_dissenting=_num(pr.get("n_dissenting")),
            original_mp=_num(specimen.get("original_mp")),
            mp_anchor_original=_num(anchor_orig),
            anchor_frame=anchor_frame,
            anchor_formula=anchor_formula,
            anchor_formula_symbolic=anchor_formula_symbolic,
            mp_anchor_working=_num(anchor_work),
            anchor_dropped=_num(pr.get("anchor_dropped")),
            pct_vs_anchor=_num(pr.get("pct_vs_anchor")),
            method=pr.get("method"),
            n_clusters=_num(pr.get("n_clusters")),
            peer_spread_pct=_num(None if pr.get("spread") is None
                                 else 100.0 * pr["spread"]),
            peers_agree=_num(pr.get("agree")),
            n_ruler_crops=len(crop_rows), n_measured=n_meas,
            n_skipped=sum(1 for r in crop_rows if r["status"] == "skipped_class"),
            n_failed=sum(1 for r in crop_rows
                         if r["status"] in ("failed", "unreadable")),
            n_used=sum(1 for r in crop_rows if r["verdict"] == "used"),
            n_rejected=sum(1 for r in crop_rows if r["verdict"] == "rejected"),
            qc_image_path=None,
        )

        record = dict(schema_version=self.ENGINE_VERSION,
                      image=image_row, crops=crop_rows)

        # Render from the RECORD, never from the live engine output. If this
        # succeeds the DB provably holds enough to redraw the panel later.
        qc = None
        if self.write_qc:
            qc = self.render_qc(record, reconcile=pr, live=live)
            if qc is not None and image_name:
                self.qc_dir.mkdir(parents=True, exist_ok=True)
                flag = "" if status == "published" else "UNCERTIFIED__"
                p = self.qc_dir / f"{flag}{Path(image_name).stem}.png"
                qc.save(p)
                image_row["qc_image_path"] = str(p)
        record["qc_image"] = qc
        return record

    # ---- one crop ----------------------------------------------------------
    def _measure_crop(self, index, c, specimen, anchor_work):
        """-> (crop_row, live) where `live` carries only in-memory rasters."""
        det = c.get("detection_id")
        cls = c.get("ruler_class") or "METRIC_MM"
        cp = c.get("crop_path")
        work_scale = float(specimen.get("work_scale") or 1.0)
        sp = spec_of(cls)

        row = {k: None for k in crop_columns()}
        row.update(
            specimen_id=specimen.get("specimen_id"), detection_id=det,
            crop_index=index, crop_path=cp, det_conf=_num(c.get("det_conf")),
            x1=_num(c.get("x1")), y1=_num(c.get("y1")),
            x2=_num(c.get("x2")), y2=_num(c.get("y2")),
            ruler_class=cls, cls_conf=_num(c.get("cls_conf")),
            class_systems="+".join(sp.get("systems") or ()) or None,
            class_layout=sp.get("layout"),
            class_units="|".join(sp.get("units") or []) or None,
            admissible_units="|".join(_admissible(cls)) or None,
            verdict=None, verdict_note=None)
        live = dict(detection_id=det, rot=None, lab=None, tile=None,
                    res=None, s=None, groups=None)

        # A class the registry marks unsupported (FP / messy) is recorded and
        # skipped BEFORE any pixels are touched -- no CF is meaningful for it.
        sk = is_skipped(cls)
        if sk:
            row.update(status="skipped_class", status_reason=sk)
            row["tile_four_path"] = c.get("tile_four_path")   # pre-made by RulerClassifier
            return row, live

        g = cv2.imread(cp, cv2.IMREAD_GRAYSCALE) if cp and os.path.exists(cp) else None
        if g is None:
            row.update(status="unreadable",
                       status_reason=f"crop file could not be read: {cp}")
            return row, live
        row.update(crop_w=int(g.shape[1]), crop_h=int(g.shape[0]))
        if g.shape[0] > g.shape[1]:
            g = cv2.rotate(g, cv2.ROTATE_90_CLOCKWISE)

        row["tile_four_path"] = c.get("tile_four_path")       # pre-made by RulerClassifier

        try:
            res = analyse(g, cls, cf_anchor=anchor_work)
            plain = analyse(g, cls, cf_anchor=None)
        except Exception as e:                       # a crash is a recorded outcome
            row.update(status="failed",
                       status_reason=f"{type(e).__name__}: {e}")
            return row, live
        if res is None or res.get("skipped"):
            row.update(status="failed",
                       status_reason=((res or {}).get("skip_reason")
                                      or "no periodic tick lattice could be recovered"))
            return row, live

        s = summarise(res)
        s_plain = summarise(plain) if plain else None
        pxcm = float(s["pxcm"])
        rec = res.get("rec") or {}
        groups = list(rec["groups"]) if rec.get("ok") else []
        tr = res.get("trans")
        # A transition ruler resolves through its own 10:1 split rather than the
        # unit ladder, so surface those two readings as groups for the comb overlay.
        if not groups and tr and s.get("trans_used"):
            for pk, uk, spx in (("pxcm_fine", "unit_fine", tr["P_fine"]),
                                ("pxcm_coarse", "unit_coarse", tr["P_coarse"])):
                if tr.get(uk):
                    groups.append(dict(unit=tr[uk], system=system_of(tr[uk]), mults=[1],
                                       n_ticks=0, est_pxcm=tr[pk], spacing=spx,
                                       averaged=False, spread_within=0.0,
                                       merge_ok=True, anchor=s["first_tick_x"]))

        lab = build_masks(res)
        band = s["band"]
        _, L_cm, L_in = ruler_bars(res["rot"], s["first_tick_x"], pxcm, band)

        row.update(
            status="measured", status_reason=None,
            pxcm=_num(pxcm), pxcm_original=_num(pxcm / (work_scale or 1.0)),
            pxcm_no_anchor=_num(None if s_plain is None else s_plain["pxcm"]),
            anchor_changed_read=int(bool(s_plain is not None and
                                         abs(s_plain["pxcm"] / max(pxcm, 1e-9) - 1) > 0.03)),
            pct_vs_anchor=_num(_pct(pxcm, anchor_work)),
            rotation_deg=_num(s["angle"]),
            band_y0=int(band[0]), band_y1=int(band[1]),
            band_kind=s["band_kind"], band_periodicity=_num(s["band_score"]),
            P_acf_px=_num(s["P_acf"]), P_base_px=_num(s["P"]),
            period_read_as=s["period_name"],
            ladder="|".join(map(str, s["ladder"])) or None,
            first_tick_x=_num(s["first_tick_x"]),
            n_ticks=_num(s["n_ticks"]), n_kept=_num(s["kept"]),
            keep_ratio=_num(s["keep_ratio"]), occupancy=_num(s["occupancy"]),
            salvaged=_num(s["salvaged"]), lattice_fallback=_num(s["fallback"]),
            layout=s["layout"], relation=s["relation"],
            systems="+".join(s["systems"]) or None,
            n_units=len(groups),
            units="|".join(g_["unit"] for g_ in groups) or None,
            unit_estimates="|".join(f"{g_['est_pxcm']:.3f}" for g_ in groups) or None,
            n_corroborating=_num(s["n_corroborating"]),
            rec_agree=_num(s["rec_agree"]),
            cross_unit_spread_pct=_num(None if s["rec_spread"] is None
                                       else 100.0 * s["rec_spread"]),
            # index rather than unpack: the tuple grew a 7th element (the rejection reason) and
            # records written before that are still 6 long.
            cf_contributions=";".join(
                f"{c[0]}:{c[1]}:{c[2]}:{c[3]}:{'U' if c[4] else 'R'}"
                for c in s["cf_contributions"]) or None,
            cf_n_used=_num(s["cf_n_used"]), cf_n_rejected=_num(s["cf_n_rejected"]),
            cf_spread_pct=_num(None if s["cf_spread"] is None
                               else 100.0 * s["cf_spread"]),
            cf_fallback=_num(s["cf_fallback"]),
            implied_len_cm=_num(s["implied_len_cm"]),
            bands_evaluated=_num(s["bands_evaluated"]),
            n_band_clusters=_num(s["n_band_clusters"]),
            n_bands_agree=_num(s["n_bands_agree"]),
            band_spread_pct=_num(None if s["band_spread"] is None
                                 else 100.0 * s["band_spread"]),
            band_cf=_num(s["band_cf"]), bands_thin=_num(s["bands_thin"]),
            shadows="|".join(f"{d['ratio']}x@{d['cf']}"
                             for d in (s["shadows"] or [])) or None,
            stacked_pair=(None if not s.get("stacked_pair")
                          else f"{s['stacked_pair']['ratio']}x"),
            trans_ratio=_num(None if not tr else tr["ratio"]),
            trans_ratio_err_pct=_num(None if not tr else 100.0 * tr["ratio_err"]),
            trans_expect=_num(None if not tr else tr["expect"]),
            trans_split_x=_num(None if not tr else tr["split_x"]),
            trans_order=(None if not tr else tr["order"]),
            trans_used=_num(s.get("trans_used")),
            trans_P_fine=_num(None if not tr else tr.get("P_fine")),
            trans_P_coarse=_num(None if not tr else tr.get("P_coarse")),
            trans_unit_fine=(None if not tr else tr.get("unit_fine")),
            trans_unit_coarse=(None if not tr else tr.get("unit_coarse")),
            trans_pxcm_fine=_num(None if not tr else tr.get("pxcm_fine")),
            trans_pxcm_coarse=_num(None if not tr else tr.get("pxcm_coarse")),
            groups_json=_json([
                dict(unit=g_["unit"], system=g_.get("system"),
                     mults=list(g_.get("mults") or []),
                     est_pxcm=g_.get("est_pxcm"), spacing=g_.get("spacing"),
                     n_ticks=g_.get("n_ticks"), averaged=g_.get("averaged"),
                     spread_within=g_.get("spread_within"),
                     merge_ok=g_.get("merge_ok"),
                     anchor=g_.get("anchor")) for g_ in groups]),
            summary_json=_json(s), rec_json=_json(rec if rec else None),
            px_per_cm_bar=_num(L_cm), px_per_inch_bar=_num(L_in),
            rot_path=self._save_gray(res["rot"], det, "rot"),
            tick_mask_path=self._save_gray(lab, det, "ticks"))

        live.update(rot=res["rot"], lab=lab, res=res, s=s, s_plain=s_plain,
                    groups=groups)
        return row, live

    # ---- QC ----------------------------------------------------------------
    def render_qc(self, record, reconcile=None, live=None):
        """Rebuild the sheet's QC panel from the RECORD alone.

        `live` is an optional in-memory shortcut for the rasters we just computed;
        when it is absent (the after-the-fact reconstruction path) every raster is
        loaded from the paths in the record. Both paths run the same drawing code,
        so a panel that renders live is a panel that renders from the DB.
        """
        img = record["image"]
        crops = record["crops"]
        if not crops:
            return None
        by_det = {l["detection_id"]: l for l in (live or [])}
        pr = reconcile or self._reconcile_from_record(record)

        panels, entries = [], []
        for r in crops:
            lv = by_det.get(r["detection_id"]) or {}
            # Always the STORED thumbnail, never the in-memory full-res tile: the
            # live panel and the panel rebuilt from the DB must be the same image.
            tile = _load_rgb(r.get("tile_four_path"))
            qrow = dict(image_name=img.get("image_name"),
                        detection_id=r["detection_id"],
                        det_conf=float(r.get("det_conf") or 0.0),
                        cls_conf=r.get("cls_conf"),
                        work_scale=img.get("work_scale"),
                        anchor=img.get("mp_anchor_working"),
                        anchor_original=img.get("mp_anchor_original"),
                        anchor_frame=img.get("anchor_frame"),
                        anchor_formula=img.get("anchor_formula"),
                        original_mp=img.get("original_mp"),
                        ruler_class=r.get("ruler_class"))
            cls = r["ruler_class"]

            if r["status"] != "measured":
                why = (f"NOT A SUPPORTED CLASS -- no conversion factor is attempted.\n"
                       f"{r['status_reason']}" if r["status"] == "skipped_class"
                       else f"CF DETERMINATION FAILED -- {r['status_reason']}")
                panels.append(QC.class_section(qrow, cls, r.get("crop_path"),
                                               failure=why, tile_img=tile))
                entries.append(dict(key=f"det{r['detection_id']}", rot=None,
                                    verdict=r.get("verdict")))
                continue

            rot = lv.get("rot")
            if rot is None:
                rot = _load_gray(r.get("rot_path"))
            lab = lv.get("lab")
            if lab is None:
                lab = _load_gray(r.get("tick_mask_path"))
            s = _summary_from_row(r)
            # A measured crop whose rasters or summary have gone missing is shown
            # as exactly that. It is never silently dropped from the stack, or the
            # panel would quietly disagree with n_ruler_crops in the image row.
            if rot is None or s is None:
                miss = [n for n, v in (("rot_path", rot), ("summary_json", s))
                        if v is None]
                panels.append(QC.class_section(
                    qrow, cls, r.get("crop_path"), tile_img=tile,
                    failure="QC CANNOT BE REBUILT for this crop -- the stored "
                            f"{', '.join(miss)} is missing. The measurement itself "
                            f"stands: {r['pxcm']:.2f} px/cm."))
                entries.append(dict(key=f"det{r['detection_id']}", rot=None,
                                    verdict=r.get("verdict")))
                continue
            band = tuple(s["band"])
            pxcm = float(r["pxcm"])
            groups = _groups_from_json(r.get("groups_json"))
            s_plain = (None if r.get("pxcm_no_anchor") is None
                       else dict(pxcm=float(r["pxcm_no_anchor"])))
            res_like = dict(rot=rot, rec=_rec_from_row(r),
                            trans=s.get("trans"), skipped=False)

            ov_mask = mask_overlay(rot, lab if lab is not None else
                                   np.zeros(rot.shape, np.uint8))
            ov_comb = comb_overlay(rot, groups, pxcm, band)
            ov_bars, L_cm, L_in = ruler_bars(rot, r.get("first_tick_x") or 0, pxcm, band)
            panel, _flip = QC.build_panel(qrow, s, s_plain, res_like, groups, pxcm,
                                          ov_mask, ov_comb, ov_bars, (L_cm, L_in))
            panel = QC.stack_sections([
                QC.class_section(qrow, cls, r.get("crop_path"), tile_img=tile), panel])
            panels.append(panel)
            entries.append(dict(key=f"det{r['detection_id']}", rot=rot,
                                x0=r.get("first_tick_x") or 0, pxcm=pxcm, band=band,
                                cf=pxcm, ruler_class=cls, verdict=r.get("verdict")))

        if not panels:
            return None
        # Emitted whenever the sheet has more than one ruler panel, even if none of
        # them yielded a strip to draw: the block still carries the per-crop verdict
        # table and the reason no CF was certified, which is the audit trail for a
        # sheet whose rulers were all rejected.
        # Also rendered for a SINGLE-ruler sheet whose CF was withheld: that sheet is about to be
        # measured with the MP anchor instead, and the block is the only place that says so and
        # shows the predicted scale against the ruler.
        recon = None
        drawable = [e for e in entries if e.get("rot") is not None]
        withheld = pr.get("cf_px_per_cm") is None and img.get("mp_anchor_working")
        # Whether the stage actually SUBSTITUTED the MP anchor (use_CF_predicted_by_MP on) or left
        # the sheet without a CF. Read from the record, so a panel rebuilt later says the same.
        applied = img.get("cf_source") == CF_SOURCE_MP
        if drawable and (len(panels) > 1 or withheld):
            recon = QC.build_recon_section(img.get("image_name"), drawable, pr,
                                           img.get("mp_anchor_working"),
                                           anchor_formula=img.get("anchor_formula"),
                                           fallback_applied=applied)
        # The CF summary is unconditional: every panel ends with the two numbers, whether the
        # sheet published a measured CF or fell back to the prediction.
        cf_summary = QC.build_cf_summary_section(
            pr.get("cf_px_per_cm"), img.get("mp_anchor_working"),
            img.get("anchor_formula_symbolic"), fallback_applied=applied)
        return QC.stack_parent(panels, recon, cf_summary=cf_summary)

    def _reconcile_from_record(self, record):
        """The reconciliation dict the recon section needs, rebuilt from the row."""
        img = record["image"]
        return dict(
            cf_px_per_cm=img.get("cf_px_per_cm"),
            cf_px_per_cm_measured=img.get("cf_px_per_cm_measured"),
            ok=(img.get("status") == "published"),
            confidence=img.get("confidence"),
            confidence_reasons=json.loads(img.get("confidence_reasons_json") or "[]"),
            method=img.get("method"),
            spread=(None if img.get("peer_spread_pct") is None
                    else img["peer_spread_pct"] / 100.0),
            pct_vs_anchor=img.get("pct_vs_anchor"),
            reason=None,
            per_crop=[dict(key=f"det{r['detection_id']}", cf=r.get("pxcm"),
                           ruler_class=r.get("ruler_class"),
                           weight=r.get("n_kept"), verdict=r.get("verdict"),
                           note=r.get("verdict_note"),
                           pct_vs_parent=r.get("pct_vs_parent"),
                           pct_vs_anchor=r.get("pct_vs_anchor"))
                      for r in record["crops"]])

    # ---- persistence -------------------------------------------------------
    @staticmethod
    def write_db(con: sqlite3.Connection, record: dict) -> None:
        """Idempotent upsert of one parent image's whole trail.

        Delete-then-insert per specimen inside the caller's transaction, matching
        how ProjectDB writes archival_detection: a re-run replaces the sheet's rows
        wholesale, so a torn or repeated write can never leave two versions behind
        or blend a previous run's crops into this one's.
        """
        img, crops = record["image"], record["crops"]
        sid = img["specimen_id"]
        con.execute("DELETE FROM ruler_CF_lattice_crop WHERE specimen_id=?", (sid,))
        con.execute("DELETE FROM ruler_CF_lattice WHERE specimen_id=?", (sid,))
        icols = [c for c in image_columns() if c in img]
        con.execute(f"INSERT INTO ruler_CF_lattice ({','.join(icols)}) "
                    f"VALUES ({','.join('?' * len(icols))})",
                    [_num(img[c]) for c in icols])
        ccols = crop_columns()
        con.executemany(
            f"INSERT INTO ruler_CF_lattice_crop ({','.join(ccols)}) "
            f"VALUES ({','.join('?' * len(ccols))})",
            [[_num(r.get(c)) for c in ccols] for r in crops])

    @staticmethod
    def read_record(con: sqlite3.Connection, specimen_id: int) -> dict | None:
        """The inverse of write_db: the record as the DB holds it, for re-rendering."""
        con.row_factory = sqlite3.Row
        i = con.execute("SELECT * FROM ruler_CF_lattice WHERE specimen_id=?",
                        (specimen_id,)).fetchone()
        if i is None:
            return None
        cs = con.execute("SELECT * FROM ruler_CF_lattice_crop WHERE specimen_id=? "
                         "ORDER BY crop_index", (specimen_id,)).fetchall()
        return dict(image=dict(i), crops=[dict(c) for c in cs])

    # ---- rasters -----------------------------------------------------------
    def _dir(self, sub):
        d = self.artifact_dir / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_gray(self, arr, det, kind):
        if arr is None or not self.write_rasters:
            return None
        p = self._dir(kind) / f"det{det}__{kind}.png"
        Image.fromarray(np.asarray(arr).astype(np.uint8)).save(p)
        return str(p)


# ------------------------------------------------------- record -> QC helpers --
def _admissible(cls):
    from .units import admissible_units
    try:
        return admissible_units(cls)
    except Exception:
        return []


def _load_gray(p):
    if not p or not os.path.exists(p):
        return None
    return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)


def _load_rgb(p):
    if not p or not os.path.exists(p):
        return None
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _groups_from_json(js):
    try:
        return json.loads(js) if js else []
    except (TypeError, ValueError):
        return []


def _split(v, cast=str):
    return [cast(x) for x in v.split("|")] if v else []


def _summary_from_row(r):
    """The engine's `summarise()` output as stored. No field is reconstructed.

    Returns None when the row carries no summary, and the caller renders the crop
    as unrenderable rather than drawing a panel out of invented defaults -- a
    plausible-looking panel built from placeholders is worse than no panel, because
    it cannot be told apart from a real one.
    """
    js = r.get("summary_json")
    if not js:
        return None
    try:
        s = json.loads(js)
    except (TypeError, ValueError):
        return None
    if isinstance(s.get("band"), list):          # JSON has no tuples
        s["band"] = tuple(s["band"])
    if isinstance(s.get("stacked_pair"), list):
        s["stacked_pair"] = tuple(s["stacked_pair"])
    s["cf_contributions"] = [tuple(c) for c in (s.get("cf_contributions") or [])]
    return s


def _rec_from_row(r):
    js = r.get("rec_json")
    if not js:
        return None
    try:
        return json.loads(js)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------- driver ----
def load_specimens(db: Path):
    """Every specimen in an LM3 run, with its Ruler boxes. Shape of the caller LM3
    will replace this with -- kept here so the class can be exercised standalone."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    specs = {r["specimen_id"]: dict(r) for r in con.execute(
        "select specimen_id, image_name, width working_width, height working_height, "
        "work_scale, original_mp, cf_px_per_cm_predicted_by_mp from specimen")}
    crops = {}
    for r in con.execute(
            "select d.detection_id, d.specimen_id, d.conf det_conf, d.crop_path, "
            "d.x1, d.y1, d.x2, d.y2, rc.unit_type ruler_class, rc.conf cls_conf "
            "from archival_detection d "
            "left join ruler_classification rc using(detection_id) "
            "where d.cls_name='Ruler' order by d.detection_id"):
        crops.setdefault(r["specimen_id"], []).append(dict(r))
    con.close()
    return [(s, crops.get(sid, [])) for sid, s in sorted(specs.items())]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="herbcode2")
    ap.add_argument("--artifact-dir", default=None)
    ap.add_argument("--qc-dir", default=None)
    ap.add_argument("--out-db", default=None,
                    help="write the two tables here (default: the run DB itself)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-qc", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="prove statelessness, idempotency and DB-sufficiency")
    a = ap.parse_args()

    run_db = HERE / "lm3_runs" / a.run_name / f"{a.run_name}.sqlite"
    if not run_db.exists():
        sys.exit(f"no run DB at {run_db}")
    art = Path(a.artifact_dir or (HERE / "lm3_runs" / a.run_name / "ruler_cf_lattice"))
    eng = RulerCFLattice(art, qc_dir=a.qc_dir, write_qc=not a.no_qc)

    if a.selftest:
        return _selftest(eng, run_db)

    items = load_specimens(run_db)
    if a.limit:
        items = items[:a.limit]
    out_db = Path(a.out_db) if a.out_db else run_db
    con = sqlite3.connect(out_db)
    eng.ensure_schema(con)
    n = {"published": 0, "withheld": 0, "no_reading": 0, "no_ruler": 0}
    t0 = time.time()
    for i, (spec, crops) in enumerate(items, 1):
        rec = eng.process_specimen(spec, crops)
        with con:
            eng.write_db(con, rec)
        im = rec["image"]
        n[im["status"]] = n.get(im["status"], 0) + 1
        cf = im["cf_px_per_cm"]
        print(f"  [{i:3d}/{len(items)}] {im['status']:<10s} {im['confidence']:<6s} "
              f"CF={'-' if cf is None else '%.2f' % cf:>8s} "
              f"({im['n_measured']}/{im['n_ruler_crops']} crops)  "
              f"{str(im['image_name'])[:44]}")
    con.close()
    print(f"\n{len(items)} specimens in {time.time()-t0:.0f}s -> {out_db}")
    print(f"  {n}")
    print(f"  rasters: {art}")
    return 0


def _selftest(eng, run_db):
    """The three properties the LM3 integration depends on."""
    items = load_specimens(run_db)
    multi = [x for x in items if len(x[1]) > 1][:2] or items[:2]
    single = [x for x in items if len(x[1]) == 1][:1]
    probe = (multi + single)[:3]
    if len(probe) < 2:
        sys.exit("need at least 2 specimens to self-test")

    def strip(rec):                       # everything except wall-clock
        r = json.loads(_json({k: v for k, v in rec["image"].items()
                              if k not in ("created_at", "runtime_ms")}))
        return _json(dict(image=r, crops=json.loads(_json(rec["crops"]))))

    print("1. NO CARRY-OVER BETWEEN PARENT IMAGES")
    before = _json(sorted(eng.__dict__.keys()))
    a1 = eng.process_specimen(*probe[0])
    _ = eng.process_specimen(*probe[1])
    a2 = eng.process_specimen(*probe[0])
    same = strip(a1) == strip(a2)
    print(f"   A -> B -> A reproduces A exactly: {same}")
    print(f"   instance attributes unchanged   : {before == _json(sorted(eng.__dict__.keys()))}")
    if not same:
        sys.exit("!! FAILED: a record changed after processing another specimen")

    print("\n2. DB ROUND-TRIP IS LOSSLESS")
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE specimen (specimen_id INTEGER PRIMARY KEY)")
    con.execute("CREATE TABLE archival_detection (detection_id INTEGER PRIMARY KEY)")
    eng.ensure_schema(con)
    with con:
        eng.write_db(con, a1)
        eng.write_db(con, a1)              # twice: must not duplicate
    back = RulerCFLattice.read_record(con, a1["image"]["specimen_id"])
    n_rows = con.execute("select count(*) from ruler_CF_lattice_crop").fetchone()[0]
    print(f"   re-writing the same record leaves {n_rows} crop rows "
          f"(expected {len(a1['crops'])}): {n_rows == len(a1['crops'])}")
    miss = [c for c in image_columns()
            if c in a1["image"] and _num(a1["image"][c]) != back["image"].get(c)]
    print(f"   image columns that did not round-trip: {miss or 'none'}")

    print("\n3. THE QC PANEL REBUILDS FROM THE DB ALONE")
    live = eng.render_qc(a1)
    from_db = eng.render_qc(back)
    if live is None or from_db is None:
        print("   (this specimen renders no panel)")
    else:
        la, fa = np.asarray(live), np.asarray(from_db)
        ok = la.shape == fa.shape and np.array_equal(la, fa)
        print(f"   live panel {la.shape} vs DB-rebuilt {fa.shape}: "
              f"{'PIXEL-IDENTICAL' if ok else 'DIFFERENT'}")
        if not ok and la.shape == fa.shape:
            print(f"   max abs delta {int(np.abs(la.astype(int)-fa.astype(int)).max())}")
    con.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
