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
import inspect
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

from leafmachine3.core.records import CF_SOURCE_FP, CF_SOURCE_MP, CF_SOURCE_RULER

from . import fieldprism as FP
from .analysis import analyse, build_masks, summarise
from .sheet_cf import _harmonic_of, reconcile_parent
from .units import is_fieldprism, is_skipped, spec_of, system_of
from . import qc as QC
from .qc import mask_overlay, comb_overlay, ruler_bars

HERE = Path(__file__).resolve().parent

# Bump when a change alters numeric output, so a DB row always says which engine
# produced it and a mixed-version project is detectable with one GROUP BY.
ENGINE_VERSION = "lattice-2026.10.09-fp"

CM_PER_INCH = 2.54

# Reconciliation weight of one FieldPrism marker. A lattice crop's weight is its kept-tick count;
# a marker has no ticks, so it gets a fixed, well-supported weight: far above
# DISTANCE_WEIGHT_FLOOR / DISAGREEMENT_FLOOR (6), so a marker always counts as real evidence, and
# comparable to a good 4-10 cm lattice read, so neither kind silently swamps the other.
FP_RECONCILE_WEIGHT = 40


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

    qc_image_path         TEXT,               -- stacked QC panel for the whole sheet

    -- Which reference every reading on this sheet was checked against. 'megapixels' = the MP
    -- prediction (mp_anchor_working); 'fieldprism' = the sheet's FieldPrism markers (the MP
    -- prediction is then recorded in mp_anchor_* for audit only and never used, not even as a
    -- fallback, because FieldPrism sheets come in several sizes); 'none' = no anchor at all.
    anchor_source         TEXT,               -- megapixels | fieldprism | none
    anchor_cf_working     REAL,               -- the anchor actually used, working frame
    fp_detected           INTEGER             -- 1 = the sheet has FP crops and the FieldPrism path ran
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

-- ruler_FP_marker : ONE row per FieldPrism (FP) ruler crop -- the app-style square finder's
-- verdict on that marker. Square centers (TL/TR/C/BL, and the PREDICTED center of the empty BR
-- cell) are WORKING-frame px, so the Reporter can label them exactly like the FieldPrism app.
-- Stores no files: crop_path stays in archival_detection / ruler_CF_lattice_crop.
CREATE TABLE IF NOT EXISTS ruler_FP_marker (
    fp_marker_id     INTEGER PRIMARY KEY,
    specimen_id      INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    detection_id     INTEGER NOT NULL REFERENCES archival_detection(detection_id) ON DELETE CASCADE,
    crop_index       INTEGER,                 -- same index as the ruler_CF_lattice_crop row
    det_conf         REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    roi_x0 INTEGER, roi_y0 INTEGER, roi_x1 INTEGER, roi_y1 INTEGER,
    status           TEXT NOT NULL,           -- measured | failed | unreadable
    status_reason    TEXT,
    valid            INTEGER,                 -- 1 = passed the per-marker geometric checks
    validation_json  TEXT,
    verdict          TEXT,                    -- used | rejected | skipped
    verdict_note     TEXT,
    n_peaks          INTEGER,
    holes_filled     INTEGER,
    peak_area_ratio  REAL,
    tl_x REAL, tl_y REAL, tr_x REAL, tr_y REAL, c_x REAL, c_y REAL, bl_x REAL, bl_y REAL,
    br_x REAL, br_y REAL,                     -- PREDICTED center of the empty BR cell = TR + BL - TL
    pitch_h_px       REAL,
    pitch_v_px       REAL,
    pxcm             REAL,
    pxcm_original    REAL,
    pct_vs_fp        REAL,
    orientation_deg  INTEGER,                 -- app vote 0|90|180|270
    sheet_corner     TEXT,                    -- TL|TR|BL|BR or NULL
    UNIQUE (detection_id)
);
CREATE INDEX IF NOT EXISTS ix_rfpm_spec ON ruler_FP_marker (specimen_id, crop_index);

-- ruler_FP_sheet : ONE row per sheet that has FP crops -- which printed FieldPrism sheet the
-- markers belong to (catalog in fieldprism_sheets.json), the similarity fit, the reconstructed
-- (inferred) markers and the FieldPrism anchor CF. The *_json columns are JSON text.
CREATE TABLE IF NOT EXISTS ruler_FP_sheet (
    specimen_id      INTEGER PRIMARY KEY REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    catalog_version  TEXT,
    n_fp_detected    INTEGER,
    n_fp_measured    INTEGER,
    n_fp_valid       INTEGER,
    n_fp_used        INTEGER,
    n_fp_rejected    INTEGER,
    n_fp_inferred    INTEGER,
    sheet_status     TEXT NOT NULL,           -- identified | ambiguous | undetermined | unrecognized
    sheet_type       TEXT,
    sheet_label      TEXT,
    corners_ambiguous INTEGER,
    sheet_candidates_json TEXT,
    orientation_deg  INTEGER,
    fit_rotation_deg REAL,
    fit_scale_px_per_mm REAL,
    fit_tx           REAL,
    fit_ty           REAL,
    fit_rms_mm       REAL,
    fit_max_mm       REAL,
    fit_scale_dev_mm REAL,
    fit_cost_mm      REAL,
    cf_px_per_cm_sheet_fit REAL,
    cf_px_per_cm_marker_mean REAL,
    cf_px_per_cm_fp  REAL,
    cf_source_detail TEXT,                    -- sheet_fit | marker_mean
    fp_peer_spread_pct REAL,
    fp_confidence    TEXT,
    fp_reasons_json  TEXT,
    corners_json     TEXT,
    page_corners_json TEXT,
    fpfit_margins_json TEXT
);

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
_FP_MARKER_COLS = None
_FP_SHEET_COLS = None


def _col_decls(table: str) -> list[tuple[str, str]]:
    """(name, declaration) of every column parsed out of SCHEMA_SQL -- the DDL is the single source."""
    body = SCHEMA_SQL.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1]
    out = []
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
                out.append((decl.split()[0], decl))
    return out


def _cols(table: str) -> list[str]:
    """Column names parsed out of SCHEMA_SQL -- the DDL is the single source."""
    return [name for name, _ in _col_decls(table)]


def schema_tables() -> list[str]:
    """Every table SCHEMA_SQL declares, in declaration order."""
    return [part.split("(", 1)[0].strip()
            for part in SCHEMA_SQL.split("CREATE TABLE IF NOT EXISTS ")[1:]]


def migration_columns(table: str) -> list[tuple[str, str]]:
    """(name, ALTER-safe type) for every column of `table`, for adding missing ones to an old DB.

    `ALTER TABLE ... ADD COLUMN` cannot add a PRIMARY KEY or UNIQUE column, nor a NOT NULL one
    without a default, so only the bare type survives; the constraints are the table's business
    when it is created fresh, and every column here is written by the engine on every row."""
    out = []
    for name, decl in _col_decls(table):
        parts = decl.split()
        typ = parts[1] if len(parts) > 1 and parts[1].isalpha() else ""
        out.append((name, typ.upper()))
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


def fp_marker_columns() -> list[str]:
    global _FP_MARKER_COLS
    if _FP_MARKER_COLS is None:
        _FP_MARKER_COLS = [c for c in _cols("ruler_FP_marker") if c != "fp_marker_id"]
    return _FP_MARKER_COLS


def fp_sheet_columns() -> list[str]:
    global _FP_SHEET_COLS
    if _FP_SHEET_COLS is None:
        _FP_SHEET_COLS = _cols("ruler_FP_sheet")
    return _FP_SHEET_COLS


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
                 tile_format="jpg", tile_quality=80, tile_store_px=512,
                 fp_enabled=True, fp_peer_tol=FP.FP_PEER_TOL,
                 fp_anchor_tol=FP.FP_ANCHOR_TOL, fp_allow_single_marker=True):
        self.artifact_dir = Path(artifact_dir)
        self.qc_dir = Path(qc_dir) if qc_dir else self.artifact_dir / "qc"
        self.write_qc = bool(write_qc)
        self.write_rasters = bool(write_rasters)
        self.squarify_sz = int(squarify_sz)
        self.squarify_method = str(squarify_method)
        self.anchor_tol = float(anchor_tol)
        self.min_frame_cm = min_frame_cm          # None = ruler_sheet_cf's own default
        # FieldPrism (FP) markers. When enabled, a sheet with FP crops is anchored on the FieldPrism
        # geometry instead of the MP prediction: fp_peer_tol is how closely the markers must agree
        # with each other, fp_anchor_tol how closely every reading (FP or tick ruler) must agree
        # with that anchor in the sheet reconciliation, and fp_allow_single_marker whether one
        # valid marker alone may carry a published CF.
        self.fp_enabled = bool(fp_enabled)
        self.fp_peer_tol = float(fp_peer_tol)
        self.fp_anchor_tol = float(fp_anchor_tol)
        self.fp_allow_single_marker = bool(fp_allow_single_marker)
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
            fieldprism=dict(
                enabled=self.fp_enabled, peer_tol=self.fp_peer_tol,
                anchor_tol=self.fp_anchor_tol,
                allow_single_marker=self.fp_allow_single_marker,
                reconcile_weight=FP_RECONCILE_WEIGHT,
                catalog_version=_fp_catalog_version()),
        )

    # ---- schema ------------------------------------------------------------
    def ensure_schema(self, con: sqlite3.Connection) -> None:
        con.executescript(self.SCHEMA_SQL)

    # ---- the one entry point ----------------------------------------------
    def process_specimen(self, specimen: dict, crops: list) -> dict:
        """Measure every ruler crop of ONE parent image and fuse them into one CF.

        specimen : {specimen_id, image_name, work_scale, working_width,
                    working_height, cf_px_per_cm_predicted_by_mp, working_path}
        crops    : [{detection_id, crop_path, ruler_class, cls_conf, det_conf,
                     x1, y1, x2, y2}]  -- every Ruler box on the sheet, in any order

        Returns {"schema_version", "image": {...}, "crops": [{...}],
                 "fp_sheet": {...}|None, "fp_markers": [{...}], "qc_image": PIL.Image|None}.
        The dict sections use exactly the column names in SCHEMA_SQL.

        FIELDPRISM. When FP is enabled and any crop is classified FP, the working image
        (``working_path``) is read ONCE and every FP crop is measured by fieldprism.py FIRST. The
        FieldPrism CF (the sheet fit, or the agreeing markers' mean) then replaces the MP
        prediction as the anchor for EVERYTHING on the sheet -- the tick-lattice reads of any
        non-FP rulers and the reconciliation -- because FieldPrism sheets come in several sizes
        and the MP model assumes one. The MP value is still recorded in mp_anchor_* for audit.
        A sheet without FP crops (or with FP disabled) runs exactly as before.
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

        # ---- FieldPrism FIRST: its geometry becomes the sheet's anchor -------
        # On a sheet with FP crops the MP prediction is never an anchor (FieldPrism sheets come in
        # several sizes, so a resolution-based guess is meaningless there). The anchor is the
        # FieldPrism CF when the markers form an agreeing cluster (confidence high or medium);
        # markers that disagree with each other (low) or that all failed give NO anchor at all.
        fp_idx = ([i for i, c in enumerate(ordered) if is_fieldprism(_cls_of(c))]
                  if self.fp_enabled else [])
        fp = None
        fp_authoritative = False
        if fp_idx:
            fp = self._analyze_fieldprism(specimen, [ordered[i] for i in fp_idx])
            fp_authoritative = bool(fp["anchor_cf"]) and fp["confidence"] in ("high", "medium")
            anchor_used = fp["anchor_cf"] if fp_authoritative else None
            anchor_source = "fieldprism" if anchor_used else "none"
        else:
            anchor_used = anchor_work
            anchor_source = "megapixels" if anchor_work else "none"
        fp_by_det = {m["detection_id"]: m for m in (fp["markers"] if fp else [])}

        crop_rows, live = [], []
        for i, c in enumerate(ordered):
            row, live_entry = self._measure_crop(i, c, specimen, anchor_used,
                                                 fp_marker=fp_by_det.get(c.get("detection_id")))
            crop_rows.append(row)
            live.append(live_entry)

        # ---- fuse the sheet's crops into ONE CF ----------------------------
        extra_reasons = []
        if fp_authoritative:
            # FieldPrism DECIDES. Its own peer step already judged the markers against each other
            # (the median-based cluster); feeding them through reconcile's greedy 3% clustering a
            # second time could split an agreeing cluster or absorb a marker FieldPrism rejected.
            # Every other ruler is judged against the FieldPrism CF, which is also the value.
            pr = self._fieldprism_decision(crop_rows, fp, float(anchor_used))
        else:
            if fp is not None:
                # No usable FieldPrism anchor: plain peer reconciliation with no anchor at all. The
                # MIN_FRAME_CM drop is an MP-model guard and does not apply (frame_width_px=None).
                kw = dict(anchor=None, anchor_tol=self.fp_anchor_tol,
                          frame_width_px=None, anchor_name="FieldPrism anchor")
            else:
                kw = dict(anchor=anchor_work, anchor_tol=self.anchor_tol,
                          frame_width_px=frame_w)
            pr = reconcile_parent(
                [dict(key=f"det{r['detection_id']}", cf=r["pxcm"],
                      weight=(FP_RECONCILE_WEIGHT if _is_fp_row(r) else (r["n_kept"] or 0)),
                      ruler_class=r["ruler_class"],
                      implied_len_cm=(None if _is_fp_row(r) else r.get("implied_len_cm")),
                      skipped=(r["status"] != "measured"),
                      skip_reason=r["status_reason"]) for r in crop_rows], **kw)

        by_key = {c.get("key"): c for c in pr.get("per_crop", [])}
        for r in crop_rows:
            v = by_key.get(f"det{r['detection_id']}") or {}
            r["verdict"] = v.get("verdict")
            r["verdict_note"] = v.get("note")
            r["pct_vs_parent"] = _num(v.get("pct_vs_parent"))

        fp_used = any(_is_fp_row(r) and r["verdict"] == "used" for r in crop_rows)
        if fp is not None:
            extra_reasons = [f"FieldPrism: {x}" for x in fp["sheet"].get("fp_reasons") or []]
            fp_conf = fp["confidence"]
            # Without an anchor, reconcile can still certify FP markers by their mutual agreement
            # alone -- but FieldPrism already found those markers disagree, so a CF they carry is
            # withheld.
            if (not fp_authoritative and fp_used and fp_conf != "high"
                    and pr.get("confidence") == "high"):
                pr = dict(pr, confidence="medium", cf_px_per_cm=None, ok=False)
                extra_reasons.append(
                    f"confidence capped at medium: the FieldPrism markers are only "
                    f"{fp_conf or 'not'} confidence, so the CF they carry is withheld")

        if extra_reasons:       # the live QC panel must read exactly what the DB row will say
            pr = dict(pr, confidence_reasons=list(pr.get("confidence_reasons") or [])
                      + extra_reasons)
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
        if cf is None:
            cf_source = None                                    # the stage stamps CF_SOURCE_MP
        else:
            cf_source = CF_SOURCE_FP if fp_used else CF_SOURCE_RULER
        if cf is not None:
            fallback = ""
        elif fp is not None:
            fallback = "none"          # the MP prediction is never a fallback on a FieldPrism sheet
        else:
            fallback = "mp_anchor" if anchor_work else "none"

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
            cf_source=cf_source,
            fallback=fallback,
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
            anchor_source=anchor_source,
            anchor_cf_working=_num(anchor_used),
            fp_detected=int(fp is not None),
        )

        fp_sheet, fp_markers = None, []
        if fp is not None:
            fp_sheet = FP.sheet_row(fp, spec_id, _fp_catalog_version())
            idx_of = {r["detection_id"]: r["crop_index"] for r in crop_rows}
            row_of = {r["detection_id"]: r for r in crop_rows}
            for m in fp["markers"]:
                mr = FP.marker_row(m, spec_id, idx_of.get(m["detection_id"]), work_scale)
                # When FieldPrism decided the sheet, the crop verdicts ARE its verdicts. Otherwise
                # (markers that disagree, no anchor) the sheet reconciliation has the last word on
                # a marker FieldPrism used; an FP-level rejection or skip is final either way.
                cr = row_of.get(m["detection_id"]) or {}
                if mr["verdict"] == "used" and cr.get("verdict"):
                    mr["verdict"] = cr["verdict"]
                    if cr["verdict"] != "used":
                        mr["verdict_note"] = cr.get("verdict_note")
                fp_markers.append(mr)
            fp_markers.sort(key=lambda r: (r["crop_index"] is None, r["crop_index"] or 0))
            # The sheet row's counters must agree with the marker rows they summarize.
            fp_sheet["n_fp_used"] = sum(1 for r in fp_markers if r["verdict"] == "used")
            fp_sheet["n_fp_rejected"] = sum(1 for r in fp_markers if r["verdict"] == "rejected")

        record = dict(schema_version=self.ENGINE_VERSION,
                      image=image_row, crops=crop_rows,
                      fp_sheet=fp_sheet, fp_markers=fp_markers)

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

    # ---- FieldPrism ----------------------------------------------------------
    def _fieldprism_decision(self, crop_rows, fp, anchor):
        """The sheet decision when FieldPrism markers form an agreeing cluster.

        Returns a dict shaped like `reconcile_parent`'s, so the image row and the QC panels read
        it the same way. The VALUE is the FieldPrism CF (the sheet fit when the sheet type is
        identified -- 146-249 mm baselines, the scale the app rectified the image to -- else the
        agreeing markers' mean). FP markers keep the FieldPrism step's verdicts; every other ruler
        is a witness, used if it reads within ``fp_anchor_tol`` of that CF and rejected otherwise.
        Only FieldPrism confidence "high" publishes.
        """
        tol = self.fp_anchor_tol
        detail = fp["sheet"].get("cf_source_detail") or "marker_mean"
        fp_by_det = {m["detection_id"]: m for m in fp["markers"]}
        per_crop, used_vals, witnesses, dissent = [], [], [], []
        for r in crop_rows:
            key = f"det{r['detection_id']}"
            px = r.get("pxcm")
            pct = None if not px else 100.0 * (float(px) / anchor - 1.0)
            if _is_fp_row(r):
                m = fp_by_det.get(r["detection_id"]) or {}
                verdict = m.get("verdict") or "skipped"
                if r["status"] != "measured" and verdict == "used":
                    verdict = "skipped"
                note = m.get("verdict_note") or (r["status_reason"] if verdict == "skipped"
                                                 else None)
                if verdict == "rejected":
                    dissent.append(key)
            elif r["status"] != "measured" or not px:
                verdict, note = "skipped", r["status_reason"]
            elif abs(float(px) / anchor - 1.0) <= tol:
                verdict, note = "used", None
                witnesses.append(key)
            else:
                h = _harmonic_of(float(px), anchor)
                verdict = "rejected"
                note = (f"{h:g}x the FieldPrism CF -- wrong harmonic" if h
                        else f"disagrees with the FieldPrism CF ({pct:+.1f}%)")
                dissent.append(key)
            if verdict == "used":
                used_vals.append(float(px))
            per_crop.append(dict(key=key, cf=px, ruler_class=r.get("ruler_class"),
                                 weight=(FP_RECONCILE_WEIGHT if _is_fp_row(r) else r.get("n_kept")),
                                 verdict=verdict, note=note,
                                 pct_vs_parent=(pct if verdict != "skipped" else None),
                                 pct_vs_anchor=(pct if verdict != "skipped" else None)))
        spread = ((max(used_vals) - min(used_vals)) / (sum(used_vals) / len(used_vals))
                  if len(used_vals) > 1 else None)
        confidence = fp["confidence"]
        reasons = [f"CF = FieldPrism {detail.replace('_', ' ')} {anchor:.2f} px/cm (the "
                   f"FieldPrism geometry is authoritative on this sheet)"]
        if witnesses:
            reasons.append(f"{len(witnesses)} other ruler(s) corroborate the FieldPrism CF within "
                           f"+/-{tol:.0%} ({', '.join(witnesses)})")
        if dissent:
            reasons.append(f"{len(dissent)} reading(s) disagree and were rejected "
                           f"({', '.join(dissent)})")
        if confidence != "high":
            reasons.append(f"CF withheld: the FieldPrism markers are only {confidence} confidence")
        publish = confidence == "high"
        return dict(n_crops=len(crop_rows), n_live=len(used_vals), anchor=anchor,
                    anchor_dropped=False,
                    cf_px_per_cm=(anchor if publish else None), cf_px_per_cm_measured=anchor,
                    ok=publish, confidence=confidence, confidence_reasons=reasons,
                    anchor_supported=True, anchor_log_dist=0.0, length_backed=False,
                    win_ruler_len_cm=None,
                    corroborated_by_peers=bool(len(used_vals) >= 2),
                    n_dissenting=len(dissent), method=f"fieldprism_{detail}",
                    spread=spread, agree=(spread is not None and spread <= 2 * tol),
                    n_clusters=1, pct_vs_anchor=0.0, per_crop=per_crop, reason=None)

    def _analyze_fieldprism(self, specimen, fp_crops):
        """analyze_fieldprism over the sheet's FP crops, reading the working image ONCE.

        An unreadable working image is a recorded outcome, not a crash: every marker comes back
        'unreadable' and the sheet 'undetermined', with the reason, and there is no anchor."""
        boxes = [dict(detection_id=c.get("detection_id"), det_conf=c.get("det_conf"),
                      x1=float(c["x1"]), y1=float(c["y1"]),
                      x2=float(c["x2"]), y2=float(c["y2"])) for c in fp_crops]
        wp = specimen.get("working_path")
        img = cv2.imread(str(wp), cv2.IMREAD_COLOR) if wp and os.path.exists(str(wp)) else None
        if img is None:
            return _unreadable_fieldprism(boxes, f"working image could not be read: {wp}")
        return FP.analyze_fieldprism(
            img, boxes, image_wh=(int(img.shape[1]), int(img.shape[0])),
            peer_tol=self.fp_peer_tol, allow_single_marker=self.fp_allow_single_marker)

    def _fp_crop_row(self, row, c, specimen, m, anchor_work):
        """Fill a ruler_CF_lattice_crop row for one FieldPrism marker (no pixels are read here).

        A marker the FieldPrism step used, or rejected for disagreeing with the other markers,
        is 'measured' and goes to the sheet reconciliation (a peer-rejected one is rejected
        there too). A failed / invalid / duplicate marker is 'failed' (or 'unreadable')."""
        work_scale = float(specimen.get("work_scale") or 1.0)
        W, H = specimen.get("working_width"), specimen.get("working_height")
        bx1, by1 = max(0, int(round(c["x1"]))), max(0, int(round(c["y1"])))
        bx2 = int(round(c["x2"])) if not W else min(int(W), int(round(c["x2"])))
        by2 = int(round(c["y2"])) if not H else min(int(H), int(round(c["y2"])))
        row.update(crop_w=max(0, bx2 - bx1), crop_h=max(0, by2 - by1),
                   layout="fieldprism", n_kept=None,
                   tile_four_path=c.get("tile_four_path"))
        if m["status"] == "measured" and m.get("verdict") in ("used", "rejected"):
            pxcm = float(m["pxcm"])
            row.update(status="measured", status_reason=None,
                       pxcm=_num(pxcm), pxcm_original=_num(pxcm / (work_scale or 1.0)),
                       pct_vs_anchor=_num(_pct(pxcm, anchor_work)),
                       rotation_deg=_num(m.get("orientation_deg")))
        else:
            # failed -> why the squares were not found; measured but invalid / duplicate -> the
            # FieldPrism verdict note ("failed validation: ...", "duplicate detection ...")
            why = (m.get("verdict_note") if m["status"] == "measured" else m.get("status_reason"))
            why = why or "FieldPrism marker not usable"
            row.update(status=("unreadable" if m["status"] == "unreadable" else "failed"),
                       status_reason=why)
        return row

    # ---- one crop ----------------------------------------------------------
    def _measure_crop(self, index, c, specimen, anchor_work, fp_marker=None):
        """-> (crop_row, live) where `live` carries only in-memory rasters.

        `fp_marker` is this crop's analyze_fieldprism marker when the FieldPrism path ran; an FP
        crop is then filled from it and never reaches the tick lattice."""
        det = c.get("detection_id")
        cls = _cls_of(c)
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

        # FieldPrism markers never reach the tick lattice: measured by fieldprism.py (already done
        # in process_specimen), or, with FP disabled, recorded as skipped.
        if is_fieldprism(cls):
            if fp_marker is not None:
                return self._fp_crop_row(row, c, specimen, fp_marker, anchor_work), live
            row.update(status="skipped_class",
                       status_reason="FieldPrism marker (FP) -- FieldPrism measurement is "
                                     "disabled (modules.ruler_cf.fieldprism.enabled)",
                       tile_four_path=c.get("tile_four_path"))
            return row, live

        # A class the registry marks unsupported (messy / unknown) is recorded and
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
        fp_sheet = record.get("fp_sheet")
        fp_markers = list(record.get("fp_markers") or [])
        fp_by_det = {m.get("detection_id"): m for m in fp_markers}
        # A sheet the FieldPrism path ran on is anchored on the FieldPrism CF (possibly none), so
        # every section names THAT anchor; any other sheet renders exactly as before.
        fp_sheet_run = bool(img.get("fp_detected")) or img.get("anchor_source") == "fieldprism"
        anchor_qc = (img.get("anchor_cf_working") if fp_sheet_run
                     else img.get("mp_anchor_working"))

        panels, entries = [], []
        n_fp_panel = 0
        for r in crops:
            lv = by_det.get(r["detection_id"]) or {}
            fpm = fp_by_det.get(r["detection_id"])
            if fpm is not None and hasattr(QC, "fp_marker_section"):
                n_fp_panel += 1
                panels.append(QC.fp_marker_section(
                    fpm, r.get("crop_path"), (r.get("x1"), r.get("y1"), r.get("x2"), r.get("y2")),
                    title_index=n_fp_panel))
                entries.append(dict(key=f"det{r['detection_id']}", rot=None,
                                    verdict=r.get("verdict")))
                continue
            # Always the STORED thumbnail, never the in-memory full-res tile: the
            # live panel and the panel rebuilt from the DB must be the same image.
            tile = _load_rgb(r.get("tile_four_path"))
            qrow = dict(image_name=img.get("image_name"),
                        detection_id=r["detection_id"],
                        det_conf=float(r.get("det_conf") or 0.0),
                        cls_conf=r.get("cls_conf"),
                        work_scale=img.get("work_scale"),
                        anchor=anchor_qc,
                        anchor_source=("fieldprism" if fp_sheet_run else "megapixels"),
                        anchor_original=img.get("mp_anchor_original"),
                        anchor_frame=img.get("anchor_frame"),
                        anchor_formula=img.get("anchor_formula"),
                        original_mp=img.get("original_mp"),
                        ruler_class=r.get("ruler_class"))
            cls = r["ruler_class"]

            if r["status"] != "measured" or fpm is not None:
                why = (f"NOT A SUPPORTED CLASS -- no conversion factor is attempted.\n"
                       f"{r['status_reason']}" if r["status"] == "skipped_class"
                       else f"CF DETERMINATION FAILED -- {r['status_reason']}"
                       if r["status"] != "measured"
                       else f"FieldPrism marker: {r.get('pxcm') or 0:.2f} px/cm "
                            f"(verdict {r.get('verdict') or '-'})")
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

        if fp_sheet is not None and hasattr(QC, "fp_sheet_section"):
            panels.append(QC.fp_sheet_section(fp_sheet, fp_markers))

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
        withheld = pr.get("cf_px_per_cm") is None and anchor_qc
        # Whether the stage actually SUBSTITUTED the MP anchor (use_CF_predicted_by_MP on) or left
        # the sheet without a CF. Read from the record, so a panel rebuilt later says the same.
        applied = img.get("cf_source") == CF_SOURCE_MP
        # The FieldPrism keywords are passed ONLY for a FieldPrism sheet (and only to a QC module
        # that takes them), so every other sheet makes exactly the calls it always made.
        recon_kw = ({"anchor_source": "fieldprism"} if fp_sheet_run and
                    _accepts(QC.build_recon_section, "anchor_source") else {})
        if drawable and (len(panels) > 1 or withheld):
            recon = QC.build_recon_section(img.get("image_name"), drawable, pr,
                                           anchor_qc,
                                           anchor_formula=img.get("anchor_formula"),
                                           fallback_applied=applied, **recon_kw)
        # The CF summary is unconditional: every panel ends with the two numbers, whether the
        # sheet published a measured CF or fell back to the prediction.
        sum_kw = ({"anchor_source": "fieldprism", "fp_sheet": fp_sheet} if fp_sheet_run and
                  _accepts(QC.build_cf_summary_section, "fp_sheet") else {})
        cf_summary = QC.build_cf_summary_section(
            pr.get("cf_px_per_cm"), anchor_qc,
            img.get("anchor_formula_symbolic"), fallback_applied=applied, **sum_kw)
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
        # children before parents; the FieldPrism tables are replaced wholesale too, so a sheet
        # re-run without FP crops (or with FP disabled) leaves no stale marker/sheet rows behind
        con.execute("DELETE FROM ruler_FP_marker WHERE specimen_id=?", (sid,))
        con.execute("DELETE FROM ruler_FP_sheet WHERE specimen_id=?", (sid,))
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
        sheet = record.get("fp_sheet")
        if sheet:
            scols = [c for c in fp_sheet_columns() if c in sheet]
            con.execute(f"INSERT INTO ruler_FP_sheet ({','.join(scols)}) "
                        f"VALUES ({','.join('?' * len(scols))})",
                        [_num(sheet[c]) for c in scols])
        markers = record.get("fp_markers") or []
        if markers:
            mcols = fp_marker_columns()
            con.executemany(
                f"INSERT INTO ruler_FP_marker ({','.join(mcols)}) "
                f"VALUES ({','.join('?' * len(mcols))})",
                [[_num(m.get(c)) for c in mcols] for m in markers])

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
        try:
            fs = con.execute("SELECT * FROM ruler_FP_sheet WHERE specimen_id=?",
                             (specimen_id,)).fetchone()
            fm = con.execute("SELECT * FROM ruler_FP_marker WHERE specimen_id=? "
                             "ORDER BY crop_index, detection_id", (specimen_id,)).fetchall()
        except sqlite3.OperationalError:     # a DB from before FieldPrism, opened read-only
            fs, fm = None, []
        return dict(image=dict(i), crops=[dict(c) for c in cs],
                    fp_sheet=(None if fs is None else dict(fs)),
                    fp_markers=[dict(m) for m in fm])

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


# --------------------------------------------------------- FieldPrism helpers --
def _cls_of(c):
    """The crop's classifier class; a NULL classification defaults to METRIC_MM (as before)."""
    return c.get("ruler_class") or "METRIC_MM"


def _is_fp_row(r):
    """A ruler_CF_lattice_crop row that the FieldPrism path filled (not a skipped FP crop)."""
    return r.get("class_layout") == "fieldprism" and r.get("status") != "skipped_class"


def _fp_catalog_version():
    try:
        return FP.load_sheet_catalog().get("catalog_version")
    except (OSError, ValueError):            # a missing catalog is recorded, never fatal here
        return None


def _unreadable_fieldprism(boxes, reason):
    """analyze_fieldprism's output shape for a sheet whose working image cannot be read."""
    markers = []
    for b in boxes:
        markers.append(dict(
            status="unreadable", status_reason=reason, roi=None, n_peaks=0, holes_filled=0,
            peak_area_ratio=None, peak_areas=[], roles=None, br=None, pitch_h_px=None,
            pitch_v_px=None, pxcm=None, orientation_deg=None, valid=False, validation={},
            validation_reasons=[], detection_id=b["detection_id"],
            det_conf=_num(b.get("det_conf")), x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"],
            verdict="skipped", verdict_note=reason, sheet_corner=None, pct_vs_fp=None))
    sheet = dict(
        status="undetermined", sheet_type=None, label=None, corners_ambiguous=None,
        candidates=[], assignment={}, fit=None, cf_px_per_cm_sheet_fit=None,
        orientation_deg=None, corners={}, page_corners_px=None, fpfit_margins_mm=None,
        n_fp_detected=len(markers), n_fp_measured=0, n_fp_valid=0, n_fp_used=0,
        n_fp_rejected=0, n_fp_inferred=0, cf_px_per_cm_marker_mean=None,
        cf_px_per_cm_fp=None, cf_source_detail=None, fp_peer_spread_pct=None,
        fp_confidence=None, fp_reasons=[reason])
    return dict(markers=markers, sheet=sheet, anchor_cf=None, confidence=None)


# ------------------------------------------------------- record -> QC helpers --
def _admissible(cls):
    from .units import admissible_units
    try:
        return admissible_units(cls)
    except Exception:
        return []


def _accepts(fn, name):
    """True when `fn` takes the keyword `name` (a QC module that predates FieldPrism does not)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


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
