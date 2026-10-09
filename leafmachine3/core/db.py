"""leafmachine3.core.db -- the ONLY module that issues raw SQL.

One :class:`ProjectDB` wraps a single per-project SQLite database (WAL mode). Every
stage reads and writes through these typed methods; no stage embeds SQL of its own.
The pipeline collector is the SINGLE writer, so a specimen's row writes and its
``image_status`` checkpoint are grouped together inside :meth:`ProjectDB.transaction`
(``BEGIN IMMEDIATE`` / ``COMMIT``) -- a crash mid-write can never leave a half-persisted
specimen marked ``done``.

Design notes
------------
* The connection runs in autocommit mode (``isolation_level=None``); each statement
  commits on its own unless wrapped in an explicit :meth:`transaction`. This keeps the
  writer methods usable both standalone (tests) and grouped (the collector).
* Reads return :class:`sqlite3.Row` objects (mapping- and index-accessible).
* Idempotency: detections are delete-then-insert per specimen, leaf instances upsert on
  ``(detection_id, instance_index)`` after a per-specimen delete, and the resume queries
  are set-based snapshots taken before dispatch (race-free under a single writer).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from leafmachine3.core.records import (
    CF_SOURCE_RULER,
    CropRef,
    DetRow,
    Grounded,
    LeafRow,
    PhenologyResult,
    RulerClassRow,
    SpecimenRecord,
)

log = logging.getLogger("leafmachine3.db")

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

_DETECTION_TABLES = ("archival_detection", "plant_detection")
# Tables that carry a ``specimen_id`` column and can be scanned for "which specimens have
# rows here" (used by phenology over detections and metric_grounding over segmentation).
_ROW_TABLES = _DETECTION_TABLES + ("leaf_segmentation",)

# Fallback canonical stage keys, in pipeline order. The single source of truth is
# ``leafmachine3.pipeline.STAGE_ORDER``; we prefer that when importable but keep this
# list so the DB layer can seed ``project_status`` even if the pipeline module (which
# imports every stage) is not yet importable.
_CANONICAL_STAGE_KEYS: tuple[str, ...] = (
    "mp_conversion_factor",
    "archival_detector",
    "plant_detector",
    "specimen_segmenter",
    "phenology_detector",
    "ruler_classifier",
    "ruler_cf",
    "leaf_segmenter",
    "morphology",
    "landmark_detector",
    "landmark_measurements",
    "leaf_orientation",
    "petiole_width",
    "bilateral_symmetry",
    "metric_grounding",
    "reporter",
    "ect",
    "momocs",
)

# Duplicated specimen/leaf columns each stage OWNS -- nulled when that stage is reset.
_OWNED_SPECIMEN_COLS: dict[str, tuple[str, ...]] = {
    "mp_conversion_factor": ("cf_px_per_cm_predicted_by_mp", "original_mp"),
    "ruler_classifier": ("ruler_class_type",),
    "phenology_detector": ("has_leaves", "has_flowers", "has_fruits"),
    "ruler_cf": ("cf_px_per_cm", "cf_source", "ruler_unit_type"),   # CF write-back; nulled on ruler_cf reset
}
_OWNED_LEAF_COLS: dict[str, tuple[str, ...]] = {
    "metric_grounding": ("area_cm2", "perimeter_cm", "bbox_w_cm", "bbox_h_cm"),
}
# leaf_landmark_measurement columns a stage OWNS (updates in place) -- nulled when that stage is reset.
_OWNED_LM_MEASURE_COLS: dict[str, tuple[str, ...]] = {
    "metric_grounding": ("lamina_trace_length_cm", "lamina_extent_cm", "lamina_tip_base_length_cm",
                         "leaf_width_cm", "petiole_trace_length_cm"),
}
# leaf_morphology columns a stage OWNS (updates in place) -- nulled when that stage is reset.
_OWNED_MORPH_COLS: dict[str, tuple[str, ...]] = {
    "leaf_orientation": ("oriented_leaf_success", "oriented_leaf_rotation_angle_degreesCW",
                         "rotated_bbox_length", "rotated_bbox_width"),
}
# leaf_petiole columns a stage OWNS (updates in place) -- nulled when that stage is reset.
_OWNED_PETIOLE_COLS: dict[str, tuple[str, ...]] = {
    "metric_grounding": ("width_cm", "length_cm"),
}


def _stage_keys() -> list[str]:
    """Canonical stage keys in order -- from the pipeline if importable, else the fallback."""
    try:
        from leafmachine3.pipeline import STAGE_ORDER  # noqa: WPS433 (local import by design)

        return [cls.key for cls in STAGE_ORDER]
    except Exception:  # pragma: no cover - pipeline pulls in every stage/model import
        return list(_CANONICAL_STAGE_KEYS)


class ProjectDB:
    """Typed gateway to one project's SQLite database."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    # ---- lifecycle ------------------------------------------------------- #
    @classmethod
    def open_or_create(cls, path: Path | str) -> "ProjectDB":
        """Open (creating if absent) the DB, apply the schema, and seed the ledger."""
        db = cls(path)
        db.connect()
        db.init_schema()
        return db

    def connect(self) -> sqlite3.Connection:
        """Open the connection in autocommit/WAL mode with dict-accessible rows."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: ingest and the CPU-stage in-process thread pool call
        # through this single connection from worker threads. CPython's sqlite3 defaults to
        # SERIALIZED threading, so the connection's own mutex serializes those calls safely.
        conn = sqlite3.connect(
            str(self.path), timeout=30, isolation_level=None, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ProjectDB is not connected; call connect() first")
        return self._conn

    # Columns added to `specimen` after the initial schema shipped. `CREATE TABLE IF NOT EXISTS`
    # cannot add a column to an existing DB, so these are ALTER-backfilled idempotently at init.
    _SPECIMEN_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("cf_px_per_cm_predicted_by_mp", "REAL"),
        ("original_mp", "REAL"),
        ("ruler_class_type", "TEXT"),
        # Explicit "pixels were discarded" flag. `normalized` cannot answer this: it is also 1 for a
        # small non-JPEG that was only format-converted. Backfilled from the dims, which are exact.
        ("downsampled", "INTEGER NOT NULL DEFAULT 0"),
        # Which CF specimen.cf_px_per_cm holds (CF_SOURCE_* below). Backfilled from the CF itself:
        # before this column existed only a published ruler CF was ever written there.
        ("cf_source", "TEXT"),
    )
    # Columns added to BOTH detection tables after the initial schema (same ALTER-backfill reason).
    _DETECTION_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("suppressed", "INTEGER NOT NULL DEFAULT 0"),
        ("suppressed_by", "INTEGER"),
        ("suppress_overlap", "REAL"),
    )
    # Column added to ruler_classification (pre-made four-tile collage path).
    _RULER_CLASS_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("squarify_path", "TEXT"),
    )
    # Columns added to leaf_petiole (cm-grounded petiole length; width_cm shipped with the table).
    _PETIOLE_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("length_cm", "REAL"),
        ("leaf_mass_per_area", "REAL"),
    )
    # Column added to leaf_ect (radial ECT + leaf-outline overlay PNG).
    _ECT_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("overlay_png", "TEXT"),
    )
    # Columns added to leaf_landmark_measurement: the cm-grounded twins of the five LENGTH metrics
    # (MetricGrounding fills them when a CF exists). Angles/curvature are degrees -- no cm twin.
    _LM_MEASURE_MIGRATIONS: tuple[tuple[str, str], ...] = (
        ("lamina_trace_length_cm", "REAL"),
        ("lamina_extent_cm", "REAL"),
        ("lamina_tip_base_length_cm", "REAL"),
        ("leaf_width_cm", "REAL"),
        ("petiole_trace_length_cm", "REAL"),
    )
    # Columns REMOVED from leaf_morphology. They were declared but never written -- MetricGrounding
    # grounds this leaf's area/perimeter/bbox on `leaf_segmentation` (same leaf_id), which is the one
    # home for cm values. Dropped rather than left in place so a `SELECT *` consumer (and every CSV
    # export) stops shipping four all-NULL columns that read as a bug. Idempotent: a DB created after
    # this change simply has none of them.
    _MORPH_DROPS: tuple[str, ...] = ("area_cm2", "perimeter_cm", "length_cm", "width_cm")

    def init_schema(self) -> None:
        """Apply ``schema.sql`` (idempotent), migrate new columns, seed ``project_status`` + landmarks."""
        self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        have = self._table_columns("specimen")
        for col, decl in self._SPECIMEN_MIGRATIONS:
            if col not in have:
                self._exec(f"ALTER TABLE specimen ADD COLUMN {col} {decl}")
                if col == "downsampled":
                    # ALTER can only supply the constant default, which would assert "nothing was
                    # ever downsampled" about every pre-existing row. The dims already record the
                    # truth exactly, so backfill from them rather than ship a column that lies.
                    self._exec("UPDATE specimen SET downsampled = "
                               "(original_width IS NOT NULL AND width IS NOT NULL "
                               " AND width <> original_width)")
                elif col == "cf_source":
                    self._exec("UPDATE specimen SET cf_source = ? WHERE cf_px_per_cm IS NOT NULL",
                               (CF_SOURCE_RULER,))
        for table in _DETECTION_TABLES:
            cols = self._table_columns(table)
            for col, decl in self._DETECTION_MIGRATIONS:
                if col not in cols:
                    self._exec(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        rc_cols = self._table_columns("ruler_classification")
        for col, decl in self._RULER_CLASS_MIGRATIONS:
            if col not in rc_cols:
                self._exec(f"ALTER TABLE ruler_classification ADD COLUMN {col} {decl}")
        pet_cols = self._table_columns("leaf_petiole")
        for col, decl in self._PETIOLE_MIGRATIONS:
            if col not in pet_cols:
                self._exec(f"ALTER TABLE leaf_petiole ADD COLUMN {col} {decl}")
        ect_cols = self._table_columns("leaf_ect")
        for col, decl in self._ECT_MIGRATIONS:
            if col not in ect_cols:
                self._exec(f"ALTER TABLE leaf_ect ADD COLUMN {col} {decl}")
        lmm_cols = self._table_columns("leaf_landmark_measurement")
        for col, decl in self._LM_MEASURE_MIGRATIONS:
            if col not in lmm_cols:
                self._exec(f"ALTER TABLE leaf_landmark_measurement ADD COLUMN {col} {decl}")
        morph_cols = self._table_columns("leaf_morphology")
        for col in self._MORPH_DROPS:
            if col in morph_cols:
                # DROP COLUMN needs SQLite >= 3.35 and refuses on an indexed/constrained column.
                # Neither applies here, but a failure must never take the run down over four dead
                # columns -- log and carry on, the export selects columns explicitly either way.
                try:
                    self._exec(f"ALTER TABLE leaf_morphology DROP COLUMN {col}")
                except sqlite3.Error as exc:   # pragma: no cover - old SQLite only
                    log.warning("could not drop dead leaf_morphology column %s: %s", col, exc)
        # Lattice ruler-CF tables (ruler_CF_lattice + ruler_CF_lattice_crop + view): the engine's
        # SCHEMA_SQL is the single DDL source, so code and tables cannot drift. Lazy-imported to
        # keep core/ free of an inference/ import at module load.
        # Columns first: SCHEMA_SQL also creates indexes, which would fail on an old table that
        # lacks an indexed column; then the script creates whatever tables/indexes are missing.
        from leafmachine3.inference.ruler_lattice import SCHEMA_SQL as _LATTICE_SCHEMA_SQL
        self._migrate_lattice_tables()
        self.conn.executescript(_LATTICE_SCHEMA_SQL)
        for order, key in enumerate(_stage_keys(), start=1):
            self._exec(
                "INSERT OR IGNORE INTO project_status(stage_key, stage_order) VALUES (?, ?)",
                (key, order),
            )
        from leafmachine3.core import landmarks as _lm     # seed the self-describing keypoint schema
        for i, name in enumerate(_lm.KPT_NAMES):
            self._exec("INSERT OR IGNORE INTO landmark_schema(kpt_index, name, grp) VALUES (?, ?, ?)",
                       (i, name, _lm.KPT_GROUP[name]))
        for edge_id, (a, b, kind) in enumerate(_lm.SKELETON):
            self._exec("INSERT OR IGNORE INTO landmark_skeleton(edge_id, a_index, b_index, kind) "
                       "VALUES (?, ?, ?, ?)", (edge_id, _lm.KPT_INDEX[a], _lm.KPT_INDEX[b], kind))

    def _migrate_lattice_tables(self) -> None:
        """Add every column the engine's SCHEMA_SQL declares but an older DB's table lacks.

        GENERIC on purpose: the engine DDL is the single source for its tables, so a column added
        there (e.g. ruler_CF_lattice.anchor_source for FieldPrism) reaches existing DBs with no
        per-column migration list to keep in step. Constraints are dropped (ALTER cannot add a
        PRIMARY KEY / UNIQUE / bare NOT NULL column); only the bare type is applied."""
        from leafmachine3.inference.ruler_lattice import migration_columns, schema_tables
        for table in schema_tables():
            have = self._table_columns(table)
            if not have:                       # not created yet: SCHEMA_SQL creates it whole
                continue
            for col, typ in migration_columns(table):
                if col not in have:
                    self._exec(f"ALTER TABLE {table} ADD COLUMN {col} {typ}".rstrip())

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """``BEGIN IMMEDIATE`` / ``COMMIT``; ``ROLLBACK`` on any error.

        The collector wraps a specimen's row-writes AND its ``mark_image_done`` in one of
        these so a torn write cannot leave the specimen checkpointed as done.
        """
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    # ---- low-level helpers ---------------------------------------------- #
    def _exec(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def _one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchone()

    def _table_columns(self, table: str) -> set[str]:
        return {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    # ---- specimen (raw/metadata) ---------------------------------------- #
    def upsert_specimen(self, rec: SpecimenRecord) -> int:
        """Insert or update a specimen (keyed on ``image_stem``); return its id."""
        self._exec(
            """
            INSERT INTO specimen (
                image_name, image_stem, original_path, working_path,
                width, height, original_width, original_height, work_scale,
                orig_size_bytes, orig_mtime, normalized, downsampled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_stem) DO UPDATE SET
                image_name      = excluded.image_name,
                original_path   = excluded.original_path,
                working_path    = excluded.working_path,
                width           = excluded.width,
                height          = excluded.height,
                original_width  = excluded.original_width,
                original_height = excluded.original_height,
                work_scale      = excluded.work_scale,
                orig_size_bytes = excluded.orig_size_bytes,
                orig_mtime      = excluded.orig_mtime,
                normalized      = excluded.normalized,
                downsampled     = excluded.downsampled
            """,
            (
                rec.image_name,
                rec.image_stem,
                rec.original_path,
                rec.working_path,
                int(rec.width),
                int(rec.height),
                int(rec.original_width),
                int(rec.original_height),
                float(rec.work_scale),
                int(rec.orig_size_bytes),
                float(rec.orig_mtime),
                int(bool(rec.normalized)),
                int(bool(rec.downsampled)),
            ),
        )
        row = self._one("SELECT specimen_id FROM specimen WHERE image_stem = ?", (rec.image_stem,))
        return int(row["specimen_id"])

    def clear_specimens(self) -> None:
        """Delete every specimen; ``ON DELETE CASCADE`` wipes all method rows."""
        self._exec("DELETE FROM specimen")

    def get_specimen(self, specimen_id: int) -> Optional[sqlite3.Row]:
        return self._one("SELECT * FROM specimen WHERE specimen_id = ?", (specimen_id,))

    def iter_specimens(self) -> list[sqlite3.Row]:
        """All specimen rows, ordered by id (input to the detector stages)."""
        return self._query("SELECT * FROM specimen ORDER BY specimen_id")

    def iter_specimen_ids(self) -> list[int]:
        return [int(r["specimen_id"]) for r in self._query("SELECT specimen_id FROM specimen ORDER BY specimen_id")]

    def ingest_signatures(self) -> dict[str, tuple[int, float]]:
        """``{original_path: (size_bytes, mtime)}`` for ingest change-detection."""
        rows = self._query("SELECT original_path, orig_size_bytes, orig_mtime FROM specimen")
        return {
            str(r["original_path"]): (int(r["orig_size_bytes"] or 0), float(r["orig_mtime"] or 0.0))
            for r in rows
        }

    def set_specimen_cf(
        self,
        specimen_id: int,
        cf_px_per_cm: Optional[float],
        *,
        unit_type: Optional[str] = None,
        source: Optional[str] = CF_SOURCE_RULER,
    ) -> None:
        """Write the sheet's CF (WORKING frame), where it came from, and the rulers' unit-type.

        ``source`` is ``CF_SOURCE_RULER`` for a high-confidence 'published' lattice CF, or
        ``CF_SOURCE_MP`` when ``modules.ruler_cf.use_CF_predicted_by_MP`` substituted the
        megapixel prediction for a sheet with no ruler or a lattice that did not pass. With the
        option off such a sheet is never written, so ``cf_px_per_cm`` stays NULL (a visible
        absence) rather than carrying an unknown unit-misnaming error.
        """
        cf = None if cf_px_per_cm is None else float(cf_px_per_cm)
        self._exec(
            "UPDATE specimen SET cf_px_per_cm = ?, cf_source = ?, ruler_unit_type = ? "
            "WHERE specimen_id = ?",
            (cf, None if cf is None or not source else str(source),
             None if not unit_type else str(unit_type), specimen_id),
        )

    def set_specimen_mp_cf(
        self, specimen_id: int, cf_px_per_cm: Optional[float], original_mp: Optional[float] = None,
    ) -> None:
        """Store the megapixels + resolution->CF linear prediction (mp_conversion_factor) on the specimen."""
        self._exec(
            "UPDATE specimen SET cf_px_per_cm_predicted_by_mp = ?, original_mp = ? WHERE specimen_id = ?",
            (None if cf_px_per_cm is None else float(cf_px_per_cm),
             None if original_mp is None else float(original_mp), specimen_id),
        )

    def specimen_cf(self, specimen_id: int) -> Optional[float]:
        row = self._one("SELECT cf_px_per_cm FROM specimen WHERE specimen_id = ?", (specimen_id,))
        if row is None or row["cf_px_per_cm"] is None:
            return None
        return float(row["cf_px_per_cm"])

    # ---- resume queries (set-based; race-free) --------------------------- #
    def eligible_specimens(self, stage: str, depends_on: Sequence[str] = ()) -> list[int]:
        """Specimens whose EVERY ``depends_on`` stage has a ``done`` image_status row.

        An empty ``depends_on`` means every specimen is eligible.
        """
        deps = tuple(depends_on)
        if not deps:
            return self.iter_specimen_ids()
        placeholders = ",".join("?" for _ in deps)
        rows = self._query(
            f"""
            SELECT s.specimen_id
              FROM specimen s
             WHERE (
                     SELECT COUNT(*) FROM image_status i
                      WHERE i.specimen_id = s.specimen_id
                        AND i.state = 'done'
                        AND i.stage_key IN ({placeholders})
                   ) = ?
             ORDER BY s.specimen_id
            """,
            (*deps, len(deps)),
        )
        return [int(r["specimen_id"]) for r in rows]

    def done_ids(self, stage: str) -> set[int]:
        """Specimen ids already ``done`` for ``stage`` in ``image_status``."""
        rows = self._query(
            "SELECT specimen_id FROM image_status WHERE stage_key = ? AND state = 'done'",
            (stage,),
        )
        return {int(r["specimen_id"]) for r in rows}

    def pending_specimens(self, stage: str, *, depends_on: Sequence[str] = ()) -> list[int]:
        """Eligible specimens for ``stage`` MINUS those already done for it."""
        done = self.done_ids(stage)
        return [sid for sid in self.eligible_specimens(stage, depends_on) if sid not in done]

    # ---- per-method writers (called by the collector) -------------------- #
    def record_detections(self, table: str, specimen_id: int, rows: Sequence[DetRow]) -> list[int]:
        """DELETE the specimen's rows in ``table`` then INSERT ``rows`` (idempotent re-run)."""
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        self._exec(f"DELETE FROM {table} WHERE specimen_id = ?", (specimen_id,))
        ids: list[int] = []
        for r in rows:
            x1, y1, x2, y2 = r.xyxy
            cur = self._exec(
                f"""
                INSERT INTO {table}
                    (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2, tag, crop_path,
                     suppressed, suppress_overlap)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    specimen_id,
                    int(r.cls_id),
                    r.cls_name,
                    float(r.conf),
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    r.tag,
                    r.crop_path,
                    int(bool(getattr(r, "suppressed", False))),
                    None if getattr(r, "suppress_overlap", None) is None else float(r.suppress_overlap),
                ),
            )
            new_id = int(cur.lastrowid)
            r.detection_id = new_id
            ids.append(new_id)
        # second pass: resolve each suppressed row's keeper index -> the keeper's detection_id
        for r in rows:
            idx = getattr(r, "suppressed_by_index", None)
            if getattr(r, "suppressed", False) and idx is not None and 0 <= idx < len(ids):
                self._exec(
                    f"UPDATE {table} SET suppressed_by = ? WHERE detection_id = ?",
                    (ids[idx], r.detection_id),
                )
        return ids

    def record_phenology(self, specimen_id: int, result: PhenologyResult) -> None:
        """Write the phenology row and mirror the presence flags onto the specimen."""
        has_leaves, n_leaves = bool(result.leaves[0]), int(result.leaves[1])
        has_flowers, n_flowers = bool(result.flowers[0]), int(result.flowers[1])
        has_fruits, n_fruits = bool(result.fruits[0]), int(result.fruits[1])
        self._exec(
            """
            INSERT INTO phenology
                (specimen_id, has_leaves, has_flowers, has_fruits, n_leaves, n_flowers, n_fruits)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(specimen_id) DO UPDATE SET
                has_leaves = excluded.has_leaves,   has_flowers = excluded.has_flowers,
                has_fruits = excluded.has_fruits,   n_leaves    = excluded.n_leaves,
                n_flowers  = excluded.n_flowers,    n_fruits    = excluded.n_fruits
            """,
            (
                specimen_id,
                int(has_leaves),
                int(has_flowers),
                int(has_fruits),
                n_leaves,
                n_flowers,
                n_fruits,
            ),
        )
        self._exec(
            "UPDATE specimen SET has_leaves = ?, has_flowers = ?, has_fruits = ? WHERE specimen_id = ?",
            (int(has_leaves), int(has_flowers), int(has_fruits), specimen_id),
        )

    def record_ruler_classifications(self, specimen_id: int, rows: Sequence[RulerClassRow]) -> None:
        """Upsert one ``ruler_classification`` per Ruler crop (keyed on ``detection_id``)."""
        for r in rows:
            self._exec(
                """
                INSERT INTO ruler_classification
                    (specimen_id, detection_id, unit_type, votes_json, conf, squarify_path)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(detection_id) DO UPDATE SET
                    unit_type     = excluded.unit_type,
                    votes_json    = excluded.votes_json,
                    conf          = excluded.conf,
                    squarify_path = excluded.squarify_path
                """,
                (
                    specimen_id,
                    int(r.detection_id),
                    r.unit_type,
                    json.dumps(r.votes) if r.votes is not None else None,
                    r.conf,
                    getattr(r, "squarify_path", None),
                ),
            )

    def set_specimen_ruler_class(self, specimen_id: int, ruler_class_type: Optional[str]) -> None:
        """Store the ruler-classifier's per-specimen consensus unit-type on the specimen row."""
        self._exec(
            "UPDATE specimen SET ruler_class_type = ? WHERE specimen_id = ?",
            (None if not ruler_class_type else str(ruler_class_type), specimen_id),
        )

    def record_ruler_cf_lattice(self, record: dict) -> None:
        """Idempotent upsert of one sheet's lattice ruler-CF trail (image row + per-crop rows).

        Delegates to the engine's own delete-then-insert (its SCHEMA_SQL is the DDL source). The
        caller (Stage.persist) already runs inside ProjectDB's per-specimen transaction, so this
        must NOT open a nested one -- it writes on the live connection like every other record_*."""
        from leafmachine3.inference.ruler_lattice import RulerCFLattice
        RulerCFLattice.write_db(self.conn, record)

    def ruler_cf_lattice_record(self, specimen_id: int) -> Optional[dict]:
        """The lattice record as the DB holds it (``{image, crops}``), for re-rendering QC, or None."""
        from leafmachine3.inference.ruler_lattice import RulerCFLattice
        return RulerCFLattice.read_record(self.conn, specimen_id)

    def record_leaf_instances(self, specimen_id: int, rows: Sequence[LeafRow]) -> None:
        """DELETE the specimen's leaf rows then upsert on ``(detection_id, instance_index)``."""
        self._exec("DELETE FROM leaf_segmentation WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            bx1, by1, bx2, by2 = r.bbox
            self._exec(
                """
                INSERT INTO leaf_segmentation
                    (specimen_id, detection_id, instance_index, parent_instance_index,
                     cls_id, cls_name, conf, mask_format, mask_data,
                     frame_width, frame_height, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     num_parts, area_px, perimeter_px)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(detection_id, instance_index) DO UPDATE SET
                    specimen_id           = excluded.specimen_id,
                    parent_instance_index = excluded.parent_instance_index,
                    cls_id                = excluded.cls_id,
                    cls_name              = excluded.cls_name,
                    conf                  = excluded.conf,
                    mask_format           = excluded.mask_format,
                    mask_data             = excluded.mask_data,
                    frame_width           = excluded.frame_width,
                    frame_height          = excluded.frame_height,
                    bbox_x1 = excluded.bbox_x1, bbox_y1 = excluded.bbox_y1,
                    bbox_x2 = excluded.bbox_x2, bbox_y2 = excluded.bbox_y2,
                    num_parts    = excluded.num_parts,
                    area_px      = excluded.area_px,
                    perimeter_px = excluded.perimeter_px
                """,
                (
                    specimen_id,
                    int(r.detection_id),
                    int(r.instance_index),
                    r.parent_instance_index,
                    int(r.cls_id),
                    r.cls_name,
                    r.conf,
                    r.mask_format,
                    r.mask_data,
                    int(r.frame_width),
                    int(r.frame_height),
                    float(bx1),
                    float(by1),
                    float(bx2),
                    float(by2),
                    int(r.num_parts),
                    r.area_px,
                    r.perimeter_px,
                ),
            )

    def set_leaf_metrics_cm(self, grounded: Sequence[Grounded]) -> None:
        """Fill the cm-grounded metric columns for the given leaf instances."""
        for g in grounded:
            self._exec(
                """
                UPDATE leaf_segmentation
                   SET area_cm2 = ?, perimeter_cm = ?, bbox_w_cm = ?, bbox_h_cm = ?
                 WHERE leaf_id = ?
                """,
                (g.area_cm2, g.perimeter_cm, g.bbox_w_cm, g.bbox_h_cm, int(g.leaf_id)),
            )

    def set_petiole_metrics_cm(self, rows: Sequence[Any]) -> None:
        """Fill the cm-grounded petiole columns (MetricGrounding). ``rows`` are
        ``(leaf_id, width_cm, length_cm)`` triples."""
        for leaf_id, width_cm, length_cm in rows:
            self._exec(
                "UPDATE leaf_petiole SET width_cm = ?, length_cm = ? WHERE leaf_id = ?",
                (width_cm, length_cm, int(leaf_id)),
            )

    def set_landmark_metrics_cm(self, rows: Sequence[Any]) -> None:
        """Fill the cm-grounded landmark LENGTH columns (MetricGrounding). ``rows`` are
        ``(measure_id, lamina_trace_cm, lamina_extent_cm, tip_base_cm, leaf_width_cm, petiole_trace_cm)``.

        Keyed on ``measure_id`` rather than ``(detection_id, instance_index)`` because the row's own
        primary key cannot be ambiguous, and the landmark instance space is NOT the segmentation
        instance space (see the leaf_landmark_measurement schema comment).
        """
        for measure_id, trace_cm, extent_cm, tip_base_cm, width_cm, petiole_cm in rows:
            self._exec(
                """
                UPDATE leaf_landmark_measurement
                   SET lamina_trace_length_cm = ?, lamina_extent_cm = ?,
                       lamina_tip_base_length_cm = ?, leaf_width_cm = ?, petiole_trace_length_cm = ?
                 WHERE measure_id = ?
                """,
                (trace_cm, extent_cm, tip_base_cm, width_cm, petiole_cm, int(measure_id)),
            )

    def record_leaf_morphology(self, specimen_id: int, rows: Sequence[Any]) -> None:
        """DELETE the specimen's morphology rows then insert one per leaf instance."""
        self._exec("DELETE FROM leaf_morphology WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            cx1, cy1, cx2, cy2 = r.crop_box
            bx1, by1, bx2, by2 = r.bbox
            cen_x, cen_y = r.centroid
            cir_x, cir_y, cir_r = r.circle
            self._exec(
                """
                INSERT INTO leaf_morphology
                    (leaf_id, specimen_id, detection_id, instance_index, cls_name,
                     crop_x1, crop_y1, crop_x2, crop_y2,
                     area_px, perimeter_px,
                     lamina_area_incl_holes_px, lamina_area_excl_holes_px, lamina_hole_area_px, n_holes,
                     centroid_x, centroid_y,
                     convex_hull_area, convexity, concavity, circularity, aspect_ratio, n_vertices,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     rotate_angle, rotated_bbox_dim_max, rotated_bbox_dim_min, rotated_bbox_json,
                     circle_cx, circle_cy, circle_radius)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r.leaf_id), specimen_id, int(r.detection_id), int(r.instance_index), r.cls_name,
                 cx1, cy1, cx2, cy2,
                 r.area_px, r.perimeter_px,
                 r.lamina_area_incl_holes_px, r.lamina_area_excl_holes_px, r.lamina_hole_area_px, int(r.n_holes),
                 cen_x, cen_y,
                 r.convex_hull_area, r.convexity, r.concavity, r.circularity, r.aspect_ratio, r.n_vertices,
                 bx1, by1, bx2, by2,
                 r.rotate_angle, r.dim_max, r.dim_min, r.rotated_bbox_json,
                 cir_x, cir_y, cir_r),
            )

    def leaf_morphology(self, specimen_id: int) -> list[sqlite3.Row]:
        """All leaf_morphology rows for a specimen (rotated bbox, shape metrics, links)."""
        return self._query(
            "SELECT * FROM leaf_morphology WHERE specimen_id = ? ORDER BY leaf_id",
            (specimen_id,),
        )

    def record_leaf_petioles(self, specimen_id: int, rows: Sequence[Any]) -> None:
        """DELETE the specimen's petiole rows then insert one per leaf with a petiole (PetioleWidth)."""
        self._exec("DELETE FROM leaf_petiole WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            self._exec(
                """
                INSERT INTO leaf_petiole
                    (leaf_id, specimen_id, detection_id, instance_index,
                     width_px, length_px, n_samples, touches_leaf, measure_location,
                     width_segment_json, sample_segments_json, leaf_mass_per_area)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r.leaf_id), specimen_id, int(r.detection_id), int(r.instance_index),
                 r.width_px, r.length_px, int(r.n_samples), 1 if r.touches_leaf else 0, r.measure_location,
                 json.dumps(r.width_segment) if r.width_segment is not None else None,
                 json.dumps(r.sample_segments) if r.sample_segments else None,
                 r.leaf_mass_per_area),
            )

    def leaf_petioles(self, specimen_id: int) -> list[sqlite3.Row]:
        """All leaf_petiole rows for a specimen (one per leaf with a petiole)."""
        return self._query(
            "SELECT * FROM leaf_petiole WHERE specimen_id = ? ORDER BY leaf_id", (specimen_id,)
        )

    def set_leaf_orientation(self, rows: Sequence[Any]) -> None:
        """Update the orientation columns on each leaf's morphology row (LeafOrientation stage).

        Also fills ``rotated_bbox_length`` / ``rotated_bbox_width`` -- the same two side lengths as
        ``dim_max``/``dim_min`` but assigned by the tip->base axis rather than by which is longer."""
        for r in rows:
            self._exec(
                """
                UPDATE leaf_morphology
                   SET oriented_leaf_success = ?, oriented_leaf_rotation_angle_degreesCW = ?,
                       rotated_bbox_length = ?, rotated_bbox_width = ?
                 WHERE leaf_id = ?
                """,
                (1 if r.success else 0, r.angle_cw,
                 getattr(r, "bbox_length", None), getattr(r, "bbox_width", None), int(r.leaf_id)),
            )

    def record_leaf_landmarks(self, specimen_id: int, rows: Sequence[Any]) -> None:
        """DELETE the specimen's landmark rows then insert one per (crop x instance x keypoint)."""
        self._exec("DELETE FROM leaf_landmark WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            self._exec(
                """
                INSERT INTO leaf_landmark
                    (specimen_id, detection_id, instance_index, kpt_index, kpt_name,
                     x, y, x_crop, y_crop, conf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (specimen_id, int(r.detection_id), int(r.instance_index), int(r.kpt_index),
                 r.kpt_name, r.x, r.y, r.x_crop, r.y_crop, r.conf),
            )

    def leaf_landmarks(self, specimen_id: int) -> list[sqlite3.Row]:
        """All leaf_landmark rows for a specimen (keypoints in working coords, links to crop)."""
        return self._query(
            "SELECT * FROM leaf_landmark WHERE specimen_id = ? "
            "ORDER BY detection_id, instance_index, kpt_index",
            (specimen_id,),
        )

    def specimens_with_landmarks(self) -> list[int]:
        """Specimen ids that have any predicted keypoints (input to landmark_measurements)."""
        return [
            int(r["specimen_id"])
            for r in self._query("SELECT DISTINCT specimen_id FROM leaf_landmark ORDER BY specimen_id")
        ]

    def record_leaf_landmark_measurements(self, specimen_id: int, rows: Sequence[Any]) -> None:
        """DELETE the specimen's measurement rows then insert one per leaf instance."""
        self._exec("DELETE FROM leaf_landmark_measurement WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            self._exec(
                """
                INSERT INTO leaf_landmark_measurement
                    (specimen_id, detection_id, instance_index,
                     lamina_trace_length, lamina_extent, lamina_tip_base_length, leaf_width,
                     apex_angle, apex_angle_type, base_angle, base_angle_type,
                     petiole_trace_length, lamina_curvature, curvature_point,
                     lamina_centroid_x, lamina_centroid_y, n_present)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (specimen_id, int(r.detection_id), int(r.instance_index),
                 r.lamina_trace_length, r.lamina_extent, r.lamina_tip_base_length, r.leaf_width,
                 r.apex_angle, r.apex_angle_type, r.base_angle, r.base_angle_type,
                 r.petiole_trace_length, r.lamina_curvature, r.curvature_point,
                 r.lamina_centroid_x, r.lamina_centroid_y, int(r.n_present)),
            )

    def leaf_landmark_measurements(self, specimen_id: int) -> list[sqlite3.Row]:
        """All derived landmark-measurement rows for a specimen (one per leaf instance)."""
        return self._query(
            "SELECT * FROM leaf_landmark_measurement WHERE specimen_id = ? "
            "ORDER BY detection_id, instance_index",
            (specimen_id,),
        )

    def record_report_manifest(self, specimen_id: int, written: Sequence[Any]) -> None:
        """Record the artifact paths Reporter wrote (so ``--restart`` can delete them).

        Each entry is either a path string or a ``(path, kind)`` pair.
        """
        for entry in written:
            if isinstance(entry, (tuple, list)):
                path, kind = str(entry[0]), (str(entry[1]) if len(entry) > 1 else None)
            else:
                path, kind = str(entry), None
            self._exec(
                """
                INSERT INTO report_manifest (specimen_id, path, kind) VALUES (?, ?, ?)
                ON CONFLICT(specimen_id, path) DO UPDATE SET kind = excluded.kind
                """,
                (specimen_id, path, kind),
            )

    # ---- reader helpers for collect_items -------------------------------- #
    def detections(self, table: str, specimen_id: int, *, include_suppressed: bool = False) -> list[sqlite3.Row]:
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        keep = "" if include_suppressed else "AND suppressed = 0"
        return self._query(
            f"SELECT * FROM {table} WHERE specimen_id = ? {keep} ORDER BY detection_id",
            (specimen_id,),
        )

    def detection_boxes(self, specimen_id: int, table: str, *, include_suppressed: bool = False) -> dict[int, tuple]:
        """Map ``detection_id -> (x1, y1, x2, y2)`` (working coords) for one specimen's KEPT boxes.

        Used by the Reporter to recover each leaf crop's parent-frame box for per-crop mask exports.
        """
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        keep = "" if include_suppressed else "AND suppressed = 0"
        return {
            int(r["detection_id"]): (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            for r in self._query(
                f"SELECT detection_id, x1, y1, x2, y2 FROM {table} WHERE specimen_id = ? {keep}",
                (specimen_id,),
            )
        }

    def crops(
        self,
        table: str,
        specimen_id: int,
        *,
        cls_name: Optional[str] = None,
        cls_in: Optional[Sequence[str]] = None,
        include_suppressed: bool = False,
    ) -> list[CropRef]:
        """Saved detection crops for a specimen as :class:`CropRef` (frame dims from specimen).

        Suppressed (same-class duplicate) boxes have no crop and are excluded by default."""
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        clause, params = self._class_filter(cls_name, cls_in)
        keep = "" if include_suppressed else "AND d.suppressed = 0"
        rows = self._query(
            f"""
            SELECT d.detection_id, d.specimen_id, d.cls_name, d.crop_path,
                   d.x1, d.y1, d.x2, d.y2,
                   s.width AS frame_width, s.height AS frame_height
              FROM {table} d
              JOIN specimen s ON s.specimen_id = d.specimen_id
             WHERE d.specimen_id = ? AND d.crop_path IS NOT NULL {keep} {clause}
             ORDER BY d.detection_id
            """,
            (specimen_id, *params),
        )
        return [
            CropRef(
                detection_id=int(r["detection_id"]),
                specimen_id=int(r["specimen_id"]),
                cls_name=str(r["cls_name"]),
                crop_path=str(r["crop_path"]),
                x1=float(r["x1"] or 0.0),
                y1=float(r["y1"] or 0.0),
                x2=float(r["x2"] or 0.0),
                y2=float(r["y2"] or 0.0),
                frame_width=int(r["frame_width"] or 0),
                frame_height=int(r["frame_height"] or 0),
            )
            for r in rows
        ]

    def specimens_with_rows(self, table: str, *, include_suppressed: bool = False) -> list[int]:
        assert table in _ROW_TABLES, f"unsupported row table {table!r}"
        # only the detection tables carry a `suppressed` flag; a specimen counts only if it has a KEPT row
        keep = "WHERE suppressed = 0" if (table in _DETECTION_TABLES and not include_suppressed) else ""
        rows = self._query(
            f"SELECT DISTINCT specimen_id FROM {table} {keep} ORDER BY specimen_id"
        )
        return [int(r["specimen_id"]) for r in rows]

    def specimens_with_crops(
        self,
        table: str,
        *,
        cls_name: Optional[str] = None,
        cls_in: Optional[Sequence[str]] = None,
        include_suppressed: bool = False,
    ) -> list[int]:
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        clause, params = self._class_filter(cls_name, cls_in)
        keep = "" if include_suppressed else "AND suppressed = 0"
        rows = self._query(
            f"""
            SELECT DISTINCT specimen_id FROM {table}
             WHERE crop_path IS NOT NULL {keep} {clause}
             ORDER BY specimen_id
            """,
            tuple(params),
        )
        return [int(r["specimen_id"]) for r in rows]

    def ruler_classifications(self, specimen_id: int) -> list[sqlite3.Row]:
        return self._query(
            "SELECT * FROM ruler_classification WHERE specimen_id = ? ORDER BY ruler_class_id",
            (specimen_id,),
        )

    def ruler_lattice_crops(self, specimen_id: int) -> list[dict]:
        """The Ruler crops for the lattice CF stage, in the engine's expected dict shape: box +
        detection conf + classifier verdict + the pre-made four-tile collage path. Suppressed
        duplicate boxes and crop-less rows are excluded."""
        rows = self._query(
            """
            SELECT d.detection_id, d.conf AS det_conf, d.crop_path,
                   d.x1, d.y1, d.x2, d.y2,
                   rc.unit_type AS ruler_class, rc.conf AS cls_conf,
                   rc.squarify_path AS tile_four_path
              FROM archival_detection d
              LEFT JOIN ruler_classification rc ON rc.detection_id = d.detection_id
             WHERE d.specimen_id = ? AND d.cls_name = 'Ruler'
               AND d.crop_path IS NOT NULL AND d.suppressed = 0
             ORDER BY d.detection_id
            """,
            (specimen_id,),
        )
        return [dict(r) for r in rows]

    def leaf_instances(self, specimen_id: int) -> list[sqlite3.Row]:
        return self._query(
            "SELECT * FROM leaf_segmentation WHERE specimen_id = ? ORDER BY leaf_id",
            (specimen_id,),
        )

    # ---- specimen mask (SpecimenSegmenter) ------------------------------- #
    def record_specimen_mask(self, specimen_id: int, stem: str, result, dirs) -> None:
        """Write the final + removed mask PNGs under ``dirs.masks`` and upsert the row.

        ``result`` is a :class:`~leafmachine3.core.records.SpecimenMaskResult` (PNG bytes +
        working-frame sizes + paper-sample centers). Keyed on ``specimen_id`` so a re-run
        overwrites in place.
        """
        masks_dir = Path(dirs.masks)
        masks_dir.mkdir(parents=True, exist_ok=True)
        final_path = masks_dir / f"{stem}__SpecimenMask.png"
        refined_path = masks_dir / f"{stem}__SpecimenRefined.png"
        final_path.write_bytes(result.final_png)
        refined_path.write_bytes(result.removed_png)
        centers_json = json.dumps([[int(x), int(y)] for (x, y) in (result.centers or [])])
        self._exec(
            """
            INSERT INTO specimen_mask
                (specimen_id, mask_format, mask_path, refined_path, sample_centers_json,
                 frame_width, frame_height, area_frac, model_name)
            VALUES (?, 'png_path', ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(specimen_id) DO UPDATE SET
                mask_path           = excluded.mask_path,
                refined_path        = excluded.refined_path,
                sample_centers_json = excluded.sample_centers_json,
                frame_width         = excluded.frame_width,
                frame_height        = excluded.frame_height,
                area_frac           = excluded.area_frac,
                model_name          = excluded.model_name
            """,
            (specimen_id, str(final_path), str(refined_path), centers_json,
             int(result.frame_width), int(result.frame_height),
             float(result.area_frac), str(result.model_name)),
        )

    def specimen_mask(self, specimen_id: int) -> Optional[sqlite3.Row]:
        return self._one("SELECT * FROM specimen_mask WHERE specimen_id = ?", (specimen_id,))

    # ---- leaf ECT (ECT stage) -------------------------------------------- #
    # -- bilateral symmetry ------------------------------------------------- #
    def record_bilateral_one(self, row: dict) -> None:
        """Upsert ONE leaf's ``bilateral_symmetry`` row (keyed on leaf_id).

        Upsert rather than delete-then-insert because the stage FANS OUT -- each leaf is its own
        WorkItem, so a per-specimen delete would clobber siblings already written by other workers.
        """
        self._exec(
            """
            INSERT INTO bilateral_symmetry
                (leaf_id, specimen_id, detection_id, instance_index,
                 si_a, a_star, dice, sinuosity,
                 archetype_score, term_symmetry, term_integrity, term_completeness, term_trace,
                 gates_pass, is_archetypal, reasons_json,
                 largest_frac, solidity, perimeter_ratio, hole_frac,
                 kpt_conf_mean, kpt_conf_min, n_midvein_kpts, truncated,
                 angle_cw, crop_w, crop_h, mask_w, mask_h,
                 tip_x, tip_y, base_x, base_y, midvein_json, n_bins, qc_png)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(leaf_id) DO UPDATE SET
                specimen_id=excluded.specimen_id, detection_id=excluded.detection_id,
                instance_index=excluded.instance_index,
                si_a=excluded.si_a, a_star=excluded.a_star, dice=excluded.dice,
                sinuosity=excluded.sinuosity, archetype_score=excluded.archetype_score,
                term_symmetry=excluded.term_symmetry, term_integrity=excluded.term_integrity,
                term_completeness=excluded.term_completeness, term_trace=excluded.term_trace,
                gates_pass=excluded.gates_pass, is_archetypal=excluded.is_archetypal,
                reasons_json=excluded.reasons_json,
                largest_frac=excluded.largest_frac, solidity=excluded.solidity,
                perimeter_ratio=excluded.perimeter_ratio, hole_frac=excluded.hole_frac,
                kpt_conf_mean=excluded.kpt_conf_mean, kpt_conf_min=excluded.kpt_conf_min,
                n_midvein_kpts=excluded.n_midvein_kpts, truncated=excluded.truncated,
                angle_cw=excluded.angle_cw, crop_w=excluded.crop_w, crop_h=excluded.crop_h,
                mask_w=excluded.mask_w, mask_h=excluded.mask_h,
                tip_x=excluded.tip_x, tip_y=excluded.tip_y,
                base_x=excluded.base_x, base_y=excluded.base_y,
                midvein_json=excluded.midvein_json, n_bins=excluded.n_bins, qc_png=excluded.qc_png
            """,
            (int(row["leaf_id"]), int(row["specimen_id"]), row.get("detection_id"),
             row.get("instance_index"),
             row.get("si_a"), row.get("a_star"), row.get("dice"), row.get("sinuosity"),
             row.get("archetype_score"), row.get("term_symmetry"), row.get("term_integrity"),
             row.get("term_completeness"), row.get("term_trace"),
             1 if row.get("gates_pass") else 0, 1 if row.get("is_archetypal") else 0,
             row.get("reasons_json"),
             row.get("largest_frac"), row.get("solidity"), row.get("perimeter_ratio"),
             row.get("hole_frac"), row.get("kpt_conf_mean"), row.get("kpt_conf_min"),
             row.get("n_midvein_kpts"), 1 if row.get("truncated") else 0,
             row.get("angle_cw"), row.get("crop_w"), row.get("crop_h"),
             row.get("mask_w"), row.get("mask_h"),
             row.get("tip_x"), row.get("tip_y"), row.get("base_x"), row.get("base_y"),
             row.get("midvein_json"), row.get("n_bins"), row.get("qc_png")),
        )

    def bilateral_symmetry(self, specimen_id: int) -> list:
        """All ``bilateral_symmetry`` rows for a specimen (one per measured leaf)."""
        return self._query(
            "SELECT * FROM bilateral_symmetry WHERE specimen_id = ? ORDER BY leaf_id",
            (int(specimen_id),),
        )

    def set_bilateral_qc_png(self, leaf_id: int, path: str) -> None:
        """Record the QC image the Reporter wrote for one leaf."""
        self._exec("UPDATE bilateral_symmetry SET qc_png = ? WHERE leaf_id = ?",
                   (str(path), int(leaf_id)))

    def record_leaf_ect_one(self, row: dict) -> None:
        """Upsert ONE leaf's ``leaf_ect`` row (keyed on leaf_id). Used by the fanout ECT stage where
        each leaf is its own WorkItem, so a per-specimen delete-then-insert would clobber siblings."""
        self._exec(
            """
            INSERT INTO leaf_ect
                (leaf_id, specimen_id, detection_id, instance_index, mask_includes,
                 h5_path, radial_png, ect_png, overlay_png, n_outline_points, num_dirs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(leaf_id) DO UPDATE SET
                specimen_id=excluded.specimen_id, detection_id=excluded.detection_id,
                instance_index=excluded.instance_index, mask_includes=excluded.mask_includes,
                h5_path=excluded.h5_path, radial_png=excluded.radial_png, ect_png=excluded.ect_png,
                overlay_png=excluded.overlay_png,
                n_outline_points=excluded.n_outline_points, num_dirs=excluded.num_dirs
            """,
            (int(row["leaf_id"]), int(row["specimen_id"]), row.get("detection_id"),
             row.get("instance_index"), row.get("mask_includes"), row.get("h5_path"),
             row.get("radial_png"), row.get("ect_png"), row.get("overlay_png"),
             row.get("n_outline_points"), row.get("num_dirs")),
        )

    def record_leaf_ect(self, specimen_id: int, rows) -> None:
        """Delete-then-insert this specimen's ``leaf_ect`` rows (one per oriented Leaf_WHOLE leaf)."""
        self._exec("DELETE FROM leaf_ect WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            self._exec(
                """
                INSERT INTO leaf_ect
                    (leaf_id, specimen_id, detection_id, instance_index, mask_includes,
                     h5_path, radial_png, ect_png, overlay_png, n_outline_points, num_dirs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r["leaf_id"]), int(specimen_id), r.get("detection_id"), r.get("instance_index"),
                 r.get("mask_includes"), r.get("h5_path"), r.get("radial_png"), r.get("ect_png"),
                 r.get("overlay_png"), r.get("n_outline_points"), r.get("num_dirs")),
            )

    def leaf_ect(self, specimen_id: int) -> list[sqlite3.Row]:
        return self._query("SELECT * FROM leaf_ect WHERE specimen_id = ? ORDER BY leaf_id", (specimen_id,))

    def record_leaf_momocs(self, specimen_id: int, rows) -> None:
        """Delete-then-insert this specimen's ``leaf_momocs`` rows (one per exported leaf image)."""
        self._exec("DELETE FROM leaf_momocs WHERE specimen_id = ?", (specimen_id,))
        for r in rows:
            self._exec(
                """
                INSERT INTO leaf_momocs
                    (leaf_id, specimen_id, detection_id, instance_index, tree, mask_includes,
                     source_mask, mask_path, json_path, n_outline_points, image_width, image_height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r["leaf_id"]), int(specimen_id), r.get("detection_id"), r.get("instance_index"),
                 r.get("tree"), r.get("mask_includes"), r.get("source_mask"), r.get("mask_path"),
                 r.get("json_path"), r.get("n_outline_points"), r.get("image_width"), r.get("image_height")),
            )

    def leaf_momocs(self, specimen_id: int) -> list[sqlite3.Row]:
        return self._query("SELECT * FROM leaf_momocs WHERE specimen_id = ? ORDER BY leaf_id", (specimen_id,))

    def leaf_momocs_json_paths(self) -> list[str]:
        """Every sheet-level Momit JSON the Momocs stage wrote, in specimen order (run-level rebuild)."""
        return [r["json_path"] for r in self._query(
            "SELECT json_path FROM leaf_momocs WHERE json_path IS NOT NULL "
            "GROUP BY json_path ORDER BY MIN(specimen_id)")]

    # ---- CSV export readers (reports/Data) -------------------------------- #
    # Whole-project, run-once reads backing `reporting.data_export`. They live here because this is
    # the only module allowed to issue SQL; the exporter owns the column ORDER, units and prose.
    # Every selected column carries an explicit alias -- the joins collide on `area_px`, `cls_name`,
    # `conf`, `instance_index` and `detection_id`, so an un-aliased SELECT would silently drop one
    # side. `reporting.data_export` names those aliases, and a test asserts the two agree.

    #: Tables the passthrough exporter may dump verbatim. A whitelist, not a parameter: the table
    #: name is interpolated into the SQL, so it must never come from config or user input.
    EXPORT_PASSTHROUGH_TABLES: tuple[str, ...] = (
        "ruler_CF_lattice", "ruler_CF_lattice_crop", "ruler_FP_marker", "ruler_FP_sheet",
        "project_status",
    )

    def export_specimen_rows(self) -> list[sqlite3.Row]:
        """One row per input image: identity, frame, both CFs, phenology, sheet-mask QC, CF verdict."""
        return self._query(
            """
            SELECT s.specimen_id, s.image_name, s.image_stem, s.original_path, s.working_path,
                   s.width  AS working_width,  s.height AS working_height,
                   s.original_width, s.original_height, s.work_scale,
                   s.normalized, s.downsampled, s.original_mp,
                   s.cf_px_per_cm            AS cf_px_per_cm,
                   s.cf_source               AS cf_source,
                   s.cf_px_per_cm_predicted_by_mp,
                   s.ruler_unit_type, s.ruler_class_type,
                   s.ingested_at,
                   ph.has_leaves, ph.has_flowers, ph.has_fruits,
                   ph.n_leaves AS n_leaf_boxes, ph.n_flowers AS n_flower_boxes,
                   ph.n_fruits AS n_fruit_boxes,
                   sm.area_frac  AS specimen_mask_area_frac,
                   sm.model_name AS specimen_mask_model,
                   rcf.status     AS ruler_cf_status,
                   rcf.confidence AS ruler_cf_confidence,
                   rcf.cf_px_per_cm_measured,
                   rcf.n_ruler_crops, rcf.n_used AS n_ruler_crops_used,
                   rcf.anchor_source AS ruler_cf_anchor_source,
                   fps.sheet_type      AS fp_sheet_type,
                   fps.sheet_status    AS fp_sheet_status,
                   fps.orientation_deg AS fp_orientation_deg,
                   fps.n_fp_detected   AS fp_n_markers_detected,
                   fps.n_fp_used       AS fp_n_markers_used,
                   fps.n_fp_inferred   AS fp_n_markers_inferred,
                   fps.cf_px_per_cm_fp AS fp_cf_px_per_cm,
                   fps.fp_confidence   AS fp_confidence
              FROM specimen s
              LEFT JOIN phenology        ph  ON ph.specimen_id  = s.specimen_id
              LEFT JOIN specimen_mask    sm  ON sm.specimen_id  = s.specimen_id
              LEFT JOIN ruler_CF_lattice rcf ON rcf.specimen_id = s.specimen_id
              LEFT JOIN ruler_FP_sheet   fps ON fps.specimen_id = s.specimen_id
             ORDER BY s.specimen_id
            """
        )

    def export_leaf_rows(self) -> list[sqlite3.Row]:
        """One row per segmented LEAF instance -- the master measurement table.

        Anchored on ``leaf_segmentation`` (the row that defines a leaf instance) and LEFT JOINed to
        every per-leaf measurement table, so a leaf missing landmarks or a petiole still appears with
        NULLs rather than vanishing. ``leaf_landmark_measurement`` joins on
        ``(detection_id, instance_index)`` -- the convention petiole_width and the Reporter already
        use -- because landmarks are keyed by the POSE instance in a crop, not the segmentation
        instance. Suppressed detection boxes are excluded, matching every other read in this module.

        ``leaf_ect`` is deliberately NOT joined: the ECT stage runs after the Reporter (it consumes
        the Reporter's oriented masks), so at export time its rows are either absent or left over
        from a previous run. Exporting them would ship a column that is stale exactly when it looks
        freshest. ECT products are addressable by the same crop box token this table emits.
        """
        return self._query(
            """
            SELECT s.specimen_id, s.image_name, s.image_stem,
                   s.original_path, s.working_path,
                   s.width  AS working_width, s.height AS working_height,
                   s.original_width, s.original_height, s.work_scale, s.downsampled, s.original_mp,
                   s.cf_px_per_cm AS cf_px_per_cm,
                   s.cf_source    AS cf_source,
                   s.cf_px_per_cm_predicted_by_mp,
                   s.ruler_unit_type, s.ruler_class_type,
                   rcf.anchor_source AS ruler_cf_anchor_source,
                   fps.sheet_type      AS fp_sheet_type,
                   fps.sheet_status    AS fp_sheet_status,
                   fps.orientation_deg AS fp_orientation_deg,
                   fps.n_fp_detected   AS fp_n_markers_detected,
                   fps.n_fp_used       AS fp_n_markers_used,
                   fps.n_fp_inferred   AS fp_n_markers_inferred,
                   fps.cf_px_per_cm_fp AS fp_cf_px_per_cm,
                   fps.fp_confidence   AS fp_confidence,
                   s.has_leaves, s.has_flowers, s.has_fruits,

                   pd.detection_id,
                   pd.cls_name AS detection_class,
                   pd.conf     AS detection_conf,
                   pd.x1 AS crop_x1, pd.y1 AS crop_y1, pd.x2 AS crop_x2, pd.y2 AS crop_y2,
                   pd.crop_path,

                   ls.leaf_id, ls.instance_index,
                   ls.conf      AS segmentation_conf,
                   ls.num_parts AS n_mask_parts,
                   ls.area_cm2      AS lamina_area_incl_holes_cm2,
                   ls.perimeter_cm  AS lamina_perimeter_cm,
                   ls.bbox_w_cm, ls.bbox_h_cm,

                   m.area_px       AS lamina_area_incl_holes_px,
                   m.perimeter_px  AS lamina_perimeter_px,
                   m.lamina_area_excl_holes_px, m.lamina_hole_area_px, m.n_holes,
                   m.centroid_x, m.centroid_y,
                   m.convex_hull_area AS convex_hull_area_px,
                   m.convexity, m.concavity, m.circularity, m.aspect_ratio, m.n_vertices,
                   m.bbox_x1, m.bbox_y1, m.bbox_x2, m.bbox_y2,
                   m.rotate_angle,
                   m.rotated_bbox_dim_max AS rotated_bbox_dim_max_px,
                   m.rotated_bbox_dim_min AS rotated_bbox_dim_min_px,
                   m.rotated_bbox_length  AS rotated_bbox_length_px,
                   m.rotated_bbox_width   AS rotated_bbox_width_px,
                   m.circle_cx, m.circle_cy, m.circle_radius AS circle_radius_px,
                   m.oriented_leaf_success, m.oriented_leaf_rotation_angle_degreesCW,

                   lm.lamina_trace_length      AS lamina_trace_length_px,
                   lm.lamina_extent            AS lamina_extent_px,
                   lm.lamina_tip_base_length   AS lamina_tip_base_length_px,
                   lm.leaf_width               AS leaf_width_px,
                   lm.petiole_trace_length     AS petiole_trace_length_px,
                   lm.lamina_trace_length_cm, lm.lamina_extent_cm,
                   lm.lamina_tip_base_length_cm, lm.leaf_width_cm AS landmark_leaf_width_cm,
                   lm.petiole_trace_length_cm,
                   lm.apex_angle, lm.apex_angle_type, lm.base_angle, lm.base_angle_type,
                   lm.lamina_curvature, lm.curvature_point,
                   lm.n_present AS n_landmarks_present,

                   p.width_px  AS petiole_width_px,
                   p.length_px AS petiole_length_px,
                   p.width_cm  AS petiole_width_cm,
                   p.length_cm AS petiole_length_cm,
                   p.n_samples AS petiole_n_samples,
                   p.touches_leaf AS petiole_touches_leaf,
                   p.measure_location AS petiole_measure_location,
                   p.leaf_mass_per_area,

                   b.si_a, b.a_star, b.dice, b.sinuosity,
                   b.archetype_score, b.term_symmetry, b.term_integrity,
                   b.term_completeness, b.term_trace,
                   b.gates_pass, b.is_archetypal,
                   b.largest_frac, b.solidity, b.perimeter_ratio, b.hole_frac,
                   b.kpt_conf_mean, b.kpt_conf_min, b.n_midvein_kpts, b.truncated
              FROM leaf_segmentation ls
              JOIN specimen        s  ON s.specimen_id   = ls.specimen_id
              JOIN plant_detection pd ON pd.detection_id = ls.detection_id
              LEFT JOIN ruler_CF_lattice rcf ON rcf.specimen_id = ls.specimen_id
              LEFT JOIN ruler_FP_sheet   fps ON fps.specimen_id = ls.specimen_id
              LEFT JOIN leaf_morphology   m ON m.leaf_id = ls.leaf_id
              LEFT JOIN leaf_petiole      p ON p.leaf_id = ls.leaf_id
              LEFT JOIN bilateral_symmetry b ON b.leaf_id = ls.leaf_id
              LEFT JOIN leaf_landmark_measurement lm
                     ON lm.detection_id   = ls.detection_id
                    AND lm.instance_index = ls.instance_index
             WHERE ls.cls_name = 'Leaf' AND pd.suppressed = 0
             ORDER BY ls.specimen_id, ls.detection_id, ls.instance_index
            """
        )

    def export_detection_rows(self) -> list[sqlite3.Row]:
        """One row per detection box from BOTH detectors, tagged by ``source``.

        Suppressed duplicates are INCLUDED here (with their flags) -- unlike every consuming read.
        This file is the record of what the detectors proposed, and "this box was considered and
        rejected as a duplicate" is part of that record.
        """
        return self._query(
            """
            SELECT 'archival' AS source, s.image_stem AS image_stem,
                   d.specimen_id AS specimen_id, d.detection_id AS detection_id,
                   d.cls_name AS cls_name, d.conf AS conf,
                   d.x1 AS x1, d.y1 AS y1, d.x2 AS x2, d.y2 AS y2,
                   d.tag AS tag, d.crop_path AS crop_path,
                   d.suppressed AS suppressed, d.suppressed_by AS suppressed_by,
                   d.suppress_overlap AS suppress_overlap
              FROM archival_detection d JOIN specimen s ON s.specimen_id = d.specimen_id
            UNION ALL
            SELECT 'plant' AS source, s.image_stem AS image_stem,
                   d.specimen_id AS specimen_id, d.detection_id AS detection_id,
                   d.cls_name AS cls_name, d.conf AS conf,
                   d.x1 AS x1, d.y1 AS y1, d.x2 AS x2, d.y2 AS y2,
                   d.tag AS tag, d.crop_path AS crop_path,
                   d.suppressed AS suppressed, d.suppressed_by AS suppressed_by,
                   d.suppress_overlap AS suppress_overlap
              FROM plant_detection d JOIN specimen s ON s.specimen_id = d.specimen_id
             ORDER BY specimen_id, source, detection_id
            """
        )

    def export_landmark_rows(self) -> list[sqlite3.Row]:
        """One row per predicted KEYPOINT (31 per leaf), self-described by the seeded schema."""
        return self._query(
            """
            SELECT s.image_stem, k.specimen_id, k.detection_id, k.instance_index,
                   k.kpt_index, k.kpt_name, sch.grp AS kpt_group,
                   k.x, k.y, k.x_crop, k.y_crop, k.conf,
                   pd.x1 AS crop_x1, pd.y1 AS crop_y1, pd.x2 AS crop_x2, pd.y2 AS crop_y2
              FROM leaf_landmark k
              JOIN specimen s   ON s.specimen_id = k.specimen_id
              LEFT JOIN plant_detection pd  ON pd.detection_id = k.detection_id
              LEFT JOIN landmark_schema sch ON sch.kpt_index   = k.kpt_index
             ORDER BY k.specimen_id, k.detection_id, k.instance_index, k.kpt_index
            """
        )

    def export_table(self, table: str) -> list[sqlite3.Row]:
        """Dump one whitelisted audit table verbatim (every column, table order).

        Used for the ruler-CF audit trail and the stage ledger, whose value is completeness rather
        than a curated column set. ``table`` MUST be in :data:`EXPORT_PASSTHROUGH_TABLES`.
        """
        if table not in self.EXPORT_PASSTHROUGH_TABLES:
            raise ValueError(f"table {table!r} is not exportable; expected one of "
                             f"{self.EXPORT_PASSTHROUGH_TABLES}")
        order = {"ruler_CF_lattice": "ORDER BY specimen_id",
                 "ruler_CF_lattice_crop": "ORDER BY specimen_id, crop_index",
                 "ruler_FP_marker": "ORDER BY specimen_id, crop_index",
                 "ruler_FP_sheet": "ORDER BY specimen_id",
                 "project_status": "ORDER BY stage_order"}[table]
        return self._query(f"SELECT * FROM {table} {order}")

    def export_table_columns(self, table: str) -> list[str]:
        """Column names of a whitelisted passthrough table, in table order.

        Lets the exporter write a correct HEADER for a table that happens to be empty -- a zero-row
        CSV with columns says "exported, nothing matched"; a zero-BYTE one says nothing at all.
        """
        if table not in self.EXPORT_PASSTHROUGH_TABLES:
            raise ValueError(f"table {table!r} is not exportable; expected one of "
                             f"{self.EXPORT_PASSTHROUGH_TABLES}")
        return [r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")]

    def export_stage_errors(self) -> list[sqlite3.Row]:
        """Every per-image stage failure, so a run's errors are visible without opening the log."""
        return self._query(
            """
            SELECT s.image_stem, i.specimen_id, i.stage_key, i.state, i.no_work,
                   i.updated_at, i.error_msg
              FROM image_status i JOIN specimen s ON s.specimen_id = i.specimen_id
             WHERE i.state = 'error'
             ORDER BY i.stage_key, i.specimen_id
            """
        )

    def overlay_detections(self, specimen_id: int, *, include_suppressed: bool = False) -> list[dict]:
        """Union of KEPT archival + plant boxes for the Reporter overlay.

        Returns dicts with ``detection_id``, ``cls_name``, ``conf``, ``xyxy=(x1,y1,x2,y2)``,
        ``source``. ``detection_id`` is unique only WITHIN a source (the two tables number
        independently); the Reporter uses it to match archival Ruler boxes to ruler_FP_marker rows.
        Suppressed (same-class duplicate) boxes are excluded by default.
        """
        keep = "" if include_suppressed else "AND suppressed = 0"
        out: list[dict] = []
        for table, source in (("archival_detection", "archival"), ("plant_detection", "plant")):
            for r in self._query(
                f"SELECT detection_id, cls_name, conf, x1, y1, x2, y2 FROM {table} "
                f"WHERE specimen_id = ? {keep} ORDER BY detection_id",
                (specimen_id,),
            ):
                out.append(
                    {
                        "detection_id": int(r["detection_id"]),
                        "cls_name": str(r["cls_name"]),
                        "conf": float(r["conf"]),
                        "xyxy": (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])),
                        "source": source,
                    }
                )
        return out

    @staticmethod
    def _class_filter(
        cls_name: Optional[str], cls_in: Optional[Sequence[str]]
    ) -> tuple[str, tuple[Any, ...]]:
        """Build an optional ``AND cls_name ...`` clause and its bind params."""
        if cls_name is not None:
            return " AND cls_name = ?", (cls_name,)
        if cls_in:
            names = tuple(cls_in)
            placeholders = ",".join("?" for _ in names)
            return f" AND cls_name IN ({placeholders})", names
        return "", ()

    # ---- ledgers --------------------------------------------------------- #
    def mark_image_done(self, specimen_id: int, stage: str, *, no_work: bool = False) -> None:
        """Checkpoint a specimen ``done`` for a stage and bump ``project_status.n_done``."""
        self._exec(
            """
            INSERT INTO image_status (specimen_id, stage_key, state, no_work, updated_at)
            VALUES (?, ?, 'done', ?, datetime('now'))
            ON CONFLICT(specimen_id, stage_key) DO UPDATE SET
                state = 'done', no_work = excluded.no_work,
                updated_at = datetime('now'), error_msg = NULL
            """,
            (specimen_id, stage, int(bool(no_work))),
        )
        self._exec(
            "UPDATE project_status SET n_done = n_done + 1 WHERE stage_key = ?",
            (stage,),
        )

    def mark_image_error(self, specimen_id: int, stage: str, msg: Optional[str]) -> None:
        self._exec(
            """
            INSERT INTO image_status (specimen_id, stage_key, state, no_work, updated_at, error_msg)
            VALUES (?, ?, 'error', 0, datetime('now'), ?)
            ON CONFLICT(specimen_id, stage_key) DO UPDATE SET
                state = 'error', updated_at = datetime('now'), error_msg = excluded.error_msg
            """,
            (specimen_id, stage, msg),
        )

    def mark_stage_running(self, stage: str, n_total: int) -> None:
        """Mark a stage running; seed ``n_done`` from the already-checkpointed specimens."""
        n_done = len(self.done_ids(stage))
        self._exec(
            """
            UPDATE project_status
               SET state = 'running', n_total = ?, n_done = ?,
                   started_at = datetime('now'), finished_at = NULL, error_msg = NULL
             WHERE stage_key = ?
            """,
            (int(n_total), n_done, stage),
        )

    def mark_stage_done(self, stage: str) -> None:
        self._exec(
            "UPDATE project_status SET state = 'done', finished_at = datetime('now') WHERE stage_key = ?",
            (stage,),
        )

    def mark_stage_error(self, stage: str, msg: Optional[str] = None) -> None:
        self._exec(
            """
            UPDATE project_status
               SET state = 'error', finished_at = datetime('now'), error_msg = ?
             WHERE stage_key = ?
            """,
            (msg, stage),
        )

    def mark_stage_complete_no_work(self, stage: str) -> None:
        """Mark a DISABLED stage complete: ``done`` for every specimen (``no_work=1``).

        This satisfies downstream ``depends_on`` gating without the stage ever running.
        """
        with self.transaction():
            self._exec(
                """
                UPDATE project_status
                   SET state = 'done', n_total = (SELECT COUNT(*) FROM specimen),
                       n_done = (SELECT COUNT(*) FROM specimen),
                       started_at = datetime('now'), finished_at = datetime('now'),
                       error_msg = NULL
                 WHERE stage_key = ?
                """,
                (stage,),
            )
            self._exec(
                """
                INSERT INTO image_status (specimen_id, stage_key, state, no_work, updated_at)
                SELECT specimen_id, ?, 'done', 1, datetime('now') FROM specimen WHERE true
                ON CONFLICT(specimen_id, stage_key) DO UPDATE SET
                    state = 'done', no_work = 1, updated_at = datetime('now'), error_msg = NULL
                """,
                (stage,),
            )

    def stage_state(self, stage: str) -> str:
        row = self._one("SELECT state FROM project_status WHERE stage_key = ?", (stage,))
        return str(row["state"]) if row is not None else "pending"

    def reclaim_running(self) -> None:
        """Crash recovery: any stage left ``running`` reverts to ``pending``."""
        self._exec("UPDATE project_status SET state = 'pending' WHERE state = 'running'")

    # ---- settings-hash + restart cascade --------------------------------- #
    def stage_settings_hash(self, stage: str) -> Optional[str]:
        row = self._one("SELECT settings_hash FROM project_status WHERE stage_key = ?", (stage,))
        return None if row is None or row["settings_hash"] is None else str(row["settings_hash"])

    def set_stage_settings_hash(self, stage: str, h: Optional[str]) -> None:
        self._exec("UPDATE project_status SET settings_hash = ? WHERE stage_key = ?", (h, stage))

    def reset_stages(self, keys: Iterable[str], stages_by_key: dict[str, Any]) -> None:
        """``--restart``: purge each stage's output rows, crops, artifacts, and ledger.

        For every key, in ONE transaction: delete its ``owns_tables`` rows (unlinking any
        tracked crop / report files first; FK cascade cleans children), null the duplicated
        specimen/leaf columns it owns, delete its ``image_status`` rows, and reset its
        ``project_status`` to ``pending``. Dependents are expected to already be in ``keys``
        (the caller passes the transitive closure).
        """
        keys = list(keys)
        with self.transaction():
            for key in keys:
                stage = stages_by_key.get(key)
                owns = tuple(getattr(stage, "owns_tables", ()) or ())
                for table in owns:
                    self._unlink_table_artifacts(table)
                    self._exec(f"DELETE FROM {table}")
                if key == "reporter":
                    self._unlink_report_manifest()
                    self._exec("DELETE FROM report_manifest")
                for col in _OWNED_SPECIMEN_COLS.get(key, ()):  # null duplicated specimen cols
                    self._exec(f"UPDATE specimen SET {col} = NULL")
                for col in _OWNED_LEAF_COLS.get(key, ()):      # null cm-grounded leaf metrics
                    self._exec(f"UPDATE leaf_segmentation SET {col} = NULL")
                for col in _OWNED_MORPH_COLS.get(key, ()):     # null orientation fields on morphology
                    self._exec(f"UPDATE leaf_morphology SET {col} = NULL")
                for col in _OWNED_PETIOLE_COLS.get(key, ()):   # null cm-grounded petiole metrics
                    self._exec(f"UPDATE leaf_petiole SET {col} = NULL")
                for col in _OWNED_LM_MEASURE_COLS.get(key, ()):  # null cm-grounded landmark lengths
                    self._exec(f"UPDATE leaf_landmark_measurement SET {col} = NULL")
                self._exec("DELETE FROM image_status WHERE stage_key = ?", (key,))
                self._exec(
                    """
                    UPDATE project_status
                       SET state = 'pending', n_total = 0, n_done = 0, settings_hash = NULL,
                           started_at = NULL, finished_at = NULL, error_msg = NULL
                     WHERE stage_key = ?
                    """,
                    (key,),
                )

    # Path columns a table OWNS (safe to delete when THAT table's stage resets). Default = every
    # path column is this stage's own artifact. A table that stores FOREIGN references must be
    # listed explicitly with only its owned columns, or resetting it would delete another stage's
    # files: ruler_CF_lattice_crop.crop_path -> archival_detection's Ruler crop, .tile_four_path ->
    # ruler_classification's pre-made tile; only rot_path/tick_mask_path are the lattice's own.
    _DEFAULT_ARTIFACT_COLS: tuple[str, ...] = (
        "crop_path", "mask_path", "refined_path", "h5_path", "radial_png", "ect_png", "overlay_png",
        "squarify_path", "qc_png", "json_path",
    )
    _OWNED_ARTIFACT_COLS: dict[str, tuple[str, ...]] = {
        "ruler_CF_lattice_crop": ("rot_path", "tick_mask_path"),
        # The FieldPrism tables store no files (marker geometry and sheet fit only).
        "ruler_FP_marker": (),
        "ruler_FP_sheet": (),
    }

    def _unlink_table_artifacts(self, table: str) -> None:
        """Delete files this table's stage OWNS before its rows are purged on ``--restart``.

        Only columns the table owns are unlinked (see ``_OWNED_ARTIFACT_COLS``): a foreign-reference
        path column (another stage's artifact stored here for provenance) is never deleted."""
        cols = self._table_columns(table)
        for col in self._OWNED_ARTIFACT_COLS.get(table, self._DEFAULT_ARTIFACT_COLS):
            if col not in cols:
                continue
            for r in self._query(f"SELECT {col} FROM {table} WHERE {col} IS NOT NULL"):
                _safe_unlink(r[col])

    def _unlink_report_manifest(self) -> None:
        for r in self._query("SELECT path FROM report_manifest WHERE path IS NOT NULL"):
            _safe_unlink(r["path"])


def _safe_unlink(path: Optional[str]) -> None:
    """Best-effort delete of a tracked artifact file (never raises)."""
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - filesystem edge cases
        log.warning("could not delete artifact %s: %s", path, exc)
