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
    CropRef,
    DetRow,
    Grounded,
    LeafRow,
    PhenologyResult,
    RulerCFRow,
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
    "archival_detector",
    "plant_detector",
    "phenology_detector",
    "ruler_classifier",
    "ruler_cf",
    "leaf_segmenter",
    "morphology",
    "landmark_detector",
    "landmark_measurements",
    "metric_grounding",
    "reporter",
)

# Duplicated specimen/leaf columns each stage OWNS -- nulled when that stage is reset.
_OWNED_SPECIMEN_COLS: dict[str, tuple[str, ...]] = {
    "phenology_detector": ("has_leaves", "has_flowers", "has_fruits"),
    "ruler_cf": ("cf_px_per_cm", "selected_ruler_cf_id", "ruler_unit_type"),
}
_OWNED_LEAF_COLS: dict[str, tuple[str, ...]] = {
    "metric_grounding": ("area_cm2", "perimeter_cm", "bbox_w_cm", "bbox_h_cm"),
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

    def init_schema(self) -> None:
        """Apply ``schema.sql`` (idempotent), seed ``project_status`` + the landmark reference tables."""
        self.conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
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
                orig_size_bytes, orig_mtime, normalized
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                normalized      = excluded.normalized
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
        ruler_cf_id: Optional[int] = None,
        unit_type: Optional[str] = None,
    ) -> None:
        """Duplicate the winning CF onto the specimen and flag its ``ruler_cf`` provenance."""
        self._exec(
            """
            UPDATE specimen
               SET cf_px_per_cm = ?, selected_ruler_cf_id = ?, ruler_unit_type = ?
             WHERE specimen_id = ?
            """,
            (cf_px_per_cm, ruler_cf_id, unit_type, specimen_id),
        )
        if ruler_cf_id is not None:
            self._exec("UPDATE ruler_cf SET is_selected = 1 WHERE ruler_cf_id = ?", (ruler_cf_id,))

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
                    (specimen_id, cls_id, cls_name, conf, x1, y1, x2, y2, tag, crop_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                ),
            )
            new_id = int(cur.lastrowid)
            r.detection_id = new_id
            ids.append(new_id)
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
                    (specimen_id, detection_id, unit_type, votes_json, conf)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(detection_id) DO UPDATE SET
                    unit_type  = excluded.unit_type,
                    votes_json = excluded.votes_json,
                    conf       = excluded.conf
                """,
                (
                    specimen_id,
                    int(r.detection_id),
                    r.unit_type,
                    json.dumps(r.votes) if r.votes is not None else None,
                    r.conf,
                ),
            )

    def record_ruler_cf(self, specimen_id: int, rows: Sequence[RulerCFRow]) -> list[int]:
        """Upsert per-ruler CF measurements (keyed on ``ruler_class_id``); return their ids."""
        ids: list[int] = []
        for r in rows:
            self._exec(
                """
                INSERT INTO ruler_cf
                    (specimen_id, ruler_class_id, unit_type, minimum_unit, second_unit,
                     px_per_mm, cf_px_per_cm, cf_px_per_inch, agreement, n_ticks, is_valid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ruler_class_id) DO UPDATE SET
                    unit_type      = excluded.unit_type,
                    minimum_unit   = excluded.minimum_unit,
                    second_unit    = excluded.second_unit,
                    px_per_mm      = excluded.px_per_mm,
                    cf_px_per_cm   = excluded.cf_px_per_cm,
                    cf_px_per_inch = excluded.cf_px_per_inch,
                    agreement      = excluded.agreement,
                    n_ticks        = excluded.n_ticks,
                    is_valid       = excluded.is_valid
                """,
                (
                    specimen_id,
                    int(r.ruler_class_id),
                    r.unit_type,
                    r.minimum_unit,
                    r.second_unit,
                    r.px_per_mm,
                    r.cf_px_per_cm,
                    r.cf_px_per_inch,
                    r.agreement,
                    r.n_ticks,
                    int(bool(r.is_valid)),
                ),
            )
            row = self._one(
                "SELECT ruler_cf_id FROM ruler_cf WHERE ruler_class_id = ?",
                (int(r.ruler_class_id),),
            )
            new_id = int(row["ruler_cf_id"])
            r.ruler_cf_id = new_id
            ids.append(new_id)
        return ids

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
                     area_px, perimeter_px, centroid_x, centroid_y,
                     convex_hull_area, convexity, concavity, circularity, aspect_ratio, n_vertices,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                     rotate_angle, rotated_bbox_dim_max, rotated_bbox_dim_min, rotated_bbox_json,
                     circle_cx, circle_cy, circle_radius)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (int(r.leaf_id), specimen_id, int(r.detection_id), int(r.instance_index), r.cls_name,
                 cx1, cy1, cx2, cy2,
                 r.area_px, r.perimeter_px, cen_x, cen_y,
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
                     petiole_trace_length, lamina_curvature,
                     lamina_centroid_x, lamina_centroid_y, n_present)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (specimen_id, int(r.detection_id), int(r.instance_index),
                 r.lamina_trace_length, r.lamina_extent, r.lamina_tip_base_length, r.leaf_width,
                 r.apex_angle, r.apex_angle_type, r.base_angle, r.base_angle_type,
                 r.petiole_trace_length, r.lamina_curvature,
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
    def detections(self, table: str, specimen_id: int) -> list[sqlite3.Row]:
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        return self._query(
            f"SELECT * FROM {table} WHERE specimen_id = ? ORDER BY detection_id",
            (specimen_id,),
        )

    def detection_boxes(self, specimen_id: int, table: str) -> dict[int, tuple]:
        """Map ``detection_id -> (x1, y1, x2, y2)`` (working coords) for one specimen's boxes.

        Used by the Reporter to recover each leaf crop's parent-frame box for per-crop mask exports.
        """
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        return {
            int(r["detection_id"]): (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]))
            for r in self._query(
                f"SELECT detection_id, x1, y1, x2, y2 FROM {table} WHERE specimen_id = ?",
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
    ) -> list[CropRef]:
        """Saved detection crops for a specimen as :class:`CropRef` (frame dims from specimen)."""
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        clause, params = self._class_filter(cls_name, cls_in)
        rows = self._query(
            f"""
            SELECT d.detection_id, d.specimen_id, d.cls_name, d.crop_path,
                   d.x1, d.y1, d.x2, d.y2,
                   s.width AS frame_width, s.height AS frame_height
              FROM {table} d
              JOIN specimen s ON s.specimen_id = d.specimen_id
             WHERE d.specimen_id = ? AND d.crop_path IS NOT NULL {clause}
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

    def specimens_with_rows(self, table: str) -> list[int]:
        assert table in _ROW_TABLES, f"unsupported row table {table!r}"
        rows = self._query(
            f"SELECT DISTINCT specimen_id FROM {table} ORDER BY specimen_id"
        )
        return [int(r["specimen_id"]) for r in rows]

    def specimens_with_crops(
        self,
        table: str,
        *,
        cls_name: Optional[str] = None,
        cls_in: Optional[Sequence[str]] = None,
    ) -> list[int]:
        assert table in _DETECTION_TABLES, f"unknown detection table {table!r}"
        clause, params = self._class_filter(cls_name, cls_in)
        rows = self._query(
            f"""
            SELECT DISTINCT specimen_id FROM {table}
             WHERE crop_path IS NOT NULL {clause}
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

    def leaf_instances(self, specimen_id: int) -> list[sqlite3.Row]:
        return self._query(
            "SELECT * FROM leaf_segmentation WHERE specimen_id = ? ORDER BY leaf_id",
            (specimen_id,),
        )

    def overlay_detections(self, specimen_id: int) -> list[dict]:
        """Union of archival + plant boxes for the Reporter overlay.

        Returns dicts with ``cls_name``, ``conf``, ``xyxy=(x1,y1,x2,y2)``, ``source``.
        """
        out: list[dict] = []
        for table, source in (("archival_detection", "archival"), ("plant_detection", "plant")):
            for r in self._query(
                f"SELECT cls_name, conf, x1, y1, x2, y2 FROM {table} WHERE specimen_id = ? ORDER BY detection_id",
                (specimen_id,),
            ):
                out.append(
                    {
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

    def _unlink_table_artifacts(self, table: str) -> None:
        """Delete any files referenced by a ``crop_path`` column in ``table``."""
        if "crop_path" not in self._table_columns(table):
            return
        for r in self._query(f"SELECT crop_path FROM {table} WHERE crop_path IS NOT NULL"):
            _safe_unlink(r["crop_path"])

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
