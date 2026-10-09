"""Lattice ruler conversion-factor engine, ported from LM3_Ruler_Segmentation.

Pure cv2/numpy/PIL/math tick-lattice measurement (no models, no weights). The
public surface is :class:`RulerCFLattice` (one class, one record, one QC panel) plus
the column helpers its schema is parsed from (the two lattice tables and the two
FieldPrism tables, ``ruler_FP_marker`` / ``ruler_FP_sheet``; FieldPrism markers are
measured by :mod:`.fieldprism`). The four-tile squarify tile
is pre-made by the RulerClassifier stage (``_ruler_squarify/``) and handed in per
crop, so this package never squarifies.

Credit: the tick-lattice method is the LM3_Ruler_Segmentation research pipeline
(ruler_units / ruler_lattice / ruler_analysis / ruler_sheet_cf / ruler_qc).
"""
from .engine import (
    RulerCFLattice,
    ENGINE_VERSION,
    SCHEMA_SQL,
    FP_RECONCILE_WEIGHT,
    image_columns,
    crop_columns,
    fp_marker_columns,
    fp_sheet_columns,
    schema_tables,
    migration_columns,
)

__all__ = [
    "RulerCFLattice",
    "ENGINE_VERSION",
    "SCHEMA_SQL",
    "FP_RECONCILE_WEIGHT",
    "image_columns",
    "crop_columns",
    "fp_marker_columns",
    "fp_sheet_columns",
    "schema_tables",
    "migration_columns",
]
