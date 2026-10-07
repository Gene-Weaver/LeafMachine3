"""Lattice ruler conversion-factor engine, ported from LM3_Ruler_Segmentation.

Pure cv2/numpy/PIL/math tick-lattice measurement (no models, no weights). The
public surface is :class:`RulerCFLattice` (one class, one record, one QC panel) plus
the column helpers its two-table schema is parsed from. The four-tile squarify tile
is pre-made by the RulerClassifier stage (``_ruler_squarify/``) and handed in per
crop, so this package never squarifies.

Credit: the tick-lattice method is the LM3_Ruler_Segmentation research pipeline
(ruler_units / ruler_lattice / ruler_analysis / ruler_sheet_cf / ruler_qc).
"""
from .engine import (
    RulerCFLattice,
    ENGINE_VERSION,
    SCHEMA_SQL,
    image_columns,
    crop_columns,
)

__all__ = [
    "RulerCFLattice",
    "ENGINE_VERSION",
    "SCHEMA_SQL",
    "image_columns",
    "crop_columns",
]
