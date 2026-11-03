"""Green index formulas, one file per published version, plus the pixel selection they run on.

Each formula lives in a file named exactly for its version (``Green_Index_Signorelli_2025.py``), so
a reported number can be traced to the code and the citation behind it. The Green Index stage,
:mod:`leafmachine3.modules.green_index`, runs them on every oriented leaf cutout.
"""
from __future__ import annotations

from leafmachine3.core.green_index import (
    Excess_Green_Thrash_2022,
    Green_Chromatic_Coordinate_Woebbecke_1995,
    Green_Index_Signorelli_2025,
    Green_Leaf_Index_Louhaichi_2001,
    Greenness_Lazarevic_2022,
    Two_Fold_Red_Excess_Agarwal_2025,
    Vegetation_Index_2_Kawashima_1998,
)
from leafmachine3.core.green_index._common import GreenIndexMethod

#: Every reported index, in column order. Adding a version: add its file, then list it here.
METHODS: tuple[GreenIndexMethod, ...] = (
    Green_Index_Signorelli_2025.METHOD,
    Green_Chromatic_Coordinate_Woebbecke_1995.METHOD,
    Excess_Green_Thrash_2022.METHOD,
    Two_Fold_Red_Excess_Agarwal_2025.METHOD,
    Vegetation_Index_2_Kawashima_1998.METHOD,
    Green_Leaf_Index_Louhaichi_2001.METHOD,
    Greenness_Lazarevic_2022.METHOD,
)
