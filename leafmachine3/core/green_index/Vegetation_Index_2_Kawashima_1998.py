"""Vegetation_Index_2_Kawashima_1998.

Kawashima & Nakatani (1998). Annals of Botany.
https://doi.org/10.1006/anbo.1997.0544

    VI2 = (R - B) / (R + B)

Also called Normalized Difference Pigment Index (NDPI). It was developed to estimate chlorophyll and
correlates negatively with it, so a higher value does not mean a greener leaf.
"""
from __future__ import annotations

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb, _safe_divide


def calculate_vegetation_index_2(r, g, b):
    r, g, b = _rgb(r, g, b)

    return _safe_divide(r - b, r + b)


METHOD = GreenIndexMethod(
    name="Vegetation_Index_2_Kawashima_1998",
    column="vegetation_index_2_kawashima_1998",
    short="VI2",
    formula="VI2 = (R - B) / (R + B)",
    citation="Kawashima & Nakatani (1998), Annals of Botany",
    url="https://doi.org/10.1006/anbo.1997.0544",
    calculate=calculate_vegetation_index_2,
    axis=(-0.5, 1.0),
    direction="higher = more red, less blue",
)
