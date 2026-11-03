"""Greenness_Lazarevic_2022.

Lazarevic et al. (2022). GCB Bioenergy.
https://onlinelibrary.wiley.com/doi/full/10.1111/gcbb.12999

    GR = (2G - R - B) / (2R + G + B)

Alternative formulation used by some commercial phenotyping systems under the name Green Leaf Index.
"""
from __future__ import annotations

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb, _safe_divide


def calculate_greenness(r, g, b):
    r, g, b = _rgb(r, g, b)

    numerator = 2 * g - r - b
    denominator = 2 * r + g + b

    return _safe_divide(numerator, denominator)


METHOD = GreenIndexMethod(
    name="Greenness_Lazarevic_2022",
    column="greenness_lazarevic_2022",
    short="GR",
    formula="GR = (2G - R - B) / (2R + G + B)",
    citation="Lazarevic et al. (2022), GCB Bioenergy",
    url="https://onlinelibrary.wiley.com/doi/full/10.1111/gcbb.12999",
    calculate=calculate_greenness,
    axis=(-0.3, 0.4),
    direction="higher = greener",
)
