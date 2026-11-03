"""Green_Leaf_Index_Louhaichi_2001.

Louhaichi et al. (2001). Geocarto International.
https://doi.org/10.1080/10106040108542184

    GLI = (2G - R - B) / (2G + R + B)
"""
from __future__ import annotations

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb, _safe_divide


def calculate_green_leaf_index(r, g, b):
    r, g, b = _rgb(r, g, b)

    numerator = 2 * g - r - b
    denominator = 2 * g + r + b

    return _safe_divide(numerator, denominator)


METHOD = GreenIndexMethod(
    name="Green_Leaf_Index_Louhaichi_2001",
    column="green_leaf_index_louhaichi_2001",
    short="GLI",
    formula="GLI = (2G - R - B) / (2G + R + B)",
    citation="Louhaichi et al. (2001), Geocarto International",
    url="https://doi.org/10.1080/10106040108542184",
    calculate=calculate_green_leaf_index,
    axis=(-0.3, 0.4),
    direction="higher = greener",
)
