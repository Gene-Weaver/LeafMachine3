"""Green_Index_Signorelli_2025.

Signorelli et al. (2025). Plant, Cell & Environment.
https://onlinelibrary.wiley.com/doi/full/10.1111/pce.70102
Preprint, Eq. 12: https://doi.org/10.1101/2023.08.23.554481

    GI = preGI / (12 * (1 - preGI))
    preGI = ((255-|G-165|) + (255-|R-37.5|) + (255-|B-37.5|)) / (3*255)

The index scores closeness to one reference green, R 37.5, G 165, B 37.5. The factor of 12 scaled the
authors' 24-color card to about 0-1, but the formula is not capped there: an 8-bit pixel can reach
63.67, at (38, 165, 38). R and B cannot equal 37.5 on an 8-bit pixel, so the denominator is never zero.
"""
from __future__ import annotations

import numpy as np

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb, _safe_divide


def calculate_green_index(r, g, b):
    r, g, b = _rgb(r, g, b)

    green = 255 - np.abs(g - 165)
    red = 255 - np.abs(r - 37.5)
    blue = 255 - np.abs(b - 37.5)

    numerator = (green + red + blue) / (3 * 255)
    denominator = 12 * (1 - numerator)

    return _safe_divide(numerator, denominator)


METHOD = GreenIndexMethod(
    name="Green_Index_Signorelli_2025",
    column="green_index_signorelli_2025",
    short="GI",
    formula="GI = preGI / (12 * (1 - preGI)); preGI = ((255-|G-165|) + (255-|R-37.5|) + (255-|B-37.5|)) / (3*255)",
    citation="Signorelli et al. (2025), Plant, Cell & Environment",
    url="https://onlinelibrary.wiley.com/doi/full/10.1111/pce.70102",
    calculate=calculate_green_index,
    axis=(0.0, 1.0),
    direction="higher = closer to reference green",
    target_rgb=(37.5, 165.0, 37.5),
)
