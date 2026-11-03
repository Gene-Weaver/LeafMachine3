"""Green_Chromatic_Coordinate_Woebbecke_1995.

Woebbecke et al. (1995). Transactions of the ASAE.
https://doi.org/10.13031/2013.27838
https://www.ars.usda.gov/research/publications/publication/?seqNo115=418040

    GCC = G / (R + G + B)

Also called Normalized Greenness Index (NGI). Greenness_Index_Jia_2004 is the same formula
(https://doi.org/10.1016/j.fcr.2004.02.014), so it is an alias here rather than a separate column.
"""
from __future__ import annotations

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb, _safe_divide


def calculate_green_chromatic_coordinate(r, g, b):
    r, g, b = _rgb(r, g, b)

    return _safe_divide(g, r + g + b)


# Greenness_Index_Jia_2004 (same as GCC)
def calculate_greenness_index(r, g, b):
    return calculate_green_chromatic_coordinate(r, g, b)


METHOD = GreenIndexMethod(
    name="Green_Chromatic_Coordinate_Woebbecke_1995",
    column="green_chromatic_coordinate_woebbecke_1995",
    short="GCC",
    formula="GCC = G / (R + G + B)",
    citation="Woebbecke et al. (1995), Transactions of the ASAE",
    url="https://doi.org/10.13031/2013.27838",
    calculate=calculate_green_chromatic_coordinate,
    axis=(0.2, 0.5),
    direction="higher = greener",
)
