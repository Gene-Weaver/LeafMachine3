"""Excess_Green_Thrash_2022.

Thrash et al. (2022). Applications in Plant Sciences.
https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11502

    ExG = 2G - R - B

Original index associated with Woebbecke et al. (1995), https://doi.org/10.13031/2013.27838, whose
raw-RGB Vegetation_Index_Woebbecke_1995 is the same formula, so it is an alias here rather than a
separate column.

Thrash et al. used ExG values of 29 to 202 to identify green pixels and estimate foliar area. Dried
herbarium tissue is mostly browner than that window: it held no pixels at all on about half of the
leaves tested. So the leaf's main value is the mean ExG of all its used pixels, and the strict
window result is reported beside it.
"""
from __future__ import annotations

import numpy as np

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb

THRASH_WINDOW: tuple[int, int] = (29, 202)


def calculate_excess_green(r, g, b):
    r, g, b = _rgb(r, g, b)

    return 2 * g - r - b


def excess_green(r, g, b):
    """Thrash et al. (2022) implementation, on one leaf's pixels.

    The published version takes a whole image and a plant count. Here the lamina mask has already
    selected the leaf, and each image is one leaf, so it takes pixel values and has no plant count.

    Returns:
        exg: Per-pixel Excess Green values
        mask: Boolean mask (29 <= ExG <= 202)
        green_pixels: Number of retained pixels
        mean_greenness: Mean ExG of retained pixels (NaN when none are retained)
    """
    exg = calculate_excess_green(r, g, b)
    mask = (exg >= THRASH_WINDOW[0]) & (exg <= THRASH_WINDOW[1])

    green_pixels = np.count_nonzero(mask)
    mean_greenness = np.mean(exg[mask]) if green_pixels else np.nan

    return exg, mask, green_pixels, mean_greenness


# Vegetation_Index_Woebbecke_1995 (raw-RGB form of ExG)
def calculate_vegetation_index(r, g, b):
    return calculate_excess_green(r, g, b)


METHOD = GreenIndexMethod(
    name="Excess_Green_Thrash_2022",
    column="excess_green_thrash_2022",
    short="ExG",
    formula="ExG = 2G - R - B",
    citation="Thrash et al. (2022), Applications in Plant Sciences",
    url="https://bsapubs.onlinelibrary.wiley.com/doi/10.1002/aps3.11502",
    calculate=calculate_excess_green,
    axis=(-120.0, 202.0),
    direction="higher = greener",
    window=(float(THRASH_WINDOW[0]), float(THRASH_WINDOW[1])),
)
