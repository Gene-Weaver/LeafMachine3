"""What the green index formula files share: safe arithmetic and the description of one method."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def _rgb(r, g, b):
    """Convert RGB values to float to prevent uint8 overflow."""
    return (np.asarray(c, dtype=float) for c in (r, g, b))


def _safe_divide(numerator, denominator):
    """Return NaN where denominator is zero. Supports scalars and arrays."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            np.asarray(denominator) != 0,
            np.asarray(numerator) / denominator,
            np.nan
        )
    return result.item() if np.ndim(result) == 0 else result


@dataclass(frozen=True)
class GreenIndexMethod:
    """One published index: its formula, where it came from, and how the summary image shows it."""
    name: str                             # the file's name: Green_Index_Signorelli_2025
    column: str                           # green_index.csv + leaf_green_index column
    short: str                            # label on the summary image: GI
    formula: str
    citation: str
    url: str
    calculate: Callable                   # (r, g, b) -> per-pixel values
    axis: tuple[float, float]             # fixed x range of the histogram, so leaves compare across images
    direction: str                        # what a higher value means, printed on the summary image
    target_rgb: Optional[tuple[float, float, float]] = None   # a reference color, when the index has one
    window: Optional[tuple[float, float]] = None              # a published value window, shaded on the histogram
