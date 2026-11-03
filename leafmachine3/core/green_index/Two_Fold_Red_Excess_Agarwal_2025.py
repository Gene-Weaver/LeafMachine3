"""Two_Fold_Red_Excess_Agarwal_2025.

Agarwal et al. (2025). Plant Methods.
https://link.springer.com/article/10.1186/s13007-025-01339-y

    TREx = (R-G) + (R-B) = 2R-G-B

This index rises with red, so a higher value does not mean a greener leaf.
"""
from __future__ import annotations

from leafmachine3.core.green_index._common import GreenIndexMethod, _rgb


def calculate_two_fold_red_excess(r, g, b):
    r, g, b = _rgb(r, g, b)

    return 2 * r - g - b


METHOD = GreenIndexMethod(
    name="Two_Fold_Red_Excess_Agarwal_2025",
    column="two_fold_red_excess_agarwal_2025",
    short="TREx",
    formula="TREx = (R-G) + (R-B) = 2R-G-B",
    citation="Agarwal et al. (2025), Plant Methods",
    url="https://link.springer.com/article/10.1186/s13007-025-01339-y",
    calculate=calculate_two_fold_red_excess,
    axis=(-120.0, 300.0),
    direction="higher = redder",
)
