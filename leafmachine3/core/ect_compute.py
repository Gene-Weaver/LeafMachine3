"""Euler Characteristic Transform (ECT) for one oriented Leaf_WHOLE mask.

Uses the MODERN ``ect`` package NATIVELY (github.com/MunchLab/ect): build an ``EmbeddedGraph`` cycle
from the leaf outline, center its bounding box on the origin, scale to the unit circle, and compute
the ECT matrix. No PCA re-projection (that would fight the required tip-up orientation) and none of
LeafMachine2's older ECT code.

The input mask is NOT reconstructed here -- LM3 already produces clean oriented leaf masks in the
Reporter's ``Leaf_Oriented`` products; the ECT stage loads one of those PNGs and passes the boolean
mask in. The ``mask_includes`` label + which product to load is decided by
:func:`ect_product_for` from the user's petiole/holes toggles.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from leafmachine3.core.leaf_outline import douglas_peucker, extract_contour

# (include_petiole, include_holes) -> (mask_includes, leaf_product_key, Leaf_Oriented subfolder, SEG friendly).
# include_holes == True  => holes are PUNCHED OUT of the lamina (the reporter's holes-removed masks).
# include_holes == False => holes are FILLED / unioned in (the reporter's solid-silhouette masks).
_ECT_PRODUCT: dict[tuple[bool, bool], tuple[str, str, str, str]] = {
    (False, False): ("lamina",               "lamina_holes_mask",         "Lamina_Holes_Mask",        "laminaHoles"),
    (False, True):  ("lamina_hole",           "lamina_mask",               "Lamina_Mask",              "lamina"),
    (True,  False): ("lamina_petiole",        "lamina_petiole_holes_mask", "LaminaPetiole_Holes_Mask", "laminaPetioleHoles"),
    (True,  True):  ("lamina_petiole_hole",   "lamina_petiole_mask",       "LaminaPetiole_Mask",       "laminaPetiole"),
}


@dataclass(frozen=True)
class EctProduct:
    mask_includes: str      # "lamina" | "lamina_petiole" | "lamina_hole" | "lamina_petiole_hole"
    product_key: str        # the report leaf_products key that must be exported (Oriented)
    folder: str             # reports/Leaf_Oriented/<folder>/
    seg_friendly: str       # filename friendly used by the reporter (SEG-<friendly>)


def ect_product_for(include_petiole: bool, include_holes: bool) -> EctProduct:
    return EctProduct(*_ECT_PRODUCT[(bool(include_petiole), bool(include_holes))])


@dataclass
class EctResult:
    ect_matrix: np.ndarray          # (num_thresholds, num_dirs) integer ECT -- rows thresholds, cols dirs
    thetas: np.ndarray              # (num_dirs,) direction angles (radians)
    thresholds: np.ndarray          # (num_thresholds,)
    bound_radius: float
    outline_norm: np.ndarray        # (N, 2) unit-circle normalized outline (exactly what the ECT used)
    outline_literal: np.ndarray     # (N, 2) raw px outline of the loaded oriented mask (crop frame)
    outline_simple: Optional[np.ndarray]   # (M, 2) Douglas-Peucker of the normalized outline, or None
    n_points: int


def compute_ect(
    mask,
    *,
    num_dirs: int = 360,
    bound_radius: float = 1.0,
    want_simple: bool = True,
    simplify_tolerance: float = 0.0025,
    simplify_cutoff: int = 500,
) -> Optional[EctResult]:
    """Compute the ECT of a binary leaf ``mask``. Returns ``None`` for an empty/degenerate mask."""
    from ect import ECT, EmbeddedGraph

    contour = extract_contour(mask)                 # ordered px boundary (x, y), largest external
    if contour is None:
        return None
    g = EmbeddedGraph()
    g.add_cycle(contour)
    g.center_coordinates(center_type="bounding_box")   # bbox center -> origin (LM2's ECT centering)
    g.scale_coordinates(float(bound_radius))           # farthest node -> bound_radius (unit circle)
    thresholds = np.linspace(0.0, float(bound_radius), int(num_dirs))
    result = ECT(num_dirs=int(num_dirs), thresholds=thresholds, bound_radius=float(bound_radius)).calculate(g)

    outline_norm = np.asarray(g.coord_matrix, dtype=float)      # same order/count as `contour`
    simple = None
    if want_simple:
        simple = douglas_peucker(outline_norm, simplify_tolerance, simplify_cutoff)
    return EctResult(
        ect_matrix=np.asarray(result.T),
        thetas=np.asarray(result.directions.thetas, dtype=float),
        thresholds=np.asarray(result.thresholds, dtype=float),
        bound_radius=float(bound_radius),
        outline_norm=outline_norm,
        outline_literal=contour,
        outline_simple=simple,
        n_points=int(len(outline_norm)),
    )


# -- tip-up display convention -----------------------------------------------------
# The ECT is computed on the Reporter's ORIENTED mask (tip at top, base at bottom) and the outline
# comes straight from cv2.findContours, so the coordinates are IMAGE coords with y pointing DOWN --
# only translated (bbox center -> origin) and uniformly scaled, never flipped. The ect package's
# direction vector is omega(theta) = (cos theta, sin theta) in that same frame, so the direction
# that points at the TIP (toward -y, the top of the image) is theta = 3*pi/2.
#
# A point (x, y) of a y-down frame is drawn on a y-up screen at angle atan2(-y, x), so direction
# omega(theta) belongs at screen angle -theta. Re-basing the direction axis on the tip,
#
#     phi = (theta - TIP_DIRECTION_RADIANS) mod 2*pi,
#
# makes phi the angle measured CLOCKWISE ON SCREEN starting from the tip: phi=0 North (tip),
# pi/2 East, pi South (base), 3*pi/2 West. Every ECT visual uses phi so they share one origin.
TIP_DIRECTION_RADIANS: float = 1.5 * np.pi


def tip_up_index(thetas) -> int:
    """Index of the direction in ``thetas`` that points at the leaf tip (nearest ``3*pi/2``)."""
    return int(np.argmin(np.abs(np.asarray(thetas, dtype=float) - TIP_DIRECTION_RADIANS)))


def tip_up_directions(ect_matrix, thetas):
    """Re-base an ECT matrix's DIRECTION axis (its columns) on the leaf tip.

    Returns ``(matrix_phi, phi, phi_ref)`` where ``matrix_phi[i, j]`` is chi at ``thresholds[i]``
    for ``phi[j]``, ``phi`` ascends ``0 .. 2*pi`` clockwise-on-screen from the tip, and ``phi_ref``
    is the raw theta that became ``phi = 0`` (exactly ``3*pi/2`` when ``num_dirs % 4 == 0``).
    """
    thetas = np.asarray(thetas, dtype=float)
    k = tip_up_index(thetas)
    phi_ref = float(thetas[k])
    phi = np.roll((thetas - phi_ref) % (2.0 * np.pi), -k)      # ascending 0 .. 2*pi
    return np.roll(np.asarray(ect_matrix), -k, axis=1), phi, phi_ref


def outline_polar(outline_norm, phi_ref: float, *, closed: bool = False):
    """Normalized outline (y-DOWN image coords) -> ``(phi, r)`` on the tip-up polar axes.

    With ``closed=True`` the ring is closed (first point repeated) and ``phi`` is UNWRAPPED rather
    than folded into ``[0, 2*pi)``. That is required for plotting: Matplotlib interpolates polar
    line segments IN THETA, so a contour step across the 2*pi seam (359.99deg -> 0.01deg) would
    otherwise sweep a bogus arc nearly all the way around the disc.
    """
    p = np.asarray(outline_norm, dtype=float)
    if closed and len(p):
        p = np.concatenate([p, p[:1]])
    x, y = p[:, 0], p[:, 1]
    phi = np.arctan2(y, x) - float(phi_ref)
    phi = np.unwrap(phi) if closed else phi % (2.0 * np.pi)
    return phi, np.hypot(x, y)
