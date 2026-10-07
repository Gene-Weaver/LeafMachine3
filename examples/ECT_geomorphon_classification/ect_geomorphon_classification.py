#!/usr/bin/env python
"""Geomorphon classification of a leaf's ECT matrix -- a faithful port of the LeafMachine2 method.

Treats the ECT matrix as a TERRAIN and runs a geomorphon classifier over it: every cell is compared
against its local neighborhood and labeled flat / ridge-peak / valley-pit, which on a leaf's ECT
separates LOBES (broad, flat-topped excursions) from TEETH (narrow serrations). The per-leaf
``fraction_teeth`` / ``fraction_lobe`` that fall out are what LeafMachine2's ridgeline and margin
analyses are built on.

Ported from LeafMachine2 (``leafmachine2/ect_methods/utils_metrics.py``):

    classify_geomorphon(matrix, search_distance=1, flatness_threshold=1, threshold=2)
    visualize_geomorphon_stages(matrix, summary, False, path, fullname, shape, dpi=400)

and driven with LM2's own ECT recipe from ``leafmachine2/ect_methods/leaf_ect.py``
(``LeafECT.compute_ect_for_contour``), because the classifier's integer thresholds are tuned to the
matrix that recipe produces:

    contour -> shapely simplify(0.001) above 2000 pts -> EmbeddedGraph.add_cycle
            -> center_coordinates("bounding_box") -> project_coordinates("pca") -> scale_coordinates()
            -> ECT(num_dirs=N, num_thresh=N).calculate(G, override_bound_radius=1)

``N`` is swept over ``DIRS`` -- 256 (LM2's own default), 360 (LM3's default) and 720 (what LM3
currently ships) -- so the same leaf can be compared across resolutions. Each run writes one 4x4
summary sheet: rows 1-2 are LM2's original Cartesian figure, rows 3-4 repeat those six panels in
the polar projection.

The TRANSFORM is deliberately LM2's, not ``leafmachine3.core.ect_compute``'s -- LM3's differs in
three ways that would each invalidate the geomorphon's tuning:

  * LM3 skips the PCA projection (it needs the Reporter's tip-up orientation preserved);
  * LM3 sweeps thresholds over ``[0, bound_radius]``, LM2 over ``[-1, 1]`` -- the geomorphon's
    ``threshold=2`` and its "leaf bounds" notion assume the LM2 range;
  * LM3 stores the matrix transposed (thresholds x directions) relative to LM2.

The RENDERING, however, is LM3's throughout -- both what is shown and how:

  * SELECTION (radial only). ``compute_ect`` sweeps ``linspace(0, bound_radius, n)``, so an LM3
    matrix only ever holds NON-NEGATIVE thresholds; LM2 sweeps ``-1..1``. The RADIAL panels take
    LM3's ``thresholds >= 0`` half, which is what lets them use ``set_rlim([0, bound_radius])`` and
    so puts the threshold axis on the same scale as the contour's own ``|p|`` -- support tangency
    reads correctly again. The CARTESIAN panels keep the full ``-1..1`` matrix: an image needs no
    non-negative axis, and the whole matrix keeps them square (N x N) instead of half-height. The
    price is that the two halves of the sheet cover different threshold extents; the footnote says so.
  * DIRECTION AXIS. Cartesian and polar panels alike re-base the columns on the tip with
    ``tip_up_directions``, and the Cartesian ones run threshold bottom-up (``origin="lower"``) --
    together that is exactly ``render_cartesian_ect``'s ``_colorize(np.flipud(tip_up(...)))``.
  * POLAR CONSTRUCTION. The radial panels are built as ``_polar_ect_axes`` builds them, using LM3's
    ``outline_polar`` and the transform's real ``thetas`` / ``thresholds``; see that function below.

The CLASSIFICATION is never restricted: ``classify_geomorphon`` runs over the whole LM2 matrix, so
the reported fractions stay LM2's. Each sheet's footnote says so.

Usage
-----
    # one mask -> ./<mask stem>/
    python examples/ECT_geomorphon_classification/ect_geomorphon_classification.py mask.png

    # the whole sample set, each leaf into its own folder beside this script
    python examples/ECT_geomorphon_classification/ect_geomorphon_classification.py \
        examples/ECT_geomorphon_classification/ECT_geomorphon_classification_masks/*.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors                      # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leafmachine3.core.ect_compute import (                   # noqa: E402
    outline_polar,
    tip_up_directions,
    tip_up_index,
)
from leafmachine3.core.leaf_outline import extract_contour   # noqa: E402

# LM2 LeafECT defaults (leaf_ect.py:13) and its simplify settings (compute_ect_for_contour).
# 256 is LM2's own default; 360 is LM3's default num_dirs and 720 is what LM3 currently ships.
# num_thresh always tracks num_dirs, exactly as LM2 pairs them.
DIRS = (256, 360, 720)
SIMPLIFY_TOLERANCE = 0.001
SIMPLIFY_CUTOFF = 2000
# LM2 geomorphon call site (utils_metrics.py:1764).
SEARCH_DISTANCE = 1
FLATNESS_THRESHOLD = 1
THRESHOLD = 2
DPI = 400


# --------------------------------------------------------------------------------------------- #
# LM2's ECT recipe
# --------------------------------------------------------------------------------------------- #
def lm2_ect_matrix(mask: np.ndarray, num_dirs: int):
    """``(ect_matrix, contour_points, normalized_coords)`` as ``LeafECT.compute_ect_for_contour``.

    ``contour_points`` is the raw pixel contour LM2 draws in the Contour panel; ``normalized_coords``
    is the graph AFTER centering, PCA projection and scaling -- i.e. the exact frame the ECT was
    computed in, which is the only frame in which a polar overlay lines up with the transform.
    """
    from ect import ECT, EmbeddedGraph
    from shapely.geometry import LineString

    points = extract_contour(mask)
    if points is None or len(points) < 3:
        return None, None, None, None
    # LM2 only simplifies ABOVE the cutoff; below it the raw contour goes straight through.
    if len(points) >= SIMPLIFY_CUTOFF:
        simplified = LineString(points).simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
        points = np.asarray(simplified.coords)

    g = EmbeddedGraph()
    g.add_cycle(points)
    g.center_coordinates(center_type="bounding_box")
    g.project_coordinates(projection_type="pca")     # LM2 projects; LM3 deliberately does not
    g.scale_coordinates()
    result = ECT(num_dirs=num_dirs, num_thresh=num_dirs).calculate(g, override_bound_radius=1)
    # thetas/thresholds are the REAL axes of the transform, not index ranges -- the radial panels
    # need them (see the polar block below), and LM2 throws them away by keeping only the matrix.
    axes_ = (np.asarray(result.directions.thetas, dtype=float),
             np.asarray(result.thresholds, dtype=float))
    return np.asarray(result), points, np.asarray(g.coord_matrix, dtype=float), axes_


# --------------------------------------------------------------------------------------------- #
# Port of utils_metrics.define_leaf_bounds / classify_geomorphon
# --------------------------------------------------------------------------------------------- #
def define_leaf_bounds(matrix):
    rows, cols = matrix.shape
    n_cells = rows * cols
    n_cells_leaf = np.count_nonzero(matrix)
    return n_cells, n_cells_leaf, n_cells - n_cells_leaf


def classify_geomorphon(matrix_og, search_distance=SEARCH_DISTANCE,
                        flatness_threshold=FLATNESS_THRESHOLD, threshold=THRESHOLD):
    """Classify ECT cells as flat/lobe (1), ridge-peak transition (2), or valley-pit/teeth (3).

    Returns ``(geomorphon_classes, class_summary)``. The summary carries the intermediate stages the
    visual needs plus the ``fraction_teeth`` / ``fraction_lobe`` / ``fraction_flat`` LM2 records.
    """
    n_cells, n_cells_leaf, n_cells_bg = define_leaf_bounds(matrix_og)

    matrix_og = matrix_og.T
    matrix = np.where(matrix_og >= threshold, matrix_og, 0)
    rows, cols = matrix.shape
    geomorphon_classes = np.zeros_like(matrix, dtype=int)

    for i in range(rows):
        for j in range(cols):
            center_height = matrix[i, j]
            if center_height == 0:
                continue                                  # below threshold -> background

            min_row, max_row = max(i - search_distance, 0), min(i + search_distance + 1, rows)
            min_col, max_col = max(j - search_distance, 0), min(j + search_distance + 1, cols)
            neighborhood = matrix[min_row:max_row, min_col:max_col]
            relative_heights = neighborhood - center_height
            max_height, min_height = np.max(relative_heights), np.min(relative_heights)

            if abs(max_height) < flatness_threshold and abs(min_height) < flatness_threshold:
                geomorphon_classes[i, j] = 1                                  # Flat (lobes)
            elif max_height > abs(min_height):
                flat_cells = np.abs(neighborhood - center_height) < flatness_threshold
                if np.any(flat_cells):
                    flat_original = matrix_og[min_row:max_row, min_col:max_col][flat_cells]
                    # LM2 keeps a Plateau class (4) here but never colors it; preserved verbatim.
                    geomorphon_classes[i, j] = 4 if center_height > np.max(flat_original) else 2
                else:
                    geomorphon_classes[i, j] = 2                              # Ridge/Peak
            else:
                geomorphon_classes[i, j] = 3                                  # Valley/Pit (teeth)

    initial_classes = geomorphon_classes.copy()

    # A class-3 cell touching a class-1 cell becomes class 1: teeth adjacent to a lobe are lobe.
    search_matrix = geomorphon_classes.copy()
    updated_matrix = geomorphon_classes.copy()
    for i in range(rows):
        for j in range(cols):
            if search_matrix[i, j] == 1:
                for ni in range(max(i - 1, 0), min(i + 2, rows)):
                    for nj in range(max(j - 1, 0), min(j + 2, cols)):
                        if search_matrix[ni, nj] == 3:
                            updated_matrix[ni, nj] = 1
    geomorphon_classes = updated_matrix
    intermediate_classes = geomorphon_classes.copy()

    # Resolve the transition class: toward lobe if any lobe exists, toward teeth otherwise.
    if np.any(geomorphon_classes == 1):
        geomorphon_classes[geomorphon_classes == 2] = 1
    else:
        geomorphon_classes[geomorphon_classes == 2] = 3
    final_classes = geomorphon_classes.copy()

    class_labels = {0: "background", 1: "lobe", 2: "transition", 3: "teeth"}
    class_summary = {label: 0 for label in class_labels.values()}
    unique, counts = np.unique(geomorphon_classes, return_counts=True)
    for class_id, count in zip(unique, counts):
        if class_id in class_labels:
            class_summary[class_labels[class_id]] = count

    class_summary["n_cells"] = n_cells
    class_summary["n_cells_leaf"] = n_cells_leaf
    class_summary["n_cells_bg"] = n_cells_bg
    class_summary["fraction_teeth"] = class_summary["teeth"] / n_cells_leaf
    class_summary["fraction_lobe"] = class_summary["lobe"] / n_cells_leaf
    class_summary["fraction_flat"] = (class_summary["background"] - n_cells_bg) / n_cells_leaf
    class_summary["thresholded_matrix"] = matrix
    class_summary["initial_classes"] = initial_classes
    class_summary["post_transition_classes"] = intermediate_classes
    class_summary["final_classes"] = final_classes
    return geomorphon_classes, class_summary



# --------------------------------------------------------------------------------------------- #
# Port of utils_metrics.visualize_geomorphon_stages, widened to a 4x4 summary sheet
#
# Rows 1-2 are LM2's original figure verbatim (Cartesian). Rows 3-4 repeat the SAME six panels in
# the polar projection LM2 uses in visualize_geomorphon_stages_with_polar (theta = direction,
# r = threshold, zero at North, clockwise) -- so the ECT is shown both unrolled and as the disc.
#
# Two deliberate departures from LM2's polar helper, both so rows 3-4 really are rows 1-2:
#   * LM2 hardcodes vmin=0, vmax=6 there; here the polar rows reuse the SAME Normalize as the
#     Cartesian rows, or the two halves of the sheet would not be comparable;
#   * LM2 derives its polar angles from the matrix's COLUMN count and its radii from the ROW count
#     (harmless only because it always runs square 256x256). Here angles come from the direction
#     axis and radii from the threshold axis explicitly, so any num_dirs works.
# --------------------------------------------------------------------------------------------- #
CLASS_COLORS = {0: "white", 1: "blue", 2: "orange", 3: "green", 5: "black"}
CLASS_LABELS = {0: "Background", 1: "Lobe", 2: "Transition", 3: "Serrate", 5: "Leaf Bounds"}


#: Extra whole-figure spin applied to every radial panel, in DEGREES counter-clockwise, ON TOP of
#: LM3's placement. 0 = exactly what leafmachine3.reporting.ect_viz._polar_ect_axes draws.
EXTRA_ROTATION_DEG = 0.0

#: Turn the whole CARTESIAN half of the sheet (rows 1-2) half a turn.
#:
#: For the five MATRIX panels it is ``np.rot90(arr, 2)`` before ``imshow``, which composes correctly
#: with ``origin="lower"``: reversing both array axes reverses both image axes, so the drawn panel
#: is the 180-degree rotation of what it would otherwise be (verified pixel-exact at 1 cell : 1 px).
#: For the Contour panel -- a line plot, not an image -- it is a point reflection through the
#: outline's own bounding-box center, which turns the shape while leaving it in the same extent, so
#: the axis ticks still read the original pixel range.
#:
#: The RADIAL panels (rows 3-4), including the radial contour, are untouched; their own knob is
#: EXTRA_ROTATION_DEG.
CARTESIAN_ROTATE_180 = True


def _polar_ect_axes(fig, cell, matrix, thetas, thresholds, title, *, cmap=None, norm=None):
    """One radial panel, built the way ``leafmachine3.reporting.ect_viz._polar_ect_axes`` builds it.

    Three things this gets from LM3 that a naive polar plot gets wrong:

      * the RADIUS is the transform's real ``thresholds`` array, not ``linspace(0, 1, num_thresh)``.
        Under LM2's recipe those run -1..1 (``override_bound_radius=1`` -> ``linspace(-r, r, n)``),
        so the inner half of the disc is the NEGATIVE thresholds and the rim is +1;
      * the DIRECTION axis is re-based with ``tip_up_directions`` so column 0 is the direction
        nearest ``3*pi/2``, and ``phi`` -- not a raw index ramp -- is the angular coordinate;
      * ``theta_direction(-1)`` with ``theta_offset(-phi_ref)`` puts direction theta at screen angle
        exactly ``-theta``, which is where a y-down direction vector belongs on a y-up screen.
        ``set_theta_zero_location("N")`` is off by a quarter turn from this.

    Caveat worth knowing: LM3's tip-up convention assumes the Reporter's oriented mask, where the
    leaf tip really is at ``3*pi/2``. LM2's recipe PCA-projects first, so the re-base lands on the
    shape's principal axis rather than the botanical tip. The construction is still LM3's; only the
    meaning of "up" differs, and it is at least consistent across every panel in the sheet.
    """
    matrix_phi, phi, phi_ref = tip_up_directions(matrix, thetas)
    ax = fig.add_subplot(cell, projection="polar")
    PHI, R = np.meshgrid(phi, np.asarray(thresholds))
    ax.pcolormesh(PHI, R, matrix_phi, cmap=cmap, norm=norm, shading="auto")
    ax.set_theta_direction(-1)
    ax.set_theta_offset(-phi_ref + np.radians(EXTRA_ROTATION_DEG))
    ax.set_rlim([float(np.min(thresholds)), float(np.max(thresholds))])
    _label_polar(ax, title)
    return ax, phi_ref


def _label_polar(ax, title):
    ax.set_title(title)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
    ax.set_yticklabels([])
    ax.grid(alpha=0.25)


def visualize_geomorphon_stages(matrix_og, geom_classes_summary, path_geomorphon, name,
                                leaf_data, coords_norm, thetas, thresholds, dpi=DPI):
    """LM2's summary sheet: contour, raw + thresholded ECT, and the three classification stages,
    given once in Cartesian (rows 1-2) and once in polar (rows 3-4)."""
    fraction_teeth = f"{round(geom_classes_summary['fraction_teeth'], 4)}"
    fraction_lobe = f"{round(geom_classes_summary['fraction_lobe'], 4)}"
    fraction_flat = f"{round(geom_classes_summary['fraction_flat'], 4)}"
    n_cells = f"{round(geom_classes_summary['n_cells'], 4)}"
    n_cells_leaf = f"{round(geom_classes_summary['n_cells_leaf'], 4)}"
    n_cells_bg = f"{round(geom_classes_summary['n_cells_bg'], 4)}"

    matrix_og = matrix_og.T                                   # -> (num_thresh, num_dirs)
    thresholded_matrix = geom_classes_summary["thresholded_matrix"]
    initial_classes = geom_classes_summary["initial_classes"]
    post_transition_classes = geom_classes_summary["post_transition_classes"]
    final_classes = geom_classes_summary["final_classes"]

    leaf_bounds = np.where(matrix_og >= 1, 1, 0)
    modified_initial_classes = initial_classes.copy()
    modified_initial_classes[(initial_classes == 0) & (leaf_bounds == 1)] = 5
    modified_post_transition_classes = post_transition_classes.copy()
    modified_post_transition_classes[(post_transition_classes == 0) & (leaf_bounds == 1)] = 5
    modified_final_classes = final_classes.copy()
    modified_final_classes[(final_classes == 0) & (leaf_bounds == 1)] = 5

    # -- LM3's SELECTION, applied to the RADIAL panels only ---------------------------------------
    # leafmachine3.core.ect_compute.compute_ect sweeps `np.linspace(0, bound_radius, num_dirs)`, so
    # an LM3 matrix only ever holds NON-NEGATIVE thresholds. LM2's recipe sweeps -1..1, so half of
    # its rows are a region LM3 never shows.
    #
    # The Cartesian panels keep the FULL -1..1 matrix: nothing about an image needs a non-negative
    # axis, and showing all of it keeps them square (N x N) rather than half-height. The radial
    # panels take the >= 0 selection, because a polar radius cannot go negative without remapping --
    # and taking it is what lets them use LM3's `set_rlim([0, bound_radius])`, which puts the
    # threshold axis on the same scale as the contour's own |p| so support tangency reads correctly.
    # Consequence, spelled out in the footnote: the two halves of the sheet cover different extents.
    #
    # The CLASSIFICATION is untouched either way: classify_geomorphon runs over the whole LM2
    # matrix, so the fractions below remain LM2's.
    thresholds = np.asarray(thresholds)
    keep = thresholds >= 0.0
    thresholds_radial = thresholds[keep]
    sel = lambda m: m[keep]                      # noqa: E731 -- LM3's row selection, radial only

    colormap = plt.cm.hot
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=np.max(matrix_og))
    cmap = mcolors.ListedColormap([CLASS_COLORS[key] for key in sorted(CLASS_COLORS.keys())])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(CLASS_COLORS) + 0.5),
                                ncolors=len(CLASS_COLORS))
    stages = (("Initial Classification", modified_initial_classes),
              ("Post-Transition Classification", modified_post_transition_classes),
              ("Final Classification", modified_final_classes))
    geomorphon_text = (
        f"Fraction Teeth: {fraction_teeth}\n"
        f"Fraction Lobe: {fraction_lobe}\n"
        f"Fraction Flat: {fraction_flat}\n"
        f"N Cells: {n_cells}\n"
        f"N Cells (Leaf): {n_cells_leaf}\n"
        f"N Cells (BG): {n_cells_bg}"
    )
    selection_note = (
        f"Fractions: FULL LM2 matrix, thresholds -1..1.\n"
        f"Cartesian: full matrix, all {keep.size} threshold rows.\n"
        f"Radial: LM3's selection, thresholds 0..1\n"
        f"({keep.sum()}/{keep.size} rows). Both tip-up."
    )

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(4, 4, hspace=0.18, wspace=0.15)
    fig.suptitle(f"{name} - Geomorphon Classification Stages", fontsize=20, fontweight="bold",
                 y=0.94)

    def legend_cell(ax, with_text: bool):
        """Stats at the top, class key in the middle, selection footnote at the bottom -- the three
        are pinned to separate bands of the cell so the taller stats block cannot land on the key."""
        ax.axis("off")
        handles = [plt.Line2D([0], [0], color=CLASS_COLORS[k], lw=4, label=CLASS_LABELS[k])
                   for k in CLASS_COLORS]
        legend = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.62),
                           title="Classes")
        if with_text:
            ax.text(0.0, 1.0, geomorphon_text, transform=ax.transAxes, fontsize=14,
                    va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
            ax.text(0.0, 0.22, selection_note, transform=ax.transAxes, fontsize=11,
                    va="top", ha="left", fontstyle="italic", color="#333333")
        ax.add_artist(legend)

    # -- rows 1-2: Cartesian, on LM3's axes ------------------------------------------------------ #
    # LM3's render_cartesian_ect is `_colorize(np.flipud(tip_up_directions(m, thetas)[0]))`, i.e. the
    # direction axis re-based on the tip and the threshold axis running BOTTOM-UP. Both are applied
    # here (origin="lower" is the flipud), which is also what makes rows 3-4 a true polar restatement
    # of rows 1-2 -- the polar panels already re-base, so leaving these on raw column order would
    # have shown two different orderings of the same matrix side by side.
    def roll(m):
        """LM3's tip-up column roll, plus the optional half turn (CARTESIAN_ROTATE_180)."""
        out = tip_up_directions(m, thetas)[0]
        return np.rot90(out, 2) if CARTESIAN_ROTATE_180 else out

    leaf_bounds_c = roll(leaf_bounds)

    contour_xy = np.asarray(leaf_data, dtype=float)
    if CARTESIAN_ROTATE_180:
        # Point reflection through the outline's own bbox center: p -> 2c - p. A true 180 deg turn
        # that lands the shape back in the SAME bounding box, so the tick labels keep reading the
        # original pixel range instead of running backwards as inverted axes would.
        center = (contour_xy.min(axis=0) + contour_xy.max(axis=0)) / 2.0
        contour_xy = 2.0 * center - contour_xy

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(contour_xy[:, 0], contour_xy[:, 1], lw=3, color="black", alpha=0.5)
    ax.fill(contour_xy[:, 0], contour_xy[:, 1], lw=0, color="black", alpha=0.2)
    ax.set_title("Contour", fontsize=14, fontstyle="italic")
    ax.set_aspect("equal", adjustable="datalim")

    for col, (title, data) in enumerate((("Original ECT Matrix", matrix_og),
                                         ("Ignore Leaf Bounds ECT Matrix", thresholded_matrix)), 1):
        data = roll(data)
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data, cmap=colormap, norm=normalize, origin="lower")
        ax.set_title(title)
        ax.axis("off")
        overlay = np.zeros((*data.shape, 4))
        overlay[..., 3] = (data == 0) & (leaf_bounds_c == 0)
        ax.imshow(overlay, origin="lower")

    cbar_host = fig.add_subplot(gs[0, 3])
    cbar_host.axis("off")
    cax = cbar_host.inset_axes([0.06, 0.05, 0.09, 0.9])
    cbar = fig.colorbar(im, cax=cax, orientation="vertical")
    cax.set_title("ECT")
    ticks = np.arange(np.floor(thresholded_matrix.min()), np.ceil(thresholded_matrix.max()) + 1, 1)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks.astype(int))

    for col, (title, data) in enumerate(stages):
        ax = fig.add_subplot(gs[1, col])
        ax.imshow(roll(data), cmap=cmap, norm=norm, origin="lower")
        ax.set_title(title)
        ax.axis("off")
    legend_cell(fig.add_subplot(gs[1, 3]), with_text=True)

    # -- rows 3-4: the same six panels, radial (LM3's construction) ----------------------------- #
    phi_ref = float(thetas[tip_up_index(thetas)])

    ax = fig.add_subplot(gs[2, 0], projection="polar")
    if coords_norm is not None and len(coords_norm):
        # LM3's own outline_polar: closed=True unwraps phi, because polar line segments interpolate
        # IN THETA and a step across the 2*pi seam would otherwise sweep the long way round the disc.
        phi_o, r_o = outline_polar(coords_norm, phi_ref, closed=True)
        ax.plot(phi_o, r_o, lw=3, color="black", alpha=0.6)
        ax.fill(phi_o, r_o, lw=0, color="black", alpha=0.2)
        # The contour's radius is |p|, which lives on 0..1 -- NOT the -1..1 threshold axis the ECT
        # panels use. It gets its own rlim so the shape is not squashed into the outer half.
        ax.set_rlim(0.0, float(np.max(r_o)) or 1.0)
    ax.set_theta_direction(-1)
    ax.set_theta_offset(-phi_ref + np.radians(EXTRA_ROTATION_DEG))
    _label_polar(ax, "Contour (radial)")

    for col, (title, data) in enumerate((("Original ECT Matrix (radial)", matrix_og),
                                         ("Ignore Leaf Bounds ECT Matrix (radial)",
                                          thresholded_matrix)), 1):
        ax, _ = _polar_ect_axes(fig, gs[2, col], sel(data), thetas, thresholds_radial, title,
                                cmap=colormap, norm=normalize)

    cbar_host = fig.add_subplot(gs[2, 3])
    cbar_host.axis("off")
    cax = cbar_host.inset_axes([0.06, 0.05, 0.09, 0.9])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=normalize, cmap=colormap), cax=cax,
                        orientation="vertical")
    cax.set_title("ECT")
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ticks.astype(int))

    for col, (title, data) in enumerate(stages):
        _polar_ect_axes(fig, gs[3, col], sel(data), thetas, thresholds_radial,
                        f"{title} (radial)", cmap=cmap, norm=norm)
    legend_cell(fig.add_subplot(gs[3, 3]), with_text=True)

    plt.savefig(path_geomorphon, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- #
def run_one(mask_path: Path, out_dir: Path, num_dirs: int) -> dict | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"cannot read mask: {mask_path}")
    stem = mask_path.stem
    matrix, points, coords_norm, (thetas, thresholds) = lm2_ect_matrix(mask > 127, num_dirs)
    if matrix is None:
        print(f"  {stem} d{num_dirs}: empty or degenerate mask, skipped")
        return None

    _classes, summary = classify_geomorphon(matrix)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}__geomorphon__d{num_dirs}.png"
    visualize_geomorphon_stages(matrix, summary, out_path, f"{stem}  (num_dirs={num_dirs})",
                                points, coords_norm, thetas, thresholds)
    print(f"  {stem} d{num_dirs}: matrix {matrix.shape}, {len(points)} contour pts | "
          f"teeth {summary['fraction_teeth']:.4f}  lobe {summary['fraction_lobe']:.4f}  "
          f"flat {summary['fraction_flat']:.4f}  -> {out_path.name}", flush=True)
    return {"stem": stem, "num_dirs": num_dirs,
            **{k: summary[k] for k in ("fraction_teeth", "fraction_lobe", "fraction_flat",
                                       "n_cells", "n_cells_leaf", "n_cells_bg")}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks", nargs="+", type=Path, help="full path(s) to binary mask PNG(s)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output root (default: beside this script); one subfolder per mask stem")
    ap.add_argument("--dirs", type=int, nargs="+", default=list(DIRS),
                    help=f"direction counts, num_thresh tracks each "
                         f"(default: {' '.join(map(str, DIRS))})")
    args = ap.parse_args(argv)

    root = args.out if args.out is not None else Path(__file__).resolve().parent
    rows = []
    for mask_path in args.masks:
        mask_path = mask_path.resolve()
        print(f"{mask_path.name} -> {root / mask_path.stem}", flush=True)
        for num_dirs in args.dirs:
            row = run_one(mask_path, root / mask_path.stem, num_dirs)
            if row:
                rows.append(row)

    if rows:
        print(f"\n{'leaf':<24} {'dirs':>5} {'teeth':>9} {'lobe':>9} {'flat':>9}")
        for r in rows:
            print(f"{r['stem']:<24} {r['num_dirs']:5d} {r['fraction_teeth']:9.4f} "
                  f"{r['fraction_lobe']:9.4f} {r['fraction_flat']:9.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
