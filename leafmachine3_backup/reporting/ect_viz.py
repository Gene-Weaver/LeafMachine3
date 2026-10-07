"""ECT visualizations: a RADIAL (polar) view, a non-radial (Cartesian) view, and a radial view with
the leaf outline overlaid.

All three share ONE direction origin: the matrix's direction axis is re-based on the leaf tip by
:func:`leafmachine3.core.ect_compute.tip_up_directions`, so ``phi = 0`` is the tip for every visual.
The radial views then put the tip at North (matching the oriented tip-up leaf masks) and the
Cartesian view reads left-to-right as the radial view unrolled from the top, clockwise.

============================================================================================
CREDIT: The radial/polar ECT rendering here is adapted from Dan Chitwood's project
``ect_to_shape_CNN`` (https://github.com/DanChitwood/ect_to_shape_CNN), specifically the
``save_grayscale_radial_ect`` polar ``pcolormesh`` construction in ``0_radial_ect_and_masks.py``
(directions around theta, ECT thresholds as the radius, North = 0deg, clockwise). Only the
VISUALIZATION concept is borrowed (the CNN pieces are not used); the colormap is user-configurable
here rather than fixed grayscale. Thank you to Dan Chitwood / the Chitwood Lab.
============================================================================================
"""
from __future__ import annotations

import numpy as np

from leafmachine3.core.ect_compute import outline_polar, tip_up_directions

# ---------------------------------------------------------------------------------------------
# RESOLUTION CONTRACT: every ECT product is exactly ``num_dirs x num_dirs`` pixels -- the size of
# the ECT matrix itself. Nothing here takes a ``size`` or a ``dpi``: raising ``modules.ect.num_dirs``
# is the ONLY way to change the output resolution, so a render can never silently down-sample the
# transform it is showing. The Cartesian view is therefore a literal 1-matrix-cell-to-1-pixel
# colormap of the matrix (no plotting library in the raster path at all), and the polar views are
# drawn onto a canvas sized to the same num_dirs so the direction axis is never resampled down.
# Callers save these as PNG, which is lossless -- no ECT product is ever JPEG-compressed.
# ---------------------------------------------------------------------------------------------


#: White core of the overlay outline, in PIXELS per 256 px of canvas (the historical 1.2 pt @ 110
#: dpi look); the black casing is twice this. See :func:`render_radial_ect_overlay`.
_OUTLINE_PX_PER_256: float = 1.2 * 110.0 / 72.0


def ect_render_size(ect_matrix) -> int:
    """The mandated pixel size of every ECT render: the matrix's direction count (``num_dirs``)."""
    return int(np.asarray(ect_matrix).shape[1])


def visual_log(ect_matrix) -> np.ndarray:
    """DISPLAY-ONLY signed ``log1p`` of an ECT matrix -- ``modules.ect.apply_log_to_visual_for_bold_color``.

    Chi is dominated by a long tail of large |values|, so a linear ``Normalize`` squeezes the dense
    mid-range into one or two shades and the interference structure washes out. Compressing that
    tail spreads the mid-range across the whole colormap and the structure pops.

    SIGNED, because chi runs negative through zero to positive on non-convex shapes (holes, deep
    sinuses) -- a plain ``log`` would produce nan/-inf there. ``sign(x) * log1p(|x|)`` is monotonic
    over the whole range, fixes 0 at 0, and is smooth through it.

    CRITICAL: this is a rendering transform ONLY. The caller passes the RESULT to the renderers and
    keeps the untouched :attr:`~leafmachine3.core.ect_compute.EctResult.ect_matrix` for the .h5 and
    the ``leaf_ect`` row, so no stored ECT value is ever log-scaled. Because the renderers autoscale
    per image, the color ramp is not comparable across leaves either way -- only the .h5 matrix is.
    """
    a = np.asarray(ect_matrix, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def _colorize(values, cmap: str) -> np.ndarray:
    """Colormap a 2-D array to BGR uint8, ONE array cell per pixel -- no interpolation, no resize.

    Autoscaling matches what ``imshow``/``pcolormesh`` do by default (``Normalize`` over the data
    range), so the palette reads identically to the polar views next to it.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    a = np.asarray(values, dtype=float)
    rgb = colormaps[cmap](Normalize()(a))[..., :3]
    return np.ascontiguousarray((rgb[..., ::-1] * 255.0).round().astype(np.uint8))


# The object-oriented Figure API (NOT pyplot) is used for the polar views: pyplot keeps a global
# figure registry that is not thread-safe and leaks figures unless explicitly closed. A bare Figure
# + an Agg canvas has no global state, so these renderers are safe to call from many threads AND
# from process-pool workers, and each figure is freed as soon as it goes out of scope.
def _square_figure(size: int):
    """A ``size x size`` PIXEL figure at Matplotlib's DEFAULT dpi (dpi is never overridden here)."""
    from matplotlib.figure import Figure

    fig = Figure()
    fig.set_size_inches(size / fig.dpi, size / fig.dpi)
    return fig


def _fig_to_bgr(fig, size: int) -> np.ndarray:
    """Rasterize ``fig`` to BGR at its native canvas resolution, forced to exactly ``size``.

    The RGBA buffer is read straight off the Agg canvas -- no PNG round-trip and, critically, no
    ``bbox_inches="tight"``, which re-crops to the drawn artists and would make the output size
    depend on what was plotted. The canvas is already ``size`` px by construction; the pad/crop
    below only ever absorbs a sub-pixel float rounding of ``size / dpi * dpi``.
    """
    import cv2
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)        # attach a renderer without touching pyplot's global state
    canvas.draw()
    bgr = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    h, w = bgr.shape[:2]
    if (h, w) != (size, size):
        out = np.full((size, size, 3), 255, np.uint8)
        out[:min(h, size), :min(w, size)] = bgr[:min(h, size), :min(w, size)]
        bgr = out
    return bgr


def _polar_ect_axes(fig, ect_matrix, thetas, thresholds, bound_radius, cmap: str):
    """Draw the tip-up polar ECT onto a new axes of ``fig``; returns ``(ax, phi_ref)``."""
    matrix_phi, phi, phi_ref = tip_up_directions(ect_matrix, thetas)
    ax = fig.add_subplot(projection="polar")
    PHI, R = np.meshgrid(phi, np.asarray(thresholds))
    ax.pcolormesh(PHI, R, matrix_phi, cmap=cmap)
    # Screen angle = offset + direction*phi = -phi_ref - phi = -theta, which is exactly where a
    # y-down direction vector belongs on a y-up screen. Equivalent to zero_location "N" whenever
    # num_dirs % 4 == 0 (phi_ref == 3*pi/2), but exact for any num_dirs.
    ax.set_theta_direction(-1)          # clockwise
    ax.set_theta_offset(-phi_ref)       # phi = 0 (the tip) at North
    ax.set_rlim([0.0, float(bound_radius)])
    ax.axis("off")
    return ax, phi_ref


def render_radial_ect(ect_matrix, thetas, thresholds, bound_radius, *,
                      cmap: str = "viridis") -> np.ndarray:
    """Polar ECT: directions around, ``thresholds`` as radius (Chitwood's radial ECT), tip at North.

    Returns a BGR image that is exactly ``num_dirs x num_dirs`` px -- see the resolution contract at
    the top of this module. ``cmap`` is any Matplotlib colormap name.
    """
    size = ect_render_size(ect_matrix)
    fig = _square_figure(size)
    _polar_ect_axes(fig, ect_matrix, thetas, thresholds, bound_radius, cmap)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _fig_to_bgr(fig, size)


def render_radial_ect_overlay(ect_matrix, thetas, thresholds, bound_radius, outline_norm, *,
                              cmap: str = "viridis") -> np.ndarray:
    """:func:`render_radial_ect` with the leaf outline drawn on top, in the SAME tip-up frame.

    The ECT's colored extent at any angle is the support ``h(omega) = max_p p.omega``, and
    ``h >= |p|`` always, so the outline nests inside the colored envelope and touches it wherever
    the outline's outward normal matches that direction -- that tangency is the visual confirmation
    the two are aligned. Returns a BGR image, exactly ``num_dirs x num_dirs`` px.
    """
    import matplotlib.patheffects as pe

    size = ect_render_size(ect_matrix)
    fig = _square_figure(size)
    ax, phi_ref = _polar_ect_axes(fig, ect_matrix, thetas, thresholds, bound_radius, cmap)
    # closed=True also UNWRAPS phi -- see outline_polar: polar line segments interpolate in theta,
    # so a contour step across the 2*pi seam would otherwise sweep the long way around the disc.
    phi_o, r_o = outline_polar(outline_norm, phi_ref, closed=True)
    # Matplotlib line widths are in POINTS, which would make the stroke depend on the ambient dpi
    # AND thin to a hair as num_dirs grows. Fix the stroke in PIXELS instead (a ~1.8 px white core
    # in a ~3.7 px black casing per 256 px of canvas) and convert once, so the outline keeps the
    # same weight relative to the disc at every num_dirs and under any default dpi.
    lw = (_OUTLINE_PX_PER_256 * size / 256.0) * 72.0 / fig.dpi
    ax.plot(phi_o, r_o, lw=lw, color="white", solid_joinstyle="round",
            path_effects=[pe.Stroke(linewidth=2.0 * lw, foreground="black"), pe.Normal()])
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return _fig_to_bgr(fig, size)


def render_cartesian_ect(ect_matrix, thetas, *, cmap: str = "viridis") -> np.ndarray:
    """Non-radial ECT: the matrix (thresholds x directions) as an image. Returns a BGR image.

    ONE MATRIX CELL IS ONE PIXEL -- the returned image is the matrix's own shape, colormapped
    directly, so the raw transform is recoverable from the PNG and nothing is ever resampled. (The
    old Matplotlib ``imshow`` path is gone precisely because it rasterized into a fixed-size figure.)

    The direction axis is re-based on the tip, so reading left-to-right (``phi = 0 .. 2*pi``) is the
    radial view unrolled from the top, clockwise: tip, right side, base, left side, tip. Rows run
    bottom-up in threshold (the ``origin="lower"`` convention the radial views share), so the array
    is flipped once on the way out.
    """
    matrix_phi, _phi, _phi_ref = tip_up_directions(ect_matrix, thetas)
    return _colorize(np.flipud(matrix_phi), cmap)
