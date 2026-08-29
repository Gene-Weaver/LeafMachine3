"""Tests for the ECT visuals' shared TIP-UP direction convention (core.ect_compute + reporting.ect_viz).

The ECT is computed on the Reporter's oriented (tip-up) mask in y-DOWN image coords, so the direction
that points at the tip is theta = 3*pi/2. Every visual re-bases the matrix's direction axis on that
direction, which is what keeps the radial views tip-at-North and the Cartesian view an unrolled copy
of them. These tests pin that convention down end to end.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("ect")
pytest.importorskip("matplotlib")

from leafmachine3.core.ect_compute import (  # noqa: E402
    TIP_DIRECTION_RADIANS,
    compute_ect,
    outline_polar,
    tip_up_directions,
    tip_up_index,
)
from leafmachine3.reporting.ect_viz import (  # noqa: E402
    render_cartesian_ect,
    render_radial_ect,
    render_radial_ect_overlay,
    visual_log,
)

# A convex, tip-up synthetic leaf: apex at the TOP (min y, image coords), base at the bottom, widest
# across the middle. The bbox is symmetric about x=150 so the tip sits exactly on the +/-y axis once
# centered, which makes the expected angles exact.
_LEAF_POLY = np.array([(150, 30), (200, 90), (230, 150), (200, 220),
                       (150, 270), (100, 220), (70, 150), (100, 90)], np.int32)


def _leaf_mask(size: int = 300) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    cv2.fillPoly(m, [_LEAF_POLY], 255)
    return m


def _ect(num_dirs: int = 64):
    return compute_ect(_leaf_mask() > 127, num_dirs=num_dirs, want_simple=False)


def _support_from_matrix(column, thresholds) -> float:
    """Last threshold at which chi is still non-zero == the support h(omega) for that direction."""
    nz = np.nonzero(np.asarray(column))[0]
    return float(thresholds[nz[-1]]) if len(nz) else 0.0


@pytest.mark.parametrize("n", [360, 64, 50])   # 50 is NOT divisible by 4 -> phi_ref != exactly 3*pi/2
def test_tip_up_directions_rebases_columns_on_the_tip(n: int) -> None:
    thetas = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    marked = np.tile(np.arange(n), (n, 1))          # cell value == its ORIGINAL column index
    matrix_phi, phi, phi_ref = tip_up_directions(marked, thetas)

    k = tip_up_index(thetas)
    assert phi_ref == pytest.approx(float(thetas[k]))
    assert abs(phi_ref - TIP_DIRECTION_RADIANS) <= (np.pi / n) + 1e-12   # nearest direction to the tip
    assert np.all(matrix_phi[:, 0] == k)                                 # tip column moved to index 0
    assert phi[0] == pytest.approx(0.0)
    assert np.all(np.diff(phi) > 0) and phi[-1] < 2 * np.pi              # ascending, no wrap
    assert matrix_phi.shape == marked.shape


def test_tip_up_is_exact_for_360_dirs() -> None:
    """At the shipped num_dirs=360 the re-base is a clean np.roll(-270) and phi == the raw thetas."""
    thetas = np.linspace(0.0, 2 * np.pi, 360, endpoint=False)
    _m, phi, phi_ref = tip_up_directions(np.zeros((360, 360)), thetas)
    assert tip_up_index(thetas) == 270
    assert phi_ref == pytest.approx(1.5 * np.pi)
    assert np.allclose(phi, thetas)


def test_rebased_zero_direction_measures_the_tip() -> None:
    """The regression guard: after re-basing, direction phi=0 is the one pointing at the leaf tip.

    Its support (last non-zero threshold) must equal the leaf's extent toward the tip -- and must NOT
    equal it for the RAW column 0, which points along +x instead.
    """
    res = _ect(num_dirs=64)
    step = float(res.thresholds[1] - res.thresholds[0])
    tip_extent = float(-res.outline_norm[:, 1].min())     # y is DOWN, so the tip is at min y
    side_extent = float(res.outline_norm[:, 0].max())
    assert tip_extent > side_extent                        # the synthetic leaf really is taller than wide

    matrix_phi, _phi, _phi_ref = tip_up_directions(res.ect_matrix, res.thetas)
    assert _support_from_matrix(matrix_phi[:, 0], res.thresholds) == pytest.approx(tip_extent, abs=1.5 * step)
    assert _support_from_matrix(res.ect_matrix[:, 0], res.thresholds) == pytest.approx(side_extent, abs=1.5 * step)


def test_ect_matrix_rows_are_thresholds() -> None:
    """``EctResult.ect_matrix`` is [threshold, direction]; the support check only works that way."""
    res = _ect(num_dirs=64)
    x, y = res.outline_norm[:, 0], res.outline_norm[:, 1]
    step = float(res.thresholds[1] - res.thresholds[0])
    support = np.array([np.max(x * np.cos(t) + y * np.sin(t)) for t in res.thetas])
    by_col = np.array([_support_from_matrix(res.ect_matrix[:, j], res.thresholds) for j in range(len(res.thetas))])
    assert np.abs(by_col - support).max() <= 1.5 * step


def test_outline_polar_puts_the_tip_at_north() -> None:
    res = _ect(num_dirs=64)
    _m, _phi, phi_ref = tip_up_directions(res.ect_matrix, res.thetas)
    phi, r = outline_polar(res.outline_norm, phi_ref)
    x, y = res.outline_norm[:, 0], res.outline_norm[:, 1]

    def wrapped(v):                       # distance to 0 across the 2*pi seam
        return abs((v + np.pi) % (2 * np.pi) - np.pi)

    assert wrapped(phi[int(np.argmin(y))]) < np.radians(2)                  # tip   -> phi ~ 0    (N)
    assert wrapped(phi[int(np.argmax(y))] - np.pi) < np.radians(2)          # base  -> phi ~ pi   (S)
    assert wrapped(phi[int(np.argmax(x))] - np.pi / 2) < np.radians(2)      # right -> phi ~ pi/2 (E)
    assert wrapped(phi[int(np.argmin(x))] - 3 * np.pi / 2) < np.radians(2)  # left  -> phi ~ 3pi/2 (W)
    assert r.max() == pytest.approx(1.0, abs=1e-6)                          # fills rlim exactly


def test_outline_polar_closed_is_unwrapped_for_plotting() -> None:
    """Polar line segments interpolate IN THETA, so the plotted angles must never jump by >= pi."""
    res = _ect(num_dirs=64)
    _m, _phi, phi_ref = tip_up_directions(res.ect_matrix, res.thetas)
    phi, r = outline_polar(res.outline_norm, phi_ref, closed=True)

    assert len(phi) == len(res.outline_norm) + 1                # ring closed
    assert phi[-1] == pytest.approx(phi[0] + 2 * np.pi) or phi[-1] == pytest.approx(phi[0] - 2 * np.pi)
    assert r[0] == pytest.approx(r[-1])
    assert np.abs(np.diff(phi)).max() < np.pi                   # no seam sweep across the disc


def test_renderers_return_bgr_images_and_the_overlay_draws() -> None:
    res = _ect(num_dirs=64)
    radial = render_radial_ect(res.ect_matrix, res.thetas, res.thresholds, res.bound_radius)
    cart = render_cartesian_ect(res.ect_matrix, res.thetas)
    overlay = render_radial_ect_overlay(res.ect_matrix, res.thetas, res.thresholds, res.bound_radius,
                                        res.outline_norm)
    for img in (radial, cart, overlay):
        assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
        assert min(img.shape[:2]) > 32
    # the outline actually rendered (and did so without changing the framing of the polar disc)
    assert overlay.shape == radial.shape
    assert not np.array_equal(overlay, radial)
    assert (overlay != radial).any(axis=2).sum() > 50           # a real stroke, not a stray pixel


@pytest.mark.parametrize("num_dirs", [64, 128, 257, 360])
def test_every_render_is_exactly_num_dirs_square(num_dirs: int) -> None:
    """The resolution contract: no render may ever be anything but ``num_dirs x num_dirs`` px.

    The renderers take no size/dpi knob at all, so raising ``modules.ect.num_dirs`` is the only way
    to change the output resolution and a render can never silently down-sample the transform.
    """
    res = _ect(num_dirs=num_dirs)
    assert res.ect_matrix.shape == (num_dirs, num_dirs)         # thresholds are tied to directions
    imgs = (
        render_radial_ect(res.ect_matrix, res.thetas, res.thresholds, res.bound_radius),
        render_cartesian_ect(res.ect_matrix, res.thetas),
        render_radial_ect_overlay(res.ect_matrix, res.thetas, res.thresholds, res.bound_radius,
                                  res.outline_norm),
    )
    for img in imgs:
        assert img.shape == (num_dirs, num_dirs, 3)


def test_cartesian_render_is_one_matrix_cell_per_pixel() -> None:
    """The Cartesian view is a LITERAL colormap of the matrix -- never resampled into a figure.

    Every distinct chi value must map to exactly one color and vice versa, which is what makes the
    ECT recoverable from the PNG.
    """
    res = _ect(num_dirs=128)
    matrix_phi, _phi, _ref = tip_up_directions(res.ect_matrix, res.thetas)
    expected = np.flipud(matrix_phi)                            # renders use the origin="lower" convention
    img = render_cartesian_ect(res.ect_matrix, res.thetas)
    assert img.shape[:2] == expected.shape

    by_value: dict[int, set] = {}
    for value, pixel in zip(expected.ravel(), img.reshape(-1, 3)):
        by_value.setdefault(int(value), set()).add(tuple(int(c) for c in pixel))
    assert all(len(colors) == 1 for colors in by_value.values())    # one chi value -> one color
    assert len({next(iter(c)) for c in by_value.values()}) == len(by_value)   # and no two share one


# --------------------------------------------------------------------------- #
# apply_log_to_visual_for_bold_color -- a DISPLAY transform, never a stored one
# --------------------------------------------------------------------------- #

def test_visual_log_is_signed_monotonic_and_fixes_zero() -> None:
    """Chi crosses zero, so the display log has to be signed -- a plain log would be nan/-inf."""
    values = np.array([[-40.0, -7.0, -1.0, 0.0, 1.0, 7.0, 40.0]])
    out = visual_log(values)

    assert np.isfinite(out).all()
    assert out[0, 3] == 0.0                                     # zero is fixed
    assert np.array_equal(np.sign(out), np.sign(values))        # sign preserved either side of it
    assert np.all(np.diff(out[0]) > 0)                          # monotonic across the whole range
    # it really is a compression: the big-|chi| tail shrinks relative to the mid-range
    assert abs(out[0, -1] / out[0, -2]) < abs(values[0, -1] / values[0, -2])


def test_visual_log_does_not_mutate_the_matrix_it_is_given() -> None:
    """The stage keeps rendering from `res.ect_matrix` after this call -- it must be untouched."""
    res = _ect(num_dirs=64)
    before = res.ect_matrix.copy()
    scaled = visual_log(res.ect_matrix)

    assert np.array_equal(res.ect_matrix, before)               # no in-place scaling
    assert scaled is not res.ect_matrix
    assert res.ect_matrix.dtype.kind in "iu"                    # stored ECT stays integer chi


def test_the_log_toggle_changes_the_picture_and_nothing_else() -> None:
    """Both branches of the toggle, at the resolution and framing the contract mandates."""
    res = _ect(num_dirs=64)
    raw = render_cartesian_ect(res.ect_matrix, res.thetas)
    bold = render_cartesian_ect(visual_log(res.ect_matrix), res.thetas)

    assert bold.shape == raw.shape == (64, 64, 3)               # log never changes the geometry
    assert not np.array_equal(bold, raw)                        # ...but it does change the colors
    # the point of the toggle: the mid-range stops piling into one shade
    assert len(np.unique(bold.reshape(-1, 3), axis=0)) >= len(np.unique(raw.reshape(-1, 3), axis=0))
