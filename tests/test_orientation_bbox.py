"""Orientation-aware split of the rotated bbox into tip->base length vs perpendicular width.

``rotated_bbox_dim_max``/``dim_min`` are the GEOMETRIC long/short sides, which is the wrong answer
for a leaf that is wider than it is long -- on real data that is over half of them. These tests pin
that ``length`` follows the tip->base axis rather than whichever side happens to be longer.
"""
from __future__ import annotations

import math

from leafmachine3.core.orientation import length_width_from_box

_TALL = [[0, 0], [10, 0], [10, 40], [0, 40]]     # 10 wide x 40 tall
_WIDE = [[0, 0], [40, 0], [40, 10], [0, 10]]     # 40 wide x 10 tall


def test_upright_leaf_takes_the_vertical_side_as_length():
    length, width = length_width_from_box(_TALL, 0.0)      # already tip-up
    assert (length, width) == (40.0, 10.0)


def test_sideways_leaf_takes_the_horizontal_side_as_length():
    """angle_cw=90 means the leaf must turn 90 deg to stand up, so its LONG axis is horizontal."""
    length, width = length_width_from_box(_TALL, 90.0)
    assert (length, width) == (10.0, 40.0)


def test_wider_than_long_leaf_reports_length_below_width():
    """The case dim_max/dim_min cannot express: an upright leaf broader than it is tall."""
    length, width = length_width_from_box(_WIDE, 0.0)
    assert length == 10.0 and width == 40.0
    assert length < width


def test_result_is_independent_of_180_degree_flips():
    """Only |vertical component| is compared, so tip-up vs tip-down cannot swap the assignment."""
    for angle in (0.0, 180.0, 360.0):
        assert length_width_from_box(_TALL, angle) == (40.0, 10.0)


def test_sides_are_the_boxes_own_lengths_for_an_oblique_box():
    """A rotated (non-axis-aligned) box still returns its true side lengths, not a re-measurement."""
    t = math.radians(30.0)
    c, s = math.cos(t), math.sin(t)
    corners = [[0, 0], [10 * c, 10 * s], [10 * c - 40 * s, 10 * s + 40 * c], [-40 * s, 40 * c]]
    length, width = length_width_from_box(corners, 30.0)
    assert math.isclose(max(length, width), 40.0, abs_tol=1e-6)
    assert math.isclose(min(length, width), 10.0, abs_tol=1e-6)


def test_unusable_input_returns_none():
    assert length_width_from_box(_TALL, None) == (None, None)      # unoriented leaf
    assert length_width_from_box(None, 0.0) == (None, None)
    assert length_width_from_box([[0, 0], [1, 1]], 0.0) == (None, None)          # not 4 corners
    assert length_width_from_box([[0, 0], [0, 0], [0, 0], [0, 0]], 0.0) == (None, None)  # degenerate
