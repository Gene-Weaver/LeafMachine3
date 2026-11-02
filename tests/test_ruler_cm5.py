"""METRIC_CM5: a ruler with a tick every 2 mm (5 per cm), numbered every cm.

The 2 mm unit exists only for the class that prints it. Everything here guards that it cannot reach
another class's ladder, and that the lattice reads it, on a synthetic ruler, at several resolutions.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from leafmachine3.inference.ruler_lattice import units as u
from leafmachine3.inference.ruler_lattice.analysis import _divisions_of_lags, analyse, summarise
from leafmachine3.inference.ruler_lattice.lattice import MIN_PERIOD
from leafmachine3.inference.ruler_lattice.qc import PRETTY

#: The classifier's 19 classes (label_map.json of the pinned Hub model).
MODEL_CLASSES = ("FP", "GRID_CM", "METRIC_CM", "METRIC_CM2", "METRIC_CM4", "METRIC_CM5", "METRIC_CM_BLOCK",
                 "METRIC_CM_STAGGER", "METRIC_MM", "METRIC_MM2", "METRIC_MM_CM", "METRIC_MM_CM_BLOCK",
                 "METRIC_MM_IN16", "STD_IN16", "STD_IN16_IN8_IN2", "STD_IN32_IN16", "STD_IN8", "STD_IN8_CM", "messy")


def ruler(px_per_cm: float, length_cm: float = 42.0, height: int = 140) -> np.ndarray:
    """A light strip with dark ticks every 2 mm, taller every cm, tallest every 10 cm."""
    width = int(round(length_cm * px_per_cm))
    img = np.full((height, width), 205, np.float32)
    tick = max(1, int(round(px_per_cm / 40)))
    for k in range(int(length_cm * 5)):
        x = int(round((k + 0.5) * px_per_cm / 5))
        top = 30 if k % 50 == 0 else 55 if k % 5 == 0 else 85
        img[top:height - 12, x:x + tick] = 70
    rng = np.random.default_rng(1402)
    img = cv2.GaussianBlur(img, (0, 0), 0.8) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_every_model_class_is_registered():
    for name in MODEL_CLASSES:
        assert not u.spec_of(name).get("unknown"), name


def test_the_class_reads_only_what_the_ruler_prints():
    assert u.unit_mm("metric__5_CM") == 2.0
    assert u.class_units("METRIC_CM5") == ["metric__5_CM"]
    assert u.admissible_units("METRIC_CM5") == ["metric__5_CM", "metric__CM"]    # no 2.5 mm or 5 mm tick exists
    assert PRETTY["metric__5_CM"] == "2 mm"


@pytest.mark.parametrize("name", [c for c in MODEL_CLASSES if c != "METRIC_CM5"])
def test_no_other_class_can_read_2_mm(name):
    assert "metric__5_CM" not in u.admissible_units(name)


def test_the_2_mm_rung_is_not_on_the_shared_ladder():
    assert "metric__5_CM" not in u.real_units() and "metric__5_CM" not in u.real_units("metric")
    assert u.match_named(2.0, "metric") is None
    assert u.match_named(2.0, "metric", declared=["metric__5_CM"]) == "metric__5_CM"
    # the anchor gate of every class hangs off this ratio; 2.5 mm / 2 mm would make it 1.25
    assert u.min_rung_ratio() == pytest.approx(1.2598, abs=1e-4)


@pytest.mark.parametrize("px_per_cm", [70.0, 52.5, 105.0, 35.0])
def test_the_lattice_recovers_the_scale(px_per_cm):
    res = analyse(ruler(px_per_cm), "METRIC_CM5", cf_anchor=px_per_cm * 1.03)
    assert abs(summarise(res)["pxcm"] / px_per_cm - 1.0) < 0.02


def test_a_lag_no_printed_period_explains_is_divided():
    px_per_cm = 52.0
    x = np.arange(2000, dtype=np.float32)
    comb = np.cos(2 * math.pi * x / (px_per_cm / 5))                     # ticks every 10.4 px
    four_mm = 2 * px_per_cm / 5
    got = _divisions_of_lags(comb, [four_mm], "METRIC_CM5", px_per_cm)
    assert len(got) == 1 and got[0] == pytest.approx(px_per_cm / 5, rel=0.02)


def test_a_lag_a_printed_period_explains_is_left_alone():
    # a true 1 cm lag / 4 is a phantom 2.5 mm comb that names as 2 mm and still passes ANCHOR_TOL
    x = np.arange(2000, dtype=np.float32)
    comb = np.cos(2 * math.pi * x / 14.0)
    assert _divisions_of_lags(comb, [70.0], "METRIC_CM5", 70.0) == []


def test_nothing_is_divided_without_an_anchor():
    x = np.arange(2000, dtype=np.float32)
    assert _divisions_of_lags(np.cos(2 * math.pi * x / 10.4), [20.8], "METRIC_CM5", None) == []


def test_a_period_under_the_search_floor_is_not_offered():
    anchor = 5 * (MIN_PERIOD - 1)                                       # 2 mm would be 6 px
    x = np.arange(2000, dtype=np.float32)
    assert _divisions_of_lags(np.cos(2 * math.pi * x / 6.0), [12.0], "METRIC_CM5", anchor) == []
