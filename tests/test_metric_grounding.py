"""MetricGrounding: px -> cm for leaf metrics, petiole width/length AND the landmark lengths.

The CF is published only for high-confidence sheets, so the critical property is that a specimen
WITHOUT a CF is left entirely NULL rather than grounded against a guess.
"""
from __future__ import annotations

import types

from leafmachine3.modules.metric_grounding import MetricGrounding
from leafmachine3.core.stage import WorkItem


def _stage():
    return MetricGrounding(types.SimpleNamespace(modules={}, stage=lambda k: {}))


def _leaf(leaf_id=1, area=400.0, perim=80.0):
    return {"leaf_id": leaf_id, "area_px": area, "perimeter_px": perim,
            "bbox_x1": 0.0, "bbox_y1": 0.0, "bbox_x2": 20.0, "bbox_y2": 40.0}


def _petiole(leaf_id=1, width=20.0, length=100.0):
    return {"leaf_id": leaf_id, "width_px": width, "length_px": length}


def _measure(measure_id=1, trace=300.0, extent=280.0, tip_base=290.0, width=150.0, petiole=50.0):
    return {"measure_id": measure_id, "lamina_trace_length": trace, "lamina_extent": extent,
            "lamina_tip_base_length": tip_base, "leaf_width": width,
            "petiole_trace_length": petiole}


def _payload(cf=10.0, leaves=None, petioles=None, measures=None):
    return (cf,
            [_leaf()] if leaves is None else leaves,
            [_petiole()] if petioles is None else petioles,
            [_measure()] if measures is None else measures)


def test_grounds_leaf_and_petiole_metrics_with_a_cf():
    grounded, pet, _lm = _stage().infer(WorkItem(1, _payload()), None)

    g = grounded[0]
    assert g.area_cm2 == 4.0            # 400 px / 10^2   (area divides by cf SQUARED)
    assert g.perimeter_cm == 8.0        # 80 px / 10
    assert g.bbox_w_cm == 2.0 and g.bbox_h_cm == 4.0

    leaf_id, width_cm, length_cm = pet[0]
    assert leaf_id == 1
    assert width_cm == 2.0              # 20 px / 10   (a length, so cf to the FIRST power)
    assert length_cm == 10.0            # 100 px / 10


def test_grounds_landmark_lengths_with_a_cf():
    """All five landmark LENGTHS divide by the CF once; the row is keyed by its own measure_id."""
    _g, _p, lm = _stage().infer(WorkItem(1, _payload()), None)

    measure_id, trace_cm, extent_cm, tip_base_cm, width_cm, petiole_cm = lm[0]
    assert measure_id == 1
    assert trace_cm == 30.0             # 300 px / 10
    assert extent_cm == 28.0
    assert tip_base_cm == 29.0
    assert width_cm == 15.0
    assert petiole_cm == 5.0


def test_no_cf_grounds_nothing():
    """A withheld/failed ruler CF must leave every *_cm column NULL, never a guessed value."""
    grounded, pet, lm = _stage().infer(WorkItem(1, _payload(cf=None)), None)
    assert grounded == [] and pet == [] and lm == []


def test_missing_pixel_values_stay_none():
    """An occluded keypoint leaves a NULL px metric -- it must stay NULL, not become 0.0."""
    grounded, pet, lm = _stage().infer(
        WorkItem(1, _payload(
            leaves=[_leaf(area=None, perim=None)],
            petioles=[_petiole(width=None, length=None)],
            measures=[_measure(trace=None, extent=None, tip_base=None, width=None, petiole=None)],
        )), None)
    assert grounded[0].area_cm2 is None and grounded[0].perimeter_cm is None
    assert pet[0][1] is None and pet[0][2] is None
    assert all(v is None for v in lm[0][1:])


def test_partially_occluded_leaf_grounds_only_what_it_measured():
    """A row with SOME metrics present grounds those and leaves the rest NULL."""
    _g, _p, lm = _stage().infer(
        WorkItem(1, _payload(measures=[_measure(width=None, petiole=None)])), None)
    _mid, trace_cm, extent_cm, tip_base_cm, width_cm, petiole_cm = lm[0]
    assert (trace_cm, extent_cm, tip_base_cm) == (30.0, 28.0, 29.0)
    assert width_cm is None and petiole_cm is None


def test_specimen_with_no_petioles_still_grounds_leaves():
    grounded, pet, lm = _stage().infer(WorkItem(1, _payload(petioles=[])), None)
    assert grounded and pet == [] and lm
