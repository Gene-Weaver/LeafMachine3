"""Tests for the Ruler Classifier: squarify preprocessing, per-specimen consensus, DB round-trip."""
from __future__ import annotations

import numpy as np

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import RulerClassRow, SpecimenRecord
from leafmachine3.inference.ruler_squarify import RulerSquarifier
from leafmachine3.modules.ruler_classifier import _specimen_ruler_class


def _row(unit_type: str) -> RulerClassRow:
    return RulerClassRow(detection_id=0, unit_type=unit_type, votes={}, conf=None)


def test_specimen_ruler_class_consensus() -> None:
    assert _specimen_ruler_class([_row("METRIC_MM"), _row("METRIC_MM"), _row("STD_IN8")]) == "METRIC_MM"
    assert _specimen_ruler_class([_row("UNKNOWN"), _row("METRIC_CM")]) == "METRIC_CM"   # UNKNOWN ignored
    assert _specimen_ruler_class([_row("UNKNOWN"), _row("UNKNOWN")]) is None
    assert _specimen_ruler_class([]) is None


def test_set_specimen_ruler_class_roundtrip(tmp_path) -> None:
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    sid = db.upsert_specimen(SpecimenRecord(
        image_name="a.jpg", image_stem="a", original_path="/o/a.jpg", working_path="/w/a.jpg"))
    db.set_specimen_ruler_class(sid, "METRIC_MM")
    assert db.get_specimen(sid)["ruler_class_type"] == "METRIC_MM"
    db.set_specimen_ruler_class(sid, None)
    assert db.get_specimen(sid)["ruler_class_type"] is None


def test_squarify_tile_four_shape_determinism_grayscale() -> None:
    """tile_four @ sz=720 -> 1440x1440; deterministic (augment=False); channels grayscale-equal
    (color input is normalized to grayscale, matching the grayscale crops the members trained on)."""
    sq = RulerSquarifier(sz=720, method="tile_four", augment=False)
    strip = np.zeros((80, 800, 3), np.uint8)         # a wide COLOR ruler strip (w/h = 10 > 4 -> stacks)
    strip[:, ::20] = (0, 0, 255)                     # red "ticks"
    out = sq.transform(strip)
    assert out.shape == (1440, 1440, 3)              # 2*sz square
    assert np.array_equal(out, sq.transform(strip))  # deterministic
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_squarify_makes_portrait_horizontal() -> None:
    """A PORTRAIT strip is rotated to landscape internally, so the collage is square regardless."""
    sq = RulerSquarifier(sz=256, method="tile_four", augment=False)
    out = sq.transform(np.full((800, 80, 3), 127, np.uint8))
    assert out.shape == (512, 512, 3)                # 2*sz, orientation-agnostic


# ---------------------------------------------------------------- the model
def _write_model(path, n_out: int, classes=None) -> str:
    """A tiny real ONNX classifier [N,3,896,896] -> [N, n_out], plus label_map.json beside it."""
    import json
    import os

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    os.makedirs(path, exist_ok=True)
    w = numpy_helper.from_array(np.arange(3 * n_out, dtype=np.float32).reshape(3, n_out), "w")
    axes = numpy_helper.from_array(np.array([2, 3], np.int64), "axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["m"], keepdims=0),
         helper.make_node("MatMul", ["m", "w"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 896, 896])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", n_out])],
        [w, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    target = os.path.join(path, "model.onnx")
    onnx.save(model, target)
    with open(os.path.join(path, "label_map.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": classes or [f"C{i}" for i in range(n_out)]}, fh)
    return target


def test_the_classifier_sees_the_training_input_and_names_its_top_classes(tmp_path) -> None:
    from leafmachine3.inference.ruler_classifier import RulerClassifierModel

    model = RulerClassifierModel(_write_model(str(tmp_path), 4))
    assert model.ok and model.classes == ["C0", "C1", "C2", "C3"]
    strip = np.zeros((80, 800, 3), np.uint8)
    strip[:, ::20] = 255
    assert model.squarify_tile(strip).shape == (896, 896, 3)      # tile_four at sz=448
    out = model.predict(strip)
    # a mostly black strip normalizes to a negative mean, so the fixture's smallest column wins
    assert out["label"] == "C0" and [c for c, _ in out["top3"]] == ["C0", "C1", "C2"]
    assert 0.25 < out["conf"] <= 1.0 and out["conf"] >= out["top3"][1][1]


def test_the_classifier_refuses_a_class_count_mismatch(tmp_path) -> None:
    """A model whose output size differs from the label map must not load (it would misname classes)."""
    from leafmachine3.inference.ruler_classifier import RulerClassifierModel

    bad = RulerClassifierModel(_write_model(str(tmp_path), 3, classes=["A", "B"]))
    assert not bad.ok and bad.predict(np.zeros((40, 200, 3), np.uint8)) == {"label": "UNKNOWN"}


def test_a_missing_model_degrades_to_unknown(tmp_path) -> None:
    from leafmachine3.inference.ruler_classifier import RulerClassifierModel

    model = RulerClassifierModel(str(tmp_path / "absent" / "model.onnx"))
    assert not model.ok and model.predict(np.zeros((40, 200, 3), np.uint8)) == {"label": "UNKNOWN"}


def test_old_ensemble_settings_warn_and_still_get_the_model(tmp_path) -> None:
    """A settings file from before 3.0.1056 names the ensemble; it must not change what runs."""
    import yaml

    from leafmachine3.core.config import Config

    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({"project": {"input": {"dirs": ["x"]}}, "modules": {"ruler_classifier": {
        "enabled": True, "models_dir": "models/ruler_classifier", "ensemble_members": ["a"],
        "min_conf": 0.35, "squarify": {"sz": 720, "method": "tile_four"}}}}))
    cfg = Config.load(path)
    retired = dict(cfg.retired_settings())
    assert {f"modules.ruler_classifier.{k}" for k in ("models_dir", "ensemble_members", "min_conf", "squarify")} <= set(retired)
    assert cfg.stage("ruler_classifier")["model"]["path"] == "models/ruler_classifier/model.onnx"
