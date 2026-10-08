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


# ---------------------------------------------------------------- label map / model agreement
def _write_member(model_dir, n_out: int) -> None:
    """A tiny real ONNX classifier [1,3,224,224] -> [batch, n_out] plus its metadata.json."""
    import json
    import os

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    os.makedirs(os.path.join(model_dir, "exported"), exist_ok=True)
    w = numpy_helper.from_array(np.ones((3, n_out), np.float32), "w")
    axes = numpy_helper.from_array(np.array([2, 3], np.int64), "axes")
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["x", "axes"], ["m"], keepdims=0),
         helper.make_node("MatMul", ["m", "w"], ["y"])],
        "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3, 224, 224])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", n_out])],
        [w, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    onnx.save(model, os.path.join(model_dir, "exported", "model.onnx"))
    with open(os.path.join(model_dir, "metadata.json"), "w") as fh:
        json.dump({"family": "yolo", "imgsz": 224}, fh)


def _write_label_map(path, classes) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"classes": list(classes)}, fh)


def test_label_map_prefers_installed_copy(tmp_path) -> None:
    """The installer's label_map.json (same Hub revisions as the members) wins over a training-tree splits/ copy."""
    from leafmachine3.inference.ruler_ensemble import _load_label_map, ensemble_files

    _write_label_map(str(tmp_path / "label_map.json"), ["A", "B"])
    _write_label_map(str(tmp_path / "splits" / "label_map.json"), ["A", "B", "C"])
    assert _load_label_map(str(tmp_path)) == ["A", "B"]
    assert ensemble_files(str(tmp_path))[-1] == str(tmp_path / "label_map.json")
    (tmp_path / "label_map.json").unlink()
    assert _load_label_map(str(tmp_path)) == ["A", "B", "C"]    # training tree: splits/ only


def test_member_refuses_class_count_mismatch(tmp_path) -> None:
    """A model whose output size differs from the label map must not load (it would misname classes)."""
    import pytest

    from leafmachine3.inference.ruler_ensemble import RulerEnsemble, _Member

    _write_member(str(tmp_path / "m"), n_out=3)
    with pytest.raises(ValueError, match="outputs 3 classes but the label map has 2"):
        _Member(str(tmp_path / "m"), ["CPUExecutionProvider"], ["A", "B"])
    assert _Member(str(tmp_path / "m"), ["CPUExecutionProvider"], ["A", "B", "C"]).classes == ["A", "B", "C"]

    for name in ("yolo26x_cls_224", "yolo26n_cls_224", "dinov2_frozen_mlp"):
        _write_member(str(tmp_path / "ens" / name), n_out=3)
    _write_label_map(str(tmp_path / "ens" / "label_map.json"), ["A", "B"])
    bad = RulerEnsemble(str(tmp_path / "ens"))
    assert not bad.ok and bad.predict(np.zeros((40, 200, 3), np.uint8)) == {"ensemble": "UNKNOWN"}
    _write_label_map(str(tmp_path / "ens" / "label_map.json"), ["A", "B", "C"])
    assert RulerEnsemble(str(tmp_path / "ens")).ok
