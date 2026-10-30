"""The native CoreML route (``coreml_fp32`` packages): everything except Core ML itself, on any OS.

Core ML runs only on a Mac, so the package's network is stood in for by the ONNX twin cut at the same
split (``LM3_COREML_SIMULATE``); the contract reading, shape coverage, CPU fallback, numpy tail and
routing are the real code.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from leafmachine3.inference import coreml
from leafmachine3.inference import coreml_native as cn
from tests.test_coreml import _end2end_graph, _feat, needs_onnx

REPO = Path(__file__).resolve().parents[1]


def _exporter():
    spec = importlib.util.spec_from_file_location("export_coreml_fp32", REPO / "tools/modelhub/export_coreml_fp32.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _package(root: Path, *, onnx_bytes: bytes | None = None, shapes=((1, 900, 13),), tail="end2end",
             outputs=("boxes", "scores"), split=(4, 9), max_det=300, onnx_out=("output0",), **override):
    """A model dir with ``model.onnx`` and a ``coreml_fp32`` package whose program is a stub."""
    root.mkdir(parents=True, exist_ok=True)
    pkg = root / "coreml_fp32" / "model.mlpackage"
    program = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
    program.parent.mkdir(parents=True)
    program.write_bytes(b"ml program")
    (pkg / "Manifest.json").write_text("{}")
    if onnx_bytes is not None:
        (root / "model.onnx").write_bytes(onnx_bytes)
    contract = {
        "format": cn.FORMAT, "schema": "1", "mlmodel_sha256": hashlib.sha256(b"ml program").hexdigest(),
        "input": {"name": "feat", "shapes": [list(s) for s in shapes]}, "outputs": list(outputs),
        "tail": tail, "split": list(split), "max_det": max_det,
        "onnx": {"inputs": [["feat", ["batch", "anchors", sum(split)]]],
                 "outputs": [[n, ["batch", max_det, 6]] for n in onnx_out], "metadata": {"task": "detect"}},
    }
    contract.update(override)
    (pkg.parent / "model.json").write_text(json.dumps(contract))
    return root / "model.onnx", pkg


# --------------------------------------------------------------------------------------------------
# Finding and reading packages
# --------------------------------------------------------------------------------------------------
def test_the_package_is_found_beside_its_onnx_twin(tmp_path):
    onnx_path, pkg = _package(tmp_path / "plant_detector")
    assert cn.package_for(onnx_path) == pkg
    # the ruler layout: <member>/exported/model.onnx, package at <member>/coreml_fp32/
    _, pkg2 = _package(tmp_path / "ruler" / "yolo26x_cls_224")
    assert cn.package_for(tmp_path / "ruler" / "yolo26x_cls_224" / "exported" / "model.onnx") == pkg2
    assert cn.package_for(tmp_path / "elsewhere" / "model.onnx") is None


@pytest.mark.parametrize("override,message", [
    ({"format": "coreml"}, "not a coreml_fp32 package"),
    ({"schema": "99"}, "schema"),
    ({"mlmodel_sha256": "0" * 64}, "does not match its model.json"),
])
def test_a_package_that_is_not_this_contract_is_refused(tmp_path, override, message):
    _, pkg = _package(tmp_path / "m", **override)
    with pytest.raises(cn.CoremlPackageError, match=message):
        cn.read_package(pkg)


def test_a_package_without_its_contract_is_refused(tmp_path):
    _, pkg = _package(tmp_path / "m")
    (pkg.parent / "model.json").unlink()
    with pytest.raises(cn.CoremlPackageError, match="cannot read"):
        cn.read_package(pkg)


def test_the_contract_is_read_without_coremltools(tmp_path):
    _, pkg = _package(tmp_path / "m", shapes=((1, 900, 13), (1, 400, 13)))
    info = cn.read_package(pkg)
    assert info.shapes == {(1, 900, 13), (1, 400, 13)} and info.outputs == ("boxes", "scores")
    assert info.tail == "end2end" and info.split == (4, 9) and info.max_det == 300
    assert info.metadata == {"task": "detect"} and info.onnx_outputs[0][0] == "output0"


# --------------------------------------------------------------------------------------------------
# Running (simulated Core ML)
# --------------------------------------------------------------------------------------------------
@needs_onnx
@pytest.mark.parametrize("nc,extra", [(9, 0), (3, 32), (1, 93)])     # detect, segment, pose
@pytest.mark.parametrize("ties", [False, True])
def test_the_native_session_equals_the_full_onnx_graph(tmp_path, monkeypatch, nc, extra, ties):
    import onnxruntime as ort

    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    graph = _end2end_graph(nc, extra, k=300)
    split = (4, nc) + ((extra,) if extra else ())
    onnx_path, pkg = _package(tmp_path / "m", onnx_bytes=graph, shapes=((1, 900, sum(split)),), split=split,
                              outputs=("boxes", "scores") + (("extra",) if extra else ()))
    sess = cn.NativeSession(pkg, onnx_path)
    x = _feat(nc, extra, ties=ties)
    got = sess.run(None, {"feat": x})[0]
    want = ort.InferenceSession(graph, providers=["CPUExecutionProvider"]).run(None, {"feat": x})[0]
    np.testing.assert_array_equal(got, want)
    assert sess.get_providers() == [cn.PROVIDER] and sess.stats["first_ms"] and not sess.stats["cpu_fallback"]


@needs_onnx
def test_a_shape_the_package_lacks_runs_the_onnx_twin_on_the_cpu(tmp_path, monkeypatch):
    import onnxruntime as ort

    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    graph = _end2end_graph(9, 0, k=300)
    onnx_path, pkg = _package(tmp_path / "m", onnx_bytes=graph, shapes=((1, 900, 13),))
    sess = cn.NativeSession(pkg, onnx_path)
    x = _feat(9, 0, anchors=640)
    got = sess.run(None, {"feat": x})[0]
    want = ort.InferenceSession(graph, providers=["CPUExecutionProvider"]).run(None, {"feat": x})[0]
    np.testing.assert_array_equal(got, want)
    assert sess.stats["cpu_fallback"] == [[1, 640, 13]] and not sess.stats["first_ms"]


@needs_onnx
def test_session_for_routes_to_the_package_or_to_the_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    graph = _end2end_graph(9, 0, k=300)
    onnx_path, _ = _package(tmp_path / "with", onnx_bytes=graph)
    assert isinstance(cn.session_for(onnx_path), cn.NativeSession)
    (tmp_path / "without").mkdir()
    (tmp_path / "without" / "model.onnx").write_bytes(graph)
    assert cn.session_for(tmp_path / "without" / "model.onnx") is None       # no package: the CPU
    monkeypatch.setenv("LM3_COREML", "0")
    assert cn.session_for(onnx_path) is None                                  # CoreML switched off


@needs_onnx
def test_a_broken_package_falls_back_to_the_cpu_with_a_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    onnx_path, _ = _package(tmp_path / "m", onnx_bytes=_end2end_graph(9, 0, k=300), mlmodel_sha256="0" * 64)
    assert cn.session_for(onnx_path) is None
    assert "does not match its model.json" in caplog.text


# --------------------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------------------
def test_the_route_switch(monkeypatch):
    monkeypatch.delenv("LM3_COREML_ROUTE", raising=False)
    assert coreml.route() == "native"
    monkeypatch.setenv("LM3_COREML_ROUTE", "ort")
    assert coreml.route() == "ort"
    monkeypatch.setenv("LM3_COREML_ROUTE", "bogus")
    assert coreml.route() == "native"


@needs_onnx
def test_the_native_route_never_falls_back_to_the_onnxruntime_route(tmp_path, monkeypatch):
    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    monkeypatch.delenv("LM3_COREML_ROUTE", raising=False)
    (tmp_path / "model.onnx").write_bytes(_end2end_graph(9, 0, k=300))
    assert coreml.yolo_session(str(tmp_path / "model.onnx"), coreml.providers()) is None   # no package: CPU
    monkeypatch.setenv("LM3_COREML_ROUTE", "ort")
    assert isinstance(coreml.yolo_session(str(tmp_path / "model.onnx"), coreml.providers()), coreml.End2EndSession)


def test_one_tail_implementation():
    assert coreml.end2end_tail is cn.end2end_tail


# --------------------------------------------------------------------------------------------------
# The exporter's contract with the runtime
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("imgsz", [640, 1024, 1280])
def test_the_package_shapes_cover_every_letterbox_shape(imgsz):
    from leafmachine3.inference import ultra_replacements as ultra_rep

    shapes = _exporter().rect_shapes(imgsz)
    assert len(shapes) == 2 * imgsz // 32 - 1 and shapes[0] == (1, 3, imgsz, imgsz) and len(set(shapes)) == len(shapes)
    rng = np.random.default_rng(imgsz)
    for h, w in [(3000, 2000), (2000, 3000), (5000, 37), (37, 5000), (800, 800)] + \
            [tuple(rng.integers(20, 6000, 2)) for _ in range(300)]:
        im = ultra_rep.preprocess(np.zeros((h, w, 3), np.uint8), (imgsz, imgsz), 32, True)
        assert tuple(im.shape) in shapes, (h, w)


def test_the_exporter_covers_the_default_models_that_run_on_coreml(tmp_path):
    jobs = _exporter().lock_jobs(tmp_path / "models", tmp_path / "pt", tmp_path / "out")
    by_stage = {}
    for stage, key, kind, _ckpt, onnx_path, pkg in jobs:
        by_stage.setdefault(stage, []).append(key)
        assert kind == ("unetpp" if key.startswith("unetpp_") else "yolo")
        # the package lands where the runtime looks for it, relative to the ONNX twin
        rel_onnx = onnx_path.relative_to(tmp_path / "models")
        rel_pkg = pkg.relative_to(tmp_path / "out")
        model_dir = rel_onnx.parent.parent if rel_onnx.parent.name == "exported" else rel_onnx.parent
        assert rel_pkg == model_dir / cn.PACKAGE
    assert set(by_stage) == {"archival_detector", "plant_detector", "landmark_detector", "leaf_segmenter",
                             "specimen_segmenter", "ruler_classifier"}
    assert by_stage["ruler_classifier"] == ["yolo26x_cls_224"]          # the n and DINOv2 members stay on the CPU


# --------------------------------------------------------------------------------------------------
# Fewer shape changes, same rows
# --------------------------------------------------------------------------------------------------
def test_crops_are_grouped_by_the_input_shape_they_get():
    from types import SimpleNamespace

    from leafmachine3.inference import ultra_replacements as ultra_rep

    rng = np.random.default_rng(3)
    crops = []
    for _ in range(400):
        w, h = (int(v) for v in rng.integers(30, 2500, 2))
        crops.append(SimpleNamespace(x1=10.0, y1=20.0, x2=10.0 + w, y2=20.0 + h))
    order = cn.crop_shape_order(crops)
    assert sorted(order) == list(range(len(crops)))
    for imgsz in (640, 1024):
        seq = [tuple(ultra_rep.preprocess(np.zeros((int(c.y2 - c.y1), int(c.x2 - c.x1), 3), np.uint8),
                                          (imgsz, imgsz), 32, True).shape) for c in (crops[i] for i in order)]
        runs = [s for k, s in enumerate(seq) if k == 0 or s != seq[k - 1]]
        assert len(runs) == len(set(seq)), "each input shape should come up in one run"


def test_cpu_only_is_refused_for_a_package_with_several_shapes(tmp_path, monkeypatch):
    """Core ML's CPU path crashes the process on an enumerated-shape package (BNNS, macOS 26)."""
    monkeypatch.delenv("LM3_COREML_SIMULATE", raising=False)
    monkeypatch.setenv("LM3_COREML_UNITS", "CPUOnly")
    onnx_path, pkg = _package(tmp_path / "m", onnx_bytes=b"", shapes=((1, 900, 13), (1, 400, 13)))
    with pytest.raises(cn.CoremlPackageError, match="CPU path crashes"):
        cn.NativeSession(pkg, onnx_path)
