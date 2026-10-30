"""The macOS CoreML path (leafmachine3/inference/coreml.py), tested where CoreML does not exist.

What CoreML itself computes can only be checked on a Mac (tools/macos/README.md). What is pinned
here is everything around it: the options onnxruntime 1.20.1 actually reads, which models go to
CoreML, the executor's one-worker plan, and -- the part the box-bug fix rests on -- that cutting an
end2end graph before its top-k tail and redoing the tail in numpy gives EXACTLY the full graph's
output.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

import numpy as np
import pytest

from leafmachine3.core.device import Device
from leafmachine3.inference import coreml

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------------------------------
# providers / options
# --------------------------------------------------------------------------------------------------
def test_options_use_the_flags_key_onnxruntime_1_20_reads(monkeypatch):
    """1.20.1 ignores named keys ({"ModelFormat": ...}); only ``flags`` reaches CoreML."""
    for var in ("LM3_COREML_UNITS", "LM3_COREML_FORMAT"):
        monkeypatch.delenv(var, raising=False)
    opts = coreml.provider_options()
    assert set(opts) == {"flags"}
    flags = opts["flags"].split("|")
    assert set(flags) == {"COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES", "COREML_FLAG_CREATE_MLPROGRAM",
                          "COREML_FLAG_USE_CPU_AND_GPU"}


def test_option_switches(monkeypatch):
    monkeypatch.setenv("LM3_COREML_UNITS", "ALL")
    monkeypatch.setenv("LM3_COREML_FORMAT", "NeuralNetwork")
    assert coreml.provider_options()["flags"] == "COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES"
    monkeypatch.setenv("LM3_COREML_UNITS", "CPUOnly")
    assert "COREML_FLAG_USE_CPU_ONLY" in coreml.provider_options()["flags"]
    monkeypatch.setenv("LM3_COREML_UNITS", "bogus")
    assert "COREML_FLAG_USE_CPU_AND_GPU" in coreml.provider_options()["flags"]


def test_doctor_probe_passes_the_same_coreml_options():
    from leafmachine3 import doctor_probe

    assert doctor_probe._provider_options("CoreMLExecutionProvider") == coreml.provider_options()
    assert doctor_probe._provider_options("CUDAExecutionProvider") == {"device_id": 0}


def test_coreml_device_hands_out_coreml_first():
    provs = Device("coreml").ort_providers()
    assert coreml.is_coreml(provs) and provs[-1] == "CPUExecutionProvider"
    assert not coreml.is_coreml(Device("cpu").ort_providers())
    assert not coreml.is_coreml(Device("cuda", 1).ort_providers())


def test_which_models_go_to_coreml():
    cml = coreml.providers()
    assert coreml.for_model(cml, True) == cml
    assert coreml.for_model(cml, False) == ["CPUExecutionProvider"]
    cuda = Device("cuda").ort_providers()
    assert coreml.for_model(cuda, False) == cuda              # only a CoreML list is ever rewritten
    assert coreml.COREML_RULER_MEMBERS == {"x224"}             # n224 x0.9, dinov2 x0.55 on an M2 Pro
    assert coreml.specimen_on_coreml("letterbox_imagenet", "unetpp_effb7_1024_dinov3_vith16plus_2048_selftrain")
    assert coreml.specimen_on_coreml("ultralytics_seg", "yolo26x_seg_1280")
    assert not coreml.specimen_on_coreml("stretch_imagenet", "birefnet_hr_swinl_1024")


def test_availability_switches(monkeypatch):
    monkeypatch.delenv("LM3_COREML_SIMULATE", raising=False)
    monkeypatch.setenv("LM3_COREML", "0")
    assert not coreml.available()
    monkeypatch.delenv("LM3_COREML")
    monkeypatch.setattr(coreml.sys, "platform", "linux")
    assert not coreml.available()
    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    assert coreml.available()


# --------------------------------------------------------------------------------------------------
# executor plan
# --------------------------------------------------------------------------------------------------
def _dm(monkeypatch, gpus, devices="auto", kind="cuda"):
    from leafmachine3.core.executor import DeviceManager

    cfg = types.SimpleNamespace(compute=types.SimpleNamespace(mock=False, devices=devices),
                                hardware=types.SimpleNamespace(stage=lambda key: {"workers": 3}))
    dm = DeviceManager(cfg, types.SimpleNamespace(key="plant_detector", device_kind=kind))
    monkeypatch.setattr(dm, "_resolve_gpus", lambda: list(gpus))
    return dm


def test_mac_plans_one_coreml_worker(monkeypatch):
    monkeypatch.delenv("LM3_COREML_SIMULATE", raising=False)
    monkeypatch.setattr(coreml, "available", lambda: True)
    assert _dm(monkeypatch, []).plan() == [Device("coreml")]


def test_coreml_never_overrides_cuda_or_an_explicit_cpu_choice(monkeypatch):
    monkeypatch.delenv("LM3_COREML_SIMULATE", raising=False)
    monkeypatch.setattr(coreml, "available", lambda: True)
    monkeypatch.setattr("leafmachine3.core.executor._free_vram_mb", lambda idx: 48000.0)
    assert all(d.kind == "cuda" for d in _dm(monkeypatch, [0, 1]).plan())
    assert _dm(monkeypatch, [], devices="cpu").plan() == [Device("cpu")] * 3
    assert _dm(monkeypatch, [], kind="cpu").plan() == [Device("cpu")] * 3


def test_without_coreml_a_gpu_less_plan_stays_on_the_cpu(monkeypatch):
    monkeypatch.setattr(coreml, "available", lambda: False)
    assert _dm(monkeypatch, []).plan() == [Device("cpu")] * 3


def test_a_coreml_plan_runs_in_process():
    from leafmachine3.core.executor import StageExecutor

    ex = StageExecutor(types.SimpleNamespace(compute=types.SimpleNamespace(mock=False)),
                       types.SimpleNamespace(key="x", device_kind="cuda", cpu_parallel="thread", fanout=False))
    assert ex._use_inprocess([Device("coreml")], 10)
    assert not ex._use_inprocess([Device("cuda", 0)], 10)


# --------------------------------------------------------------------------------------------------
# the end2end tail: a synthetic graph with the YOLO26 export's tail ops
# --------------------------------------------------------------------------------------------------
try:
    import onnx
    import onnxruntime as ort
except ImportError:          # onnx is a Linux test dependency; the Windows/macOS legs skip these
    onnx = ort = None
needs_onnx = pytest.mark.skipif(onnx is None, reason="needs the onnx package")


def _end2end_graph(nc: int, extra: int, k: int) -> bytes:
    """``feat`` (b, anchors, 4+nc+extra) -> Split -> the YOLO26 end2end tail -> ``output0`` (b, k, 6+extra).

    The same op chain the Ultralytics export emits: ReduceMax -> TopK -> GatherElements -> Flatten ->
    TopK -> Div / Mod -> index gathers -> Concat. Boxes and extra come from the second-stage anchor.
    """
    from onnx import TensorProto, helper

    sizes = [4, nc] + ([extra] if extra else [])
    outs = ["boxes", "scores"] + (["extra"] if extra else [])
    inits = [helper.make_tensor("sizes", TensorProto.INT64, [len(sizes)], sizes),
             helper.make_tensor("k", TensorProto.INT64, [1], [k]),
             helper.make_tensor("nc", TensorProto.INT64, [], [nc]),
             helper.make_tensor("axis_last", TensorProto.INT64, [1], [-1]),
             helper.make_tensor("rep_nc", TensorProto.INT64, [3], [1, 1, nc]),
             helper.make_tensor("rep_4", TensorProto.INT64, [3], [1, 1, 4])]
    if extra:
        inits.append(helper.make_tensor("rep_e", TensorProto.INT64, [3], [1, 1, extra]))
    n = helper.make_node
    nodes = [
        n("Split", ["feat", "sizes"], outs, axis=-1),
        n("ReduceMax", ["scores", "axis_last"], ["smax"], keepdims=0),
        n("TopK", ["smax", "k"], ["_v1", "idx"], axis=-1, largest=1, sorted=1),
        n("Unsqueeze", ["idx", "axis_last"], ["idx3"]),
        n("Tile", ["idx3", "rep_nc"], ["idx_nc"]),
        n("GatherElements", ["scores", "idx_nc"], ["cand"], axis=1),
        n("Flatten", ["cand"], ["flat"], axis=1),
        n("TopK", ["flat", "k"], ["vals", "fi"], axis=-1, largest=1, sorted=1),
        n("Div", ["fi", "nc"], ["rank"]),
        n("Mod", ["fi", "nc"], ["cls_i"], fmod=0),
        n("GatherElements", ["idx", "rank"], ["anchor"], axis=1),
        n("Unsqueeze", ["anchor", "axis_last"], ["anchor3"]),
        n("Tile", ["anchor3", "rep_4"], ["a4"]),
        n("GatherElements", ["boxes", "a4"], ["box_k"], axis=1),
        n("Unsqueeze", ["vals", "axis_last"], ["vals3"]),
        n("Unsqueeze", ["cls_i", "axis_last"], ["cls3i"]),
        n("Cast", ["cls3i"], ["cls3"], to=TensorProto.FLOAT),
    ]
    cat = ["box_k", "vals3", "cls3"]
    if extra:
        nodes += [n("Tile", ["anchor3", "rep_e"], ["ae"]),
                  n("GatherElements", ["extra", "ae"], ["extra_k"], axis=1)]
        cat.append("extra_k")
    nodes.append(n("Concat", cat, ["output0"], axis=-1))
    graph = helper.make_graph(
        nodes, "end2end_tail",
        [helper.make_tensor_value_info("feat", TensorProto.FLOAT, ["batch", "anchors", sum(sizes)])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, ["batch", k, 6 + extra])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    model.metadata_props.add(key="task", value="detect")
    onnx.checker.check_model(model)
    return model.SerializeToString()


def _feat(nc: int, extra: int, anchors: int = 900, ties: bool = False, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    feat = rng.random((1, anchors, 4 + nc + extra), dtype=np.float32)
    if ties:      # many equal scores: the order among equals must match onnxruntime's TopK
        feat[..., 4:4 + nc] = np.round(feat[..., 4:4 + nc] * 4) / 4
    return feat


@needs_onnx
@pytest.mark.parametrize("nc,extra", [(9, 0), (3, 32), (1, 93)])     # detect, segment, pose
@pytest.mark.parametrize("ties", [False, True])
def test_cut_plus_numpy_tail_equals_the_full_graph(tmp_path, nc, extra, ties):
    path = tmp_path / "m.onnx"
    path.write_bytes(_end2end_graph(nc, extra, k=300))
    feat = _feat(nc, extra, ties=ties)
    full = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"feat": feat})[0]
    sess = coreml.End2EndSession(str(path), ["CPUExecutionProvider"])
    assert sess.tail.split_sizes == tuple([4, nc] + ([extra] if extra else []))
    assert sess.tail.max_det == 300
    got = sess.run(None, {"feat": feat})[0]
    assert got.shape == full.shape == (1, 300, 6 + extra)
    np.testing.assert_array_equal(got, full)
    assert sess.get_outputs()[0].name == "output0" and sess.get_modelmeta().custom_metadata_map["task"] == "detect"


@needs_onnx
def test_k_is_capped_by_the_anchor_count(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(_end2end_graph(9, 0, k=300))
    feat = _feat(9, 0, anchors=120)
    with pytest.raises(Exception):   # the graph itself refuses k > anchors; the numpy tail caps it
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"feat": feat})
    out = coreml.end2end_tail(feat[..., :4], feat[..., 4:], None, 300)
    assert out.shape == (1, 120, 6)


@needs_onnx
def test_a_graph_without_the_tail_is_refused(tmp_path):
    from onnx import TensorProto, helper

    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "plain",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    with pytest.raises(coreml.CoremlUnsupported):
        coreml.find_end2end_tail(model)


@needs_onnx
def test_one_session_per_input_shape_with_dims_pinned(tmp_path, monkeypatch):
    path = tmp_path / "m.onnx"
    path.write_bytes(_end2end_graph(9, 0, k=300))
    seen = []
    real = coreml.make_session
    monkeypatch.setattr(coreml, "make_session", lambda m, p, d=None: seen.append(d) or real(m, p, d))
    sess = coreml.End2EndSession(str(path), ["CPUExecutionProvider"])
    for anchors in (900, 900, 1200):
        sess.run(None, {"feat": _feat(9, 0, anchors=anchors)})
    assert seen == [{"batch": 1, "anchors": 900}, {"batch": 1, "anchors": 1200}]


@needs_onnx
def test_simulation_swaps_coreml_for_the_cpu_provider(tmp_path, monkeypatch):
    path = tmp_path / "m.onnx"
    path.write_bytes(_end2end_graph(9, 0, k=300))
    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    sess = coreml.make_session(str(path), coreml.providers(), {"batch": 1, "anchors": 900})
    assert sess.get_providers() == ["CPUExecutionProvider"]


# --------------------------------------------------------------------------------------------------
# the real exports, when installed (LM3_MODELS_DIR or <repo>/models)
# --------------------------------------------------------------------------------------------------
def _models_root() -> Path:
    return Path(os.environ.get("LM3_MODELS_DIR") or REPO / "models")


@needs_onnx
@pytest.mark.parametrize("name,task", [("plant_detector", "detect"), ("archival_detector", "detect"),
                                       ("leaf_segmenter", "segment"), ("landmark_detector", "pose")])
def test_real_exports_cut_path_matches_the_full_graph(name, task, monkeypatch):
    path = _models_root() / name / "model.onnx"
    if not path.exists():
        pytest.skip(f"{path} not installed")
    from leafmachine3.inference import coreml_check
    from leafmachine3.inference import ultra_replacements as ultra_rep

    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    monkeypatch.setenv("LM3_COREML_ROUTE", "ort")
    img = ultra_rep.imread(coreml_check.calibration_images()[0])
    cut = ultra_rep.YOLO(path, task=task, providers=coreml.providers())
    ref = ultra_rep.YOLO(path, task=task, providers=["CPUExecutionProvider"])
    assert isinstance(cut.session, coreml.End2EndSession)
    imgsz = ref.metadata["imgsz"]
    a = ref.predict(img, conf=0.1, imgsz=imgsz, retina_masks=True)[0]
    b = cut.predict(img, conf=0.1, imgsz=imgsz, retina_masks=True)[0]
    # CoreML is fed the CPU's input tensor: the minimum rectangle, not a square
    cpu_shape = ultra_rep.preprocess(img, ultra_rep.check_imgsz(imgsz, ref.stride), ref.stride, ref.auto_letterbox).shape
    assert list(cut.session._sessions) == [tuple(cpu_shape)] and cpu_shape[2] != cpu_shape[3]
    np.testing.assert_array_equal(a.boxes.xyxy, b.boxes.xyxy)
    np.testing.assert_array_equal(a.boxes.conf, b.boxes.conf)
    if task == "segment" and a.masks is not None:
        np.testing.assert_array_equal(a.masks.data, b.masks.data)
    if task == "pose":
        np.testing.assert_array_equal(a.keypoints.data, b.keypoints.data)


def test_dynamic_mode_drops_the_static_shapes_flag(monkeypatch):
    monkeypatch.setenv("LM3_COREML_DYNAMIC", "1")
    assert "STATIC" not in coreml.provider_options()["flags"]


def test_session_cache_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "m.onnx"
    if onnx is None:
        pytest.skip("needs the onnx package")
    path.write_bytes(_end2end_graph(9, 0, k=300))
    monkeypatch.setenv("LM3_COREML_MAX_SESSIONS", "2")
    sess = coreml.End2EndSession(str(path), ["CPUExecutionProvider"])
    for anchors in (900, 1000, 900, 1100, 1000):
        sess.run(None, {"feat": _feat(9, 0, anchors=anchors)})
    assert sess.n_built == 4 and [k[1] for k in sess._sessions] == [1100, 1000]   # 900 was least recent
