"""The macOS Core ML path (leafmachine3/inference/coreml.py), tested where Core ML does not exist.

What Core ML itself computes can only be checked on a Mac (tools/macos/README.md). What is pinned
here is everything around it: which models go to Core ML, the executor's one-worker plan, the
doctor's Core ML probe, and -- what every ``coreml_fp32`` package rests on -- that cutting an end2end
graph before its top-k tail and redoing the tail in numpy gives EXACTLY the full graph's output.
(The packages themselves: tests/test_coreml_native.py.)
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
def test_the_doctor_probes_core_ml_natively(monkeypatch):
    """Core ML is not an onnxruntime provider in LM3: check 6 runs its own Core ML probe."""
    from leafmachine3 import doctor_probe

    monkeypatch.setattr(doctor_probe, "run_coreml_probe", lambda: {"bound": "CoreMLExecutionProvider", "native": True})
    assert doctor_probe.run_probe("CoreMLExecutionProvider")["native"]
    assert doctor_probe._provider_options("CUDAExecutionProvider") == {"device_id": 0}
    assert doctor_probe._provider_options("CPUExecutionProvider") == {}


def test_the_core_ml_probe_reference_is_a_padded_conv_and_relu():
    from leafmachine3 import doctor_probe

    rng = np.random.default_rng(0)
    x, w, b = rng.random((1, 3, 6, 5)), rng.normal(size=(2, 3, 3, 3)), rng.normal(size=2)
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    want = np.zeros((1, 2, 6, 5))
    for o in range(2):
        for i in range(6):
            for j in range(5):
                want[0, o, i, j] = max(0.0, b[o] + (w[o] * xp[0, :, i:i + 3, j:j + 3]).sum())
    np.testing.assert_allclose(doctor_probe._probe_conv_reference(x, w, b), want, rtol=1e-12, atol=1e-12)


def test_the_marker_list_never_carries_onnxruntime_options():
    assert coreml.providers() == ["CoreMLExecutionProvider", "CPUExecutionProvider"]


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
    from leafmachine3.inference import coreml_native as cn

    path = tmp_path / "m.onnx"
    path.write_bytes(_end2end_graph(nc, extra, k=300))
    feat = _feat(nc, extra, ties=ties)
    full = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"feat": feat})[0]
    split = cn.find_end2end_split(onnx.load(str(path)))
    assert split is not None and len(split.output) == (3 if extra else 2)
    parts = ort.InferenceSession(cn.onnx_split_model(path), providers=["CPUExecutionProvider"]).run(None, {"feat": feat})
    assert [p.shape[-1] for p in parts] == [4, nc] + ([extra] if extra else [])
    got = coreml.end2end_tail(parts[0], parts[1], parts[2] if extra else None, 300)
    assert got.shape == full.shape == (1, 300, 6 + extra)
    np.testing.assert_array_equal(got, full)


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
def test_a_graph_without_the_tail_has_no_split(tmp_path):
    from onnx import TensorProto, helper

    from leafmachine3.inference import coreml_native as cn

    graph = helper.make_graph([helper.make_node("Relu", ["x"], ["y"])], "plain",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    assert cn.find_end2end_split(model) is None
    path = tmp_path / "plain.onnx"
    onnx.save(model, str(path))
    with pytest.raises(cn.CoremlPackageError):
        cn.onnx_split_model(path)


# --------------------------------------------------------------------------------------------------
# the real exports and their packages, when installed (LM3_MODELS_DIR or <repo>/models)
# --------------------------------------------------------------------------------------------------
def _models_root() -> Path:
    return Path(os.environ.get("LM3_MODELS_DIR") or REPO / "models")


@needs_onnx
@pytest.mark.parametrize("name,task", [("plant_detector", "detect"), ("archival_detector", "detect"),
                                       ("leaf_segmenter", "segment"), ("landmark_detector", "pose")])
def test_real_exports_native_path_matches_the_full_graph(name, task, monkeypatch):
    """The simulated native path (the ONNX graph cut where the package is + the numpy tail) gives the
    CPU's exact detections, fed the CPU's exact input tensor."""
    path = _models_root() / name / "model.onnx"
    if not (path.parent / "coreml_fp32" / "model.json").exists():
        pytest.skip(f"no coreml_fp32 package beside {path}")
    from leafmachine3.inference import coreml_check
    from leafmachine3.inference import coreml_native as cn
    from leafmachine3.inference import ultra_replacements as ultra_rep

    monkeypatch.setenv("LM3_COREML_SIMULATE", "1")
    img = ultra_rep.imread(coreml_check.calibration_images()[0])
    cut = ultra_rep.YOLO(path, task=task, providers=coreml.providers())
    ref = ultra_rep.YOLO(path, task=task, providers=["CPUExecutionProvider"])
    assert isinstance(cut.session, cn.NativeSession)
    imgsz = ref.metadata["imgsz"]
    a = ref.predict(img, conf=0.1, imgsz=imgsz, retina_masks=True)[0]
    b = cut.predict(img, conf=0.1, imgsz=imgsz, retina_masks=True)[0]
    assert not cut.session.stats["cpu_fallback"]            # the CPU's rect shape is in the package
    np.testing.assert_array_equal(a.boxes.xyxy, b.boxes.xyxy)
    np.testing.assert_array_equal(a.boxes.conf, b.boxes.conf)
    if task == "segment" and a.masks is not None:
        np.testing.assert_array_equal(a.masks.data, b.masks.data)
    if task == "pose":
        np.testing.assert_array_equal(a.keypoints.data, b.keypoints.data)
