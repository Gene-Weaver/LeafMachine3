#!/usr/bin/env python
"""Export an LM3 model to ``coreml_fp32``: the CoreML package LM3 runs natively on a Mac.

This is the CoreML export for every LM3 model published to the Hub from now on. The older ``coreml/``
folders (Ultralytics' default export: float16, one fixed square size) stay as they are for use outside
LM3. A ``coreml_fp32`` package gives the CPU's answers because:

* weights and arithmetic are float32 (``compute_precision=FLOAT32``; coremltools defaults to float16);
* its input is a float32 tensor -- exactly the tensor LM3's CPU path builds (letterbox, RGB, /255,
  ImageNet normalization for UNet++) -- not a CoreML image input;
* one package holds every input shape LM3 feeds the model. A YOLO detector, segmenter or pose model gets
  the minimum-rectangle letterbox (long side ``imgsz``, short side a multiple of 32), so the package
  enumerates all ``2 * imgsz / 32 - 1`` of those shapes; fixed-size models (the ruler member, UNet++)
  have their one shape;
* a YOLO26 end2end network stops at its head's split -- ``boxes``, ``scores``, ``extra`` (mask
  coefficients or keypoints) and ``protos`` -- and LM3 runs the top-k tail in numpy, exactly the ONNX
  graph's tail (``leafmachine3/inference/coreml_native.py``);
* input and output names, and the metadata, are the ONNX twin's, plus ``lm3_*`` keys (format, source
  checkpoint sha256, tail, outputs).

Before converting, the traced network is compared with the published ONNX model on the CPU (the split
tensors of an end2end graph, the whole output otherwise) at several input shapes, so a package is never
built from a network that differs from the ONNX model. Conversion runs on Linux or macOS; only running a
package needs a Mac. On a Mac the package is then loaded on the GPU in a child process and run on the
same inputs, compared with the ONNX model again (``load_test`` in model.json): a package that does not
load, or disagrees, is removed and reported, and ``all`` goes on with the next model.

Two rewrites make the network convert and compile, both the same arithmetic (the ONNX comparison proves
it for each package): Proto26's resize-to-size becomes an integer scale factor, and Attention's
``v @ attn^T`` becomes ``(attn @ v^T)^T`` -- Apple's GPU compiler (MPSGraph, macOS 26) aborts on the
original form, in every YOLO26 model, including Ultralytics' own CoreML exports.

Environment: torch, ultralytics 8.4.79 (the version LM3's ONNX exports were made with), coremltools 9.0,
onnx and onnxruntime; segmentation_models_pytorch for UNet++. On a Mac, without touching LM3's own
environment (uv layers these over it in a throwaway environment):

    uv run --frozen --no-sync --with coremltools==9.0 --with torch==2.5.1 --with ultralytics==8.4.79 \\
        python tools/modelhub/export_coreml_fp32.py yolo best.pt model.onnx OUT/coreml_fp32/model.mlpackage

Usage:
    export_coreml_fp32.py all --models DIR [--pytorch DIR] [--out DIR] [--only STAGE ...]
    export_coreml_fp32.py yolo   CHECKPOINT ONNX_TWIN OUT.mlpackage      # detect | segment | pose | classify
    export_coreml_fp32.py unetpp CHECKPOINT ONNX_TWIN OUT.mlpackage      # UNet++ specimen segmenter

``all`` exports every default model in LM3's model lock that runs on CoreML (see :data:`CPU_ONLY`):
ONNX twins from ``--models`` (an LM3 models folder), checkpoints from ``--pytorch`` (default the same
folder; ``lm3 models install --formats pytorch,meta --dest DIR`` fetches them), packages to
``<out>/<model dir>/coreml_fp32/`` -- by default beside the ONNX twins, where LM3 looks for them.

Beside the package the exporter writes ``model.json`` (``coreml_fp32/model.json``): the contract LM3's
runtime reads -- input name and shapes, outputs, tail, the ONNX twin's names, shapes and metadata, the
package program's sha256 -- plus the provenance (checkpoint and ONNX hashes, versions, the ONNX check).
Publish the whole ``coreml_fp32/`` folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np

FORMAT = "coreml_fp32"
SCHEMA = "1"                  # bump when the package contract (names, cut, metadata) changes
MIN_TARGET = "macOS13"        # ML Program + enumerated shapes; every Apple-silicon Mac runs it
#: largest |torch - onnxruntime| allowed per output, relative to max(that output's largest |value|, 1). Measured:
#: YOLO26 <= 6e-5, UNet++ (EfficientNet-B7, logits to ~130) 1.1e-4 with identical masks; a wrong network is O(1).
CHECK_TOL = 1e-3


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rect_shapes(imgsz: int, stride: int = 32) -> list[tuple[int, int, int, int]]:
    """Every input shape LM3's letterbox gives a YOLO model: long side ``imgsz``, short side a multiple
    of ``stride`` (``ultra_replacements.letterbox`` with ``auto=True``). Square first."""
    sides = range(stride, imgsz + 1, stride)
    return ([(1, 3, imgsz, imgsz)] + [(1, 3, imgsz, w) for w in sides if w != imgsz]
            + [(1, 3, h, imgsz) for h in sides if h != imgsz])


def onnx_twin(path: str | Path) -> dict[str, Any]:
    import onnx

    m = onnx.load(str(path), load_external_data=False)

    def io(values):
        return [(v.name, [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim]) for v in values]

    return {"path": str(path), "sha256": sha256(path), "inputs": io(m.graph.input), "outputs": io(m.graph.output),
            "metadata": {p.key: p.value for p in m.metadata_props}}


# --------------------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------------------
def yolo_network(checkpoint: str):
    """The checkpoint's network prepared as Ultralytics' exporter prepares it for ONNX (fused, float32,
    ``export=True``, ``dynamic=True`` so anchors follow the input shape, C2f ``forward_split``), with the
    end2end head stopped at its split. Returns ``(module, task, output_names, split_sizes, max_det)``."""
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO
    from ultralytics.nn.modules import C2f, Classify, Detect
    from ultralytics.nn.modules.block import Attention, Proto, Proto26

    yolo = YOLO(checkpoint)
    task, model = yolo.task, yolo.model
    for p in model.parameters():
        p.requires_grad = False
    model = model.eval().float().fuse()
    sizes: list[int] = []
    max_det = 300
    for m in model.modules():
        if isinstance(m, Classify):
            m.export = True
            m.format = "onnx"
        elif isinstance(m, Detect):
            m.dynamic, m.export, m.format, m.max_det, m.shape = True, True, "onnx", max_det, None
            m.xyxy = False
            if not m.end2end:
                raise SystemExit("not an end2end YOLO26 head: LM3's runtime has no NMS")
            sizes = [4, m.nc] + ([m.nm] if hasattr(m, "nm") else []) + ([m.nk] if hasattr(m, "nk") else [])
            # stop at the split the end2end tail starts with; LM3 runs the tail in numpy
            m.postprocess = types.MethodType(lambda self, preds, s=tuple(sizes): tuple(preds.split(list(s), dim=-1)), m)
        elif isinstance(m, C2f):
            m.forward = m.forward_split
    strides = [int(s) for s in model.stride.tolist()]
    for m in model.modules():
        if isinstance(m, Proto26):
            # Proto26 resizes the coarser maps to the stride-8 map's size; coremltools cannot convert a
            # resize to a computed size. Every LM3 input side is a multiple of 32, so that size is always
            # an exact integer multiple (x2, x4) and nearest-neighbor by that factor gives the same pixels.
            # The ONNX comparison below proves it.
            factors = tuple(float(strides[i + 1] / strides[0]) for i in range(len(m.feat_refine)))

            def proto_forward(self, x, return_semantic=True, factors=factors):
                feat = x[0]
                for i, f in enumerate(self.feat_refine):
                    feat = feat + F.interpolate(f(x[i + 1]), scale_factor=factors[i], mode="nearest")
                return Proto.forward(self, self.feat_fuse(feat))

            m.forward = types.MethodType(proto_forward, m)
        elif isinstance(m, Attention):
            m.forward = types.MethodType(_attention_forward, m)

    class Flat(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            y = self.net(x)
            if isinstance(y, tuple) and len(y) == 2 and isinstance(y[0], tuple):   # segment: ((b, s, e), protos)
                return (*y[0], y[1])
            return y

    if task == "classify":
        return Flat(model).eval(), task, None, [], 0
    names = ["boxes", "scores"] + (["extra"] if len(sizes) == 3 else []) + (["protos"] if task == "segment" else [])
    return Flat(model).eval(), task, names, sizes, max_det


def _attention_forward(self, x):
    """Ultralytics ``Attention.forward`` (8.4.79) with its second matmul written as ``(attn @ v^T)^T``
    instead of ``v @ attn^T`` -- the same products and sums. Apple's GPU compiler (MPSGraph) aborts the
    whole process ("MLIR pass manager failed") on the original form under macOS 26; this one compiles."""
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2)
    attn = ((q.transpose(-2, -1) @ k) * self.scale).softmax(dim=-1)
    o = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
    return self.proj(o.reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W)))


def unetpp_network(checkpoint: str):
    """UNet++ from an LM3 specimen checkpoint (the EMA weights when present, as every published UNet++)."""
    import segmentation_models_pytorch as smp
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = (ck.get("cfg") or {}) if isinstance(ck, dict) else {}
    arch, enc = cfg.get("arch", "UnetPlusPlus"), cfg.get("encoder", "efficientnet-b7")
    model = getattr(smp, arch)(encoder_name=enc, encoder_weights=None, in_channels=3, classes=1, activation=None)
    state = next((ck[k] for k in ("ema", "model", "state_dict") if isinstance(ck, dict) and k in ck), ck)
    model.load_state_dict(state)
    return model.eval()


# --------------------------------------------------------------------------------------------------
# The ONNX comparison
# --------------------------------------------------------------------------------------------------
def onnx_split_session(path: str):
    """An onnxruntime CPU session on the ONNX twin cut at its end2end split (same rule as the runtime:
    the last Split whose second output feeds ReduceMax -> TopK); outputs are the split tensors, then
    every other graph output (the segmenter's protos)."""
    import onnx
    import onnxruntime as ort
    from onnx.utils import Extractor

    m = onnx.load(path)
    consumers: dict[str, list] = {}
    for n in m.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    split = None
    for n in reversed(m.graph.node):
        if n.op_type == "Split" and len(n.output) in (2, 3):
            if any(c.op_type == "TopK" for rm in consumers.get(n.output[1], []) if rm.op_type == "ReduceMax"
                   for c in consumers.get(rm.output[0], [])):
                split = n
                break
    if split is None:
        raise SystemExit(f"{path}: no end2end split found")
    known = {v.name for v in m.graph.value_info} | {o.name for o in m.graph.output}
    for name in split.output:
        if name not in known:
            m.graph.value_info.append(onnx.helper.make_tensor_value_info(name, m.graph.input[0].type.tensor_type.elem_type, None))
    outputs = list(split.output) + [o.name for o in m.graph.output[1:]]
    cut = Extractor(m).extract_model([i.name for i in m.graph.input], outputs)
    return ort.InferenceSession(cut.SerializeToString(), providers=["CPUExecutionProvider"])


def check_inputs(shape: tuple, images: Sequence[Path], i: int, normalize=None) -> np.ndarray:
    """A real sheet resized to ``shape`` (natural image statistics), or uniform noise without images."""
    if not images:
        return np.random.default_rng(i).random(shape, dtype=np.float32)
    import cv2

    img = cv2.imread(str(images[i % len(images)]), cv2.IMREAD_COLOR)
    x = cv2.resize(img, (shape[3], shape[2]), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255.0
    if normalize is not None:
        x = (x - normalize[0]) / normalize[1]
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def compare_with_onnx(traced, sess, shapes: Sequence[tuple], images: Sequence[Path] = (), normalize=None,
                      keep: list | None = None) -> list[dict]:
    """Traced network vs the ONNX session on the CPU, one input per shape. ``keep`` collects
    ``(shape, input, onnx outputs)`` for the Mac load test."""
    import torch

    rows = []
    in_name = sess.get_inputs()[0].name
    for i, shape in enumerate(shapes):
        x = check_inputs(shape, images, i, normalize)
        with torch.no_grad():
            got = traced(torch.from_numpy(x))
        got = [t.numpy() for t in (got if isinstance(got, (tuple, list)) else [got])]
        ref = sess.run(None, {in_name: x})
        if keep is not None:
            keep.append((shape, x, ref))
        worst = []
        for g, r in zip(got, ref):
            if g.shape != r.shape:
                raise SystemExit(f"shape {shape}: traced output {g.shape} vs ONNX {r.shape}")
            worst.append(float(np.abs(g - r).max()) / max(float(np.abs(r).max()), 1.0))
        rows.append({"shape": list(shape), "max_rel_diff": worst, "ok": max(worst) <= CHECK_TOL})
        print(f"  check {shape}: max |torch - onnx| / max(|onnx|, 1) per output {['%.1e' % w for w in worst]}",
              flush=True)
    return rows


# --------------------------------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------------------------------
def _numpy2_cast_compat() -> None:
    """coremltools 9.0 folds a traced size computation into a constant with ``int(array)``; NumPy >= 2.4
    refuses that for a one-element array that is not 0-d (the ruler member's attention, ``N = H * W``,
    at its fixed size). Same constant, via ``.item()``."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.frontend.torch import ops

    if getattr(ops._cast, "lm3_compat", False):
        return
    original = ops._cast

    def _cast(context, node, dtype, dtype_name):
        x = ops._get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const() and isinstance(x.val, np.ndarray) and x.val.size == 1:
            context.add(mb.const(val=dtype(x.val.item()), name=node.name), node.name)
            return None
        return original(context, node, dtype, dtype_name)

    _cast.lm3_compat = True
    ops._cast = _cast


def convert(traced, input_name: str, shapes: Sequence[tuple], output_names: Sequence[str]):
    import coremltools as ct

    _numpy2_cast_compat()
    shape = (ct.EnumeratedShapes(shapes=[tuple(s) for s in shapes], default=tuple(shapes[0]))
             if len(shapes) > 1 else tuple(shapes[0]))
    return ct.convert(
        traced, convert_to="mlprogram", compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=getattr(ct.target, MIN_TARGET),
        inputs=[ct.TensorType(name=input_name, shape=shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=n, dtype=np.float32) for n in output_names],
        skip_model_load=True)          # loading compiles for the GPU; done in a child (mac_load_test)


def _rel_diffs(got: Sequence[np.ndarray], ref: Sequence[np.ndarray]) -> list[float]:
    return [float(np.abs(np.asarray(g, np.float32) - r).max()) / max(float(np.abs(r).max()), 1.0)
            for g, r in zip(got, ref)]


def mac_load_test(pkg: Path, samples: list, timeout: int = 1800) -> dict:
    """On a Mac: load the package on CPU+GPU in a child process and run the check inputs through it.

    A Core ML compiler failure kills the process it happens in (SIGABRT), so the load runs in a child:
    a failure is reported instead of ending the exporter. Off a Mac (or with
    LM3_EXPORT_SKIP_LOAD_TEST=1) nothing runs."""
    if sys.platform != "darwin" or os.environ.get("LM3_EXPORT_SKIP_LOAD_TEST") == "1":
        return {"skipped": "not on a Mac" if sys.platform != "darwin" else "LM3_EXPORT_SKIP_LOAD_TEST=1"}
    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "in.npz", Path(td) / "out.npz"
        np.savez(inp, **{f"x{i}": x for i, (_shape, x, _ref) in enumerate(samples)})
        cmd = [sys.executable, str(Path(__file__).resolve()), "load-test", str(pkg), str(inp), str(outp)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"no answer after {timeout} s"}
        if r.returncode != 0:
            tail = [ln for ln in (r.stderr or "").splitlines() if ln.strip()][-6:]
            return {"ok": False, "returncode": r.returncode, "error": " | ".join(tail)}
        info = json.loads(r.stdout.strip().splitlines()[-1])
        got = np.load(outp)
        rows = [{"shape": list(shape), "max_rel_diff": _rel_diffs([got[f"y{i}_{j}"] for j in range(len(ref))], ref)}
                for i, (shape, _x, ref) in enumerate(samples)]
    worst = max(max(r["max_rel_diff"]) for r in rows)
    return {"ok": worst <= CHECK_TOL, "load_s": info["load_s"], "ms": info["ms"], "coreml_vs_onnx": rows,
            "max_rel_diff": worst}


def _load_test_child(pkg: str, inp: str, outp: str) -> int:
    """The child side of :func:`mac_load_test`: load, predict each input, save the outputs."""
    import coremltools as ct

    t0 = time.perf_counter()
    model = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    load_s = time.perf_counter() - t0
    spec = model.get_spec()
    in_name = spec.description.input[0].name
    out_names = [o.name for o in spec.description.output]
    xs = np.load(inp)
    out, ms = {}, []
    for i in range(len(xs.files)):
        t0 = time.perf_counter()
        y = model.predict({in_name: xs[f"x{i}"]})
        ms.append(round((time.perf_counter() - t0) * 1000.0, 1))
        for j, n in enumerate(out_names):
            out[f"y{i}_{j}"] = np.asarray(y[n], np.float32)
    np.savez(outp, **out)
    print(json.dumps({"load_s": round(load_s, 2), "ms": ms}))
    return 0


def versions() -> dict[str, str]:
    import importlib.metadata as md

    out = {}
    for p in ("coremltools", "torch", "ultralytics", "segmentation-models-pytorch", "onnxruntime"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    return out


IMAGENET = (np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32))


def export(kind: str, checkpoint: str, onnx_path: str, out: str, n_check: int = 4,
           check_images: Sequence[Path] = ()) -> dict:
    import torch

    twin = onnx_twin(onnx_path)
    meta = dict(twin["metadata"])
    in_name = twin["inputs"][0][0]
    onnx_out_names = [n for n, _ in twin["outputs"]]
    t0 = time.perf_counter()
    if kind == "yolo":
        net, task, cut_names, sizes, max_det = yolo_network(checkpoint)
        imgsz = int(json.loads(meta.get("imgsz", "[640, 640]").replace("'", '"'))[0])
        stride = int(meta.get("stride", 32))
        if task == "classify":
            shapes, out_names, tail = [(1, 3, imgsz, imgsz)], onnx_out_names, "none"
        else:
            shapes, out_names, tail = rect_shapes(imgsz, stride), cut_names, "end2end"
    elif kind == "unetpp":
        net, task, sizes, max_det = unetpp_network(checkpoint), "specimen", [], 0
        shapes = [tuple(int(d) for d in twin["inputs"][0][1])]
        out_names, tail = onnx_out_names, "none"
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    example = torch.from_numpy(np.random.default_rng(1).random(shapes[0], dtype=np.float32))
    with torch.no_grad():
        traced = torch.jit.trace(net, example, check_trace=False)
    print(f"{Path(checkpoint).name}: {task}, {len(shapes)} input shape(s), outputs {out_names}; traced in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)

    # the traced network must equal the published ONNX model before it is converted
    import onnxruntime as ort

    sess = onnx_split_session(onnx_path) if tail == "end2end" else ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    pick = [shapes[i] for i in np.unique(np.linspace(0, len(shapes) - 1, max(1, n_check)).round().astype(int))]
    samples: list = []
    check = compare_with_onnx(traced, sess, pick, check_images, IMAGENET if kind == "unetpp" else None, samples)
    if not all(r["ok"] for r in check):
        raise SystemExit(f"the traced network differs from {onnx_path} by more than {CHECK_TOL} -- not converting")
    del sess

    t1 = time.perf_counter()
    ml = convert(traced, in_name, shapes, out_names)
    print(f"converted in {time.perf_counter() - t1:.0f} s", flush=True)
    src_sha = sha256(checkpoint)
    lm3 = {"lm3_format": FORMAT, "lm3_format_schema": SCHEMA, "lm3_tail": tail,
           "lm3_outputs": ",".join(out_names), "lm3_onnx_outputs": ",".join(onnx_out_names),
           "lm3_source_sha256": src_sha, "lm3_onnx_sha256": twin["sha256"],
           "lm3_onnx_io": json.dumps({"inputs": twin["inputs"], "outputs": twin["outputs"]}),
           "lm3_shapes": f"rect:{shapes[0][2]}:{int(meta.get('stride', 32))}" if len(shapes) > 1 else "fixed",
           "lm3_exporter": ", ".join(f"{k} {v}" for k, v in versions().items())}
    if tail == "end2end":
        lm3.update(lm3_split=",".join(map(str, sizes)), lm3_max_det=str(max_det))
    for k, v in {**meta, **lm3}.items():
        ml.user_defined_metadata[k] = str(v)
    if meta.get("author"):
        ml.author = meta["author"]
    if meta.get("license"):
        ml.license = meta["license"]
    if meta.get("description"):
        ml.short_description = meta["description"]
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    ml.save(str(out_p))
    record = {"format": FORMAT, "schema": SCHEMA, "package": out_p.name,
              "mlmodel_sha256": sha256(out_p / "Data" / "com.apple.CoreML" / "model.mlmodel"),
              "kind": kind, "task": task, "input": {"name": in_name, "shapes": [list(s) for s in shapes]},
              "outputs": out_names, "tail": tail, "split": sizes if tail == "end2end" else [],
              "max_det": max_det if tail == "end2end" else 0,
              "onnx": {"sha256": twin["sha256"], "inputs": twin["inputs"], "outputs": twin["outputs"], "metadata": meta},
              "source": {"file": Path(checkpoint).name, "sha256": src_sha},
              "onnx_check": check, "exporter": versions(),
              "package_bytes": sum(p.stat().st_size for p in out_p.rglob("*") if p.is_file()),
              "seconds": round(time.perf_counter() - t0, 1)}
    t2 = time.perf_counter()
    record["load_test"] = mac_load_test(out_p, samples)
    if record["load_test"].get("ok") is False:
        shutil.rmtree(out_p, ignore_errors=True)
        lt = record["load_test"]
        raise SystemExit(f"{out_p} failed the load test on this Mac and was removed: "
                         + (lt.get("error") or f"CoreML vs ONNX max diff {lt.get('max_rel_diff'):.1e} > {CHECK_TOL}"))
    if "ok" in record["load_test"]:
        lt = record["load_test"]
        print(f"  load test on this Mac: loaded in {lt['load_s']:.1f} s, CoreML vs ONNX max diff "
              f"{lt['max_rel_diff']:.1e}, predict ms {lt['ms']} ({time.perf_counter() - t2:.0f} s)", flush=True)
    (out_p.parent / "model.json").write_text(json.dumps(record, indent=1))
    print(f"wrote {out_p} ({record['package_bytes'] / 1e6:.0f} MB) + model.json in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return record


#: Lock models that stay on the CPU on a Mac (no faster on CoreML, or no CoreML conversion): no package.
CPU_ONLY = frozenset({"yolo26n_cls_224", "dinov2_frozen_mlp"})


def lock_jobs(models: Path, pytorch: Path, out: Path, only: Sequence[str] = ()) -> list[tuple]:
    """``(stage, model_key, kind, checkpoint, onnx, package)`` for every default lock model that runs on
    CoreML. The package goes to ``<model dir>/coreml_fp32/model.mlpackage``, the model dir being the
    folder of ``model.onnx`` (or of ``exported/model.onnx``) -- ``coreml_native.package_for``'s rule."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from leafmachine3.modelhub.registry import load_lock

    jobs = []
    for stage, action in load_lock().actions.items():
        if only and stage not in only:
            continue
        for unit in action.units:
            key = unit.model_key or ""
            onnx_f = next((f for f in unit.files if f.format == "onnx"), None)
            pt_f = next((f for f in unit.files if f.format == "pytorch"), None)
            if key in CPU_ONLY or key.startswith("birefnet") or onnx_f is None or pt_f is None:
                continue
            model_dir = Path(onnx_f.dest).parent
            if model_dir.name == "exported":
                model_dir = model_dir.parent
            kind = "unetpp" if key.startswith("unetpp_") else "yolo"
            jobs.append((stage, key, kind, pytorch / pt_f.dest, models / onnx_f.dest,
                         out / model_dir / FORMAT / "model.mlpackage"))
    return jobs


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["load-test"]:                # internal: the child process of mac_load_test
        return _load_test_child(*argv[1:4])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    one = {}
    for kind in ("yolo", "unetpp"):
        one[kind] = sub.add_parser(kind, help=f"one {kind} model")
        one[kind].add_argument("checkpoint", help="the published PyTorch checkpoint (pytorch/ folder of the Hub repo)")
        one[kind].add_argument("onnx", help="the published ONNX twin (onnx/ folder): names, metadata and the check")
        one[kind].add_argument("out", help="output package, e.g. OUT/coreml_fp32/model.mlpackage")
    every = sub.add_parser("all", help="every default model in the lock that runs on CoreML")
    every.add_argument("--models", required=True, help="LM3 models folder with the ONNX twins installed")
    every.add_argument("--pytorch", default=None, help="folder with the checkpoints (default: --models)")
    every.add_argument("--out", default=None, help="where to write <model dir>/coreml_fp32/ (default: --models)")
    every.add_argument("--only", nargs="*", default=(), help="stages to export (default: all)")
    for p in [*one.values(), every]:
        p.add_argument("--check-shapes", type=int, default=6, help="input shapes compared with the ONNX model")
        p.add_argument("--check-images", default=str(Path(__file__).resolve().parents[2] / "examples" / "images"),
                       help="sheets resized to each checked shape (default examples/images; noise if none)")
    a = ap.parse_args(argv)
    folder = Path(a.check_images)
    images = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if folder.is_dir() else []
    if a.kind != "all":
        export(a.kind, a.checkpoint, a.onnx, a.out, a.check_shapes, images)
        return 0
    models = Path(a.models).expanduser()
    jobs = lock_jobs(models, Path(a.pytorch or models).expanduser(), Path(a.out or models).expanduser(), a.only)
    missing = [str(f) for job in jobs for f in job[3:5] if not Path(f).is_file()]
    if missing:
        raise SystemExit("missing (install the onnx and pytorch formats first):\n  " + "\n  ".join(missing))
    failed = []
    for stage, key, kind, ckpt, onnx_path, out in jobs:
        print(f"\n=== {stage} ({key})", flush=True)
        try:
            export(kind, str(ckpt), str(onnx_path), str(out), a.check_shapes, images)
        except (SystemExit, Exception) as exc:  # noqa: BLE001 - report and go on with the next model
            failed.append(f"{stage}: {exc}")
            print(f"FAILED {stage}: {exc}", flush=True)
    print(f"\n{len(jobs) - len(failed)} package(s) written" + (f", {len(failed)} failed:" if failed else ""), flush=True)
    for f in failed:
        print(f"  {f}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
