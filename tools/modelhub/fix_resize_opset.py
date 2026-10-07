#!/usr/bin/env python3
"""Convert YOLO26 ONNX exports from opset 19 to 18 so their Resize layers run on the GPU.

Why: onnxruntime 1.20's CUDA execution provider registers its Resize kernel only up to opset 18.
Ultralytics exported LM3's YOLO26 models (archival / plant / landmark detectors, leaf segmenter) at
opset 19, so every upsampling Resize falls back to the CPU and onnxruntime brackets it with
device<->host copies -- the "N Memcpy nodes are added to the graph main_graph for
CUDAExecutionProvider" warning printed for every worker session. The Resize attributes these graphs
use (nearest / asymmetric / floor) mean the same thing at opset 18, and onnx.version_converter
rewrites the opset declaration, so the converted graph computes the same function.

Nothing is trusted on faith. For every model this tool:
  * converts with onnx.version_converter and runs onnx.checker;
  * requires every ``metadata_props`` key (what ultra_replacements reads) to survive unchanged;
  * runs the original and the converted graph on the SAME random inputs, at two input sizes, on the
    CPU and on CUDA (default provider options, i.e. what LM3 runs), and requires every output to be
    BITWISE identical;
  * requires the converted graph to have no operator without a CUDA kernel and to copy no data
    between host and device (int64 shape values onnxruntime keeps on the CPU by design are fine);
  * times both on CUDA.
A model that fails any check is reported and NOT written. Output: <out>/<relative path> plus
<out>/report.json (old/new sha256, checks, timings), ready to publish as a new Hub revision.

Needs `onnx` (the dev group: `uv run --group full python tools/modelhub/fix_resize_opset.py ...`) and,
for the CUDA checks, a GPU with the CUDA libraries on LD_LIBRARY_PATH (run it through `lm3`'s
environment; see core/cuda_libs.py).

    python tools/modelhub/fix_resize_opset.py --models models --out converted_models \\
        archival_detector/model.onnx plant_detector/model.onnx landmark_detector/model.onnx \\
        leaf_segmenter/model.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import version_converter

TARGET_OPSET = 18
CUDA = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
CPU = ["CPUExecutionProvider"]

_PLACEMENT_PROBE = r'''
import sys, onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 0; so.log_verbosity_level = 0
ort.InferenceSession(sys.argv[1], so, providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
'''


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 23), b""):
            h.update(block)
    return h.hexdigest()


#: Ops that only do shape arithmetic. onnxruntime keeps these on the CPU on purpose, so a tensor
#: computed purely from Shape outputs through them is a few int64s, not image data.
_SHAPE_OPS = {"Shape", "Gather", "Slice", "Unsqueeze", "Squeeze", "Concat", "Cast", "Constant",
              "Add", "Sub", "Mul", "Div", "Floor", "Ceil", "Reshape", "Identity"}


def _shape_derived(model: onnx.ModelProto, tensor: str) -> bool:
    """True when ``tensor`` is computed only from Shape outputs and constants (no data flows in)."""
    producer = {o: n for n in model.graph.node for o in n.output}
    inits = {i.name for i in model.graph.initializer}
    seen, stack, saw_shape = set(), [tensor], False
    while stack:
        t = stack.pop()
        if t in seen or not t:
            continue
        seen.add(t)
        node = producer.get(t)
        if node is None:
            if t in inits:
                continue
            return False                                  # a graph input: image data
        if node.op_type == "Shape":
            saw_shape = True
            continue                                      # reads only the shape of its input
        if node.op_type not in _SHAPE_OPS:
            return False
        stack.extend(node.input)
    return saw_shape


def placement(path: Path) -> dict:
    """Ask onnxruntime (in a child, at verbose logging) which ops lack a CUDA kernel and what it copies.

    ``data_copies`` are copies of tensors that carry data (the CPU fallback this tool exists to
    remove); ``shape_copies`` are int64 shape values onnxruntime computes on the CPU by design, which a
    graph with dynamic input sizes cannot avoid.
    """
    r = subprocess.run([sys.executable, "-c", _PLACEMENT_PROBE, str(path)], capture_output=True, text=True)
    copied = re.findall(r"Add Memcpy(?:FromHost after|ToHost before) (\S+) for CUDAExecutionProvider", r.stderr)
    model = onnx.load(path, load_external_data=False)
    shape = sorted(t for t in set(copied) if _shape_derived(model, t))
    return {"no_cuda_kernel_ops": sorted(set(re.findall(r"CUDA kernel not found in registries for Op type: (\w+)", r.stderr))),
            "memcpy_nodes": len(copied), "shape_copies": shape,
            "data_copies": sorted(set(copied) - set(shape))}


def _session(path: Path, providers) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=providers)


def _inputs(model: onnx.ModelProto) -> list[np.ndarray]:
    """Two input sizes from the export's own imgsz: square, and a 3:4 letterboxed sheet shape."""
    meta = {p.key: p.value for p in model.metadata_props}
    size = max(int(v) for v in re.findall(r"\d+", meta.get("imgsz", "640")))
    rng = np.random.default_rng(0)
    return [rng.random((1, 3, size, size), dtype=np.float32),
            rng.random((1, 3, size, (size * 3 // 4) // 32 * 32), dtype=np.float32)]


def compare(orig: Path, conv: Path, xs: list[np.ndarray]) -> dict:
    out = {}
    for name, prov in (("cpu", CPU), ("cuda", CUDA)):
        a, b = _session(orig, prov), _session(conv, prov)
        iname = a.get_inputs()[0].name
        diffs = []
        for x in xs:
            for oa, ob in zip(a.run(None, {iname: x}), b.run(None, {iname: x})):
                diffs.append(0.0 if np.array_equal(oa, ob) else float(np.abs(oa - ob).max()))
        out[f"{name}_bitwise_identical"] = all(d == 0.0 for d in diffs)
        out[f"{name}_max_abs_diff"] = max(diffs)
    timing = {}
    for label, path in (("original", orig), ("converted", conv)):
        s = _session(path, CUDA)
        iname, x = s.get_inputs()[0].name, xs[0]
        for _ in range(3):
            s.run(None, {iname: x})
        runs = []
        for _ in range(20):
            t0 = time.perf_counter()
            s.run(None, {iname: x})
            runs.append(time.perf_counter() - t0)
        timing[f"cuda_median_ms_{label}"] = round(1000 * float(np.median(runs)), 2)
    return {**out, **timing}


def convert_one(models: Path, rel: str, out: Path) -> dict:
    src = models / rel
    model = onnx.load(src)
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    rec: dict = {"model": rel, "source_sha256": sha256(src), "source_opset": opset}
    if opset <= TARGET_OPSET:
        rec["result"] = f"skipped: already opset {opset}"
        return rec
    converted = version_converter.convert_version(model, TARGET_OPSET)
    onnx.checker.check_model(converted)
    before = {p.key: p.value for p in model.metadata_props}
    after = {p.key: p.value for p in converted.metadata_props}
    rec["metadata_preserved"] = before == after
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, dst)
    rec["placement_before"], rec["placement_after"] = placement(src), placement(dst)
    rec.update(compare(src, dst, _inputs(model)))
    ok = (rec["metadata_preserved"] and rec["cpu_bitwise_identical"] and rec["cuda_bitwise_identical"]
          and not rec["placement_after"]["no_cuda_kernel_ops"] and not rec["placement_after"]["data_copies"])
    if ok:
        rec["result"], rec["converted_sha256"] = "converted", sha256(dst)
    else:
        dst.unlink()
        rec["result"] = "REJECTED: a check failed; nothing written"
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", type=Path, required=True, help="the installed models folder")
    ap.add_argument("--out", type=Path, required=True, help="where converted files and report.json go")
    ap.add_argument("files", nargs="+", help="model paths relative to --models")
    args = ap.parse_args(argv)
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise SystemExit("the CUDA checks need onnxruntime-gpu with a GPU")
    args.out.mkdir(parents=True, exist_ok=True)
    report = [convert_one(args.models, rel, args.out) for rel in args.files]
    (args.out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for r in report:
        extra = ""
        if "cuda_median_ms_original" in r:
            extra = (f"  cuda {r['cuda_median_ms_original']} -> {r['cuda_median_ms_converted']} ms"
                     f"  memcpy {r['placement_before']['memcpy_nodes']} -> {r['placement_after']['memcpy_nodes']}")
        print(f"{r['model']:32s} {r['result']}{extra}")
    return 0 if all(r["result"] != "REJECTED: a check failed; nothing written" for r in report) else 1


if __name__ == "__main__":
    sys.exit(main())
