"""Native CoreML on macOS: run a model's ``coreml_fp32`` package on the Apple GPU.

A ``coreml_fp32`` package (made by ``tools/modelhub/export_coreml_fp32.py``) is the model's network in
float32, taking exactly the tensor LM3's CPU path builds, with every input shape LM3 feeds it in one
package. :class:`NativeSession` stands in for ``onnxruntime.InferenceSession`` on it -- same input and
output names and shapes as the ONNX twin, same metadata -- so ``ultra_rep.YOLO``, the specimen
segmenter and the ruler member run it unchanged. For a YOLO26 end2end model the package stops at the
head's split and :func:`end2end_tail` runs the top-k tail in numpy, exactly as the ONNX graph does.

The package lives beside its ONNX twin: ``<model dir>/coreml_fp32/model.mlpackage`` where the model
dir holds ``model.onnx`` (or ``exported/model.onnx``) -- :func:`package_for`. It is compiled once per
version into a cache (``LM3_COREML_CACHE``, default ``~/Library/Caches/LeafMachine3/coreml``) and
loaded from there on every later run.

An input shape the package does not list (a custom ``imgsz``) runs that one input through the ONNX
twin on the CPU, with a warning, so results never depend on which path ran.

Environment switches (testing; the defaults are what a Mac user gets):

* ``LM3_COREML=0`` -- never use CoreML.
* ``LM3_COREML_UNITS`` -- ``CPUAndGPU`` (default) | ``CPUOnly`` | ``ALL``. ``ALL`` adds the Neural
  Engine, which computes in float16 only: a speed probe, never a setting for real work.
* ``LM3_COREML_SIMULATE=1`` -- off a Mac, run this whole path with the ONNX twin (cut at the same
  split) on the CPU in place of CoreML, so everything except CoreML itself is tested on Linux.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger("leafmachine3.inference.coreml_native")

FORMAT = "coreml_fp32"
PACKAGE = Path(FORMAT) / "model.mlpackage"
SCHEMAS = ("1",)                              # package contracts this runtime reads
PROVIDER = "CoreML"                           # what get_providers() reports

_UNITS = {"CPUAndGPU": "CPU_AND_GPU", "CPUOnly": "CPU_ONLY", "ALL": "ALL"}


class CoremlPackageError(RuntimeError):
    """The package is missing, unreadable, or not a contract this runtime knows."""


# --------------------------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------------------------
def simulated() -> bool:
    return os.environ.get("LM3_COREML_SIMULATE", "") == "1"


def available() -> bool:
    """True on a Mac with coremltools (or when simulating), unless ``LM3_COREML=0``."""
    if os.environ.get("LM3_COREML", "") == "0":
        return False
    if simulated():
        return True
    if sys.platform != "darwin":
        return False
    try:
        import coremltools  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def compute_units() -> str:
    units = os.environ.get("LM3_COREML_UNITS", "CPUAndGPU")
    if units not in _UNITS:
        log.warning("LM3_COREML_UNITS=%r is not one of %s; using CPUAndGPU", units, sorted(_UNITS))
        units = "CPUAndGPU"
    return units


def package_for(onnx_path: str | os.PathLike) -> Optional[Path]:
    """The installed ``coreml_fp32`` package of the model whose ONNX file is ``onnx_path``, or None."""
    p = Path(onnx_path)
    for model_dir in (p.parent, p.parent.parent):
        pkg = model_dir / PACKAGE
        if (pkg / "Manifest.json").is_file():
            return pkg
    return None


# --------------------------------------------------------------------------------------------------
# The YOLO26 end2end tail (also used by the onnxruntime route in coreml.py)
# --------------------------------------------------------------------------------------------------
def _topk(a: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """onnxruntime ``TopK(largest=1, sorted=1)`` on the last axis; equal values keep index order."""
    idx = np.argsort(-a, axis=-1, kind="stable")[..., :k]
    return np.take_along_axis(a, idx, -1), idx


def end2end_tail(boxes: np.ndarray, scores: np.ndarray, extra: Optional[np.ndarray], max_det: int) -> np.ndarray:
    """The Ultralytics YOLO26 end2end postprocess in numpy -> (b, k, 4 + 1 + 1 [+ e]).

    The graph's own two-stage top-k: keep the ``k`` anchors with the highest class score, then the
    ``k`` best (anchor, class) pairs among them; each row is box, score, class[, extra of its anchor].
    """
    b, n, nc = scores.shape
    k = min(int(max_det), n)
    _, anchors = _topk(scores.max(-1), k)                                       # (b, k)
    cand = np.take_along_axis(scores, np.repeat(anchors[..., None], nc, -1), 1)  # (b, k, nc)
    vals, flat = _topk(cand.reshape(b, -1), k)
    anchor = np.take_along_axis(anchors, flat // nc, 1)
    cls = (flat % nc).astype(scores.dtype)
    parts = [np.take_along_axis(boxes, np.repeat(anchor[..., None], boxes.shape[-1], -1), 1),
             vals[..., None], cls[..., None]]
    if extra is not None:
        parts.append(np.take_along_axis(extra, np.repeat(anchor[..., None], extra.shape[-1], -1), 1))
    return np.concatenate(parts, -1)


def find_end2end_split(model: Any):
    """The ``Split`` node an Ultralytics YOLO26 end2end tail starts with: the last Split whose second
    output (the class scores) feeds a ``ReduceMax`` that feeds a ``TopK``. None if there is none."""
    consumers: dict[str, list] = {}
    for n in model.graph.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    for n in reversed(list(model.graph.node)):
        if n.op_type == "Split" and len(n.output) in (2, 3):
            for rm in consumers.get(n.output[1], []):
                if rm.op_type == "ReduceMax" and any(c.op_type == "TopK" for c in consumers.get(rm.output[0], [])):
                    return n
    return None


def onnx_split_model(onnx_path: str | os.PathLike) -> bytes:
    """The ONNX graph cut at its end2end split: outputs are the split tensors, then every other graph
    output (the segmenter's protos) -- the network a ``coreml_fp32`` package holds."""
    import onnx
    from onnx.utils import Extractor

    model = onnx.load(str(onnx_path))
    split = find_end2end_split(model)
    if split is None:
        raise CoremlPackageError(f"{onnx_path}: no YOLO26 end2end split found")
    # the Extractor needs a type for each tensor it cuts at
    known = {v.name for v in model.graph.value_info} | {o.name for o in model.graph.output}
    elem = model.graph.input[0].type.tensor_type.elem_type
    for name in split.output:
        if name not in known:
            model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, elem, None))
    outputs = list(split.output) + [o.name for o in model.graph.output[1:]]
    return Extractor(model).extract_model([i.name for i in model.graph.input], outputs).SerializeToString()


# --------------------------------------------------------------------------------------------------
# Packages
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Arg:
    """The slice of onnxruntime's ``NodeArg`` LM3 reads."""
    name: str
    shape: list
    type: str = "tensor(float)"


@dataclass(frozen=True)
class _ModelMeta:
    custom_metadata_map: dict


@dataclass(frozen=True)
class PackageInfo:
    """A package's contract, from ``coreml_fp32/model.json`` (written by the exporter; no coremltools needed)."""
    path: Path
    metadata: dict                      # the ONNX twin's metadata (what ultra_rep parses)
    input_name: str
    shapes: frozenset                   # every (1, 3, H, W) the package accepts
    outputs: tuple[str, ...]            # the package's output names
    onnx_inputs: list                   # [(name, shape)] of the ONNX twin
    onnx_outputs: list
    tail: str                           # "end2end" | "none"
    split: tuple[int, ...]
    max_det: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_package(path: str | os.PathLike) -> PackageInfo:
    """Read a ``coreml_fp32`` package's contract, or raise :class:`CoremlPackageError`."""
    pkg = Path(path)
    contract_path = pkg.parent / "model.json"
    program = pkg / "Data" / "com.apple.CoreML" / "model.mlmodel"
    if not (pkg / "Manifest.json").is_file() or not program.is_file():
        raise CoremlPackageError(f"{pkg} is not a complete .mlpackage")
    try:
        c = json.loads(contract_path.read_text())
    except (OSError, ValueError) as exc:
        raise CoremlPackageError(f"cannot read {contract_path}: {exc}") from exc
    if c.get("format") != FORMAT:
        raise CoremlPackageError(f"{pkg} is not a {FORMAT} package; the older coreml/ exports are float16 "
                                 "and are not run by LM3")
    if str(c.get("schema")) not in SCHEMAS:
        raise CoremlPackageError(f"{pkg} uses package schema {c.get('schema')!r}; this LM3 reads "
                                 f"{', '.join(SCHEMAS)} -- update LM3 or reinstall the models")
    if c.get("mlmodel_sha256") != _sha256(program):
        raise CoremlPackageError(f"{pkg} does not match its model.json (program hash differs); reinstall it")
    onnx = c.get("onnx") or {}
    return PackageInfo(
        path=pkg, metadata={k: str(v) for k, v in (onnx.get("metadata") or {}).items()},
        input_name=c["input"]["name"], shapes=frozenset(tuple(int(d) for d in s) for s in c["input"]["shapes"]),
        outputs=tuple(c["outputs"]), onnx_inputs=[tuple(x) for x in onnx.get("inputs", [])],
        onnx_outputs=[tuple(x) for x in onnx.get("outputs", [])], tail=c.get("tail", "none"),
        split=tuple(int(x) for x in c.get("split") or ()), max_det=int(c.get("max_det") or 0))


def _cache_root() -> Path:
    env = os.environ.get("LM3_COREML_CACHE")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "LeafMachine3" / "coreml"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "leafmachine3" / "coreml"


def _package_key(pkg: Path) -> str:
    """Changes whenever the package's content does: hash of every file's path, size and mtime, plus the
    program file's bytes (small)."""
    h = hashlib.sha256()
    for f in sorted(p for p in pkg.rglob("*") if p.is_file()):
        st = f.stat()
        h.update(f"{f.relative_to(pkg)}:{st.st_size}:{st.st_mtime_ns}\n".encode())
        if f.name == "model.mlmodel":
            h.update(f.read_bytes())
    return h.hexdigest()[:24]


def compiled_model(pkg: Path) -> Path:
    """The package compiled to ``.mlmodelc`` -- compiled once, then reused from the cache."""
    import coremltools as ct

    dest = _cache_root() / f"{pkg.parent.parent.name}-{_package_key(pkg)}.mlmodelc"
    if dest.is_dir():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tmp = Path(ct.utils.compile_model(str(pkg)))
    staged = dest.with_name(dest.name + f".tmp{os.getpid()}")
    shutil.move(str(tmp), str(staged))
    try:
        os.replace(staged, dest)
    except OSError:                      # another process won the race: use theirs
        shutil.rmtree(staged, ignore_errors=True)
    log.info("CoreML: compiled %s in %.1f s -> %s", pkg, time.perf_counter() - t0, dest)
    return dest


class _SimulatedModel:
    """Off a Mac (``LM3_COREML_SIMULATE=1``): the ONNX twin on the CPU, cut where the package is."""

    def __init__(self, info: PackageInfo, onnx_path: str) -> None:
        import onnxruntime as ort

        model = onnx_split_model(onnx_path) if info.tail == "end2end" else str(onnx_path)
        self.session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.names = info.outputs

    def predict(self, feed: dict) -> dict:
        return dict(zip(self.names, self.session.run(None, feed)))


class NativeSession:
    """Stands in for ``ort.InferenceSession`` on a ``coreml_fp32`` package (see the module docstring)."""

    def __init__(self, package: str | os.PathLike, onnx_path: str | os.PathLike) -> None:
        self.info = read_package(package)
        self.onnx_path = str(onnx_path)
        self.label = os.path.join(self.info.path.parent.parent.name, FORMAT)
        if not self.info.onnx_inputs or not self.info.onnx_outputs:
            raise CoremlPackageError(f"{package}: model.json names no ONNX inputs/outputs")
        self._inputs = [_Arg(n, list(s)) for n, s in self.info.onnx_inputs]
        self._outputs = [_Arg(n, list(s)) for n, s in self.info.onnx_outputs]
        self._meta = _ModelMeta(dict(self.info.metadata))
        self._lock = threading.Lock()
        self._cpu = None                 # the ONNX twin on the CPU, built on the first unlisted shape
        self._warned: set = set()
        self._seen: set = set()
        #: predict() milliseconds: the first call at each input shape, and every later call
        self.stats: dict[str, list] = {"first_ms": [], "steady_ms": [], "cpu_fallback": []}
        t0 = time.perf_counter()
        if simulated() and sys.platform != "darwin":
            self.model = _SimulatedModel(self.info, self.onnx_path)
        else:
            import coremltools as ct

            units = getattr(ct.ComputeUnit, _UNITS[compute_units()])
            self.model = ct.models.CompiledMLModel(str(compiled_model(self.info.path)), units)
        self.load_seconds = time.perf_counter() - t0
        log.info("CoreML: %s loaded in %.1f s (%s, %d input shape(s), tail=%s)", self.label, self.load_seconds,
                 compute_units(), len(self.info.shapes), self.info.tail)

    # --- the InferenceSession surface LM3 reads ---
    def get_inputs(self) -> list[_Arg]:
        return self._inputs

    def get_outputs(self) -> list[_Arg]:
        return self._outputs

    def get_modelmeta(self) -> _ModelMeta:
        return self._meta

    def get_providers(self) -> list[str]:
        return [PROVIDER]

    def _cpu_run(self, output_names, feed):
        if self._cpu is None:
            import onnxruntime as ort

            from leafmachine3.inference.providers import session_options

            self._cpu = ort.InferenceSession(self.onnx_path, session_options(), providers=["CPUExecutionProvider"])
        return self._cpu.run(output_names, feed)

    def run(self, output_names: Optional[Sequence[str]], feed: dict) -> list:
        names = list(output_names or [o.name for o in self._outputs])
        x = np.ascontiguousarray(feed[self._inputs[0].name], dtype=np.float32)
        shape = tuple(int(s) for s in x.shape)
        if shape not in self.info.shapes:
            if shape not in self._warned:
                self._warned.add(shape)
                log.warning("CoreML: %s has no input shape %s; running that input on the CPU (ONNX)",
                            self.label, "x".join(map(str, shape)))
            self.stats["cpu_fallback"].append(list(shape))
            return self._cpu_run(names, {self._inputs[0].name: x})
        with self._lock:
            t0 = time.perf_counter()
            out = self.model.predict({self.info.input_name: x})
            ms = (time.perf_counter() - t0) * 1000.0
            self.stats["steady_ms" if shape in self._seen else "first_ms"].append(ms)
            self._seen.add(shape)
        if self.info.tail == "end2end":
            parts = [np.asarray(out[n], np.float32) for n in self.info.outputs]
            n_split = len(self.info.split)
            extra = parts[2] if n_split == 3 else None
            result = {self._outputs[0].name: end2end_tail(parts[0], parts[1], extra, self.info.max_det)}
            result.update(zip((o.name for o in self._outputs[1:]), parts[n_split:]))
        else:
            result = {name: np.asarray(out[name], np.float32) for name in self.info.outputs}
        return [result[n] for n in names]


def session_for(onnx_path: str | os.PathLike) -> Optional[NativeSession]:
    """A :class:`NativeSession` for this ONNX model's package when CoreML is available and the package
    is installed and readable; None otherwise (the caller runs the ONNX model on the CPU)."""
    if not available():
        return None
    pkg = package_for(onnx_path)
    if pkg is None:
        log.info("CoreML: no %s package beside %s; it runs on the CPU", FORMAT, onnx_path)
        return None
    try:
        return NativeSession(pkg, onnx_path)
    except CoremlPackageError as exc:
        log.warning("CoreML: %s; %s runs on the CPU", exc, os.path.basename(str(onnx_path)))
        return None
