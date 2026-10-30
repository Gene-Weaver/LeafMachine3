"""Apple GPU (CoreML) execution on macOS -- the one place LM3 decides how a model runs on CoreML.

Two routes (``LM3_COREML_ROUTE``):

* ``native`` (default): the model's ``coreml_fp32`` package, run by Core ML itself
  (``coreml_native.py``): float32, every LM3 input shape in one compiled model. A model without an
  installed package runs on the CPU.
* ``ort``: onnxruntime's CoreML execution provider on the ONNX file -- the first attempt, kept until the
  native route is verified on a Mac, then removed. The rest of this docstring describes it.

Three facts shape the ``ort`` route. The first is read from the onnxruntime source; the other two were measured on
an M2 Pro with onnxruntime 1.20.1 (2026-10-29):

1. **Options.** onnxruntime 1.20.1's Python binding reads ONE CoreML option, ``flags``: a string of
   ``COREML_FLAG_*`` names. Named keys such as ``ModelFormat`` are silently ignored. 1.21+ still reads
   ``flags`` first, so this module only ever passes ``flags`` (:func:`provider_options`).
2. **Static shapes.** CoreML fails on LM3's graphs with dynamic H/W ("Error in building plan" --
   reported with option keys 1.20.1 ignores, so in its default NeuralNetwork format; MLProgram with
   dynamic shapes is untested, see ``LM3_COREML_DYNAMIC``). Every session built here pins each
   symbolic input dimension to the one shape it will see (free-dimension overrides).
3. **The YOLO26 end2end tail.** CoreML returns the right scores but wrong boxes (duplicated, or x ~ 0)
   from the top-k / integer-index tail of the end2end exports (Split -> ReduceMax -> TopK ->
   GatherElements -> TopK -> Div/Mod -> Gather -> GatherElements -> Concat), on the GPU, on CoreML's CPU
   and in both model formats. :class:`End2EndSession` cuts the graph at that Split, runs the network on
   CoreML and the tail in numpy (:func:`end2end_tail`). On the CPU provider the numpy tail reproduces
   the full graph exactly (tests/test_coreml.py), so the cut changes where the tail runs, not what it
   computes.

**Same answers as every other platform.** A Mac must produce what Linux and Windows produce, to
float32 rounding. Two things guarantee the inputs and the arithmetic:

* *The input tensor is the CPU's.* A YOLO model gets exactly the letterbox it gets everywhere (long
  side to ``imgsz``, the short side padded only to the next multiple of 32), so input shapes vary with
  each image's aspect ratio. Padding differently is not a numerical no-op -- square padding vs this,
  CPU vs CPU on the 23 example sheets, measured 237/258 leaves matched, leaf area moved 7.2% at the 95th
  percentile and 2 sheets changed ruler unit (CUDA vs CPU: 258/258, 0.01%, none) -- so CoreML never
  pads differently. Varying shapes cost CoreML compile time or memory instead: per-shape sessions
  (default) or one dynamic session (``LM3_COREML_DYNAMIC``).
* *float32 throughout.* The default -- an ML Program on ``CPUAndGPU`` -- keeps the graph's float32
  types. The Neural Engine computes only in float16, and the NeuralNetwork format may use float16 on
  the GPU, so ``LM3_COREML_UNITS=ALL`` and ``LM3_COREML_FORMAT=NeuralNetwork`` are speed probes that
  give different answers, never settings for real work.

Which models go to CoreML is decided here too: the YOLO26n and DINOv2 ruler members were no faster
on CoreML than on the CPU, so they stay on the CPU (:data:`COREML_RULER_MEMBERS`); the UNet++ specimen
models and every YOLO26 end2end export go to CoreML; anything else (BiRefNet) stays on the CPU.

Environment switches (testing and comparison only; the defaults are what a Mac user gets):

* ``LM3_COREML=0`` -- never use CoreML; GPU stages plan CPU workers as before.
* ``LM3_COREML_DYNAMIC=1`` -- one CoreML session per model that takes any input shape (no pinned
  H/W, no static-shapes flag) instead of one per shape.
* ``LM3_COREML_MAX_SESSIONS`` -- per-shape sessions kept per model before the least recently used is
  dropped (default 8). Each holds a compiled copy of the network (~240 MB of float32 weights for a
  YOLO26x model); a model sees at most ``2 * imgsz / 32 - 1`` shapes (79 at 1280, 63 at 1024, 39 at
  640), and crops arrive in no useful order, so a cache smaller than that recompiles: on the 23 example
  sheets the leaf segmenter saw 40 shapes and rebuilt 295 sessions with 8 kept, 55 with 32, 40 with 48.
* ``LM3_COREML_UNITS`` -- ``CPUAndGPU`` (default) | ``ALL`` (adds the Neural Engine) | ``CPUOnly``.
* ``LM3_COREML_FORMAT`` -- ``MLProgram`` (default) | ``NeuralNetwork``.
* ``LM3_COREML_SIMULATE=1`` -- run this whole path where CoreML does not exist: everything is the same
  except that the provider handed to onnxruntime is the CPU one. It lets the macOS code path be tested
  on Linux (tools/macos/README.md); never set it for real work.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

log = logging.getLogger("leafmachine3.inference.coreml")

COREML_EP = "CoreMLExecutionProvider"
CPU_EP = "CPUExecutionProvider"
CPU_PROVIDERS = (CPU_EP,)

#: Ruler-ensemble members that run on CoreML (the others measured no faster there: x0.9 / x0.55).
COREML_RULER_MEMBERS = frozenset({"x224"})

_UNIT_FLAGS = {"CPUAndGPU": "COREML_FLAG_USE_CPU_AND_GPU", "CPUOnly": "COREML_FLAG_USE_CPU_ONLY", "ALL": ""}


class CoremlUnsupported(RuntimeError):
    """This model cannot take the CoreML path; the caller runs it on the CPU instead."""


# --------------------------------------------------------------------------------------------------
# Availability and providers
# --------------------------------------------------------------------------------------------------
def simulated() -> bool:
    """``LM3_COREML_SIMULATE=1``: the CoreML code path with the CPU provider underneath (testing)."""
    return os.environ.get("LM3_COREML_SIMULATE", "") == "1"


def route() -> str:
    """``native`` (default) or ``ort`` -- see the module docstring."""
    r = os.environ.get("LM3_COREML_ROUTE", "native").strip().lower()
    if r not in ("native", "ort"):
        log.warning("LM3_COREML_ROUTE=%r is not native or ort; using native", r)
        return "native"
    return r


def available() -> bool:
    """True on a Mac that can run the selected route (or when simulating), unless ``LM3_COREML=0``."""
    if os.environ.get("LM3_COREML", "") == "0":
        return False
    if simulated():
        return True
    if sys.platform != "darwin":
        return False
    if route() == "native":
        from leafmachine3.inference import coreml_native

        return coreml_native.available()
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001 - ORT optional at import time
        return False
    return COREML_EP in ort.get_available_providers()


def provider_options() -> dict:
    """CoreML options in the ``flags`` form that onnxruntime 1.20.1 reads (see module docstring)."""
    units = os.environ.get("LM3_COREML_UNITS", "CPUAndGPU")
    if units not in _UNIT_FLAGS:
        log.warning("LM3_COREML_UNITS=%r is not one of %s; using CPUAndGPU", units, sorted(_UNIT_FLAGS))
        units = "CPUAndGPU"
    flags = [] if dynamic() else ["COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES"]
    if os.environ.get("LM3_COREML_FORMAT", "MLProgram") != "NeuralNetwork":
        flags.append("COREML_FLAG_CREATE_MLPROGRAM")
    if _UNIT_FLAGS[units]:
        flags.append(_UNIT_FLAGS[units])
    return {"flags": "|".join(flags)}


def providers() -> list:
    """The provider list a CoreML device hands to the inference wrappers: CoreML, CPU behind it."""
    return [(COREML_EP, provider_options()), CPU_EP]


def _name(provider: Any) -> str:
    return provider[0] if isinstance(provider, (tuple, list)) else str(provider)


def is_coreml(provider_list: Optional[Sequence]) -> bool:
    """Does this provider list ask for CoreML first?"""
    return bool(provider_list) and _name(provider_list[0]) == COREML_EP


def for_model(provider_list: Optional[Sequence], on_coreml: bool) -> list:
    """``provider_list`` unchanged, except a CoreML list becomes CPU for a model kept off CoreML."""
    provs = list(provider_list) if provider_list else list(CPU_PROVIDERS)
    return provs if (on_coreml or not is_coreml(provs)) else list(CPU_PROVIDERS)


def specimen_on_coreml(workflow: str, key: str) -> bool:
    """UNet++ (verified on an M2 Pro) and YOLO26-seg (via the cut tail) run on CoreML; BiRefNet does not."""
    return workflow == "ultralytics_seg" or key.startswith("unetpp_")


def dynamic() -> bool:
    """``LM3_COREML_DYNAMIC=1``: one CoreML session per model for every input shape."""
    return os.environ.get("LM3_COREML_DYNAMIC", "") == "1"


def _rss_gb() -> Optional[float]:
    """This process's resident memory in GB, for the session log (what a cached session costs)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:  # noqa: BLE001 - logging only
        return None


# --------------------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------------------
def _onnx():
    try:
        import onnx
    except ImportError as exc:   # the macos extra installs it; anything else should not get here
        raise CoremlUnsupported("the CoreML path needs the `onnx` package (uv sync --frozen --extra macos)") from exc
    return onnx


def input_dim_overrides(model: Any, shape: Sequence[int]) -> dict[str, int]:
    """Map each symbolic dim of the first graph input to its value in ``shape`` (e.g. height -> 1280)."""
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if len(dims) != len(shape):
        raise ValueError(f"input has {len(dims)} dims, shape {tuple(shape)} has {len(shape)}")
    return {d.dim_param: int(s) for d, s in zip(dims, shape) if d.dim_param}


def make_session(model: Any, provider_list: Sequence, dim_overrides: Optional[dict] = None):
    """``ort.InferenceSession`` for ``model`` (path or serialized bytes) with every symbolic input dim
    pinned. Under ``LM3_COREML_SIMULATE`` the CoreML entry is replaced by the CPU provider."""
    import onnxruntime as ort

    from leafmachine3.inference.providers import session_options

    so = session_options()
    for name, value in (dim_overrides or {}).items():
        so.add_free_dimension_override_by_name(name, int(value))
    provs = list(provider_list)
    if simulated():
        provs = [p for p in provs if _name(p) != COREML_EP] or [CPU_EP]
    return ort.InferenceSession(model if isinstance(model, bytes) else str(model), so, providers=provs)


def make_static_session(model_path: str, provider_list: Sequence, shape: Sequence[int]):
    """A session for a single-input model that always sees ``shape`` (ruler member, UNet++)."""
    model = _onnx().load(str(model_path))
    overrides = input_dim_overrides(model, shape)
    del model
    sess = make_session(model_path, provider_list, overrides)
    _warn_if_not_coreml(sess, provider_list, _label(model_path))
    return sess


def yolo_session(model_path: str, provider_list: Sequence):
    """The CoreML session for a YOLO26 end2end model on a CoreML device, or None: the caller then runs
    the ONNX model on the CPU."""
    if route() == "native":
        from leafmachine3.inference import coreml_native

        return coreml_native.session_for(model_path)
    try:
        return End2EndSession(model_path, provider_list)
    except CoremlUnsupported as exc:
        log.warning("CoreML: %s cannot run on CoreML (%s); using the CPU", _label(model_path), exc)
        return None


def static_session(model_path: str, provider_list: Sequence, shape: Sequence[int]):
    """The CoreML session for a fixed-input model (ruler member, UNet++), or None (CPU)."""
    if route() == "native":
        from leafmachine3.inference import coreml_native

        return coreml_native.session_for(model_path)
    return make_static_session(model_path, provider_list, shape)


def is_coreml_session(sess: Any) -> bool:
    """Is this session one of the CoreML stand-ins (either route)?"""
    from leafmachine3.inference import coreml_native

    return isinstance(sess, (End2EndSession, coreml_native.NativeSession))


def _label(model_path) -> str:
    """``plant_detector/model.onnx`` -- every LM3 model file is called model.onnx."""
    p = os.path.normpath(str(model_path))
    return os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p))


def _warn_if_not_coreml(sess, provider_list: Sequence, label: str) -> None:
    if is_coreml(provider_list) and not simulated() and sess.get_providers()[0] != COREML_EP:
        log.warning("CoreML requested for %s but onnxruntime bound %s", label, sess.get_providers()[0])


# --------------------------------------------------------------------------------------------------
# The YOLO26 end2end tail
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TailSpec:
    """Where an end2end graph is cut and what the numpy tail needs to rebuild ``output0``."""
    split_outputs: tuple[str, ...]      # boxes (b, n, 4), scores (b, n, nc)[, extra (b, n, e)]
    split_sizes: tuple[int, ...]        # (4, nc[, e])
    max_det: int                        # k of both TopKs (300 in every LM3 export)
    output0: str                        # the graph output the tail builds
    passthrough: tuple[str, ...]        # other graph outputs (the segmenter's protos)


def _const(model: Any, name: str) -> Optional[np.ndarray]:
    onnx = _onnx()
    for init in model.graph.initializer:
        if init.name == name:
            return onnx.numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type == "Constant" and name in node.output:
            for attr in node.attribute:
                if attr.name == "value":
                    return onnx.numpy_helper.to_array(attr.t)
    return None


def find_end2end_tail(model: Any) -> TailSpec:
    """Locate the end2end tail of an Ultralytics YOLO26 export, or raise :class:`CoremlUnsupported`.

    The tail starts at the last ``Split`` whose second output (the class scores) feeds a ``ReduceMax``
    that feeds a ``TopK``. Its sizes must add up to the width of ``output0`` plus the two columns the
    tail adds (score, class).
    """
    nodes = list(model.graph.node)
    consumers: dict[str, list] = {}
    for n in nodes:
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    def feeds(tensor: str, op: str) -> list:
        return [n for n in consumers.get(tensor, []) if n.op_type == op]

    split = None
    for n in reversed(nodes):
        if n.op_type == "Split" and len(n.output) in (2, 3):
            if any(feeds(rm.output[0], "TopK") for rm in feeds(n.output[1], "ReduceMax")):
                split = n
                break
    if split is None:
        raise CoremlUnsupported("no YOLO26 end2end top-k tail found")
    sizes = _const(model, split.input[1]) if len(split.input) > 1 else None
    if sizes is None or len(sizes) != len(split.output) or int(sizes[0]) != 4:
        raise CoremlUnsupported(f"unexpected end2end Split sizes {sizes}")
    sizes = tuple(int(s) for s in sizes)

    ks = [_const(model, n.input[1]) for n in nodes if n.op_type == "TopK" and len(n.input) > 1]
    ks = {int(np.asarray(k).reshape(-1)[0]) for k in ks if k is not None}
    if len(ks) != 1:
        raise CoremlUnsupported(f"expected one top-k size in the tail, found {sorted(ks)}")
    max_det = ks.pop()

    outputs = [o.name for o in model.graph.output]
    out0 = outputs[0]
    width = model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value
    expected = 4 + 2 + (sizes[2] if len(sizes) == 3 else 0)
    if width and width != expected:
        raise CoremlUnsupported(f"{out0} has width {width}; the tail would build {expected}")
    return TailSpec(tuple(split.output), sizes, max_det, out0, tuple(outputs[1:]))


# The numpy tail: one implementation, shared with the native route.
from leafmachine3.inference.coreml_native import _topk, end2end_tail  # noqa: E402,F401


@dataclass(frozen=True)
class _Arg:
    """The slice of onnxruntime's ``NodeArg`` that ``ultra_rep.YOLO`` reads."""
    name: str
    shape: list
    type: str


@dataclass(frozen=True)
class _ModelMeta:
    custom_metadata_map: dict


_ELEM_TYPES = {1: "tensor(float)", 10: "tensor(float16)", 7: "tensor(int64)"}


def _args(values) -> list[_Arg]:
    return [_Arg(v.name, [d.dim_param or d.dim_value for d in v.type.tensor_type.shape.dim],
                 _ELEM_TYPES.get(v.type.tensor_type.elem_type, "tensor(float)")) for v in values]


class End2EndSession:
    """Stands in for ``ort.InferenceSession`` on an end2end YOLO26 graph headed for CoreML.

    The network up to the tail ``Split`` runs in a CoreML session built for the exact input shape
    (one per shape, least recently used dropped past ``LM3_COREML_MAX_SESSIONS``; with
    ``LM3_COREML_DYNAMIC`` one for all); the tail runs in numpy.
    It exposes what ``ultra_rep.YOLO`` uses: ``get_inputs / get_outputs / get_modelmeta /
    get_providers / run``. If CoreML cannot build a session for a shape, that shape runs the same cut
    graph on the CPU (with a warning), which is still exact.
    """

    def __init__(self, model_path: str, provider_list: Sequence) -> None:
        onnx = _onnx()
        from onnx.utils import Extractor

        model = onnx.load(str(model_path))
        self.tail = find_end2end_tail(model)
        self.label = _label(model_path)
        self._inputs = _args(model.graph.input)
        self._outputs = _args(model.graph.output)
        self._meta = _ModelMeta({p.key: p.value for p in model.metadata_props})
        self._dim_params = [d.dim_param for d in model.graph.input[0].type.tensor_type.shape.dim]
        # The Extractor needs a type for each tensor it cuts at; an export without stored shapes
        # gets an untyped-shape declaration in the input's element type.
        known = {v.name for v in model.graph.value_info} | {o.name for o in model.graph.output}
        elem = model.graph.input[0].type.tensor_type.elem_type
        for name in self.tail.split_outputs:
            if name not in known:
                model.graph.value_info.append(onnx.helper.make_tensor_value_info(name, elem, None))
        cut = Extractor(model).extract_model([i.name for i in model.graph.input],
                                             list(self.tail.split_outputs) + list(self.tail.passthrough))
        self._cut = cut.SerializeToString()
        del cut
        self._providers = list(provider_list)
        self._sessions: OrderedDict[tuple, Any] = OrderedDict()   # least recently used first
        self._max_sessions = max(1, int(os.environ.get("LM3_COREML_MAX_SESSIONS", "8") or 8))
        self.n_built = 0
        self._bound: list[str] = [_name(p) for p in self._providers]
        self._lock = threading.Lock()
        del model
        log.info("CoreML: %s cut before its end2end tail (%d outputs, top-k %d); the tail runs in numpy",
                 self.label, len(self.tail.split_outputs), self.tail.max_det)

    # --- the InferenceSession surface ultra_rep reads ---
    def get_inputs(self) -> list[_Arg]:
        return self._inputs

    def get_outputs(self) -> list[_Arg]:
        return self._outputs

    def get_modelmeta(self) -> _ModelMeta:
        return self._meta

    def get_providers(self) -> list[str]:
        return list(self._bound)

    def _session_for(self, shape: tuple) -> Any:
        key = (shape[0],) if dynamic() else shape      # dynamic: one session for every H/W
        with self._lock:
            sess = self._sessions.get(key)
            if sess is not None:
                self._sessions.move_to_end(key)
                return sess
            overrides = {p: int(s) for p, s in zip(self._dim_params, key) if p}
            t0 = time.perf_counter()
            try:
                sess = make_session(self._cut, self._providers, overrides)
                _warn_if_not_coreml(sess, self._providers, self.label)
            except Exception as exc:  # noqa: BLE001 - a CoreML build failure must not lose the item
                log.warning("CoreML could not build %s for input %s (%s: %s); running that shape on the CPU",
                            self.label, shape, type(exc).__name__, exc)
                sess = make_session(self._cut, CPU_PROVIDERS, overrides)
            self.n_built += 1
            rss = _rss_gb()
            log.info("CoreML: %s session #%d for input %s bound %s in %.1f s (%d alive, process RSS %s)",
                     self.label, self.n_built, "dynamic" if dynamic() else "x".join(map(str, shape)),
                     sess.get_providers()[0], time.perf_counter() - t0, len(self._sessions) + 1,
                     f"{rss:.1f} GB" if rss is not None else "?")
            self._bound = sess.get_providers()
            self._sessions[key] = sess
            while len(self._sessions) > self._max_sessions:
                old, _ = self._sessions.popitem(last=False)
                log.info("CoreML: %s dropped the session for input %s (LM3_COREML_MAX_SESSIONS=%d)",
                         self.label, "x".join(map(str, old)), self._max_sessions)
            return sess

    def run(self, output_names: Optional[Sequence[str]], feed: dict) -> list:
        x = feed[self._inputs[0].name]
        outs = self._session_for(tuple(int(s) for s in x.shape)).run(None, feed)
        n = len(self.tail.split_outputs)
        extra = outs[2] if n == 3 else None
        result = {self.tail.output0: end2end_tail(outs[0], outs[1], extra, self.tail.max_det)}
        result.update(zip(self.tail.passthrough, outs[n:]))
        return [result[name] for name in (output_names or [o.name for o in self._outputs])]
