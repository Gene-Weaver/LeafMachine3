"""Apple GPU (Core ML) execution on macOS -- the one place LM3 decides which models run on Core ML.

On a Mac, a model runs on the Apple GPU from its ``coreml_fp32`` package, with Core ML itself
(``coreml_native.py``): float32 throughout, the CPU path's exact input tensor, every input shape LM3
uses in one compiled model. A model without an installed package runs its ONNX file on the CPU.

**Same answers as every other platform.** A Mac must produce what Linux and Windows produce, to
float32 rounding:

* *The input tensor is the CPU's.* A YOLO model gets exactly the letterbox it gets everywhere (long
  side to ``imgsz``, the short side padded only to the next multiple of 32), so input shapes vary with
  each image's aspect ratio, and the package lists every one of them. Padding differently is not a
  numerical no-op -- square padding vs this, CPU vs CPU on the 23 example sheets, measured 237/258
  leaves matched, leaf area moved 7.2% at the 95th percentile and 2 sheets changed ruler unit (CUDA vs
  CPU: 258/258, 0.01%, none).
* *float32 throughout.* Packages are float32 ML Programs run on ``CPUAndGPU``. The Neural Engine
  computes only in float16, so ``LM3_COREML_UNITS=ALL`` is a speed probe that gives different
  answers, never a setting for real work.

onnxruntime's own CoreML execution provider is not used: it returned wrong boxes from the YOLO26
end2end tail, and compiling a session per input shape (~1.9 s and ~1.1 GB each, M2 Pro) made a run
slower than the CPU. :func:`providers` still names ``CoreMLExecutionProvider``, but only as the marker
that tells the inference wrappers to ask this module for a session; no onnxruntime session is ever
created with it.

Which models go to Core ML is decided here too: the YOLO26n and DINOv2 ruler members were no faster
on the Apple GPU than on the CPU, so they stay on the CPU (:data:`COREML_RULER_MEMBERS`); the UNet++
specimen models and every YOLO26 end2end export with a package go to Core ML; anything else (BiRefNet)
stays on the CPU.

Environment switches (testing and comparison only; the defaults are what a Mac user gets):

* ``LM3_COREML=0`` -- never use Core ML; GPU stages plan CPU workers as on any other machine.
* ``LM3_COREML_UNITS`` -- ``CPUAndGPU`` (default) | ``CPUOnly`` | ``ALL`` (see ``coreml_native.py``).
* ``LM3_COREML_CACHE`` -- where compiled packages are kept (``coreml_native.py``).
* ``LM3_COREML_SIMULATE=1`` -- run this whole path where Core ML does not exist, with the ONNX model cut
  at the package's split on the CPU in place of Core ML. It lets the macOS code path be tested on
  Linux; never set it for real work.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional, Sequence

from leafmachine3.inference import coreml_native

log = logging.getLogger("leafmachine3.inference.coreml")

COREML_EP = "CoreMLExecutionProvider"
CPU_EP = "CPUExecutionProvider"
CPU_PROVIDERS = (CPU_EP,)

#: Ruler-ensemble members that run on Core ML (the others measured no faster there: x0.9 / x0.55).
COREML_RULER_MEMBERS = frozenset({"x224"})

#: The tail helpers live with the packages that need them.
end2end_tail = coreml_native.end2end_tail


def simulated() -> bool:
    """``LM3_COREML_SIMULATE=1``: the Core ML code path with the CPU underneath (testing)."""
    return coreml_native.simulated()


def available() -> bool:
    """True on a Mac that can run ``coreml_fp32`` packages (or when simulating), unless ``LM3_COREML=0``."""
    if os.environ.get("LM3_COREML", "") == "0":
        return False
    if simulated():
        return True
    return sys.platform == "darwin" and coreml_native.available()


def providers() -> list:
    """The provider list a Core ML device hands to the inference wrappers: the Core ML marker first,
    the CPU behind it (see the module docstring)."""
    return [COREML_EP, CPU_EP]


def _name(provider: Any) -> str:
    return provider[0] if isinstance(provider, (tuple, list)) else str(provider)


def is_coreml(provider_list: Optional[Sequence]) -> bool:
    """Does this provider list ask for Core ML first?"""
    return bool(provider_list) and _name(provider_list[0]) == COREML_EP


def for_model(provider_list: Optional[Sequence], on_coreml: bool) -> list:
    """``provider_list`` unchanged, except a Core ML list becomes CPU for a model kept off Core ML."""
    provs = list(provider_list) if provider_list else list(CPU_PROVIDERS)
    return provs if (on_coreml or not is_coreml(provs)) else list(CPU_PROVIDERS)


def specimen_on_coreml(workflow: str, key: str) -> bool:
    """UNet++ (verified on an M2 Pro) and YOLO26-seg run on Core ML; BiRefNet does not."""
    return workflow == "ultralytics_seg" or key.startswith("unetpp_")


def yolo_session(model_path: str, provider_list: Sequence):
    """The Core ML session for a YOLO26 model on a Core ML device, or None: the caller then runs the
    ONNX model on the CPU."""
    return coreml_native.session_for(model_path)


def static_session(model_path: str, provider_list: Sequence, shape: Sequence[int]):
    """The Core ML session for a fixed-input model (ruler member, UNet++), or None (CPU)."""
    return coreml_native.session_for(model_path)


def is_coreml_session(sess: Any) -> bool:
    """Is this session a Core ML stand-in for an onnxruntime session?"""
    return isinstance(sess, coreml_native.NativeSession)
