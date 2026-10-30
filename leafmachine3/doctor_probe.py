"""The accelerator probe `lm3 doctor` runs in a CHILD process (check 6).

Why a child: the CUDA provider is only loadable if ``LD_LIBRARY_PATH`` names the nvidia-* wheel
libraries when the process STARTS (the dynamic loader reads it once). ``machine3`` solves that by
re-exec'ing itself; this probe calls the very same function, so it binds the provider exactly the
way a pipeline run will -- including failing the same way. A doctor that tested the provider in its
own already-started process would pass where the pipeline silently falls back to the CPU.

``--coreml-models [settings]`` (macOS, check 7) runs the real models on CoreML and on the CPU and
compares them; see :func:`coreml_models`.

Run as ``python -m leafmachine3.doctor_probe <ExecutionProvider>``. The LAST line on stdout is one
JSON object: requested / available / bound provider, the max abs difference against a CPU reference
run of the same model, and any error. onnxruntime's own diagnostics go to stderr, which the doctor
captures to name the library that failed to load.
"""
from __future__ import annotations

import base64
import json
import sys

#: tools/release/make_doctor_probe.py -- Conv(3->8, 3x3) + ReLU on 1x3x32x32; IR 8, opset 17; 1078 bytes.
PROBE_ONNX_B64 = (
    "CAgSCmxtMy1kb2N0b3I6nwgKJQoBeAoBdwoBYhIBYyIEQ29udioRCgRwYWRzQAFAAUABQAGgAQcKDAoBYxIBeSIEUmVsdRIQbG0zX2RvY3Rvcl9wcm9iZSrwBggICAMIAwgDEAFCAXdK4AaamZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD6amZk+mpmZvs3MTL7NzMy9AAAAAM3MzD3NzEw+mpmZPpqZmb7NzEy+zczMvQAAAADNzMw9zcxMPpqZmT6amZm+zcxMvs3MzL0AAAAAzczMPc3MTD4qKQgIEAFCAWJKIM3MTL4lSRK++YqvvaEO6ryhDuo8+YqvPSVJEj7NzEw+WhsKAXgSFgoUCAESEAoCCAEKAggDCgIIIAoCCCBiGwoBeRIWChQIARIQCgIIAQoCCAgKAgggCgIIIEIECgAQEQ=="
)

def _provider_options(provider: str) -> dict:
    """The options LM3's own sessions pass. CoreML: the ``flags`` form, the only one onnxruntime
    1.20.1 reads (a ``{"ModelFormat": ...}`` dict is silently ignored there)."""
    if provider == "CoreMLExecutionProvider":
        from leafmachine3.inference.coreml import provider_options

        return provider_options()
    return {"device_id": 0} if provider == "CUDAExecutionProvider" else {}


def probe_model_bytes() -> bytes:
    return base64.b64decode(PROBE_ONNX_B64)


def probe_input():
    import numpy as np

    return (np.arange(3 * 32 * 32, dtype=np.float32).reshape(1, 3, 32, 32) % 13) / 13.0


def run_probe(provider: str) -> dict:
    """Bind provider the way LM3 does (requested first, CPU behind it) and run the probe model."""
    out: dict = {"requested": provider, "available": [], "bound": None, "max_abs_diff": None, "error": None}
    try:
        import onnxruntime as ort

        out["ort_version"] = ort.__version__
        out["available"] = list(ort.get_available_providers())
        model, x = probe_model_bytes(), probe_input()
        chain = ([(provider, _provider_options(provider))] if provider != "CPUExecutionProvider" else []) \
            + ["CPUExecutionProvider"]
        sess = ort.InferenceSession(model, providers=chain)
        out["bound"] = sess.get_providers()[0]
        y = sess.run(None, {"x": x})[0]
        ref = ort.InferenceSession(model, providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
        out["max_abs_diff"] = float(abs(y - ref).max())
    except Exception as exc:  # noqa: BLE001 - every failure is a finding, reported not raised
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def placement_session(model_path: str) -> None:
    """Create a CUDA session for ``model_path`` at VERBOSE onnxruntime logging and return.

    The parent (``lm3 doctor --models``) reads this process's stderr for onnxruntime's
    "CUDA kernel not found in registries for Op type: X" lines: the ops that would run on the CPU.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 0
    so.log_verbosity_level = 0
    ort.InferenceSession(model_path, so, providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                                   "CPUExecutionProvider"])


def coreml_models(settings: str | None) -> dict:
    """``lm3 doctor --models`` on a Mac: every CoreML-bound model vs its CPU reference on the packaged
    calibration sheets (leafmachine3/inference/coreml_check.py). Run in a child because CoreML compiles
    each model, which takes memory the doctor should not keep."""
    import os

    out: dict = {"results": [], "error": None}
    try:
        from leafmachine3.core.config import Config
        from leafmachine3.inference import coreml_check

        path = settings or "LM3_settings.yaml"
        if not os.path.exists(path):
            out["error"] = f"no settings file at {path} (run from the LeafMachine3 folder or pass --config)"
            return out
        out["results"] = coreml_check.check_models(Config.load(path), coreml_check.calibration_images()[:3],
                                                   log=lambda s: print(s, file=sys.stderr))
    except Exception as exc:  # noqa: BLE001 - every failure is a finding, reported not raised
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--coreml-models":
        print(json.dumps(coreml_models(args[1] if len(args) > 1 else None)))
        return 0
    placement = bool(args) and args[0] == "--placement"
    provider = "CUDAExecutionProvider" if placement else (args[0] if args else "CPUExecutionProvider")
    if provider == "CUDAExecutionProvider" and sys.platform.startswith("linux"):
        # machine3's own loader-path step. It re-execs `sys.executable + sys.argv`, so make argv the
        # `-m` form first; otherwise the child would run this file as a bare script.
        sys.argv = ["-m", "leafmachine3.doctor_probe", *args]
        from leafmachine3.machine3 import _exec_with_cuda_libpath

        _exec_with_cuda_libpath()
    if placement:
        placement_session(args[1])
        print(json.dumps({"placement": args[1]}))
        return 0
    print(json.dumps(run_probe(provider)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
