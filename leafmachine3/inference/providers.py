"""ONNXRuntime execution-provider selection — GPU-first, fail-loud.

NVIDIA CUDA (or TensorRT) is chosen whenever its execution provider (EP) is genuinely
available; a non-CUDA GPU path is next; CPU is used ONLY when the machine exposes no
usable accelerator. A GPU that is *present* but whose EP fails to bind is treated as a
hard configuration error (unless the user opts into CPU fallback) rather than a silent
slow CPU run.

Heavy imports (``onnxruntime``) are deferred so this module imports cleanly on a host
that only ever runs the mock backends.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Any

log = logging.getLogger("leafmachine3.inference.providers")


def _available_providers() -> set[str]:
    """Return the set of EPs this onnxruntime build exposes (empty if ORT is absent)."""
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001 - ORT optional at import time
        return set()
    return set(ort.get_available_providers())


def _get(section: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a Section/dict/attr-bearing object, tolerating any of them."""
    if section is None:
        return default
    if hasattr(section, "get"):
        try:
            return section.get(key, default)
        except Exception:  # noqa: BLE001
            pass
    return getattr(section, key, default)


def _tensorrt_enabled(cfg: Any) -> bool:
    return bool(_get(cfg.compute, "tensorrt", False))


def _allow_cpu_fallback(cfg: Any) -> bool:
    """CPU fallback flag lives under ``compute.onnxruntime`` (with a top-level alias)."""
    ort_cfg = _get(cfg.compute, "onnxruntime", None)
    if ort_cfg is not None:
        val = _get(ort_cfg, "allow_cpu_fallback", None)
        if val is not None:
            return bool(val)
    return bool(_get(cfg.compute, "allow_cpu_fallback", False))


def nvidia_gpu_present() -> bool:
    """Probe the *hardware* for an NVIDIA GPU, independent of the installed ORT build.

    Uses ``pynvml`` when importable, else falls back to invoking ``nvidia-smi``. Any
    failure is interpreted as "no GPU" so CPU-only hosts never trip the fail-loud path.
    """
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        try:
            return pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # noqa: BLE001 - pynvml optional / may fail on CPU host
        pass

    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        out = subprocess.run(
            [smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return False


#: onnxruntime's session-creation log level for LM3's own sessions: 3 = ERROR.
#:
#: At its default (WARNING) onnxruntime prints, once per session and so once per worker, how it
#: placed the graph -- "N Memcpy nodes are added to the graph", "Some nodes were not assigned to the
#: preferred execution providers". For LM3's models those were diagnosed on 2026-10-07: the real CPU
#: fallback (opset-19 Resize in the YOLO exports) is fixed by tools/modelhub/fix_resize_opset.py and
#: checked by `lm3 doctor --models`; what remains is int64 shape arithmetic onnxruntime keeps on the
#: CPU by design. Errors still print -- "Failed to load library", "Failed to create
#: CUDAExecutionProvider" -- and a session that falls back to the CPU entirely is refused by
#: make_session / reported by the doctor. Set LM3_ORT_LOG_SEVERITY (0 verbose .. 4 fatal) to see
#: placements again when debugging a model.
SESSION_LOG_SEVERITY = 3


def session_options():
    """``onnxruntime.SessionOptions`` for every LM3 inference session (see SESSION_LOG_SEVERITY)."""
    import os

    import onnxruntime as ort

    so = ort.SessionOptions()
    try:
        so.log_severity_level = int(os.environ.get("LM3_ORT_LOG_SEVERITY", SESSION_LOG_SEVERITY))
    except ValueError:
        so.log_severity_level = SESSION_LOG_SEVERITY
    return so


def build_providers(cfg: Any) -> list:
    """Build the GPU-first EP ladder for the current onnxruntime build and config."""
    avail = _available_providers()
    ladder: list = []
    # 1. NVIDIA first: TensorRT > CUDA when both are present and TRT is requested.
    if "TensorrtExecutionProvider" in avail and _tensorrt_enabled(cfg):
        ladder.append(("TensorrtExecutionProvider", {"device_id": 0}))
    if "CUDAExecutionProvider" in avail:
        ladder.append(("CUDAExecutionProvider", {"device_id": 0}))
    # 2. non-NVIDIA GPU paths (no CUDA install required).
    if "DmlExecutionProvider" in avail:
        ladder.append("DmlExecutionProvider")        # Windows: any DX12 GPU
    if "CoreMLExecutionProvider" in avail:
        ladder.append("CoreMLExecutionProvider")     # Apple GPU / ANE
    if "OpenVINOExecutionProvider" in avail:
        ladder.append("OpenVINOExecutionProvider")   # Intel iGPU / NPU
    # 3. CPU backstop — reached only when no accelerator EP is available.
    ladder.append("CPUExecutionProvider")
    return ladder


def make_session(model_path, cfg: Any):
    """Create an ``ort.InferenceSession`` bound to the best available provider.

    Raises ``RuntimeError`` when an NVIDIA GPU is present yet ORT bound the CPU provider
    (a misconfiguration) unless ``compute.onnxruntime.allow_cpu_fallback`` is set.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), session_options(), providers=build_providers(cfg))
    bound = sess.get_providers()[0]
    log.info("ONNXRuntime bound EP: %s", bound)
    if (
        bound == "CPUExecutionProvider"
        and not _allow_cpu_fallback(cfg)
        and nvidia_gpu_present()
    ):
        raise RuntimeError(
            "NVIDIA GPU detected but ONNXRuntime bound the CPU provider. Install "
            "onnxruntime-gpu and put the CUDA libs on LD_LIBRARY_PATH (the loader reads "
            "it at process start), or set compute.onnxruntime.allow_cpu_fallback: true "
            "to run on CPU anyway."
        )
    return sess
