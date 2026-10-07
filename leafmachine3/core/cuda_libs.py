"""Where the nvidia-* wheels put the CUDA runtime libraries onnxruntime-gpu needs.

``nvidia`` is a NAMESPACE shared by every nvidia-* wheel. Several of those wheels also ship an
``nvidia/__init__.py`` and list it in their own RECORD, so uninstalling ANY of them deletes the file
for all of them. After that ``nvidia.__file__`` is ``None``, and a lookup built on it finds nothing:
the CUDA library dirs never reach ``LD_LIBRARY_PATH``, onnxruntime cannot load its CUDA provider,
and every ONNX stage silently runs on the CPU.

That happened on 2026-10-07 simply by switching a lock-built environment from the development
variant back to production (``uv sync --extra gpu --group full`` then ``uv sync --extra gpu``: uv
removed torch's extra nvidia wheels and with them the shared ``__init__.py``); ``lm3 doctor`` check
6 caught it. ``__path__`` is defined for regular and namespace packages alike, so it survives.
"""
from __future__ import annotations

from pathlib import Path


def nvidia_lib_dirs() -> list[str]:
    """Sorted ``lib/`` directories of the installed nvidia-* wheels (empty if there are none)."""
    try:
        import nvidia  # type: ignore
    except Exception:  # noqa: BLE001 - no nvidia wheels installed (cpu / macos variants)
        return []
    roots = [Path(p) for p in getattr(nvidia, "__path__", ()) or ()]
    return sorted({str(d) for root in roots for d in root.glob("*/lib") if d.is_dir()})
