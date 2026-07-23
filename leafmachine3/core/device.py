"""Execution-device descriptor shared by the executor, stages, and inference wrappers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    """One execution slot. Workers hard-pin ``CUDA_VISIBLE_DEVICES`` to ``index``, so inside a
    worker the chosen GPU is always ordinal 0 (hence ``torch_str`` / ort device_id 0)."""
    kind: str = "cpu"      # "cuda" | "cpu"
    index: int = 0         # physical GPU ordinal (ignored for cpu)

    @property
    def is_cuda(self) -> bool:
        return self.kind == "cuda"

    @property
    def torch_str(self) -> str:
        return "cuda:0" if self.is_cuda else "cpu"

    def ort_providers(self) -> list:
        if self.is_cuda:
            return [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
