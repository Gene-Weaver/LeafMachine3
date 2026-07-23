"""Real object-detector backend — an exported YOLO26 model run through Ultralytics.

``YoloDetector`` wraps the *exported* artifact (``.onnx`` / ``.pt`` / ``.torchscript`` /
OpenVINO / CoreML) via Ultralytics' ``YOLO(..., task="detect")``, which handles the
per-backend I/O and the YOLO26 end-to-end (NMS-free) postprocess uniformly. It returns
the shared :class:`~leafmachine3.core.records.Detection` shape so detector stages persist
identical rows regardless of the export format.

Ultralytics is imported lazily inside the constructor so this module imports cleanly on a
host that only ever exercises the mock backends.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np

from leafmachine3.core.records import Detection

log = logging.getLogger("leafmachine3.inference.detector")


class YoloDetector:
    """Format-agnostic detector over an exported YOLO26 model.

    Parameters
    ----------
    model_path:
        Path to the exported detection artifact.
    class_names:
        Optional class-name list (index order). ``None`` uses the model's embedded names.
    conf, iou:
        Confidence and NMS-IoU thresholds forwarded to ``predict``.
    imgsz:
        Inference image size (long side). ``None`` uses the export's default.
    device:
        Torch device string (``"cuda:0"`` / ``"cpu"``).
    max_det:
        Maximum detections retained per image.
    """

    def __init__(
        self,
        model_path,
        class_names: Optional[Sequence[str]] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        device: Optional[str] = None,
        max_det: Optional[int] = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"detector model artifact not found: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")
        self.class_names = self._resolve_class_names(class_names)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = imgsz
        self.device = device
        self.max_det = max_det
        log.info(
            "YoloDetector loaded: %s (nc=%d, imgsz=%s, device=%s)",
            os.path.basename(self.model_path), len(self.class_names), imgsz, device,
        )

    def _resolve_class_names(self, class_names: Optional[Sequence[str]]) -> dict[int, str]:
        if class_names is not None:
            return {i: str(n) for i, n in enumerate(class_names)}
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, (list, tuple)):
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    def _predict_kwargs(self, extra: dict) -> dict:
        kw: dict[str, Any] = dict(conf=self.conf, iou=self.iou, verbose=False)
        if self.imgsz is not None:
            kw["imgsz"] = self.imgsz
        if self.device is not None:
            kw["device"] = self.device
        if self.max_det is not None:
            kw["max_det"] = self.max_det
        kw.update(extra)
        return kw

    def _to_detections(self, result) -> list[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        out: list[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            out.append(
                Detection(
                    cls_id=int(k),
                    cls_name=self.class_names.get(int(k), str(int(k))),
                    conf=float(c),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return out

    def predict(self, source, **kwargs) -> list[Detection]:
        """Detect on one image (path | BGR ``np.ndarray`` | PIL). Returns ``list[Detection]``."""
        if isinstance(source, np.ndarray):
            source = np.ascontiguousarray(source)
        results = self.model.predict(source, **self._predict_kwargs(kwargs))
        return self._to_detections(results[0]) if results else []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"YoloDetector(nc={len(self.class_names)}, conf={self.conf}, iou={self.iou}, "
            f"path={os.path.basename(self.model_path)})"
        )
