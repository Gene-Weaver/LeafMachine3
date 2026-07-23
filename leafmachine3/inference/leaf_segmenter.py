"""Real instance-segmentation backend — an exported YOLO26-seg model via Ultralytics.

``YoloSegmenter`` runs an exported segmentation artifact through ``YOLO(..., task=
"segment")`` and returns one :class:`Instance` per detected instance, each carrying its
polygon in **input-image (crop) coordinates**. Leaf-segmenter stages then re-base the
polygon to the parent/working frame with ``offset_polygon``.

Ultralytics is imported lazily so this module imports cleanly without it (mock path).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

log = logging.getLogger("leafmachine3.inference.leaf_segmenter")


@dataclass
class Instance:
    """One segmented instance in INPUT-image (crop) coordinates.

    ``polygon`` is an ``Nx2`` float array of ``[x, y]`` points.
    """
    cls_id: int
    cls_name: str
    conf: float
    polygon: np.ndarray


class YoloSegmenter:
    """Format-agnostic instance segmenter over an exported YOLO26-seg model."""

    def __init__(
        self,
        model_path,
        imgsz: Optional[int] = None,
        conf: float = 0.30,
        iou: float = 0.50,
        retina_masks: bool = True,
        device: Optional[str] = None,
        max_det: Optional[int] = None,
    ) -> None:
        from ultralytics import YOLO

        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"segmenter model artifact not found: {self.model_path}")
        self.model = YOLO(self.model_path, task="segment")
        self.imgsz = imgsz
        self.conf = float(conf)
        self.iou = float(iou)
        self.retina_masks = bool(retina_masks)
        self.device = device
        self.max_det = max_det
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.class_names = {int(k): str(v) for k, v in names.items()}
        elif isinstance(names, (list, tuple)):
            self.class_names = {i: str(n) for i, n in enumerate(names)}
        else:
            self.class_names = {}
        log.info(
            "YoloSegmenter loaded: %s (nc=%d, imgsz=%s, device=%s)",
            os.path.basename(self.model_path), len(self.class_names), imgsz, device,
        )

    def _predict_kwargs(self, extra: dict) -> dict:
        kw: dict[str, Any] = dict(
            conf=self.conf, iou=self.iou, retina_masks=self.retina_masks, verbose=False,
        )
        if self.imgsz is not None:
            kw["imgsz"] = self.imgsz
        if self.device is not None:
            kw["device"] = self.device
        if self.max_det is not None:
            kw["max_det"] = self.max_det
        kw.update(extra)
        return kw

    def _to_instances(self, result) -> list[Instance]:
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or boxes is None or masks.xy is None:
            return []
        polys = masks.xy  # list of Nx2 arrays in input-image coords
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        out: list[Instance] = []
        for poly, c, k in zip(polys, confs, clss):
            arr = np.asarray(poly, dtype=float).reshape(-1, 2)
            if len(arr) < 3:
                continue
            out.append(
                Instance(
                    cls_id=int(k),
                    cls_name=self.class_names.get(int(k), str(int(k))),
                    conf=float(c),
                    polygon=arr,
                )
            )
        return out

    def predict(self, source, **kwargs) -> list[Instance]:
        """Segment one image (path | BGR ``np.ndarray`` | PIL). Returns ``list[Instance]``."""
        if isinstance(source, np.ndarray):
            source = np.ascontiguousarray(source)
        results = self.model.predict(source, **self._predict_kwargs(kwargs))
        return self._to_instances(results[0]) if results else []

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"YoloSegmenter(nc={len(self.class_names)}, conf={self.conf}, iou={self.iou}, "
            f"path={os.path.basename(self.model_path)})"
        )
