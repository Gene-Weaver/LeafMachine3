"""Real instance-segmentation backend — an exported YOLO26-seg model via Ultralytics.

``YoloSegmenter`` runs an exported segmentation artifact through ``YOLO(..., task=
"segment")`` and returns :class:`Instance` objects carrying a polygon in **input-image
(crop) coordinates**. Leaf-segmenter stages then re-base the polygon to the parent/
working frame with ``offset_polygon``.

Masks are built from the **raster** ``results.masks.data``, NOT ``results.masks.xy``.
``masks.xy`` runs Ultralytics' ``masks2segments(strategy="all")``, which concatenates a
single instance's disconnected mask contours into ONE polygon ring — so when a mask has
islands (a torn fragment, a neighbouring leaf poking in) the ring bridges them, drawing
thin "stringing". Working from the raster avoids that entirely. Per-class reduction
(``build_instances``): a class named ``Hole`` keeps ALL of its components; every other
class (``Leaf``, ``Petiole``) keeps only its single LARGEST connected component.

Ultralytics is imported lazily so this module imports cleanly without it (mock path).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import cv2
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


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Boolean mask of the single largest connected component (empty in -> empty out)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros(mask.shape, dtype=bool)
    return lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def _components(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    """Every connected component with area >= ``min_area`` px, as boolean masks."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [lab == i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def _outer_polygons(mask: np.ndarray) -> list[np.ndarray]:
    """Outer (RETR_EXTERNAL) contours of ``mask`` as Nx2 float [x, y] arrays."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        arr = c.reshape(-1, 2).astype(float)
        if len(arr) >= 3:
            out.append(arr)
    return out


def build_instances(data, cls_ids, confs, class_names, hw, min_area_px: int = 64) -> list[Instance]:
    """Reduce YOLO raster instance masks (``results.masks.data``) to clean Instances.

    Avoids the ``masks.xy`` stringing artifact (see module docstring). Per class:
      * a class named ``Hole`` -> keep ALL components (only sub-``min_area_px`` speckle dropped);
      * every other class (``Leaf``, ``Petiole``) -> keep only its single LARGEST component.
    Detections of the same class are unioned first, then reduced; each emitted polygon is
    an outer contour in crop coordinates. Instances are ordered Leaf -> other -> Hole so the
    stage's Hole/Petiole->Leaf attachment sees the Leaf first. ``data`` is ``[n, mh, mw]``;
    each mask is nearest-resized to ``hw`` when the model returned proto-resolution masks.
    """
    h, w = hw
    acc: dict[int, dict] = {}
    for i in range(len(cls_ids)):
        c = int(cls_ids[i])
        m = (np.asarray(data[i]) > 0.5).astype(np.uint8)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        d = acc.setdefault(c, {"name": class_names.get(c, str(c)),
                               "mask": np.zeros((h, w), bool), "conf": 0.0})
        d["mask"] |= m.astype(bool)
        d["conf"] = max(d["conf"], float(confs[i]))

    def rank(cid: int) -> int:
        nm = acc[cid]["name"].lower()
        return 0 if nm == "leaf" else (2 if nm == "hole" else 1)

    out: list[Instance] = []
    for cid in sorted(acc, key=rank):
        d = acc[cid]
        if d["name"].lower() == "hole":
            comps = _components(d["mask"], min_area_px)
        else:
            comps = [_largest_component(d["mask"])] if d["mask"].any() else []
        for comp in comps:
            for poly in _outer_polygons(comp):
                out.append(Instance(cls_id=cid, cls_name=d["name"], conf=d["conf"], polygon=poly))
    return out


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
        min_area_px: int = 64,
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
        self.min_area_px = int(min_area_px)
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
        # Raster masks (masks.data), NOT masks.xy -> no polygon-concatenation stringing.
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or boxes is None or getattr(masks, "data", None) is None or len(boxes) == 0:
            return []
        data = masks.data.cpu().numpy()                 # [n, mh, mw] raster instance masks
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        h, w = result.orig_shape                        # native crop size (retina_masks -> native)
        return build_instances(data, clss, confs, self.class_names, (h, w), self.min_area_px)

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
