"""Leaf-landmark pose inference (yolo26x-pose, 31-kpt mid15_pet5).

Wraps the exported pose model via Ultralytics AutoBackend. The model is trained on leaf crops
with a WHITE_PAD_FRAC white border, so ``predict`` re-adds that border, runs inference, then
subtracts the offset → keypoints come back in the INPUT crop's pixel frame (as if the padding
were never added). Feed it a plant-detector leaf crop, NOT a raw herbarium sheet.

``predict`` returns a list of leaf instances; each is ``{kpt_name: (x, y, conf)}`` in the crop's
pixel coordinates. Keypoint names/order come from :mod:`leafmachine3.core.landmarks`.
"""
from __future__ import annotations

import os

from leafmachine3.core.landmarks import KPT_NAMES, WHITE_PAD_FRAC


class LeafLandmarkPose:
    """Exported yolo26x-pose backend that returns per-leaf named keypoints (crop frame)."""

    def __init__(self, model_path, conf: float = 0.25, iou: float = 0.45, imgsz=640,
                 device=None, white_pad: float = WHITE_PAD_FRAC):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"landmark pose model not found: {self.model_path}")
        self.model = YOLO(self.model_path, task="pose")
        self.conf, self.iou, self.imgsz, self.device = conf, iou, imgsz, device
        self.white_pad = float(white_pad)

    @staticmethod
    def _pad_white(img, frac):
        """Add a white border of ``frac`` of each dim per side. Returns (padded, px, py)."""
        import cv2

        h, w = img.shape[:2]
        px, py = round(w * frac), round(h * frac)
        out = cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return out, px, py

    def predict(self, source):
        """Detect leaf keypoints. Returns ``list[{kpt_name: (x, y, conf)}]`` in crop coords."""
        import cv2
        import numpy as np

        img = source if isinstance(source, np.ndarray) else cv2.imread(str(source))
        if img is None:
            raise FileNotFoundError(f"cannot read leaf crop: {source}")
        padded, px, py = self._pad_white(img, self.white_pad)

        kw = dict(conf=self.conf, iou=self.iou, verbose=False)
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        if self.device is not None:
            kw["device"] = self.device
        res = self.model.predict(padded, **kw)
        if not res:
            return []
        kp = getattr(res[0], "keypoints", None)
        if kp is None or kp.data is None:
            return []
        data = kp.data.cpu().numpy()                      # (n_leaves, 31, 3): x, y, conf (padded frame)
        leaves = []
        for inst in data:
            leaf = {}
            for i, (x, y, c) in enumerate(inst):
                if c > 0 and (x > 0 or y > 0):            # skip not-visible keypoints
                    leaf[KPT_NAMES[i]] = (float(x) - px, float(y) - py, float(c))   # -> crop frame
            leaves.append(leaf)
        return leaves
