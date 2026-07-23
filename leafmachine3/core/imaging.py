"""Image IO + polygon geometry helpers shared by modules and the Reporter.

Images are handled as BGR ``np.ndarray`` (OpenCV). Masks are stored as ``polygon_xy``
(a JSON ring of ``[x, y]`` points) in the parent/working coordinate frame; the helpers
here encode/decode, transform, measure, and rasterize them with no heavy dependencies.
"""
from __future__ import annotations

import json
import os

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - cv2 always present at runtime
    cv2 = None


# ---- image IO ------------------------------------------------------------------
def read_image(path):
    """Read an image as a BGR ``np.ndarray`` (raises if unreadable)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    return img


def save_image(img, path, quality: int = 100):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    ext = os.path.splitext(str(path))[1].lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    cv2.imwrite(str(path), img, params)
    return str(path)


def save_crop(img, xyxy, stem: str, tag: str, crops_dir) -> str:
    """Crop ``img`` to ``xyxy`` and save as ``<stem>__<TAG>__x1-y1-x2-y2.jpg`` (LM2 naming)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    os.makedirs(str(crops_dir), exist_ok=True)
    path = os.path.join(str(crops_dir), f"{stem}__{tag}__{x1}-{y1}-{x2}-{y2}.jpg")
    crop = img[y1:y2, x1:x2]
    if crop.size:
        cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 100])
    return path


# ---- polygon geometry (parent/working coords) ----------------------------------
def _pts(poly) -> np.ndarray:
    return np.asarray(poly, dtype=float).reshape(-1, 2)


def encode_polygon(poly) -> str:
    """Serialize an Nx2 ring to a compact JSON string for the ``mask_data`` column."""
    return json.dumps([[round(float(x), 2), round(float(y), 2)] for x, y in _pts(poly)])


def decode_polygon(data: str) -> np.ndarray:
    return np.asarray(json.loads(data), dtype=float).reshape(-1, 2)


def offset_polygon(poly, ox: float, oy: float) -> np.ndarray:
    return _pts(poly) + np.array([ox, oy], dtype=float)


def scale_polygon(poly, s: float) -> np.ndarray:
    return _pts(poly) * float(s)


def polygon_bbox(poly):
    p = _pts(poly)
    if len(p) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max()))


def polygon_area(poly) -> float:
    """Shoelace area in px²."""
    p = _pts(poly)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_perimeter(poly) -> float:
    p = _pts(poly)
    if len(p) < 2:
        return 0.0
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def polygon_mask(poly, shape_hw) -> np.ndarray:
    """Rasterize one ring into a boolean mask of ``shape_hw`` (H, W[, C])."""
    m = np.zeros(shape_hw[:2], dtype=np.uint8)
    p = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    if len(p) >= 3:
        cv2.fillPoly(m, [p], 1)
    return m.astype(bool)


def polygon_centroid(poly):
    p = _pts(poly)
    return (float(p[:, 0].mean()), float(p[:, 1].mean())) if len(p) else (0.0, 0.0)


def composite(img, mask, bg=0):
    """Return an image with ``img`` pixels where ``mask`` is True and ``bg`` elsewhere."""
    out = np.full_like(img, bg)
    out[mask] = img[mask]
    return out
