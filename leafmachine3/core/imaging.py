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
#: The largest image, in pixels, LM3 will decode with Pillow: 1 gigapixel (e.g. 25,000 x 40,000).
#:
#: Pillow's own default (MAX_IMAGE_PIXELS = 89,478,485) is a defense for web servers decoding
#: untrusted uploads: it WARNS above that and raises DecompressionBombError above twice that
#: (178,956,970 px). Herbarium scans are routinely 100-200 MP, so with the default every sheet above
#: 89 MP printed a warning and every sheet above 179 MP raised -- which ingest caught as a decode
#: failure and quarantined as "corrupt", silently dropping the specimen from the run. The guard is
#: kept, not disabled: a crafted file claiming 10+ gigapixels is still refused.
MAX_IMAGE_PIXELS = 1_000_000_000


def configure_pillow():
    """Raise Pillow's decompression-bomb limit to :data:`MAX_IMAGE_PIXELS`; return ``PIL.Image``.

    Every module that decodes ORIGINAL images with Pillow imports Image through this, so the limit is
    defined once. Never lowers a limit someone set higher (or disabled) on purpose.
    """
    from PIL import Image  # noqa: PLC0415 - Pillow is only needed by the callers that decode originals

    current = Image.MAX_IMAGE_PIXELS
    if current is not None and current < MAX_IMAGE_PIXELS:
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    return Image


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


def crop_filename(stem: str, label: str, xyxy, ext: str = "jpg") -> str:
    """Build the canonical crop filename ``<stem>__<label>__x1_y1_x2_y2.<ext>``.

    ``label`` is the class part, e.g. ``BBOX-ruler`` or ``SEG-leaf``. The four bbox
    corners are integers joined by single ``_`` (the three logical parts are joined by
    ``__``), so the parent stem and coords keep their own single underscores and the name
    round-trips through :func:`parse_crop_filename`.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    suffix = f".{str(ext).lstrip('.')}" if ext else ""
    return f"{stem}__{label}__{x1}_{y1}_{x2}_{y2}{suffix}"


def parse_crop_filename(name: str) -> dict | None:
    """Inverse of :func:`crop_filename`. Returns stem, prefix (BBOX/SEG), friendly class,
    label, and the integer ``xyxy`` — enough to reinsert a crop/mask into its parent.
    """
    base = str(name).rsplit(".", 1)[0]
    parts = base.split("__")
    if len(parts) < 3:
        return None
    coords, label, stem = parts[-1], parts[-2], "__".join(parts[:-2])
    try:
        xyxy = tuple(int(v) for v in coords.split("_"))
    except ValueError:
        return None
    if len(xyxy) != 4:
        return None
    prefix, _, friendly = label.partition("-")
    return {"stem": stem, "prefix": prefix, "friendly": friendly, "label": label, "xyxy": xyxy}


def save_crop(img, xyxy, stem: str, label: str, crops_dir, ext: str = "jpg", quality: int = 100) -> str:
    """Crop ``img`` to ``xyxy`` and save as ``<stem>__<label>__x1_y1_x2_y2.<ext>``.

    The filename encodes the RAW detection box (so it matches the DB row and can be
    reinserted into the parent); the pixel slice is clamped to the image bounds.
    """
    h, w = img.shape[:2]
    rx1, ry1, rx2, ry2 = (int(round(v)) for v in xyxy)
    cx1, cy1 = max(0, rx1), max(0, ry1)
    cx2, cy2 = min(w, rx2), min(h, ry2)
    os.makedirs(str(crops_dir), exist_ok=True)
    name = crop_filename(stem, label, (rx1, ry1, rx2, ry2), ext)
    path = os.path.join(str(crops_dir), name)
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if name.lower().endswith((".jpg", ".jpeg")) else []
        cv2.imwrite(path, crop, params)
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


# ---- rotation + content-fit (leaf orientation products) ------------------------
def rotate_image(img, angle_cw: float, bg: int = 0, nearest: bool = False):
    """Rotate ``img`` CLOCKWISE by ``angle_cw`` degrees, expanding the canvas to fit the whole
    rotated image; blank area filled with ``bg``. ``nearest`` (for masks) avoids interpolated edges.

    cv2's rotation angle is counter-clockwise-positive, so a clockwise angle is passed negated.
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), -float(angle_cw), 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(round(h * sin + w * cos))
    nh = int(round(h * cos + w * sin))
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    border = (bg, bg, bg) if img.ndim == 3 else bg
    return cv2.warpAffine(img, m, (nw, nh), flags=flags, borderValue=border)


def mask_bbox(mask) -> tuple[int, int, int, int] | None:
    """Tight ``(x1, y1, x2, y2)`` bounding box of a boolean/uint8 mask's True pixels, or ``None``."""
    m = np.asarray(mask)
    ys, xs = np.where(m > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_to_box(img, box, pad: int = 0):
    """Slice ``img`` to ``box=(x1,y1,x2,y2)`` (optionally padded, clamped to bounds)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    return img[y1:y2, x1:x2]
