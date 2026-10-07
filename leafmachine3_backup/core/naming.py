"""Crop / mask filename class-labels, driven by the ``naming`` block in LM3_settings.yaml.

A crop or mask file is named ``<stem>__<PREFIX>-<friendly>__x1_y1_x2_y2.<ext>`` where
``PREFIX`` is ``BBOX`` for detection boxes and ``SEG`` for segmentation masks, and
``friendly`` is the user-facing class name from ``naming.friendly_names`` (e.g. the real
class ``Leaf_WHOLE`` -> ``leaf``). Everything falls back gracefully when the config omits
the ``naming`` block, so runs never break on a missing mapping.

Every prefix here is distinct in a way that survives dropping the extension: the whole point
is that a user can pour every derived file from a run into one directory without anything
overwriting anything else. So a binary mask and its RGB cutout twin -- same class, same box,
different only in ``.png`` vs ``.jpg`` -- get DIFFERENT prefixes (``SEG`` / ``SEGRGB``,
``MaskFull`` / ``MaskRGBFull``) rather than relying on the suffix to tell them apart.
"""
from __future__ import annotations

BBOX = "bbox"
SEG = "seg"
SEG_RGB = "seg_rgb"
MASK_FULL = "mask_full"
MASK_RGB_FULL = "mask_rgb_full"
LANDMARK = "landmark"
PETIOLE = "petiole"
BILATERAL = "bilateral"

_PREFIX_KEYS = {BBOX: ("bbox_prefix", "BBOX"), SEG: ("seg_prefix", "SEG"),
                SEG_RGB: ("seg_rgb_prefix", "SEGRGB"),
                MASK_FULL: ("mask_full_prefix", "MaskFull"),
                MASK_RGB_FULL: ("mask_rgb_full_prefix", "MaskRGBFull"),
                LANDMARK: ("landmark_prefix", "LM"),
                PETIOLE: ("petiole_prefix", "PET"),
                BILATERAL: ("bilateral_prefix", "BSYM")}


def friendly_name(cfg, cls_name: str) -> str:
    """Map a real class name to its user-facing friendly name (identity if unmapped)."""
    mapping = _get(cfg, "naming", "friendly_names")
    if mapping is not None:
        v = _getk(mapping, cls_name, None)
        if v:
            return str(v)
    return str(cls_name)


def crop_label(cfg, kind: str, cls_name: str) -> str:
    """Return the filename class-label: ``BBOX-ruler`` (bbox), ``SEG-leaf`` / ``SEGRGB-leaf``
    (per-crop binary mask / RGB cutout), ``MaskFull-leaf`` / ``MaskRGBFull-leaf`` (the same
    pair on the full image), or ``LM-leaf`` (landmark — per-leaf landmark overlays)."""
    key, default = _PREFIX_KEYS.get(kind, _PREFIX_KEYS[BBOX])
    prefix = str(_get(cfg, "naming", key) or default)
    return f"{prefix}-{friendly_name(cfg, cls_name)}"


# -- tolerant access (works on Config/Section dot-access OR plain dicts) ------------
def _getk(obj, key, default=None):
    if obj is None:
        return default
    try:
        if hasattr(obj, "get"):
            v = obj.get(key, default)
            return default if v is None else v
    except Exception:
        pass
    try:
        v = getattr(obj, key, default)
        return default if v is None else v
    except Exception:
        return default


def _get(cfg, *path):
    cur = cfg
    for p in path:
        cur = _getk(cur, p, None)
        if cur is None:
            return None
    return cur
