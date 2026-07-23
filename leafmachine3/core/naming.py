"""Crop / mask filename class-labels, driven by the ``naming`` block in LM3_settings.yaml.

A crop or mask file is named ``<stem>__<PREFIX>-<friendly>__x1_y1_x2_y2.<ext>`` where
``PREFIX`` is ``BBOX`` for detection boxes and ``SEG`` for segmentation masks, and
``friendly`` is the user-facing class name from ``naming.friendly_names`` (e.g. the real
class ``Leaf_WHOLE`` -> ``leaf``). Everything falls back gracefully when the config omits
the ``naming`` block, so runs never break on a missing mapping.
"""
from __future__ import annotations

BBOX = "bbox"
SEG = "seg"


def friendly_name(cfg, cls_name: str) -> str:
    """Map a real class name to its user-facing friendly name (identity if unmapped)."""
    mapping = _get(cfg, "naming", "friendly_names")
    if mapping is not None:
        v = _getk(mapping, cls_name, None)
        if v:
            return str(v)
    return str(cls_name)


def crop_label(cfg, kind: str, cls_name: str) -> str:
    """Return the filename class-label, e.g. ``BBOX-ruler`` (kind='bbox') or ``SEG-leaf``."""
    if kind == SEG:
        prefix = str(_get(cfg, "naming", "seg_prefix") or "SEG")
    else:
        prefix = str(_get(cfg, "naming", "bbox_prefix") or "BBOX")
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
