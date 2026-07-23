"""Summary_Image overlay renderer for the LeafMachine3 Reporter.

Reconstructs the LeafMachine2 "custom overlay" look on demand from stored records and
the ORIGINAL image pixels: archival/plant detection boxes, leaf-segmentation instance
masks (alpha-blended fill + outline), per-box class labels, and a conversion-factor
banner. Every color, line width and per-part visibility flag is driven by an
:class:`~leafmachine3.reporting.palette.OverlayStyle` (built from ``report.overlay`` in
``LM3_settings.yaml``), so users fully restyle the overlay without touching code.

All stored geometry is in the WORKING (analysis) coordinate frame. The Reporter draws
onto the full-resolution original, so every coordinate is multiplied by ``1/work_scale``
before it is rendered.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - cv2 is always present at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from leafmachine3.core.imaging import decode_polygon, scale_polygon
from leafmachine3.reporting.palette import RGB, OverlayStyle

log = logging.getLogger("leafmachine3.overlay")

_FONT = cv2.FONT_HERSHEY_SIMPLEX if cv2 is not None else 0
_BASE_FONT_SCALE = 0.6


# -- small tolerant accessors (rows are sqlite3.Row | dataclass | dict) ------------
def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a sqlite3.Row, mapping, or object with attributes."""
    if row is None:
        return default
    if hasattr(row, key):
        v = getattr(row, key)
        return default if v is None else v
    try:
        v = row[key]
    except Exception:
        return default
    return default if v is None else v


def _bgr(color: RGB) -> tuple[int, int, int]:
    """Convert a config-space ``[R, G, B]`` triple to OpenCV BGR ints."""
    r, g, b = (int(c) for c in color)
    return (b, g, r)


def build_summary_image(
    image_bgr: np.ndarray,
    detections: Sequence[dict],
    leaves: Sequence[Any],
    cf_px_per_cm: Optional[float],
    style: OverlayStyle,
    work_scale: float = 1.0,
) -> np.ndarray:
    """Render the Summary_Image overlay onto a copy of ``image_bgr``.

    Args:
        image_bgr: The ORIGINAL image (BGR ``np.ndarray``) to draw onto.
        detections: Box dicts with ``cls_name``, ``conf``, ``xyxy=(x1,y1,x2,y2)`` and
            ``source`` in ``{'archival', 'plant'}``. Coordinates are in the working frame.
        leaves: Leaf-segmentation rows exposing ``cls_name``, ``mask_format`` and
            ``mask_data`` (a ``polygon_xy`` JSON ring in working coordinates).
        cf_px_per_cm: Derived pixels-per-cm conversion factor, or ``None`` if unknown.
        style: Resolved :class:`OverlayStyle` (colors / flags / per-class visibility).
        work_scale: Working-to-original scale of the stored geometry. Coordinates are
            multiplied by ``1/work_scale`` to map onto the original pixels.

    Returns:
        A new BGR ``np.ndarray`` the same shape as ``image_bgr``.
    """
    out = image_bgr.copy()
    scale = 1.0 / float(work_scale or 1.0)

    # Masks are painted first (under boxes/labels) so outlines stay crisp on top.
    if style.draw_masks:
        _draw_masks(out, leaves, style, scale)
    _draw_boxes(out, detections, style, scale)
    if style.draw_cf_banner and cf_px_per_cm:
        _draw_cf_banner(out, float(cf_px_per_cm), style)
    return out


# -- masks -------------------------------------------------------------------------
def _draw_masks(out: np.ndarray, leaves: Sequence[Any], style: OverlayStyle, scale: float) -> None:
    """Alpha-blend filled instance polygons, then stroke their outlines opaquely."""
    layer = out.copy()
    touched = np.zeros(out.shape[:2], dtype=np.uint8)
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    for row in leaves:
        cls_name = str(_row_get(row, "cls_name", ""))
        if not style.show(cls_name):
            continue
        if str(_row_get(row, "mask_format", "polygon_xy")) != "polygon_xy":
            continue  # coco_rle is not produced by this pipeline; skip defensively
        mask_data = _row_get(row, "mask_data")
        if not mask_data:
            continue
        try:
            poly = scale_polygon(decode_polygon(str(mask_data)), scale)
        except Exception:
            log.warning("skipping unreadable mask polygon for cls=%s", cls_name)
            continue
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        bgr = _bgr(style.color_for(cls_name))
        cv2.fillPoly(layer, [pts], bgr)
        cv2.fillPoly(touched, [pts], 1)
        outlines.append((pts, bgr))

    if touched.any():
        alpha = max(0.0, min(1.0, style.alpha))
        blended = cv2.addWeighted(layer, alpha, out, 1.0 - alpha, 0.0)
        out[touched.astype(bool)] = blended[touched.astype(bool)]
    for pts, bgr in outlines:
        cv2.polylines(out, [pts], True, bgr, max(1, style.line_width_mask), cv2.LINE_AA)


# -- boxes + labels ----------------------------------------------------------------
def _draw_boxes(out: np.ndarray, detections: Sequence[dict], style: OverlayStyle, scale: float) -> None:
    for d in detections:
        source = str(d.get("source", "plant"))
        if source == "archival" and not style.draw_boxes_archival:
            continue
        if source == "plant" and not style.draw_boxes_plant:
            continue
        cls_name = str(d.get("cls_name", ""))
        if not style.show(cls_name):
            continue
        x1, y1, x2, y2 = (float(v) * scale for v in d["xyxy"])
        p1, p2 = (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))
        bgr = _bgr(style.color_for(cls_name))
        lw = style.line_width_archival if source == "archival" else style.line_width_plant
        cv2.rectangle(out, p1, p2, bgr, max(1, lw), cv2.LINE_AA)
        if style.draw_labels:
            text = cls_name
            if style.draw_confidence and d.get("conf") is not None:
                text = f"{cls_name} {float(d['conf']):.2f}"
            _draw_label(out, text, p1, bgr, style)


def _draw_label(
    out: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    box_bgr: tuple[int, int, int],
    style: OverlayStyle,
) -> None:
    """Draw ``text`` in a filled class-colored tab just above the box top-left."""
    font_scale = _BASE_FONT_SCALE * max(0.1, style.font_scale)
    thickness = max(1, int(round(font_scale * 2)))
    (tw, th), base = cv2.getTextSize(text, _FONT, font_scale, thickness)
    x, y = anchor
    top = max(0, y - th - base - 4)
    x2 = min(out.shape[1], x + tw + 6)
    cv2.rectangle(out, (x, top), (x2, top + th + base + 4), box_bgr, -1, cv2.LINE_AA)
    cv2.putText(
        out, text, (x + 3, top + th + 2), _FONT, font_scale,
        _bgr(style.label_text_color), thickness, cv2.LINE_AA,
    )


# -- conversion-factor banner ------------------------------------------------------
def _draw_cf_banner(out: np.ndarray, cf_px_per_cm: float, style: OverlayStyle) -> None:
    """Print the derived conversion factor in a banner at the top-left of the sheet."""
    text = f"CF: {cf_px_per_cm:.2f} px/cm"
    font_scale = _BASE_FONT_SCALE * 1.4 * max(0.1, style.font_scale)
    thickness = max(1, int(round(font_scale * 2)))
    (tw, th), base = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = int(round(6 * max(0.5, style.font_scale)))
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + base + 2 * pad), _bgr(style.cf_banner_color), -1)
    cv2.putText(
        out, text, (pad, th + pad), _FONT, font_scale,
        (0, 0, 0), thickness, cv2.LINE_AA,
    )
