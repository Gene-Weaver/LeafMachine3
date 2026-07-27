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

import json
import logging
from typing import Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - cv2 is always present at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from leafmachine3.core.imaging import decode_polygon, scale_polygon
from leafmachine3.core.landmarks import KPT_GROUP, MIDVEIN_N, SKELETON
from leafmachine3.reporting.palette import RGB, LandmarkStyle, OverlayStyle

log = logging.getLogger("leafmachine3.overlay")

_FONT = cv2.FONT_HERSHEY_SIMPLEX if cv2 is not None else 0
_BASE_FONT_SCALE = 0.6
_LEAF_DET_CLASSES = {"Leaf_WHOLE", "Leaf_PARTIAL"}   # plant detections replaced by rotated boxes


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
    morphology: Sequence[Any] = (),
    landmarks: Sequence[Any] = (),
    landmark_style: Optional[LandmarkStyle] = None,
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

    # In "rotated" mode leaf boxes come from Morphology's rotated (min) bounding box; the
    # axis-aligned YOLO leaf boxes are suppressed. Falls back to YOLO if no morphology exists.
    rotated_mode = style.box_style == "rotated" and bool(morphology)

    # Masks are painted first (under boxes/labels) so outlines stay crisp on top.
    if style.draw_masks:
        _draw_masks(out, leaves, style, scale)
    _draw_boxes(out, detections, style, scale, skip_plant_leaf=rotated_mode)
    if rotated_mode and style.draw_boxes_plant:
        _draw_rotated_boxes(out, morphology, style, scale)
    if style.draw_cf_banner and cf_px_per_cm:
        _draw_cf_banner(out, float(cf_px_per_cm), style)
    # Landmarks go ON TOP of masks + boxes so every leaf's keypoints stay visible.
    if style.draw_landmarks and landmarks:
        _draw_landmarks(out, landmarks, landmark_style or LandmarkStyle(), scale)
    return out


# -- landmarks (keypoints + skeleton) ----------------------------------------------
def _group_landmarks(landmarks: Sequence[Any]) -> dict[tuple[int, int], list[Any]]:
    """Group keypoint rows by ``(detection_id, instance_index)`` -- one leaf per group."""
    groups: dict[tuple[int, int], list[Any]] = {}
    for r in landmarks or []:
        key = (int(_row_get(r, "detection_id", -1)), int(_row_get(r, "instance_index", 0)))
        groups.setdefault(key, []).append(r)
    return groups


def _draw_landmark_skeleton(
    out: np.ndarray,
    pt: dict[str, tuple[float, float]],
    conf: dict[str, float],
    lm_style: LandmarkStyle,
) -> None:
    """Draw the skeleton edges then the keypoints. ``pt`` is name -> (x, y) in target pixels."""

    def ok(name: str) -> bool:
        return name in pt and conf.get(name, 1.0) >= lm_style.min_conf

    if lm_style.draw_skeleton:
        lw = max(1, lm_style.line_width)
        for a, b, kind in SKELETON:
            if kind == "lamina_length":
                continue   # the extent chord is drawn below between the first/last PRESENT midvein
            if ok(a) and ok(b):
                pa = (int(round(pt[a][0])), int(round(pt[a][1])))
                pb = (int(round(pt[b][0])), int(round(pt[b][1])))
                cv2.line(out, pa, pb, _bgr(lm_style.color_for_kind(kind)), lw, cv2.LINE_AA)
        # lamina_extent (white line): chord between the first and last PRESENT midvein points, so it
        # matches the lamina_extent metric exactly even when an endpoint keypoint is occluded.
        present_mv = [pt[f"midvein_{i}"] for i in range(MIDVEIN_N) if ok(f"midvein_{i}")]
        if len(present_mv) >= 2:
            pa = (int(round(present_mv[0][0])), int(round(present_mv[0][1])))
            pb = (int(round(present_mv[-1][0])), int(round(present_mv[-1][1])))
            cv2.line(out, pa, pb, _bgr(lm_style.color_for_kind("lamina_length")), lw, cv2.LINE_AA)
    if lm_style.draw_points:
        r = max(1, lm_style.point_radius)
        for name, (x, y) in pt.items():
            if not ok(name):
                continue
            c = _bgr(lm_style.color_for_group(KPT_GROUP.get(name, "lamina")))
            center = (int(round(x)), int(round(y)))
            cv2.circle(out, center, r, c, -1, cv2.LINE_AA)
            cv2.circle(out, center, r, (0, 0, 0), 1, cv2.LINE_AA)   # thin dark ring for contrast


def _draw_landmarks(
    out: np.ndarray, landmarks: Sequence[Any], lm_style: LandmarkStyle, scale: float
) -> None:
    """Draw every leaf's keypoints/skeleton on the summary image (working coords * scale)."""
    for _key, rows in _group_landmarks(landmarks).items():
        pt: dict[str, tuple[float, float]] = {}
        conf: dict[str, float] = {}
        for r in rows:
            name = str(_row_get(r, "kpt_name", ""))
            x, y = _row_get(r, "x", None), _row_get(r, "y", None)
            if x is None or y is None:
                continue
            pt[name] = (float(x) * scale, float(y) * scale)
            conf[name] = float(_row_get(r, "conf", 1.0) or 0.0)
        _draw_landmark_skeleton(out, pt, conf, lm_style)


def build_leaf_landmark_overlay(
    crop_bgr: np.ndarray,
    landmark_rows: Sequence[Any],
    measure_lines: Sequence[str],
    lm_style: LandmarkStyle,
) -> np.ndarray:
    """Per-leaf overlay: draw the keypoints/skeleton on the leaf crop (crop-frame coords) plus a
    panel of the derived measurements. Used for the ``Overlay_Landmarks`` output."""
    out = crop_bgr.copy()
    pt: dict[str, tuple[float, float]] = {}
    conf: dict[str, float] = {}
    for r in landmark_rows:
        name = str(_row_get(r, "kpt_name", ""))
        xc, yc = _row_get(r, "x_crop", None), _row_get(r, "y_crop", None)
        if xc is None or yc is None:
            continue
        pt[name] = (float(xc), float(yc))
        conf[name] = float(_row_get(r, "conf", 1.0) or 0.0)
    _draw_landmark_skeleton(out, pt, conf, lm_style)
    if measure_lines:
        _draw_measure_panel(out, list(measure_lines), lm_style)
    return out


def _draw_measure_panel(out: np.ndarray, lines: list[str], lm_style: LandmarkStyle) -> None:
    """Draw a translucent dark panel of measurement text at the crop's top-left."""
    h, w = out.shape[:2]
    fs = max(0.4, min(0.9, w / 520.0))
    thick = max(1, int(round(fs * 1.6)))
    pad = int(round(6 * fs))
    sizes = [cv2.getTextSize(t, _FONT, fs, thick)[0] for t in lines]
    line_h = max((s[1] for s in sizes), default=12) + int(round(6 * fs))
    box_w = min(w, (max((s[0] for s in sizes), default=10)) + 2 * pad)
    box_h = min(h, line_h * len(lines) + 2 * pad)
    panel = out[0:box_h, 0:box_w].copy()
    dark = np.zeros_like(panel)
    cv2.addWeighted(dark, 0.55, panel, 0.45, 0.0, panel)
    out[0:box_h, 0:box_w] = panel
    y = pad + line_h - int(round(6 * fs))
    for t in lines:
        cv2.putText(out, t, (pad, y), _FONT, fs, _bgr(lm_style.label_color), thick, cv2.LINE_AA)
        y += line_h


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
def _draw_boxes(
    out: np.ndarray, detections: Sequence[dict], style: OverlayStyle, scale: float,
    skip_plant_leaf: bool = False,
) -> None:
    for d in detections:
        source = str(d.get("source", "plant"))
        if source == "archival" and not style.draw_boxes_archival:
            continue
        if source == "plant" and not style.draw_boxes_plant:
            continue
        cls_name = str(d.get("cls_name", ""))
        if skip_plant_leaf and source == "plant" and cls_name in _LEAF_DET_CLASSES:
            continue  # drawn as a rotated bounding box from Morphology instead
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


# -- rotated (minimum) bounding boxes from Morphology ------------------------------
def _draw_rotated_boxes(out: np.ndarray, morphology: Sequence[Any], style: OverlayStyle, scale: float) -> None:
    """Draw each leaf's rotated minimum bounding box (LM2 procedure) as a tilted rectangle."""
    for m in morphology:
        cls_name = str(_row_get(m, "cls_name", "Leaf"))
        if not style.show(cls_name):
            continue
        raw = _row_get(m, "rotated_bbox_json")
        if not raw:
            continue
        try:
            corners = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        pts = (np.asarray(corners, dtype=float) * scale).round().astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        bgr = _bgr(style.color_for(cls_name))
        cv2.polylines(out, [pts], True, bgr, max(1, style.line_width_plant), cv2.LINE_AA)
        if style.draw_labels:
            top = min(range(len(pts)), key=lambda i: int(pts[i][0][1]))   # topmost corner
            _draw_label(out, cls_name, (int(pts[top][0][0]), int(pts[top][0][1])), bgr, style)


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
