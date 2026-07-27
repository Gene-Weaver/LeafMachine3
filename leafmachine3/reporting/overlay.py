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

from leafmachine3.core.imaging import decode_polygon, mask_bbox, scale_polygon
from leafmachine3.core.landmarks import KPT_GROUP, MIDVEIN_N, SKELETON
from leafmachine3.reporting.palette import RGB, LandmarkStyle, OverlayStyle, PetioleStyle

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
    landmark_measurements: Sequence[Any] = (),
    petioles: Sequence[Any] = (),
    petiole_style: Optional[PetioleStyle] = None,
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
        _draw_landmarks(out, landmarks, landmark_style or LandmarkStyle(), scale,
                        _curvature_by_detection(landmark_measurements))
    # Petiole width bands (purple) go on top too.
    if style.draw_petiole and petioles:
        _draw_petiole_widths(out, petioles, petiole_style or PetioleStyle(), scale)
    return out


def _ipt(p) -> tuple[int, int]:
    return (int(round(p[0])), int(round(p[1])))


def _draw_petiole_widths(out: np.ndarray, petioles: Sequence[Any], pet_style: PetioleStyle, scale: float) -> None:
    """Draw each measured petiole's reported width segment as a purple band (working coords * scale)."""
    for r in petioles:
        raw = _row_get(r, "width_segment_json", None)
        if not raw:
            continue
        try:
            seg = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not seg or len(seg) < 2:
            continue
        p1 = _ipt((seg[0][0] * scale, seg[0][1] * scale))
        p2 = _ipt((seg[1][0] * scale, seg[1][1] * scale))
        cv2.line(out, p1, p2, _bgr(pet_style.width_color), max(2, pet_style.band_thickness), cv2.LINE_AA)


def build_leaf_petiole_overlay(
    crop_bgr: np.ndarray,
    class_polys: dict[str, list],
    sample_segments: Sequence[Any],
    width_segment: Optional[Any],
    measure_lines: Sequence[str],
    seg_colors: dict[str, RGB],
    pet_style: PetioleStyle,
) -> np.ndarray:
    """Per-leaf petiole overlay: the leaf crop with the Leaf + Petiole masks filled (NO outline, so
    the mask reaches the true edge), the per-sample width probes (light purple), the reported width
    band (purple), and a measurement panel. All geometry is in the crop frame."""
    out = crop_bgr.copy()
    layer = out.copy()
    touched = np.zeros(out.shape[:2], dtype=np.uint8)
    for cls in ("Leaf", "Petiole"):                       # filled, no outline -> fills to the edge
        color = _bgr(seg_colors.get(cls, (0, 255, 0)))
        for poly in class_polys.get(cls, []):
            pts = np.round(np.asarray(poly, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
            if len(pts) >= 3:
                cv2.fillPoly(layer, [pts], color)
                cv2.fillPoly(touched, [pts], 1)
    if touched.any():
        a = max(0.0, min(1.0, pet_style.mask_alpha))
        blended = cv2.addWeighted(layer, a, out, 1.0 - a, 0.0)
        out[touched.astype(bool)] = blended[touched.astype(bool)]

    for s in sample_segments or []:                       # light-purple sample probes
        if s and len(s) >= 2:
            cv2.line(out, _ipt(s[0]), _ipt(s[1]), _bgr(pet_style.sample_color),
                     max(1, pet_style.sample_thickness), cv2.LINE_AA)
    if width_segment and len(width_segment) >= 2:         # reported width band (blue)
        cv2.line(out, _ipt(width_segment[0]), _ipt(width_segment[1]), _bgr(pet_style.width_color),
                 max(2, pet_style.band_thickness), cv2.LINE_AA)
    if measure_lines:
        _draw_measure_panel(out, list(measure_lines), pet_style.label_color)

    # second half: a pixelated blow-up of the petiole RGB cutout with a 1-px width line, to the right
    right = _petiole_zoom_panel(crop_bgr, class_polys.get("Petiole", []), width_segment, out.shape[0], pet_style)
    if right is not None:
        out = np.hstack([out, right])
    return out


def _petiole_zoom_panel(
    crop_bgr: np.ndarray,
    petiole_polys: list,
    width_segment: Optional[Any],
    target_h: int,
    pet_style: PetioleStyle,
) -> Optional[np.ndarray]:
    """Fitted, full-color crop around the petiole -- the petiole keeps its original pixels while the
    background is tinted toward ``zoom_bg_color`` (so context stays visible but the petiole stands
    out) -- **rotated so the petiole's long axis is vertical** (a long petiole would otherwise blow
    up to a huge width when scaled to the crop height), nearest-neighbour blown up to ``target_h``,
    with the reported width location drawn as a 1-original-pixel line in ``width_color`` so the exact
    pixels the width spans are checkable."""
    if not petiole_polys:
        return None
    h, w = crop_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in petiole_polys:
        pts = np.round(np.asarray(poly, dtype=float)).astype(np.int32).reshape(-1, 1, 2)
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 1)
    if not mask.any():
        return None

    img = crop_bgr.astype(np.float32).copy()                 # full color; petiole keeps its pixels
    tint = max(0.0, min(1.0, pet_style.zoom_bg_tint))
    if tint > 0:                                             # background tinted toward zoom_bg_color
        bg = ~mask.astype(bool)
        img[bg] = img[bg] * (1.0 - tint) + np.array(_bgr(pet_style.zoom_bg_color), np.float32) * tint
    img = img.astype(np.uint8)

    # rotate the panel to the angle that MINIMISES the petiole's horizontal extent, so a long (or
    # long-and-curved) petiole doesn't blow up the panel width when it is scaled to the crop height.
    # A brute-force sweep (never worse than no rotation) beats PCA, which aligns the chord and can
    # actually widen a curved petiole by turning its bulge sideways.
    ys, xs = np.where(mask)
    dx = xs.astype(np.float32) - float(xs.mean())
    dy = ys.astype(np.float32) - float(ys.mean())
    best_ang, best_w = 0.0, None
    for a in range(0, 180, 3):
        th = float(np.radians(a))
        ext = float(np.ptp(np.cos(th) * dx + np.sin(th) * dy))   # width after cv2 rotation by a deg
        if best_w is None or ext < best_w:
            best_w, best_ang = ext, float(a)
    cx, cy = w / 2.0, h / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), best_ang, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(round(h * sin + w * cos)), int(round(h * cos + w * sin))
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    rimg = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    rmask = cv2.warpAffine(mask, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=0) > 0

    box = mask_bbox(rmask)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    cut = rimg[y1:y2, x1:x2].copy()
    if width_segment and len(width_segment) >= 2:           # 1-px width line, rotated + fitted into place

        def _tf(p):
            x = m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]
            y = m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]
            return _ipt((x - x1, y - y1))

        cv2.line(cut, _tf(width_segment[0]), _tf(width_segment[1]), _bgr(pet_style.width_color), 1, cv2.LINE_8)
    ch, cw = cut.shape[:2]
    if ch == 0 or cw == 0:
        return None
    new_w = max(1, int(round(cw * (target_h / float(ch)))))
    return cv2.resize(cut, (new_w, int(target_h)), interpolation=cv2.INTER_NEAREST)   # pixelated blow-up


def _curvature_by_detection(measurements: Sequence[Any]) -> dict[tuple[int, int], int]:
    """Map ``(detection_id, instance_index) -> curvature_point`` (midvein index) for bend drawing."""
    out: dict[tuple[int, int], int] = {}
    for mm in measurements or []:
        cp = _row_get(mm, "curvature_point", None)
        if cp is not None:
            key = (int(_row_get(mm, "detection_id", -1)), int(_row_get(mm, "instance_index", 0)))
            out[key] = int(cp)
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
    curvature_idx: Optional[int] = None,
) -> None:
    """Draw the skeleton edges then the keypoints. ``pt`` is name -> (x, y) in target pixels.

    ``curvature_idx`` (the ``curvature_point`` midvein index) adds the two gray bend lines from the
    midvein ends to that vertex -- the arms whose angle ``lamina_curvature`` measures.
    """

    def ok(name: str) -> bool:
        return name in pt and conf.get(name, 1.0) >= lm_style.min_conf

    def ipt(p) -> tuple[int, int]:
        return (int(round(p[0])), int(round(p[1])))

    if lm_style.draw_skeleton:
        lw = max(1, lm_style.line_width)
        present_mv = [pt[f"midvein_{i}"] for i in range(MIDVEIN_N) if ok(f"midvein_{i}")]
        # 1) curvature bend arms FIRST, UNDERNEATH everything (black by default): two lines from the
        #    midvein ends to the most-bent vertex -- the arms of the lamina_curvature angle. The
        #    cyan midvein + white extent are drawn after, so they sit on top of these.
        if len(present_mv) >= 2 and curvature_idx is not None and ok(f"midvein_{curvature_idx}"):
            v = ipt(pt[f"midvein_{curvature_idx}"])
            bend = _bgr(lm_style.curvature_color)
            cv2.line(out, ipt(present_mv[0]), v, bend, lw, cv2.LINE_AA)
            cv2.line(out, ipt(present_mv[-1]), v, bend, lw, cv2.LINE_AA)
        # 2) skeleton edges (cyan midvein, petiole, apex, base, width) ON TOP of the bend arms.
        for a, b, kind in SKELETON:
            if kind == "lamina_length":
                continue   # the extent chord is drawn below between the first/last PRESENT midvein
            if ok(a) and ok(b):
                cv2.line(out, ipt(pt[a]), ipt(pt[b]), _bgr(lm_style.color_for_kind(kind)), lw, cv2.LINE_AA)
        # 3) lamina_extent (white line): chord between the first/last PRESENT midvein points, on top,
        #    so it matches the metric exactly even when an endpoint keypoint is occluded.
        if len(present_mv) >= 2:
            cv2.line(out, ipt(present_mv[0]), ipt(present_mv[-1]),
                     _bgr(lm_style.color_for_kind("lamina_length")), lw, cv2.LINE_AA)
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
    out: np.ndarray, landmarks: Sequence[Any], lm_style: LandmarkStyle, scale: float,
    curv_by_det: Optional[dict[tuple[int, int], int]] = None,
) -> None:
    """Draw every leaf's keypoints/skeleton on the summary image (working coords * scale)."""
    curv_by_det = curv_by_det or {}
    for key, rows in _group_landmarks(landmarks).items():
        pt: dict[str, tuple[float, float]] = {}
        conf: dict[str, float] = {}
        for r in rows:
            name = str(_row_get(r, "kpt_name", ""))
            x, y = _row_get(r, "x", None), _row_get(r, "y", None)
            if x is None or y is None:
                continue
            pt[name] = (float(x) * scale, float(y) * scale)
            conf[name] = float(_row_get(r, "conf", 1.0) or 0.0)
        _draw_landmark_skeleton(out, pt, conf, lm_style, curvature_idx=curv_by_det.get(key))


def build_leaf_landmark_overlay(
    crop_bgr: np.ndarray,
    landmark_rows: Sequence[Any],
    measure_lines: Sequence[str],
    lm_style: LandmarkStyle,
    curvature_idx: Optional[int] = None,
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
    _draw_landmark_skeleton(out, pt, conf, lm_style, curvature_idx=curvature_idx)
    if measure_lines:
        _draw_measure_panel(out, list(measure_lines), lm_style.label_color)
    return out


def _draw_measure_panel(out: np.ndarray, lines: list[str], label_color: RGB = (255, 255, 255)) -> None:
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
        cv2.putText(out, t, (pad, y), _FONT, fs, _bgr(label_color), thick, cv2.LINE_AA)
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
