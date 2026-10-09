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
import math
from typing import Any, Optional, Sequence

import numpy as np

try:  # pragma: no cover - cv2 is always present at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from leafmachine3.core import records as _records
from leafmachine3.core.imaging import decode_polygon, mask_bbox, scale_polygon
from leafmachine3.core.records import CF_SOURCE_MP, CF_SOURCE_RULER
from leafmachine3.core.landmarks import KPT_GROUP, MIDVEIN_N, SKELETON
from leafmachine3.reporting import fieldprism_viz as fpv
from leafmachine3.reporting.palette import (
    LEAF_DET_CLASSES, RGB, CFScalebarStyle, FieldPrismStyle, GroupStyle, LandmarkStyle, OverlayStyle,
    PetioleStyle, SpecimenStyle,
)

# A CF published from FieldPrism markers. Read defensively so this module also imports against a
# records.py that predates FieldPrism support.
CF_SOURCE_FP = getattr(_records, "CF_SOURCE_FP", "measured_from_fieldprism")

log = logging.getLogger("leafmachine3.overlay")

_FONT = cv2.FONT_HERSHEY_SIMPLEX if cv2 is not None else 0
_BASE_FONT_SCALE = 0.6
_LEAF_DET_CLASSES = LEAF_DET_CLASSES   # plant detections replaced by rotated boxes

# Box labels are white text centered in the box, auto-fit to the box but clamped to a font scale
# range that is RELATIVE to the image resolution: the clamp is anchored at REF_LONG (the long side
# of the 2592px reference overlay) and scales linearly with the actual long side, so labels look the
# same relative size on a 2000px sheet and an 8688px sheet. font_scale multiplies the clamp bounds.
_REF_LONG = 2592
_REF_FS_LO, _REF_FS_HI = 0.30, 1.70

_CM_PER_INCH = 2.54
_RULER_CLS = "Ruler"      # the ArchivalDetector class the in-ruler CF bars are laid over


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
    cf_style: Optional[CFScalebarStyle] = None,
    cf_source: Optional[str] = None,
    cf_note: Optional[str] = None,
    fieldprism: Optional[dict] = None,
    fp_style: Optional[FieldPrismStyle] = None,
) -> np.ndarray:
    """Render the Summary_Image overlay onto a copy of ``image_bgr``.

    Args:
        image_bgr: The ORIGINAL image (BGR ``np.ndarray``) to draw onto.
        detections: Box dicts with ``cls_name``, ``conf``, ``xyxy=(x1,y1,x2,y2)`` and
            ``source`` in ``{'archival', 'plant'}``. Coordinates are in the working frame.
        leaves: Leaf-segmentation rows exposing ``cls_name``, ``mask_format`` and
            ``mask_data`` (a ``polygon_xy`` JSON ring in working coordinates).
        cf_px_per_cm: Derived pixels-per-cm conversion factor **in the working frame**, or
            ``None`` if unknown. Rescaled to the original frame here for the scale overlays.
        style: Resolved :class:`OverlayStyle` (colors / flags / per-class visibility).
        work_scale: Working-to-original scale of the stored geometry. Coordinates are
            multiplied by ``1/work_scale`` to map onto the original pixels.
        cf_style: Resolved :class:`CFScalebarStyle` for the two optional CF scale overlays.
        cf_source: ``specimen.cf_source``. When the CF was PREDICTED from megapixels rather than
            measured from a ruler, the scale overlays must not pass for a measurement: the banner
            says so, the 1 cm / 1 inch raft goes in the top-left corner (under the banner)
            instead of over the detected rulers -- none of which produced this CF -- and the
            exterior checkerboard switches to its ``*_predicted`` colors (black / 50% gray).
            A measured CF is labeled "(measured from ruler)" in the banner.
        cf_note: Optional second banner line saying WHY a predicted CF was used -- see
            :func:`cf_fallback_reason`. Ignored unless the CF is predicted.
        fieldprism: ``{"sheet": ruler_FP_sheet row | None, "markers": [ruler_FP_marker rows]}``
            (see :func:`~leafmachine3.reporting.fieldprism_viz.fieldprism_from_record`), or
            ``None``. When the sheet has FieldPrism markers (and ``style.draw_fieldprism``), the
            FP detector boxes are not drawn at all (no fill, border, label or ruler raft); the
            markers are labeled like the FieldPrism app (TL/TR/C/BL, predicted BR cell, "1 cm =");
            reconstructed markers are drawn dashed; one 1 cm + 1 inch raft goes in the top-left
            corner under the banner whenever a CF is shown, and the sheet badge sits right of it.
            ``None`` renders exactly what this function drew before FieldPrism support.
        fp_style: Resolved :class:`FieldPrismStyle`; the app colors when omitted.

    Returns:
        A new BGR ``np.ndarray``, the same shape as ``image_bgr`` -- EXCEPT under
        ``style.insert_cf_exterior``, which appends a checkerboard margin to the top and left
        edges and so returns a taller/wider image.
    """
    out = image_bgr.copy()
    scale = 1.0 / float(work_scale or 1.0)
    fp = fieldprism if (style.draw_fieldprism and fpv.has_fp_markers(fieldprism)) else None
    is_fp_box = fpv.fp_box_matcher(fp) if fp else None

    # In "rotated" mode leaf boxes come from Morphology's rotated (min) bounding box; the
    # axis-aligned YOLO leaf boxes are suppressed. Falls back to YOLO if no morphology exists.
    rotated_mode = style.box_style == "rotated" and bool(morphology)

    # Masks are painted first (under boxes/labels) so outlines stay crisp on top.
    if style.draw_masks:
        _draw_masks(out, leaves, style, scale)
    # FieldPrism markers are shown the app's way (TL/TR/C/BL labels + the green BR cell), so their
    # detector boxes -- fill, border and "Ruler 0.93" label -- are left off this overlay entirely.
    box_dets = [d for d in detections if not is_fp_box(d)] if is_fp_box else detections
    _draw_boxes(out, box_dets, style, scale, skip_plant_leaf=rotated_mode)
    if rotated_mode and style.group("leaf").visible:
        _draw_rotated_boxes(out, morphology, style, scale)
    predicted = cf_source == CF_SOURCE_MP
    banner_bottom = 0
    if style.draw_cf_banner and cf_px_per_cm:
        banner_bottom = _draw_cf_banner(out, float(cf_px_per_cm), style, cf_source=cf_source,
                                        note=cf_note if predicted else None)
    # Landmarks go ON TOP of masks + boxes so every leaf's keypoints stay visible.
    if style.draw_landmarks and landmarks:
        _draw_landmarks(out, landmarks, landmark_style or LandmarkStyle(), scale,
                        _curvature_by_detection(landmark_measurements))
    # Petiole width bands (purple) go on top too.
    if style.draw_petiole and petioles:
        _draw_petiole_widths(out, petioles, petiole_style or PetioleStyle(), scale)

    # The two CF scale overlays LAST: the ruler rafts must cover whatever was drawn over the
    # rulers, and the exterior checkerboard changes the canvas size, so nothing may follow it.
    # cf_px_per_cm is a WORKING-frame value; * scale puts it in the original frame drawn on here.
    # A FieldPrism sheet also gets the corner raft under a MEASURED CF: its FP boxes carry no raft
    # (the marker labels are their scale), so the corner is where the sheet's 1 cm / 1 inch lives;
    # any non-FP ruler still gets its own raft. The sheet badge then goes right of the corner raft,
    # and the FP markers are drawn last so their app-style lines can steer clear of all three.
    cf_orig = (float(cf_px_per_cm) * scale) if cf_px_per_cm else None
    raft = None
    cfs = cf_style or CFScalebarStyle()
    if cf_orig and cf_orig > 0 and style.insert_cf_in_rulers:
        if predicted or fp:
            raft = _draw_corner_cf_raft(out, cf_orig, cfs, top=banner_bottom)
        if not predicted:
            _draw_ruler_cf_bars(out, detections, cf_orig, cfs, scale, skip=is_fp_box)
    if fp:
        fps = fp_style or FieldPrismStyle()
        taken = [] if raft is None else [(0, banner_bottom, raft[0], raft[1])]
        if banner_bottom:
            taken.append((0, 0, _cf_banner_layout(float(cf_px_per_cm), style, cf_source,
                                                  cf_note if predicted else None)[0], banner_bottom))
        badge = _draw_fp_sheet_badge(out, fp, fps, style, banner_bottom, raft[0] if raft else None)
        if badge is not None:
            taken.append(badge)
        fpv.draw_fp_markers(out, fp, fps, scale, avoid=taken)
    if cf_orig and cf_orig > 0 and style.insert_cf_exterior:
        out = _append_cf_exterior(out, cf_orig, cfs, predicted=predicted)
    return out


def _draw_fp_sheet_badge(out: np.ndarray, fp: dict, fps: FieldPrismStyle, style: OverlayStyle,
                         top: int, raft_right: Optional[int]) -> Optional[tuple[int, int, int, int]]:
    """The FieldPrism sheet badge, top-aligned with the corner raft immediately to its right, or
    flush left under the banner when no raft was drawn; returns its box. Sized like the box labels:
    reference px scaled by the image resolution and ``style.font_scale``."""
    text = fpv.fp_badge_text(fp.get("sheet"), fp.get("markers") or [])
    if not text:
        return None
    rr = _res_ratio(out)
    px = max(8.0, fps.badge_font_px * rr * max(0.1, float(style.font_scale)))
    x = 0 if raft_right is None else raft_right + max(4, int(round(8 * rr)))
    return fpv.draw_fp_badge(out, text, x, max(0, int(top)), px, fps)


# -- CF scale overlays -------------------------------------------------------------
def _fill_solid(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, bgr: tuple[int, int, int]) -> None:
    """Paint a solid, fully opaque ``bgr`` rectangle, clipped to the image (no-op if fully outside)."""
    h, w = out.shape[:2]
    xa, ya = max(0, min(x1, x2)), max(0, min(y1, y2))
    xb, yb = min(w, max(x1, x2)), min(h, max(y1, y2))
    if xa < xb and ya < yb:
        out[ya:yb, xa:xb] = bgr


def _draw_ruler_cf_bars(
    out: np.ndarray, detections: Sequence[dict], cf_px_per_cm: float,
    cfs: CFScalebarStyle, scale: float, skip: Optional[Any] = None,
) -> None:
    """Lay a 1 cm + 1 inch scale raft over every detected Ruler (``insert_cf_in_rulers``).

    Each raft is a solid ``raft_color`` rectangle carrying two solid bars -- one ``round(cf)`` px
    long in ``cm_color``, one ``round(cf * 2.54)`` px long in ``inch_color`` -- with a ``brim`` px
    gutter around and between them, so the raft is exactly the bars' bounding box plus the brim.

    The bars run along the ruler bbox's LONG axis (so they lie parallel to the ruler's own
    graduations); the raft is centered across the short axis and flush with the ruler's left edge
    on a landscape ruler, its top edge on a portrait one. Everything is drawn opaquely: a scale bar
    blended with the ruler underneath would be unmeasurable at its ends.

    ``cf_px_per_cm`` is already in the ORIGINAL frame; ``scale`` maps the working-frame boxes onto it.
    ``skip(d)`` true leaves that ruler bare -- a FieldPrism marker, whose app labels are its scale.
    """
    cm_px = max(1, int(round(cf_px_per_cm)))
    inch_px = max(1, int(round(cf_px_per_cm * _CM_PER_INCH)))
    brim = max(0, int(cfs.brim))
    res_ratio = _res_ratio(out)

    for d in detections:
        if str(d.get("source", "")) != "archival" or str(d.get("cls_name", "")) != _RULER_CLS:
            continue
        if skip is not None and skip(d):
            continue
        xyxy = d.get("xyxy")
        if not xyxy:
            continue
        x1, y1, x2, y2 = (float(v) * scale for v in xyxy)
        box_w, box_h = abs(x2 - x1), abs(y2 - y1)
        if box_w < 1 or box_h < 1:
            continue
        horizontal = box_w >= box_h
        short_len = min(box_w, box_h)

        # bar width across the ruler: the configured reference-resolution width, shrunk if the two
        # bars + three brims would not fit inside the ruler's short dimension.
        thick = max(1, _scaled_lw(cfs.bar_thickness, res_ratio))
        if 2 * thick + 3 * brim > short_len:
            thick = max(1, int((short_len - 3 * brim) // 2))
        raft_short = 2 * thick + 3 * brim

        # raft origin: flush with the near edge along the long axis, centered on the short axis
        if horizontal:
            rx, ry = int(round(x1)), int(round((y1 + y2) / 2.0 - raft_short / 2.0))
        else:
            rx, ry = int(round((x1 + x2) / 2.0 - raft_short / 2.0)), int(round(y1))
        _draw_raft(out, rx, ry, horizontal, thick, brim, cm_px, inch_px, cfs)


def _draw_corner_cf_raft(
    out: np.ndarray, cf_px_per_cm: float, cfs: CFScalebarStyle, top: int = 0,
) -> tuple[int, int]:
    """Lay ONE 1 cm + 1 inch raft in the top-left corner, for a CF predicted from megapixels (or on a
    FieldPrism sheet, whose markers carry no raft); return its far ``(right, bottom)`` (exclusive).

    The predicted CF came from no ruler on the sheet, so it must not be drawn over one -- a raft
    sitting on a detected ruler reads as "this ruler was measured". It goes flush left at ``top``
    (the bottom of the CF banner, which also lives in the corner; 0 when there is no banner),
    horizontal, with the same bars, colors and reference-scaled thickness as the ruler rafts.
    ``cf_px_per_cm`` is in the ORIGINAL frame.
    """
    cm_px = max(1, int(round(cf_px_per_cm)))
    inch_px = max(1, int(round(cf_px_per_cm * _CM_PER_INCH)))
    brim = max(0, int(cfs.brim))
    thick = max(1, _scaled_lw(cfs.bar_thickness, _res_ratio(out)))
    _draw_raft(out, 0, max(0, int(top)), True, thick, brim, cm_px, inch_px, cfs)
    return max(cm_px, inch_px) + 2 * brim, max(0, int(top)) + 2 * thick + 3 * brim


def _draw_raft(
    out: np.ndarray, rx: int, ry: int, horizontal: bool, thick: int, brim: int,
    cm_px: int, inch_px: int, cfs: CFScalebarStyle,
) -> None:
    """Paint one raft with its top-left corner at ``(rx, ry)``: a solid ``raft_color`` backing,
    then the 1 cm and 1 inch bars along its long axis, ``brim`` px apart and from the edges."""
    raft_short = 2 * thick + 3 * brim
    raft_long = max(cm_px, inch_px) + 2 * brim
    bars = ((0, cm_px, cfs.cm_color), (thick + brim, inch_px, cfs.inch_color))
    if horizontal:
        _fill_solid(out, rx, ry, rx + raft_long, ry + raft_short, _bgr(cfs.raft_color))
        for offset, length, color in bars:
            by = ry + brim + offset
            _fill_solid(out, rx + brim, by, rx + brim + length, by + thick, _bgr(color))
    else:
        _fill_solid(out, rx, ry, rx + raft_short, ry + raft_long, _bgr(cfs.raft_color))
        for offset, length, color in bars:
            bx = rx + brim + offset
            _fill_solid(out, bx, ry + brim, bx + thick, ry + brim + length, _bgr(color))


def _append_cf_exterior(
    out: np.ndarray, cf_px_per_cm: float, cfs: CFScalebarStyle, predicted: bool = False,
) -> np.ndarray:
    """Append a 1 cm checkerboard band to the top and left of the sheet (``insert_cf_exterior``).

    The band is ``exterior_cells`` cells thick and lives entirely OUTSIDE the original pixels -- the
    sheet is pasted at ``(margin, margin)`` on a larger canvas -- so it can never obscure content::

        O X O X O X O X ...
        X O X O X O X O ...
        O X
        X O
        ...

    One global checkerboard governs both bands (cell ``(i, j)`` is light when ``i + j`` is even), so
    the shared top-left corner agrees with itself and the two bands read as one continuous ring.

    Cell edges are ``round(k * cf)``, NOT ``k * round(cf)``: every boundary lands on a whole pixel
    (each cell is drawn exactly), while no rounding error accumulates along the band, so the 30th
    cell still starts at the true 30 cm mark.

    ``predicted`` (a CF predicted from megapixels, not measured from a ruler) swaps in
    ``exterior_light_predicted`` / ``exterior_dark_predicted`` -- black and 50% gray by default -- so
    a predicted scale is never mistaken for a measured one at a glance. Only the DARK cells are painted -- the canvas is
    pre-filled ``exterior_light`` -- which is also what makes the far edge behave as asked: a dark
    cell clipped by the image boundary is drawn cut short, and where the parity instead puts a light
    cell against the boundary nothing is drawn there at all, so no light cell ever renders truncated.
    """
    cells = max(1, int(cfs.exterior_cells))

    def edge(k: int) -> int:
        """Pixel offset of the k-th cell boundary from the outer corner of the band."""
        return int(round(k * cf_px_per_cm))

    margin = edge(cells)
    if margin < 1:                                  # a CF this small has no drawable cells
        return out
    h, w = out.shape[:2]
    new_h, new_w = h + margin, w + margin
    canvas = np.empty((new_h, new_w, 3), dtype=out.dtype)
    light = cfs.exterior_light_predicted if predicted else cfs.exterior_light
    canvas[:] = _bgr(light)
    canvas[margin:, margin:] = out

    dark = _bgr(cfs.exterior_dark_predicted if predicted else cfs.exterior_dark)
    for i in range(cells):                          # top band: `cells` rows across the full width
        ya, yb = edge(i), min(edge(i + 1), margin)
        j = 0
        while edge(j) < new_w:
            if (i + j) % 2:
                canvas[ya:yb, edge(j):min(edge(j + 1), new_w)] = dark
            j += 1
    for j in range(cells):                          # left band: `cells` columns down the full height
        xa, xb = edge(j), min(edge(j + 1), margin)
        i = 0
        while edge(i) < new_h:
            if (i + j) % 2:
                canvas[edge(i):min(edge(i + 1), new_h), xa:xb] = dark
            i += 1
    return canvas


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
def _res_ratio(out: np.ndarray) -> float:
    """Long side of the image relative to the reference overlay (font/line-width scaling anchor)."""
    h, w = out.shape[:2]
    return max(h, w) / float(_REF_LONG)


def _scaled_lw(line_width: int, res_ratio: float) -> int:
    """A configured line width (px at the reference resolution) scaled to the actual image size."""
    return max(1, int(round(line_width * res_ratio)))


def _fill_rect(out: np.ndarray, x1: float, y1: float, x2: float, y2: float,
               bgr: tuple[int, int, int], alpha: float) -> None:
    """Alpha-blend a solid ``bgr`` fill into the axis-aligned box region (clipped to the image)."""
    a = max(0.0, min(1.0, alpha))
    if a <= 0:
        return
    h, w = out.shape[:2]
    xa, ya = max(0, int(round(min(x1, x2)))), max(0, int(round(min(y1, y2))))
    xb, yb = min(w, int(round(max(x1, x2)))), min(h, int(round(max(y1, y2))))
    if xa >= xb or ya >= yb:
        return
    roi = out[ya:yb, xa:xb]
    layer = np.empty_like(roi)
    layer[:] = bgr
    cv2.addWeighted(layer, a, roi, 1.0 - a, 0.0, roi)


def _fill_poly(out: np.ndarray, pts: np.ndarray, bgr: tuple[int, int, int], alpha: float) -> None:
    """Alpha-blend a solid ``bgr`` fill inside an int polygon ``pts`` (Nx1x2), within its bbox."""
    a = max(0.0, min(1.0, alpha))
    if a <= 0 or len(pts) < 3:
        return
    x, y, bw, bh = cv2.boundingRect(pts)
    H, W = out.shape[:2]
    xa, ya, xb, yb = max(0, x), max(0, y), min(W, x + bw), min(H, y + bh)
    if xa >= xb or ya >= yb:
        return
    roi = out[ya:yb, xa:xb]
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts - np.array([[[xa, ya]]], dtype=np.int32)], 1)
    m = mask.astype(bool)
    if not m.any():
        return
    layer = np.empty_like(roi)
    layer[:] = bgr
    blended = cv2.addWeighted(layer, a, roi, 1.0 - a, 0.0)
    roi[m] = blended[m]


def _label_text(d: dict, cls_name: str, style: OverlayStyle) -> str:
    if style.draw_confidence and d.get("conf") is not None:
        return f"{cls_name} {float(d['conf']):.2f}"
    return cls_name


def _draw_centered_label(
    out: np.ndarray, cx: float, cy: float, text: str, box_w: float, box_h: float,
    angle_deg: float, res_ratio: float, style: OverlayStyle,
) -> None:
    """Draw ``text`` centered at ``(cx, cy)``, rotated by ``angle_deg``, auto-fit to the box but
    clamped to a resolution-relative font-scale range (scaled by ``style.font_scale``)."""
    if not text or box_w < 4 or box_h < 4:
        return
    fscale = max(0.05, float(style.font_scale))
    (w1, h1), bl1 = cv2.getTextSize(text, _FONT, 1.0, 2)
    th1 = h1 + bl1
    c, s = abs(math.cos(math.radians(angle_deg))), abs(math.sin(math.radians(angle_deg)))
    foot_w, foot_h = w1 * c + th1 * s, w1 * s + th1 * c
    fs = min(box_w * 0.88 / max(foot_w, 1e-3), box_h * 0.88 / max(foot_h, 1e-3))
    fs = float(np.clip(fs, _REF_FS_LO * res_ratio * fscale, _REF_FS_HI * res_ratio * fscale))
    thick = max(1, int(round(fs * 2)))
    (tw, th), bl = cv2.getTextSize(text, _FONT, fs, thick)
    pad = max(2, int(round(4 * fs)))
    canvas = np.zeros((th + bl + 2 * pad, tw + 2 * pad, 3), np.uint8)
    color = _bgr(style.label_text_color)
    # Draw WHITE so the canvas is a pure grayscale coverage map used as the alpha below; the label
    # is then composited TOWARD `color`. Drawing in `color` here would fold the color into its own
    # per-channel alpha and only render correctly for white.
    cv2.putText(canvas, text, (pad, th + pad), _FONT, fs, (255, 255, 255), thick, cv2.LINE_AA)
    if abs(angle_deg) > 0.5:
        h0, w0 = canvas.shape[:2]
        m = cv2.getRotationMatrix2D((w0 / 2.0, h0 / 2.0), angle_deg, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
        m[0, 2] += nw / 2.0 - w0 / 2.0
        m[1, 2] += nh / 2.0 - h0 / 2.0
        canvas = cv2.warpAffine(canvas, m, (nw, nh), flags=cv2.INTER_LINEAR)
    ch, cw = canvas.shape[:2]
    x0, y0 = int(round(cx - cw / 2.0)), int(round(cy - ch / 2.0))
    H, W = out.shape[:2]
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + cw), min(H, y0 + ch)
    if xa >= xb or ya >= yb:
        return
    sub = canvas[ya - y0:yb - y0, xa - x0:xb - x0].astype(np.float32) / 255.0
    roi = out[ya:yb, xa:xb].astype(np.float32)
    out[ya:yb, xa:xb] = (roi * (1.0 - sub) + np.asarray(color, np.float32) * sub).astype(np.uint8)


def _draw_boxes(
    out: np.ndarray, detections: Sequence[dict], style: OverlayStyle, scale: float,
    skip_plant_leaf: bool = False,
) -> None:
    """Draw the axis-aligned detection boxes, styled per group (leaf / plant / archival): fills
    first (under), then borders, then centered labels (on top)."""
    res_ratio = _res_ratio(out)
    items: list[tuple[dict, str, GroupStyle, float, float, float, float]] = []
    for d in detections:
        source = str(d.get("source", "plant"))
        cls_name = str(d.get("cls_name", ""))
        if skip_plant_leaf and source == "plant" and cls_name in _LEAF_DET_CLASSES:
            continue  # drawn as a rotated bounding box from Morphology instead
        if not style.show(cls_name):
            continue
        g = style.group_for(source, cls_name)
        if not g.visible:
            continue
        x1, y1, x2, y2 = (float(v) * scale for v in d["xyxy"])
        items.append((d, cls_name, g, x1, y1, x2, y2))

    for d, cls_name, g, x1, y1, x2, y2 in items:
        if g.fill:
            _fill_rect(out, x1, y1, x2, y2, _bgr(style.color_for(cls_name)), g.fill_alpha)
    for d, cls_name, g, x1, y1, x2, y2 in items:
        if g.border:
            cv2.rectangle(out, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                          _bgr(style.color_for(cls_name)), _scaled_lw(g.line_width, res_ratio), cv2.LINE_AA)
    if style.draw_labels:
        for d, cls_name, g, x1, y1, x2, y2 in items:
            bw, bh = x2 - x1, y2 - y1
            angle = 90.0 if bh > bw else 0.0   # tall boxes: rotate the label 90deg CCW
            _draw_centered_label(out, (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                                 _label_text(d, cls_name, style), bw, bh, angle, res_ratio, style)


# -- rotated (minimum) bounding boxes from Morphology ------------------------------
def _draw_rotated_boxes(out: np.ndarray, morphology: Sequence[Any], style: OverlayStyle, scale: float) -> None:
    """Draw each leaf's rotated minimum bounding box (LM2 procedure) as a tilted rectangle, styled by
    the ``leaf`` group, with a label centered along the box's long axis."""
    res_ratio = _res_ratio(out)
    g = style.group("leaf")
    boxes: list[tuple[str, np.ndarray, np.ndarray]] = []
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
        fcorners = np.asarray(corners, dtype=float) * scale
        pts = fcorners.round().astype(np.int32).reshape(-1, 1, 2)
        if len(pts) < 3:
            continue
        boxes.append((cls_name, fcorners, pts))

    for cls_name, _fc, pts in boxes:
        if g.fill:
            _fill_poly(out, pts, _bgr(style.color_for(cls_name)), g.fill_alpha)
    for cls_name, _fc, pts in boxes:
        if g.border:
            cv2.polylines(out, [pts], True, _bgr(style.color_for(cls_name)),
                          _scaled_lw(g.line_width, res_ratio), cv2.LINE_AA)
    if style.draw_labels:
        for cls_name, fc, _pts in boxes:
            if len(fc) < 4:
                continue
            cx, cy = float(fc[:, 0].mean()), float(fc[:, 1].mean())
            e0, e1 = fc[1] - fc[0], fc[2] - fc[1]
            n0, n1 = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
            long_e, long_len, short_len = (e0, n0, n1) if n0 >= n1 else (e1, n1, n0)
            ang = (math.degrees(math.atan2(long_e[1], long_e[0])) + 90) % 180 - 90   # keep upright
            _draw_centered_label(out, cx, cy, cls_name, long_len, short_len, -ang, res_ratio, style)


# -- conversion-factor banner ------------------------------------------------------
def cf_fallback_reason(lattice_image: Optional[dict]) -> Optional[str]:
    """Why a sheet fell back to the megapixel CF, as a short banner line; None if unknown.

    Read from the sheet's ``ruler_CF_lattice`` row. ``no_reading`` covers two different failures,
    told apart by the crop counts: every ruler was a type the lattice does not handle yet
    (``n_skipped``), or the rulers were attempted and could not be read (``n_failed``). A
    ``withheld`` sheet shows the rejected reading (working-frame px per cm, the banner's frame).
    """
    img = lattice_image or {}
    status = img.get("status")
    if status == "no_ruler":
        return "missing ruler"
    if status == "withheld":
        meas = img.get("cf_px_per_cm_measured")
        return "ruler failed validation" + ("" if meas is None else f" ({float(meas):.2f} px)")
    if status == "no_reading":
        n = int(img.get("n_ruler_crops") or 0)
        if n and int(img.get("n_skipped") or 0) == n:
            return "unsupported ruler"
        if n and int(img.get("n_failed") or 0) == n:
            return "unreadable ruler"
        return "unusable ruler"
    return None


def _draw_cf_banner(out: np.ndarray, cf_px_per_cm: float, style: OverlayStyle,
                    cf_source: Optional[str] = None, note: Optional[str] = None) -> int:
    """Print the conversion factor in a banner at the top-left of the sheet; return its bottom y.

    Line 1 names the CF's source -- "(measured from ruler)", "(measured from FieldPrism)" or
    "(predicted from megapixels)" -- so the number is never read as a measurement it isn't.
    ``note`` (why the prediction was used) becomes a second line in the same font; the banner grows
    to fit both."""
    right, bottom, lines, sizes, font_scale, thickness, pad, line_h = _cf_banner_layout(
        cf_px_per_cm, style, cf_source, note)
    cv2.rectangle(out, (0, 0), (right - 1, bottom), _bgr(style.cf_banner_color), -1)
    for k, ((_tw, th), _base) in enumerate(sizes):
        cv2.putText(
            out, lines[k], (pad, pad + k * (line_h + pad) + th), _FONT, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA,
        )
    return bottom + 1          # cv2.rectangle is inclusive of its far corner


def _cf_banner_layout(cf_px_per_cm: float, style: OverlayStyle, cf_source: Optional[str] = None,
                      note: Optional[str] = None):
    """The banner's text and geometry: ``(right, bottom, lines, sizes, font_scale, thickness, pad,
    line_h)``, ``right`` exclusive, ``bottom`` inclusive (as ``cv2.rectangle`` paints it)."""
    label = {CF_SOURCE_RULER: " (measured from ruler)",
             CF_SOURCE_FP: " (measured from FieldPrism)",
             CF_SOURCE_MP: " (predicted from megapixels)"}.get(cf_source, "")
    lines = [f"CF: {cf_px_per_cm:.2f} px/cm{label}"] + ([note] if note else [])
    font_scale = _BASE_FONT_SCALE * 1.4 * max(0.1, style.font_scale)
    thickness = max(1, int(round(font_scale * 2)))
    sizes = [cv2.getTextSize(t, _FONT, font_scale, thickness) for t in lines]
    pad = int(round(6 * max(0.5, style.font_scale)))
    line_h = max(th + base for (_tw, th), base in sizes)
    width = max(tw for (tw, _th), _base in sizes)
    bottom = len(lines) * line_h + (len(lines) - 1) * pad + 2 * pad
    return width + 2 * pad + 1, bottom, lines, sizes, font_scale, thickness, pad, line_h


# -- specimen segmentation overlay -------------------------------------------------
def _blend_mask(out: np.ndarray, mask_bool: np.ndarray, bgr: tuple[int, int, int], alpha: float) -> None:
    """Alpha-blend a solid ``bgr`` fill into the ``mask_bool`` region of ``out`` in place."""
    a = max(0.0, min(1.0, alpha))
    if a <= 0 or not mask_bool.any():
        return
    layer = out.copy()
    layer[mask_bool] = bgr
    cv2.addWeighted(layer, a, out, 1.0 - a, 0.0, out)


def build_specimen_overlay(
    image_bgr: np.ndarray,
    final_mask: np.ndarray,
    removed_mask: np.ndarray,
    centers: Sequence[Any],
    style: SpecimenStyle,
) -> np.ndarray:
    """Render the 2-panel ``Overlay_Specimen_Segmentation`` view onto the ORIGINAL image.

    LEFT panel: the sheet with the final specimen mask filled (``mask_color``), the paperclean
    removed region tinted (``refined_color``), and the paper-sampling locations boxed
    (``sample_box_color``). RIGHT panel: the final masked RGB cutout on black. ``final_mask`` /
    ``removed_mask`` are working-frame rasters ({0,255} or {0,1}); ``centers`` are paper-sample box
    centers in the working (mask) frame. Both are scaled onto the display panel here.
    """
    H, W = image_bgr.shape[:2]
    fh, fw = final_mask.shape[:2]
    md = int(style.display_max_dim)
    s = min(1.0, md / float(max(H, W))) if md > 0 else 1.0
    dw, dh = max(1, int(round(W * s))), max(1, int(round(H * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    img = cv2.resize(image_bgr, (dw, dh), interpolation=interp) if s != 1.0 else image_bgr.copy()
    fm = cv2.resize(final_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
    rm = cv2.resize(removed_mask.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0

    left = img.copy()
    _blend_mask(left, fm, _bgr(style.mask_color), style.mask_alpha)
    _blend_mask(left, rm, _bgr(style.refined_color), style.refined_alpha)
    if style.outline and fm.any():
        cnts, _ = cv2.findContours(fm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left, cnts, -1, _bgr(style.mask_color), max(1, int(0.0018 * max(dh, dw))))
    # blue paper-sampling boxes (centers are in the working/mask frame -> display frame)
    csx, csy = dw / float(max(1, fw)), dh / float(max(1, fh))
    bx = max(5, int(0.010 * max(dh, dw)))
    box_bgr = _bgr(style.sample_box_color)
    for c in centers or []:
        x, y = int(round(float(c[0]) * csx)), int(round(float(c[1]) * csy))
        cv2.rectangle(left, (x - bx, y - bx), (x + bx, y + bx), box_bgr, max(2, bx // 3), cv2.LINE_AA)

    right = np.zeros_like(img)
    right[fm] = img[fm]                                   # final masked RGB cutout on black
    return np.hstack([left, right])
