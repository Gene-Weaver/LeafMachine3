"""FieldPrism (FP) marker drawing shared by the Summary overlay and ``Overlay_FieldPrism``.

A clone of the FieldPrism app's FPfit overlay -- Android ``RulerDeskewPrecise.drawFpOverlay``
(RulerDeskewPrecise.kt:478-575) and iOS ``drawFpOverlayContent`` (RulerDeskewPrecise.swift:874-988)
-- drawn from the STORED ``ruler_FP_marker`` / ``ruler_FP_sheet`` rows, never from a live analysis.
Per marker, with ``S`` = that marker's px/cm:

* the PREDICTED center of the empty BR cell (``TR + BL - TL``) gets an axis-aligned square of side
  ``S``, filled green with a black ``1·s`` outline, drawn BEFORE the labels;
* "TL" red, "TR" yellow, "BL" white and "C" cyan, bold sans at ``0.70·S`` with a black shadow, each
  centered (by ink bounds, as Android does) on its detected square center. The squares themselves get
  no outline or dot -- text only, like the app;
* ``"1 cm = %.0f px"`` in white above the TL label at ``x = xTL - w/4``, baseline
  ``yTL - h(TL) - 5·s``. The app draws on an image it has already rotated upright; LM3 draws on the
  image as photographed, so the formula is applied to the marker's ON-IMAGE top-left cell (TL on an
  upright sheet, so identical there) and the line stays above the marker at 90/180/270 degrees.

LM3 additions (not in the app): a marker the CF does not use keeps its labels but its BR cell is only
outlined, in red, with a "rejected" note; a marker RECONSTRUCTED from the sheet geometry (a corner
the camera never measured) is drawn as dashed magenta cell outlines with dimmed labels and an
"inferred" note; and a sheet badge ("FieldPrism Letter | 4 markers") on black, magenta text -- the
app's only sheet-label precedent (CaptureActivity ``createFRfitOverlayBitmap``).

Everything is drawn onto OpenCV BGR arrays; :class:`FieldPrismStyle` colors are config RGB. Text is
rendered with PIL (DejaVu Sans Bold) because Hershey has no bold sans face and the app's labels are
the thing being cloned. All stored geometry is in the WORKING frame; ``scale`` maps it onto the
image drawn on (1.0 for every Reporter output).
"""
from __future__ import annotations

import json
import logging
import math
import os
from functools import lru_cache
from typing import Any, Callable, Optional, Sequence

import numpy as np

try:  # pragma: no cover - cv2 is always present at runtime
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from PIL import Image, ImageDraw, ImageFont

from leafmachine3.reporting.palette import RGB, FieldPrismStyle

log = logging.getLogger("leafmachine3.fieldprism_viz")

_REF_LONG = 2592                 # same reference long side as overlay._REF_LONG
_ROLES = ("TL", "TR", "C", "BL")
_LABEL_ORDER = ("TL", "TR", "BL", "C")   # the app's drawTinyLabel order
_LEGEND_BG_ALPHA = 150.0 / 255.0         # app COLOR_LEGEND_BG: black at alpha 150
_SHADOW_SIGMA = 2.0                      # app setShadowLayer(2f, 0, 0, BLACK)
_SHEET_AMBIGUITY_MM = 1.0                # fallback for fieldprism.AMBIGUITY_MARGIN_MM
# fallback display names when the sheet catalog cannot be read (sheet_type ids are case-insensitive)
_KNOWN_LABELS = {"letter": "Letter", "legal": "Legal", "tabloid": "Tabloid",
                 "a3": "A3", "a4": "A4", "a5": "A5"}


# -- record access -----------------------------------------------------------------
def fieldprism_from_record(record: Any) -> Optional[dict]:
    """``{"sheet": row | None, "markers": [rows]}`` from a ruler-lattice record, or ``None``.

    The record is ``ReportBundle.ruler_lattice`` (``{image, crops, fp_sheet, fp_markers}``). Records
    written before FieldPrism support have neither key, and a sheet without FP rulers has an empty
    marker list and no sheet row -- both return ``None`` so callers draw exactly what they always did.
    """
    if not record or not hasattr(record, "get"):
        return None
    sheet = record.get("fp_sheet")
    markers = [dict(m) for m in (record.get("fp_markers") or [])]
    if not sheet and not markers:
        return None
    return {"sheet": dict(sheet) if sheet else None, "markers": markers}


def has_fp_markers(fieldprism: Optional[dict]) -> bool:
    """True when the sheet carries at least one FieldPrism-classified ruler."""
    if not fieldprism:
        return False
    if fieldprism.get("markers"):
        return True
    sheet = fieldprism.get("sheet") or {}
    return int(_num(sheet.get("n_fp_detected")) or 0) > 0


def fp_box_matcher(fieldprism: Optional[dict]) -> Callable[[dict], bool]:
    """A predicate: is this overlay detection dict one of the sheet's FP markers?

    Only an archival ``Ruler`` box can be one: ``detection_id`` is unique only WITHIN a source
    (``archival_detection`` and ``plant_detection`` number independently), so a plant box or an
    archival box of another class is never a marker, whatever its id or coordinates. A dict
    without ``source`` / ``cls_name`` (an older bundle) is not excluded by the missing key.
    Matched by ``detection_id``. A detection dict without one (a bundle from before
    ``overlay_detections`` carried it) falls back to its box matching a marker row's stored
    ``x1..y2`` to half a pixel -- both come verbatim from ``archival_detection``.
    """
    markers = (fieldprism or {}).get("markers") or []
    ids = {int(m["detection_id"]) for m in markers if m.get("detection_id") is not None}
    boxes = [tuple(float(m[k]) for k in ("x1", "y1", "x2", "y2")) for m in markers
             if all(m.get(k) is not None for k in ("x1", "y1", "x2", "y2"))]

    def is_fp(d: dict) -> bool:
        # guards BOTH the id path and the coordinate fallback below
        if str(d.get("source", "archival")) != "archival" or str(d.get("cls_name", "Ruler")) != "Ruler":
            return False
        did = d.get("detection_id")
        if did is not None:
            return int(did) in ids
        xyxy = d.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            return False
        return any(all(abs(float(a) - b) <= 0.5 for a, b in zip(xyxy, box)) for box in boxes)

    return is_fp


# -- the sheet badge ---------------------------------------------------------------
def fp_badge_text(sheet: Optional[dict], markers: Sequence[dict] = ()) -> Optional[str]:
    """The one-line sheet badge, e.g. ``FieldPrism Letter | 4 markers``.

    ``identified``    -> ``FieldPrism Letter | 4 markers`` / ``FieldPrism Letter | 3 + 1 inferred``
    ``ambiguous``     -> ``FieldPrism Letter? (or Legal)``
    ``undetermined``  -> ``FieldPrism sheet: undetermined | 1 marker`` (fewer than 2 usable markers)
    ``unrecognized``  -> ``FieldPrism sheet: unrecognized``
    Marker counts come from the stored corners (observed vs reconstructed); without them, from the
    ``n_fp_used`` / ``n_fp_inferred`` counts, and with no sheet row at all from the marker verdicts.
    """
    if not sheet:
        if not markers:
            return None
        n = sum(1 for m in markers if _is_used(m))
        return f"FieldPrism sheet: unknown | {_count(n, 0)}"
    status = str(sheet.get("sheet_status") or "")
    corners = _parse_json(sheet.get("corners_json")) or {}
    if isinstance(corners, dict) and corners:
        n_obs = sum(1 for c in corners.values() if isinstance(c, dict) and c.get("observed"))
        n_inf = sum(1 for c in corners.values()
                    if isinstance(c, dict) and not c.get("observed") and c.get("squares"))
    else:
        used = sheet.get("n_fp_used")
        n_obs = int(_num(used)) if used is not None else sum(1 for m in markers if _is_used(m))
        n_inf = int(_num(sheet.get("n_fp_inferred")) or 0)
    if status == "identified":
        return f"FieldPrism {_sheet_name(sheet.get('sheet_label'), sheet.get('sheet_type'))} | {_count(n_obs, n_inf)}"
    if status == "ambiguous":
        best, alts = _ambiguous_names(sheet)
        if best is None:
            return "FieldPrism sheet: ambiguous"
        return f"FieldPrism {best}?" + (f" (or {', '.join(alts)})" if alts else "")
    if status == "unrecognized":
        return "FieldPrism sheet: unrecognized"
    return f"FieldPrism sheet: {status or 'undetermined'} | {_count(n_obs, n_inf)}"


def _count(n_obs: int, n_inf: int) -> str:
    if n_inf:
        return f"{n_obs} + {n_inf} inferred"
    return f"{n_obs} marker" + ("" if n_obs == 1 else "s")


def _ambiguous_names(sheet: dict) -> tuple[Optional[str], list[str]]:
    """The best-ranked sheet name and the other sheet types still within the ambiguity margin."""
    cands = [c for c in (_parse_json(sheet.get("sheet_candidates_json")) or []) if isinstance(c, dict)]
    best_type = sheet.get("sheet_type") or (cands[0].get("sheet_type") if cands else None)
    if best_type is None:
        return None, []
    best = _sheet_name(sheet.get("sheet_label") if sheet.get("sheet_type") else None, best_type)
    costs = [_num(c.get("cost_mm")) for c in cands if _num(c.get("cost_mm")) is not None]
    limit = (min(costs) + _ambiguity_margin()) if costs else math.inf
    alts: list[str] = []
    for c in cands:
        t = c.get("sheet_type")
        cost = _num(c.get("cost_mm"))
        if t is None or str(t).lower() == str(best_type).lower() or (cost is not None and cost > limit):
            continue
        name = _sheet_name(None, t)
        if name not in alts:
            alts.append(name)
    return best, alts[:2]


def _ambiguity_margin() -> float:
    try:
        from leafmachine3.inference.ruler_lattice import fieldprism as _fp
        return float(getattr(_fp, "AMBIGUITY_MARGIN_MM", _SHEET_AMBIGUITY_MM))
    except Exception:
        return _SHEET_AMBIGUITY_MM


def _sheet_name(label: Any, sheet_type: Any) -> str:
    """The display name of a sheet: its stored label, else the catalog's, else a known name."""
    if label:
        return str(label)
    if sheet_type is None:
        return "?"
    key = str(sheet_type)
    cat = _catalog_labels()
    return cat.get(key) or cat.get(key.lower()) or _KNOWN_LABELS.get(key.lower(), key)


@lru_cache(maxsize=1)
def _catalog_labels() -> dict:
    """``{sheet_type: label}`` from the FieldPrism sheet catalog; empty if it cannot be read."""
    try:
        from leafmachine3.inference.ruler_lattice.fieldprism import load_sheet_catalog
        cat = load_sheet_catalog()
    except Exception:
        return {}
    sheets = cat.get("sheets", cat) if isinstance(cat, dict) else cat
    out: dict = {}
    items = sheets.items() if isinstance(sheets, dict) else (
        (s.get("sheet_type") or s.get("id") or s.get("key"), s) for s in (sheets or []) if isinstance(s, dict))
    for key, spec in items:
        if key is not None and isinstance(spec, dict) and spec.get("label"):
            out[str(key)] = str(spec["label"])
            out[str(key).lower()] = str(spec["label"])
    return out


# -- drawing: markers --------------------------------------------------------------
def draw_fp_markers(
    out: np.ndarray, fieldprism: dict, style: FieldPrismStyle, scale: float = 1.0,
    *, draw_inferred: bool = True, avoid: Sequence[Sequence[float]] = (),
) -> None:
    """Draw every FP marker app-style onto ``out`` (BGR, in place).

    Reconstructed markers go first, so a measured marker's labels always sit on top. A marker that
    was not measured, or failed the per-marker geometry checks (``valid`` false: its square roles
    cannot be trusted), gets nothing beyond its detection box -- its corner is drawn as inferred
    when the sheet fit reconstructed it.

    ``avoid`` lists ``(x1, y1, x2, y2)`` boxes already holding text (the CF banner, the corner raft,
    the sheet badge, the legend). A marker's "1 cm =" / "inferred" line that would land on one moves
    below the marker instead of being painted over it -- the app has the same collision between its
    legend and the top-left marker's line, and simply overdraws.
    """
    s = _res_ratio(out)
    sheet = fieldprism.get("sheet") or {}
    if draw_inferred and sheet:
        for corner in _inferred_corners(sheet):
            _draw_inferred_marker(out, corner, style, scale, s, avoid)
    for m in fieldprism.get("markers") or []:
        sq = marker_squares(m)
        if sq is None or str(m.get("status") or "") != "measured" or _is_invalid(m):
            continue
        sq = {k: (x * scale, y * scale) for k, (x, y) in sq.items()}
        pxcm = _num(m.get("pxcm"))
        S = pxcm * scale if pxcm is not None and pxcm > 0 else _pitch_side(sq)
        if S > 1:
            _draw_measured_marker(out, m, sq, S, style, s, avoid)


def marker_squares(m: dict) -> Optional[dict]:
    """``{TL, TR, C, BL, BR: (x, y)}`` working-frame square centers of a marker row, or ``None``.

    BR is the stored prediction, or ``TR + BL - TL`` when an older row lacks it."""
    pts: dict = {}
    for role in _ROLES:
        x, y = _num(m.get(f"{role.lower()}_x")), _num(m.get(f"{role.lower()}_y"))
        if x is None or y is None:
            return None
        pts[role] = (x, y)
    bx, by = _num(m.get("br_x")), _num(m.get("br_y"))
    if bx is None or by is None:
        bx = pts["TR"][0] + pts["BL"][0] - pts["TL"][0]
        by = pts["TR"][1] + pts["BL"][1] - pts["TL"][1]
    pts["BR"] = (bx, by)
    return pts


def _pitch_side(sq: dict) -> float:
    """1 cm in px from a marker's square centers: half the mean TL->TR / TL->BL pitch (20 mm)."""
    a = math.dist(sq["TL"], sq["TR"])
    b = math.dist(sq["TL"], sq["BL"])
    return (a + b) / 4.0


def _is_used(m: dict) -> bool:
    v = m.get("verdict")
    if v is not None:
        return str(v) == "used"
    return bool(m.get("valid")) and str(m.get("status") or "") == "measured"


def _is_invalid(m: dict) -> bool:
    """Measured, but explicitly failed the per-marker checks (``valid`` stored as 0/False)."""
    v = m.get("valid")
    return v is not None and not bool(v)


class _Lines:
    """Stacks extra text lines under a marker, falling back to beside it at the image's bottom.

    "Under" and "beside" are taken from ALL five cells on the image, not from the BL/BR roles: on a
    sheet photographed at 90/180/270 degrees the role rows are not the image rows."""

    def __init__(self, out: np.ndarray, sq: dict, S: float):
        self.out, self.sq, self.S = out, sq, S
        self.y = max(p[1] for p in sq.values()) + S / 2.0       # bottom edge of the marker's cells
        self.side = 0

    def draw(self, text: str, px: float, rgb: RGB, alpha: float = 1.0) -> None:
        out, sq, S = self.out, self.sq, self.S
        w, h = _ink_size(text, px)
        gap = 0.35 * S
        if self.y + gap + h <= out.shape[0] - 2:
            self.y += gap + h
            cx = (sq["TL"][0] + sq["BR"][0]) / 2.0
            draw_text(out, text, (_clamp_x(out, cx - w / 2.0, w), self.y), px, rgb,
                      mode="baseline", alpha=alpha)
            return
        x = max(p[0] for p in sq.values()) + S / 2.0 + gap          # bottom-edge marker: beside it
        if x + w > out.shape[1] - 2:
            x = min(p[0] for p in sq.values()) - S / 2.0 - gap - w
        y = sq["C"][1] + h / 2.0 + self.side * (h + gap)
        self.side += 1
        draw_text(out, text, (_clamp_x(out, x, w), y), px, rgb, mode="baseline", alpha=alpha)


def _draw_top_line(out: np.ndarray, text: str, sq: dict, px: float, rgb: RGB, s: float,
                   avoid: Sequence[Sequence[float]], below: _Lines, alpha: float = 1.0) -> None:
    """The app's per-marker line above the marker (see :func:`_top_line_box`), kept inside the
    image; moved under the marker if it hits ``avoid`` or the marker's own labels / BR cell."""
    x, y, box = _top_line_box(out, text, sq, px, s)
    blocked = (*avoid, *_own_text_boxes(sq, px, below.S))
    if box[1] < 0 or any(_overlaps(box, r) for r in blocked):
        below.draw(text, px, rgb, alpha)
    else:
        draw_text(out, text, (x, y), px, rgb, mode="baseline", alpha=alpha)


def _top_left_cell(sq: dict) -> tuple[float, float]:
    """Center of the marker's ON-IMAGE top-left corner cell: of the four corner cells (TL, TR, BL
    and the empty BR) the one with the smallest ``x + y``.

    The app draws on an image it has already rotated upright, where that cell IS TL. LM3 draws on
    the image as photographed: on a sheet at 90/180/270 degrees TL sits at another corner of the
    marker, and "above TL" lands on the marker's middle row. Ties go to TL, and on an upright
    marker (tilt under 45 degrees) TL always wins, so the upright output is the app's exactly."""
    return min((sq[r] for r in ("TL", "TR", "BL", "BR") if r in sq), key=lambda p: p[0] + p[1])


def _own_text_boxes(sq: dict, px: float, S: float) -> list[tuple]:
    """The ink boxes of a marker's own TL/TR/BL/C labels and the middle half of its BR cell --
    what its "1 cm =" / "inferred" line must never cover."""
    boxes = [_centered_ink_box(r, sq[r], px) for r in _LABEL_ORDER if r in sq]
    if "BR" in sq:
        q, (bx, by) = S / 4.0, sq["BR"]
        boxes.append((bx - q, by - q, bx + q, by + q))
    return boxes


def _centered_ink_box(text: str, center: tuple[float, float], px: float) -> tuple[float, ...]:
    """The ink box ``draw_text(..., mode="center")`` gives ``text`` at ``center`` (no drawing)."""
    l, t, r, b = _font(_font_px(px)).getbbox(text, anchor="ls")
    ox, oy = center[0] - (l + r) / 2.0, center[1] - (t + b) / 2.0
    return (ox + l, oy + t, ox + r, oy + b)


def _top_line_box(out: np.ndarray, text: str, sq: dict, px: float, s: float):
    """Origin ``(x, y)`` and approximate box of a marker's line above its top row.

    The app's formula (``x = xTL - w/4``, baseline ``yTL - h(TL) - 5·s``) applied to the marker's
    on-image top-left cell (:func:`_top_left_cell`) instead of the TL role: identical to the app on
    an upright sheet, and still above the marker on a rotated one."""
    w, h = _ink_size(text, px)
    tl_h = _ink_size("TL", px)[1]
    ax, ay = _top_left_cell(sq)
    x, y = _clamp_x(out, ax - w / 4.0, w), ay - tl_h - 5.0 * s
    return x, y, (x, y - h, x + w, y)


def _marker_text_boxes(out: np.ndarray, fieldprism: dict, style: FieldPrismStyle,
                       scale: float) -> list[tuple]:
    """Where every drawn marker's line above its top row would go (before any relocation)."""
    s = _res_ratio(out)
    boxes = []
    for corner in _inferred_corners(fieldprism.get("sheet") or {}):
        sq = {k: (x * scale, y * scale) for k, (x, y) in corner["squares"].items()}
        S = _pitch_side(sq)
        if S > 1:
            boxes.append(_top_line_box(out, "inferred", sq, style.text_size_frac * S, s)[2])
    for m in fieldprism.get("markers") or []:
        sq = marker_squares(m)
        if sq is None or str(m.get("status") or "") != "measured" or _is_invalid(m):
            continue
        sq = {k: (x * scale, y * scale) for k, (x, y) in sq.items()}
        pxcm = _num(m.get("pxcm"))
        S = pxcm * scale if pxcm is not None and pxcm > 0 else _pitch_side(sq)
        if S > 1:
            boxes.append(_top_line_box(out, "1 cm = %.0f px" % S, sq, style.text_size_frac * S, s)[2])
    return boxes


def _draw_measured_marker(out: np.ndarray, m: dict, sq: dict, S: float,
                          style: FieldPrismStyle, s: float, avoid: Sequence[Sequence[float]]) -> None:
    used = _is_used(m)
    lw = max(1, int(round(s)))
    x1, y1, x2, y2 = _cell(sq["BR"], S)
    if used:                                       # app: fill (no AA) then a black 1*s stroke
        _fill(out, x1, y1, x2, y2, _bgr(style.br_used))
        _rect(out, x1, y1, x2, y2, (0, 0, 0), lw)
    else:                                          # LM3: a marker the CF does not use -> red outline
        _rect(out, x1, y1, x2, y2, _bgr(style.br_rejected), max(2, int(round(2 * s))))

    px = style.text_size_frac * S
    for role in _LABEL_ORDER:
        draw_text(out, role, sq[role], px, style.label_color(role), mode="center")
    below = _Lines(out, sq, S)
    _draw_top_line(out, "1 cm = %.0f px" % S, sq, px, (255, 255, 255), s, avoid, below)
    if not used:
        below.draw("rejected" if str(m.get("verdict") or "") == "rejected" else "not used",
                   px * 0.8, style.br_rejected)


def _inferred_corners(sheet: dict) -> list[dict]:
    """The sheet corners whose marker was RECONSTRUCTED (``observed`` false) and has square centers."""
    corners = _parse_json(sheet.get("corners_json")) or {}
    out = []
    if not isinstance(corners, dict):
        return out
    for name, c in corners.items():
        if not isinstance(c, dict) or c.get("observed"):
            continue
        squares = c.get("squares") or {}
        try:
            sq = {r: (float(squares[r][0]), float(squares[r][1])) for r in (*_ROLES, "BR")}
        except Exception:
            continue
        out.append({"corner": name, "squares": sq})
    return out


def _draw_inferred_marker(out: np.ndarray, corner: dict, style: FieldPrismStyle, scale: float,
                          s: float, avoid: Sequence[Sequence[float]]) -> None:
    """A reconstructed marker: dashed outlines of its four filled cells and the empty BR cell,
    dimmed labels, and "inferred" where a measured marker carries its "1 cm =" line."""
    sq = {k: (x * scale, y * scale) for k, (x, y) in corner["squares"].items()}
    S = _pitch_side(sq)
    if S <= 1:
        return
    color = _bgr(style.inferred)
    lw = max(1, int(round(2 * s)))
    for role in (*_ROLES, "BR"):
        _dashed_rect(out, *_cell(sq[role], S), color, lw, dash=max(4.0, S / 6.0))
    px = style.text_size_frac * S
    for role in _LABEL_ORDER:
        draw_text(out, role, sq[role], px, style.label_color(role), mode="center",
                  alpha=style.inferred_label_alpha)
    _draw_top_line(out, "inferred", sq, px, style.inferred, s, avoid, _Lines(out, sq, S))


def draw_page_outline(out: np.ndarray, sheet: Optional[dict], style: FieldPrismStyle,
                      scale: float = 1.0) -> None:
    """The identified page's outline (``page_corners_json``: TL, TR, BR, BL), thin magenta."""
    pts = _parse_json((sheet or {}).get("page_corners_json"))
    try:
        poly = np.asarray(pts, dtype=float).reshape(-1, 2) * scale
    except Exception:
        return
    if len(poly) < 3 or not np.all(np.isfinite(poly)):
        return
    lw = max(1, int(round(2 * _res_ratio(out))))
    cv2.polylines(out, [np.round(poly).astype(np.int32).reshape(-1, 1, 2)], True,
                  _bgr(style.inferred), lw, cv2.LINE_AA)


# -- drawing: badge + legend -------------------------------------------------------
def fp_badge_box(text: str, x: float, y: float, px: float) -> tuple[int, int, int, int]:
    """The ``(x1, y1, x2, y2)`` a badge of ``text`` at ``px`` occupies with its top-left at ``(x, y)``."""
    w, h = _ink_size(text, px)
    pad = max(2, int(round(0.35 * px)))
    x1, y1 = int(round(x)), int(round(y))
    return x1, y1, x1 + int(round(w)) + 2 * pad, y1 + int(round(h)) + 2 * pad


def draw_fp_badge(out: np.ndarray, text: str, x: float, y: float, px: float,
                  style: FieldPrismStyle) -> tuple[int, int, int, int]:
    """Paint the sheet badge with its top-left at ``(x, y)``; return its ``(x1, y1, x2, y2)``.

    Black at ``badge_alpha`` behind ``badge_text``-colored bold text (no shadow: it sits on black)."""
    x1, y1, x2, y2 = fp_badge_box(text, x, y, px)
    _blend_rect(out, x1, y1, x2, y2, (0, 0, 0), style.badge_alpha)
    draw_text(out, text, ((x1 + x2) / 2.0, (y1 + y2) / 2.0), px, style.badge_text,
              mode="center", shadow=False)
    return x1, y1, x2, y2


def fp_legend_box(out: np.ndarray, pxcm: float, style: FieldPrismStyle) -> tuple[int, int, int, int]:
    """The ``(x1, y1, x2, y2)`` of the app legend for ``pxcm`` on ``out``."""
    s = _res_ratio(out)
    w, h = _ink_size("1 cm = %.1f px" % pxcm, style.text_size_frac * pxcm)
    margin, pad = 20.0 * s, 16.0 * s
    return (int(round(margin)), int(round(margin)),
            int(round(margin + w + 2 * pad)), int(round(margin + h + 2 * pad)))


def draw_fp_legend(out: np.ndarray, pxcm: float, style: FieldPrismStyle) -> tuple[int, int, int, int]:
    """The app's global legend: ``"1 cm = %.1f px"`` in a rounded black (alpha 150) box at
    ``(20, 20)·s``, padded ``16·s``, corner radius ``12·s``, text ``0.70·S``. Returns the box."""
    s = _res_ratio(out)
    x1, y1, x2, y2 = fp_legend_box(out, pxcm, style)
    _blend_rounded_rect(out, x1, y1, x2, y2, int(round(12 * s)), (0, 0, 0), _LEGEND_BG_ALPHA)
    draw_text(out, "1 cm = %.1f px" % pxcm, ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
              style.text_size_frac * pxcm, (255, 255, 255), mode="center")
    return x1, y1, x2, y2


def fp_anchor_pxcm(fieldprism: dict, cf_px_per_cm: Optional[float] = None) -> Optional[float]:
    """The px/cm the FP legend reports: the sheet's FieldPrism CF, else ``cf_px_per_cm``, else the
    marker mean, else the mean of the used markers' px/cm (all working frame)."""
    sheet = fieldprism.get("sheet") or {}
    for v in (sheet.get("cf_px_per_cm_fp"), cf_px_per_cm, sheet.get("cf_px_per_cm_marker_mean")):
        if _num(v) is not None and float(v) > 0:
            return float(v)
    vals = [float(m["pxcm"]) for m in fieldprism.get("markers") or []
            if _is_used(m) and _num(m.get("pxcm")) is not None]
    return float(np.mean(vals)) if vals else None


def build_fieldprism_overlay(
    image_bgr: np.ndarray, fieldprism: dict, style: Optional[FieldPrismStyle] = None,
    *, cf_px_per_cm: Optional[float] = None, work_scale: float = 1.0,
) -> np.ndarray:
    """Render ``Overlay_FieldPrism``: the app's FPfit overlay on a copy of ``image_bgr``.

    The page outline (thin magenta) under everything, then the reconstructed and measured markers,
    then the app legend top-left and the sheet badge. The badge goes directly right of the legend,
    in the legend's text size shrunk (to no less than half) to end before the next FP box; only when
    it cannot fit there does it go under the legend -- on an FPfit crop the top-left marker starts
    right below the legend, so "under" would cover it.
    ``cf_px_per_cm`` (working frame) is the legend's fallback when the sheet has no FP CF.
    """
    st = style or FieldPrismStyle()
    out = image_bgr.copy()
    scale = 1.0 / float(work_scale or 1.0)
    sheet = fieldprism.get("sheet")
    markers = fieldprism.get("markers") or []
    s = _res_ratio(out)
    pxcm = fp_anchor_pxcm(fieldprism, cf_px_per_cm)
    pxcm = pxcm * scale if pxcm else None

    legend = fp_legend_box(out, pxcm, st) if pxcm else None
    text = fp_badge_text(sheet, markers)
    badge = None
    if text:
        px = st.text_size_frac * pxcm if pxcm else st.badge_font_px * s
        if legend is None:
            badge = fp_badge_box(text, 20.0 * s, 20.0 * s, px)
        else:
            # obstacles: every FP box and every marker's own "1 cm =" / "inferred" line
            taken = [tuple(float(m[k]) * scale for k in ("x1", "y1", "x2", "y2")) for m in markers
                     if all(_num(m.get(k)) is not None for k in ("x1", "y1", "x2", "y2"))]
            taken = [b for b in taken + _marker_text_boxes(out, fieldprism, st, scale)
                     if not _overlaps(b, legend)]
            x0 = legend[2] + 10.0 * s
            # right of the legend, shrunk (to half the legend's text size at most) to end before
            # the first obstacle in the legend's band, or the image edge
            limit = min([float(out.shape[1]) - 10.0 * s]
                        + [b[0] - 10.0 * s for b in taken
                           if b[2] > x0 and b[1] < legend[3] and b[3] > legend[1]])
            full = fp_badge_box(text, x0, legend[1], px)
            fit = px * min(1.0, (limit - x0) / max(1.0, float(full[2] - full[0])))
            if fit >= 0.5 * px:
                px = fit
                badge = fp_badge_box(text, x0, legend[1], px)
            if badge is None or any(_overlaps(badge, b) for b in taken):
                px = st.text_size_frac * pxcm
                badge = fp_badge_box(text, legend[0], legend[3] + 10.0 * s, px)

    draw_page_outline(out, sheet, st, scale)
    draw_fp_markers(out, fieldprism, st, scale, avoid=[r for r in (legend, badge) if r])
    if legend is not None:
        draw_fp_legend(out, pxcm, st)
    if badge is not None:
        draw_fp_badge(out, text, badge[0], badge[1], px, st)
    return out


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _clamp_x(out: np.ndarray, x: float, w: float) -> float:
    """Keep a ``w``-wide line inside the image horizontally (the app lets edge markers' text clip)."""
    return max(2.0, min(float(x), out.shape[1] - 2.0 - w))


# -- primitives --------------------------------------------------------------------
_FONT_DIR = "/usr/share/fonts/truetype/dejavu"


@lru_cache(maxsize=128)
def _font(px: int) -> ImageFont.FreeTypeFont:
    """Bold sans at ``px`` (em size): DejaVu Sans Bold, as ``ruler_lattice.qc._font`` finds DejaVu --
    the system copy, else matplotlib's bundled one -- with the mono face as a last resort."""
    candidates = [os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf"),
                  os.path.join(_FONT_DIR, "DejaVuSansMono-Bold.ttf")]
    try:
        import matplotlib
        mp = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        candidates += [os.path.join(mp, "DejaVuSans-Bold.ttf"), os.path.join(mp, "DejaVuSansMono-Bold.ttf")]
    except Exception:
        pass
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, px)
    return ImageFont.load_default(px)


def _font_px(px: float) -> int:
    return max(6, int(round(px)))


def _ink_size(text: str, px: float) -> tuple[float, float]:
    l, t, r, b = _font(_font_px(px)).getbbox(text, anchor="ls")
    return float(r - l), float(b - t)


def draw_text(
    out: np.ndarray, text: str, org: tuple[float, float], px: float, rgb: RGB, *,
    mode: str = "center", alpha: float = 1.0, shadow: bool = True, measure_only: bool = False,
) -> tuple[float, float, float, float]:
    """Draw ``text`` (bold sans, ``px`` em size) onto BGR ``out``; return its ink box.

    ``mode="center"`` centers the INK bounds on ``org`` (Android: ``x - w/2``, baseline ``y + h/2``);
    ``mode="baseline"`` puts the left of the text origin and the baseline at ``org``. The black
    shadow is the app's (blur 2, offset 0), strengthened slightly so white text stays readable on
    white paper. ``measure_only`` returns the box without drawing.
    """
    font = _font(_font_px(px))
    l, t, r, b = font.getbbox(text, anchor="ls")
    if mode == "center":
        ox, oy = org[0] - (l + r) / 2.0, org[1] - (t + b) / 2.0
    else:
        ox, oy = float(org[0]), float(org[1])
    ink = (ox + l, oy + t, ox + r, oy + b)
    if measure_only or not text:
        return ink
    m = int(math.ceil(3 * _SHADOW_SIGMA)) + 2 if shadow else 1
    X0, Y0 = int(math.floor(ox + l)) - m, int(math.floor(oy + t)) - m
    fx, fy = ox - math.floor(ox), oy - math.floor(oy)          # keep the sub-pixel origin
    w, h = int(r - l) + 2 * m + 2, int(b - t) + 2 * m + 2
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).text((m - l + fx, m - t + fy), text, font=font, fill=255, anchor="ls")
    cov = np.asarray(canvas, dtype=np.float32) / 255.0

    H, W = out.shape[:2]
    xa, ya, xb, yb = max(0, X0), max(0, Y0), min(W, X0 + w), min(H, Y0 + h)
    if xa >= xb or ya >= yb:
        return ink
    cov = cov[ya - Y0:yb - Y0, xa - X0:xb - X0]
    a = max(0.0, min(1.0, float(alpha)))
    roi = out[ya:yb, xa:xb].astype(np.float32)
    if shadow:
        full = np.asarray(canvas, dtype=np.uint8)
        halo = cv2.dilate(full, np.ones((3, 3), np.uint8))
        sh = cv2.GaussianBlur(halo.astype(np.float32) / 255.0, (0, 0), _SHADOW_SIGMA)
        sh = np.clip(sh * 1.5, 0.0, 1.0)[ya - Y0:yb - Y0, xa - X0:xb - X0]
        roi *= (1.0 - a * sh)[..., None]
    k = (a * cov)[..., None]
    color = np.asarray(_bgr(rgb), np.float32)
    roi = roi * (1.0 - k) + color * k
    out[ya:yb, xa:xb] = np.clip(np.round(roi), 0, 255).astype(np.uint8)
    return ink


def _cell(center: tuple[float, float], S: float) -> tuple[int, int, int, int]:
    """Integer ``(x1, y1, x2, y2)`` (exclusive far edge) of the side-``S`` square at ``center``."""
    half = S / 2.0
    return (int(round(center[0] - half)), int(round(center[1] - half)),
            int(round(center[0] + half)), int(round(center[1] + half)))


def _fill(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, bgr) -> None:
    H, W = out.shape[:2]
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if xa < xb and ya < yb:
        out[ya:yb, xa:xb] = bgr


def _rect(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, bgr, lw: int) -> None:
    """Outline painted INSIDE the cell ``[x1, x2) x [y1, y2)``, ``lw`` px wide, no antialiasing."""
    lw = max(1, min(lw, (x2 - x1) // 2 or 1, (y2 - y1) // 2 or 1))
    _fill(out, x1, y1, x2, y1 + lw, bgr)
    _fill(out, x1, y2 - lw, x2, y2, bgr)
    _fill(out, x1, y1, x1 + lw, y2, bgr)
    _fill(out, x2 - lw, y1, x2, y2, bgr)


def _dashed_rect(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, bgr, lw: int,
                 dash: float) -> None:
    """A dashed outline of ``[x1, x2) x [y1, y2)``; dashes start at every corner."""
    pts = [(x1, y1), (x2 - 1, y1), (x2 - 1, y2 - 1), (x1, y2 - 1)]
    for (ax, ay), (bx, by) in zip(pts, pts[1:] + pts[:1]):
        length = math.hypot(bx - ax, by - ay)
        if length < 1:
            continue
        n = max(1, int(round(length / (2 * dash))))
        step = length / n
        for k in range(n):
            t0, t1 = k * step / length, (k * step + step / 2.0) / length
            p0 = (int(round(ax + (bx - ax) * t0)), int(round(ay + (by - ay) * t0)))
            p1 = (int(round(ax + (bx - ax) * t1)), int(round(ay + (by - ay) * t1)))
            cv2.line(out, p0, p1, bgr, lw, cv2.LINE_8)


def _blend_rect(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, bgr, alpha: float) -> None:
    H, W = out.shape[:2]
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    a = max(0.0, min(1.0, float(alpha)))
    if xa >= xb or ya >= yb or a <= 0:
        return
    roi = out[ya:yb, xa:xb].astype(np.float32)
    out[ya:yb, xa:xb] = np.round(roi * (1 - a) + np.asarray(bgr, np.float32) * a).astype(np.uint8)


def _blend_rounded_rect(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, r: int,
                        bgr, alpha: float) -> None:
    H, W = out.shape[:2]
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if xa >= xb or ya >= yb:
        return
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(mask, (r, 0), (x2 - x1 - 1 - r, y2 - y1 - 1), 255, -1)
    cv2.rectangle(mask, (0, r), (x2 - x1 - 1, y2 - y1 - 1 - r), 255, -1)
    for cx, cy in ((r, r), (x2 - x1 - 1 - r, r), (r, y2 - y1 - 1 - r), (x2 - x1 - 1 - r, y2 - y1 - 1 - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
    k = (mask[ya - y1:yb - y1, xa - x1:xb - x1].astype(np.float32) / 255.0 * float(alpha))[..., None]
    roi = out[ya:yb, xa:xb].astype(np.float32)
    out[ya:yb, xa:xb] = np.round(roi * (1 - k) + np.asarray(bgr, np.float32) * k).astype(np.uint8)


def _res_ratio(out: np.ndarray) -> float:
    h, w = out.shape[:2]
    return max(h, w) / float(_REF_LONG)


def _bgr(color: RGB) -> tuple[int, int, int]:
    r, g, b = (int(c) for c in color)
    return (b, g, r)


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _parse_json(v: Any) -> Any:
    """A JSON column as stored (string) or already parsed; ``None`` when unreadable."""
    if v is None or isinstance(v, (dict, list)):
        return v
    try:
        return json.loads(v)
    except Exception:
        log.debug("unreadable FieldPrism JSON column: %r", v)
        return None
