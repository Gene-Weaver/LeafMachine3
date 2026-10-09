"""Ruler QC rendering: overlays, panel sections, and the per-sheet stack.

Everything that turns a measured ruler into something a human can check. Split in
two halves, previously two modules:

  PRIMITIVES  mask_overlay / comb_overlay / ruler_bars draw one overlay onto one
              deskewed strip. Fonts, colours and the page width live here too, so
              every QC image in the project looks the same.
  SECTIONS    class_section (what the classifier decided + the four-tile collage it
              actually saw), build_panel (the measurement itself), recon_strip and
              build_recon_section (how several rulers on one sheet were reconciled
              into a single CF), and stack_parent (the final page).
  FIELDPRISM  fp_marker_section (one FieldPrism marker: its 1 cm squares labeled
              exactly like the FieldPrism app, plus the geometric checks) and
              fp_sheet_section (which printed sheet the markers belong to, drawn
              to scale, and the CF it implies).

There is no CLI here. `ruler_CF_with_lattice_detection.RulerCFLattice` owns the
driving, and calls into this module so the live panel and the panel rebuilt later
from the project DB are produced by the same code.
"""

from __future__ import annotations

import functools
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .units import unit_mm, spec_of, admissible_units
from .units import ANCHOR_TOL, PLAUSIBLE_LEN_CM


# Squarify is MANDATORY, not decorative: all three classifier ensemble members were
# trained on pre-squarified square crops, so the four-tile collage IS the image the
# class label was derived from, and the QC panel shows it for exactly that reason.
# In the LM3 integration the four-tile tile is PRE-MADE by the RulerClassifier (in
# _ruler_squarify/) and always passed to class_section() as ``tile_img``; this lazy
# squarifier is only a fallback for the rare call with no stored tile, and is loaded
# via LM3's own port so this module has no hard-coded paths and no import-time model.
_SQ = None


def _squarify_fallback(crop_path, tile_h):
    global _SQ
    try:
        if _SQ is None:
            from leafmachine3.inference.ruler_squarify import RulerSquarifier
            _SQ = RulerSquarifier(sz=720, method="tile_four", augment=False)
        tile = cv2.cvtColor(_SQ.transform(crop_path), cv2.COLOR_BGR2RGB)
        return cv2.resize(tile, (tile_h, tile_h), interpolation=cv2.INTER_AREA)
    except Exception:
        return np.full((tile_h, tile_h, 3), 240, np.uint8)

W_OUT, PAD = 1700, 14
# Panels B/C and the header rows: colour == a RECONCILED PHYSICAL UNIT.
UNIT_COLORS = [
    (32, 178, 96),     # finest unit
    (238, 108, 32),
    (56, 132, 255),
    (196, 64, 200),
    (240, 200, 40),
]
# Panel A: colour == TICK LEVEL (how long the tick is), which is a DIFFERENT
# thing from a physical unit. Deliberately a separate palette so the two can
# never be confused at a glance.
LEVEL_COLORS = [
    (0, 150, 160),     # level 1 = every tick on the base lattice
    (170, 110, 220),   # level 2 = systematically longer ticks
    (200, 140, 40),
    (120, 180, 60),
    (220, 90, 140),
]
CM_COLOR, IN_COLOR = (32, 178, 96), (56, 132, 255)
# When no ruler yielded a publishable CF the sheet falls back to the MP-predicted anchor. Those
# bars are drawn in RED / ORANGE instead of green / blue so a predicted scale can never be mistaken
# at a glance for a measured one -- they are a regression estimate, not a reading off this ruler.
MP_CM_COLOR, MP_IN_COLOR = (220, 38, 38), (249, 115, 22)
PRETTY = {"metric__2_MM": "0.5 mm", "metric__MM": "1 mm", "metric__4_CM": "2.5 mm",
          "metric__4_MM": "2.5 mm",          # pre-rename records still render
          "metric__2_CM": "5 mm", "metric__CM": "1 cm",
          "std__32_IN": '1/32"', "std__16_IN": '1/16"', "std__8_IN": '1/8"',
          "std__4_IN": '1/4"', "std__2_IN": '1/2"', "std__IN": '1"'}

_FD = "/usr/share/fonts/truetype/dejavu"
def _font(sz, bold=False):
    for p in (os.path.join(_FD, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"),
              os.path.join(_FD, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    import matplotlib
    mp = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    return ImageFont.truetype(os.path.join(
        mp, "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"), sz)

F_T, F_B, F_S, F_XS = _font(21, True), _font(17), _font(15), _font(13)


def mask_overlay(gray, lab):
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    for lv in range(1, len(LEVEL_COLORS) + 1):
        m = lab == lv
        if m.any():
            rgb[m] = 0.30 * rgb[m] + 0.70 * np.array(LEVEL_COLORS[lv - 1], np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def comb_overlay(gray, groups, pxcm, band):
    """1-px combs from the PREDICTED scale, one per reconciled unit."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    H, W = gray.shape
    y0, y1 = band
    nb = max(1, len(groups))
    for i, g in enumerate(groups):
        col = UNIT_COLORS[i % len(UNIT_COLORS)]
        step = unit_mm(g["unit"]) / 10.0 * pxcm          # predicted px per unit
        if not np.isfinite(step) or step < 1.5:
            continue
        anchor = float(g.get("anchor") or 0.0)
        # each unit gets its own horizontal lane so overlapping combs stay legible
        lane0 = int(y0 + (y1 - y0) * i / nb)
        lane1 = int(y0 + (y1 - y0) * (i + 1) / nb)
        lane0, lane1 = max(0, lane0), min(H, max(lane0 + 3, lane1))
        k0 = int(math.floor(-anchor / step))
        k1 = int(math.ceil((W - anchor) / step))
        for k in range(k0, k1 + 1):
            x = int(round(anchor + k * step))
            if 0 <= x < W:
                rgb[lane0:lane1, x] = col
        cv2.line(rgb, (int(anchor), 0), (int(anchor), H - 1), (255, 255, 255), 1)
    return rgb


def ruler_bars(gray, x0, pxcm, band):
    """Panel D: from the first accepted tick, one 10-px-thick bar exactly
    round(px/cm) long and, 10 px below it, one exactly round(px/inch) long.

    Written with direct array slicing rather than any drawing primitive, so the
    bars carry NO border or anti-aliasing and their pixel length is exactly the
    predicted value -- laying the prediction physically against the ruler.
    """
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    H, W = gray.shape
    L_cm = int(round(pxcm))
    L_in = int(round(pxcm * 2.54))
    # A tick whose cell straddles the left edge has a NEGATIVE centre. Passing
    # that straight into a slice gives rgb[y:y1, -4:142], which numpy reads as
    # "4 from the end" and evaluates to an EMPTY selection -- so both bars were
    # silently not drawn on 10/69 panels. Clamp before slicing.
    x0 = max(0, int(round(x0)))
    y = int(np.clip(band[0], 0, max(0, H - 32)))
    for (yy, L, col) in ((y, L_cm, CM_COLOR), (y + 20, L_in, IN_COLOR)):
        y1 = min(H, yy + 10)
        x1 = min(W, x0 + L)
        if y1 > yy and x1 > x0:
            rgb[yy:y1, x0:x1] = col
    return rgb, L_cm, L_in


def fit_w(img, w, max_h=None):
    h = max(1, int(round(img.shape[0] * w / img.shape[1])))
    if max_h and h > max_h:
        w = max(1, int(round(w * max_h / h))); h = max_h
    # NEAREST on downscale so 1-px comb lines survive
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


CLASS_TILE_H = 260          # four-tile render height, matched to the ruler strips

VERDICT_ALPHA = 0.50        # translucency of the big USED / REJECTED watermark
VERDICT_HEIGHT = 0.80       # glyph height as a fraction of the strip height
REJECTED_BAR_ALPHA = 0.50   # a rejected ruler's 1 cm / 1 inch bars are half-strength
USED_COLOR = (32, 178, 96)
REJECTED_COLOR = (194, 65, 12)


def class_section(row, cls, crop_path, tile_h=CLASS_TILE_H, failure=None,
                  tile_img=None):
    """First section of every ruler panel: WHAT the classifier decided, and the
    four-tile squarified collage it actually saw.

    For a crop that cannot yield a CF -- an unsupported class (FP / messy) or a
    failed determination -- this is the ONLY section emitted, with the reason.

    `tile_img` is an already-squarified RGB array. Pass it to rebuild a panel from
    a stored tile instead of re-deriving one from the crop, so the QC image can be
    reconstructed from the project DB alone even if the crop file is gone.
    """
    sp = spec_of(cls)
    if tile_img is not None:
        tile = cv2.resize(np.asarray(tile_img), (tile_h, tile_h),
                          interpolation=cv2.INTER_AREA)
    else:
        tile = _squarify_fallback(crop_path, tile_h)

    units = ", ".join(PRETTY.get(u, u) for u in (sp.get("units") or [])) or "-"
    adm = ", ".join(PRETTY.get(u, u) for u in admissible_units(cls)) or "-"
    lines = [("1 -- Ruler Class", F_T, (15, 15, 20)),
             (f"{row.get('image_name')}   [detection {row['detection_id']}, "
              f"ruler conf {row['det_conf']:.2f}]", F_S, (60, 62, 70)),
             (f"RulerClassifier -> {cls}"
              f"{'' if row.get('cls_conf') is None else '  (%.2f)' % row['cls_conf']}",
              F_B, (194, 65, 12) if failure else (21, 128, 61)),
             (f"systems  : {'+'.join(sp.get('systems') or ()) or '-'}", F_XS, (90, 95, 105)),
             (f"layout   : {sp.get('layout')}"
              f"{'   row period %g cm' % sp['row_period_cm'] if sp.get('row_period_cm') else ''}"
              f"{'   known ratio %g' % sp['ratio'] if sp.get('ratio') else ''}",
              F_XS, (90, 95, 105)),
             (f"units printed : {units}", F_XS, (90, 95, 105)),
             (f"units admissible : {adm}", F_XS, (90, 95, 105))]
    if failure:
        lines.append(("", F_XS, (90, 95, 105)))
        for ln in failure.split("\n"):
            lines.append((ln, F_S, (194, 65, 12)))
    lines.append(("", F_XS, (90, 95, 105)))
    lines.append(("The square below is the four-tile squarified collage the classifier "
                  "actually receives (LM2 tile_four, 1440x1440).", F_XS, (120, 125, 135)))

    LH = {id(F_T): 30, id(F_B): 24, id(F_S): 20, id(F_XS): 18}
    head = 10 + sum(LH[id(f)] for _, f, _ in lines) + 8
    H = head + tile_h + 14
    im = Image.new("RGB", (W_OUT, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    y = 10
    for txt, f, c in lines:
        if txt:
            d.text((PAD, y), txt, font=f, fill=c)
        y += LH[id(f)]
    im.paste(Image.fromarray(tile), (PAD, head))
    return im


def stack_sections(parts):
    """Vertically stack section images into one panel."""
    W = max(p.size[0] for p in parts)
    H = sum(p.size[1] for p in parts)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for p in parts:
        im.paste(p, (0, y)); y += p.size[1]
    return im


def _font_at_height(px, text="REJECTED"):
    """A bold font whose rendered glyph height is about `px` pixels."""
    size = max(8, int(px * 1.3))
    f = _font(size, bold=True)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    bb = probe.textbbox((0, 0), text, font=f)
    h = bb[3] - bb[1]
    if h > 0:
        size = max(8, int(round(size * px / h)))
        f = _font(size, bold=True)
    return f


def recon_strip(gray, x0, pxcm, band, verdict):
    """One ruler for the reconciliation comparison, composited in z-order:
    ruler image -> huge translucent USED/REJECTED watermark -> the predicted
    1 cm / 1 inch bars ON TOP (half-strength when the ruler was rejected)."""
    H, W = gray.shape
    im = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)).convert("RGBA")

    label = {"used": "USED", "rejected": "REJECTED"}.get(verdict)
    if label:
        col = USED_COLOR if verdict == "used" else REJECTED_COLOR
        f = _font_at_height(max(10, int(VERDICT_HEIGHT * H)), label)
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        bb = d.textbbox((0, 0), label, font=f)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        d.text(((W - tw) // 2 - bb[0], (H - th) // 2 - bb[1]), label, font=f,
               fill=col + (int(round(255 * VERDICT_ALPHA)),))
        im = Image.alpha_composite(im, layer)

    arr = np.asarray(im.convert("RGB")).copy()
    L_cm, L_in = int(round(pxcm)), int(round(pxcm * 2.54))
    x0 = max(0, int(round(x0)))
    y = int(np.clip(band[0], 0, max(0, H - 32)))
    a = REJECTED_BAR_ALPHA if verdict == "rejected" else 1.0
    for (yy, L, c) in ((y, L_cm, CM_COLOR), (y + 20, L_in, IN_COLOR)):
        y1, x1 = min(H, yy + 10), min(W, x0 + L)
        if y1 > yy and x1 > x0:
            reg = arr[yy:y1, x0:x1].astype(np.float32)
            arr[yy:y1, x0:x1] = np.clip(
                a * np.array(c, np.float32) + (1.0 - a) * reg, 0, 255).astype(np.uint8)
    return arr





def mp_fallback_strip(gray, x0, anchor_pxcm, band):
    """A ruler strip carrying the MP-PREDICTED 1 cm / 1 inch bars in red / orange.

    Same geometry as :func:`recon_strip` so the two are directly comparable, but the colors and the
    watermark say PREDICTED. Laying the prediction physically against the ruler's own graduations is
    the point: it lets a reader see immediately how far the fallback is from the truth on this
    sheet, which a bare px/cm number cannot convey.
    """
    H, W = gray.shape
    im = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)).convert("RGBA")
    f = _font_at_height(max(10, int(VERDICT_HEIGHT * H)), "PREDICTED")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    bb = d.textbbox((0, 0), "PREDICTED", font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((W - tw) // 2 - bb[0], (H - th) // 2 - bb[1]), "PREDICTED", font=f,
           fill=MP_CM_COLOR + (int(round(255 * VERDICT_ALPHA)),))
    arr = np.asarray(Image.alpha_composite(im, layer).convert("RGB")).copy()
    L_cm, L_in = int(round(anchor_pxcm)), int(round(anchor_pxcm * 2.54))
    x0 = max(0, int(round(x0)))
    y = int(np.clip(band[0], 0, max(0, H - 32)))
    for (yy, L, c) in ((y, L_cm, MP_CM_COLOR), (y + 20, L_in, MP_IN_COLOR)):
        y1, x1 = min(H, yy + 10), min(W, x0 + L)
        if y1 > yy and x1 > x0:
            arr[yy:y1, x0:x1] = c
    return arr


def build_panel(row, s_anch, s_plain, res, groups, pxcm, ov_mask, ov_comb, ov_bars, bars):
    A = fit_w(ov_mask, W_OUT - 2 * PAD, max_h=300)
    B = fit_w(ov_comb, W_OUT - 2 * PAD, max_h=300)
    D = fit_w(ov_bars, W_OUT - 2 * PAD, max_h=300)
    coarse = max([unit_mm(g["unit"]) / 10.0 * pxcm for g in groups] or [pxcm])
    win = int(min(ov_comb.shape[1],
                  max(6 * coarse, 0.30 * ov_comb.shape[1], 40 * s_anch["P"])))
    C = fit_w(ov_comb[:, max(0, ov_comb.shape[1] - win):], W_OUT - 2 * PAD, max_h=300)

    anchor = row.get("anchor")
    fp_anchor = row.get("anchor_source") == "fieldprism"
    flipped = (s_plain is not None
               and abs(s_plain["pxcm"] / max(pxcm, 1e-9) - 1.0) > 0.03)
    d_anchor = (100 * (pxcm / anchor - 1.0)) if anchor else None
    tr = s_anch.get("trans")

    lines = [(f"{row.get('image_name')}   [detection {row['detection_id']}, "
              f"ruler conf {row['det_conf']:.2f}]", F_T, (15, 15, 20)),
             (f"RulerClassifier -> {row.get('ruler_class') or '?'}"
              f"{'' if row.get('cls_conf') is None else ' (%.2f)' % row['cls_conf']}"
              f"   [{s_anch['layout']}: "
              f"{', '.join(PRETTY.get(u, u) for u in s_anch['class_units_declared']) or '-'}]"
              f"   MP {row.get('original_mp')}"
              f"   rotation {s_anch['angle']:+.2f}deg"
              f"   band {s_anch['band'][0]}-{s_anch['band'][1]} [{s_anch['band_kind']}]"
              f"   periodicity {s_anch['band_score']:.2f}", F_XS, (90, 95, 105)),
             (f"P_acf {s_anch['P_acf']:.2f}px -> {s_anch['period_name']}"
              f"   base lattice {s_anch['P']:.2f}px"
              f"   ticks {s_anch['kept']}/{s_anch['n_ticks']}"
              f"{'   [FALLBACK]' if s_anch['fallback'] else ''}"
              f"{'   [SALVAGED]' if s_anch['salvaged'] else ''}", F_XS, (90, 95, 105)),
             (f"RECONCILED: {s_anch['relation']}   "
              f"systems={'+'.join(s_anch['systems']) or '-'}   "
              f"corroborating-pairs={s_anch['n_corroborating']}   "
              f"averaged-same-unit={s_anch['n_averaged']}   cross-unit spread="
              f"{'n/a' if s_anch['rec_spread'] is None else format(100*s_anch['rec_spread'], '.2f') + '%'}"
              f"  -> {'AGREE' if s_anch['rec_agree'] else 'no cross-check'}",
              F_S, (21, 128, 61) if s_anch["rec_agree"] else (150, 100, 30))]
    if tr:
        lines.append((f"TRANSITION: fine {tr['P_fine']:.2f}px | coarse {tr['P_coarse']:.2f}px "
                      f"at x={tr['split_x']}  ->  ratio {tr['ratio']:.3f} vs expected "
                      f"{tr['expect']:.0f} = {100*tr['ratio_err']:.2f}% off"
                      f"{'   [USED as scale]' if s_anch.get('trans_used') else '   [not used]'}",
                      F_XS, (21, 128, 61) if s_anch.get("trans_used") else (150, 100, 30)))
    for i, g in enumerate(groups):
        step = unit_mm(g["unit"]) / 10.0 * pxcm
        lines.append((
            f"    {PRETTY.get(g['unit'], g['unit']):>7s} ({g['system']:6s})  "
            f"levels x{','.join(str(m) for m in g['mults']):<8s} "
            f"measured {g['spacing']:7.2f}px   predicted {step:7.2f}px   "
            f"px/cm {g['est_pxcm']:7.2f}   n={g['n_ticks']:<4d}"
            f"{'  AVERAGED (%.2f%%)' % (100*g['spread_within']) if g['averaged'] else ''}",
            F_XS, UNIT_COLORS[i % len(UNIT_COLORS)]))

    ws = row.get("work_scale") or 1.0
    lines.append((f"CLASS ADMITS ({'+'.join(s_anch['systems']) or '-'}): "
                  f"{', '.join(PRETTY.get(u, u) for u in s_anch['admissible'])}"
                  f"      -- nothing outside this set can be assigned",
                  F_XS, (90, 95, 105)))
    lines.append((f"ADMISSION BOUNDS: a reading is kept only if it implies a ruler "
                  f"{PLAUSIBLE_LEN_CM[0]:g}-{PLAUSIBLE_LEN_CM[1]:g} cm long "
                  f"AND sits within +/-{100 * ANCHOR_TOL:.0f}% of the "
                  f"{'FieldPrism' if fp_anchor else 'MP'} anchor",
                  F_XS, (90, 95, 105)))
    for c in s_anch["cf_contributions"]:
        # 7th element (the reason) was added later; records written before it are 6-tuples.
        u, est, n, srcx, usedx, pct = c[:6]
        why = c[6] if len(c) > 6 else None
        verdict = "USED" if usedx else f"REJECTED -- {why or 'out of bounds'}"
        lines.append((f"    CF from {PRETTY.get(u, str(u)):>7s}  {est:8.2f} px/cm  n={n:<4d} "
                      f"[{srcx}]  {'' if pct is None else 'anchor %+6.1f%%' % pct}  {verdict}",
                      F_XS, (21, 128, 61) if usedx else (194, 65, 12)))
    lines.append((f"FUSED CF from {s_anch['cf_n_used']} unit(s), {s_anch['cf_n_rejected']} "
                  f"rejected; spread "
                  f"{'n/a' if s_anch['cf_spread'] is None else format(100*s_anch['cf_spread'], '.2f')+'%'}"
                  f"{'   [FALLBACK: nothing passed bounds]' if s_anch['cf_fallback'] else ''}",
                  F_S, (150, 100, 30) if s_anch["cf_fallback"] else (21, 128, 61)))
    # The frame, stated once and up front. Every measurement LM3 makes -- every tick spacing, every
    # CF, every published number -- is in the WORKING frame: the resized copy the pipeline actually
    # analyzes. Original-frame values appear ONLY as provenance, never as an input, and the panel
    # used to convert back and forth between the two on adjacent lines without saying which was
    # authoritative.
    anchor_frame = str(row.get("anchor_frame") or "original")
    if fp_anchor:
        # On a FieldPrism sheet the ruler was read against the FieldPrism CF, measured in the
        # working frame from this sheet's own markers; the MP prediction played no part.
        src = (f"FIELDPRISM-ANCHOR {'-' if anchor is None else '%.2f' % anchor} px/cm "
               f"(WORKING frame, measured from this sheet's FieldPrism markers)")
    elif anchor_frame == "working":
        src = (f"MP-ANCHOR {'-' if anchor is None else '%.2f' % anchor} px/cm "
               f"(WORKING frame, predicted directly from the working image)")
    else:
        src = (f"MP-ANCHOR {row.get('anchor_original')} px/cm (original) x work_scale "
               f"{ws:.4f} = {'-' if anchor is None else '%.2f' % anchor} px/cm (WORKING frame)")
    lines.append((f"FRAME: all px/cm below are WORKING-frame (the resized image LM3 measures on); "
                  f"original-frame values are provenance only.   work_scale {ws:.4f}",
                  F_XS, (90, 95, 105)))
    lines.append((src, F_XS, (90, 95, 105)))
    lines.append((f"PRED {pxcm:.2f} px/cm (working)"
                  f"{'' if d_anchor is None else '  [%+.1f%% vs anchor]' % d_anchor}"
                  f"   =  {pxcm/ws:.2f} px/cm in the ORIGINAL frame"
                  f"      PRED without anchor "
                  f"{'-' if s_plain is None else '%.2f' % s_plain['pxcm']}"
                  f"{'   << ANCHOR CHANGED THE READ' if flipped else '   (same)'}",
                  F_B, (194, 65, 12) if flipped else (21, 128, 61)))

    LH = {id(F_T): 27, id(F_B): 24, id(F_S): 20, id(F_XS): 18}
    head = 8 + sum(LH[id(f)] for _, f, _ in lines) + 10
    n_lv = max(1, len(s_anch["ladder"]) + 1)
    H = head + A.shape[0] + B.shape[0] + C.shape[0] + D.shape[0] + 4 * 26 + 22 + 3 * PAD
    im = Image.new("RGB", (W_OUT, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    yy = 8
    for txt, f, c in lines:
        if f is F_XS and txt.startswith("    "):
            d.rectangle([PAD, yy + 3, PAD + 12, yy + 13], fill=c)
            d.text((PAD + 18, yy), txt.strip(), font=f, fill=c)
        else:
            d.text((PAD, yy), txt, font=f, fill=c)
        yy += LH[id(f)]

    y = head
    d.text((PAD, y), "A colours = TICK LEVEL (not a unit):", font=F_XS, fill=(90, 95, 105))
    lx = PAD + int(d.textlength("A colours = TICK LEVEL (not a unit):", font=F_XS)) + 12
    for i in range(n_lv):
        c = LEVEL_COLORS[i % len(LEVEL_COLORS)]
        d.rectangle([lx, y + 2, lx + 12, y + 12], fill=c)
        lab = ("level 1 (every base tick)" if i == 0
               else f"level {i+1} (x{s_anch['ladder'][i-1]} longer)")
        d.text((lx + 17, y - 1), lab, font=F_XS, fill=c)
        lx += 26 + int(d.textlength(lab, font=F_XS))
    y += 22
    for img, cap in (
            (A, "2A - DETECTED tick masks, coloured by TICK LEVEL (legend above)"),
            (B, "2B - PREDICTED combs from the anchored px/cm, anchored on each unit's "
                "first accepted tick. Drift from the real ticks = scale error."),
            (C, "2C - right-hand end of B at full scale: accumulated drift"),
            (D, f"2D - PREDICTED RULE from the first accepted tick: GREEN = 1 cm = exactly "
                f"{bars[0]} px, BLUE (10 px below) = 1 inch = exactly {bars[1]} px. "
                f"Check these against the ruler's own graduations.")):
        d.text((PAD, y), cap, font=F_XS, fill=(120, 125, 135))
        y += 20
        im.paste(Image.fromarray(img), (PAD, y))
        y += img.shape[0] + 6
    return im, flipped


def build_cf_summary_section(measured_cf, anchor_cf, formula_symbolic=None, fallback_applied=False,
                             *, anchor_source="megapixels", fp_sheet=None):
    """The last block on every panel: the two numbers a reader came for, stated plainly.

    Everything above this is the audit trail -- which unit was named, which crop won, why a reading
    was rejected. This says only what the sheet ends up with: the CF the ruler methods produced (or
    that they produced nothing), and the megapixel regression's prediction alongside the equation
    that generated it. It is rendered for EVERY sheet, published or not, so the answer is always in
    the same place at the same end of the image.

    The last line says which of the two the sheet is actually measured with. ``fallback_applied``
    is True when the ruler_cf stage substituted the prediction (``use_CF_predicted_by_MP``).

    On a sheet with FieldPrism markers (``anchor_source="fieldprism"``, or any ``fp_sheet`` row)
    ``anchor_cf`` is the anchor actually used -- the FieldPrism CF, never the megapixel prediction
    -- and the block says so; ``fp_sheet`` (a ruler_FP_sheet row) adds the sheet type and the
    sheet-fit / marker-mean breakdown. With the defaults the output is unchanged.
    """
    if anchor_source == "fieldprism" or fp_sheet is not None:
        return _fp_cf_summary_section(measured_cf, anchor_cf, anchor_source, fp_sheet)
    W = W_OUT
    meas = "none" if measured_cf is None else f"{float(measured_cf):.2f} px/cm"
    eq = formula_symbolic or "megapixel regression"
    pred = "none" if anchor_cf is None else f"{float(anchor_cf):.2f} px/cm"
    if measured_cf is not None:
        used = (f"CF used for this sheet: {float(measured_cf):.2f} px/cm -- measured from the ruler",
                (21, 128, 61))
    elif fallback_applied and anchor_cf is not None:
        used = (f"CF used for this sheet: {float(anchor_cf):.2f} px/cm -- PREDICTED from megapixels "
                f"(use_CF_predicted_by_MP is on)", (150, 100, 30))
    else:
        used = ("CF used for this sheet: none -- cm measurements left empty "
                "(use_CF_predicted_by_MP is off)", (194, 65, 12))
    lines = [
        ("Pixel to Metric Conversion Factor", F_T, (15, 15, 20)),
        (f"Measured CF: {meas}", F_B,
         (194, 65, 12) if measured_cf is None else (21, 128, 61)),
        (f"Predicted CF ({eq}): {pred}", F_B, (90, 95, 105)),
        (used[0], F_B, used[1]),
    ]
    LH = {id(F_T): 34, id(F_B): 26}
    H = 12 + sum(LH[id(f)] for _, f, _ in lines) + 12
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    y = 12
    for txt, f, col in lines:
        d.text((PAD, y), txt, font=f, fill=col)
        y += LH[id(f)]
    return im


def stack_parent(panels, recon_img, out_path=None, cf_summary=None):
    """One PNG per parent image: every ruler panel stacked vertically, then (when
    the sheet has more than one ruler) the Multiple Ruler Reconciliation section, and finally the
    Pixel to Metric Conversion Factor summary -- each below a medium-thick black rule.

    Returns the assembled image. `out_path` is optional so a caller that wants the
    image in memory (the RulerCFLattice class) uses this exact layout rather than
    reimplementing the stacking and drifting from it by a couple of pixels.
    """
    tail = [x for x in (recon_img, cf_summary) if x is not None]
    W = max(p.size[0] for p in panels + tail)
    RULE = 6                                   # medium-thick separator
    H = sum(p.size[1] for p in panels) + (len(panels) - 1) * 2
    for t in tail:
        H += RULE + 10 + t.size[1]
    im = Image.new("RGB", (W, H), (255, 255, 255))
    y = 0
    for i, p in enumerate(panels):
        im.paste(p, (0, y)); y += p.size[1]
        if i < len(panels) - 1:
            ImageDraw.Draw(im).rectangle([0, y, W, y + 1], fill=(215, 215, 220))
            y += 2
    for t in tail:                             # recon (when present), then ALWAYS the CF summary
        ImageDraw.Draw(im).rectangle([0, y, W, y + RULE - 1], fill=(0, 0, 0))
        y += RULE + 10
        im.paste(t, (0, y))
        y += t.size[1]
    if out_path is not None:
        im.save(out_path)
    return im


def build_recon_section(image_name, entries, pr, anchor, anchor_formula=None, fallback_applied=False,
                        *, anchor_source="megapixels"):
    """The 'Multiple Ruler Reconciliation' block: every ruler's own CF, its
    predicted 1 cm / 1 inch bars stacked for direct visual comparison, and an
    explicit account of how the single parent CF was arrived at.

    ``anchor_source`` names what ``anchor`` is. "megapixels" (the default) is the
    MP regression and renders exactly as before. "fieldprism" is the FieldPrism
    sheet CF: the header says "FieldPrism anchor", and a withheld sheet says there
    is no megapixel fallback instead of drawing one (the stage never applies the
    MP prediction to a sheet with FieldPrism markers). Any other value (e.g. the
    engine's "none" for a sheet with no anchor) renders exactly like "megapixels",
    so a sheet without FieldPrism markers looks the same as before."""
    W = W_OUT
    fp = anchor_source == "fieldprism"
    head_lines = []
    head_lines.append(("3 -- Multiple Ruler Reconciliation", F_T, (15, 15, 20)))
    # FieldPrism markers carry no deskewed strip, so `entries` (the drawable crops) undercounts a
    # FieldPrism sheet; the reconciliation table in `pr` lists every crop.
    n_fp = sum(1 for c in pr.get("per_crop", []) if c.get("ruler_class") == "FP")
    n_crops = len(pr.get("per_crop") or entries) if fp else len(entries)
    head_lines.append((
        f"{image_name}   --   {n_crops} ruler crops on this sheet"
        f"{' (%d FieldPrism marker%s)' % (n_fp, '' if n_fp == 1 else 's') if fp and n_fp else ''}"
        f". Every crop measures the "
        f"SAME sheet, so all must reconcile to ONE conversion factor.", F_S, (60, 62, 70)))
    cf = pr.get("cf_px_per_cm")
    meas = pr.get("cf_px_per_cm_measured")
    anchor_name = "FieldPrism anchor" if fp else "MP anchor"
    head_lines.append((
        f"{anchor_name} (working frame) {('-' if anchor is None else '%.2f px/cm' % anchor)}"
        f"      method: {pr.get('method') or '-'}"
        f"      peer spread "
        f"{'n/a' if pr.get('spread') is None else format(100*pr['spread'], '.2f') + '%'}"
        f"      -> PARENT CF {('NONE' if cf is None else '%.2f px/cm' % cf)}"
        f"{'' if pr.get('pct_vs_anchor') is None else '  (%+.1f%% vs anchor)' % pr['pct_vs_anchor']}",
        F_B, (21, 128, 61) if pr.get("ok") else (194, 65, 12)))
    # A withheld sheet must say on the image what it read, why that was not trusted,
    # and what the consumer gets instead -- otherwise a null CF looks like a crash.
    if cf is None:
        read = (f"The lattice read {meas:.2f} px/cm but it did not clear the gate"
                if meas is not None else "No ruler crop produced a usable reading")
        head_lines.append((
            f"CF WITHHELD (confidence: {pr.get('confidence')}).  {read}, so NO measured "
            f"conversion factor is published for this sheet.", F_B, (194, 65, 12)))
        if fp:
            head_lines.append((
                "NO MEGAPIXEL FALLBACK on a FieldPrism sheet -- FieldPrism sheets come in several "
                "sizes, so the MP prediction is never substituted; this sheet's cm measurements "
                "stay empty.", F_S, (194, 65, 12)))
        elif anchor is not None:
            head_lines.append((
                "FALLBACK APPLIED (use_CF_predicted_by_MP is on) -- every downstream measurement "
                "on this sheet uses the MP-PREDICTED anchor instead of a ruler reading:"
                if fallback_applied else
                "FALLBACK NOT APPLIED (use_CF_predicted_by_MP is off) -- this sheet's cm "
                "measurements stay empty. With it on, the sheet would use the MP-PREDICTED anchor:",
                F_S, (194, 65, 12)))
            head_lines.append((
                f"    {anchor_formula or ('cf = %.2f px/cm (megapixel regression)' % anchor)}"
                f"        [predicted from the image's megapixels, NOT measured from this ruler]",
                F_B, (194, 65, 12)))
            head_lines.append((
                "    The RED (1 cm) and ORANGE (1 inch) bars in the final strip below are drawn at "
                "that predicted scale. Green/blue bars are measured; red/orange are predicted.",
                F_XS, (194, 65, 12)))
    for why in (pr.get("confidence_reasons") or ([pr["reason"]] if pr.get("reason") else [])):
        head_lines.append((f"    - {why}", F_S,
                           (60, 62, 70) if pr.get("ok") else (194, 65, 12)))
    for c in pr.get("per_crop", []):
        col = {"used": (21, 128, 61), "rejected": (194, 65, 12)}.get(c["verdict"], (120, 125, 135))
        head_lines.append((
            f"    {c['key']:<8s} {c.get('ruler_class') or '-':<16s} "
            f"CF {('-' if c.get('cf') is None else '%8.2f' % c['cf'])} px/cm   "
            f"{'' if c.get('pct_vs_parent') is None else 'vs parent %+7.2f%%' % c['pct_vs_parent']}   "
            f"{'' if c.get('pct_vs_anchor') is None else 'vs anchor %+7.2f%%' % c['pct_vs_anchor']}   "
            f"-> {c['verdict'].upper()}{'  (' + c['note'] + ')' if c.get('note') else ''}",
            F_XS, col))
    head_lines.append((
        "Bars below: each ruler's PREDICTED 1 cm (green) and 1 inch (blue), drawn at that "
        "ruler's own CF and stacked for direct comparison. Equal-length bars = the crops agree.",
        F_XS, (120, 125, 135)))
    if fp and n_fp:
        head_lines.append((
            "FieldPrism markers have no strip here: their 1 cm squares are shown in the FieldPrism "
            "Marker panels above, and the sheet geometry in the FieldPrism Sheet panel.",
            F_XS, (120, 125, 135)))

    LH = {id(F_T): 30, id(F_B): 24, id(F_S): 20, id(F_XS): 18}
    head = 10 + sum(LH[id(f)] for _, f, _ in head_lines) + 8
    strips, labels = [], []
    for e in entries:
        if e.get("rot") is None:
            continue
        strips.append(fit_w(recon_strip(e["rot"], e["x0"], e["pxcm"], e["band"],
                                        e.get("verdict")),
                            W_OUT - 2 * PAD, max_h=240))
        labels.append(f"{e['key']}  {e.get('ruler_class') or ''}  "
                      f"CF {e['pxcm']:.2f} px/cm  ->  {(e.get('verdict') or '?').upper()}")
    # the MP fallback (applied or not -- the header says which), laid against a real ruler
    if cf is None and anchor is not None and entries and not fp:
        e0 = entries[0]
        strips.append(fit_w(mp_fallback_strip(e0["rot"], e0["x0"], float(anchor), e0["band"]),
                            W_OUT - 2 * PAD, max_h=240))
        labels.append(f"MP FALLBACK   1 cm (red) / 1 inch (orange) at the PREDICTED "
                      f"{float(anchor):.2f} px/cm   --   {anchor_formula or 'megapixel regression'}")
    sh = [s.shape[0] for s in strips]
    H = head + sum(h + 22 for h in sh) + 12
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    y = 10
    for txt, f, c in head_lines:
        if f is F_XS and txt.startswith("    "):
            d.rectangle([PAD, y + 3, PAD + 12, y + 13], fill=c)
            d.text((PAD + 18, y), txt.strip(), font=f, fill=c)
        else:
            d.text((PAD, y), txt, font=f, fill=c)
        y += LH[id(f)]
    y = head
    for lab, strip in zip(labels, strips):
        d.text((PAD, y), lab, font=F_XS, fill=(90, 95, 105))
        y += 20
        im.paste(Image.fromarray(strip), (PAD, y))
        y += strip.shape[0] + 2
    return im


# =========================================================================== #
# FIELDPRISM
# =========================================================================== #
# A FieldPrism (FP) marker is a 3x3 grid of 10 mm cells with the TL, TR, C and BL cells printed and
# the BR cell empty; four translated copies sit near the corners of a printed sheet. LM3 measures
# each marker independently (fieldprism.measure_marker) and identifies the sheet from where the
# markers are (fieldprism.identify_sheet). The two sections below are the QC for those two steps.
#
# Label colors are the FieldPrism app's own (RulerDeskewPrecise.kt:25-33 drawFpOverlay), so a crop
# here reads exactly like the app's FPfit overlay and like the Reporter's summary overlay.
FP_ROLE_COLORS = {"TL": (255, 0, 0), "TR": (255, 255, 0), "BL": (255, 255, 255),
                  "C": (0, 255, 255)}
FP_BR_USED = (0, 255, 0)          # predicted empty BR cell of a marker the sheet CF used
FP_BR_REJECTED = (255, 0, 0)      # ... of a marker that was measured but not used (outline only)
FP_INFERRED = (255, 0, 255)       # reconstructed markers and the page outline
FP_LABEL_FRAC = 0.70              # app text size = 0.70 x (px per cm)
FP_CROP_H = 400                   # display height of the marker crop in fp_marker_section
FP_SCHEMATIC_H = 520              # display height of the two page schematics in fp_sheet_section
FP_FIT_MAX_W = 700                # widest the working-image fit schematic may get
FP_ROLES = ("TL", "TR", "C", "BL")
FP_CORNERS = ("TL", "TR", "BL", "BR")
_FP_OK, _FP_BAD, _FP_WARN, _FP_GRAY = (21, 128, 61), (194, 65, 12), (150, 100, 30), (120, 125, 135)

# The printed layout, used only when the sheet catalog (fieldprism_sheets.json) cannot be read.
# Source: FieldPrism QR_code_builder/build_PDF_PageSizes.py -- PageInfo W x H in integer mm
# (PS:5-23), marker top-left corner x = 20 | W-50, y = 23 | H-54 (xy_drawMarker, PS:77-101), and
# the square centers relative to that corner (drawMarker, build_PDF_utils.py:200-229).
_FP_RULE_SHEETS = {          # key: (label, PageInfo W x H, physical page W x H)
    "A5": ("A5", (148, 210), (148.0, 210.0)),
    "A4": ("A4", (210, 297), (210.0, 297.0)),
    "A3": ("A3", (297, 420), (297.0, 420.0)),
    "Letter": ("Letter", (216, 279), (215.9, 279.4)),
    "Legal": ("Legal", (216, 356), (215.9, 355.6)),
    "Tabloid": ("Tabloid", (279, 432), (279.0, 432.0)),
}
_FP_SQUARE_OFFSETS_MM = {"TL": (5.0, 5.0), "TR": (25.0, 5.0), "C": (15.0, 15.0),
                         "BL": (5.0, 25.0), "BR": (25.0, 25.0)}

# Fallback limits for the per-marker checks when a row carries no validation_json (the values
# fieldprism.py documents; read from that module when it is importable).
_FP_DEFAULT_LIMITS = {"pitch_ratio": 0.05, "right_angle": 3.0, "c_mid": 0.03, "c_diag": 0.05,
                      "peak_area": 0.3}


def _fp_limits():
    try:
        from . import fieldprism as _fpm
        return {"pitch_ratio": float(_fpm.MAX_PITCH_RATIO_ERR),
                "right_angle": float(_fpm.MAX_RIGHT_ANGLE_ERR_DEG),
                "c_mid": float(_fpm.MAX_C_MID_ERR), "c_diag": float(_fpm.MAX_C_DIAG_ERR),
                "peak_area": float(_fpm.MIN_PEAK_AREA_RATIO)}
    except Exception:
        return dict(_FP_DEFAULT_LIMITS)


def _fp_font(sz):
    """Bold proportional sans for the on-image labels (the app uses the system bold font);
    the panel text stays in the monospace F_* fonts above."""
    p = os.path.join(_FD, "DejaVuSans-Bold.ttf")
    if os.path.exists(p):
        return ImageFont.truetype(p, max(6, int(sz)))
    return _font(max(6, int(sz)), bold=True)


def _fp_json(v, default=None):
    """A JSON column as stored (a string) or already parsed; anything unreadable -> default."""
    if v is None:
        return default
    if isinstance(v, (bytes, str)):
        try:
            v = json.loads(v)
        except (TypeError, ValueError):
            return default
    return default if v is None else v


def _fp_num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fp_pt(v):
    """[x, y] / (x, y) / {"x":, "y":} -> (float, float), else None."""
    if isinstance(v, dict):
        v = (v.get("x"), v.get("y"))
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        x, y = _fp_num(v[0]), _fp_num(v[1])
        if x is not None and y is not None:
            return (x, y)
    return None


def _fp_wh(v):
    """[w, h] / {"w","h"} / {"width","height"} -> (float, float), else None."""
    if isinstance(v, dict):
        v = (v.get("w", v.get("width")), v.get("h", v.get("height")))
    return _fp_pt(v)


def _fp_marker_roles(m):
    """{role: (x, y)} from a ruler_FP_marker row (WORKING frame); BR is the stored predicted cell,
    or TR + BL - TL when only the three printed corners are stored."""
    out = {}
    for r in FP_ROLES + ("BR",):
        x, y = _fp_num(m.get(f"{r.lower()}_x")), _fp_num(m.get(f"{r.lower()}_y"))
        if x is not None and y is not None:
            out[r] = (x, y)
    if "BR" not in out and all(k in out for k in ("TL", "TR", "BL")):
        out["BR"] = (out["TR"][0] + out["BL"][0] - out["TL"][0],
                     out["TR"][1] + out["BL"][1] - out["TL"][1])
    return out


# ---- the sheet model: catalog first, the layout rule as a fallback ----------------------------
@functools.lru_cache(maxsize=1)
def _fp_catalog():
    """The parsed sheet catalog, via fieldprism.load_sheet_catalog when that module exists, else
    read straight from fieldprism_sheets.json; None when neither is available."""
    try:
        from . import fieldprism as _fpm
        cat = _fpm.load_sheet_catalog()
        if cat:
            return cat
    except Exception:
        pass
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fieldprism_sheets.json")
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _fp_catalog_entry(cat, sheet_type):
    """The catalog's record for `sheet_type` (case-insensitive), whether the catalog keys its
    sheets by type in a dict or lists them with a key/sheet_type/name field."""
    if not isinstance(cat, dict) or not sheet_type:
        return None
    sheets = cat.get("sheets", cat)
    want = str(sheet_type).lower()
    if isinstance(sheets, dict):
        for k, v in sheets.items():
            if str(k).lower() == want and isinstance(v, dict):
                return v
        sheets = [v for v in sheets.values() if isinstance(v, dict)]
    if isinstance(sheets, list):
        for v in sheets:
            if isinstance(v, dict) and any(str(v.get(k, "")).lower() == want
                                           for k in ("sheet_type", "key", "id", "name", "label")):
                return v
    return None


def _fp_rule_model(sheet_type):
    key = next((k for k in _FP_RULE_SHEETS if k.lower() == str(sheet_type or "").lower()), None)
    if key is None:
        return None
    label, (W, H), page = _FP_RULE_SHEETS[key]
    corner = {"TL": (20.0, 23.0), "TR": (W - 50.0, 23.0), "BL": (20.0, H - 54.0),
              "BR": (W - 50.0, H - 54.0)}
    return dict(sheet_type=key, label=label, layout_mm=(float(W), float(H)), page_mm=page,
                marker_corner_mm=corner, source="layout rule")


def _fp_sheet_model(sheet_type):
    """Everything the to-scale schematic needs for one sheet type, or None if it is unknown:
    label, page_mm (physical), layout_mm (PageInfo), marker_corner_mm {corner: (x, y)}, the square
    centers {corner: {role: (x, y)}} and the two 10 cm bars. Catalog values win; any field the
    catalog does not carry (or that fails to parse) comes from the layout rule."""
    rule = _fp_rule_model(sheet_type)
    ent = _fp_catalog_entry(_fp_catalog(), sheet_type)
    if ent is None and rule is None:
        return None
    model = dict(rule) if rule else dict(sheet_type=str(sheet_type), label=str(sheet_type))
    if ent is not None:
        model["source"] = "sheet catalog"
        if ent.get("label"):
            model["label"] = str(ent["label"])
        for k in ("layout_mm", "page_mm"):
            wh = _fp_wh(ent.get(k))
            if wh:
                model[k] = wh
        mc = ent.get("marker_corner_mm")
        if isinstance(mc, dict):
            pts = {c: _fp_pt(mc.get(c)) for c in FP_CORNERS}
            if all(pts.values()):
                model["marker_corner_mm"] = pts
        sc = ent.get("square_centers_mm")
        if isinstance(sc, dict):
            sq = {}
            for c in FP_CORNERS:
                roles = sc.get(c) if isinstance(sc.get(c), dict) else {}
                pts = {r: _fp_pt(roles.get(r)) for r in FP_ROLES + ("BR",)}
                if all(pts.values()):
                    sq[c] = pts
            if len(sq) == 4:
                model["square_centers_mm"] = sq
    if "layout_mm" not in model:
        if "page_mm" not in model:
            return None
        model["layout_mm"] = model["page_mm"]
    model.setdefault("page_mm", model["layout_mm"])
    W, H = model["layout_mm"]
    if "marker_corner_mm" not in model:
        model["marker_corner_mm"] = {"TL": (20.0, 23.0), "TR": (W - 50.0, 23.0),
                                     "BL": (20.0, H - 54.0), "BR": (W - 50.0, H - 54.0)}
        model["source"] = model.get("source", "layout rule")
    if "square_centers_mm" not in model:
        model["square_centers_mm"] = {
            c: {r: (x + dx, y + dy) for r, (dx, dy) in _FP_SQUARE_OFFSETS_MM.items()}
            for c, (x, y) in model["marker_corner_mm"].items()}
    # 10 cm bars (xy_draw10cm, PS:103-125): 1 mm wide, x 20.4 -> 120, at y = 55 and H - 22.
    model["bars_mm"] = [(20.4, 55.0, 120.0, 1.0), (20.4, H - 22.0, 120.0, 1.0)]
    bars = (ent or {}).get("scale_bar_mm")
    if isinstance(bars, dict):
        got = []
        for b in bars.values():
            if isinstance(b, dict):
                x0, x1, y = _fp_num(b.get("x0")), _fp_num(b.get("x1")), _fp_num(b.get("y"))
                if None not in (x0, x1, y):
                    got.append((x0, y, x1, _fp_num(b.get("line_width")) or 1.0))
        if got:
            model["bars_mm"] = got
    return model


# ---- drawing helpers ---------------------------------------------------------------------------
def _fp_dashed_line(d, p0, p1, fill, width=1, dash=6, gap=4):
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L <= 0:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    t = 0.0
    while t < L:
        t1 = min(L, t + dash)
        d.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * t1, y0 + uy * t1)], fill=fill, width=width)
        t = t1 + gap


def _fp_dashed_poly(d, pts, fill, width=1, dash=6, gap=4):
    for i in range(len(pts)):
        _fp_dashed_line(d, pts[i], pts[(i + 1) % len(pts)], fill, width, dash, gap)


def _fp_square(cx, cy, side):
    h = side / 2.0
    return [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]


def _fp_label(d, xy, text, color, font, alpha=255):
    """App-style label: bold, centered on the square center, with a black shadow (the app's
    blur-2 shadow, rendered as a stroke)."""
    sw = max(1, int(round(font.size / 16)))
    d.text(xy, text, font=font, fill=tuple(color) + (alpha,), anchor="mm",
           stroke_width=sw, stroke_fill=(0, 0, 0, alpha))


def _fp_draw_marker(layer, pts, S, *, style, labels=True, cm_text=None, outline_w=1):
    """Draw one marker in the app's FPfit style onto an RGBA `layer` (display pixels).

    `pts` {role: (x, y)} includes the predicted BR; `S` is display px per cm. `style` is "used"
    (BR filled green, black outline), "rejected" (BR outlined red), or "inferred" (dashed magenta
    outlines of all five cells, labels at reduced alpha). The BR square is drawn BEFORE the labels,
    as in drawFpOverlay."""
    d = ImageDraw.Draw(layer)
    alpha = 150 if style == "inferred" else 255
    if style == "inferred":
        for r in FP_ROLES + ("BR",):
            if r in pts:
                _fp_dashed_poly(d, _fp_square(*pts[r], S), FP_INFERRED + (255,),
                                width=max(1, outline_w + 1), dash=max(3, int(S / 8)),
                                gap=max(2, int(S / 12)))
    elif "BR" in pts:
        sq = _fp_square(*pts["BR"], S)
        if style == "used":
            d.polygon(sq, fill=FP_BR_USED + (255,), outline=(0, 0, 0, 255), width=outline_w)
        else:
            d.polygon(sq, outline=FP_BR_REJECTED + (255,), width=max(2, 2 * outline_w))
    if not labels:
        return
    f = _fp_font(FP_LABEL_FRAC * S)
    for r in FP_ROLES:
        if r in pts:
            _fp_label(d, pts[r], r, FP_ROLE_COLORS[r], f, alpha)
    if cm_text and "TL" in pts:
        x, y, _bb = _fp_cm_text_geom(d, _fp_top_left_cell(pts), f, cm_text)
        d.text((x, y), cm_text, font=f, fill=(255, 255, 255, alpha), anchor="ls",
               stroke_width=max(1, int(round(f.size / 16))), stroke_fill=(0, 0, 0, alpha))


def _fp_top_left_cell(pts):
    """Center of the marker's ON-IMAGE top-left corner cell: of TL, TR, BL and the predicted BR,
    the one with the smallest x + y (ties go to TL).

    The app draws on an image it has already rotated upright, where that cell IS TL; the crop here
    is as photographed, so on a sheet at 90/180/270 degrees TL is another corner of the marker and
    "above TL" would land on its middle row, over the C label. Same rule as
    ``reporting.fieldprism_viz._top_left_cell``."""
    return min((pts[r] for r in ("TL", "TR", "BL", "BR") if r in pts), key=lambda p: p[0] + p[1])


def _fp_cm_text_geom(d, anchor, f, text):
    """Where the app puts its per-marker "1 cm = N px" text: white, x = xA - textW/4, baseline
    yA - inkH("TL") - 5, with A the marker's on-image top-left cell (:func:`_fp_top_left_cell`;
    TL on an upright sheet, exactly the app's formula). Returns (x, baseline_y, ink bbox)."""
    bb = d.textbbox((0, 0), "TL", font=f, anchor="ls")
    x = anchor[0] - d.textlength(text, font=f) / 4.0
    y = anchor[1] - (bb[3] - bb[1]) - 5
    return x, y, d.textbbox((x, y), text, font=f, anchor="ls")


def _fp_marker_style(m):
    """'used' | 'rejected' | 'failed' | 'skipped' for one ruler_FP_marker row."""
    status, verdict = m.get("status"), m.get("verdict")
    if status != "measured":
        return "failed"
    if verdict == "rejected" or not m.get("valid"):
        return "rejected"            # failed the per-marker checks, or disagreed with its peers
    if verdict == "skipped":
        return "skipped"             # valid, but not used (e.g. a second box on the same marker)
    return "used"


def _fp_check_values(m, roles):
    """The per-marker checks as {check: (value, limit, ok, unit)}: validation_json first (whatever
    the check names, matched by meaning), then recomputed from the stored centers for anything the
    row does not carry. Checks validation_json has that this panel does not know are returned
    under their own names so nothing the engine judged is hidden."""
    val = _fp_json(m.get("validation_json"), {}) or {}
    if not isinstance(val, dict):
        val = {}
    lim = _fp_limits()

    def pick(*needles, avoid=()):
        for k in val:
            kl = str(k).lower()
            if all(n in kl for n in needles) and not any(a in kl for a in avoid):
                return k
        return None

    keys = {"pitch_ratio": pick("pitch", "ratio") or pick("a_b") or pick("ab_ratio"),
            "right_angle": pick("angle", avoid=("orient",)),
            "c_mid": pick("mid"),
            "c_diag": pick("diag"),
            "peak_area": pick("peak") or pick("area"),
            "orientation": pick("orient")}

    calc = {}
    if all(r in roles for r in ("TL", "TR", "BL")):
        tl, tr, bl = (np.asarray(roles[r], float) for r in ("TL", "TR", "BL"))
        a, b = float(np.linalg.norm(tr - tl)), float(np.linalg.norm(bl - tl))
        if a > 0 and b > 0:
            pitch = (a + b) / 2.0
            calc["pitch_ratio"] = abs(a / b - 1.0)
            cosang = float(np.clip(np.dot(tr - tl, bl - tl) / (a * b), -1.0, 1.0))
            calc["right_angle"] = abs(90.0 - math.degrees(math.acos(cosang)))
            if "C" in roles:
                c = np.asarray(roles["C"], float)
                calc["c_mid"] = float(np.linalg.norm(c - (tr + bl) / 2.0)) / pitch
                calc["c_diag"] = abs(float(np.linalg.norm(tl - c)) * math.sqrt(2.0) / pitch - 1.0)
    if _fp_num(m.get("peak_area_ratio")) is not None:
        calc["peak_area"] = _fp_num(m.get("peak_area_ratio"))

    out = {}
    for chk in ("pitch_ratio", "right_angle", "c_mid", "c_diag", "peak_area"):
        k = keys.get(chk)
        v = val.get(k) if k is not None else None
        if isinstance(v, dict) and _fp_num(v.get("value")) is not None:
            value, limit = _fp_num(v.get("value")), _fp_num(v.get("limit"))
            ok = v.get("ok")
        else:
            value, limit, ok = calc.get(chk), lim.get(chk), None
        if limit is None:
            limit = lim.get(chk)
        if ok is None and value is not None and limit is not None:
            ok = value >= limit if chk == "peak_area" else value <= limit
        out[chk] = (value, limit, ok)
    used = {k for k in keys.values() if k}
    extra = {k: v for k, v in val.items() if k not in used and isinstance(v, dict)}
    return out, extra, (val.get(keys["orientation"]) if keys["orientation"] else None)


_FP_CHECK_NAMES = {"pitch_ratio": "|a/b - 1|", "right_angle": "right angle", "c_mid": "C-mid",
                   "c_diag": "C-diagonal", "peak_area": "peak-area ratio"}


def _fp_fmt_pct(v, nd=2):
    return "-" if v is None else f"{100.0 * v:.{nd}f} %"


def _fp_ok_text(ok):
    return "-" if ok is None else ("ok" if ok else "FAIL")


def _fp_lines(d, lines, x, y, LH):
    """Draw (text, font, color) lines; an indented F_XS line gets the color swatch, as in the
    other sections, and a tab-led line is aligned with the swatched lines' text (a table header)."""
    for txt, f, c in lines:
        if txt:
            if txt.startswith("\t"):
                d.text((x + 18, y), txt[1:], font=f, fill=c)
            elif f is F_XS and txt.startswith("    "):
                d.rectangle([x, y + 3, x + 12, y + 13], fill=c)
                d.text((x + 18, y), txt.strip(), font=f, fill=c)
            else:
                d.text((x, y), txt, font=f, fill=c)
        y += LH[id(f)]
    return y


def _fp_load_crop(crop_path):
    if not crop_path or not os.path.exists(str(crop_path)):
        return None
    img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def fp_marker_section(marker, crop_path, crop_box, *, title_index=None):
    """One FieldPrism marker: the crop with its four printed 1 cm squares labeled exactly as the
    FieldPrism app labels them (TL red, TR yellow, C cyan, BL white, and the PREDICTED empty BR cell
    as a 1 cm square -- green with a black outline when the marker was used, red outline when it
    was not), next to every geometric check that decided whether the marker could be trusted.

    `marker` is a ruler_FP_marker row (dict). `crop_box` is the archival box (x1, y1, x2, y2) in
    WORKING px; crop pixel (u, v) is working pixel (max(0, round(x1)) + u, max(0, round(y1)) + v).
    A missing crop file still renders (a gray placeholder of the box's size), as does a row with no
    centers (a failed marker), so the panel can always be rebuilt from the DB.
    """
    m = dict(marker or {})
    bx1, by1, bx2, by2 = (float(v) for v in crop_box)
    ox, oy = max(0, int(round(bx1))), max(0, int(round(by1)))
    crop = _fp_load_crop(crop_path)
    box_w, box_h = max(8, int(round(bx2)) - ox), max(8, int(round(by2)) - oy)
    if crop is None:
        crop = np.full((box_h, box_w, 3), 225, np.uint8)
    ch, cw = crop.shape[:2]
    # The crop is cut from the working image at the box, so its size IS the box size; if a crop on
    # disk was cut at another resolution, map the working-frame centers onto it proportionally.
    sx, sy = cw / float(box_w), ch / float(box_h)
    if abs(sx - 1.0) < 0.02 and abs(sy - 1.0) < 0.02:
        sx = sy = 1.0
    max_w = int(0.42 * W_OUT)
    k = min(FP_CROP_H / float(ch), max_w / float(cw))
    dw, dh = max(1, int(round(cw * k))), max(1, int(round(ch * k)))
    view = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
    view = Image.fromarray(view).convert("RGBA")

    style = _fp_marker_style(m)
    roles = _fp_marker_roles(m)
    pxcm = _fp_num(m.get("pxcm"))
    disp = {r: ((x - ox) * sx * k, (y - oy) * sy * k) for r, (x, y) in roles.items()}
    if disp and pxcm and style != "failed":
        S = pxcm * sx * k
        cm_text = f"1 cm = {pxcm:.0f} px"
        if "TL" in disp:
            # A tight crop leaves no room for the app's "1 cm" text above TL: widen the canvas
            # with a dark border (the app draws on the whole image) instead of clipping it.
            probe = ImageDraw.Draw(view)
            _x, _y, bb = _fp_cm_text_geom(probe, _fp_top_left_cell(disp),
                                          _fp_font(FP_LABEL_FRAC * S), cm_text)
            pl, pt = max(0, int(math.ceil(4 - bb[0]))), max(0, int(math.ceil(4 - bb[1])))
            pr = max(0, int(math.ceil(bb[2] + 4 - dw)))
            if pl or pt or pr:
                canvas = Image.new("RGBA", (dw + pl + pr, dh + pt), (34, 34, 40, 255))
                canvas.paste(view, (pl, pt))
                view, dw, dh = canvas, dw + pl + pr, dh + pt
                disp = {r: (x + pl, y + pt) for r, (x, y) in disp.items()}
        layer = Image.new("RGBA", view.size, (0, 0, 0, 0))
        _fp_draw_marker(layer, disp, S, style=style if style != "skipped" else "rejected",
                        cm_text=cm_text)
        view = Image.alpha_composite(view, layer)
    view = view.convert("RGB")

    # ---- the text ----
    det = m.get("detection_id")
    conf = _fp_num(m.get("det_conf"))
    status, verdict = m.get("status") or "?", m.get("verdict")
    vchecks, extra, orient_check = _fp_check_values(m, roles)
    failed_checks = [_FP_CHECK_NAMES.get(n, n) for n, (_v, _l, ok) in vchecks.items()
                     if ok is False]
    reasons = _fp_json(m.get("validation_reasons"), None) or []
    title = "1 -- FieldPrism Marker" + ("" if title_index is None else f" {title_index}")
    if style == "failed":
        state = (f"NOT MEASURED ({status}) -- {m.get('status_reason') or 'no reason recorded'}",
                 _FP_BAD)
    else:
        v = ("VALID" if m.get("valid") else
             "FAILED VALIDATION" + (f" ({', '.join(failed_checks)})" if failed_checks else ""))
        state = (f"MEASURED, {v}  ->  {(verdict or 'no verdict').upper()}",
                 {"used": _FP_OK, "rejected": _FP_BAD, "skipped": _FP_GRAY}[style])
    note = m.get("verdict_note") or (m.get("status_reason") if style != "failed" else None)
    head = [(title, F_T, (15, 15, 20)),
            (f"{m['image_name'] + '   ' if m.get('image_name') else ''}"
             f"[detection {'-' if det is None else det}, ruler conf "
             f"{'-' if conf is None else '%.2f' % conf}]   RulerClassifier -> FP   "
             f"box ({bx1:.0f}, {by1:.0f}) - ({bx2:.0f}, {by2:.0f})"
             + (f"   ROI ({m.get('roi_x0')}, {m.get('roi_y0')}) - ({m.get('roi_x1')}, "
                f"{m.get('roi_y1')})" if m.get("roi_x0") is not None else ""),
             F_S, (60, 62, 70)),
            (state[0], F_B, state[1])]
    # one line per reason: a verdict note joins several with "; " and would run off the page
    parts = [p.strip() for p in str(note or "").split("; ") if p.strip()]
    parts += [str(r) for r in (reasons if isinstance(reasons, list) else [reasons])
              if str(r) not in parts]
    for p in parts:
        head.append((f"    {p}", F_XS, state[1]))
    head.append(("Labels as in the FieldPrism app: TL red, TR yellow, C cyan, BL white; the green "
                 "square is the PREDICTED empty BR cell (TR + BL - TL), 1 cm wide.", F_XS, _FP_GRAY))
    head.append(("CF = (|TR - TL| + |BL - TL|) / 2 / 2.0 cm  (square-center pitch is 20 mm; ink "
                 "edges bloom, so edge length is never used).", F_XS, _FP_GRAY))

    def row(name, value, limit="", res=None, col=None):
        txt = f"{name:<22s}{value:>13s}   {limit:<12s}{'' if res is None else _fp_ok_text(res)}"
        c = col or (_FP_GRAY if res is None else (_FP_OK if res else _FP_BAD))
        return (txt, F_S, c)

    def fv(v, fmt):
        return "-" if v is None else fmt % v

    ph, pv = _fp_num(m.get("pitch_h_px")), _fp_num(m.get("pitch_v_px"))
    vr, lr, okr = vchecks["pitch_ratio"]
    va, la, oka = vchecks["right_angle"]
    vm, lm, okm = vchecks["c_mid"]
    vd, ld, okd = vchecks["c_diag"]
    vp, lp, okp = vchecks["peak_area"]
    orient = m.get("orientation_deg")
    orient_txt = "-" if orient is None else f"{int(orient)} deg"
    if isinstance(orient_check, dict):
        # only an orientation check the engine actually ran can pass or fail the row
        orient_row = row("orientation vote", orient_txt, "upright = 0", orient_check.get("ok"))
    elif orient in (None, 0):
        orient_row = row("orientation vote", orient_txt, "upright = 0", col=_FP_GRAY)
    else:
        # informational: a rotated sheet is supported (the empty BR cell gives the rotation) and
        # never rejects a marker, so it is a warning like the sheet section's, not a FAIL
        orient_row = (f"{'orientation vote':<22s}{orient_txt:>13s}   {'upright = 0':<12s}rotated",
                      F_S, _FP_WARN)
    pct = _fp_num(m.get("pct_vs_fp"))
    pxo = _fp_num(m.get("pxcm_original"))
    table = [(f"{'check':<22s}{'value':>13s}   {'limit':<12s}result", F_S, (15, 15, 20)),
             row("pitch TL->TR (h)", fv(ph, "%.2f px")),
             row("pitch TL->BL (v)", fv(pv, "%.2f px")),
             row("|a/b - 1|", _fp_fmt_pct(vr), "<= " + _fp_fmt_pct(lr, 1) if lr is not None else "",
                 okr),
             row("right-angle error", fv(va, "%.2f deg"),
                 "<= " + fv(la, "%.1f deg") if la is not None else "", oka),
             row("C-mid error", _fp_fmt_pct(vm), "<= " + _fp_fmt_pct(lm, 1) if lm is not None else "",
                 okm),
             row("C-diagonal error", _fp_fmt_pct(vd),
                 "<= " + _fp_fmt_pct(ld, 1) if ld is not None else "", okd),
             row("peak-area ratio", fv(vp, "%.3f"), ">= " + fv(lp, "%.2f") if lp is not None else "",
                 okp)]
    for name, v in extra.items():
        table.append(row(str(name)[:22], fv(_fp_num(v.get("value")), "%.4g"),
                         fv(_fp_num(v.get("limit")), "%.4g"), v.get("ok")))
    table += [row("peaks / holes filled",
                  f"{'-' if m.get('n_peaks') is None else m['n_peaks']} / "
                  f"{'-' if m.get('holes_filled') is None else m['holes_filled']}"),
              orient_row,
              ("", F_XS, _FP_GRAY),
              row("px/cm (working)", fv(pxcm, "%.2f"), col=(15, 15, 20)),
              row("px/cm (original)", fv(pxo, "%.2f"), col=_FP_GRAY),
              row("vs FieldPrism anchor", fv(pct, "%+.2f %%"), "",
                  col=_FP_GRAY if pct is None else (15, 15, 20)),
              row("sheet corner", str(m.get("sheet_corner") or "-"), "",
                  col=(15, 15, 20) if m.get("sheet_corner") else _FP_GRAY)]

    LH = {id(F_T): 30, id(F_B): 24, id(F_S): 20, id(F_XS): 18}
    h_head = 10 + sum(LH[id(f)] for _, f, _ in head) + 8
    h_table = sum(LH[id(f)] for _, f, _ in table)
    H = h_head + max(dh + 6, h_table) + PAD
    im = Image.new("RGB", (W_OUT, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    _fp_lines(d, head, PAD, 10, LH)
    # a colored frame says the verdict at a glance without covering the app-style labels
    frame = {"used": _FP_OK, "rejected": _FP_BAD, "failed": _FP_BAD, "skipped": _FP_GRAY}[style]
    d.rectangle([PAD - 3, h_head - 3, PAD + dw + 2, h_head + dh + 2], outline=frame, width=3)
    im.paste(view, (PAD, h_head))
    _fp_lines(d, table, PAD + dw + 32, h_head, LH)
    return im


# ---- the sheet ---------------------------------------------------------------------------------
_FP_STATUS_COLOR = {"identified": _FP_OK, "ambiguous": _FP_WARN, "undetermined": _FP_WARN,
                    "unrecognized": _FP_BAD}


def _fp_corner_map(sheet, markers):
    """{corner: {"observed": bool, "detection_id", "squares": {role: (x, y)}, "marker": row|None}}
    from corners_json, completed from the marker rows' sheet_corner for anything it lacks."""
    by_det = {m.get("detection_id"): m for m in markers or [] if isinstance(m, dict)}
    out = {}
    corners = _fp_json(sheet.get("corners_json"), {}) or {}
    if isinstance(corners, dict):
        for c in FP_CORNERS:
            e = corners.get(c)
            if not isinstance(e, dict):
                continue
            det = e.get("detection_id")
            sq = e.get("squares") if isinstance(e.get("squares"), dict) else {}
            out[c] = {"observed": bool(e.get("observed")), "detection_id": det,
                      "squares": {r: p for r, p in ((r, _fp_pt(sq.get(r)))
                                                    for r in FP_ROLES + ("BR",)) if p},
                      "marker": by_det.get(det)}
    for m in markers or []:
        c = m.get("sheet_corner") if isinstance(m, dict) else None
        if c in FP_CORNERS and c not in out:
            out[c] = {"observed": True, "detection_id": m.get("detection_id"),
                      "squares": _fp_marker_roles(m), "marker": m}
    # An inferred corner often HAS a detection that was dropped (failed or rejected): name it, by
    # the box that contains the reconstructed C square.
    taken = {e.get("detection_id") for e in out.values() if e.get("observed")}
    for e in out.values():
        cpt = e["squares"].get("C")
        if e["observed"] or cpt is None:
            continue
        for m in markers or []:
            if not isinstance(m, dict) or m.get("detection_id") in taken:
                continue
            bx = [_fp_num(m.get(k)) for k in ("x1", "y1", "x2", "y2")]
            if None not in bx and bx[0] <= cpt[0] <= bx[2] and bx[1] <= cpt[1] <= bx[3]:
                e["near"] = m
                break
    return out


def _fp_candidate_rows(cands, chosen_type, chosen_assign, n=5):
    lines = [(f"\t{'#':<3s}{'sheet':<14s}{'markers -> corners':<44s}{'cost':>7s}{'rms':>7s}"
              f"{'scale':>8s}{'rot':>8s}{'inside':>8s}{'margins':>9s}", F_XS, (15, 15, 20))]
    for i, c in enumerate(cands[:n]):
        if not isinstance(c, dict):
            continue
        asg = c.get("assignment") if isinstance(c.get("assignment"), dict) else {}
        order = {k: j for j, k in enumerate(FP_CORNERS)}
        txt = " ".join(f"{v}:det{k}" for k, v in sorted(asg.items(),
                                                       key=lambda kv: order.get(kv[1], 9)))

        def num(key, fmt, c=c):
            return "-" if _fp_num(c.get(key)) is None else fmt % _fp_num(c[key])
        ins = c.get("inside_image")
        best = (c.get("sheet_type") == chosen_type
                and (not chosen_assign or {str(k): v for k, v in asg.items()}
                     == {str(k): v for k, v in chosen_assign.items()}))
        lines.append((f"    {i + 1:<3d}{str(c.get('sheet_type') or '-')[:13]:<14s}{txt[:43]:<44s}"
                      f"{num('cost_mm', '%7.2f')}{num('rms_mm', '%7.2f')}"
                      f"{num('scale_dev_mm', '%8.2f')}{num('rotation_deg', '%+8.2f')}"
                      f"{('-' if ins is None else 'yes' if ins else 'no'):>8s}"
                      f"{num('margin_spread_mm', '%9.2f')}",
                      F_XS, _FP_OK if best else (90, 95, 105)))
    if len(lines) == 1:
        lines.append(("    (no candidate hypotheses recorded)", F_XS, _FP_GRAY))
    return lines


def _fp_page_schematic(model, cmap, margins, *, height=FP_SCHEMATIC_H):
    """The printed sheet to scale (mm): page outline, the 10 cm bars, and each corner's marker --
    observed markers printed solid (BR green = used, red outline = measured but not used),
    reconstructed ones as dashed magenta outlines, unseen ones as faint outlines. The FPfit crop
    implied by the margins is the dashed blue rectangle. Corner captions sit outside the page."""
    pw, ph = model["page_mm"]
    ox, oy = 20, 44                       # room above the page for the title + top captions
    k = (height - 2 * oy) / float(ph)
    caps = {}
    for c in FP_CORNERS:
        e = cmap.get(c)
        m, det = (e or {}).get("marker"), (e or {}).get("detection_id")
        if e is None:
            state, col = "not seen", _FP_GRAY
        elif not e["observed"]:
            near = e.get("near")
            state, col = "inferred", FP_INFERRED
            if near is not None:
                state += (f" (det{near.get('detection_id')} "
                          f"{near.get('verdict') or _fp_marker_style(near)})")
        else:
            st = _fp_marker_style(m) if m else "used"
            state = (m.get("verdict") or st) if m else "observed"
            col = _FP_OK if st == "used" else _FP_BAD
        caps[c] = (f"{c}{'' if det is None else ' det%s' % det}  {state}", col)

    def cw(c):
        return F_XS.getlength(caps[c][0])
    # wide enough that a left and a right caption never overlap
    W = int(max(round(pw * k), cw("TL") + cw("TR") + 16, cw("BL") + cw("BR") + 16)) + 2 * ox
    H = height
    im = Image.new("RGB", (W, H), (244, 244, 246))
    d = ImageDraw.Draw(im)

    def P(x, y):
        return (ox + x * k, oy + y * k)

    d.rectangle([*P(0, 0), *P(pw, ph)], fill=(255, 255, 255), outline=(110, 110, 120), width=1)
    for x0, y, x1, lw in model["bars_mm"]:
        d.rectangle([*P(x0, y - lw / 2), *P(x1, y + lw / 2)], fill=(150, 150, 155))
    # the FPfit crop: margins are measured from the outermost square centers (sheet frame)
    if isinstance(margins, dict) and all(_fp_num(margins.get(s)) is not None
                                         for s in ("left", "right", "top", "bottom")):
        cen = [p for c in FP_CORNERS for r, p in model["square_centers_mm"][c].items() if r != "BR"]
        xs, ys = [p[0] for p in cen], [p[1] for p in cen]
        lft, r = min(xs) - _fp_num(margins["left"]), max(xs) + _fp_num(margins["right"])
        t, b = min(ys) - _fp_num(margins["top"]), max(ys) + _fp_num(margins["bottom"])
        _fp_dashed_poly(d, [P(lft, t), P(r, t), P(r, b), P(lft, b)], (56, 132, 255), width=2,
                        dash=8, gap=5)
    side = 10.0 * k
    for c in FP_CORNERS:
        sq = model["square_centers_mm"][c]
        e = cmap.get(c)
        m = (e or {}).get("marker")
        if e is None:
            for r in FP_ROLES:
                d.polygon(_fp_square(*P(*sq[r]), side), outline=(190, 190, 196))
        elif not e["observed"]:
            for r in FP_ROLES + ("BR",):
                _fp_dashed_poly(d, _fp_square(*P(*sq[r]), side), FP_INFERRED, width=2,
                                dash=4, gap=3)
        else:
            st = _fp_marker_style(m) if m else "used"
            for r in FP_ROLES:
                d.polygon(_fp_square(*P(*sq[r]), side), fill=(20, 20, 24))
            br = _fp_square(*P(*sq["BR"]), side)
            if st == "used":
                d.polygon(br, fill=FP_BR_USED, outline=(0, 0, 0))
            else:
                d.polygon(br, outline=FP_BR_REJECTED, width=2)
        lab, col = caps[c]
        ty = oy - 18 if c in ("TL", "TR") else P(0, ph)[1] + 3
        tx = ox if c in ("TL", "BL") else max(P(pw, 0)[0], W - ox) - d.textlength(lab, font=F_XS)
        d.text((max(2, tx), ty), lab, font=F_XS, fill=col)
    lw, lh = model["layout_mm"]
    d.text((ox, 4), f"{model['label']}  {pw:g} x {ph:g} mm  [{model.get('source', '')}]",
           font=F_XS, fill=(60, 62, 70))
    d.text((ox, H - 20), f"same-role square pitch {lw - 70:g} x {lh - 77:g} mm", font=F_XS,
           fill=(90, 95, 105))
    return im


def _fp_fit_schematic(sheet, cmap, markers, *, height=FP_SCHEMATIC_H):
    """The fit in the WORKING image (px), on a dark ground like the app overlay: the page outline
    implied by the fit (magenta), every corner's squares filled in the app's role colors (observed)
    or as dashed magenta outlines (reconstructed), and each measured center as a cross."""
    pts = []
    page = [p for p in (_fp_pt(q) for q in (_fp_json(sheet.get("page_corners_json"), []) or []))
            if p]
    pts += page
    for e in cmap.values():
        pts += list(e["squares"].values())
    meas = []
    for m in markers or []:
        if isinstance(m, dict) and m.get("status") == "measured":
            meas += [p for r, p in _fp_marker_roles(m).items() if r != "BR"]
    pts += meas
    # One marker alone has no sheet-scale geometry to show (its own panel already shows it).
    if len(page) < 4 and sum(1 for e in cmap.values() if e["squares"]) < 2:
        return None
    # one printed square = 1 cm: the fit's scale, else (no fit) the markers' own mean px/cm
    pxcm = [_fp_num(m.get("pxcm")) for m in markers or [] if isinstance(m, dict)]
    pxcm = [v for v in pxcm if v]
    s_mm = _fp_num(sheet.get("fit_scale_px_per_mm"))
    cm = 10.0 * s_mm if s_mm else (float(np.mean(pxcm)) if pxcm else 0.0)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs) - cm / 2, max(xs) + cm / 2, min(ys) - cm / 2, max(ys) + cm / 2
    span_x, span_y = max(1.0, x1 - x0), max(1.0, y1 - y0)
    ox, oy = 24, 34
    k = min((height - 2 * oy) / span_y, (FP_FIT_MAX_W - 2 * ox) / span_x)
    W = int(round(span_x * k)) + 2 * ox
    im = Image.new("RGB", (max(W, 300), height), (34, 34, 40))
    d = ImageDraw.Draw(im, "RGBA")

    def P(p):
        return (ox + (p[0] - x0) * k, oy + (p[1] - y0) * k)

    if len(page) == 4:
        q = [P(p) for p in page]
        d.line(q + q[:1], fill=FP_INFERRED, width=1)
    side = max(4.0, cm * k)
    for e in cmap.values():
        sq = e["squares"]
        if e["observed"]:
            st = _fp_marker_style(e["marker"]) if e.get("marker") else "used"
            for r in FP_ROLES:
                if r in sq:
                    d.polygon(_fp_square(*P(sq[r]), side), fill=FP_ROLE_COLORS[r],
                              outline=(0, 0, 0))
            if "BR" in sq:
                br = _fp_square(*P(sq["BR"]), side)
                if st == "used":
                    d.polygon(br, fill=FP_BR_USED, outline=(0, 0, 0))
                else:
                    d.polygon(br, outline=FP_BR_REJECTED, width=2)
        else:
            for r in FP_ROLES + ("BR",):
                if r in sq:
                    _fp_dashed_poly(d, _fp_square(*P(sq[r]), side), FP_INFERRED, width=2,
                                    dash=4, gap=3)
    for p in meas:
        x, y = P(p)
        for w, c in ((3, (0, 0, 0)), (1, (255, 255, 255))):
            d.line([(x - 6, y), (x + 6, y)], fill=c, width=w)
            d.line([(x, y - 6), (x, y + 6)], fill=c, width=w)
    d.text((8, 8), "fit in the working image (px)", font=F_XS, fill=(220, 220, 225))
    d.text((8, height - 22), "TL red  TR yellow  C cyan  BL white   + measured", font=F_XS,
           fill=(200, 200, 205))
    return im


def fp_sheet_section(sheet, markers):
    """Which printed FieldPrism sheet the markers belong to, and the CF that geometry gives.

    `sheet` is a ruler_FP_sheet row (dict; the *_json columns as stored strings or already parsed)
    and `markers` the sheet's ruler_FP_marker rows. Shows the sheet type and status, the marker
    counts, orientation and fit quality, the CF three ways (sheet fit, marker mean, the anchor
    actually used), the FieldPrism confidence with its reasons, the top 5 ranked sheet hypotheses,
    and two schematics: the catalog sheet drawn to scale with observed vs inferred markers, and
    the fit in the working image. Every field may be missing; the section still renders.
    """
    s = dict(sheet or {})
    markers = [m for m in (markers or []) if isinstance(m, dict)]
    status = str(s.get("sheet_status") or "undetermined")
    cands = _fp_json(s.get("sheet_candidates_json"), []) or []
    cands = cands if isinstance(cands, list) else []
    reasons = _fp_json(s.get("fp_reasons_json"), []) or []
    reasons = reasons if isinstance(reasons, list) else [reasons]
    margins = _fp_json(s.get("fpfit_margins_json"), None)
    margins = margins if isinstance(margins, dict) else None
    cmap = _fp_corner_map(s, markers)

    shown_type = s.get("sheet_type") or next(
        (c.get("sheet_type") for c in cands if isinstance(c, dict) and c.get("sheet_type")), None)
    model = _fp_sheet_model(shown_type) if shown_type else None
    label = s.get("sheet_label") or (model or {}).get("label") or shown_type
    others = []
    for c in cands:
        t = c.get("sheet_type") if isinstance(c, dict) else None
        if t and t != shown_type and t not in others:
            others.append(t)
    if status == "identified":
        head_txt = f"Sheet: FieldPrism {label}  --  IDENTIFIED"
    elif status == "ambiguous":
        head_txt = (f"Sheet: FieldPrism {label}?  --  AMBIGUOUS"
                    + (f" (or {', '.join(others[:3])})" if others else ""))
    elif status == "unrecognized":
        head_txt = "Sheet: unrecognized  --  no catalog sheet fits the marker positions"
    else:
        head_txt = "Sheet: undetermined  --  fewer than 2 usable markers"
    head_col = _FP_STATUS_COLOR.get(status, _FP_WARN)

    n = {k: s.get(k) for k in ("n_fp_detected", "n_fp_measured", "n_fp_valid", "n_fp_used",
                               "n_fp_rejected", "n_fp_inferred")}

    def cnt(k):
        return "-" if n[k] is None else str(n[k])

    def fnum(k, fmt):
        return "-" if _fp_num(s.get(k)) is None else fmt % _fp_num(s[k])
    orient = s.get("orientation_deg")

    lines = [("FieldPrism Sheet Identification", F_T, (15, 15, 20)),
             (head_txt, F_B, head_col)]
    if s.get("corners_ambiguous"):
        lines.append(("Layout not certain: the sheet size is known, but which marker sits at which "
                      "corner (or which printing of the layout, e.g. Legal vs legacy Legal) is not -- "
                      "no markers are inferred"
                      + (" and the sheet fit is not used as the CF." if
                         s.get("cf_source_detail") != "sheet_fit" else "."),
                      F_S, _FP_WARN))
    lines += [
        (f"markers: {cnt('n_fp_detected')} detected, {cnt('n_fp_measured')} measured, "
         f"{cnt('n_fp_valid')} valid, {cnt('n_fp_used')} used, {cnt('n_fp_rejected')} rejected, "
         f"{cnt('n_fp_inferred')} inferred", F_S, (60, 62, 70)),
        (f"orientation {'-' if orient is None else '%d deg' % int(orient)}"
         f"{'' if orient in (None, 0) else ' (NOT upright: image is rotated)'}   "
         f"fit rotation {fnum('fit_rotation_deg', '%+.2f deg')}   "
         f"scale {fnum('fit_scale_px_per_mm', '%.3f px/mm')}   rms {fnum('fit_rms_mm', '%.2f mm')}   "
         f"max {fnum('fit_max_mm', '%.2f mm')}   scale dev {fnum('fit_scale_dev_mm', '%.2f mm')}   "
         f"cost {fnum('fit_cost_mm', '%.2f mm')}",
         F_S, _FP_WARN if orient not in (None, 0) else (60, 62, 70))]
    if margins:
        vals = [_fp_num(margins.get(k)) for k in ("left", "right", "top", "bottom")]
        if all(v is not None for v in vals):
            lines.append((f"FPfit margins (outermost square centers to the image edges): left "
                          f"{vals[0]:.1f}  right {vals[1]:.1f}  top {vals[2]:.1f}  bottom "
                          f"{vals[3]:.1f} mm   (spread {max(vals) - min(vals):.1f} mm)",
                          F_S, (60, 62, 70)))
    cf_fp = _fp_num(s.get("cf_px_per_cm_fp"))
    cf_fit, cf_mean = _fp_num(s.get("cf_px_per_cm_sheet_fit")), _fp_num(s.get("cf_px_per_cm_marker_mean"))
    detail = s.get("cf_source_detail")

    def rel(v):
        return "" if v is None or not cf_fp else f"   ({100.0 * (v / cf_fp - 1.0):+.2f}% vs anchor)"
    spread = _fp_num(s.get("fp_peer_spread_pct"))
    lines += [
        ("", F_XS, _FP_GRAY),
        (f"CF from the sheet fit      {'-' if cf_fit is None else '%8.2f px/cm' % cf_fit}{rel(cf_fit)}"
         f"{'   [anchor]' if detail == 'sheet_fit' else ''}", F_S,
         (15, 15, 20) if detail == "sheet_fit" else (90, 95, 105)),
        (f"CF from the marker mean    {'-' if cf_mean is None else '%8.2f px/cm' % cf_mean}"
         f"{rel(cf_mean)}{'' if spread is None else '   peer spread %.2f%%' % spread}"
         f"{'   [anchor]' if detail == 'marker_mean' else ''}", F_S,
         (15, 15, 20) if detail == "marker_mean" else (90, 95, 105)),
        (f"FieldPrism anchor (working frame): {'NONE' if cf_fp is None else '%.2f px/cm' % cf_fp}"
         f"{'' if not detail else '  from the ' + str(detail).replace('_', ' ')}"
         f"   --   confidence {s.get('fp_confidence') or 'none'}",
         F_B, _FP_OK if (cf_fp is not None and s.get("fp_confidence") == "high")
         else (_FP_WARN if cf_fp is not None else _FP_BAD))]
    for r in reasons:
        lines.append((f"    - {r}", F_S, (60, 62, 70) if s.get("fp_confidence") == "high"
                      else _FP_WARN))
    lines.append(("", F_XS, _FP_GRAY))
    lines.append(("Ranked sheet hypotheses (top 5; cost = rms + scale deviation, mm):", F_S,
                  (60, 62, 70)))
    asg = _fp_json(s.get("assignment_json"), None)
    if not isinstance(asg, dict):
        asg = {str(e["detection_id"]): c for c, e in cmap.items()
               if e.get("observed") and e.get("detection_id") is not None}
    lines += _fp_candidate_rows(cands, shown_type, asg)
    if model is None:
        lines.append((f"No page schematic: {'no sheet type to draw' if not shown_type else 'sheet %s is not in the catalog' % shown_type}.",
                      F_XS, _FP_GRAY))

    LH = {id(F_T): 30, id(F_B): 24, id(F_S): 20, id(F_XS): 18}
    h_head = 10 + sum(LH[id(f)] for _, f, _ in lines) + 10
    pics = []
    if model is not None:
        pics.append(_fp_page_schematic(model, cmap, margins))
    fit = _fp_fit_schematic(s, cmap, markers)
    if fit is not None:
        pics.append(fit)
    legend = [("Schematic key", F_S, (15, 15, 20)),
              ("    observed marker: printed squares solid", F_XS, (20, 20, 24)),
              ("    BR green = predicted empty cell of a USED marker", F_XS, FP_BR_USED),
              ("    BR red outline = measured but not used", F_XS, FP_BR_REJECTED),
              ("    dashed magenta = INFERRED (reconstructed) marker", F_XS, FP_INFERRED),
              ("    dashed blue = the FPfit crop implied by the margins", F_XS, (56, 132, 255)),
              ("    faint = a corner no marker was seen at", F_XS, (170, 170, 175))]
    h_pics = max([p.size[1] for p in pics] + [0])
    H = h_head + h_pics + 2 * PAD
    im = Image.new("RGB", (W_OUT, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    _fp_lines(d, lines, PAD, 10, LH)
    x = PAD
    for p in pics:
        if x + p.size[0] > W_OUT - PAD:      # a very wide fit never pushes past the page
            p = p.resize((max(1, W_OUT - PAD - x), max(1, int(round(p.size[1] * (W_OUT - PAD - x)
                                                                     / p.size[0])))))
        im.paste(p, (x, h_head))
        x += p.size[0] + 20
    if pics and x < W_OUT - 300:
        y = h_head
        for txt, f, c in legend:                 # swatch in the key color, text always dark
            if txt.startswith("    "):
                d.rectangle([x + 10, y + 2, x + 24, y + 14], fill=c, outline=(90, 95, 105))
                d.text((x + 32, y), txt.strip(), font=f, fill=(60, 62, 70))
            else:
                d.text((x + 10, y), txt, font=f, fill=c)
            y += LH[id(f)]
    return im


def _fp_cf_summary_section(measured_cf, anchor_cf, anchor_source, fp_sheet):
    """build_cf_summary_section for a sheet with FieldPrism markers: the anchor is the FieldPrism
    geometry, and the megapixel prediction is never substituted, so the block says exactly that."""
    s = dict(fp_sheet or {})
    meas = "none" if measured_cf is None else f"{float(measured_cf):.2f} px/cm"
    detail = {"sheet_fit": "sheet fit", "marker_mean": "marker mean"}.get(
        s.get("cf_source_detail"), s.get("cf_source_detail"))
    anchor_txt = "none" if anchor_cf is None else f"{float(anchor_cf):.2f} px/cm"
    if measured_cf is not None:
        used = (f"CF used for this sheet: {float(measured_cf):.2f} px/cm -- measured, anchored on "
                f"the FieldPrism markers", _FP_OK)
    else:
        used = ("CF used for this sheet: none -- cm measurements left empty (the megapixel "
                "prediction is never used on a FieldPrism sheet)", _FP_BAD)
    lines = [("Pixel to Metric Conversion Factor", F_T, (15, 15, 20)),
             (f"Measured CF: {meas}", F_B, _FP_BAD if measured_cf is None else _FP_OK),
             (f"FieldPrism anchor{' (' + detail + ')' if detail else ''}: {anchor_txt}"
              if anchor_source == "fieldprism" or anchor_cf is not None else
              "FieldPrism anchor: none -- no FieldPrism CF could be established", F_B, (90, 95, 105))]
    if fp_sheet is not None:
        st = s.get("sheet_status") or "undetermined"
        lab = s.get("sheet_label") or s.get("sheet_type")
        n_used, n_inf = s.get("n_fp_used"), s.get("n_fp_inferred")
        mk = ("" if n_used is None else f"{n_used} marker{'' if n_used == 1 else 's'}"
              + (f" + {n_inf} inferred" if n_inf else ""))
        sheet_txt = (f"FieldPrism {lab}" if st == "identified" and lab else
                     f"FieldPrism {lab}? ({st})" if lab else f"FieldPrism sheet: {st}")
        fit, mean = _fp_num(s.get("cf_px_per_cm_sheet_fit")), _fp_num(s.get("cf_px_per_cm_marker_mean"))
        lines.append((f"{sheet_txt}{'  |  ' + mk if mk else ''}"
                      f"  |  sheet fit {'-' if fit is None else '%.2f' % fit}"
                      f"  |  marker mean {'-' if mean is None else '%.2f' % mean} px/cm"
                      f"  |  confidence {s.get('fp_confidence') or 'none'}",
                      F_S, _FP_STATUS_COLOR.get(st, _FP_WARN)))
    lines.append((used[0], F_B, used[1]))
    lines.append(("FieldPrism sheets come in several sizes, so the megapixel prediction is not used "
                  "as the anchor here; the printed marker geometry is.", F_XS, _FP_GRAY))
    LH = {id(F_T): 34, id(F_B): 26, id(F_S): 22, id(F_XS): 20}
    H = 12 + sum(LH[id(f)] for _, f, _ in lines) + 12
    im = Image.new("RGB", (W_OUT, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    y = 12
    for txt, f, col in lines:
        d.text((PAD, y), txt, font=f, fill=col)
        y += LH[id(f)]
    return im
