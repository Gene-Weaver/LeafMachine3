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

There is no CLI here. `ruler_CF_with_lattice_detection.RulerCFLattice` owns the
driving, and calls into this module so the live panel and the panel rebuilt later
from the project DB are produced by the same code.
"""

from __future__ import annotations

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
    lines = [(f"1 -- Ruler Class", F_T, (15, 15, 20)),
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
                  f"AND sits within +/-{100 * ANCHOR_TOL:.0f}% of the MP anchor",
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
    if anchor_frame == "working":
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


def build_cf_summary_section(measured_cf, anchor_cf, formula_symbolic=None):
    """The last block on every panel: the two numbers a reader came for, stated plainly.

    Everything above this is the audit trail -- which unit was named, which crop won, why a reading
    was rejected. This says only what the sheet ends up with: the CF the ruler methods produced (or
    that they produced nothing), and the megapixel regression's prediction alongside the equation
    that generated it. It is rendered for EVERY sheet, published or not, so the answer is always in
    the same place at the same end of the image.
    """
    W = W_OUT
    meas = "none" if measured_cf is None else f"{float(measured_cf):.2f} px/cm"
    eq = formula_symbolic or "megapixel regression"
    pred = "none" if anchor_cf is None else f"{float(anchor_cf):.2f} px/cm"
    lines = [
        ("Pixel to Metric Conversion Factor", F_T, (15, 15, 20)),
        (f"Measured CF: {meas}", F_B,
         (194, 65, 12) if measured_cf is None else (21, 128, 61)),
        (f"Predicted CF ({eq}): {pred}", F_B, (90, 95, 105)),
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


def build_recon_section(image_name, entries, pr, anchor, anchor_formula=None):
    """The 'Multiple Ruler Reconciliation' block: every ruler's own CF, its
    predicted 1 cm / 1 inch bars stacked for direct visual comparison, and an
    explicit account of how the single parent CF was arrived at."""
    W = W_OUT
    head_lines = []
    head_lines.append(("3 -- Multiple Ruler Reconciliation", F_T, (15, 15, 20)))
    head_lines.append((
        f"{image_name}   --   {len(entries)} ruler crops on this sheet. Every crop measures the "
        f"SAME sheet, so all must reconcile to ONE conversion factor.", F_S, (60, 62, 70)))
    cf = pr.get("cf_px_per_cm")
    meas = pr.get("cf_px_per_cm_measured")
    head_lines.append((
        f"MP anchor (working frame) {('-' if anchor is None else '%.2f px/cm' % anchor)}"
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
        if anchor is not None:
            head_lines.append((
                f"FALLBACK -- every downstream measurement on this sheet uses the MP-PREDICTED "
                f"anchor instead of a ruler reading:", F_S, (194, 65, 12)))
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
    # the fallback the sheet will actually be measured with, laid against a real ruler
    if cf is None and anchor is not None and entries:
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
