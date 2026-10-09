"""FieldPrism (FP) photogrammetric markers: 1 cm squares, roles, sheet type, conversion factor.

A FieldPrism field sheet carries four identical markers, one near each page corner. Each
marker is a 3x3 grid of 10 mm cells with the TL, TR, C (middle) and BL cells filled and the
BR cell EMPTY. All four markers are translated copies with the same orientation, so the
missing BR cell of ANY marker tells which way the page is turned, and the relative marker
positions tell which sheet size (A5/A4/A3/Letter/Legal/Tabloid) was printed.

This module is the pure-numpy/cv2 core. It has no DB access and every output is a plain,
picklable dict (Python floats/ints), in WORKING-frame pixels (the frame of
`archival_detection.x1..y2`). LM3 assumes the FPfit images are already rectified, so
nothing here warps or deskews.

APP PORT (what is cloned exactly). The square finder and the role assignment are a faithful
port of the FieldPrism apps; the two apps are the same algorithm line for line:

    KT = FieldPrism_Anrdoid/app/src/main/java/com/leafmachine/fieldprism/RulerDeskewPrecise.kt
    SW = FieldPrism_iOS/FieldPrism/FieldPrism/ImageProcessing/RulerDeskewPrecise.swift
    MM = FieldPrism_iOS/FieldPrism/FieldPrism/OpenCVWrapper.mm

  * constants                      KT:14-22, SW:11-16
  * ROI with pad (computeRoiWithPad) KT:401-411, SW:413
  * square centers (ClusterProcessor.centersFromMat) KT:1439-1501 == MM:146-261
  * roles (RoleAssigner.assign)      KT:1503-1539 == SW:729-772
  * orientation (estimateOptimalRotation) KT:307-327 == SW:312-332
  * per-marker CF (1-cluster AFFINE path) SW:1107-1130: px/cm = ((|TR-TL| + |BL-TL|)/2) / 2

  Grayscale follows iOS (MM:155, RGBA2GRAY = correct weights); in Python that is
  `cv2.COLOR_BGR2GRAY` on the BGR array `cv2.imread` returns. Android runs BGR2GRAY on an
  RGBA Mat (KT:1448), which swaps the R/B weights -- an app bug, not reproduced.

LM3 HARDENING (deliberate deviations, each flagged `LM3 DEVIATION` where it happens):

  1. Small-hole fill. After keeping the largest component, holes that do not touch the ROI
     border and are smaller than HOLE_FILL_MAX_FRAC of the component are filled. A white
     speck inside a square otherwise splits its distance-transform plateau and drags the
     center (15_1 TR marker: C off by 15 px = 1.3 mm). Filling ALL holes is harmful: a twig
     that encloses a large region (5_1 TR) then shifts C by 4% of the pitch.
     The fill can also CREATE a plateau the apps never see: on 5_1's twig-crossed TL marker
     the app finder (no fill) finds only 3 distance-transform plateaus and drops the marker
     ("found 3 square candidates (need 4)"), while the fill closes 43 twig-gap holes and a
     spurious 4th plateau appears (area 152 vs ~1950). So whenever the fill changed the
     plateau count, the app-faithful (unfilled) count is computed too (`n_peaks_app`), and a
     marker that reaches 4 squares only because of the fill says so in its status_reason
     and validation (check `app_finder_squares`).
  2. Per-marker geometric validation (pitch ratio, right angle, C at the midpoint and on the
     diagonal, peak-area balance). The apps accept ANY 4 peaks. Validation catches bad
     4-peak layouts, whether they come from the image or from deviation 1 (the fill-made
     5_1 TL layout above has a/b = 0.85 and unbalanced plateaus 0.08, so it is rejected).
  3. Sheet identification, missing-marker reconstruction and the peer/confidence rules are
     LM3 additions; the apps never identify the sheet from the image.

Ink edges image ~1.6% oversized (bloom), so the scale ALWAYS comes from square-CENTER pitch,
never from square edge length or area.
"""
from __future__ import annotations

import functools
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np

# ---- app constants (KT:14-22, SW:11-16) ---------------------------------------------------
SQUARE_PITCH_MM = 20.0        # TL->TR and TL->BL square-center distance
SQUARE_EDGE_MM = 10.0         # one cell; never used for scale (ink bloom), documentation only
DT_PEAK_FRAC = 0.55           # distance-transform plateau threshold (fraction of the max)
CROP_PAD_PX = 20              # pad added around each detector box before the square finder
MAX_DETECTIONS = 10           # the apps keep the 10 largest ruler boxes (KT:72-73, SW:71)

# ---- LM3 hardening ------------------------------------------------------------------------
HOLE_FILL_MAX_FRAC = 0.01     # fill holes (not touching the ROI border) < 1% of the component
# Per-marker validation limits. Measured on the 8 markers of the two FPfit test images the
# good markers sit at <= 0.013 / 0.06 deg / 0.005 / 0.008 / >= 0.33 (the 0.33 is the 15_1 TR
# speck marker BEFORE hole filling). The 5_1 TL marker is one the apps drop (3 plateaus);
# only LM3's hole fill gives it a 4th, and that layout fails at 0.155 pitch ratio / 0.08 area.
MAX_PITCH_RATIO_ERR = 0.05    # | |TR-TL| / |BL-TL| - 1 |
MAX_RIGHT_ANGLE_ERR_DEG = 3.0  # | angle(TR-TL, BL-TL) - 90 |
MAX_C_MID_ERR = 0.03          # |C - (TR+BL)/2| / pitch
MAX_C_DIAG_ERR = 0.05         # | |TL-C| * sqrt(2) / pitch - 1 |
MIN_PEAK_AREA_RATIO = 0.3     # min/max of the 4 peak plateau areas

# ---- sheet identification -----------------------------------------------------------------
# A hypothesis (sheet x marker->corner assignment) is judged on two SIZE-INDEPENDENT terms:
#   rms_mm  = positional residual of the similarity fit, on the sheet (mm). Shape evidence.
#   rel_dev = |ln(s_fit / s_pitch)|: how far the fit's scale (from the long marker baselines)
#             disagrees with the markers' own 20 mm square pitch. Size evidence.
# rel_dev used to be charged as |ln(..)| * span_mm against one absolute mm budget, so the same
# percentage error cost 1.65x more on A3/Tabloid (411 mm diagonal) than on Letter (249 mm) and
# real A3/Tabloid captures came out "unrecognized".
#
# MAX_SHEET_RMS_MM = 2.0: the true sheet fits at 0.2-0.5 mm on the real FPfit images (also
# with one marker blanked) and at <= 0.8 mm on synthetic A3/Tabloid pages with 15_1's 1.6%
# keystone (<= 1.0 mm at 3 px/mm with 1 px center noise), so 2 mm keeps 2x headroom, while
# a wrong layout of the right scale (other aspect) misses by several mm.
# MAX_SHEET_SCALE_DEV = 0.02 (2%): the marker pitch reads +0.4-0.5% above the layout scale on
# both FPfit images (15_1: 115.33 vs 114.72 px/cm, 5_1: 92.13 vs 91.76) and a 1.6% keystone
# adds up to ~0.5% on a subset, so the true sheet sits at <= 1.0% (synthetic, every sheet and
# 2/3/4-marker subset, 12 px/mm). The closest wrong sheet that only scale can reject is
# Letter vs A4 on a horizontal pair: 146 vs 140 mm = 4.2%, which the same bias + keystone
# pulls down to 2.9% at worst. 2% sits in the 1.0% / 2.9% gap; wider sheet-size gaps (A3 vs
# Tabloid vertical 3.4%, A4 vs A3 ~41%, A5 vs Tabloid 2.7x) are farther out. FP_PEER_TOL
# (3%) would be too loose here. (At 3 px/mm with 1 px center noise both distributions
# overlap -- true up to 2.2%, wrong down to 0.6% -- and no threshold separates them; the
# ranking still picks the right sheet there in the stress runs.)
# Both gates must hold for a hypothesis to be admissible.
MAX_SHEET_RMS_MM = 2.0
MAX_SHEET_SCALE_DEV = 0.02
# Ranking (and the ambiguity margin) use one size-independent cost in mm:
#   scale_dev_mm = max(0, rel_dev - SCALE_DEV_FLOOR) * REF_SPAN_MM
#   cost_mm      = rms_mm + scale_dev_mm
# REF_SPAN_MM is a fixed reference length, ~ the Letter marker diagonal (249 mm), so the scale
# term weighs a Letter diagonal as it did before and larger sheets are no longer penalized for
# their size (the floor below lowers every cost by up to 1.25 mm).
# SCALE_DEV_FLOOR = 0.5% is the systematic pitch-vs-layout bias seen on both real images:
# a scale difference smaller than that is not evidence. Without the floor the bias alone
# (0.5% = 1.25 mm of cost, more than AMBIGUITY_MARGIN_MM) makes the SMALLER of two layouts
# that differ by < 1% in scale win outright: a true Legal diagonal pair came out
# "identified Legal (legacy)" (diagonals 314.9 vs 313.1 mm) with its inferred markers 6 mm
# off. With the floor both stay in the pool and the extent/symmetry tie-breaks or the
# Legal-variant rule decide. Pairs that only scale can tell apart differ by >= 3.4%
# (Letter vs A4 4.2%), which the floor barely dents.
REF_SPAN_MM = 250.0
SCALE_DEV_FLOOR = 0.005
# The largest cost an admissible hypothesis can carry (implied by the two gates; it is not a
# third gate unless identify_sheet(max_cost_mm=...) asks for one).
MAX_SHEET_COST_MM = MAX_SHEET_RMS_MM + (MAX_SHEET_SCALE_DEV - SCALE_DEV_FLOOR) * REF_SPAN_MM  # 5.75
AMBIGUITY_MARGIN_MM = 1.0     # hypotheses within best + this are tied and go to tie-breaks
MAX_FIT_ROTATION_ERR_DEG = 5.0  # fit rotation vs the markers' orientation vote
# A hypothesis counts as "FPfit-symmetric" when the four margins between its outermost square
# centers and the image edges differ by at most this. The FPfit export pads every side
# equally (paddingCm + 0.5 cm), so symmetry is evidence; the padding itself is not trusted.
FPFIT_SYMMETRY_MAX_MM = 4.0
# Legal vs Legal (legacy) on an FPfit crop: both layouts can pass the 4 mm symmetry test
# (a 6 mm x-spacing difference split over two margins), so among FPfit-symmetric layouts of
# ONE family a layout is dropped when its best margin spread is more than this above the
# most symmetric one. Synthetic Legal/Legal_legacy crops (keystone 1.6%, pitch bias 0.5%,
# 3-12 px/mm) put the true layout >= 1.1 mm below its sibling. Different families are never
# split this way: they stay "ambiguous" (spec), and without FPfit symmetry it never applies.
VARIANT_SYMMETRY_MARGIN_MM = 1.0

# ---- settings defaults --------------------------------------------------------------------
FP_PEER_TOL = 0.03            # FP markers within 3% of their cluster median agree
FP_ANCHOR_TOL = 0.03          # reconcile tolerance around the FieldPrism anchor (engine)

CATALOG_PATH = Path(__file__).with_name("fieldprism_sheets.json")
ROLES = ("TL", "TR", "C", "BL")
CORNERS = ("TL", "TR", "BL", "BR")
_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# ============================================================================================
# helpers
# ============================================================================================
def _f(x):
    """Python float (or None) -- keeps outputs picklable/JSON-safe and numpy-free."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _pt(p):
    return [float(p[0]), float(p[1])]


def _wrap180(deg):
    """Wrap an angle to (-180, 180]."""
    d = (float(deg) + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


@functools.lru_cache(maxsize=4)
def _load_catalog_cached(path_str):
    with open(path_str, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_sheet_catalog(path=None) -> dict:
    """The literal FieldPrism sheet geometry (fieldprism_sheets.json), parsed once and cached.

    Treat the returned dict as read-only: it is shared between callers.
    """
    return _load_catalog_cached(str(Path(path) if path else CATALOG_PATH))


# ============================================================================================
# 1. squares inside one marker (app ClusterProcessor)
# ============================================================================================
def _to_gray(roi):
    if roi.ndim == 2:
        return roi
    if roi.shape[2] == 4:
        return cv2.cvtColor(roi, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def _fill_small_holes(big, max_frac):
    """LM3 DEVIATION 1: fill holes of `big` that do not touch the ROI border and are small.

    Holes are 4-connected background components (the dual of the 8-connected foreground).
    Returns (mask, n_filled).
    """
    comp_area = int(np.count_nonzero(big))
    inv = (big == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = big.shape[:2]
    limit = max_frac * comp_area
    fill = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in st[i])
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue                                   # touches the ROI border: background
        if area < limit:
            fill.append(i)
    if not fill:
        return big, 0
    out = big.copy()
    out[np.isin(lab, fill)] = 255
    return out, len(fill)


def _dt_plateaus(big):
    """App plateau step: L2 distance transform (mask 5) -> normalize to 0..255 (CV_8U) ->
    keep > 0.55*max -> 3x3 open -> 8-connected components. Returns (n, stats, centroids)."""
    dt = cv2.distanceTransform(big, cv2.DIST_L2, 5)
    dtn = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    thr = DT_PEAK_FRAC * float(dtn.max())
    _, pk = cv2.threshold(dtn, thr, 255, cv2.THRESH_BINARY)   # strictly > floor(thr) for 8U
    pk = cv2.morphologyEx(pk, cv2.MORPH_OPEN, _KERNEL)
    n2, _, st2, cen2 = cv2.connectedComponentsWithStats(pk, connectivity=8)
    return n2, st2, cen2


def find_marker_squares(roi_bgr, *, fill_small_holes=True) -> dict:
    """Centers of the 4 filled squares of one marker (ROI-local px).

    Exact port of ClusterProcessor.centersFromMat (KT:1439-1501 == MM:146-261):
    gray -> Otsu -> invert -> 3x3 dilate -> largest 8-connected component -> L2 distance
    transform (mask 5) -> normalize to 0..255 (CV_8U) -> keep > 0.55*max -> 3x3 open ->
    connected components -> the 4 largest plateaus by area; their centroids are the square
    centers. The only change is the optional small-hole fill (LM3 DEVIATION 1).

    `n_peaks_app` is the plateau count the apps would see (no fill). It equals `n_peaks`
    unless holes were filled; then the unfilled mask is run through the plateau step as
    well, so a marker whose 4th square exists only after the fill can be told apart.
    """
    out = {"ok": False, "reason": None, "centers": [], "peak_areas": [], "n_peaks": 0,
           "n_peaks_app": 0, "holes_filled": 0, "component_area": 0}
    if roi_bgr is None or roi_bgr.size == 0:
        out["reason"] = "empty ROI"
        return out
    gray = _to_gray(roi_bgr)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    b = cv2.bitwise_not(b)                                     # dark squares -> 255
    d = cv2.dilate(b, _KERNEL)                                 # joins corner-touching squares
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, connectivity=8)
    if n <= 1:
        out["reason"] = "no dark foreground in the ROI"
        return out
    # largest component; first maximum wins (strict > in KT/MM)
    largest = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    big = np.where(lab == largest, 255, 0).astype(np.uint8)
    out["component_area"] = int(st[largest, cv2.CC_STAT_AREA])
    unfilled = big
    if fill_small_holes:
        big, out["holes_filled"] = _fill_small_holes(big, HOLE_FILL_MAX_FRAC)
    n2, st2, cen2 = _dt_plateaus(big)
    out["n_peaks"] = int(n2 - 1)
    out["n_peaks_app"] = (int(_dt_plateaus(unfilled)[0] - 1) if out["holes_filled"]
                          else out["n_peaks"])
    if n2 <= 1:
        out["reason"] = "no distance-transform peaks"
        return out
    # Kotlin sortedByDescending is stable (Swift's NSMutableArray sort is not; ties only).
    cands = sorted(range(1, n2), key=lambda i: -int(st2[i, cv2.CC_STAT_AREA]))
    if len(cands) < 4:
        out["reason"] = f"found {len(cands)} square candidates (need 4)"
        return out
    top = cands[:4]
    out["centers"] = [_pt(cen2[i]) for i in top]
    out["peak_areas"] = [int(st2[i, cv2.CC_STAT_AREA]) for i in top]
    out["ok"] = True
    return out


# ============================================================================================
# 2. roles and orientation (app RoleAssigner / estimateOptimalRotation)
# ============================================================================================
def assign_roles(pts) -> dict | None:
    """Label 4 square centers TL/TR/C/BL. Exact port of RoleAssigner (KT:1503-1539 == SW:729-772).

    C is the point nearest the mean (first minimum wins). Of the other three, the pair with
    the smallest cosine around C is TR-C-BL (collinear), so the remaining one is TL. TR vs BL
    by the sign of cross(TL-C, pA-C) in y-down image coordinates (> 0 -> pA is TR).
    """
    if pts is None or len(pts) != 4:
        return None
    P = [(float(p[0]), float(p[1])) for p in pts]
    mx = sum(p[0] for p in P) / 4.0
    my = sum(p[1] for p in P) / 4.0
    c_idx, min_d = 0, float("inf")
    for i, p in enumerate(P):
        dd = (p[0] - mx) ** 2 + (p[1] - my) ** 2
        if dd < min_d:
            min_d, c_idx = dd, i
    C = P[c_idx]
    others = [p for i, p in enumerate(P) if i != c_idx]
    best, min_cos = None, 1.0
    for i, j in ((0, 1), (0, 2), (1, 2)):
        v1 = (others[i][0] - C[0], others[i][1] - C[1])
        v2 = (others[j][0] - C[0], others[j][1] - C[1])
        m1, m2 = math.hypot(*v1), math.hypot(*v2)
        if m1 > 0 and m2 > 0:
            cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)
            if cos < min_cos:                                   # strict <, as in the apps
                min_cos, best = cos, (i, j)
    if best is None:
        return None
    tl_idx = next(k for k in range(3) if k not in best)
    TL = others[tl_idx]
    pA, pB = others[best[0]], others[best[1]]
    cross = (TL[0] - C[0]) * (pA[1] - C[1]) - (TL[1] - C[1]) * (pA[0] - C[0])
    TR, BL = (pA, pB) if cross > 0 else (pB, pA)
    return {"TL": _pt(TL), "TR": _pt(TR), "C": _pt(C), "BL": _pt(BL)}


def orientation_vote(roles) -> int | None:
    """App orientation vote (estimateOptimalRotation, KT:307-327 == SW:312-332).

    `roles` is one roles dict or a list of them (the app votes over all markers). Per marker,
    with (dx, dy) = TL - C: (-,-) -> 0, (-,+) -> 90, (+,+) -> 180, (+,-) -> 270; an exact
    zero casts no vote. The winner is scanned in [0, 90, 180, 270] order with a strict >, so
    ties go to the earlier angle. The value is the CLOCKWISE rotation that makes the sheet
    upright (equivalently the counterclockwise rotation the sheet shows in the image).
    Returns None when no marker voted.
    """
    if roles is None:
        return None
    items = [roles] if isinstance(roles, dict) else [r for r in roles if r]
    votes = [0, 0, 0, 0]
    for r in items:
        dx = r["TL"][0] - r["C"][0]
        dy = r["TL"][1] - r["C"][1]
        if dx < 0 and dy < 0:
            votes[0] += 1
        elif dx > 0 and dy < 0:
            votes[3] += 1
        elif dx > 0 and dy > 0:
            votes[2] += 1
        elif dx < 0 and dy > 0:
            votes[1] += 1
    if sum(votes) == 0:
        return None
    best_rot, max_votes = 0, -1
    for i, ang in enumerate((0, 90, 180, 270)):
        if votes[i] > max_votes:
            max_votes, best_rot = votes[i], ang
    return best_rot


# ============================================================================================
# 3. one marker, end to end
# ============================================================================================
def _roi_with_pad(w, h, box, pad):
    """computeRoiWithPad (KT:401-411): floor/ceil the box, pad, clamp; None if degenerate."""
    x1, y1, x2, y2 = box
    left = max(0, int(math.floor(x1)) - pad)
    top = max(0, int(math.floor(y1)) - pad)
    right = min(w, int(math.ceil(x2)) + pad)
    bottom = min(h, int(math.ceil(y2)) + pad)
    if right - left <= 1 or bottom - top <= 1:
        return None
    return left, top, right, bottom


def _validate(roles, a, b, peak_areas):
    """LM3 DEVIATION 2: per-marker geometric checks. Returns (validation, reasons)."""
    TL, TR, C, BL = (np.asarray(roles[k], float) for k in ROLES)
    pitch = (a + b) / 2.0
    vh, vv = TR - TL, BL - TL
    cosang = float(np.dot(vh, vv) / (a * b))
    angle = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
    checks = {
        "pitch_ratio": (abs(a / b - 1.0), MAX_PITCH_RATIO_ERR, "le",
                        "TL-TR and TL-BL pitches differ by {:.1%} (limit {:.0%})"),
        "right_angle_deg": (abs(angle - 90.0), MAX_RIGHT_ANGLE_ERR_DEG, "le",
                            "corner angle is off 90 deg by {:.2f} deg (limit {:.1f})"),
        "c_mid": (float(np.linalg.norm(C - (TR + BL) / 2.0)) / pitch, MAX_C_MID_ERR, "le",
                  "C is {:.1%} of the pitch from the TR-BL midpoint (limit {:.0%})"),
        "c_diag": (abs(float(np.linalg.norm(TL - C)) * math.sqrt(2.0) / pitch - 1.0),
                   MAX_C_DIAG_ERR, "le",
                   "TL-C distance is off the diagonal by {:.1%} (limit {:.0%})"),
        "peak_area_ratio": (min(peak_areas) / max(peak_areas) if max(peak_areas) > 0 else 0.0,
                            MIN_PEAK_AREA_RATIO, "ge",
                            "square plateaus are unbalanced: min/max area {:.2f} (limit {:.2f})"),
    }
    validation, reasons = {}, []
    for name, (val, lim, op, msg) in checks.items():
        ok = val <= lim if op == "le" else val >= lim
        validation[name] = {"value": _f(val), "limit": _f(lim), "ok": bool(ok)}
        if not ok:
            reasons.append(msg.format(val, lim))
    return validation, reasons


def _failed_marker(reason):
    return {"status": "failed", "status_reason": reason, "roi": None, "n_peaks": 0,
            "n_peaks_app": None, "holes_filled": 0, "peak_area_ratio": None, "peak_areas": [],
            "roles": None, "br": None, "pitch_h_px": None, "pitch_v_px": None, "pxcm": None,
            "orientation_deg": None, "valid": False, "validation": {}, "validation_reasons": []}


def measure_marker(image_bgr, box_xyxy, *, pad_px=CROP_PAD_PX, fill_small_holes=True) -> dict:
    """Find, label and validate the squares of the marker inside one detector box.

    status "measured" = 4 squares found and roles assigned; `valid` = the geometric checks
    passed as well. Coordinates are in the image (working) frame.

    When the small-hole fill changed what the app finder would see, the marker says so
    (`n_peaks_app` is the app's plateau count): a marker that has 4 squares only after the
    fill gets status_reason "app finder: N square candidates; 4th plateau only after LM3 hole
    fill" and a failed, informational `app_finder_squares` entry in `validation` (the note is
    added to validation_reasons too when the geometric checks already reject the marker; it
    does not by itself invalidate a marker whose geometry passes). A marker the fill pushes
    BELOW 4 plateaus while the app finder has 4 keeps the failure but names the app count.
    """
    m = _failed_marker(None)
    h, w = image_bgr.shape[:2]
    roi = _roi_with_pad(w, h, [float(v) for v in box_xyxy], int(pad_px))
    if roi is None:
        m["status_reason"] = "empty ROI (box outside the image)"
        return m
    x0, y0, x1, y1 = roi
    m["roi"] = [int(x0), int(y0), int(x1), int(y1)]
    sq = find_marker_squares(image_bgr[y0:y1, x0:x1], fill_small_holes=fill_small_holes)
    m["n_peaks"] = sq["n_peaks"]
    m["n_peaks_app"] = sq["n_peaks_app"]
    m["holes_filled"] = sq["holes_filled"]
    n_app = sq["n_peaks_app"]
    if not sq["ok"]:
        m["status_reason"] = sq["reason"]
        if sq["holes_filled"] and n_app >= 4 > sq["n_peaks"]:
            m["status_reason"] += (f" after LM3 hole fill ({sq['holes_filled']} holes); "
                                   f"the app finder (no fill) has {n_app}")
        return m
    m["peak_areas"] = list(sq["peak_areas"])
    pa = sq["peak_areas"]
    m["peak_area_ratio"] = _f(min(pa) / max(pa)) if max(pa) > 0 else 0.0
    roles = assign_roles([[x + x0, y + y0] for x, y in sq["centers"]])
    if roles is None:
        m["status_reason"] = "role assignment failed"
        return m
    TL, TR, BL = (np.asarray(roles[k], float) for k in ("TL", "TR", "BL"))
    a = float(np.linalg.norm(TR - TL))
    bb = float(np.linalg.norm(BL - TL))
    if a <= 1e-6 or bb <= 1e-6:
        m["status_reason"] = "degenerate square layout"
        return m
    m["status"] = "measured"
    m["roles"] = roles
    m["br"] = _pt(TR + BL - TL)                     # predicted center of the empty BR cell
    m["pitch_h_px"] = _f(a)
    m["pitch_v_px"] = _f(bb)
    m["pxcm"] = _f((a + bb) / 2.0 / (SQUARE_PITCH_MM / 10.0))   # SW:1107-1130 affine path
    m["orientation_deg"] = orientation_vote(roles)
    m["validation"], m["validation_reasons"] = _validate(roles, a, bb, pa)
    m["valid"] = not m["validation_reasons"]
    if sq["holes_filled"] and n_app < 4:
        made = "4th plateau" if n_app == 3 else f"plateaus {n_app + 1}-4"
        note = (f"app finder: {n_app} square candidate{'' if n_app == 1 else 's'}; "
                f"{made} only after LM3 hole fill ({sq['holes_filled']} holes filled)")
        m["status_reason"] = note
        m["validation"]["app_finder_squares"] = {"value": _f(n_app), "limit": 4.0, "ok": False}
        if not m["valid"]:
            m["validation_reasons"].insert(0, note)
    return m


# ============================================================================================
# 4. sheet identification
# ============================================================================================
def _umeyama2d(src, dst):
    """Least-squares similarity dst ~ s*R@src + t (no reflection). Returns (s, R, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = float((xs ** 2).sum()) / len(src)
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S)) / var_s if var_s > 0 else 0.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def project_sheet_points(fit, pts_mm):
    """Map sheet points (mm, page top-left origin, y down) into image px with a `fit` dict.

    x_img = s*(cos(th)*x - sin(th)*y) + tx,  y_img = s*(sin(th)*x + cos(th)*y) + ty, with
    th = -radians(fit["rotation_deg"]) because rotation_deg is counterclockwise AS SEEN on
    screen while pixel y points down.
    """
    th = -math.radians(fit["rotation_deg"])
    s = fit["scale_px_per_mm"]
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    P = np.asarray(pts_mm, float).reshape(-1, 2)
    return (s * P @ R.T + np.array([fit["tx"], fit["ty"]])).tolist()


def _dedupe_markers(markers):
    """Drop markers whose TL square repeats an earlier one (two boxes on one marker)."""
    kept = []
    for m in markers:
        tl = np.asarray(m["roles"]["TL"], float)
        pitch_px = (m.get("pxcm") or 0.0) * SQUARE_PITCH_MM / 10.0
        if any(np.linalg.norm(tl - np.asarray(k["roles"]["TL"], float)) < 0.25 * pitch_px
               for k in kept):
            continue
        kept.append(m)
    return kept


def _empty_sheet(status, orientation=None):
    return {"status": status, "sheet_type": None, "label": None, "corners_ambiguous": None,
            "candidates": [], "assignment": {}, "fit": None, "cf_px_per_cm_sheet_fit": None,
            "orientation_deg": orientation, "corners": {}, "page_corners_px": None,
            "fpfit_margins_mm": None, "tied_labels": []}


def identify_sheet(markers, image_wh, *, catalog=None, max_cost_mm=MAX_SHEET_COST_MM,
                   ambiguity_margin_mm=AMBIGUITY_MARGIN_MM, max_rms_mm=MAX_SHEET_RMS_MM,
                   max_scale_dev=MAX_SHEET_SCALE_DEV) -> dict:
    """Which sheet, which corner is each marker, and where are the missing markers.

    markers: [{"detection_id", "roles": {"TL","TR","C","BL"}, "pxcm"}] -- valid, agreeing
    markers only. Every catalog sheet x every injective marker->corner assignment is fitted
    with a 2-D similarity (model square centers in mm -> observed px). Per hypothesis:
      rms_mm        = rms residual / s                    (s = fitted px per mm)
      rel_dev       = |ln(s / s_pitch)|                   (s_pitch = mean(pxcm)/10)
      scale_dev_pct = 100 * rel_dev                       (raw; the admissibility gate)
      scale_dev_mm  = max(0, rel_dev - SCALE_DEV_FLOOR) * REF_SPAN_MM  (the scale
                      disagreement beyond the 0.5% systematic pitch bias, expressed as a
                      length over a FIXED 250 mm reference span -- not over the sheet's own
                      span -- so it means the same thing on every sheet size)
      cost_mm       = rms_mm + scale_dev_mm               (ranking + ambiguity margin)
    rotation_deg is the page's counterclockwise rotation as seen on screen, (-180, 180]; it
    is in the same convention as the app vote (orientation_vote), and hypotheses more than
    MAX_FIT_ROTATION_ERR_DEG from the markers' vote are discarded.

    Admissible: rms_mm <= max_rms_mm AND rel_dev <= max_scale_dev (AND cost_mm <=
    max_cost_mm, which the defaults already imply). No admissible hypothesis ->
    "unrecognized". Otherwise the tie-break pool is EVERY hypothesis, admissible or not,
    with cost <= best admissible cost + ambiguity_margin_mm, so a true sheet slightly over
    a gate still competes with a cheaper wrong one. An admissible layout of the same family
    under the same corner assignment (Legal vs Legal_legacy) joins the pool whatever its
    cost: the two differ by less than the pitch bias. The pool then keeps (1) the
    hypotheses whose reconstructed squares are all inside the image, if any, then (2) the
    FPfit-symmetric ones (margin spread <= FPFIT_SYMMETRY_MAX_MM), if any; when those are
    all one family but several layouts, a layout whose best margin spread is more than
    VARIANT_SYMMETRY_MARGIN_MM above the most symmetric one is dropped. The chosen top can
    therefore carry a cost slightly above the gates; `fit` reports what it is.

    More than one sheet FAMILY left -> "ambiguous"; one family -> "identified". A corner
    tie that cannot be broken sets corners_ambiguous. Current and legacy Legal are one
    family: their marker spacings differ by 6 mm in x (146 vs 140) and 1 mm in y (279 vs
    280), so a horizontal pair, the image extent or the margin symmetry usually separates
    them (a vertical or diagonal pair in an ordinary photo does not: the 0.4-0.6% scale gap
    is within the pitch bias). When both are still in the pool the result stays
    "identified" with the family label ("Legal") -- both candidates are Legal paper -- but
    corners_ambiguous is set (no inferred markers, no page outline: they would sit up to
    6 mm off) and cf_px_per_cm_sheet_fit is None (the two layouts differ by up to 0.4% in
    scale), so the anchor falls back to the marker mean. `tied_labels` lists the sheet
    labels left in the final pool (best first).
    """
    cat = catalog or load_sheet_catalog()
    sheets = cat["sheets"]
    W, H = (float(image_wh[0]), float(image_wh[1]))
    mk = [m for m in (markers or []) if m.get("roles") and m.get("pxcm")]
    mk = _dedupe_markers(mk)
    vote = orientation_vote([m["roles"] for m in mk]) if mk else None
    if len(mk) < 2:
        return _empty_sheet("undetermined", vote)
    if len(mk) > 4:
        out = _empty_sheet("unrecognized", vote)
        out["reason"] = f"{len(mk)} distinct markers; a sheet has 4"
        return out

    obs = np.array([[m["roles"][r] for r in ROLES] for m in mk], float)       # (n, 4, 2)
    s_pitch = float(np.mean([m["pxcm"] for m in mk])) / 10.0
    hyps = []
    for key, sh in sheets.items():
        sc = sh["square_centers_mm"]
        for perm in itertools.permutations(CORNERS, len(mk)):
            model = np.array([[sc[c][r] for r in ROLES] for c in perm], float)
            s, R, t = _umeyama2d(model.reshape(-1, 2), obs.reshape(-1, 2))
            if s <= 0:
                continue
            rot = _wrap180(-math.degrees(math.atan2(R[1, 0], R[0, 0])))
            if vote is not None and abs(_wrap180(rot - vote)) > MAX_FIT_ROTATION_ERR_DEG:
                continue
            res = np.linalg.norm((s * model.reshape(-1, 2) @ R.T + t) - obs.reshape(-1, 2), axis=1)
            rms_mm = float(np.sqrt(np.mean(res ** 2))) / s
            rel_dev = abs(math.log(s / s_pitch))
            scale_dev = max(0.0, rel_dev - SCALE_DEV_FLOOR) * REF_SPAN_MM
            cost = rms_mm + scale_dev
            # every filled square of all four markers, reconstructed through the fit
            allsq = np.array([sc[c][r] for c in CORNERS for r in ROLES], float)
            px = s * allsq @ R.T + t
            inside = bool(np.all((px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)))
            margins = {"left": float(px[:, 0].min()) / s, "right": (W - float(px[:, 0].max())) / s,
                       "top": float(px[:, 1].min()) / s, "bottom": (H - float(px[:, 1].max())) / s}
            spread = max(margins.values()) - min(margins.values())
            hyps.append({
                "sheet_type": key, "label": sh["label"], "family": sh.get("family", key),
                "assignment": {m["detection_id"]: c for m, c in zip(mk, perm)},
                "cost_mm": cost, "rms_mm": rms_mm, "scale_dev_mm": scale_dev,
                "rel_dev": rel_dev,
                "admissible": bool(rms_mm <= max_rms_mm and rel_dev <= max_scale_dev
                                   and (max_cost_mm is None or cost <= max_cost_mm)),
                "max_mm": float(res.max()) / s, "rotation_deg": rot, "inside_image": inside,
                "margin_spread_mm": spread, "_margins": margins, "_s": s, "_R": R, "_t": t,
                "_perm": perm,
            })

    hyps.sort(key=lambda h: h["cost_mm"])
    admissible = [h for h in hyps if h["admissible"]]
    pool = []
    if admissible:
        limit = admissible[0]["cost_mm"] + ambiguity_margin_mm
        pool = [h for h in hyps if h["cost_mm"] <= limit]           # admissible or not
        # Layouts of one family (Legal / Legal_legacy: <= 0.6% apart in scale, <= 1.2 deg in
        # direction) are never separated by cost -- that gap is within the pitch bias and
        # pitch noise -- so an admissible sibling layout under the same corner assignment
        # joins the pool and only extent / symmetry can split them.
        def _akey(h):
            return h["family"], tuple(sorted(h["assignment"].items(), key=lambda kv: str(kv[0])))
        in_pool = {id(h) for h in pool}
        keys = {_akey(h) for h in pool}
        pool += [h for h in admissible if id(h) not in in_pool and _akey(h) in keys]
        pool.sort(key=lambda h: h["cost_mm"])
        pool = [h for h in pool if h["inside_image"]] or pool
        sym = [h for h in pool if h["margin_spread_mm"] <= FPFIT_SYMMETRY_MAX_MM]
        if sym:
            pool = sym
            if len({h["family"] for h in pool}) == 1:
                # one family, several layouts (Legal vs Legal_legacy): keep the layouts
                # whose best margin spread is within VARIANT_SYMMETRY_MARGIN_MM of the best
                spread = {}
                for h in pool:
                    spread[h["sheet_type"]] = min(spread.get(h["sheet_type"], math.inf),
                                                  h["margin_spread_mm"])
                lo = min(spread.values())
                keep = {t for t, v in spread.items() if v <= lo + VARIANT_SYMMETRY_MARGIN_MM}
                pool = [h for h in pool if h["sheet_type"] in keep]
    pool_ids = {id(h) for h in pool}
    ranked = pool + [h for h in admissible if id(h) not in pool_ids] \
        + [h for h in hyps if not h["admissible"] and id(h) not in pool_ids]

    def public(h):
        return {"sheet_type": h["sheet_type"], "label": h["label"], "family": h["family"],
                "assignment": dict(h["assignment"]), "cost_mm": _f(h["cost_mm"]),
                "rms_mm": _f(h["rms_mm"]), "scale_dev_mm": _f(h["scale_dev_mm"]),
                "scale_dev_pct": _f(100.0 * h["rel_dev"]),
                "rotation_deg": _f(h["rotation_deg"]), "inside_image": bool(h["inside_image"]),
                "margin_spread_mm": _f(h["margin_spread_mm"]),
                "admissible": bool(h["admissible"]), "in_pool": id(h) in pool_ids}

    candidates = [public(h) for h in ranked[:10]]
    if not pool:
        out = _empty_sheet("unrecognized", vote)
        out["candidates"] = candidates
        return out

    top = pool[0]
    types = list(dict.fromkeys(h["sheet_type"] for h in pool))           # rank order
    families = {h["family"] for h in pool}
    status = "identified" if len(families) == 1 else "ambiguous"
    # Same family, more than one layout (Legal vs Legal_legacy): report the family, guess
    # nothing (inferred markers / page outline / single-layout fit CF would be up to 6 mm /
    # 0.4% off).
    variant_tie = status == "identified" and len(types) > 1
    corner_sets = {tuple(sorted(h["assignment"].items(), key=lambda kv: str(kv[0])))
                   for h in pool if h["sheet_type"] == top["sheet_type"]}
    corners_ambiguous = status == "identified" and (variant_tie or len(corner_sets) > 1)
    sh = sheets[top["sheet_type"]]
    sheet_type, label = top["sheet_type"], top["label"]
    if variant_tie:
        fam = top["family"]
        sheet_type = fam if fam in sheets else top["sheet_type"]
        label = sheets[sheet_type]["label"]
    s, R, t = top["_s"], top["_R"], top["_t"]
    fit = {"scale_px_per_mm": _f(s), "rotation_deg": _f(top["rotation_deg"]),
           "tx": _f(t[0]), "ty": _f(t[1]), "rms_mm": _f(top["rms_mm"]),
           "max_mm": _f(top["max_mm"]), "scale_dev_mm": _f(top["scale_dev_mm"]),
           "scale_dev_pct": _f(100.0 * top["rel_dev"]), "cost_mm": _f(top["cost_mm"])}

    # Corners. Observed corners carry the MEASURED square centers (BR = TR + BL - TL, as on the
    # marker rows); inferred corners are the fit's reconstruction of the model squares. They
    # are only reconstructed when the sheet and its corners are unambiguous -- a guessed
    # marker drawn on the overlay would be worse than none.
    by_corner = {c: m for m, c in zip(mk, top["_perm"])}
    corners = {}
    for c in CORNERS:
        m = by_corner.get(c)
        if m is not None:
            sq = {r: _pt(m["roles"][r]) for r in ROLES}
            sq["BR"] = _pt(np.asarray(sq["TR"]) + np.asarray(sq["BL"]) - np.asarray(sq["TL"]))
            corners[c] = {"observed": True, "detection_id": m["detection_id"], "squares": sq}
        elif status == "identified" and not corners_ambiguous:
            model = np.array([sh["square_centers_mm"][c][r] for r in ROLES + ("BR",)], float)
            px = s * model @ R.T + t
            corners[c] = {"observed": False, "detection_id": None,
                          "squares": {r: _pt(p) for r, p in zip(ROLES + ("BR",), px)}}
    page_corners = None
    if status == "identified" and not corners_ambiguous:
        pw, ph = sh["page_mm"]
        pc = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], float)
        page_corners = [_pt(p) for p in s * pc @ R.T + t]

    return {
        "status": status,
        "sheet_type": sheet_type,
        "label": label,
        "corners_ambiguous": bool(corners_ambiguous),
        "candidates": candidates,
        "assignment": dict(top["assignment"]),
        "fit": fit,
        "cf_px_per_cm_sheet_fit": (_f(s * 10.0) if status == "identified" and not variant_tie
                                   else None),
        "orientation_deg": int(round(top["rotation_deg"] / 90.0) * 90) % 360,
        "corners": corners,
        "page_corners_px": page_corners,
        "fpfit_margins_mm": {k: _f(v) for k, v in top["_margins"].items()},
        "tied_labels": [sheets[k]["label"] for k in types],
    }


# ============================================================================================
# 5. one image: markers -> peers -> sheet -> anchor CF
# ============================================================================================
def _peer_cluster(vals, tol):
    """Largest subset whose members all lie within `tol` of the subset's median.

    vals: [(key, pxcm)]. Ties in size go to the tighter subset. Exhaustive up to 12 markers
    (a sheet has 4), a center-scan beyond that.
    """
    n = len(vals)
    if n < 2:
        return []

    def ok(sub):
        med = float(np.median([v for _, v in sub]))
        return med > 0 and all(abs(v / med - 1.0) <= tol for _, v in sub)

    def spread(sub):
        xs = [v for _, v in sub]
        return (max(xs) - min(xs)) / float(np.mean(xs))

    if n <= 12:
        for k in range(n, 1, -1):
            good = [list(c) for c in itertools.combinations(vals, k) if ok(c)]
            if good:
                return min(good, key=spread)
        return []
    best = []
    for _, c in vals:
        sub = [kv for kv in vals if abs(kv[1] / c - 1.0) <= tol]
        if len(sub) >= 2 and ok(sub) and (len(sub) > len(best)
                                          or (len(sub) == len(best) and spread(sub) < spread(best))):
            best = sub
    return best


def analyze_fieldprism(image_bgr, fp_crops, *, image_wh=None, peer_tol=FP_PEER_TOL,
                       allow_single_marker=True, catalog=None) -> dict:
    """Measure every FP crop of one image and derive the FieldPrism anchor CF.

    1) measure_marker per crop; 2) valid markers -> peer cluster (largest set within
    peer_tol of its median, >= 2 agree); the others are rejected; 3) identify_sheet on the
    cluster; 4) anchor = the sheet-fit CF when the sheet is identified (and not split
    between the two Legal layouts), else the cluster's mean marker CF; 5) confidence:
    "high" when >= 2 markers agree, or exactly one valid marker and allow_single_marker;
    "medium" for one valid marker otherwise; "low" when >= 2 are valid but none agree (all
    of them are then used, and the anchor is their mean unless the sheet still fits); None
    without a valid marker. "high" drops to "medium" when an identified sheet's fit CF
    differs from the marker mean by > peer_tol. Inferred markers and `sheet_corner` are only
    filled when the sheet AND its corners are unambiguous.
    """
    if image_wh is None:
        image_wh = (int(image_bgr.shape[1]), int(image_bgr.shape[0]))
    crops = list(fp_crops or [])
    # The apps keep the 10 largest boxes (KT:72-73, SW:71); the rest are skipped, not measured.
    order = sorted(range(len(crops)), key=lambda i: -((crops[i]["x2"] - crops[i]["x1"])
                                                      * (crops[i]["y2"] - crops[i]["y1"])))
    keep = set(order[:MAX_DETECTIONS])
    markers = []
    for i, c in enumerate(crops):
        box = (c["x1"], c["y1"], c["x2"], c["y2"])
        m = (measure_marker(image_bgr, box) if i in keep else
             _failed_marker(f"beyond the {MAX_DETECTIONS} largest FieldPrism boxes"))
        m.update({"detection_id": c["detection_id"], "det_conf": _f(c.get("det_conf")),
                  "x1": _f(c["x1"]), "y1": _f(c["y1"]), "x2": _f(c["x2"]), "y2": _f(c["y2"]),
                  "verdict": None, "verdict_note": None, "sheet_corner": None, "pct_vs_fp": None})
        markers.append(m)

    # Failed / invalid -> skipped. Two boxes on one marker -> the lower-confidence one skipped.
    valid = []
    for m in sorted(markers, key=lambda m: -(m["det_conf"] or 0.0)):
        if m["status"] != "measured":
            m["verdict"], m["verdict_note"] = "skipped", m["status_reason"]
            continue
        if not m["valid"]:
            m["verdict"] = "skipped"
            m["verdict_note"] = "failed validation: " + "; ".join(m["validation_reasons"])
            continue
        if _dedupe_markers(valid + [m])[-1] is not m:
            m["verdict"], m["verdict_note"] = "skipped", "duplicate detection of the same marker"
            continue
        valid.append(m)
    pos = {id(m): i for i, m in enumerate(markers)}
    valid.sort(key=lambda m: pos[id(m)])

    reasons = []
    cluster = _peer_cluster([(id(m), m["pxcm"]) for m in valid], peer_tol)
    cl_ids = {k for k, _ in cluster}
    if len(cluster) >= 2:
        used = [m for m in valid if id(m) in cl_ids]
        med = float(np.median([m["pxcm"] for m in used]))
        for m in valid:
            if id(m) in cl_ids:
                m["verdict"] = "used"
            else:
                m["verdict"] = "rejected"
                m["verdict_note"] = (f"disagrees with the other FieldPrism markers: "
                                     f"{m['pxcm']:.2f} px/cm is {100 * (m['pxcm'] / med - 1):+.1f}% "
                                     f"from their median {med:.2f}")
    else:
        used = list(valid)
        for m in used:
            m["verdict"] = "used"
        if len(used) >= 2:
            sp = (max(m["pxcm"] for m in used) - min(m["pxcm"] for m in used)) \
                / float(np.mean([m["pxcm"] for m in used]))
            note = f"FieldPrism markers disagree (spread {100 * sp:.1f}%)"
            for m in used:
                m["verdict_note"] = note
            reasons.append(note + "; none agree within " + f"{100 * peer_tol:.0f}%")

    sheet = identify_sheet([{"detection_id": m["detection_id"], "roles": m["roles"],
                             "pxcm": m["pxcm"]} for m in used], image_wh, catalog=catalog)
    pxs = [m["pxcm"] for m in used]
    marker_mean = float(np.mean(pxs)) if pxs else None
    if sheet["status"] == "identified" and sheet["cf_px_per_cm_sheet_fit"]:
        anchor, detail = sheet["cf_px_per_cm_sheet_fit"], "sheet_fit"
    elif marker_mean is not None:
        anchor, detail = marker_mean, "marker_mean"
    else:
        anchor, detail = None, None

    n_valid = len(valid)
    if n_valid == 0:
        conf = None
        reasons.append("no FieldPrism marker passed validation" if markers
                       else "no FieldPrism marker detected")
    elif len(cluster) >= 2:
        conf = "high"
        reasons.append(f"{len(cluster)} FieldPrism markers agree within {100 * peer_tol:.0f}%")
    elif n_valid == 1:
        conf = "high" if allow_single_marker else "medium"
        reasons.append("single valid FieldPrism marker" +
                       ("" if allow_single_marker else " (single-marker CF not allowed)"))
    else:
        conf = "low"
    if (conf == "high" and sheet["status"] == "identified" and marker_mean
            and sheet["cf_px_per_cm_sheet_fit"]):
        dev = sheet["cf_px_per_cm_sheet_fit"] / marker_mean - 1.0
        if abs(dev) > peer_tol:
            conf = "medium"
            reasons.append(f"sheet-fit CF differs from the marker mean by {100 * dev:+.1f}% "
                           f"(limit {100 * peer_tol:.0f}%)")
    tied = list(sheet.get("tied_labels") or [])
    if sheet["status"] == "ambiguous":
        alts = [lb for lb in tied if lb != sheet["label"]]
        reasons.append(f"sheet type ambiguous: {sheet['label']} or {', '.join(alts)}")
    elif sheet["status"] == "identified" and len(tied) > 1:
        reasons.append(f"{sheet['label']} sheet, but its layouts ({' / '.join(tied)}) fit "
                       f"equally: no markers inferred, anchor = marker mean")
    elif sheet["status"] == "unrecognized":
        reasons.append("marker layout matches no known FieldPrism sheet")

    assign = sheet["assignment"] if (sheet["status"] == "identified"
                                     and not sheet["corners_ambiguous"]) else {}
    for m in markers:
        m["sheet_corner"] = assign.get(m["detection_id"])
        if anchor and m["pxcm"]:
            m["pct_vs_fp"] = _f(100.0 * (m["pxcm"] / anchor - 1.0))

    spread_pct = None
    if len(pxs) >= 2:
        spread_pct = _f(100.0 * (max(pxs) - min(pxs)) / float(np.mean(pxs)))
    n_inferred = sum(1 for c in sheet["corners"].values() if not c["observed"])
    sheet = dict(sheet)
    sheet.update({
        "n_fp_detected": len(markers),
        "n_fp_measured": sum(1 for m in markers if m["status"] == "measured"),
        "n_fp_valid": n_valid,
        "n_fp_used": len(used),
        "n_fp_rejected": sum(1 for m in markers if m["verdict"] == "rejected"),
        "n_fp_inferred": n_inferred,
        "cf_px_per_cm_marker_mean": _f(marker_mean),
        "cf_px_per_cm_fp": _f(anchor),
        "cf_source_detail": detail,
        "fp_peer_spread_pct": spread_pct,
        "fp_confidence": conf,
        "fp_reasons": reasons,
    })
    return {"markers": markers, "sheet": sheet, "anchor_cf": _f(anchor), "confidence": conf}


# ============================================================================================
# 6. DB rows (ruler_FP_marker / ruler_FP_sheet, spec §4)
# ============================================================================================
def _json(v):
    return None if v is None else json.dumps(v)


def marker_row(m, specimen_id, crop_index, work_scale) -> dict:
    """One analyze_fieldprism marker -> a ruler_FP_marker row (column names exactly)."""
    roles = m.get("roles") or {}
    br = m.get("br")
    ws = float(work_scale or 1.0)
    pxcm = m.get("pxcm")
    row = {
        "specimen_id": int(specimen_id),
        "detection_id": int(m["detection_id"]),
        "crop_index": None if crop_index is None else int(crop_index),
        "det_conf": _f(m.get("det_conf")),
        "x1": _f(m.get("x1")), "y1": _f(m.get("y1")), "x2": _f(m.get("x2")), "y2": _f(m.get("y2")),
        "roi_x0": None, "roi_y0": None, "roi_x1": None, "roi_y1": None,
        "status": m["status"],
        "status_reason": m.get("status_reason"),
        "valid": int(bool(m.get("valid"))) if m["status"] == "measured" else None,
        "validation_json": _json(m.get("validation") or None),
        "verdict": m.get("verdict"),
        "verdict_note": m.get("verdict_note"),
        "n_peaks": m.get("n_peaks"),
        "holes_filled": m.get("holes_filled"),
        "peak_area_ratio": _f(m.get("peak_area_ratio")),
        "br_x": _f(br[0]) if br else None, "br_y": _f(br[1]) if br else None,
        "pitch_h_px": _f(m.get("pitch_h_px")), "pitch_v_px": _f(m.get("pitch_v_px")),
        "pxcm": _f(pxcm),
        "pxcm_original": _f(pxcm / ws) if pxcm else None,
        "pct_vs_fp": _f(m.get("pct_vs_fp")),
        "orientation_deg": m.get("orientation_deg"),
        "sheet_corner": m.get("sheet_corner"),
    }
    if m.get("roi"):
        row["roi_x0"], row["roi_y0"], row["roi_x1"], row["roi_y1"] = (int(v) for v in m["roi"])
    for r in ROLES:
        p = roles.get(r)
        row[f"{r.lower()}_x"] = _f(p[0]) if p else None
        row[f"{r.lower()}_y"] = _f(p[1]) if p else None
    return row


def sheet_row(analysis, specimen_id, catalog_version) -> dict:
    """analyze_fieldprism(...) -> the ruler_FP_sheet row (JSON columns as strings)."""
    s = analysis["sheet"]
    fit = s.get("fit") or {}
    ca = s.get("corners_ambiguous")
    return {
        "specimen_id": int(specimen_id),
        "catalog_version": catalog_version,
        "n_fp_detected": s.get("n_fp_detected"), "n_fp_measured": s.get("n_fp_measured"),
        "n_fp_valid": s.get("n_fp_valid"), "n_fp_used": s.get("n_fp_used"),
        "n_fp_rejected": s.get("n_fp_rejected"), "n_fp_inferred": s.get("n_fp_inferred"),
        "sheet_status": s["status"],
        "sheet_type": s.get("sheet_type"),
        "sheet_label": s.get("label"),
        "corners_ambiguous": None if ca is None else int(bool(ca)),
        "sheet_candidates_json": _json(s.get("candidates") or []),
        "orientation_deg": s.get("orientation_deg"),
        "fit_rotation_deg": _f(fit.get("rotation_deg")),
        "fit_scale_px_per_mm": _f(fit.get("scale_px_per_mm")),
        "fit_tx": _f(fit.get("tx")), "fit_ty": _f(fit.get("ty")),
        "fit_rms_mm": _f(fit.get("rms_mm")), "fit_max_mm": _f(fit.get("max_mm")),
        "fit_scale_dev_mm": _f(fit.get("scale_dev_mm")), "fit_cost_mm": _f(fit.get("cost_mm")),
        "cf_px_per_cm_sheet_fit": _f(s.get("cf_px_per_cm_sheet_fit")),
        "cf_px_per_cm_marker_mean": _f(s.get("cf_px_per_cm_marker_mean")),
        "cf_px_per_cm_fp": _f(s.get("cf_px_per_cm_fp")),
        "cf_source_detail": s.get("cf_source_detail"),
        "fp_peer_spread_pct": _f(s.get("fp_peer_spread_pct")),
        "fp_confidence": s.get("fp_confidence"),
        "fp_reasons_json": _json(s.get("fp_reasons") or []),
        "corners_json": _json(s.get("corners") or {}),
        "page_corners_json": _json(s.get("page_corners_px")),
        "fpfit_margins_json": _json(s.get("fpfit_margins_mm")),
    }
