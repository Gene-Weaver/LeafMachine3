"""FieldPrism core: square finder, roles, validation, sheet identification, peer/confidence rules.

Every image here is SYNTHETIC: the literal sheet geometry from fieldprism_sheets.json is drawn
at several px/mm (squares, the 10 cm scale bars and the caption that falls inside the left
markers' empty cells), optionally with noise, blur, specks, holes and thin occluding lines,
and rotated with np.rot90. The live check on the two real FPfit images is
tests/live_fieldprism_check.py (run by hand, not collected).
"""
from __future__ import annotations

import itertools
import json
import pickle

import cv2
import numpy as np
import pytest

from leafmachine3.inference.ruler_lattice import fieldprism as fp

CAT = fp.load_sheet_catalog()
SHEETS = list(CAT["sheets"])
ROLES = fp.ROLES
CORNERS = fp.CORNERS
BIG_PHOTO = (40.0, 60.0, 150.0, 260.0)   # asymmetric margins (mm) around an FPfit crop
PAIRS = {  # 2-marker subsets by kind
    "horizontal": [("TL", "TR"), ("BL", "BR")],
    "vertical": [("TL", "BL"), ("TR", "BR")],
    "diagonal": [("TL", "BR"), ("TR", "BL")],
}


# ============================================================================================
# synthetic rendering
# ============================================================================================
def _rot_point(x, y, w, h, k):
    """Where np.rot90(img, k) (counterclockwise k*90) puts the pixel (x, y)."""
    for _ in range(k % 4):
        x, y, w, h = y, w - 1 - x, h, w
    return x, y


def _frame(sheet, pad_mm=15.0):
    """FPfit crop of a sheet: outermost square centers pad_mm from the image edges."""
    sc = CAT["sheets"][sheet]["square_centers_mm"]
    pts = np.array([sc[c][r] for c in CORNERS for r in ROLES], float)
    return pts.min(0) - pad_mm, pts.max(0) + pad_mm


class Page:
    """A rendered sheet (or part of one) plus its ground truth, in image px."""

    def __init__(self, sheet, ppmm, corners=CORNERS, *, pad_mm=15.0,
                 marker_scale=None, extras=True, paper=232, ink=25):
        self.sheet, self.ppmm = sheet, ppmm
        sh = CAT["sheets"][sheet]
        self.lo, hi = _frame(sheet, pad_mm)
        self.W = int(round((hi[0] - self.lo[0]) * ppmm))
        self.H = int(round((hi[1] - self.lo[1]) * ppmm))
        img = np.full((self.H, self.W), paper, np.uint8)
        self.truth = {}          # corner -> role -> [x, y] px (BR = empty cell)
        self.boxes = {}          # corner -> [x1, y1, x2, y2]
        marker_scale = marker_scale or {}
        for c in CORNERS:
            k = marker_scale.get(c, 1.0)
            sq = sh["square_centers_mm"][c]
            cen = np.array(sq["C"], float)
            self.truth[c] = {r: self.px(cen + (np.array(sq[r]) - cen) * k) for r in ROLES + ("BR",)}
            mx, my = sh["marker_corner_mm"][c]
            ext = 15 * k
            self.boxes[c] = [self.px((cen[0] - ext, 0))[0], self.px((0, cen[1] - ext))[1],
                             self.px((cen[0] + ext, 0))[0], self.px((0, cen[1] + ext))[1]]
            if c not in corners:
                continue
            for r in ROLES:
                p = cen + (np.array(sq[r]) - cen) * k
                self._rect(img, p[0] - 5 * k, p[1] - 5 * k, p[0] + 5 * k, p[1] + 5 * k, ink)
        if extras:
            for pos, bar in sh["scale_bar_mm"].items():
                self._rect(img, bar["x0"] - 0.5, bar["y"] - 0.5, bar["x1"] + 0.5, bar["y"] + 0.5, ink)
                for t in sh["text_mm"][pos][:1]:          # caption ink inside the left markers
                    org = self.px((t["x"], t["baseline_y"]))
                    cv2.putText(img, "10cm - " + sheet.split("_")[0].lower(),
                                (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_DUPLEX,
                                5.6 * ppmm / 22.0, ink, max(1, int(ppmm * 0.5)), cv2.LINE_AA)
        self.gray = img

    def px(self, p_mm):
        return [(p_mm[0] - self.lo[0]) * self.ppmm - 0.5, (p_mm[1] - self.lo[1]) * self.ppmm - 0.5]

    def _rect(self, img, x0, y0, x1, y1, val):
        a, b = self.px((x0, y0)), self.px((x1, y1))
        poly = np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]]) + 0.5
        cv2.fillPoly(img, [np.round(poly * 256).astype(np.int32)], int(val), cv2.LINE_AA, shift=8)

    def bgr(self, *, noise=0.0, blur=0.0, seed=0):
        img = self.gray.astype(np.float32)
        if blur:
            img = cv2.GaussianBlur(img, (0, 0), blur)
        if noise:
            img += np.random.default_rng(seed).normal(0, noise, img.shape)
        return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    def crops(self, corners=CORNERS, *, jitter_px=0.0, seed=0, k=0):
        rng = np.random.default_rng(seed)
        out = []
        for i, c in enumerate(corners):
            x1, y1, x2, y2 = (v + rng.uniform(-jitter_px, jitter_px) for v in self.boxes[c])
            pts = [_rot_point(x, y, self.W, self.H, k) for x, y in ((x1, y1), (x2, y2))]
            out.append({"detection_id": 100 + CORNERS.index(c), "det_conf": 0.9 - 0.01 * i,
                        "x1": min(p[0] for p in pts), "y1": min(p[1] for p in pts),
                        "x2": max(p[0] for p in pts), "y2": max(p[1] for p in pts)})
        return out

    def truth_rot(self, corner, role, k):
        x, y = self.truth[corner][role]
        return _rot_point(x, y, self.W, self.H, k)

    def wh(self, k=0):
        return (self.W, self.H) if k % 2 == 0 else (self.H, self.W)


def _rot(img, k):
    return np.ascontiguousarray(np.rot90(img, k))


def _geom_markers(sheet, corners, ppmm=10.0, k=0, *, noise_px=0.0, seed=0, extra_mm=None):
    """identify_sheet input straight from the catalog geometry (no rendering).

    extra_mm = (left, top, right, bottom) margin added to the FPfit crop: an ordinary,
    asymmetric photo that carries no image-extent evidence."""
    page_lo, page_hi = _frame(sheet, 15.0)
    if extra_mm is not None:
        page_lo = page_lo - np.array(extra_mm[:2], float)
        page_hi = page_hi + np.array(extra_mm[2:], float)
    W = int(round((page_hi[0] - page_lo[0]) * ppmm))
    H = int(round((page_hi[1] - page_lo[1]) * ppmm))
    rng = np.random.default_rng(seed)
    sc = CAT["sheets"][sheet]["square_centers_mm"]
    out, truth = [], {}
    for c in CORNERS:
        truth[c] = {}
        for r in ROLES + ("BR",):
            x = (sc[c][r][0] - page_lo[0]) * ppmm
            y = (sc[c][r][1] - page_lo[1]) * ppmm
            truth[c][r] = _rot_point(x, y, W, H, k)
    for c in corners:
        roles = {r: [truth[c][r][0] + rng.normal(0, noise_px), truth[c][r][1] + rng.normal(0, noise_px)]
                 for r in ROLES}
        out.append({"detection_id": 100 + CORNERS.index(c), "roles": roles, "pxcm": 10 * ppmm})
    wh = (W, H) if k % 2 == 0 else (H, W)
    return out, wh, truth


# ============================================================================================
# catalog
# ============================================================================================
def test_catalog_literal_geometry():
    assert CAT["catalog_version"]
    assert set(SHEETS) == {"A5", "A4", "A3", "Letter", "Legal", "Tabloid", "Legal_legacy"}
    assert "Custom" not in SHEETS
    assert [k for k, s in CAT["sheets"].items() if s["legacy"]] == ["Legal_legacy"]
    expect = {"A5": (78, 133), "A4": (140, 220), "A3": (227, 343), "Letter": (146, 202),
              "Legal": (146, 279), "Tabloid": (209, 355), "Legal_legacy": (140, 280)}
    for key, sh in CAT["sheets"].items():
        assert (sh["delta_x_mm"], sh["delta_y_mm"]) == expect[key]
        if sh["layout_mm"]:
            W, H = sh["layout_mm"]
            assert (sh["delta_x_mm"], sh["delta_y_mm"]) == (W - 70, H - 77)
            assert sh["marker_corner_mm"] == {"TL": [20, 23], "TR": [W - 50, 23],
                                              "BL": [20, H - 54], "BR": [W - 50, H - 54]}
        for c, (mx, my) in sh["marker_corner_mm"].items():
            sq = sh["square_centers_mm"][c]
            assert sq == {"TL": [mx + 5, my + 5], "TR": [mx + 25, my + 5], "C": [mx + 15, my + 15],
                          "BL": [mx + 5, my + 25], "BR": [mx + 25, my + 25]}
        assert sh["pdf_check"]["n_squares"] == 16 and sh["pdf_check"]["max_center_err_mm"] < 0.01


def test_catalog_is_cached():
    assert fp.load_sheet_catalog() is fp.load_sheet_catalog()


# ============================================================================================
# square finder + roles + orientation on rendered markers
# ============================================================================================
@pytest.mark.parametrize("ppmm", [4.0, 6.5, 9.2, 11.5])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_measure_marker_rendered_rotated(ppmm, k):
    page = Page("Letter", ppmm)
    img = _rot(page.bgr(noise=6, blur=0.6 * ppmm / 4, seed=int(ppmm * 10) + k), k)
    for crop, c in zip(page.crops(jitter_px=1.5 * ppmm, seed=k, k=k), CORNERS):
        m = fp.measure_marker(img, (crop["x1"], crop["y1"], crop["x2"], crop["y2"]))
        assert m["status"] == "measured" and m["valid"], (c, m["validation_reasons"])
        for r in ROLES:          # roles survive rotation; centers within ~0.1 mm
            assert np.hypot(*np.subtract(m["roles"][r], page.truth_rot(c, r, k))) < max(0.8, 0.1 * ppmm)
        assert np.hypot(*np.subtract(m["br"], page.truth_rot(c, "BR", k))) < max(1.2, 0.12 * ppmm)
        assert m["pxcm"] == pytest.approx(10 * ppmm, rel=0.015 if ppmm < 5 else 0.008)
        assert m["orientation_deg"] == 90 * k
        assert fp.orientation_vote(m["roles"]) == 90 * k


def test_find_marker_squares_gray_and_bgra_inputs():
    page = Page("A5", 8.0)
    x1, y1, x2, y2 = (int(v) for v in page.boxes["TL"])
    roi = page.gray[y1 - 20:y2 + 20, x1 - 20:x2 + 20]
    a = fp.find_marker_squares(roi)
    b = fp.find_marker_squares(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGRA))
    assert a["ok"] and b["ok"] and a["centers"] == b["centers"] and a["n_peaks"] == 4


def test_assign_roles_exact_port_properties():
    pts = {"TL": [0.0, 0.0], "TR": [20.0, 0.0], "C": [10.0, 10.0], "BL": [0.0, 20.0]}
    for perm in itertools.permutations(ROLES):
        assert fp.assign_roles([pts[r] for r in perm]) == pts
    assert fp.assign_roles([[0, 0]] * 3) is None
    assert fp.assign_roles([[5, 5]] * 4) is None          # all magnitudes 0 -> no pair


def test_orientation_vote_rules():
    up = {"TL": [0, 0], "C": [10, 10], "TR": [20, 0], "BL": [0, 20]}
    assert fp.orientation_vote(up) == 0
    assert fp.orientation_vote({"TL": [0, 20], "C": [10, 10]}) == 90
    assert fp.orientation_vote({"TL": [20, 20], "C": [10, 10]}) == 180
    assert fp.orientation_vote({"TL": [20, 0], "C": [10, 10]}) == 270
    assert fp.orientation_vote({"TL": [10, 0], "C": [10, 10]}) is None   # exact zero: no vote
    # list vote, tie -> the earlier angle in [0, 90, 180, 270]
    assert fp.orientation_vote([{"TL": [20, 20], "C": [10, 10]}, {"TL": [0, 20], "C": [10, 10]}]) == 90
    assert fp.orientation_vote([up, up, {"TL": [20, 20], "C": [10, 10]}]) == 0


def test_robust_to_caption_noise_blur_and_thin_line():
    page = Page("Letter", 7.0)
    img = page.bgr(noise=10, blur=1.4, seed=3)
    for c in CORNERS:                          # a thin twig-like line across every marker
        x1, y1, x2, y2 = (int(v) for v in page.boxes[c])
        cv2.line(img, (x1 - 10, y1 + 30), (x2 + 10, y2 - 60), (110, 110, 110), 2, cv2.LINE_AA)
    for crop, c in zip(page.crops(), CORNERS):
        m = fp.measure_marker(img, (crop["x1"], crop["y1"], crop["x2"], crop["y2"]))
        assert m["valid"], (c, m["validation_reasons"])
        assert m["pxcm"] == pytest.approx(70.0, rel=0.01)


# ============================================================================================
# LM3 hardening: hole fill + validation
# ============================================================================================
def _marker_img(ppmm=10.0):
    page = Page("A4", ppmm, corners=("TL",), extras=False)
    return page, page.bgr(noise=3, seed=1)


def test_small_hole_fill_recovers_center_of_specked_square():
    page, img = _marker_img()
    cx, cy = page.truth["TL"]["C"]
    cv2.circle(img, (int(cx + 18), int(cy - 15)), 8, (232, 232, 232), -1)    # white speck in C
    box = page.boxes["TL"]
    raw = fp.measure_marker(img, box, fill_small_holes=False)
    fixed = fp.measure_marker(img, box)
    assert fixed["holes_filled"] >= 1 and raw["holes_filled"] == 0
    err_raw = np.hypot(*np.subtract(raw["roles"]["C"], (cx, cy)))
    err_fix = np.hypot(*np.subtract(fixed["roles"]["C"], (cx, cy)))
    assert err_raw > 4.0 and err_fix < 1.0
    assert fixed["valid"] and fixed["peak_area_ratio"] > 0.9


def test_large_enclosed_region_is_not_filled(monkeypatch):
    page, img = _marker_img()
    (x, y), (_, y2) = page.truth["TL"]["TL"], page.truth["TL"]["BL"]
    # a thin twig loop from the TL square to the BL square encloses ~17% of the component
    cv2.rectangle(img, (int(x - 40), int(y + 30)), (int(x + 30), int(y2 - 30)), (40, 40, 40), 3)
    assert fp.find_marker_squares(img)["holes_filled"] == 0
    monkeypatch.setattr(fp, "HOLE_FILL_MAX_FRAC", 0.5)          # proves the hole is there
    assert fp.find_marker_squares(img)["holes_filled"] >= 1


def _riddle(img, center_px, half_px, step=12, dot=3):
    """Pepper a square with tiny white dots: its distance-transform peak collapses (no plateau
    for the app finder), while LM3's small-hole fill restores it."""
    cx, cy = center_px
    for x in range(int(cx - half_px) + 6, int(cx + half_px) - 4, step):
        for y in range(int(cy - half_px) + 6, int(cy + half_px) - 4, step):
            img[y:y + dot, x:x + dot] = 232
    return img


def test_app_finder_count_when_fill_makes_the_fourth_square():
    # The apps (no fill) see 3 plateaus and drop the marker; LM3's fill gives 4. The marker
    # still measures (geometry is fine) but says what the app would have done.
    img, box = _render_marker_custom(GOOD)
    _riddle(img, (400, 200), 50)                                   # the TR square
    app = fp.find_marker_squares(img, fill_small_holes=False)
    lm3 = fp.find_marker_squares(img)
    assert app["n_peaks"] == 3 and not app["ok"] and app["n_peaks_app"] == 3
    assert lm3["ok"] and lm3["n_peaks"] == 4 and lm3["n_peaks_app"] == 3 and lm3["holes_filled"]
    m = fp.measure_marker(img, box, pad_px=0)
    assert m["status"] == "measured" and m["valid"] and m["n_peaks_app"] == 3
    assert m["status_reason"].startswith("app finder: 3 square candidates; 4th plateau only "
                                         "after LM3 hole fill")
    assert m["validation"]["app_finder_squares"] == {"value": 3.0, "limit": 4.0, "ok": False}
    assert m["validation_reasons"] == []                  # informational for a valid marker
    row = fp.marker_row(dict(m, detection_id=7), 1, 0, 1.0)
    assert row["status_reason"] == m["status_reason"] and row["n_peaks"] == 4
    assert json.loads(row["validation_json"])["app_finder_squares"]["value"] == 3.0
    # no fill involved -> no note, n_peaks_app == n_peaks
    clean = fp.measure_marker(*_render_marker_custom(GOOD), pad_px=0)
    assert clean["status_reason"] is None and clean["n_peaks_app"] == clean["n_peaks"] == 4
    assert "app_finder_squares" not in clean["validation"]


def test_app_finder_note_leads_the_reasons_of_an_invalid_fill_made_marker():
    # Like 5_1's twig-crossed TL marker: the fill-made 4-square layout is also geometrically
    # bad, so the verdict should lead with the app's reason.
    img, box = _render_marker_custom(GOOD, warp=[[1.08, 0], [0, 1]])   # stretched 8% in x
    _riddle(img, (408, 200), 50)                                       # the (moved) TR square
    m = fp.measure_marker(img, box, pad_px=0)
    assert m["status"] == "measured" and not m["valid"]
    assert m["validation_reasons"][0].startswith("app finder: 3 square candidates")
    assert len(m["validation_reasons"]) >= 2
    res = fp.analyze_fieldprism(img, [{"detection_id": 1, "x1": box[0], "y1": box[1],
                                       "x2": box[2], "y2": box[3], "det_conf": 0.9}])
    assert res["markers"][0]["verdict_note"].startswith("failed validation: app finder: 3")


def _render_marker_custom(squares_mm, ppmm=10.0, size_mm=60, warp=None):
    """One marker with arbitrary square centers/sides (mm) on a blank card, optionally warped
    by a 2x2 linear map about the card center (squares must still touch, as the app needs)."""
    n = int(size_mm * ppmm)
    img = np.full((n, n), 232, np.uint8)
    for (x, y, s) in squares_mm:
        poly = np.array([[x - s / 2, y - s / 2], [x + s / 2, y - s / 2], [x + s / 2, y + s / 2],
                         [x - s / 2, y + s / 2]]) * ppmm
        if warp is not None:
            c = n / 2.0
            poly = (poly - c) @ np.asarray(warp, float).T + c
        cv2.fillPoly(img, [np.round(poly * 256).astype(np.int32)], 25, cv2.LINE_AA, shift=8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [0, 0, n - 1, n - 1]


GOOD = [(20, 20, 10), (40, 20, 10), (30, 30, 10), (20, 40, 10)]
BIG = [(20, 20, 12), (40, 20, 12), (30, 30, 12), (20, 40, 12)]     # overlapping 12 mm squares


@pytest.mark.parametrize("squares,warp,check", [
    (GOOD, [[1.08, 0], [0, 1]], "pitch_ratio"),                     # stretched 8% in x
    (GOOD, [[1, 0.08], [0, 1]], "right_angle_deg"),                 # sheared 4.6 deg
    (BIG[:2] + [(30, 31.2, 12)] + BIG[3:], None, "c_mid"),          # C 1.2 mm off the midpoint
    (BIG[:2] + [(31.5, 31.5, 12)] + BIG[3:], None, "c_diag"),       # C 1.5 mm along the diagonal
    (BIG[:3] + [(22, 38, 8)], None, "peak_area_ratio"),             # an undersized BL square
])
def test_validation_rejects_bad_geometry(squares, warp, check):
    img, box = _render_marker_custom(squares, warp=warp)
    m = fp.measure_marker(img, box, pad_px=0)
    assert m["status"] == "measured", m["status_reason"]
    assert not m["valid"] and not m["validation"][check]["ok"]
    assert m["validation_reasons"]


@pytest.mark.parametrize("squares", [GOOD, BIG])
def test_validation_accepts_good_marker(squares):
    m = fp.measure_marker(*_render_marker_custom(squares), pad_px=0)
    assert m["valid"] and all(v["ok"] for v in m["validation"].values())
    assert m["pxcm"] == pytest.approx(100.0, rel=0.005)


def test_failed_marker_reports_reason():
    img, box = _render_marker_custom([(20, 20, 10), (40, 20, 10), (20, 40, 10)])   # no C (3 squares)
    m = fp.measure_marker(img, box, pad_px=0)
    assert m["status"] == "failed" and "need 4" in m["status_reason"] and not m["valid"]
    blank = fp.measure_marker(np.full((100, 100, 3), 255, np.uint8), (10, 10, 60, 60))
    assert blank["status"] == "failed" and blank["status_reason"]
    off = fp.measure_marker(np.full((100, 100, 3), 255, np.uint8), (500, 500, 600, 600))
    assert off["status"] == "failed" and off["roi"] is None


def test_roi_matches_app_compute_roi_with_pad():
    m = fp.measure_marker(np.full((300, 400, 3), 255, np.uint8), (10.7, 30.2, 150.1, 290.9))
    assert m["roi"] == [0, 10, 171, 300]       # floor-20 clamped, ceil+20 clamped


# ============================================================================================
# sheet identification (geometry only)
# ============================================================================================
@pytest.mark.parametrize("sheet", SHEETS)
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_identify_all_markers_and_three(sheet, k):
    for n in (4, 3):
        for corners in itertools.combinations(CORNERS, n):
            mk, wh, truth = _geom_markers(sheet, corners, 6.0, k, noise_px=0.25, seed=n)
            res = fp.identify_sheet(mk, wh)
            assert res["status"] == "identified" and res["sheet_type"] == sheet, (corners, res["candidates"][:3])
            assert not res["corners_ambiguous"]
            assert res["assignment"] == {100 + CORNERS.index(c): c for c in corners}
            assert res["orientation_deg"] == 90 * k
            assert res["cf_px_per_cm_sheet_fit"] == pytest.approx(60.0, rel=0.002)
            assert res["fit"]["cost_mm"] <= fp.MAX_SHEET_COST_MM
            # reconstructed markers land on the literal geometry
            for c in set(CORNERS) - set(corners):
                assert res["corners"][c]["observed"] is False
                for r in ROLES + ("BR",):
                    assert np.hypot(*np.subtract(res["corners"][c]["squares"][r], truth[c][r])) < 1.0
            assert len(res["page_corners_px"]) == 4


@pytest.mark.parametrize("sheet", SHEETS)
@pytest.mark.parametrize("kind", list(PAIRS))
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_identify_pairs_on_fpfit_crop(sheet, kind, k):
    for pair in PAIRS[kind]:
        mk, wh, truth = _geom_markers(sheet, pair, 5.0, k, noise_px=0.2, seed=k)
        res = fp.identify_sheet(mk, wh)
        assert res["status"] == "identified", (pair, res["status"], res["candidates"][:4])
        assert res["sheet_type"] == sheet, (pair, res["candidates"][:4])
        assert not res["corners_ambiguous"]
        assert res["assignment"] == {100 + CORNERS.index(c): c for c in pair}
        assert res["orientation_deg"] == 90 * k
        for c in set(CORNERS) - set(pair):
            assert np.hypot(*np.subtract(res["corners"][c]["squares"]["TL"], truth[c]["TL"])) < 2.0


@pytest.mark.parametrize("sheet", SHEETS)
def test_identify_single_marker_is_undetermined(sheet):
    for c in CORNERS:
        mk, wh, _ = _geom_markers(sheet, (c,), 5.0)
        res = fp.identify_sheet(mk, wh)
        assert res["status"] == "undetermined" and res["sheet_type"] is None
        assert res["orientation_deg"] == 0 and res["corners"] == {}
    assert fp.identify_sheet([], (100, 100))["status"] == "undetermined"


def test_letter_vs_legal_horizontal_pair_by_image_extent():
    # FPfit Letter crop: Legal's reconstruction (77 mm lower) falls outside the image.
    mk, wh, _ = _geom_markers("Letter", ("TL", "TR"), 5.0)
    res = fp.identify_sheet(mk, wh)
    assert res["status"] == "identified" and res["sheet_type"] == "Letter"
    adm = {c["sheet_type"] for c in res["candidates"] if c["admissible"]}
    assert {"Letter", "Legal"} <= adm                     # tied on cost, split by extent
    # FPfit Legal crop: both fit inside, Legal wins on margin symmetry.
    mk, wh, _ = _geom_markers("Legal", ("BL", "BR"), 5.0)
    res = fp.identify_sheet(mk, wh)
    assert res["status"] == "identified" and res["sheet_type"] == "Legal"


def test_letter_vs_legal_horizontal_pair_ambiguous_without_extent():
    # A big, asymmetric photo gives neither extent nor symmetry evidence.
    mk, wh, _ = _geom_markers("Letter", ("TL", "TR"), 5.0, extra_mm=BIG_PHOTO)
    res = fp.identify_sheet(mk, wh)
    assert res["status"] == "ambiguous"
    assert {res["sheet_type"]} | {c["sheet_type"] for c in res["candidates"][:4]} >= {"Letter", "Legal"}
    assert res["cf_px_per_cm_sheet_fit"] is None
    assert all(v["observed"] for v in res["corners"].values())     # nothing guessed
    assert res["page_corners_px"] is None


def test_unique_horizontal_pair_in_big_image_has_ambiguous_corners():
    # A5 is the only 78 mm sheet, but top-vs-bottom cannot be told without extent.
    mk, wh, _ = _geom_markers("A5", ("BL", "BR"), 5.0, extra_mm=BIG_PHOTO)
    res = fp.identify_sheet(mk, wh)
    assert res["status"] == "identified" and res["sheet_type"] == "A5"
    assert res["corners_ambiguous"] is True
    assert res["cf_px_per_cm_sheet_fit"] == pytest.approx(50.0, rel=0.002)


def test_a4_vs_letter_horizontal_pair_resolved_by_scale():
    for sheet, other in (("A4", "Letter"), ("Letter", "A4")):
        mk, wh, _ = _geom_markers(sheet, ("TL", "TR"), 5.0, extra_mm=BIG_PHOTO)
        res = fp.identify_sheet(mk, wh)
        others = [c for c in res["candidates"] if c["sheet_type"] == other]
        assert others and all(not c["admissible"] and c["scale_dev_mm"] > 4.0 for c in others)
        assert other not in {c["sheet_type"] for c in res["candidates"] if c["admissible"]}
    # on an FPfit crop A4 is identified (Legal_legacy, also 140 mm, is split off by extent)
    mk, wh, _ = _geom_markers("A4", ("TL", "TR"), 5.0)
    assert fp.identify_sheet(mk, wh)["sheet_type"] == "A4"


def test_rotation_filter_and_convention():
    mk, wh, _ = _geom_markers("Letter", CORNERS, 5.0, 1)
    res = fp.identify_sheet(mk, wh)
    assert res["fit"]["rotation_deg"] == pytest.approx(90.0, abs=0.01)
    for c in res["candidates"]:
        assert abs(((c["rotation_deg"] - 90 + 180) % 360) - 180) <= fp.MAX_FIT_ROTATION_ERR_DEG
    # project_sheet_points reproduces the fit
    sc = CAT["sheets"]["Letter"]["square_centers_mm"]
    got = fp.project_sheet_points(res["fit"], [sc["TL"]["TL"]])[0]
    assert np.allclose(got, mk[0]["roles"]["TL"], atol=0.01)


def test_unrecognized_layout():
    # two Letter markers 30% too far apart for any sheet
    mk, wh, _ = _geom_markers("Letter", ("TL", "BR"), 5.0)
    mk[1]["roles"] = {r: [x * 1.3, y * 1.3] for r, (x, y) in mk[1]["roles"].items()}
    res = fp.identify_sheet(mk, (wh[0] * 2, wh[1] * 2))
    assert res["status"] == "unrecognized" and res["candidates"]
    assert not any(c["admissible"] for c in res["candidates"])


def test_duplicate_detections_are_collapsed():
    mk, wh, _ = _geom_markers("Letter", ("TL", "TR"), 5.0)
    dup = dict(mk[0], detection_id=999)
    res = fp.identify_sheet(mk + [dup], wh)
    assert res["status"] == "identified" and 999 not in res["assignment"]


# ============================================================================================
# sheet identification under real-like capture errors (review findings [1]-[3])
# ============================================================================================
def _real_like_markers(sheet, corners, ppmm=12.0, k=0, *, noise_px=0.5, seed=0,
                       keystone=0.016, bias=0.005, extra_mm=None):
    """identify_sheet input with the errors measured on the two real FPfit images: each
    marker's 20 mm pitch reads `bias` above the layout scale (15_1: +0.5%, 5_1: +0.4%) and a
    projective keystone makes the bottom of the page `keystone` larger than the top (15_1:
    1.6%), plus Gaussian center noise. The image is the FPfit crop (+ extra_mm margins)."""
    sc = CAT["sheets"][sheet]["square_centers_mm"]
    lo, hi = _frame(sheet, 15.0)
    if extra_mm is not None:
        lo = lo - np.array(extra_mm[:2], float)
        hi = hi + np.array(extra_mm[2:], float)
    ctr = (lo + hi) / 2.0
    g = keystone / (hi[1] - lo[1])

    def project(p):
        x, y = p[0] - ctr[0], p[1] - ctr[1]
        w = 1.0 - g * y
        return np.array([x / w + ctr[0], y / w + ctr[1]])

    W = int(round((hi[0] - lo[0]) * ppmm))
    H = int(round((hi[1] - lo[1]) * ppmm))
    rng = np.random.default_rng(seed)
    out = []
    for c in corners:
        cen = np.array(sc[c]["C"], float)
        roles = {}
        for r in ROLES:
            q = cen + (np.array(sc[c][r], float) - cen) * (1.0 + bias)
            x, y = (project(q) - lo) * ppmm + rng.normal(0, noise_px, 2)
            roles[r] = list(_rot_point(x, y, W, H, k))
        a = np.hypot(*np.subtract(roles["TR"], roles["TL"]))
        b = np.hypot(*np.subtract(roles["BL"], roles["TL"]))
        out.append({"detection_id": 100 + CORNERS.index(c), "roles": roles, "pxcm": (a + b) / 4.0})
    return out, ((W, H) if k % 2 == 0 else (H, W))


@pytest.mark.parametrize("sheet", ["A3", "Tabloid"])
@pytest.mark.parametrize("n", [2, 3, 4])
def test_large_sheets_identified_with_pitch_bias_and_keystone(sheet, n):
    # Before: the scale term was |ln(s/s_pitch)| * span_mm against a fixed 3 mm budget, so
    # A3/Tabloid (411 mm diagonal) went "unrecognized" for every 4-marker capture.
    for corners in itertools.combinations(CORNERS, n):
        for k in range(4):
            for seed in range(2):
                mk, wh = _real_like_markers(sheet, corners, 12.0, k, seed=seed)
                res = fp.identify_sheet(mk, wh)
                assert res["status"] == "identified" and res["sheet_type"] == sheet, \
                    (corners, k, res["candidates"][:3])
                assert res["assignment"] == {100 + CORNERS.index(c): c for c in corners}
                assert not res["corners_ambiguous"]
                assert res["cf_px_per_cm_sheet_fit"] == pytest.approx(120.0, rel=0.012)
                assert res["fit"]["scale_dev_pct"] < 100 * fp.MAX_SHEET_SCALE_DEV


@pytest.mark.parametrize("ppmm", [3.0, 12.0])
def test_every_sheet_and_subset_with_pitch_bias_and_keystone(ppmm):
    for sheet in SHEETS:
        for n in (2, 3, 4):
            for corners in itertools.combinations(CORNERS, n):
                for k in (0, 1):
                    mk, wh = _real_like_markers(sheet, corners, ppmm, k, seed=n + k)
                    res = fp.identify_sheet(mk, wh)
                    fam = CAT["sheets"][sheet]["family"]
                    assert res["status"] == "identified", (sheet, corners, res["candidates"][:3])
                    assert CAT["sheets"][res["sheet_type"]]["family"] == fam
                    if res["sheet_type"] != sheet:          # only a Legal-family layout tie
                        assert res["corners_ambiguous"] and res["cf_px_per_cm_sheet_fit"] is None


def test_scale_gate_is_size_independent():
    # the same 1.5% pitch-vs-layout disagreement is admissible on A5 and on A3 ...
    for sheet in ("A5", "Letter", "A3", "Tabloid"):
        mk, wh = _real_like_markers(sheet, ("TL", "BR"), 10.0, keystone=0.0, bias=0.015,
                                    noise_px=0.0)
        res = fp.identify_sheet(mk, wh)
        assert res["status"] == "identified" and res["sheet_type"] == sheet
        assert res["fit"]["scale_dev_pct"] == pytest.approx(1.5, abs=0.1)
    # ... and 2.5% is not, whatever the size
    for sheet in ("A5", "A3"):
        mk, wh = _real_like_markers(sheet, ("TL", "BR"), 10.0, keystone=0.0, bias=0.025,
                                    noise_px=0.0)
        res = fp.identify_sheet(mk, wh)
        assert res["status"] == "unrecognized"
        true = [c for c in res["candidates"] if c["sheet_type"] == sheet]
        assert true and true[0]["rms_mm"] < 0.5 and not true[0]["admissible"]


@pytest.mark.parametrize("sheet,other", [("A4", "A3"), ("A3", "A4"), ("A5", "Tabloid"),
                                         ("Tabloid", "A5"), ("Letter", "A4"), ("A4", "Letter")])
def test_wrong_size_sheets_still_rejected_by_scale(sheet, other):
    # horizontal pair in an ordinary photo: no extent/symmetry evidence, scale alone decides
    for k in range(4):
        for seed in range(3):
            mk, wh = _real_like_markers(sheet, ("TL", "TR"), 12.0, k, seed=seed,
                                        keystone=0.016 if seed % 2 else -0.016,
                                        extra_mm=BIG_PHOTO)
            res = fp.identify_sheet(mk, wh)
            wrong = [c for c in res["candidates"] if c["sheet_type"] == other]
            assert all(not c["admissible"] and not c["in_pool"] for c in wrong)
            assert all(c["scale_dev_pct"] > 2.5 for c in wrong)
            assert res["sheet_type"] != other


@pytest.mark.parametrize("pair", [("TL", "BR"), ("TR", "BL")])
def test_tabloid_diagonal_pair_is_not_identified_as_a3(pair):
    # A3 and Tabloid diagonals differ by 0.16% in length; the +0.5% pitch bias used to make
    # A3 the only hypothesis under budget although its reconstruction left the image.
    for k in range(4):
        for seed in range(3):
            mk, wh = _real_like_markers("Tabloid", pair, 12.0, k, noise_px=1.0, seed=seed)
            res = fp.identify_sheet(mk, wh)
            assert res["status"] == "identified" and res["sheet_type"] == "Tabloid", \
                res["candidates"][:3]


def test_pool_is_relative_to_best_admissible_not_only_admissible():
    # Make the TRUE Tabloid barely inadmissible on scale while the wrong A3 (reconstruction
    # outside the image) stays admissible: the inside-image tie-break must still pick Tabloid.
    mk, wh = _real_like_markers("Tabloid", ("TL", "BR"), 12.0, noise_px=0.0, keystone=0.0,
                                bias=0.0065)
    base = fp.identify_sheet(mk, wh)
    tab = next(c for c in base["candidates"] if c["sheet_type"] == "Tabloid")
    a3 = next(c for c in base["candidates"] if c["sheet_type"] == "A3")
    assert a3["scale_dev_pct"] < tab["scale_dev_pct"] and not a3["inside_image"]
    gate = (a3["scale_dev_pct"] + tab["scale_dev_pct"]) / 200.0
    res = fp.identify_sheet(mk, wh, max_scale_dev=gate)
    cands = {c["sheet_type"]: c for c in res["candidates"]}
    assert cands["A3"]["admissible"] and not cands["Tabloid"]["admissible"]
    assert res["status"] == "identified" and res["sheet_type"] == "Tabloid"
    assert cands["Tabloid"]["in_pool"]
    # nothing admissible at all -> still unrecognized
    res = fp.identify_sheet(mk, wh, max_scale_dev=0.0001)
    assert res["status"] == "unrecognized"


def _pad_photo(img, crops, left, top, right, bottom):
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(232, 232, 232))
    return out, [dict(c, x1=c["x1"] + left, x2=c["x2"] + left, y1=c["y1"] + top,
                      y2=c["y2"] + top) for c in crops]


@pytest.mark.parametrize("sheet", ["Legal", "Legal_legacy"])
def test_legal_layouts_tied_report_legal_without_guessing(sheet):
    # Vertical pair in an ordinary photo: Legal (146 x 279) and Legal (legacy) (140 x 280)
    # cannot be told apart, and before one of them was "identified" with inferred markers
    # 6.5 mm off and a sheet-fit CF 0.4% off.
    for pair in (("TL", "BL"), ("TR", "BR")):
        for k in range(4):
            for seed in range(3):
                mk, wh = _real_like_markers(sheet, pair, 10.0, k, seed=seed, keystone=0.0,
                                            bias=0.0, extra_mm=BIG_PHOTO)
                res = fp.identify_sheet(mk, wh)
                assert res["status"] == "identified"
                if res["sheet_type"] == sheet and not res["corners_ambiguous"]:
                    continue                                  # resolved, and correctly
                assert res["sheet_type"] == "Legal" and res["label"] == "Legal"
                assert res["corners_ambiguous"] is True
                assert all(v["observed"] for v in res["corners"].values())
                assert res["page_corners_px"] is None and res["cf_px_per_cm_sheet_fit"] is None
                assert set(res["tied_labels"]) == {"Legal", "Legal (legacy)"}


def test_legal_layout_tie_in_analyze_uses_marker_mean():
    page = Page("Legal_legacy", 4.0, corners=("TL", "BL"))
    img, crops = _pad_photo(page.bgr(seed=3), page.crops(("TL", "BL")), 160, 240, 600, 1040)
    res = fp.analyze_fieldprism(img, crops)
    sh = res["sheet"]
    assert sh["status"] == "identified" and sh["label"] == "Legal" and sh["corners_ambiguous"]
    assert sh["cf_px_per_cm_sheet_fit"] is None and sh["cf_source_detail"] == "marker_mean"
    assert sh["n_fp_inferred"] == 0 and res["confidence"] == "high"
    assert all(m["sheet_corner"] is None for m in res["markers"])
    assert any("Legal (legacy)" in r and "no markers inferred" in r for r in sh["fp_reasons"])
    assert res["anchor_cf"] == pytest.approx(40.0, rel=0.01)
    row = fp.sheet_row(res, 1, CAT["catalog_version"])
    assert row["sheet_type"] == "Legal" and row["corners_ambiguous"] == 1
    assert row["cf_px_per_cm_sheet_fit"] is None and row["fit_cost_mm"] is not None


@pytest.mark.parametrize("sheet", ["Legal", "Legal_legacy"])
@pytest.mark.parametrize("pair", [("TL", "BR"), ("TR", "BL"), ("TL", "BL"), ("TR", "BR")])
def test_legal_layouts_split_by_symmetry_on_fpfit_crop(sheet, pair):
    # On an FPfit crop both layouts can pass the 4 mm symmetry test; the more symmetric one
    # wins, so the exact layout is identified (never the other one with inferred markers).
    for k in range(4):
        for seed in range(2):
            mk, wh = _real_like_markers(sheet, pair, 12.0, k, seed=seed)
            res = fp.identify_sheet(mk, wh)
            assert res["status"] == "identified" and res["sheet_type"] == sheet, \
                res["candidates"][:3]
            assert not res["corners_ambiguous"]


# ============================================================================================
# analyze_fieldprism on rendered sheets
# ============================================================================================
@pytest.mark.parametrize("sheet,ppmm", [("A5", 9.0), ("A4", 5.0), ("A3", 3.5), ("Letter", 5.5),
                                         ("Legal", 4.5), ("Tabloid", 3.5), ("Legal_legacy", 4.5)])
def test_analyze_rendered_sheet_every_type(sheet, ppmm):
    page = Page(sheet, ppmm)
    for k in (0, 1, 2, 3):
        res = fp.analyze_fieldprism(_rot(page.bgr(noise=5, blur=0.7, seed=k), k), page.crops(k=k))
        sh = res["sheet"]
        assert sh["status"] == "identified" and sh["sheet_type"] == sheet, sh["candidates"][:3]
        assert sh["orientation_deg"] == 90 * k
        assert res["anchor_cf"] == pytest.approx(10 * ppmm, rel=0.004)
        assert sh["cf_source_detail"] == "sheet_fit" and res["confidence"] == "high"
        assert {m["detection_id"]: m["sheet_corner"] for m in res["markers"]} == \
            {100 + i: c for i, c in enumerate(CORNERS)}
        assert sh["n_fp_used"] == 4 and sh["n_fp_inferred"] == 0


def test_analyze_three_markers_infers_fourth():
    page = Page("Letter", 6.0, corners=("TR", "BL", "BR"))
    res = fp.analyze_fieldprism(page.bgr(noise=4, seed=2), page.crops(("TR", "BL", "BR")))
    sh = res["sheet"]
    assert sh["status"] == "identified" and sh["n_fp_used"] == 3 and sh["n_fp_inferred"] == 1
    tl = sh["corners"]["TL"]
    assert tl["observed"] is False and tl["detection_id"] is None
    for r in ROLES + ("BR",):
        assert np.hypot(*np.subtract(tl["squares"][r], page.truth["TL"][r])) < 1.5


def test_analyze_failed_marker_is_skipped_and_inferred():
    page = Page("Letter", 6.0)
    img = page.bgr(seed=4)
    x1, y1, x2, y2 = (int(v) for v in page.boxes["TL"])
    img[y1 + 2:y2 - 2, x1 + 70:x2 - 2] = 232           # wipe TR + C squares of the TL marker
    res = fp.analyze_fieldprism(img, page.crops())
    tl = next(m for m in res["markers"] if m["detection_id"] == 100)
    assert tl["verdict"] == "skipped" and tl["status"] == "failed"
    assert res["sheet"]["n_fp_used"] == 3 and res["sheet"]["n_fp_inferred"] == 1
    assert res["confidence"] == "high"


def test_peer_cluster_rejects_outlier_marker():
    page = Page("Letter", 6.0, marker_scale={"BR": 1.10})
    res = fp.analyze_fieldprism(page.bgr(seed=5), page.crops())
    br = next(m for m in res["markers"] if m["detection_id"] == 103)
    assert br["valid"] and br["verdict"] == "rejected" and "disagrees" in br["verdict_note"]
    sh = res["sheet"]
    assert sh["n_fp_rejected"] == 1 and sh["n_fp_used"] == 3 and sh["status"] == "identified"
    assert sh["corners"]["BR"]["observed"] is False and sh["n_fp_inferred"] == 1
    assert br["pct_vs_fp"] == pytest.approx(10.0, abs=1.0)
    assert res["confidence"] == "high"


def test_confidence_single_marker_rules():
    page = Page("A4", 6.0, corners=("TL",))
    img, crops = page.bgr(seed=6), page.crops(("TL",))
    hi = fp.analyze_fieldprism(img, crops)
    assert hi["confidence"] == "high" and hi["sheet"]["status"] == "undetermined"
    assert hi["sheet"]["cf_source_detail"] == "marker_mean"
    assert hi["anchor_cf"] == pytest.approx(60.0, rel=0.01)
    med = fp.analyze_fieldprism(img, crops, allow_single_marker=False)
    assert med["confidence"] == "medium" and med["anchor_cf"] == hi["anchor_cf"]


def test_confidence_low_when_two_markers_disagree():
    page = Page("Letter", 6.0, corners=("TL", "BR"), marker_scale={"BR": 1.08})
    res = fp.analyze_fieldprism(page.bgr(seed=7), page.crops(("TL", "BR")))
    assert res["confidence"] == "low"
    assert all(m["verdict"] == "used" for m in res["markers"])
    assert res["sheet"]["cf_source_detail"] == "marker_mean"


def test_confidence_none_without_valid_markers():
    res = fp.analyze_fieldprism(np.full((400, 400, 3), 255, np.uint8),
                                [{"detection_id": 1, "x1": 10, "y1": 10, "x2": 200, "y2": 200}])
    assert res["confidence"] is None and res["anchor_cf"] is None
    assert res["sheet"]["n_fp_detected"] == 1 and res["sheet"]["n_fp_valid"] == 0
    assert res["markers"][0]["verdict"] == "skipped"
    empty = fp.analyze_fieldprism(np.zeros((10, 10, 3), np.uint8), [])
    assert empty["confidence"] is None and empty["sheet"]["status"] == "undetermined"


def test_high_downgraded_when_sheet_fit_disagrees_with_markers():
    # A5 pair whose markers print 1.5% large: they agree with each other, the 78 mm sheet
    # still fits (scale 1.5% < MAX_SHEET_SCALE_DEV), but sheet fit vs marker mean differs by
    # more than peer_tol = 1%.
    page = Page("A5", 8.0, corners=("TL", "TR"), marker_scale={"TL": 1.015, "TR": 1.015})
    res = fp.analyze_fieldprism(page.bgr(seed=8), page.crops(("TL", "TR")), peer_tol=0.01)
    sh = res["sheet"]
    assert sh["status"] == "identified" and sh["sheet_type"] == "A5"
    assert res["confidence"] == "medium"
    assert any("sheet-fit CF differs" in r for r in sh["fp_reasons"])
    # 2.5% large is beyond the scale gate on ANY sheet size -> no sheet, marker-mean anchor
    page = Page("A5", 8.0, corners=("TL", "TR"), marker_scale={"TL": 1.025, "TR": 1.025})
    res = fp.analyze_fieldprism(page.bgr(seed=8), page.crops(("TL", "TR")))
    assert res["sheet"]["status"] == "unrecognized"
    assert res["sheet"]["cf_source_detail"] == "marker_mean"


def test_outputs_are_plain_and_picklable_and_rows_match_schema():
    page = Page("Letter", 5.0, corners=("TL", "TR", "BL"))
    res = fp.analyze_fieldprism(page.bgr(seed=9), page.crops(("TL", "TR", "BL")))
    assert pickle.loads(pickle.dumps(res)) == res

    def plain(o):
        if isinstance(o, dict):
            return all(plain(k) and plain(v) for k, v in o.items())
        if isinstance(o, (list, tuple)):
            return all(plain(v) for v in o)
        return o is None or type(o) in (bool, int, float, str)

    assert plain(res)
    marker_cols = {
        "specimen_id", "detection_id", "crop_index", "det_conf", "x1", "y1", "x2", "y2",
        "roi_x0", "roi_y0", "roi_x1", "roi_y1", "status", "status_reason", "valid",
        "validation_json", "verdict", "verdict_note", "n_peaks", "holes_filled", "peak_area_ratio",
        "tl_x", "tl_y", "tr_x", "tr_y", "c_x", "c_y", "bl_x", "bl_y", "br_x", "br_y",
        "pitch_h_px", "pitch_v_px", "pxcm", "pxcm_original", "pct_vs_fp", "orientation_deg",
        "sheet_corner"}
    sheet_cols = {
        "specimen_id", "catalog_version", "n_fp_detected", "n_fp_measured", "n_fp_valid",
        "n_fp_used", "n_fp_rejected", "n_fp_inferred", "sheet_status", "sheet_type", "sheet_label",
        "corners_ambiguous", "sheet_candidates_json", "orientation_deg", "fit_rotation_deg",
        "fit_scale_px_per_mm", "fit_tx", "fit_ty", "fit_rms_mm", "fit_max_mm", "fit_scale_dev_mm",
        "fit_cost_mm", "cf_px_per_cm_sheet_fit", "cf_px_per_cm_marker_mean", "cf_px_per_cm_fp",
        "cf_source_detail", "fp_peer_spread_pct", "fp_confidence", "fp_reasons_json",
        "corners_json", "page_corners_json", "fpfit_margins_json"}
    for i, m in enumerate(res["markers"]):
        row = fp.marker_row(m, 7, i, 0.5)
        assert set(row) == marker_cols and plain(row)
        assert row["pxcm_original"] == pytest.approx(row["pxcm"] / 0.5)
        assert row["br_x"] == pytest.approx(row["tr_x"] + row["bl_x"] - row["tl_x"])
        assert json.loads(row["validation_json"])["pitch_ratio"]["ok"] is True
        assert row["sheet_corner"] in CORNERS and row["valid"] == 1
    srow = fp.sheet_row(res, 7, CAT["catalog_version"])
    assert set(srow) == sheet_cols and plain(srow)
    assert srow["sheet_status"] == "identified" and srow["sheet_type"] == "Letter"
    assert srow["n_fp_inferred"] == 1 and srow["corners_ambiguous"] == 0
    corners = json.loads(srow["corners_json"])
    assert corners["BR"]["observed"] is False and len(json.loads(srow["page_corners_json"])) == 4
    assert set(json.loads(srow["fpfit_margins_json"])) == {"left", "right", "top", "bottom"}
    # a failed marker still produces a complete row
    failed = fp.analyze_fieldprism(np.full((300, 300, 3), 255, np.uint8),
                                   [{"detection_id": 5, "x1": 1, "y1": 1, "x2": 100, "y2": 100}])
    frow = fp.marker_row(failed["markers"][0], 7, 0, 1.0)
    assert set(frow) == marker_cols and frow["status"] == "failed" and frow["tl_x"] is None
    assert set(fp.sheet_row(failed, 7, "v")) == sheet_cols
