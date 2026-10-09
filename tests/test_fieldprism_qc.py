"""FieldPrism QC sections (ruler_lattice/qc.py): fp_marker_section, fp_sheet_section, and the
FieldPrism wording of the CF summary / reconciliation blocks.

The sections are rebuilt from DB rows, so every test feeds them plain row dicts -- including rows
with missing or malformed JSON columns -- and checks that they render at the panel width and SAY
the right things (ImageDraw.text is spied on). The two pre-existing functions gained keyword
arguments; with the defaults they must stay pixel-identical to the committed code, which the last
tests check against qc.py as it was BEFORE FieldPrism support (commit 26627c4, pinned so the check
stays meaningful after this work is committed), loaded as a separate module.
"""
from __future__ import annotations

import json
import subprocess
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import ImageDraw

from leafmachine3.inference.ruler_lattice import qc


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _texts(monkeypatch, fn, *a, **kw):
    """Run fn and return (image, [(text, fill)]) for every ImageDraw.text call it made."""
    seen = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        seen.append((str(text), kwargs.get("fill")))
        return real(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    im = fn(*a, **kw)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", real)
    return im, seen


def _joined(seen):
    return "\n".join(t for t, _ in seen)


BOX = (1000.4, 500.2, 1200.0, 700.0)     # working px; crop pixel (u, v) = (1000 + u, 500 + v)
PXCM = 50.0                              # pitch 100 px


@pytest.fixture()
def crop_path(tmp_path):
    """A synthetic 200 x 200 marker crop: TL/TR/C/BL 1 cm squares printed, BR empty."""
    img = np.full((200, 200, 3), 235, np.uint8)
    for cx, cy in ((50, 50), (150, 50), (100, 100), (50, 150)):
        cv2.rectangle(img, (cx - 25, cy - 25), (cx + 25, cy + 25), (20, 20, 20), -1)
    p = tmp_path / "x__BBOX-ruler__1000_500_1200_700.jpg"
    cv2.imwrite(str(p), img)
    return str(p)


def _marker(**over):
    """A measured, valid, used ruler_FP_marker row (column names per the DB contract)."""
    ox, oy = 1000.0, 500.0
    row = dict(specimen_id=1, detection_id=7, crop_index=0, det_conf=0.93,
               x1=BOX[0], y1=BOX[1], x2=BOX[2], y2=BOX[3],
               roi_x0=980, roi_y0=480, roi_x1=1220, roi_y1=720,
               status="measured", status_reason=None, valid=1,
               validation_json=json.dumps({
                   "pitch_ratio": {"value": 0.004, "limit": 0.05, "ok": True},
                   "right_angle_deg": {"value": 0.1, "limit": 3.0, "ok": True},
                   "c_mid": {"value": 0.002, "limit": 0.03, "ok": True},
                   "c_diag": {"value": 0.003, "limit": 0.05, "ok": True},
                   "peak_area_ratio": {"value": 0.97, "limit": 0.3, "ok": True}}),
               verdict="used", verdict_note=None, n_peaks=4, holes_filled=0,
               peak_area_ratio=0.97,
               tl_x=ox + 50, tl_y=oy + 50, tr_x=ox + 150, tr_y=oy + 50, c_x=ox + 100, c_y=oy + 100,
               bl_x=ox + 50, bl_y=oy + 150, br_x=ox + 150, br_y=oy + 150,
               pitch_h_px=100.0, pitch_v_px=100.0, pxcm=PXCM, pxcm_original=PXCM,
               pct_vs_fp=0.4, orientation_deg=0, sheet_corner="TL")
    row.update(over)
    return row


# --------------------------------------------------------------------------- #
# fp_marker_section
# --------------------------------------------------------------------------- #
def test_used_marker_has_app_labels_in_app_colors(monkeypatch, crop_path) -> None:
    im, seen = _texts(monkeypatch, qc.fp_marker_section, _marker(), crop_path, BOX, title_index=2)
    assert im.size[0] == qc.W_OUT
    fills = {t: f for t, f in seen}
    # the labels the FieldPrism app draws, in the app's colors (RGBA on the overlay layer)
    for role, rgb in qc.FP_ROLE_COLORS.items():
        assert role in fills and tuple(fills[role][:3]) == rgb
    assert tuple(fills["1 cm = 50 px"][:3]) == (255, 255, 255)
    txt = _joined(seen)
    for want in ("1 -- FieldPrism Marker 2", "detection 7", "MEASURED, VALID", "USED",
                 "|a/b - 1|", "right-angle error", "C-mid error", "C-diagonal error",
                 "peak-area ratio", "orientation vote", "50.00", "+0.40 %", "sheet corner"):
        assert want in txt, want
    # the predicted BR cell is a filled app-green square
    assert (np.asarray(im) == qc.FP_BR_USED).all(axis=2).sum() > 1000


def test_invalid_marker_is_flagged_and_outlined_red(monkeypatch, crop_path) -> None:
    bad = _marker(valid=0, verdict="skipped", sheet_corner=None,
                  verdict_note="failed validation: TL-TR and TL-BL pitches differ by 15.5% (limit "
                               "5%); square plateaus are unbalanced: min/max area 0.08",
                  validation_json=json.dumps({
                      "pitch_ratio": {"value": 0.155, "limit": 0.05, "ok": False},
                      "peak_area_ratio": {"value": 0.08, "limit": 0.3, "ok": False}}))
    im, seen = _texts(monkeypatch, qc.fp_marker_section, bad, crop_path, BOX)
    txt = _joined(seen)
    assert "FAILED VALIDATION (|a/b - 1|, peak-area ratio)" in txt
    assert "SKIPPED" in txt and "FAIL" in txt
    # the note is split, one reason per line
    assert any(t.startswith("square plateaus are unbalanced") for t, _ in seen)
    arr = np.asarray(im)
    assert (arr == qc.FP_BR_USED).all(axis=2).sum() == 0          # never green when not used
    assert (arr == qc.FP_BR_REJECTED).all(axis=2).sum() > 100     # red BR outline


def test_failed_marker_draws_no_labels(monkeypatch, crop_path) -> None:
    row = _marker(status="failed", status_reason="only 3 square peaks found (need 4)",
                  valid=None, verdict="skipped", validation_json=None, peak_area_ratio=None,
                  pitch_h_px=None, pitch_v_px=None, pxcm=None, pxcm_original=None,
                  pct_vs_fp=None, orientation_deg=None, sheet_corner=None,
                  **{f"{r}_{a}": None for r in ("tl", "tr", "c", "bl", "br") for a in "xy"})
    im, seen = _texts(monkeypatch, qc.fp_marker_section, row, crop_path, BOX)
    assert im.size[0] == qc.W_OUT
    assert "NOT MEASURED (failed) -- only 3 square peaks found (need 4)" in _joined(seen)
    assert not {"TL", "TR", "C", "BL"} & {t for t, _ in seen}


def test_marker_without_crop_or_json_still_renders(monkeypatch) -> None:
    """Crop file gone and no validation_json: the checks are recomputed from the centers."""
    row = _marker(validation_json=None, peak_area_ratio=None)
    for k in ("roi_x0", "pct_vs_fp", "sheet_corner", "br_x", "br_y"):
        row.pop(k)
    im, seen = _texts(monkeypatch, qc.fp_marker_section, row, "/nonexistent/crop.jpg", BOX)
    assert im.size[0] == qc.W_OUT
    txt = _joined(seen)
    assert "0.00 %" in txt                       # |a/b - 1| recomputed from equal pitches
    assert "<= 5.0 %" in txt                     # limit from fieldprism.py / the documented default
    # BR recomputed as TR + BL - TL, so the green cell is still drawn
    assert (np.asarray(im) == qc.FP_BR_USED).all(axis=2).sum() > 1000
    im2 = qc.fp_marker_section(row, None, BOX)
    assert im2.size[0] == qc.W_OUT


def test_marker_on_a_rescaled_crop_maps_centers_proportionally(tmp_path) -> None:
    """A crop file at another resolution than the box: labels still land on the squares."""
    img = np.full((100, 100, 3), 235, np.uint8)
    p = tmp_path / "half.jpg"
    cv2.imwrite(str(p), img)
    im = qc.fp_marker_section(_marker(), str(p), BOX)
    assert im.size[0] == qc.W_OUT


def _rot_marker(k, **over):
    """_marker() on a sheet photographed rotated (np.rot90(img, k): k CCW quarter turns): the role
    centers move within the 200 x 200 box, e.g. k=1 puts TL bottom-left and BR top-right."""
    row = _marker(orientation_deg=90 * k, **over)
    for r in ("tl", "tr", "c", "bl", "br"):
        u, v = row[f"{r}_x"] - 1000.0, row[f"{r}_y"] - 500.0
        for _ in range(k):
            u, v = v, 200.0 - u
        row[f"{r}_x"], row[f"{r}_y"] = 1000.0 + u, 500.0 + v
    return row


def _text_boxes(monkeypatch, fn, *a, **kw):
    """Run fn and return {text: [(xy, ink bbox)]} for every ImageDraw.text call it made."""
    seen = {}
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *args, **kwargs):
        bb = self.textbbox(xy, str(text), font=kwargs.get("font"), anchor=kwargs.get("anchor"))
        seen.setdefault(str(text), []).append((tuple(xy), bb))
        return real(self, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    fn(*a, **kw)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", real)
    return seen


def _bb_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_cm_text_stays_above_a_rotated_marker(monkeypatch, crop_path, k) -> None:
    """The crop is as photographed: at 90/180/270 degrees "above TL" is the marker's middle row,
    so the "1 cm" text is anchored on the on-image top-left cell -- TL when upright (the app)."""
    seen = _text_boxes(monkeypatch, qc.fp_marker_section, _rot_marker(k), crop_path, BOX)
    (xy, bb), = seen["1 cm = 50 px"]
    labels = {r: seen[r][0] for r in ("TL", "TR", "C", "BL")}
    for r, (_c, lb) in labels.items():
        assert not _bb_overlap(bb, lb), (k, r)
    S_disp = (labels["C"][0][0] - min(c[0] for c, _ in labels.values())) * 1.0   # one pitch / 2 = 1 cm
    top_row = min(c[1] for c, _ in labels.values())
    assert xy[1] <= top_row - S_disp / 2.0 + 1, k                # baseline above the marker's top edge
    if k == 0:
        tl = labels["TL"][0]
        f = qc._fp_font(qc.FP_LABEL_FRAC * PXCM * 2.0)            # the crop is shown at 2x
        probe = ImageDraw.Draw(qc.Image.new("RGBA", (8, 8)))
        assert xy == pytest.approx(qc._fp_cm_text_geom(probe, tl, f, "1 cm = 50 px")[:2])


def test_fp_top_left_cell_is_tl_upright_and_the_top_left_corner_rotated() -> None:
    up = {"TL": (50, 50), "TR": (150, 47), "C": (101, 99), "BL": (52, 150), "BR": (152, 147)}
    assert qc._fp_top_left_cell(up) == up["TL"]                     # slightly tilted, still TL
    for k, want in ((1, "TR"), (2, "BR"), (3, "BL")):
        row = _rot_marker(k)
        pts = {r.upper(): (row[f"{r}_x"], row[f"{r}_y"]) for r in ("tl", "tr", "c", "bl", "br")}
        assert qc._fp_top_left_cell(pts) == pts[want], k


def _orientation_row(seen):
    rows = [(t, f) for t, f in seen if t.startswith("orientation vote")]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("deg", [90, 180, 270])
def test_rotated_marker_orientation_is_a_warning_not_a_fail(monkeypatch, crop_path, deg) -> None:
    """validation_json has no orientation check, and a rotated sheet is supported: the row says
    "rotated" in the warning color, like the sheet section, never a red FAIL under a USED verdict."""
    _im, seen = _texts(monkeypatch, qc.fp_marker_section, _rot_marker(deg // 90), crop_path, BOX)
    text, fill = _orientation_row(seen)
    assert f"{deg} deg" in text and "rotated" in text and "FAIL" not in text
    assert tuple(fill) == qc._FP_WARN
    assert "MEASURED, VALID  ->  USED" in _joined(seen)


def test_upright_orientation_row_is_informational(monkeypatch, crop_path) -> None:
    _im, seen = _texts(monkeypatch, qc.fp_marker_section, _marker(), crop_path, BOX)
    text, fill = _orientation_row(seen)
    assert "0 deg" in text and "FAIL" not in text and "rotated" not in text
    assert tuple(fill) == qc._FP_GRAY


def test_orientation_row_fails_only_on_a_stored_orientation_check(monkeypatch, crop_path) -> None:
    val = json.loads(_marker()["validation_json"])
    val["orientation"] = {"value": 90, "limit": 0, "ok": False}
    row = _rot_marker(1, validation_json=json.dumps(val))
    _im, seen = _texts(monkeypatch, qc.fp_marker_section, row, crop_path, BOX)
    text, fill = _orientation_row(seen)
    assert text.rstrip().endswith("FAIL") and tuple(fill) == qc._FP_BAD


# --------------------------------------------------------------------------- #
# fp_sheet_section
# --------------------------------------------------------------------------- #
LETTER_CENTERS = qc._fp_sheet_model("Letter")["square_centers_mm"]


def _sheet_rows(observed=("TL", "TR", "BL", "BR"), status="identified", s_mm=10.0, **over):
    """A ruler_FP_sheet row + its marker rows for a Letter sheet at s_mm px/mm, unrotated."""

    def tf(p):
        return [p[0] * s_mm + 40.0, p[1] * s_mm + 30.0]

    corners, markers, asg = {}, [], {}
    for i, c in enumerate(qc.FP_CORNERS):
        sq = {r: tf(p) for r, p in LETTER_CENTERS[c].items()}
        obs = c in observed
        det = 100 + i
        corners[c] = {"observed": obs, "detection_id": det if obs else None, "squares": sq}
        xs = [p[0] for p in sq.values()]
        ys = [p[1] for p in sq.values()]
        m = dict(detection_id=det, specimen_id=1, x1=min(xs) - 60, y1=min(ys) - 60,
                 x2=max(xs) + 60, y2=max(ys) + 60, pxcm=10 * s_mm)
        if obs:
            m.update(status="measured", valid=1, verdict="used", sheet_corner=c,
                     **{f"{r.lower()}_{a}": sq[r][j] for r in ("TL", "TR", "C", "BL")
                        for j, a in enumerate("xy")})
            asg[str(det)] = c
        else:
            m.update(status="measured", valid=0, verdict="skipped", sheet_corner=None,
                     verdict_note="failed validation: pitches differ")
        markers.append(m)
    W, H = qc._fp_sheet_model("Letter")["page_mm"]
    sheet = dict(specimen_id=1, catalog_version="t", n_fp_detected=4, n_fp_measured=4,
                 n_fp_valid=len(observed), n_fp_used=len(observed), n_fp_rejected=0,
                 n_fp_inferred=4 - len(observed), sheet_status=status, sheet_type="Letter",
                 sheet_label="Letter", corners_ambiguous=0,
                 sheet_candidates_json=json.dumps([
                     {"sheet_type": "Letter", "assignment": asg, "cost_mm": 0.3, "rms_mm": 0.2,
                      "scale_dev_mm": 0.1, "rotation_deg": 0.0, "inside_image": True,
                      "margin_spread_mm": 0.4},
                     {"sheet_type": "Legal", "assignment": asg, "cost_mm": 9.8, "rms_mm": 8.1,
                      "scale_dev_mm": 1.7, "rotation_deg": 0.0, "inside_image": False,
                      "margin_spread_mm": 77.0}]),
                 orientation_deg=0, fit_rotation_deg=0.02, fit_scale_px_per_mm=s_mm, fit_tx=40.0,
                 fit_ty=30.0, fit_rms_mm=0.2, fit_max_mm=0.35, fit_scale_dev_mm=0.1,
                 fit_cost_mm=0.3, cf_px_per_cm_sheet_fit=10 * s_mm,
                 cf_px_per_cm_marker_mean=10 * s_mm * 1.004, cf_px_per_cm_fp=10 * s_mm,
                 cf_source_detail="sheet_fit", fp_peer_spread_pct=1.1, fp_confidence="high",
                 fp_reasons_json=json.dumps(["4 FieldPrism markers agree within 3%"]),
                 corners_json=json.dumps(corners),
                 page_corners_json=json.dumps([tf(p) for p in ((0, 0), (W, 0), (W, H), (0, H))]),
                 fpfit_margins_json=json.dumps({"left": 15, "right": 15, "top": 15, "bottom": 15}))
    sheet.update(over)
    return sheet, markers


def test_identified_sheet_four_markers(monkeypatch) -> None:
    sheet, markers = _sheet_rows()
    im, seen = _texts(monkeypatch, qc.fp_sheet_section, sheet, markers)
    assert im.size[0] == qc.W_OUT
    txt = _joined(seen)
    for want in ("FieldPrism Sheet Identification", "Sheet: FieldPrism Letter  --  IDENTIFIED",
                 "4 detected, 4 measured, 4 valid, 4 used, 0 rejected, 0 inferred",
                 "rms 0.20 mm", "max 0.35 mm", "scale dev 0.10 mm",
                 "CF from the sheet fit", "CF from the marker mean", "100.40 px/cm",
                 "FieldPrism anchor (working frame): 100.00 px/cm  from the sheet fit",
                 "confidence high", "4 FieldPrism markers agree within 3%",
                 "TL det100  used", "BR det103  used", "Legal",
                 "FPfit margins"):
        assert want in txt, want


def test_identified_sheet_three_plus_one_inferred(monkeypatch) -> None:
    sheet, markers = _sheet_rows(observed=("TR", "BL", "BR"))
    im, seen = _texts(monkeypatch, qc.fp_sheet_section, sheet, markers)
    txt = _joined(seen)
    assert "1 inferred" in txt
    # the dropped detection sitting at the inferred corner is named in the caption
    assert "TL  inferred (det100 skipped)" in txt
    assert (np.asarray(im) == qc.FP_INFERRED).all(axis=2).sum() > 50   # dashed magenta


@pytest.mark.parametrize("status,want", [
    ("ambiguous", "Sheet: FieldPrism Letter?  --  AMBIGUOUS (or Legal)"),
    ("undetermined", "Sheet: undetermined"),
    ("unrecognized", "Sheet: unrecognized"),
])
def test_other_statuses(monkeypatch, status, want) -> None:
    sheet, markers = _sheet_rows(status=status)
    if status != "ambiguous":
        sheet.update(sheet_type=None, sheet_label=None, cf_px_per_cm_sheet_fit=None,
                     cf_source_detail="marker_mean")
    im, seen = _texts(monkeypatch, qc.fp_sheet_section, sheet, markers)
    assert im.size[0] == qc.W_OUT
    assert want in _joined(seen)


def test_sheet_with_missing_and_malformed_json_columns(monkeypatch) -> None:
    """Nothing but the status: no counts, no fit, no JSON; then garbage JSON strings."""
    im, seen = _texts(monkeypatch, qc.fp_sheet_section, {"sheet_status": "undetermined"}, [])
    assert im.size[0] == qc.W_OUT
    assert "FieldPrism anchor (working frame): NONE" in _joined(seen)
    assert "(no candidate hypotheses recorded)" in _joined(seen)
    sheet, markers = _sheet_rows()
    for k in ("sheet_candidates_json", "fp_reasons_json", "corners_json", "page_corners_json",
              "fpfit_margins_json"):
        sheet[k] = "{not json"
    im = qc.fp_sheet_section(sheet, markers)
    assert im.size[0] == qc.W_OUT
    assert qc.fp_sheet_section(None, None).size[0] == qc.W_OUT


def test_sheet_model_layout_rule_matches_the_catalog(monkeypatch) -> None:
    """The schematic is drawn from the catalog; without it, the layout rule gives the same page."""
    with_cat = {t: qc._fp_sheet_model(t) for t in ("A5", "A4", "A3", "Letter", "Legal", "Tabloid")}
    monkeypatch.setattr(qc, "_fp_catalog", lambda: None)
    for t, m in with_cat.items():
        rule = qc._fp_sheet_model(t)
        assert rule["source"] == "layout rule"
        assert rule["layout_mm"] == pytest.approx(m["layout_mm"])
        for c in qc.FP_CORNERS:
            for r in ("TL", "TR", "C", "BL", "BR"):
                assert rule["square_centers_mm"][c][r] == pytest.approx(
                    m["square_centers_mm"][c][r]), (t, c, r)
    letter = qc._fp_sheet_model("letter")       # case-insensitive
    assert letter["square_centers_mm"]["BR"]["TL"] == (171.0, 230.0)
    assert letter["square_centers_mm"]["TL"]["BR"] == (45.0, 48.0)
    assert qc._fp_sheet_model("Custom") is None


# --------------------------------------------------------------------------- #
# FieldPrism wording of the CF summary and the reconciliation
# --------------------------------------------------------------------------- #
def test_cf_summary_fieldprism(monkeypatch) -> None:
    sheet, _ = _sheet_rows(n_fp_used=3, n_fp_inferred=1)
    im, seen = _texts(monkeypatch, qc.build_cf_summary_section, 100.2, 100.0, "k*sqrt(MP)",
                      anchor_source="fieldprism", fp_sheet=sheet)
    txt = _joined(seen)
    assert im.size[0] == qc.W_OUT
    assert "FieldPrism anchor (sheet fit): 100.00 px/cm" in txt
    assert "FieldPrism Letter  |  3 markers + 1 inferred" in txt
    assert "CF used for this sheet: 100.20 px/cm -- measured, anchored on the FieldPrism markers" in txt
    assert "megapixel regression" not in txt and "k*sqrt(MP)" not in txt
    _im, seen = _texts(monkeypatch, qc.build_cf_summary_section, None, None, None,
                       anchor_source="fieldprism", fp_sheet={"sheet_status": "unrecognized"})
    txt = _joined(seen)
    assert "CF used for this sheet: none" in txt and "never used on a FieldPrism sheet" in txt
    assert "FieldPrism sheet: unrecognized" in txt


def _gray(h=60, w=400):
    g = np.full((h, w), 200, np.uint8)
    g[:, ::10] = 40
    return g


def _recon_args(withheld=False, fp=False):
    entries = [dict(key="det1", rot=_gray(), x0=5, pxcm=101.0, band=(10, 50), cf=101.0,
                    ruler_class="METRIC_MM", verdict="used"),
               dict(key="det2", rot=_gray(70, 380), x0=3, pxcm=100.1, band=(12, 52), cf=100.1,
                    ruler_class="METRIC_CM", verdict="rejected" if withheld else "used")]
    per_crop = [dict(key="det1", ruler_class="METRIC_MM", cf=101.0, pct_vs_parent=0.4,
                     pct_vs_anchor=1.0, verdict="used"),
                dict(key="det2", ruler_class="METRIC_CM", cf=100.1, pct_vs_parent=-0.4,
                     pct_vs_anchor=0.1, verdict="used", note="agrees")]
    if fp:
        per_crop.append(dict(key="det3", ruler_class="FP", cf=100.0, pct_vs_parent=0.0,
                             pct_vs_anchor=0.0, verdict="used"))
    pr = dict(cf_px_per_cm=None if withheld else 100.5, cf_px_per_cm_measured=100.5,
              method="weighted", spread=0.009, ok=not withheld,
              pct_vs_anchor=None if withheld else 0.5, confidence="low" if withheld else "high",
              confidence_reasons=["two rulers agree"], per_crop=per_crop)
    return entries, pr


def test_recon_fieldprism_wording(monkeypatch) -> None:
    entries, pr = _recon_args(withheld=True, fp=True)
    im, seen = _texts(monkeypatch, qc.build_recon_section, "a.jpg", entries, pr, 100.0,
                      anchor_formula="cf = 0.5*sqrt(MP)", fallback_applied=False,
                      anchor_source="fieldprism")
    txt = _joined(seen)
    assert im.size[0] == qc.W_OUT
    assert "3 ruler crops on this sheet (1 FieldPrism marker)" in txt
    assert "FieldPrism anchor (working frame) 100.00 px/cm" in txt
    assert "NO MEGAPIXEL FALLBACK on a FieldPrism sheet" in txt
    # no MP wording and no red/orange MP strip ("PREDICTED" alone is the bars caption)
    for gone in ("MP anchor", "MP FALLBACK", "MP-PREDICTED", "FALLBACK APPLIED",
                 "FALLBACK NOT APPLIED"):
        assert gone not in txt, gone
    assert "PREDICTED" not in {t for t, _ in seen}          # the watermark of mp_fallback_strip
    # ... which the megapixel path, same inputs, does draw
    _im, mp = _texts(monkeypatch, qc.build_recon_section, "a.jpg", entries, pr, 100.0,
                     anchor_formula="cf = 0.5*sqrt(MP)", fallback_applied=False)
    assert "PREDICTED" in {t for t, _ in mp} and "MP anchor (working frame)" in _joined(mp)
    assert "FieldPrism markers have no strip here" in txt


# --------------------------------------------------------------------------- #
# defaults are pixel-identical to the committed code
# --------------------------------------------------------------------------- #
# The last commit whose qc.py predates FieldPrism support.
_PRE_FP_COMMIT = "26627c4"


@pytest.fixture(scope="module")
def head_qc():
    """qc.py before FieldPrism support, loaded as a separate module in the same package."""
    here = Path(qc.__file__).resolve()
    try:
        src = subprocess.run(["git", "show", f"{_PRE_FP_COMMIT}:./qc.py"], cwd=here.parent,
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git copy of qc.py at {_PRE_FP_COMMIT} not available")
    mod = types.ModuleType("leafmachine3.inference.ruler_lattice._qc_head")
    mod.__package__ = "leafmachine3.inference.ruler_lattice"
    mod.__file__ = str(here)
    exec(compile(src, "qc_head.py", "exec"), mod.__dict__)
    return mod


def _same(a, b) -> None:
    assert a.size == b.size
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("args,kw", [
    ((101.0, 97.5, "k*sqrt(MP)"), {}),
    ((None, 97.5, "k*sqrt(MP)"), {"fallback_applied": True}),
    ((None, 97.5, None), {}),
    ((None, None, None), {}),
])
def test_cf_summary_defaults_unchanged(head_qc, args, kw) -> None:
    old = head_qc.build_cf_summary_section(*args, **kw)
    _same(qc.build_cf_summary_section(*args, **kw), old)
    _same(qc.build_cf_summary_section(*args, **kw, anchor_source="megapixels", fp_sheet=None), old)


@pytest.mark.parametrize("withheld,applied,anchor", [
    (False, False, 100.0), (True, False, 100.0), (True, True, 100.0), (True, False, None)])
def test_recon_defaults_unchanged(head_qc, withheld, applied, anchor) -> None:
    entries, pr = _recon_args(withheld=withheld)
    args = ("a.jpg", entries, pr, anchor)
    kw = dict(anchor_formula="cf = 0.5*sqrt(MP)", fallback_applied=applied)
    old = head_qc.build_recon_section(*args, **kw)
    _same(qc.build_recon_section(*args, **kw), old)
    _same(qc.build_recon_section(*args, **kw, anchor_source="megapixels"), old)
    # the engine's "none" (a sheet with no anchor and no FieldPrism markers) looks the same too
    _same(qc.build_recon_section(*args, **kw, anchor_source="none"), old)
