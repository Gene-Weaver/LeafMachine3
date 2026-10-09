"""Tests for the FieldPrism (FP) marker drawing: :mod:`leafmachine3.reporting.fieldprism_viz`, its
Summary-overlay integration in :func:`leafmachine3.reporting.overlay.build_summary_image`, the
``Overlay_FieldPrism`` export and :class:`FieldPrismStyle`.

The scene is synthetic: FieldPrism markers (3x3 grids of 1 cm cells, BR cell empty) printed black on
white at 60 px/cm, with marker rows shaped like the ``ruler_FP_marker`` DB contract.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from leafmachine3.reporting import fieldprism_viz as fpv
from leafmachine3.reporting import overlay
from leafmachine3.reporting.overlay import build_summary_image
from leafmachine3.reporting.palette import CFScalebarStyle, FieldPrismStyle, OverlayStyle

S = 60.0                                    # px per cm in the synthetic scene
FP_SRC = "measured_from_fieldprism"
_GREEN, _BLACK, _RED, _MAGENTA = (0, 255, 0), (0, 0, 0), (0, 0, 255), (255, 0, 255)   # BGR
_LABEL_BGR = {"TL": (0, 0, 255), "TR": (0, 255, 255), "BL": (255, 255, 255), "C": (255, 255, 0)}
_RAFT_RGB = (1, 2, 3)                      # a raft color nothing else uses
_RELS = {"TL": (0.5, 0.5), "TR": (2.5, 0.5), "C": (1.5, 1.5), "BL": (0.5, 2.5), "BR": (2.5, 2.5)}


def _squares(ox: float, oy: float) -> dict:
    """Square centers of a marker whose top-left corner is at (ox, oy), in px."""
    return {r: (ox + dx * S, oy + dy * S) for r, (dx, dy) in _RELS.items()}


def _marker(did: int, ox: float, oy: float, **kw) -> dict:
    sq = _squares(ox, oy)
    row = {"detection_id": did, "crop_index": did, "det_conf": 0.93,
           "x1": ox - 10, "y1": oy - 10, "x2": ox + 3 * S + 10, "y2": oy + 3 * S + 10,
           "status": "measured", "valid": 1, "verdict": "used", "pxcm": S, "orientation_deg": 0}
    for r in ("TL", "TR", "C", "BL", "BR"):
        row[f"{r.lower()}_x"], row[f"{r.lower()}_y"] = sq[r]
    row.update(kw)
    return row


def _scene(h: int = 1200, w: int = 1600, markers=()) -> np.ndarray:
    """White paper with every marker's four filled cells printed black."""
    img = np.full((h, w, 3), 255, np.uint8)
    for m in markers:
        sq = _squares(m["tl_x"] - 0.5 * S, m["tl_y"] - 0.5 * S)
        for r in ("TL", "TR", "C", "BL"):
            x, y = sq[r]
            img[int(y - S / 2):int(y + S / 2), int(x - S / 2):int(x + S / 2)] = 0
    return img


def _det(m: dict, with_id: bool = True) -> dict:
    d = {"cls_name": "Ruler", "conf": m["det_conf"], "xyxy": (m["x1"], m["y1"], m["x2"], m["y2"]),
         "source": "archival"}
    if with_id:
        d["detection_id"] = m["detection_id"]
    return d


def _sheet(**kw) -> dict:
    row = {"sheet_status": "identified", "sheet_type": "letter", "sheet_label": "Letter",
           "n_fp_detected": 2, "n_fp_used": 2, "n_fp_inferred": 0, "cf_px_per_cm_fp": S,
           "corners_json": None, "page_corners_json": None, "sheet_candidates_json": None}
    row.update(kw)
    return row


def _style(**kw) -> OverlayStyle:
    base = dict(draw_masks=False, draw_cf_banner=False, draw_labels=True, insert_cf_in_rulers=True)
    base.update(kw)
    return OverlayStyle(**base)


def _cfs() -> CFScalebarStyle:
    return CFScalebarStyle(raft_color=_RAFT_RGB, bar_thickness=10, brim=3)


def _match(img: np.ndarray, bgr) -> np.ndarray:
    return np.all(img == np.array(bgr, np.uint8), axis=-1)


def _magentaish(img: np.ndarray) -> np.ndarray:
    """Magenta up to antialiasing or a nearby text shadow: G ~ 0 with strong R and B."""
    b, g, r = (img[..., k].astype(int) for k in range(3))
    return (g < 40) & (r > 150) & (b > 150)


def _window(img: np.ndarray, center, half: float) -> np.ndarray:
    x, y = center
    return img[int(y - half):int(y + half) + 1, int(x - half):int(x + half) + 1]


M1 = _marker(1, 400, 300)
M2 = _marker(2, 1100, 300)
NON_FP = {"detection_id": 9, "cls_name": "Ruler", "conf": 0.88, "xyxy": (300, 900, 1300, 960),
          "source": "archival"}


def _summary(markers=(M1, M2), sheet=None, dets=None, style=None, cf=S, cf_source=FP_SRC, **kw):
    markers = list(markers)
    img = _scene(markers=markers)
    dets = [_det(m) for m in markers] + [NON_FP] if dets is None else dets
    fp = {"sheet": _sheet() if sheet is None else sheet, "markers": markers}
    return img, build_summary_image(img, dets, [], cf, style or _style(), 1.0, cf_style=_cfs(),
                                    cf_source=cf_source, fieldprism=fp, **kw)


# -- FP boxes: not drawn at all (no fill, border, label or raft) --------------------
def test_fp_boxes_get_no_raft_and_no_centered_label(monkeypatch) -> None:
    labels: list[str] = []
    real = overlay._draw_centered_label

    def spy(out, cx, cy, text, *a, **k):
        labels.append(text)
        return real(out, cx, cy, text, *a, **k)

    monkeypatch.setattr(overlay, "_draw_centered_label", spy)
    img, out = _summary()
    assert labels == ["Ruler 0.88"]                      # only the non-FP ruler is labeled
    for m in (M1, M2):
        box = out[int(m["y1"]):int(m["y2"]), int(m["x1"]):int(m["x2"])]
        assert not _match(box, _RAFT_RGB[::-1]).any()    # no raft on a FieldPrism marker
        # no box at all: an empty paper pixel just inside the box corner is untouched (no fill,
        # no border) -- the app-style TL/TR/C/BL labels and BR cell are the FP presentation
        y, x = int(m["y1"]) + 3, int(m["x1"]) + 3
        assert np.array_equal(out[y, x], img[y, x])


def test_fp_boxes_fall_back_to_ordinary_ruler_boxes_without_fieldprism_drawing() -> None:
    """With report.overlay.draw_fieldprism off nothing else shows the markers, so their detector
    boxes are drawn like any other ruler box."""
    img, out = _summary(style=_style(draw_fieldprism=False))
    for m in (M1, M2):
        y, x = int(m["y1"]) + 3, int(m["x1"]) + 3
        assert not np.array_equal(out[y, x], img[y, x])          # the archival fill is back


def test_non_fp_ruler_still_gets_its_raft() -> None:
    _img, out = _summary()
    x1, y1, x2, y2 = (int(v) for v in NON_FP["xyxy"])
    assert _match(out[y1:y2, x1:x2], _RAFT_RGB[::-1]).any()


def test_fp_box_matched_by_coordinates_when_detection_id_is_missing() -> None:
    """A bundle from before overlay_detections carried detection_id still recognizes FP boxes."""
    dets = [_det(m, with_id=False) for m in (M1, M2)]
    _img, out = _summary(dets=dets)
    for m in (M1, M2):
        box = out[int(m["y1"]):int(m["y2"]), int(m["x1"]):int(m["x2"])]
        assert not _match(box, _RAFT_RGB[::-1]).any()
    is_fp = fpv.fp_box_matcher({"markers": [M1]})
    assert is_fp(_det(M1, with_id=False)) and is_fp(_det(M1))
    assert not is_fp(NON_FP) and not is_fp({"xyxy": (0, 0, 5, 5)})


def test_fp_box_matcher_only_matches_archival_ruler_boxes() -> None:
    """detection_id is unique only within a source: a plant box (or an archival box of another
    class) that shares an FP marker's id or coordinates is never an FP marker -- on either path."""
    is_fp = fpv.fp_box_matcher({"markers": [M1]})
    same_id = {"detection_id": M1["detection_id"], "conf": 0.7, "xyxy": (10, 10, 40, 40)}
    same_box = {"conf": 0.7, "xyxy": (M1["x1"], M1["y1"], M1["x2"], M1["y2"])}
    for src, cls in (("plant", "Flower"), ("plant", "Leaf_WHOLE"), ("plant", "Ruler"),
                     ("archival", "Label")):
        assert not is_fp({**same_id, "source": src, "cls_name": cls}), (src, cls)       # id path
        assert not is_fp({**same_box, "source": src, "cls_name": cls}), (src, cls)      # xyxy path
    assert is_fp({**same_id, "source": "archival", "cls_name": "Ruler"})
    # an older bundle without source / cls_name still matches (the keys are not required)
    assert is_fp({"detection_id": M1["detection_id"]}) and is_fp(dict(same_box))


def test_plant_box_sharing_an_fp_id_keeps_its_label(monkeypatch) -> None:
    labels: list[str] = []
    real = overlay._draw_centered_label

    def spy(out, cx, cy, text, *a, **k):
        labels.append(text)
        return real(out, cx, cy, text, *a, **k)

    monkeypatch.setattr(overlay, "_draw_centered_label", spy)
    flower = {"detection_id": M1["detection_id"], "cls_name": "Flower", "conf": 0.77,
              "xyxy": (700.0, 600.0, 900.0, 800.0), "source": "plant"}
    _summary(dets=[_det(M1), _det(M2), NON_FP, flower])
    assert sorted(labels) == ["Flower 0.77", "Ruler 0.88"]       # the FP boxes stay unlabeled


# -- app-style marker labels -------------------------------------------------------
@pytest.mark.parametrize("role", ["TL", "TR", "BL", "C"])
def test_label_colors_sit_on_the_square_centers(role) -> None:
    _img, out = _summary()
    for m in (M1, M2):
        center = (m[f"{role.lower()}_x"], m[f"{role.lower()}_y"])
        win = _window(out, center, 0.3 * S)
        assert _match(win, _LABEL_BGR[role]).sum() > 20, role      # solid glyph pixels, on the cell
        assert not _match(win, _GREEN).any()


def test_br_square_is_green_with_a_black_outline() -> None:
    _img, out = _summary()
    bx, by = int(round(M1["br_x"])), int(round(M1["br_y"]))
    x1, x2 = int(round(M1["br_x"] - S / 2)), int(round(M1["br_x"] + S / 2))
    assert tuple(out[by, bx]) == _GREEN                              # filled
    assert tuple(out[by, x1]) == _BLACK and tuple(out[by, x2 - 1]) == _BLACK   # 1*s outline
    assert tuple(out[by, x1 + 3]) == _GREEN
    assert tuple(out[by, x2 + 2]) != _GREEN                          # exactly 1 cm wide


def test_resolution_line_is_drawn_above_the_tl_label() -> None:
    """``"1 cm = 60 px"`` in white just above TL -- on white paper only its shadow differs, so look
    for the shadow's dark pixels in the band above the TL cell."""
    img, out = _summary(markers=[M2], dets=[_det(M2)])
    band = (slice(int(M2["tl_y"] - 2.2 * S), int(M2["tl_y"] - S / 2)),
            slice(int(M2["tl_x"] - 1.5 * S), int(M2["tl_x"] + 3 * S)))
    assert (out[band].astype(int).sum(-1) < 600).sum() > 50


def _rot_squares(ox: float, oy: float, k: int) -> dict:
    """Square centers of a marker on a sheet photographed rotated (``np.rot90(img, k)``: k quarter
    turns counterclockwise), occupying the 3x3-cell block whose top-left corner is (ox, oy)."""
    out = {}
    for r, (dx, dy) in _RELS.items():
        for _ in range(k):
            dx, dy = dy, 3.0 - dx                  # one CCW quarter turn inside the 3 x 3 block
        out[r] = (ox + dx * S, oy + dy * S)
    return out


def _rot_marker(did: int, ox: float, oy: float, k: int, **kw) -> dict:
    row = _marker(did, ox, oy, orientation_deg=90 * k, **kw)
    for r, (x, y) in _rot_squares(ox, oy, k).items():
        row[f"{r.lower()}_x"], row[f"{r.lower()}_y"] = x, y
    return row


def _text_spy(monkeypatch) -> list:
    """Record (text, org, ink box) for every draw_text call that paints."""
    calls: list = []
    real = fpv.draw_text

    def spy(out, text, org, *a, **k):
        ink = real(out, text, org, *a, **k)
        if not k.get("measure_only"):
            calls.append((text, tuple(org), ink))
        return ink

    monkeypatch.setattr(fpv, "draw_text", spy)
    return calls


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def test_rotated_squares_helper_matches_np_rot90() -> None:
    """k=1 puts TL bottom-left and the empty BR top-right, as on np.rot90 of an upright sheet."""
    sq = _rot_squares(0, 0, 1)
    assert sq["TL"] == (0.5 * S, 2.5 * S) and sq["BR"] == (2.5 * S, 0.5 * S)
    assert _rot_squares(0, 0, 2)["TL"] == (2.5 * S, 2.5 * S)
    assert _rot_squares(0, 0, 4) == _squares(0, 0)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_resolution_line_stays_above_a_rotated_marker(monkeypatch, k) -> None:
    """On a sheet at 90/180/270 degrees the "1 cm =" line goes above the marker's on-image top row,
    never across its middle row (over the C label); upright it is exactly the app's placement."""
    calls = _text_spy(monkeypatch)
    m = _rot_marker(2, 1100, 300, k)
    _img, out = _summary(markers=[m], dets=[_det(m)])
    line = [c for c in calls if c[0] == "1 cm = 60 px"]
    assert len(line) == 1
    _t, (x, y), ink = line[0]
    for role in ("TL", "TR", "BL", "C"):
        lab = [c for c in calls if c[0] == role]
        assert len(lab) == 1 and not _overlap(ink, lab[0][2]), (k, role)
    assert y <= 300 + 1                                      # baseline above the marker's top edge
    assert ink[0] < 1100 + 3 * S and ink[2] > 1100           # and over it, not off to a side
    if k == 0:                                               # the app: x = xTL - w/4, yTL - h(TL) - 5 s
        w, _h = fpv._ink_size("1 cm = 60 px", 0.70 * S)
        s = 1600 / 2592
        assert x == pytest.approx(m["tl_x"] - w / 4.0)
        assert y == pytest.approx(m["tl_y"] - fpv._ink_size("TL", 0.70 * S)[1] - 5.0 * s)


def test_upright_resolution_line_is_unchanged_by_the_rotation_rule() -> None:
    """The anchor is the on-image top-left cell, which is TL for an upright marker -- also when it
    is slightly tilted -- so the upright drawing is the app's TL-anchored one, pixel for pixel."""
    tilt = math.radians(4.0)
    tl = (1100.0 + 0.5 * S, 300.0 + 0.5 * S)
    sq = {}
    for r, (dx, dy) in _RELS.items():
        ux, uy = (dx - 0.5) * S, (dy - 0.5) * S
        sq[r] = (tl[0] + ux * math.cos(tilt) + uy * math.sin(tilt),
                 tl[1] - ux * math.sin(tilt) + uy * math.cos(tilt))
    assert fpv._top_left_cell(sq) == sq["TL"]
    for k in (1, 2, 3):
        rs = _rot_squares(0, 0, k)
        assert fpv._top_left_cell(rs) == min(rs.values(), key=lambda p: p[0] + p[1])
        assert fpv._top_left_cell(rs) == (0.5 * S, 0.5 * S)


def test_line_moved_below_a_180_degree_marker_clears_all_its_cells(monkeypatch) -> None:
    """A marker at the image's top edge has no room above it, so its line goes under it. "Under"
    is below ALL five cells: at 180 degrees BL/BR are the TOP row, and the old BL/BR-based bottom
    put the line across the marker's middle."""
    calls = _text_spy(monkeypatch)
    m = _rot_marker(2, 1100, 4, 2)
    _img, out = _summary(markers=[m], dets=[_det(m)])
    _t, _org, ink = [c for c in calls if c[0] == "1 cm = 60 px"][0]
    assert ink[1] > 4 + 3 * S                                # below the marker's bottom edge
    for role in ("TL", "TR", "BL", "C"):
        assert not _overlap(ink, [c for c in calls if c[0] == role][0][2]), role


@pytest.mark.parametrize("k", [1, 2, 3])
def test_inferred_line_stays_above_a_rotated_reconstructed_marker(monkeypatch, k) -> None:
    calls = _text_spy(monkeypatch)
    sq = _rot_squares(400, 700, k)
    corners = {"BL": {"observed": False, "detection_id": None,
                      "squares": {r: list(v) for r, v in sq.items()}}}
    fp = {"sheet": _sheet(n_fp_inferred=1, corners_json=json.dumps(corners)), "markers": []}
    fpv.draw_fp_markers(_scene(), fp, FieldPrismStyle())
    _t, (_x, y), ink = [c for c in calls if c[0] == "inferred"][0]
    assert y <= 700 + 1, k
    for role in ("TL", "TR", "BL", "C"):
        assert not _overlap(ink, [c for c in calls if c[0] == role][0][2]), (k, role)


def test_line_that_would_cover_its_own_labels_moves_below(monkeypatch) -> None:
    """Belt and braces: whatever the geometry, the line never paints over its own marker's labels."""
    calls = _text_spy(monkeypatch)
    m = _marker(2, 1100, 300)
    m["c_x"], m["c_y"] = m["tl_x"] + S, m["tl_y"] - 0.9 * S       # a C label right where the line goes
    fpv.draw_fp_markers(_scene(), {"sheet": None, "markers": [m]}, FieldPrismStyle())
    _t, _org, ink = [c for c in calls if c[0] == "1 cm = 60 px"][0]
    assert not _overlap(ink, [c for c in calls if c[0] == "C"][0][2])
    assert ink[1] > 300 + 3 * S                                   # moved under the marker


def test_rejected_marker_gets_a_red_outline_and_a_note(monkeypatch) -> None:
    texts: list[str] = []
    real = fpv.draw_text

    def spy(out, text, *a, **k):
        texts.append(text)
        return real(out, text, *a, **k)

    monkeypatch.setattr(fpv, "draw_text", spy)
    rej = _marker(1, 400, 300, verdict="rejected")
    _img, out = _summary(markers=[rej, M2])
    bx, by = int(round(rej["br_x"])), int(round(rej["br_y"]))
    assert tuple(out[by, bx]) != _GREEN                              # no fill
    assert tuple(out[by, int(round(rej["br_x"] - S / 2))]) == _RED  # red outline
    assert "rejected" in texts
    assert texts.count("TL") == 2                                    # labels are still drawn


def test_failed_and_invalid_markers_draw_nothing() -> None:
    """Their roles are unreliable, so no labels; and FP detector boxes are never drawn on the
    Summary overlay, so a failed marker leaves the paper untouched."""
    failed = {k: v for k, v in _marker(1, 400, 300, status="failed", verdict="skipped").items()
              if not k.startswith(("tl_", "tr_", "c_", "bl_", "br_"))}
    invalid = _marker(2, 1100, 300, valid=0, verdict="skipped")
    img = _scene(markers=[invalid])
    fp = {"sheet": _sheet(), "markers": [failed, invalid]}
    style = _style(draw_labels=False, insert_cf_in_rulers=False)
    plain = build_summary_image(img, [], [], S, style, 1.0,
                                cf_source=FP_SRC, fieldprism={"sheet": _sheet(), "markers": []})
    out = build_summary_image(img, [_det(failed), _det(invalid)], [], S, style, 1.0,
                              cf_source=FP_SRC, fieldprism=fp)
    assert np.array_equal(out, plain)               # same badge; no box, no marker pixel
    for m in (failed, invalid):
        y1, y2, x1, x2 = int(m["y1"]), int(m["y2"]), int(m["x1"]), int(m["x2"])
        assert not _match(out[y1:y2, x1:x2], _GREEN).any()


# -- reconstructed (inferred) markers ----------------------------------------------
def _inferred_sheet(ox: float = 400, oy: float = 700) -> dict:
    sq = _squares(ox, oy)
    corners = {"TL": {"observed": True, "detection_id": 1, "squares": _squares(400, 300)},
               "TR": {"observed": True, "detection_id": 2, "squares": _squares(1100, 300)},
               "BL": {"observed": False, "detection_id": None,
                      "squares": {k: list(v) for k, v in sq.items()}}}
    return _sheet(n_fp_inferred=1, corners_json=json.dumps(corners))


def test_inferred_markers_are_drawn_dashed_magenta_with_dimmed_labels(monkeypatch) -> None:
    alphas: dict[str, list[float]] = {}
    texts: list[str] = []
    real = fpv.draw_text

    def spy(out, text, *a, alpha=1.0, **k):
        alphas.setdefault(text, []).append(alpha)
        texts.append(text)
        return real(out, text, *a, alpha=alpha, **k)

    monkeypatch.setattr(fpv, "draw_text", spy)
    _img, out = _summary(sheet=_inferred_sheet())
    sq = _squares(400, 700)
    for role in ("TL", "TR", "C", "BL", "BR"):
        x, y = sq[role]
        edge = out[int(y + S / 2) - 2:int(y + S / 2) + 1, int(x - S / 2):int(x + S / 2)]   # bottom edge
        n = int(_magentaish(edge).any(axis=0).sum())
        assert S / 3 <= n < S - 5, (role, n)                         # dashed: some columns, not all
    assert "inferred" in texts
    assert min(alphas["TL"]) == pytest.approx(FieldPrismStyle().inferred_label_alpha)
    assert max(alphas["TL"]) == 1.0                                  # the measured ones stay opaque


def test_inferred_markers_can_be_left_out() -> None:
    blank = _scene(markers=[M1, M2])
    only = blank.copy()
    fpv.draw_fp_markers(only, {"sheet": _inferred_sheet(), "markers": []}, FieldPrismStyle(),
                        draw_inferred=False)
    assert np.array_equal(only, blank)


# -- the sheet badge and the corner raft --------------------------------------------
def _banner_rows(out: np.ndarray, color_rgb) -> np.ndarray:
    return np.where(_match(out[:, :2], color_rgb[::-1]).all(axis=1))[0]


def test_badge_sits_right_of_the_corner_raft_and_below_the_banner() -> None:
    banner_rgb = (255, 200, 0)
    style = _style(draw_cf_banner=True, cf_banner_color=banner_rgb)
    _img, out = _summary(style=style)
    banner_bottom = int(_banner_rows(out, banner_rgb).max()) + 1
    raft = np.argwhere(_match(out[:200], _RAFT_RGB[::-1]))
    assert raft[:, 0].min() == banner_bottom and raft[:, 1].min() == 0     # corner raft under the banner
    raft_right = int(raft[:, 1].max()) + 1
    assert raft_right == round(S * 2.54) + 2 * 3                            # 1 inch bar + brims
    mag = np.argwhere(_match(out[:200], _MAGENTA))
    assert len(mag) > 50
    assert mag[:, 1].min() > raft_right                                     # immediately right of it
    assert mag[:, 1].min() < raft_right + 40
    assert mag[:, 0].min() >= banner_bottom                                 # never on the banner
    # the badge box: black at alpha 0.6 over the white paper, top-aligned with the raft
    gap = max(4, round(8 * 1600 / 2592))
    assert tuple(out[banner_bottom + 1, raft_right + gap + 1]) == (102, 102, 102)


def test_badge_goes_top_left_under_the_banner_without_a_raft() -> None:
    banner_rgb = (255, 200, 0)
    style = _style(draw_cf_banner=True, cf_banner_color=banner_rgb, insert_cf_in_rulers=False)
    _img, out = _summary(style=style)
    banner_bottom = int(_banner_rows(out, banner_rgb).max()) + 1
    assert tuple(out[banner_bottom + 1, 1]) == (102, 102, 102)
    mag = np.argwhere(_match(out[:200], _MAGENTA))
    assert mag[:, 0].min() >= banner_bottom and mag[:, 1].min() < 30


def test_badge_at_the_very_top_without_any_cf() -> None:
    _img, out = _summary(cf=None, cf_source=None)
    assert tuple(out[1, 1]) == (102, 102, 102)
    assert _match(out[:100, :600], _MAGENTA).any()
    assert not _match(out, _RAFT_RGB[::-1]).any()


def test_banner_names_fieldprism(monkeypatch) -> None:
    seen: list[str] = []
    real = overlay.cv2.putText

    def spy(img, text, *a, **k):
        seen.append(text)
        return real(img, text, *a, **k)

    monkeypatch.setattr(overlay.cv2, "putText", spy)
    _summary(style=_style(draw_cf_banner=True, draw_labels=False))
    assert seen == ["CF: 60.00 px/cm (measured from FieldPrism)"]


def test_badge_text_variants() -> None:
    four = {c: {"observed": True, "squares": {}} for c in ("TL", "TR", "BL", "BR")}
    three = dict(four, BR={"observed": False, "squares": {"TL": [0, 0]}})
    assert fpv.fp_badge_text(_sheet(corners_json=json.dumps(four))) == "FieldPrism Letter | 4 markers"
    assert fpv.fp_badge_text(_sheet(corners_json=json.dumps(three))) == "FieldPrism Letter | 3 + 1 inferred"
    # without stored corners the counts columns are used
    assert fpv.fp_badge_text(_sheet(n_fp_used=3, n_fp_inferred=1)) == "FieldPrism Letter | 3 + 1 inferred"
    cands = [{"sheet_type": "letter", "cost_mm": 0.8}, {"sheet_type": "legal", "cost_mm": 1.2},
             {"sheet_type": "letter", "cost_mm": 1.3}, {"sheet_type": "a4", "cost_mm": 2.5}]
    amb = _sheet(sheet_status="ambiguous", sheet_candidates_json=json.dumps(cands))
    assert fpv.fp_badge_text(amb) == "FieldPrism Letter? (or Legal)"       # A4 is outside the margin
    assert fpv.fp_badge_text(_sheet(sheet_status="unrecognized", sheet_type=None, sheet_label=None)) \
        == "FieldPrism sheet: unrecognized"
    assert fpv.fp_badge_text(_sheet(sheet_status="undetermined", sheet_type=None, sheet_label=None,
                                    n_fp_used=1)) == "FieldPrism sheet: undetermined | 1 marker"
    assert fpv.fp_badge_text(_sheet(sheet_label=None, sheet_type="a4", n_fp_used=2)) \
        == "FieldPrism A4 | 2 markers"
    assert fpv.fp_badge_text(None) is None
    assert fpv.fp_badge_text(None, [M1, M2]) == "FieldPrism sheet: unknown | 2 markers"


# -- old behavior is untouched -----------------------------------------------------
def _legacy_scene():
    img = _scene(markers=[M1, M2])
    dets = [_det(M1), _det(M2), NON_FP]
    return img, dets


@pytest.mark.parametrize("cf_source", [FP_SRC, "measured_from_ruler", "predicted_from_megapixels"])
def test_no_fieldprism_data_is_byte_identical(cf_source) -> None:
    img, dets = _legacy_scene()
    style = _style(draw_cf_banner=True, insert_cf_exterior=True)
    ref = build_summary_image(img, dets, [], S, style, 1.0, cf_style=_cfs(), cf_source=cf_source)
    for fp in (None, {"sheet": None, "markers": []}):
        out = build_summary_image(img, dets, [], S, style, 1.0, cf_style=_cfs(), cf_source=cf_source,
                                  fieldprism=fp, fp_style=FieldPrismStyle())
        assert np.array_equal(out, ref)
    # draw_fieldprism off: FP data present but ignored -- the pre-FieldPrism look
    off = OverlayStyle(**{**style.__dict__, "draw_fieldprism": False})
    out = build_summary_image(img, dets, [], S, off, 1.0, cf_style=_cfs(), cf_source=cf_source,
                              fieldprism={"sheet": _sheet(), "markers": [M1, M2]})
    assert np.array_equal(out, ref)


def test_fieldprism_from_record_handles_old_records() -> None:
    assert fpv.fieldprism_from_record(None) is None
    assert fpv.fieldprism_from_record({"image": {}, "crops": []}) is None          # pre-FP record
    assert fpv.fieldprism_from_record({"image": {}, "crops": [], "fp_sheet": None, "fp_markers": []}) is None
    got = fpv.fieldprism_from_record({"fp_sheet": _sheet(), "fp_markers": [M1]})
    assert got["sheet"]["sheet_type"] == "letter" and got["markers"][0]["detection_id"] == 1
    assert fpv.has_fp_markers(got)
    assert fpv.has_fp_markers({"sheet": _sheet(n_fp_detected=1), "markers": []})
    assert not fpv.has_fp_markers({"sheet": _sheet(n_fp_detected=0), "markers": []})


def test_exterior_ring_still_wraps_an_fp_sheet() -> None:
    """The ring is appended after every FP drawing, so the badge and markers shift with the sheet."""
    margin = round(2 * S)
    style = _style(insert_cf_exterior=True)
    _img, out = _summary(style=style)
    _img2, inner = _summary()
    assert out.shape == (1200 + margin, 1600 + margin, 3)
    assert np.array_equal(out[margin:, margin:], inner)
    assert out[1, 1].tolist() == [255, 255, 255] and out[1, int(round(S)) + 1].tolist() == [0, 0, 0]


# -- Overlay_FieldPrism ------------------------------------------------------------
def test_fieldprism_overlay_draws_legend_badge_markers_and_page_outline() -> None:
    img = _scene(markers=[M1, M2])
    before = img.copy()
    page = [[100, 120], [1500, 120], [1500, 1150], [100, 1150]]
    fp = {"sheet": _inferred_sheet() | {"page_corners_json": json.dumps(page)}, "markers": [M1, M2]}
    out = fpv.build_fieldprism_overlay(img, fp, cf_px_per_cm=S)
    assert np.array_equal(img, before) and out.shape == img.shape
    s = 1600 / 2592
    assert tuple(out[int(20 * s) + 6, int(20 * s) + 6]) == (105, 105, 105)    # legend: alpha 150
    assert tuple(out[int(M1["br_y"]), int(M1["br_x"])]) == _GREEN
    assert _magentaish(out[1148:1153, 300:1300]).any(axis=0).all()            # page outline
    assert _match(out[:120, :], _MAGENTA).sum() > 50                          # the badge text


def test_fieldprism_overlay_badge_moves_right_of_the_legend() -> None:
    img = _scene(markers=[M1, M2])
    fp = {"sheet": _sheet(), "markers": [M1, M2]}
    out = fpv.build_fieldprism_overlay(img, fp, cf_px_per_cm=S)
    legend = fpv.fp_legend_box(out, S, FieldPrismStyle())
    mag = np.argwhere(_match(out, _MAGENTA))
    assert mag[:, 1].min() > legend[2] and mag[:, 0].max() < legend[3] + 5


def test_reporter_writes_overlay_fieldprism(tmp_path: Path) -> None:
    from leafmachine3.modules.reporter import Reporter

    img = _scene(markers=[M1, M2])
    b = SimpleNamespace(working_path="w.jpg", cf_px_per_cm=S, specimen_id=1)
    fp = {"sheet": _sheet(), "markers": [M1, M2]}
    got = Reporter._export_fieldprism(None, b, fp, FieldPrismStyle(), tmp_path, "sheet7", "jpg", 95,
                                      lambda _p: img)
    path = tmp_path / "Overlay" / "Overlay_FieldPrism" / "sheet7__FieldPrism.jpg"
    assert got == [(str(path), "Overlay/Overlay_FieldPrism")] and path.exists()
    # a broken record is logged, never raised
    bad = {"sheet": {"corners_json": "{not json"}, "markers": [{"status": "measured", "tl_x": "x"}]}
    assert isinstance(Reporter._export_fieldprism(None, b, bad, FieldPrismStyle(), tmp_path, "s8",
                                                  "jpg", 95, lambda _p: img), list)


# -- FieldPrismStyle -----------------------------------------------------------------
def test_fieldprism_style_defaults_are_the_app_colors(tmp_path: Path) -> None:
    from leafmachine3.core.config import Config

    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"report": {"overlay": {}}}), encoding="utf-8")
    st = FieldPrismStyle.from_config(Config.load(path))
    assert (st.tl, st.tr, st.bl, st.c) == ((255, 0, 0), (255, 255, 0), (255, 255, 255), (0, 255, 255))
    assert (st.br_used, st.br_rejected, st.inferred, st.badge_text) == (
        (0, 255, 0), (255, 0, 0), (255, 0, 255), (255, 0, 255))
    assert st.badge_alpha == 0.6 and st.text_size_frac == 0.70
    path.write_text(yaml.safe_dump({"report": {"overlay": {
        "draw_fieldprism": False, "fieldprism": {"tl": [1, 2, 3], "text_size_frac": 0.5}}}}),
        encoding="utf-8")
    cfg = Config.load(path)
    st = FieldPrismStyle.from_config(cfg)
    assert st.tl == (1, 2, 3) and st.text_size_frac == 0.5 and st.tr == (255, 255, 0)
    assert OverlayStyle.from_config(cfg).draw_fieldprism is False
    assert OverlayStyle().draw_fieldprism is True
