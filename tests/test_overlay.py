"""Tests for :mod:`leafmachine3.reporting.overlay` -- Summary_Image rendering."""
from __future__ import annotations

import numpy as np
import pytest

from leafmachine3.core.imaging import encode_polygon
from leafmachine3.reporting.overlay import _res_ratio, _scaled_lw, build_summary_image
from leafmachine3.reporting.palette import GroupStyle, OverlayStyle


def _blank(h: int = 300, w: int = 400) -> np.ndarray:
    return np.full((h, w, 3), 220, dtype=np.uint8)


def _leaf_row(cls_name: str, poly) -> dict:
    return {"cls_name": cls_name, "mask_format": "polygon_xy", "mask_data": encode_polygon(poly)}


def test_returns_same_shape_ndarray() -> None:
    img = _blank()
    detections = [
        {"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 120, 40), "source": "archival"},
        {"cls_name": "Leaf_WHOLE", "conf": 0.8, "xyxy": (60, 80, 260, 250), "source": "plant"},
    ]
    leaves = [_leaf_row("Leaf", [[80, 100], [230, 110], [220, 240], [90, 235]])]
    out = build_summary_image(img, detections, leaves, cf_px_per_cm=None,
                              style=OverlayStyle(), work_scale=1.0)
    assert isinstance(out, np.ndarray)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_does_not_mutate_input() -> None:
    img = _blank()
    before = img.copy()
    build_summary_image(
        img,
        [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 120, 40), "source": "archival"}],
        [],
        cf_px_per_cm=96.0,
        style=OverlayStyle(),
        work_scale=1.0,
    )
    assert np.array_equal(img, before)                      # drew on a copy


def test_actually_draws_something() -> None:
    img = _blank()
    detections = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 120, 40), "source": "archival"}]
    out = build_summary_image(img, detections, [], cf_px_per_cm=96.0,
                              style=OverlayStyle(), work_scale=1.0)
    assert not np.array_equal(out, img)                     # box + CF banner changed pixels


def test_work_scale_upscales_geometry() -> None:
    """With work_scale 0.5 the stored coords map onto a 2x-larger original frame."""
    img = _blank(600, 800)
    detections = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (10, 10, 100, 100), "source": "archival"}]
    out = build_summary_image(img, detections, [], cf_px_per_cm=None,
                              style=OverlayStyle(), work_scale=0.5)
    assert out.shape == img.shape                           # renders without going out of bounds


def test_group_split_archival_fills_plant_borders() -> None:
    """The core redesign contract: archival boxes are FILLED (no border), plant boxes are
    BORDERED (no fill). Checks interior vs border pixels, not just 'something drew'."""
    img = _blank(400, 400)
    detections = [
        {"cls_name": "Ruler", "conf": 0.9, "xyxy": (20, 20, 180, 180), "source": "archival"},
        {"cls_name": "Flower_ONE", "conf": 0.9, "xyxy": (220, 20, 380, 180), "source": "plant"},
    ]
    # labels/banner off so the only pixels that change are the box fill/border themselves
    style = OverlayStyle(draw_labels=False, draw_cf_banner=False)
    out = build_summary_image(img, detections, [], cf_px_per_cm=None, style=style, work_scale=1.0)
    bg = np.array([220, 220, 220], np.uint8)

    # archival Ruler: interior center is FILLED (differs from background)
    assert not np.array_equal(out[100, 100], bg)
    # plant Flower_ONE: interior center is NOT filled (equals background)...
    assert np.array_equal(out[100, 300], bg)
    # ...but its border IS drawn (a pixel on the top edge differs from background)
    assert not np.array_equal(out[20, 300], bg)


def test_group_border_and_fill_toggle_independently() -> None:
    """Turning a group's fill off (and it had no border) makes it draw nothing; adding border draws
    only the outline. Exercises border/fill independence via the config."""
    img = _blank(400, 400)
    det = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (20, 20, 180, 180), "source": "archival"}]
    bg = np.array([220, 220, 220], np.uint8)

    off = OverlayStyle(draw_labels=False, draw_cf_banner=False,
                       groups={"archival": GroupStyle(border=False, fill=False)})
    out_off = build_summary_image(img, det, [], cf_px_per_cm=None, style=off, work_scale=1.0)
    assert np.array_equal(out_off[100, 100], bg) and np.array_equal(out_off[20, 100], bg)

    border_only = OverlayStyle(draw_labels=False, draw_cf_banner=False,
                               groups={"archival": GroupStyle(border=True, fill=False)})
    out_b = build_summary_image(img, det, [], cf_px_per_cm=None, style=border_only, work_scale=1.0)
    assert np.array_equal(out_b[100, 100], bg)              # interior still background (no fill)
    assert not np.array_equal(out_b[20, 100], bg)           # border drawn


def test_res_ratio_and_scaled_lw() -> None:
    """Resolution-relative anchors: _res_ratio is 1.0 at the 2592 reference long side and scales
    linearly; _scaled_lw scales a configured width and never drops to 0."""
    assert _res_ratio(np.empty((2592, 1944, 3), np.uint8)) == 1.0
    assert _res_ratio(np.empty((1944, 2592, 3), np.uint8)) == 1.0     # long side, orientation-agnostic
    assert _res_ratio(np.empty((5184, 2000, 3), np.uint8)) == 2.0     # 2x the reference
    assert _scaled_lw(3, 1.0) == 3
    assert _scaled_lw(3, 2.0) == 6
    assert _scaled_lw(3, 0.1) >= 1                                    # never rounds down to 0


# -- CF scale overlays (insert_cf_in_rulers / insert_cf_exterior) -------------------
_CYAN_BGR, _GREEN_BGR, _WHITE_BGR, _BLACK_BGR = (255, 255, 0), (0, 255, 0), (255, 255, 255), (0, 0, 0)


def _cf_style(**kw):
    from leafmachine3.reporting.palette import CFScalebarStyle

    return CFScalebarStyle(**kw)


def _plain(**flags) -> OverlayStyle:
    """An OverlayStyle drawing nothing but what the flags turn on."""
    return OverlayStyle(draw_labels=False, draw_cf_banner=False, draw_masks=False, **flags)


def _run(color, line) -> tuple[int, int]:
    """(start, length) of the pixels matching ``color`` along a 1-D strip of BGR pixels."""
    idx = np.flatnonzero(np.all(line == np.array(color, np.uint8), axis=-1))
    return (int(idx.min()), int(len(idx))) if len(idx) else (-1, 0)


def test_ruler_bars_are_exactly_one_cm_and_one_inch() -> None:
    """The whole point of the feature: the bars measure 1 cm and 1 inch at the sheet's CF."""
    img = np.full((400, 600, 3), 128, np.uint8)
    det = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (50, 200, 350, 240), "source": "archival"}]
    out = build_summary_image(img, det, [], cf_px_per_cm=40.0, style=_plain(insert_cf_in_rulers=True),
                              work_scale=1.0, cf_style=_cf_style(bar_thickness=10, brim=3))
    ys, xs = np.where(np.all(out == np.array(_CYAN_BGR, np.uint8), axis=-1))
    cm_start, cm_len = _run(_CYAN_BGR, out[ys.min()])
    inch_start, inch_len = _run(_GREEN_BGR, out[np.where(
        np.all(out == np.array(_GREEN_BGR, np.uint8), axis=-1))[0].min()])
    assert (cm_len, inch_len) == (40, round(40 * 2.54))       # 1 cm, 1 inch -- to the pixel
    assert cm_start == inch_start == 50 + 3                   # flush with the ruler's left edge + brim


def test_ruler_bars_follow_work_scale_into_the_original_frame() -> None:
    """cf_px_per_cm is a WORKING-frame value; on a 2x original a cm must be 2x as many pixels."""
    img = np.full((800, 1200, 3), 128, np.uint8)
    det = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (50, 200, 350, 240), "source": "archival"}]
    out = build_summary_image(img, det, [], cf_px_per_cm=40.0, style=_plain(insert_cf_in_rulers=True),
                              work_scale=0.5, cf_style=_cf_style())
    ys = np.where(np.all(out == np.array(_CYAN_BGR, np.uint8), axis=-1))[0]
    assert _run(_CYAN_BGR, out[ys.min()])[1] == 80            # 40 working px/cm -> 80 original px/cm


def test_ruler_bars_run_along_the_long_axis_of_a_portrait_ruler() -> None:
    """A tall ruler gets vertical bars flush with its TOP edge, centered across its width."""
    img = np.full((400, 600, 3), 128, np.uint8)
    det = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (450, 40, 490, 340), "source": "archival"}]
    out = build_summary_image(img, det, [], cf_px_per_cm=40.0, style=_plain(insert_cf_in_rulers=True),
                              work_scale=1.0, cf_style=_cf_style(bar_thickness=10, brim=3))
    xs = np.where(np.all(out == np.array(_CYAN_BGR, np.uint8), axis=-1))[1]
    start, length = _run(_CYAN_BGR, out[:, xs.min()])
    assert length == 40 and start == 40 + 3                   # runs DOWN, flush with the top edge
    raft_xs = np.where(np.all(out == np.array(_WHITE_BGR, np.uint8), axis=-1))[1]
    assert abs((int(raft_xs.min()) + int(raft_xs.max()) + 1) / 2 - 470) <= 1   # centered on the ruler


def test_ruler_bars_only_touch_rulers_and_only_when_enabled() -> None:
    img = np.full((400, 600, 3), 128, np.uint8)
    det = [{"cls_name": "Label", "conf": 0.9, "xyxy": (50, 200, 350, 240), "source": "archival"},
           {"cls_name": "Ruler", "conf": 0.9, "xyxy": (50, 20, 350, 60), "source": "plant"}]
    out = build_summary_image(img, det, [], cf_px_per_cm=40.0, style=_plain(insert_cf_in_rulers=True),
                              work_scale=1.0, cf_style=_cf_style())
    assert not np.any(np.all(out == np.array(_CYAN_BGR, np.uint8), axis=-1))

    ruler = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (50, 200, 350, 240), "source": "archival"}]
    off = build_summary_image(img, ruler, [], cf_px_per_cm=40.0, style=_plain(), work_scale=1.0)
    assert not np.any(np.all(off == np.array(_CYAN_BGR, np.uint8), axis=-1))   # flag off -> no raft
    no_cf = build_summary_image(img, ruler, [], cf_px_per_cm=None,
                                style=_plain(insert_cf_in_rulers=True), work_scale=1.0)
    assert not np.any(np.all(no_cf == np.array(_CYAN_BGR, np.uint8), axis=-1))  # no CF, nothing to scale


def test_exterior_ring_appends_a_margin_without_touching_the_sheet() -> None:
    img = np.full((300, 500, 3), 128, np.uint8)
    out = build_summary_image(img, [], [], cf_px_per_cm=37.0, style=_plain(insert_cf_exterior=True),
                              work_scale=1.0, cf_style=_cf_style(exterior_cells=2))
    margin = round(2 * 37.0)
    assert out.shape == (300 + margin, 500 + margin, 3)
    assert np.array_equal(out[margin:, margin:], img)         # every original pixel is untouched


def test_exterior_ring_is_one_checkerboard_across_both_bands() -> None:
    """Rows read OXOXOX / XOXOXO and the left band continues the SAME parity, so the shared
    top-left corner agrees with itself instead of showing a seam."""
    cf = 37.0
    img = np.full((300, 500, 3), 128, np.uint8)
    out = build_summary_image(img, [], [], cf_px_per_cm=cf, style=_plain(insert_cf_exterior=True),
                              work_scale=1.0, cf_style=_cf_style(exterior_cells=2))

    def cell(i: int, j: int) -> str:
        px = out[int(round(i * cf)) + 1, int(round(j * cf)) + 1].tolist()
        return "X" if px == list(_BLACK_BGR) else "O"

    assert "".join(cell(0, j) for j in range(8)) == "OXOXOXOX"
    assert "".join(cell(1, j) for j in range(8)) == "XOXOXOXO"
    assert "".join(cell(i, 0) for i in range(8)) == "OXOXOXOX"   # left band, same parity
    assert "".join(cell(i, 1) for i in range(8)) == "XOXOXOXO"


def test_exterior_cell_edges_do_not_accumulate_rounding_error() -> None:
    """Boundaries are round(k*cf), so the 15th cell still starts at the true 15 cm mark -- a
    per-cell round(cf) would have drifted 15 * 0.4 px by then."""
    cf = 37.4
    img = np.full((200, 900, 3), 128, np.uint8)
    out = build_summary_image(img, [], [], cf_px_per_cm=cf, style=_plain(insert_cf_exterior=True),
                              work_scale=1.0, cf_style=_cf_style(exterior_cells=2))
    row = np.all(out[1] == np.array(_BLACK_BGR, np.uint8), axis=-1)
    starts = [int(round(k * cf)) for k in range(1, 16, 2)]     # the dark cells of row 0
    assert all(row[s] and not row[s - 1] for s in starts)


def test_exterior_ring_clips_a_dark_cell_and_never_a_light_one() -> None:
    """The far edge must cut a BLACK square short; where parity puts a light cell against the
    boundary nothing is drawn, so no light square ever renders truncated."""
    cf = 37.0
    img = np.full((300, 500, 3), 128, np.uint8)
    out = build_summary_image(img, [], [], cf_px_per_cm=cf, style=_plain(insert_cf_exterior=True),
                              work_scale=1.0, cf_style=_cf_style(exterior_cells=2))
    new_h, new_w = out.shape[:2]
    last_j = max(j for j in range(new_w) if round(j * cf) < new_w)
    dark_row, light_row = (0, 1) if (last_j % 2) else (1, 0)
    assert out[int(round(dark_row * cf)) + 1, new_w - 1].tolist() == list(_BLACK_BGR)
    assert out[int(round(light_row * cf)) + 1, new_w - 1].tolist() == list(_WHITE_BGR)
    last_i = max(i for i in range(new_h) if round(i * cf) < new_h)
    dark_col, light_col = (0, 1) if (last_i % 2) else (1, 0)
    assert out[new_h - 1, int(round(dark_col * cf)) + 1].tolist() == list(_BLACK_BGR)
    assert out[new_h - 1, int(round(light_col * cf)) + 1].tolist() == list(_WHITE_BGR)


# -- a CF PREDICTED from megapixels (cf_source = predicted_from_megapixels) --------
_GRAY_BGR = (128, 128, 128)
_PRED = "predicted_from_megapixels"


def test_predicted_cf_ring_is_black_and_gray_not_black_and_white() -> None:
    """Same checkerboard geometry, but light cells are 50% gray so it can't pass for a measured scale."""
    cf = 37.0
    img = np.full((300, 500, 3), 60, np.uint8)
    out = build_summary_image(img, [], [], cf_px_per_cm=cf, style=_plain(insert_cf_exterior=True),
                              work_scale=1.0, cf_style=_cf_style(exterior_cells=2), cf_source=_PRED)
    margin = round(2 * cf)
    assert out.shape == (300 + margin, 500 + margin, 3)
    assert np.array_equal(out[margin:, margin:], img)

    def px(i: int, j: int) -> list:
        return out[int(round(i * cf)) + 1, int(round(j * cf)) + 1].tolist()

    assert [px(0, j) for j in range(4)] == [list(_GRAY_BGR), list(_BLACK_BGR)] * 2
    assert not np.any(np.all(out[:margin] == np.array(_WHITE_BGR, np.uint8), axis=-1))


def test_measured_cf_ring_keeps_black_and_white() -> None:
    cf = 37.0
    img = np.full((300, 500, 3), 60, np.uint8)
    out = build_summary_image(img, [], [], cf_px_per_cm=cf, style=_plain(insert_cf_exterior=True),
                              work_scale=1.0, cf_style=_cf_style(exterior_cells=2),
                              cf_source="measured_from_ruler")
    assert out[1, 1].tolist() == list(_WHITE_BGR)


def test_predicted_cf_raft_goes_top_left_and_never_on_a_ruler() -> None:
    """The prediction came from no ruler, so a raft on a detected ruler would claim a measurement."""
    img = np.full((400, 600, 3), 128, np.uint8)
    det = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (250, 200, 550, 240), "source": "archival"}]
    out = build_summary_image(img, det, [], cf_px_per_cm=40.0, style=_plain(insert_cf_in_rulers=True),
                              work_scale=1.0, cf_style=_cf_style(bar_thickness=10, brim=3),
                              cf_source=_PRED)
    ys, xs = np.where(np.all(out == np.array(_CYAN_BGR, np.uint8), axis=-1))
    assert ys.min() == 3 and xs.min() == 3                    # corner raft: origin (0, 0) + brim
    assert _run(_CYAN_BGR, out[ys.min()])[1] == 40            # still exactly 1 cm
    gy = np.where(np.all(out == np.array(_GREEN_BGR, np.uint8), axis=-1))[0]
    assert _run(_GREEN_BGR, out[gy.min()])[1] == round(40 * 2.54)
    on_ruler = np.all(out[200:240, 250:550] == np.array(_CYAN_BGR, np.uint8), axis=-1)
    assert not on_ruler.any()                                 # no bar on the ruler


def test_predicted_cf_raft_sits_below_the_banner() -> None:
    """The banner owns (0, 0) too; the raft must start under it, not be painted over by it."""
    img = np.full((400, 600, 3), 128, np.uint8)
    style = OverlayStyle(draw_labels=False, draw_cf_banner=True, draw_masks=False,
                         insert_cf_in_rulers=True, cf_banner_color=(255, 200, 0))   # not raft-white
    out = build_summary_image(img, [], [], cf_px_per_cm=40.0, style=style, work_scale=1.0,
                              cf_style=_cf_style(bar_thickness=10, brim=3), cf_source=_PRED)
    banner = np.array(style.cf_banner_color[::-1], np.uint8)        # RGB config -> BGR
    banner_rows = np.where(np.all(out[:, 0] == banner, axis=-1))[0]
    raft_rows = np.where(np.all(out[:, 0] == np.array(_WHITE_BGR, np.uint8), axis=-1))[0]
    assert len(banner_rows) and len(raft_rows)
    assert raft_rows.min() == banner_rows.max() + 1


def test_measured_cf_raft_stays_on_the_ruler() -> None:
    """Unchanged behavior for a measured CF: over the ruler, nothing in the corner."""
    img = np.full((400, 600, 3), 128, np.uint8)
    det = [{"cls_name": "Ruler", "conf": 0.9, "xyxy": (250, 200, 550, 240), "source": "archival"}]
    out = build_summary_image(img, det, [], cf_px_per_cm=40.0, style=_plain(insert_cf_in_rulers=True),
                              work_scale=1.0, cf_style=_cf_style(bar_thickness=10, brim=3),
                              cf_source="measured_from_ruler")
    xs = np.where(np.all(out == np.array(_CYAN_BGR, np.uint8), axis=-1))[1]
    assert xs.min() == 250 + 3
    assert np.array_equal(out[:150, :200], img[:150, :200])


def _banner_text(monkeypatch, **kw) -> list[str]:
    """Every string the summary overlay puts on the image (only the banner, with labels off)."""
    from leafmachine3.reporting import overlay

    seen: list[str] = []
    real = overlay.cv2.putText

    def spy(img, text, *a, **k):
        seen.append(text)
        return real(img, text, *a, **k)

    monkeypatch.setattr(overlay.cv2, "putText", spy)
    style = OverlayStyle(draw_labels=False, draw_cf_banner=True, draw_masks=False)
    build_summary_image(np.full((400, 1400, 3), 128, np.uint8), [], [], cf_px_per_cm=40.0,
                        style=style, work_scale=1.0, **kw)
    return seen


def test_banner_names_the_cf_source(monkeypatch) -> None:
    assert _banner_text(monkeypatch, cf_source="measured_from_ruler") == [
        "CF: 40.00 px/cm (measured from ruler)"]
    assert _banner_text(monkeypatch, cf_source=_PRED) == [
        "CF: 40.00 px/cm (predicted from megapixels)"]
    assert _banner_text(monkeypatch) == ["CF: 40.00 px/cm"]       # caller that passes no source


def test_predicted_banner_carries_the_reason_line(monkeypatch) -> None:
    assert _banner_text(monkeypatch, cf_source=_PRED, cf_note="missing ruler") == [
        "CF: 40.00 px/cm (predicted from megapixels)", "missing ruler"]
    # a note on a MEASURED sheet would be a contradiction, so it is never drawn
    assert _banner_text(monkeypatch, cf_source="measured_from_ruler", cf_note="missing ruler") == [
        "CF: 40.00 px/cm (measured from ruler)"]


def test_one_line_banner_geometry_is_unchanged() -> None:
    """The second line must not move a single-line banner by a pixel (overlays stay comparable)."""
    from leafmachine3.reporting.overlay import _draw_cf_banner

    style = OverlayStyle()
    a = np.zeros((300, 1400, 3), np.uint8)
    b = np.zeros((300, 1400, 3), np.uint8)
    assert _draw_cf_banner(a, 40.0, style) == _draw_cf_banner(b, 40.0, style, note=None)
    two = _draw_cf_banner(np.zeros((300, 1400, 3), np.uint8), 40.0, style, note="missing ruler")
    assert two > _draw_cf_banner(np.zeros((300, 1400, 3), np.uint8), 40.0, style)


def test_predicted_cf_raft_sits_below_a_two_line_banner() -> None:
    img = np.full((400, 600, 3), 128, np.uint8)
    style = OverlayStyle(draw_labels=False, draw_cf_banner=True, draw_masks=False,
                         insert_cf_in_rulers=True, cf_banner_color=(255, 200, 0))
    out = build_summary_image(img, [], [], cf_px_per_cm=40.0, style=style, work_scale=1.0,
                              cf_style=_cf_style(bar_thickness=10, brim=3), cf_source=_PRED,
                              cf_note="ruler failed validation (94.82 px)")
    banner = np.array(style.cf_banner_color[::-1], np.uint8)
    banner_rows = np.where(np.all(out[:, 0] == banner, axis=-1))[0]
    raft_rows = np.where(np.all(out[:, 0] == np.array(_WHITE_BGR, np.uint8), axis=-1))[0]
    assert raft_rows.min() == banner_rows.max() + 1


@pytest.mark.parametrize("image, reason", [
    ({"status": "no_ruler", "n_ruler_crops": 0}, "missing ruler"),
    ({"status": "no_reading", "n_ruler_crops": 4, "n_skipped": 4, "n_failed": 0}, "unsupported ruler"),
    ({"status": "no_reading", "n_ruler_crops": 2, "n_skipped": 0, "n_failed": 2}, "unreadable ruler"),
    ({"status": "no_reading", "n_ruler_crops": 3, "n_skipped": 1, "n_failed": 2}, "unusable ruler"),
    ({"status": "withheld", "cf_px_per_cm_measured": 94.8213}, "ruler failed validation (94.82 px)"),
    ({"status": "withheld", "cf_px_per_cm_measured": None}, "ruler failed validation"),
    ({"status": "published"}, None),
    (None, None),
])
def test_cf_fallback_reason(image, reason) -> None:
    from leafmachine3.reporting.overlay import cf_fallback_reason

    assert cf_fallback_reason(image) == reason
