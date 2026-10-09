"""Tests for the Momocs stage: mask preparation, outline conventions, Momit JSON, report-dep override.

Each format rule guards against an import that goes wrong in R WITHOUT an error (see
``leafmachine3.core.momocs_prep``), so the tests pin the rule itself rather than an implementation detail.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from leafmachine3.core.momocs_prep import (
    merge_momit_documents,
    momit_document,
    momocs_image,
    momocs_outline,
    momocs_product_for,
    prepare_mask,
    resample_closed,
    signed_area,
)
from leafmachine3.modules.momocs import enforce_report_deps


def _leaf(h=200, w=120, hole=True, speck=True):
    """A white-on-black, edge-touching ellipse 'leaf' (like an LM3 leaf product) + a hole + a speck."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    if hole:
        cv2.circle(m, (w // 2, int(h * 0.62)), 10, 0, -1)      # on the center-down line Momocs walks
    if speck:
        m[2:5, 2:5] = 255                                       # a stray fragment in a corner
    return m


def test_product_mapping_always_holes_filled() -> None:
    assert momocs_product_for(False).folder == "Lamina_Holes_Mask"
    assert momocs_product_for(True).folder == "LaminaPetiole_Holes_Mask"
    assert momocs_product_for(True).product_key == "lamina_petiole_holes_mask"
    assert (momocs_product_for(False).mask_includes, momocs_product_for(True).mask_includes) \
        == ("lamina", "lamina_petiole")


def test_prepare_mask_pads_fills_and_keeps_one_piece() -> None:
    m = _leaf()
    p = prepare_mask(m, pad_px=10)
    assert p.shape == (m.shape[0] + 20, m.shape[1] + 20)
    assert not p[:10].any() and not p[-10:].any() and not p[:, :10].any() and not p[:, -10:].any()
    n_fg = cv2.connectedComponents(p.astype(np.uint8))[0] - 1
    n_bg = cv2.connectedComponents((~p).astype(np.uint8))[0] - 1
    assert n_fg == 1                      # the corner speck is gone
    assert n_bg == 1                      # the hole is filled: background is one region


def test_prepare_mask_empty_is_none() -> None:
    assert prepare_mask(np.zeros((50, 50), np.uint8)) is None


def test_momocs_image_is_black_leaf_on_white() -> None:
    img = momocs_image(prepare_mask(_leaf(), pad_px=10))
    assert img.dtype == np.uint8 and set(np.unique(img)) == {0, 255}
    assert img[0, 0] == 255 and img[img.shape[0] // 2, img.shape[1] // 2] == 0


def test_outline_is_y_up_clockwise_and_starts_at_the_base() -> None:
    p = prepare_mask(_leaf(), pad_px=10)
    xy = momocs_outline(p)
    assert signed_area(xy) < 0                            # clockwise in a y-up frame
    assert xy[0, 1] == xy[:, 1].min()                     # starts at the lowest point (the base, tip-up)
    rows, cols = np.nonzero(p)
    assert xy[:, 1].max() == (p.shape[0] - 1) - rows.min()   # y = (h - 1) - row: the top is the max y
    assert xy[:, 0].min() == cols.min()


def test_outline_y_up_is_not_mirrored() -> None:
    """An asymmetric shape (blob up-left) must keep its handedness in y-up coordinates."""
    m = np.zeros((100, 100), np.uint8)
    cv2.rectangle(m, (20, 20), (80, 80), 255, -1)
    cv2.circle(m, (25, 25), 15, 255, -1)                  # bulge at the top-left in the IMAGE
    xy = momocs_outline(prepare_mask(m, pad_px=5))
    top_left = xy[(xy[:, 0] < xy[:, 0].mean()) & (xy[:, 1] > xy[:, 1].mean())]
    bottom_left = xy[(xy[:, 0] < xy[:, 0].mean()) & (xy[:, 1] < xy[:, 1].mean())]
    assert top_left[:, 0].min() < bottom_left[:, 0].min()    # the bulge is still at the top-left


def test_resample_closed_spacing() -> None:
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    r = resample_closed(sq, 40)
    assert r.shape == (40, 2)
    d = np.hypot(*np.diff(np.vstack([r, r[:1]]), axis=0).T)
    assert np.allclose(d, 1.0)
    assert momocs_outline(prepare_mask(_leaf(), pad_px=10), n_points=64).shape == (64, 2)


def test_momit_document_shape() -> None:
    xy = momocs_outline(prepare_mask(_leaf(), pad_px=10))
    doc = momit_document([
        {"id": "a__or-MOMOCS-lamina__1_2_3_4", "coo": xy, "image_stem": "a", "leaf_id": 1, "cf_px_per_cm": None},
        {"id": "b__or-MOMOCS-lamina__1_2_3_4", "coo": xy, "image_stem": "b", "leaf_id": 2, "cf_px_per_cm": 3.5},
    ])
    meta = doc["metadata"]
    assert meta["n_rows"] == 2 and meta["version"] == "0.1.0"
    assert list(meta["columns"])[:2] == ["id", "coo"]
    assert meta["columns"]["coo"]["col_class"] == ["out", "coo", "list"]
    assert meta["columns"]["coo"]["elem_class"] == ["xy", "matrix", "array"]
    assert meta["columns"]["leaf_id"]["col_class"] == "integer"
    assert meta["columns"]["cf_px_per_cm"]["col_class"] == "numeric"
    row = doc["data"][0]
    assert row["cf_px_per_cm"] is None and len(row["coo"]) == len(xy) and len(row["coo"][0]) == 2
    json.dumps(doc)                                       # serializable as is (no numpy scalars)


def test_merge_momit_documents() -> None:
    xy = momocs_outline(prepare_mask(_leaf(), pad_px=10), n_points=32)
    d1 = momit_document([{"id": "a", "coo": xy, "image_stem": "a"}])
    d2 = momit_document([{"id": "b", "coo": xy, "image_stem": "b"}, {"id": "c", "coo": xy, "image_stem": "b"}])
    merged = merge_momit_documents([d1, d2])
    assert merged["metadata"]["n_rows"] == 3
    assert [r["id"] for r in merged["data"]] == ["a", "b", "c"]
    assert merge_momit_documents([])["metadata"]["n_rows"] == 0


def test_enforce_report_deps_forces_the_product(mock_config_path) -> None:
    from leafmachine3.core.config import Config

    cfg = Config.load(mock_config_path)
    cfg.modules["momocs"] = {"enabled": True, "include_petiole": True, "oriented": False}
    enforce_report_deps(cfg)
    lp = cfg.report["leaf_products"]
    assert lp["enabled"] is True and lp["original"] is True
    assert lp["products"]["lamina_petiole_holes_mask"] is True     # opt-in product forced on


def test_enforce_report_deps_off_when_disabled(mock_config_path) -> None:
    from leafmachine3.core.config import Config

    cfg = Config.load(mock_config_path)
    cfg.modules["momocs"] = {"enabled": False, "include_petiole": True}
    before = json.dumps(cfg.report.get("leaf_products"), sort_keys=True, default=str)
    enforce_report_deps(cfg)
    assert json.dumps(cfg.report.get("leaf_products"), sort_keys=True, default=str) == before
