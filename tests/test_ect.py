"""Tests for the ECT stage: native ect compute, include->product mapping, report-dep override, h5."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

pytest.importorskip("ect")
pytest.importorskip("h5py")

from leafmachine3.core.ect_compute import compute_ect, ect_product_for  # noqa: E402
from leafmachine3.modules.ect import _write_ect_h5, enforce_report_deps  # noqa: E402


def _disk(size=300, r=120):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), r, 255, -1)
    return m


def test_compute_ect_disk() -> None:
    res = compute_ect(_disk() > 127, num_dirs=64, want_simple=True, simplify_cutoff=50)
    assert res.ect_matrix.shape == (64, 64)
    assert res.thetas.shape == (64,) and res.thresholds.shape == (64,)
    assert abs(float(np.hypot(*res.outline_norm.T).max()) - 1.0) < 0.02   # scaled to the unit circle
    assert res.outline_literal.shape[1] == 2
    assert res.outline_simple is not None and len(res.outline_simple) < res.n_points   # DP reduced points


def test_ect_product_mapping() -> None:
    """The 4 (petiole, holes) combos map to the right mask_includes label + Leaf_Oriented product."""
    assert (ect_product_for(False, False).mask_includes, ect_product_for(False, False).folder) \
        == ("lamina", "Lamina_Holes_Mask")                       # holes FILLED
    assert (ect_product_for(False, True).mask_includes, ect_product_for(False, True).folder) \
        == ("lamina_hole", "Lamina_Mask")                        # holes PUNCHED
    assert (ect_product_for(True, False).mask_includes, ect_product_for(True, False).folder) \
        == ("lamina_petiole", "LaminaPetiole_Holes_Mask")        # +petiole, holes filled (opt-in product)
    assert (ect_product_for(True, True).mask_includes, ect_product_for(True, True).folder) \
        == ("lamina_petiole_hole", "LaminaPetiole_Mask")         # +petiole, holes punched


def test_enforce_report_deps_forces_mask(mock_config_path) -> None:
    from leafmachine3.core.config import Config

    cfg = Config.load(mock_config_path)
    cfg.modules["ect"] = {"enabled": True, "include_petiole": True, "include_holes": False}
    enforce_report_deps(cfg)
    lp = cfg.report["leaf_products"]
    assert lp["oriented"] is True and lp["enabled"] is True
    assert lp["products"]["lamina_petiole_holes_mask"] is True    # the opt-in product ECT needs is forced on


def test_write_ect_h5_sections(tmp_path) -> None:
    import h5py

    res = compute_ect(_disk() > 127, num_dirs=64, want_simple=True, simplify_cutoff=50)
    r = {
        "result": res, "mask_includes": "lamina", "stem": "a",
        "leaf": {"leaf_id": 1, "detection_id": 2, "instance_index": 0,
                 "det_box": (0, 0, 300, 300), "angle_cw": 12.5},
        "specimen": {"image_name": "a.jpg", "image_stem": "a", "cf_px_per_cm": 100.0,
                     "cf_px_per_cm_predicted_by_mp": 81.0, "original_width": 3000,
                     "original_height": 4000, "original_mp": 12.0},
    }
    p = tmp_path / "leaf.h5"
    _write_ect_h5(p, r, {"export_literal_coords": True, "simplify_tolerance": 0.0025})

    with h5py.File(str(p), "r") as f:
        assert {"image_metadata", "ect_data", "leaf_outline", "leaf_outline_simple"}.issubset(set(f.keys()))
        im = f["image_metadata"].attrs
        assert im["parent_image_filename"] == "a.jpg" and im["mask_includes"] == "lamina"
        assert abs(float(im["cf_px_per_cm_predicted_by_mp"]) - 81.0) < 1e-6      # linear-predicted CF
        assert abs(float(im["cf_px_per_cm"]) - 100.0) < 1e-6                     # final CF
        assert int(im["original_width"]) == 3000 and int(im["original_height"]) == 4000
        assert abs(float(im["original_mp"]) - 12.0) < 1e-6                       # MP
        ed = f["ect_data"]
        assert ed["matrix"].shape == (64, 64) and "thetas" in ed and "thresholds" in ed
        lo = f["leaf_outline"]
        assert lo["coords"].shape[1] == 2 and "coords_literal" in lo             # literal coords exported
        assert bool(lo.attrs["normalized_unit_circle"]) and lo.attrs["mask_includes"] == "lamina"
