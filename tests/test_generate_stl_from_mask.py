"""Tests for the generate_stl_from_mask postprocessing tool."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")           # skip if optional postproc deps are absent
pytest.importorskip("shapely")

from leafmachine3.postprocessing.generate_stl_from_mask import (  # noqa: E402
    _parse_colors,
    generate_stl,
    run,
)


def _write(tmp_path, name, img):
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


def _disk(size, cx, cy, r, val=255):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (cx, cy), r, int(val), -1)
    return m


def test_circle_becomes_a_cylinder(tmp_path) -> None:
    """A circular mask extrudes to a cylinder: longest dim == length_mm, z == thickness_mm."""
    p = _write(tmp_path, "circle.png", _disk(400, 200, 200, 150))
    out = tmp_path / "circle.stl"
    res = generate_stl(p, out, length_mm=150.0, thickness_mm=2.0, simplify_tolerance_px=1.0)
    assert out.exists() and res["watertight"]
    x, y, z = res["size_mm"]
    assert abs(z - 2.0) < 1e-6                       # thickness
    assert abs(max(x, y) - 150.0) < 0.5              # longest dim scaled to length_mm
    assert abs(x - y) < 2.0                          # a circle -> ~square footprint
    mesh = trimesh.load(str(out))
    # volume ~ area(circle @ 150mm diameter) * 2mm; footprint diameter 150 -> r=75
    assert math.isclose(mesh.volume, math.pi * 75 ** 2 * 2.0, rel_tol=0.05)


def test_fill_holes_toggle(tmp_path) -> None:
    """A donut: fill_holes=True yields a solid disk (more volume) than fill_holes=False."""
    donut = _disk(400, 200, 200, 150)
    cv2.circle(donut, (200, 200), 60, 0, -1)         # punch a hole
    p = _write(tmp_path, "donut.png", donut)
    filled = generate_stl(p, tmp_path / "f.stl", length_mm=150, thickness_mm=2, fill_holes=True)
    holed = generate_stl(p, tmp_path / "h.stl", length_mm=150, thickness_mm=2, fill_holes=False)
    vf = trimesh.load(str(tmp_path / "f.stl")).volume
    vh = trimesh.load(str(tmp_path / "h.stl")).volume
    assert vf > vh                                   # filling the hole adds material
    assert holed["watertight"] and filled["watertight"]
    # the hole (r=60 of R=150 -> 16% of area) is removed when not filled
    assert math.isclose(vh / vf, 1 - (60 / 150) ** 2, rel_tol=0.06)


def test_length_scale_and_thickness(tmp_path) -> None:
    """A wide rectangle: longest side maps to length_mm, the short side scales proportionally."""
    m = np.zeros((300, 600, 3), np.uint8)
    m[100:200, 100:500] = (255, 255, 255)            # 400 x 100 px white rectangle
    p = _write(tmp_path, "rect.png", m)
    res = generate_stl(p, tmp_path / "rect.stl", length_mm=200.0, thickness_mm=3.0,
                       simplify_tolerance_px=0.0)
    x, y, z = res["size_mm"]
    assert abs(max(x, y) - 200.0) < 0.5              # 400px longest -> 200mm
    assert abs(min(x, y) - 50.0) < 0.5               # 100px -> 50mm (proportional)
    assert abs(z - 3.0) < 1e-6


def test_color_selection_union(tmp_path) -> None:
    """Color(s) pick the foreground; a list of colors is unioned into one mask."""
    img = np.zeros((200, 200, 3), np.uint8)          # BGR canvas
    img[20:80, 20:80] = (0, 0, 255)                  # a RED square (RGB 255,0,0)
    img[120:180, 120:180] = (255, 255, 255)          # a WHITE square
    p = _write(tmp_path, "colors.png", img)
    # white only -> 1 part; red only -> 1 part; both -> 2 parts
    assert generate_stl(p, tmp_path / "w.stl", colors=["white"], min_area_px=10)["n_parts"] == 1
    assert generate_stl(p, tmp_path / "r.stl", colors=[[255, 0, 0]], min_area_px=10)["n_parts"] == 1
    assert generate_stl(p, tmp_path / "b.stl", colors=["white", [255, 0, 0]], min_area_px=10)["n_parts"] == 2


def test_parse_colors_forms() -> None:
    assert _parse_colors("white") == [(255, 255, 255)]
    assert _parse_colors([255, 255, 255]) == [(255, 255, 255)]          # single RGB
    assert _parse_colors([[255, 0, 0], "black"]) == [(255, 0, 0), (0, 0, 0)]
    assert _parse_colors([[1, 2, 3, 255]]) == [(1, 2, 3)]               # RGBA -> alpha ignored


def test_run_de_dupes_and_uses_output_dir(tmp_path) -> None:
    p = _write(tmp_path, "circle.png", _disk(200, 100, 100, 70))
    outdir = tmp_path / "out"
    results = run({"length_mm": 100, "thickness_mm": 2}, paths=[str(p), str(p)], output_dir=str(outdir))
    assert len(results) == 1                          # duplicate path collapsed
    assert (outdir / "circle.stl").exists()
