"""JPEG XL (.jxl) originals are ingested like any other non-JPEG: decoded, converted to a JPEG
working copy in _tmp_original, and registered. Pillow 12.2 has no JXL codec; core/imaging.py
registers the pinned pillow-jxl-plugin. Also covers the 16-bit grayscale rescale (_to_rgb8), which
JXL made common but which equally fixes 16-bit PNG/TIFF (convert("RGB") clipped them to white).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from leafmachine3.core.config import Config, builtin_defaults
from leafmachine3.core.db import ProjectDB
from leafmachine3.core.dirs import build_dirs
from leafmachine3.core.imaging import configure_pillow
from leafmachine3.core.ingest import SUPPORTED, ImageIngestor, _to_rgb8

from .conftest import build_mock_config, fresh_out_dir

pytest.importorskip("pillow_jxl")
Image = configure_pillow()


def _sheet(w: int = 640, h: int = 480) -> np.ndarray:
    """A smooth synthetic RGB 'sheet' (gradients compress well and survive lossy coding)."""
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    return np.stack([np.broadcast_to(x, (h, w)), np.broadcast_to(y, (h, w)),
                     np.full((h, w), 128, np.float32)], -1).astype(np.uint8)


def _run_ingest(images: Path, tmp_path: Path, run_name: str) -> dict:
    cfg_dict = build_mock_config(images, fresh_out_dir(run_name), run_name=run_name)
    cfg_dict["project"]["input"]["image_extensions"] = [".jpg", ".png", ".jxl"]
    cfg_path = tmp_path / f"{run_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    cfg = Config.load(cfg_path)
    db = ProjectDB.open_or_create(build_dirs(cfg).db_path)
    ImageIngestor(cfg, db).run()
    rows = {r["image_stem"]: dict(r) for r in db.conn.execute("SELECT * FROM specimen")}
    db.close()
    return rows


def test_jxl_is_registered_and_on_by_default():
    assert Image.registered_extensions().get(".jxl") == "JXL"
    assert ".jxl" in SUPPORTED
    assert ".jxl" in builtin_defaults()["project"]["input"]["image_extensions"]


def test_a_jxl_sheet_is_ingested_as_a_jpeg_working_copy(tmp_path):
    images = tmp_path / "in"
    images.mkdir()
    src = _sheet()
    Image.fromarray(src).save(images / "sheet.jxl", lossless=True)

    rows = _run_ingest(images, tmp_path, "jxl_ingest")
    assert "sheet" in rows
    row = rows["sheet"]
    assert (row["original_width"], row["original_height"]) == (640, 480)
    assert (row["width"], row["height"]) == (640, 480)
    assert row["normalized"] and not row["downsampled"]

    working = Path(row["working_path"])
    assert working.suffix == ".jpg"
    with Image.open(working) as im:                    # follows the symlink into _tmp_original
        assert im.format == "JPEG"
        got = np.asarray(im.convert("RGB"), dtype=np.int16)
    assert np.abs(got - src).mean() < 2.0              # only JPEG q100 loss on top of lossless JXL


@pytest.mark.parametrize("ext", [".jxl", ".png"])
def test_16_bit_grayscale_is_rescaled_not_clipped(tmp_path, ext):
    """A mid-gray 16-bit scan (32768) must land near 128, not at 255 as convert("RGB") gave."""
    images = tmp_path / "in"
    images.mkdir()
    a = np.full((240, 320), 32768, np.uint16)
    a[:, :160] = 8192                                   # two tones so a constant output fails too
    Image.fromarray(a).save(images / f"gray16{ext}", **({"lossless": True} if ext == ".jxl" else {}))

    rows = _run_ingest(images, tmp_path, f"gray16_{ext[1:]}")
    with Image.open(rows["gray16"]["working_path"]) as im:
        got = np.asarray(im.convert("RGB"), dtype=np.float32)
    assert abs(got[:, 200:].mean() - 32768 / 257) < 2.0
    assert abs(got[:, :120].mean() - 8192 / 257) < 2.0


def test_to_rgb8_float_range_and_passthrough():
    f = Image.fromarray(np.full((4, 4), 0.5, np.float32))
    assert f.mode == "F"
    assert np.asarray(_to_rgb8(f))[0, 0].tolist() == [128, 128, 128]
    rgb = Image.fromarray(_sheet(8, 8))
    assert _to_rgb8(rgb) is rgb
