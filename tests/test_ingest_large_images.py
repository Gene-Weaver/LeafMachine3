"""Herbarium-sized originals must be ingested, not quarantined (core/imaging.py MAX_IMAGE_PIXELS).

Pillow's default decompression-bomb guard warns above 89,478,485 px and raises above 178,956,970 px.
Ingest caught that exception as a decode failure, so every sheet larger than ~179 MP was quarantined
as "corrupt" and silently left out of the run (found 2026-10-07 from a DecompressionBombWarning in an
install test). These tests build PNGs of the exact sizes involved WITHOUT holding them in memory: the
pixel data is a zlib stream of zero rows, so a 180 MP file is a few hundred KB.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
import yaml

from leafmachine3.core.config import Config
from leafmachine3.core.db import ProjectDB
from leafmachine3.core.dirs import build_dirs
from leafmachine3.core.imaging import MAX_IMAGE_PIXELS
from leafmachine3.core.ingest import ImageIngestor

from .conftest import build_mock_config, fresh_out_dir


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def write_blank_png(path: Path, w: int, h: int, *, pixels: bool = True) -> Path:
    """An 8-bit grayscale PNG of w x h zeros. ``pixels=False`` writes only a token IDAT: enough for a
    header-level size check, which is where Pillow's guard fires (at open, before any decoding)."""
    comp = zlib.compressobj(9)
    row = b"\x00" * (w + 1)                     # filter byte 0 + w zero samples
    if pixels:
        idat = b"".join(comp.compress(row) for _ in range(h)) + comp.flush()
    else:
        idat = comp.compress(row) + comp.flush()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b""))
    return path


def _run_ingest(images: Path, tmp_path: Path, run_name: str) -> tuple[dict, Path]:
    cfg_dict = build_mock_config(images, fresh_out_dir(run_name), run_name=run_name)
    cfg_path = tmp_path / f"{run_name}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    cfg = Config.load(cfg_path)
    db = ProjectDB.open_or_create(build_dirs(cfg).db_path)
    ImageIngestor(cfg, db).run()
    rows = {r["image_stem"]: dict(r) for r in db.conn.execute("SELECT * FROM specimen")}
    db.close()
    return rows, build_dirs(cfg).root / "INVALID"


def test_importing_ingest_raises_the_pillow_limit():
    from PIL import Image

    assert Image.MAX_IMAGE_PIXELS is None or Image.MAX_IMAGE_PIXELS >= MAX_IMAGE_PIXELS
    assert MAX_IMAGE_PIXELS >= 1_000_000_000


def test_a_180_megapixel_sheet_is_probed_not_rejected(tmp_path):
    """15,000 x 12,000 = 180 MP: just over Pillow's default error threshold (178,956,970)."""
    png = write_blank_png(tmp_path / "big.png", 15_000, 12_000)
    meta = ImageIngestor._probe(png)
    assert meta.wh == (15_000, 12_000)


def test_an_absurd_image_is_quarantined_as_too_large_not_corrupt(tmp_path):
    """A header claiming 2.25 gigapixels exceeds 2 x MAX_IMAGE_PIXELS; it must be refused, and the
    quarantine marker must say why, instead of calling a valid file corrupt."""
    images = tmp_path / "in"
    images.mkdir()
    write_blank_png(images / "absurd.png", 50_000, 45_000, pixels=False)
    rows, invalid = _run_ingest(images, tmp_path, "too_large_probe")
    assert "absurd" not in rows
    markers = list(invalid.glob("absurd_*.txt"))
    assert [m.name.split(".")[-2] for m in markers] == ["too_large"]
    text = markers[0].read_text(encoding="utf-8")
    assert "2250000000 pixels" in text and f"{MAX_IMAGE_PIXELS:,}" in text


@pytest.mark.slow
def test_a_180_megapixel_sheet_is_ingested_end_to_end(tmp_path):
    """The full path: probe, decode, normalize to the working size, register. ~1-2 GB of RAM."""
    images = tmp_path / "in"
    images.mkdir()
    write_blank_png(images / "big_sheet.png", 15_000, 12_000)
    rows, invalid = _run_ingest(images, tmp_path, "large_ingest")
    assert "big_sheet" in rows, f"not ingested; quarantined: {[m.name for m in invalid.glob('*')]}"
    row = rows["big_sheet"]
    assert (row["original_width"], row["original_height"]) == (15_000, 12_000)
    assert max(row["width"], row["height"]) <= 3200 or row["downsampled"]
