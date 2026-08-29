"""The working-frame contract: LM3 analyzes, measures and reports in ONE frame.

Ingest may resize an input down to ``ingest.max_working_dim``. Everything after that -- every
detection box, polygon, keypoint, the published CF, and now every Reporter output -- lives in that
resized WORKING frame. The original's *dimensions* are kept (the MP->CF regression is fit on them);
its *pixels* are read exactly once, by ingest.

Two things here are easy to get wrong and impossible to notice later:

  * ``normalized`` and ``downsampled`` are different questions. Ingest writes a copy when the input
    is not a JPEG *or* is too big, so ``normalized`` alone cannot tell you whether pixels were
    discarded. Every input in a typical herbarium set is a big JPEG, which makes the two flags
    agree by coincidence and hides the difference until someone feeds LM3 a small TIFF.
  * A Reporter output rendered on the original looks *better* -- it is simply bigger -- while
    carrying upscaled masks that were never measured at that resolution. Nothing downstream
    complains. Only a dimension assertion catches it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from leafmachine3.core.config import Config
from leafmachine3.core.db import ProjectDB
from leafmachine3.core.dirs import build_dirs
from leafmachine3.core.ingest import ImageIngestor
from leafmachine3.core.records import SpecimenRecord

from .conftest import build_mock_config, fresh_out_dir, make_specimen_image


def _write_cfg(images: Path, out: Path, tmp_path: Path, run_name: str, max_dim: int) -> Path:
    """A real mock config (so ingest/build_dirs see a genuine Config) with the cap overridden."""
    cfg = build_mock_config(images, out, run_name=run_name)
    cfg["ingest"]["max_working_dim"] = max_dim
    path = tmp_path / f"{run_name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _ingest_only(cfg_path: Path) -> dict[str, dict]:
    """Run ONLY ingest and return ``{image_stem: specimen row}``."""
    cfg = Config.load(cfg_path)
    db = ProjectDB.open_or_create(build_dirs(cfg).db_path)
    ImageIngestor(cfg, db).run()
    rows = {r["image_stem"]: dict(r) for r in db.conn.execute("SELECT * FROM specimen")}
    db.close()
    return rows


# --------------------------------------------------------------------------- #
# normalized vs downsampled
# --------------------------------------------------------------------------- #

def test_normalized_and_downsampled_are_different_questions(tmp_path: Path) -> None:
    """A small PNG is CONVERTED but not resized; a big JPEG is resized but not converted.

    This is the whole reason ``downsampled`` exists: ``normalized`` is 1 for both, so it cannot
    answer "were pixels thrown away" -- the only question that matters for the frame.
    """
    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "small_png.png"), np.full((300, 200, 3), 200, np.uint8))
    cv2.imwrite(str(src / "small_jpeg.jpg"), np.full((300, 200, 3), 200, np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(src / "big_jpeg.jpg"), np.full((900, 600, 3), 200, np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    rows = _ingest_only(_write_cfg(src, fresh_out_dir("test_flags"), tmp_path, "flags", max_dim=600))

    png = rows["small_png"]
    assert (png["normalized"], png["downsampled"]) == (1, 0)      # converted only, no pixels lost
    assert (png["width"], png["height"]) == (png["original_width"], png["original_height"]) == (200, 300)
    assert png["work_scale"] == pytest.approx(1.0)

    big = rows["big_jpeg"]
    assert (big["normalized"], big["downsampled"]) == (1, 1)      # resized (and therefore rewritten)
    assert max(big["width"], big["height"]) == 600
    assert (big["original_width"], big["original_height"]) == (600, 900)
    assert big["work_scale"] == pytest.approx(600 / 900)

    small = rows["small_jpeg"]
    assert (small["normalized"], small["downsampled"]) == (0, 0)  # passed straight through


def test_downsampled_backfills_from_the_dims_on_an_existing_db(tmp_path: Path) -> None:
    """The ALTER can only supply its constant default, which would claim nothing was ever resized.

    A DB predating the column must come back with the truth, not with zeros -- the dims already
    record it exactly, so the migration derives it rather than guessing.
    """
    db = ProjectDB(tmp_path / "old.sqlite")
    db.connect()
    db.init_schema()
    for stem, (w, h, ow, oh) in {"resized": (800, 1000, 4000, 5000),
                                 "untouched": (900, 700, 900, 700)}.items():
        db.upsert_specimen(SpecimenRecord(
            image_name=f"{stem}.jpg", image_stem=stem, original_path=f"/o/{stem}.jpg",
            working_path=f"/w/{stem}.jpg", width=w, height=h,
            original_width=ow, original_height=oh, work_scale=max(w, h) / max(ow, oh)))
    db.conn.execute("ALTER TABLE specimen DROP COLUMN downsampled")     # simulate the older schema
    assert "downsampled" not in [r[1] for r in db.conn.execute("PRAGMA table_info(specimen)")]

    db.init_schema()                                                    # re-open -> migrate + backfill
    got = {r["image_stem"]: r["downsampled"] for r in
           db.conn.execute("SELECT image_stem, downsampled FROM specimen")}
    assert got == {"resized": 1, "untouched": 0}

    db.init_schema()                                                    # idempotent: no re-ALTER, no reset
    assert {r["image_stem"]: r["downsampled"] for r in
            db.conn.execute("SELECT image_stem, downsampled FROM specimen")} == got
    db.close()


# --------------------------------------------------------------------------- #
# the Reporter renders in the working frame and never opens an original
# --------------------------------------------------------------------------- #

def test_report_bundle_cannot_carry_an_original_path() -> None:
    """Structural, not conventional: the field is gone, so a future edit cannot quietly re-read
    originals in the Reporter without first putting it back and saying why."""
    from leafmachine3.core.records import ReportBundle

    assert "original_path" not in ReportBundle.__dataclass_fields__


def test_reporter_outputs_are_working_sized_and_survive_a_deleted_original(tmp_path: Path) -> None:
    """End-to-end on a resized sheet: every rendered output matches the WORKING dims, and the run
    completes with the original file deleted after ingest.

    Both halves matter. Sizes alone would still pass if the Reporter opened the original and
    happened to downscale it; deleting the original alone would pass if it rendered at some third
    size. Together they pin the frame AND the dependency.
    """
    from leafmachine3.machine3 import machine3

    src = tmp_path / "in"
    make_specimen_image(src / "sheet.jpg", seed=1)                      # 700x900
    out = fresh_out_dir("test_working_frame")
    cfg_path = _write_cfg(src, out, tmp_path, "wframe", max_dim=450)    # force a 0.5x downsample

    cfg = Config.load(cfg_path)
    dirs = build_dirs(cfg)
    db = ProjectDB.open_or_create(dirs.db_path)
    ImageIngestor(cfg, db).run()
    row = dict(next(iter(db.conn.execute("SELECT * FROM specimen"))))
    db.close()

    assert row["downsampled"] == 1
    assert (row["width"], row["height"]) == (350, 450)
    assert (row["original_width"], row["original_height"]) == (700, 900)

    (src / "sheet.jpg").unlink()                                        # the original is now GONE
    machine3(cfg_path)                                                  # ...and the run still completes

    rendered = [p for p in dirs.reports.rglob("*")
                if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    assert rendered, "the Reporter wrote nothing, so this test asserts nothing"
    long_side = max(row["width"], row["height"])
    oversized = []
    for p in rendered:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Overlay_Specimen_Segmentation is two panels hstacked (sheet | cutout), so its width is
        # legitimately 2x a panel's. Measure the PANEL, not the sheet of paper it is printed on.
        if p.parent.name == "Overlay_Specimen_Segmentation":
            w //= 2
        if max(h, w) > long_side:
            oversized.append(f"{p.relative_to(dirs.reports)} {w}x{h}")
    assert not oversized, (
        f"outputs exceed the working frame ({row['width']}x{row['height']}), so something is "
        f"still rendering on the original: {oversized}"
    )

    summary = next(iter((dirs.reports / "Overlay" / "Overlay_Summary").glob("*__Overlay.*")))
    sh, sw = cv2.imread(str(summary)).shape[:2]
    assert (sw, sh) == (row["width"], row["height"])                    # 1:1 with what was analyzed

    spec = next(iter((dirs.reports / "Overlay" / "Overlay_Specimen_Segmentation").glob("*__SpecimenSeg.*")))
    ph, pw = cv2.imread(str(spec)).shape[:2]
    assert (pw, ph) == (2 * row["width"], row["height"])                # two working-sized panels
