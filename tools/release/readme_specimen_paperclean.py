#!/usr/bin/env python3
"""Build the README's paper-cleanup comparison figures (section 4, Specimen Segmenter) in ``docs/readme_github/specimen_paperclean/``.

One figure per sheet, drawn with LM3's own pieces so it shows exactly what a run produces: the ingest working image
(long side capped at ingest.max_working_dim, LANCZOS, JPEG at ingest.jpg_quality), the specimen_segmenter backend for
the configured model key, then ``core.paper_removal.run_paperclean`` with each method, rendered with the Reporter's
``build_specimen_overlay`` in the ``report.overlay_specimen`` style:

    columns: no paper cleanup | LLR_Hyst_KeepBlobs (the default) | HSV (the original method)
    top row: overlay (green = kept, red = removed by the cleanup, blue = paper-sampling boxes); bottom row: cutout

The sheets (CONFIG below) are bundled examples (``examples/images``) or CC0 sheets from the LM3 CC0 evaluation sets;
``provenance.json`` records each one's source, license and GBIF record. The README shows the figures as a 2 x 2 table
with one caption per sheet.

    .venv/bin/python tools/release/readme_specimen_paperclean.py [--model <onnx>] [--key <model key>] [--device cuda:0]

Without --model/--key it uses the specimen_segmenter model in LM3_settings.yaml. ONNX on the GPU needs the venv's
CUDA libraries on LD_LIBRARY_PATH (else onnxruntime silently falls back to the CPU, which is only slower).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from leafmachine3.core.config import Config  # noqa: E402
from leafmachine3.core.imaging import configure_pillow, read_image  # noqa: E402
from leafmachine3.core.paper_removal import METHOD_HSV, METHOD_LLR_HYST_KEEPBLOBS, run_paperclean  # noqa: E402
from leafmachine3.inference.specimen_models import DEFAULT_SPECIMEN_MODEL_KEY, SPECIMEN_MODELS  # noqa: E402
from leafmachine3.inference.specimen_segmenter import build_specimen_segmenter  # noqa: E402
from leafmachine3.reporting.overlay import build_specimen_overlay  # noqa: E402
from leafmachine3.reporting.palette import SpecimenStyle  # noqa: E402
from PIL import ImageDraw, ImageFont  # noqa: E402

Image = configure_pillow()

# ---------------------------------------------------------------- CONFIG
# (file name, image path, source, license, GBIF record). Order = README grid order (row by row).
CC0 = Path("/datac/Labelbox_Dump/datasets")
SHEETS = [
    ("01_Quercus_helvetica_Z_2575336428", ROOT / "examples/images/Z_2575336428_Fagaceae_Quercus_helvetica.jpg",
     "bundled example", "see examples/images", "https://www.gbif.org/occurrence/2575336428"),
    ("02_Catalpa_erubescens_PH_3865012217", ROOT / "examples/images/PH_3865012217_Bignoniaceae_Catalpa_erubescens.jpg",
     "bundled example", "see examples/images", "https://www.gbif.org/occurrence/3865012217"),
    ("03_Quercus_anceps_LYN_4134274989",
     CC0 / "LM3_Specimen_CC0Evaluation/v-0/images/test/dnmogprz__LYN_4134274989_Fagaceae_Quercus_anceps.jpg",
     "LM3 CC0 evaluation set (specimen)", "CC0 1.0 (Lynchburg College, Ramsey-Freer Herbarium)",
     "https://www.gbif.org/occurrence/4134274989"),
    ("04_Quercus_hypoleuca_L_2515539380",
     CC0 / "LM3_Plant_CC0Evaluation/v-0/images/test/dnmogprz__L_2515539380_Fagaceae_Quercus_hypoleuca.jpg",
     "LM3 CC0 evaluation set (plant)", "CC0 1.0 (Naturalis Biodiversity Center)",
     "https://www.gbif.org/occurrence/2515539380"),
]
COLUMNS = [("No paper cleanup", None), ("LLR (default)", METHOD_LLR_HYST_KEEPBLOBS), ("HSV", METHOD_HSV)]
TILE_H, PAD, HEAD = 1000, 16, 132        # tile height, gaps, column-header band (px, before the final resize)
OUT_W = 1500                             # figure width written


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F_BIG, F_SMALL = font(54), font(38)


def working_image(path: Path, max_dim: int, quality: int, tmp: Path) -> np.ndarray:
    """The image an LM3 run segments (core/ingest.py ``_normalize`` for non-JPEG or oversized originals)."""
    with Image.open(path) as im:
        if path.suffix.lower() in (".jpg", ".jpeg") and max(im.size) <= max_dim:
            return read_image(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max(im.size) > max_dim:
            s = max_dim / float(max(im.size))
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
        dst = tmp / f"{path.stem}_work.jpg"
        im.save(dst, format="JPEG", quality=quality)
    return read_image(dst)


def figure(img: np.ndarray, raw: np.ndarray, res: dict, style: SpecimenStyle) -> tuple[Image.Image, dict]:
    tiles, stats = [], {"kept_pct_of_sheet": {}, "removed_pct_of_mask": {}}
    for name, method in COLUMNS:
        if method is None:
            final, removed, centers = raw, np.zeros_like(raw), []
            note = f"{raw.mean() * 100:.1f}% of the sheet"
        else:
            final, removed, centers = res[method]
            share = removed.sum() / max(1, int(raw.sum())) * 100
            note = f"removes {share:.1f}% of the mask"
            stats["removed_pct_of_mask"][method] = round(float(share), 2)
        stats["kept_pct_of_sheet"][method or "none"] = round(float(final.mean() * 100), 2)
        both = build_specimen_overlay(img, final * 255, removed * 255, centers, style)
        w = both.shape[1] // 2
        tiles.append((name, note, both[:, :w], both[:, w:]))
    h0, w0 = tiles[0][2].shape[:2]
    def paste(im, x, y, tw, th):
        r = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)), (x, y))
    if w0 <= h0:      # portrait sheet: one column per method, overlay above cutout
        tw = int(round(w0 * TILE_H / h0))
        W, H = 3 * tw + 4 * PAD, HEAD + 2 * TILE_H + 3 * PAD
        canvas = Image.new("RGB", (W, H), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)
        for j, (name, note, ov, cut) in enumerate(tiles):
            x = PAD + j * (tw + PAD)
            dr.text((x + 6, 10), name, fill=(245, 245, 245), font=F_BIG)
            dr.text((x + 6, 76), note, fill=(175, 175, 175), font=F_SMALL)
            for i, im in enumerate((ov, cut)):
                paste(im, x, HEAD + PAD + i * (TILE_H + PAD), tw, TILE_H)
    else:             # landscape sheet: one row per method, overlay beside cutout (keeps the figure near-square)
        tw = int(round(w0 * TILE_H / h0))
        W, H = 2 * tw + 3 * PAD, 3 * (HEAD + TILE_H) + 4 * PAD
        canvas = Image.new("RGB", (W, H), (18, 18, 18))
        dr = ImageDraw.Draw(canvas)
        for i, (name, note, ov, cut) in enumerate(tiles):
            y = PAD + i * (HEAD + TILE_H + PAD)
            dr.text((PAD + 6, y), name, fill=(245, 245, 245), font=F_BIG)
            dr.text((PAD + 6 + dr.textlength(name, font=F_BIG) + 30, y + 14), note, fill=(175, 175, 175), font=F_SMALL)
            for j, im in enumerate((ov, cut)):
                paste(im, PAD + j * (tw + PAD), y + HEAD - 40, tw, TILE_H)
    if W > OUT_W:
        canvas = canvas.resize((OUT_W, round(H * OUT_W / W)), Image.LANCZOS)
    return canvas, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--settings", default=str(ROOT / "LM3_settings.yaml"))
    ap.add_argument("--model", default="", help="specimen segmenter ONNX (default: the one LM3_settings.yaml names)")
    ap.add_argument("--key", default="", help="its model key (default: the one LM3_settings.yaml names)")
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--out", default=str(ROOT / "docs/readme_github/specimen_paperclean"))
    a = ap.parse_args()
    if a.device.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device.split(":")[1] if ":" in a.device else "0")
    cfg = Config.load(a.settings)
    sc = cfg.stage("specimen_segmenter")
    key = a.key or (sc.get("model") or {}).get("key") or DEFAULT_SPECIMEN_MODEL_KEY
    model_path = a.model or str((ROOT / (sc.get("model") or {}).get("path", "")).resolve())
    spec = SPECIMEN_MODELS.get(key) or dataclasses.replace(SPECIMEN_MODELS[DEFAULT_SPECIMEN_MODEL_KEY], key=key, label=key)
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if a.device.startswith("cuda") else ["CPUExecutionProvider"]
    model = build_specimen_segmenter(spec, model_path, providers=prov, conf=float(sc.get("conf", 0.5)), paperclean=False)
    style = SpecimenStyle.from_config(cfg)
    max_dim, quality = int(cfg.ingest.max_working_dim), int(cfg.ingest.jpg_quality)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    with tempfile.TemporaryDirectory() as td:
        for name, path, source, lic, gbif in SHEETS:
            img = working_image(Path(path), max_dim, quality, Path(td))
            raw = (model.mask(img) > 0).astype(np.uint8)
            res = {}
            for m in (METHOD_LLR_HYST_KEEPBLOBS, METHOD_HSV):
                final, removed, centers = run_paperclean(img, raw, m)
                res[m] = ((final > 0).astype(np.uint8), (removed > 0).astype(np.uint8), centers)
            fig, stats = figure(img, raw, res, style)
            fig.save(out / f"{name}.jpg", quality=85)
            recs.append({"file": f"{name}.jpg", "source_image": str(path), "source": source, "license": lic, "gbif": gbif,
                         "working_image_size": [img.shape[1], img.shape[0]], **stats})
            print(f"{name}.jpg  " + json.dumps(stats), flush=True)
    (out / "provenance.json").write_text(json.dumps(
        {"what": "README section 4: the specimen mask with no paper cleanup, with LLR_Hyst_KeepBlobs (default) and with the "
                 "HSV method; top = Reporter overlay, bottom = cutout. Rendered by tools/release/readme_specimen_paperclean.py.",
         "model_key": key, "model": model_path, "provider": model.session.get_providers()[0],
         "rendered": dt.date.today().isoformat(), "sheets": recs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
