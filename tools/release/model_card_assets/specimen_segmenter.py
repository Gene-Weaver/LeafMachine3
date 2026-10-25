#!/usr/bin/env python3
"""Render the specimen_segmenter model-card panels: earlier UNet++ models next to the one being released, same sheets.

The released model is ``--key`` (its ONNX ``--model``; default the v2 key, so the v2 card's commands below still
reproduce its panels). Earlier models come from ``--v1-masks`` (the v1 card's SAVED masks) and/or ``--baseline
KEY=model.onnx`` (re-run through LM3's backend). A key LM3's registry does not know yet (a model being released) runs
with the v2 key's spec (same letterbox_imagenet workflow) under its own name. Two sets, both drawn like the
unetpp_effb7_1024 card's panels (sheet on the left, then one cut-out per model on black, the share of the sheet kept
under each):

``examples``  the 20 sheets of the unetpp_effb7_1024 card (docs/hf_docs/specimen_segmenter/compare_provenance.json):
    Original | <earlier models> | <released model>. Saved v1 masks are not re-run; the script checks that they
    reproduce the published kept-% numbers. The models run on the same input those masks came from (the dataset
    image at native resolution), raw masks, no paper removal.
    -> docs/hf_docs/specimen_segmenter/<key>/compare_NN_<split>_<specimen>.jpg + compare_provenance.json

``failures``  every image in --in (sheets the v1 UNet++ failed on), or only the picks of an earlier card
    (--picks-from <failure_provenance.json>: same sheets, same panel kinds, same selection notes), every model run the
    way an LM3 run does: the ingest working image (long side capped at ingest.max_working_dim with LANCZOS, re-saved
    as JPEG at ingest.jpg_quality; core/ingest.py ``_normalize``), the stage's backend for each model key, then the
    paperclean follow-up (core/paper_removal.py ``run_paperclean``; --paperclean settings = the method the
    settings file selects, LLR_Hyst_KeepBlobs by default). Two panels per sheet:
      failure_<split>_<specimen>__raw_masks.jpg           Original | <each model>                  (model masks)
      failure_<split>_<specimen>__with_paper_removal.jpg  Original | <each model> + paper removal   (red = deleted by it)
    -> --out + failure_provenance.json

LM3 runtime env (ONNX on the GPU needs the venv's CUDA libs on LD_LIBRARY_PATH, else silent CPU fallback):
    # v2 card (as rendered for it; --paperclean hsv was the only method then)
    .venv/bin/python tools/release/model_card_assets/specimen_segmenter.py examples --model <v2 model.onnx> \
        --v1-masks <dir> [<dir> ...] --label "<model, iteration>" --device cuda:0
    .venv/bin/python tools/release/model_card_assets/specimen_segmenter.py failures --model <v2 model.onnx> \
        --v1-model <v1 model.onnx> --in <dir> --out <dir> --manifest <failures manifest.csv> --label "..." --paperclean hsv
    # v8 card: v2 as the baseline, the v2 card's failure picks
    ... examples --key unetpp_effb7_1024_dinov3_vith16plus_2048_consensus --model <v8.onnx> \
        --baseline unetpp_effb7_1024_dinov3_vith16plus_2048=<v2.onnx> --label "..." --device cuda:0
    ... failures --key unetpp_effb7_1024_dinov3_vith16plus_2048_consensus --model <v8.onnx> \
        --baseline unetpp_effb7_1024_dinov3_vith16plus_2048=<v2.onnx> --picks-from <v2 docs>/failure_provenance.json \
        --in <dir> --out <docs dir> --manifest <manifest.csv> --label "..." --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import glob
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageDraw, ImageFont

from _common import ROOT, device
from leafmachine3.core.config import Config
from leafmachine3.core.imaging import configure_pillow, read_image
from leafmachine3.core.paper_removal import (
    METHOD_HSV, METHOD_LLR_HYST_KEEPBLOBS, METHOD_NONE, paperclean_method_from_cfg, run_paperclean,
)
from leafmachine3.inference.specimen_models import SPECIMEN_MODELS
from leafmachine3.inference.specimen_segmenter import build_specimen_segmenter

Image = configure_pillow()

V1_KEY = "unetpp_effb7_1024"
V2_KEY = "unetpp_effb7_1024_dinov3_vith16plus_2048"
V8_KEY = "unetpp_effb7_1024_dinov3_vith16plus_2048_consensus"
V9_KEY = "unetpp_effb7_1024_dinov3_vith16plus_2048_selftrain"
NEW_KEY = V2_KEY                    # default --key: the v2 card's renders stay reproducible
V1_DOCS = ROOT / "docs" / "hf_docs" / "specimen_segmenter"
DATASETS = Path("/datac/Labelbox_Dump/datasets")
IMAGE_GLOBS = ("LM3_Specimen_Segmentation-specimen-unetMasksFromSam3-paperclean/images/*/{stem}.jpg",
               "LM3_Specimen_Segmentation-dense-weak-bbox-from-plant-archival/images/*/{stem}.jpg")
NAMES = {V1_KEY: "UNet++ v1", V2_KEY: "UNet++ v2 (DINOv3)", V8_KEY: "UNet++ v8 (DINOv3, consensus)",
         V9_KEY: "UNet++ v9 (self-trained)"}
H, PAD, BAR, JPEG_QUALITY = 1000, 12, 76, 88
REMOVED_RGB = (235, 60, 60)                                   # reporting/palette.py refined_color (paperclean red)


def font(size: int):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


F, FS = font(25), font(19)


def cutout(img: np.ndarray, keep: np.ndarray, removed: np.ndarray | None = None) -> np.ndarray:
    """Kept pixels on black; pixels paper removal deleted, if given, tinted red."""
    out = np.zeros_like(img)
    out[keep] = img[keep]
    if removed is not None and removed.any():
        red = np.array(REMOVED_RGB[::-1], np.float32)
        out[removed] = (img[removed].astype(np.float32) * 0.45 + red * 0.55).astype(np.uint8)
    return out


def panel(img: np.ndarray, tiles: list[tuple[str, np.ndarray, str]], path: Path, rows: int = 1) -> None:
    """[Original | tiles...] at height H on a dark canvas (v1 card style); title, then the note on a second line.
    rows > 1 wraps the tiles into a grid (row by row, the last row padded), so many models stay legible."""
    h, w = img.shape[:2]
    s = H / h
    pw = int(round(w * s))
    cols = [("Original", img, "")] + tiles
    per = -(-len(cols) // max(1, rows))
    nrow = -(-len(cols) // per)
    canvas = Image.new("RGB", (per * pw + (per + 1) * PAD, nrow * (H + BAR + PAD) + PAD), (18, 18, 18))
    dr = ImageDraw.Draw(canvas)
    for j, (title, im, note) in enumerate(cols):
        r = cv2.resize(im, (pw, H), interpolation=cv2.INTER_AREA)
        x = PAD + (j % per) * (pw + PAD)
        y = PAD + (j // per) * (H + BAR + PAD)
        canvas.paste(Image.fromarray(cv2.cvtColor(r, cv2.COLOR_BGR2RGB)), (x, y))
        dr.text((x + 4, y + H + 6), title, fill=(240, 240, 240), font=F)
        if note:
            dr.text((x + 4, y + H + 42), note, fill=(170, 170, 170), font=FS)
    canvas.save(path, quality=JPEG_QUALITY)


def pct(m: np.ndarray) -> float:
    return round(float(m.mean()) * 100, 1)


def split_name(stem: str) -> str:
    return stem.split("__", 1)[1] if "__" in stem else stem


def spec_for(key: str):
    """The registry spec, or -- for a key being released before LM3 registers it -- the v2 spec under that name."""
    if key in SPECIMEN_MODELS:
        return SPECIMEN_MODELS[key]
    print(f"note: {key} is not in LM3's registry yet; running it with the {V2_KEY} workflow", flush=True)
    return dataclasses.replace(SPECIMEN_MODELS[V2_KEY], key=key, label=NAMES.get(key, key))


def load_models(keys_paths: dict[str, str], dev: str) -> dict:
    prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if dev.startswith("cuda") else ["CPUExecutionProvider"]
    return {k: build_specimen_segmenter(spec_for(k), p, providers=prov, conf=0.5, paperclean=False)
            for k, p in keys_paths.items()}


def baselines(a) -> dict[str, str]:
    """--baseline KEY=onnx entries (and the legacy --v1-model) in order."""
    out = {}
    if getattr(a, "v1_model", None):
        out[V1_KEY] = a.v1_model
    for b in a.baseline or []:
        k, p = b.split("=", 1)
        out[k] = p
    return out


def method_for(choice: str, settings: str) -> str:
    """--paperclean llr|hsv, or 'settings' = what modules.specimen_segmenter selects (LLR when the pass is off there)."""
    if choice == "llr":
        return METHOD_LLR_HYST_KEEPBLOBS
    if choice == "hsv":
        return METHOD_HSV
    sc = Config.load(settings).stage("specimen_segmenter")
    m = paperclean_method_from_cfg(bool(sc.get("paperclean", True)), bool(sc.get("LLR_Hyst_KeepBlobs", True)))
    return METHOD_LLR_HYST_KEEPBLOBS if m == METHOD_NONE else m


def working_image(path: Path, max_dim: int, quality: int, tmp: Path) -> np.ndarray:
    """The image an LM3 run segments: core/ingest.py ``_normalize`` for non-JPEG or oversized originals, else the file."""
    with Image.open(path) as im:
        if path.suffix.lower() in (".jpg", ".jpeg") and max(im.size) <= max_dim:
            return read_image(path)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max(im.size) > max_dim:
            sc = max_dim / float(max(im.size))
            im = im.resize((max(1, round(im.width * sc)), max(1, round(im.height * sc))), Image.LANCZOS)
        dst = tmp / f"{path.stem}_tmp.jpg"
        im.save(dst, format="JPEG", quality=quality)
    return read_image(dst)


def examples(a) -> None:
    v1_prov = json.loads((V1_DOCS / "compare_provenance.json").read_text())
    docs = Path(a.docs or V1_DOCS / a.key)
    docs.mkdir(parents=True, exist_ok=True)
    base = baselines(a)
    if not a.v1_masks and not base:
        raise SystemExit("give --v1-masks and/or --baseline KEY=model.onnx: the panels compare against an earlier model")
    models = load_models(base | {a.key: a.model}, a.device)
    order = ([V1_KEY] if a.v1_masks else []) + list(base) + [a.key]
    method = method_for(a.paperclean, a.settings)
    recs = []
    for k, ex in enumerate(v1_prov["examples"], 1):
        stem = ex["stem"]
        src = next((p for g in IMAGE_GLOBS for p in glob.glob(str(DATASETS / g.format(stem=stem)))), None)
        mp = next((Path(d) / f"{stem}.png" for d in a.v1_masks if (Path(d) / f"{stem}.png").exists()), None)
        if src is None or (a.v1_masks and mp is None):
            raise SystemExit(f"{stem}: image {src} / v1 mask {mp} not found")
        img = read_image(src)
        h, w = img.shape[:2]
        masks = {}
        if a.v1_masks:
            m1 = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if m1.shape != (h, w):
                m1 = cv2.resize(m1, (w, h), interpolation=cv2.INTER_NEAREST)
            masks[V1_KEY] = m1 > 127
            pub = ex["percent_of_sheet_kept"]["UNet++"]
            if abs(pct(masks[V1_KEY]) - pub) > 0.05:
                raise SystemExit(f"{stem}: saved v1 mask keeps {pct(masks[V1_KEY])}% but the published card says {pub}%")
        for key, m in models.items():
            masks[key] = m.mask(img) > 0
        kept = {key: pct(masks[key]) for key in order}
        tiles = []
        for key in order:
            tiles.append((NAMES.get(key, key), cutout(img, masks[key]), f"{kept[key]}% kept"))
            if key in a.paperclean_after:
                f = run_paperclean(img, masks[key].astype(np.uint8), method)[0] > 0
                rm = masks[key] & ~f
                kept[f"{key}+paperclean"] = pct(f)
                tiles.append((f"{NAMES.get(key, key)} + paper removal", cutout(img, f, rm), f"{pct(f)}% kept, {pct(rm)}% removed"))
        name = f"compare_{k:02d}_{ex['dataset_split']}_{ex['specimen']}.jpg"
        panel(img, tiles, docs / name, rows=a.rows)
        rec = {"file": name, "stem": stem, "specimen": ex["specimen"], "project_key": ex["project_key"],
               "dataset_split": ex["dataset_split"], "image_size": [w, h]}
        if a.v1_masks:
            rec["v1_mask"] = str(mp)
        recs.append(rec | {"percent_of_sheet_kept": kept, "selection": ex.get("selection")})
        print(f"{name}: " + "  ".join(f"{NAMES.get(key, key)} {v:5.1f}%" for key, v in kept.items()), flush=True)
    cols = " | ".join(NAMES.get(key, key) for key in order)
    src_note = []
    if a.v1_masks:
        src_note.append("v1 = the saved masks behind that card (not re-run)")
    src_note.append("the others = the exported ONNX through LM3's specimen backend on the same input (the dataset image at "
                    "native resolution)")
    pc_note = (f" Tiles marked '+ paper removal' are the preceding model's mask after LM3's paper removal ({method}); red = "
               "what it deleted." if a.paperclean_after else "")
    prov = {"what": f"Background removal on the 20 sheets of the unetpp_effb7_1024 card: Original | {cols}; black = removed. "
                    f"Raw model masks unless marked.{pc_note} {'; '.join(src_note)}. "
                    "Rendered by tools/release/model_card_assets/specimen_segmenter.py examples.",
            "models": ({V1_KEY: "unet/runs/unet_masksFromSam3_paperclean_control_1024/best.pt (saved masks)"} if a.v1_masks else {})
            | dict(base) | {a.key: a.label},
            "rendered": dt.date.today().isoformat(), "examples": recs}
    (docs / "compare_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


def failures(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(a.settings)
    max_dim, quality = int(cfg.ingest.max_working_dim), int(cfg.ingest.jpg_quality)
    method = method_for(a.paperclean, a.settings)
    splits = {}
    if a.manifest:
        for r in csv.DictReader(open(a.manifest)):
            splits[r["stem"]] = r.get("dense_weak_split") or r.get("split") or ""
    base_models = baselines(a)
    if not base_models:
        raise SystemExit("give --baseline KEY=model.onnx (or --v1-model): the panels compare against an earlier model")
    models = load_models(base_models | {a.key: a.model}, a.device)
    images = {p.stem: p for p in sorted(Path(a.inp).iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")}
    # (stem, group, panel kinds): every image, or exactly the earlier card's picks with the same panels
    src = json.loads(Path(a.picks_from).read_text()) if a.picks_from else None
    if src is None:
        jobs = [(stem, "examples", ("raw", "pc")) for stem in images]
    else:
        jobs = [(e["stem"], g, tuple(k for k, suf in (("raw", "__raw_masks.jpg"), ("pc", "__with_paper_removal.jpg"))
                                    if any(f.endswith(suf) for f in e["files"])))
                for g in ("examples", "paper_removal_extra") for e in src.get(g, [])]
        missing = [s for s, _, _ in jobs if s not in images]
        if missing:
            raise SystemExit(f"picks not found in --in: {missing}")
    recs: dict[str, list] = {"examples": [], "paper_removal_extra": []}
    with tempfile.TemporaryDirectory(prefix="lm3_specseg_card_") as td:
        for stem, group, kinds in jobs:
            img = working_image(images[stem], max_dim, quality, Path(td))
            raw, final, removed = {}, {}, {}
            for k, m in models.items():
                raw[k] = m.mask(img)
                f, r, _ = run_paperclean(img, raw[k], method)
                final[k], removed[k] = f > 0, r > 0
                raw[k] = raw[k] > 0
            split = splits.get(stem, "")
            base = f"failure_{split + '_' if split else ''}{split_name(stem)}"
            files = []
            if "raw" in kinds:
                panel(img, [(NAMES.get(k, k), cutout(img, raw[k]), f"{pct(raw[k])}% kept") for k in models],
                      out / f"{base}__raw_masks.jpg", rows=a.rows)
                files.append(f"{base}__raw_masks.jpg")
            if "pc" in kinds:
                panel(img, [(f"{NAMES.get(k, k)} + paper removal", cutout(img, final[k], removed[k]),
                             f"{pct(final[k])}% kept, {pct(removed[k])}% removed") for k in models],
                      out / f"{base}__with_paper_removal.jpg", rows=a.rows)
                files.append(f"{base}__with_paper_removal.jpg")
            rec = {"stem": stem, "split": split, "files": files,
                   "working_image_size": [img.shape[1], img.shape[0]],
                   "percent_of_sheet_kept": {k: {"raw": pct(raw[k]), "with_paper_removal": pct(final[k]),
                                                 "removed_by_paper_removal": pct(removed[k])} for k in models}}
            recs[group].append(rec)
            print(f"{base}: " + "  ".join(f"{NAMES.get(k, k)} {v['raw']:5.1f}% -> {v['with_paper_removal']:5.1f}%"
                                          for k, v in rec["percent_of_sheet_kept"].items()), flush=True)
    cols = " | ".join(NAMES.get(k, k) for k in models)
    prov = {"what": f"Sheets the v1 UNet++ failed on. __raw_masks: Original | {cols} (model masks, black = removed). "
                    "__with_paper_removal: the same after LM3's paper-removal follow-up (red = deleted by it). "
                    "Every model runs as in an LM3 run: the ingest working image, the stage's backend, paperclean. "
                    "Rendered by tools/release/model_card_assets/specimen_segmenter.py failures.",
            "models": dict(base_models) | {a.key: a.label}, "paperclean_method": method,
            "input": str(Path(a.inp).resolve()),
            "ingest": {"max_working_dim": max_dim, "jpg_quality": quality},
            "rendered": dt.date.today().isoformat(), "examples": recs["examples"]}
    if src is not None:
        prov["selection"] = src.get("selection")
        if recs["paper_removal_extra"]:
            prov["paper_removal_extra"] = recs["paper_removal_extra"]
            prov["selection_paper_removal_extra"] = src.get("selection_paper_removal_extra")
        prov["picks_from"] = str(Path(a.picks_from).resolve())
    (out / "failure_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("examples", "failures"):
        p = sub.add_parser(name)
        p.add_argument("--key", default=NEW_KEY, help="model key being released (default: the v2 key)")
        p.add_argument("--model", required=True, help="ONNX of --key")
        p.add_argument("--label", required=True, help="model description written into the provenance file")
        p.add_argument("--baseline", action="append", default=[], metavar="KEY=ONNX", help="earlier model to show (repeat)")
        p.add_argument("--device", default="cpu", help="cpu or cuda:N")
        p.add_argument("--rows", type=int, default=1, help="wrap the tiles into this many rows")
        p.add_argument("--settings", default=str(ROOT / "LM3_settings.yaml"), help="ingest sizes and the paper-removal method")
        p.add_argument("--paperclean", default="settings", choices=("settings", "llr", "hsv"),
                       help="paper-removal method (settings = what --settings selects; LLR_Hyst_KeepBlobs by default)")
    ex = sub.choices["examples"]
    ex.add_argument("--v1-masks", nargs="+", default=[], help="folders holding the v1 card's saved <stem>.png masks")
    ex.add_argument("--docs", default="", help="output folder (default docs/hf_docs/specimen_segmenter/<key>)")
    ex.add_argument("--paperclean-after", nargs="*", default=[], metavar="KEY",
                    help="add a '+ paper removal' tile after each of these models' raw tile")
    fa = sub.choices["failures"]
    fa.add_argument("--v1-model", default="", help=f"{V1_KEY} ONNX (same as --baseline {V1_KEY}=...)")
    fa.add_argument("--picks-from", default="", help="an earlier card's failure_provenance.json: render only its picks")

    fa.add_argument("--in", dest="inp", required=True)
    fa.add_argument("--out", required=True)
    fa.add_argument("--manifest", default="", help="CSV with stem + dense_weak_split/split columns (file names carry the split)")

    a = ap.parse_args()
    device(a.device)
    examples(a) if a.cmd == "examples" else failures(a)


if __name__ == "__main__":
    main()
