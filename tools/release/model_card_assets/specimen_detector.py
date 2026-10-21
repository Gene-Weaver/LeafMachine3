#!/usr/bin/env python3
"""Render the specimen_detector model-card examples: predicted vs hand-drawn Specimen boxes on test sheets.

The Specimen detector is not a LeafMachine3 pipeline stage (it built the dense weak labels behind the
specimen-segmenter training data), so there is no stage loader or Reporter style to reuse. Each panel shows
one test-split sheet of datasets/LM3_Plant_Specimen with the hand-drawn Specimen boxes (white, dashed) and the
model's boxes from the exported ONNX at the confidence the dataset build used (cyan, solid, labeled with the
confidence), shown at 1600 px. Sheets are picked at random with a fixed seed, one per institution, herbarium
sheets only (stems of the form <project>__<INST>_<id>_<Family>_<Genus>_<species>), at least two of them with
several Specimen boxes. Writes docs/hf_docs/specimen_detector/overlay_NN_<sheet>.jpg + overlay_provenance.json.

Runs in the training env (ultralytics):
    .venv_sync_labelbox/bin/python LM3/tools/release/model_card_assets/specimen_detector.py \
        --model LM3_Specimen_Detector/models/yolo26n_det_640/exported_last/model.onnx --label "yolo26n_det_640 last.pt (epoch 300)"
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import random
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]                      # the LM3 checkout
DOCS = ROOT / "docs" / "hf_docs" / "specimen_detector"
DATA = Path("/datac/Labelbox_Dump/datasets/LM3_Plant_Specimen")
SHEET = re.compile(r"^[a-z0-9]{8}__([A-Z]+)_(\d+)_([A-Z][a-z]+aceae)_([A-Z][a-z]+)_([a-z-]+)")
LONG_SIDE, JPEG_QUALITY = 1600, 88
CYAN, WHITE, BLACK = (255, 255, 0), (255, 255, 255), (0, 0, 0)


def gt_boxes(stem: str, w: int, h: int) -> list[tuple[float, float, float, float]]:
    p = DATA / "labels" / "test" / f"{stem}.txt"
    out = []
    for ln in p.read_text().splitlines() if p.exists() else []:
        _, cx, cy, bw, bh = map(float, ln.split()[:5])
        out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def dashed_rect(img, x1, y1, x2, y2, color, t, dash):
    for (ax, ay), (bx, by) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(bx - ax, by - ay) // dash))
        for k in range(0, n, 2):
            s, e = k / n, min(1.0, (k + 1) / n)
            cv2.line(img, (int(ax + (bx - ax) * s), int(ay + (by - ay) * s)),
                     (int(ax + (bx - ax) * e), int(ay + (by - ay) * e)), color, t, cv2.LINE_AA)


def render(img, gts, preds, scale):
    t = max(2, round(4 * scale))
    for x1, y1, x2, y2 in gts:                                # hand-drawn: white dashed with a dark halo
        dashed_rect(img, *(int(v) for v in (x1, y1, x2, y2)), BLACK, t + 2, 28 * scale)
        dashed_rect(img, *(int(v) for v in (x1, y1, x2, y2)), WHITE, t, 28 * scale)
    fs = 1.1 * scale
    for (x1, y1, x2, y2), c in preds:                         # model: cyan solid + confidence label
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), CYAN, t, cv2.LINE_AA)
        label = f"Specimen {c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, max(1, t // 2))
        y0 = int(y1) if int(y1) - th - 8 < 0 else int(y1) - th - 8
        cv2.rectangle(img, (int(x1), y0), (int(x1) + tw + 8, y0 + th + 8), CYAN, -1)
        cv2.putText(img, label, (int(x1) + 4, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, BLACK, max(1, t // 2), cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="Specimen detector ONNX")
    ap.add_argument("--label", required=True, help="model description written into the provenance file")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--docs", default=str(DOCS))
    a = ap.parse_args()
    from ultralytics import YOLO
    docs = Path(a.docs)
    docs.mkdir(parents=True, exist_ok=True)
    for old in docs.glob("overlay_*.jpg"):
        old.unlink()

    cands = [Path(p) for p in sorted(glob.glob(str(DATA / "images" / "test" / "*.jpg")))]
    cands = [p for p in cands if SHEET.match(p.stem)]
    random.Random(a.seed).shuffle(cands)
    multi = [p for p in cands if len(gt_boxes(p.stem, 1, 1)) > 1]
    picks, inst = [], set()
    for p in multi[:20] + cands:                              # a few multi-specimen sheets first, then the rest
        i = SHEET.match(p.stem).group(1)
        if i in inst:
            continue
        if p in multi[:20] and sum(q in multi for q in picks) >= 2:
            continue
        picks.append(p); inst.add(i)
        if len(picks) == a.n:
            break

    model = YOLO(a.model, task="detect")
    examples = []
    for k, p in enumerate(picks, 1):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        r = model.predict(str(p), conf=a.conf, imgsz=640, verbose=False)[0]
        preds = [(tuple(b), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist())]
        gts = gt_boxes(p.stem, w, h)
        out = render(img.copy(), gts, preds, max(h, w) / 3000)
        s = LONG_SIDE / max(h, w)
        if s < 1:
            out = cv2.resize(out, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        m = SHEET.match(p.stem)
        name = f"overlay_{k:02d}_{p.stem.split('__', 1)[1]}.jpg"
        cv2.imwrite(str(docs / name), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        examples.append({"file": name, "image": p.stem, "split": "test", "institution": m.group(1),
                         "accession": f"{m.group(1)}_{m.group(2)}", "species": f"{m.group(4)} {m.group(5)}",
                         "hand_drawn_boxes": len(gts), "predicted_boxes": len(preds),
                         "confidences": [round(c, 3) for _, c in preds], "bytes": (docs / name).stat().st_size})
        print(f"{name}: {len(gts)} hand-drawn, {len(preds)} predicted", flush=True)
    prov = {"what": "Specimen detector boxes (cyan, solid, confidence) vs hand-drawn Specimen boxes (white, dashed) on "
                    "test-split sheets of datasets/LM3_Plant_Specimen, rendered by "
                    "tools/release/model_card_assets/specimen_detector.py.",
            "model": a.label, "conf": a.conf, "seed": a.seed, "rendered": dt.date.today().isoformat(), "examples": examples}
    (docs / "overlay_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")


if __name__ == "__main__":
    main()
