#!/usr/bin/env python3
"""Evaluate the landmark-as-class detector ("YOLO hack", e.g. yolo26n_det_640) and render its card assets.

Writes the metrics JSON the card's tables come from, and docs/hf_docs/landmark_detector/<key>/:
  * per-split instance histogram of the dataset;
  * Ultralytics val (box mAP, per-class AP50, speed) on val and test;
  * a point metric on test: a predicted point is correct if it lies within r (2.5% / 5% of the crop's long
    side) of an unmatched labeled point of the same class, greedy by confidence, swept over conf thresholds
    (this model's confidences top out near 0.2, so the usual 0.25 returns almost nothing);
  * overlay_NN_<genus>.jpg: predicted points at the best-F1 threshold on 10 test crops (one per genus,
    herbarium-named, >= 20 labeled points, long side >= 400 px), plus legend.png and overlay_provenance.json.

    python tools/release/model_card_assets/landmark_detect_points.py --weights <best.pt> \
        --dataset <datasets/LM3_Landmarks_Detect> --metrics-out <json> [--docs <dir>] [--device 0] [--skip-val]
Needs ultralytics (the training env), not the LM3 runtime.
"""
import argparse, json, os, random, re, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description="Evaluate + render the landmark detect model's card assets")
ap.add_argument("--weights", required=True)
ap.add_argument("--dataset", required=True, help="YOLO dataset dir with data.yaml, images/, labels/")
ap.add_argument("--metrics-out", required=True)
ap.add_argument("--docs", default=str(Path(__file__).resolve().parents[3] / "docs" / "hf_docs" / "landmark_detector" / "yolo26n_det_640"))
ap.add_argument("--device", default="0")
ap.add_argument("--skip-val", action="store_true", help="reuse the *_box metrics already in --metrics-out")
a = ap.parse_args()
DS, OUT_JSON, DOCS = a.dataset, a.metrics_out, a.docs
VAL_DIR = os.path.join(os.path.dirname(os.path.abspath(OUT_JSON)), "val_runs")
NAMES = ["Lamina Tip", "Lamina Base", "Petiole Tip", "Lobe Tip", "Width", "Midvein Trace",
         "Petiole Trace", "Apex Angle", "Base Angle"]
COLORS = ["#e6194b", "#3cb44b", "#ffe119", "#f58231", "#911eb4", "#4363d8", "#46f0f0", "#f032e6", "#bcf60c"]
CONF, IMGSZ = 0.01, 640
THRESHOLDS = (0.01, 0.02, 0.03, 0.05, 0.08)
RADII = (0.025, 0.05)          # point match radius, fraction of the image's long side

from ultralytics import YOLO

model = YOLO(a.weights)
prev = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {}
res = {"model": "yolo26n_det_640", "iteration": 2, "weights": a.weights,
       "imgsz": IMGSZ, "classes": NAMES}

# ---- instance histogram per split ----
hist = {}
for split in ("train", "val", "test"):
    c = np.zeros(len(NAMES), int); n = 0
    d = f"{DS}/labels/{split}"
    for f in os.listdir(d):
        n += 1
        for line in open(f"{d}/{f}"):
            if line.strip():
                c[int(line.split()[0])] += 1
    hist[split] = {"images": n, "instances": dict(zip(NAMES, c.tolist()))}
res["dataset"] = hist
res.update({k: v for k, v in prev.items() if k.endswith('_box')})

# ---- Ultralytics val (box metrics, per-class AP50, speed) ----
for split in (() if a.skip_val else ("val", "test")):
    m = model.val(data=f"{DS}/data.yaml", split=split, imgsz=IMGSZ, batch=32, device=a.device, plots=False,
                  verbose=False, project=VAL_DIR, name=split, exist_ok=True)
    b = m.box
    res[f"{split}_box"] = {"map50": float(b.map50), "map50_95": float(b.map), "precision": float(b.mp),
                           "recall": float(b.mr),
                           "ap50_per_class": {NAMES[i]: float(b.ap50[j]) for j, i in enumerate(b.ap_class_index)},
                           "speed_ms": {k: float(v) for k, v in m.speed.items()}}
    print(split, res[f"{split}_box"]["map50"], m.speed, flush=True)

# ---- point-level metric on test: a prediction is a hit if within r of an unmatched same-class GT point ----
imgs = sorted(os.listdir(f"{DS}/images/test"))
preds_by_img, gts_by_img, shapes = {}, {}, {}
for i in range(0, len(imgs), 64):
    batch = [f"{DS}/images/test/{f}" for f in imgs[i:i + 64]]
    for f, r in zip(imgs[i:i + 64], model.predict(batch, imgsz=IMGSZ, conf=CONF, device=a.device, verbose=False)):
        h, w = r.orig_shape; shapes[f] = (w, h)
        preds_by_img[f] = [(int(c), float(x), float(y), float(s_)) for (x, y, _, _), c, s_ in
                           zip(r.boxes.xywh.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        G = []
        lab = f"{DS}/labels/test/{os.path.splitext(f)[0]}.txt"
        if os.path.exists(lab):
            for line in open(lab):
                if line.strip():
                    c, x, y = line.split()[:3]; G.append((int(c), float(x) * w, float(y) * h))
        gts_by_img[f] = G

def score(thr, rad):
    tp = np.zeros(len(NAMES)); npred = np.zeros(len(NAMES)); ngt = np.zeros(len(NAMES))
    for f in imgs:
        P = [p for p in preds_by_img[f] if p[3] >= thr]; G = gts_by_img[f]; R = rad * max(shapes[f])
        for c, *_ in P: npred[c] += 1
        for c, *_ in G: ngt[c] += 1
        for c in range(len(NAMES)):
            gc = [g for g in G if g[0] == c]; used = [False] * len(gc)
            for _, px, py, _ in sorted([p for p in P if p[0] == c], key=lambda p: -p[3]):
                best, bd = -1, R
                for j, (_, gx, gy) in enumerate(gc):
                    d = ((px - gx) ** 2 + (py - gy) ** 2) ** 0.5
                    if not used[j] and d <= bd: best, bd = j, d
                if best >= 0: used[best] = True; tp[c] += 1
    def prf(t, p, g):
        pr = t / p if p else 0.0; rc = t / g if g else 0.0
        return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}
    out = prf(tp.sum(), npred.sum(), ngt.sum()); out["preds_per_image"] = npred.sum() / len(imgs)
    out["per_class"] = {n: prf(tp[c], npred[c], ngt[c]) for c, n in enumerate(NAMES)}
    return out

sweep = {f"conf{t}_r{r}": score(t, r) for t in THRESHOLDS for r in RADII}
best_thr = max(THRESHOLDS, key=lambda t: sweep[f"conf{t}_r{RADII[0]}"]["f1"])
res["test_points"] = {"radius_frac_of_long_side": list(RADII), "n_images": len(imgs),
                      "gt_per_image": sum(len(g) for g in gts_by_img.values()) / len(imgs),
                      "best_conf_by_f1": best_thr, "sweep": sweep}
for k, v in sweep.items(): print(k, round(v["precision"], 3), round(v["recall"], 3), round(v["f1"], 3), round(v["preds_per_image"], 1))
preds_by_img = {f: [p for p in P if p[3] >= best_thr] for f, P in preds_by_img.items()}
os.makedirs(os.path.dirname(os.path.abspath(OUT_JSON)), exist_ok=True)
json.dump(res, open(OUT_JSON, "w"), indent=1)

# ---- examples: 10 test crops, one per genus, herbarium-named, >= 20 GT points, long side >= 400 ----
pat = re.compile(r"__(?:Leaf_WHOLE__)?([A-Z]+)_(\d+)_[A-Z][a-z]+_([A-Z][a-z]+)_([a-z-]+)__")
cands = []
for f in imgs:
    m = pat.search(f)
    lab = f"{DS}/labels/test/{os.path.splitext(f)[0]}.txt"
    if not m or not os.path.exists(lab):
        continue
    n = sum(1 for l in open(lab) if l.strip())
    w, h = Image.open(f"{DS}/images/test/{f}").size
    if n >= 20 and max(w, h) >= 400:
        cands.append((f, m.group(3), m.group(4), f"{m.group(1)}_{m.group(2)}"))
random.seed(7); random.shuffle(cands)
picked, seen = [], set()
for c in cands:
    if c[1] not in seen:
        seen.add(c[1]); picked.append(c)
    if len(picked) == 10:
        break
os.makedirs(DOCS, exist_ok=True)
prov = {"model": os.path.abspath(a.weights), "conf": best_thr, "imgsz": IMGSZ, "split": "test",
        "dataset": "datasets/LM3_Landmarks_Detect", "examples": []}
for k, (f, genus, sp, acc) in enumerate(picked, 1):
    im = Image.open(f"{DS}/images/test/{f}").convert("RGB")
    s = 900 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    d = ImageDraw.Draw(im); rad = max(4, round(0.009 * max(im.size)))
    for c, x, y, _ in sorted(preds_by_img[f], key=lambda p: p[0] in (5, 6)):   # traces underneath
        x, y = x * s, y * s
        d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=COLORS[c], outline="white", width=2)
    name = f"overlay_{k:02d}_{genus}.jpg"
    im.save(f"{DOCS}/{name}", quality=92)
    prov["examples"].append({"file": name, "source": f, "species": f"{genus} {sp}", "accession": acc,
                             "n_pred": len(preds_by_img[f])})
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
leg = Image.new("RGB", (len(NAMES) * 190, 44), "white"); d = ImageDraw.Draw(leg)
for i, (n, c) in enumerate(zip(NAMES, COLORS)):
    x = 12 + i * 190
    d.ellipse([x, 12, x + 20, 32], fill=c, outline="#555"); d.text((x + 28, 9), n, fill="black", font=font)
leg.save(f"{DOCS}/legend.png")
json.dump(prov, open(f"{DOCS}/overlay_provenance.json", "w"), indent=1)
print("rendered", [p["file"] for p in prov["examples"]])
