#!/usr/bin/env python3
"""Render the example panels for the two BiRefNet leaf models (leaf_segmenter alternates, not LM3 stages).

One panel per leaf crop, four columns: the crop | the lamina + petiole model's mask | the lamina-only model's
mask | the two combined (lamina from the lamina-only model, petiole = the bulk of what the lamina + petiole
model adds, after a morphological opening that drops the thin ring where the two models' edges disagree;
the method of LM3_Leaf_Segmentation/birefnet_compare_to_yolo26/mask_utils.bulk_petiole). Masks use the
LeafMachine3 segmentation palette (Leaf green, Petiole magenta). The crops are the plant_detector Leaf_WHOLE
crops the leaf_segmenter card uses (docs/hf_docs/leaf_segmenter/io_provenance.json) plus the most confident
large Leaf_WHOLE crop of four more specimens from that card's overlay set, read from COPIES of the example
runs' databases. Both models run their export contract: RGB resized to 1024 x 1024, /255, ONNX -> probability,
resized back to the crop, threshold 0.5. Small crops are drawn at native size and shown at a long side of 600.

Writes docs/hf_docs/leaf_segmenter/birefnet/panel_NN_<specimen>.jpg + panel_provenance.json; a later run
re-renders the crops listed in that provenance file.

    .venv_LM3/bin/python tools/release/model_card_assets/birefnet_leaf.py \
        --leaf-model <lamina+petiole model.onnx> --lamina-model <lamina-only model.onnx> --label "<models, iteration>" [--device cpu]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np

from _common import ROOT, Runs

DOCS = ROOT / "docs" / "hf_docs" / "leaf_segmenter" / "birefnet"
LEAF_DOCS = ROOT / "docs" / "hf_docs" / "leaf_segmenter"
EXTRA = ["VT_2573061663_Ulmaceae_Ulmus_americana", "CM_2859339684_Salicaceae_Populus_grandidentata",
         "NY_1928597599_Rosaceae_Crataegus_douglasii", "MNHN_731826796_Fagaceae_Quercus_rosacea"]
SIZE, MIN_LONG_SIDE, JPEG_QUALITY, ALPHA = 1024, 600, 92, 0.45
LEAF_BGR, PETIOLE_BGR = (0, 255, 46), (150, 0, 255)          # palette Leaf (46,255,0) / Petiole (255,0,150) RGB
TITLES = ("input crop", "lamina + petiole model", "lamina-only model", "combined: lamina | petiole")


class Model:
    def __init__(self, path: str, device: str):
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=prov)

    def mask(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        x = np.ascontiguousarray(np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None])
        p = self.sess.run(None, {"input_rgb01": x})[0].reshape(SIZE, SIZE)
        return cv2.resize(p, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5


def bulk_petiole(leaf, lamina, open_radius=4, min_area_frac=8e-4, close_radius=2):
    raw = (leaf & ~lamina).astype(np.uint8)
    if not raw.any():
        return np.zeros_like(leaf)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
    core = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.zeros_like(core)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(64, int(min_area_frac * leaf.size)):
            keep[lab == i] = 1
    if close_radius and keep.any():
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
        keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, kc)
    return keep.astype(bool) & leaf


def paint(img, layers):
    out = img.astype(np.float32)
    for m, color in layers:
        out[m] = out[m] * (1 - ALPHA) + np.array(color, np.float32) * ALPHA
    out = out.astype(np.uint8)
    for m, color in layers:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, color, 1, cv2.LINE_AA)
    return out


def panel(tiles):
    h, w = tiles[0].shape[:2]
    band, gap = 40, 8
    cols = []
    for t, title in zip(tiles, TITLES):
        head = np.full((band, w, 3), 34, np.uint8)
        fs = 0.6 * max(1.0, w / 600)
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.putText(head, title, (max(4, (w - tw) // 2), (band + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (235, 235, 235), 1, cv2.LINE_AA)
        cols += [np.vstack([head, t]), np.full((h + band, gap, 3), 255, np.uint8)]
    return np.hstack(cols[:-1])


def pick_extra(runs: Runs, specimen: str, source: str) -> dict:
    db = runs.db(source)
    sid, _ = runs.specimen(db, specimen)
    conf = {int(r["detection_id"]): float(r["conf"]) for r in
            db._query("SELECT detection_id, conf FROM plant_detection WHERE specimen_id = ?", (sid,))}
    crops = [c for c in db.crops("plant_detection", sid, cls_in=("Leaf_WHOLE",)) if conf[c.detection_id] >= 0.7]
    c = max(crops, key=lambda c: (c.x2 - c.x1) * (c.y2 - c.y1))
    return {"specimen": specimen, "detection_id": c.detection_id, "source_run": source,
            "plant_detector_conf": round(conf[c.detection_id], 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leaf-model", required=True, help="lamina + petiole BiRefNet ONNX")
    ap.add_argument("--lamina-model", required=True, help="lamina-only BiRefNet ONNX")
    ap.add_argument("--label", required=True, help="models description written into the provenance file")
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N")
    ap.add_argument("--docs", default=str(DOCS))
    a = ap.parse_args()
    docs = Path(a.docs)
    docs.mkdir(parents=True, exist_ok=True)
    prov_path = docs / "panel_provenance.json"
    runs = Runs()
    try:
        if prov_path.exists():
            examples = json.loads(prov_path.read_text())["examples"]
        else:                                                   # first render: the leaf card's crops + 4 more
            io = json.loads((LEAF_DOCS / "io_provenance.json").read_text())["examples"]
            ov = {e["specimen"]: e["source_run"] for e in json.loads((LEAF_DOCS / "overlay_provenance.json").read_text())["examples"]}
            examples = [{k: e[k] for k in ("specimen", "detection_id", "source_run", "plant_detector_conf")} for e in io]
            examples += [pick_extra(runs, s, ov[s]) for s in EXTRA]
        crops = []
        for ex in examples:
            db = runs.db(ex["source_run"])
            sid, _ = runs.specimen(db, ex["specimen"])
            crop = next(c for c in db.crops("plant_detection", sid, cls_in=("Leaf_WHOLE",)) if c.detection_id == ex["detection_id"])
            crops.append(cv2.imread(crop.crop_path, cv2.IMREAD_COLOR))
        masks = {}
        for key, path in (("leaf", a.leaf_model), ("lamina", a.lamina_model)):   # one 1 GB model in memory at a time
            m = Model(path, a.device)
            masks[key] = [m.mask(c) for c in crops]
            del m
        for k, (ex, img) in enumerate(zip(examples, crops), 1):
            leaf, lamina = masks["leaf"][k - 1], masks["lamina"][k - 1]
            pet = bulk_petiole(leaf, lamina)
            tiles = [img, paint(img, [(leaf, LEAF_BGR)]), paint(img, [(lamina, LEAF_BGR)]),
                     paint(img, [(lamina, LEAF_BGR), (pet, PETIOLE_BGR)])]
            s = MIN_LONG_SIDE / max(img.shape[:2])
            if s > 1:
                size = (round(img.shape[1] * s), round(img.shape[0] * s))
                tiles = [cv2.resize(t, size, interpolation=cv2.INTER_CUBIC) for t in tiles]
            name = f"panel_{k:02d}_{ex['specimen']}.jpg"
            cv2.imwrite(str(docs / name), panel(tiles), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            ex.update(file=name, crop_size_px=[img.shape[1], img.shape[0]],
                      pixels={"lamina_plus_petiole": int(leaf.sum()), "lamina": int(lamina.sum()), "petiole_bulk": int(pet.sum())})
            print(f"{name}: {ex['pixels']}", flush=True)
        prov = {"what": "BiRefNet leaf models on plant_detector Leaf_WHOLE crops: input | lamina + petiole model | lamina-only "
                        "model | combined (petiole = bulk of the difference), rendered by "
                        "tools/release/model_card_assets/birefnet_leaf.py.",
                "model": a.label, "rendered": dt.date.today().isoformat(), "examples": examples}
        prov_path.write_text(json.dumps(prov, indent=1) + "\n")
    finally:
        runs.close()


if __name__ == "__main__":
    main()
