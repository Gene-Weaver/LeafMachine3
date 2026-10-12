#!/usr/bin/env python3
"""Re-render the leaf_segmenter model-card assets with a given model.

Two asset sets live in docs/hf_docs/leaf_segmenter/, each described by a provenance file that names
the specimens, the example run they come from, and (for the I/O pairs) the plant_detector crop:

* ``io_NN_<specimen>_{input_crop,output_masks}.jpg`` (io_provenance.json): the stored ``Leaf_WHOLE``
  crop exactly as LeafMachine3 saved it, and the same crop with the model's Leaf / Petiole / Hole masks.
* ``overlay_NN_<specimen>.jpg`` (overlay_provenance.json): every leaf crop of the specimen segmented
  and drawn on the working image, shown at 1600 px.

Rendering uses the pipeline's own pieces, so the look matches the Reporter's Overlay_Summary:
``load_segmenter`` (the leaf_segmenter stage's model loader and settings from LM3_settings.yaml, with
the model path overridden), the stage's polygon re-basing, and ``build_summary_image`` with masks only
(no boxes, labels, confidences, landmarks or CF banner). The example runs are read from COPIES of
their SQLite files, so the runs themselves are never touched. The provenance files are updated with
the new instance counts and the model that rendered them.

    .venv_LM3/bin/python tools/release/model_card_assets/leaf_segmenter.py \
        --model /path/to/model.onnx --label "yolo26x_seg_1024 iteration 2" [--device cuda:1|cpu]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import replace
from pathlib import Path

import cv2

from _common import ROOT, Runs, device, load_config, published
from leafmachine3.core.imaging import encode_polygon, offset_polygon
from leafmachine3.inference.factory import load_segmenter
from leafmachine3.reporting.overlay import build_summary_image
from leafmachine3.reporting.palette import OverlayStyle

DOCS = ROOT / "docs" / "hf_docs" / "leaf_segmenter"
OVERLAY_LONG_SIDE = 1600
IO_MIN_LONG_SIDE = 600       # small crops are shown at a long side of 600 px (drawn at native size, then scaled)
JPEG_QUALITY = 92


class _Leaf:
    """The three fields build_summary_image reads from a leaf_segmentation row."""
    def __init__(self, cls_name: str, poly):
        self.cls_name, self.mask_format, self.mask_data = cls_name, "polygon_xy", encode_polygon(poly)


def segment(model, crop, offset: bool) -> list[_Leaf]:
    """The leaf_segmenter stage's per-crop step: model.predict + polygon re-basing."""
    out = []
    for inst in model.predict(crop.crop_path):
        poly = offset_polygon(inst.polygon, crop.x1, crop.y1) if offset else inst.polygon
        out.append(_Leaf(inst.cls_name, poly))
    return out


def counts(leaves) -> dict:
    c = {"Leaf": 0, "Petiole": 0, "Hole": 0}
    for leaf in leaves:
        c[leaf.cls_name] = c.get(leaf.cls_name, 0) + 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="leaf segmenter ONNX to render with")
    ap.add_argument("--label", required=True, help="model description written into the provenance files")
    ap.add_argument("--settings", default=str(ROOT / "LM3_settings.yaml"))
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--docs", default=str(DOCS))
    ap.add_argument("--all", action="store_true", help="also render pruned candidates (to pick again)")
    a = ap.parse_args()
    docs = Path(a.docs)

    cfg = load_config(a.settings, "leaf_segmenter", a.model)
    stage = cfg.stage("leaf_segmenter")
    classes = ("Leaf_WHOLE", "Leaf_PARTIAL") if stage.get("include_partial", False) else ("Leaf_WHOLE",)
    model = load_segmenter(cfg, device(a.device))
    style = replace(OverlayStyle.from_config(cfg), draw_labels=False, draw_confidence=False, draw_cf_banner=False)
    runs = Runs()
    stamp = dt.date.today().isoformat()
    try:
        # ---- input/output pairs -------------------------------------------------------------
        io = json.loads((docs / "io_provenance.json").read_text())
        for ex in io["examples"]:
            db = runs.db(ex["source_run"])
            sid, _ = runs.specimen(db, ex["specimen"])
            crop = next(c for c in db.crops("plant_detection", sid, cls_in=classes) if c.detection_id == ex["detection_id"])
            img = cv2.imread(crop.crop_path, cv2.IMREAD_COLOR)
            leaves = segment(model, crop, offset=False)                # crop coordinates
            out = build_summary_image(img, [], leaves, None, style, 1.0)
            k = IO_MIN_LONG_SIDE / max(img.shape[:2])
            if k > 1:
                size = (round(img.shape[1] * k), round(img.shape[0] * k))
                img, out = (cv2.resize(x, size, interpolation=cv2.INTER_CUBIC) for x in (img, out))
            cv2.imwrite(str(docs / ex["input"]), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            cv2.imwrite(str(docs / ex["output"]), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            ex["instances"] = counts(leaves)
            print(f"io  {ex['specimen']}: {ex['instances']}", flush=True)
        io["model"], io["rendered"] = a.label, stamp
        (docs / "io_provenance.json").write_text(json.dumps(io, indent=1) + "\n")

        # ---- whole-specimen overlays -----------------------------------------------------------
        ov = json.loads((docs / "overlay_provenance.json").read_text())
        for ex in ov["examples"]:
            if not published(docs, ov, ex["file"], a.all):
                continue
            db = runs.db(ex["source_run"])
            sid, working = runs.specimen(db, ex["specimen"])
            leaves = [leaf for crop in db.crops("plant_detection", sid, cls_in=classes)
                      for leaf in segment(model, crop, offset=True)]   # working-frame coordinates
            img = cv2.imread(working, cv2.IMREAD_COLOR)
            if img is None:
                print(f"SKIP {ex['file']}: working image missing ({working})", flush=True)
                continue
            out = build_summary_image(img, [], leaves, None, style, 1.0)
            s = OVERLAY_LONG_SIDE / max(out.shape[:2])
            if s < 1:
                out = cv2.resize(out, (round(out.shape[1] * s), round(out.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(docs / ex["file"]), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            ex["instances"] = counts(leaves)
            print(f"ovl {ex['specimen']}: {ex['instances']}", flush=True)
        ov["model"], ov["rendered"] = a.label, stamp
        ov["what"] = ("leaf_segmenter instance masks (Leaf / Petiole / Hole) drawn on the LM3 working image with the "
                      "Reporter's build_summary_image (masks only; boxes/landmarks/labels/CF off), rendered by "
                      "tools/release/model_card_assets/leaf_segmenter.py.")
        (docs / "overlay_provenance.json").write_text(json.dumps(ov, indent=1) + "\n")
    finally:
        runs.close()


if __name__ == "__main__":
    main()
