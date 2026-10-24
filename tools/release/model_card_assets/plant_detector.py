#!/usr/bin/env python3
"""Re-render the plant_detector model-card example grid with a given model.

The set lives in docs/hf_docs/plant_detector/ and is described by overlay_provenance.json, which names
the specimens and the example run they come from. For each specimen the model detects on the run's
working image exactly as the plant_detector stage does (``load_detector`` with the stage's settings from
LM3_settings.yaml, only the model path overridden, then the stage's same-class duplicate suppression),
and the kept boxes are drawn with the Reporter's ``build_summary_image`` in the style the card has used
since iteration 1 (axis-aligned boxes with class + confidence labels; masks, landmarks and CF banner off),
shown at 1600 px. The example run is read from a COPY of its SQLite file, so it is never touched. The
provenance file is updated with the new box counts and the model that rendered them.

    .venv/bin/python tools/release/model_card_assets/plant_detector.py \
        --model /path/to/model.onnx --label "yolo26x_det_1280 iteration 2" [--device cuda:0|cpu]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2

from _common import ROOT, Runs, device, load_config, published
from leafmachine3.core.box_dedup import suppress_duplicate_boxes
from leafmachine3.core.imaging import read_image
from leafmachine3.inference.factory import load_detector
from leafmachine3.modules.detector_common import _dedup_settings
from leafmachine3.reporting.overlay import build_summary_image
from leafmachine3.reporting.palette import GroupStyle, OverlayStyle

DOCS = ROOT / "docs" / "hf_docs" / "plant_detector"
STAGE = "plant_detector"
OVERLAY_LONG_SIDE = 1600
JPEG_QUALITY = 88
# the iteration-1 grid's look (2026-10-06 renderer): detector boxes only, archival filled, plant outlined
STYLE = dict(draw_masks=False, draw_landmarks=False, draw_petiole=False, draw_cf_banner=False, box_style="yolo",
             groups={"archival": GroupStyle(border=True, fill=True, fill_alpha=0.30, line_width=3),
                     "plant": GroupStyle(border=True, fill=False, line_width=3),
                     "leaf": GroupStyle(border=True, fill=False, line_width=3)})


def detect(cfg, model, img) -> list[dict]:
    """The plant_detector stage's per-specimen step (detector_common.run_detection) without crops or DB rows."""
    dets = model.predict(img)
    enabled, overlap, metric = _dedup_settings(cfg, STAGE)
    decisions = suppress_duplicate_boxes(dets, overlap=overlap, metric=metric) if enabled else [None] * len(dets)
    return [{"cls_name": d.cls_name, "conf": float(d.conf), "xyxy": tuple(d.xyxy), "source": "plant"}
            for d, dec in zip(dets, decisions) if dec is None or not dec.suppressed]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="plant detector ONNX to render with")
    ap.add_argument("--label", required=True, help="model description written into the provenance file")
    ap.add_argument("--settings", default=str(ROOT / "LM3_settings.yaml"))
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--docs", default=str(DOCS))
    ap.add_argument("--all", action="store_true", help="also render pruned candidates (to pick again)")
    a = ap.parse_args()
    docs = Path(a.docs)

    cfg = load_config(a.settings, STAGE, a.model)
    model = load_detector(cfg, STAGE, device(a.device))
    style = OverlayStyle(**STYLE)
    runs = Runs()
    try:
        ov = json.loads((docs / "overlay_provenance.json").read_text())
        for ex in ov["examples"]:
            if not published(docs, ov, ex["file"], a.all):
                continue
            db = runs.db(ex["source_run"])
            _, working = runs.specimen(db, ex["specimen"])
            img = read_image(working)
            boxes = detect(cfg, model, img)
            out = build_summary_image(img, boxes, [], None, style, 1.0)
            s = OVERLAY_LONG_SIDE / max(out.shape[:2])
            if s < 1:
                out = cv2.resize(out, (round(out.shape[1] * s), round(out.shape[0] * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(docs / ex["file"]), out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            ex["n_boxes"], ex["classes"] = len(boxes), sorted({b["cls_name"] for b in boxes})
            ex["bytes"] = (docs / ex["file"]).stat().st_size
            print(f"{ex['specimen']}: {len(boxes)} boxes {ex['classes']}", flush=True)
        ov["model"], ov["rendered"] = a.label, dt.date.today().isoformat()
        ov["what"] = ("plant_detector boxes drawn on the LM3 working image with the Reporter's build_summary_image "
                      "(boxes + labels only; masks/landmarks/CF off), detected with the plant_detector stage's settings "
                      "and duplicate suppression; rendered by tools/release/model_card_assets/plant_detector.py.")
        (docs / "overlay_provenance.json").write_text(json.dumps(ov, indent=1) + "\n")
    finally:
        runs.close()


if __name__ == "__main__":
    main()
