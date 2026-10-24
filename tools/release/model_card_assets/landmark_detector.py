#!/usr/bin/env python3
"""Re-render the landmark_detector (pose) model-card overlays with a given model.

docs/hf_docs/landmark_detector/overlay_provenance.json lists the examples: the parent specimen, the
plant_detector leaf box, and the Reporter overlay of the example run it was taken from. Each is redrawn
the way the pipeline draws Overlay_Landmarks:

  landmark_detector stage   load_landmark_pose (settings from LM3_settings.yaml, model path overridden),
                            model.predict(crop) -> keypoints, re-based crop -> working frame
  landmark_measurements     compute_measurements on the keypoints with conf >= min_kpt_conf
  Reporter                  the leaf box re-cut from the working image, build_leaf_landmark_overlay
                            with the measurement panel (_landmark_measure_lines)

The example runs are read from copies of their databases (never modified). The provenance file records
the model that rendered the images.

    .venv/bin/python tools/release/model_card_assets/landmark_detector.py \
        --model /path/to/model.onnx --label "yolo26x_pose_640 iteration 2" [--device cuda:1|cpu]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2

from _common import ROOT, Runs, device, load_config, published
from leafmachine3.core.landmark_metrics import compute_measurements
from leafmachine3.core.landmarks import KPT_INDEX
from leafmachine3.core.records import LandmarkMeasureRow, LandmarkRow
from leafmachine3.inference import load_landmark_pose
from leafmachine3.modules.reporter import _landmark_measure_lines
from leafmachine3.reporting.overlay import build_leaf_landmark_overlay
from leafmachine3.reporting.palette import LandmarkStyle

DOCS = ROOT / "docs" / "hf_docs" / "landmark_detector"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="landmark pose ONNX to render with")
    ap.add_argument("--label", required=True, help="model description written into the provenance file")
    ap.add_argument("--settings", default=str(ROOT / "LM3_settings.yaml"))
    ap.add_argument("--device", default="cuda:0", help="cuda:N or cpu")
    ap.add_argument("--docs", default=str(DOCS))
    ap.add_argument("--all", action="store_true", help="also render pruned candidates (to pick again)")
    a = ap.parse_args()
    docs = Path(a.docs)

    cfg = load_config(a.settings, "landmark_detector", a.model)
    stage = cfg.stage("landmark_detector")
    classes = ("Leaf_WHOLE", "Leaf_PARTIAL") if stage.get("include_partial", False) else ("Leaf_WHOLE",)
    min_conf = float(cfg.stage("landmark_measurements").get("min_kpt_conf", 0.25))
    quality = int(cfg.report.get("formats", {}).get("jpg_quality", 100)) if hasattr(cfg, "report") else 100
    model = load_landmark_pose(cfg, device(a.device))
    lm_style = LandmarkStyle.from_config(cfg)
    runs = Runs()
    try:
        prov = json.loads((docs / "overlay_provenance.json").read_text())
        for ex in prov["examples"]:
            if not published(docs, prov, ex["file"], a.all):
                continue
            db = runs.db(ex["source"])
            sid, working_path = runs.specimen(db, ex["parent_specimen"])
            want = [int(round(v)) for v in ex["leaf_crop_bbox_px_x1y1x2y2"]]
            crop = next((c for c in db.crops("plant_detection", sid, cls_in=classes)
                         if [int(round(v)) for v in (c.x1, c.y1, c.x2, c.y2)] == want), None)
            if crop is None:
                print(f"SKIP {ex['file']}: no leaf crop with box {want} in {ex['parent_specimen']}", flush=True)
                continue
            # landmark_detector stage: keypoints for this crop (first instance = the leaf)
            leaves = model.predict(crop.crop_path)
            if not leaves:
                print(f"SKIP {ex['file']}: no keypoints", flush=True)
                continue
            rows = [LandmarkRow(detection_id=crop.detection_id, instance_index=0, kpt_index=KPT_INDEX[name],
                                kpt_name=name, x=float(xc) + crop.x1, y=float(yc) + crop.y1,
                                x_crop=float(xc), y_crop=float(yc), conf=float(cf))
                    for name, (xc, yc, cf) in leaves[0].items() if name in KPT_INDEX]
            # landmark_measurements stage
            m = compute_measurements({r.kpt_name: (r.x, r.y) for r in rows if r.conf >= min_conf})
            cx, cy = m.lamina_centroid if m.lamina_centroid is not None else (None, None)
            mrow = LandmarkMeasureRow(
                detection_id=crop.detection_id, instance_index=0, lamina_trace_length=m.lamina_trace_length,
                lamina_extent=m.lamina_extent, lamina_tip_base_length=m.lamina_tip_base_length,
                leaf_width=m.leaf_width, apex_angle=m.apex_angle, apex_angle_type=m.apex_angle_type,
                base_angle=m.base_angle, base_angle_type=m.base_angle_type,
                petiole_trace_length=m.petiole_trace_length, lamina_curvature=m.lamina_curvature,
                curvature_point=m.curvature_point, lamina_centroid_x=cx, lamina_centroid_y=cy, n_present=m.n_present)
            # Reporter: the box re-cut from the working image, keypoints + measurement panel
            working = cv2.imread(working_path, cv2.IMREAD_COLOR)
            if working is None:
                print(f"SKIP {ex['file']}: working image missing ({working_path})", flush=True)
                continue
            h, w = working.shape[:2]
            x1, y1, x2, y2 = want
            leaf = working[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            img = build_leaf_landmark_overlay(leaf, rows, _landmark_measure_lines(mrow), lm_style,
                                              curvature_idx=int(m.curvature_point) if m.curvature_point is not None else None)
            cv2.imwrite(str(docs / ex["file"]), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            ex["size"] = [int(img.shape[1]), int(img.shape[0])]
            ex["bytes"] = (docs / ex["file"]).stat().st_size
            ex["n_keypoints_present"] = int(m.n_present)
            print(f"{ex['file']}: {m.n_present} keypoints present", flush=True)
        prov["model"], prov["rendered"] = a.label, dt.date.today().isoformat()
        prov["what"] = ("LeafMachine3 Reporter 'Overlay_Landmarks' renders: landmark_detector keypoints drawn on the "
                        "plant_detector leaf crop (midvein trace, petiole trace, apex/base angle lines, width line) + "
                        "landmark_measurements text box, rendered by tools/release/model_card_assets/landmark_detector.py.")
        (docs / "overlay_provenance.json").write_text(json.dumps(prov, indent=1) + "\n")
    finally:
        runs.close()


if __name__ == "__main__":
    main()
