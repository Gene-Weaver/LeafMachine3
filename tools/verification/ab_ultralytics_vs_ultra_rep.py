"""A/B: ultralytics.YOLO vs ultra_rep.YOLO on identical inputs/kwargs (needs the models + ultralytics).

Usage: python tools/verification/ab_ultralytics_vs_ultra_rep.py [cpu|cuda:0]
Run it ONLY with device cuda:0 in a venv that has onnxruntime-gpu: on a CPU device ultralytics' ONNX backend
pip-installs the CPU onnxruntime over onnxruntime-gpu (check_requirements). Set LD_LIBRARY_PATH to the
nvidia/*/lib dirs first (machine3 does this via re-exec; a bare python process does not).
"""
import glob, os, sys, time, json
import numpy as np, cv2
os.environ.setdefault("YOLO_VERBOSE", "False")
dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"          # "cpu" | "cuda:0"
B = "/datac/Labelbox_Dump/LM3/examples_out/test_ultralytics_removal_for_inference_baseline"
sheets = sorted(glob.glob(f"{B}/_working/*.jpg"))
leafs = sorted(glob.glob(f"{B}/_crops/*__BBOX-leaf__*.jpg"))[:80]
print(f"device={dev} sheets={len(sheets)} leaf_crops={len(leafs)}")

from ultralytics import YOLO as UL
from leafmachine3.inference import ultra_replacements as ultra_rep
providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"] if dev.startswith("cuda") else ["CPUExecutionProvider"]

def pad_white(img, frac=0.10):
    h, w = img.shape[:2]; px, py = round(w*frac), round(h*frac)
    return cv2.copyMakeBorder(img, py, py, px, px, cv2.BORDER_CONSTANT, value=(255,255,255))

CASES = {
  "plant_detector":    dict(task="detect",  inputs=sheets, kw=dict(conf=0.3,  iou=0.55, imgsz=1280, max_det=1000)),
  "archival_detector": dict(task="detect",  inputs=sheets, kw=dict(conf=0.25, iou=0.6,  imgsz=1280, max_det=100)),
  "leaf_segmenter":    dict(task="segment", inputs=leafs,  kw=dict(conf=0.3,  iou=0.5,  imgsz=1024, max_det=100, retina_masks=True)),
  "landmark_detector": dict(task="pose",    inputs=leafs,  kw=dict(conf=0.25, iou=0.45, imgsz=640), pad=True),
}
summary = {}
for name, c in CASES.items():
    path = f"/datac/Labelbox_Dump/LM3/models/{name}/model.onnx"
    ul = UL(path, task=c["task"]); rp = ultra_rep.YOLO(path, task=c["task"], providers=providers)
    assert {int(k): v for k, v in ul.names.items()} == rp.names, "names differ"
    stats = dict(n_inputs=0, n_det_mismatch=0, n_det_total=0, box_maxabs=0.0, conf_maxabs=0.0, cls_mismatch=0,
                 kpt_maxabs=0.0, mask_px_diff=0, mask_px_total=0, mask_min_iou=1.0, masks_compared=0, t_ul=0.0, t_rp=0.0)
    for p in c["inputs"]:
        img = cv2.imread(p)
        if c.get("pad"): img = pad_white(img)
        t0 = time.perf_counter(); ru = ul.predict(img, device=dev, verbose=False, **c["kw"])[0]; t1 = time.perf_counter()
        rr = rp.predict(img, **c["kw"])[0]; t2 = time.perf_counter()
        stats["t_ul"] += t1 - t0; stats["t_rp"] += t2 - t1; stats["n_inputs"] += 1
        bu = ru.boxes; nu = len(bu); nr = len(rr.boxes); stats["n_det_total"] += nu
        assert ru.orig_shape == tuple(rr.orig_shape), (ru.orig_shape, rr.orig_shape)
        if nu != nr:
            stats["n_det_mismatch"] += 1; print(f"  COUNT MISMATCH {name} {os.path.basename(p)}: ul={nu} rp={nr}"); continue
        if nu == 0: continue
        xu, cu, ku = bu.xyxy.cpu().numpy(), bu.conf.cpu().numpy(), bu.cls.cpu().numpy()
        stats["box_maxabs"] = max(stats["box_maxabs"], float(np.abs(xu - rr.boxes.xyxy).max()))
        stats["conf_maxabs"] = max(stats["conf_maxabs"], float(np.abs(cu - rr.boxes.conf).max()))
        stats["cls_mismatch"] += int((ku.astype(int) != rr.boxes.cls.astype(int)).sum())
        if c["task"] == "pose":
            kpu = ru.keypoints.data.cpu().numpy(); stats["kpt_maxabs"] = max(stats["kpt_maxabs"], float(np.abs(kpu - rr.keypoints.data).max()))
        if c["task"] == "segment":
            mu = ru.masks.data.cpu().numpy().astype(bool); mr = rr.masks.data.astype(bool)
            assert mu.shape == mr.shape, (mu.shape, mr.shape)
            for a, b in zip(mu, mr):
                inter, union = int((a & b).sum()), int((a | b).sum())
                stats["mask_px_diff"] += int((a ^ b).sum()); stats["mask_px_total"] += int(a.sum())
                stats["mask_min_iou"] = min(stats["mask_min_iou"], inter / union if union else 1.0); stats["masks_compared"] += 1
    summary[name] = stats
    print(name, json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in stats.items()}))
json.dump(summary, open(f"/tmp/ab_{dev.replace(':','')}.json", "w"), indent=1)
