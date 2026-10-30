"""Does each model give the same answer on CoreML as on the CPU? (``lm3 doctor --models`` on a Mac,
and ``tools/macos/verify_coreml.py``.)

Every model the settings select is loaded twice: the CPU path every other platform runs (full graph)
and the CoreML path exactly as a pipeline worker builds it. Both get the same input tensors.
Both run on the same images and are compared:

* detectors / leaf segmenter / landmarks: reference detections (conf >= ``REQUIRED_CONF``) matched to
  CoreML detections of the same class by box IoU; matched-box IoU, score difference, mask IoU
  (segmenter) and keypoint distance (landmarks);
* specimen segmenter: whole-sheet mask IoU;
* ruler member: top-1 class and largest probability difference.

The leaf segmenter and the landmark model run on leaf crops cut from the reference plant
detector's leaf boxes, as in a real run. Verdicts: ``ok``, ``warn`` (agrees, but less closely than
expected), ``fail`` (wrong answers -- the YOLO box bug looks like this). ``ok`` means float32 rounding:
its limits sit a little outside what CUDA vs the CPU gives on the 23 example sheets (``ROUNDING``).
"""
from __future__ import annotations

import time
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

REQUIRED_CONF = 0.40      # a reference detection this confident must be found on CoreML
REF_CONF = 0.25           # the pipeline's usual detector threshold
TEST_CONF = 0.15          # search the CoreML output a little lower, so a score at the edge is not "missing"
MATCH_IOU = 0.50
N_LEAF_CROPS = 4          # leaf crops per sheet for the segmenter / landmark checks

OK, WARN, FAIL = "ok", "warn", "fail"

#: The ``ok`` limits. CUDA vs CPU, same check, 23 example sheets (2026-10-30): box IoU min 0.997,
#: score diff max 0.035, leaf mask IoU min 0.995, keypoint error max 0.3 px, specimen mask IoU min
#: 0.99987, ruler probability diff max 0.0005.
ROUNDING = {"box_iou_min": 0.99, "dconf_max": 0.05, "mask_iou_min": 0.98, "kpt_err_px_max": 2.0,
            "specimen_iou_min": 0.999, "ruler_prob_diff_max": 0.01}


def calibration_images() -> list[Path]:
    """The six packaged sheets (``leafmachine3/setup/calibration_images``), present in every install."""
    root = resources.files("leafmachine3.setup.calibration_images")
    return sorted(Path(str(p)) for p in root.iterdir() if str(p).lower().endswith((".jpg", ".jpeg", ".png")))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match(ref, test) -> list[tuple[int, int, float]]:
    """Greedy same-class matching, most confident reference first -> ``[(ref_i, test_j, iou)]``."""
    pairs, used = [], set()
    for i in np.argsort(-ref.boxes.conf):
        best, best_j = MATCH_IOU, None
        for j in range(len(test.boxes)):
            if j in used or test.boxes.cls[j] != ref.boxes.cls[i]:
                continue
            iou = _iou(ref.boxes.xyxy[i], test.boxes.xyxy[j])
            if iou >= best:
                best, best_j = iou, j
        if best_j is not None:
            used.add(best_j)
            pairs.append((int(i), best_j, best))
    return pairs


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def compare_yolo(ref_res, test_res) -> dict:
    """Agreement of one image's CoreML result with its CPU reference."""
    pairs = match(ref_res, test_res)
    required = [i for i in range(len(ref_res.boxes)) if ref_res.boxes.conf[i] >= REQUIRED_CONF]
    found = {i for i, _, _ in pairs}
    out: dict[str, Any] = {
        "n_ref": len(ref_res.boxes), "n_coreml": len(test_res.boxes),
        "required": len(required), "required_found": sum(1 for i in required if i in found),
        "ious": [p[2] for p in pairs],
        "dconf": [abs(float(ref_res.boxes.conf[i]) - float(test_res.boxes.conf[j])) for i, j, _ in pairs],
    }
    if ref_res.masks is not None and test_res.masks is not None:
        out["mask_ious"] = [_mask_iou(ref_res.masks.data[i], test_res.masks.data[j]) for i, j, _ in pairs]
    if ref_res.keypoints is not None and test_res.keypoints is not None:
        errs = []
        for i, j, _ in pairs:
            a, b = ref_res.keypoints.data[i], test_res.keypoints.data[j]
            vis = (a[:, 2] > 0.5) & (b[:, 2] > 0.5)
            if vis.any():
                errs.append(float(np.linalg.norm(a[vis, :2] - b[vis, :2], axis=1).max()))
        out["kpt_err_px"] = errs
    return out


def summarize_yolo(per_image: Sequence[dict]) -> dict:
    """Pool per-image comparisons into one verdict for the model."""
    def pool(key):
        return [v for d in per_image for v in d.get(key, [])]

    ious, dconf, miou, kerr = pool("ious"), pool("dconf"), pool("mask_ious"), pool("kpt_err_px")
    req = sum(d["required"] for d in per_image)
    found = sum(d["required_found"] for d in per_image)
    s = {
        "images": len(per_image), "required": req, "required_found": found,
        "n_ref": sum(d["n_ref"] for d in per_image), "n_coreml": sum(d["n_coreml"] for d in per_image),
        "box_iou_min": min(ious) if ious else None, "box_iou_mean": float(np.mean(ious)) if ious else None,
        "dconf_max": max(dconf) if dconf else None,
        "mask_iou_min": min(miou) if miou else None, "mask_iou_mean": float(np.mean(miou)) if miou else None,
        "kpt_err_px_max": max(kerr) if kerr else None,
    }
    recall = found / req if req else 1.0
    if req and (recall < 0.90 or (s["box_iou_mean"] or 0) < 0.90):
        verdict, why = FAIL, f"found {found}/{req} confident detections, mean box IoU {s['box_iou_mean']}"
    elif recall < 1.0 or (s["box_iou_min"] is not None and s["box_iou_min"] < ROUNDING["box_iou_min"]) \
            or (s["dconf_max"] or 0) > ROUNDING["dconf_max"] \
            or (s["mask_iou_min"] is not None and s["mask_iou_min"] < ROUNDING["mask_iou_min"]) \
            or (s["kpt_err_px_max"] or 0) > ROUNDING["kpt_err_px_max"]:
        verdict, why = WARN, "agrees, but beyond float32 rounding"
    else:
        verdict, why = OK, ""
    s.update(verdict=verdict, why=why)
    return s


def _timed(fn: Callable, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def leaf_crops(img: np.ndarray, plant_res, n: int = N_LEAF_CROPS) -> list[np.ndarray]:
    """The ``n`` most confident leaf boxes of a plant-detector result, cropped from ``img``."""
    names = {k: v.lower() for k, v in plant_res.names.items()}
    order = [i for i in np.argsort(-plant_res.boxes.conf) if "leaf" in names.get(int(plant_res.boxes.cls[i]), "")]
    crops = []
    for i in order[:n]:
        x1, y1, x2, y2 = (int(round(v)) for v in plant_res.boxes.xyxy[i])
        if x2 - x1 > 16 and y2 - y1 > 16:
            crops.append(np.ascontiguousarray(img[y1:y2, x1:x2]))
    return crops


def _model_paths(cfg) -> dict[str, Any]:
    from leafmachine3.inference.specimen_models import resolve_specimen_model

    def path(stage):
        return cfg.resolve_model_path(cfg.stage(stage)["model"]["path"])

    spec, _ = resolve_specimen_model(cfg.stage("specimen_segmenter"))
    ruler = cfg.stage("ruler_classifier")
    return {
        "archival_detector": path("archival_detector"), "plant_detector": path("plant_detector"),
        "leaf_segmenter": path("leaf_segmenter"), "landmark_detector": path("landmark_detector"),
        "specimen_segmenter": (spec, path("specimen_segmenter")),
        "ruler_classifier": cfg.resolve_model_path(ruler.get("models_dir") or "models/ruler_classifier"),
    }


def check_models(cfg, images: Sequence[Path], log: Callable[[str], None] = lambda s: None,
                 only: Optional[Sequence[str]] = None) -> list[dict]:
    """Compare every CoreML-bound model with its CPU reference on ``images`` -> one dict per model."""
    from leafmachine3.inference import coreml
    from leafmachine3.inference import ultra_replacements as ultra_rep

    cpu, cml = list(coreml.CPU_PROVIDERS), coreml.providers()
    paths = _model_paths(cfg)
    sheets = [(p.name, ultra_rep.imread(p)) for p in images]
    sheets = [(n, im) for n, im in sheets if im is not None]
    results: list[dict] = []
    crops: list[np.ndarray] = []

    def want(name):
        return not only or name in only

    for name, task in (("plant_detector", "detect"), ("archival_detector", "detect"),
                       ("leaf_segmenter", "segment"), ("landmark_detector", "pose")):
        if not want(name) and not (name == "plant_detector" and (want("leaf_segmenter") or want("landmark_detector"))):
            continue
        log(f"{name}: loading CPU reference and CoreML session ...")
        test = ultra_rep.YOLO(paths[name], task=task, providers=cml)
        ref = ultra_rep.YOLO(paths[name], task=task, providers=cpu)
        imgsz = ref.metadata.get("imgsz") or 640
        inputs = [im for _, im in sheets] if task == "detect" else crops
        kw = dict(imgsz=imgsz, retina_masks=True)
        per_image, t_ref, t_test = [], [], []
        if inputs:
            test.predict(inputs[0], conf=TEST_CONF, **kw)            # compile / warm up, not timed
        for im in inputs:
            r, tr = _timed(ref.predict, im, conf=REF_CONF, **kw)
            t, tt = _timed(test.predict, im, conf=TEST_CONF, **kw)
            per_image.append(compare_yolo(r[0], t[0]))
            t_ref.append(tr)
            t_test.append(tt)
            if name == "plant_detector":
                crops.extend(leaf_crops(im, r[0]))
        if want(name):
            s = summarize_yolo(per_image)
            s.update(model=name, path=str(paths[name]), bound=test.session.get_providers()[0],
                     sessions_built=getattr(test.session, "n_built", None),
                     ms_cpu=float(np.mean(t_ref)) if t_ref else None,
                     ms_coreml=float(np.mean(t_test)) if t_test else None)
            results.append(s)
            log(_line(s))
        del ref, test

    if want("specimen_segmenter"):
        from leafmachine3.inference.specimen_segmenter import build_specimen_segmenter

        spec, mpath = paths["specimen_segmenter"]
        log(f"specimen_segmenter ({spec.key}): loading ...")
        ref = build_specimen_segmenter(spec, mpath, providers=cpu, paperclean=False)
        test = build_specimen_segmenter(spec, mpath, providers=cml, paperclean=False)
        if sheets:
            test.mask(sheets[0][1])
        ious, t_ref, t_test = [], [], []
        for _, im in sheets:
            a, tr = _timed(ref.mask, im)
            b, tt = _timed(test.mask, im)
            ious.append(_mask_iou(a, b))
            t_ref.append(tr)
            t_test.append(tt)
        m = min(ious) if ious else None
        verdict = OK if m is None or m >= ROUNDING["specimen_iou_min"] else WARN if m >= 0.95 else FAIL
        bound = getattr(getattr(test, "session", None), "get_providers", lambda: ["(via ultra_rep)"])()[0]
        s = dict(model="specimen_segmenter", key=spec.key, path=str(mpath), images=len(ious), bound=bound,
                 mask_iou_min=m, mask_iou_mean=float(np.mean(ious)) if ious else None,
                 ms_cpu=float(np.mean(t_ref)) if t_ref else None, ms_coreml=float(np.mean(t_test)) if t_test else None,
                 verdict=verdict, why="" if verdict == OK else f"specimen mask IoU down to {m:.4f}")
        results.append(s)
        log(_line(s))
        del ref, test

    if want("ruler_classifier"):
        results.extend(_check_ruler(paths["ruler_classifier"], [im for _, im in sheets], cpu, cml, log))
    return results


def _check_ruler(models_dir: str, sheets: Sequence[np.ndarray], cpu, cml, log) -> list[dict]:
    """Each CoreML ruler member vs its CPU twin, on squarified tiles cut from the sheets."""
    import os

    from leafmachine3.inference import coreml
    from leafmachine3.inference.ruler_ensemble import _load_label_map, _Member, resolve_members
    from leafmachine3.inference.ruler_squarify import RulerSquarifier

    classes = _load_label_map(models_dir)
    sq = RulerSquarifier(sz=720, method="tile_four", augment=False)
    tiles = []
    for im in sheets:   # the bottom strip of a sheet is the usual ruler position; any texture will do
        h = im.shape[0]
        tiles.append(cv2.cvtColor(sq.transform(np.ascontiguousarray(im[int(h * 0.85):, :])), cv2.COLOR_BGR2RGB))
    out = []
    for key, dir_name in resolve_members(None):
        if key not in coreml.COREML_RULER_MEMBERS:
            continue
        log(f"ruler_classifier {key}: loading ...")
        ref = _Member(os.path.join(models_dir, dir_name), cpu, classes)
        test = _Member(os.path.join(models_dir, dir_name), cml, classes)
        same, dmax, t_ref, t_test = 0, 0.0, [], []
        if tiles:
            test.session.run(None, {test._input_name: test._preprocess(tiles[0])})
        for rgb in tiles:
            x = ref._preprocess(rgb)
            a, tr = _timed(ref.session.run, None, {ref._input_name: x})
            b, tt = _timed(test.session.run, None, {test._input_name: x})
            a, b = np.asarray(a[0], np.float64), np.asarray(b[0], np.float64)
            same += int(a.argmax() == b.argmax())
            dmax = max(dmax, float(np.abs(a - b).max()))
            t_ref.append(tr)
            t_test.append(tt)
        verdict = FAIL if same < len(tiles) else WARN if dmax > ROUNDING["ruler_prob_diff_max"] else OK
        s = dict(model=f"ruler_classifier {key}", path=os.path.join(models_dir, dir_name), images=len(tiles),
                 bound=test.session.get_providers()[0], top1_same=same, prob_diff_max=dmax,
                 ms_cpu=float(np.mean(t_ref)) if t_ref else None, ms_coreml=float(np.mean(t_test)) if t_test else None,
                 verdict=verdict, why="" if verdict == OK else f"top-1 same on {same}/{len(tiles)}, max prob diff {dmax:.3f}")
        out.append(s)
        log(_line(s))
    return out


def _fmt(v, f="{:.3f}") -> str:
    return "-" if v is None else f.format(v)


def _line(s: dict) -> str:
    speed = ""
    if s.get("ms_cpu") and s.get("ms_coreml"):
        speed = f"  cpu {s['ms_cpu']:.0f} ms, coreml {s['ms_coreml']:.0f} ms (x{s['ms_cpu'] / s['ms_coreml']:.1f})"
    if "box_iou_mean" in s:
        body = (f"found {s['required_found']}/{s['required']}, box IoU min {_fmt(s['box_iou_min'])} "
                f"mean {_fmt(s['box_iou_mean'])}, dconf max {_fmt(s['dconf_max'])}")
        if s.get("mask_iou_min") is not None:
            body += f", mask IoU min {_fmt(s['mask_iou_min'])}"
        if s.get("kpt_err_px_max") is not None:
            body += f", kpt err max {_fmt(s['kpt_err_px_max'], '{:.1f}')} px"
    elif "top1_same" in s:
        body = f"top-1 same {s['top1_same']}/{s['images']}, prob diff max {_fmt(s['prob_diff_max'])}"
    else:
        body = f"mask IoU min {_fmt(s.get('mask_iou_min'), '{:.4f}')} mean {_fmt(s.get('mask_iou_mean'), '{:.4f}')}"
    return f"[{s['verdict']:4s}] {s['model']:22s} on {s['bound']}: {body}{speed}"
