"""Regenerate ``leafmachine3/modelhub/models.lock.yaml`` from the Hugging Face Hub.

For every default model repo this reads the repo's current commit and its ``manifest.json`` (which
carries the per-file SHA-256 written at upload time) and pins both into the lock. The runtime then
installs exactly those commits: a new model on the Hub changes nothing until this script is re-run
and the lock is committed with an LM3 release (DEPLOYMENT_PLAN.md section 7.2).

Run from the checkout with a logged-in ``huggingface_hub``:

    python tools/modelhub/build_lock.py            # rewrite the lock
    python tools/modelhub/build_lock.py --check    # exit 1 if the lock is stale

Placeholders: an action whose repo is not on the Hub yet is listed with ``placeholder: true`` and no
revision; the installer reports it as pending and never tries to fetch it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LOCK = HERE.parents[1] / "leafmachine3" / "modelhub" / "models.lock.yaml"
NAMESPACE = "phyloforfun"

#: action -> list of (repo_id, dest_dir, dest rules). ``dest`` is relative to the models root and
#: MUST match the relative ``models/...`` paths in LM3_settings.yaml.
#: ``files`` maps a repo path to (dest, format). Formats: onnx | json | torchscript | pytorch | meta,
#: plus whatever other format folders the repo ships (see FORMAT_FOLDERS). ``meta`` files are always
#: installed; the others are selected with --formats (default: onnx + json, the runtime formats).
#: ``dir`` is the unit's folder under the models root: every auto-discovered format lands in it.
#: ``activatable: False`` marks a stage whose model the GUI must not let the user swap (it always
#: runs its default); the Models tab renders those with a locked selector.
#: Provenance files are marked optional: installed, but their absence is not "missing" (the stage runs
#: without them). That is every ``training_metadata.json`` sidecar, and the conversion-factor fit data
#: and script, which tests/test_mp_conversion_factor.py refits to prove model.json is reproducible.
OPTIONAL_SUFFIXES = ("training_metadata.json", "fit_data.csv", "fit_mp_cf.py")

#: Format folders every repo may ship. Files under these are discovered from the repo manifest and
#: added to the unit automatically (dest = <unit dir>/<repo path>), so a format that appears on the
#: Hub later is picked up by regenerating the lock, not by editing this table. Multi-file formats
#: (CoreML .mlpackage, OpenVINO) are listed file by file; the installer is per file anyway.
FORMAT_FOLDERS = ("onnx", "coreml", "openvino", "torchscript", "pytorch")
DEFAULT_FORMATS = ["onnx", "json"]
#: Stages whose model the user may not swap from the GUI.
LOCKED_STAGES = frozenset({"mp_conversion_factor", "ruler_classifier", "landmark_detector"})


def _is_optional(dest: str) -> bool:
    return dest.endswith(OPTIONAL_SUFFIXES)


DEFAULTS: dict[str, dict] = {
    "archival_detector": {"units": [{
        "repo_id": f"{NAMESPACE}/lm3_archival_detector__yolo26x_det_1280",
        "dir": "archival_detector",
        "files": {"onnx/model.onnx": ("archival_detector/model.onnx", "onnx"),
                  "torchscript/model.torchscript": ("archival_detector/model.torchscript", "torchscript"),
                  "pytorch/best.pt": ("archival_detector/best.pt", "pytorch"),
                  "training_metadata.json": ("archival_detector/training_metadata.json", "meta")}}]},
    "plant_detector": {"units": [{
        "repo_id": f"{NAMESPACE}/lm3_plant_detector__yolo26x_det_1280",
        "dir": "plant_detector",
        "files": {"onnx/model.onnx": ("plant_detector/model.onnx", "onnx"),
                  "torchscript/model.torchscript": ("plant_detector/model.torchscript", "torchscript"),
                  "pytorch/best.pt": ("plant_detector/best.pt", "pytorch"),
                  "training_metadata.json": ("plant_detector/training_metadata.json", "meta")}}]},
    "landmark_detector": {"units": [{
        "repo_id": f"{NAMESPACE}/lm3_landmark_detector__yolo26x_pose_640",
        "dir": "landmark_detector",
        "files": {"onnx/model.onnx": ("landmark_detector/model.onnx", "onnx"),
                  "torchscript/model.torchscript": ("landmark_detector/model.torchscript", "torchscript"),
                  "pytorch/best.pt": ("landmark_detector/best.pt", "pytorch"),
                  "training_metadata.json": ("landmark_detector/training_metadata.json", "meta")}}]},
    "leaf_segmenter": {"units": [{
        "repo_id": f"{NAMESPACE}/lm3_leaf_segmenter__yolo26x_seg_1024",
        "dir": "leaf_segmenter",
        "files": {"onnx/model.onnx": ("leaf_segmenter/model.onnx", "onnx"),
                  "torchscript/model.torchscript": ("leaf_segmenter/model.torchscript", "torchscript"),
                  "pytorch/best.pt": ("leaf_segmenter/best.pt", "pytorch"),
                  "training_metadata.json": ("leaf_segmenter/training_metadata.json", "meta")}}]},
    # Destination keeps the file name LM3_settings.yaml already points at, so no settings change.
    "specimen_segmenter": {"units": [{
        "repo_id": f"{NAMESPACE}/lm3_specimen_segmenter__unetpp_effb7_1024",
        "dir": "specimen_segmenter",
        "files": {"onnx/model.onnx": ("specimen_segmenter/unet_masksFromSam3_paperclean_control_1024.onnx", "onnx"),
                  "torchscript/model.torchscript.pt": ("specimen_segmenter/unet_masksFromSam3_paperclean_control_1024.torchscript.pt", "torchscript"),
                  "pytorch/best.pt": ("specimen_segmenter/best.pt", "pytorch"),
                  "training_metadata.json": ("specimen_segmenter/training_metadata.json", "meta")}}]},
    # Not a network: a one-parameter fit shipped as json -- the "json" runtime format, one of the
    # defaults, so it is installed alongside the ONNX graphs.
    "mp_conversion_factor": {"units": [{
        "repo_id": f"{NAMESPACE}/lm3_mp_conversion_factor__sqrt_fit",
        "dir": "mp_conversion_factor",
        "files": {"json/model.json": ("mp_conversion_factor/model.json", "json"),
                  "fit/fit_data.csv": ("mp_conversion_factor/fit_data.csv", "meta"),
                  "fit/fit_mp_cf.py": ("mp_conversion_factor/fit_mp_cf.py", "meta")}}]},
    # The ensemble: each member is its own repo; the runtime (inference/ruler_ensemble.py) expects
    # <models_dir>/<member>/exported/model.onnx + <models_dir>/<member>/metadata.json and ONE shared
    # <models_dir>/label_map.json. The same label_map.json ships in every member repo (same sha), so
    # it is listed once, on the tie-break member.
    "ruler_classifier": {"units": [
        {"repo_id": f"{NAMESPACE}/lm3_ruler_classifier_ensemble__yolo26x_cls_224",
         "dir": "ruler_classifier/yolo26x_cls_224",
         "files": {"onnx/model.onnx": ("ruler_classifier/yolo26x_cls_224/exported/model.onnx", "onnx"),
                   "pytorch/best.pt": ("ruler_classifier/yolo26x_cls_224/weights/best.pt", "pytorch"),
                   "training_metadata.json": ("ruler_classifier/yolo26x_cls_224/metadata.json", "meta"),
                   "label_map.json": ("ruler_classifier/label_map.json", "meta")}},
        {"repo_id": f"{NAMESPACE}/lm3_ruler_classifier_ensemble__yolo26n_cls_224",
         "dir": "ruler_classifier/yolo26n_cls_224",
         "files": {"onnx/model.onnx": ("ruler_classifier/yolo26n_cls_224/exported/model.onnx", "onnx"),
                   "pytorch/best.pt": ("ruler_classifier/yolo26n_cls_224/weights/best.pt", "pytorch"),
                   "training_metadata.json": ("ruler_classifier/yolo26n_cls_224/metadata.json", "meta")}},
        {"repo_id": f"{NAMESPACE}/lm3_ruler_classifier_ensemble__dinov2_frozen_mlp",
         "dir": "ruler_classifier/dinov2_frozen_mlp",
         "files": {"onnx/model.onnx": ("ruler_classifier/dinov2_frozen_mlp/exported/model.onnx", "onnx"),
                   "pytorch/ckpt_best.pt": ("ruler_classifier/dinov2_frozen_mlp/weights/ckpt_best.pt", "pytorch"),
                   "training_metadata.json": ("ruler_classifier/dinov2_frozen_mlp/metadata.json", "meta")}},
    ]},
}

#: Published NON-default models: installed only on request (`lm3 models install --model STAGE=KEY`),
#: never reported missing. Destinations live under <stage>/<model_key>/ so they never collide with a
#: default. ``settings`` = extra lines the CLI prints alongside the path (and key, for keyed stages).
ALTERNATES: dict[str, dict[str, dict]] = {
    "specimen_segmenter": {
        "birefnet_hr_swinl_1024": {"units": [{
            "repo_id": f"{NAMESPACE}/lm3_specimen_segmenter__birefnet_hr_swinl_1024",
        "dir": "specimen_segmenter/birefnet_hr_swinl_1024",
            "files": {"onnx/model.onnx": ("specimen_segmenter/birefnet_hr_swinl_1024/model.onnx", "onnx"),
                      "pytorch/epoch_145.pth": ("specimen_segmenter/birefnet_hr_swinl_1024/epoch_145.pth", "pytorch"),
                      "training_metadata.json": ("specimen_segmenter/birefnet_hr_swinl_1024/training_metadata.json", "meta")}}]},
        "yolo26x_seg_1280": {"units": [{
            "repo_id": f"{NAMESPACE}/lm3_specimen_segmenter__yolo26x_seg_1280",
        "dir": "specimen_segmenter/yolo26x_seg_1280",
            "files": {"onnx/model.onnx": ("specimen_segmenter/yolo26x_seg_1280/model.onnx", "onnx"),
                      "torchscript/model.torchscript": ("specimen_segmenter/yolo26x_seg_1280/model.torchscript", "torchscript"),
                      "pytorch/best.pt": ("specimen_segmenter/yolo26x_seg_1280/best.pt", "pytorch"),
                      "training_metadata.json": ("specimen_segmenter/yolo26x_seg_1280/training_metadata.json", "meta")}}]},
    },
    "archival_detector": {
        "yolo26n_det_640": {"settings": {"imgsz": 640}, "units": [{
            "repo_id": f"{NAMESPACE}/lm3_archival_detector__yolo26n_det_640",
        "dir": "archival_detector/yolo26n_det_640",
            "files": {"onnx/model.onnx": ("archival_detector/yolo26n_det_640/model.onnx", "onnx"),
                      "torchscript/model.torchscript": ("archival_detector/yolo26n_det_640/model.torchscript", "torchscript"),
                      "pytorch/best.pt": ("archival_detector/yolo26n_det_640/best.pt", "pytorch"),
                      "training_metadata.json": ("archival_detector/yolo26n_det_640/training_metadata.json", "meta")}}]},
    },
    "plant_detector": {
        "yolo26n_det_640": {"settings": {"imgsz": 640}, "units": [{
            "repo_id": f"{NAMESPACE}/lm3_plant_detector__yolo26n_det_640",
        "dir": "plant_detector/yolo26n_det_640",
            "files": {"onnx/model.onnx": ("plant_detector/yolo26n_det_640/model.onnx", "onnx"),
                      "torchscript/model.torchscript": ("plant_detector/yolo26n_det_640/model.torchscript", "torchscript"),
                      "pytorch/best.pt": ("plant_detector/yolo26n_det_640/best.pt", "pytorch"),
                      "training_metadata.json": ("plant_detector/yolo26n_det_640/training_metadata.json", "meta")}}]},
    },
}

#: Not on the Hub yet. Listed so status/UI know they are expected; the installer skips them.
#: ``local`` points at the file in this checkout so its sha/bytes can be pinned ahead of upload.
PLACEHOLDERS: dict[str, dict] = {
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit_files(u: dict, manifest: dict) -> list[dict]:
    """The unit's pinned files: the explicit table, plus every file the repo ships under a format folder.

    Discovery is what keeps the lock (and so the Models tab) in step with the Hub without edits here:
    a repo that gains an ``openvino/`` export is picked up on the next regeneration. Explicit entries
    win over discovered ones for the same repo path, which is how a default keeps its historical
    destination name (the UNet++ ONNX file LM3_settings.yaml already points at).
    """
    files = []
    seen = set()
    for src, (dest, fmt) in u["files"].items():
        rec = manifest["files"].get(src)
        if rec is None:
            raise SystemExit(f"{u['repo_id']}: {src} is not in the repo manifest")
        files.append({"src": src, "dest": dest, "format": fmt, "sha256": rec["sha256"], "bytes": rec["bytes"],
                      "optional": _is_optional(dest)})
        seen.add(src)
    udir = u.get("dir")
    if udir:
        for src in sorted(manifest["files"]):
            top = src.split("/", 1)[0]
            if src in seen or top not in FORMAT_FOLDERS or "/" not in src:
                continue
            rec = manifest["files"][src]
            files.append({"src": src, "dest": f"{udir}/{src}", "format": top, "sha256": rec["sha256"],
                          "bytes": rec["bytes"], "optional": False})
    return files


def _pin_units(api, hf_hub_download, spec: dict) -> tuple[list[dict], dict]:
    """Pin a spec's units; also the stage settings the variant implies (its input size, from the
    manifest), merged under any explicit ``settings`` in the spec. Activating a variant from the GUI
    writes these; switching back to the default restores the default's, so a nano detector's 640 does
    not linger after the 1280 default is chosen again."""
    units = []
    settings: dict = {}
    for u in spec["units"]:
        info = api.model_info(u["repo_id"], files_metadata=True)
        manifest = json.loads(Path(hf_hub_download(u["repo_id"], "manifest.json", revision=info.sha)).read_text())
        units.append({"repo_id": u["repo_id"], "revision": info.sha, "model_key": manifest["lm3"].get("model_key"),
                      "files": _unit_files(u, manifest)})
        if len(spec["units"]) == 1 and isinstance(manifest.get("imgsz"), int):
            settings["imgsz"] = manifest["imgsz"]
    settings.update(spec.get("settings") or {})
    return units, settings


def build() -> dict:
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415 - tooling only

    api = HfApi()
    actions: dict[str, dict] = {}
    for action, spec in DEFAULTS.items():
        units, settings = _pin_units(api, hf_hub_download, spec)
        actions[action] = {"required": True, "activatable": action not in LOCKED_STAGES,
                           **({"settings": settings} if settings else {}), "units": units}
    checkout = HERE.parents[1]
    for action, spec in PLACEHOLDERS.items():
        units = []
        for u in spec["units"]:
            files = []
            for src, (dest, fmt) in u["files"].items():
                local = checkout / u["local"][src]
                rec = {"sha256": _sha256(local), "bytes": local.stat().st_size} if local.is_file() else {"sha256": None, "bytes": None}
                files.append({"src": src, "dest": dest, "format": fmt, **rec})
            units.append({"repo_id": u["repo_id"], "revision": None, "model_key": None, "files": files})
        actions[action] = {"required": True, "placeholder": True, "units": units}
    alternates: dict[str, dict] = {}
    for stage, models in ALTERNATES.items():
        for k, spec in models.items():
            units, settings = _pin_units(api, hf_hub_download, spec)
            alternates.setdefault(stage, {})[k] = ({"settings": settings} if settings else {}) | {"units": units}
    version = (HERE.parents[1] / "VERSION").read_text(encoding="utf-8").strip()   # the one LM3 version
    return {"schema_version": 1, "lm3_version": version, "hub_namespace": NAMESPACE,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "default_formats": list(DEFAULT_FORMATS), "actions": actions, "alternates": alternates}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed lock differs from the Hub")
    ap.add_argument("--out", default=str(LOCK))
    a = ap.parse_args(argv)
    lock = build()
    text = "# GENERATED by tools/modelhub/build_lock.py -- do not hand-edit. See DEPLOYMENT_PLAN.md section 7.2.\n" + yaml.safe_dump(lock, sort_keys=False)
    out = Path(a.out)
    if a.check:
        old = yaml.safe_load(out.read_text()) if out.exists() else None
        same = old is not None and {k: v for k, v in old.items() if k != "generated_at"} == {k: v for k, v in lock.items() if k != "generated_at"}
        print("lock is up to date" if same else "lock is STALE")
        return 0 if same else 1
    out.write_text(text)
    n = sum(len(u["files"]) for a_ in lock["actions"].values() for u in a_["units"])
    print(f"wrote {out} ({len(lock['actions'])} actions, {n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
