"""Check the CoreML path model by model on a Mac: same answers as the CPU, and how much faster.

Every model the settings select that LM3 sends to CoreML is run twice on the same images -- once
through the CoreML path exactly as a pipeline worker builds it (its ``coreml_fp32`` package), once on
the CPU as every other platform runs it, with the same input tensors -- and compared
(leafmachine3/inference/coreml_check.py). Prints the environment, one line per model (with load time
and first-call vs later-call milliseconds), and exits 1 if any model gives different answers.

Usage (from the LeafMachine3 folder, installed with ``uv sync --frozen --extra macos``):
    uv run --frozen --no-sync python tools/macos/verify_coreml.py --n 23 --json out.json

On Linux it runs only with LM3_COREML_SIMULATE=1 (each package's network stood in for by its ONNX twin
on the CPU), which checks everything except Core ML itself.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="LM3_settings.yaml", help="settings file that names the models")
    ap.add_argument("--images", default="examples/images", help="folder of sheets (default examples/images)")
    ap.add_argument("--n", type=int, default=6, help="how many sheets to use (sorted order; default 6)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="model names to check (plant_detector archival_detector leaf_segmenter landmark_detector "
                         "specimen_segmenter ruler_classifier)")
    ap.add_argument("--json", default=None, help="also write every number to this file")
    args = ap.parse_args(argv)
    # one line per CoreML compile: input shape, seconds, sessions alive, process memory
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.WARNING, format="  %(message)s", stream=sys.stdout)
    logging.getLogger("leafmachine3.inference.coreml").setLevel(logging.INFO)

    import onnxruntime as ort

    from leafmachine3.core.config import Config
    from leafmachine3.inference import coreml, coreml_check

    env = {k: v for k, v in os.environ.items() if k.startswith("LM3_COREML")}
    print(f"platform      {platform.platform()} ({platform.machine()})")
    print(f"onnxruntime   {ort.__version__}  providers {ort.get_available_providers()}")
    from leafmachine3.inference import coreml_native

    try:
        import coremltools
        ct_version = coremltools.__version__
    except Exception:  # noqa: BLE001
        ct_version = "not installed"
    print(f"coreml        available={coreml.available()} simulated={coreml.simulated()} "
          f"units={coreml_native.compute_units()} coremltools {ct_version}")
    print(f"switches      {env or '(defaults)'}")
    if not coreml.available():
        print("CoreML is not available here (on Linux, set LM3_COREML_SIMULATE=1 to test the code path).")
        return 2

    folder = Path(args.images)
    exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)[: args.n] if folder.is_dir() else []
    if not images:
        images = coreml_check.calibration_images()[: args.n]
        print(f"no images in {folder}; using the {len(images)} packaged calibration sheets")
    print(f"images        {len(images)} from {images[0].parent if images else '-'}\n")

    t0 = time.perf_counter()
    rows = coreml_check.check_models(Config.load(args.config), images, log=print, only=args.only)
    took = time.perf_counter() - t0
    bad = [r for r in rows if r["verdict"] == coreml_check.FAIL]
    warn = [r for r in rows if r["verdict"] == coreml_check.WARN]
    print(f"\n{len(rows)} model(s) checked in {took:.0f} s: {len(rows) - len(bad) - len(warn)} ok, "
          f"{len(warn)} warn, {len(bad)} fail")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "platform": platform.platform(), "onnxruntime": ort.__version__, "coremltools": ct_version,
            "units": coreml_native.compute_units(), "switches": env,
            "images": [str(p) for p in images], "results": rows}, indent=1, default=str))
        print(f"wrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
