"""Fingerprint every step between an input image and the first network output, to compare platforms.

Two machines that run the same images through the same models should produce the same numbers. When
they don't, this finds the first step that differs. Run it on each machine and compare the JSON files:

    python tools/verification/input_fingerprint.py --images examples/images --out fp_linux.json
    python tools/verification/input_fingerprint.py --images examples/images --out fp_mac.json
    python tools/verification/input_fingerprint.py --compare fp_linux.json fp_mac.json

Per image it records the SHA-256 of:

* ``file``: the image file's bytes;
* ``decode_cv2``: the pixels LM3 decodes (``ultra_replacements.imread``, OpenCV);
* ``decode_pil``: the same file decoded by Pillow (a second JPEG decoder);
* ``ingest_resized`` / ``ingest_jpeg`` / ``ingest_decoded``: for images ingest downsizes, the Pillow
  LANCZOS resize, the JPEG it writes and that JPEG decoded again (the working image);
* ``letterbox``: the working image letterboxed to ``--imgsz``, as the detectors see it;
* ``resize_exact``: the same resize with ``cv2.INTER_LINEAR_EXACT``, OpenCV's platform-exact variant.

Once per machine it also runs the archival detector's ONNX file on the CPU on a seeded random tensor
(``ort_random``), so differences in the network arithmetic alone show up separately from the inputs.

``--dump DIR`` writes the arrays for ``--dump-images`` (and the ONNX output) as .npz files, so the
size of a difference can be measured, not only detected.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from leafmachine3.inference import ultra_replacements as ultra_rep  # noqa: E402

JPEG_EXTS = {".jpg", ".jpeg"}
IMAGE_EXTS = JPEG_EXTS | {".png", ".tif", ".tiff", ".bmp", ".webp"}


def sha(a) -> str:
    if isinstance(a, np.ndarray):
        a = np.ascontiguousarray(a).tobytes()
    return hashlib.sha256(a).hexdigest()[:16]


def versions() -> dict:
    from PIL import features
    out = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
           "numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__,
           "pillow_libjpeg": features.version("jpg")}
    info = cv2.getBuildInformation()
    out["opencv_jpeg"] = next((ln.split(":", 1)[1].strip() for ln in info.splitlines()
                               if ln.strip().startswith("JPEG:")), "")
    out["opencv_cpu_baseline"] = next((ln.split(":", 1)[1].strip() for ln in info.splitlines()
                                       if "Baseline:" in ln), "")
    try:
        import onnxruntime as ort
        out["onnxruntime"] = ort.__version__
    except ImportError:
        pass
    return out


def ingest_copy(path: Path, max_dim: int, quality: int):
    """``Ingest._normalize`` in memory: (resized RGB array or None, JPEG bytes or None)."""
    with Image.open(path) as im:
        if path.suffix.lower() in JPEG_EXTS and max(im.size) <= max_dim:
            return None, None
        if im.mode != "RGB":
            im = im.convert("RGB")
        if max(im.size) > max_dim:
            scale = max_dim / float(max(im.size))
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return np.asarray(im), buf.getvalue()


def fingerprint(args) -> dict:
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    dump = Path(args.dump) if args.dump else None
    wanted = set(args.dump_images.split(",")) if args.dump_images else set()
    if dump:
        dump.mkdir(parents=True, exist_ok=True)
    out = {"versions": versions(), "settings": {"max_dim": args.max_dim, "quality": args.quality,
                                                "imgsz": args.imgsz}, "images": {}}
    for p in images:
        rec = {}
        raw = p.read_bytes()
        rec["file"] = sha(raw)
        dec = ultra_rep.imread(p)
        rec["decode_cv2"] = sha(dec)
        with Image.open(p) as im:
            rec["decode_pil"] = sha(np.asarray(im.convert("RGB")))
        resized, jpeg = ingest_copy(p, args.max_dim, args.quality)
        work = dec
        if jpeg is not None:
            rec["ingest_resized"] = sha(resized)
            rec["ingest_jpeg"] = sha(jpeg)
            work = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            rec["ingest_decoded"] = sha(work)
        lb = ultra_rep.letterbox(work, (args.imgsz, args.imgsz), stride=32, auto=True)
        rec["letterbox"] = sha(lb)
        r = args.imgsz / max(work.shape[:2])
        size = (round(work.shape[1] * r), round(work.shape[0] * r))
        rec["resize_exact"] = sha(cv2.resize(work, size, interpolation=cv2.INTER_LINEAR_EXACT))
        rec["shape"] = list(work.shape)
        out["images"][p.name] = rec
        print(p.name, " ".join(f"{k}={v}" for k, v in rec.items() if k != "shape"), flush=True)
        if dump and p.stem in wanted:
            arrays = {"decode_cv2": dec, "letterbox": lb}
            if jpeg is not None:
                arrays.update(ingest_resized=resized, ingest_jpeg=np.frombuffer(jpeg, np.uint8), ingest_decoded=work)
            np.savez_compressed(dump / f"{p.stem}.npz", **arrays)
    if args.onnx:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1                    # one thread: no reduction-order effects
        sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
        x = np.random.default_rng(0).random((1, 3, args.imgsz, args.imgsz * 3 // 4), dtype=np.float32)
        outs = sess.run(None, {sess.get_inputs()[0].name: x})
        out["ort_random"] = {"input": sha(x), "onnx": sha(Path(args.onnx).read_bytes()),
                             "outputs": [sha(o) for o in outs]}
        print("ort_random", out["ort_random"], flush=True)
        if dump:
            np.savez_compressed(dump / "ort_random.npz", **{f"out{i}": o for i, o in enumerate(outs)})
    return out


def compare(a_path: str, b_path: str) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (a_path, b_path))
    print("versions:")
    for k in sorted(set(a["versions"]) | set(b["versions"])):
        va, vb = a["versions"].get(k), b["versions"].get(k)
        print(f"  {'  ' if va == vb else '!='} {k}: {va} | {vb}")
    steps = ["file", "decode_cv2", "decode_pil", "ingest_resized", "ingest_jpeg", "ingest_decoded", "letterbox",
             "resize_exact"]
    counts = {s: [0, 0] for s in steps}
    print("first step that differs, per image:")
    for name in sorted(set(a["images"]) & set(b["images"])):
        ra, rb = a["images"][name], b["images"][name]
        first = None
        for s in steps:
            if s in ra and s in rb:
                counts[s][0] += 1
                same = ra[s] == rb[s]
                counts[s][1] += same
                if not same and first is None:
                    first = s
        print(f"  {name}: {first or 'all the same'}")
    print("steps the same on both machines:")
    for s, (n, same) in counts.items():
        if n:
            print(f"  {s:15s} {same}/{n}")
    if "ort_random" in a and "ort_random" in b:
        oa, ob = a["ort_random"], b["ort_random"]
        print("onnx on a fixed random input:", "same model" if oa["onnx"] == ob["onnx"] else "DIFFERENT model",
              "| outputs", "identical" if oa["outputs"] == ob["outputs"] else "differ")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--images", help="folder of input images")
    ap.add_argument("--out", help="write the fingerprint JSON here")
    ap.add_argument("--max-dim", type=int, default=3200, help="ingest.max_working_dim")
    ap.add_argument("--quality", type=int, default=100, help="ingest.jpg_quality")
    ap.add_argument("--imgsz", type=int, default=1280, help="letterbox long side (the detectors' imgsz)")
    ap.add_argument("--onnx", help="an ONNX model to run on a seeded random input (archival detector)")
    ap.add_argument("--dump", help="write arrays for --dump-images here (.npz)")
    ap.add_argument("--dump-images", default="", help="comma-separated image stems to dump")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"), help="compare two fingerprint files")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.images or not args.out:
        ap.error("--images and --out are required (or use --compare)")
    res = fingerprint(args)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
