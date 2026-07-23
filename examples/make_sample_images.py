"""Generate a few synthetic herbarium-ish images for trying LeafMachine3 without real data.

Each image is a pale "sheet" carrying a green leaf blob, a dark ruler strip, and a label
rectangle -- enough structure for a smoke run. With ``compute.mock: true`` in
``LM3_settings.yaml`` you can run the whole pipeline against these immediately::

    python examples/make_sample_images.py            # writes examples/images/*.jpg
    machine3 --config LM3_settings.yaml              # (set compute.mock: true first)

Once real exported models are placed under ``models/``, flip ``compute.mock`` back to
``false`` and rerun to get real detections/segmentations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_image(path: Path, seed: int) -> Path:
    """Write one deterministic synthetic specimen JPEG to ``path``."""
    rng = np.random.default_rng(seed)
    h, w = 1200, 900
    img = np.full((h, w, 3), 232, dtype=np.uint8)                     # pale sheet
    # leaf blob (BGR green), rotated a little differently per seed
    angle = int(rng.integers(0, 60))
    cv2.ellipse(img, (int(0.36 * w), int(0.46 * h)), (180, 300), angle, 0, 360, (40, 130, 45), -1)
    cv2.ellipse(img, (int(0.36 * w), int(0.46 * h)), (180, 300), angle, 0, 360, (25, 90, 30), 4)
    # petiole
    cv2.line(img, (int(0.36 * w), int(0.76 * h)), (int(0.40 * w), int(0.92 * h)), (30, 90, 35), 10)
    # ruler strip along the bottom
    cv2.rectangle(img, (60, h - 90), (w - 60, h - 50), (60, 60, 60), -1)
    for x in range(80, w - 60, 40):                                  # tick marks
        cv2.line(img, (x, h - 90), (x, h - 70), (230, 230, 230), 2)
    # a label + a barcode-ish block
    cv2.rectangle(img, (int(0.60 * w), int(0.10 * h)), (int(0.93 * w), int(0.32 * h)), (250, 250, 240), -1)
    cv2.rectangle(img, (int(0.62 * w), int(0.34 * h)), (int(0.80 * w), int(0.40 * h)), (20, 20, 20), -1)
    # a little noise so specimens differ
    noise = rng.integers(0, 14, size=(h, w, 3), dtype=np.uint8)
    img = cv2.subtract(img, noise)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic herbarium sample images.")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "images"),
        help="output directory for the sample JPEGs",
    )
    parser.add_argument("--count", type=int, default=4, help="how many images to generate")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    written = [make_image(out_dir / f"sample_{i:02d}.jpg", seed=i + 1) for i in range(args.count)]
    print(f"wrote {len(written)} sample image(s) to {out_dir}")
    for p in written:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
