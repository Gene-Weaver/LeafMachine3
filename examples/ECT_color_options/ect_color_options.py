#!/usr/bin/env python
"""Render one binary leaf mask's ECT across every candidate palette, direction count, and scaling.

This is the palette-picking harness behind ``modules.ect.palette`` and
``modules.ect.apply_log_to_visual_for_bold_color``: point it at a binary mask PNG and it writes the
full cross-product of

    {360, 720 directions} x {17 palettes} x {linear, log} x {cartesian, radial}

so the options can be compared side by side on a REAL leaf instead of guessed at from a colorbar.

The log is a SIGNED ``log1p`` applied to the matrix immediately before coloring and NOWHERE else --
chi runs negative through zero to positive on non-convex shapes, so a plain log would blow up. It
compresses the long tail of large |chi| so the dense mid-range spreads across the whole colormap
instead of piling into one or two shades. This mirrors exactly what LM3 does when
``modules.ect.apply_log_to_visual_for_bold_color`` is on: the visual changes, the stored ECT matrix
never does.

Every render is exactly ``num_dirs x num_dirs`` px -- see the resolution contract in
``leafmachine3.reporting.ect_viz``. The Cartesian view is one matrix cell per pixel, so the
transform is recoverable from the PNG.

Usage
-----
    # one mask -> ./<mask stem>/
    python examples/ECT_color_options/ect_color_options.py /path/to/leaf_mask.png

    # the whole sample set, each leaf into its own folder beside this script
    python examples/ECT_color_options/ect_color_options.py \
        examples/ECT_color_options/ECT_color_options_masks/*.png

    # somewhere else, and only the two direction counts you care about
    python examples/ECT_color_options/ect_color_options.py mask.png -o /tmp/out --dirs 720
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Run from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from leafmachine3.core.ect_compute import compute_ect          # noqa: E402
from leafmachine3.core.imaging import save_image               # noqa: E402
from leafmachine3.reporting.ect_viz import (                   # noqa: E402
    render_cartesian_ect,
    render_radial_ect,
    visual_log,
)

#: Perceptually-uniform, sequential, and single-hue families worth auditing for the ECT.
PALETTES: tuple[str, ...] = (
    "bone", "gray", "pink", "magma", "viridis", "cividis", "winter", "cool", "summer", "spring",
    "YlGn", "Blues", "Greens", "Purples", "Greys", "Oranges", "Reds",
)
DIRS: tuple[int, ...] = (360, 720)
SCALES: tuple[str, ...] = ("linear", "log")


def render_one(mask_path: Path, out_dir: Path, dirs=DIRS, palettes=PALETTES,
               scales=SCALES) -> list[Path]:
    """Write the full palette/direction/scaling cross-product for ``mask_path`` into ``out_dir``."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"cannot read mask: {mask_path}")
    stem = mask_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for num_dirs in dirs:
        res = compute_ect(mask > 127, num_dirs=num_dirs, want_simple=False)
        if res is None:
            raise SystemExit(f"empty or degenerate mask: {mask_path}")
        print(f"  {stem}  num_dirs={num_dirs}: matrix {res.ect_matrix.shape}, "
              f"chi [{res.ect_matrix.min()}, {res.ect_matrix.max()}], "
              f"{len(np.unique(res.ect_matrix))} distinct values", flush=True)
        for scale in scales:
            # The ONE place the log is applied: on the way to the renderer, never into res.
            matrix = visual_log(res.ect_matrix) if scale == "log" else res.ect_matrix
            for cmap in palettes:
                views = {
                    "cartesian": render_cartesian_ect(matrix, res.thetas, cmap=cmap),
                    "radial": render_radial_ect(matrix, res.thetas, res.thresholds,
                                                res.bound_radius, cmap=cmap),
                }
                for view, img in views.items():
                    p = out_dir / f"{stem}__ECT-{view}__d{num_dirs}__{cmap}__{scale}.png"
                    save_image(img, p)                  # PNG -> lossless, never resized
                    written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks", nargs="+", type=Path,
                    help="full path(s) to binary mask PNG(s) -- a leaf silhouette, white on black")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output root (default: beside this script). Each mask gets its own "
                         "subfolder named after the mask stem.")
    ap.add_argument("--dirs", type=int, nargs="+", default=list(DIRS),
                    help=f"direction counts to render (default: {' '.join(map(str, DIRS))})")
    ap.add_argument("--palettes", nargs="+", default=list(PALETTES),
                    help="Matplotlib colormap names, CASE-SENSITIVE (default: all 17)")
    ap.add_argument("--scales", nargs="+", choices=SCALES, default=list(SCALES),
                    help="linear and/or log (default: both)")
    args = ap.parse_args(argv)

    root = args.out if args.out is not None else Path(__file__).resolve().parent
    total = 0
    for mask_path in args.masks:
        mask_path = mask_path.resolve()
        out_dir = root / mask_path.stem if args.out is None else root / mask_path.stem
        print(f"{mask_path.name} -> {out_dir}", flush=True)
        written = render_one(mask_path, out_dir, tuple(args.dirs), tuple(args.palettes),
                             tuple(args.scales))
        print(f"  {len(written)} images", flush=True)
        total += len(written)
    print(f"\n{total} images written under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
