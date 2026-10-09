#!/usr/bin/env python
"""Render and time each leaf's ECT at 180, 360 (the default), and 720 directions.

The companion to ``examples/ECT_color_options/ect_color_options.py``: same sample masks, same
renderers, but the palette is fixed to the LM3 default (``magma`` with the visual log) and the
direction count varies. ``num_dirs`` sets both axes of the ECT -- the number of directions and the
number of thresholds -- so every render is ``num_dirs x num_dirs`` px:

    180  one direction every other degree
    360  one direction per degree (``modules.ect.num_dirs`` default)
    720  two directions per degree

For each mask and direction count it writes the three images the ECT stage writes (Cartesian,
radial, radial + outline) and times the stage's per-leaf work -- ``compute_ect`` plus the three
renders -- ``--repeats`` times (default 10) after one untimed warm-up, single process. Per-run times
go to ``timings.csv`` and the per-resolution means to ``timings.md``.

Usage
-----
    # the whole sample set, each leaf into its own folder beside this script
    python examples/ECT_resolution_options/ect_resolution_comparison.py \
        examples/ECT_color_options/ECT_color_options_masks/*.png
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ECT_color_options"))

from ect_color_options import (                                # noqa: E402
    compute_ect,
    render_cartesian_ect,
    render_radial_ect,
    save_image,
    visual_log,
)
from leafmachine3.reporting.ect_viz import render_radial_ect_overlay   # noqa: E402

DIRS: tuple[int, ...] = (180, 360, 720)
PALETTE = "magma"
REPEATS = 10


def run_once(mask, num_dirs: int):
    """One leaf's ECT-stage work at ``num_dirs``: returns (compute_s, render_s, images)."""
    t0 = time.perf_counter()
    res = compute_ect(mask, num_dirs=num_dirs)
    t1 = time.perf_counter()
    viz = visual_log(res.ect_matrix)
    images = {
        "cartesian": render_cartesian_ect(viz, res.thetas, cmap=PALETTE),
        "radial": render_radial_ect(viz, res.thetas, res.thresholds, res.bound_radius,
                                    cmap=PALETTE),
        "radial-overlay": render_radial_ect_overlay(viz, res.thetas, res.thresholds,
                                                    res.bound_radius, res.outline_norm,
                                                    cmap=PALETTE),
    }
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1, images


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("masks", nargs="+", type=Path,
                    help="binary mask PNG(s) -- a leaf silhouette, white on black")
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).resolve().parent,
                    help="output root (default: beside this script); one subfolder per mask")
    ap.add_argument("--dirs", type=int, nargs="+", default=list(DIRS),
                    help=f"direction counts (default: {' '.join(map(str, DIRS))})")
    ap.add_argument("--repeats", type=int, default=REPEATS,
                    help=f"timed runs per mask and direction count (default: {REPEATS})")
    args = ap.parse_args(argv)

    rows = []
    for mask_path in args.masks:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f"cannot read mask: {mask_path}")
        mask = mask > 127
        stem = mask_path.stem
        out_dir = args.out / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for num_dirs in args.dirs:
            _, _, images = run_once(mask, num_dirs)                  # warm-up, untimed
            for view, img in images.items():
                save_image(img, out_dir / f"{stem}__ECT-{view}__d{num_dirs}__{PALETTE}__log.png")
            for rep in range(args.repeats):
                c, r, _ = run_once(mask, num_dirs)
                rows.append({"mask": stem, "num_dirs": num_dirs, "repeat": rep,
                             "compute_s": c, "render_s": r, "total_s": c + r})
            mine = [x for x in rows if x["mask"] == stem and x["num_dirs"] == num_dirs]
            print(f"{stem:22s} d{num_dirs:<4d} compute {statistics.mean(x['compute_s'] for x in mine):.3f}s"
                  f"  render {statistics.mean(x['render_s'] for x in mine):.3f}s", flush=True)

    with open(args.out / "timings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [f"Mean seconds per leaf over {len(args.masks)} masks x {args.repeats} runs "
             f"(palette {PALETTE}, log), single process.", "",
             "| num_dirs | image size | compute_ect | 3 renders | total |", "|---|---|---|---|---|"]
    for num_dirs in args.dirs:
        sel = [x for x in rows if x["num_dirs"] == num_dirs]
        c, r, t = (statistics.mean(x[k] for x in sel) for k in ("compute_s", "render_s", "total_s"))
        lines.append(f"| {num_dirs} | {num_dirs} x {num_dirs} px | {c:.3f} | {r:.3f} | {t:.3f} |")
    (args.out / "timings.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
