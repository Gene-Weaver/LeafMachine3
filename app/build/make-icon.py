#!/usr/bin/env python3
"""Regenerate app/build/icon.png (1024x1024 RGBA).

electron-builder derives every other icon size from this one file -- the Windows .ico and the
macOS .icns are generated at build time, so this is the only image the repo has to carry.
Run:  python3 app/build/make-icon.py
"""
from __future__ import annotations

import math
import pathlib

from PIL import Image, ImageDraw

SIZE = 1024
OUT = pathlib.Path(__file__).resolve().parent / "icon.png"

BG_TOP = (24, 61, 45)       # deep herbarium green
BG_BOTTOM = (12, 34, 26)
LEAF = (126, 200, 118)
LEAF_EDGE = (58, 122, 62)
VEIN = (24, 61, 45)


def rounded_background() -> Image.Image:
    """Vertical gradient inside a rounded square (macOS/Windows both mask it further)."""
    grad = Image.new("RGB", (1, SIZE))
    for y in range(SIZE):
        t = y / (SIZE - 1)
        grad.putpixel((0, y), tuple(
            round(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)
        ))
    grad = grad.resize((SIZE, SIZE))

    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, SIZE - 1, SIZE - 1), radius=round(SIZE * 0.22), fill=255
    )
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    img.paste(grad, (0, 0), mask)
    return img


def leaf_outline(cx: float, cy: float, half_len: float, half_wid: float) -> list[tuple[float, float]]:
    """A simple ovate leaf: two mirrored quarter-lobes tapering to a point at each end."""
    pts: list[tuple[float, float]] = []
    steps = 160
    for side in (1, -1):
        rng = range(steps + 1) if side == 1 else range(steps, -1, -1)
        for i in rng:
            t = i / steps                       # 0 = tip, 1 = base
            y = cy - half_len + 2 * half_len * t
            # width profile: fat just below the middle, sharp at both ends
            w = half_wid * math.sin(math.pi * t) ** 0.75 * (1.0 - 0.25 * (t - 0.5) ** 2)
            pts.append((cx + side * w, y))
    return pts


def main() -> None:
    img = rounded_background()
    d = ImageDraw.Draw(img)

    cx, cy = SIZE / 2, SIZE / 2 + SIZE * 0.02
    pts = leaf_outline(cx, cy, SIZE * 0.36, SIZE * 0.215)
    d.polygon(pts, fill=LEAF, outline=LEAF_EDGE)

    # midrib
    d.line(
        [(cx, cy - SIZE * 0.35), (cx, cy + SIZE * 0.35)],
        fill=VEIN, width=round(SIZE * 0.016),
    )
    # secondary veins, splayed off the midrib both ways
    for k in range(1, 6):
        t = k / 6.0
        y0 = cy - SIZE * 0.30 + 2 * SIZE * 0.30 * t
        reach = SIZE * 0.155 * math.sin(math.pi * t) ** 0.6
        drop = SIZE * 0.075
        for side in (1, -1):
            d.line(
                [(cx, y0), (cx + side * reach, y0 + drop)],
                fill=VEIN, width=round(SIZE * 0.010),
            )

    # petiole
    d.line(
        [(cx, cy + SIZE * 0.34), (cx, cy + SIZE * 0.42)],
        fill=LEAF_EDGE, width=round(SIZE * 0.022),
    )

    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
