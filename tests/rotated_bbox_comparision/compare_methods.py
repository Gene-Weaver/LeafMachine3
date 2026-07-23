"""Rotated-bounding-box method comparison harness (manual / visual, not a pytest test).

Renders a labeled 2x3 panel for a leaf mask comparing six ways to fit the rotated
("length x width") bounding box:

  1. LM2 fit_min_bbox      -- the production default (leafmachine3.core.morphometrics, method="lm2")
  2. cv2.minAreaRect       -- the production option (method="minarearect")
  3. Feret axis + extents  -- Tier 1: orient by the max-Feret (longest-chord) axis, measure hull extents
  4. PCA principal axis    -- Tier 2: orient by the mask's second-moment principal axis
  5. cv2.fitEllipse axis   -- Tier 2: orient by a least-squares ellipse major axis
  6. Hybrid                -- Feret, unless it disagrees with PCA by > 20 deg, then PCA

Methods 3-6 are prototypes implemented HERE only (the repo ships lm2 + minarearect). Each
panel labels the box length/width (px), aspect ratio, long-axis tilt, and box-area / mask-area
(1.0 = perfectly tight).

The input masks (and their RGB-crop backdrops) are the copies in ``inputs/`` next to this
script, so the comparison is self-contained and reproducible; if a copy is missing it falls
back to a live run's ``examples_out/<run>/reports/...`` output.

Run:
    PYTHONPATH=<repo> python tests/rotated_bbox_comparision/compare_methods.py           # all three leaves
    python tests/rotated_bbox_comparision/compare_methods.py /path/to/mask.png out.png "Species"
"""
from __future__ import annotations

import glob
import math
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))                 # .../LM3
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from leafmachine3.core.morphometrics import polygon_morphology  # production LM2 + cv2 paths  # noqa: E402

INPUTS = os.path.join(_HERE, "inputs")                          # self-contained copies (preferred)
RUN_GLOB = os.path.join(_REPO, "examples_out", "*", "reports")   # fallback: a live run's output

# (exact mask filename, species label, output filename) -- located by glob across runs.
LEAVES = [
    ("A_2351160546_Bignoniaceae_Catalpa_speciosa__SEG-leaf__1372_308_1818_929.png",
     "Catalpa speciosa (cordate / heart-shaped)", "panel_catalpa_cordate.png"),
    ("MNHN_438777132_Platanaceae_Platanus_macrophylla__SEG-leaf__35_739_2008_2992.png",
     "Platanus macrophylla (palmate / lobed, maple-like)", "panel_platanus_lobed.png"),
    ("US_1321818753_Rubiaceae_Posoqueria_mutisii__SEG-leaf__1284_301_2334_1529.png",
     "Posoqueria mutisii (simple / entire -- a 'normal' leaf)", "panel_posoqueria_normal.png"),
]


# ---- geometry helpers (the prototype "improved" methods live here) ----------------
def rot(P, phi):
    c, s = math.cos(phi), math.sin(phi)
    P = np.asarray(P, float)
    return np.column_stack([P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c])


def oriented_box(points, theta_deg):
    """Corners of the axis-aligned bbox of ``points`` measured in the frame at ``theta_deg``."""
    th = math.radians(theta_deg)
    Pr = rot(points, -th)
    mn, mx = Pr.min(0), Pr.max(0)
    corners_r = np.array([[mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]])
    return rot(corners_r, th)


def feret_axis(hull):
    best, pa, pb = -1.0, hull[0], hull[0]
    for i in range(len(hull)):
        d = ((hull - hull[i]) ** 2).sum(1)
        j = int(d.argmax())
        if d[j] > best:
            best, pa, pb = d[j], hull[i], hull[j]
    return math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))


def pca_axis(mask):
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs, ys]).astype(float)
    pts -= pts.mean(0)
    w, v = np.linalg.eigh(np.cov(pts.T))
    vec = v[:, int(w.argmax())]
    return math.degrees(math.atan2(vec[1], vec[0]))


def box_metrics(corners):
    c = np.asarray(corners, float)
    edges = [c[(i + 1) % 4] - c[i] for i in range(4)]
    lens = [float(np.hypot(*e)) for e in edges]
    i_long = int(np.argmax(lens))
    L, W = lens[i_long], lens[(i_long + 1) % 4]
    tilt = math.degrees(math.atan2(edges[i_long][1], edges[i_long][0]))
    return L, W, ((tilt + 90) % 180) - 90


# ---- panel rendering -------------------------------------------------------------
def _find_mask(fname: str) -> str | None:
    local = os.path.join(INPUTS, fname)                          # prefer the copied-in input
    if os.path.exists(local):
        return local
    hits = sorted(glob.glob(os.path.join(RUN_GLOB, "Binary_Masks", "Binary_Masks__Leaf", fname)))
    return hits[0] if hits else None


def _find_rgb(mask_basename: str) -> str | None:
    """Exact RGB crop for a mask (same coords): SEG-leaf.png -> BBOX-leaf.jpg."""
    rgb_name = mask_basename.replace("SEG-leaf", "BBOX-leaf").replace(".png", ".jpg")
    local = os.path.join(INPUTS, rgb_name)                       # prefer the copied-in backdrop
    if os.path.exists(local):
        return local
    for pattern in (os.path.join(RUN_GLOB, "Crops", "RGB__leaf", rgb_name),
                    os.path.join(_REPO, "examples_out", "*", "crops", rgb_name)):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


def render_panel(mask_path: str, species: str, out_path: str) -> None:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    stem = os.path.basename(mask_path).split("__SEG-leaf__")[0]
    rgb_path = _find_rgb(os.path.basename(mask_path))
    rgb = cv2.imread(rgb_path) if rgb_path else None
    if rgb is None:                                             # no crop -> colorized mask backdrop
        rgb = np.zeros((*mask.shape, 3), np.uint8)
        rgb[mask > 0] = (60, 160, 60)
    elif rgb.shape[:2] != mask.shape:                          # edge-clamped crop -> top-left pad to frame
        canvas = np.zeros((*mask.shape, 3), np.uint8)
        h, w = min(rgb.shape[0], mask.shape[0]), min(rgb.shape[1], mask.shape[1])
        canvas[:h, :w] = rgb[:h, :w]
        rgb = canvas
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    poly = cnt.reshape(-1, 2).astype(float)
    hull = cv2.convexHull(cnt).reshape(-1, 2).astype(float)
    mask_area = float(cv2.contourArea(cnt))

    results = []
    m = polygon_morphology(poly, method="lm2")
    results.append(("LM2 fit_min_bbox  (current default)", np.asarray(m.rotated_bbox, float)))
    m2 = polygon_morphology(poly, method="minarearect")
    results.append(("cv2.minAreaRect  (current option)", np.asarray(m2.rotated_bbox, float)))
    fth = feret_axis(hull)
    results.append(("Feret axis + hull extents  (Tier 1)", oriented_box(hull, fth)))
    pth = pca_axis(mask)
    results.append(("PCA principal axis  (Tier 2)", oriented_box(hull, pth)))
    (_, _), (_, _), eang = cv2.fitEllipse(cnt)
    results.append(("cv2.fitEllipse axis  (Tier 2)", oriented_box(hull, eang)))
    disagree = abs(((fth - pth + 90) % 180) - 90)
    hang = fth if disagree < 20 else pth
    results.append((f"Hybrid (Feret vs PCA disagree {disagree:.0f} deg -> {'Feret' if disagree < 20 else 'PCA'})",
                    oriented_box(hull, hang)))

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 13.5))
    fig.suptitle(f"Rotated-bbox methods on a {species} leaf\n"
                 f"mask area = {mask_area:,.0f} px^2   --   box red, leaf outline green, long axis dashed",
                 fontsize=14, y=0.98)
    for ax, (title, corners) in zip(axes.ravel(), results):
        ax.imshow(rgb)
        ax.plot(np.append(poly[:, 0], poly[0, 0]), np.append(poly[:, 1], poly[0, 1]),
                color="#2ecc40", lw=1.3, alpha=0.9)
        c = np.asarray(corners, float)
        ax.plot(np.append(c[:, 0], c[0, 0]), np.append(c[:, 1], c[0, 1]), color="#ff2d2d", lw=2.6)
        L, W, tilt = box_metrics(corners)
        edges = [c[(i + 1) % 4] - c[i] for i in range(4)]
        il = int(np.argmax([np.hypot(*e) for e in edges]))
        mid1 = (c[il] + c[(il + 3) % 4]) / 2
        mid2 = (c[(il + 1) % 4] + c[(il + 2) % 4]) / 2
        ax.plot([mid1[0], mid2[0]], [mid1[1], mid2[1]], "--", color="#ffdc00", lw=1.6)
        ax.set_title(f"{title}\nL={L:.0f}  W={W:.0f}  AR={L / max(W, 1):.2f}  "
                     f"tilt={tilt:+.0f} deg  box/mask={(L * W) / mask_area:.2f}x", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"  {'method':40} {'L':>6} {'W':>6} {'AR':>5} {'tilt':>6} {'box/mask':>9}")
    for title, corners in results:
        L, W, tilt = box_metrics(corners)
        print(f"  {title[:40]:40} {L:6.0f} {W:6.0f} {L / max(W, 1):5.2f} {tilt:+6.0f} {(L * W) / mask_area:9.2f}")


def main() -> int:
    if len(sys.argv) >= 3:                                       # explicit: mask.png out.png [species]
        species = sys.argv[3] if len(sys.argv) > 3 else "leaf"
        render_panel(sys.argv[1], species, sys.argv[2])
        return 0
    missing = 0
    for fname, species, out_name in LEAVES:
        mp = _find_mask(fname)
        if not mp:
            print(f"SKIP (mask not found -- run the pipeline first): {fname}")
            missing += 1
            continue
        render_panel(mp, species, os.path.join(_HERE, out_name))
    return 1 if missing == len(LEAVES) else 0


if __name__ == "__main__":
    raise SystemExit(main())
