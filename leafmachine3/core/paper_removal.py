"""Informed HSV paper-removal from within a specimen foreground mask (locked settings).

Ported verbatim from the ``LM3_Specimen_Segmentation`` training project's
``common/paper_removal.py`` (the single source of truth used to build the paperclean
training set). Runs as the follow-up step after the UNet specimen mask: it removes
white/yellow/gray paper the mask wrongly includes -- interior inter-branch gaps and the
edge halo -- WITHOUT assuming a fixed paper color:

  1. Sample 5x5 pixel grids a RANDOM 2-5% of min(H,W) OUTSIDE the mask (buffered off the
     plant edge); robustly reject plant-contaminated / non-uniform grids.
  2. Model paper as a per-channel HSV value-RANGE (robust percentiles), widened a bit.
  3. Remove mask pixels inside that range, but ONLY where they form a substantial
     contiguous region -- protecting paper-colored dried leaves. Speckle-open and min-gap
     both scale with resolution.

The sampling is seeded (``np.random.default_rng(0)``), so the result -- and the sampled
patch centers used for the QC overlay -- is fully reproducible for a given (image, mask).
"""
from __future__ import annotations

import cv2
import numpy as np

# --- locked settings (identical to the training project) --------------------------
N_PATCHES = 20
PCT_LO, PCT_HI = 0.02, 0.05          # sample a random 2-5% of min(H,W) outside the mask
GRID = 2                             # 5x5 sampling grid (2*GRID+1)
RANGE_PLO, RANGE_PHI = 1, 99         # robust per-channel paper-range percentiles
PAPER_WIDEN = 3.5                    # multiplicative expansion of each channel's half-range
MARGIN_HSV = (4, 3, 6)               # additive H,S,V margin (S small -> leaves stay safe)
MIN_GAP_FRAC = 0.0025 ** 2           # min gap = (0.25% of linear dim)^2 of area, resolution-scaled
MIN_GAP_FLOOR_PX = 1.0               # never sub-pixel
MIN_KEEP_FRAC = 3e-4                 # drop foreground specks smaller than this
OPEN_FRAC = 0.0015                   # speckle-open kernel = this * min(H,W)


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def remove_small_blobs(mask: np.ndarray, min_frac: float = 5e-4) -> np.ndarray:
    """Drop connected foreground components smaller than ``min_frac`` of the image."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    H, W = m.shape[:2]
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    min_area = min_frac * H * W
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def sample_paper(img_hsv: np.ndarray, mask: np.ndarray, return_centers: bool = False):
    """Sample paper pixels outside ``mask``. ``return_centers=True`` also returns the kept
    sample-box centers ``[(x, y), ...]`` (native/mask-frame coords) for QC/visualization."""
    none = (None, []) if return_centers else None
    m = (mask > 0).astype(np.uint8)
    H, W = m.shape
    mindim = min(H, W)
    lo, hi = max(3, int(PCT_LO * mindim)), max(4, int(PCT_HI * mindim))
    bg = cv2.dilate(m, _disk(GRID)) == 0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return none
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    cx, cy = cnt.mean(0)
    rng = np.random.default_rng(0)
    patches, centers = [], []
    for i in np.linspace(0, len(cnt) - 1, N_PATCHES * 2).astype(int):
        px, py = cnt[i]
        d = np.array([px - cx, py - cy], float)
        nrm = np.linalg.norm(d)
        if nrm < 1:
            continue
        d /= nrm
        off = int(rng.uniform(lo, hi))
        sx, sy = int(px + d[0] * off), int(py + d[1] * off)
        x0, x1 = max(0, sx - GRID), min(W, sx + GRID + 1)
        y0, y1 = max(0, sy - GRID), min(H, sy + GRID + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        pix = img_hsv[y0:y1, x0:x1][bg[y0:y1, x0:x1]].astype(np.float32)
        if len(pix) >= (2 * GRID + 1) ** 2 // 2:
            patches.append(pix)
            centers.append((sx, sy))
    if len(patches) < 3:
        return none
    meds = np.array([np.median(p, 0) for p in patches])
    consensus = np.median(meds, 0)
    dev = np.linalg.norm(meds - consensus, axis=1)
    mad = np.median(dev) + 1e-6
    spread = np.array([p.std(0).sum() for p in patches])
    keep = [i for i in range(len(patches))
            if dev[i] < 3.0 * mad and spread[i] < 2.5 * (np.median(spread) + 1e-6)]
    if len(keep) < 3:
        keep = list(np.argsort(dev)[:max(3, N_PATCHES // 2)])
    keep = keep[:N_PATCHES]
    paper = np.concatenate([patches[i] for i in keep], 0)
    return (paper, [centers[i] for i in keep]) if return_centers else paper


def paper_range(paper_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.percentile(paper_px, RANGE_PLO, axis=0)
    hi = np.percentile(paper_px, RANGE_PHI, axis=0)
    c = (lo + hi) / 2.0
    half = (hi - lo) / 2.0 * PAPER_WIDEN + np.array(MARGIN_HSV, np.float32)
    plo = np.maximum(c - half, [0, 0, 0])
    phi = np.minimum(c + half, [180, 255, 255])
    return plo.astype(np.float32), phi.astype(np.float32)


def _keep_large(binary: np.ndarray, min_frac: float) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    H, W = binary.shape
    min_area = max(MIN_GAP_FLOOR_PX, min_frac * H * W)
    out = np.zeros_like(binary)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return out


def remove_paper(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a paper-removed copy of ``mask`` (uint8 {0,1}). If no paper can be sampled
    (e.g. the mask fills the frame), returns the mask unchanged."""
    final, _removed, _centers = paperclean(img_bgr, mask)
    return final


def paperclean(img_bgr: np.ndarray, mask: np.ndarray):
    """Run paper-removal and return everything the QC overlay needs.

    Returns ``(final, removed, centers)`` where ``final`` is the cleaned mask (uint8 {0,1}),
    ``removed`` is the pixels deleted from ``mask`` (uint8 {0,1}; the "refined" region drawn
    red), and ``centers`` is the list of paper-sampling box centers ``[(x, y), ...]`` in the
    mask's pixel frame (drawn as blue boxes). If nothing could be sampled the mask is returned
    unchanged with an empty removed region and no centers.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    empty = np.zeros_like(m)
    if m.sum() == 0:
        return m, empty, []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    paper, centers = sample_paper(hsv, m, return_centers=True)
    if paper is None:
        return m, empty, []
    plo, phi = paper_range(paper)
    H, W = m.shape
    open_k = max(3, int(OPEN_FRAC * min(H, W)))
    inrange = np.all((hsv >= plo) & (hsv <= phi), axis=2)
    pp = (inrange & (m > 0)).astype(np.uint8)
    pp = cv2.morphologyEx(pp, cv2.MORPH_OPEN, _disk(max(1, open_k // 2)))
    pp = _keep_large(pp, MIN_GAP_FRAC)
    out = ((m > 0) & (pp == 0)).astype(np.uint8)
    final = remove_small_blobs(out, MIN_KEEP_FRAC)
    removed = ((m > 0) & (final == 0)).astype(np.uint8)
    return final, removed, [(int(x), int(y)) for (x, y) in centers]
