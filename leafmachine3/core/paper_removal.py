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

Two methods share that sampler and the morphological finish (``run_paperclean`` picks one by
name, ``modules.specimen_segmenter.LLR_Hyst_KeepBlobs`` in the settings):

* ``METHOD_HSV`` -- the original above: an HSV value-range widened x3.5 around the samples.
* ``METHOD_LLR_HYST_KEEPBLOBS`` (default since 3.0.1043) -- ``paperclean_llr_hyst_keepblobs``:
  paper and plant are each modeled as a Gaussian in CIE Lab (the plant from the mask's own eroded
  interior), every pixel scores a log-likelihood ratio, and a candidate region (LLR > 1) is removed
  only when at least 10% of it is confidently paper (LLR > 4) -- hysteresis, so a pale leaf is not
  deleted for brushing the paper's color. It also skips the final small-blob drop, which was
  deleting every isolated leaf tip under ~3e-4 of the sheet regardless of color. Chosen on 100
  held-out sheets (training project ``ctxclean/paperclean_variants.py``, variant V23): paper
  precision 0.81 -> 0.91, sheets losing > 25% of their plant 13 -> 5, isolated tips lost 55 -> 0.
"""
from __future__ import annotations

import cv2
import numpy as np

METHOD_HSV = "hsv"
METHOD_LLR_HYST_KEEPBLOBS = "LLR_Hyst_KeepBlobs"
METHOD_NONE = "none"                  # recorded when paperclean is switched off
PAPERCLEAN_METHODS = (METHOD_HSV, METHOD_LLR_HYST_KEEPBLOBS)

# --- LLR hysteresis settings (locked; the harness's V23) ----------------------------
LLR_SEED_TAU = 4.0                   # a region is paper only if >= LLR_MIN_SEED of it scores above this
LLR_GROW_TAU = 1.0                   # ...and it extends over everything above this
LLR_MIN_SEED = 0.10
PLANT_INTERIOR_FRAC = 0.01           # plant color sample = mask eroded by this * min(H,W)
PLANT_SAMPLE_MAX = 50000             # at most this many interior pixels fit the plant Gaussian
LAB_VAR_FLOOR = 4.0                  # variance floor per Lab channel (units^2)
MIN_PAPER_SAMPLES = 25
MIN_PLANT_SAMPLES = 50

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


# --------------------------------------------------------------------------- #
# LLR hysteresis, keep blobs
# --------------------------------------------------------------------------- #
def _hsv_samples_to_lab(paper_hsv: np.ndarray) -> np.ndarray:
    """The sampler's float HSV pixels as float32 OpenCV Lab (L 0-255, a/b offset 128)."""
    hsv8 = np.clip(paper_hsv, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv8, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


def _gauss_fit(x: np.ndarray):
    return x.mean(0), x.var(0) + np.float32(LAB_VAR_FLOOR)


def _gauss_logp(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * (((x - mu) ** 2) / var).sum(-1) - 0.5 * np.log(var).sum()


def plant_interior(mask: np.ndarray, frac: float = PLANT_INTERIOR_FRAC) -> np.ndarray:
    """Pixels deep inside the mask (eroded by ``frac`` of min(H, W)): the plant's own color sample."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    H, W = m.shape
    r = max(3, int(frac * min(H, W)))
    return cv2.erode(m, _disk(r)) > 0


def llr_paper_region(img_bgr: np.ndarray, mask: np.ndarray, paper_hsv: np.ndarray):
    """The hysteresis candidate region (bool, inside ``mask``) of pixels that read as paper rather
    than plant, or ``None`` when either color sample is too small to fit."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    paper_lab = _hsv_samples_to_lab(paper_hsv)
    inside = plant_interior(m)
    if len(paper_lab) < MIN_PAPER_SAMPLES or int(inside.sum()) < MIN_PLANT_SAMPLES:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    plant_lab = lab[inside]
    if len(plant_lab) > PLANT_SAMPLE_MAX:
        pick = np.random.default_rng(0).choice(len(plant_lab), PLANT_SAMPLE_MAX, replace=False)
        plant_lab = plant_lab[pick]
    mu_pa, var_pa = _gauss_fit(paper_lab)
    mu_pl, var_pl = _gauss_fit(plant_lab)
    llr = _gauss_logp(lab, mu_pa, var_pa) - _gauss_logp(lab, mu_pl, var_pl)
    wide = (llr > LLR_GROW_TAU) & (m > 0)
    seed = (llr > LLR_SEED_TAU) & wide
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(wide.astype(np.uint8), 8)
    seeds_per = np.bincount(lab_cc[seed], minlength=n)
    keep = [i for i in range(1, n) if seeds_per[i] >= LLR_MIN_SEED * stats[i, cv2.CC_STAT_AREA]]
    return np.isin(lab_cc, keep)


def paperclean_llr_hyst_keepblobs(img_bgr: np.ndarray, mask: np.ndarray):
    """Lab log-likelihood-ratio paper removal with hysteresis; same contract as :func:`paperclean`.

    Returns ``(final, removed, centers)``. The paper is sampled exactly as the HSV method samples
    it (same boxes, same centers for the overlay); the sheet is then scored against a paper and a
    plant Gaussian in Lab. The morphological finish is the original's speckle-open and minimum
    paper-region size, but the final small-blob drop is deliberately NOT applied: every component
    the network predicted survives unless its color says paper.
    """
    m = (np.asarray(mask) > 0).astype(np.uint8)
    empty = np.zeros_like(m)
    if m.sum() == 0:
        return m, empty, []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    paper, centers = sample_paper(hsv, m, return_centers=True)
    centers = [(int(x), int(y)) for (x, y) in centers] if centers else []
    if paper is None:
        return m, empty, centers
    region = llr_paper_region(img_bgr, m, paper)
    if region is None:
        return m, empty, centers
    H, W = m.shape
    open_k = max(3, int(OPEN_FRAC * min(H, W)))
    pp = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, _disk(max(1, open_k // 2)))
    pp = _keep_large(pp, MIN_GAP_FRAC)
    final = ((m > 0) & (pp == 0)).astype(np.uint8)
    removed = ((m > 0) & (final == 0)).astype(np.uint8)
    return final, removed, centers


def run_paperclean(img_bgr: np.ndarray, mask: np.ndarray, method: str = METHOD_LLR_HYST_KEEPBLOBS):
    """Dispatch on ``method`` (one of :data:`PAPERCLEAN_METHODS`); same return as :func:`paperclean`."""
    if method == METHOD_LLR_HYST_KEEPBLOBS:
        return paperclean_llr_hyst_keepblobs(img_bgr, mask)
    if method == METHOD_HSV:
        return paperclean(img_bgr, mask)
    raise ValueError(f"unknown paperclean method {method!r}; expected one of {PAPERCLEAN_METHODS}")


def paperclean_method_from_cfg(paperclean_on: bool, llr_hyst_keepblobs: bool) -> str:
    """The method name the two settings select: ``none`` when the pass is off, else the
    LLR method when ``LLR_Hyst_KeepBlobs`` is true, else the original HSV method."""
    if not paperclean_on:
        return METHOD_NONE
    return METHOD_LLR_HYST_KEEPBLOBS if llr_hyst_keepblobs else METHOD_HSV
