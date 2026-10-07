"""Mask-quality diagnostics that are INDEPENDENT of symmetry, plus an "archetypal leaf" composite.

The experiment's hypothesis is that symmetry can stand in for mask quality -- that a highly
symmetric leaf is a clean, whole, well-segmented, *archetypal* leaf. A hypothesis cannot be tested
against itself, so everything in :class:`QualityDiagnostics` is computed from the silhouette,
the holes and the keypoint confidences ONLY. Not one field looks at the left/right split, at the
axis frame, or at any symmetry metric. That is what makes ``corr(symmetry, diagnostic)`` in the
report an honest measurement rather than a tautology.

:func:`archetype_score` is the opposite thing: it deliberately FUSES symmetry with the diagnostics
into one ranking number. **Do not use it to validate the symmetry hypothesis** -- it contains
symmetry, so it will always agree with symmetry. Score with ``weights={"symmetry": 0.0}`` to get
the symmetry-free composite (a zero weight drops the term out of the geometric mean exactly), and
correlate THAT against the symmetry metrics.

What each diagnostic can and cannot tell you
--------------------------------------------
Three of these are shape descriptors that a botanist would call *lobing*, not *quality*:
``solidity``, ``perimeter_ratio`` and ``boundary_roughness`` all rise for a deeply lobed or finely
serrate leaf exactly as they rise for a torn or noisy mask. They are reported because the report
needs them to tell those two cases apart by eye, and they are deliberately kept OUT of
:func:`archetype_score` so the composite stays interpretable and does not quietly punish oaks.

The genuinely quality-bearing fields are ``n_components``, ``largest_frac``, ``hole_frac``,
``truncated`` and the keypoint-confidence group. ``flush_edge_frac`` sits between the two lists: it
is meant as a cut-off detector but measures leaf SIZE far more strongly than truncation on the real
cohort (r = +0.848 against ``1 / sqrt(area_px)``), which is why the completeness term it feeds
carries only 0.05 of the composite -- see :func:`_completeness_term`.

One measured caveat on the composite, worth knowing before trusting a ranking
-----------------------------------------------------------------------------
Run over the 105 leaves of the test project, the three quality terms saturate at exactly 1.0 for 59
of them: most masks simply have nothing wrong with them, and the terms fire only on the messy tail
(27-component debris, 3.5% holes, a 0.35-confidence midvein point). So with these weights
``archetype_score`` is in practice a SYMMETRY ranking with quality vetoes bolted on, not an even
blend -- which is fine for "surface the best leaves", and is one more reason the score cannot be
used as independent evidence for the symmetry hypothesis. The diagnostics can; the score cannot.

Frame: image coords, y DOWN, the tip-up oriented-mask frame from :mod:`geometry`.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .axes import AxisFrame
from .geometry import MIDVEIN_NAMES, OrientedLeaf

# --------------------------------------------------------------------------------------------- #
# Diagnostic tuning constants (measurement definitions, not scoring)
# --------------------------------------------------------------------------------------------- #

# Connectivity for component labeling. 8-connectivity, NOT the scipy default 4: polygon
# rasterization routinely leaves one-pixel diagonal necks inside a single leaf, and 4-connectivity
# would report those as separate components -- a fake defect in an otherwise perfect mask.
_CONN8 = np.ones((3, 3), bool)

N_CONTOUR = 512            # equal-arclength samples of the outline used for smoothing
ROUGHNESS_SIGMA_FRAC = 0.005   # Gaussian bandwidth as a FRACTION of outline length (see below)

# Total midvein trace points the pose model can emit: lamina_tip + midvein_0..14 + lamina_base.
N_MIDVEIN_MAX = len(MIDVEIN_NAMES)      # == 17


@dataclass(frozen=True)
class QualityDiagnostics:
    """Per-leaf mask diagnostics. Every field is symmetry-free by construction.

    Degenerate input never raises: a field that cannot be defined (empty mask, no contour, no
    keypoints) is ``nan`` for floats and ``0`` for counts.

    Attributes
    ----------
    n_components:
        Connected components (8-connected) of the silhouette. ``1`` for a clean leaf; ``>1`` means
        debris, a second leaf caught in the crop, or a mask split by a tear.
    largest_frac:
        Largest component area / total silhouette area, in ``(0, 1]``. ``1.0`` means the whole mask
        is one blob; ``0.97`` means 3% of the "leaf" is stray pixels somewhere else.
    solidity:
        Outer-contour area / convex-hull area, on the LARGEST component only (debris is already
        reported by the two fields above; letting it drag the hull would double-count it).
        **This is a lobing signal at least as much as a quality signal.** A clean entire leaf sits
        near 0.95-0.99, but a clean deeply lobed leaf -- an oak, a maple -- is legitimately down
        near 0.6-0.7 with nothing whatever wrong with its mask. Never threshold it alone.
    border_contact_frac:
        Fraction of the silhouette's boundary pixels that lie on the edge of the oriented mask, i.e.
        evidence the leaf was sliced off by the detection box rather than ending in its own margin.
        Reported for the correlation study, but NOT the field the composite scores on -- see
        ``flush_edge_frac`` for why. Real leaves in the test project run 0.001-0.22, median 0.018.
    flush_edge_frac:
        The largest fraction of any ONE of the four mask edges that the silhouette covers, and the
        signal :func:`archetype_score` uses for completeness. The oriented mask is
        content-fit to the silhouette's tight bounding box, so the mask necessarily touches all four
        edges; what distinguishes a cut leaf is that it lies FLUSH along a whole edge. The
        difference matters: a clean synthetic ellipse measures ``border_contact_frac`` 0.112 purely
        because a smooth convex curve is tangent to its bounding box over a long arc, so scoring on
        that number would systematically penalize every blunt-apex leaf as "cut off"; the same
        ellipse measures ``flush_edge_frac`` 0.11 against ~1.0 for a genuinely chopped mask.
        **MEASURED SIZE CONFOUND, read before using this as a truncation flag.** On the 105 real
        leaves it correlates with ``1 / sqrt(area_px)`` at r = +0.848 and with ``log(area_px)`` at
        r = -0.728: a small mask is a coarse mask, one row of its edge is a larger fraction of a
        short edge, and staircase quantization puts more of the outline exactly on the border. So
        the number is mostly reporting how SMALL the leaf is, not how cut off it is, and it clears
        ``FLUSH_EDGE_FREE`` for exactly 1 of the 105. ``border_contact_frac`` carries the same
        confound (r = +0.841 against ``1 / sqrt(area_px)``). Neither is a clean truncation detector;
        see :func:`_completeness_term` for what that costs the composite.
        One caveat remains for both fields: the mask has been ROTATED tip-up, so a cut made along a
        sheet edge only stays flush with a mask edge when the rotation is near a multiple of 90
        degrees. At other angles the cut becomes a slanted straight chord that this axis-aligned
        test under-reads, which is why ``truncated`` is carried alongside as the geometric,
        rotation-proof flag.
    truncated:
        The detection box was clipped by the sheet edge (from :class:`OrientedLeaf`). A leaf that ran
        off the sheet cannot be archetypal, so this is a hard gate in :func:`is_archetypal`.
        It is only *evidence* in :func:`archetype_score`: a box may overhang the sheet by a pixel
        while the leaf itself is entirely inside, so it is graded, not vetoed.
    perimeter_ratio:
        ``perimeter / (2 * sqrt(pi * area))`` -- the dissection index of Shi et al. (2020), the
        reciprocal square root of circularity. Exactly 1 for a disk and rising with any departure
        from it. It rises for lobing, for serration AND for a ragged mask, so like ``solidity`` it
        is a shape descriptor first. Perimeter and area are both taken from the same digitized outer
        contour; on a pixel grid that outline is a staircase, which inflates the perimeter by
        roughly 5% for a smooth curve, so a clean ellipse measures ~1.1 rather than 1.0. Compare
        leaves to each other, not to the ideal.
    boundary_roughness:
        ``perimeter(raw outline) / perimeter(low-pass outline)``, where the low-pass outline is the
        contour resampled to ``N_CONTOUR`` equal-arclength points and convolved with a periodic
        Gaussian of sigma ``ROUGHNESS_SIGMA_FRAC`` of the outline LENGTH. Because the bandwidth is a
        fraction of the shape's own perimeter rather than a pixel count, the measure is
        dimensionless and scale-free: the same leaf scanned at twice the resolution scores the same.
        It rises with everything finer than about 1% of the perimeter and is flat for everything
        coarser, so smooth lobes -- however deep -- pass through the filter untouched and do not
        register. The honest limitation: fine marginal TEETH are also finer than the bandwidth, so a
        serrate leaf reads as rough. Read with ``solidity`` (lobing lowers it, noise barely does) to
        separate real margin complexity from pixel-level noise.
    hole_frac:
        Hole area / silhouette area -- insect damage, decay, or segmentation dropouts.
    kpt_conf_mean, kpt_conf_min:
        Pose confidence over the MIDVEIN TRACE keypoints only, because those are the points the
        midvein axis is fitted to; a confident lobe tip would not make a bad axis good. Computed
        over the stations with a USABLE (finite) confidence, so one nan-valued confidence can no
        longer poison both statistics into nan and silently delete the trace penalty; both are nan
        only when no station has a usable confidence at all.
    n_midvein_kpts:
        Midvein trace points that survived the loader's confidence filter AND carry a usable
        (finite) confidence, out of ``N_MIDVEIN_MAX`` (17). A short trace means a short or wandering
        axis. Counted on the same stations the two confidence statistics average over, so
        :func:`_trace_term` can reconstruct the full-skeleton picture from the three numbers.
    area_px:
        Silhouette area in pixels (holes filled). Size context, and the scale against which
        pixel-level noise should be judged.
    aspect:
        Silhouette bounding-box height / width in the tip-up frame, so it reads as
        length / width for an oriented leaf.
    axis_aspect:
        ``axis arclength / (2 * max |u|)`` from the axis frame, i.e. length over full width measured
        along the real midline instead of the bounding box. ``nan`` when no frame is supplied.
        Context only -- it is never scored, and it is the one field derived from the axis frame.
    """

    specimen_id: int
    leaf_id: int
    detection_id: int

    n_components: int
    largest_frac: float
    solidity: float
    border_contact_frac: float
    flush_edge_frac: float
    truncated: bool
    perimeter_ratio: float
    boundary_roughness: float
    hole_frac: float
    kpt_conf_mean: float
    kpt_conf_min: float
    n_midvein_kpts: int
    area_px: int
    aspect: float
    axis_aspect: float

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON/CSV-friendly record -- one row of the report's diagnostics table."""
        return asdict(self)


# --------------------------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------------------------- #

def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    """``(largest component, n_components, largest_frac)`` for a boolean mask."""
    from scipy import ndimage as ndi

    total = int(mask.sum())
    if total == 0:
        return mask, 0, float("nan")
    lab, n = ndi.label(mask, structure=_CONN8)
    if n <= 1:
        return mask, int(n), 1.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    k = int(counts.argmax())
    return lab == k, int(n), float(counts[k] / total)


def _outer_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Longest external contour of ``mask`` as an ``(N, 1, 2)`` int32 array, or None.

    ``CHAIN_APPROX_NONE``: the roughness measure needs every boundary pixel, not a simplified
    polygon -- simplification is exactly the signal being measured.
    """
    import cv2

    m = np.ascontiguousarray(mask.astype(np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) >= 3]
    if not cnts:
        return None
    return max(cnts, key=lambda c: len(c))


def _resample_closed(pts: np.ndarray, n: int) -> Optional[np.ndarray]:
    """Resample a CLOSED polyline to ``n`` points at equal arclength (last point != first)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 3:
        return None
    loop = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return None
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, cum, loop[:, 0]), np.interp(t, cum, loop[:, 1])])


def _closed_perimeter(pts: np.ndarray) -> float:
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 2:
        return float("nan")
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Silhouette pixels with at least one background 4-neighbor, image edge counting as background.

    ``border_value=0`` (the scipy default) is what makes an on-edge pixel count as boundary, which
    is the whole point here: a leaf cut by the crop has a wall of boundary pixels on the edge.
    """
    from scipy import ndimage as ndi

    if not mask.any():
        return np.zeros_like(mask)
    cross = ndi.generate_binary_structure(2, 1)
    return mask & ~ndi.binary_erosion(mask, structure=cross)


def _border_stats(mask: np.ndarray) -> tuple[float, float]:
    """``(border_contact_frac, flush_edge_frac)`` -- two views of "does the mask hit the crop edge".

    The first normalizes by the outline length and so is inflated by mere tangency; the second asks
    how much of a single edge LINE the mask covers, which only a straight cut can drive toward 1.
    """
    b = _boundary_pixels(mask)
    n = int(b.sum())
    if n == 0:
        return float("nan"), float("nan")
    edge = np.zeros_like(mask)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    contact = float((b & edge).sum() / n)
    flush = max(float(mask[0, :].mean()), float(mask[-1, :].mean()),
                float(mask[:, 0].mean()), float(mask[:, -1].mean()))
    return contact, flush


def _shape_stats(mask: np.ndarray) -> tuple[float, float, float]:
    """``(solidity, perimeter_ratio, boundary_roughness)`` from one component's outer contour."""
    import cv2
    from scipy.ndimage import gaussian_filter1d

    nan3 = (float("nan"), float("nan"), float("nan"))
    cnt = _outer_contour(mask)
    if cnt is None:
        return nan3

    perim = float(cv2.arcLength(cnt, True))
    area = float(cv2.contourArea(cnt))
    if perim <= 0 or area <= 0:
        return nan3

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else float("nan")

    # 2*sqrt(pi*A) is the perimeter of the disk of equal area, so the ratio is 1 for a disk.
    perimeter_ratio = perim / (2.0 * math.sqrt(math.pi * area))

    res = _resample_closed(cnt.reshape(-1, 2).astype(float), N_CONTOUR)
    if res is None:
        return solidity, perimeter_ratio, float("nan")
    sigma = ROUGHNESS_SIGMA_FRAC * N_CONTOUR      # bandwidth in SAMPLES == fraction of arclength
    smooth = gaussian_filter1d(res, sigma, axis=0, mode="wrap")
    sp = _closed_perimeter(smooth)
    roughness = perim / sp if sp and sp > 0 else float("nan")
    return solidity, perimeter_ratio, roughness


# --------------------------------------------------------------------------------------------- #
# the diagnostic entry point
# --------------------------------------------------------------------------------------------- #

def compute_quality(leaf: OrientedLeaf, frame: Optional[AxisFrame] = None) -> QualityDiagnostics:
    """All symmetry-free diagnostics for one leaf.

    ``frame`` is optional and supplies ``axis_aspect`` only; every other field is measured on the
    masks alone, so a leaf whose axis could not be built is still fully diagnosed.
    """
    sil = np.asarray(leaf.silhouette, bool)
    area_px = int(sil.sum())

    largest, n_comp, largest_frac = _largest_component(sil)
    # Shape stats on the largest component: a stray blob's contribution is already reported by
    # n_components/largest_frac, and dragging the convex hull out to reach it would report the same
    # defect a second time as fake lobing.
    solidity, perim_ratio, roughness = _shape_stats(largest) if area_px else (
        float("nan"), float("nan"), float("nan"))

    holes = np.asarray(leaf.holes, bool)
    hole_frac = float(holes.sum() / area_px) if area_px else float("nan")

    if area_px:
        ys, xs = np.nonzero(sil)
        h = float(ys.max() - ys.min() + 1)
        w = float(xs.max() - xs.min() + 1)
        aspect = h / w if w > 0 else float("nan")
    else:
        aspect = float("nan")

    # nan-tolerant: np.mean/np.min over a set holding ONE nan confidence returns nan for both
    # statistics, and a nan sub-score is dropped from the archetype mean -- so a single unusable
    # confidence used to delete the whole trace penalty and let the leaf score as if its trace were
    # perfect. Average over the stations whose confidence is usable instead (== nanmean/nanmin), and
    # count only those, so mean, min and count all describe the same set of stations.
    confs = np.asarray([leaf.kpt_conf[n] for n in MIDVEIN_NAMES if n in leaf.kpt_conf], float)
    confs = confs[np.isfinite(confs)]
    kpt_mean = float(confs.mean()) if confs.size else float("nan")
    kpt_min = float(confs.min()) if confs.size else float("nan")
    contact, flush = _border_stats(sil) if area_px else (float("nan"), float("nan"))

    axis_aspect = float("nan")
    if frame is not None and frame.length > 0:
        umax = float(np.abs(frame.u).max()) if frame.u.size else 0.0
        if umax > 0:
            axis_aspect = float(frame.length / (2.0 * umax))

    return QualityDiagnostics(
        specimen_id=int(leaf.specimen_id), leaf_id=int(leaf.leaf_id),
        detection_id=int(leaf.detection_id),
        n_components=n_comp, largest_frac=largest_frac, solidity=solidity,
        border_contact_frac=contact, flush_edge_frac=flush,
        truncated=bool(leaf.truncated), perimeter_ratio=perim_ratio,
        boundary_roughness=roughness, hole_frac=hole_frac,
        kpt_conf_mean=kpt_mean, kpt_conf_min=kpt_min, n_midvein_kpts=int(confs.size),
        area_px=area_px, aspect=aspect, axis_aspect=axis_aspect,
    )


# --------------------------------------------------------------------------------------------- #
# Archetype scoring
#
# EVERY constant below is a FIRST PROPOSAL, not an established value. They were chosen to be
# plausible against one verified example leaf (midvein Dice 0.9735, chord Dice 0.9052) and then
# sanity-checked against the distribution of real diagnostics; they have NOT been validated against
# human judgement of which leaves look archetypal. The intended workflow is: rank leaves by
# archetype_score, LOOK at the ranked contact sheet, and move these numbers until the ranking
# matches what the eye says. Treat them as a starting point that is meant to be edited.
# --------------------------------------------------------------------------------------------- #

# -- symmetry term ---------------------------------------------------------------------------- #
# Calibrated against the cohort, NOT against the one verified example leaf. That leaf (midvein Dice
# 0.9735) turns out to sit at the cohort MAXIMUM: the 105 real leaves run Dice 0.13-0.977 with a
# median of 0.895, and |SI_A| 0.027-0.878 with a median of 0.142. Ramps anchored on the example leaf
# drove 49 of 105 leaves to a hard 0 and destroyed all ranking in the bottom half, so the zero
# points sit below the 10th percentile and only genuinely broken masks veto.
SYM_DICE_ZERO = 0.70     # straightened left/right Dice at which the symmetry sub-score hits 0
SYM_DICE_ONE = 0.97      # ... and at which it saturates at 1 (the best real leaf measured)
SYM_SIA_ZERO = 0.35      # |SI_A| (signed area imbalance) at which the sub-score hits 0
SYM_SIA_ONE = 0.03       # ... and at which it saturates at 1 (the most balanced real leaf measured)
SYM_AXIS = "midvein"     # the axis the composite scores; the chord axis is the control, not the ruler

# -- integrity term --------------------------------------------------------------------------- #
LARGEST_FRAC_ZERO = 0.90     # largest component holds <=90% of the area -> integrity 0
LARGEST_FRAC_ONE = 1.0
COMPONENT_PENALTY_K = 0.5    # component sub-score = 1 / (1 + K * (n_components - 1))
HOLE_FRAC_ZERO = 0.05        # >=5% of the lamina lost to holes -> 0
HOLE_FRAC_ONE = 0.0

# -- completeness term ------------------------------------------------------------------------ #
# Scored on flush_edge_frac, not border_contact_frac: a clean ellipse is tangent to its content-fit
# box over enough of its outline to read 0.11 border contact with nothing wrong with it.
# Both of those are however confounded with leaf SIZE rather than with truncation (measured
# correlations in the flush_edge_frac field docstring), so this term is deliberately weighted low --
# see _completeness_term and ARCHETYPE_WEIGHTS.
FLUSH_EDGE_FREE = 0.25       # deadband: a blunt apex legitimately covers a quarter of its box edge
FLUSH_EDGE_ZERO = 0.75       # three quarters of one edge covered -> that edge is a cut, not a margin
TRUNCATED_PENALTY = 0.25     # multiplier when the detection box was clipped by the sheet edge

# -- trace-confidence term -------------------------------------------------------------------- #
# Nudged to span the range actually observed on the test project (mean 0.76-0.995, worst-point
# 0.35-0.98) so the term discriminates instead of saturating at 1 for every leaf. The mean ramp is
# applied to the COVERAGE-WEIGHTED mean (see _trace_term), which equals the plain mean for the
# complete 17-station trace every leaf in the test project has, so the calibration is unchanged.
KPT_CONF_MEAN_ZERO = 0.50
KPT_CONF_MEAN_ONE = 0.95
KPT_CONF_MIN_ZERO = 0.25     # == the loader's default filter, so a barely-surviving point scores 0
KPT_CONF_MIN_ONE = 0.85
# Where the COUNT sub-score bottoms out. 2.0, not the 6.0 first proposed: geometry.OrientedLeaf
# .midvein() returns an axis from 3 points, so 3 points is a coarse axis, not an absent one, and 2
# is the count at which the axis genuinely ceases to exist. The old 6.0 hard-vetoed (sub-score
# exactly 0, and a 0 annihilates a geometric mean) every 3-6 point trace that the geometry, the
# axis fit and the symmetry metrics had all accepted and measured. Grading them instead leaves the
# categorical rejection to ARCHETYPE_MIN_MIDVEIN_KPTS in is_archetypal, which is where the other
# categorical judgments live and where the threshold can be changed without silently zeroing a
# published score.
MIDVEIN_KPTS_ZERO = 2.0
MIDVEIN_KPTS_ONE = 15.0

#: Weights of the four top-level terms in the weighted GEOMETRIC mean. Geometric, not arithmetic,
#: because "archetypal" is a conjunction: a beautifully symmetric leaf that is torn in half is not
#: archetypal, and no amount of symmetry should buy that back. Each term contributes ``t ** w``, so
#: a term at 0 vetoes the leaf outright and a small term is punished far harder than an arithmetic
#: mean would punish it. Set a weight to 0.0 to drop a term entirely -- in particular
#: ``{"symmetry": 0.0}`` yields the symmetry-free composite the report needs to test the hypothesis
#: without circularity. The weights are RELATIVE -- ``_gmean`` divides by whatever they sum to --
#: so lowering one raises every other one's share.
ARCHETYPE_WEIGHTS: dict[str, float] = {
    "symmetry": 0.40,       # the hypothesis under test -- weighted highest ON PURPOSE, see caveat
    "integrity": 0.25,      # is it one whole undamaged blob
    # 0.05, cut from the 0.20 first proposed. The term cannot deliver 0.20 of an honest completeness
    # judgment: the only truncation evidence in the stored oriented products is size-confounded
    # (flush_edge_frac, r = +0.848 against 1/sqrt(area_px)) and it fires on 1 of 105 real leaves.
    # The weight now reflects what the term really is -- a flag for the rare genuinely clipped
    # detection box, backed by the hard `truncated` gate in is_archetypal -- rather than an equal
    # partner in the ranking. See _completeness_term for why nothing better is computable here.
    "completeness": 0.05,   # is all of it actually in frame (see caveat -- weak evidence)
    "trace": 0.15,          # do we trust the axis the symmetry was measured against
}

#: Hard gates for :func:`is_archetypal`. A leaf must clear the score threshold AND every gate.
#: Measured acceptance on the 105 real leaves of the test project, with real symmetry metrics, so
#: the threshold can be dialed in without re-deriving the table:
#: 0.70 -> 69%, 0.80 -> 56%, 0.85 -> 44%, 0.90 -> 30%, 0.92 -> 22%, 0.95 -> 10%.
#: 0.92 is shipped on the judgement that "archetypal" should name a select minority rather than the
#: merely-unbroken majority -- the structural gates below already reject the broken ones. This is
#: the single most likely constant to want changing after looking at the ranked gallery.
ARCHETYPE_MIN_SCORE = 0.92
ARCHETYPE_MAX_COMPONENTS = 1
ARCHETYPE_MIN_LARGEST_FRAC = 0.995
ARCHETYPE_MAX_HOLE_FRAC = 0.02
ARCHETYPE_MAX_FLUSH_EDGE = FLUSH_EDGE_ZERO
ARCHETYPE_MIN_MIDVEIN_KPTS = 10

#: A sub-score below this is worth explaining in the returned reasons.
REASON_SCORE_THRESHOLD = 0.98


def _ramp(x: float, zero_at: float, one_at: float) -> float:
    """Clipped linear map: ``zero_at -> 0``, ``one_at -> 1``. Works in either direction.

    NaN in, NaN out -- a missing measurement must not masquerade as a passing one.
    """
    if x is None:
        return float("nan")
    x = float(x)
    if not math.isfinite(x) or zero_at == one_at:
        return float("nan")
    return float(min(1.0, max(0.0, (x - zero_at) / (one_at - zero_at))))


def _gmean(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Weighted geometric mean, ignoring NaN entries and zero weights. NaN if nothing is left."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan")
    v, w = np.clip(v[ok], 0.0, 1.0), w[ok]
    if (v <= 0).any():
        return 0.0                       # one vetoing term; log would be -inf
    return float(np.exp(float(np.sum(w * np.log(v)) / float(w.sum()))))


def _amean(values: Sequence[float]) -> float:
    """Unweighted ARITHMETIC mean of the finite entries, clipped to ``[0, 1]``. NaN if none are.

    Used inside :func:`_trace_term` only. Across the four top-level terms the mean is geometric
    because "archetypal" is a conjunction of independent properties; the three trace inputs are
    three views of ONE property (how much of the midvein the pose model actually pinned down), and a
    zero in one of them must not annihilate a trace the geometry can still fit an axis to.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.clip(v, 0.0, 1.0).mean()) if v.size else float("nan")


def _sym_value(sym: dict, name: str, axis: str) -> tuple[float, str]:
    """Pull one symmetry metric out of ``sym``, tolerating the layouts metrics.py might use.

    Accepts, in order: ``sym[axis][name]``, ``sym[f"{axis}_{name}"]``, ``sym[f"{name}_{axis}"]``,
    ``sym[name]``; all case-insensitively, so ``SI_A``/``si_a`` and ``dice``/``Dice`` both resolve.
    Returns ``(value, axis_actually_used)`` with ``nan`` and ``""`` when nothing matches, rather
    than guessing -- a missing metric is dropped from the mean, never defaulted to a passing value.
    """
    if not isinstance(sym, dict):
        return float("nan"), ""

    def _num(x: Any) -> float:
        try:
            f = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return f if math.isfinite(f) else float("nan")

    flat = {str(k).lower(): v for k, v in sym.items()}
    nested = flat.get(axis.lower())
    if isinstance(nested, dict):
        sub = {str(k).lower(): v for k, v in nested.items()}
        if name.lower() in sub:
            return _num(sub[name.lower()]), axis
    for key in (f"{axis}_{name}", f"{name}_{axis}", name):
        if key.lower() in flat:
            return _num(flat[key.lower()]), axis
    return float("nan"), ""


def _symmetry_term(sym: dict) -> tuple[float, list[str]]:
    """Sub-score from the midvein-axis Dice and |SI_A|, falling back to the chord axis loudly."""
    reasons: list[str] = []
    axis = SYM_AXIS
    dice, used = _sym_value(sym, "dice", axis)
    sia, used_a = _sym_value(sym, "SI_A", axis)
    if not used and not used_a:
        # The chord axis systematically over-reads asymmetry on a curved leaf (verified: -0.0867 vs
        # -0.0216 on the same leaf), so scoring on it is a fallback worth announcing.
        dice, used = _sym_value(sym, "dice", "chord")
        sia, used_a = _sym_value(sym, "SI_A", "chord")
        if used or used_a:
            axis = "chord"
            reasons.append("no midvein-axis symmetry; scored on the chord axis, which over-reads "
                           "asymmetry for a curved midvein")
    # SI_A is mean(|per-bin imbalance|) -- UNSIGNED by construction (metrics.py), and on the 105 real
    # leaves it runs 0.027-0.878 with not one negative value. Printing it as "{:+.3f}" therefore
    # promised a direction it cannot carry: the leading + was an artifact of the format string, not
    # a side. A_star is the signed total imbalance (-0.892..+0.417 here, 52 of 105 negative, + means
    # the viewer's LEFT is larger), so the direction sentence is built from A_star and SI_A is
    # reported unsigned as the magnitude it is.
    astar, _used_s = _sym_value(sym, "A_star", axis) if (used or used_a) else (float("nan"), "")

    d_score = _ramp(dice, SYM_DICE_ZERO, SYM_DICE_ONE)
    a_score = _ramp(abs(sia) if math.isfinite(sia) else float("nan"), SYM_SIA_ZERO, SYM_SIA_ONE)
    if math.isfinite(d_score) and d_score < REASON_SCORE_THRESHOLD:
        reasons.append(f"{axis} half-lamina Dice {dice:.3f} (archetypal from {SYM_DICE_ONE:.2f})")
    if math.isfinite(a_score) and a_score < REASON_SCORE_THRESHOLD:
        side = ""
        if math.isfinite(astar):
            side = (f"; net A* {astar:+.3f}, {abs(astar) * 100:.1f}% more lamina on the "
                    f"{'left' if astar > 0 else 'right'}")
        reasons.append(f"{axis} area imbalance SI_A {abs(sia):.3f} unsigned "
                       f"({abs(sia) * 100:.1f}% of the lamina on one side in the average bin){side}")
    score = _gmean([d_score, a_score])
    if not math.isfinite(score):
        reasons.append("symmetry metrics unavailable; the leaf CANNOT be scored (the symmetry term "
                       "is the hypothesis, so it is never renormalized away)")
    return score, reasons


def _integrity_term(q: QualityDiagnostics) -> tuple[float, list[str]]:
    """Sub-score from fragmentation and holes: is the mask ONE WHOLE piece of leaf?"""
    reasons: list[str] = []
    frac = _ramp(q.largest_frac, LARGEST_FRAC_ZERO, LARGEST_FRAC_ONE)
    comp = 1.0 / (1.0 + COMPONENT_PENALTY_K * max(0, q.n_components - 1)) if q.n_components else float("nan")
    hole = _ramp(q.hole_frac, HOLE_FRAC_ZERO, HOLE_FRAC_ONE)

    if q.n_components and q.n_components > 1:
        pct = q.largest_frac * 100 if math.isfinite(q.largest_frac) else float("nan")
        reasons.append(f"silhouette split into {q.n_components} components "
                       f"(largest holds {pct:.1f}% of the area)")
    elif math.isfinite(frac) and frac < REASON_SCORE_THRESHOLD:
        reasons.append(f"largest component holds only {q.largest_frac * 100:.1f}% of the mask")
    if math.isfinite(hole) and hole < REASON_SCORE_THRESHOLD:
        reasons.append(f"holes cover {q.hole_frac * 100:.1f}% of the silhouette")
    score = _gmean([frac, comp, hole])
    if not math.isfinite(score):
        reasons.append("integrity undefined (empty mask); term dropped from the score")
    return score, reasons


def _completeness_term(q: QualityDiagnostics) -> tuple[float, list[str]]:
    """Sub-score for "all of the leaf is actually in the picture" -- WEAK EVIDENCE, read this.

    **No sound size-independent truncation measure is available from the data at hand, and this
    term is not one.** A genuine detector would ask what fraction of the outline lies on the
    ORIGINAL crop border -- the detection box intersected with the sheet, before the tip-up rotation
    and the content-fit -- but :class:`~.geometry.OrientedLeaf` keeps only the post-fit masks, the
    det box and ``truncated``; the pre-fit geometry and the sheet dimensions needed to tell "this
    border is the sheet edge" from "this border is where the detector chose to stop" are not
    carried, and the content fit guarantees the mask touches all four edges of the frame it does
    keep. So the term is scored on ``flush_edge_frac``, which is a leaf-SIZE measurement dressed as
    a cut-off detector: r = +0.848 against ``1 / sqrt(area_px)`` on the 105 real leaves, and exactly
    ONE of them ends up below 1.0. Its weight in :data:`ARCHETYPE_WEIGHTS` was cut to 0.05 to match
    that: the term is a flag for a genuinely clipped detection box (``truncated``, also a hard gate
    in :func:`is_archetypal`), not a graded completeness judgment. Fixing it properly means storing
    the pre-fit border contact in ``OrientedLeaf``, not re-tuning the constants here.
    """
    reasons: list[str] = []
    score = _ramp(q.flush_edge_frac, FLUSH_EDGE_ZERO, FLUSH_EDGE_FREE)
    if math.isfinite(score) and score < REASON_SCORE_THRESHOLD:
        reasons.append(f"mask lies flush along {q.flush_edge_frac * 100:.0f}% of one crop edge "
                       "(leaf likely cut off, but the number is size-confounded -- see the "
                       "flush_edge_frac docstring)")
    if q.truncated:
        reasons.append("detection box was clipped by the sheet edge (leaf may run off the sheet)")
        # nan stays nan. Substituting a 1.0 base here fabricated a finite 0.25 for a leaf with NO
        # measurement at all (an empty mask has no boundary and no flush edge), which reads as a
        # measured quarter-credit rather than as "undefined" and drags a real number into the mean.
        score = score * TRUNCATED_PENALTY
    if not math.isfinite(score):
        reasons.append("completeness undefined (no boundary); term dropped from the score")
    return score, reasons


def _trace_term(q: QualityDiagnostics) -> tuple[float, list[str]]:
    """Sub-score for "the midvein axis the symmetry was measured against is trustworthy".

    Count and confidence are scored JOINTLY, over all :data:`N_MIDVEIN_MAX` stations of the fixed
    pose skeleton rather than over the surviving points alone, because scoring the survivors alone
    made the term NON-MONOTONIC IN DATA QUALITY: deleting the single weakest keypoint of a 17-point
    trace took the term from 0.00 to 1.00 and flipped ``is_archetypal`` to True (measured), since the
    min-confidence penalty walked out with the point that caused it while the count barely moved.
    The pose model emits the whole skeleton every time, so a station that is not here is a station
    the model was not confident about -- absent evidence, not neutral evidence -- and it is scored
    as zero confidence. Every input below is then monotone non-decreasing in every station's
    confidence, so removing information can only lower the term, never raise it.
    """
    reasons: list[str] = []
    n = min(int(q.n_midvein_kpts), N_MIDVEIN_MAX)
    if n <= 0 or not math.isfinite(q.kpt_conf_mean) or not math.isfinite(q.kpt_conf_min):
        # No usable trace at all is an ABSENT measurement, not a bad one: let the term drop out and
        # let the hard gate in is_archetypal do the rejecting, rather than fabricating a zero -- or,
        # worse, a 1.0, which is what a nan-poisoned confidence pair used to produce here.
        return float("nan"), ["no usable midvein keypoint confidences; term dropped from the score"]

    # Coverage-weighted mean: the mean confidence over all N_MIDVEIN_MAX stations with the missing
    # ones contributing 0. Identical to the plain mean for a complete trace -- which is every leaf of
    # the test project -- so the ramp calibration is untouched.
    cov_mean = n * q.kpt_conf_mean / N_MIDVEIN_MAX
    # Worst station. A station that is not here was dropped for being below the loader's confidence
    # filter, so it is worse than anything that survived it; an incomplete trace takes the floor of
    # the ramp instead of the best-of-the-survivors minimum it used to be handed. Zero and the
    # filter threshold score the same here (KPT_CONF_MIN_ZERO is that threshold), so scoring an
    # unplaced station as zero everywhere costs nothing and keeps one rule for both statistics.
    worst = q.kpt_conf_min if n >= N_MIDVEIN_MAX else 0.0
    mean_s = _ramp(cov_mean, KPT_CONF_MEAN_ZERO, KPT_CONF_MEAN_ONE)
    min_s = _ramp(worst, KPT_CONF_MIN_ZERO, KPT_CONF_MIN_ONE)
    n_s = _ramp(float(n), MIDVEIN_KPTS_ZERO, MIDVEIN_KPTS_ONE)

    if math.isfinite(mean_s) and mean_s < REASON_SCORE_THRESHOLD:
        reasons.append(f"midvein keypoint confidence {q.kpt_conf_mean:.2f} mean, "
                       f"{q.kpt_conf_min:.2f} worst")
    if n < N_MIDVEIN_MAX:
        reasons.append(f"only {n}/{N_MIDVEIN_MAX} midvein trace points survived the confidence "
                       "filter (the missing ones are scored as unplaced, not ignored)")
    # _amean, not _gmean: see _amean. A trace missing stations is less trustworthy, not unusable,
    # and vetoing it here would re-create in the confidence inputs exactly the hard zero that
    # MIDVEIN_KPTS_ZERO was lowered to remove.
    score = _amean([mean_s, min_s, n_s])
    if not math.isfinite(score):
        reasons.append("midvein trace sub-scores undefined; term dropped from the score")
    return score, reasons


def archetype_subscores(sym: dict, qual: QualityDiagnostics) -> tuple[dict[str, float], list[str]]:
    """The four term scores in ``[0, 1]`` (NaN = unavailable) and every reason string raised.

    Exposed separately from :func:`archetype_score` so the report can plot, re-weight or audit the
    terms without re-deriving them, and so a symmetry-free composite is one call away.
    """
    s, r_s = _symmetry_term(sym if isinstance(sym, dict) else {})
    i, r_i = _integrity_term(qual)
    c, r_c = _completeness_term(qual)
    t, r_t = _trace_term(qual)
    return {"symmetry": s, "integrity": i, "completeness": c, "trace": t}, [*r_s, *r_i, *r_c, *r_t]


def archetype_score(sym: dict, qual: QualityDiagnostics, *,
                    weights: Optional[dict[str, float]] = None) -> tuple[float, list[str]]:
    """Composite "how archetypal is this leaf" score in ``[0, 1]`` plus the reasons for its penalties.

    ``sym`` is the metrics dict for one leaf (see :func:`_sym_value` for the key layouts accepted);
    ``qual`` comes from :func:`compute_quality`. ``weights`` overrides :data:`ARCHETYPE_WEIGHTS`
    key by key; unknown keys are ignored and a weight of 0 removes that term exactly.

    The four terms are combined as a WEIGHTED GEOMETRIC MEAN, so each contributes ``term ** weight``
    and a single term at 0 drives the whole score to 0. That is the intended semantics of
    "archetypal": it is a conjunction of properties, not a tally that a strength elsewhere can
    compensate for. A term whose inputs are missing is dropped and the remaining weights are
    renormalized, with a reason recorded -- a missing measurement never scores as a passing one.

    **The symmetry term is the one exception, and it is disqualifying.** Drop-and-renormalize is the
    right rule for a nice-to-have term; for the term that IS the hypothesis it is catastrophic --
    a leaf whose symmetry was never measured used to score a clean 1.0 off the three quality terms
    alone and outrank every leaf whose symmetry actually WAS measured. So when symmetry carries any
    weight and cannot be evaluated, this returns ``nan`` rather than a composite of what is left.
    Ask for the symmetry-free composite explicitly with ``weights={"symmetry": 0.0}`` and the
    disqualification lifts with it, because then no symmetry claim is being made.

    Returns ``nan`` when no term could be evaluated at all. The hard yes/no lives in
    :func:`is_archetypal`, kept separate so this function's return type stays a plain
    ``(score, reasons)`` pair for the report.

    Every threshold behind these numbers is a first proposal to be tuned against the ranked visual
    output -- see the constants block above.
    """
    w = dict(ARCHETYPE_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    terms, reasons = archetype_subscores(sym, qual)
    if _symmetry_required(w) and not math.isfinite(terms["symmetry"]):
        return float("nan"), reasons
    score = _gmean([terms[k] for k in w], [w[k] for k in w])
    return score, reasons


def _symmetry_required(weights: Optional[dict[str, float]]) -> bool:
    """Does this weighting actually claim to be scoring symmetry?

    True unless the caller deliberately zeroed the symmetry weight, which is the documented way to
    ask for the symmetry-free composite. Only then may a missing symmetry measurement be ignored.
    """
    if not weights:
        return ARCHETYPE_WEIGHTS["symmetry"] > 0
    try:
        return float(weights.get("symmetry", ARCHETYPE_WEIGHTS["symmetry"])) > 0
    except (TypeError, ValueError):
        return True


def is_archetypal(sym: dict, qual: QualityDiagnostics, *,
                  weights: Optional[dict[str, float]] = None,
                  score: Optional[float] = None) -> bool:
    """Hard accept/reject: a good composite score AND every structural gate cleared.

    The gates exist because the score is a smooth trade-off and some defects are categorical: a leaf
    in two pieces, one that ran off the sheet, or one whose symmetry was never measured at all, is
    not archetypal at any score. Pass ``score`` to reuse an already-computed value. Thresholds:
    :data:`ARCHETYPE_MIN_SCORE` and the ``ARCHETYPE_*`` constants -- again, first proposals meant to
    be tuned against the ranked output.
    """
    # An unmeasured symmetry disqualifies, and it has to be re-checked here rather than trusted to
    # come back as a nan score: a caller passing a precomputed `score` would otherwise walk straight
    # past the disqualification archetype_score applies.
    if _symmetry_required(weights) and not math.isfinite(_symmetry_term(
            sym if isinstance(sym, dict) else {})[0]):
        return False
    if score is None:
        score, _ = archetype_score(sym, qual, weights=weights)
    # numbers.Real, not float: a np.float32/np.float64 score is a perfectly good score, and
    # isinstance(np.float32(0.99), float) is False, so the strict check silently rejected every
    # leaf whose score arrived out of a numpy array. bool is a Real too, and is not a score.
    if isinstance(score, bool) or not isinstance(score, numbers.Real):
        return False
    score = float(score)
    if not (math.isfinite(score) and score >= ARCHETYPE_MIN_SCORE):
        return False
    if qual.truncated or qual.area_px <= 0:
        return False
    if qual.n_components > ARCHETYPE_MAX_COMPONENTS:
        return False
    if qual.n_midvein_kpts < ARCHETYPE_MIN_MIDVEIN_KPTS:
        return False
    for value, limit, worse_is_higher in (
        (qual.largest_frac, ARCHETYPE_MIN_LARGEST_FRAC, False),
        (qual.hole_frac, ARCHETYPE_MAX_HOLE_FRAC, True),
        (qual.flush_edge_frac, ARCHETYPE_MAX_FLUSH_EDGE, True),
    ):
        if not math.isfinite(value):
            return False
        if (value > limit) if worse_is_higher else (value < limit):
            return False
    return True
