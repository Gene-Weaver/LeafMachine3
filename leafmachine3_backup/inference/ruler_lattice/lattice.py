"""Lattice engine v2.

Changes from v1, all driven by inspecting the v1 QC output:

1. ROTATION, not shear. v1 sheared the crop to make ticks vertical, which skews
   the ruler and leaves the mask running across the ticks. v2 estimates the tick
   orientation, rotates the crop on an EXPANDED canvas with black padding, and
   carries a validity mask so the padding never contributes to any projection.

2. VERTICALITY-WEIGHTED tick response. |dI/dx| alone fires on digits, ruler
   borders and shadows. Weighting by (Rx-Ry)/(Rx+Ry) keeps thin vertical bars and
   suppresses horizontal edges and the curved strokes of numerals -- this is the
   alphanumeric rejection v1 only claimed to have.

3. ADAPTIVE bands from a row-wise period map, instead of fixed halves/thirds.
   A crop is a stack of scales; the band boundaries have to follow the scales.

4. MULTI-LEVEL ticks. Real rulers encode several units in ONE row by varying
   tick LENGTH (1/16 short, 1/8 taller, 1/4 taller still, 1 inch tallest). v2
   detects every tick on the finest lattice, measures each tick's length, and
   finds which residue classes k = r (mod m) are systematically longer. Those
   multipliers m are the coarser units, and they cross-validate against the known
   unit table in ruler_units.
"""
from __future__ import annotations

import math
import numpy as np
import cv2

EPS = 1e-6
# Asked to drop 7 -> 3; measured the sweep on the 69 GT crops instead of guessing:
#   3 -> 51/69 within 3%, p90 58.6%      4 -> 53/69, p90 55.7%
#   5 -> 53/69, p90 55.7%                7 -> 53/69, p90 58.9%
# The DBG crops that motivated lowering this are fixed identically at EVERY value
# 3..7 -- their bug was `class_units()` returning a metric default, not this floor.
# 7 keeps the best within-1% (61%) on the GT set, so it stays. Kept at Will's call.
# 3 px is only 1.5 px of tick + 1.5 px of gap; the period is still recoverable from
# many cycles, but a per-tick MASK there is ~1 px wide and not usable, and JPEG 8x8
# blocking harmonics land at 4 / 2.7 px.
MIN_PERIOD = 7.0
MAX_LEVELS = 5


# --------------------------------------------------------------------------- #
# 1. Response
# --------------------------------------------------------------------------- #
def gradients(gray):
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 0.9)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy


def verticality(gx, gy):
    """+1 for a purely vertical edge, 0 isotropic, -1 purely horizontal."""
    rx, ry = np.abs(gx), np.abs(gy)
    return (rx - ry) / (rx + ry + EPS)


def body_response(gray):
    """Horizontal-edge response: the ruler's long top/bottom borders and the
    block-ruler rails. Mirror image of tick_response."""
    gx, gy = gradients(gray)
    v = verticality(gx, gy)
    return np.abs(gy) * np.clip(-v, 0.0, 1.0)


def tick_response(gray):
    """Vertical-bar response. Zero on horizontal edges (ruler borders, shadows)
    and heavily attenuated on the curved/oblique strokes of printed numerals."""
    gx, gy = gradients(gray)
    v = verticality(gx, gy)
    return np.abs(gx) * np.clip(v, 0.0, 1.0)


def text_penalty(gray, ksize=None):
    """Local fraction of edge energy that is NOT vertical -- high over lettering
    and numerals, near zero over tick marks."""
    gx, gy = gradients(gray)
    rx, ry = np.abs(gx), np.abs(gy)
    k = ksize or 9
    sx = cv2.blur(rx, (k, k))
    sy = cv2.blur(ry, (k, k))
    return sy / (sx + sy + EPS)


# --------------------------------------------------------------------------- #
# 2. Orientation + rotation with black padding
# --------------------------------------------------------------------------- #
def rotate_bound(img, angle_deg, border=0.0, interp=cv2.INTER_CUBIC):
    """Rotate CCW by angle_deg onto an expanded canvas, padding with `border`."""
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(math.ceil(h * sin + w * cos))
    nh = int(math.ceil(h * cos + w * sin))
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(img, M, (nw, nh), flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def edge_orientation(gray, keep=0.30, max_tilt=35.0):
    """Magnitude-weighted median tilt of the near-VERTICAL edges (the ticks),
    plus the near-HORIZONTAL edges (the ruler body) as a PCA-style cross-check."""
    gx, gy = gradients(gray)
    mag = np.hypot(gx, gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 90.0) % 180.0 - 90.0   # (-90, 90]
    strong = mag >= np.percentile(mag, 100 * (1 - keep))

    def wmedian(sel):
        if sel.sum() < 30:
            return None, 0.0
        a, w = ang[sel], mag[sel] ** 2
        o = np.argsort(a); a, w = a[o], w[o]
        c = np.cumsum(w)
        t = float(a[np.searchsorted(c, 0.5 * c[-1])])
        inl = np.abs(ang - t) < 6.0
        return t, float((mag[inl] ** 2).sum() / max((mag ** 2).sum(), EPS))

    tick_t, tick_c = wmedian(strong & (np.abs(ang) < max_tilt))
    # horizontal edges: gradient near +/-90 deg
    body_t, body_c = wmedian(strong & (np.abs(ang) > 90.0 - max_tilt))
    if body_t is not None:                       # express as a tilt from horizontal
        body_t = body_t - math.copysign(90.0, body_t)
    return (tick_t or 0.0), tick_c, body_t, body_c


def valid_profile(resp, valid, y0, y1, min_cov=0.6):
    """Column mean of `resp` over rows [y0,y1) using only valid pixels.

    Coverage is measured against the BEST-COVERED COLUMN, not against the slice
    height. rotate_bound expands the canvas as the angle grows, so a fixed
    fraction of the canvas is a moving target: a 1228x189 crop rotated 6 deg sits
    on a 317-row canvas while its content is still only 191 rows, i.e. 0.60 of the
    canvas. Measured against the canvas, that crop lost every valid column past
    6 deg -- and AT 6 deg only 46 of 1228 columns survived, a sliver whose CV^2
    spiked to 74 against ~2 for the true interior optimum. Normalising by the best
    column makes the test scale-free: interior columns stay ~1.0 at any angle and
    only genuinely clipped corners drop out.
    """
    r = resp[y0:y1]
    v = valid[y0:y1].astype(np.float32)
    num = (r * v).sum(axis=0)
    den = v.sum(axis=0)
    ref = float(den.max()) if den.size else 0.0
    if ref <= 0:
        return None, None
    cov = den / ref
    ok = cov >= min_cov
    if ok.sum() < 40:
        return None, None
    prof = np.zeros_like(num)
    prof[ok] = num[ok] / np.maximum(den[ok], EPS)
    xs = np.where(ok)[0]
    return prof[xs[0]:xs[-1] + 1], xs[0]


def bandpass(p, win=121):
    win = max(5, min(int(win), (len(p) // 2) * 2 - 1))
    if win % 2 == 0:
        win += 1
    lo = cv2.blur(np.asarray(p, np.float32).reshape(-1, 1), (1, win)).ravel()
    q = np.asarray(p, np.float64) - lo
    s = q.std()
    return q / s if s > EPS else q


def _acf(profile):
    """Normalised autocorrelation of the band-passed profile, plus the usable
    lag range. Split out of `periodicity` so the sub-harmonic search can reuse
    exactly the same curve the argmax was taken from."""
    if profile is None or len(profile) < 40:
        return None, 0, 0
    p = bandpass(profile)
    n = len(p)
    f = np.fft.rfft(p * np.hanning(n), n=2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    if a[0] <= EPS:
        return None, 0, 0
    return a / a[0], int(MIN_PERIOD), min(n // 3, n - 2)


def _interp_lag(a, lag):
    """Sub-sample the ACF peak at integer `lag` by parabolic interpolation."""
    n = len(a)
    lag = float(lag)
    i = int(lag)
    if 0 < i < n - 1:
        d = a[i - 1] - 2 * a[i] + a[i + 1]
        if abs(d) > 1e-12:
            lag += float(np.clip(0.5 * (a[i - 1] - a[i + 1]) / d, -0.5, 0.5))
    return lag


def periodicity(profile, min_p=MIN_PERIOD):
    if profile is None or len(profile) < 40:
        return 0.0, float("nan")
    p = bandpass(profile)
    n = len(p)
    f = np.fft.rfft(p * np.hanning(n), n=2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    if a[0] <= EPS:
        return 0.0, float("nan")
    a = a / a[0]
    lo, hi = int(min_p), min(n // 3, n - 2)
    if hi <= lo + 1:
        return 0.0, float("nan")
    seg = a[lo:hi + 1]
    k = int(np.argmax(seg))
    lag = lo + k
    if 0 < lag < n - 1:
        d = a[lag - 1] - 2 * a[lag] + a[lag + 1]
        if abs(d) > 1e-12:
            lag += float(np.clip(0.5 * (a[lag - 1] - a[lag + 1]) / d, -0.5, 0.5))
    return float(seg[k]), float(lag)


# --------------------------------------------------------------------------- #
# Sub-harmonics of the ACF argmax
# --------------------------------------------------------------------------- #
# A tick comb of period P autocorrelates at EVERY multiple of P and at NO
# divisor of P -- shift a comb by P/2 and it lands squarely in the gaps, which is
# a TROUGH. The implication runs one way only: a real ACF peak at lag/d proves
# that `lag` is a harmonic, while a peak at `lag` says nothing at all about
# whether `lag` is fundamental. A bare argmax is therefore biased COARSE, and on
# a real tick row the fundamental and its harmonics are separated by far less
# than the measurement noise -- on one crop P, 2P and 3P scored 0.9502 / 0.9558 /
# 0.9559, so the FUNDAMENTAL scored lowest and argmax took 3P. Once a multiple is
# locked, naming it is hopeless: 3 mm is not a printed unit, so it gets rounded
# to the nearest name (2.5 mm or 5 mm) and a 20-60% scale error is baked in.
#
# So the sub-harmonics are not a correction applied to the argmax -- they are
# extra CANDIDATE periods handed to the class-aware naming step, which then
# decides among them on physical evidence (implied ruler length, the MP CF
# anchor, the class's finest printed unit). That matters because the smallest
# supported lag is not always right either: on a 1/8-in class whose row also
# carries 1/16-in ticks the fundamental IS the 1/16 lattice, but the class cannot
# name it and reading it as 1/8 in halves the scale. Only the naming step knows
# that; the ACF does not.
HARM_DIVS = (5, 4, 3, 2)
# A divisor lag must retain this fraction of the argmax correlation to be offered
# as a candidate at all. Deliberately high: a sub-lattice that is genuinely
# printed correlates almost as well as its own harmonics, whereas a spurious one
# (every other cell empty) does not. It is also what protects the classes whose
# COARSE row period is real by design -- on a CM_STAGGER / *_BLOCK ruler the lag
# at half the row period sits in the anti-correlation trough, nowhere near the
# peak, so no sub-harmonic is ever offered.
FUND_RATIO = 0.80
# The ACF is sampled on INTEGER lags, so a sub-period almost never lands on one:
# a fundamental of 7.35 px appears as a peak at lag 7, which is 6.8% away from
# 14.7/2. The admissible offset is therefore an ABSOLUTE half-sample plus a small
# relative slack -- a pure percentage is far too permissive at long lags and far
# too tight at short ones.
FUND_POS_ABS = 0.75
FUND_POS_REL = 0.04


def subharmonic_lags(profile, ratio=FUND_RATIO, divs=HARM_DIVS, max_steps=4):
    """ACF argmax lag, then every sub-harmonic lag the ACF itself supports.

    Returned coarsest first, sub-sample interpolated. The walk is CHAINED
    (lag -> lag/3 -> lag/6) rather than testing lag/d for every d in one shot,
    because each step is verified against the ACF locally: an argmax sitting on
    the 6th harmonic reaches the fundamental via 6 -> 2 -> 1, and lag/6 tested
    directly would fall outside the position tolerance once integer-lag
    quantisation is compounded over the whole jump.
    """
    a, lo, hi = _acf(profile)
    if a is None or hi <= lo + 1:
        return []
    n = len(a)
    cur = lo + int(np.argmax(a[lo:hi + 1]))
    ref = float(a[cur])
    out = [cur]
    for _ in range(max_steps):
        nxt = None
        for d in divs:                   # largest divisor first = biggest step down
            c0 = cur / float(d)
            if c0 < lo:
                continue
            w = max(1, int(math.ceil(FUND_POS_REL * c0)))
            i0 = max(lo, int(math.floor(c0)) - w)
            i1 = min(n - 2, int(math.ceil(c0)) + w)
            if i1 <= i0:
                continue
            j = i0 + int(np.argmax(a[i0:i1 + 1]))
            if j <= 0 or j >= n - 1:
                continue
            if abs(j - c0) > max(FUND_POS_ABS, FUND_POS_REL * c0):
                continue                 # not actually at cur/d
            if not (a[j] >= a[j - 1] and a[j] >= a[j + 1]):
                continue                 # on a slope, not a peak in its own right
            if a[j] >= ratio * ref:
                nxt = j
                break
        if nxt is None or nxt >= cur:
            break
        cur = nxt
        out.append(cur)
    return [_interp_lag(a, c) for c in out]


# --------------------------------------------------------------------------- #
# The ACF peak COMB -- a period the argmax search structurally cannot return
# --------------------------------------------------------------------------- #
COMB_PEAK_FLOOR = 0.12   # an ACF local maximum under this fraction of the argmax is noise
COMB_MIN_PEAKS = 4       # fewer maxima than this is a coincidence, not a comb
COMB_EVENNESS = 0.12     # median gap deviation allowed before the series is not a comb


def comb_lag(profile):
    """Tick period inferred from the SPACING of the ACF's peaks, or None.

    ``MIN_PERIOD`` floors the ACF search at 7 px, so a ruler whose finest graduation images
    smaller than that has a fundamental OUTSIDE the search window -- not merely missed, but
    unreturnable by argmax or by any sub-harmonic descent, both of which can only report a lag
    they are allowed to look at. Measured case: A_1989713254 prints 1 mm ticks at 5.93 px and was
    read at 94.82 px/cm against a truth of 59.26.

    What survives above the floor is the comb of that fundamental's harmonics -- 12, 17, 23, 29,
    35, ... -- evenly spaced by the fundamental itself. So the period is recoverable as the SPACING
    even when it is invisible as a lag, and this returns values below MIN_PERIOD by design.

    The spacing is taken as the slope of peak position against peak INDEX. Positions advance by
    exactly one period per index regardless of which harmonic the series starts on, so the starting
    k never has to be identified. Fitting ``round(peak/P)`` multiples instead was tried and is
    worse: ACF lags are integers, the k assignment locks onto that half-pixel quantisation, and on
    the crop above it converged to 6.08 px with a 1.2 px residual where the index regression gives
    5.84 px.

    Evaluated on 896 ruler crops from 549 human-measured sheets (see
    ``leafmachine3/modules/experiments/ACF_comb``): offered as an extra candidate it lifts sub-5%
    readings 858 -> 870 and halves catastrophic (>25%) readings 15 -> 7, with 18 crops moving from
    loss to win against 6 moving the other way, none worse than 8%.
    """
    a, lo, hi = _acf(profile)
    if a is None or hi <= lo + 1:
        return None
    hi = min(hi, len(a) - 2)
    if hi <= lo:
        return None
    ref = float(np.max(a[lo:hi + 1]))
    if ref <= 0:
        return None
    peaks = [i for i in range(max(lo, 1), hi)
             if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > COMB_PEAK_FLOOR * ref]
    if len(peaks) < COMB_MIN_PEAKS:
        return None
    y = np.asarray(peaks, dtype=float)
    P = float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])
    if not np.isfinite(P) or P < 1.0:
        return None
    if float(np.median(np.abs(np.diff(y) - P))) > COMB_EVENNESS * P:
        return None                      # ragged maxima are not a comb; do not fit one
    return P


def projection_sharpness(rot, valid, w0):
    """Deskew objective: how PEAKY is the column projection of the tick response?

    When the ticks are truly vertical every tick concentrates into one column and
    the projection is a comb of tall narrow spikes; misaligned, each tick smears
    over several columns and the projection flattens. Coefficient of variation
    squared measures exactly that, and it is scale-free.

    This deliberately does NOT use the autocorrelation peak. ACF peak height is
    biased toward COARSE periodicity -- it will happily rotate a ruler by 19 deg
    to trade a sharp mm comb for a smoother cm comb -- which is the mistake that
    produced the wrong angles in the first v2 run.
    """
    def cv2_of(prof):
        if prof is None or len(prof) < 8:
            return None
        m = float(prof.mean())
        return None if m <= EPS else float(prof.var()) / (m * m)

    # ticks vertical -> the COLUMN projection of the vertical-edge response peaks
    pv, _ = valid_profile(tick_response(rot), valid, 0, rot.shape[0])
    if pv is None or len(pv) < max(40, 0.5 * w0):
        return -1.0
    sv = cv2_of(pv)
    if sv is None:
        return -1.0

    # ruler level -> the ROW projection of the horizontal-edge response peaks.
    # Ticks are perpendicular to the ruler body, so one rotation satisfies both;
    # scoring both makes the estimate robust on block rulers, which have few thin
    # vertical strokes but very strong horizontal rails.
    br = body_response(rot)
    v = valid.astype(np.float32)
    num = (br * v).sum(axis=1)
    den = v.sum(axis=1)
    ok = den >= 0.6 * rot.shape[1]
    sh = cv2_of(num[ok] / np.maximum(den[ok], EPS)) if ok.sum() >= 8 else None

    return sv if sh is None else sv + sh


def orient(gray, span=20.0, coarse=0.5, fine=0.05):
    """Rotate so the ticks stand vertical.

    A wide sweep on the sharpness objective, rather than a narrow refinement
    around an edge-orientation guess: that guess is unreliable exactly where it
    matters most (block rulers have few vertical strokes, so tick-orientation
    coherence is low), and anchoring to it capped the correction at a few degrees.
    """
    t0, tc, body, bc = edge_orientation(gray)
    ones = np.ones_like(gray, np.float32)
    g32 = gray.astype(np.float32)
    w0 = gray.shape[1]
    cache = {}

    def score_of(angle):
        key = round(float(angle), 4)
        if key not in cache:
            rot = rotate_bound(g32, key, 0.0)
            val = rotate_bound(ones, key, 0.0, cv2.INTER_NEAREST) > 0.5
            cache[key] = projection_sharpness(rot, val, w0)
        return cache[key]

    grid = np.arange(-span, span + 1e-9, coarse)
    scored = [(float(a), score_of(a)) for a in grid]
    valid_a = [a for a, v in scored if v > -1.0]
    # An argmax at the extreme of the valid span is not trustworthy even after the
    # coverage fix: it has no interior neighbour to corroborate it. Require a valid
    # neighbour on BOTH sides before accepting a candidate.
    interior = {a for a in valid_a
                if (a - coarse) in set(valid_a) and (a + coarse) in set(valid_a)}
    pool = [(a, v) for a, v in scored if a in interior] or \
           [(a, v) for a, v in scored if v > -1.0] or scored
    best = float(max(pool, key=lambda t: t[1])[0])
    for a in np.arange(best - coarse, best + coarse + 1e-9, fine):
        if score_of(a) > score_of(best):
            best = float(a)

    ang = float(best)
    rot = rotate_bound(g32, ang, 0.0)
    val = rotate_bound(ones, ang, 0.0, cv2.INTER_NEAREST) > 0.5
    return dict(angle=ang, score=score_of(ang), tick_tilt=t0, tick_coh=tc,
                body_tilt=body, body_coh=bc,
                rot=np.clip(rot, 0, 255).astype(np.uint8), valid=val)


# --------------------------------------------------------------------------- #
# 3. Adaptive bands
# --------------------------------------------------------------------------- #
def comb_support(p_bp, pp):
    n = len(p_bp)
    if not np.isfinite(pp) or pp < MIN_PERIOD or pp > n / 3.0:
        return -1e9, 0.0
    xs = np.arange(n, dtype=np.float64)
    z = np.sum(np.clip(p_bp, 0, None) * np.exp(-2j * np.pi * xs / pp))
    phase = ((-np.angle(z) / (2 * np.pi)) * pp) % pp
    k = np.arange(int((n - 1 - phase) / pp) + 1)
    if len(k) < 4:
        return -1e9, phase

    def s(off):
        pos = phase + k * pp + off
        pos = pos[(pos >= 0) & (pos < n - 1)]
        if len(pos) < 3:
            return np.array([0.0])
        i0 = pos.astype(int); fr = pos - i0
        return p_bp[i0] * (1 - fr) + p_bp[i0 + 1] * fr

    return float(np.median(s(0.0)) - np.median(np.concatenate([s(pp / 2), s(-pp / 2)]))), float(phase)


def refine_period(p_bp, pp, frac=0.06, steps=121):
    best = (-1e9, pp, 0.0)
    for q in np.linspace(pp * (1 - frac), pp * (1 + frac), steps):
        sc, ph = comb_support(p_bp, q)
        if sc > best[0]:
            best = (sc, float(q), ph)
    return best[1], best[2], best[0]


def candidate_bands(rot, valid):
    """Fixed 1/2/3/4 splits UNION the adaptive runs.

    Measured on the 69 GT crops: fixed splits alone pick the scale-bearing band
    better than adaptive runs alone (72% vs 65% within 3%), because a fixed split
    always offers the whole-crop band and clean halves, while an adaptive run can
    merge two scales or clip one. Taking the union and choosing by periodicity
    gets both -- adaptive runs still win where the scales genuinely stratify.
    """
    H = rot.shape[0]
    out = []
    for nb in (1, 2, 3, 4):
        for i in range(nb):
            y0, y1 = i * H // nb, (i + 1) * H // nb
            if y1 - y0 >= 8:
                out.append((y0, y1, "fixed"))
    for (y0, y1, sc, per) in adaptive_bands(rot, valid):
        if y1 - y0 >= 8:
            out.append((y0, y1, "adaptive"))
    seen, uniq = set(), []
    for y0, y1, kind in out:
        if (y0, y1) not in seen:
            seen.add((y0, y1)); uniq.append((y0, y1, kind))
    return uniq


def adaptive_bands(rot, valid, min_score=0.55):
    """Row-wise period map -> contiguous runs of rows that share one scale."""
    H = rot.shape[0]
    resp = tick_response(rot)
    win = max(8, H // 12)
    step = max(2, win // 4)
    rows = []
    for y in range(0, max(1, H - win + 1), step):
        prof, _ = valid_profile(resp, valid, y, y + win)
        # Raw argmax on purpose -- see subharmonic_lags(). This row map never
        # names a unit; it only asks "do these two neighbouring row windows share
        # ONE scale?" by comparing log(p_i / p_j). That test needs the windows to
        # report the SAME harmonic, not the physically fundamental one: a
        # sub-harmonic step is a discrete switch that can fire on one window and
        # not the next, splitting a single band in half for no reason.
        s, p = periodicity(prof)
        rows.append((y, y + win, s, p))
    if not rows:
        return [(0, H, 0.0, float("nan"))]

    bands, cur = [], []
    for (y0, y1, s, p) in rows:
        good = s >= min_score and np.isfinite(p)
        if good and cur and abs(math.log(p / cur[-1][3])) < 0.18:
            cur.append((y0, y1, s, p))
        elif good:
            if cur:
                bands.append(cur)
            cur = [(y0, y1, s, p)]
        else:
            if cur:
                bands.append(cur); cur = []
    if cur:
        bands.append(cur)

    out = []
    for b in bands:
        y0 = min(r[0] for r in b); y1 = max(r[1] for r in b)
        out.append((y0, min(y1, H), float(max(r[2] for r in b)),
                    float(np.median([r[3] for r in b]))))
    if not out:
        prof, _ = valid_profile(resp, valid, 0, H)
        s, p = periodicity(prof)
        out = [(0, H, s, p)]
    out.sort(key=lambda t: -t[2])
    return out


# --------------------------------------------------------------------------- #
# 4. Tick extraction with length, on the finest lattice
# --------------------------------------------------------------------------- #
def descend_finest(i_bp, P0, min_ratio=0.5, divisors=(2, 3, 4, 5, 8, 10, 16)):
    """The ACF usually locks a coarse, smooth lattice. Step down to the finest
    sub-lattice that still carries real comb support on the intensity profile."""
    _, ph0, sup0 = refine_period(i_bp, P0)
    best = (P0, ph0, 1)
    for d in divisors:
        q = P0 / d
        if q < MIN_PERIOD:
            continue
        qq, qph, sq = refine_period(i_bp, q, frac=0.03, steps=61)
        if sq > min_ratio * sup0 and d > best[2]:
            best = (qq, qph, d)
    return best


def band_profiles(rot, valid, band):
    y0, y1 = band
    resp_prof, off_r = valid_profile(tick_response(rot), valid, y0, y1)
    inten_prof, off_i = valid_profile(rot.astype(np.float32), valid, y0, y1)
    if resp_prof is None or inten_prof is None:
        return None
    q = bandpass(inten_prof)
    pol = -1.0 if abs(q.min()) > abs(q.max()) else 1.0
    return dict(resp=resp_prof, resp_off=off_r,
                inten=inten_prof, inten_off=off_i, i_bp=q * pol, pol=pol)


def cell_peaks(dev, band, P, phase, x_off, W):
    """Per-cell peak ink amplitude, before any thresholding. Used to set an
    adaptive amplitude floor and to test for an over-descended lattice."""
    y0, y1 = band
    out = []
    k0 = int(math.floor((0 - phase) / P))
    k1 = int(math.ceil((W - x_off - phase) / P))
    for k in range(k0, k1 + 1):
        c = x_off + phase + k * P
        a, b = int(round(c - P / 2)), int(round(c + P / 2))
        a, b = max(0, a), min(W, b)
        if b - a < 3:
            continue
        seg = dev[y0:y1, a:b]
        out.append((k, float(seg.max()) if seg.size else 0.0))
    return out


def alternation(peaks, max_m=5, min_ratio=0.35):
    """Is this lattice a factor m TOO FINE?

    Minor and major ticks are printed in the same ink, so their peak amplitudes
    are comparable -- what differs is length. But the phantom cells of an
    over-descended lattice have almost no ink at all. So if the cells NOT in some
    residue class carry < min_ratio of the ink of those in it, the true lattice is
    m times coarser. Returns the offending m, or None.
    """
    if len(peaks) < 8:
        return None
    ks = np.array([k for k, _ in peaks])
    pv = np.array([v for _, v in peaks], float)
    for m in range(2, max_m + 1):
        if len(peaks) // m < 3:
            break
        best = None
        for r in range(m):
            sel = (ks % m) == r
            if sel.sum() < 3 or (~sel).sum() < 3:
                continue
            on, off = np.median(pv[sel]), np.median(pv[~sel])
            if on <= 0:
                continue
            if best is None or off / on < best[0]:
                best = (off / on, r)
        if best and best[0] < min_ratio:
            return m
    return None


def extract_ticks(rot, valid, band, P, phase, x_off, pol, txt_pen,
                  min_amp=4.0, max_text=0.80):
    """One tick per lattice cell: its x centre, vertical extent, length, ink
    strength, and rejection flags. Length is what encodes the unit hierarchy."""
    y0, y1 = band
    H, W = rot.shape
    img = rot.astype(np.float32)
    # background along x only, so vertical bars survive and illumination does not
    bg = cv2.GaussianBlur(img, (0, 0), sigmaX=max(2.0, P * 0.8), sigmaY=0.6)
    dev = (img - bg) * pol
    dev[~valid] = 0.0

    # adaptive amplitude floor: a real tick on THIS ruler, not an absolute
    # grey-level constant. p75 survives even when half the cells are phantoms.
    pk = cell_peaks(dev, band, P, phase, x_off, W)
    if pk:
        ref = float(np.percentile([v for _, v in pk], 75))
        min_amp = max(min_amp, 0.35 * ref)

    ticks = []
    n_cells = 0
    k0 = int(math.floor((0 - phase) / P))
    k1 = int(math.ceil((W - x_off - phase) / P))
    for k in range(k0, k1 + 1):
        c = x_off + phase + k * P
        a, b = int(round(c - P / 2)), int(round(c + P / 2))
        a, b = max(0, a), min(W, b)
        if b - a < 3:
            continue
        n_cells += 1
        cell = dev[:, a:b]
        colmax = cell.max(axis=1)                     # per-row strength
        inband = colmax[y0:y1]
        if inband.size == 0:
            continue
        peak = float(inband.max())
        if peak < min_amp:
            continue
        thr = max(min_amp, 0.40 * peak)
        # grow vertically from the strongest row inside the band
        ys = y0 + int(np.argmax(inband))
        t, bmt = ys, ys
        while t - 1 >= 0 and colmax[t - 1] >= thr:
            t -= 1
        while bmt + 1 < H and colmax[bmt + 1] >= thr:
            bmt += 1
        length = bmt - t + 1
        seg = cell[t:bmt + 1]
        mk = (seg >= thr).astype(np.uint8)
        if mk.sum() == 0:
            continue
        xs = np.where(mk.any(axis=0))[0]
        ys_ = np.where(mk.any(axis=1))[0]
        wpx = xs[-1] - xs[0] + 1
        hpx = ys_[-1] - ys_[0] + 1
        tp = float(txt_pen[t:bmt + 1, a:b][mk > 0].mean())

        # Morphological shape gate. Every feature we want is a filled RECTANGLE
        # (picket-fence tick, or a block-ruler cell edge), so:
        #   rectangularity = area / bbox area  -> ~1 for a bar, ~0.4-0.6 for a digit
        #   row-width CV                       -> ~0 for a bar, large for a digit
        # This is a far better discriminator than an edge-orientation heuristic,
        # and unlike perimeter^2/area it does not penalise thin bars (a 3x40 bar
        # has a high P^2/A purely from its aspect ratio).
        box = mk[ys_[0]:ys_[-1] + 1, xs[0]:xs[-1] + 1]
        area = float(box.sum())
        rect = area / max(1.0, float(box.shape[0] * box.shape[1]))
        rw = box.sum(axis=1).astype(float)
        rw_cv = float(rw.std() / max(rw.mean(), EPS))

        # Shape gate. Everything we want is a filled RECTANGLE -- either a thin
        # picket-fence tick or a block-ruler cell. So a wide component is only
        # rejected when it is also SHORT, which is the signature of a horizontal
        # shadow or border smear; a wide, TALL component is a legitimate block.
        band_h = max(1, y1 - y0)
        wide_and_short = (wpx > 0.85 * (b - a)) and (hpx < 0.45 * band_h)
        # `rw_cv > 0.85` was here and never fired: removing it leaves all 69 GT
        # crops bit-identical in pxcm. rect < 0.40 already subsumes it, because a
        # blob with wildly varying row widths cannot fill 40% of its bbox.
        reject = wide_and_short or (rect < 0.40) or (tp > max_text)
        ticks.append(dict(k=k, xc=c, a=a, b=b, y0=t, y1=bmt + 1, length=length,
                          peak=peak, width=wpx, height=hpx, text=tp,
                          rect=rect, rw_cv=rw_cv, mask=mk, reject=reject))
    # n_cells counts EVERY lattice position examined, including the ones with no
    # tick at all. Occupancy = len(ticks)/n_cells is what exposes an over-descended
    # lattice: a spurious x2 sub-lattice leaves every other cell empty, so it sits
    # near 0.5 while a true lattice sits near 1.0.
    return ticks, n_cells, pk


# --------------------------------------------------------------------------- #
# 5. Tick LEVELS -- several units encoded in one row by tick length
# --------------------------------------------------------------------------- #
def find_levels(ticks, max_m=16, min_gain=0.18):
    """Which residue classes k = r (mod m) are systematically LONGER?

    That is how a ruler puts 1/16, 1/8, 1/4, 1/2 and 1 inch on a single row.
    Returns a nested, ascending list of (m, r, gain).
    """
    good = [t for t in ticks if not t["reject"]]
    if len(good) < 8:
        return []
    ks = np.array([t["k"] for t in good])
    ln = np.array([t["length"] for t in good], float)
    base = float(np.median(ln))
    if base <= 0:
        return []

    found = []
    for m in range(2, max_m + 1):
        if len(good) // m < 2:
            break
        best = None
        for r in range(m):
            sel = (ks % m) == r
            oth = ~sel
            if sel.sum() < 2 or oth.sum() < 2:
                continue
            gain = (np.median(ln[sel]) - np.median(ln[oth])) / base
            if best is None or gain > best[0]:
                best = (gain, r, int(sel.sum()))
        if best and best[0] >= min_gain:
            found.append((m, best[1], float(best[0])))

    # keep a NESTED ladder: each level must be a multiple of the previous one
    found.sort(key=lambda t: t[0])
    ladder = []
    for m, r, g in found:
        if not ladder or m % ladder[-1][0] == 0:
            # MEASURED spacing of this level, from its own tick positions. This
            # is what makes cross-validation real: computing it as m*P_base would
            # be algebraically identical to the base estimate and could never
            # disagree with it.
            xs = np.array([t["xc"] for t in good if t["k"] % m == r], float)
            xs.sort()
            spacing = float(np.median(np.diff(xs))) if len(xs) >= 3 else float("nan")
            ladder.append((m, r, g, spacing))
        if len(ladder) >= MAX_LEVELS - 1:
            break
    return ladder


def level_spacing_ok(m, spacing, P_base, tol=0.06):
    """Does a level's measured spacing really equal m x the base period?"""
    if not np.isfinite(spacing) or P_base <= 0:
        return False
    return abs(spacing / (m * P_base) - 1.0) <= tol


def level_of(tick, ladder):
    """Deepest level index a tick belongs to (0 = finest)."""
    lv = 0
    for i, (m, r, *_rest) in enumerate(ladder, start=1):
        if tick["k"] % m == r:
            lv = i
    return lv


# --------------------------------------------------------------------------- #
# Transition rulers: ONE row whose spacing changes along x (mm run -> cm run)
# --------------------------------------------------------------------------- #
def detect_transition(i_bp, expect_ratio=10.0, tol=0.20, n_splits=17, margin=0.02):
    """Find the x where a `transition` ruler switches from its fine run to its
    coarse run, and measure both periods.

    A metric__MM_CM ruler does not interleave mm and cm ticks -- it prints mm
    across one part of the strip and cm across the rest. So the two periods live
    in different x ranges and a single global comb sees a blend of them. Locating
    the split gives two INDEPENDENT period measurements whose ratio is known a
    priori (10 for mm->cm), which is a cross-validation that needs no ground
    truth: if P_coarse / P_fine comes out at 10, both measurements are confirmed.
    """
    n = len(i_bp)
    if n < 120:
        return None
    # margin 0.02: the transition often sits very close to one end of the crop, so
    # the split is searched across almost the whole profile. The per-side minimum
    # of 50 samples below is what actually stops a degenerate split.
    lo, hi = int(margin * n), int((1 - margin) * n)
    best = None
    for s in np.linspace(lo, hi, n_splits):
        s = int(s)
        left, right = i_bp[:s], i_bp[s:]
        if len(left) < 50 or len(right) < 50:
            continue
        sl, pl = periodicity(left)
        sr, pr = periodicity(right)
        if not (np.isfinite(pl) and np.isfinite(pr)):
            continue
        pl, _, supl = refine_period(left, pl)
        pr, _, supr = refine_period(right, pr)
        if supl <= 0 or supr <= 0:
            continue
        fine, coarse = (pl, pr) if pl <= pr else (pr, pl)
        ratio = coarse / max(fine, EPS)
        err = abs(ratio / expect_ratio - 1.0)
        if err > tol:
            continue
        score = (sl + sr) - 2.0 * err          # periodic on both sides, right ratio
        if best is None or score > best[0]:
            best = (score, s, fine, coarse, ratio, sl, sr,
                    "fine-left" if pl <= pr else "fine-right")
    if best is None:
        return None
    _, s, fine, coarse, ratio, sl, sr, order = best
    return dict(split_x=s, P_fine=fine, P_coarse=coarse, ratio=ratio,
                ratio_err=abs(ratio / expect_ratio - 1.0), expect=expect_ratio,
                score_left=sl, score_right=sr, order=order)
