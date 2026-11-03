"""Which lamina pixels count: a white-peak check that finds mounting tape and other near-white matter.

Tape is often glued across a leaf, inside its mask, and it moves every color index. On a whiteness
histogram of one leaf it shows up as its own peak on the bright side, with a trough between it and
the leaf. Whiteness is min(R, G, B): high only when all three channels are high, so a bright yellow
or green leaf does not read as white (its blue channel is low) while tape does.

    white peak   the lowest significant peak at WHITE_MIN or above
    leaf peak    the tallest peak at least MIN_PEAK_SEPARATION below it. NOT the tallest peak overall:
                 a narrow tape spike is often taller than the broad hump of the leaf.
    trough       the middle of the flat bottom of the valley between the two
    excluded     pixels whiter than the trough

White may outnumber leaf, because tape over a curved or broken leaf can cover most of what is left.
That is also what a pale, woolly leaf with one darker fold looks like, so above half of the lamina the
evidence has to be stronger: the excluded side must be near-neutral, like tape, and the kept side must
be colored. Above MAX_EXCLUDED_FRAC nothing is excluded.

These values were set by reviewing contact sheets of 45,180 leaves from three runs. There are no
hand-labeled tape masks behind them. The known limit is tape on a pale leaf, whose peak can merge with
the leaf's own, which is then left in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_LAMINA_PX = 400               # smaller leaves have too noisy a histogram to trust
SMOOTH_SIGMA = 2.5                # histogram smoothing, in gray levels
WHITE_MIN = 170                   # the white peak must sit at or above this whiteness
MIN_PEAK_SEPARATION = 30          # gray levels between the leaf peak and the white peak
VALLEY_RATIO = 0.7                # the trough must be below this fraction of the white peak's height
MAX_PEAK_CHROMA = 60              # median max(RGB)-min(RGB) around the white peak; aged tape is creamy
MIN_EXCLUDED_FRAC = 0.003         # a smaller white side is not worth removing
MAX_EXCLUDED_FRAC = 0.8           # more than this has only been seen on pale leaves, never on tape
MAJORITY_FRAC = 0.5               # above this the excluded side outnumbers the leaf, so the bar rises:
MAJORITY_MAX_WHITE_CHROMA = 24    # ...the excluded side must be near-neutral, like tape
MAJORITY_MIN_LEAF_CHROMA = 20     # ...and the kept side must be colored: brown, red or green
FLAT_BOTTOM_TOL = 0.1             # valley bottom = within this fraction of the white peak's height of the minimum
MIN_PEAK_REL_HEIGHT = 0.25        # a white peak this small beside the tallest white peak is noise
MIN_PEAK_HEIGHT = 1e-4            # share of pixels per gray level; below this a bump is not a peak
PEAK_CHROMA_WINDOW = 8            # gray levels either side of the white peak used to read its color
_HIST_PAD = 12                    # zero padding, so a tape spike clipped at 255 is still a peak

STATUS_OFF = "off"
STATUS_CUT = "cut"


@dataclass(frozen=True)
class WhitePeakResult:
    """The decision for one leaf. ``cut`` is the trough's whiteness, or None when nothing is excluded."""
    status: str
    cut: Optional[int] = None
    leaf_peak: Optional[int] = None
    white_peak: Optional[int] = None


def whiteness(px: np.ndarray) -> np.ndarray:
    """min(R, G, B) of an N x 3 uint8 pixel array."""
    return px.min(axis=1).astype(np.uint8)


def chroma(px: np.ndarray) -> np.ndarray:
    """max(RGB) - min(RGB) of an N x 3 uint8 pixel array: 0 is neutral gray or white."""
    return px.max(axis=1).astype(float) - px.min(axis=1)


def find_white_cut(px: np.ndarray) -> WhitePeakResult:
    """Decide whether this leaf's N x 3 uint8 lamina pixels hold a white peak to exclude.

    ``status`` is ``cut`` or the reason nothing was excluded, which ends up in green_index.csv.
    """
    # Here, not at module level: importing scipy.signal would add half a second to every pipeline import.
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    n = len(px)
    if n < MIN_LAMINA_PX:
        return WhitePeakResult("too_few_px")
    w = whiteness(px)
    hist = np.bincount(w, minlength=256).astype(float) / n
    padded = np.concatenate([np.zeros(_HIST_PAD), hist, np.zeros(_HIST_PAD)])
    s = gaussian_filter1d(padded, SMOOTH_SIGMA, mode="constant")
    peaks, _ = find_peaks(s, height=MIN_PEAK_HEIGHT)
    peaks = peaks - _HIST_PAD
    peaks = peaks[(peaks >= 0) & (peaks < 256)]
    s = s[_HIST_PAD:_HIST_PAD + 256]

    white = [int(x) for x in peaks if x >= WHITE_MIN]
    if not white:
        return WhitePeakResult("no_white_peak")
    tallest = max(s[x] for x in white)
    wp = next(x for x in white if s[x] >= MIN_PEAK_REL_HEIGHT * tallest)
    below = [int(x) for x in peaks if x <= wp - MIN_PEAK_SEPARATION]
    if not below:
        return WhitePeakResult("no_leaf_peak", white_peak=wp)      # the leaf itself is pale
    leaf = max(below, key=lambda x: s[x])

    def result(status: str, cut: Optional[int] = None) -> WhitePeakResult:
        return WhitePeakResult(status, cut, leaf, wp)

    trough, valley = _trough(s, leaf, wp)
    if valley >= VALLEY_RATIO:
        return result("shallow_valley")
    near = np.abs(w.astype(int) - wp) <= PEAK_CHROMA_WINDOW
    if near.any() and float(np.median(chroma(px[near]))) > MAX_PEAK_CHROMA:
        return result("colored_peak")
    over = w > trough
    frac = float(over.mean())
    if frac < MIN_EXCLUDED_FRAC:
        return result("too_small")
    if frac > MAX_EXCLUDED_FRAC:
        return result("too_large")
    if frac > MAJORITY_FRAC:
        c = chroma(px)
        if (float(np.median(c[over])) > MAJORITY_MAX_WHITE_CHROMA
                or float(np.median(c[~over])) < MAJORITY_MIN_LEAF_CHROMA):
            return result("pale_leaf_not_tape")
    return result(STATUS_CUT, trough)


def _trough(s: np.ndarray, leaf: int, white: int) -> tuple[int, float]:
    """The middle of the valley's flat bottom between two peaks, and its height relative to the white peak.

    The middle, because the valley between a dark leaf and bright tape is often long and empty, and the
    first minimum would sit right against the leaf.
    """
    seg = s[leaf:white + 1]
    low = float(seg.min())
    i = int(np.argmin(seg))
    flat = low + FLAT_BOTTOM_TOL * (s[white] - low)
    lo = hi = i
    while lo > 0 and seg[lo - 1] <= flat:
        lo -= 1
    while hi < len(seg) - 1 and seg[hi + 1] <= flat:
        hi += 1
    return leaf + (lo + hi) // 2, low / float(s[white])
