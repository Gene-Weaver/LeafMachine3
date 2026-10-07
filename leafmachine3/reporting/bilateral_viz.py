"""Bilateral-symmetry QC panel -- Reporter-side rendering.

Three panels per leaf, which together separate the two things a low score can mean:

1. **Oriented mask**, left half tinted blue and right orange, the traced midvein solid green and the
   straight tip->base chord dashed. A midvein running down one SIDE of the silhouette is the case to
   look at, and the panel exists because it has three quite different causes that the numbers alone
   cannot separate: a **folded leaf** (the midrib really is at the edge -- normal, and common on
   herbarium sheets), a genuinely **asymmetric leaf** (oblique bases are normal in many taxa), or a
   **mis-traced midvein**. Only the first two are honest leaves; all three score low, which is
   correct for "pick an exemplar" but must not be read as "the mask is broken".
2. **Straightened** ``(s, u)`` view: the same halves with the midvein pulled to a vertical line, so
   the halves can be compared without the midvein's own curvature counting as asymmetry.
3. **Mirrored overlap**: left half against the reflected right half. The agreed area is muted grey;
   where they disagree keeps the half's own colour, so a blue fringe means the LEFT half overhangs
   and an orange one means the right does.

Never imports pyplot. The Reporter runs in a worker pool, and pyplot carries global figure state
that is not safe there; the object-oriented ``Figure`` + ``FigureCanvasAgg`` API has no such state.

Output is JPEG. The panel is a rendered figure, not data -- every number in it is already in the
``bilateral_symmetry`` table -- so it is sized for someone flicking through a gallery.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from leafmachine3.core.bilateral import build_axis, profiles, straighten

# the timing.html / guide house theme, so QC images match the rest of the LM3 reports
BG, PANEL, INK, LINE = "#101012", "#191a1d", "#e8e8ea", "#2a2b30"
LEFT_C, RIGHT_C, AXIS_C, BAD_C, CHORD_C = "#38bdf8", "#fb923c", "#4ade80", "#f87171", "#6f757f"
MIDVEIN_C = "#ffffff"      # the axis itself, over tinted halves -- white reads on blue AND orange
AGREE_C = "#4b5563"        # where the mirrored halves agree: muted, so the DISAGREEMENT stands out


def _rgb(hex_s: str) -> np.ndarray:
    h = hex_s.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0


def render_qc_panel(silhouette: np.ndarray, row: Any, *, dpi: int = 110) -> Optional[np.ndarray]:
    """Render the 3-up panel for one leaf and return it as an RGB uint8 array (or ``None``).

    ``row`` is a ``bilateral_symmetry`` row: it carries the midvein polyline and tip/base in ORIENTED
    coords, so the frame is rebuilt directly here with no re-derivation from landmarks and no
    dependence on the confidence filter still being set the way the stage had it.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    mask = np.asarray(silhouette, bool)
    if not mask.any():
        return None
    try:
        midvein = np.asarray(json.loads(row["midvein_json"] or "[]"), float)
    except Exception:  # noqa: BLE001 - a malformed row must not take the Reporter down
        return None
    if midvein.ndim != 2 or len(midvein) < 3:
        return None

    tip = np.array([row["tip_x"], row["tip_y"]], float)
    base = np.array([row["base_x"], row["base_y"]], float)
    frame = build_axis(mask, "midvein", midvein=midvein, tip=tip, base=base)
    if frame is None:
        return None
    prof = profiles(frame, n_bins=int(row["n_bins"] or 200))
    left_g, right_g = straighten(frame)

    fig = Figure(figsize=(10.5, 4.2), dpi=dpi, facecolor=BG)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for ax in axes:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(LINE)
        ax.tick_params(colors=INK, labelsize=7)

    # ---- 1: oriented mask, halves tinted, midvein solid + chord dashed --------------------
    h, w = mask.shape
    canvas = np.zeros((h, w, 3), float)
    side = np.zeros((h, w), np.int8)
    ij = np.rint(frame.xy).astype(int)
    np.clip(ij[:, 0], 0, w - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, h - 1, out=ij[:, 1])
    side[ij[:, 1], ij[:, 0]] = np.where(frame.u > 0, 1, np.where(frame.u < 0, -1, 0))
    canvas[side == 1] = _rgb(LEFT_C)
    canvas[side == -1] = _rgb(RIGHT_C)
    canvas[(side == 0) & mask] = _rgb(INK)
    axes[0].imshow(canvas, interpolation="nearest")
    axes[0].plot(frame.path[:, 0], frame.path[:, 1], color=MIDVEIN_C, lw=1.6, label="midvein")
    axes[0].plot([tip[0], base[0]], [tip[1], base[1]], color=CHORD_C, lw=1.0, ls="--", label="chord")
    axes[0].scatter([tip[0], base[0]], [tip[1], base[1]], s=18, c=[MIDVEIN_C], zorder=3)
    axes[0].set_title(f"oriented  ·  left={LEFT_C_NAME}  right={RIGHT_C_NAME}", color=INK, fontsize=8)
    axes[0].legend(loc="lower right", fontsize=6, facecolor=PANEL, edgecolor=LINE, labelcolor=INK)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # ---- 2: straightened (s, u) ------------------------------------------------------------
    strt = np.zeros((left_g.shape[0], left_g.shape[1] * 2, 3), float)
    strt[:, :left_g.shape[1]] = _rgb(LEFT_C) * left_g[:, ::-1, None]
    strt[:, left_g.shape[1]:] = _rgb(RIGHT_C) * right_g[:, :, None]
    # The (s, u) grid is a FIXED n_s x n_u lattice, so drawing it "equal" renders every leaf as the
    # same square no matter its real shape. One row spans length/n_s px of arclength and one column
    # spans umax/n_u px across, so this aspect restores the leaf's true proportions.
    n_s, n_u = left_g.shape
    umax = float(np.abs(frame.u).max()) if frame.u.size else 0.0
    px_row = frame.length / max(1, n_s)
    px_col = (umax / max(1, n_u)) if umax > 0 else px_row
    axes[1].imshow(strt, aspect=(px_row / px_col if px_col > 0 else 1.0), interpolation="nearest")
    axes[1].axvline(left_g.shape[1] - 0.5, color=MIDVEIN_C, lw=1.2)
    axes[1].set_title("straightened  ·  midvein vertical", color=INK, fontsize=8)
    axes[1].set_ylabel("tip (0)  →  base (1)", color=INK, fontsize=7)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # ---- 3: mirrored overlap ---------------------------------------------------------------
    # Same palette as the other two panels. Colouring the two halves' EXCLUSIVE regions separately
    # says WHICH side overhangs, which a single difference colour throws away; the agreed region is
    # deliberately muted so the eye goes to the disagreement.
    both = left_g & right_g
    ov = np.zeros((*left_g.shape, 3), float)
    ov[both] = _rgb(AGREE_C)
    ov[left_g & ~right_g] = _rgb(LEFT_C)
    ov[right_g & ~left_g] = _rgb(RIGHT_C)
    axes[2].imshow(ov, aspect=(px_row / px_col if px_col > 0 else 1.0), interpolation="nearest")
    dice = _num(row["dice"])
    si_a = _num(row["si_a"])
    score = _num(row["archetype_score"])
    axes[2].set_title(f"mirrored overlap  ·  dice {_fmt(dice)}", color=INK, fontsize=8)
    axes[2].set_xticks([]); axes[2].set_yticks([])

    gate = "PASS" if int(row["gates_pass"] or 0) else "VETOED"
    gate_c = AXIS_C if gate == "PASS" else BAD_C
    head = f"archetype {_fmt(score)}   ·   si_a {_fmt(si_a)}   ·   gates {gate}"
    fig.suptitle(head, color=gate_c, fontsize=10, y=0.985)
    # Reasons go on their own line, elided -- they run to hundreds of characters and a suptitle
    # silently draws straight off the canvas rather than wrapping.
    reasons = _reasons(row)
    if reasons:
        fig.text(0.5, 0.925, _elide(reasons, 150), color=INK, fontsize=7,
                 ha="center", va="top", alpha=0.75)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    canvas_agg = fig.canvas
    canvas_agg.draw()
    buf = np.asarray(canvas_agg.buffer_rgba())[:, :, :3].copy()
    fig.clf()
    return buf


LEFT_C_NAME, RIGHT_C_NAME = "blue", "orange"


def _num(v: Any) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def _fmt(x: float) -> str:
    return "-" if not np.isfinite(x) else f"{x:.3f}"


def _elide(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _reasons(row: Any, limit: int = 2) -> str:
    try:
        rs = json.loads(row["reasons_json"] or "[]")
    except Exception:  # noqa: BLE001
        return ""
    return "; ".join(str(r) for r in rs[:limit])
