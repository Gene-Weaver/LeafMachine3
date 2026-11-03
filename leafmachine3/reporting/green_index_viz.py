"""Green Index summary image: one per oriented leaf, written to ``Overlay/Overlay_Green_Index``.

The oriented lamina cutout sits on the left with every excluded pixel painted red at 60%. To its
right, one tile per index: a histogram of that index over the used pixels, with each bar colored by
the actual pixels in its bin, a line at the leaf's value, the value printed, and three color tiles
for this index's lowest 10% of pixels, the pixels nearest the leaf's value, and its highest 10%. The
last tile is the white-peak check itself, so a cut can be audited from the picture.

Every histogram keeps a fixed x range per index, so two leaves can be compared across images.

Never imports pyplot. The stage renders in a process pool and pyplot carries global figure state;
the object-oriented ``Figure`` + ``FigureCanvasAgg`` API has none. The image is a rendered figure,
not data: every number in it is in the ``leaf_green_index`` table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from leafmachine3.core.green_index._common import GreenIndexMethod
from leafmachine3.core.green_index.pixel_exclusion import STATUS_OFF, WhitePeakResult, whiteness

# the timing.html / guide house theme, so QC images match the rest of the LM3 reports
BG, PANEL, INK, LINE = "#101012", "#191a1d", "#e8e8ea", "#2a2b30"
MUTED, RED_LINE = "#a0a4ab", "#ff3b30"

EXCLUDED_RGB = np.array([255, 0, 0], dtype=float)
EXCLUDED_ALPHA = 0.6
_INDEX_BINS = 40
_WHITENESS_BINS = 51


@dataclass(frozen=True)
class IndexTile:
    """One index on one leaf, ready to draw."""
    method: GreenIndexMethod
    values: np.ndarray            # per-pixel values over the used pixels, aligned with the panel's pixels
    final: Optional[float]        # the leaf's value; None when no pixel gave a finite value
    note: str = ""                # replaces the method's direction text when set


def paint_excluded(rgb: np.ndarray, excluded: np.ndarray) -> np.ndarray:
    """A copy of ``rgb`` with the ``excluded`` pixels painted red at 60%."""
    out = rgb.copy()
    if excluded.any():
        out[excluded] = ((1 - EXCLUDED_ALPHA) * out[excluded] + EXCLUDED_ALPHA * EXCLUDED_RGB).astype(np.uint8)
    return out


def render_summary_panel(
    rgb: np.ndarray,
    lamina: np.ndarray,
    used: np.ndarray,
    tiles: Sequence[IndexTile],
    white_peak: WhitePeakResult,
    *,
    title: str,
    dpi: int = 110,
) -> np.ndarray:
    """Draw one leaf's summary image and return it as an RGB uint8 array.

    ``rgb`` is the cutout, ``lamina`` and ``used`` boolean masks of its size. Each tile's ``values``
    run over ``rgb[used]``, in array order.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    n_lamina, n_used = int(lamina.sum()), int(used.sum())
    fig = Figure(figsize=(14, 6.6), dpi=dpi, facecolor=BG)
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(4, 3, width_ratios=[1.05, 1, 1], left=0.015, right=0.99, top=0.93, bottom=0.05,
                            hspace=0.42, wspace=0.05)
    fig.suptitle(title, color=INK, fontsize=8, x=0.015, ha="left", y=0.985)

    leaf = fig.add_subplot(grid[:, 0])
    leaf.imshow(paint_excluded(rgb, lamina & ~used))
    leaf.set_axis_off()
    leaf.set_title(_leaf_caption(n_lamina, n_used), color=INK, fontsize=7, loc="left")

    pixels = rgb[used].astype(float)
    slots = [grid[row, col] for row in range(4) for col in (1, 2)]
    for tile, slot in zip(tiles, slots):
        _draw_index_tile(fig.add_subplot(slot), tile, pixels)
    _draw_white_peak_tile(fig.add_subplot(slots[len(tiles)]), rgb[lamina], white_peak, n_lamina, n_used)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def _leaf_caption(n_lamina: int, n_used: int) -> str:
    share = 100.0 * n_used / max(1, n_lamina)
    if n_used == n_lamina:
        return f"{n_lamina:,} lamina pixels, all used"
    return f"{n_lamina:,} lamina pixels; {n_used:,} used ({share:.1f}%); red = excluded"


def _style(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(LINE)
    ax.tick_params(colors=INK, labelsize=6, length=2)
    ax.set_yticks([])


def _colored_bars(ax, values: np.ndarray, colors: np.ndarray, lo: float, hi: float, bins: int) -> float:
    """Histogram of ``values`` with each bar the mean color of its pixels; returns the tallest bar.

    Values outside the axis land in the end bins, because the shape of the leaf's distribution matters
    more here than an outlier's position, and the printed value is never clipped.
    """
    from matplotlib.collections import PolyCollection

    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(values, edges) - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins).astype(float)
    shown = np.nonzero(counts)[0]
    frac = counts / max(1.0, counts.sum())
    means = np.stack([np.bincount(idx, weights=colors[:, k], minlength=bins) for k in range(3)], axis=1)
    bar_colors = means[shown] / counts[shown, None] / 255.0
    polygons = [[(edges[i], 0), (edges[i], frac[i]), (edges[i + 1], frac[i]), (edges[i + 1], 0)] for i in shown]
    ax.add_collection(PolyCollection(polygons, facecolors=bar_colors, edgecolors="none"))
    ax.set_xlim(lo, hi)
    tallest = float(frac.max()) if len(frac) else 0.0
    ax.set_ylim(0, max(tallest * 1.75, 1e-3))
    return tallest


def _draw_index_tile(ax, tile: IndexTile, pixels: np.ndarray) -> None:
    m = tile.method
    _style(ax)
    ax.set_title(m.name, color=INK, fontsize=7, loc="left", pad=3)
    ok = np.isfinite(tile.values)
    values, colors = tile.values[ok], pixels[ok]
    lo, hi = m.axis
    if m.window is not None:
        ax.axvspan(m.window[0], min(m.window[1], hi), color="#ffffff", alpha=0.06, lw=0)
    if len(values):
        _colored_bars(ax, values, colors, lo, hi, _INDEX_BINS)
    else:
        ax.set_xlim(lo, hi)
    value = "n/a" if tile.final is None else f"{tile.final:.3f}"
    ax.text(0.98, 0.93, f"{m.short} = {value}", transform=ax.transAxes, ha="right", va="top", color=INK,
            fontsize=10, fontweight="bold")
    ax.text(0.98, 0.76, tile.note or m.direction, transform=ax.transAxes, ha="right", va="top", color=MUTED,
            fontsize=6)
    if tile.final is None:
        return
    ax.axvline(min(max(tile.final, lo), hi), color="#ffffff", lw=1.2)
    _draw_swatches(ax, _swatches(values, colors, tile.final, m.target_rgb))


def _swatches(values: np.ndarray, colors: np.ndarray, final: float, target) -> list[tuple[str, np.ndarray]]:
    """This index's lowest 10% of pixels, those nearest the leaf's value, and its highest 10%, as colors."""
    n = len(values)
    k = max(1, n // 10)
    order = np.argpartition(values, kth=(k - 1, max(k - 1, n - k)))
    nearest = np.argpartition(np.abs(values - final), kth=k - 1)[:k]
    out = [("low 10%", colors[order[:k]].mean(axis=0) / 255.0),
           ("at value", colors[nearest].mean(axis=0) / 255.0),
           ("high 10%", colors[order[-k:]].mean(axis=0) / 255.0)]
    if target is not None:
        out.append(("target", np.asarray(target, dtype=float) / 255.0))
    return out


def _draw_swatches(ax, swatches: list[tuple[str, np.ndarray]]) -> None:
    from matplotlib.patches import Rectangle

    for i, (label, color) in enumerate(swatches):
        x = 0.015 + i * 0.1
        ax.add_patch(Rectangle((x, 0.72), 0.09, 0.2, transform=ax.transAxes, facecolor=color, edgecolor=INK, lw=0.4))
        ax.text(x + 0.045, 0.70, label, transform=ax.transAxes, ha="center", va="top", color=INK, fontsize=5)


def _draw_white_peak_tile(ax, lamina_px: np.ndarray, result: WhitePeakResult, n_lamina: int, n_used: int) -> None:
    _style(ax)
    ax.set_title("White-peak check: whiteness = min(R, G, B)", color=INK, fontsize=7, loc="left", pad=3)
    _colored_bars(ax, whiteness(lamina_px).astype(float), lamina_px.astype(float), 0, 255, _WHITENESS_BINS)
    if result.cut is not None:
        ax.axvspan(result.cut, 255, color=RED_LINE, alpha=0.18, lw=0)
        ax.axvline(result.cut, color=RED_LINE, lw=1.2)
    share = 100.0 * (n_lamina - n_used) / max(1, n_lamina)
    text = "check off" if result.status == STATUS_OFF else f"{result.status}: -{share:.1f}%"
    ax.text(0.98, 0.93, text, transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=6.5)
