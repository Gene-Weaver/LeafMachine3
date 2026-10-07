"""Matplotlib figure builders for the bilateral-symmetry report.

Every builder returns a ``data:image/png;base64,...`` string rather than writing a file, because the
report is a SINGLE self-contained HTML document -- it has to survive being emailed or dropped in a
shared folder with no sidecar image directory.

Palette, spacing and typography are lifted from ``leafmachine3/setup/timing.py`` so the two reports
read as one system: dark ``#191a1d`` panels, ``#e8e8ea`` ink, ``#2a2b30`` rules, monospace numerals.
Figures carry no title -- the surrounding HTML supplies every heading and caption.

ONE COLOR CONTRACT, held everywhere: the viewer's LEFT half of the leaf is ``acc2`` blue and the
RIGHT half is ``acc`` orange. Because every mask is tip-up oriented (see geometry.py), "left" is the
same physical side in every figure, so a reader can carry the color across panels without a legend.
Where a figure has no sides (histograms, scatter), blue is the primary series and orange the
comparison it is being judged against.

pyplot is never imported. It keeps a global figure registry and an interactive state machine, both
of which leak between calls and are unsafe once the driver renders figures from more than one
thread; ``Figure`` + ``FigureCanvasAgg`` is the same renderer with none of that.

Dependencies are deliberately limited to numpy, matplotlib and this package's geometry/axes: the
cohort builders take plain ``list[dict]`` rows and plain arrays, so metrics.py, shape.py and
quality.py can evolve their field sets without touching this module.
"""
from __future__ import annotations

import base64
import io
import math
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .axes import AxisFrame, Profiles, straighten

if TYPE_CHECKING:  # geometry pulls in cv2 via core.imaging; a plotting module should not need it
    from .geometry import OrientedLeaf

# --- timing.py's :root palette, verbatim ---------------------------------------------------- #
BG = "#101012"
PANEL = "#191a1d"
PANEL2 = "#1f2024"
INK = "#e8e8ea"
MUTE = "#9ca3af"
DIM = "#6f757f"
LINE = "#2a2b30"
ACC = "#fb923c"    # orange -- the viewer's RIGHT half
ACC2 = "#38bdf8"   # blue   -- the viewer's LEFT half
ACC3 = "#4ade80"   # green  -- the midvein axis / "good" flag
WARN = "#fbbf24"
BAD = "#f87171"    # red    -- disagreement between the halves

LEFT_C, RIGHT_C = ACC2, ACC
MONO = "DejaVu Sans Mono"
DPI = 110
# width/height cap for the per-leaf thumbnails: a degenerate mask (a 3 x 900 sliver) would otherwise
# ask for a figure meters wide, and 105 of these are embedded in one HTML page
MAX_THUMB_ASPECT = 4.0

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #
def to_data_uri(fig: Figure) -> str:
    """Render a figure to a base64 PNG data URI (the only way a figure leaves this module)."""
    FigureCanvasAgg(fig)  # attaching the Agg canvas is what makes savefig work without pyplot
    # the figure's OWN dpi wins: the per-leaf thumbnails deliberately ask for a cheaper one, and
    # hard-coding DPI here would silently discard that for all 105 of them
    dpi = float(fig.get_dpi())
    if not (np.isfinite(dpi) and dpi > 0):
        dpi = float(DPI)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.16,
                facecolor=fig.get_facecolor(), edgecolor="none")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _figure(w: float, h: float) -> Figure:
    fig = Figure(figsize=(w, h), dpi=DPI, facecolor=PANEL)
    fig.patch.set_alpha(1.0)
    return fig


def _style(ax, *, grid: Optional[str] = "both", labelsize: float = 8.5) -> None:
    """The house look for one axes: panel fill, hairline rules, muted monospace numerals."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(LINE)
        sp.set_linewidth(1.0)
    ax.tick_params(colors=MUTE, labelcolor=MUTE, labelsize=labelsize, length=3, width=0.8,
                   labelfontfamily=MONO)
    if grid:
        ax.grid(True, axis=grid, color=LINE, lw=0.7, alpha=0.85)
        ax.set_axisbelow(True)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_color(MUTE)
        lab.set_size(9.5)


def _image_axes(ax) -> None:
    """A framed image panel: keep the hairline border, drop the ticks."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(LINE)
        sp.set_linewidth(1.0)
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(ax, *, loc: str = "best", ncol: int = 1, fontsize: float = 8.2, **kw):
    leg = ax.legend(loc=loc, ncol=ncol, fontsize=fontsize, framealpha=0.95,
                    facecolor=PANEL2, edgecolor=LINE, labelcolor=INK,
                    borderpad=0.5, handlelength=1.6, **kw)
    leg.get_frame().set_linewidth(0.8)
    return leg


def _note(ax, text: str, *, x: float = 0.98, y: float = 0.97, ha: str = "right",
          va: str = "top", color: str = INK, size: float = 8.4) -> None:
    """A small monospace callout pinned in axes coordinates (numbers the reader needs in-frame)."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=color, fontsize=size,
            family=MONO, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.42", facecolor=PANEL2, edgecolor=LINE, linewidth=0.8))


def _empty(ax, msg: str = "no data") -> None:
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            color=DIM, fontsize=10, family=MONO)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(LINE)


def _placeholder(msg: str, w: float = 7.2, h: float = 2.0, *, dpi: float = DPI) -> str:
    """A figure-shaped 'nothing to plot' card, so a thin cohort degrades instead of raising.

    ``dpi`` exists so a placeholder standing in for a cheap thumbnail comes out the same pixel
    height as the thumbnails it sits beside.
    """
    fig = _figure(w, h)
    fig.set_dpi(dpi)
    ax = fig.add_subplot()
    ax.set_facecolor(PANEL)
    _empty(ax, msg)
    return to_data_uri(fig)


def _col(rows: Sequence[dict], key: str) -> np.ndarray:
    """One column of the cohort as float, non-numeric coerced to nan (never raises)."""
    out = np.full(len(rows), _NAN)
    for i, r in enumerate(rows):
        v = r.get(key) if isinstance(r, dict) else None
        try:
            # a flag is not a measurement -- np.bool_ has to be dropped alongside Python's bool,
            # otherwise the same column plots as nan or as 1.0/0.0 depending on who wrote the row
            if v is None or isinstance(v, (bool, np.bool_)):
                continue
            out[i] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _flag(rows: Sequence[dict], key: str) -> np.ndarray:
    """One boolean column; a nan flag means UNKNOWN and counts as False (``bool(nan)`` is True)."""
    out = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        v = r.get(key) if isinstance(r, dict) else None
        try:
            if v is None or v != v:      # v != v is the nan test that also works for numpy scalars
                continue
            out[i] = bool(v)
        except (TypeError, ValueError):
            continue
    return out


def _first_key(rows: Sequence[dict], candidates: Sequence[str]) -> Optional[str]:
    """First candidate that any row actually carries -- tolerates the driver's naming choice."""
    for k in candidates:
        if any(isinstance(r, dict) and r.get(k) is not None for r in rows):
            return k
    return None


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), so Spearman is computed without pulling in scipy."""
    order = np.argsort(a, kind="mergesort")
    r = np.empty(a.size, float)
    r[order] = np.arange(1.0, a.size + 1.0)
    sa = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sa[j + 1] == sa[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return _NAN
    xs, ys = x - x.mean(), y - y.mean()
    den = float(np.sqrt((xs * xs).sum() * (ys * ys).sum()))
    return float((xs * ys).sum() / den) if den > 0 else _NAN


def _diag_ok(col: np.ndarray) -> bool:
    """Can a column be correlated at all? Same 3-finite-values bar as :func:`_spearman`, plus the
    requirement that the values actually vary -- a constant column ranks every leaf identically."""
    v = col[np.isfinite(col)]
    return bool(v.size >= 3 and float(v.max() - v.min()) > 0)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return _NAN
    return _pearson(_rank(x[ok]), _rank(y[ok]))


def _fmt(v: float, nd: int = 3) -> str:
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def _pad(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return (0.0, 1.0)
    if hi <= lo:
        d = max(abs(hi), 1.0) * 0.05
        return (lo - d, hi + d)
    d = (hi - lo) * frac
    return (lo - d, hi + d)


def _num(v: Any, default: float, lo: float, hi: float) -> float:
    """A caller-supplied size/dpi clamped into a renderable range (nothing here may raise)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return float(np.clip(f, lo, hi)) if np.isfinite(f) else default


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b, _ = to_rgba(color)
    return (r, g, b, alpha)


def _axis_label(frame: Optional[AxisFrame]) -> str:
    """Name an axis from the frame itself -- callers fall back to the chord when a leaf has no
    midvein, so the slot a frame arrives in says nothing about what it is."""
    kind = getattr(frame, "kind", None)
    return f"{kind} axis" if isinstance(kind, str) and kind else "axis"


def _side_raster(frame: AxisFrame, *, px_per_cell: float = 2.0,
                 max_s: int = 256, max_u: int = 200) -> tuple[Optional[np.ndarray], float]:
    """Signed ``(s, u)`` occupancy grid: 0 empty, 1 left, 2 right.

    Built by :func:`axes.straighten`'s BACKWARD sampling -- each cell asks the mask what is at its
    own ``(s, |u|)`` -- rather than by scattering mask pixels forward into the grid. Forward scatter
    leaves unsampled cells that read as holes in the straightened blade as soon as a leaf has fewer
    pixels per half than the grid has cells (measured on a 404-px leaf: 28.5% of the blade's span
    empty, against 0.3% here), and it lets a left and a right pixel land in the SAME cell with the
    winner decided by array order (6317 such cells over the 105-leaf cohort). It is exactly the
    approach ``straighten`` documents as wrong, and this panel sits next to the folded view that
    already uses ``straighten``: a picture that contradicts the metric beside it is worse than a
    slower one.

    Cell size is still tied to the leaf's own pixel scale, so a big leaf gets a finer grid.
    """
    if frame is None or frame.u.size == 0:
        return None, 0.0
    umax = float(np.abs(frame.u).max())
    if umax <= 0:
        return None, 0.0
    n_s = int(np.clip(frame.length / px_per_cell, 40, max_s))
    n_h = max(12, int(np.clip(2.0 * umax / px_per_cell, 24, max_u)) // 2)   # cells per HALF
    gl, gr = straighten(frame, n_s=n_s, n_u=n_h)
    # columns run u = -umax .. +umax to match the imshow extent, so the right half is laid down in
    # reverse (|u| decreasing toward the axis) and the left half keeps its |u| order
    g = np.zeros((n_s, 2 * n_h), np.uint8)
    g[:, :n_h][gr[:, ::-1]] = 2
    g[:, n_h:][gl] = 1
    return g, umax


def _code_rgba(codes: np.ndarray, table: dict[int, tuple[float, float, float, float]]) -> np.ndarray:
    out = np.zeros(codes.shape + (4,), float)
    for code, rgba in table.items():
        out[codes == code] = rgba
    return out


# --------------------------------------------------------------------------- #
# per-leaf panel
# --------------------------------------------------------------------------- #
def fig_leaf_panel(leaf: "OrientedLeaf", frame_chord: Optional[AxisFrame],
                   frame_mid: Optional[AxisFrame], prof_mid: Optional[Profiles]) -> str:
    """Three views of ONE leaf: as measured, straightened, and folded onto itself.

    Read left to right, the panels answer "where is the axis", "what do the two halves look like
    once the curve is taken out", and "how much of the blade fails to mirror".
    """
    fig = _figure(12.8, 4.5)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.0), wspace=0.16,
                          left=0.02, right=0.98, top=0.97, bottom=0.10)
    ax0, ax1, ax2 = (fig.add_subplot(gs[0, i]) for i in range(3))

    frame_side = frame_mid if frame_mid is not None else frame_chord
    sil = np.asarray(getattr(leaf, "silhouette", np.zeros((1, 1), bool)), bool)

    # --- (1) the oriented leaf with both candidate axes over it --------------------------------
    _image_axes(ax0)
    if sil.any():
        h, w = sil.shape
        rgba = np.zeros((h, w, 4), float)
        rgba[sil] = _rgba(MUTE, 0.30)   # fallback wash if no frame is available to split sides
        if frame_side is not None and tuple(frame_side.shape) == (h, w):
            xy = frame_side.xy.astype(int)
            xs, ys = np.clip(xy[:, 0], 0, w - 1), np.clip(xy[:, 1], 0, h - 1)
            is_left = frame_side.u > 0
            rgba[ys[is_left], xs[is_left]] = _rgba(LEFT_C, 0.62)
            rgba[ys[~is_left], xs[~is_left]] = _rgba(RIGHT_C, 0.62)
        holes = np.asarray(getattr(leaf, "holes", np.zeros_like(sil)), bool)
        if holes.shape == sil.shape and holes.any():
            rgba[holes] = _rgba(BG, 0.92)
        ax0.imshow(rgba, interpolation="nearest", origin="upper")
        ax0.set_xlim(-0.5, w - 0.5)
        ax0.set_ylim(h - 0.5, -0.5)
    else:
        _empty(ax0, "empty mask")

    handles: list[Any] = []
    if frame_chord is not None:
        ax0.plot(frame_chord.path[:, 0], frame_chord.path[:, 1], ls=(0, (5, 3)), lw=1.5,
                 color=DIM, zorder=3)
        handles.append(Line2D([], [], color=DIM, ls=(0, (5, 3)), lw=1.5,
                              label=_axis_label(frame_chord)))
    # a leaf with no midvein arrives with the CHORD frame in both slots: draw and name it once,
    # never as a green "midvein axis" the leaf does not have
    if frame_mid is not None and frame_mid is not frame_chord:
        ax0.plot(frame_mid.path[:, 0], frame_mid.path[:, 1], lw=2.0, color=ACC3, zorder=4)
        handles.append(Line2D([], [], color=ACC3, lw=2.0, label=_axis_label(frame_mid)))
    try:
        tb = leaf.tip_base() if hasattr(leaf, "tip_base") else None
    except Exception:       # a figure never raises: losing the tip/base markers costs two glyphs
        tb = None
    if tb is not None:
        (tx, ty), (bx, by) = tb[0], tb[1]
        ax0.scatter([tx], [ty], s=52, marker="v", color=INK, edgecolor=BG, linewidth=0.9, zorder=5)
        ax0.scatter([bx], [by], s=52, marker="o", color=INK, edgecolor=BG, linewidth=0.9, zorder=5)
        ax0.annotate("tip", (tx, ty), textcoords="offset points", xytext=(8, -2),
                     color=MUTE, fontsize=8, family=MONO)
        ax0.annotate("base", (bx, by), textcoords="offset points", xytext=(8, 2),
                     color=MUTE, fontsize=8, family=MONO)
    handles += [Patch(facecolor=_rgba(LEFT_C, 0.62), edgecolor="none", label="left half"),
                Patch(facecolor=_rgba(RIGHT_C, 0.62), edgecolor="none", label="right half")]
    # the key sits BELOW the mask: a leaf fills its own bounding box, so no in-panel corner is
    # reliably free, and covering lamina here would hide the very asymmetry being shown
    ax0.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=2,
               fontsize=7.8, framealpha=0.95, facecolor=PANEL2, edgecolor=LINE, labelcolor=INK,
               borderpad=0.45, handlelength=1.6, columnspacing=1.4).get_frame().set_linewidth(0.8)

    # --- (2) the same leaf straightened: the midvein becomes a vertical line --------------------
    _style(ax1, grid="both")
    grid, umax = _side_raster(frame_side)
    if grid is None:
        _empty(ax1, "no axis frame")
    else:
        img = _code_rgba(grid, {1: _rgba(LEFT_C, 0.78), 2: _rgba(RIGHT_C, 0.78)})
        ax1.imshow(img, extent=(-umax, umax, 1.0, 0.0), aspect="auto",
                   interpolation="nearest", zorder=1)
        ax1.axvline(0.0, color=ACC3, lw=1.8, zorder=3)
        if prof_mid is not None:
            used = np.asarray(prof_mid.n_used, float) > 0
            s = np.asarray(prof_mid.s, float)
            wl = np.where(used, np.asarray(prof_mid.w_l, float), np.nan)
            wr = np.where(used, np.asarray(prof_mid.w_r, float), np.nan)
            ax1.plot(wl, s, color=LEFT_C, lw=1.1, alpha=0.95, zorder=4)
            ax1.plot(-wr, s, color=RIGHT_C, lw=1.1, alpha=0.95, zorder=4)
        # u increases to the viewer's LEFT, so the axis is reversed to keep blue on the left
        ax1.set_xlim(umax * 1.04, -umax * 1.04)
        ax1.set_ylim(1.0, 0.0)
        ax1.set_xlabel("signed offset u (px)", color=MUTE, fontsize=9)
        ax1.set_ylabel("s — tip (0) → base (1)", color=MUTE, fontsize=9)
        kind = frame_side.kind if frame_side is not None else "?"
        _note(ax1, f"{kind} frame\nmargin envelope overlaid", x=0.03, ha="left", size=7.8)

    # --- (3) fold the leaf on its axis: what fails to mirror ------------------------------------
    _style(ax2, grid="both")
    if frame_side is None:
        _empty(ax2, "no axis frame")
    else:
        umax2 = float(np.abs(frame_side.u).max()) if frame_side.u.size else 0.0
        n_s = int(np.clip(frame_side.length / 2.0, 40, 256))
        n_u = int(np.clip(umax2 / 2.0, 24, 160))
        gl, gr = straighten(frame_side, n_s=n_s, n_u=n_u)
        inter, sym = gl & gr, gl ^ gr
        codes = np.where(inter, 1, np.where(sym, 2, 0)).astype(np.uint8)
        img = _code_rgba(codes, {1: _rgba(MUTE, 0.34), 2: _rgba(BAD, 0.95)})
        ax2.imshow(img, extent=(0.0, max(umax2, 1e-6), 1.0, 0.0), aspect="auto",
                   interpolation="nearest", zorder=1)
        if prof_mid is not None:
            used = np.asarray(prof_mid.n_used, float) > 0
            s = np.asarray(prof_mid.s, float)
            ax2.plot(np.where(used, np.asarray(prof_mid.w_l, float), np.nan), s,
                     color=LEFT_C, lw=1.3, zorder=4, label="left margin")
            ax2.plot(np.where(used, np.asarray(prof_mid.w_r, float), np.nan), s,
                     color=RIGHT_C, lw=1.3, zorder=4, label="right (reflected)")
        inter_n, tot_n = float(inter.sum()), float(gl.sum() + gr.sum())
        dice = 2.0 * inter_n / tot_n if tot_n > 0 else _NAN
        ax2.set_xlim(0.0, max(umax2, 1e-6) * 1.04)
        ax2.set_ylim(1.0, 0.0)
        ax2.set_xlabel("|u| (px), right half reflected onto left", color=MUTE, fontsize=9)
        _note(ax2, f"Dice {_fmt(dice, 4)}\nSD   {_fmt(1.0 - dice, 4)}")
        hs = [Patch(facecolor=_rgba(MUTE, 0.34), edgecolor="none", label="both halves"),
              Patch(facecolor=_rgba(BAD, 0.95), edgecolor="none", label="symmetric difference")]
        if prof_mid is not None:
            hs += [Line2D([], [], color=LEFT_C, lw=1.3, label="left margin"),
                   Line2D([], [], color=RIGHT_C, lw=1.3, label="right margin")]
        ax2.legend(handles=hs, loc="lower right", fontsize=7.8, framealpha=0.95, facecolor=PANEL2,
                   edgecolor=LINE, labelcolor=INK, borderpad=0.45,
                   handlelength=1.6).get_frame().set_linewidth(0.8)

    return to_data_uri(fig)


# --------------------------------------------------------------------------- #
# per-leaf profiles
# --------------------------------------------------------------------------- #
def fig_width_profiles(prof: Profiles, *, title_l: str = "left", title_r: str = "right") -> str:
    """The two margin-envelope half-widths against arclength, with their gap shaded.

    Plotted tip -> base (the axis direction, not the growth direction) so it lines up row-for-row
    with every other s-indexed figure in the report.
    """
    fig = _figure(9.0, 3.5)
    ax = fig.add_subplot()
    _style(ax, grid="both")

    if prof is None or np.asarray(prof.s).size == 0:
        _empty(ax, "no profile")
        return to_data_uri(fig)

    s = np.asarray(prof.s, float)
    used = np.asarray(prof.n_used, float) > 0
    wl = np.where(used, np.asarray(prof.w_l, float), np.nan)
    wr = np.where(used, np.asarray(prof.w_r, float), np.nan)
    both = np.isfinite(wl) & np.isfinite(wr)

    if both.any():
        ax.fill_between(s, np.where(both, wl, 0.0), np.where(both, wr, 0.0), where=both,
                        color=_rgba(BAD, 0.20), linewidth=0, zorder=1,
                        label="|w$_L$ − w$_R$|")
    ax.plot(s, wl, color=LEFT_C, lw=1.8, zorder=3, label=f"w$_L$(s) — {title_l}")
    ax.plot(s, wr, color=RIGHT_C, lw=1.8, zorder=3, label=f"w$_R$(s) — {title_r}")

    ax.set_xlim(0.0, 1.0)
    top = np.nanmax(np.concatenate([wl, wr])) if both.any() else 1.0
    ax.set_ylim(0.0, (top if np.isfinite(top) and top > 0 else 1.0) * 1.12)
    ax.set_xlabel("s along the axis — tip (0) → base (1)", color=MUTE, fontsize=9.5)
    ax.set_ylabel("half-width, px", color=MUTE, fontsize=9.5)
    ax.text(0.0, -0.155, "TIP", transform=ax.transAxes, ha="left", va="top",
            color=DIM, fontsize=8, family=MONO)
    ax.text(1.0, -0.155, "BASE", transform=ax.transAxes, ha="right", va="top",
            color=DIM, fontsize=8, family=MONO)

    if both.any():
        mae = float(np.abs(wl[both] - wr[both]).mean())
        _note(ax, f"MAE {mae:7.2f} px\nn    {int(both.sum()):5d} bins")
    _legend(ax, loc="upper left", ncol=1)
    return to_data_uri(fig)


def fig_asymmetry_profile(s: np.ndarray, a_area: np.ndarray, a_width: np.ndarray,
                          cumulative: np.ndarray) -> str:
    """Where along the leaf the asymmetry sits, and whether it accumulates or cancels.

    The top panel is instantaneous imbalance (sign = which side is larger); the bottom is its running
    total, which is what separates a persistently lopsided leaf from one that merely trades sides.
    """
    s = np.asarray(s, float).ravel()

    def _align(a) -> np.ndarray:
        """Coerce a curve to the s grid; a caller that binned differently gets nan, not a crash."""
        if a is None:
            return np.full(s.shape, _NAN)
        v = np.asarray(a, float).ravel()
        # a different length means a different binning: keeping the first k samples would plot the
        # curve's TIP end against the whole axis and annotate the wrong C(1)
        return v if v.size == s.size else np.full(s.shape, _NAN)

    aa, aw, cc = _align(a_area), _align(a_width), _align(cumulative)

    fig = _figure(9.0, 5.4)
    gs = fig.add_gridspec(2, 1, height_ratios=(1.45, 1.0), hspace=0.10)
    top = fig.add_subplot(gs[0])
    bot = fig.add_subplot(gs[1], sharex=top)

    if s.size == 0 or not np.isfinite(aa).any():
        _empty(top, "no profile")
        _empty(bot, "")
        return to_data_uri(fig)

    # --- instantaneous imbalance ---------------------------------------------------------------
    _style(top, grid="both")
    top.axhline(0.0, color=LINE, lw=1.2, zorder=1)
    filled = np.where(np.isfinite(aa), aa, 0.0)
    top.fill_between(s, 0.0, filled, where=filled > 0, color=_rgba(LEFT_C, 0.34),
                     linewidth=0, interpolate=True, zorder=2)
    top.fill_between(s, 0.0, filled, where=filled < 0, color=_rgba(RIGHT_C, 0.34),
                     linewidth=0, interpolate=True, zorder=2)
    top.plot(s, aa, color=INK, lw=1.5, zorder=4, label="a$_A$(s)  area")
    if np.isfinite(aw).any():
        top.plot(s, aw, color=MUTE, lw=1.3, ls=(0, (4, 2.5)), zorder=3, label="a$_w$(s)  width")

    lim = float(np.nanmax(np.abs(np.concatenate([aa, aw])))) if np.isfinite(aa).any() else 1.0
    lim = lim if np.isfinite(lim) and lim > 0 else 1.0
    top.set_ylim(-lim * 1.28, lim * 1.28)
    top.set_ylabel("(L − R) / (L + R)", color=MUTE, fontsize=9.5)
    top.text(0.012, 0.955, "left-heavy ▲", transform=top.transAxes, ha="left", va="top",
             color=LEFT_C, fontsize=8.2, family=MONO)
    top.text(0.012, 0.045, "right-heavy ▼", transform=top.transAxes, ha="left", va="bottom",
             color=RIGHT_C, fontsize=8.2, family=MONO)
    top.tick_params(labelbottom=False)
    _legend(top, loc="upper right", ncol=2)

    # --- running total -------------------------------------------------------------------------
    _style(bot, grid="both")
    bot.axhline(0.0, color=LINE, lw=1.2, zorder=1)
    if np.isfinite(cc).any():
        cf = np.where(np.isfinite(cc), cc, 0.0)
        bot.fill_between(s, 0.0, cf, where=cf > 0, color=_rgba(LEFT_C, 0.28), linewidth=0,
                         interpolate=True, zorder=2)
        bot.fill_between(s, 0.0, cf, where=cf < 0, color=_rgba(RIGHT_C, 0.28), linewidth=0,
                         interpolate=True, zorder=2)
        bot.plot(s, cc, color=ACC3, lw=1.8, zorder=4)
        fin = np.flatnonzero(np.isfinite(cc))
        k = int(fin[-1])
        bot.scatter([s[k]], [cc[k]], s=42, color=ACC3, edgecolor=BG, linewidth=0.9, zorder=5)
        bot.annotate(f"C(1) = {cc[k]:+.4f}", (s[k], cc[k]), textcoords="offset points",
                     xytext=(-8, 12), ha="right", color=INK, fontsize=8.6, family=MONO,
                     bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL2, edgecolor=LINE,
                               linewidth=0.8))
        cl = float(np.nanmax(np.abs(cc)))
        cl = cl if np.isfinite(cl) and cl > 0 else 1.0
        bot.set_ylim(-cl * 1.55, cl * 1.55)
    else:
        _empty(bot, "no cumulative curve")

    bot.set_xlim(0.0, 1.0)
    bot.set_ylabel("C(s) cumulative", color=MUTE, fontsize=9.5)
    bot.set_xlabel("s along the axis — tip (0) → base (1)", color=MUTE, fontsize=9.5)
    return to_data_uri(fig)


# --------------------------------------------------------------------------- #
# cohort figures
# --------------------------------------------------------------------------- #
def fig_metric_distributions(rows: list[dict], keys: list[str]) -> str:
    """Small multiples: one histogram per metric over the whole cohort, median marked.

    The point is shape, not precision -- a metric whose cohort distribution is a spike carries no
    ranking information no matter how principled its definition.
    """
    keys = [k for k in (keys or [])]
    if not rows or not keys:
        return _placeholder("no cohort data")

    n = len(keys)
    ncols = 1 if n <= 1 else 2 if n <= 4 else 3 if n <= 6 else 4
    nrows = int(math.ceil(n / ncols))
    fig = _figure(3.35 * ncols, 2.45 * nrows)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.62, wspace=0.30)

    for i, key in enumerate(keys):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        _style(ax, grid="y", labelsize=7.8)
        ax.set_title(str(key), loc="left", color=INK, fontsize=9.2, family=MONO, pad=17)
        v = _col(rows, key)
        v = v[np.isfinite(v)]
        if v.size == 0:
            _empty(ax, "no values")
            continue
        nb = int(np.clip(round(2.0 * math.sqrt(v.size)), 6, 26))
        # a constant metric would give a zero-width range; one bar still shows it exists
        counts, _edges, _p = ax.hist(v, bins=(1 if float(v.max() - v.min()) <= 0 else nb),
                                     color=_rgba(ACC2, 0.85), edgecolor=PANEL, linewidth=0.6)
        med = float(np.median(v))
        ax.axvline(med, color=ACC, lw=1.5, ls=(0, (4, 2.2)), zorder=5)
        ax.set_ylim(0.0, float(np.max(counts)) * 1.12 if np.size(counts) else 1.0)
        ax.set_ylabel("leaves", color=MUTE, fontsize=8.4)
        # median goes ABOVE the axes: inside the panel it collides with whichever bar is tallest
        ax.text(0.0, 1.015, f"med {med:.4g}   n {v.size}", transform=ax.transAxes,
                ha="left", va="bottom", color=ACC, fontsize=7.8, family=MONO)

    return to_data_uri(fig)


def fig_chord_vs_midvein(rows: list[dict], key_chord: str, key_mid: str, label: str) -> str:
    """The experiment's headline: one metric measured both ways, per leaf.

    Distance from the diagonal is the price of the straight-chord assumption that silhouette-only
    methods (Shi 2018, Wang 2018) are forced to make. Points above it are leaves those methods call
    more asymmetric than the traced midvein says they are.

    Points are colored by POSITION relative to the drawn diagonal (``y > x``), which is the rule the
    report's caption describes. For a non-negative metric (SI_A, Dice) that is also "the chord
    reports more asymmetry"; for a SIGNED metric (A*) it is not -- a right-heavy leaf the chord
    exaggerates has a larger magnitude but sits BELOW the line -- so the legend names the position
    and leaves the verdict to the |chord| - |midvein| readout in the corner.
    """
    if not rows:
        return _placeholder("no cohort data")
    x = _col(rows, key_mid)      # midvein on x, chord on y -- matches the report's caption
    y = _col(rows, key_chord)
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) == 0:
        return _placeholder(f"no paired values for {label}")
    x, y = x[ok], y[ok]

    fig = _figure(6.6, 6.0)
    ax = fig.add_subplot()
    _style(ax, grid="both")

    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    lo, hi = _pad(lo, hi, 0.07)
    ax.plot([lo, hi], [lo, hi], color=DIM, ls=(0, (5, 3)), lw=1.2, zorder=2, label="y = x")

    above = y > x   # the chord reads HIGHER than the midvein: literally above the line just drawn
    ax.scatter(x[above], y[above], s=34, color=_rgba(ACC, 0.85), edgecolor=BG, linewidth=0.5,
               zorder=4, label=f"chord reads higher (n={int(above.sum())})")
    ax.scatter(x[~above], y[~above], s=34, color=_rgba(ACC2, 0.85), edgecolor=BG, linewidth=0.5,
               zorder=4, label=f"chord reads lower  (n={int((~above).sum())})")

    d = np.abs(y) - np.abs(x)
    med = float(np.median(d))
    base = float(np.median(np.abs(y)))
    pct = 100.0 * med / base if base > 0 else _NAN
    _note(ax, f"median |chord| − |midvein|\n  {med:+.4f}"
              + (f"  ({pct:+.1f}% of |chord|)" if np.isfinite(pct) else ""),
          x=0.03, y=0.97, ha="left")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{label} — midvein axis", color=MUTE, fontsize=9.5)
    ax.set_ylabel(f"{label} — chord axis", color=MUTE, fontsize=9.5)
    _legend(ax, loc="lower right")
    return to_data_uri(fig)


def fig_taylor(rows: list[dict], mean_key: str, var_key: str) -> str:
    """Taylor's power law for leaf asymmetry, Wang et al. (2018) Forests 9:500.

    Per leaf, the mean and variance of the strip-wise absolute left/right difference. If
    ``V = alpha * M^beta`` holds, the cloud is a straight line in log-log with slope ``beta`` -- the
    published result is ``beta`` close to 2, i.e. asymmetry variance scales with the square of its
    magnitude, so a single leaf's mean already predicts its spread.
    """
    if not rows:
        return _placeholder("no cohort data")
    m = _col(rows, mean_key)
    v = _col(rows, var_key)
    ok = np.isfinite(m) & np.isfinite(v) & (m > 0) & (v > 0)   # log-log needs strictly positive
    fig = _figure(6.8, 5.4)
    ax = fig.add_subplot()
    _style(ax, grid="both")
    if int(ok.sum()) < 3:
        _empty(ax, f"need 3+ positive pairs (have {int(ok.sum())})")
        return to_data_uri(fig)

    m, v = m[ok], v[ok]
    lx, ly = np.log10(m), np.log10(v)
    ax.scatter(m, v, s=36, color=_rgba(ACC2, 0.85), edgecolor=BG, linewidth=0.5, zorder=4,
               label=f"leaves (n={m.size})")

    beta = intercept = r2 = _NAN
    if float(lx.max() - lx.min()) > 0:
        beta, intercept = (float(c) for c in np.polyfit(lx, ly, 1))
        pred = beta * lx + intercept
        ss_res = float(((ly - pred) ** 2).sum())
        ss_tot = float(((ly - ly.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else _NAN
        gx = np.logspace(lx.min(), lx.max(), 64)
        ax.plot(gx, 10.0 ** intercept * gx ** beta, color=ACC, lw=1.8, zorder=5,
                label="OLS fit in log-log")

    ax.set_xscale("log")
    ax.set_yscale("log")
    _note(ax, f"log V = β·log M + log α\nβ     {_fmt(beta, 4)}\n"
              f"log α {_fmt(intercept, 4)}\nR²    {_fmt(r2, 4)}",
          x=0.03, y=0.97, ha="left")
    ax.set_xlabel(f"M$_D$  mean strip |L − R|   [{mean_key}]", color=MUTE, fontsize=9.5)
    ax.set_ylabel(f"V$_D$  variance of strip |L − R|   [{var_key}]", color=MUTE, fontsize=9.5)
    _legend(ax, loc="lower right")
    return to_data_uri(fig)


def fig_score_vs_symmetry(rows: list[dict], *, key_score: str = "archetype_score",
                          key_flag: str = "is_archetypal",
                          key_sym: Optional[str] = None) -> str:
    """Does symmetry alone find the clean masks?

    If the archetypal leaves formed a tight low-SI cluster the score could be replaced by one
    symmetry number; overlap between the two groups means symmetry is necessary but not sufficient.
    """
    if not rows:
        return _placeholder("no cohort data")
    key_sym = key_sym or _first_key(rows, ("midvein_SI_A", "SI_A_midvein", "mid_SI_A",
                                           "SI_A_mid", "SI_A"))
    if key_sym is None:
        return _placeholder("no SI_A column in the cohort rows")

    x = _col(rows, key_sym)
    y = _col(rows, key_score)
    flag = _flag(rows, key_flag)
    ok = np.isfinite(x) & np.isfinite(y)

    fig = _figure(6.8, 5.4)
    ax = fig.add_subplot()
    _style(ax, grid="both")
    if not ok.any():
        _empty(ax, "no paired score / symmetry values")
        return to_data_uri(fig)

    x, y, flag = x[ok], y[ok], flag[ok]
    ax.scatter(x[~flag], y[~flag], s=36, color=_rgba(MUTE, 0.65), edgecolor=BG, linewidth=0.5,
               zorder=3, label=f"not archetypal (n={int((~flag).sum())})")
    ax.scatter(x[flag], y[flag], s=42, color=_rgba(ACC3, 0.92), edgecolor=BG, linewidth=0.5,
               zorder=4, label=f"archetypal (n={int(flag.sum())})")

    rho = _spearman(x, y)
    sep = ""
    if flag.any() and (~flag).any():
        sep = f"\nmedian SI_A  {np.median(x[flag]):.4f} / {np.median(x[~flag]):.4f}"
    _note(ax, f"Spearman ρ  {_fmt(rho, 3)}{sep}", x=0.97, y=0.03, ha="right", va="bottom")

    ax.set_xlim(*_pad(float(x.min()), float(x.max())))
    ax.set_ylim(*_pad(float(y.min()), float(y.max())))
    ax.set_xlabel(f"SI$_A$ on the midvein axis   [{key_sym}]", color=MUTE, fontsize=9.5)
    ax.set_ylabel(f"archetype score   [{key_score}]", color=MUTE, fontsize=9.5)
    _legend(ax, loc="upper right")
    return to_data_uri(fig)


def fig_correlation_matrix(rows: list[dict], keys: list[str]) -> str:
    """Spearman correlation between every pair of metrics -- the redundancy map.

    Rank correlation rather than Pearson because several of these metrics are bounded ratios with
    long tails, and the question here is only whether two metrics rank the cohort the same way. A
    block of |rho| near 1 means those metrics are one measurement wearing several names.
    """
    keys = [k for k in (keys or [])]
    if not rows or len(keys) < 2:
        return _placeholder("need 2+ metrics to correlate")

    cols = [_col(rows, k) for k in keys]
    n = len(keys)
    m = np.full((n, n), _NAN)
    for i in range(n):
        # a column with no finite values -- or a constant one -- has no rank order to correlate, so
        # its diagonal stays blank rather than showing a confident 1.00 over a row of em dashes
        m[i, i] = 1.0 if _diag_ok(cols[i]) else _NAN
        for j in range(i + 1, n):
            r = _spearman(cols[i], cols[j])
            m[i, j] = m[j, i] = r

    side = float(np.clip(0.46 * n + 2.0, 4.2, 14.0))
    fig = _figure(side + 1.1, side)
    ax = fig.add_subplot()
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(LINE)

    # diverging through the panel color: a metric pair with no rank agreement fades into the page
    cmap = LinearSegmentedColormap.from_list("lm3_div", [ACC, PANEL2, ACC2])
    cmap.set_bad(PANEL)
    im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, vmin=-1.0, vmax=1.0,
                   interpolation="nearest")

    fs = float(np.clip(11.0 - 0.32 * n, 5.0, 9.0))
    for i in range(n):
        for j in range(n):
            r = m[i, j]
            txt = "—" if not np.isfinite(r) else f"{r:+.2f}".replace("+", " ")
            col = BG if (np.isfinite(r) and abs(r) > 0.55) else INK
            ax.text(j, i, txt, ha="center", va="center", color=col, fontsize=fs, family=MONO)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=fs + 0.6, family=MONO, color=MUTE)
    ax.set_yticklabels(keys, fontsize=fs + 0.6, family=MONO, color=MUTE)
    ax.set_xticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1.0), minor=True)
    ax.grid(which="minor", color=PANEL, lw=1.2)
    ax.tick_params(which="both", length=0, colors=MUTE)

    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.03, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_edgecolor(LINE)
    cb.ax.tick_params(colors=MUTE, labelsize=8, labelfontfamily=MONO)
    cb.set_label("Spearman ρ", color=MUTE, fontsize=9)
    return to_data_uri(fig)


def fig_leaf_split(leaf: "OrientedLeaf", frame_chord: Optional[AxisFrame],
                   frame_mid: Optional[AxisFrame], *, height_in: float = 2.6,
                   dpi_override: int = 78) -> str:
    """JUST the left panel of :func:`fig_leaf_panel`, small enough to embed for EVERY leaf.

    The interactive scatter needs one thumbnail per point, so this is deliberately cheap: no
    legend, no annotation, no straightened or folded view -- only the oriented mask with its two
    halves tinted and both candidate axes drawn. That is the view that tells a bad midvein trace
    (green axis hugging a margin) apart from a genuinely lopsided leaf at a glance.
    """
    # a figure never raises: a zero or non-numeric height makes Figure reject the aspect outright
    height_in = _num(height_in, 2.6, 0.4, 12.0)
    dpi = _num(dpi_override, 78.0, 20.0, 600.0)

    sil = np.asarray(getattr(leaf, "silhouette", np.zeros((1, 1), bool)), bool)
    if not sil.any():
        return _placeholder("empty mask", 1.4, height_in, dpi=dpi)
    h, w = sil.shape
    fig_w = max(0.7, min(height_in * MAX_THUMB_ASPECT, height_in * w / max(1, h)))
    fig = Figure(figsize=(fig_w, height_in), dpi=dpi)
    fig.patch.set_facecolor(PANEL)
    ax = fig.add_axes((0, 0, 1, 1))
    _image_axes(ax)
    ax.set_facecolor(PANEL)

    frame_side = frame_mid if frame_mid is not None else frame_chord
    rgba = np.zeros((h, w, 4), float)
    rgba[sil] = _rgba(MUTE, 0.30)
    if frame_side is not None and tuple(frame_side.shape) == (h, w):
        xy = frame_side.xy.astype(int)
        xs, ys = np.clip(xy[:, 0], 0, w - 1), np.clip(xy[:, 1], 0, h - 1)
        is_left = frame_side.u > 0
        rgba[ys[is_left], xs[is_left]] = _rgba(LEFT_C, 0.62)
        rgba[ys[~is_left], xs[~is_left]] = _rgba(RIGHT_C, 0.62)
    holes = np.asarray(getattr(leaf, "holes", np.zeros_like(sil)), bool)
    if holes.shape == sil.shape and holes.any():
        rgba[holes] = _rgba(BG, 0.92)
    ax.imshow(rgba, interpolation="nearest", origin="upper")
    if frame_chord is not None:
        ax.plot(frame_chord.path[:, 0], frame_chord.path[:, 1], ls=(0, (4, 3)), lw=1.1,
                color=DIM, zorder=3)
    # green means midvein everywhere in this report; when the caller has only a chord it can arrive
    # in both slots, and drawing it green here would invent a midvein trace the leaf has not got
    if frame_mid is not None and frame_mid is not frame_chord:
        ax.plot(frame_mid.path[:, 0], frame_mid.path[:, 1], lw=1.6, color=ACC3, zorder=4)

    # letterbox rather than stretch: the figure aspect is capped above, so a freakishly wide mask is
    # centered in the panel at its true proportions instead of emitting a meters-wide PNG
    aspect = fig_w / height_in
    view_w, view_h = float(w), float(h)
    if view_w / view_h > aspect:
        view_h = view_w / aspect
    else:
        view_w = view_h * aspect
    cx, cy = 0.5 * w - 0.5, 0.5 * h - 0.5
    ax.set_xlim(cx - 0.5 * view_w, cx + 0.5 * view_w)
    ax.set_ylim(cy + 0.5 * view_h, cy - 0.5 * view_h)
    return to_data_uri(fig)
