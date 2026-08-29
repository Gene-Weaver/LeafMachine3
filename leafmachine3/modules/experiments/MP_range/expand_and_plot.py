"""MP_range step 2 -- expand each labeled sheet across the MP range and plot MP vs CF.

Takes the 20 hand-labeled originals from :mod:`label_server` and renders each one across a common
megapixel ladder: down through 4 geometric steps to ~1 MP, and up to every other sheet's native
resolution. The result is full artificial coverage -- every parent present at every real MP -- from
20 real measurements.

WHY THE DERIVED LABELS ARE EXACT, NOT ESTIMATES
    A uniform resize scales every distance in the image by the same factor, so it scales the
    conversion factor by that factor too. If a sheet measures 116.60 px/cm at 2946x5000 and is
    resized to 1473x2500, it measures exactly 58.30 px/cm. There is nothing to re-measure, and
    re-measuring would only add click noise to a quantity that is known analytically. The CF is
    therefore derived from the ACTUAL integer output dimensions (not the requested target), so the
    rounding every resize performs is carried into the answer instead of being assumed away.

WHAT THE PLOT IS FOR
    ``specimen.cf_px_per_cm_predicted_by_mp`` comes from a one-feature LINEAR fit,
    ``cf = slope * MP + intercept``. This experiment draws the truth for each sheet as MP varies,
    which is a square root: a sheet of fixed physical size imaged at 4x the pixel count has 2x the
    px/cm. A straight line can approximate that over a narrow band and cannot outside it -- and the
    intercept term means the line predicts a large positive CF at 0 MP, where the truth is 0.

Run (after pressing Finished in the labeler)::

    python -m leafmachine3.modules.experiments.MP_range.expand_and_plot
    python -m leafmachine3.modules.experiments.MP_range.expand_and_plot --write-images

Writes ``mp_cf_grid.csv`` (every parent x every ladder rung) and ``comparison.png``. Image files are
written only with ``--write-images`` -- the ladder is ~480 renders and several GB, and the plot
does not need them; they exist for anyone who wants to re-measure or refit on real pixels.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: Repo root, so the production artifact + its fit data can be found from anywhere.
REPO = HERE.parents[3]
MODEL_JSON = REPO / "models" / "mp_conversion_factor" / "model.json"
FIT_DATA = REPO / "models" / "mp_conversion_factor" / "fit_data.csv"


def shipped_model() -> tuple[float, float, str]:
    """The coefficients PRODUCTION actually uses, read through LM3's own loader.

    Read rather than hardcoded so that refitting the model and re-running this script cannot
    disagree -- the line drawn on the figure is by construction the line the pipeline anchors
    against. The loader already falls back to its baked coefficients if the artifact is missing.
    """
    from leafmachine3.inference.mp_conversion_factor import load_model

    m = load_model(MODEL_JSON)
    return m.slope, m.intercept, m.source


def fit_data() -> tuple[np.ndarray, np.ndarray]:
    """The (MP, CF) pairs the shipped line was fit on, or empty arrays if the CSV is absent."""
    if not FIT_DATA.exists():
        return np.array([]), np.array([])
    mp, cf = [], []
    with FIT_DATA.open() as fh:
        for row in csv.DictReader(fh):
            try:
                mp.append(float(row["mp"]))
                cf.append(float(row["cf"]))
            except (KeyError, TypeError, ValueError):
                continue
    return np.array(mp), np.array(cf)


# ------------------------------------------------------------------ the ladder
def build_ladder(native_mps: list[float], down_steps: int, floor_mp: float) -> list[float]:
    """The common target ladder: every real native MP, plus a geometric descent to ``floor_mp``.

    The real natives are what "full coverage at the true real MPs" means -- every parent gets a
    point at every resolution the corpus actually contains, which is what makes the parents
    comparable at a shared x. The descent below the smallest native is geometric rather than linear
    because CF is multiplicative in the resize factor: even steps in ratio give even spacing on the
    log axis where the relationship is a straight line.
    """
    reals = sorted(set(round(m, 4) for m in native_mps))
    lo = reals[0]
    ratio = (floor_mp / lo) ** (1.0 / down_steps)
    down = [round(lo * ratio ** k, 4) for k in range(1, down_steps + 1)]
    return sorted(set(down + reals))


def expand(labels: list[dict], ladder: list[float]) -> list[dict]:
    """One row per (parent, ladder rung) with the exact output dims and the derived CF."""
    rows: list[dict] = []
    for lab in labels:
        w0, h0, cf0 = int(lab["width"]), int(lab["height"]), float(lab["px_per_cm"])
        mp0 = w0 * h0 / 1e6
        for target in ladder:
            s = math.sqrt(target / mp0)
            w = max(1, int(round(w0 * s)))
            h = max(1, int(round(h0 * s)))
            # The achieved scale, not the requested one: integer dims never land exactly on the
            # target, and the sqrt of the area ratio is the scale a distance in any direction sees.
            actual_s = math.sqrt((w * h) / (w0 * h0))
            rows.append({
                "parent": lab["name"],
                "native_mp": round(mp0, 4),
                "native_cf": round(cf0, 4),
                "target_mp": target,
                "width": w,
                "height": h,
                "mp": round(w * h / 1e6, 6),
                "cf_px_per_cm": round(cf0 * actual_s, 6),
                "scale": round(actual_s, 6),
                # Identify the native rung by UNCHANGED DIMENSIONS, not by matching the target
                # MP: the ladder is rounded to 4 dp while mp0 is not, so 4865x6105 -> 29.700825
                # never equals its own rung 29.7008 and the real measurement went unmarked.
                "kind": "native" if (w == w0 and h == h0) else ("up" if s > 1 else "down"),
            })
    return rows


def write_images(rows: list[dict], manifest: list[dict], out_dir: Path, quality: int) -> None:
    """Materialize every rung as a JPEG under ``out_dir/<parent stem>/``."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    src = {m["name"]: m["path"] for m in manifest}
    by_parent: dict[str, list[dict]] = {}
    for r in rows:
        by_parent.setdefault(r["parent"], []).append(r)
    total = len(rows)
    n = 0
    for parent, group in by_parent.items():
        with Image.open(src[parent]) as im:
            im = im.convert("RGB")
            dest = out_dir / Path(parent).stem
            dest.mkdir(parents=True, exist_ok=True)
            for r in group:
                n += 1
                p = dest / f"{Path(parent).stem}__mp{r['mp']:09.4f}__{r['width']}x{r['height']}.jpg"
                if p.exists():
                    continue
                im.resize((r["width"], r["height"]), Image.LANCZOS).save(p, quality=quality)
                print(f"    [{n}/{total}] {p.name}", flush=True)


# ------------------------------------------------------------------ fits
def fit_linear(mp: np.ndarray, cf: np.ndarray) -> tuple[float, float]:
    """``cf = a*MP + b`` -- the form LM3 ships, refit on this set's native points."""
    a, b = np.polyfit(mp, cf, 1)
    return float(a), float(b)


def fit_sqrt(mp: np.ndarray, cf: np.ndarray) -> float:
    """``cf = k*sqrt(MP)`` -- the physically correct form, least squares through the origin.

    A sheet of physical width ``W`` cm imaged at ``w`` px has ``cf = w/W``, and ``MP = w*h/1e6``,
    so for a fixed aspect ratio ``cf`` is proportional to ``sqrt(MP)``. The single free parameter
    absorbs the population's mean physical size and aspect; there is no intercept to fit because a
    zero-pixel image has a zero conversion factor.
    """
    x = np.sqrt(mp)
    return float((x @ cf) / (x @ x))


def fit_power(mp: np.ndarray, cf: np.ndarray) -> tuple[float, float]:
    """``cf = a*MP^b`` with b FREE, fit in log space.

    The diagnostic that settles the argument without assuming it: if the exponent lands on 1/2 when
    nothing constrains it, the square-root form is what the data has been saying all along.
    """
    b, la = np.polyfit(np.log(mp), np.log(cf), 1)
    return float(np.exp(la)), float(b)


def fit_stats(pred: np.ndarray, truth: np.ndarray) -> dict:
    """RMSE / R^2 / percentage-error summary for one candidate form."""
    res = truth - pred
    ss = float(((truth - truth.mean()) ** 2).sum())
    pe = np.abs(100.0 * res / truth)
    return {"rmse": float(np.sqrt(res @ res / len(truth))),
            "r2": float(1 - (res @ res) / ss) if ss else float("nan"),
            "mape": float(pe.mean()), "p95": float(np.percentile(pe, 95))}


def pct_err(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return 100.0 * (pred - truth) / truth


def scale_equivariance_check(labels: list[dict], k: float, slope: float, intercept: float,
                             cap: int = 3200) -> None:
    """Show that the sqrt anchor can be computed from the WORKING image alone, and the linear one cannot.

    LM3 needs the anchor in the working frame. Today it predicts in the ORIGINAL frame and multiplies
    by work_scale, which is why ``specimen.original_width/height`` are load-bearing. With the sqrt
    form that round trip disappears::

        MP_work = MP_orig * ws^2        (area scales as the square of a linear resize)
        sqrt(MP_work) = sqrt(MP_orig) * ws
        k*sqrt(MP_orig) * ws  ==  k*sqrt(MP_work)          <- ws cancels exactly

    The linear form cannot do this, and it is specifically the INTERCEPT that breaks it: scaling
    ``a*MP + b`` by ws scales b too, while evaluating it at MP_work does not. Only a function that
    is homogeneous in the linear scale factor survives the frame change, and ``k*sqrt(MP)`` is the
    one the physics already asked for.
    """
    print(f"\n  anchor computed from the WORKING image alone (long side capped to {cap}px):")
    print(f"    {'sheet':32s} {'ws':>6s} | {'sqrt via orig':>13s} {'sqrt work-only':>14s} {'diff':>8s}"
          f" | {'lin via orig':>12s} {'lin work-only':>13s} {'diff':>7s}")
    ds, dl = [], []
    for lab in labels:
        w0, h0 = int(lab["width"]), int(lab["height"])
        ws = min(1.0, cap / max(w0, h0))
        w, h = max(1, round(w0 * ws)), max(1, round(h0 * ws))
        mo, mw = w0 * h0 / 1e6, w * h / 1e6
        s_via, s_only = k * math.sqrt(mo) * ws, k * math.sqrt(mw)
        l_via, l_only = (slope * mo + intercept) * ws, slope * mw + intercept
        ds.append(abs(s_only / s_via - 1) * 100)
        dl.append(abs(l_only / l_via - 1) * 100)
        print(f"    {lab['name'][:32]:32s} {ws:6.3f} | {s_via:13.3f} {s_only:14.3f} {ds[-1]:7.4f}%"
              f" | {l_via:12.2f} {l_only:13.2f} {dl[-1]:6.1f}%")
    print(f"\n    sqrt   : mean {sum(ds)/len(ds):.4f}%  max {max(ds):.4f}%   -> work_scale CANCELS;"
          f" the original dims are not needed")
    print(f"    linear : mean {sum(dl)/len(dl):.1f}%    max {max(dl):.1f}%     -> the intercept does not"
          f" scale; original dims REQUIRED")


# ------------------------------------------------------------------ plot
def plot(rows: list[dict], labels: list[dict], out_png: Path, dark: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    # Surfaces and ink. Text wears ink tokens; the series color lives only in the marks.
    surface = "#12151a" if dark else "#ffffff"
    ink, ink2, ink3 = ("#e9ecf2", "#aab3c2", "#5d6675") if dark else ("#1b1f27", "#5a6472", "#c6ccd6")
    ref = "#e05a4f" if dark else "#c0392b"          # the shipped model -- a reserved reference color
    alt = "#f5f7fa" if dark else "#111827"          # corrected sqrt form -- achromatic,
                                                    # so it cannot be mistaken for a viridis series

    parents = sorted({r["parent"] for r in rows}, key=lambda p: next(
        x["native_mp"] for x in rows if x["parent"] == p))
    # SEQUENTIAL ramp keyed to native MP, not 20 categorical hues. The parents have a real order,
    # so the ramp both gives every parent its own color AND lets a reader place a curve on the
    # scale without a 20-entry legend -- which is the only honest way past 8 series.
    norm = Normalize(min(r["native_mp"] for r in rows), max(r["native_mp"] for r in rows))
    cmap = plt.get_cmap("viridis")
    color = {p: cmap(norm(next(x["native_mp"] for x in rows if x["parent"] == p))) for p in parents}

    nat_mp = np.array([lab["mp"] for lab in labels], float)
    nat_cf = np.array([lab["px_per_cm"] for lab in labels], float)
    a, b = fit_linear(nat_mp, nat_cf)
    slope, intercept, src = shipped_model()
    fmp, fcf = fit_data()
    lo, hi = (float(fmp.min()), float(fmp.max())) if fmp.size else (14.6, 36.2)
    # The candidate replacement is refit on the SAME 708 sheets production was fit on, not on the
    # 20 labeled here -- 20 sheets settle the functional form, they do not set a coefficient.
    k = fit_sqrt(fmp, fcf) if fmp.size else fit_sqrt(nat_mp, nat_cf)

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 5.3), facecolor=surface)
    for ax in axes:
        ax.set_facecolor(surface)
        ax.grid(True, color=ink3, lw=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink3)
        ax.tick_params(colors=ink2, labelsize=9)
        ax.xaxis.label.set_color(ink2)
        ax.yaxis.label.set_color(ink2)

    grid = np.linspace(0.5, max(r["mp"] for r in rows) * 1.03, 400)

    # -- A: the truth curves against the shipped line --------------------------------
    ax = axes[0]
    ax.axvspan(lo, hi, color=ink3, alpha=0.30, lw=0, zorder=0)
    ax.text((lo + hi) / 2, 0.985, f"production fit range\n{lo:.1f}–{hi:.1f} MP", ha="center",
            va="top", transform=ax.get_xaxis_transform(), fontsize=8, color=ink2, linespacing=1.4)
    if fmp.size:                                     # the cloud the production line was fit through
        ax.plot(fmp, fcf, ".", ms=3, color=ink3, alpha=0.85, zorder=1,
                label=f"fit data (n={fmp.size})")
    for p in parents:
        g = sorted((r for r in rows if r["parent"] == p), key=lambda r: r["mp"])
        ax.plot([r["mp"] for r in g], [r["cf_px_per_cm"] for r in g],
                color=color[p], lw=1.6, alpha=0.85, zorder=2)
        n = next((r for r in g if r["kind"] == "native"), None)
        if n:                                            # the one real measurement on this curve
            ax.plot(n["mp"], n["cf_px_per_cm"], "o", ms=8, color=color[p],
                    mec=surface, mew=1.8, zorder=7)      # surface ring so it reads over the lines
    ax.plot(grid, slope * grid + intercept, color=ref, lw=2.6, ls="--", zorder=6,
            label=f"PRODUCTION  cf = {slope:.2f}·MP + {intercept:.1f}")
    ax.plot(grid, k * np.sqrt(grid), color=alt, lw=2.4, zorder=5,
            label=f"corrected  cf = {k:.2f}·√MP  (refit on the 708)")
    ax.set_xlabel("megapixels"), ax.set_ylabel("conversion factor  (px / cm)")
    ax.set_title("Each sheet's true CF across the MP range", color=ink, fontsize=11, loc="left", pad=10)
    ax.set_xlim(0, grid[-1]), ax.set_ylim(0, None)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(ink2)

    # -- B: log-log, where the correct form is a straight line -----------------------
    ax = axes[1]
    for p in parents:
        g = sorted((r for r in rows if r["parent"] == p), key=lambda r: r["mp"])
        ax.plot([r["mp"] for r in g], [r["cf_px_per_cm"] for r in g],
                color=color[p], lw=1.6, alpha=0.85, zorder=2)
    if fmp.size:
        ax.plot(fmp, fcf, ".", ms=3, color=ink3, alpha=0.85, zorder=1)
    ax.plot(grid, slope * grid + intercept, color=ref, lw=2.6, ls="--", zorder=6)
    ax.plot(grid, k * np.sqrt(grid), color=alt, lw=2.4, zorder=5)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.annotate("production", (grid[-1], slope * grid[-1] + intercept), color=ref, fontsize=8.5,
                ha="right", va="bottom", xytext=(0, 6), textcoords="offset points")
    # label the sqrt line at the LEFT end -- at the right end both reference lines converge and
    # the two annotations sat on top of each other
    ax.annotate("corrected  √MP", (grid[0], k * np.sqrt(grid[0])), color=alt, fontsize=8.5,
                ha="left", va="top", xytext=(4, -4), textcoords="offset points")
    ax.set_xlabel("megapixels  (log)"), ax.set_ylabel("px / cm  (log)")
    ax.set_title("Log-log: truth is slope ½; the line is not", color=ink, fontsize=11, loc="left", pad=10)

    # -- C: what the shipped model costs you -----------------------------------------
    ax = axes[2]
    ax.axvspan(lo, hi, color=ink3, alpha=0.30, lw=0, zorder=0)
    ax.axhline(0, color=ink2, lw=1, zorder=1)
    for p in parents:
        g = sorted((r for r in rows if r["parent"] == p), key=lambda r: r["mp"])
        m = np.array([r["mp"] for r in g]), np.array([r["cf_px_per_cm"] for r in g])
        ax.plot(m[0], pct_err(slope * m[0] + intercept, m[1]),
                color=color[p], lw=1.6, alpha=0.85, zorder=2)
    ax.set_xscale("log")
    ax.text(0.985, 0.985, f"production fit range\n{lo:.1f}–{hi:.1f} MP", ha="right", va="top",
            transform=ax.transAxes, fontsize=8, color=ink2, linespacing=1.4)
    ax.set_xlabel("megapixels  (log)"), ax.set_ylabel("production anchor error  (%)")
    ax.set_title("Error explodes below the fit range", color=ink, fontsize=11, loc="left", pad=10)

    # -- D: the same question asked of production's OWN 708 training sheets ----------
    ax = axes[3]
    if fmp.size:
        pa, pb = fit_linear(fmp, fcf)
        pk = fit_sqrt(fmp, fcf)
        amp, bexp = fit_power(fmp, fcf)
        lin_s, sq_s = fit_stats(pa * fmp + pb, fcf), fit_stats(pk * np.sqrt(fmp), fcf)
        wide = np.linspace(0.5, 60, 400)
        ax.axvspan(lo, hi, color=ink3, alpha=0.30, lw=0, zorder=0)
        ax.plot(fmp, fcf, ".", ms=3.5, color=ink3, alpha=0.9, zorder=1, label=f"708 fit sheets")
        ax.plot(wide, slope * wide + intercept, color=ref, lw=2.6, ls="--", zorder=4,
                label=f"production linear   RMSE {lin_s['rmse']:.2f}  R² {lin_s['r2']:.3f}")
        ax.plot(wide, pk * np.sqrt(wide), color=alt, lw=2.4, zorder=3,
                label=f"√MP refit  k={pk:.2f}   RMSE {sq_s['rmse']:.2f}  R² {sq_s['r2']:.3f}")
        # The exponent nobody constrained. This is the whole argument in one number.
        ax.text(0.035, 0.955, f"free power fit:  cf = {amp:.2f}·MP$^{{{bexp:.3f}}}$\n"
                              f"theory says the exponent is 0.500",
                transform=ax.transAxes, fontsize=8.5, color=ink, va="top", linespacing=1.5)
        ax.text((lo + hi) / 2, 0.985, "where production\nhas data", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=8, color=ink2, linespacing=1.4)
        ax.set_xlim(0, 60), ax.set_ylim(0, None)
        leg = ax.legend(loc="lower right", frameon=False, fontsize=8)
        for t in leg.get_texts():
            t.set_color(ink2)
    ax.set_xlabel("megapixels"), ax.set_ylabel("px / cm")
    ax.set_title("Refit on production's own 708 sheets", color=ink, fontsize=11, loc="left", pad=10)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axes, fraction=0.016, pad=0.012)
    cb.set_label("parent sheet — native megapixels", color=ink2, fontsize=9)
    cb.ax.tick_params(colors=ink2, labelsize=8)
    cb.outline.set_visible(False)

    fig.suptitle(f"LM3 megapixel → conversion-factor anchor ({src}) vs. ground truth   "
                 f"({len(parents)} hand-labeled sheets, {len(rows)} derived points)",
                 color=ink, fontsize=12.5, x=0.008, ha="left", y=0.985)
    fig.savefig(out_png, dpi=150, facecolor=surface, bbox_inches="tight")
    print(f"  wrote {out_png}")


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Expand labeled sheets across the MP range and plot.")
    ap.add_argument("--dir", type=Path, default=HERE)
    ap.add_argument("--down-steps", type=int, default=4, help="geometric steps down to --floor-mp")
    ap.add_argument("--floor-mp", type=float, default=1.0, help="the bottom rung, in megapixels")
    ap.add_argument("--write-images", action="store_true", help="materialize every rung as a JPEG")
    ap.add_argument("--images-out", type=Path, default=None)
    ap.add_argument("--jpg-quality", type=int, default=92)
    ap.add_argument("--dark", action="store_true", help="also render a dark-surface figure")
    args = ap.parse_args(argv)

    labels_path = args.dir / "labels.json"
    if not labels_path.exists():
        raise SystemExit(f"no labels at {labels_path} -- run label_server.py first")
    blob = json.loads(labels_path.read_text())
    labels = sorted(blob["labels"].values(), key=lambda r: r["mp"])
    if not labels:
        raise SystemExit("labels.json has no labeled sheets")
    if not blob.get("finished"):
        print("  note: the labeler was not marked Finished; using what is there")
    manifest = json.loads((args.dir / "manifest.json").read_text())

    ladder = build_ladder([lab["mp"] for lab in labels], args.down_steps, args.floor_mp)
    rows = expand(labels, ladder)
    print(f"  {len(labels)} labeled sheets x {len(ladder)} rungs = {len(rows)} points")
    print(f"  ladder: {', '.join(f'{m:g}' for m in ladder)}")

    csv_path = args.dir / "mp_cf_grid.csv"
    with csv_path.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"  wrote {csv_path}")

    if args.write_images:
        out = args.images_out or (args.dir / "renders")
        px = sum(r["width"] * r["height"] for r in rows)
        print(f"  writing {len(rows)} images (~{px / 1e9:.1f} gigapixels) under {out} ...")
        write_images(rows, manifest, out, args.jpg_quality)

    plot(rows, labels, args.dir / "comparison.png")
    if args.dark:
        plot(rows, labels, args.dir / "comparison_dark.png", dark=True)

    # -- the numbers the figure is making a case about --------------------------------
    nat_mp = np.array([lab["mp"] for lab in labels], float)
    nat_cf = np.array([lab["px_per_cm"] for lab in labels], float)
    k = fit_sqrt(nat_mp, nat_cf)
    a, b = fit_linear(nat_mp, nat_cf)
    mp = np.array([r["mp"] for r in rows])
    cf = np.array([r["cf_px_per_cm"] for r in rows])
    slope, intercept, src = shipped_model()
    fmp, fcf = fit_data()
    lo, hi = (float(fmp.min()), float(fmp.max())) if fmp.size else (14.6, 36.2)
    shipped = pct_err(slope * mp + intercept, cf)
    sqrt_err = pct_err(k * np.sqrt(mp), cf)
    inside = (mp >= lo) & (mp <= hi)

    print(f"\n  production anchor ({src}):  cf = {slope:.4f}*MP + {intercept:.2f}"
          f"   [fit on MP {lo:.1f}-{hi:.1f}]")
    if fmp.size:
        print(f"\n  REFIT on production's own {fmp.size} sheets:")
        pa, pb = fit_linear(fmp, fcf)
        pk = fit_sqrt(fmp, fcf)
        amp, bexp = fit_power(fmp, fcf)
        A = np.column_stack([np.sqrt(fmp), np.ones_like(fmp)])
        (ck, cc), *_ = np.linalg.lstsq(A, fcf, rcond=None)
        cands = [("linear   cf = a*MP + b", pa * fmp + pb, f"a={pa:.4f} b={pb:.3f}"),
                 ("sqrt     cf = k*sqrt(MP)", pk * np.sqrt(fmp), f"k={pk:.4f}"),
                 ("sqrt+c   cf = k*sqrt(MP)+c", ck * np.sqrt(fmp) + cc, f"k={ck:.4f} c={cc:.3f}"),
                 ("power    cf = a*MP^b", amp * fmp ** bexp, f"a={amp:.4f} b={bexp:.4f}")]
        print(f"    {'form':28s} {'params':28s} {'RMSE':>6s} {'R2':>7s} {'mean|%|':>8s} {'p95':>6s}")
        for nm, pred, params in cands:
            st = fit_stats(pred, fcf)
            print(f"    {nm:28s} {params:28s} {st['rmse']:6.2f} {st['r2']:7.4f} "
                  f"{st['mape']:7.2f}% {st['p95']:5.2f}%")
        print(f"\n    free exponent b = {bexp:.4f}  (theory: 0.5) | sqrt+c intercept c = {cc:.3f}"
              f"  (theory: 0)")
        print(f"    extrapolation:  {'MP':>6s} {'linear':>9s} {'sqrt':>9s}")
        for m in (1.0, 3.0, 8.0, 25.0, 60.0, 101.0):
            print(f"                    {m:6.1f} {slope * m + intercept:9.2f} {pk * math.sqrt(m):9.2f}")
    if fmp.size:
        scale_equivariance_check(labels, fit_sqrt(fmp, fcf), slope, intercept)
    print(f"\n  refit on these {len(labels)} natives:")
    print(f"    linear (shipped form)  cf = {a:.3f}*MP + {b:.2f}")
    print(f"    sqrt   (correct form)  cf = {fit_sqrt(nat_mp, nat_cf):.3f}*sqrt(MP)")
    print(f"\n  {'':22s} {'mean |err|':>11s} {'p95 |err|':>10s} {'max |err|':>10s}")
    for name, err, mask in (("shipped, inside range", shipped, inside),
                            ("shipped, outside range", shipped, ~inside),
                            ("shipped, all rungs", shipped, np.ones_like(inside)),
                            ("sqrt refit, all rungs", sqrt_err, np.ones_like(inside))):
        e = np.abs(err[mask.astype(bool)])
        if e.size:
            print(f"  {name:22s} {e.mean():10.1f}% {np.percentile(e, 95):9.1f}% {e.max():9.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
