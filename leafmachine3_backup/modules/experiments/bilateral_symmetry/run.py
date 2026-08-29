"""Driver: measure bilateral symmetry over an already-computed run and write the HTML report.

    .venv_LM3/bin/python -m leafmachine3.modules.experiments.bilateral_symmetry.run \
        --run examples_out/testing_up_to_specimen_seg

Reads only; writes ``modules/experiments/bilateral_symmetry.html`` plus a per-leaf CSV next to it.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from leafmachine3.core.db import ProjectDB
from leafmachine3.modules.experiments.bilateral_symmetry import figures as F
from leafmachine3.modules.experiments.bilateral_symmetry.axes import build_axis, profiles
from leafmachine3.modules.experiments.bilateral_symmetry.geometry import load_leaves
from leafmachine3.modules.experiments.bilateral_symmetry.metrics import (
    area_asymmetry_profile,
    compute_metrics,
    cumulative_imbalance,
    width_asymmetry_profile,
)
from leafmachine3.modules.experiments.bilateral_symmetry.quality import (
    archetype_score,
    archetype_subscores,
    compute_quality,
)
from leafmachine3.modules.experiments.bilateral_symmetry.report import render_report
from leafmachine3.modules.experiments.bilateral_symmetry.shape import compute_shape

log = logging.getLogger("leafmachine3.bilateral_symmetry")

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent                     # modules/experiments/
DEFAULT_RUN = "examples_out/testing_up_to_specimen_seg"
# Exemplar cut. quality.py ships a much stricter default; on a cohort of already-clean masks that
# selects a top slice rather than a usability bar, so the driver default is set to the bar that
# was actually wanted here. Structural gates always apply on top of it.
DEFAULT_MIN_SCORE = 0.50

# the metrics carried into the cohort figures / table (one representative per family)
TABLE_KEYS: list[tuple[str, str, str]] = [
    ("mid_SI_A", "SI_A", "Shi et al. standardized index on the midvein axis (0 = perfect)"),
    ("mid_WSI_A", "WSI_A", "area-weighted = symmetric-difference area / lamina area"),
    ("mid_A_star", "A*", "signed total imbalance, + = viewer's left half larger"),
    ("mid_dice", "Dice", "overlap of the straightened, mirrored halves (1 = perfect)"),
    ("mid_hausdorff_norm", "Hausdorff/L", "worst-case margin mismatch, normalized by axis length"),
    ("mid_r_w", "r_w", "correlation of the two half-width profiles"),
    ("mid_C_max", "C_max", "largest running signed imbalance along the axis"),
    ("mid_sinuosity", "sinuosity", "midvein arclength / straight tip-base distance"),
    ("d_SI_A", "ΔSI_A", "chord minus midvein: apparent asymmetry caused by a curved midvein"),
    ("solidity", "solidity", "mask area / convex hull area (also a lobing signal)"),
    ("archetype_score", "score", "composite archetypal-leaf score in [0,1]"),
]

DIST_KEYS = ["mid_SI_A", "mid_WSI_A", "mid_A_star", "mid_dice", "mid_r_w", "mid_C_max",
             "mid_hausdorff_norm", "mid_sinuosity", "solidity", "archetype_score"]

CORR_KEYS = ["mid_SI_A", "mid_WSI_A", "mid_RMSE_w", "mid_NRMSE_w", "mid_r_w", "mid_dice",
             "mid_iou", "mid_hausdorff_norm", "mid_d_c_norm", "mid_C_max", "mid_I_C",
             "mid_aA_SI_apex", "mid_aA_SI_base", "mid_sinuosity", "solidity", "perimeter_ratio"]


# --- exemplar gate policy -------------------------------------------------------------------
# Owned by the driver, not quality.py, so it can be tuned without touching the scoring module.
# Three of quality.py's stock gates are deliberately NOT applied here:
#   holes       -- the shape under test is the HOLES-FILLED silhouette (LM3's `lamina` /
#                  Lamina_Holes_Mask product), so an insect hole is not part of the outline whose
#                  symmetry is being measured. hole_frac stays in the CSV as a diagnostic but is
#                  zeroed before scoring so it cannot influence the composite either.
#   n_components-- NOTE this is NOT because the holes are filled: filling a hole removes an interior
#                  void, it cannot merge two disjoint blobs, and measured on this cohort the counts
#                  are identical on the holes-filled and holes-punched masks for all 105 leaves.
#                  The raw count is dropped because it counts SPECKS: 18 leaves have >1 component
#                  but 12 of those have under 0.1% of their area outside the main blob, and one
#                  leaf scoring 0.96 was vetoed by a second component holding 0.001%. largest_frac
#                  keeps the part that matters -- it flags the 2 genuinely fragmented masks.
#   flush_edge  -- measured r = 0.85 against 1/sqrt(area_px), i.e. it is a leaf-SIZE proxy rather
#                  than a cut-off detector; `truncated` answers that question directly.
GATE_MIN_LARGEST_FRAC = 0.995
GATE_MIN_MIDVEIN_KPTS = 10


def gates_pass(q: Any) -> tuple[bool, list[str]]:
    """Structural vetoes -- faults that disqualify a leaf regardless of how symmetric it looks."""
    reasons: list[str] = []
    if bool(getattr(q, "truncated", False)):
        reasons.append("detection box was clipped by the sheet edge")
    if not float(getattr(q, "area_px", 0) or 0) > 0:
        reasons.append("empty mask")
    lf = float(getattr(q, "largest_frac", float("nan")) or float("nan"))
    if not (np.isfinite(lf) and lf >= GATE_MIN_LARGEST_FRAC):
        reasons.append(f"largest component holds only {lf:.4f} of the mask area")
    nk = float(getattr(q, "n_midvein_kpts", 0) or 0)
    if nk < GATE_MIN_MIDVEIN_KPTS:
        reasons.append(f"only {nk:.0f} midvein keypoints survived the confidence filter")
    return (not reasons), reasons


def _f(v: Any, nd: int = 3) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(x) else f"{x:.{nd}f}"


def measure_leaf(leaf, *, n_bins: int, min_score: float = DEFAULT_MIN_SCORE) -> Optional[dict]:
    """All metrics for one leaf on BOTH axes, flattened into a single row dict."""
    mv, tb = leaf.midvein(), leaf.tip_base()
    if tb is None:
        return None
    tip, base = tb
    shape_mask = leaf.silhouette          # holes filled: outline symmetry, not damage

    frames = {
        "chord": build_axis(shape_mask, "chord", tip=tip, base=base),
        "mid": build_axis(shape_mask, "midvein", midvein=mv, tip=tip, base=base) if mv is not None else None,
    }
    if frames["chord"] is None:
        return None

    row: dict[str, Any] = {
        "specimen_id": leaf.specimen_id, "leaf_id": leaf.leaf_id,
        "detection_id": leaf.detection_id, "stem": leaf.stem,
        "name": f"{leaf.stem}#{leaf.leaf_id}",
        "truncated": int(leaf.truncated),
    }
    # quality.archetype_score resolves metrics as sym[axis][name] first, so hand it the nested
    # layout keyed by quality.SYM_AXIS ("midvein") while the flat row keeps the short mid_/chord_
    # prefixes the table, figures and CSV use.
    nested: dict[str, dict] = {}
    for pre, fr in frames.items():
        if fr is None:
            continue
        pr = profiles(fr, n_bins=n_bins)
        flat = {**compute_metrics(fr, pr).as_dict(), **compute_shape(fr, pr).as_dict()}
        nested["midvein" if pre == "mid" else pre] = flat
        for k, v in flat.items():
            row[f"{pre}_{k}"] = v

    q = compute_quality(leaf, frames["mid"] or frames["chord"])
    row.update(q.as_dict())          # the REAL hole_frac is reported, just not scored
    # Holes are filled in the mask being measured, so zero them before scoring: otherwise the
    # integrity sub-score penalizes damage that is not present in the shape under test.
    q_score = replace(q, hole_frac=0.0)
    score, reasons = archetype_score(nested, q_score)
    subs, _ = archetype_subscores(nested, q_score)
    gates_ok, gate_reasons = gates_pass(q)
    row["archetype_score"] = score
    row["gates_pass"] = gates_ok
    row["is_archetypal"] = bool(gates_ok and np.isfinite(score) and score >= min_score)
    row["reasons"] = gate_reasons + reasons
    for k, v in subs.items():
        row[f"term_{k}"] = v

    # chord - midvein deltas: apparent asymmetry that is really a curved midvein
    for k in ("SI_A", "WSI_A", "A_star", "dice", "iou", "NRMSE_w", "C_max", "r_w"):
        c, m = row.get(f"chord_{k}"), row.get(f"mid_{k}")
        row[f"d_{k}"] = (float(c) - float(m)) if (c is not None and m is not None
                                                  and np.isfinite(c) and np.isfinite(m)) else float("nan")
    row["_leaf"] = leaf
    row["_frames"] = frames
    return row


def collect(db, *, min_kpt_conf: float, n_bins: int, limit: Optional[int],
            min_score: float = DEFAULT_MIN_SCORE) -> list[dict]:
    rows: list[dict] = []
    for sid in db.specimens_with_rows("leaf_segmentation"):
        for leaf in load_leaves(db, sid, min_kpt_conf=min_kpt_conf):
            try:
                r = measure_leaf(leaf, n_bins=n_bins, min_score=min_score)
            except Exception as e:                       # one bad leaf must not kill the cohort
                log.warning("leaf %s failed: %s", leaf.leaf_id, e)
                continue
            if r is not None:
                rows.append(r)
        if limit and len(rows) >= limit:
            break
    return rows


def _finite(rows: list[dict], key: str) -> np.ndarray:
    v = np.array([r.get(key, np.nan) for r in rows], float)
    return v[np.isfinite(v)]


def _median(rows: list[dict], key: str) -> float:
    v = _finite(rows, key)
    return float(np.median(v)) if v.size else float("nan")


def _leaf_card(r: dict) -> dict:
    fr = r["_frames"]
    mid = fr.get("mid") or fr["chord"]
    pr = profiles(mid)
    panel = F.fig_leaf_panel(r["_leaf"], fr["chord"], mid, pr)
    chips = [
        ("SI_A", _f(r.get("mid_SI_A"))), ("Dice", _f(r.get("mid_dice"))),
        ("A*", _f(r.get("mid_A_star"))), ("ΔSI_A", _f(r.get("d_SI_A"))),
        ("sinuosity", _f(r.get("mid_sinuosity"), 4)), ("score", _f(r.get("archetype_score"))),
    ]
    return {"name": r["name"], "panel": panel, "chips": chips,
            "is_archetypal": bool(r.get("is_archetypal")), "truncated": bool(r.get("truncated")),
            "reasons": r.get("reasons") or []}


def build_context(rows: list[dict], run_name: str, *, gallery_n: int, db_path: str) -> dict:
    ok = [r for r in rows if np.isfinite(r.get("mid_SI_A", np.nan))]
    n_arch = sum(1 for r in rows if r.get("is_archetypal"))
    med_si_c, med_si_m = _median(rows, "chord_SI_A"), _median(rows, "mid_SI_A")
    med_dice_c, med_dice_m = _median(rows, "chord_dice"), _median(rows, "mid_dice")
    d_si = _finite(rows, "d_SI_A")
    d_dice = _finite(rows, "d_dice")

    stats = [
        ("leaves measured", len(rows)),
        ("specimens", len({r["specimen_id"] for r in rows})),
        ("median SI_A (midvein)", _f(med_si_m)),
        ("median Dice (midvein)", _f(med_dice_m)),
        ("median SI_A (chord)", _f(med_si_c)),
        ("median Dice (chord)", _f(med_dice_c)),
        ("median ΔSI_A", _f(float(np.median(d_si)) if d_si.size else float("nan"))),
        ("flagged archetypal", f"{n_arch} / {len(rows)}"),
    ]

    figs = {}
    try:
        figs["distributions"] = F.fig_metric_distributions(rows, DIST_KEYS)
        figs["correlation"] = F.fig_correlation_matrix(rows, [k for k in CORR_KEYS
                                                              if _finite(rows, k).size > 3])
        figs["chord_vs_midvein"] = F.fig_chord_vs_midvein(rows, "chord_SI_A", "mid_SI_A", "SI_A")
        figs["taylor"] = F.fig_taylor(rows, "mid_M_D", "mid_V_D")
        figs["taylor_norm"] = F.fig_taylor(rows, "mid_M_d", "mid_V_d")
        figs["score_vs_symmetry"] = F.fig_score_vs_symmetry(rows)
    except Exception as e:
        log.warning("figure build failed: %s", e)

    # One thumbnail per leaf so EVERY point in the scatter is clickable -- the whole cohort, not
    # just the leaves that made a gallery. Kept small (fig_leaf_split, ~78 dpi) since 105 of them
    # are embedded in the page.
    scatter_points: list[dict] = []
    for r in rows:
        x, y = r.get("mid_SI_A"), r.get("archetype_score")
        if not (np.isfinite(x if x is not None else np.nan)
                and np.isfinite(y if y is not None else np.nan)):
            continue
        fr = r["_frames"]
        try:
            thumb = F.fig_leaf_split(r["_leaf"], fr.get("chord"), fr.get("mid"))
        except Exception as e:
            log.warning("thumb failed for %s: %s", r["name"], e)
            continue
        scatter_points.append({
            "name": r["name"], "x": float(x), "y": float(y),
            "arch": bool(r.get("is_archetypal")), "gates": bool(r.get("gates_pass", True)),
            "chips": _f(r.get("mid_SI_A")), "dice": _f(r.get("mid_dice")),
            "astar": _f(r.get("mid_A_star")), "img": thumb,
        })
    log.info("built %d clickable leaf thumbnails", len(scatter_points))

    by_score = sorted(ok, key=lambda r: r.get("archetype_score", 0.0), reverse=True)
    by_mover = sorted([r for r in ok if np.isfinite(r.get("d_SI_A", np.nan))],
                      key=lambda r: r["d_SI_A"], reverse=True)
    galleries = [
        ("Most archetypal", "Highest composite score. These are the candidates for a curated, "
         "&ldquo;clean mask&rdquo; training or reference set.",
         [_leaf_card(r) for r in by_score[:gallery_n]]),
        ("Least archetypal", "Lowest composite score. There are <b>three different causes</b> in "
         "here and the panels tell them apart: a genuinely messy <i>mask</i> (stray blobs, torn "
         "margin); a genuinely asymmetric <i>leaf</i> (oblique base &mdash; normal in many taxa, and "
         "the score is wrong to penalize it); and a failed <i>midvein trace</i>, which shows up "
         "unmistakably as the green axis running along a margin instead of down the middle, with "
         "nearly all the lamina on one side. Read the left panel of each before trusting the score.",
         [_leaf_card(r) for r in by_score[-gallery_n:][::-1]]),
        ("Largest chord-midvein gap", "Leaves where the straight-chord assumption inflates "
         "asymmetry the most. These are exactly the leaves a silhouette-only method misjudges.",
         [_leaf_card(r) for r in by_mover[:gallery_n]]),
    ]

    thead = "".join(f'<th title="{t}">{h}</th>' for _k, h, t in
                    [("name", "leaf", "specimen stem and leaf id")] + [(k, h, t) for k, h, t in TABLE_KEYS])
    trows = []
    for r in by_score:
        cells = "".join(f'<td class="num">{_f(r.get(k), 4 if k == "mid_sinuosity" else 3)}</td>'
                        for k, _h, _t in TABLE_KEYS)
        flag = ' <span class="badge good">arch</span>' if r.get("is_archetypal") else ""
        trows.append(f'<tr><td class="mono">{r["name"]}{flag}</td>{cells}</tr>')
    table_html = (f'<div class="tblwrap"><table><thead><tr>{thead}</tr></thead>'
                  f'<tbody>{"".join(trows)}</tbody></table></div>')

    return {
        "run_name": run_name,
        "subtitle": ("How symmetric is each leaf lamina about its own midvein, which measurements "
                     "capture that, and can symmetry be used to pick out clean, archetypal masks? "
                     "Measured on the tip-up <span class='mono'>Leaf_Oriented</span> silhouettes so "
                     "left and right mean the same thing for every leaf."),
        "stats": stats,
        "figures": figs,
        "scatter_points": scatter_points,
        "galleries": galleries,
        "table_html": table_html,
        "cohort_note": f"{len(rows)} leaves from {db_path}",
        "generated": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "method_html": _METHOD,
        "findings": _findings(rows, med_si_c, med_si_m, med_dice_c, med_dice_m, d_si, d_dice),
        "axis_note": _axis_note(rows, d_si, d_dice),
        "corr_note": _CORR_NOTE,
        "taylor_note": _taylor_note(rows),
        "score_note": _SCORE_NOTE,
        "limitations": _LIMITATIONS,
    }


def _findings(rows, med_si_c, med_si_m, med_dice_c, med_dice_m, d_si, d_dice) -> list[str]:
    out = []
    if np.isfinite(med_si_c) and np.isfinite(med_si_m) and med_si_m > 0:
        out.append(
            f"Measuring against the <b>traced midvein</b> instead of a straight chord drops median "
            f"SI<sub>A</sub> from <b>{med_si_c:.3f}</b> to <b>{med_si_m:.3f}</b> "
            f"({100 * (med_si_c - med_si_m) / med_si_c:.0f}% lower) and raises median Dice from "
            f"{med_dice_c:.3f} to <b>{med_dice_m:.3f}</b>. That difference is not biology &mdash; "
            f"it is the published straight-axis methods charging a curved midvein as asymmetry.")
    if d_si.size:
        frac = float((d_si > 0).mean())
        out.append(f"The chord axis reports MORE asymmetry than the midvein axis for "
                   f"<b>{100 * frac:.0f}%</b> of leaves (median &Delta;SI<sub>A</sub> = "
                   f"{np.median(d_si):+.3f}, max {d_si.max():+.3f}).")
    sin = _finite(rows, "mid_sinuosity")
    if sin.size and d_si.size:
        both = [(r.get("mid_sinuosity"), r.get("d_SI_A")) for r in rows]
        both = np.array([(a, b) for a, b in both if np.isfinite(a) and np.isfinite(b)])
        if len(both) > 3:
            from scipy.stats import spearmanr
            rho, p = spearmanr(both[:, 0], both[:, 1])
            out.append(f"Midvein <b>sinuosity predicts</b> that penalty (Spearman &rho; = "
                       f"<b>{rho:+.2f}</b>, p = {p:.1e}, n = {len(both)}): the more curved the "
                       f"midvein, the more a straight-chord method overstates asymmetry.")

    # Does the composite actually add anything over symmetry alone on THIS cohort?
    from scipy.stats import spearmanr
    sc = np.array([r.get("archetype_score", np.nan) for r in rows], float)
    si = np.array([r.get("mid_SI_A", np.nan) for r in rows], float)
    m = np.isfinite(sc) & np.isfinite(si)
    if m.sum() > 5:
        rho2, _p2 = spearmanr(sc[m], si[m])
        sat = []
        for t in ("integrity", "completeness", "trace"):
            v = np.array([r.get(f"term_{t}", np.nan) for r in rows], float)
            v = v[np.isfinite(v)]
            if v.size:
                sat.append(f"{t} {100 * float((v >= 0.999).mean()):.0f}%")
        out.append(
            f"The composite archetype score tracks SI<sub>A</sub> at Spearman &rho; = "
            f"<b>{rho2:+.2f}</b> &mdash; on this cohort it is <i>almost entirely</i> the symmetry "
            f"term. That is a property of the data, not a flaw in the score: the independent "
            f"quality terms saturate at 1.0 for most leaves ({', '.join(sat)}), because these masks "
            f"are already structurally clean. Symmetry is the only signal that varies here, so it "
            f"does the ranking by default &mdash; and the galleries are the check on whether a "
            f"low-symmetry leaf is a bad mask or simply an asymmetric leaf.")

    # Extreme imbalance is usually a MIDVEIN TRACE failure, not an asymmetric leaf.
    a = np.array([r.get("mid_A_star", np.nan) for r in rows], float)
    km = np.array([r.get("kpt_conf_min", np.nan) for r in rows], float)
    dc = np.array([r.get("mid_dice", np.nan) for r in rows], float)
    ext = np.isfinite(a) & (np.abs(a) > 0.4)
    if ext.any() and np.isfinite(km).sum() > 5:
        rest = np.isfinite(a) & ~ext & np.isfinite(km)
        out.append(
            f"<b>{int(ext.sum())} leaf(s) have |A*| &gt; 0.4</b> &mdash; one half more than 2.3&times; "
            f"the other, which is not a real simple-leaf shape. Their worst midvein keypoint "
            f"confidence is {np.nanmedian(km[ext]):.2f} against {np.nanmedian(km[rest]):.2f} for "
            f"everything else, and their median Dice is {np.nanmedian(dc[ext]):.2f}. These are "
            f"<i>midvein trace failures</i>, not asymmetric leaves &mdash; the pose model ran the "
            f"midvein along a margin, so the lamina ended up almost entirely on one side. Across the "
            f"whole cohort Dice and worst-keypoint confidence correlate at Spearman &rho; = "
            f"{spearmanr(dc[np.isfinite(dc) & np.isfinite(km)], km[np.isfinite(dc) & np.isfinite(km)])[0]:+.2f}. "
            f"So the most immediately useful application of these metrics may not be mask QC at all: "
            f"they are a cheap, automatic detector for <b>bad landmark traces</b>, which matters "
            f"while the pose model is still alpha.")
    return out


def _axis_note(rows, d_si, d_dice) -> str:
    if not d_si.size:
        return ""
    worst = max((r for r in rows if np.isfinite(r.get("d_SI_A", np.nan))),
                key=lambda r: r["d_SI_A"], default=None)
    extra = ""
    if worst is not None:
        extra = (f" The most affected leaf is <span class='mono'>{worst['name']}</span>: "
                 f"SI<sub>A</sub> {worst['chord_SI_A']:.3f} on the chord vs "
                 f"{worst['mid_SI_A']:.3f} on the midvein, with sinuosity "
                 f"{worst.get('mid_sinuosity', float('nan')):.4f}.")
    return ('<div class="note"><b>Why this is the interesting panel.</b> Shi et al. (2018), Wang et '
            'al. (2018) and Guo et al. (2020) all divide the lamina with a straight base&rarr;apex '
            'line, because a silhouette is all they have. LM3 traces the midvein, so the same leaf '
            'can be measured both ways and the difference attributed. A leaf whose midvein bows is '
            'penalized twice by the chord method: once because the chord cuts across the blade, and '
            'again because the strips perpendicular to it are no longer perpendicular to the leaf.'
            + extra + '</div>')


def _fit_loglog(rows, mk: str, vk: str) -> Optional[tuple[float, float, float, int]]:
    m = np.array([r.get(mk, np.nan) for r in rows], float)
    v = np.array([r.get(vk, np.nan) for r in rows], float)
    good = np.isfinite(m) & np.isfinite(v) & (m > 0) & (v > 0)
    if good.sum() < 4:
        return None
    lm, lv = np.log10(m[good]), np.log10(v[good])
    b, a = np.polyfit(lm, lv, 1)
    return float(b), float(a), float(np.corrcoef(lm, lv)[0, 1] ** 2), int(good.sum())


def _taylor_note(rows) -> str:
    raw = _fit_loglog(rows, "mid_M_D", "mid_V_D")
    nrm = _fit_loglog(rows, "mid_M_d", "mid_V_d")
    if raw is None:
        return ""
    b, _a, r2, n = raw
    # is M_D just tracking leaf size? if so beta==2 is arithmetic, not biology
    area = np.array([r.get("mid_lamina_area_px", np.nan) for r in rows], float)
    md = np.array([r.get("mid_M_D", np.nan) for r in rows], float)
    ok = np.isfinite(area) & np.isfinite(md) & (area > 0) & (md > 0)
    rho = float(np.corrcoef(np.log10(area[ok]), np.log10(md[ok]))[0, 1]) if ok.sum() > 3 else float("nan")

    second = ""
    if nrm is not None:
        b2, _a2, r2b, n2 = nrm
        second = (f' Refitting on the <b>size-normalized</b> differences '
                  f'd<sub>i</sub> = |A<sub>Li</sub>&minus;A<sub>Ri</sub>|/(A<sub>Li</sub>+A<sub>Ri</sub>) '
                  f'gives &beta; = <b>{b2:.2f}</b> (R&sup2; = {r2b:.2f}, n = {n2}) &mdash; that is the '
                  f'number with the size effect removed, and the one worth interpreting.')
    return ('<div class="note"><b>Read this one skeptically.</b> Wang et al. (2018) fit '
            'log V<sub>D</sub> = log &alpha; + &beta; log M<sub>D</sub> across leaves; &beta; is a '
            f'<i>cohort</i> parameter, not a per-leaf score. On raw pixel&sup2; areas this cohort '
            f'gives &beta; = <b>{b:.2f}</b> (R&sup2; = {r2:.2f}, n = {n}) &mdash; but that is very '
            'nearly an arithmetic identity rather than a result. D<sub>i</sub> has units of area, so '
            'scaling a leaf up by k multiplies M<sub>D</sub> by k&sup2; and V<sub>D</sub> by '
            'k&#8308;, forcing &beta; &rarr; 2 for any cohort of similar shapes at different sizes. '
            f'Here log M<sub>D</sub> correlates r = <b>{rho:.2f}</b> with log lamina area, so the '
            'raw fit is largely measuring how big the leaves are.' + second + '</div>')


_CORR_NOTE = (
    '<div class="note"><b>How to use this.</b> Metrics inside a near-1 block are measuring the same '
    'thing; keeping all of them just triple-counts one signal in any downstream score. The families '
    'worth keeping are the ones that stay <i>uncorrelated</i>: a strip-based magnitude '
    '(SI<sub>A</sub> or WSI<sub>A</sub>), a whole-shape overlap (Dice), a worst-case term '
    '(Hausdorff), a positional term (r<sub>w</sub> or D<sub>c</sub>), and the axis geometry '
    '(sinuosity). AR is deliberately included to show the failure the 2018 paper warns about: it can '
    'sit near 1.0 while the local indices are large. Two blocks are redundant by construction and '
    'are kept only as a check that the matrix is behaving: Dice and IoU are a deterministic '
    'reparameterization of each other (IoU = Dice/(2&minus;Dice)), so they must show &rho; = 1.00; '
    'and WSI<sub>A</sub> tracks SI<sub>A</sub> at &rho; &asymp; 0.95 because they differ only in '
    'weighting. Sinuosity, by contrast, is nearly uncorrelated with every symmetry metric &mdash; it '
    'is independent information, which is exactly why it predicts the chord-vs-midvein gap above '
    'rather than the asymmetry itself.</div>')

_SCORE_NOTE = (
    '<div class="note"><b>Click any point</b> to see that leaf\'s oriented mask with its two halves '
    'tinted and both candidate axes drawn &mdash; every leaf in the cohort is clickable, not just '
    'the ones in the galleries below. That left-hand view is the fastest way to tell the three '
    'failure modes apart: a green midvein axis running down the middle with unequal halves is a '
    'genuinely asymmetric leaf; a green axis hugging one margin is a bad landmark trace; a ragged '
    'or multi-part silhouette is a bad mask.<br><br>'
    '<b>The three tiers.</b> <span style="color:var(--bad)">Vetoed</span> leaves had a score term '
    'driven to zero by a structural fault (extra components, holes, truncation, an unusable midvein '
    'trace) &mdash; these are the only hard rejects. '
    '<span style="color:var(--acc3)">Usable</span> is everything else. Ringed points are '
    '<b>exemplars</b>: usable AND above the score cut, which is a policy knob '
    '(<span class="mono">--min-score</span>), not a measurement. The structural gates apply at every '
    'threshold; only the score cut moves.<br><br>'
    '<b>Honest reading.</b> The archetype score deliberately combines symmetry '
    'with <i>independent</i> mask-quality evidence (stray-component fraction, truncation, convexity, '
    'landmark-trace confidence) as a weighted geometric mean, so any one bad term vetoes a leaf. '
    'Holes are excluded from both the gates and the score: the shape under test is the holes-filled '
    'silhouette, so an insect hole is not part of the outline being measured. The raw component '
    '<i>count</i> is excluded too &mdash; it vetoes single-pixel specks &mdash; and replaced by the '
    'fraction of area outside the main blob.<br><br>'
    'And the honest result: this scatter <b>is</b> a tight monotone band. On a cohort whose masks '
    'are already structurally clean, the other three terms saturate at 1.0 for most leaves, so the '
    'composite is very nearly the symmetry term alone. That is a fact about this data, not a '
    'validation of the score &mdash; the independent terms would only start separating leaves on a '
    'messier sweep. Treat the weights and thresholds as a first proposal to be tuned against the '
    'galleries below, not as established values.</div>')

_METHOD = """
<p class="body">Every leaf is measured in the <b>oriented</b> frame, so &ldquo;left&rdquo; and
&ldquo;right&rdquo; mean the same thing for every leaf in the cohort. The Reporter's
<span class="mono">Leaf_Oriented</span> masks are produced by a crop &rarr; rotate &rarr; content-fit
chain; the landmarks live in whole-sheet coordinates, so that chain is replayed exactly on the
keypoints. The rebuilt silhouette matches the saved PNG at <b>IoU 1.000000</b>, which is what
guarantees the midvein really lies on the mask it is measuring.</p>

<p class="body">Two axes are then built from tip to base:</p>
<div class="eq">chord    : the straight lamina_base &rarr; lamina_tip line &mdash; what Shi et al. and Wang et al. must use
midvein  : lamina_tip, midvein_0..14, lamina_base &mdash; smoothing-spline fitted, resampled to equal arclength</div>

<p class="body">Each axis induces a curvilinear coordinate system on the mask: <b>s</b> is normalized
arclength (0 at the tip, 1 at the base) and <b>u</b> is the signed perpendicular offset, positive to
the viewer's left. Equal-arclength strips drawn perpendicular to a <i>curved</i> axis would overlap
on the inside of a bend and leave gaps on the outside, so instead each lamina pixel is assigned to
its <b>nearest point on the axis</b>. That is a Voronoi partition of the polyline: it tiles the
lamina exactly once, with no overlap and no gaps, and the per-strip areas therefore sum back to the
total lamina area exactly &mdash; a property that is asserted in the tests.</p>

<p class="body">The lamina used is the <b>hole-filled silhouette</b>: the question is whether the
<i>outline</i> mirrors, and insect or decay holes are damage to a leaf that grew at its full size.
Hole area is measured separately and fed to the quality score instead. For the whole-shape overlap
measures the two halves are reflected <b>after</b> straightening into (s, u) &mdash; reflecting
across a single Euclidean line would count a curved midvein as asymmetry, which is precisely the
artifact this experiment is trying to isolate.</p>
"""

_LIMITATIONS = [
    "The midvein comes from an <b>alpha</b> pose model. Where the trace is wrong, the midvein axis "
    "is wrong, and the leaf will look asymmetric for a reason that has nothing to do with the leaf. "
    "Landmark confidence is carried into the score for exactly this reason, but a confidently wrong "
    "trace is not detectable here.",
    "&ldquo;Left&rdquo; is the viewer's left of the oriented mask. It is <b>not</b> a botanical "
    "anodic/cathodic call, so the directional-asymmetry literature (Chitwood et al. 2012; Martinez "
    "et al. 2016) cannot be tested with these signs &mdash; a cohort mean near zero is expected even "
    "if strong directional asymmetry exists, because the sheet orientation is arbitrary.",
    "A leaf can be genuinely, biologically asymmetric (oblique bases are normal in many taxa) and "
    "still have a perfect mask. Low symmetry is evidence of a messy mask only in combination with "
    "the independent quality diagnostics; the galleries are there to check that combination by eye.",
    "Fluctuating-asymmetry interpretation is deliberately avoided. M&aacute;jekov&aacute; et al. "
    "(2024) reviewed 51 studies and found leaf FA is not a reliable stress indicator, so nothing "
    "here should be read as an environmental signal.",
    "The cohort is small and taxonomically narrow. Thresholds tuned on it will not transfer "
    "unexamined to a broad herbarium sweep.",
]


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="bilateral_symmetry")
    p.add_argument("--run", default=DEFAULT_RUN, help="run directory containing <name>.sqlite")
    p.add_argument("--out", default=str(OUT_DIR / "bilateral_symmetry.html"))
    p.add_argument("--min-kpt-conf", type=float, default=0.25)
    p.add_argument("--bins", type=int, default=200, help="arclength strips per leaf")
    p.add_argument("--gallery-n", type=int, default=6)
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                   help="archetype score needed for the exemplar tier (gates always apply)")
    p.add_argument("--limit", type=int, default=None, help="stop after N leaves (smoke test)")
    a = p.parse_args(argv)

    run_dir = Path(a.run)
    dbs = sorted(run_dir.glob("*.sqlite"))
    if not dbs:
        p.error(f"no .sqlite in {run_dir}")
    db = ProjectDB.open_or_create(dbs[0])

    log.info("measuring leaves in %s", dbs[0])
    rows = collect(db, min_kpt_conf=a.min_kpt_conf, n_bins=a.bins, limit=a.limit,
                   min_score=a.min_score)
    log.info("%d leaves measured", len(rows))
    if not rows:
        log.error("no measurable leaves (need oriented masks + landmarks)")
        return 1

    ctx = build_context(rows, run_dir.name, gallery_n=a.gallery_n, db_path=str(dbs[0]))
    out = Path(a.out)
    out.write_text(render_report(ctx), encoding="utf-8")

    csv_path = out.with_suffix(".csv")
    keys = [k for k in rows[0] if not k.startswith("_") and k != "reasons"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})

    log.info("report -> %s  (%.1f MB)", out, out.stat().st_size / 1e6)
    log.info("csv    -> %s  (%d rows x %d cols)", csv_path, len(rows), len(keys))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
