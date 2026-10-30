"""How close are two LM3 runs of the same images that are NOT expected to be identical?

For runs that should match exactly (same code path, same provider) use compare_run_databases.py.
This one is for runs that differ by float rounding -- CoreML or CUDA vs the CPU -- and asks how much.
It reads each run's reports/Data CSVs, matches detections and measured leaves between the runs by box
IoU (same image, same class), and reports:

* detections: how many of A's boxes B also found (all, and conf >= 0.4), matched-box IoU, score change;
* leaves: how many of A's measured leaves B also measured, and per measurement the relative change
  (median / 95th percentile / max) over matched leaves;
* sheets: per-sheet counts, conversion factor (px/cm), ruler unit type, specimen-mask area.

Usage:
    python tools/verification/compare_runs_close.py RUN_A RUN_B [--label TEXT] [--json PATH]

RUN_A / RUN_B are run folders (runs/<name>) or their reports/Data folders. A is the reference.
Report only: the exit status is 0 unless a run cannot be read.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MATCH_IOU = 0.5
CONFIDENT = 0.4
LEAF_METRICS = [
    "lamina_area_incl_holes_px", "lamina_perimeter_px", "convexity", "circularity", "aspect_ratio",
    "rotated_bbox_length_px", "rotated_bbox_width_px", "lamina_trace_length_px", "lamina_tip_base_length_px",
    "leaf_width_px", "petiole_trace_length_px", "petiole_width_px", "apex_angle", "base_angle",
    "lamina_area_incl_holes_cm2",
]
SHEET_COUNTS = ["n_archival_detections", "n_plant_detections", "n_leaf_instances", "n_leaves_measured",
                "n_leaves_with_landmarks", "n_leaves_with_petiole", "n_leaves_archetypal", "n_ruler_crops"]


def data_dir(p: str) -> Path:
    p = Path(p)
    for cand in (p, p / "reports" / "Data"):
        if (cand / "specimen_summary.csv").exists():
            return cand
    sys.exit(f"no reports/Data/specimen_summary.csv under {p}")


def read(d: Path, name: str) -> list[dict]:
    path = d / f"{name}.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def greedy(ra: list, rb: list, box, group, order) -> list[tuple[int, int, float]]:
    """Match rows of A to rows of B in the same group by box IoU, A's ``order`` first."""
    by_group = defaultdict(list)
    for j, r in enumerate(rb):
        by_group[group(r)].append(j)
    used, pairs = set(), []
    for i in sorted(range(len(ra)), key=lambda i: order(ra[i])):
        best, best_j = MATCH_IOU, None
        for j in by_group.get(group(ra[i]), []):
            if j not in used:
                v = iou(box(ra[i]), box(rb[j]))
                if v >= best:
                    best, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            pairs.append((i, best_j, best))
    return pairs


def stats(values) -> dict:
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max()), "min": float(v.min())}


def compare_detections(a: list, b: list) -> dict:
    box = lambda r: (num(r["x1"]), num(r["y1"]), num(r["x2"]), num(r["y2"]))       # noqa: E731
    out = {}
    for source in sorted({r["source"] for r in a} | {r["source"] for r in b}):
        ra = [r for r in a if r["source"] == source]
        rb = [r for r in b if r["source"] == source]
        pairs = greedy(ra, rb, box, lambda r: (r["image_stem"], r["cls_name"]), lambda r: -num(r["conf"]))
        conf_a = [i for i, r in enumerate(ra) if num(r["conf"]) >= CONFIDENT]
        found = {i for i, _, _ in pairs}
        out[source] = {
            "n_a": len(ra), "n_b": len(rb), "matched": len(pairs),
            "confident_a": len(conf_a), "confident_matched": sum(1 for i in conf_a if i in found),
            "iou": stats([p[2] for p in pairs]),
            "dconf": stats([abs(num(ra[i]["conf"]) - num(rb[j]["conf"])) for i, j, _ in pairs]),
        }
    return out


def compare_leaves(a: list, b: list) -> dict:
    box = lambda r: (num(r["crop_x1"]), num(r["crop_y1"]), num(r["crop_x2"]), num(r["crop_y2"]))   # noqa: E731
    pairs = greedy(a, b, box, lambda r: (r["image_stem"], r.get("instance_index", "0")),
                   lambda r: -num(r.get("lamina_area_incl_holes_px")))
    metrics = {}
    for m in LEAF_METRICS:
        if not a or m not in a[0]:
            continue
        rel = []
        for i, j, _ in pairs:
            va, vb = num(a[i][m]), num(b[j][m])
            if np.isfinite(va) and np.isfinite(vb):
                rel.append(abs(vb - va) / max(abs(va), 1e-9))
        s = stats(rel)
        if s["n"]:
            s["within_1pct"] = float(np.mean(np.asarray(rel) <= 0.01))
        metrics[m] = s
    return {"n_a": len(a), "n_b": len(b), "matched": len(pairs), "metrics": metrics}


def compare_sheets(a: list, b: list) -> dict:
    bmap = {r["image_stem"]: r for r in b}
    common = [r for r in a if r["image_stem"] in bmap]
    out = {"images_a": len(a), "images_b": len(b), "common": len(common), "counts": {}, "differs": defaultdict(list)}
    for c in SHEET_COUNTS:
        same = sum(1 for r in common if r.get(c) == bmap[r["image_stem"]].get(c))
        out["counts"][c] = same
        out["differs"][c] = [f"{r['image_stem']}: {r.get(c)} -> {bmap[r['image_stem']].get(c)}"
                             for r in common if r.get(c) != bmap[r["image_stem"]].get(c)]
    cf = [abs(num(bmap[r["image_stem"]]["cf_px_per_cm"]) - num(r["cf_px_per_cm"])) / num(r["cf_px_per_cm"])
          for r in common if num(r.get("cf_px_per_cm")) > 0]
    out["cf_rel"] = stats(cf)
    out["cf_within_0_5pct"] = int(sum(1 for v in cf if v <= 0.005))
    out["ruler_unit_same"] = sum(1 for r in common if r.get("ruler_unit_type") == bmap[r["image_stem"]].get("ruler_unit_type"))
    out["differs"]["ruler_unit_type"] = [f"{r['image_stem']}: {r.get('ruler_unit_type')} -> {bmap[r['image_stem']].get('ruler_unit_type')}"
                                         for r in common if r.get("ruler_unit_type") != bmap[r["image_stem"]].get("ruler_unit_type")]
    out["mask_area_frac_abs"] = stats([abs(num(bmap[r["image_stem"]]["specimen_mask_area_frac"]) - num(r["specimen_mask_area_frac"]))
                                       for r in common])
    out["differs"] = {k: v for k, v in out["differs"].items() if v}
    return out


def _f(s: dict, k: str, fmt: str = "{:.3f}") -> str:
    return fmt.format(s[k]) if s.get("n") else "-"


def render(label: str, rep: dict) -> str:
    lines = [f"== {label}"]
    for source, d in rep["detections"].items():
        lines.append(f"detections {source:9s} A {d['n_a']:5d}  B {d['n_b']:5d}  matched {d['matched']}/{d['n_a']} "
                     f"(conf>={CONFIDENT}: {d['confident_matched']}/{d['confident_a']})  IoU median {_f(d['iou'], 'median')} "
                     f"min {_f(d['iou'], 'min')}  |dconf| median {_f(d['dconf'], 'median')} max {_f(d['dconf'], 'max')}")
    lv = rep["leaves"]
    lines.append(f"leaves     measured A {lv['n_a']}  B {lv['n_b']}  matched {lv['matched']}/{lv['n_a']}")
    lines.append(f"  {'metric':28s} {'n':>5s} {'median':>9s} {'p95':>9s} {'max':>9s} {'<=1%':>6s}   (relative change B vs A)")
    for m, s in lv["metrics"].items():
        lines.append(f"  {m:28s} {s['n']:5d} {_f(s, 'median', '{:9.4f}')} {_f(s, 'p95', '{:9.4f}')} {_f(s, 'max', '{:9.4f}')} "
                     f"{s.get('within_1pct', float('nan')) * 100:5.1f}%")
    sh = rep["sheets"]
    lines.append(f"sheets     {sh['common']} common (A {sh['images_a']}, B {sh['images_b']})")
    lines.append("  same count on: " + ", ".join(f"{c.replace('n_', '')} {v}/{sh['common']}" for c, v in sh["counts"].items()))
    lines.append(f"  cf_px_per_cm within 0.5% on {sh['cf_within_0_5pct']}/{sh['cf_rel'].get('n', 0)} "
                 f"(rel change median {_f(sh['cf_rel'], 'median', '{:.4f}')}, max {_f(sh['cf_rel'], 'max', '{:.4f}')}); "
                 f"ruler unit same {sh['ruler_unit_same']}/{sh['common']}; specimen mask area |diff| max "
                 f"{_f(sh['mask_area_frac_abs'], 'max', '{:.4f}')}")
    for k, v in sh["differs"].items():
        lines.append(f"  {k} differs on {len(v)}: " + "; ".join(v[:6]) + (" ..." if len(v) > 6 else ""))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", help="reference run folder (or its reports/Data)")
    ap.add_argument("b", help="run to compare against it")
    ap.add_argument("--label", default=None)
    ap.add_argument("--json", default=None, help="also write the full report here")
    args = ap.parse_args(argv)
    da, db = data_dir(args.a), data_dir(args.b)
    rep = {
        "a": str(da), "b": str(db),
        "detections": compare_detections(read(da, "detections"), read(db, "detections")),
        "leaves": compare_leaves(read(da, "leaf_measurements"), read(db, "leaf_measurements")),
        "sheets": compare_sheets(read(da, "specimen_summary"), read(db, "specimen_summary")),
    }
    print(render(args.label or f"{Path(args.a).name} (A) vs {Path(args.b).name} (B)", rep))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
