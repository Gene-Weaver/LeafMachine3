"""MP_range step 3 -- run the ruler-CF gate under both anchors and audit every decision.

The 708 rows of ``models/mp_conversion_factor/fit_data.csv`` are 549 unique sheets with HUMAN ruler
measurements (``ruler_manual.csv`` ``cm_1_avg``), so they are independent ground truth for the
lattice -- not lattice output fed back to itself. That makes them the right set to ask the question
this experiment exists for: if the MP anchor changes form, do the gate's publish/withhold decisions
still make sense, and are the CFs it publishes closer to or further from the truth?

Compares two completed runs that differ ONLY in the anchor:

  linear   cf = 2.674*MP + 67.52   evaluated on the ORIGINAL dims, then scaled by work_scale
  sqrt     cf = 27.112*sqrt(MP)    evaluated on the WORKING dims, already in the working frame

Ground truth is in the original frame, so it is compared against ``cf_px_per_cm_original``.

A NOTE ON THE STANDARD BEING APPLIED
    153 of the 549 sheets were measured twice by hand, and those repeats disagree by 1.29% on
    average (p95 3.84%, worst 28.6%). No model can be scored tighter than that, so "correct" here
    means within a tolerance well above the human noise floor, and a disagreement of a few percent
    is not evidence against either anchor.

Run::

    python -m leafmachine3.modules.experiments.MP_range.gate_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIT_DATA = REPO / "models" / "mp_conversion_factor" / "fit_data.csv"

#: A published CF further than this from the human measurement is a bad publish. Chosen well above
#: the 1.29% human repeat-measurement noise so the verdict is about the model, not the labeler.
TOL = 0.10


def ground_truth() -> dict[str, float]:
    """``{image_stem: mean human px/cm}`` in the ORIGINAL frame; repeats averaged."""
    acc: dict[str, list[float]] = {}
    with FIT_DATA.open() as fh:
        for r in csv.DictReader(fh):
            acc.setdefault(r["filename"], []).append(float(r["cf"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def load_run(db_path: Path) -> dict[str, dict]:
    """``{image_stem: row}`` joining the specimen row to its lattice record."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = {}
    for r in con.execute("""
        SELECT s.image_stem, s.width, s.height, s.original_width, s.original_height,
               s.work_scale, s.original_mp, s.cf_px_per_cm_predicted_by_mp AS anchor_stored,
               s.cf_px_per_cm AS published_work,
               i.cf_px_per_cm_original AS published_orig, i.cf_px_per_cm_measured AS measured_work,
               i.status, i.confidence, i.anchor_supported, i.n_dissenting, i.pct_vs_anchor
        FROM specimen s LEFT JOIN ruler_CF_lattice i USING(specimen_id)"""):
        rows[r["image_stem"]] = dict(r)
    con.close()
    return rows


def audit(gt: dict[str, float], run: dict[str, dict], name: str) -> dict:
    """Per-sheet verdicts for one run."""
    out = {}
    for stem, g in gt.items():
        r = run.get(stem)
        if r is None:
            continue
        ws = r["work_scale"] or 1.0
        gt_work = g * ws                                  # truth, moved into the frame the gate sees
        anchor = r["anchor_stored"]
        # the anchor as the ENGINE sees it: the sqrt run stores a working-frame value already
        anchor_work = (anchor if name == "sqrt" else (anchor * ws if anchor else None))
        meas = r["measured_work"]
        pub = r["published_orig"]
        out[stem] = {
            "gt": g, "gt_work": gt_work, "work_scale": ws, "mp": r["original_mp"],
            "anchor_work": anchor_work,
            "anchor_err": (anchor_work / gt_work - 1) if (anchor_work and gt_work) else None,
            "measured_err": (meas / gt_work - 1) if (meas and gt_work) else None,
            "published_err": (pub / g - 1) if (pub and g) else None,
            "status": r["status"] or "no_ruler",
            "confidence": r["confidence"],
            "published": pub is not None,
        }
    return out


def _stats(vals) -> str:
    a = np.abs(np.array([v for v in vals if v is not None], float))
    if not a.size:
        return "      n/a"
    return f"n={a.size:4d} mean {100*a.mean():5.1f}% med {100*np.median(a):5.1f}% p95 {100*np.percentile(a,95):5.1f}%"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Audit the ruler-CF gate under both MP anchors.")
    ap.add_argument("--out", type=Path, default=REPO / "examples_out")
    ap.add_argument("--linear-run", default="mp_anchor_linear")
    ap.add_argument("--sqrt-run", default="mp_anchor_sqrt")
    args = ap.parse_args(argv)

    gt = ground_truth()
    runs = {}
    for name, run in (("linear", args.linear_run), ("sqrt", args.sqrt_run)):
        db = args.out / run / f"{run}.sqlite"
        if not db.exists():
            raise SystemExit(f"missing run database: {db}")
        runs[name] = audit(gt, load_run(db), name)

    shared = sorted(set(runs["linear"]) & set(runs["sqrt"]))
    print(f"  {len(shared)} sheets with human ground truth in both runs\n")

    # -- 1. how good is the ANCHOR itself -------------------------------------------
    print("  ANCHOR vs human truth (working frame -- what the gate actually compares against)")
    for name in ("linear", "sqrt"):
        print(f"    {name:7s} {_stats([runs[name][s]['anchor_err'] for s in shared])}")

    # -- 2. what the lattice MEASURED (identical in both runs; the anchor only judges it) --
    print("\n  LATTICE MEASUREMENT vs human truth (before the gate)")
    print(f"    {'':7s} {_stats([runs['linear'][s]['measured_err'] for s in shared])}")

    # -- 3. gate decisions -----------------------------------------------------------
    print("\n  GATE DECISIONS")
    for name in ("linear", "sqrt"):
        c = Counter(runs[name][s]["status"] for s in shared)
        pub = sum(1 for s in shared if runs[name][s]["published"])
        print(f"    {name:7s} published {pub:4d}/{len(shared)} ({100*pub/len(shared):4.1f}%)   "
              f"{dict(sorted(c.items()))}")

    # -- 4. accuracy of what each run SHIPS ------------------------------------------
    print("\n  PUBLISHED CF vs human truth (original frame -- the number downstream uses)")
    for name in ("linear", "sqrt"):
        errs = [runs[name][s]["published_err"] for s in shared if runs[name][s]["published"]]
        bad = sum(1 for e in errs if e is not None and abs(e) > TOL)
        print(f"    {name:7s} {_stats(errs)}   worse than {TOL:.0%}: {bad}")

    # -- 5. the decisions that CHANGED ------------------------------------------------
    print("\n  DECISION CHANGES (linear -> sqrt)")
    changed = [s for s in shared if runs["linear"][s]["published"] != runs["sqrt"][s]["published"]]
    gained = [s for s in changed if runs["sqrt"][s]["published"]]
    lost = [s for s in changed if runs["linear"][s]["published"]]
    print(f"    unchanged {len(shared)-len(changed):4d}   newly published {len(gained):3d}   "
          f"newly withheld {len(lost):3d}")

    def verdict(rows, stems, label):
        if not stems:
            return
        print(f"\n    {label} ({len(stems)}):")
        print(f"      {'sheet':44s} {'MP':>6s} {'meas err':>9s} {'anchor err L':>13s} {'anchor err S':>13s}")
        good = 0
        for s in sorted(stems, key=lambda x: abs(rows[x]["measured_err"] or 9))[:25]:
            m = runs["linear"][s]["measured_err"]
            al, asq = runs["linear"][s]["anchor_err"], runs["sqrt"][s]["anchor_err"]
            good += int(m is not None and abs(m) <= TOL)
            print(f"      {s[:44]:44s} {runs['linear'][s]['mp'] or 0:6.1f} "
                  f"{100*(m or 0):8.1f}% {100*(al or 0):12.1f}% {100*(asq or 0):12.1f}%")
        allgood = sum(1 for s in stems
                      if runs["linear"][s]["measured_err"] is not None
                      and abs(runs["linear"][s]["measured_err"]) <= TOL)
        print(f"      -> {allgood}/{len(stems)} of these had a measurement within {TOL:.0%} of truth")

    verdict(runs["sqrt"], gained, "NEWLY PUBLISHED under sqrt")
    verdict(runs["linear"], lost, "NEWLY WITHHELD under sqrt")

    # -- 6. the bottom line: correct publishes vs bad publishes -----------------------
    print("\n  BOTTOM LINE")
    print(f"    {'':7s} {'published':>10s} {'within tol':>11s} {'BAD publish':>12s} {'missed good':>12s}")
    for name in ("linear", "sqrt"):
        pub = [s for s in shared if runs[name][s]["published"]]
        okp = sum(1 for s in pub if abs(runs[name][s]["published_err"] or 9) <= TOL)
        badp = len(pub) - okp
        missed = sum(1 for s in shared
                     if not runs[name][s]["published"]
                     and runs[name][s]["measured_err"] is not None
                     and abs(runs[name][s]["measured_err"]) <= TOL)
        print(f"    {name:7s} {len(pub):10d} {okp:11d} {badp:12d} {missed:12d}")

    dump = args.out / "mp_anchor_gate_eval.csv"
    with dump.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stem", "mp", "work_scale", "gt_cf_original", "measured_err",
                     "anchor_err_linear", "anchor_err_sqrt", "status_linear", "status_sqrt",
                     "published_linear", "published_sqrt", "published_err_linear", "published_err_sqrt"])
        for s in shared:
            L, S = runs["linear"][s], runs["sqrt"][s]
            wr.writerow([s, L["mp"], L["work_scale"], round(L["gt"], 3),
                         L["measured_err"], L["anchor_err"], S["anchor_err"],
                         L["status"], S["status"], L["published"], S["published"],
                         L["published_err"], S["published_err"]])
    print(f"\n  wrote {dump}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
