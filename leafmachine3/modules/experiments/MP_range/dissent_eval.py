"""MP_range step 4 -- can the dissent veto be loosened without letting bad CFs through?

The gate audit found that of 26 sheets the lattice withheld, **22 had a measurement within 10% of
the human ground truth**, and **19 of those were vetoed purely by a dissenting crop while the MP
anchor agreed with the winner**. The rule is currently unconditional::

    dissent = [every live non-winning crop with weight >= DISAGREEMENT_FLOOR]
    if dissent: confidence = "low"          # -> the sheet publishes nothing

So a sheet with three rulers where two agree, the anchor confirms them, and the third is misread
loses its CF entirely. This module replays alternative rules against the STORED lattice records and
scores each one against the human measurements.

NOTHING HERE TOUCHES PRODUCTION. ``reconcile_parent`` is a pure function of the per-crop rows plus
the anchor, all of which are in the database, so every variant is evaluated offline. The baseline
variant is checked against the decisions production actually stored before any variant is trusted --
if the replay cannot reproduce the shipped run, its verdicts about the others mean nothing.

Run (after mp_anchor_linear / mp_anchor_sqrt exist)::

    python -m leafmachine3.modules.experiments.MP_range.dissent_eval
"""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from pathlib import Path

from leafmachine3.inference.ruler_lattice.sheet_cf import (
    DISAGREEMENT_FLOOR, MIN_FRAME_CM, PEER_TOL, _harmonic_of, reconcile_parent,
)
from leafmachine3.inference.ruler_lattice.units import RUNG_HALF_LOG

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIT_DATA = REPO / "models" / "mp_conversion_factor" / "fit_data.csv"
TOL = 0.10                     # a published CF further than this from the human value is a bad publish
ANCHOR_TOL = 0.25              # matches modules.ruler_cf.anchor_tol in the shipped settings


# --------------------------------------------------------------------------- #
# the variants
# --------------------------------------------------------------------------- #
def v_baseline(win_cf, anchor, by_anchor, dissenters, win_weight):
    """Production: ANY well-supported non-winning crop vetoes the sheet."""
    return bool(dissenters)


def v_anchor_overrules(win_cf, anchor, by_anchor, dissenters, win_weight):
    """A dissenter cannot veto a winner the ANCHOR already confirms.

    The module's own thesis is "the anchor remains the absolute reference". If the winning CF sits
    within half a unit-ladder rung of a plausible anchor, an independent measurement says it is
    right; a sibling crop that disagrees is then the odd one out, not evidence against it.
    """
    return bool(dissenters) and not by_anchor


def v_harmonic_ignored(win_cf, anchor, by_anchor, dissenters, win_weight):
    """A dissenter sitting on a clean 2x / 5x / 10x of the winner is a KNOWN misnaming, not a peer.

    That is the single failure mode the unit ladder exists to model: the crop read the right ticks
    and named the wrong unit. Treating it as an unresolvable second opinion throws away the sheet
    for the one kind of disagreement the engine can actually explain.
    """
    real = [d for d in dissenters if _harmonic_of(d["cf"], win_cf) is None]
    return bool(real)


def v_dissent_must_be_plausible(win_cf, anchor, by_anchor, dissenters, win_weight):
    """Only a dissenter that is ITSELF anchor-admissible counts as a second opinion.

    A crop that disagrees with the winner AND with the anchor is simply a bad reading. Requiring a
    dissenter to be credible before it can veto is the difference between "two plausible readings
    conflict" and "one reading is broken".
    """
    if not anchor:
        return bool(dissenters)
    real = [d for d in dissenters if abs(d["cf"] / anchor - 1.0) <= ANCHOR_TOL]
    return bool(real)


def v_dissent_must_outweigh(win_cf, anchor, by_anchor, dissenters, win_weight):
    """Only a dissenter carrying at least as much tick evidence as the whole winning cluster."""
    return any((d.get("weight") or 0) >= win_weight for d in dissenters)


def v_anchor_or_harmonic(win_cf, anchor, by_anchor, dissenters, win_weight):
    """Anchor-confirmed winners are safe, AND harmonic dissenters never count."""
    if by_anchor:
        return False
    return bool([d for d in dissenters if _harmonic_of(d["cf"], win_cf) is None])


def v_anchor_and_plausible(win_cf, anchor, by_anchor, dissenters, win_weight):
    """Anchor-confirmed winners are safe, AND only anchor-admissible dissenters count otherwise."""
    if by_anchor:
        return False
    if not anchor:
        return bool(dissenters)
    return bool([d for d in dissenters if abs(d["cf"] / anchor - 1.0) <= ANCHOR_TOL])


VARIANTS = {
    "baseline (production)": v_baseline,
    "anchor overrules dissent": v_anchor_overrules,
    "harmonic dissent ignored": v_harmonic_ignored,
    "dissenter must be plausible": v_dissent_must_be_plausible,
    "dissenter must outweigh winner": v_dissent_must_outweigh,
    "anchor overrules + harmonic": v_anchor_or_harmonic,
    "anchor overrules + plausible": v_anchor_and_plausible,
}


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #
def reconcile(sheet) -> dict:
    """Production's clustering + ranking for one sheet. Cached per sheet: it does not depend on the
    dissent rule, so every variant judges exactly the same winner."""
    return reconcile_parent(sheet["crops"], anchor=sheet["anchor"], anchor_tol=ANCHOR_TOL,
                            frame_width_px=sheet["frame_width_px"])


def decide(pr: dict, crops: list, veto) -> float | None:
    """The published CF under one dissent rule, or None if the sheet is withheld.

    Only the confidence step is re-derived; the winner comes from production's own
    ``reconcile_parent`` above, so a variant differs from production in the dissent rule and in
    nothing else. Re-implementing the clustering here would let the harness drift from the code it
    is supposed to be reasoning about.
    """
    cf = pr.get("cf_px_per_cm_measured")
    if cf is None:
        return None
    used_anchor = pr.get("anchor")
    anchor_log = pr.get("anchor_log_dist")
    by_anchor = bool(anchor_log is not None and anchor_log <= RUNG_HALF_LOG)
    by_anchor_weak = bool(anchor_log is not None and not by_anchor
                          and anchor_log <= math.log1p(ANCHOR_TOL))
    by_peers = bool(pr.get("corroborated_by_peers"))

    # reconcile_parent labels winning crops "used" (not "kept" -- "kept" is the per-TICK counter).
    # Getting this wrong empties the winner set, makes every crop look like a dissenter, and the
    # baseline replay silently withholds everything; the fidelity check above is what caught it.
    win_keys = {c["key"] for c in pr.get("per_crop", []) if c.get("verdict") == "used"}
    live = [c for c in crops if not c.get("skipped") and c.get("cf")]
    winners = [c for c in live if c["key"] in win_keys] or live[:1]
    win_weight = sum(max(1, c.get("weight") or 1) for c in winners)
    dissenters = [c for c in live
                  if c["key"] not in win_keys and (c.get("weight") or 0) >= DISAGREEMENT_FLOOR]

    if veto(cf, used_anchor, by_anchor, dissenters, win_weight):
        return None
    if by_anchor or (used_anchor is None and by_peers):
        return cf
    return None                     # "medium" and "low" both publish nothing


def load(db_path: Path, anchor_frame: str) -> dict[str, dict]:
    """Per-sheet crops + the working-frame anchor, straight out of a completed run."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sheets: dict[str, dict] = {}
    for r in con.execute("""
        SELECT s.specimen_id, s.image_stem, s.width, s.work_scale,
               s.cf_px_per_cm_predicted_by_mp AS anchor_stored, i.status AS stored_status
        FROM specimen s JOIN ruler_CF_lattice i USING(specimen_id)"""):
        a = r["anchor_stored"]
        ws = r["work_scale"] or 1.0
        sheets[r["image_stem"]] = {
            "specimen_id": r["specimen_id"],
            "anchor": (a if anchor_frame == "working" else (a * ws if a else None)),
            "frame_width_px": r["width"], "work_scale": ws,
            "stored_status": r["stored_status"], "crops": [],
        }
    by_id = {v["specimen_id"]: v for v in sheets.values()}
    for r in con.execute("""SELECT specimen_id, detection_id, pxcm, n_kept, ruler_class, status,
                                   status_reason FROM ruler_CF_lattice_crop"""):
        s = by_id.get(r["specimen_id"])
        if s is not None:
            s["crops"].append({"key": f"det{r['detection_id']}", "cf": r["pxcm"],
                               "weight": r["n_kept"] or 0, "ruler_class": r["ruler_class"],
                               "skipped": (r["status"] != "measured"),
                               "skip_reason": r["status_reason"]})
    con.close()
    return sheets


def ground_truth() -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    with FIT_DATA.open() as fh:
        for r in csv.DictReader(fh):
            acc.setdefault(r["filename"], []).append(float(r["cf"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay dissent-veto variants offline.")
    ap.add_argument("--out", type=Path, default=REPO / "examples_out")
    args = ap.parse_args(argv)

    gt = ground_truth()
    for run, frame in (("mp_anchor_linear", "original"), ("mp_anchor_sqrt", "working")):
        db = args.out / run / f"{run}.sqlite"
        if not db.exists():
            print(f"  skipping {run} (no database)")
            continue
        sheets = {k: v for k, v in load(db, frame).items() if k in gt}
        print(f"\n{'='*104}\n  {run}   ({len(sheets)} sheets with human ground truth)\n{'='*104}")

        # -- faithfulness check: the baseline replay must reproduce the shipped decisions --
        pr = {stem: reconcile(sh) for stem, sh in sheets.items()}
        mism = [stem for stem, sh in sheets.items()
                if (decide(pr[stem], sh["crops"], v_baseline) is not None)
                != (sh["stored_status"] == "published")]
        print(f"  replay fidelity: baseline reproduces {len(sheets)-len(mism)}/{len(sheets)} stored "
              f"decisions" + ("" if not mism else f"   <-- {len(mism)} MISMATCH, variants are suspect"))

        print(f"\n  {'variant':32s} {'publish':>8s} {'correct':>8s} {'BAD':>5s} {'missed good':>12s} "
              f"{'net vs baseline':>16s}")
        base = None
        for name, veto in VARIANTS.items():
            pub = ok = bad = missed = 0
            for stem, sh in sheets.items():
                cf = decide(pr[stem], sh["crops"], veto)
                g = gt[stem] * sh["work_scale"]                    # truth in the working frame
                m = pr[stem].get("cf_px_per_cm_measured")
                if cf is not None:
                    pub += 1
                    ok += int(abs(cf / g - 1) <= TOL)
                    bad += int(abs(cf / g - 1) > TOL)
                elif m is not None and abs(m / g - 1) <= TOL:
                    missed += 1
            net = "" if base is None else f"{ok-base[0]:+d} good / {bad-base[1]:+d} bad"
            if base is None:
                base = (ok, bad)
            print(f"  {name:32s} {pub:8d} {ok:8d} {bad:5d} {missed:12d} {net:>16s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
