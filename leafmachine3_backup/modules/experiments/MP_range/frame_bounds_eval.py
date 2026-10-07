"""MP_range step 5 -- the physical-plausibility bound on the sheet, replayed offline.

Production drops the MP anchor when it implies an impossibly SMALL sheet::

    implied_cm = frame_width_px / anchor          # sheet_cf.py
    if implied_cm < MIN_FRAME_CM:  anchor = None  # MIN_FRAME_CM = 20.0

Two problems with that as written:

  1. It divides the image WIDTH. On a portrait sheet width is the short side, which is what the
     20 cm floor is about -- but a scanned sheet that lands in landscape has width as the LONG
     side, so the same test is comparing ~43 cm against a 20 cm floor and can never fire. Using
     ``min(width, height)`` makes the test mean the same thing at any orientation.
  2. It is one-sided. Nothing rejects an anchor implying an impossibly LARGE sheet -- and that is
     the direction a harmonic error actually goes: a CF read 2x too small implies a sheet 2x too
     big. A standard plate is 29 x 41 cm; with a generous buffer the maximum probable image is
     45 x 60 cm, and nothing today enforces it.

This replays four bound configurations against the 549 human-measured sheets. NOTHING IS CHANGED:
``reconcile_parent``'s internal check is bypassed by passing ``frame_width_px=None``, and the drop
decision is made here instead, so each variant is exactly production-minus-that-one-rule.

Run::

    python -m leafmachine3.modules.experiments.MP_range.frame_bounds_eval
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from leafmachine3.inference.ruler_lattice.sheet_cf import MIN_FRAME_CM, reconcile_parent
from leafmachine3.inference.ruler_lattice.units import RUNG_HALF_LOG

from .dissent_eval import ANCHOR_TOL, TOL, ground_truth, load

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]

#: A standard herbarium plate is 29 x 41 cm; the buffer zone around the mounted sheet puts the
#: maximum probable IMAGE at 45 x 60 cm. Short side against 45, long side against 60.
MAX_SHORT_CM, MAX_LONG_CM = 45.0, 60.0


def implied(w: int, h: int, cf: float) -> tuple[float, float]:
    """(short side, long side) of the sheet in cm implied by ``cf``, orientation-independent."""
    return min(w, h) / cf, max(w, h) / cf


# --------------------------------------------------------------------------- #
# the bound configurations -- each returns True when the anchor should be DROPPED
# --------------------------------------------------------------------------- #
def b_production(w, h, cf):
    """Exactly what ships: image WIDTH only, 20 cm floor, no ceiling."""
    return (w / cf) < MIN_FRAME_CM


def b_none(w, h, cf):
    """Case 1 -- no minimum, no maximum. The anchor is never dropped on size."""
    return False


def b_max_only(w, h, cf):
    """Case 2 -- no minimum, maximum only (45 x 60 cm)."""
    short, long = implied(w, h, cf)
    return short > MAX_SHORT_CM or long > MAX_LONG_CM


def b_min_only(w, h, cf):
    """Case 3 -- keep the 20 cm minimum, on the SHORT side, no maximum."""
    return implied(w, h, cf)[0] < MIN_FRAME_CM


def b_min_and_max(w, h, cf):
    """Case 4 -- both bounds (the combination the other three bracket)."""
    return b_min_only(w, h, cf) or b_max_only(w, h, cf)


CASES = {
    "production (width, min 20)": b_production,
    "1. min-dim, NO bounds": b_none,
    "2. min-dim, MAX only (45x60)": b_max_only,
    "3. min-dim, MIN only (20)": b_min_only,
    "4. min-dim, MIN + MAX": b_min_and_max,
}


def publish(crops, anchor, dropped) -> tuple[float | None, float | None]:
    """(published CF, measured CF) with the size rule already applied by the caller.

    ``frame_width_px=None`` disables reconcile_parent's own plausibility check, so the only size
    rule in force is the one under test.
    """
    pr = reconcile_parent(crops, anchor=(None if dropped else anchor),
                          anchor_tol=ANCHOR_TOL, frame_width_px=None)
    return pr.get("cf_px_per_cm"), pr.get("cf_px_per_cm_measured")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay sheet-size plausibility bounds offline.")
    ap.add_argument("--out", type=Path, default=REPO / "examples_out")
    ap.add_argument("--run", default="mp_anchor_linear")
    args = ap.parse_args(argv)

    gt = ground_truth()
    db = args.out / args.run / f"{args.run}.sqlite"
    sheets = {k: v for k, v in load(db, "original").items() if k in gt}

    # working dims, needed for the orientation-independent test
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    dims = {r["image_stem"]: (r["width"], r["height"])
            for r in con.execute("SELECT image_stem, width, height FROM specimen")}
    con.close()

    land = sum(1 for w, h in dims.values() if w > h)
    print(f"  {len(sheets)} sheets   |   landscape (width > height): {land}\n")

    print(f"  {'case':30s} {'dropped':>8s} {'publish':>8s} {'correct':>8s} {'BAD':>4s} "
          f"{'missed good':>12s} {'net vs prod':>14s}")
    base = None
    for name, rule in CASES.items():
        dropped = pub = ok = bad = missed = 0
        for stem, sh in sheets.items():
            w, h = dims[stem]
            a = sh["anchor"]
            d = bool(a) and rule(w, h, a)
            dropped += d
            cf, meas = publish(sh["crops"], a, d)
            g = gt[stem] * sh["work_scale"]
            if cf is not None:
                pub += 1
                ok += int(abs(cf / g - 1) <= TOL)
                bad += int(abs(cf / g - 1) > TOL)
            elif meas is not None and abs(meas / g - 1) <= TOL:
                missed += 1
        net = "" if base is None else f"{ok-base[0]:+d} good / {bad-base[1]:+d} bad"
        if base is None:
            base = (ok, bad)
        print(f"  {name:30s} {dropped:8d} {pub:8d} {ok:8d} {bad:4d} {missed:12d} {net:>14s}")

    # -- what the bounds would reject if applied to the MEASURED CF too ------------------
    print("\n  If the same bounds were applied to the MEASURED CF (they are not today):")
    for label, rule in (("min 20 (short side)", b_min_only), ("max 45x60", b_max_only),
                        ("both", b_min_and_max)):
        hit_good = hit_bad = 0
        for stem, sh in sheets.items():
            w, h = dims[stem]
            _, meas = publish(sh["crops"], sh["anchor"], False)
            if meas is None or not rule(w, h, meas):
                continue
            g = gt[stem] * sh["work_scale"]
            if abs(meas / g - 1) <= TOL:
                hit_good += 1
            else:
                hit_bad += 1
        print(f"    {label:22s} would reject {hit_good + hit_bad:3d} readings: "
              f"{hit_bad} genuinely BAD, {hit_good} actually good")

    # -- the observed envelope, from the human measurements ------------------------------
    shorts, longs = [], []
    for stem, sh in sheets.items():
        w, h = dims[stem]
        s, l = implied(w, h, gt[stem] * sh["work_scale"])
        shorts.append(s), longs.append(l)
    shorts.sort(), longs.sort()
    print(f"\n  observed sheet envelope from the human CFs (working frame, orientation-independent):")
    print(f"    short side  {shorts[0]:5.1f} .. {shorts[-1]:5.1f} cm      "
          f"long side  {longs[0]:5.1f} .. {longs[-1]:5.1f} cm")
    print(f"    headroom to the proposed ceiling: short {MAX_SHORT_CM - shorts[-1]:.1f} cm, "
          f"long {MAX_LONG_CM - longs[-1]:.1f} cm")
    print(f"    headroom to the 20 cm floor:      short {shorts[0] - MIN_FRAME_CM:.1f} cm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
