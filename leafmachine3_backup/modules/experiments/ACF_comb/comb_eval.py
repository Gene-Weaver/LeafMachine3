"""ACF_comb -- recover the tick fundamental from the ACF's PEAK COMB, not its argmax.

WHY

``A_1989713254`` is a plain metric ruler: 1 mm ticks with longer 1 cm ticks. The engine read it at
94.82 px/cm when the truth is 59.26 -- a 60% error that the old MP anchor happened to endorse. The
cause is not the naming step and not the tick detector. It is this:

    lattice.MIN_PERIOD = 7.0        # the ACF search window starts at lag 7 px

At 59.26 px/cm one millimetre is **5.93 px**, which is BELOW that floor. The fundamental is not
merely missed, it is outside the search window and cannot be returned by any amount of sub-harmonic
descent. What the ACF does show is a comb of peaks at

    12, 17, 23, 29, 35, 40, 46, 52, 58, 63, 69, 75, ...

every one of them a harmonic k*5.93 for k = 2, 3, 4, 5, ... They are uniformly spaced by ~5.9 px, so
**the fundamental appears as the GAP between peaks, never as a peak**. The argmax (17 px) is simply
the tallest harmonic, and naming it afterwards is hopeless: 2.87 mm is not a printed graduation, so
it rounds to 2.5 mm and the error is locked in.

THE METHOD

Two ways to use the whole peak set instead of one peak, scored side by side here:

``ladder``  Walk the peaks from the finest upward. Take peak i as the candidate finest graduation
            and require the REST of the peaks to be integer multiples of it -- a real comb. The
            first peak that passes wins. This is the "first non-noisy peak, then check the combs"
            rule, and it is honest about the floor: it can only ever return a peak it can see.

``comb``    Fit the peak POSITIONS to k*P over consecutive integers k, solving for P directly. This
            can return a P below MIN_PERIOD, because P is inferred from the spacing rather than
            observed as a lag -- which is the only way to reach 5.93 px on this crop.

Both are then held to the same admission the engine already uses: the resulting px/cm must name an
ADMISSIBLE unit for the ruler's class and sit within ANCHOR_TOL of the MP anchor.

NOTHING HERE IS WIRED INTO PRODUCTION. This replays the stored ``rot`` rasters of completed runs and
scores every method against the human ruler measurements, so a change can be judged before it ships.

Run::

    python -m leafmachine3.modules.experiments.ACF_comb.comb_eval            # the 549 GT corpus
    python -m leafmachine3.modules.experiments.ACF_comb.comb_eval --sheet A_1989713254
"""
from __future__ import annotations

import argparse
import csv
import math
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

from leafmachine3.inference.ruler_lattice.lattice import MIN_PERIOD, _acf
from leafmachine3.inference.ruler_lattice.units import (
    ANCHOR_TOL, MINIMUM_UNITS, admissible_units, canon_unit, unit_mm,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
FIT_DATA = REPO / "models" / "mp_conversion_factor" / "fit_data.csv"

PEAK_FLOOR = 0.12        # an ACF local maximum below this fraction of the argmax is noise
MIN_PEAKS = 4            # fewer than this is not a comb, it is a coincidence
COMB_TOL = 0.12          # a peak counts as a multiple of P within this relative slack


# --------------------------------------------------------------------------- #
# peak extraction
# --------------------------------------------------------------------------- #
def acf_peaks(profile: np.ndarray) -> tuple[list[int], float]:
    """Every credible local maximum of the ACF, finest first, plus the argmax value."""
    a, lo, hi = _acf(profile)
    if a is None or hi <= lo + 1:
        return [], 0.0
    hi = min(hi, len(a) - 2)
    amax = lo + int(np.argmax(a[lo:hi + 1]))
    ref = float(a[amax])
    if ref <= 0:
        return [], 0.0
    peaks = [i for i in range(lo, hi)
             if a[i] >= a[i - 1] and a[i] >= a[i + 1] and a[i] > PEAK_FLOOR * ref]
    return peaks, ref


def fundamental_ladder(peaks: list[int]) -> float | None:
    """First peak from the fine end whose integer multiples explain the rest of the comb.

    "Take the finest credible peak as the minimum graduation and check the others are its
    multiples; if they are not, step up one peak and try again."

    The multiples must be DISTINCT and must reach at least 3x. Without that, any peak near the
    coarse end passes trivially: its few remaining neighbours all sit within COMB_TOL of 1x itself,
    which scores as a perfect comb of one. On A_1989713254 that returned lag 146 -- the second
    coarsest peak -- instead of failing honestly.
    """
    for i, p in enumerate(peaks):
        if p <= 0:
            continue
        others = peaks[i:]
        ks = {round(q / p) for q in others
              if abs(q / p - round(q / p)) <= COMB_TOL and round(q / p) >= 1}
        hits = sum(1 for q in others if abs(q / p - round(q / p)) <= COMB_TOL and round(q / p) >= 1)
        if len(ks) >= MIN_PEAKS and max(ks) >= 3 and hits >= 0.7 * len(others):
            return float(p)
    return None


def fundamental_comb(peaks: list[int]) -> float | None:
    """The comb SPACING, from a least-squares fit of peak position against peak index.

    The peaks of a tick comb sit at ``k*P`` for consecutive integers k, so their positions advance
    by exactly one P per index regardless of which k the series starts at. Regressing position on
    index therefore recovers P without ever having to identify k -- and without requiring P itself
    to be an observable lag, which is what makes a sub-MIN_PERIOD fundamental reachable.

    Fitting the multiples directly (round(peak/P) and least-square the assignment) was tried first
    and is worse: the ACF returns integer lags, so each peak carries up to half a pixel of
    quantisation, the k assignment locks onto that noise, and on the A_1989713254 crop it converged
    to 6.08 px with a 1.2 px median residual. The index regression uses only the SPACING, where the
    quantisation averages out over the series, and returns 5.82 px on the same peaks.
    """
    if len(peaks) < MIN_PEAKS:
        return None
    y = np.asarray(peaks, float)
    x = np.arange(len(y), dtype=float)
    P = float(np.polyfit(x, y, 1)[0])                 # slope = one period per index step
    if P < 1.0:
        return None
    # the series must actually be evenly spaced -- a ragged peak set is not a comb
    gaps = np.diff(y)
    if np.median(np.abs(gaps - P)) > COMB_TOL * P:
        return None
    return P


# --------------------------------------------------------------------------- #
# naming: turn a period into px/cm using only units the class can show
# --------------------------------------------------------------------------- #
def name_period(P: float, unit_class: str, anchor: float | None) -> tuple[float, str] | None:
    """Best (px/cm, unit) for a period, admissible units only, anchor-consistent when possible."""
    adm = [canon_unit(u) for u in (admissible_units(unit_class) or [])]
    adm = [u for u in adm if u in MINIMUM_UNITS]
    if not adm:
        return None
    cands = [(P / (unit_mm(u) / 10.0), u) for u in adm]
    if anchor and anchor > 0:
        near = [c for c in cands if abs(c[0] / anchor - 1.0) <= ANCHOR_TOL]
        if near:                          # closest to the anchor among the admissible namings
            return min(near, key=lambda c: abs(math.log(c[0] / anchor)))
        return None
    return min(cands, key=lambda c: abs(c[0] - P * 10))     # no anchor: prefer the 1-unit reading


# --------------------------------------------------------------------------- #
# corpus replay
# --------------------------------------------------------------------------- #
def ground_truth() -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    with FIT_DATA.open() as fh:
        for r in csv.DictReader(fh):
            acc.setdefault(r["filename"], []).append(float(r["cf"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def profile_of(row) -> np.ndarray | None:
    """The tick profile the ACF runs on, rebuilt from the stored deskewed strip + band."""
    rot = cv2.imread(str(row["rot_path"]), cv2.IMREAD_GRAYSCALE)
    if rot is None:
        return None
    y0, y1 = int(row["band_y0"] or 0), int(row["band_y1"] or rot.shape[0])
    y0, y1 = max(0, y0), min(rot.shape[0], max(y0 + 1, y1))
    band = rot[y0:y1].astype(np.float32)
    if band.size == 0:
        return None
    prof = band.mean(axis=0)
    return prof.max() - prof                    # ticks are dark -> make them peaks


def evaluate(db: Path, only: str | None, verbose: bool) -> None:
    gt = ground_truth()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = """SELECT s.image_stem, s.work_scale AS ws, s.cf_px_per_cm_predicted_by_mp AS a_raw,
                  k.detection_id, k.rot_path, k.band_y0, k.band_y1,
                  k.ruler_class, k.pxcm, k.P_acf_px, k.n_kept
           FROM ruler_CF_lattice_crop k
           JOIN specimen s USING(specimen_id)
           JOIN ruler_CF_lattice i USING(specimen_id)
           WHERE k.status = 'measured' AND k.rot_path IS NOT NULL"""
    has_frame = any(r[1] == "anchor_frame"
                    for r in con.execute("PRAGMA table_info(ruler_CF_lattice)"))
    if has_frame:
        q = q.replace("k.detection_id,", "i.anchor_frame, k.detection_id,")
    rows = [dict(r) for r in con.execute(q)]
    for r in rows:                       # pre-rename runs predate the column: they were all linear
        r.setdefault("anchor_frame", "original")
    con.close()
    if only:
        rows = [r for r in rows if only in r["image_stem"]]

    tally = {m: {"n": 0, "err": []} for m in ("current", "ladder", "comb")}
    paired: list[dict] = []
    below_floor = 0
    for r in rows:
        g = gt.get(r["image_stem"])
        if g is None and not only:
            continue
        ws = r["ws"] or 1.0
        gt_work = (g * ws) if g else None
        anchor = r["a_raw"] * (1.0 if (r["anchor_frame"] == "working") else ws) if r["a_raw"] else None
        prof = profile_of(r)
        if prof is None:
            continue
        peaks, _ref = acf_peaks(prof)
        if len(peaks) < MIN_PEAKS:
            continue

        got = {"current": r["pxcm"]}
        for meth, fn in (("ladder", fundamental_ladder), ("comb", fundamental_comb)):
            P = fn(peaks)
            named = name_period(P, r["ruler_class"], anchor) if P else None
            got[meth] = named[0] if named else None
            if meth == "comb" and P and P < MIN_PERIOD:
                below_floor += 1

        if verbose:
            P_l, P_c = fundamental_ladder(peaks), fundamental_comb(peaks)
            print(f"\n{r['image_stem'][:50]}  det{r['detection_id']}  class={r['ruler_class']}")
            print(f"  peaks[:12] {peaks[:12]}   gaps {list(np.diff(peaks[:8]))}")
            print(f"  anchor {anchor:.2f}" if anchor else "  anchor -")
            print(f"  ladder P={P_l}  comb P={None if P_c is None else round(P_c,3)}"
                  f"   (MIN_PERIOD={MIN_PERIOD})")
            for m in ("current", "ladder", "comb"):
                v = got[m]
                e = (f"{100*(v/gt_work-1):+7.2f}%" if (v and gt_work) else "    n/a")
                print(f"    {m:8s} {('%8.2f' % v) if v else '       -'} px/cm   err {e}")
            if gt_work:
                print(f"    truth    {gt_work:8.2f} px/cm")

        if gt_work:
            rec = {"stem": r["image_stem"], "det": r["detection_id"]}
            for m, v in got.items():
                rec[m] = (abs(v / gt_work - 1) if v else None)
                if v:
                    tally[m]["n"] += 1
                    tally[m]["err"].append(abs(v / gt_work - 1))
            paired.append(rec)

    if only:
        return
    print(f"\n  {len(rows)} measured crops with a ground-truth sheet and >= {MIN_PEAKS} ACF peaks")
    print(f"  comb returned a fundamental BELOW MIN_PERIOD={MIN_PERIOD} on {below_floor} crops "
          f"(unreachable by the current search)\n")
    print(f"  {'method':9s} {'n':>5s} {'mean|err|':>10s} {'median':>8s} {'p90':>8s} "
          f"{'<=2%':>7s} {'<=10%':>7s} {'>25%':>6s}")
    for m in ("current", "ladder", "comb"):
        e = np.array(tally[m]["err"])
        if not e.size:
            print(f"  {m:9s} {0:5d}        n/a")
            continue
        print(f"  {m:9s} {e.size:5d} {100*e.mean():9.2f}% {100*np.median(e):7.2f}% "
              f"{100*np.percentile(e,90):7.2f}% {100*(e<=0.02).mean():6.1f}% "
              f"{100*(e<=0.10).mean():6.1f}% {100*(e>0.25).mean():5.1f}%")

    # -- PAIRED: the table above lets each method answer on a different subset -------------
    print("\n  PAIRED on the crops where BOTH answer (the only fair comparison):")
    for m in ("ladder", "comb"):
        pair = [(p["current"], p[m]) for p in paired if p["current"] and p[m]]
        if not pair:
            continue
        ec = np.array([abs(a) for a, _ in pair]); em = np.array([abs(b) for _, b in pair])
        better = int((em < ec).sum()); worse = int((em > ec).sum())
        print(f"    current vs {m:6s}  n={len(pair):4d}   mean {100*ec.mean():5.2f}% -> "
              f"{100*em.mean():5.2f}%   >25%: {int((ec>0.25).sum())} -> {int((em>0.25).sum())}"
              f"   ({better} crops improved, {worse} regressed)")

    # -- the failures that matter: where the current engine is catastrophically wrong ------
    cat = [p for p in paired if p["current"] and p["current"] > 0.25]
    print(f"\n  CATASTROPHIC current readings (>25% from human truth): {len(cat)}")
    fixed = [p for p in cat if p["comb"] and p["comb"] <= 0.10]
    still = [p for p in cat if p["comb"] and p["comb"] > 0.25]
    silent = [p for p in cat if not p["comb"]]
    print(f"    comb brings within 10% : {len(fixed)}")
    print(f"    comb still >25%        : {len(still)}")
    print(f"    comb declines to answer: {len(silent)}   (the crop is simply withheld)")
    for p in sorted(cat, key=lambda x: -x["current"])[:8]:
        cm = f"{100*p['comb']:+7.2f}%" if p["comb"] else "      -"
        print(f"      {p['stem'][:44]:46s} current {100*p['current']:+8.1f}%   comb {cm}")

    # -- and the risk side: does comb ever break something the engine had right? -----------
    broke = [p for p in paired if p["current"] is not None and p["current"] <= 0.02
             and p["comb"] and p["comb"] > 0.10]
    print(f"\n  REGRESSIONS (current was within 2%, comb worse than 10%): {len(broke)}")
    for p in sorted(broke, key=lambda x: -x["comb"])[:8]:
        print(f"      {p['stem'][:44]:46s} current {100*p['current']:+7.2f}%   "
              f"comb {100*p['comb']:+8.2f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Replay ACF peak-comb period recovery.")
    ap.add_argument("--out", type=Path, default=REPO / "examples_out")
    ap.add_argument("--run", default="mp_anchor_linear")
    ap.add_argument("--sheet", default=None, help="verbose trace for one sheet")
    args = ap.parse_args(argv)
    db = args.out / args.run / f"{args.run}.sqlite"
    if not db.exists():
        raise SystemExit(f"missing run database: {db}")
    evaluate(db, args.sheet, verbose=bool(args.sheet))
    return 0


if __name__ == "__main__":
    sys.exit(main())
