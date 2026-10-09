"""LIVE FieldPrism check on the two FPfit example images (not collected by pytest).

    cd /datac/Labelbox_Dump/LM3
    .venv/bin/python tests/live_fieldprism_check.py

Reads the Ruler boxes of the finished `_fp_baseline` run, runs `analyze_fieldprism` on each
image as-is, then simulates harder cases ON THESE TWO IMAGES ONLY: markers blanked out
(painted white) down to every 2-marker pair and to 1 marker, and the whole image rotated
90/180/270 (np.rot90 with the boxes transformed). It prints per-marker roles/CF/validation
and the sheet result, and ends with a PASS/FAIL line per expectation.
"""
from __future__ import annotations

import itertools
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leafmachine3.inference.ruler_lattice import fieldprism as fp  # noqa: E402

IMAGES = {
    "15_1_FPfit": ROOT / "examples/images/15_1_FPfit.JPG",
    "5_1_FPfit": ROOT / "examples/images/5_1_FPfit.JPG",
}
DB = ROOT / "examples_out/_fp_baseline/_fp_baseline.sqlite"
FAILS: list[str] = []


def check(cond, what):
    print(("  PASS " if cond else "  FAIL ") + what)
    if not cond:
        FAILS.append(what)


def load_boxes():
    con = sqlite3.connect(str(DB))
    rows = con.execute(
        "SELECT s.image_stem, d.detection_id, d.x1, d.y1, d.x2, d.y2, d.conf "
        "FROM archival_detection d JOIN specimen s ON s.specimen_id = d.specimen_id "
        "WHERE d.cls_name = 'Ruler' AND d.suppressed = 0 ORDER BY d.detection_id").fetchall()
    con.close()
    out = {}
    for stem, did, x1, y1, x2, y2, conf in rows:
        out.setdefault(stem, []).append({"detection_id": did, "x1": x1, "y1": y1, "x2": x2,
                                         "y2": y2, "det_conf": conf})
    return out


def rot_point(x, y, w, h, k):
    """np.rot90(img, k) (counterclockwise k*90) applied to a point."""
    for _ in range(k % 4):
        x, y, w, h = y, w - 1 - x, h, w
    return x, y


def rot_boxes(crops, w, h, k):
    out = []
    for c in crops:
        pts = [rot_point(x, y, w, h, k) for x, y in
               ((c["x1"], c["y1"]), (c["x2"], c["y1"]), (c["x2"], c["y2"]), (c["x1"], c["y2"]))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        out.append(dict(c, x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
    return out


def report(res, title):
    sh = res["sheet"]
    print(f"\n== {title}")
    for m in res["markers"]:
        r = m["roles"] or {}
        fails = [k for k, v in m["validation"].items() if not v["ok"]]
        print(f"  det {m['detection_id']:>3}  {m['status']:<8} valid={str(m['valid']):<5} "
              f"verdict={m['verdict']:<8} corner={str(m['sheet_corner']):<4} "
              f"pxcm={m['pxcm'] if m['pxcm'] is None else round(m['pxcm'], 2)!s:<7} "
              f"orient={m['orientation_deg']} holes={m['holes_filled']} "
              f"app_peaks={m.get('n_peaks_app')} "
              f"TL={[round(v, 1) for v in r['TL']] if r else None} "
              f"{'FAILS ' + ','.join(fails) + ' ' if fails else ''}{m['status_reason'] or ''}")
    fit = sh.get("fit") or {}
    print(f"  sheet: status={sh['status']} type={sh['sheet_type']} corners_ambiguous="
          f"{sh['corners_ambiguous']} orientation={sh['orientation_deg']} "
          f"used={sh['n_fp_used']} inferred={sh['n_fp_inferred']}")
    if fit:
        print(f"  fit: rot={fit['rotation_deg']:.2f} rms={fit['rms_mm']:.3f}mm "
              f"max={fit['max_mm']:.3f}mm scale_dev={fit['scale_dev_pct']:.2f}% "
              f"(cost term {fit['scale_dev_mm']:.3f}mm) cost={fit['cost_mm']:.3f}mm")
    print("  candidates: " + "; ".join(
        f"{c['sheet_type']}{'*' if c['admissible'] else ''} {c['cost_mm']:.2f}mm "
        f"in={int(c['inside_image'])} sym={c['margin_spread_mm']:.1f}"
        for c in sh["candidates"][:5]))
    print(f"  CF: sheet_fit={sh['cf_px_per_cm_sheet_fit']} marker_mean="
          f"{sh['cf_px_per_cm_marker_mean']} anchor={res['anchor_cf']} "
          f"({sh['cf_source_detail']}) spread={sh['fp_peer_spread_pct']}% "
          f"confidence={res['confidence']} reasons={sh['fp_reasons']}")
    for c, v in sh["corners"].items():
        if not v["observed"]:
            print(f"  inferred {c} marker: TL square at {[round(q, 1) for q in v['squares']['TL']]}")
    return sh


def corner_map(res):
    return {m["detection_id"]: m["sheet_corner"] for m in res["markers"]}


def main():
    boxes = load_boxes()
    base = {}
    for stem, path in IMAGES.items():
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        crops = boxes[stem]
        res = fp.analyze_fieldprism(img, crops)
        sh = report(res, f"{stem} ({w}x{h}) as-is")
        base[stem] = res
        check(sh["status"] == "identified" and sh["sheet_type"] == "Letter", f"{stem}: Letter identified")
        others = [c for c in sh["candidates"] if c["sheet_type"] != "Letter"]
        check(not any(c["admissible"] or c["in_pool"] for c in others),
              f"{stem}: no other sheet type admissible or tied")
        cf = res["anchor_cf"]
        if stem.startswith("15_1"):
            px = sorted(round(m["pxcm"], 1) for m in res["markers"])
            check(sh["n_fp_used"] == 4 and sh["n_fp_inferred"] == 0, "15_1: 4 markers used")
            # The reference values came from a moment-centroid script whose per-square mask
            # admits the touching corner of C, which biases its pitch ~0.5% low; the app's
            # plateau centers agree with clean moment centroids, so 1% is the honest bound.
            check(all(abs(a / b - 1) < 0.01 for a, b in zip(px, [113.8, 114.1, 115.4, 115.6])),
                  f"15_1: per-marker CF {px} within 1% of 113.8/114.1/115.4/115.6")
            check(abs(cf - 114.7) < 0.5, f"15_1: sheet-fit CF {cf:.2f} ~ 114.7")
        else:
            check(sh["n_fp_used"] == 3 and sh["n_fp_inferred"] == 1, "5_1: 3 used + 1 inferred")
            bad = [m for m in res["markers"] if m["verdict"] != "used"]
            check(len(bad) == 1 and bad[0]["status"] == "measured" and not bad[0]["valid"]
                  and sh["corners"]["TL"]["observed"] is False,
                  "5_1: twig-crossed TL marker measured but rejected by validation -> inferred")
            # The apps drop this marker (3 plateaus); only LM3's hole fill makes a 4th.
            check(len(bad) == 1 and bad[0]["n_peaks_app"] == 3 and bad[0]["n_peaks"] == 4
                  and (bad[0]["status_reason"] or "").startswith("app finder: 3 square candidates")
                  and "app finder" in (bad[0]["verdict_note"] or ""),
                  "5_1: TL marker says the app finder sees 3 squares (4th only after hole fill)")
            check(91.4 <= cf <= 92.2, f"5_1: CF {cf:.2f} in ~91.7-91.9")
            inf = [v for v in sh["corners"].values() if not v["observed"]]
            tl = inf[0]["squares"]["TL"] if inf else [1e9, 1e9]
            check(np.hypot(tl[0] - 138, tl[1] - 138) < 4,
                  f"5_1: inferred TL-marker TL square {[round(v, 1) for v in tl]} ~ (138,138)")

        # ---- blank markers out (paint the box white) --------------------------------------
        good = [m["detection_id"] for m in res["markers"] if m["verdict"] == "used"]
        corner_of = corner_map(res)
        for keep_n in (2, 1):
            for keep in itertools.combinations(good, keep_n):
                im2 = img.copy()
                for c in crops:
                    if c["detection_id"] not in keep:
                        x1, y1 = int(max(0, c["x1"] - 25)), int(max(0, c["y1"] - 25))
                        x2, y2 = int(min(w, c["x2"] + 25)), int(min(h, c["y2"] + 25))
                        im2[y1:y2, x1:x2] = 255
                kc = [c for c in crops if c["detection_id"] in keep]
                r2 = fp.analyze_fieldprism(im2, kc)
                pair = "+".join(sorted(corner_of[d] for d in keep))
                s2 = report(r2, f"{stem} blanked to {pair}")
                if keep_n == 2:
                    ok = (s2["status"] == "identified" and s2["sheet_type"] == "Letter"
                          and all(corner_map(r2)[d] == corner_of[d] for d in keep))
                    check(ok, f"{stem} pair {pair}: Letter + same corners")
                    check(abs(r2["anchor_cf"] / cf - 1) < 0.01,
                          f"{stem} pair {pair}: CF {r2['anchor_cf']:.2f} within 1% of {cf:.2f}")
                else:
                    check(s2["status"] == "undetermined" and r2["confidence"] == "high",
                          f"{stem} single {pair}: undetermined, high (single marker allowed)")

        # ---- rotate the whole image -------------------------------------------------------
        for k in (1, 2, 3):
            im3 = np.ascontiguousarray(np.rot90(img, k))
            r3 = fp.analyze_fieldprism(im3, rot_boxes(crops, w, h, k))
            s3 = report(r3, f"{stem} np.rot90 k={k}")
            check(s3["status"] == "identified" and s3["sheet_type"] == "Letter"
                  and corner_map(r3) == corner_of, f"{stem} rot90 k={k}: Letter + same corners")
            check(s3["orientation_deg"] == 90 * k, f"{stem} rot90 k={k}: orientation {s3['orientation_deg']}")
            check(abs(r3["anchor_cf"] / cf - 1) < 0.002,
                  f"{stem} rot90 k={k}: CF {r3['anchor_cf']:.2f} vs {cf:.2f}")
    print("\n" + ("ALL LIVE CHECKS PASSED" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}"))
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
