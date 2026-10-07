"""Compare two LM3 run databases table by table on natural keys (not autoincrement ids).

Usage: python tools/verification/compare_run_databases.py A.sqlite B.sqlite [--label TEXT] [--json PATH]

Exit status: 0 when every table matches, 1 when any differs -- so it can gate a script.
--json writes the full per-column report; nothing is written unless asked (the old default wrote to
/tmp under a file name built from the label, and a label containing the input paths had slashes).
"""
import argparse, sqlite3, sys, json, math
from pathlib import Path
import numpy as np
_ap = argparse.ArgumentParser(description="Compare two LM3 run databases table by table.")
_ap.add_argument("a", help="the reference run's .sqlite")
_ap.add_argument("b", help="the run to check against it")
_ap.add_argument("--label", default=None, help="heading for the printout (default: the two run names)")
_ap.add_argument("--json", default=None, help="also write the full report to this path")
_args = _ap.parse_args()
A, B = _args.a, _args.b
label = _args.label or f"{Path(A).stem} vs {Path(B).stem}"

def load(path):
    con = sqlite3.connect(path); con.row_factory = sqlite3.Row
    spec = {r["specimen_id"]: r["image_stem"] for r in con.execute("select specimen_id, image_stem from specimen")}
    det = {}
    for t in ("plant_detection", "archival_detection"):   # ids are per-table autoincrements: keep them apart
        det[t] = {}
        for r in con.execute(f"select detection_id, specimen_id, cls_name, x1, y1, x2, y2 from {t}"):
            det[t][r["detection_id"]] = (spec[r["specimen_id"]], r["cls_name"], round(r["x1"]), round(r["y1"]), round(r["x2"]), round(r["y2"]))
    return con, spec, det

NUMERIC_TOL = 1e-6
# table -> (key columns beyond specimen/detection, columns to ignore)
TABLES = {
    "specimen": ([], {"specimen_id", "original_path", "working_path", "ingested_at", "orig_mtime"}),
    "archival_detection": (["cls_name", "x1", "y1", "x2", "y2"], {"detection_id", "specimen_id", "crop_path", "suppressed_by"}),
    "plant_detection": (["cls_name", "x1", "y1", "x2", "y2"], {"detection_id", "specimen_id", "crop_path", "suppressed_by"}),
    "phenology": ([], {"specimen_id"}),
    "ruler_classification": ([], {"ruler_class_id", "specimen_id", "detection_id", "squarify_path"}),
    "specimen_mask": ([], {"specimen_id", "mask_path", "refined_path", "created_at"}),
    "leaf_segmentation": (["instance_index", "cls_name"], {"leaf_id", "specimen_id", "detection_id", "mask_data"}),
    "leaf_morphology": (["instance_index", "cls_name"], {"morph_id", "leaf_id", "specimen_id", "detection_id", "created_at", "rotated_bbox_json"}),
    "leaf_landmark": (["instance_index", "kpt_index"], {"landmark_id", "specimen_id", "detection_id"}),
    "leaf_landmark_measurement": (["instance_index"], {"measure_id", "specimen_id", "detection_id", "created_at"}),
    "leaf_petiole": (["instance_index"], {"petiole_id", "leaf_id", "specimen_id", "detection_id", "created_at", "width_segment_json", "sample_segments_json"}),
    "bilateral_symmetry": (["instance_index"], {"bsym_id", "leaf_id", "specimen_id", "detection_id", "created_at", "qc_png", "midvein_json", "reasons_json"}),
    "leaf_ect": (["instance_index"], {"leaf_id", "specimen_id", "detection_id", "created_at", "h5_path", "radial_png", "ect_png", "overlay_png"}),
    "image_status": (["stage_key"], {"specimen_id", "updated_at"}),
}

def rows_by_key(con, spec, det, table, keycols):
    cols = [r[1] for r in con.execute(f"pragma table_info({table})")]
    out = {}
    for r in con.execute(f"select * from {table}"):
        r = dict(zip(cols, r))
        key = [spec[r["specimen_id"]]] if "specimen_id" in r else []
        if "detection_id" in r and table not in ("archival_detection", "plant_detection"):
            src = "archival_detection" if table == "ruler_classification" else "plant_detection"
            key.append(det[src].get(r["detection_id"], ("?", r["detection_id"])))
        for k in keycols:
            v = r[k]; key.append(round(v) if isinstance(v, float) else v)
        key = tuple(key)
        if key in out: key = key + (len([k for k in out if k[:len(key)] == key]),)   # dedupe collisions
        out[key] = r
    return out

ca, sa, da = load(A); cb, sb, db = load(B)
report = {}
worst = []
for table, (keycols, ignore) in TABLES.items():
    ra, rb = rows_by_key(ca, sa, da, table, keycols), rows_by_key(cb, sb, db, table, keycols)
    only_a, only_b = sorted(set(ra) - set(rb), key=str), sorted(set(rb) - set(ra), key=str)
    common = set(ra) & set(rb)
    col_stats = {}
    for k in common:
        for c, va in ra[k].items():
            if c in ignore or c in keycols: continue
            vb = rb[k].get(c)
            st = col_stats.setdefault(c, {"n": 0, "mismatch": 0, "maxabs": 0.0, "maxrel": 0.0})
            st["n"] += 1
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
                if any(isinstance(v, float) and math.isnan(v) for v in (va, vb)):
                    if not (isinstance(va, float) and isinstance(vb, float) and math.isnan(va) and math.isnan(vb)): st["mismatch"] += 1
                    continue
                d = abs(va - vb); st["maxabs"] = max(st["maxabs"], d)
                st["maxrel"] = max(st["maxrel"], d / max(abs(va), abs(vb), 1e-9))
                if d > NUMERIC_TOL: st["mismatch"] += 1
            elif va != vb:
                st["mismatch"] += 1
    bad = {c: s for c, s in col_stats.items() if s["mismatch"]}
    report[table] = {"rows_a": len(ra), "rows_b": len(rb), "only_a": len(only_a), "only_b": len(only_b), "common": len(common),
                     "cols_with_mismatch": {c: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()} for c, s in bad.items()}}
    if only_a or only_b or bad: worst.append((table, only_a[:3], only_b[:3]))
print(f"=== {label}")
for t, r in report.items():
    flag = "OK " if (r["only_a"] == 0 and r["only_b"] == 0 and not r["cols_with_mismatch"]) else "DIFF"
    line = f"{flag} {t:26s} rows {r['rows_a']:5d}/{r['rows_b']:5d} only_a={r['only_a']} only_b={r['only_b']}"
    if r["cols_with_mismatch"]:
        line += "\n      " + "\n      ".join(f"{c}: {s['mismatch']}/{s['n']} differ, maxabs={s['maxabs']}, maxrel={s['maxrel']}" for c, s in r["cols_with_mismatch"].items())
    print(line)
for t, oa, ob in worst:
    if oa or ob: print(f"   {t} examples only_a={oa} only_b={ob}")
if _args.json:
    with open(_args.json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
sys.exit(1 if worst else 0)
