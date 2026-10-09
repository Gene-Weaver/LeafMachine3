"""Momocs stage -- export every leaf in the formats the R morphometrics packages Momocs and Momocs2 read.

Runs AFTER the Reporter (and after ECT), because it CONSUMES the Reporter's holes-filled leaf-product
masks rather than re-rendering leaves. Per leaf it writes a Momocs-ready image; per sheet a Momit JSON;
per run a grouping table and one combined JSON:

    reports/Leaf_Momocs/<stem>__or-MOMOCS-lamina__x_y_x_y.jpg      black leaf on white, padded
    reports/Leaf_Momocs/Momit_JSON/<stem>.json                     this sheet's outlines (Momit format)
    reports/Leaf_Momocs/momocs_fac.csv                             one row per image (the Momocs $fac)
    reports/Leaf_Momocs/momocs_outlines.json                       every outline in the run (Momit format)

``write_fac_csv`` / ``write_momit_json`` switch the two run-level files. The per-sheet JSONs are always
written: they are what the run-level files are rebuilt from when a run resumes.

In R, either route gives the same outlines:

    Momocs  : coo <- import_jpg(list.files("Leaf_Momocs", "jpg$", full.names = TRUE))
              fac <- read.csv("Leaf_Momocs/momocs_fac.csv"); O <- Out(coo, fac = fac[match(names(coo), fac$id), ])
    Momocs2 : tb  <- Momit::from_json("Leaf_Momocs/momocs_outlines.json")
    both    : O   <- Momit::to_Momocs(tb)

The format rules (polarity, padding, filled holes, y-up, start point) and the reasons for each are in
:mod:`leafmachine3.core.momocs_prep`. ``include_petiole`` / ``oriented`` choose the Reporter product
read, and :func:`enforce_report_deps` makes sure the Reporter exports it.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from leafmachine3.core.imaging import crop_filename, save_image
from leafmachine3.core.momocs_prep import (
    merge_momit_documents,
    momit_document,
    momocs_image,
    momocs_outline,
    momocs_product_for,
    prepare_mask,
)
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.reporting.leaf_products import leaf_product_label

log = logging.getLogger("leafmachine3.momocs")

_LEAF_SEG = "Leaf"            # leaf_morphology.cls_name for a whole leaf instance
MOMOCS_DIR = "Leaf_Momocs"    # reports/<MOMOCS_DIR>/
JSON_SUBDIR = "Momit_JSON"
FAC_CSV = "momocs_fac.csv"
RUN_JSON = "momocs_outlines.json"


class Momocs(PipelineStage):
    """Write Momocs/Momocs2-ready leaf images and outlines from the Reporter's leaf-product masks."""

    key: str = "momocs"
    name: str = "Momocs"
    device_kind: str = "cpu"
    cpu_parallel: str = "thread"   # cv2 read/pad/write releases the GIL; per-leaf work is milliseconds
    io_bound: bool = True          # dominated by small file writes, like the Reporter
    depends_on: tuple[str, ...] = ("reporter",)   # consumes the Reporter's leaf-product masks
    owns_tables: tuple[str, ...] = ("leaf_momocs",)

    def _settings(self) -> dict:
        c = self.cfg.stage(self.key)
        g = c.get if hasattr(c, "get") else (lambda k, d: d)
        return {
            "include_petiole": bool(g("include_petiole", False)),
            "oriented": bool(g("oriented", True)),
            "pad_px": max(0, int(g("pad_px", 10))),
            "largest_component_only": bool(g("largest_component_only", True)),
            "jpg_quality": min(100, max(1, int(g("jpg_quality", 100)))),
            "outline_points": max(0, int(g("outline_points", 0))),
            "write_fac_csv": bool(g("write_fac_csv", True)),
            "write_momit_json": bool(g("write_momit_json", True)),
        }

    def collect_items(self, project) -> list[WorkItem]:
        """ONE WorkItem PER SHEET: its leaves are a handful of millisecond-scale image writes, and a
        sheet-level item lets the worker write that sheet's Momit JSON in one go."""
        s = self._settings()
        prod = momocs_product_for(s["include_petiole"])
        tree = "Leaf_Oriented" if s["oriented"] else "Leaf_Original"
        reports = Path(project.dirs.reports)
        mask_ext = str(_get(self.cfg.report, "formats", "mask_ext", default="png")).lstrip(".")
        src_dir = reports / tree / prod.folder
        # Must match the Reporter's leaf-product filename exactly, tree tag included (or-SEG-laminaHoles).
        src_label = leaf_product_label(tree, "SEG", prod.seg_friendly)

        items: list[WorkItem] = []
        for sid in project.db.specimens_with_rows("leaf_segmentation"):
            spec = _row_to_dict(project.db.get_specimen(sid))
            stem = str(spec.get("image_stem", sid))
            crop_boxes = project.db.detection_boxes(sid, "plant_detection")
            leaves, seen = [], set()
            morph = sorted(project.db.leaf_morphology(sid),          # lowest instance first per box
                           key=lambda m: (int(_g(m, "detection_id", -1)), int(_g(m, "instance_index", 0))))
            for m in morph:
                if str(_g(m, "cls_name", "")) != _LEAF_SEG:
                    continue
                oriented_ok = bool(int(_g(m, "oriented_leaf_success", 0) or 0))
                if s["oriented"] and not oriented_ok:     # Leaf_Oriented has no file for this leaf
                    continue
                did = int(_g(m, "detection_id", -1))
                box = crop_boxes.get(did)
                # The Reporter writes ONE leaf-product file per detection box, so a second instance in
                # the same box would only duplicate the first leaf's image.
                if not box or did in seen:
                    continue
                seen.add(did)
                det_box = tuple(int(round(v)) for v in box)
                leaves.append({
                    "leaf_id": int(_g(m, "leaf_id", -1)), "detection_id": did,
                    "instance_index": int(_g(m, "instance_index", 0)), "det_box": det_box,
                    "mask_path": str(src_dir / crop_filename(stem, src_label, det_box, mask_ext)),
                    "angle_cw": _g(m, "oriented_leaf_rotation_angle_degreesCW", None),
                })
            if leaves:
                items.append(WorkItem(sid, {"specimen": spec, "stem": stem, "leaves": leaves, "tree": tree,
                                            "mask_includes": prod.mask_includes,
                                            "momocs_friendly": prod.momocs_friendly,
                                            "reports_dir": str(reports)}))
        return items

    def infer(self, item: WorkItem, model: Any):
        """Write this sheet's Momocs images (+ its Momit JSON) and return the small per-leaf rows."""
        import cv2

        s = self._settings()
        p = item.payload
        spec, stem, tree = p["specimen"], p["stem"], p["tree"]
        out_dir = Path(p["reports_dir"]) / MOMOCS_DIR
        label = leaf_product_label(tree, "MOMOCS", p["momocs_friendly"])
        json_path = out_dir / JSON_SUBDIR / f"{stem}.json"

        rows, records, missing = [], [], 0
        for lf in p["leaves"]:
            mask = cv2.imread(lf["mask_path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                missing += 1
                continue
            padded = prepare_mask(mask, pad_px=s["pad_px"], fill_holes=True,
                                  largest_only=s["largest_component_only"])
            if padded is None:                       # an empty image would hang Momocs import_jpg
                log.warning("Momocs: empty or degenerate mask, skipping leaf: %s", lf["mask_path"])
                continue
            outline = momocs_outline(padded, s["outline_points"])
            inst = int(lf.get("instance_index", 0) or 0)
            # One image per detection box, like the Reporter mask it comes from, so no instance suffix.
            img_path = out_dir / crop_filename(stem, label, lf["det_box"], "jpg")
            save_image(momocs_image(padded), img_path, quality=s["jpg_quality"])
            x1, y1, x2, y2 = lf["det_box"]
            rec = {
                "id": img_path.stem,                 # == the name Momocs import_jpg gives this shape
                "coo": outline,
                "image": img_path.name, "image_stem": stem, "specimen_id": int(item.specimen_id),
                "leaf_id": lf["leaf_id"], "detection_id": lf["detection_id"], "instance_index": inst,
                "tree": tree, "mask_includes": p["mask_includes"],
                "box_x1": x1, "box_y1": y1, "box_x2": x2, "box_y2": y2,
                "orientation_angle_degreesCW": _num(lf["angle_cw"]),
                "outline_points": int(len(outline)), "pad_px": s["pad_px"],
                "image_width": int(padded.shape[1]), "image_height": int(padded.shape[0]),
                "cf_px_per_cm": _num(spec.get("cf_px_per_cm")), "cf_source": spec.get("cf_source") or "none",
                "cf_px_per_cm_predicted_by_mp": _num(spec.get("cf_px_per_cm_predicted_by_mp")),
                "work_scale": _num(spec.get("work_scale")),
            }
            records.append(rec)
            rows.append({
                "leaf_id": lf["leaf_id"], "specimen_id": item.specimen_id, "detection_id": lf["detection_id"],
                "instance_index": inst, "tree": tree, "mask_includes": p["mask_includes"],
                "source_mask": lf["mask_path"], "mask_path": str(img_path),
                "json_path": str(json_path),
                "n_outline_points": int(len(outline)),
                "image_width": int(padded.shape[1]), "image_height": int(padded.shape[0]),
            })
        if missing:
            # With include_petiole, a leaf with no detected petiole has no LaminaPetiole product, by
            # design (see the settings help). Otherwise a missing mask means the Reporter output is gone.
            (log.info if s["include_petiole"] else log.warning)(
                "Momocs: %d of %d leaf mask(s) missing for %s%s", missing, len(p["leaves"]), stem,
                " (leaves without a petiole are left out)" if s["include_petiole"] else "")
        if records:                                  # always: the run-level files are rebuilt from these
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(momit_document(records)))
        return rows

    def persist(self, project, item: WorkItem, rows) -> None:
        """Replace this sheet's ``leaf_momocs`` rows (the images + JSON were written in ``infer``)."""
        project.db.record_leaf_momocs(item.specimen_id, rows or [])

    def run(self, project, ctx) -> None:
        """Per-sheet work, then the run-level grouping table + combined JSON, rebuilt from every
        sheet's JSON so a resumed run still lists the sheets finished in earlier sessions."""
        super().run(project, ctx)
        self.write_run_files(project)

    def write_run_files(self, project) -> None:
        s = self._settings()
        out_dir = Path(project.dirs.reports) / MOMOCS_DIR
        docs = []
        for jp in project.db.leaf_momocs_json_paths():
            try:
                docs.append(json.loads(Path(jp).read_text()))
            except (OSError, ValueError) as exc:
                log.warning("Momocs: could not read sheet JSON %s (%s); left out of the run files", jp, exc)
        merged = merge_momit_documents(docs)
        if s["write_momit_json"]:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / RUN_JSON).write_text(json.dumps(merged))
        if s["write_fac_csv"]:
            out_dir.mkdir(parents=True, exist_ok=True)
            cols = [c for c in merged["metadata"]["columns"] if c != "coo"]
            with open(out_dir / FAC_CSV, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in merged["data"]:
                    w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
        log.info("Momocs: %d leaf image(s) across %d sheet(s) -> %s", merged["metadata"]["n_rows"], len(docs), out_dir)


# -- report-dependency enforcement -------------------------------------------------
def enforce_report_deps(cfg) -> None:
    """When Momocs is enabled, force the Reporter to export the holes-filled leaf product it reads.

    ``include_petiole`` picks the product and ``oriented`` the tree; both are switched on in
    ``report.leaf_products`` without disabling anything the user already asked for.
    """
    if not cfg.is_enabled("momocs"):
        return
    c = cfg.stage("momocs")
    g = c.get if hasattr(c, "get") else (lambda k, d: d)
    prod = momocs_product_for(bool(g("include_petiole", False)))
    oriented = bool(g("oriented", True))
    rep = cfg.report
    lp = rep.get("leaf_products")
    if not isinstance(lp, dict):
        lp = {}
        rep["leaf_products"] = lp
    lp["enabled"] = True
    lp["oriented" if oriented else "original"] = True
    products = lp.get("products")
    if not isinstance(products, dict):
        products = {}
        lp["products"] = products
    products[prod.product_key] = True
    log.info("Momocs enabled -> forcing Reporter leaf-product %r (%s)", prod.product_key,
             "Oriented" if oriented else "Original")


# -- tolerant accessors ------------------------------------------------------------
def _num(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _row_to_dict(row) -> dict:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    try:
        return {k: row[k] for k in row.keys()}
    except Exception:
        return {}


def _g(row, key, default=None):
    if row is None:
        return default
    if hasattr(row, key):
        v = getattr(row, key)
        return default if v is None else v
    try:
        v = row[key]
    except Exception:
        return default
    return default if v is None else v


def _get(obj, *path, default=None):
    cur = obj
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p, None) if hasattr(cur, "get") else getattr(cur, p, None)
    return default if cur is None else cur
