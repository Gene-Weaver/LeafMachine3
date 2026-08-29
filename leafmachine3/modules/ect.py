"""ECT stage -- Euler Characteristic Transform of each oriented Leaf_WHOLE leaf.

Runs AFTER the Reporter, because it CONSUMES the Reporter's oriented leaf-product masks
(``reports/Leaf_Oriented/...``) rather than re-deriving them: LM3 already produces clean oriented
leaf masks, so this stage just loads the right one per the user's ``include_petiole`` / ``include_holes``
toggles, computes the ECT natively with the modern ``ect`` package, and writes, per leaf:

    reports/Leaf_Data/Coordinates/<stem>__ECT__x_y_x_y.h5           (image_metadata / ect_data /
                                                                     leaf_outline [/ leaf_outline_simple])
    reports/Leaf_Data/Oriented_Leaf_ECT/<stem>__ECT__x_y_x_y.png               (Cartesian ECT)
    reports/Leaf_Data/Oriented_Leaf_Radial_ECT/<stem>__ECT-radial__x_y_x_y.png (polar ECT)
    reports/Leaf_Data/Oriented_Leaf_Radial_ECT_Overlay/<stem>__ECT-radial-overlay__x_y_x_y.png
                                                                    (polar ECT + leaf outline)

The three PNGs carry distinct ID tokens so they survive being flattened into one directory with
the rest of the run. The ``.h5`` deliberately shares the Cartesian PNG's token: it is data rather
than a picture, and its extension is what separates the two.

All three visuals share ONE direction origin -- the tip -- so they line up with each other AND with
the tip-up oriented masks they came from (see ``core.ect_compute.TIP_DIRECTION_RADIANS``). The
matrix stored in the .h5 stays RAW (theta = 0..2pi as the ``ect`` package returns it); the tip-up
re-basing is display-only and is described by the ``ect_data`` attrs.

The ECT ``include`` settings OVERRIDE ``report.leaf_products`` (see :func:`enforce_report_deps`) so the
exact mask ECT needs is guaranteed to be exported by the Reporter beforehand.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from leafmachine3.core.ect_compute import compute_ect, ect_product_for, tip_up_index
from leafmachine3.core.imaging import crop_filename, save_image
from leafmachine3.core.stage import PipelineStage, WorkItem
from leafmachine3.reporting.leaf_products import leaf_product_label
from leafmachine3.reporting.ect_viz import (
    render_cartesian_ect,
    render_radial_ect,
    render_radial_ect_overlay,
    visual_log,
)

log = logging.getLogger("leafmachine3.ect")

_LEAF_SEG = "Leaf"   # leaf_morphology.cls_name for a whole leaf instance


class ECT(PipelineStage):
    """Compute + store the ECT for each oriented Leaf_WHOLE leaf (loads the Reporter's oriented masks)."""

    key: str = "ect"
    name: str = "ECT"
    device_kind: str = "cpu"
    cpu_parallel: str = "process"   # matplotlib viz is GIL-bound AND pyplot is thread-unsafe -> processes
    fanout: bool = True             # one WorkItem PER LEAF -> N workers each pull ~total/N leaves
    # ~per-leaf cost at the default num_dirs=360 (compute_ect ~3ms + 2 polar figs ~130ms; the
    # Cartesian view is a straight numpy colormap). Renders are num_dirs x num_dirs px, so the real
    # cost grows with num_dirs (~0.45s at 720) -- this stays the conservative floor, and it only
    # sets hardware_setup's min_pool_items (the batch below which spawning the pool isn't worth it).
    est_item_seconds: float = 0.17
    depends_on: tuple[str, ...] = ("reporter",)   # consumes the Reporter's Leaf_Oriented masks
    owns_tables: tuple[str, ...] = ("leaf_ect",)

    def _settings(self) -> dict:
        c = self.cfg.stage(self.key)
        g = c.get if hasattr(c, "get") else (lambda k, d: d)
        return {
            "num_dirs": int(g("num_dirs", 360)),
            "bound_radius": float(g("bound_radius", 1.0)),
            "include_petiole": bool(g("include_petiole", False)),
            "include_holes": bool(g("include_holes", False)),
            "export_literal_coords": bool(g("export_literal_coords", False)),
            "outline_simple": bool(g("outline_simple", True)),
            "simplify_tolerance": float(g("simplify_tolerance", 0.0025)),
            "simplify_cutoff": int(g("simplify_cutoff", 500)),
            "radial_viz": bool(g("radial_viz", True)),
            "cartesian_viz": bool(g("cartesian_viz", True)),
            "radial_overlay_viz": bool(g("radial_overlay_viz", True)),
            # Matplotlib colormap name, CASE-SENSITIVE ("Greens" is valid, "greens" is not).
            "palette": str(g("palette", "Greens")),
            # DISPLAY ONLY -- log-scales the matrix on the way to the renderers so the palette
            # spreads over the dense mid-range. Never reaches the .h5 or the leaf_ect row.
            "apply_log_to_visual_for_bold_color": bool(g("apply_log_to_visual_for_bold_color", False)),
        }

    def collect_items(self, project) -> list[WorkItem]:
        """ONE WorkItem PER ORIENTED LEAF (fanout): the pool then load-balances every leaf across
        the process workers. Payload carries the leaf, its expected oriented-mask path, and the
        reports dir the worker writes its own .h5 + PNGs into (so only a small row is returned)."""
        s = self._settings()
        prod = ect_product_for(s["include_petiole"], s["include_holes"])
        reports = Path(project.dirs.reports)
        mask_ext = str(_get(self.cfg.report, "formats", "mask_ext", default="png")).lstrip(".")
        oriented_dir = reports / "Leaf_Oriented" / prod.folder
        # Must match the Reporter's leaf-product filename exactly, tree tag included (or-SEG-lamina).
        label = leaf_product_label("Leaf_Oriented", "SEG", prod.seg_friendly)

        items: list[WorkItem] = []
        for sid in project.db.specimens_with_rows("leaf_segmentation"):
            spec = _row_to_dict(project.db.get_specimen(sid))
            stem = str(spec.get("image_stem", sid))
            crop_boxes = project.db.detection_boxes(sid, "plant_detection")
            for m in project.db.leaf_morphology(sid):
                if str(_g(m, "cls_name", "")) != _LEAF_SEG:
                    continue
                if not int(_g(m, "oriented_leaf_success", 0) or 0):   # only ORIENTED leaves
                    continue
                did = int(_g(m, "detection_id", -1))
                box = crop_boxes.get(did)
                if not box:
                    continue
                det_box = tuple(int(round(v)) for v in box)
                leaf = {
                    "leaf_id": int(_g(m, "leaf_id", -1)), "detection_id": did,
                    "instance_index": int(_g(m, "instance_index", 0)),
                    "det_box": det_box,
                    "mask_path": str(oriented_dir / crop_filename(stem, label, det_box, mask_ext)),
                    "angle_cw": _g(m, "oriented_leaf_rotation_angle_degreesCW", None),
                }
                items.append(WorkItem(sid, {"specimen": spec, "stem": stem, "leaf": leaf,
                                            "mask_includes": prod.mask_includes,
                                            "reports_dir": str(reports)}))
        return items

    def infer(self, item: WorkItem, model: Any):
        """Compute ONE leaf's ECT + visuals AND write its .h5 + PNGs (in the worker), returning only
        the small ``leaf_ect`` row -- so the rendered images never cross the process boundary."""
        import cv2

        s = self._settings()
        p = item.payload
        lf = p["leaf"]
        mask = cv2.imread(lf["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            log.warning("ECT: oriented mask not found, skipping leaf: %s", lf["mask_path"])
            return None
        res = compute_ect(
            mask > 127, num_dirs=s["num_dirs"], bound_radius=s["bound_radius"],
            want_simple=s["outline_simple"], simplify_tolerance=s["simplify_tolerance"],
            simplify_cutoff=s["simplify_cutoff"],
        )
        if res is None:
            return None
        # The ONLY place the display log is applied: a separate matrix handed to the renderers.
        # `res` -- and therefore the .h5 matrix and the leaf_ect row -- stays the raw integer ECT,
        # so turning this toggle on can never change a stored value or a downstream analysis.
        viz = visual_log(res.ect_matrix) if s["apply_log_to_visual_for_bold_color"] else res.ect_matrix
        radial = (render_radial_ect(viz, res.thetas, res.thresholds, res.bound_radius,
                                    cmap=s["palette"]) if s["radial_viz"] else None)
        cart = (render_cartesian_ect(viz, res.thetas, cmap=s["palette"])
                if s["cartesian_viz"] else None)
        overlay = (render_radial_ect_overlay(viz, res.thetas, res.thresholds,
                                             res.bound_radius, res.outline_norm, cmap=s["palette"])
                   if s["radial_overlay_viz"] else None)

        reports = Path(p["reports_dir"])
        coords_dir = reports / "Leaf_Data" / "Coordinates"
        radial_dir = reports / "Leaf_Data" / "Oriented_Leaf_Radial_ECT"
        cart_dir = reports / "Leaf_Data" / "Oriented_Leaf_ECT"
        overlay_dir = reports / "Leaf_Data" / "Oriented_Leaf_Radial_ECT_Overlay"
        # >1 leaf instance can share a crop (det box) -> fold the instance into the stem so files
        # stay distinct (mirrors the Reporter's per-leaf landmark-overlay naming).
        inst = int(lf.get("instance_index", 0) or 0)
        file_stem = p["stem"] if inst == 0 else f"{p['stem']}__i{inst}"
        box = lf["det_box"]
        h5_path = coords_dir / crop_filename(file_stem, "ECT", box, "h5")
        cart_path = cart_dir / crop_filename(file_stem, "ECT", box, "png") if cart is not None else None
        radial_path = (radial_dir / crop_filename(file_stem, "ECT-radial", box, "png")
                       if radial is not None else None)
        overlay_path = (overlay_dir / crop_filename(file_stem, "ECT-radial-overlay", box, "png")
                        if overlay is not None else None)

        r = {"leaf": lf, "result": res, "radial": radial, "cartesian": cart,
             "specimen": p["specimen"], "stem": p["stem"], "mask_includes": p["mask_includes"]}
        _write_ect_h5(h5_path, r, s)
        if radial_path is not None:
            save_image(radial, radial_path)
        if cart_path is not None:
            save_image(cart, cart_path)
        if overlay_path is not None:
            save_image(overlay, overlay_path)
        return {
            "leaf_id": lf["leaf_id"], "specimen_id": item.specimen_id,
            "detection_id": lf["detection_id"], "instance_index": lf["instance_index"],
            "mask_includes": p["mask_includes"], "h5_path": str(h5_path),
            "radial_png": (str(radial_path) if radial_path else None),
            "ect_png": (str(cart_path) if cart_path else None),
            "overlay_png": (str(overlay_path) if overlay_path else None),
            "n_outline_points": res.n_points, "num_dirs": len(res.thetas),
        }

    def persist(self, project, item: WorkItem, row) -> None:
        """Record ONE leaf's ``leaf_ect`` row (upsert on leaf_id). The .h5/PNGs were already written
        by the worker in ``infer``; a leaf that produced nothing (missing mask) returns None."""
        if row is not None:
            project.db.record_leaf_ect_one(row)


# -- h5 writer ---------------------------------------------------------------------
def _write_ect_h5(path: Path, r: dict, s: dict) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    res, lf = r["result"], r["leaf"]
    norm = res.outline_norm
    ys = norm[:, 1]
    top = norm[int(np.argmin(ys))]          # oriented tip-up: topmost point (image y-down)
    bottom = norm[int(np.argmax(ys))]
    lit = res.outline_literal
    with h5py.File(str(path), "w") as f:
        # image_metadata: most of the specimen row + the required extras
        g = f.create_group("image_metadata")
        meta = dict(r["specimen"])
        meta.update({
            "parent_image_filename": meta.get("image_name"),
            "mask_includes": r["mask_includes"],
            "leaf_id": lf["leaf_id"], "detection_id": lf["detection_id"],
            "instance_index": lf["instance_index"],
            "orientation_angle_degreesCW": lf["angle_cw"],
            "ect_num_dirs": len(res.thetas), "ect_bound_radius": res.bound_radius,
        })
        for k, v in meta.items():
            g.attrs[k] = _h5v(v)

        # ect_data
        e = f.create_group("ect_data")
        e.create_dataset("matrix", data=res.ect_matrix)
        e.create_dataset("thetas", data=res.thetas)
        e.create_dataset("thresholds", data=res.thresholds)
        e.attrs["num_dirs"] = int(len(res.thetas))
        e.attrs["num_thresholds"] = int(len(res.thresholds))
        e.attrs["bound_radius"] = float(res.bound_radius)
        # `matrix` is stored RAW (theta = 0..2pi, as the ect package returns it) so it stays
        # reproducible and comparable across runs. The tip-up re-basing LM3's visuals use is
        # display-only; a consumer reproduces it with np.roll(matrix, -theta_up_index, axis=1).
        e.attrs["matrix_axes"] = "threshold,direction"
        k = tip_up_index(res.thetas)
        e.attrs["theta_up_index"] = int(k)
        e.attrs["theta_up_radians"] = float(res.thetas[k])
        e.attrs["display_convention"] = ("phi = (theta - theta_up) mod 2pi, measured clockwise "
                                         "on screen from the leaf tip")

        # leaf_outline (always) -- normalized to the unit circle by default; LM2-style position header
        o = f.create_group("leaf_outline")
        o.create_dataset("coords", data=norm)
        if s["export_literal_coords"]:
            o.create_dataset("coords_literal", data=lit)   # raw px outline of the loaded oriented mask
        o.attrs["normalized_unit_circle"] = True
        o.attrs["n_points"] = int(res.n_points)
        o.attrs["max_extent_px"] = float(max(np.ptp(lit[:, 0]), np.ptp(lit[:, 1])))
        o.attrs["x_min_px"] = float(lit[:, 0].min())
        o.attrs["y_min_px"] = float(lit[:, 1].min())
        o.attrs["orientation_angle_degreesCW"] = _h5v(lf["angle_cw"])
        o.attrs["cf_px_per_cm"] = _h5v(r["specimen"].get("cf_px_per_cm"))
        o.attrs["cf_px_per_cm_predicted_by_mp"] = _h5v(r["specimen"].get("cf_px_per_cm_predicted_by_mp"))
        # WORKING dims, to match every other px value in this group. These used to be the ORIGINAL
        # dims sitting beside working-frame max_extent_px / x_min_px / cf_px_per_cm, so anyone
        # normalizing an outline by image_width was off by exactly 1/work_scale on a resized sheet.
        # The original dims are still on the specimen row for provenance; they are not this frame.
        o.attrs["image_width"] = _h5v(r["specimen"].get("width"))
        o.attrs["image_height"] = _h5v(r["specimen"].get("height"))
        o.attrs["frame"] = "working"
        o.attrs["work_scale"] = _h5v(r["specimen"].get("work_scale"))
        o.attrs["top_xy"] = top
        o.attrs["bottom_xy"] = bottom
        o.attrs["tip_xy"] = top          # oriented tip-up: tip ~= topmost, base ~= bottommost
        o.attrs["base_xy"] = bottom
        o.attrs["mask_includes"] = r["mask_includes"]

        # leaf_outline_simple (optional) -- Douglas-Peucker of the normalized outline
        if res.outline_simple is not None:
            sg = f.create_group("leaf_outline_simple")
            sg.create_dataset("coords", data=res.outline_simple)
            sg.attrs["normalized_unit_circle"] = True
            sg.attrs["n_points"] = int(len(res.outline_simple))
            sg.attrs["simplify_tolerance"] = float(s["simplify_tolerance"])
            sg.attrs["method"] = "douglas_peucker"


def _h5v(v):
    """Coerce a value to something h5py can store as an attribute (None -> NaN)."""
    if v is None:
        return np.nan
    if isinstance(v, (str, bytes, int, float, np.integer, np.floating, np.ndarray)):
        return v
    return str(v)


# -- report-dependency enforcement -------------------------------------------------
def enforce_report_deps(cfg) -> None:
    """When ECT is enabled, force the Reporter to export the oriented leaf-product mask ECT needs.

    ECT's ``include_petiole`` / ``include_holes`` decide which ``Leaf_Oriented`` product is required;
    this turns that product on (and ``oriented`` on) in ``report.leaf_products`` so the mask exists at
    ECT time -- the ECT settings override the report mask-export settings, without disabling anything
    the user already asked for.
    """
    if not cfg.is_enabled("ect"):
        return
    c = cfg.stage("ect")
    g = c.get if hasattr(c, "get") else (lambda k, d: d)
    prod = ect_product_for(bool(g("include_petiole", False)), bool(g("include_holes", False)))
    rep = cfg.report
    lp = rep.get("leaf_products")
    if not isinstance(lp, dict):
        lp = {}
        rep["leaf_products"] = lp
    lp["enabled"] = True
    lp["oriented"] = True
    products = lp.get("products")
    if not isinstance(products, dict):
        products = {}
        lp["products"] = products
    products[prod.product_key] = True
    log.info("ECT enabled -> forcing Reporter leaf-product %r (Oriented) for mask_includes=%s",
             prod.product_key, prod.mask_includes)


# -- tolerant accessors ------------------------------------------------------------
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
