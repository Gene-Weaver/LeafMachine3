"""Stage 11 -- Petiole Width (CPU post-process).

Measure each leaf's petiole width from its ``Petiole`` segmentation mask and the landmark petiole
centerline (``lamina_base`` -> ``petiole_0..4`` -> ``petiole_tip``): the perpendicular thickness of
the mask at several points near the blade junction, reported as the MEDIAN (``core.petiole``).
Results go to ``leaf_petiole`` (one row per leaf that has a petiole), with the sample + reported
width segments (working coords) the Reporter draws in the petiole overlays.

Also derives ``leaf_mass_per_area`` -- an LMA proxy in g/m^2 from the Royer petiole-width scaling
``log10(LMA) = 3.070 + 0.382 * log10(PW^2 / A)`` -- onto the same ``leaf_petiole`` row.

Runs on the RAW Petiole mask for now; once ``LM3_Specimen_Segmentation`` lands (TODO #5) it should
consume the EDGE-REFINED petiole mask, since a thin petiole is very sensitive to leftover paper.
Widths are pixels; ``width_cm`` is filled later by MetricGrounding when a ruler CF exists.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from leafmachine3.core.imaging import decode_polygon, offset_polygon, polygon_mask
from leafmachine3.core.landmarks import PETIOLE_N
from leafmachine3.core.petiole import DEFAULT_FRACTIONS, measure_petiole
from leafmachine3.core.records import PetioleRow
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.petiole_width")

_LEAF, _PETIOLE = "Leaf", "Petiole"
# ordered petiole centerline landmark names: blade junction -> ... -> free tip
_CENTERLINE_NAMES = ("lamina_base", *[f"petiole_{i}" for i in range(PETIOLE_N)], "petiole_tip")


class PetioleWidth(PipelineStage):
    """Per-leaf petiole width (median perpendicular thickness near the blade junction)."""

    key: str = "petiole_width"
    name: str = "Petiole Width"
    device_kind: str = "cpu"
    # morphology supplies the lamina area the LMA proxy divides by (it runs at slot 7, well before
    # this stage at slot 11), so no reordering is implied by depending on it.
    depends_on: tuple[str, ...] = ("leaf_segmenter", "landmark_detector", "morphology")
    owns_tables: tuple[str, ...] = ("leaf_petiole",)

    def _min_kpt_conf(self) -> float:
        return float(_get(self.cfg.stage(self.key), "min_kpt_conf", default=0.25))

    def _touch_dist(self) -> int:
        return int(_get(self.cfg.stage(self.key), "touch_dist_px", default=20))

    def collect_items(self, project) -> list[WorkItem]:
        return [
            WorkItem(sid, (project.db.leaf_instances(sid),
                           project.db.leaf_landmarks(sid),
                           project.db.detection_boxes(sid, "plant_detection"),
                           _area_by_leaf(project.db.leaf_morphology(sid))))
            for sid in project.db.specimens_with_rows("leaf_segmentation")
        ]

    def infer(self, item: WorkItem, model: Any) -> list[PetioleRow]:
        seg, landmarks, crop_boxes, area_by_leaf = item.payload
        thr = self._min_kpt_conf()
        touch = self._touch_dist()

        pet_polys, leaf_polys = _polys_by_key(seg)     # (det, parent)->[poly] ; (det, inst)->[poly]
        center_by = _centerline_by_detection(landmarks, thr)   # det -> {name: (x_crop, y_crop)}

        rows: list[PetioleRow] = []
        for r in seg:
            if str(_row_get(r, "cls_name", "")) != _LEAF:
                continue
            det = int(_row_get(r, "detection_id", -1))
            inst = int(_row_get(r, "instance_index", 0))
            if inst != 0:
                continue        # the landmark centerline is the crop's PRIMARY (instance-0) trace;
                                # don't apply it to a secondary leaf in a rare multi-leaf-per-crop crop
            pet = pet_polys.get((det, inst))
            box = crop_boxes.get(det)
            if not pet or not box:
                continue                               # no petiole mask (or no crop box) -> no row
            x1, y1 = int(round(box[0])), int(round(box[1]))
            cw, ch = max(1, int(round(box[2])) - x1), max(1, int(round(box[3])) - y1)

            def raster(polys):
                m = np.zeros((ch, cw), dtype=bool)
                for p in polys:
                    m |= polygon_mask(offset_polygon(p, -x1, -y1), (ch, cw))
                return m

            pet_mask = raster(pet)
            leaf_mask = raster(leaf_polys.get((det, inst), []))
            # Require the blade-junction anchor (lamina_base) so fraction 0 is the true junction and
            # "near base" sampling + length_px are meaningful; without it, don't measure a width.
            named = center_by.get(det, {})
            centerline = ([named[n] for n in _CENTERLINE_NAMES if n in named]
                          if "lamina_base" in named else [])       # crop coords

            pw = measure_petiole(pet_mask, leaf_mask, centerline, fractions=DEFAULT_FRACTIONS, touch_dist=touch)

            def to_working(s):
                return [[p[0] + x1, p[1] + y1] for p in s] if s else None

            leaf_id = int(_row_get(r, "leaf_id", -1))
            rows.append(PetioleRow(
                leaf_id=leaf_id, detection_id=det, instance_index=inst,
                width_px=pw.width_px, length_px=pw.length_px, n_samples=pw.n_samples,
                touches_leaf=pw.touches_leaf, measure_location=pw.measure_location,
                width_segment=to_working(pw.width_segment),
                sample_segments=[to_working(s) for s in pw.sample_segments],
                leaf_mass_per_area=_lma(pw.width_px, area_by_leaf.get(leaf_id)),
            ))
        return rows

    def persist(self, project, item: WorkItem, rows: list[PetioleRow]) -> None:
        project.db.record_leaf_petioles(item.specimen_id, rows)


# -- leaf mass per area ------------------------------------------------------------
# Royer petiole-width scaling: log10(LMA) = LMA_A + LMA_B * log10(PW^2 / A), LMA in g/m^2.
LMA_A: float = 3.070
LMA_B: float = 0.382


def _lma(width_px: float | None, area_px: float | None) -> float | None:
    """LMA proxy in g/m^2 from petiole width and lamina area, or ``None`` if not computable.

    ``PW^2 / A`` is a RATIO OF AREAS and therefore dimensionless, so pixels give exactly the same
    value as mm or cm (``(pw/cf)^2 / (area/cf^2) == pw^2 / area``). That is why this needs no ruler
    conversion factor and is filled in even for specimens with no CF -- unlike ``width_cm``.

    The logs are base 10, which is what the published fit uses: on a typical sheet (~2 mm petiole,
    ~50 cm^2 lamina) base 10 gives ~77 g/m^2, squarely in the range for real leaves, whereas natural
    logs would give ~1.4 g/m^2.
    """
    if not width_px or not area_px or width_px <= 0 or area_px <= 0:
        return None                                    # no petiole measured, or a degenerate mask
    return float(10.0 ** (LMA_A + LMA_B * math.log10((width_px * width_px) / area_px)))


def _area_by_leaf(morph_rows) -> dict[int, float]:
    """``{leaf_id: lamina area px}`` from leaf_morphology.

    Uses ``area_px`` -- the full silhouette INSIDE the outer boundary, holes included -- because the
    scaling relates petiole width to the area of lamina the petiole once supported; insect/decay
    holes are damage to a leaf that grew at its full size.
    """
    out: dict[int, float] = {}
    for r in morph_rows or ():
        leaf_id = _row_get(r, "leaf_id", None)
        area = _row_get(r, "area_px", None)
        if leaf_id is not None and area:
            out[int(leaf_id)] = float(area)
    return out


# -- helpers -----------------------------------------------------------------------
def _polys_by_key(seg) -> tuple[dict, dict]:
    """Petiole polys keyed by (detection_id, parent leaf instance); Leaf polys by (det, instance)."""
    pet: dict[tuple[int, int], list] = {}
    leaf: dict[tuple[int, int], list] = {}
    for r in seg:
        cls = str(_row_get(r, "cls_name", ""))
        if cls not in (_LEAF, _PETIOLE):
            continue
        if str(_row_get(r, "mask_format", "polygon_xy")) != "polygon_xy":
            continue
        data = _row_get(r, "mask_data")
        if not data:
            continue
        try:
            poly = decode_polygon(str(data))
        except Exception:
            continue
        det = int(_row_get(r, "detection_id", -1))
        if cls == _PETIOLE:
            parent = _row_get(r, "parent_instance_index", None)
            key = (det, int(parent) if parent is not None else 0)
            pet.setdefault(key, []).append(poly)
        else:
            leaf.setdefault((det, int(_row_get(r, "instance_index", 0))), []).append(poly)
    return pet, leaf


def _centerline_by_detection(landmarks, thr: float) -> dict:
    center: dict[int, dict] = {}
    for r in landmarks:
        if int(_row_get(r, "instance_index", 0)) != 0:
            continue
        conf = _row_get(r, "conf", None)
        if conf is None or float(conf) < thr:
            continue
        xc, yc = _row_get(r, "x_crop", None), _row_get(r, "y_crop", None)
        if xc is None or yc is None:
            continue
        center.setdefault(int(_row_get(r, "detection_id", -1)), {})[str(_row_get(r, "kpt_name", ""))] = (
            float(xc), float(yc))
    return center


def _row_get(row: Any, key: str, default: Any = None) -> Any:
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


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        try:
            v = obj.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    v = getattr(obj, key, default)
    return default if v is None else v
