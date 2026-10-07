"""Load one leaf into the ORIENTED-mask pixel frame: silhouette + landmarks, exactly aligned.

The Reporter's ``Leaf_Oriented`` products are produced by a three-step chain (crop -> rotate ->
content-fit). Landmarks are stored in WORKING (parent-sheet) coords, so to draw a midvein on an
oriented mask the same chain has to be replayed on the points. This module replays it exactly and
rebuilds the mask from the stored polygons rather than reading the PNG, so mask and landmarks are
guaranteed to share one frame by construction.

Verified against a real run: the rebuilt silhouette matches the saved
``Leaf_Oriented/Lamina_Holes_Mask/*.png`` at **IoU 1.000000**, and ``lamina_tip`` lands at the top
edge of that mask with ``lamina_base`` at the bottom.

Frame convention (as everywhere in LM3): image coords, **y DOWN**, so the tip is at MIN y. "Left"
and "right" are the viewer's left/right of the tip-up mask, which is consistent across leaves
precisely because the mask is oriented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from leafmachine3.core.imaging import (
    crop_to_box,
    decode_polygon,
    mask_bbox,
    offset_polygon,
    polygon_mask,
    rotate_image,
)
from leafmachine3.core.landmarks import MIDVEIN_N

_LEAF, _HOLE, _PETIOLE = "Leaf", "Hole", "Petiole"

# The midvein trace as the pose model emits it: tip -> base. `lamina_tip` and `lamina_base` cap it.
MIDVEIN_NAMES: tuple[str, ...] = (
    "lamina_tip", *[f"midvein_{i}" for i in range(MIDVEIN_N)], "lamina_base",
)


@dataclass
class OrientedLeaf:
    """One leaf in the oriented-mask pixel frame (y down, tip up)."""

    specimen_id: int
    leaf_id: int
    detection_id: int
    stem: str
    silhouette: np.ndarray          # (H, W) bool -- lamina outline, holes FILLED
    lamina: np.ndarray              # (H, W) bool -- silhouette minus holes (tissue only)
    holes: np.ndarray               # (H, W) bool
    petiole: Optional[np.ndarray]   # (H, W) bool in the same frame, or None
    kpts: dict[str, tuple[float, float]]     # name -> (x, y) in this frame, conf-filtered
    kpt_conf: dict[str, float]
    angle_cw: float
    det_box: tuple[int, int, int, int]
    crop_shape: tuple[int, int]     # (h, w) of the pre-rotation crop
    truncated: bool                 # the det box was clipped by the sheet edge

    @property
    def shape(self) -> tuple[int, int]:
        return self.silhouette.shape

    def midvein(self) -> Optional[np.ndarray]:
        """(N, 2) tip->base midvein polyline from the available trace keypoints, or None."""
        pts = [self.kpts[n] for n in MIDVEIN_NAMES if n in self.kpts]
        return np.asarray(pts, float) if len(pts) >= 3 else None

    def tip_base(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if "lamina_tip" in self.kpts and "lamina_base" in self.kpts:
            return np.asarray(self.kpts["lamina_tip"], float), np.asarray(self.kpts["lamina_base"], float)
        return None


def oriented_affine(crop_shape: tuple[int, int], angle_cw: float) -> tuple[np.ndarray, tuple[int, int]]:
    """The exact affine ``core.imaging.rotate_image`` applies, plus the expanded canvas size.

    Returned as ``(M, (nh, nw))`` where ``M`` is 2x3 and maps crop-frame points to rotated-canvas
    points. Mirrors rotate_image line for line so points and pixels cannot diverge.
    """
    import cv2

    ch, cw = crop_shape
    cx, cy = cw / 2.0, ch / 2.0
    m = cv2.getRotationMatrix2D((cx, cy), -float(angle_cw), 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(round(ch * sin + cw * cos))
    nh = int(round(ch * cos + cw * sin))
    m[0, 2] += nw / 2.0 - cx
    m[1, 2] += nh / 2.0 - cy
    return m, (nh, nw)


def apply_affine(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine to an (N, 2) point array."""
    p = np.asarray(pts, float).reshape(-1, 2)
    return p @ m[:, :2].T + m[:, 2]


def load_leaves(db, specimen_id: int, *, min_kpt_conf: float = 0.25,
                require_orientation: bool = True) -> list[OrientedLeaf]:
    """Every oriented ``Leaf`` instance of one specimen, in its own oriented-mask frame."""
    spec = db.get_specimen(specimen_id)
    if spec is None:
        return []
    stem = str(spec["image_stem"])
    W, H = int(spec["width"]), int(spec["height"])

    seg = db.leaf_instances(specimen_id)
    boxes = db.detection_boxes(specimen_id, "plant_detection")
    morph = {int(r["detection_id"]): r for r in db.leaf_morphology(specimen_id)}

    kpt_by_det: dict[int, dict[str, tuple[float, float, float]]] = {}
    for r in db.leaf_landmarks(specimen_id):
        if int(r["instance_index"] or 0) != 0:
            continue
        conf = r["conf"]
        if conf is None or float(conf) < min_kpt_conf:
            continue
        if r["x"] is None or r["y"] is None:
            continue
        kpt_by_det.setdefault(int(r["detection_id"]), {})[str(r["kpt_name"])] = (
            float(r["x"]), float(r["y"]), float(conf))

    polys: dict[tuple[int, str], list] = {}
    leaf_ids: dict[int, int] = {}
    for r in seg:
        cls = str(r["cls_name"])
        if cls not in (_LEAF, _HOLE, _PETIOLE):
            continue
        if str(r["mask_format"] or "polygon_xy") != "polygon_xy" or not r["mask_data"]:
            continue
        # instance 0 only: the landmark trace belongs to the crop's primary leaf (mirrors
        # petiole_width's reasoning for multi-leaf crops).
        if cls == _LEAF and int(r["instance_index"] or 0) != 0:
            continue
        try:
            poly = decode_polygon(str(r["mask_data"]))
        except Exception:
            continue
        det = int(r["detection_id"])
        polys.setdefault((det, cls), []).append(poly)
        if cls == _LEAF:
            leaf_ids[det] = int(r["leaf_id"])

    out: list[OrientedLeaf] = []
    for det, leaf_id in sorted(leaf_ids.items()):
        m = morph.get(det)
        box = boxes.get(det)
        if box is None or m is None:
            continue
        ok = bool(m["oriented_leaf_success"] or 0)
        angle = m["oriented_leaf_rotation_angle_degreesCW"]
        if require_orientation and not (ok and angle is not None):
            continue
        angle = float(angle or 0.0)

        # --- crop frame (exactly as reporter._export_leaf_products) ---
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        ch, cw = cy2 - cy1, cx2 - cx1
        if ch <= 0 or cw <= 0:
            continue

        def raster(key: str) -> np.ndarray:
            acc = np.zeros((ch, cw), bool)
            for p in polys.get((det, key), []):
                acc |= polygon_mask(offset_polygon(p, -cx1, -cy1), (ch, cw))
            return acc

        sil = raster(_LEAF)
        if not sil.any():
            continue
        hole = raster(_HOLE)
        pet = raster(_PETIOLE)

        # --- rotate, then content-fit on the SILHOUETTE (the Lamina_Holes_Mask product's box) ---
        def rot(mask: np.ndarray) -> np.ndarray:
            return rotate_image(mask.astype(np.uint8), angle, bg=0, nearest=True) > 0

        rsil, rhole, rpet = rot(sil), rot(hole), rot(pet)
        fit = mask_bbox(rsil)
        if fit is None:
            continue

        def cut(mask: np.ndarray) -> np.ndarray:
            return crop_to_box(mask.astype(np.uint8), fit, 0) > 0

        aff, _canvas = oriented_affine((ch, cw), angle)
        named = kpt_by_det.get(det, {})
        kpts: dict[str, tuple[float, float]] = {}
        confs: dict[str, float] = {}
        if named:
            names = list(named)
            src = np.array([[named[n][0] - cx1, named[n][1] - cy1] for n in names], float)
            dst = apply_affine(aff, src) - np.array([fit[0], fit[1]], float)
            for n, p in zip(names, dst):
                kpts[n] = (float(p[0]), float(p[1]))
                confs[n] = named[n][2]

        sil_c = cut(rsil)
        out.append(OrientedLeaf(
            specimen_id=int(specimen_id), leaf_id=leaf_id, detection_id=det, stem=stem,
            silhouette=sil_c, lamina=sil_c & ~cut(rhole), holes=cut(rhole) & sil_c,
            petiole=(cut(rpet) if rpet.any() else None),
            kpts=kpts, kpt_conf=confs, angle_cw=angle, det_box=(x1, y1, x2, y2),
            crop_shape=(ch, cw), truncated=bool(x1 < 0 or y1 < 0 or x2 > W or y2 > H),
        ))
    return out
