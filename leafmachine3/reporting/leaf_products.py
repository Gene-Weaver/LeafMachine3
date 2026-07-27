"""Per-leaf product rendering (Original + Oriented trees) for the Reporter.

The highest-value leaf outputs, each produced in a non-oriented (``Original``) and an upright
(``Oriented``, rotated tip-up) form:

    bbox                 the Plant_Detector RGB leaf crop (NOT content-fitted)
    lamina_mask          binary Leaf mask, holes removed, content-fitted
    lamina_petiole_mask  binary Leaf + Petiole mask, holes removed, content-fitted
    lamina_holes_mask    binary solid Leaf silhouette (holes FILLED in), content-fitted
    lamina_rgb           RGB cutout of the lamina (background removed), content-fitted
    lamina_petiole_rgb   RGB cutout of the lamina + petiole, content-fitted
    lamina_holes_rgb     RGB cutout of the lamina WITH its holes painted a flat color, content-fitted

Everything except the bbox crop is cropped tight to its mask ("fitted"). ``lamina_petiole_*`` are
only produced when a petiole mask exists. In ``lamina_holes_rgb`` the leaf tissue keeps its original
pixels and the holes are filled with ``hole_color`` (default RGB ``(10, 10, 10)``, distinct from the
black background) so downstream users can recover the holes by a simple color threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from leafmachine3.core.imaging import composite, crop_to_box, mask_bbox, rotate_image


@dataclass(frozen=True)
class LeafProduct:
    key: str
    folder: str          # subfolder under Original/ and Oriented/
    prefix: str          # filename PREFIX (BBOX / SEG / RGB)
    friendly: str        # filename friendly class (leaf / lamina / laminaPetiole / laminaHoles)
    kind: str            # "bbox" | "mask" | "cutout" | "cutout_holes"
    mask_key: Optional[str]   # which mask defines this product's fit box ("lamina"/"laminapet"/"silhouette")
    needs_petiole: bool
    is_mask: bool        # png (mask) vs jpg (rgb)


PRODUCTS: tuple[LeafProduct, ...] = (
    LeafProduct("bbox", "Leaf_BBox", "BBOX", "leaf", "bbox", None, False, False),
    LeafProduct("lamina_mask", "Lamina_Mask", "SEG", "lamina", "mask", "lamina", False, True),
    LeafProduct("lamina_petiole_mask", "LaminaPetiole_Mask", "SEG", "laminaPetiole", "mask", "laminapet", True, True),
    LeafProduct("lamina_holes_mask", "Lamina_Holes_Mask", "SEG", "laminaHoles", "mask", "silhouette", False, True),
    LeafProduct("lamina_rgb", "Lamina_RGB", "RGB", "lamina", "cutout", "lamina", False, False),
    LeafProduct("lamina_petiole_rgb", "LaminaPetiole_RGB", "RGB", "laminaPetiole", "cutout", "laminapet", True, False),
    LeafProduct("lamina_holes_rgb", "Lamina_Holes_RGB", "RGB", "laminaHoles", "cutout_holes", "silhouette", False, False),
)
PRODUCT_KEYS: frozenset[str] = frozenset(p.key for p in PRODUCTS)


def render_leaf_products(
    crop_bgr: np.ndarray,
    masks: dict[str, Optional[np.ndarray]],
    *,
    angle_cw: Optional[float] = None,
    bg: int = 0,
    want: Optional[set[str]] = None,
    pad: int = 0,
    hole_color: tuple[int, int, int] = (10, 10, 10),
) -> dict[str, np.ndarray]:
    """Render the requested products from a leaf crop + its masks (all in the crop frame).

    ``masks`` holds boolean arrays the size of ``crop_bgr``: ``lamina`` (Leaf minus holes),
    ``laminapet`` (Leaf+Petiole minus holes, or None), ``silhouette`` (Leaf with holes filled), and
    ``hole`` (the holes). ``angle_cw`` None => Original; a value => rotate crop + masks upright first.
    Returns ``{product_key: image}`` (masks are uint8 0/255). ``lamina_petiole_*`` are omitted when
    ``laminapet`` is None. ``hole_color`` is BGR/RGB-symmetric grey for the ``*_holes`` cutout.
    """
    want = PRODUCT_KEYS if want is None else set(want)

    def _rot_mask(m):
        if m is None:
            return None
        return rotate_image(np.asarray(m, np.uint8), angle_cw, bg=0, nearest=True) > 0

    crop = crop_bgr
    m = dict(masks)
    if angle_cw is not None:                                   # rotate crop + every mask together
        crop = rotate_image(crop_bgr, angle_cw, bg=bg)
        m = {k: _rot_mask(v) for k, v in masks.items()}

    boxes = {k: (mask_bbox(m.get(k)) if m.get(k) is not None else None)
             for k in ("lamina", "laminapet", "silhouette")}

    out: dict[str, np.ndarray] = {}
    for p in PRODUCTS:
        if p.key not in want:
            continue
        if p.needs_petiole and m.get("laminapet") is None:
            continue
        if p.kind == "bbox":
            out[p.key] = crop                                  # NOT fitted (keeps the rectangle)
            continue
        box = boxes.get(p.mask_key)
        if box is None:
            continue                                           # empty mask -> nothing to fit
        if p.kind == "mask":
            out[p.key] = crop_to_box(np.asarray(m[p.mask_key], np.uint8) * 255, box, pad)
        elif p.kind == "cutout":
            out[p.key] = crop_to_box(composite(crop, m[p.mask_key], bg=bg), box, pad)
        elif p.kind == "cutout_holes":
            img = np.full_like(crop, bg)
            lamina = m.get("lamina")
            if lamina is not None:
                img[lamina] = crop[lamina]                     # leaf tissue keeps original pixels
            hole = m.get("hole")
            if hole is not None:
                img[hole & m["silhouette"]] = hole_color       # holes painted a flat, thresholdable color
            out[p.key] = crop_to_box(img, box, pad)
    return out
