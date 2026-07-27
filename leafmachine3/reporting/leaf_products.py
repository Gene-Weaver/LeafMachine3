"""Per-leaf product rendering (Original + Oriented trees) for the Reporter.

The five highest-value leaf outputs, each produced in a non-oriented (``Original``) and an upright
(``Oriented``, rotated tip-up) form:

    bbox                 the Plant_Detector RGB leaf crop (NOT content-fitted)
    lamina_mask          binary Leaf mask (holes removed), content-fitted
    lamina_petiole_mask  binary Leaf + Petiole mask (holes removed), content-fitted
    lamina_rgb           RGB cutout of the lamina (background removed), content-fitted
    lamina_petiole_rgb   RGB cutout of the lamina + petiole, content-fitted

Everything except the bbox crop is cropped tight to its mask ("fitted to minimize blank space").
The ``lamina_petiole_*`` products are only produced when a petiole mask exists for that leaf.
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
    friendly: str        # filename friendly class (leaf / lamina / laminaPetiole)
    kind: str            # "bbox" | "mask" | "cutout"
    needs_petiole: bool
    is_mask: bool        # png (mask) vs jpg (rgb)


PRODUCTS: tuple[LeafProduct, ...] = (
    LeafProduct("bbox", "Leaf_BBox", "BBOX", "leaf", "bbox", False, False),
    LeafProduct("lamina_mask", "Lamina_Mask", "SEG", "lamina", "mask", False, True),
    LeafProduct("lamina_petiole_mask", "LaminaPetiole_Mask", "SEG", "laminaPetiole", "mask", True, True),
    LeafProduct("lamina_rgb", "Lamina_RGB", "RGB", "lamina", "cutout", False, False),
    LeafProduct("lamina_petiole_rgb", "LaminaPetiole_RGB", "RGB", "laminaPetiole", "cutout", True, False),
)
PRODUCT_KEYS: frozenset[str] = frozenset(p.key for p in PRODUCTS)


def render_leaf_products(
    crop_bgr: np.ndarray,
    lamina_mask: np.ndarray,
    laminapet_mask: Optional[np.ndarray],
    *,
    angle_cw: Optional[float] = None,
    bg: int = 0,
    want: Optional[set[str]] = None,
    pad: int = 0,
) -> dict[str, np.ndarray]:
    """Render the requested products from a leaf crop + its masks (all in the crop frame).

    ``angle_cw`` None => Original; a value => rotate the crop and masks upright first. Masks are
    boolean arrays the same H×W as ``crop_bgr``. Returns ``{product_key: image}`` (masks are uint8
    0/255). ``lamina_petiole_*`` are omitted when ``laminapet_mask`` is None.
    """
    want = PRODUCT_KEYS if want is None else set(want)
    crop, lam, lampet = crop_bgr, lamina_mask, laminapet_mask
    if angle_cw is not None:                                   # rotate crop + masks upright together
        crop = rotate_image(crop_bgr, angle_cw, bg=bg)
        lam = rotate_image(np.asarray(lamina_mask, np.uint8), angle_cw, bg=0, nearest=True) > 0
        lampet = (
            None if laminapet_mask is None
            else rotate_image(np.asarray(laminapet_mask, np.uint8), angle_cw, bg=0, nearest=True) > 0
        )

    lam_box = mask_bbox(lam)
    lampet_box = mask_bbox(lampet) if lampet is not None else None

    out: dict[str, np.ndarray] = {}
    for p in PRODUCTS:
        if p.key not in want:
            continue
        petiole_variant = p.friendly == "laminaPetiole"
        if p.needs_petiole and lampet is None:
            continue
        mask = lampet if petiole_variant else lam
        box = lampet_box if petiole_variant else lam_box
        if p.kind == "bbox":
            out[p.key] = crop                                  # NOT fitted (keeps the rectangle)
        elif box is None:
            continue                                           # empty mask -> nothing to fit
        elif p.kind == "mask":
            out[p.key] = crop_to_box(np.asarray(mask, np.uint8) * 255, box, pad)
        elif p.kind == "cutout":
            out[p.key] = crop_to_box(composite(crop, mask, bg=bg), box, pad)
    return out
