"""Momocs / Momocs2 export primitives: leaf mask -> Momocs-ready image + outline + Momit JSON.

Pure functions (no DB, no config) used by :mod:`leafmachine3.modules.momocs`. Every convention
here was checked against the R packages themselves (Momocs 1.5.0, Momocs2 0.1.0, Momit 0.1.0) on
real LM3 masks; each one closes a failure that happens WITHOUT an error on the R side:

* **Black leaf on white, padded.** Momocs ``import_jpg`` reads JPEG only and treats dark pixels as
  the shape, so LM3's white-on-black masks trace the background. Momit ``from_mask`` flood-fills
  the background from the top-left pixel, so a leaf touching the image edge (LM3 leaf products are
  cropped tight) splits the background and the wrong region is traced. A white border fixes both.
* **Holes always filled.** Momocs walks straight down from the image center to find the first
  edge; an open hole on that line is traced instead of the leaf (0.8% of its area in testing). Both
  packages trace only the outer outline, so filling loses nothing.
* **One connected piece.** Both importers trace a single shape; a stray fragment can be picked up
  instead of the leaf. The largest component is kept.
* **y increases upward.** Coordinates in image order (y down) load as mirror-image leaves.
* **Outline starts at the base and runs clockwise.** On a tip-up (Leaf_Oriented) leaf that is where
  Momocs ``import_jpg`` starts too, so the image route and the coordinate route agree.
* **Never an empty image.** ``import_jpg`` loops forever on an image with no dark pixels, so a
  degenerate mask yields ``None`` and no file.

The Momit JSON layout mirrors what ``Momit::to_json`` writes, so ``Momit::from_json`` restores a
Momocs2 table with an ``out`` coordinate column, and ``Momit::to_Momocs`` turns that table into a
legacy Momocs ``Out`` with the remaining columns as its ``$fac``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from leafmachine3.core.leaf_outline import extract_contour

#: The Momit JSON format version this writer targets (``metadata.version`` in ``Momit::to_json``).
MOMIT_JSON_VERSION = "0.1.0"


@dataclass(frozen=True)
class MomocsProduct:
    """Which Reporter leaf product Momocs reads, and how its own files are labeled."""

    mask_includes: str      # lamina | lamina_petiole   (fac column + DB column)
    product_key: str        # report.leaf_products key the Reporter must export
    folder: str             # reports/<tree>/<folder>/
    seg_friendly: str       # the Reporter's SEG-<friendly> filename token
    momocs_friendly: str    # this stage's MOMOCS-<friendly> filename token


def momocs_product_for(include_petiole: bool) -> MomocsProduct:
    """The holes-FILLED leaf product to read (see the module notes on why holes are always filled)."""
    if include_petiole:
        return MomocsProduct("lamina_petiole", "lamina_petiole_holes_mask", "LaminaPetiole_Holes_Mask",
                             "laminaPetioleHoles", "laminaPetiole")
    return MomocsProduct("lamina", "lamina_holes_mask", "Lamina_Holes_Mask", "laminaHoles", "lamina")


def prepare_mask(mask, *, pad_px: int = 10, fill_holes: bool = True,
                 largest_only: bool = True) -> Optional[np.ndarray]:
    """Binary leaf mask -> padded boolean mask ready for :func:`momocs_image` / :func:`momocs_outline`.

    ``mask`` is any array whose foreground is > 0 (an LM3 leaf-product PNG read as grayscale).
    Returns ``None`` when nothing usable is left (empty mask, or fewer than 3 boundary points).
    """
    import cv2

    fg = (np.asarray(mask) > 127 if np.asarray(mask).dtype == np.uint8 else np.asarray(mask) > 0)
    fg = fg.astype(np.uint8)
    if not fg.any():
        return None
    if largest_only:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n > 2:
            fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    if fill_holes:
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        fg = cv2.drawContours(np.zeros_like(fg), cnts, -1, 1, thickness=cv2.FILLED)
    pad = max(0, int(pad_px))
    out = np.pad(fg.astype(bool), pad, constant_values=False)
    return out if extract_contour(out) is not None else None


def momocs_image(padded: np.ndarray) -> np.ndarray:
    """Padded boolean mask -> uint8 image, leaf BLACK (0) on WHITE (255): what ``import_jpg`` expects."""
    return np.where(np.asarray(padded, bool), 0, 255).astype(np.uint8)


def momocs_outline(padded: np.ndarray, n_points: int = 0) -> Optional[np.ndarray]:
    """Outer outline of a padded mask in Momocs conventions, as an ``(N, 2)`` float array.

    Frame: the padded image's pixels, ``x = column``, ``y = (height - 1) - row`` (y up), so the
    coordinates line up with the exported image. The first point is the lowest outline point (the
    base on a tip-up leaf; ties go to the point nearest the outline's mean x) and the outline runs
    clockwise. ``n_points > 0`` resamples it to that many points evenly spaced along its length;
    ``0`` keeps every boundary pixel, which is what Momit ``from_mask`` returns.

    The outline runs through boundary-pixel CENTERS, while Momocs ``import_jpg`` traces about one
    pixel further out, so the two routes differ by a band ~1 px wide: under 2% of the area on
    leaves of ordinary size, but up to half the area on 10 px specks.
    """
    c = extract_contour(padded)
    if c is None:
        return None
    h = np.asarray(padded).shape[0]
    xy = np.column_stack([c[:, 0], (h - 1) - c[:, 1]]).astype(float)
    if signed_area(xy) > 0:                       # counter-clockwise in a y-up frame -> reverse
        xy = xy[::-1]
    if n_points and n_points > 0:
        xy = resample_closed(xy, int(n_points))
    low = np.flatnonzero(xy[:, 1] <= xy[:, 1].min() + 1e-9)
    start = int(low[np.argmin(np.abs(xy[low, 0] - xy[:, 0].mean()))])
    return np.roll(xy, -start, axis=0)


def signed_area(xy: np.ndarray) -> float:
    """Shoelace signed area: positive = counter-clockwise in a y-up frame."""
    x, y = xy[:, 0], xy[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)) / 2.0


def resample_closed(xy: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed outline to ``n`` points evenly spaced along its length (start point kept)."""
    closed = np.vstack([xy, xy[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


# -- Momit JSON -----------------------------------------------------------------------------------
def _col_meta(values: list) -> dict:
    """Momit column metadata for a plain (non-coordinate) column, inferred from its values."""
    vals = [v for v in values if v is not None]
    if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
        return {"col_class": "logical", "type": "logical"}
    if vals and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in vals):
        return {"col_class": "integer", "type": "integer"}
    if vals and all(isinstance(v, (int, float, np.integer, np.floating)) for v in vals):
        return {"col_class": "numeric", "type": "double"}
    return {"col_class": "character", "type": "character"}


def _plain(v: Any) -> Any:
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def momit_document(records: Iterable[dict], *, coo_key: str = "coo") -> dict:
    """Assemble a ``Momit::from_json`` document from per-leaf records.

    Each record holds an ``id``, the outline under ``coo_key`` (``(N, 2)`` array or list of pairs)
    and any number of plain fac columns. Column order follows the first record; ``id`` and the
    outline come first, as ``Momit::to_json`` writes them for a Momocs2 table.
    """
    records = list(records)
    cols: list[str] = []
    for r in records:
        for k in r:
            if k not in cols:
                cols.append(k)
    cols = ["id", coo_key] + [c for c in cols if c not in ("id", coo_key)]
    data = []
    for r in records:
        row = {}
        for c in cols:
            v = r.get(c)
            if c == coo_key:
                v = [[_plain(a), _plain(b)] for a, b in np.asarray(v, float).tolist()] if v is not None else None
            else:
                v = _plain(v)
            row[c] = v
        data.append(row)
    meta_cols: dict[str, dict] = {}
    for c in cols:
        if c == coo_key:
            first = next((r[c] for r in data if r[c]), [])
            meta_cols[c] = {"col_class": ["out", "coo", "list"], "elem_class": ["xy", "matrix", "array"],
                            "type": "matrix", "dims": [len(first), 2]}
        else:
            meta_cols[c] = _col_meta([r[c] for r in data])
    return {
        "metadata": {"version": MOMIT_JSON_VERSION, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "n_rows": len(data), "columns": meta_cols},
        "data": data,
    }


def merge_momit_documents(docs: Iterable[dict]) -> dict:
    """Concatenate several Momit documents (e.g. one per sheet) into one, recomputing the metadata."""
    rows: list[dict] = []
    for d in docs:
        rows.extend(d.get("data", []))
    if not rows:
        return momit_document([])
    return momit_document(rows)
