"""Same-class duplicate-box suppression, shared by the ArchivalDetector and PlantDetector.

YOLO26 runs an end-to-end (NMS-free) postprocess, so two boxes for the SAME object (e.g. one
ruler detected twice) can both survive into the results. This does a light, class-aware pass that
rejects the lower-confidence box of any same-class pair whose overlap is >= a threshold, so only one
copy of each detection flows downstream.

Overlap metric:
  * ``"min"`` (default) -- intersection / area of the SMALLER box. Reads as "the smaller box is
    >=X% overlapped", and crucially catches a small box fully NESTED inside a larger one (whose IoU
    can be far below the threshold) -- the common duplicate shape from an NMS-free detector.
  * ``"iou"`` -- classic intersection / union.

Pure-Python (no numpy/cv2), so it stays trivially reusable and unit-testable. Inputs are any objects
exposing ``.xyxy`` (x1, y1, x2, y2), ``.conf``, and a class key (``.cls_id`` or ``.cls_name``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

__all__ = ["BoxDecision", "suppress_duplicate_boxes", "box_overlap"]


@dataclass(frozen=True)
class BoxDecision:
    """Outcome for one input box. ``keeper_index``/``overlap`` are set only when suppressed."""
    suppressed: bool = False
    keeper_index: Optional[int] = None      # index (in the input list) of the box that suppressed this one
    overlap: Optional[float] = None         # the overlap fraction that triggered suppression

_KEEP = BoxDecision()


def box_overlap(a, b, metric: str = "min") -> float:
    """Overlap of two boxes ``(x1, y1, x2, y2)`` by ``metric`` ('min' = intersection/smaller-area, or 'iou')."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if metric == "iou":
        union = area_a + area_b - inter
        return inter / union if union > 0.0 else 0.0
    denom = min(area_a, area_b)              # "min": containment of the smaller box
    return inter / denom if denom > 0.0 else 0.0


def _class_key(det: Any):
    cid = getattr(det, "cls_id", None)
    return cid if cid is not None else getattr(det, "cls_name", None)


def suppress_duplicate_boxes(
    dets: Sequence[Any], *, overlap: float = 0.95, metric: str = "min",
) -> list[BoxDecision]:
    """Greedy, class-aware duplicate suppression.

    Keep the highest-confidence box; reject any lower-confidence box of the SAME class that overlaps
    an already-kept box by ``>= overlap``. Returns one :class:`BoxDecision` per input box, in the
    input order. Ties (equal confidence) keep the earlier box deterministically.
    """
    n = len(dets)
    decisions: list[BoxDecision] = [_KEEP] * n
    if n < 2 or overlap is None:
        return decisions
    # process high-confidence first; stable order breaks confidence ties by original position
    order = sorted(range(n), key=lambda i: float(getattr(dets[i], "conf", 0.0) or 0.0), reverse=True)
    kept: list[int] = []
    for i in order:
        ci = _class_key(dets[i])
        best_k, best_ov = None, 0.0
        for k in kept:
            if _class_key(dets[k]) != ci:
                continue
            ov = box_overlap(dets[i].xyxy, dets[k].xyxy, metric)
            if ov >= overlap and ov > best_ov:
                best_k, best_ov = k, ov
        if best_k is None:
            kept.append(i)
        else:
            decisions[i] = BoxDecision(suppressed=True, keeper_index=best_k, overlap=round(best_ov, 4))
    return decisions
