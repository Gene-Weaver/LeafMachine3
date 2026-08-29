"""leafmachine3.core.bilateral -- bilateral-symmetry geometry, metrics and the archetype score.

Shared by the ``bilateral_symmetry`` STAGE (which computes and persists) and the REPORTER (which
redraws the QC panel), exactly as ``core.petiole`` is shared by ``PetioleWidth`` and
``Reporter._export_petiole_overlays``. Nothing here touches the DB or writes files.

What it measures, and why the midvein axis
------------------------------------------
Symmetry is measured about the TRACED MIDVEIN, not the straight tip->base chord the published
methods must use (they have only a silhouette). On the experiment cohort the chord overstated
asymmetry for 62% of leaves -- median ``si_a`` 0.151 vs 0.142, Dice 0.857 vs 0.895 -- and midvein
sinuosity predicted the size of that penalty (Spearman +0.47, p=4e-07). A curved midvein treated as
straight manufactures asymmetry that is not there.

Read the score as a QUALITY indicator first. On a cohort of already-clean masks the composite tracks
``si_a`` at Spearman -0.97, because integrity/completeness/trace saturate at 1.0 for 94/99/76% of
leaves -- symmetry is the only term that varies, so it IS the ranking. That is a fact about the
data, not a validation of the composite. In practice low symmetry has flagged bad LANDMARK TRACES at
least as often as bad masks: the lowest-Dice leaf in the cohort was not asymmetric, its midvein had
been traced along the margin. A leaf can also be genuinely asymmetric and perfectly masked (oblique
bases are normal in many taxa), so low symmetry is evidence of a bad mask only alongside the
structural diagnostics -- which is what the QC panel is for.

Every constant here is a first proposal to be tuned against the QC images, not an established value.

Frame conventions (all three must hold or the metrics are silently wrong)
------------------------------------------------------------------------
* Image coords, **y DOWN**; every mask is tip-up oriented, so the tip is at MIN y.
* ``s`` = normalized arclength, **0 at the tip, 1 at the base**.
* ``u`` = signed perpendicular offset in px, **positive to the viewer's LEFT**. Because the mask is
  oriented, "left" is the same physical side for every leaf.

Pixels are assigned to their NEAREST point on the axis (a Voronoi partition of the polyline), not to
equal-arclength perpendicular strips -- strips along a curved axis overlap on the inside of a bend
and gap on the outside. The Voronoi partition tiles the lamina exactly once, so
``area_l.sum() + area_r.sum() == mask.sum()`` exactly. On-axis pixels (``u == 0``) split half to each
side for areas and are excluded from margins; lumping them on one side made a perfectly
mirror-symmetric mask report ``a_star = -0.009`` instead of 0.

The shape measured is the **holes-filled silhouette** (``Lamina_Holes_Mask``): the question is
whether the OUTLINE mirrors, and an insect hole is not part of that outline.

Degenerate input yields ``nan``, never ``0.0`` -- a 1-px mask must not read as "perfectly
symmetric", and a leaf whose symmetry could not be measured must never outrank one where it was.

Ported from ``modules/experiments/bilateral_symmetry/`` (geometry, axes, metrics, shape, quality,
and run.py's gate policy). The bodies are carried over unchanged; that code is the verified part.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np

from leafmachine3.core.imaging import (
    crop_to_box,
    decode_polygon,
    encode_polygon,
    mask_bbox,
    offset_polygon,
    polygon_mask,
    rotate_image,
)
from leafmachine3.core.landmarks import MIDVEIN_N

_NAN = float("nan")
_LEAF, _HOLE, _PETIOLE = "Leaf", "Hole", "Petiole"

#: |u| below this counts as ON the axis (see ``profiles``). Pixel coords are O(1e2), so double
#: precision noise is ~1e-12 px; a real offset is never smaller than a milli-pixel.
ON_AXIS_EPS = 1e-9

#: Minimum lamina pixels per arclength bin. ``si_a`` averages a per-bin ratio, and a ratio built
#: from a handful of pixels is noise with a POSITIVE expected absolute value -- so a fixed bin count
#: makes small leaves look asymmetric. Measured on 538 real leaves at a fixed 200 bins, si_a
#: correlated with 1/sqrt(area) at Spearman +0.84 (median si_a 0.55 for leaves under 10k px against
#: 0.08 above 80k): the metric was mostly reporting SIZE. Choosing the bin count so each bin holds
#: at least this many pixels cuts that to +0.42 and brings the size buckets into line
#: (0.18/0.11/0.11/0.08). The experiment never saw this -- its cohort was 105 leaves with a median
#: lamina of 72,882 px, where a fixed 200 bins is already ~360 px/bin.
MIN_PX_PER_BIN = 500
MIN_BINS = 12


# ==========================================================================
# ported from experiments/bilateral_symmetry/geometry.py
# ==========================================================================

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


def leaf_inputs(db, specimen_id: int, *, min_kpt_conf: float = 0.25,
                require_orientation: bool = True) -> list[dict]:
    """The DB half of :func:`load_leaves`: small, picklable ingredients, one dict per leaf.

    Split out because LM3 worker processes are PURE -- they never touch the DB -- so the parent
    reads, and the worker rasterizes. Shipping the ingredients (encoded polygon strings, the box,
    the angle, the keypoints) instead of the built masks keeps the WorkItem payload at ~kB and puts
    the expensive rasterize + rotate inside the pool instead of serializing it in the parent.
    """
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

    out: list[dict] = []
    for det, leaf_id in sorted(leaf_ids.items()):
        m = morph.get(det)
        box = boxes.get(det)
        if box is None or m is None:
            continue
        ok = bool(m["oriented_leaf_success"] or 0)
        angle = m["oriented_leaf_rotation_angle_degreesCW"]
        if require_orientation and not (ok and angle is not None):
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        out.append({
            "specimen_id": int(specimen_id), "leaf_id": int(leaf_id), "detection_id": int(det),
            "stem": stem, "sheet_wh": (W, H), "det_box": (x1, y1, x2, y2),
            "angle_cw": float(angle or 0.0),
            "polys": {k[1]: [encode_polygon(pp) for pp in v]
                      for k, v in polys.items() if k[0] == det},
            "kpts": kpt_by_det.get(det, {}),
        })
    return out


def build_leaf(inp: dict) -> Optional[OrientedLeaf]:
    """The PURE half of :func:`load_leaves`: ingredients -> one leaf in its oriented-mask frame.

    Safe to call inside a worker process. Replays the Reporter's crop -> rotate -> content-fit chain
    on both the polygons and the keypoints, so mask and landmarks share one frame by construction.
    """
    W, H = inp["sheet_wh"]
    x1, y1, x2, y2 = inp["det_box"]
    angle = float(inp["angle_cw"])
    # crop EXACTLY as reporter._export_leaf_products does -- the filename uses the UNCLAMPED box,
    # the pixel origin is the CLAMPED one. Swapping them shifts every keypoint off the mask on any
    # leaf whose detection box overruns the sheet edge.
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(W, x2), min(H, y2)
    ch, cw = cy2 - cy1, cx2 - cx1
    if ch <= 0 or cw <= 0:
        return None

    decoded = {k: [decode_polygon(t) for t in v] for k, v in inp["polys"].items()}

    def raster(key: str) -> np.ndarray:
        acc = np.zeros((ch, cw), bool)
        for pp in decoded.get(key, []):
            acc |= polygon_mask(offset_polygon(pp, -cx1, -cy1), (ch, cw))
        return acc

    sil = raster(_LEAF)
    if not sil.any():
        return None
    hole, pet = raster(_HOLE), raster(_PETIOLE)

    def rot(mask: np.ndarray) -> np.ndarray:
        return rotate_image(mask.astype(np.uint8), angle, bg=0, nearest=True) > 0

    rsil, rhole, rpet = rot(sil), rot(hole), rot(pet)
    fit = mask_bbox(rsil)                       # content-fit on the SILHOUETTE (holes filled)
    if fit is None:
        return None

    def cut(mask: np.ndarray) -> np.ndarray:
        return crop_to_box(mask.astype(np.uint8), fit, 0) > 0

    aff, _canvas = oriented_affine((ch, cw), angle)
    named = inp["kpts"]
    kpts: dict[str, tuple[float, float]] = {}
    confs: dict[str, float] = {}
    if named:
        names = list(named)
        src = np.array([[named[n][0] - cx1, named[n][1] - cy1] for n in names], float)
        dst = apply_affine(aff, src) - np.array([fit[0], fit[1]], float)
        for n, pt in zip(names, dst):
            kpts[n] = (float(pt[0]), float(pt[1]))
            confs[n] = named[n][2]

    sil_c = cut(rsil)
    return OrientedLeaf(
        specimen_id=int(inp["specimen_id"]), leaf_id=int(inp["leaf_id"]),
        detection_id=int(inp["detection_id"]), stem=str(inp["stem"]),
        silhouette=sil_c, lamina=sil_c & ~cut(rhole), holes=cut(rhole) & sil_c,
        petiole=(cut(rpet) if rpet.any() else None),
        kpts=kpts, kpt_conf=confs, angle_cw=angle, det_box=(x1, y1, x2, y2),
        crop_shape=(ch, cw), truncated=bool(x1 < 0 or y1 < 0 or x2 > W or y2 > H),
    )


def load_leaves(db, specimen_id: int, *, min_kpt_conf: float = 0.25,
                require_orientation: bool = True) -> list[OrientedLeaf]:
    """Every oriented ``Leaf`` instance of one specimen. Convenience wrapper for the Reporter/tests."""
    built = (build_leaf(i) for i in leaf_inputs(
        db, specimen_id, min_kpt_conf=min_kpt_conf, require_orientation=require_orientation))
    return [b for b in built if b is not None]


# ==========================================================================
# ported from experiments/bilateral_symmetry/axes.py
# ==========================================================================

# axis resampling / binning defaults
N_AXIS = 512          # resampled axis vertices (the Voronoi sites)


N_BINS = 200          # arclength bins the profiles are reported on


SPLINE_SMOOTH = 3.0   # smoothing factor per point for the midvein spline fit


@dataclass
class AxisFrame:
    """A leaf midline plus the curvilinear coordinates it induces on the mask pixels."""

    kind: str                    # "chord" | "midvein"
    path: np.ndarray             # (N_AXIS, 2) equal-arclength axis points, tip -> base
    tangent: np.ndarray          # (N_AXIS, 2) unit tangents (tip -> base)
    length: float                # axis arclength, px
    chord_length: float          # straight tip->base distance, px
    xy: np.ndarray               # (P, 2) mask pixel coords (x, y)
    s: np.ndarray                # (P,) normalized arclength 0..1 of each mask pixel
    u: np.ndarray                # (P,) signed perpendicular offset, px (+ = LEFT, 0 = ON the axis)
    shape: tuple[int, int]       # (H, W) of the mask this frame was built on

    @property
    def sinuosity(self) -> float:
        """Axis arclength / straight tip-base distance. 1.0 for a perfectly straight midvein."""
        return float(self.length / self.chord_length) if self.chord_length > 0 else float("nan")

    def max_chord_deviation(self) -> float:
        """Largest perpendicular departure of the axis from its own tip->base chord, px."""
        a, b = self.path[0], self.path[-1]
        d = b - a
        n = float(np.hypot(*d))
        if n <= 0:
            return float("nan")
        d = d / n
        rel = self.path - a
        return float(np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]).max())

    def integrated_curvature(self) -> float:
        """Total absolute turning of the axis in radians (0 for a straight axis)."""
        t = self.tangent
        ang = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        return float(np.abs(np.diff(ang)).sum())


def _resample_polyline(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` points at equal arclength."""
    p = np.asarray(pts, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return np.repeat(p[:1], n, axis=0)
    t = np.linspace(0.0, cum[-1], n)
    return np.column_stack([np.interp(t, cum, p[:, 0]), np.interp(t, cum, p[:, 1])])


def _smooth_midvein(pts: np.ndarray, n: int) -> np.ndarray:
    """Equal-arclength resample of a smoothing-spline fit through the midvein keypoints.

    The pose model's midvein points jitter by a few pixels; without smoothing that jitter becomes
    spurious local curvature and rotates the perpendicular, which would show up as fake asymmetry.
    Falls back to a plain polyline resample if the spline cannot be fitted (too few points, or
    duplicate/collinear input).
    """
    p = np.asarray(pts, float)
    if len(p) < 4:
        return _resample_polyline(p, n)
    try:
        from scipy.interpolate import splev, splprep

        # k=3 needs >=4 points; s scales with point count (splprep's smoothing is a sum of squares)
        tck, _ = splprep([p[:, 0], p[:, 1]], s=SPLINE_SMOOTH * len(p), k=min(3, len(p) - 1))
        dense = np.column_stack(splev(np.linspace(0.0, 1.0, max(n, 4 * len(p))), tck))
        if not np.isfinite(dense).all():
            return _resample_polyline(p, n)
        return _resample_polyline(dense, n)
    except Exception:
        return _resample_polyline(p, n)


def _project(path: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-point projection of pixels onto the axis -> ``(s_index_frac, signed_offset)``.

    Each pixel is assigned to its nearest axis VERTEX (a Voronoi partition of the polyline, so the
    lamina is tiled exactly once), then refined to a sub-vertex arclength position and an exact
    signed perpendicular offset using the local tangent.
    """
    from scipy.spatial import cKDTree

    n = len(path)
    _d, j = cKDTree(path).query(xy, k=1)
    j = np.clip(np.asarray(j, int), 0, n - 1)

    # local tangent at the matched vertex (central difference, unit length)
    t = np.gradient(path, axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn <= 0] = 1.0
    t = t / tn[:, None]

    d = xy - path[j]
    tj = t[j]
    along = d[:, 0] * tj[:, 0] + d[:, 1] * tj[:, 1]          # tangential residual, px
    cross = tj[:, 0] * d[:, 1] - tj[:, 1] * d[:, 0]          # signed perpendicular, + = screen-left

    # vertex spacing in px -> convert the tangential residual into a fractional vertex offset, so s
    # is continuous across vertices instead of quantized to 1/N_AXIS.
    step = float(np.hypot(*np.diff(path, axis=0).T).mean()) if n > 1 else 1.0
    s_idx = j + (along / step if step > 0 else 0.0)
    return np.clip(s_idx / max(1, n - 1), 0.0, 1.0), cross


def build_axis(mask: np.ndarray, kind: str, *, midvein: Optional[np.ndarray] = None,
               tip: Optional[np.ndarray] = None, base: Optional[np.ndarray] = None,
               n_axis: int = N_AXIS) -> Optional[AxisFrame]:
    """Build a ``chord`` or ``midvein`` frame for ``mask`` (bool, the shape being measured)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return None

    if kind == "chord":
        if tip is None or base is None:
            return None
        path = _resample_polyline(np.vstack([np.asarray(tip, float), np.asarray(base, float)]), n_axis)
    elif kind == "midvein":
        if midvein is None or len(midvein) < 3:
            return None
        path = _smooth_midvein(np.asarray(midvein, float), n_axis)
    else:
        raise ValueError(f"unknown axis kind: {kind!r}")

    seg = np.hypot(*np.diff(path, axis=0).T)
    length = float(seg.sum())
    if length <= 0:
        return None
    chord_len = float(np.hypot(*(path[-1] - path[0])))

    t = np.gradient(path, axis=0)
    tn = np.hypot(t[:, 0], t[:, 1])
    tn[tn <= 0] = 1.0
    t = t / tn[:, None]

    ys, xs = np.nonzero(m)
    xy = np.column_stack([xs, ys]).astype(float)
    s, u = _project(path, xy)
    return AxisFrame(kind=kind, path=path, tangent=t, length=length, chord_length=chord_len,
                     xy=xy, s=s, u=u, shape=m.shape)


@dataclass
class Profiles:
    """Binned left/right profiles along the axis -- the input to every scalar metric."""

    s: np.ndarray            # (n_bins,) bin centers, 0 (tip) .. 1 (base)
    area_l: np.ndarray       # (n_bins,) lamina pixel count left of the axis in each bin
    area_r: np.ndarray
    w_l: np.ndarray          # (n_bins,) margin envelope: max perpendicular offset, left, px
    w_r: np.ndarray
    ds_px: float             # bin width in px of arclength
    n_used: np.ndarray       # (n_bins,) total pixels in the bin (area_l + area_r)


def profiles(frame: AxisFrame, n_bins: int = N_BINS) -> Profiles:
    """Bin a frame's pixels into equal-arclength strips and split each strip left/right.

    Areas are exact pixel counts (the Voronoi partition tiles the lamina once), so
    ``area_l.sum() + area_r.sum()`` is the lamina's total pixel area. Widths are the margin
    ENVELOPE (max |u| on that side), which is what a caliper would measure and is not simply
    ``area / ds`` for a lobed leaf with a sinus.

    ON-AXIS PIXELS. ``u`` is exactly 0 for any pixel the axis passes through, which happens for a
    whole pixel column whenever the axis is axis-aligned on integer coordinates -- the case of an
    exactly mirror-symmetric synthetic mask. Lumping them with one side (they used to fall to the
    right, since ``u > 0`` is false at 0) made such a mask report a nonzero ``A_star``. They are
    split HALF to each side instead, which keeps both properties that matter: the strip areas still
    sum EXACTLY to the lamina area, and mirroring the mask (which negates ``u`` and leaves the
    on-axis set alone) still negates every signed measure exactly. They are left out of the margin
    envelopes entirely -- a pixel ON the axis lies on neither margin.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, frame.s, side="right") - 1, 0, n_bins - 1)
    # ON-AXIS TOLERANCE, not an exact ``== 0`` test. The chord path is exactly straight, so a pixel
    # the axis passes through gets u == 0 bit-exactly; the MIDVEIN path comes out of a smoothing
    # spline, which leaves a ~5e-14 px wobble. That is geometrically nothing, but an exact test
    # sends those pixels left/right by the sign of the noise: on a perfectly mirror-symmetric
    # synthetic mask only 39 of 221 on-axis pixels were recognized, giving a_star = -0.0025 and
    # si_a = 0.050 where both must be 0. Since production measures on the midvein, and si_a IS the
    # ranking, that bias is not tolerable. ON_AXIS_EPS sits far above double-precision noise at
    # pixel magnitudes and far below any real geometric offset.
    left = frame.u > ON_AXIS_EPS
    right = frame.u < -ON_AXIS_EPS
    on_axis = ~(left | right)

    half = 0.5 * np.bincount(idx[on_axis], minlength=n_bins).astype(float)
    area_l = np.bincount(idx[left], minlength=n_bins).astype(float) + half
    area_r = np.bincount(idx[right], minlength=n_bins).astype(float) + half

    w_l = np.zeros(n_bins)
    w_r = np.zeros(n_bins)
    np.maximum.at(w_l, idx[left], frame.u[left])
    np.maximum.at(w_r, idx[right], -frame.u[right])

    centers = 0.5 * (edges[:-1] + edges[1:])
    return Profiles(s=centers, area_l=area_l, area_r=area_r, w_l=w_l, w_r=w_r,
                    ds_px=float(frame.length / n_bins), n_used=area_l + area_r)


def straighten(frame: AxisFrame, *, n_s: int = 256, n_u: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the two half-laminae into a common ``(s, u)`` grid, right half MIRRORED.

    Returns ``(left, right)`` boolean grids of shape ``(n_s, n_u)``: row = arclength bin, column =
    |perpendicular offset| bin, both scaled by the SAME ``u`` range. Reflecting in ``(s, u)`` rather
    than across one Euclidean line is what makes the overlap measures meaningful for a leaf whose
    midvein is curved -- the two halves are compared after the leaf is straightened.

    Each grid CELL is sampled by mapping its ``(s, |u|)`` back to a sheet point
    ``path[s] +/- |u| * normal[s]`` and testing the mask there. Scattering the mask's pixels
    FORWARD into the grid instead would be resolution-dependent and badly wrong for small leaves: a
    20k-px lamina has fewer pixels per half than the grid has cells, so most cells stay empty, the
    two halves' holes do not coincide, and Dice collapses toward 0 no matter how symmetric the leaf
    actually is (measured: several leaves scored a literal 0.000 that way). Sampling backward fills
    every cell and makes the result independent of both leaf size and grid resolution.

    The backward map is the exact inverse of the forward nearest-point projection only where the
    axis's radius of curvature exceeds ``|u|``; on a strongly bowed midvein the normals converge on
    the concave side. Both halves are sampled identically, so the comparison stays fair, and leaf
    midveins are far too gently curved (sinuosity ~1.01) for the crossing to arise in practice.
    """
    umax = float(np.abs(frame.u).max()) if frame.u.size else 0.0
    empty = (np.zeros((n_s, n_u), bool), np.zeros((n_s, n_u), bool))
    if umax <= 0:
        return empty

    h, w = frame.shape
    mask = np.zeros((h, w), bool)
    ij = np.rint(frame.xy).astype(int)
    np.clip(ij[:, 0], 0, w - 1, out=ij[:, 0])
    np.clip(ij[:, 1], 0, h - 1, out=ij[:, 1])
    mask[ij[:, 1], ij[:, 0]] = True

    # cell centers: s in (0,1), |u| in (0, umax]
    s_c = (np.arange(n_s) + 0.5) / n_s
    u_c = (np.arange(n_u) + 0.5) / n_u * umax
    idx = np.clip(np.rint(s_c * (len(frame.path) - 1)).astype(int), 0, len(frame.path) - 1)
    base = frame.path[idx]                       # (n_s, 2)
    t = frame.tangent[idx]
    n_left = np.column_stack([-t[:, 1], t[:, 0]])   # cross(t, n_left) = +1 -> the viewer's LEFT

    out = []
    for sign in (+1.0, -1.0):
        p = base[:, None, :] + sign * u_c[None, :, None] * n_left[:, None, :]
        xi = np.rint(p[..., 0]).astype(int)
        yi = np.rint(p[..., 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        g = np.zeros((n_s, n_u), bool)
        g[ok] = mask[yi[ok], xi[ok]]
        out.append(g)
    return out[0], out[1]


# ==========================================================================
# ported from experiments/bilateral_symmetry/metrics.py
# ==========================================================================

# a bin whose two sides together hold less than this fraction of the lamina is too small for its
# ratio to mean anything (see module docstring)
EPS_FRAC = 1e-3


def _div(num: float, den: float) -> float:
    """Division that yields nan instead of raising or returning a meaningless 0.0."""
    return float(num) / float(den) if den else _NAN


def area_asymmetry_profile(prof: Profiles, *, eps_frac: float = EPS_FRAC) -> np.ndarray:
    """``a_A(s) = (A_Li - A_Ri) / (A_Li + A_Ri)``, nan in bins below the area guard."""
    al = np.asarray(prof.area_l, float)
    ar = np.asarray(prof.area_r, float)
    tot = al + ar
    keep = tot >= eps_frac * float(tot.sum())
    out = np.full(al.shape, _NAN)
    np.divide(al - ar, tot, out=out, where=keep & (tot > 0))
    return out


# ==========================================================================
# ported from experiments/bilateral_symmetry/shape.py
# ==========================================================================

def _overlap(frame: AxisFrame, n_s: int, n_u: int) -> tuple[float, float, float]:
    """IoU, Dice and normalized symmetric difference of the two straightened halves."""
    left, right = straighten(frame, n_s=n_s, n_u=n_u)
    inter = float(np.count_nonzero(left & right))
    union = float(np.count_nonzero(left | right))
    total = float(np.count_nonzero(left) + np.count_nonzero(right))
    iou = inter / union if union > 0 else _NAN
    dice = 2.0 * inter / total if total > 0 else _NAN
    sd = float(np.count_nonzero(left ^ right)) / total if total > 0 else _NAN
    return iou, dice, sd


# ==========================================================================
# ported from experiments/bilateral_symmetry/quality.py
# ==========================================================================

# Connectivity for component labeling. 8-connectivity, NOT the scipy default 4: polygon
# rasterization routinely leaves one-pixel diagonal necks inside a single leaf, and 4-connectivity
# would report those as separate components -- a fake defect in an otherwise perfect mask.
_CONN8 = np.ones((3, 3), bool)


N_CONTOUR = 512            # equal-arclength samples of the outline used for smoothing


ROUGHNESS_SIGMA_FRAC = 0.005   # Gaussian bandwidth as a FRACTION of outline length (see below)


# Total midvein trace points the pose model can emit: lamina_tip + midvein_0..14 + lamina_base.
N_MIDVEIN_MAX = len(MIDVEIN_NAMES)      # == 17


@dataclass(frozen=True)
class QualityDiagnostics:
    """Per-leaf mask diagnostics. Every field is symmetry-free by construction.

    Degenerate input never raises: a field that cannot be defined (empty mask, no contour, no
    keypoints) is ``nan`` for floats and ``0`` for counts.

    Attributes
    ----------
    n_components:
        Connected components (8-connected) of the silhouette. ``1`` for a clean leaf; ``>1`` means
        debris, a second leaf caught in the crop, or a mask split by a tear.
    largest_frac:
        Largest component area / total silhouette area, in ``(0, 1]``. ``1.0`` means the whole mask
        is one blob; ``0.97`` means 3% of the "leaf" is stray pixels somewhere else.
    solidity:
        Outer-contour area / convex-hull area, on the LARGEST component only (debris is already
        reported by the two fields above; letting it drag the hull would double-count it).
        **This is a lobing signal at least as much as a quality signal.** A clean entire leaf sits
        near 0.95-0.99, but a clean deeply lobed leaf -- an oak, a maple -- is legitimately down
        near 0.6-0.7 with nothing whatever wrong with its mask. Never threshold it alone.
    border_contact_frac:
        Fraction of the silhouette's boundary pixels that lie on the edge of the oriented mask, i.e.
        evidence the leaf was sliced off by the detection box rather than ending in its own margin.
        Reported for the correlation study, but NOT the field the composite scores on -- see
        ``flush_edge_frac`` for why. Real leaves in the test project run 0.001-0.22, median 0.018.
    flush_edge_frac:
        The largest fraction of any ONE of the four mask edges that the silhouette covers, and the
        signal :func:`archetype_score` uses for completeness. The oriented mask is
        content-fit to the silhouette's tight bounding box, so the mask necessarily touches all four
        edges; what distinguishes a cut leaf is that it lies FLUSH along a whole edge. The
        difference matters: a clean synthetic ellipse measures ``border_contact_frac`` 0.112 purely
        because a smooth convex curve is tangent to its bounding box over a long arc, so scoring on
        that number would systematically penalize every blunt-apex leaf as "cut off"; the same
        ellipse measures ``flush_edge_frac`` 0.11 against ~1.0 for a genuinely chopped mask.
        **MEASURED SIZE CONFOUND, read before using this as a truncation flag.** On the 105 real
        leaves it correlates with ``1 / sqrt(area_px)`` at r = +0.848 and with ``log(area_px)`` at
        r = -0.728: a small mask is a coarse mask, one row of its edge is a larger fraction of a
        short edge, and staircase quantization puts more of the outline exactly on the border. So
        the number is mostly reporting how SMALL the leaf is, not how cut off it is, and it clears
        ``FLUSH_EDGE_FREE`` for exactly 1 of the 105. ``border_contact_frac`` carries the same
        confound (r = +0.841 against ``1 / sqrt(area_px)``). Neither is a clean truncation detector;
        see :func:`_completeness_term` for what that costs the composite.
        One caveat remains for both fields: the mask has been ROTATED tip-up, so a cut made along a
        sheet edge only stays flush with a mask edge when the rotation is near a multiple of 90
        degrees. At other angles the cut becomes a slanted straight chord that this axis-aligned
        test under-reads, which is why ``truncated`` is carried alongside as the geometric,
        rotation-proof flag.
    truncated:
        The detection box was clipped by the sheet edge (from :class:`OrientedLeaf`). A leaf that ran
        off the sheet cannot be archetypal, so this is a hard gate in :func:`is_archetypal`.
        It is only *evidence* in :func:`archetype_score`: a box may overhang the sheet by a pixel
        while the leaf itself is entirely inside, so it is graded, not vetoed.
    perimeter_ratio:
        ``perimeter / (2 * sqrt(pi * area))`` -- the dissection index of Shi et al. (2020), the
        reciprocal square root of circularity. Exactly 1 for a disk and rising with any departure
        from it. It rises for lobing, for serration AND for a ragged mask, so like ``solidity`` it
        is a shape descriptor first. Perimeter and area are both taken from the same digitized outer
        contour; on a pixel grid that outline is a staircase, which inflates the perimeter by
        roughly 5% for a smooth curve, so a clean ellipse measures ~1.1 rather than 1.0. Compare
        leaves to each other, not to the ideal.
    boundary_roughness:
        ``perimeter(raw outline) / perimeter(low-pass outline)``, where the low-pass outline is the
        contour resampled to ``N_CONTOUR`` equal-arclength points and convolved with a periodic
        Gaussian of sigma ``ROUGHNESS_SIGMA_FRAC`` of the outline LENGTH. Because the bandwidth is a
        fraction of the shape's own perimeter rather than a pixel count, the measure is
        dimensionless and scale-free: the same leaf scanned at twice the resolution scores the same.
        It rises with everything finer than about 1% of the perimeter and is flat for everything
        coarser, so smooth lobes -- however deep -- pass through the filter untouched and do not
        register. The honest limitation: fine marginal TEETH are also finer than the bandwidth, so a
        serrate leaf reads as rough. Read with ``solidity`` (lobing lowers it, noise barely does) to
        separate real margin complexity from pixel-level noise.
    hole_frac:
        Hole area / silhouette area -- insect damage, decay, or segmentation dropouts.
    kpt_conf_mean, kpt_conf_min:
        Pose confidence over the MIDVEIN TRACE keypoints only, because those are the points the
        midvein axis is fitted to; a confident lobe tip would not make a bad axis good. Computed
        over the stations with a USABLE (finite) confidence, so one nan-valued confidence can no
        longer poison both statistics into nan and silently delete the trace penalty; both are nan
        only when no station has a usable confidence at all.
    n_midvein_kpts:
        Midvein trace points that survived the loader's confidence filter AND carry a usable
        (finite) confidence, out of ``N_MIDVEIN_MAX`` (17). A short trace means a short or wandering
        axis. Counted on the same stations the two confidence statistics average over, so
        :func:`_trace_term` can reconstruct the full-skeleton picture from the three numbers.
    area_px:
        Silhouette area in pixels (holes filled). Size context, and the scale against which
        pixel-level noise should be judged.
    aspect:
        Silhouette bounding-box height / width in the tip-up frame, so it reads as
        length / width for an oriented leaf.
    axis_aspect:
        ``axis arclength / (2 * max |u|)`` from the axis frame, i.e. length over full width measured
        along the real midline instead of the bounding box. ``nan`` when no frame is supplied.
        Context only -- it is never scored, and it is the one field derived from the axis frame.
    """

    specimen_id: int
    leaf_id: int
    detection_id: int

    n_components: int
    largest_frac: float
    solidity: float
    border_contact_frac: float
    flush_edge_frac: float
    truncated: bool
    perimeter_ratio: float
    boundary_roughness: float
    hole_frac: float
    kpt_conf_mean: float
    kpt_conf_min: float
    n_midvein_kpts: int
    area_px: int
    aspect: float
    axis_aspect: float

    def as_dict(self) -> dict[str, Any]:
        """Flat, JSON/CSV-friendly record -- one row of the report's diagnostics table."""
        return asdict(self)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, float]:
    """``(largest component, n_components, largest_frac)`` for a boolean mask."""
    from scipy import ndimage as ndi

    total = int(mask.sum())
    if total == 0:
        return mask, 0, float("nan")
    lab, n = ndi.label(mask, structure=_CONN8)
    if n <= 1:
        return mask, int(n), 1.0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    k = int(counts.argmax())
    return lab == k, int(n), float(counts[k] / total)


def _outer_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    """Longest external contour of ``mask`` as an ``(N, 1, 2)`` int32 array, or None.

    ``CHAIN_APPROX_NONE``: the roughness measure needs every boundary pixel, not a simplified
    polygon -- simplification is exactly the signal being measured.
    """
    import cv2

    m = np.ascontiguousarray(mask.astype(np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = [c for c in cnts if len(c) >= 3]
    if not cnts:
        return None
    return max(cnts, key=lambda c: len(c))


def _resample_closed(pts: np.ndarray, n: int) -> Optional[np.ndarray]:
    """Resample a CLOSED polyline to ``n`` points at equal arclength (last point != first)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 3:
        return None
    loop = np.vstack([p, p[:1]])
    seg = np.hypot(*np.diff(loop, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] <= 0:
        return None
    t = np.linspace(0.0, cum[-1], n, endpoint=False)
    return np.column_stack([np.interp(t, cum, loop[:, 0]), np.interp(t, cum, loop[:, 1])])


def _closed_perimeter(pts: np.ndarray) -> float:
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) < 2:
        return float("nan")
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Silhouette pixels with at least one background 4-neighbor, image edge counting as background.

    ``border_value=0`` (the scipy default) is what makes an on-edge pixel count as boundary, which
    is the whole point here: a leaf cut by the crop has a wall of boundary pixels on the edge.
    """
    from scipy import ndimage as ndi

    if not mask.any():
        return np.zeros_like(mask)
    cross = ndi.generate_binary_structure(2, 1)
    return mask & ~ndi.binary_erosion(mask, structure=cross)


def _border_stats(mask: np.ndarray) -> tuple[float, float]:
    """``(border_contact_frac, flush_edge_frac)`` -- two views of "does the mask hit the crop edge".

    The first normalizes by the outline length and so is inflated by mere tangency; the second asks
    how much of a single edge LINE the mask covers, which only a straight cut can drive toward 1.
    """
    b = _boundary_pixels(mask)
    n = int(b.sum())
    if n == 0:
        return float("nan"), float("nan")
    edge = np.zeros_like(mask)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    contact = float((b & edge).sum() / n)
    flush = max(float(mask[0, :].mean()), float(mask[-1, :].mean()),
                float(mask[:, 0].mean()), float(mask[:, -1].mean()))
    return contact, flush


def _shape_stats(mask: np.ndarray) -> tuple[float, float, float]:
    """``(solidity, perimeter_ratio, boundary_roughness)`` from one component's outer contour."""
    import cv2
    from scipy.ndimage import gaussian_filter1d

    nan3 = (float("nan"), float("nan"), float("nan"))
    cnt = _outer_contour(mask)
    if cnt is None:
        return nan3

    perim = float(cv2.arcLength(cnt, True))
    area = float(cv2.contourArea(cnt))
    if perim <= 0 or area <= 0:
        return nan3

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else float("nan")

    # 2*sqrt(pi*A) is the perimeter of the disk of equal area, so the ratio is 1 for a disk.
    perimeter_ratio = perim / (2.0 * math.sqrt(math.pi * area))

    res = _resample_closed(cnt.reshape(-1, 2).astype(float), N_CONTOUR)
    if res is None:
        return solidity, perimeter_ratio, float("nan")
    sigma = ROUGHNESS_SIGMA_FRAC * N_CONTOUR      # bandwidth in SAMPLES == fraction of arclength
    smooth = gaussian_filter1d(res, sigma, axis=0, mode="wrap")
    sp = _closed_perimeter(smooth)
    roughness = perim / sp if sp and sp > 0 else float("nan")
    return solidity, perimeter_ratio, roughness


def compute_quality(leaf: OrientedLeaf, frame: Optional[AxisFrame] = None) -> QualityDiagnostics:
    """All symmetry-free diagnostics for one leaf.

    ``frame`` is optional and supplies ``axis_aspect`` only; every other field is measured on the
    masks alone, so a leaf whose axis could not be built is still fully diagnosed.
    """
    sil = np.asarray(leaf.silhouette, bool)
    area_px = int(sil.sum())

    largest, n_comp, largest_frac = _largest_component(sil)
    # Shape stats on the largest component: a stray blob's contribution is already reported by
    # n_components/largest_frac, and dragging the convex hull out to reach it would report the same
    # defect a second time as fake lobing.
    solidity, perim_ratio, roughness = _shape_stats(largest) if area_px else (
        float("nan"), float("nan"), float("nan"))

    holes = np.asarray(leaf.holes, bool)
    hole_frac = float(holes.sum() / area_px) if area_px else float("nan")

    if area_px:
        ys, xs = np.nonzero(sil)
        h = float(ys.max() - ys.min() + 1)
        w = float(xs.max() - xs.min() + 1)
        aspect = h / w if w > 0 else float("nan")
    else:
        aspect = float("nan")

    # nan-tolerant: np.mean/np.min over a set holding ONE nan confidence returns nan for both
    # statistics, and a nan sub-score is dropped from the archetype mean -- so a single unusable
    # confidence used to delete the whole trace penalty and let the leaf score as if its trace were
    # perfect. Average over the stations whose confidence is usable instead (== nanmean/nanmin), and
    # count only those, so mean, min and count all describe the same set of stations.
    confs = np.asarray([leaf.kpt_conf[n] for n in MIDVEIN_NAMES if n in leaf.kpt_conf], float)
    confs = confs[np.isfinite(confs)]
    kpt_mean = float(confs.mean()) if confs.size else float("nan")
    kpt_min = float(confs.min()) if confs.size else float("nan")
    contact, flush = _border_stats(sil) if area_px else (float("nan"), float("nan"))

    axis_aspect = float("nan")
    if frame is not None and frame.length > 0:
        umax = float(np.abs(frame.u).max()) if frame.u.size else 0.0
        if umax > 0:
            axis_aspect = float(frame.length / (2.0 * umax))

    return QualityDiagnostics(
        specimen_id=int(leaf.specimen_id), leaf_id=int(leaf.leaf_id),
        detection_id=int(leaf.detection_id),
        n_components=n_comp, largest_frac=largest_frac, solidity=solidity,
        border_contact_frac=contact, flush_edge_frac=flush,
        truncated=bool(leaf.truncated), perimeter_ratio=perim_ratio,
        boundary_roughness=roughness, hole_frac=hole_frac,
        kpt_conf_mean=kpt_mean, kpt_conf_min=kpt_min, n_midvein_kpts=int(confs.size),
        area_px=area_px, aspect=aspect, axis_aspect=axis_aspect,
    )


# -- symmetry term ---------------------------------------------------------------------------- #
# Calibrated against the cohort, NOT against the one verified example leaf. That leaf (midvein Dice
# 0.9735) turns out to sit at the cohort MAXIMUM: the 105 real leaves run Dice 0.13-0.977 with a
# median of 0.895, and |SI_A| 0.027-0.878 with a median of 0.142. Ramps anchored on the example leaf
# drove 49 of 105 leaves to a hard 0 and destroyed all ranking in the bottom half, so the zero
# points sit below the 10th percentile and only genuinely broken masks veto.
SYM_DICE_ZERO = 0.70     # straightened left/right Dice at which the symmetry sub-score hits 0


SYM_DICE_ONE = 0.97      # ... and at which it saturates at 1 (the best real leaf measured)


SYM_SIA_ZERO = 0.35      # |SI_A| (signed area imbalance) at which the sub-score hits 0


SYM_SIA_ONE = 0.03       # ... and at which it saturates at 1 (the most balanced real leaf measured)


SYM_AXIS = "midvein"     # the axis the composite scores; the chord axis is the control, not the ruler


# -- integrity term --------------------------------------------------------------------------- #
LARGEST_FRAC_ZERO = 0.90     # largest component holds <=90% of the area -> integrity 0


LARGEST_FRAC_ONE = 1.0


COMPONENT_PENALTY_K = 0.5    # component sub-score = 1 / (1 + K * (n_components - 1))


HOLE_FRAC_ZERO = 0.05        # >=5% of the lamina lost to holes -> 0


HOLE_FRAC_ONE = 0.0


# -- completeness term ------------------------------------------------------------------------ #
# Scored on flush_edge_frac, not border_contact_frac: a clean ellipse is tangent to its content-fit
# box over enough of its outline to read 0.11 border contact with nothing wrong with it.
# Both of those are however confounded with leaf SIZE rather than with truncation (measured
# correlations in the flush_edge_frac field docstring), so this term is deliberately weighted low --
# see _completeness_term and ARCHETYPE_WEIGHTS.
FLUSH_EDGE_FREE = 0.25       # deadband: a blunt apex legitimately covers a quarter of its box edge


FLUSH_EDGE_ZERO = 0.75       # three quarters of one edge covered -> that edge is a cut, not a margin


TRUNCATED_PENALTY = 0.25     # multiplier when the detection box was clipped by the sheet edge


# -- trace-confidence term -------------------------------------------------------------------- #
# Nudged to span the range actually observed on the test project (mean 0.76-0.995, worst-point
# 0.35-0.98) so the term discriminates instead of saturating at 1 for every leaf. The mean ramp is
# applied to the COVERAGE-WEIGHTED mean (see _trace_term), which equals the plain mean for the
# complete 17-station trace every leaf in the test project has, so the calibration is unchanged.
KPT_CONF_MEAN_ZERO = 0.50


KPT_CONF_MEAN_ONE = 0.95


KPT_CONF_MIN_ZERO = 0.25     # == the loader's default filter, so a barely-surviving point scores 0


KPT_CONF_MIN_ONE = 0.85


# Where the COUNT sub-score bottoms out. 2.0, not the 6.0 first proposed: geometry.OrientedLeaf
# .midvein() returns an axis from 3 points, so 3 points is a coarse axis, not an absent one, and 2
# is the count at which the axis genuinely ceases to exist. The old 6.0 hard-vetoed (sub-score
# exactly 0, and a 0 annihilates a geometric mean) every 3-6 point trace that the geometry, the
# axis fit and the symmetry metrics had all accepted and measured. Grading them instead leaves the
# categorical rejection to ARCHETYPE_MIN_MIDVEIN_KPTS in is_archetypal, which is where the other
# categorical judgments live and where the threshold can be changed without silently zeroing a
# published score.
MIDVEIN_KPTS_ZERO = 2.0


MIDVEIN_KPTS_ONE = 15.0


#: Weights of the four top-level terms in the weighted GEOMETRIC mean. Geometric, not arithmetic,
#: because "archetypal" is a conjunction: a beautifully symmetric leaf that is torn in half is not
#: archetypal, and no amount of symmetry should buy that back. Each term contributes ``t ** w``, so
#: a term at 0 vetoes the leaf outright and a small term is punished far harder than an arithmetic
#: mean would punish it. Set a weight to 0.0 to drop a term entirely -- in particular
#: ``{"symmetry": 0.0}`` yields the symmetry-free composite the report needs to test the hypothesis
#: without circularity. The weights are RELATIVE -- ``_gmean`` divides by whatever they sum to --
#: so lowering one raises every other one's share.
ARCHETYPE_WEIGHTS: dict[str, float] = {
    "symmetry": 0.40,       # the hypothesis under test -- weighted highest ON PURPOSE, see caveat
    "integrity": 0.25,      # is it one whole undamaged blob
    # 0.05, cut from the 0.20 first proposed. The term cannot deliver 0.20 of an honest completeness
    # judgment: the only truncation evidence in the stored oriented products is size-confounded
    # (flush_edge_frac, r = +0.848 against 1/sqrt(area_px)) and it fires on 1 of 105 real leaves.
    # The weight now reflects what the term really is -- a flag for the rare genuinely clipped
    # detection box, backed by the hard `truncated` gate in is_archetypal -- rather than an equal
    # partner in the ranking. See _completeness_term for why nothing better is computable here.
    "completeness": 0.05,   # is all of it actually in frame (see caveat -- weak evidence)
    "trace": 0.15,          # do we trust the axis the symmetry was measured against
}


#: A sub-score below this is worth explaining in the returned reasons.
REASON_SCORE_THRESHOLD = 0.98


def _ramp(x: float, zero_at: float, one_at: float) -> float:
    """Clipped linear map: ``zero_at -> 0``, ``one_at -> 1``. Works in either direction.

    NaN in, NaN out -- a missing measurement must not masquerade as a passing one.
    """
    if x is None:
        return float("nan")
    x = float(x)
    if not math.isfinite(x) or zero_at == one_at:
        return float("nan")
    return float(min(1.0, max(0.0, (x - zero_at) / (one_at - zero_at))))


def _gmean(values: Sequence[float], weights: Optional[Sequence[float]] = None) -> float:
    """Weighted geometric mean, ignoring NaN entries and zero weights. NaN if nothing is left."""
    v = np.asarray(values, float)
    w = np.ones_like(v) if weights is None else np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any() or w[ok].sum() <= 0:
        return float("nan")
    v, w = np.clip(v[ok], 0.0, 1.0), w[ok]
    if (v <= 0).any():
        return 0.0                       # one vetoing term; log would be -inf
    return float(np.exp(float(np.sum(w * np.log(v)) / float(w.sum()))))


def _amean(values: Sequence[float]) -> float:
    """Unweighted ARITHMETIC mean of the finite entries, clipped to ``[0, 1]``. NaN if none are.

    Used inside :func:`_trace_term` only. Across the four top-level terms the mean is geometric
    because "archetypal" is a conjunction of independent properties; the three trace inputs are
    three views of ONE property (how much of the midvein the pose model actually pinned down), and a
    zero in one of them must not annihilate a trace the geometry can still fit an axis to.
    """
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.clip(v, 0.0, 1.0).mean()) if v.size else float("nan")


def _sym_value(sym: dict, name: str, axis: str) -> tuple[float, str]:
    """Pull one symmetry metric out of ``sym``, tolerating the layouts metrics.py might use.

    Accepts, in order: ``sym[axis][name]``, ``sym[f"{axis}_{name}"]``, ``sym[f"{name}_{axis}"]``,
    ``sym[name]``; all case-insensitively, so ``SI_A``/``si_a`` and ``dice``/``Dice`` both resolve.
    Returns ``(value, axis_actually_used)`` with ``nan`` and ``""`` when nothing matches, rather
    than guessing -- a missing metric is dropped from the mean, never defaulted to a passing value.
    """
    if not isinstance(sym, dict):
        return float("nan"), ""

    def _num(x: Any) -> float:
        try:
            f = float(x)
        except (TypeError, ValueError):
            return float("nan")
        return f if math.isfinite(f) else float("nan")

    flat = {str(k).lower(): v for k, v in sym.items()}
    nested = flat.get(axis.lower())
    if isinstance(nested, dict):
        sub = {str(k).lower(): v for k, v in nested.items()}
        if name.lower() in sub:
            return _num(sub[name.lower()]), axis
    for key in (f"{axis}_{name}", f"{name}_{axis}", name):
        if key.lower() in flat:
            return _num(flat[key.lower()]), axis
    return float("nan"), ""


def _symmetry_term(sym: dict) -> tuple[float, list[str]]:
    """Sub-score from the midvein-axis Dice and |SI_A|, falling back to the chord axis loudly."""
    reasons: list[str] = []
    axis = SYM_AXIS
    dice, used = _sym_value(sym, "dice", axis)
    sia, used_a = _sym_value(sym, "SI_A", axis)
    if not used and not used_a:
        # The chord axis systematically over-reads asymmetry on a curved leaf (verified: -0.0867 vs
        # -0.0216 on the same leaf), so scoring on it is a fallback worth announcing.
        dice, used = _sym_value(sym, "dice", "chord")
        sia, used_a = _sym_value(sym, "SI_A", "chord")
        if used or used_a:
            axis = "chord"
            reasons.append("no midvein-axis symmetry; scored on the chord axis, which over-reads "
                           "asymmetry for a curved midvein")
    # SI_A is mean(|per-bin imbalance|) -- UNSIGNED by construction (metrics.py), and on the 105 real
    # leaves it runs 0.027-0.878 with not one negative value. Printing it as "{:+.3f}" therefore
    # promised a direction it cannot carry: the leading + was an artifact of the format string, not
    # a side. A_star is the signed total imbalance (-0.892..+0.417 here, 52 of 105 negative, + means
    # the viewer's LEFT is larger), so the direction sentence is built from A_star and SI_A is
    # reported unsigned as the magnitude it is.
    astar, _used_s = _sym_value(sym, "A_star", axis) if (used or used_a) else (float("nan"), "")

    d_score = _ramp(dice, SYM_DICE_ZERO, SYM_DICE_ONE)
    a_score = _ramp(abs(sia) if math.isfinite(sia) else float("nan"), SYM_SIA_ZERO, SYM_SIA_ONE)
    if math.isfinite(d_score) and d_score < REASON_SCORE_THRESHOLD:
        reasons.append(f"{axis} half-lamina Dice {dice:.3f} (archetypal from {SYM_DICE_ONE:.2f})")
    if math.isfinite(a_score) and a_score < REASON_SCORE_THRESHOLD:
        side = ""
        if math.isfinite(astar):
            side = (f"; net A* {astar:+.3f}, {abs(astar) * 100:.1f}% more lamina on the "
                    f"{'left' if astar > 0 else 'right'}")
        reasons.append(f"{axis} area imbalance SI_A {abs(sia):.3f} unsigned "
                       f"({abs(sia) * 100:.1f}% of the lamina on one side in the average bin){side}")
    score = _gmean([d_score, a_score])
    if not math.isfinite(score):
        reasons.append("symmetry metrics unavailable; the leaf CANNOT be scored (the symmetry term "
                       "is the hypothesis, so it is never renormalized away)")
    return score, reasons


def _integrity_term(q: QualityDiagnostics) -> tuple[float, list[str]]:
    """Sub-score from fragmentation and holes: is the mask ONE WHOLE piece of leaf?"""
    reasons: list[str] = []
    frac = _ramp(q.largest_frac, LARGEST_FRAC_ZERO, LARGEST_FRAC_ONE)
    comp = 1.0 / (1.0 + COMPONENT_PENALTY_K * max(0, q.n_components - 1)) if q.n_components else float("nan")
    hole = _ramp(q.hole_frac, HOLE_FRAC_ZERO, HOLE_FRAC_ONE)

    if q.n_components and q.n_components > 1:
        pct = q.largest_frac * 100 if math.isfinite(q.largest_frac) else float("nan")
        reasons.append(f"silhouette split into {q.n_components} components "
                       f"(largest holds {pct:.1f}% of the area)")
    elif math.isfinite(frac) and frac < REASON_SCORE_THRESHOLD:
        reasons.append(f"largest component holds only {q.largest_frac * 100:.1f}% of the mask")
    if math.isfinite(hole) and hole < REASON_SCORE_THRESHOLD:
        reasons.append(f"holes cover {q.hole_frac * 100:.1f}% of the silhouette")
    score = _gmean([frac, comp, hole])
    if not math.isfinite(score):
        reasons.append("integrity undefined (empty mask); term dropped from the score")
    return score, reasons


def _completeness_term(q: QualityDiagnostics) -> tuple[float, list[str]]:
    """Sub-score for "all of the leaf is actually in the picture" -- WEAK EVIDENCE, read this.

    **No sound size-independent truncation measure is available from the data at hand, and this
    term is not one.** A genuine detector would ask what fraction of the outline lies on the
    ORIGINAL crop border -- the detection box intersected with the sheet, before the tip-up rotation
    and the content-fit -- but :class:`~.geometry.OrientedLeaf` keeps only the post-fit masks, the
    det box and ``truncated``; the pre-fit geometry and the sheet dimensions needed to tell "this
    border is the sheet edge" from "this border is where the detector chose to stop" are not
    carried, and the content fit guarantees the mask touches all four edges of the frame it does
    keep. So the term is scored on ``flush_edge_frac``, which is a leaf-SIZE measurement dressed as
    a cut-off detector: r = +0.848 against ``1 / sqrt(area_px)`` on the 105 real leaves, and exactly
    ONE of them ends up below 1.0. Its weight in :data:`ARCHETYPE_WEIGHTS` was cut to 0.05 to match
    that: the term is a flag for a genuinely clipped detection box (``truncated``, also a hard gate
    in :func:`is_archetypal`), not a graded completeness judgment. Fixing it properly means storing
    the pre-fit border contact in ``OrientedLeaf``, not re-tuning the constants here.
    """
    reasons: list[str] = []
    score = _ramp(q.flush_edge_frac, FLUSH_EDGE_ZERO, FLUSH_EDGE_FREE)
    if math.isfinite(score) and score < REASON_SCORE_THRESHOLD:
        reasons.append(f"mask lies flush along {q.flush_edge_frac * 100:.0f}% of one crop edge "
                       "(leaf likely cut off, but the number is size-confounded -- see the "
                       "flush_edge_frac docstring)")
    if q.truncated:
        reasons.append("detection box was clipped by the sheet edge (leaf may run off the sheet)")
        # nan stays nan. Substituting a 1.0 base here fabricated a finite 0.25 for a leaf with NO
        # measurement at all (an empty mask has no boundary and no flush edge), which reads as a
        # measured quarter-credit rather than as "undefined" and drags a real number into the mean.
        score = score * TRUNCATED_PENALTY
    if not math.isfinite(score):
        reasons.append("completeness undefined (no boundary); term dropped from the score")
    return score, reasons


def _trace_term(q: QualityDiagnostics) -> tuple[float, list[str]]:
    """Sub-score for "the midvein axis the symmetry was measured against is trustworthy".

    Count and confidence are scored JOINTLY, over all :data:`N_MIDVEIN_MAX` stations of the fixed
    pose skeleton rather than over the surviving points alone, because scoring the survivors alone
    made the term NON-MONOTONIC IN DATA QUALITY: deleting the single weakest keypoint of a 17-point
    trace took the term from 0.00 to 1.00 and flipped ``is_archetypal`` to True (measured), since the
    min-confidence penalty walked out with the point that caused it while the count barely moved.
    The pose model emits the whole skeleton every time, so a station that is not here is a station
    the model was not confident about -- absent evidence, not neutral evidence -- and it is scored
    as zero confidence. Every input below is then monotone non-decreasing in every station's
    confidence, so removing information can only lower the term, never raise it.
    """
    reasons: list[str] = []
    n = min(int(q.n_midvein_kpts), N_MIDVEIN_MAX)
    if n <= 0 or not math.isfinite(q.kpt_conf_mean) or not math.isfinite(q.kpt_conf_min):
        # No usable trace at all is an ABSENT measurement, not a bad one: let the term drop out and
        # let the hard gate in is_archetypal do the rejecting, rather than fabricating a zero -- or,
        # worse, a 1.0, which is what a nan-poisoned confidence pair used to produce here.
        return float("nan"), ["no usable midvein keypoint confidences; term dropped from the score"]

    # Coverage-weighted mean: the mean confidence over all N_MIDVEIN_MAX stations with the missing
    # ones contributing 0. Identical to the plain mean for a complete trace -- which is every leaf of
    # the test project -- so the ramp calibration is untouched.
    cov_mean = n * q.kpt_conf_mean / N_MIDVEIN_MAX
    # Worst station. A station that is not here was dropped for being below the loader's confidence
    # filter, so it is worse than anything that survived it; an incomplete trace takes the floor of
    # the ramp instead of the best-of-the-survivors minimum it used to be handed. Zero and the
    # filter threshold score the same here (KPT_CONF_MIN_ZERO is that threshold), so scoring an
    # unplaced station as zero everywhere costs nothing and keeps one rule for both statistics.
    worst = q.kpt_conf_min if n >= N_MIDVEIN_MAX else 0.0
    mean_s = _ramp(cov_mean, KPT_CONF_MEAN_ZERO, KPT_CONF_MEAN_ONE)
    min_s = _ramp(worst, KPT_CONF_MIN_ZERO, KPT_CONF_MIN_ONE)
    n_s = _ramp(float(n), MIDVEIN_KPTS_ZERO, MIDVEIN_KPTS_ONE)

    if math.isfinite(mean_s) and mean_s < REASON_SCORE_THRESHOLD:
        reasons.append(f"midvein keypoint confidence {q.kpt_conf_mean:.2f} mean, "
                       f"{q.kpt_conf_min:.2f} worst")
    if n < N_MIDVEIN_MAX:
        reasons.append(f"only {n}/{N_MIDVEIN_MAX} midvein trace points survived the confidence "
                       "filter (the missing ones are scored as unplaced, not ignored)")
    # _amean, not _gmean: see _amean. A trace missing stations is less trustworthy, not unusable,
    # and vetoing it here would re-create in the confidence inputs exactly the hard zero that
    # MIDVEIN_KPTS_ZERO was lowered to remove.
    score = _amean([mean_s, min_s, n_s])
    if not math.isfinite(score):
        reasons.append("midvein trace sub-scores undefined; term dropped from the score")
    return score, reasons


def archetype_subscores(sym: dict, qual: QualityDiagnostics) -> tuple[dict[str, float], list[str]]:
    """The four term scores in ``[0, 1]`` (NaN = unavailable) and every reason string raised.

    Exposed separately from :func:`archetype_score` so the report can plot, re-weight or audit the
    terms without re-deriving them, and so a symmetry-free composite is one call away.
    """
    s, r_s = _symmetry_term(sym if isinstance(sym, dict) else {})
    i, r_i = _integrity_term(qual)
    c, r_c = _completeness_term(qual)
    t, r_t = _trace_term(qual)
    return {"symmetry": s, "integrity": i, "completeness": c, "trace": t}, [*r_s, *r_i, *r_c, *r_t]


def archetype_score(sym: dict, qual: QualityDiagnostics, *,
                    weights: Optional[dict[str, float]] = None) -> tuple[float, list[str]]:
    """Composite "how archetypal is this leaf" score in ``[0, 1]`` plus the reasons for its penalties.

    ``sym`` is the metrics dict for one leaf (see :func:`_sym_value` for the key layouts accepted);
    ``qual`` comes from :func:`compute_quality`. ``weights`` overrides :data:`ARCHETYPE_WEIGHTS`
    key by key; unknown keys are ignored and a weight of 0 removes that term exactly.

    The four terms are combined as a WEIGHTED GEOMETRIC MEAN, so each contributes ``term ** weight``
    and a single term at 0 drives the whole score to 0. That is the intended semantics of
    "archetypal": it is a conjunction of properties, not a tally that a strength elsewhere can
    compensate for. A term whose inputs are missing is dropped and the remaining weights are
    renormalized, with a reason recorded -- a missing measurement never scores as a passing one.

    **The symmetry term is the one exception, and it is disqualifying.** Drop-and-renormalize is the
    right rule for a nice-to-have term; for the term that IS the hypothesis it is catastrophic --
    a leaf whose symmetry was never measured used to score a clean 1.0 off the three quality terms
    alone and outrank every leaf whose symmetry actually WAS measured. So when symmetry carries any
    weight and cannot be evaluated, this returns ``nan`` rather than a composite of what is left.
    Ask for the symmetry-free composite explicitly with ``weights={"symmetry": 0.0}`` and the
    disqualification lifts with it, because then no symmetry claim is being made.

    Returns ``nan`` when no term could be evaluated at all. The hard yes/no lives in
    :func:`is_archetypal`, kept separate so this function's return type stays a plain
    ``(score, reasons)`` pair for the report.

    Every threshold behind these numbers is a first proposal to be tuned against the ranked visual
    output -- see the constants block above.
    """
    w = dict(ARCHETYPE_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items() if k in w})

    terms, reasons = archetype_subscores(sym, qual)
    if _symmetry_required(w) and not math.isfinite(terms["symmetry"]):
        return float("nan"), reasons
    score = _gmean([terms[k] for k in w], [w[k] for k in w])
    return score, reasons


def _symmetry_required(weights: Optional[dict[str, float]]) -> bool:
    """Does this weighting actually claim to be scoring symmetry?

    True unless the caller deliberately zeroed the symmetry weight, which is the documented way to
    ask for the symmetry-free composite. Only then may a missing symmetry measurement be ignored.
    """
    if not weights:
        return ARCHETYPE_WEIGHTS["symmetry"] > 0
    try:
        return float(weights.get("symmetry", ARCHETYPE_WEIGHTS["symmetry"])) > 0
    except (TypeError, ValueError):
        return True


# ==========================================================================
# ported from experiments/bilateral_symmetry/run.py
# ==========================================================================

# --- exemplar gate policy -------------------------------------------------------------------
# Owned by the driver, not quality.py, so it can be tuned without touching the scoring module.
# Three of quality.py's stock gates are deliberately NOT applied here:
#   holes       -- the shape under test is the HOLES-FILLED silhouette (LM3's `lamina` /
#                  Lamina_Holes_Mask product), so an insect hole is not part of the outline whose
#                  symmetry is being measured. hole_frac stays in the CSV as a diagnostic but is
#                  zeroed before scoring so it cannot influence the composite either.
#   n_components-- NOTE this is NOT because the holes are filled: filling a hole removes an interior
#                  void, it cannot merge two disjoint blobs, and measured on this cohort the counts
#                  are identical on the holes-filled and holes-punched masks for all 105 leaves.
#                  The raw count is dropped because it counts SPECKS: 18 leaves have >1 component
#                  but 12 of those have under 0.1% of their area outside the main blob, and one
#                  leaf scoring 0.96 was vetoed by a second component holding 0.001%. largest_frac
#                  keeps the part that matters -- it flags the 2 genuinely fragmented masks.
#   flush_edge  -- measured r = 0.85 against 1/sqrt(area_px), i.e. it is a leaf-SIZE proxy rather
#                  than a cut-off detector; `truncated` answers that question directly.
GATE_MIN_LARGEST_FRAC = 0.995


GATE_MIN_MIDVEIN_KPTS = 10


def gates_pass(q: Any) -> tuple[bool, list[str]]:
    """Structural vetoes -- faults that disqualify a leaf regardless of how symmetric it looks."""
    reasons: list[str] = []
    if bool(getattr(q, "truncated", False)):
        reasons.append("detection box was clipped by the sheet edge")
    if not float(getattr(q, "area_px", 0) or 0) > 0:
        reasons.append("empty mask")
    lf = float(getattr(q, "largest_frac", float("nan")) or float("nan"))
    if not (np.isfinite(lf) and lf >= GATE_MIN_LARGEST_FRAC):
        reasons.append(f"largest component holds only {lf:.4f} of the mask area")
    nk = float(getattr(q, "n_midvein_kpts", 0) or 0)
    if nk < GATE_MIN_MIDVEIN_KPTS:
        reasons.append(f"only {nk:.0f} midvein keypoints survived the confidence filter")
    return (not reasons), reasons

# ==========================================================================
# production entry point (the stage and the Reporter both start here)
# ==========================================================================
@dataclass
class BilateralMeasurement:
    """Everything one leaf contributes, both the numbers and the frame needed to redraw it.

    Mirrors ``core.petiole.PetioleWidth``: a plain result object the stage maps onto a DB row, with
    no DB or filesystem knowledge of its own.
    """

    # metrics (midvein axis, holes-filled silhouette)
    si_a: float = _NAN            # Shi et al. standardized index; 0 = perfect
    a_star: float = _NAN          # signed total imbalance in [-1,1]; + = viewer's LEFT half larger
    dice: float = _NAN            # straightened mirrored-half overlap; 1 = perfect
    sinuosity: float = _NAN       # midvein arclength / straight tip-base distance
    # composite
    archetype_score: float = _NAN
    term_symmetry: float = _NAN
    term_integrity: float = _NAN
    term_completeness: float = _NAN
    term_trace: float = _NAN
    gates_pass: bool = False
    is_archetypal: bool = False
    reasons: list = None
    # quality diagnostics
    largest_frac: float = _NAN
    solidity: float = _NAN
    perimeter_ratio: float = _NAN
    hole_frac: float = _NAN
    kpt_conf_mean: float = _NAN
    kpt_conf_min: float = _NAN
    n_midvein_kpts: int = 0
    truncated: bool = False
    # frame geometry -- lets the Reporter rebuild the oriented frame with no landmark re-derivation
    angle_cw: float = 0.0
    crop_w: int = 0
    crop_h: int = 0
    mask_w: int = 0
    mask_h: int = 0
    tip_x: Optional[float] = None
    tip_y: Optional[float] = None
    base_x: Optional[float] = None
    base_y: Optional[float] = None
    midvein: Optional[list] = None    # (N,2) tip->base polyline, ORIENTED coords
    n_bins: int = 0


def bins_for(area_px: float, n_bins_max: int = N_BINS) -> int:
    """Arclength bin count for a lamina of ``area_px`` pixels -- see :data:`MIN_PX_PER_BIN`.

    ``n_bins_max`` is an UPPER bound, not a target: a big leaf gets it, a small leaf gets fewer
    but better-populated bins instead of a finely-sliced noise floor.
    """
    if not np.isfinite(area_px) or area_px <= 0:
        return int(MIN_BINS)
    return int(np.clip(round(float(area_px) / MIN_PX_PER_BIN), MIN_BINS, int(n_bins_max)))


def measure_leaf(leaf: OrientedLeaf, *, n_bins: int = N_BINS,
                 min_score: float = 0.50) -> Optional[BilateralMeasurement]:
    """Measure one oriented leaf on the midvein axis. ``None`` when there is nothing to measure.

    Only the MIDVEIN frame is built. The experiment also built a chord frame as its control, and
    ``build_axis`` still supports it, but calling it here would roughly double the cost of the stage
    for a number production does not keep. The QC panel's dashed chord line needs only the two
    endpoints, which are carried on the measurement.

    ``hole_frac`` is zeroed before scoring: the measured shape is the holes-filled silhouette, so a
    hole must not be able to reach the composite. It is still reported as a diagnostic.

    ``n_bins`` is the MAXIMUM bin count; the actual count comes from :func:`bins_for` so that every
    bin is populated enough for its ratio to mean something. Read :data:`MIN_PX_PER_BIN` before
    changing it -- a fixed count makes ``si_a`` a leaf-size proxy.
    """
    mv, tb = leaf.midvein(), leaf.tip_base()
    if tb is None or mv is None or len(mv) < 3:
        return None

    tip, base = tb
    frame = build_axis(leaf.silhouette, "midvein", midvein=mv, tip=tip, base=base)
    if frame is None:
        return None

    n_bins = bins_for(float(leaf.silhouette.sum()), n_bins)
    prof = profiles(frame, n_bins=n_bins)
    a_prof = area_asymmetry_profile(prof)
    valid = np.isfinite(a_prof)
    sum_l, sum_r = float(prof.area_l.sum()), float(prof.area_r.sum())

    si_a = float(np.abs(a_prof[valid]).mean()) if int(valid.sum()) else _NAN
    a_star = _div(sum_l - sum_r, sum_l + sum_r)
    _iou, dice, _sd = _overlap(frame, 256, 128)

    qual = compute_quality(leaf, frame)
    # the measured shape has its holes filled, so hole_frac must not reach the composite
    scored = QualityDiagnostics(**{**asdict(qual), "hole_frac": 0.0})
    sym = {"midvein": {"si_a": si_a, "dice": dice}}
    terms, reasons = archetype_subscores(sym, scored)
    score, _r = archetype_score(sym, scored)
    passed, vetoes = gates_pass(qual)
    reasons = [*vetoes, *reasons]

    return BilateralMeasurement(
        si_a=si_a, a_star=a_star, dice=dice, sinuosity=frame.sinuosity,
        archetype_score=score,
        term_symmetry=terms["symmetry"], term_integrity=terms["integrity"],
        term_completeness=terms["completeness"], term_trace=terms["trace"],
        gates_pass=passed,
        # a leaf whose symmetry could not be measured must never rank as archetypal
        is_archetypal=bool(passed and math.isfinite(score) and score >= float(min_score)),
        reasons=reasons,
        largest_frac=qual.largest_frac, solidity=qual.solidity,
        perimeter_ratio=getattr(qual, "perimeter_ratio", _NAN), hole_frac=qual.hole_frac,
        kpt_conf_mean=qual.kpt_conf_mean, kpt_conf_min=qual.kpt_conf_min,
        n_midvein_kpts=int(qual.n_midvein_kpts or 0), truncated=bool(qual.truncated),
        angle_cw=float(leaf.angle_cw), crop_h=int(leaf.crop_shape[0]), crop_w=int(leaf.crop_shape[1]),
        mask_h=int(leaf.shape[0]), mask_w=int(leaf.shape[1]),
        tip_x=float(tip[0]), tip_y=float(tip[1]), base_x=float(base[0]), base_y=float(base[1]),
        midvein=[[float(x), float(y)] for x, y in frame.path],
        n_bins=int(n_bins),
    )
