"""Shared record dataclasses — the vocabulary every stage, the DB, and inference use.

Keeping these in one module means modules and ``ProjectDB`` agree on field names by
construction. Coordinates are in the WORKING-copy pixel frame unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

XYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class Detection:
    """Return shape of the detector inference wrappers (matches the training repo)."""
    cls_id: int
    cls_name: str
    conf: float
    xyxy: XYXY


@dataclass(frozen=True)
class Unit:
    """Picklable per-specimen payload carried in a WorkItem for detector stages."""
    specimen_id: int
    stem: str
    working_path: str
    original_path: str
    crops_dir: str
    width: int = 0
    height: int = 0


@dataclass
class DetRow:
    """One detection box to persist (archival_detection / plant_detection)."""
    cls_id: int
    cls_name: str
    conf: float
    xyxy: XYXY
    tag: str = ""                      # bare tag, e.g. "R" / "LW"; crop file adds __R__
    crop_path: Optional[str] = None
    detection_id: Optional[int] = None
    # same-class duplicate suppression (core.box_dedup). A suppressed box is stored for provenance
    # but has no crop and is excluded from every downstream read.
    suppressed: bool = False
    suppressed_by_index: Optional[int] = None   # index in this batch of the keeper box (-> detection_id at insert)
    suppress_overlap: Optional[float] = None


@dataclass
class CropRef:
    """A previously-saved detection crop handed to downstream stages (ruler/leaf)."""
    detection_id: int
    specimen_id: int
    cls_name: str
    crop_path: str
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    frame_width: int = 0
    frame_height: int = 0


@dataclass
class PhenologyResult:
    leaves: tuple[bool, int]           # (present, count)
    flowers: tuple[bool, int]
    fruits: tuple[bool, int]


@dataclass
class RulerClassRow:
    detection_id: int
    unit_type: str
    votes: dict = field(default_factory=dict)
    conf: Optional[float] = None
    ruler_class_id: Optional[int] = None
    squarify_path: Optional[str] = None   # pre-made four-tile collage (_ruler_squarify), reused by the lattice CF + QC




@dataclass
class LeafRow:
    """One segmented instance mask. Geometry is in PARENT (working) coords."""
    detection_id: int
    instance_index: int
    cls_id: int
    cls_name: str
    conf: Optional[float]
    parent_instance_index: Optional[int]
    mask_format: str                   # 'polygon_xy' | 'coco_rle'
    mask_data: str
    frame_width: int
    frame_height: int
    bbox: XYXY
    num_parts: int = 1
    area_px: Optional[float] = None
    perimeter_px: Optional[float] = None
    leaf_id: Optional[int] = None


@dataclass
class SpecimenMaskResult:
    """SpecimenSegmenter inference payload (worker -> parent). Masks are PNG-encoded (small
    across the process boundary) at WORKING resolution; ``centers`` are paper-sampling box
    centers in working coords (for the QC overlay's blue boxes)."""
    final_png: bytes                   # cleaned (post-paperclean) binary mask, PNG {0,255}
    removed_png: bytes                 # pixels paperclean deleted (the red "refined" region), PNG {0,255}
    frame_width: int
    frame_height: int
    centers: list = field(default_factory=list)   # [(x, y), ...] paper-sample box centers, working coords
    area_frac: float = 0.0             # foreground fraction of the final mask (QC)
    model_name: str = ""               # which export produced it (provenance)


@dataclass
class MorphRow:
    """One leaf-instance morphology row (working-frame pixels). See core.morphometrics."""
    leaf_id: int
    detection_id: int
    instance_index: int
    cls_name: str
    crop_box: XYXY                     # plant_detection leaf box (working coords)
    area_px: float                     # area inside the Leaf outer boundary (INCLUDES holes)
    perimeter_px: float
    centroid: tuple[float, float]
    convex_hull_area: float
    convexity: float
    concavity: float
    circularity: float
    aspect_ratio: float
    n_vertices: int
    bbox: XYXY                         # axis-aligned (working coords)
    rotate_angle: float
    dim_max: float                     # leaf length (rotated long side)
    dim_min: float                     # leaf width  (rotated short side)
    rotated_bbox_json: str             # JSON [[x,y],...] 4 corners, working coords
    circle: tuple[float, float, float] # min enclosing circle (cx, cy, radius)
    # hole-aware lamina areas (area_px already includes holes; see leaf_morphology schema)
    lamina_area_incl_holes_px: float = 0.0   # == area_px (full silhouette, with holes)
    lamina_area_excl_holes_px: float = 0.0   # tissue area (holes removed)
    lamina_hole_area_px: float = 0.0         # sum of the leaf's hole areas
    n_holes: int = 0
    morph_id: Optional[int] = None


@dataclass
class LandmarkRow:
    """One predicted leaf keypoint. ``x``/``y`` are WORKING (parent) coords with the training
    white-pad removed; ``x_crop``/``y_crop`` are the crop-frame coords. See core.landmarks."""
    detection_id: int
    instance_index: int
    kpt_index: int
    kpt_name: str
    x: float
    y: float
    x_crop: float
    y_crop: float
    conf: float
    landmark_id: Optional[int] = None


@dataclass
class LandmarkMeasureRow:
    """One leaf's derived landmark measurements (see core.landmark_metrics). Every metric is
    Optional -- ``None`` when the keypoints it needs were occluded/missing. Lengths are
    WORKING-frame pixels; angles are degrees. Keyed by ``(detection_id, instance_index)``."""
    detection_id: int
    instance_index: int
    lamina_trace_length: Optional[float] = None
    lamina_extent: Optional[float] = None
    lamina_tip_base_length: Optional[float] = None
    leaf_width: Optional[float] = None
    apex_angle: Optional[float] = None
    apex_angle_type: Optional[str] = None
    base_angle: Optional[float] = None
    base_angle_type: Optional[str] = None
    petiole_trace_length: Optional[float] = None
    lamina_curvature: Optional[float] = None
    curvature_point: Optional[int] = None
    lamina_centroid_x: Optional[float] = None
    lamina_centroid_y: Optional[float] = None
    n_present: int = 0
    measure_id: Optional[int] = None


@dataclass
class PetioleRow:
    """One leaf's petiole width (PetioleWidth stage). Segments are WORKING coords. See core.petiole."""
    leaf_id: int
    detection_id: int
    instance_index: int
    width_px: Optional[float] = None
    length_px: Optional[float] = None
    n_samples: int = 0
    touches_leaf: bool = False
    measure_location: str = "none"
    width_segment: Optional[list] = None          # [[x1,y1],[x2,y2]] working coords
    sample_segments: list = field(default_factory=list)   # [[[x1,y1],[x2,y2]], ...] working coords
    leaf_mass_per_area: Optional[float] = None    # LMA proxy in g/m^2 (see PetioleWidth._lma)
    petiole_id: Optional[int] = None


@dataclass
class OrientationRow:
    """Per-leaf upright orientation (LeafOrientation stage), stored on the leaf_morphology row.
    ``angle_cw`` is clockwise degrees to bring the lamina tip up; ``None`` when ``success`` is False."""
    leaf_id: int
    success: bool
    angle_cw: Optional[float] = None
    # the rotated bbox's two side lengths split by ORIENTATION (not by which is geometrically longer):
    # length runs along the tip->base axis, width perpendicular to it. None when unoriented.
    bbox_length: Optional[float] = None
    bbox_width: Optional[float] = None


@dataclass
class Grounded:
    leaf_id: int
    area_cm2: Optional[float] = None
    perimeter_cm: Optional[float] = None
    bbox_w_cm: Optional[float] = None
    bbox_h_cm: Optional[float] = None


@dataclass
class SpecimenRecord:
    image_name: str
    image_stem: str
    original_path: str
    working_path: str
    width: int = 0
    height: int = 0
    original_width: int = 0
    original_height: int = 0
    work_scale: float = 1.0
    orig_size_bytes: int = 0
    orig_mtime: float = 0.0
    normalized: bool = False    # a _tmp_original copy was written (RGB convert and/or downscale)
    downsampled: bool = False   # pixels were DISCARDED (long side capped) -- a strict subset of the above


@dataclass
class ReportBundle:
    """Everything Reporter.infer needs, assembled on the parent (picklable)."""
    specimen_id: int
    stem: str
    work_scale: float                  # kept for provenance; no Reporter output rescales by it now
    cf_px_per_cm: Optional[float]      # WORKING frame -- the frame every output is rendered in
    detections: list[Any]              # box rows with .cls_name/.conf/.xyxy/.source ('archival'|'plant')
    leaves: list[Any]                  # leaf rows with mask geometry (parent/working coords)
    reports_dir: str
    working_path: str = ""             # for per-crop mask exports (crop-frame pixels)
    crop_boxes: dict = field(default_factory=dict)   # plant detection_id -> (x1,y1,x2,y2) working coords
    morphology: list[Any] = field(default_factory=list)   # leaf_morphology rows (rotated bbox etc.)
    landmarks: list[Any] = field(default_factory=list)    # leaf_landmark rows (keypoints, working+crop coords)
    landmark_measurements: list[Any] = field(default_factory=list)  # leaf_landmark_measurement rows
    petioles: list[Any] = field(default_factory=list)     # leaf_petiole rows (width + sample segments)
    bilateral: list[Any] = field(default_factory=list)    # bilateral_symmetry rows (metrics + QC frame)
    specimen_mask: Any = None          # specimen_mask row (final/removed PNG paths + sample centers), or None
    ruler_lattice: Any = None          # lattice ruler-CF record {image, crops} for the Overlay_Ruler_Lattice QC panel, or None
