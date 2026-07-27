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


@dataclass
class RulerCFRow:
    ruler_class_id: int
    unit_type: str
    minimum_unit: Optional[str] = None
    second_unit: Optional[str] = None
    px_per_mm: Optional[float] = None
    cf_px_per_cm: Optional[float] = None
    cf_px_per_inch: Optional[float] = None
    agreement: Optional[float] = None
    n_ticks: Optional[int] = None
    is_valid: bool = True
    ruler_cf_id: Optional[int] = None


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
class MorphRow:
    """One leaf-instance morphology row (working-frame pixels). See core.morphometrics."""
    leaf_id: int
    detection_id: int
    instance_index: int
    cls_name: str
    crop_box: XYXY                     # plant_detection leaf box (working coords)
    area_px: float
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
    normalized: bool = False


@dataclass
class ReportBundle:
    """Everything Reporter.infer needs, assembled on the parent (picklable)."""
    specimen_id: int
    stem: str
    original_path: str
    work_scale: float
    cf_px_per_cm: Optional[float]
    detections: list[Any]              # box rows with .cls_name/.conf/.xyxy/.source ('archival'|'plant')
    leaves: list[Any]                  # leaf rows with mask geometry (parent/working coords)
    reports_dir: str
    working_path: str = ""             # for per-crop mask exports (crop-frame pixels)
    crop_boxes: dict = field(default_factory=dict)   # plant detection_id -> (x1,y1,x2,y2) working coords
    morphology: list[Any] = field(default_factory=list)   # leaf_morphology rows (rotated bbox etc.)
    landmarks: list[Any] = field(default_factory=list)    # leaf_landmark rows (keypoints, working+crop coords)
    landmark_measurements: list[Any] = field(default_factory=list)  # leaf_landmark_measurement rows
