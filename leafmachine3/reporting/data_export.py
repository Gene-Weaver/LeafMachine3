"""CSV export of the project database -> ``reports/Data/``.

The Reporter's LAST step. Every other Reporter output is a picture; this one is the numbers behind
them, written straight out of the per-project SQLite so a run is analyzable without opening the DB.

    reports/Data/leaf_measurements.csv        ONE ROW PER LEAF -- the master file
    reports/Data/specimen_summary.csv         one row per input image (+ per-sheet roll-ups)
    reports/Data/phenology.csv                one row per sheet, LM2's phenology.csv layout
    reports/Data/detections.csv               one row per detection box, both detectors
    reports/Data/landmarks.csv                one row per predicted keypoint (31 per leaf)
    reports/Data/ruler_conversion_factor.csv  one row per sheet -- the CF verdict + its audit trail
    reports/Data/ruler_crops.csv              one row per candidate ruler crop
    reports/Data/run_stages.csv               one row per pipeline stage (the run ledger)
    reports/Data/stage_errors.csv             one row per per-image stage failure
    reports/Data/data_dictionary.csv          every column above: file, units, meaning

FRAME AND UNITS. Every pixel measurement LM3 stores is in the WORKING frame -- the resized copy the
stages actually analyzed, not the original. Columns are suffixed accordingly (``_px``, ``_cm``,
``_cm2``); angles are degrees and are suffixed in prose, not in the name, because that is what the
DB calls them. ``work_scale`` is on every row, so original-frame pixels are ``value_px /
work_scale``. Nothing here is rescaled on the way out.

WHAT ``_cm`` MEANS. MetricGrounding divides by ``specimen.cf_px_per_cm`` (exported as
``cf_px_per_cm``), and ``cf_source`` -- stored on the specimen, not derived here -- says which CF
that is: ``measured_from_ruler`` (the lattice stage published it at high confidence) or
``predicted_from_megapixels`` (no ruler, or the lattice did not pass, and
``modules.ruler_cf.use_CF_predicted_by_MP`` was on). ``none`` means the sheet has no CF and every
``_cm`` column is empty -- the default for those sheets, a visible absence rather than a gap
silently filled. ``cf_px_per_cm_predicted_by_mp`` is always carried alongside for comparison.

THIS IS A PROJECTION, NOT A CALCULATION. Apart from a handful of derived identity/geometry columns
(marked "derived" in the data dictionary), every value is a column the pipeline stored. The exporter
computes no new science: if a measurement is not in the DB it is not in the CSV.
"""
from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

log = logging.getLogger("leafmachine3.data_export")

#: Subfolder of ``reports/`` the bundle is written into (overridable via ``report.data.folder``).
DEFAULT_FOLDER = "Data"


@dataclass(frozen=True)
class Column:
    """One output column: its name, its unit, and what it means (straight into the dictionary)."""
    name: str
    units: str
    desc: str


# --------------------------------------------------------------------------- #
# leaf_measurements.csv -- the master file
# --------------------------------------------------------------------------- #
# Grouped the way a reader reads it: who am I -> what frame am I in -> what was measured.
_LEAF_COLUMNS: tuple[Column, ...] = (
    # -- identity ---------------------------------------------------------- #
    Column("leaf_uid", "", "derived: stable id for THIS leaf instance, "
                           "'<image_stem>__<x1>_<y1>_<x2>_<y2>__i<instance_index>'. Unlike leaf_id "
                           "it survives a re-run, because it is built from the image and the box."),
    Column("crop_file_token", "", "derived: '<image_stem>__<x1>_<y1>_<x2>_<y2>' -- the filename token "
                                  "every exported file for this leaf CROP carries (leaf products, "
                                  "landmark and petiole overlays, ECT products). Per crop, NOT per "
                                  "instance: when one crop holds several leaves they share it."),
    Column("specimen_id", "", "Row id of the parent sheet in this project's database."),
    Column("image_name", "", "Original image filename, with extension."),
    Column("image_stem", "", "Original image filename without extension; the stem every output is named by."),
    Column("original_path", "", "Immutable path to the source image LM3 ingested."),
    Column("working_path", "", "Path to the working copy every stage actually opened."),
    Column("leaf_id", "", "Row id of this leaf instance mask. Project-local; not stable across re-runs."),
    Column("detection_id", "", "Row id of the plant-detector box this leaf was segmented inside."),
    Column("instance_index", "", "Segmentation instance within the crop; 0 is the primary leaf."),
    Column("detection_class", "", "Detector class of the parent box: Leaf_WHOLE or Leaf_PARTIAL."),
    Column("detection_conf", "0-1", "Plant-detector confidence for the parent box."),
    Column("segmentation_conf", "0-1", "Leaf-segmenter confidence for this instance mask."),
    Column("n_mask_parts", "count", "Disconnected parts in the instance mask; >1 means a split leaf."),
    Column("crop_path", "", "Working leaf crop the segmenter, landmark and ruler stages read."),
    Column("crop_x1", "px", "Parent leaf box, left edge (working frame)."),
    Column("crop_y1", "px", "Parent leaf box, top edge (working frame)."),
    Column("crop_x2", "px", "Parent leaf box, right edge (working frame)."),
    Column("crop_y2", "px", "Parent leaf box, bottom edge (working frame)."),

    # -- frame + scale ------------------------------------------------------ #
    Column("working_width", "px", "Width of the working image every measurement below is in."),
    Column("working_height", "px", "Height of the working image every measurement below is in."),
    Column("original_width", "px", "Width of the untouched original, kept for provenance."),
    Column("original_height", "px", "Height of the untouched original, kept for provenance."),
    Column("work_scale", "ratio", "working / original long side. Divide any _px value by this for "
                                  "original-frame pixels. 1.0 when nothing was resized."),
    Column("downsampled", "0/1", "1 iff pixels were discarded capping the long side at ingest.max_working_dim."),
    Column("original_mp", "megapixels", "Original width*height/1e6; the input to the MP->CF regression."),
    Column("cf_source", "", "Which CF produced this row's _cm columns. 'measured_from_ruler' = the "
                            "published ruler CF; 'predicted_from_megapixels' = the megapixel "
                            "prediction, used because the sheet had no ruler or the lattice did not "
                            "pass (use_CF_predicted_by_MP on); 'none' = no CF, so every _cm column "
                            "on this row is empty."),
    Column("cf_px_per_cm", "px/cm", "The CF this row's _cm columns were grounded with (working "
                                    "frame). Measured or predicted: see cf_source. Empty when "
                                    "cf_source is 'none'."),
    Column("cf_px_per_cm_predicted_by_mp", "px/cm", "Resolution-based CF estimate from the megapixel "
                                                    "regression (~4.5 px/cm rmse), carried on every "
                                                    "row. It IS the CF in use only where cf_source is "
                                                    "'predicted_from_megapixels'."),
    Column("ruler_unit_type", "", "Unit type of the rulers that produced the published CF."),
    Column("ruler_class_type", "", "Per-sheet consensus ruler unit type from the classifier ensemble."),
    Column("has_leaves", "0/1", "Sheet-level phenology: leaves present."),
    Column("has_flowers", "0/1", "Sheet-level phenology: flowers present."),
    Column("has_fruits", "0/1", "Sheet-level phenology: fruits present."),

    # -- lamina area + outline ---------------------------------------------- #
    Column("lamina_area_incl_holes_px", "px^2", "Area inside the leaf's outer boundary -- the full "
                                                "silhouette, holes INCLUDED."),
    Column("lamina_area_excl_holes_px", "px^2", "Leaf tissue only: the silhouette with its holes "
                                                "subtracted. Divide by cf_px_per_cm^2 for cm^2."),
    Column("lamina_hole_area_px", "px^2", "Total area of the leaf's Hole instances."),
    Column("n_holes", "count", "Number of Hole instances inside this leaf."),
    Column("lamina_perimeter_px", "px", "Perimeter of the leaf's outer boundary."),
    Column("lamina_area_incl_holes_cm2", "cm^2", "lamina_area_incl_holes_px grounded by the sheet CF (see cf_source)."),
    Column("lamina_perimeter_cm", "cm", "lamina_perimeter_px grounded by the sheet CF (see cf_source)."),
    Column("convex_hull_area_px", "px^2", "Area of the leaf outline's convex hull."),
    Column("convexity", "ratio", "How close the outline is to its own convex hull; 1 = convex."),
    Column("concavity", "ratio", "Complement of convexity."),
    Column("circularity", "ratio", "Isoperimetric ratio; 1 = a perfect circle."),
    Column("aspect_ratio", "ratio", "Ratio of the outline's two principal extents."),
    Column("n_vertices", "count", "Vertices in the stored outline polygon."),
    Column("centroid_x", "px", "Leaf outline centroid, x (working frame)."),
    Column("centroid_y", "px", "Leaf outline centroid, y (working frame)."),

    # -- bounding boxes ------------------------------------------------------ #
    Column("bbox_x1", "px", "Axis-aligned mask bbox, left edge (working frame)."),
    Column("bbox_y1", "px", "Axis-aligned mask bbox, top edge (working frame)."),
    Column("bbox_x2", "px", "Axis-aligned mask bbox, right edge (working frame)."),
    Column("bbox_y2", "px", "Axis-aligned mask bbox, bottom edge (working frame)."),
    Column("bbox_w_px", "px", "derived: bbox_x2 - bbox_x1."),
    Column("bbox_h_px", "px", "derived: bbox_y2 - bbox_y1."),
    Column("bbox_w_cm", "cm", "Axis-aligned bbox width grounded by the sheet CF (see cf_source)."),
    Column("bbox_h_cm", "cm", "Axis-aligned bbox height grounded by the sheet CF (see cf_source)."),
    Column("rotate_angle", "degrees", "Rotation of the minimum-area bounding box."),
    Column("rotated_bbox_dim_max_px", "px", "Rotated bbox LONG side. Geometric, not biological: a "
                                            "leaf wider than it is long puts its width here."),
    Column("rotated_bbox_dim_min_px", "px", "Rotated bbox SHORT side, same caveat as dim_max."),
    Column("rotated_bbox_length_px", "px", "Rotated bbox side along the tip->base axis. Reserved; "
                                           "empty until the orientation-aware assignment ships."),
    Column("rotated_bbox_width_px", "px", "Rotated bbox side perpendicular to tip->base. Reserved."),
    Column("circle_cx", "px", "Minimum enclosing circle centre, x."),
    Column("circle_cy", "px", "Minimum enclosing circle centre, y."),
    Column("circle_radius_px", "px", "Minimum enclosing circle radius."),
    Column("oriented_leaf_success", "0/1", "1 if an upright (tip-up) orientation was determined."),
    Column("oriented_leaf_rotation_angle_degreesCW", "degrees",
           "Clockwise rotation that brings this leaf tip-up; what the Leaf_Oriented products used."),

    # -- landmark measurements ----------------------------------------------- #
    Column("n_landmarks_present", "count", "Confident keypoints behind the metrics below. Occluded "
                                           "keypoints leave their metric empty, never a guessed value."),
    Column("lamina_trace_length_px", "px", "Summed distance along the 15 midvein trace points."),
    Column("lamina_extent_px", "px", "Straight chord between the first and last midvein points."),
    Column("lamina_tip_base_length_px", "px", "Straight distance, lamina tip to lamina base."),
    Column("leaf_width_px", "px", "Distance between the left and right width landmarks."),
    Column("petiole_trace_length_px", "px", "Summed distance along the 5 petiole trace points."),
    Column("lamina_trace_length_cm", "cm", "lamina_trace_length_px grounded by the sheet CF (see cf_source)."),
    Column("lamina_extent_cm", "cm", "lamina_extent_px grounded by the sheet CF (see cf_source)."),
    Column("lamina_tip_base_length_cm", "cm", "lamina_tip_base_length_px grounded by the sheet CF (see cf_source)."),
    Column("landmark_leaf_width_cm", "cm", "leaf_width_px grounded by the sheet CF (see cf_source)."),
    Column("petiole_trace_length_cm", "cm", "petiole_trace_length_px grounded by the sheet CF (see cf_source)."),
    Column("apex_angle", "degrees", "Angle at the apex centre landmark."),
    Column("apex_angle_type", "", "acute (<90), obtuse (>=90), or reflex."),
    Column("base_angle", "degrees", "Angle at the base centre landmark."),
    Column("base_angle_type", "", "acute (<90), obtuse (>=90), or reflex."),
    Column("lamina_curvature", "degrees", "Maximum midvein bend: 0 is straight, larger is more curved."),
    Column("curvature_point", "index", "Midvein point index where that maximum bend occurs."),

    # -- petiole -------------------------------------------------------------- #
    Column("petiole_width_px", "px", "Median perpendicular petiole width near the blade junction."),
    Column("petiole_length_px", "px", "Petiole centerline length, lamina base to petiole tip."),
    Column("petiole_width_cm", "cm", "petiole_width_px grounded by the sheet CF (see cf_source)."),
    Column("petiole_length_cm", "cm", "petiole_length_px grounded by the sheet CF (see cf_source)."),
    Column("petiole_n_samples", "count", "Perpendicular samples the median width was taken over."),
    Column("petiole_touches_leaf", "0/1", "1 if the petiole mask actually meets the lamina."),
    Column("petiole_measure_location", "", "Where the width was taken: near_base, or none."),
    Column("leaf_mass_per_area", "g/m^2", "Leaf-mass-per-area proxy from Royer petiole scaling. "
                                          "Needs no ruler CF, so it is present even when _cm columns "
                                          "are empty."),

    # -- bilateral symmetry ---------------------------------------------------- #
    Column("si_a", "index", "Standardized asymmetry index about the traced midvein; 0 is perfect."),
    Column("a_star", "-1..1", "Signed total imbalance; positive means the viewer's LEFT half is larger."),
    Column("dice", "0-1", "Overlap of the straightened mirrored halves; 1 is perfect."),
    Column("sinuosity", "ratio", "Midvein arclength over straight tip-base distance."),
    Column("archetype_score", "0-1", "Composite leaf-quality score used to rank archetypal leaves."),
    Column("term_symmetry", "0-1", "Symmetry term of archetype_score."),
    Column("term_integrity", "0-1", "Integrity term of archetype_score."),
    Column("term_completeness", "0-1", "Completeness term of archetype_score."),
    Column("term_trace", "0-1", "Midvein-trace-quality term of archetype_score."),
    Column("gates_pass", "0/1", "1 if the leaf cleared every structural veto."),
    Column("is_archetypal", "0/1", "1 if it passed the gates AND scored above the archetype threshold."),
    Column("largest_frac", "ratio", "Fraction of mask area in its largest connected component."),
    Column("solidity", "ratio", "Mask area over its convex hull area."),
    Column("perimeter_ratio", "ratio", "Perimeter relative to an equal-area disc; high means ragged."),
    Column("hole_frac", "ratio", "Fraction of the silhouette that is holes."),
    Column("kpt_conf_mean", "0-1", "Mean confidence of the keypoints behind the symmetry axis."),
    Column("kpt_conf_min", "0-1", "Lowest keypoint confidence behind the symmetry axis."),
    Column("n_midvein_kpts", "count", "Confident midvein keypoints used for the axis."),
    Column("truncated", "0/1", "1 if the leaf runs off the edge of its crop."),
)


# --------------------------------------------------------------------------- #
# specimen_summary.csv -- one row per sheet
# --------------------------------------------------------------------------- #
_SPECIMEN_COLUMNS: tuple[Column, ...] = (
    Column("specimen_id", "", "Row id of this sheet in the project database."),
    Column("image_name", "", "Original image filename, with extension."),
    Column("image_stem", "", "Original filename without extension; the stem every output is named by."),
    Column("original_path", "", "Immutable path to the source image."),
    Column("working_path", "", "Path to the working copy every stage opened."),
    Column("working_width", "px", "Working image width -- the frame every _px measurement is in."),
    Column("working_height", "px", "Working image height."),
    Column("original_width", "px", "Original image width."),
    Column("original_height", "px", "Original image height."),
    Column("work_scale", "ratio", "working / original long side; 1.0 when nothing was resized."),
    Column("normalized", "0/1", "1 if ingest wrote a converted copy (RGB convert and/or downscale)."),
    Column("downsampled", "0/1", "1 iff pixels were actually discarded. Narrower than normalized."),
    Column("original_mp", "megapixels", "Original width*height/1e6."),
    Column("cf_source", "", "measured_from_ruler | predicted_from_megapixels (no ruler or the lattice "
                            "did not pass, and use_CF_predicted_by_MP was on) | none (no CF)."),
    Column("cf_px_per_cm", "px/cm", "The CF this sheet's _cm values use (working frame); see cf_source."),
    Column("cf_px_per_cm_predicted_by_mp", "px/cm", "Megapixel-regression CF estimate (in use only "
                                                    "where cf_source is predicted_from_megapixels)."),
    Column("cf_px_per_cm_measured", "px/cm", "What the lattice actually read, INCLUDING when the "
                                             "reading was withheld. For audit only -- never consume "
                                             "this as the sheet's CF."),
    Column("ruler_cf_status", "", "published | withheld | no_reading | no_ruler."),
    Column("ruler_cf_confidence", "", "high | medium | low. Only 'high' publishes a CF."),
    Column("ruler_unit_type", "", "Unit type of the rulers behind the published CF."),
    Column("ruler_class_type", "", "Per-sheet consensus ruler unit type from the classifier ensemble."),
    Column("n_ruler_crops", "count", "Candidate ruler crops considered on this sheet."),
    Column("n_ruler_crops_used", "count", "Ruler crops that contributed to the published CF."),
    Column("has_leaves", "0/1", "Phenology: leaves present."),
    Column("has_flowers", "0/1", "Phenology: flowers present."),
    Column("has_fruits", "0/1", "Phenology: fruits present."),
    Column("n_leaf_boxes", "count", "Leaf boxes the phenology stage counted."),
    Column("n_flower_boxes", "count", "Flower boxes the phenology stage counted."),
    Column("n_fruit_boxes", "count", "Fruit boxes the phenology stage counted."),
    Column("specimen_mask_area_frac", "ratio", "Fraction of the sheet the whole-specimen mask covers."),
    Column("specimen_mask_model", "", "Which segmenter export produced that mask."),
    Column("n_archival_detections", "count", "derived: kept archival boxes (suppressed duplicates excluded)."),
    Column("n_plant_detections", "count", "derived: kept plant boxes (suppressed duplicates excluded)."),
    Column("n_suppressed_detections", "count", "derived: boxes rejected as same-class duplicates."),
    Column("n_leaf_instances", "count", "derived: rows this sheet contributes to leaf_measurements.csv."),
    Column("n_leaves_measured", "count", "derived: leaves with morphology."),
    Column("n_leaves_oriented", "count", "derived: leaves given a tip-up orientation."),
    Column("n_leaves_with_landmarks", "count", "derived: leaves with landmark measurements."),
    Column("n_leaves_with_petiole", "count", "derived: leaves with a measured petiole."),
    Column("n_leaves_archetypal", "count", "derived: leaves flagged archetypal by bilateral symmetry."),
    Column("n_leaves_grounded_cm", "count", "derived: leaves with a cm area, i.e. covered by a published CF."),
    Column("median_lamina_area_incl_holes_px", "px^2", "derived: median over this sheet's leaves."),
    Column("median_lamina_area_incl_holes_cm2", "cm^2", "derived: median over the leaves that were grounded."),
    Column("median_archetype_score", "0-1", "derived: median leaf-quality score for this sheet."),
    Column("ingested_at", "", "When LM3 ingested the image (UTC)."),
)


# --------------------------------------------------------------------------- #
# phenology.csv -- LeafMachine2's phenology.csv, rebuilt from the LM3 database
# --------------------------------------------------------------------------- #
# LM2 wrote reports as Phenology/phenology.csv by globbing the plant detector's YOLO label
# .txt files and counting class ids. LM3 has no label files -- the same boxes are rows in
# plant_detection -- so this file is that count rebuilt from the DB, keeping LM2's column
# names and their exact order so an existing LM2 analysis script can read either one.
#
# TWO LM2 COLUMNS CANNOT BE FILLED. LM3's detector is trained on LM3_Plant_Primary (nc=9),
# which deliberately DROPPED Specimen and MERGED Leaflet into Leaf_WHOLE. The columns are
# kept so the header still matches, but they are written as BLANK, never 0: a 0 would assert
# "looked, found none", when the truth is that this detector cannot emit the class at all.
# Note the merge also means LM3's leaf_whole counts LM2's leaf_whole + leaflet.
#: LM2 column -> the LM3 plant class that fills it, or None where LM3 has no such class.
_LM2_PHENOLOGY_CLASSES: tuple[tuple[str, Optional[str]], ...] = (
    ("leaf_whole",      "Leaf_WHOLE"),
    ("leaf_partial",    "Leaf_PARTIAL"),
    ("leaflet",         None),              # merged into Leaf_WHOLE when LM3_Plant_Primary was built
    ("seed_fruit_one",  "Seed_Fruit_ONE"),
    ("seed_fruit_many", "Seed_Fruit_MANY"),
    ("flower_one",      "Flower_ONE"),
    ("flower_many",     "Flower_MANY"),
    ("bud",             "Bud"),
    ("specimen",        None),              # dropped when LM3_Plant_Primary was built
    ("roots",           "Roots"),
    ("wood",            "Wood"),
)

_UNMAPPED_LM2_CLASSES = tuple(name for name, cls in _LM2_PHENOLOGY_CLASSES if cls is None)

_PHENOLOGY_COLUMNS: tuple[Column, ...] = (
    Column("file_name", "", "LM2 column. Image filename with extension. LM2 wrote the LABEL file "
                            "name here ('<stem>.txt'); LM3 has no label files, so this is the image."),
    *(Column(name, "count",
             f"LM2 column. Kept {cls} boxes on this sheet."
             if cls else
             f"LM2 column, ALWAYS BLANK: LM3's detector has no {name!r} class "
             f"({'merged into Leaf_WHOLE' if name == 'leaflet' else 'dropped'} in LM3_Plant_Primary).")
      for name, cls in _LM2_PHENOLOGY_CLASSES),
    Column("has_leaves", "0/1", "LM2 column. From LM3's phenology stage (Leaf_WHOLE + Leaf_PARTIAL "
                                "against its min_conf/min_count), NOT LM2's accept_only_ideal_leaves rule."),
    Column("is_fertile", "0/1", "LM2 column. derived: has_flowers OR has_fruits. LM3's flower group "
                                "INCLUDES Bud, which LM2's is_fertile excluded."),
    # Past here the file leaves LM2 behind: LM3-native columns appended so the file is
    # self-joining and its two flavors of "has leaves" can be told apart.
    Column("specimen_id", "", "LM3 column. Row id of this sheet in the project database."),
    Column("image_stem", "", "LM3 column. Filename without extension; the stem every output is named by."),
    Column("has_flowers", "0/1", "LM3 column. Phenology stage: flowers (incl. Bud) present."),
    Column("has_fruits", "0/1", "LM3 column. Phenology stage: fruits present."),
    Column("n_leaf_boxes", "count", "LM3 column. Leaf boxes the phenology stage counted -- gated by its "
                                    "min_conf, so this can be lower than leaf_whole + leaf_partial."),
    Column("n_flower_boxes", "count", "LM3 column. Flower boxes the phenology stage counted (incl. Bud)."),
    Column("n_fruit_boxes", "count", "LM3 column. Fruit boxes the phenology stage counted."),
    Column("n_plant_detections", "count", "LM3 column. derived: all kept plant boxes on this sheet."),
)


# --------------------------------------------------------------------------- #
# detections.csv / landmarks.csv
# --------------------------------------------------------------------------- #
_DETECTION_COLUMNS: tuple[Column, ...] = (
    Column("source", "", "Which detector produced the box: archival or plant."),
    Column("image_stem", "", "Parent image, without extension."),
    Column("specimen_id", "", "Parent sheet row id."),
    Column("detection_id", "", "Row id of this box."),
    Column("cls_name", "", "Detected class name."),
    Column("conf", "0-1", "Detector confidence."),
    Column("x1", "px", "Box left edge (working frame)."),
    Column("y1", "px", "Box top edge (working frame)."),
    Column("x2", "px", "Box right edge (working frame)."),
    Column("y2", "px", "Box bottom edge (working frame)."),
    Column("box_w_px", "px", "derived: x2 - x1."),
    Column("box_h_px", "px", "derived: y2 - y1."),
    Column("box_area_px", "px^2", "derived: box_w_px * box_h_px."),
    Column("tag", "", "Short class tag used in crop filenames."),
    Column("crop_path", "", "Working crop cut for this box, if one was written."),
    Column("suppressed", "0/1", "1 = rejected as a same-class duplicate and excluded downstream."),
    Column("suppressed_by", "", "detection_id of the higher-confidence box that suppressed it."),
    Column("suppress_overlap", "ratio", "Overlap (intersection / smaller box) that triggered suppression."),
)

_LANDMARK_COLUMNS: tuple[Column, ...] = (
    Column("image_stem", "", "Parent image, without extension."),
    Column("specimen_id", "", "Parent sheet row id."),
    Column("detection_id", "", "Leaf crop this keypoint was predicted in."),
    Column("instance_index", "", "Pose instance within the crop; 0 is the primary leaf. NOTE this is "
                                 "the POSE instance space, which is not the segmentation instance "
                                 "space -- they coincide at 0."),
    Column("crop_file_token", "", "derived: the '<stem>__<x1>_<y1>_<x2>_<y2>' token naming this crop's files."),
    Column("kpt_index", "", "Keypoint index, 0-30."),
    Column("kpt_name", "", "Keypoint name, e.g. lamina_tip, midvein_7, petiole_0."),
    Column("kpt_group", "", "Keypoint group: lamina, apex, midvein, base, petiole or width."),
    Column("x", "px", "Keypoint x in the working (whole sheet) frame."),
    Column("y", "px", "Keypoint y in the working (whole sheet) frame."),
    Column("x_crop", "px", "Keypoint x within the leaf crop."),
    Column("y_crop", "px", "Keypoint y within the leaf crop."),
    Column("conf", "0-1", "Keypoint confidence. Landmark measurements ignore keypoints below "
                          "modules.landmark_detector.min_kpt_conf."),
    Column("crop_x1", "px", "Parent crop box, left edge."),
    Column("crop_y1", "px", "Parent crop box, top edge."),
    Column("crop_x2", "px", "Parent crop box, right edge."),
    Column("crop_y2", "px", "Parent crop box, bottom edge."),
)


_STAGE_ERROR_COLUMNS: tuple[Column, ...] = (
    Column("image_stem", "", "Image the stage failed on, without extension."),
    Column("specimen_id", "", "Row id of that sheet."),
    Column("stage_key", "", "Pipeline stage that failed, e.g. leaf_segmenter."),
    Column("state", "", "Always 'error' in this file."),
    Column("no_work", "0/1", "1 marks a specimen the stage legitimately had nothing to do for."),
    Column("updated_at", "", "When the failure was recorded."),
    Column("error_msg", "", "The recorded failure message."),
)


# --------------------------------------------------------------------------- #
# The bundle
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExportFile:
    """One CSV in the bundle.

    ``columns`` None marks a PASSTHROUGH dump: every column of the source table, in table order.
    Those files (the ruler audit trail, the stage ledger) are valuable for being complete rather than
    curated, and their columns are documented by the schema that owns them.
    """
    key: str                    # report.data.files toggle
    stem: str                   # filename without extension
    default: bool
    blurb: str                  # one line, for the data dictionary + the results browser
    columns: Optional[tuple[Column, ...]] = None
    source_table: str = ""      # passthrough only


FILES: tuple[ExportFile, ...] = (
    ExportFile("leaf_measurements", "leaf_measurements", True,
               "One row per segmented leaf: every morphology, landmark, petiole and symmetry "
               "measurement, with the metadata needed to identify the specimen and the leaf.",
               columns=_LEAF_COLUMNS),
    ExportFile("specimen_summary", "specimen_summary", True,
               "One row per input image: identity, frame, conversion factor, phenology and "
               "per-sheet roll-ups of the leaf measurements.",
               columns=_SPECIMEN_COLUMNS),
    ExportFile("phenology", "phenology", True,
               "One row per sheet in LeafMachine2's phenology.csv layout: per-class plant-organ "
               "counts plus has_leaves / is_fertile, so an LM2 phenology script runs unchanged.",
               columns=_PHENOLOGY_COLUMNS),
    ExportFile("detections", "detections", True,
               "One row per detection box from both detectors, including boxes suppressed as "
               "duplicates.",
               columns=_DETECTION_COLUMNS),
    ExportFile("landmarks", "landmarks", True,
               "One row per predicted leaf keypoint (31 per leaf), in both sheet and crop coordinates.",
               columns=_LANDMARK_COLUMNS),
    ExportFile("ruler_conversion_factor", "ruler_conversion_factor", True,
               "One row per sheet: the lattice ruler-CF verdict, its confidence gate and the "
               "reasoning behind it. Columns are the ruler_CF_lattice table.",
               source_table="ruler_CF_lattice"),
    ExportFile("ruler_crops", "ruler_crops", True,
               "One row per candidate ruler crop, including skipped and failed ones. Columns are "
               "the ruler_CF_lattice_crop table.",
               source_table="ruler_CF_lattice_crop"),
    ExportFile("run_stages", "run_stages", True,
               "One row per pipeline stage: state, counts and the settings hash that drives "
               "re-runs. Columns are the project_status table.",
               source_table="project_status"),
    ExportFile("stage_errors", "stage_errors", True,
               "One row per per-image stage failure, so a run's errors are visible without the log.",
               columns=_STAGE_ERROR_COLUMNS),
    ExportFile("data_dictionary", "data_dictionary", True,
               "Every column of every file above: which file it is in, its units, and what it means.",
               columns=(Column("file", "", "CSV the column appears in."),
                        Column("column", "", "Column name."),
                        Column("units", "", "Unit of measure, or blank for identifiers and text."),
                        Column("description", "", "What the column holds."))),
)

_FILES_BY_KEY = {f.key: f for f in FILES}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def export_data_csvs(project, cfg) -> list[Path]:
    """Write the enabled CSVs into ``reports/<report.data.folder>/`` and return the paths written.

    Idempotent: it rewrites its own files every time and deletes any bundle file a toggle has since
    turned off, so the folder always describes the current database rather than accumulating stale
    exports. Run-level artifacts cannot be tracked in ``report_manifest`` (that table is keyed by
    specimen), so this self-cleaning is what keeps ``--restart`` honest -- the same approach
    ``reports/Timing`` takes.
    """
    data_cfg = _sub(cfg.report, "data")
    if not _flag(data_cfg, "enabled", True):
        log.info("data export disabled (report.data.enabled=false)")
        return []

    folder = str(_get(data_cfg, "folder", default=DEFAULT_FOLDER) or DEFAULT_FOLDER)
    out_dir = Path(str(project.dirs.reports)) / folder
    fmt = str(_get(data_cfg, "format", default="csv") or "csv").lower().lstrip(".")
    if fmt not in ("csv", "tsv"):
        log.warning("report.data.format=%r is not csv or tsv; falling back to csv", fmt)
        fmt = "csv"
    delimiter = "\t" if fmt == "tsv" else ","
    na_rep = str(_get(data_cfg, "na_rep", default="") or "")
    precision = int(_get(data_cfg, "float_precision", default=6) or 6)
    files_cfg = _sub(data_cfg, "files")

    wanted = [f for f in FILES if _flag(files_cfg, f.key, f.default)]
    out_dir.mkdir(parents=True, exist_ok=True)
    _remove_disabled(out_dir, wanted, fmt)
    if not wanted:
        log.info("data export: every file is disabled; nothing written")
        return []

    db = project.db
    # Loaded once and shared: specimen_summary's roll-ups are computed from the SAME leaf and
    # detection rows the other files are written from, so a count in one file can never disagree
    # with the rows in another.
    leaf_rows = [dict(r) for r in db.export_leaf_rows()]
    detection_rows = [dict(r) for r in db.export_detection_rows()]
    for r in leaf_rows:
        _derive_leaf(r)
    for r in detection_rows:
        _derive_detection(r)

    builders: dict[str, Callable[[], tuple[Sequence[Column] | None, list[dict]]]] = {
        "leaf_measurements": lambda: (_LEAF_COLUMNS, leaf_rows),
        "specimen_summary": lambda: (_SPECIMEN_COLUMNS,
                                     _specimen_rows(db, leaf_rows, detection_rows)),
        "phenology": lambda: (_PHENOLOGY_COLUMNS, _phenology_rows(db, detection_rows)),
        "detections": lambda: (_DETECTION_COLUMNS, detection_rows),
        "landmarks": lambda: (_LANDMARK_COLUMNS, _landmark_rows(db)),
        "ruler_conversion_factor": lambda: (None, [dict(r) for r in db.export_table("ruler_CF_lattice")]),
        "ruler_crops": lambda: (None, [dict(r) for r in db.export_table("ruler_CF_lattice_crop")]),
        "run_stages": lambda: (None, [dict(r) for r in db.export_table("project_status")]),
        "stage_errors": lambda: (_STAGE_ERROR_COLUMNS, [dict(r) for r in db.export_stage_errors()]),
    }

    written: list[Path] = []
    passthrough_cols: dict[str, list[str]] = {}
    for spec in wanted:
        if spec.key == "data_dictionary":
            continue                                    # written last: it describes the others
        path = out_dir / f"{spec.stem}.{fmt}"
        try:
            columns, rows = builders[spec.key]()
            if columns is not None:
                names = [c.name for c in columns]
            else:
                # Prefer the rows' own keys; fall back to the schema so an EMPTY table still gets a
                # header instead of a zero-byte file.
                names = _passthrough_names(rows) or db.export_table_columns(spec.source_table)
                passthrough_cols[spec.key] = names
            _write_csv(path, names, rows, delimiter=delimiter, na_rep=na_rep, precision=precision)
        except Exception:
            # One unwritable table must not cost the user the other eight files, nor fail a run
            # whose real work (every image, every measurement) is already done and persisted.
            log.exception("data export failed for %s", path.name)
            continue
        written.append(path)
        log.info("data export: %s (%d rows)", path.name, len(rows))

    dict_spec = _FILES_BY_KEY["data_dictionary"]
    if dict_spec in wanted:
        path = out_dir / f"{dict_spec.stem}.{fmt}"
        try:
            rows = _dictionary_rows(wanted, passthrough_cols, fmt)
            _write_csv(path, [c.name for c in dict_spec.columns], rows,
                       delimiter=delimiter, na_rep=na_rep, precision=precision)
            written.append(path)
        except Exception:
            log.exception("data export failed for %s", path.name)

    log.info("data export: %d file(s) -> %s", len(written), out_dir)
    return written


# --------------------------------------------------------------------------- #
# Row builders
# --------------------------------------------------------------------------- #
def _crop_token(stem: Any, x1: Any, y1: Any, x2: Any, y2: Any) -> str:
    """``<stem>__<x1>_<y1>_<x2>_<y2>`` -- byte-identical to core.imaging.crop_filename's token.

    Rounds the same way (``int(round(v))``) so a CSV row's token always matches the file on disk.
    """
    if stem is None or any(v is None for v in (x1, y1, x2, y2)):
        return ""
    return f"{stem}__{int(round(float(x1)))}_{int(round(float(y1)))}_" \
           f"{int(round(float(x2)))}_{int(round(float(y2)))}"


def _derive_leaf(r: dict) -> None:
    """Add the derived identity + geometry columns to one leaf row, in place."""
    token = _crop_token(r.get("image_stem"), r.get("crop_x1"), r.get("crop_y1"),
                        r.get("crop_x2"), r.get("crop_y2"))
    inst = r.get("instance_index")
    r["crop_file_token"] = token
    r["leaf_uid"] = f"{token}__i{int(inst)}" if token and inst is not None else ""
    r["cf_source"] = _cf_source(r)
    r["bbox_w_px"] = _span(r.get("bbox_x1"), r.get("bbox_x2"))
    r["bbox_h_px"] = _span(r.get("bbox_y1"), r.get("bbox_y2"))


def _cf_source(r: dict) -> str:
    """The stored ``specimen.cf_source``, with NULL spelled out as 'none' for the CSV reader.

    Read from the DB, never inferred from which CF columns are filled: the ruler_cf stage is the
    only place that knows whether the value in ``cf_px_per_cm`` was measured or predicted."""
    return r.get("cf_source") or "none"


def _derive_detection(r: dict) -> None:
    w = _span(r.get("x1"), r.get("x2"))
    h = _span(r.get("y1"), r.get("y2"))
    r["box_w_px"], r["box_h_px"] = w, h
    r["box_area_px"] = None if w is None or h is None else w * h


def _span(a: Any, b: Any) -> Optional[float]:
    return None if a is None or b is None else float(b) - float(a)


def _specimen_rows(db, leaf_rows: list[dict], detection_rows: list[dict]) -> list[dict]:
    """Specimen rows plus per-sheet roll-ups computed from the already-loaded leaf/detection rows."""
    by_spec: dict[Any, list[dict]] = {}
    for r in leaf_rows:
        by_spec.setdefault(r.get("specimen_id"), []).append(r)
    det_by_spec: dict[Any, list[dict]] = {}
    for r in detection_rows:
        det_by_spec.setdefault(r.get("specimen_id"), []).append(r)

    out: list[dict] = []
    for row in db.export_specimen_rows():
        r = dict(row)
        sid = r.get("specimen_id")
        leaves = by_spec.get(sid, [])
        dets = det_by_spec.get(sid, [])
        r["cf_source"] = _cf_source(r)
        r["n_archival_detections"] = sum(1 for d in dets if d.get("source") == "archival"
                                         and not d.get("suppressed"))
        r["n_plant_detections"] = sum(1 for d in dets if d.get("source") == "plant"
                                      and not d.get("suppressed"))
        r["n_suppressed_detections"] = sum(1 for d in dets if d.get("suppressed"))
        r["n_leaf_instances"] = len(leaves)
        r["n_leaves_measured"] = sum(1 for x in leaves if x.get("lamina_area_incl_holes_px") is not None)
        r["n_leaves_oriented"] = sum(1 for x in leaves if x.get("oriented_leaf_success"))
        r["n_leaves_with_landmarks"] = sum(1 for x in leaves if x.get("n_landmarks_present") is not None)
        r["n_leaves_with_petiole"] = sum(1 for x in leaves if x.get("petiole_width_px") is not None)
        r["n_leaves_archetypal"] = sum(1 for x in leaves if x.get("is_archetypal"))
        r["n_leaves_grounded_cm"] = sum(1 for x in leaves if x.get("lamina_area_incl_holes_cm2") is not None)
        r["median_lamina_area_incl_holes_px"] = _median(
            x.get("lamina_area_incl_holes_px") for x in leaves)
        r["median_lamina_area_incl_holes_cm2"] = _median(
            x.get("lamina_area_incl_holes_cm2") for x in leaves)
        r["median_archetype_score"] = _median(x.get("archetype_score") for x in leaves)
        out.append(r)
    return out


def _phenology_rows(db, detection_rows: list[dict]) -> list[dict]:
    """LM2's phenology.csv rebuilt from plant_detection + the phenology stage's verdict.

    Counts KEPT boxes only (a box suppressed as a same-class duplicate is not a second organ),
    which is the same rule ``n_plant_detections`` uses, so the two agree row for row.

    The counts here apply NO confidence gate, matching LM2 -- it counted every line in the label
    file. The phenology stage's own ``n_*_boxes`` DO apply its ``min_conf``, so the two disagree
    whenever a low-confidence box exists. Both are in the file rather than one being reconciled
    into the other: they answer different questions, and silently picking one would hide the gate.
    """
    counts: dict[Any, dict[str, int]] = {}
    for d in detection_rows:
        if d.get("source") != "plant" or d.get("suppressed"):
            continue
        counts.setdefault(d.get("specimen_id"), {})
        cls = d.get("cls_name")
        counts[d["specimen_id"]][cls] = counts[d["specimen_id"]].get(cls, 0) + 1

    out: list[dict] = []
    for row in db.export_specimen_rows():
        sid = row["specimen_id"]
        per_class = counts.get(sid, {})
        # A sheet with no plant detections still gets a row of zeros. LM2 had no row at all for
        # such a sheet (no boxes -> no label file -> nothing to glob), so this file is a superset:
        # "we looked and found nothing" is a result, and dropping it would bias any rate computed
        # from the file by silently shrinking the denominator.
        r: dict[str, Any] = {"file_name": row["image_name"]}
        for name, cls in _LM2_PHENOLOGY_CLASSES:
            r[name] = None if cls is None else per_class.get(cls, 0)
        has_flowers, has_fruits = row["has_flowers"], row["has_fruits"]
        r["has_leaves"] = row["has_leaves"]
        # None (phenology stage never ran for this sheet) must stay absent, not become 0.
        r["is_fertile"] = (None if has_flowers is None and has_fruits is None
                           else int(bool(has_flowers) or bool(has_fruits)))
        r["specimen_id"] = sid
        r["image_stem"] = row["image_stem"]
        r["has_flowers"] = has_flowers
        r["has_fruits"] = has_fruits
        r["n_leaf_boxes"] = row["n_leaf_boxes"]
        r["n_flower_boxes"] = row["n_flower_boxes"]
        r["n_fruit_boxes"] = row["n_fruit_boxes"]
        r["n_plant_detections"] = sum(per_class.values())
        out.append(r)
    return out


def _landmark_rows(db) -> list[dict]:
    rows = [dict(r) for r in db.export_landmark_rows()]
    for r in rows:
        r["crop_file_token"] = _crop_token(r.get("image_stem"), r.get("crop_x1"), r.get("crop_y1"),
                                           r.get("crop_x2"), r.get("crop_y2"))
    return rows


def _median(values: Iterable[Any]) -> Optional[float]:
    """Median of the non-empty values, or None when nothing was measured (never 0.0 by default)."""
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _passthrough_names(rows: list[dict]) -> list[str]:
    """Column names from the first row, or empty -- the caller falls back to the table schema."""
    return list(rows[0].keys()) if rows else []


def _dictionary_rows(wanted: Sequence[ExportFile], passthrough_cols: dict[str, list[str]],
                     fmt: str) -> list[dict]:
    """One row per column of every file actually written, preceded by a row describing the file."""
    out: list[dict] = []
    for spec in wanted:
        if spec.key == "data_dictionary":
            continue
        filename = f"{spec.stem}.{fmt}"
        out.append({"file": filename, "column": "", "units": "", "description": spec.blurb})
        if spec.columns:
            for c in spec.columns:
                out.append({"file": filename, "column": c.name, "units": c.units,
                            "description": c.desc})
        else:
            src = spec.source_table
            for name in passthrough_cols.get(spec.key, []):
                out.append({"file": filename, "column": name, "units": "",
                            "description": f"Verbatim {src}.{name}; see the {src} table definition "
                                           f"for its units and meaning."})
    return out


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, names: Sequence[str], rows: Sequence[dict], *,
               delimiter: str, na_rep: str, precision: int) -> None:
    """Write ``names`` as the header and one line per row.

    A header is written even for an empty table: a zero-row CSV with columns says "this was exported
    and nothing matched", where a missing file says nothing at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=delimiter, lineterminator="\n")
        w.writerow(names)
        for row in rows:
            w.writerow([_fmt(row.get(n), na_rep, precision) for n in names])


def _fmt(v: Any, na_rep: str, precision: int) -> Any:
    """Render one value: None -> na_rep, floats rounded, everything else as-is.

    Rounding is what keeps 0.30000000000000004 out of a scientific data file. NaN and infinity are
    written as ``na_rep`` too -- they are absences that survived a float pipeline, and a spreadsheet
    reading 'nan' as text would silently poison a column's dtype.
    """
    if v is None:
        return na_rep
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return na_rep
        r = round(v, precision)
        return int(r) if r == int(r) and abs(r) < 1e15 else r
    return v


def _remove_disabled(out_dir: Path, wanted: Sequence[ExportFile], fmt: str) -> None:
    """Delete bundle files that are no longer enabled (or are in the other format), so the folder
    never mixes this run's exports with leftovers from a differently-configured one."""
    keep = {f"{f.stem}.{fmt}" for f in wanted}
    for spec in FILES:
        for ext in ("csv", "tsv"):
            stale = out_dir / f"{spec.stem}.{ext}"
            if stale.name not in keep and stale.is_file():
                try:
                    stale.unlink()
                except OSError:                       # pragma: no cover - permissions / races
                    log.warning("could not remove stale export %s", stale)


# --------------------------------------------------------------------------- #
# Tolerant config access (mirrors the Reporter's accessors: Config/Section or plain dict)
# --------------------------------------------------------------------------- #
def _getk(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        if hasattr(obj, "get"):
            v = obj.get(key, default)
            return default if v is None else v
    except Exception:
        pass
    v = getattr(obj, key, default)
    return default if v is None else v


def _sub(obj: Any, key: str) -> Any:
    return _getk(obj, key, None)


def _get(obj: Any, key: str, *, default: Any = None) -> Any:
    return _getk(obj, key, default)


def _flag(obj: Any, key: str, default: bool) -> bool:
    return bool(_getk(obj, key, default))
