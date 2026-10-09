-- ==========================================================================
-- LeafMachine3 -- one SQLite database per project (in the YAML output dir).
-- Applied once at init; idempotent (IF NOT EXISTS). Stage KEYS (project_status)
-- are distinct from TABLE names: keys = archival_detector/plant_detector/... ;
-- tables = archival_detection/plant_detection/... . One canonical key set only.
-- ==========================================================================
PRAGMA journal_mode = WAL;        -- concurrent readers while the single collector writes
PRAGMA synchronous  = NORMAL;     -- safe under WAL; big throughput win
PRAGMA foreign_keys = ON;         -- cascade + provenance integrity
PRAGMA busy_timeout = 30000;

-- specimen : raw/metadata table. ONE row per input image. Original is immutable.
-- Key cross-stage OUTPUTS are duplicated here for cheap, join-free reporting.
CREATE TABLE IF NOT EXISTS specimen (
    specimen_id        INTEGER PRIMARY KEY,
    image_name         TEXT NOT NULL,
    image_stem         TEXT NOT NULL,
    original_path      TEXT NOT NULL,          -- IMMUTABLE source
    working_path       TEXT NOT NULL,          -- symlink into the working set (what stages open)
    width              INTEGER,                -- WORKING-copy dims
    height             INTEGER,
    original_width     INTEGER,                -- ORIGINAL (pre-ingest) dims, kept for provenance and
    original_height    INTEGER,                -- because the MP->CF regression is fit on them
    work_scale         REAL NOT NULL DEFAULT 1.0,   -- working / original long-side ratio (<1 iff downsampled)
    orig_size_bytes    INTEGER,                -- ingest change-detection (moved/edited original)
    orig_mtime         REAL,
    normalized         INTEGER NOT NULL DEFAULT 0,  -- 1 if a tmp jpg copy was materialized (RGB convert AND/OR downscale)
    downsampled        INTEGER NOT NULL DEFAULT 0,  -- 1 iff pixels were DISCARDED (long side capped to
                                                    -- ingest.max_working_dim). Strictly narrower than
                                                    -- `normalized`: a small TIFF is normalized, not downsampled.
    -- duplicated key outputs (authoritative copies live in the method tables):
    original_mp        REAL,                   -- original_width*original_height/1e6 (megapixels, 4dp); from mp_conversion_factor
    cf_px_per_cm_predicted_by_mp REAL,         -- from mp_conversion_factor (resolution->CF linear predictor; runs first)
    cf_px_per_cm       REAL,                   -- the sheet's CF (WORKING frame -- the only frame LM3 measures
                                               -- in), written by ruler_cf. Which CF it is: see cf_source.
    cf_source          TEXT,                   -- from ruler_cf: 'measured_from_ruler' (lattice published it,
                                               -- high confidence) | 'predicted_from_megapixels' (no ruler or the
                                               -- lattice did not pass, and modules.ruler_cf.use_CF_predicted_by_MP
                                               -- is on) | NULL (no CF; every *_cm column stays NULL)
    ruler_unit_type    TEXT,                   -- from ruler_cf (lattice): dominant unit-type of the crops that produced the CF
    ruler_class_type   TEXT,                   -- from ruler_classifier (ensemble unit-type; per-specimen consensus across rulers)
    has_leaves         INTEGER, has_flowers INTEGER, has_fruits INTEGER,
    ingested_at        TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (image_stem),
    UNIQUE (original_path)
);

-- archival_detection / plant_detection : one row per box. xyxy in WORKING coords.
-- Idempotency is by delete-then-insert per specimen inside one txn (see ProjectDB),
-- backed by a deterministic unique key so a torn write can't duplicate.
CREATE TABLE IF NOT EXISTS archival_detection (
    detection_id INTEGER PRIMARY KEY,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    cls_id INTEGER NOT NULL, cls_name TEXT NOT NULL, conf REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    tag TEXT, crop_path TEXT,                  -- tag is BARE ("R"); crop file adds __R__
    suppressed INTEGER NOT NULL DEFAULT 0,     -- 1 = rejected as a same-class duplicate (core.box_dedup)
    suppressed_by INTEGER,                     -- detection_id of the higher-conf box that suppressed it
    suppress_overlap REAL,                     -- overlap fraction (intersection / smaller-box area) that triggered it
    UNIQUE (specimen_id, cls_id, x1, y1, x2, y2)
);
CREATE INDEX IF NOT EXISTS ix_arch_spec ON archival_detection (specimen_id, cls_name);

CREATE TABLE IF NOT EXISTS plant_detection (
    detection_id INTEGER PRIMARY KEY,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    cls_id INTEGER NOT NULL, cls_name TEXT NOT NULL, conf REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    tag TEXT, crop_path TEXT,
    suppressed INTEGER NOT NULL DEFAULT 0,     -- 1 = rejected as a same-class duplicate (core.box_dedup)
    suppressed_by INTEGER,                     -- detection_id of the higher-conf box that suppressed it
    suppress_overlap REAL,                     -- overlap fraction (intersection / smaller-box area) that triggered it
    UNIQUE (specimen_id, cls_id, x1, y1, x2, y2)
);
CREATE INDEX IF NOT EXISTS ix_plant_spec ON plant_detection (specimen_id, cls_name);

-- phenology : per-specimen presence/absence (also mirrored onto specimen.has_*).
CREATE TABLE IF NOT EXISTS phenology (
    specimen_id INTEGER PRIMARY KEY REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    has_leaves INTEGER, has_flowers INTEGER, has_fruits INTEGER,
    n_leaves INTEGER, n_flowers INTEGER, n_fruits INTEGER
);

-- ruler_classification : per Ruler crop -> unit_type + the ensemble votes.
-- squarify_path is the pre-made four-tile collage (_ruler_squarify) the classifier saw,
-- reused by the ruler_cf lattice stage + its QC panel so nothing re-squarifies.
CREATE TABLE IF NOT EXISTS ruler_classification (
    ruler_class_id INTEGER PRIMARY KEY,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id)           ON DELETE CASCADE,
    detection_id INTEGER NOT NULL REFERENCES archival_detection(detection_id) ON DELETE CASCADE,
    unit_type TEXT NOT NULL, votes_json TEXT, conf REAL,
    squarify_path TEXT,                        -- _ruler_squarify/det<id>__tile_four.jpg
    UNIQUE (detection_id)                      -- idempotent re-run
);

-- ruler_cf : the lattice conversion-factor stage's two tables (ruler_CF_lattice +
-- ruler_CF_lattice_crop) and the v_ruler_CF_lattice view are created from the engine's
-- own SCHEMA_SQL in ProjectDB.init_schema (single DDL source: inference/ruler_lattice).

-- leaf_segmentation : one row per INSTANCE mask. Polygon/RLE in PARENT (working) coords
-- so Reporter rebuilds every view. Hole/Petiole link to their Leaf via parent_instance_index.
CREATE TABLE IF NOT EXISTS leaf_segmentation (
    leaf_id INTEGER PRIMARY KEY,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id)         ON DELETE CASCADE,
    detection_id INTEGER NOT NULL REFERENCES plant_detection(detection_id) ON DELETE CASCADE,
    instance_index INTEGER NOT NULL,
    parent_instance_index INTEGER,             -- Hole/Petiole -> owning Leaf instance; Leaf -> itself
    cls_id INTEGER NOT NULL, cls_name TEXT NOT NULL, conf REAL,   -- Leaf | Petiole | Hole
    mask_format TEXT NOT NULL CHECK (mask_format IN ('polygon_xy','coco_rle')),
    mask_data TEXT NOT NULL,                   -- polygon rings JSON  OR  COCO-RLE counts (parent coords)
    frame_width INTEGER NOT NULL, frame_height INTEGER NOT NULL,  -- coord frame = working dims
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    num_parts INTEGER NOT NULL DEFAULT 1,
    area_px REAL, perimeter_px REAL,
    area_cm2 REAL, perimeter_cm REAL,          -- filled by MetricGrounding
    bbox_w_cm REAL, bbox_h_cm REAL,
    UNIQUE (detection_id, instance_index)      -- upsert -> resume-safe
);
CREATE INDEX IF NOT EXISTS ix_leaf_spec ON leaf_segmentation (specimen_id);

-- leaf_morphology : one row per leaf INSTANCE mask (Morphology stage). Scalar shape metrics
-- + the LeafMachine2 rotated (minimum) bounding box. Geometry is in WORKING (parent) coords.
-- Links back to the parent via specimen_id / detection_id / leaf_id (+ the crop box).
CREATE TABLE IF NOT EXISTS leaf_morphology (
    morph_id       INTEGER PRIMARY KEY,
    leaf_id        INTEGER NOT NULL REFERENCES leaf_segmentation(leaf_id)   ON DELETE CASCADE,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id)         ON DELETE CASCADE,
    detection_id   INTEGER NOT NULL REFERENCES plant_detection(detection_id) ON DELETE CASCADE,
    instance_index INTEGER NOT NULL,
    cls_name       TEXT NOT NULL,               -- Leaf | Petiole | Hole
    -- parent-linking crop box (the plant_detection leaf box, working coords)
    crop_x1 REAL, crop_y1 REAL, crop_x2 REAL, crop_y2 REAL,
    -- scalar morphology (working-frame pixels)
    area_px REAL, perimeter_px REAL,            -- area_px = area inside the Leaf outer boundary (INCLUDES holes)
    -- hole-aware lamina areas (the Leaf polygon is the outer silhouette, so area_px already includes
    -- holes; incl == area_px kept explicit, excl removes the holes, hole_area sums the Hole instances).
    lamina_area_incl_holes_px REAL,             -- lamina area WITH holes (== area_px; the full silhouette)
    lamina_area_excl_holes_px REAL,             -- lamina tissue area (holes removed) = incl - hole_area
    lamina_hole_area_px REAL,                    -- sum of the leaf's Hole instance areas
    n_holes INTEGER,                             -- number of Hole instances in the leaf
    centroid_x REAL, centroid_y REAL,
    convex_hull_area REAL, convexity REAL, concavity REAL, circularity REAL,
    aspect_ratio REAL, n_vertices INTEGER,
    -- axis-aligned bbox (working coords)
    bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL,
    -- rotated bounding box: rotation angle + long/short side lengths + 4 corners.
    -- dim_max/dim_min are the GEOMETRIC long/short SIDE lengths (distances between adjacent
    -- vertices) -- NOT necessarily biological length/width (some leaves are wider than long).
    rotate_angle REAL,
    rotated_bbox_dim_max REAL,                  -- long side (geometric max dim)
    rotated_bbox_dim_min REAL,                  -- short side (geometric min dim)
    rotated_bbox_json TEXT,                     -- [[x,y],...] 4 corners, working coords
    -- oriented length/width: NULL until LM3_Landmark_Detector orientation exists, then a future
    -- step assigns dim_max/dim_min to length/width via the lamina-tip axis (see TODO #1).
    rotated_bbox_length REAL,                   -- tip->base extent  (reserved; NULL for now)
    rotated_bbox_width REAL,                    -- perpendicular extent (reserved; NULL for now)
    -- minimum enclosing circle (LM2 uses its diameter to find the rotation)
    circle_cx REAL, circle_cy REAL, circle_radius REAL,
    -- NO cm columns here ON PURPOSE. MetricGrounding grounds the leaf's area/perimeter/bbox on
    -- `leaf_segmentation` (one row per instance mask, same leaf_id) -- that is the single home for
    -- this leaf's cm values. This table once declared area_cm2/perimeter_cm/length_cm/width_cm too;
    -- nothing ever wrote them, so every export shipped four all-NULL columns that read as a bug.
    -- Join leaf_segmentation ON leaf_id for cm; see _MORPH_DROPS in core/db.py.
    -- leaf orientation (LeafOrientation stage): rotate the leaf so the lamina tip is up / base down.
    oriented_leaf_success INTEGER,              -- 1 if an orientation was determined, 0 if not (no oriented output)
    oriented_leaf_rotation_angle_degreesCW REAL, -- clockwise degrees to rotate the leaf products upright
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (leaf_id)                            -- one morphology row per leaf instance
);
CREATE INDEX IF NOT EXISTS ix_morph_spec ON leaf_morphology (specimen_id);

-- landmark_schema : the pose model's 31 keypoint definitions (SEEDED once at init from
-- leafmachine3.core.landmarks). Makes stored landmarks self-describing (name + group by index).
CREATE TABLE IF NOT EXISTS landmark_schema (
    kpt_index INTEGER PRIMARY KEY,              -- 0..30
    name      TEXT NOT NULL,                    -- lamina_tip, midvein_0, apex_center, ...
    grp       TEXT NOT NULL                     -- lamina | apex | midvein | base | petiole | width
);

-- landmark_skeleton : relationships between keypoints (SEEDED once). a/b -> landmark_schema.
-- kinds: midvein / petiole (traces), apex / base (angles), width, lamina_length.
CREATE TABLE IF NOT EXISTS landmark_skeleton (
    edge_id INTEGER PRIMARY KEY,
    a_index INTEGER NOT NULL REFERENCES landmark_schema(kpt_index),
    b_index INTEGER NOT NULL REFERENCES landmark_schema(kpt_index),
    kind    TEXT NOT NULL
);

-- leaf_landmark : predicted keypoints, one row per (leaf crop x instance x keypoint).
-- x/y are in WORKING (parent) coords with the training white-pad removed (as if never added);
-- x_crop/y_crop are the crop-frame coords. Links back to the leaf crop via detection_id.
CREATE TABLE IF NOT EXISTS leaf_landmark (
    landmark_id    INTEGER PRIMARY KEY,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id)         ON DELETE CASCADE,
    detection_id   INTEGER NOT NULL REFERENCES plant_detection(detection_id) ON DELETE CASCADE,
    instance_index INTEGER NOT NULL,            -- 0 (one leaf per crop; >0 if the model emits more)
    kpt_index      INTEGER NOT NULL REFERENCES landmark_schema(kpt_index),
    kpt_name       TEXT NOT NULL,
    x REAL, y REAL,                             -- working (parent) coords, pad removed
    x_crop REAL, y_crop REAL,                   -- crop-frame coords, pad removed
    conf REAL,
    UNIQUE (detection_id, instance_index, kpt_index)
);
CREATE INDEX IF NOT EXISTS ix_landmark_spec ON leaf_landmark (specimen_id);
CREATE INDEX IF NOT EXISTS ix_landmark_det  ON leaf_landmark (detection_id);

-- leaf_landmark_measurement : derived per-leaf measurements from the keypoints (landmark_measurements
-- stage). One row per leaf instance (same key space as leaf_landmark). Lengths are WORKING-frame
-- pixels; angles are degrees. Every metric is NULLABLE -- occluded/missing keypoints => NULL, never
-- a fabricated value. See leafmachine3.core.landmark_metrics for the exact definitions.
-- The five LENGTHS are additionally grounded to cm by MetricGrounding (``*_cm``, NULL without a CF);
-- angles and curvature are already unit-free degrees and have no cm twin.
CREATE TABLE IF NOT EXISTS leaf_landmark_measurement (
    measure_id     INTEGER PRIMARY KEY,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id)         ON DELETE CASCADE,
    detection_id   INTEGER NOT NULL REFERENCES plant_detection(detection_id) ON DELETE CASCADE,
    instance_index INTEGER NOT NULL,            -- matches leaf_landmark instance
    lamina_trace_length  REAL,                  -- summed dist along midvein trace pts (midvein_0..14)
    lamina_extent        REAL,                  -- straight chord of first->last midvein pt (== curvature denom)
    lamina_tip_base_length REAL,                -- straight lamina_tip -> lamina_base (separate anchors)
    leaf_width           REAL,                  -- width_left -> width_right
    apex_angle           REAL,                  -- degrees at apex_center
    apex_angle_type      TEXT,                  -- acute | obtuse | reflex | NULL
    base_angle           REAL,                  -- degrees at base_center
    base_angle_type      TEXT,                  -- acute | obtuse | reflex | NULL
    petiole_trace_length REAL,                  -- summed dist along petiole trace pts (petiole_0..4)
    lamina_curvature     REAL,                  -- max midvein bend, degrees (0 straight, larger as it curves)
    curvature_point      INTEGER,               -- midvein index of the sharpest-bend vertex (for plotting)
    lamina_centroid_x    REAL, lamina_centroid_y REAL,   -- mean of midvein trace points (for QC/overlay)
    n_present            INTEGER,               -- how many confident keypoints fed the measurement
    -- cm-grounded lengths (MetricGrounding, when specimen.cf_px_per_cm exists) -- px / cf
    lamina_trace_length_cm   REAL,
    lamina_extent_cm         REAL,
    lamina_tip_base_length_cm REAL,
    leaf_width_cm            REAL,
    petiole_trace_length_cm  REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (detection_id, instance_index)       -- one measurement row per leaf instance
);
CREATE INDEX IF NOT EXISTS ix_lmmeasure_spec ON leaf_landmark_measurement (specimen_id);

-- leaf_petiole : per-leaf petiole width (PetioleWidth stage). Width = MEDIAN of perpendicular
-- thickness samples of the Petiole mask, taken near the blade junction along the landmark petiole
-- centerline. Segments (for the overlays) are in WORKING (parent) coords. One row per leaf that has
-- a petiole. Lengths are working-frame pixels; width_cm is filled later by MetricGrounding.
CREATE TABLE IF NOT EXISTS leaf_petiole (
    petiole_id     INTEGER PRIMARY KEY,
    leaf_id        INTEGER NOT NULL REFERENCES leaf_segmentation(leaf_id)   ON DELETE CASCADE,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id)         ON DELETE CASCADE,
    detection_id   INTEGER NOT NULL REFERENCES plant_detection(detection_id) ON DELETE CASCADE,
    instance_index INTEGER NOT NULL,
    width_px            REAL,                   -- median perpendicular petiole width
    length_px           REAL,                   -- petiole centerline length (lamina_base -> petiole_tip)
    n_samples           INTEGER,                -- number of valid perpendicular samples (median over these)
    touches_leaf        INTEGER,                -- 1 if the petiole mask is within ~touch_dist px of the leaf
    measure_location    TEXT,                   -- 'near_base' | 'none'
    width_segment_json  TEXT,                   -- reported width segment [[x1,y1],[x2,y2]] (working coords)
    sample_segments_json TEXT,                  -- all sample segments [[[x1,y1],[x2,y2]], ...] (working coords)
    width_cm            REAL,                   -- width_px / cf   (MetricGrounding, when a CF exists)
    length_cm           REAL,                   -- length_px / cf  (MetricGrounding, when a CF exists)
    leaf_mass_per_area  REAL,                   -- LMA proxy, g/m^2 (Royer petiole scaling; see PetioleWidth)
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (leaf_id)                            -- one petiole row per leaf instance
);
CREATE INDEX IF NOT EXISTS ix_petiole_spec ON leaf_petiole (specimen_id);


-- bilateral_symmetry : one row per oriented Leaf_WHOLE leaf (BilateralSymmetry stage).
-- Carries the metrics AND the frame geometry the Reporter needs to redraw the QC panel, so the
-- Reporter never re-derives the oriented frame from landmarks. Measured about the TRACED MIDVEIN on
-- the holes-filled silhouette; see core/bilateral.py for why that axis and that mask.
CREATE TABLE IF NOT EXISTS bilateral_symmetry (
    bsym_id        INTEGER PRIMARY KEY,
    leaf_id        INTEGER NOT NULL REFERENCES leaf_segmentation(leaf_id) ON DELETE CASCADE,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id)      ON DELETE CASCADE,
    detection_id   INTEGER NOT NULL REFERENCES plant_detection(detection_id) ON DELETE CASCADE,
    instance_index INTEGER NOT NULL,
    -- metrics (midvein axis, holes-filled silhouette)
    si_a            REAL,     -- Shi et al. standardized index; 0 = perfect
    a_star          REAL,     -- signed total imbalance in [-1,1]; + = viewer's LEFT half larger
    dice            REAL,     -- straightened mirrored-half overlap; 1 = perfect
    sinuosity       REAL,     -- midvein arclength / straight tip-base distance
    -- composite: the two high-value columns are archetype_score and gates_pass
    archetype_score REAL,
    term_symmetry REAL, term_integrity REAL, term_completeness REAL, term_trace REAL,
    gates_pass      INTEGER,  -- 1 = cleared every structural veto
    is_archetypal   INTEGER,  -- gates_pass AND archetype_score >= min_score
    reasons_json    TEXT,     -- human-readable veto / penalty reasons
    -- quality diagnostics (reported; only some are scored -- see core/bilateral.py)
    largest_frac REAL, solidity REAL, perimeter_ratio REAL, hole_frac REAL,
    kpt_conf_mean REAL, kpt_conf_min REAL, n_midvein_kpts INTEGER, truncated INTEGER,
    -- frame geometry: everything needed to rebuild the oriented frame for the QC panel
    angle_cw REAL,                      -- CW rotation applied (duplicated from leaf_morphology)
    crop_w INTEGER, crop_h INTEGER,     -- pre-rotation crop dims (CLAMPED to the sheet)
    mask_w INTEGER, mask_h INTEGER,     -- final oriented mask dims
    tip_x REAL, tip_y REAL,             -- ORIENTED-frame coords
    base_x REAL, base_y REAL,
    midvein_json    TEXT,     -- (N,2) tip->base polyline, ORIENTED coords
    n_bins          INTEGER,  -- arclength bins actually used (size-adaptive; see bins_for)
    qc_png          TEXT,     -- reports/Leaf_Data/Bilateral_Symmetry/<crop filename>
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (leaf_id)                    -- one row per leaf instance
);
CREATE INDEX IF NOT EXISTS ix_bsym_spec ON bilateral_symmetry (specimen_id);

-- specimen_mask : one whole-sheet plant-vs-background mask per specimen (SpecimenSegmenter stage).
-- Masks are PNG files on disk (working frame), referenced by path so the raster stays lossless.
-- mask_path = final (post-paperclean) mask; refined_path = pixels paperclean removed (red overlay);
-- sample_centers_json = paper-sampling box centers [[x,y],...] in working coords (blue overlay).
CREATE TABLE IF NOT EXISTS specimen_mask (
    specimen_id  INTEGER PRIMARY KEY REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    mask_format  TEXT NOT NULL DEFAULT 'png_path',     -- 'png_path' (final binary mask raster)
    mask_path    TEXT,                                  -- final mask PNG {0,255}, WORKING frame
    refined_path TEXT,                                  -- paperclean removed-region PNG {0,255}
    sample_centers_json TEXT,                           -- [[x,y],...] paper-sample box centers (working)
    frame_width  INTEGER NOT NULL, frame_height INTEGER NOT NULL,   -- coord frame = working dims
    area_frac    REAL,                                  -- foreground fraction of the final mask (QC)
    model_name   TEXT,                                  -- which export produced it (provenance)
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- leaf_ect : one Euler Characteristic Transform per oriented Leaf_WHOLE leaf (ECT stage). Points at
-- the per-leaf .h5 (ect matrix + outline + image metadata) and the two ECT visualization PNGs.
CREATE TABLE IF NOT EXISTS leaf_ect (
    leaf_id      INTEGER PRIMARY KEY REFERENCES leaf_segmentation(leaf_id) ON DELETE CASCADE,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    detection_id INTEGER, instance_index INTEGER,
    mask_includes TEXT,                        -- lamina | lamina_petiole | lamina_hole | lamina_petiole_hole
    h5_path      TEXT,                          -- reports/Leaf_Data/Coordinates/<leaf>.h5
    radial_png   TEXT,                          -- reports/Leaf_Data/Oriented_Leaf_Radial_ECT/<leaf>.png
    ect_png      TEXT,                          -- reports/Leaf_Data/Oriented_Leaf_ECT/<leaf>.png
    overlay_png  TEXT,                          -- reports/Leaf_Data/Oriented_Leaf_Radial_ECT_Overlay/<leaf>.png
    n_outline_points INTEGER, num_dirs INTEGER,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_ect_spec ON leaf_ect (specimen_id);

-- leaf_momocs : one Momocs/Momocs2-ready image per leaf (Momocs stage), read from the Reporter's
-- holes-filled leaf-product mask. mask_path / json_path are this stage's own files; source_mask is the
-- Reporter's (a foreign reference, never deleted with this table).
CREATE TABLE IF NOT EXISTS leaf_momocs (
    leaf_id      INTEGER PRIMARY KEY REFERENCES leaf_segmentation(leaf_id) ON DELETE CASCADE,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    detection_id INTEGER, instance_index INTEGER,
    tree          TEXT,                         -- Leaf_Oriented | Leaf_Original
    mask_includes TEXT,                         -- lamina | lamina_petiole   (holes always filled)
    source_mask  TEXT,                          -- reports/<tree>/<product>/<leaf>.png (Reporter's)
    mask_path    TEXT,                          -- reports/Leaf_Momocs/<leaf>.jpg (black leaf on white)
    json_path    TEXT,                          -- reports/Leaf_Momocs/Momit_JSON/<sheet>.json
    n_outline_points INTEGER, image_width INTEGER, image_height INTEGER,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_momocs_spec ON leaf_momocs (specimen_id);

-- project_status : stage-level ledger. Drives whole-module skip + config-drift + restart.
CREATE TABLE IF NOT EXISTS project_status (
    stage_key   TEXT PRIMARY KEY,              -- canonical STAGE_KEYS, seeded at init
    stage_order INTEGER NOT NULL,              -- 1..8
    state       TEXT NOT NULL DEFAULT 'pending'
                    CHECK (state IN ('pending','running','done','error')),
    n_total INTEGER DEFAULT 0, n_done INTEGER DEFAULT 0,
    settings_hash TEXT,                        -- hash of the resolved cfg block + model mtime/size
    started_at TEXT, finished_at TEXT, error_msg TEXT
);

-- image_status : per (specimen x stage) completion ledger for mid-stage resume.
-- no_work=1 marks a specimen a stage legitimately has nothing to do for (e.g. no
-- ruler) so depends_on gating never stalls.
CREATE TABLE IF NOT EXISTS image_status (
    specimen_id INTEGER NOT NULL REFERENCES specimen(specimen_id)      ON DELETE CASCADE,
    stage_key   TEXT    NOT NULL REFERENCES project_status(stage_key)  ON DELETE CASCADE,
    state   TEXT NOT NULL DEFAULT 'done' CHECK (state IN ('done','error')),
    no_work INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    error_msg TEXT,
    PRIMARY KEY (specimen_id, stage_key)
);
CREATE INDEX IF NOT EXISTS ix_imgstatus ON image_status (stage_key, state);

-- report_manifest : exact artifact paths Reporter wrote, so --restart deletes them precisely.
CREATE TABLE IF NOT EXISTS report_manifest (
    specimen_id INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    path TEXT NOT NULL, kind TEXT,
    PRIMARY KEY (specimen_id, path)
);
