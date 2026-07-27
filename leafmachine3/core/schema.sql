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
    original_width     INTEGER,                -- ORIGINAL dims (Reporter renders on the original)
    original_height    INTEGER,
    work_scale         REAL NOT NULL DEFAULT 1.0,   -- working / original long-side ratio
    orig_size_bytes    INTEGER,                -- ingest change-detection (moved/edited original)
    orig_mtime         REAL,
    normalized         INTEGER NOT NULL DEFAULT 0,  -- 1 if a tmp jpg copy was materialized
    -- duplicated key outputs (authoritative copies live in the method tables):
    cf_px_per_cm       REAL,                   -- from ruler_cf (the CF)
    selected_ruler_cf_id INTEGER,              -- which ruler_cf row produced the CF (provenance)
    ruler_unit_type    TEXT,
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
    UNIQUE (specimen_id, cls_id, x1, y1, x2, y2)
);
CREATE INDEX IF NOT EXISTS ix_arch_spec ON archival_detection (specimen_id, cls_name);

CREATE TABLE IF NOT EXISTS plant_detection (
    detection_id INTEGER PRIMARY KEY,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id) ON DELETE CASCADE,
    cls_id INTEGER NOT NULL, cls_name TEXT NOT NULL, conf REAL NOT NULL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    tag TEXT, crop_path TEXT,
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
CREATE TABLE IF NOT EXISTS ruler_classification (
    ruler_class_id INTEGER PRIMARY KEY,
    specimen_id  INTEGER NOT NULL REFERENCES specimen(specimen_id)           ON DELETE CASCADE,
    detection_id INTEGER NOT NULL REFERENCES archival_detection(detection_id) ON DELETE CASCADE,
    unit_type TEXT NOT NULL, votes_json TEXT, conf REAL,
    UNIQUE (detection_id)                      -- idempotent re-run
);

-- ruler_cf : per-ruler CF measurement. cf_px_per_cm = px_per_mm * 10; the WINNER is
-- flagged is_selected and duplicated onto specimen.cf_px_per_cm.
CREATE TABLE IF NOT EXISTS ruler_cf (
    ruler_cf_id INTEGER PRIMARY KEY,
    specimen_id    INTEGER NOT NULL REFERENCES specimen(specimen_id)                ON DELETE CASCADE,
    ruler_class_id INTEGER NOT NULL REFERENCES ruler_classification(ruler_class_id) ON DELETE CASCADE,
    unit_type TEXT, minimum_unit TEXT, second_unit TEXT,
    px_per_mm REAL, cf_px_per_cm REAL, cf_px_per_inch REAL,
    agreement REAL, n_ticks INTEGER, is_valid INTEGER NOT NULL DEFAULT 1,
    is_selected INTEGER NOT NULL DEFAULT 0,
    UNIQUE (ruler_class_id)
);

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
    -- grounded (nullable; a future MetricGrounding pass fills these when a CF exists)
    area_cm2 REAL, perimeter_cm REAL, length_cm REAL, width_cm REAL,
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
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (detection_id, instance_index)       -- one measurement row per leaf instance
);
CREATE INDEX IF NOT EXISTS ix_lmmeasure_spec ON leaf_landmark_measurement (specimen_id);

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
