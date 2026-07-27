# LeafMachine3 — TODO / Roadmap

Feature and add-on backlog, in priority order. Each item runs as a pipeline stage or a
post-analysis add-on after the primary pipeline:

```
ingest → ArchivalDetector → PlantDetector → PhenologyDetector → RulerClassifier
→ [RulerConversionFactor — stub] → LeafSegmenter → Morphology → LandmarkDetector
→ LandmarkMeasurements → LeafOrientation → MetricGrounding → Reporter
```

See `docs/LM3_Plan.html` for the architecture. Legend: **⛔ blocked** · **▶ ready** · **… planned**.

> **Design note — rotated bbox vs. leaf orientation (two separate concerns).**
> The Morphology rotated bounding box (currently **PCA**) is used **only for measurement** — it
> yields the leaf's width and height (`rotated_bbox_dim_min` / `dim_max`). It is *not* the leaf's
> orientation and is not used to rotate anything. **Actual leaf orientation** (which way is up,
> petiole → apex) comes from the **`LM3_Landmark_Detector`** (now integrated — keypoints in the
> `leaf_landmark` table), and *that* is what will be used to physically orient leaves for
> downstream steps (petiole width, canonical crops, etc.). Keep these decoupled: the bbox
> measures, the landmarks orient.

---

## 1. Petiole width  ▶ (ready — orientation now exists)

**Goal:** measure petiole width (px) per leaf instance and store it, mirroring LM2's petiole
project. Runs as an add-on **after Morphology**. Leaf orientation (step 3) is now DONE, so this is
unblocked; consume the oriented `Petiole` masks (or the raw masks + `oriented_leaf_rotation_angle_degreesCW`).

**Prerequisites (status):**
1. ✅ **`LM3_Landmark_Detector`** — DONE (alpha, `yolo26x_pose_640`): integrated as the
   `landmark_detector` stage → 31 keypoints per leaf in `leaf_landmark` (working coords),
   self-describing schema in `landmark_schema` / `landmark_skeleton`. (Alpha weights — will be
   retrained, but good enough to build the rest on.)
2. ◑ **`landmark_measurements`** (post-process stage) — **core metrics DONE**: `lamina_trace_length`,
   `lamina_extent`, `leaf_width`, `apex_angle`/`base_angle` (+ acute/obtuse/reflex type),
   `petiole_trace_length`, `lamina_curvature` → `leaf_landmark_measurement` table
   (`core/landmark_metrics.py`, occlusion-robust; angle convention confirmed via
   `modules/experiments/angle_checks.html`). **Still TODO for orientation:** derive the explicit
   **tip→base orientation axis** (from `lamina_tip`/`lamina_base` + midvein fit) and **lobe count**;
   these two feed #1a and the orientation step below. (LM2 reference:
   `LM3_Landmark_Detector/landmark_postprocess.py` `reassemble()`.)
3. ✅ **Leaf orientation code** — DONE: the `leaf_orientation` stage (`core/orientation.py`) computes
   the clockwise upright rotation per leaf (tip→base axis; PCA fallback with petiole/base/apex to pick
   the tip end) → `leaf_morphology.oriented_leaf_rotation_angle_degreesCW` / `oriented_leaf_success`.
   The Reporter emits `Original/` + `Oriented/` leaf-product trees from it. This also unblocks #1a.
   Petiole-width can now consume the oriented `Petiole` masks (or the raw masks + this angle).

**What / how:** consume the `Leaf` + `Petiole` instance masks per leaf crop. Skeletonize the
petiole, BFS the centerline to find its two ends + skeletal length, then measure the
*perpendicular* thickness near the blade junction (10–20 % down for longer petioles, midpoint
for short ones). Record an attachment flag (petiole within ~20 px of the leaf).

**Improve on LM2:** orient from the landmark axis instead of pre-oriented masks; take the
**median of several perpendicular widths** (LM2 samples one → noisy). Measure on the
**edge-refined `Petiole` mask** (#5): a thin petiole is very sensitive to leftover white-paper
edge pixels, so intersect the petiole mask with the whole-specimen mask before measuring width.

**Touches:** new stage `modules/petiole_width.py` + a `leaf_petiole` table
(`width_px, petiole_length_px, measure_location, touches_leaf`, FK → `leaf_id`); px → cm later
via the ruler CF (#3).

**Reference:** `leafmachine2/analysis/petiole_project_measure_petioles.py`
(`SegmentationMaskProcessor`); `_parallel` variant; `petiole_project_compare_gt_with_LM2.py`.

### 1a. Oriented length vs width  (same orientation unblocks it)

Morphology already stores the rotated box's two **side lengths** as `rotated_bbox_dim_max` /
`rotated_bbox_dim_min` — but these are the **geometric** long/short sides. Length vs width is an
*orientation* call: some leaves are wider than long, so `dim_max` is sometimes the **width**.

Two nullable columns are **already reserved in `leaf_morphology`**: `rotated_bbox_length` and
`rotated_bbox_width` (NULL until orientation exists).

**How we'll fill them (now ready — `leaf_orientation` exists):** apply the stored
`oriented_leaf_rotation_angle_degreesCW` (from `leaf_orientation`) to the rotated bbox and read off
which side runs along the tip→base axis. The box side along the **tip→base** axis
becomes `rotated_bbox_length`, the perpendicular side becomes `rotated_bbox_width` (independent of
which one is max vs min). No new geometry — just an axis-aware assignment of the two existing side
lengths, done in the orientation step.

---

## 2. Export to .xlsx  … (planned)

**Goal:** emit Excel workbooks of the results alongside the per-project SQLite DB.

**What / how:** a Reporter output option (`report.export.xlsx`) or a small exporter that reads
the project DB → one `.xlsx` per project with sheets: specimen metadata + CF, archival/plant
detections, phenology, `leaf_segmentation` + `leaf_morphology` (and later petiole / CF), plus an
aggregate summary sheet. Use `openpyxl` (add to `pyproject` extras).

**Touches:** `reporter` (or a new `export` module); `pyproject` optional dep.

---

## 3. Ruler CF — pixel ↔ metric  ▶ (ready to start; unblocks MetricGrounding)

**Goal:** finish the stubbed `RulerConversionFactor` stage so it produces `cf_px_per_cm`,
unblocking `MetricGrounding` (area cm², perimeter cm) and grounding leaf dims + petiole width.

**Needs, in order:**
1. **Background removal (semantic segmentation)** — isolate the ruler's graduations/markings
   from the ruler-crop background so tick/block detection is clean.
2. **Tick detector + block detector** — find the graduation ticks and the alternating blocks.
3. **Spacing → CF** — from tick/block spacing derive `px_per_mm` → `cf_px_per_cm`, using the
   unit system already predicted by `RulerClassifier` (METRIC_MM, STD_IN8, …); write `ruler_cf`
   rows and set `specimen.cf_px_per_cm` (schema + stage already exist, just disabled).

**Reference:** `leafmachine2/machine/utils_ruler.py` (`convert_rulers`); the
`LM3_Ruler_Distance_Groundtruth` tool in the training repo (tick-center labeling + px→cm/inch
math + cross-validation); `RulerClassifier` provides the graduation system.

**Touches:** `modules/ruler_conversion_factor.py` (enable the stub); new inference wrappers
(bg-removal seg + tick/block detectors); `ruler_cf` table (already in schema).

---

## 4. ECT — Euler Characteristic Transform shape analysis  … (planned)

**Goal:** add ECT shape descriptors for leaves (post-analysis add-on) for downstream
morphometric / phylogenetic work.

**What / how:** operate on leaf instance masks/outlines → compute the ECT → store descriptors
(and/or export). Likely heavy; an optional add-on, off by default.

**Reference:** `leafmachine2/ect_methods/` (ECT + ridgeline / PGLS analysis scripts).

**Touches:** new add-on module + a descriptor table / export.

---

## 5. Leaf Edge Refinement  ⛔ (blocked — needs `LM3_Specimen_Segmentation` trained + exported)

**Goal:** tighten each leaf-instance mask by trimming leftover background (white paper) that the
YOLO26 leaf-segmentation masks include when they aren't cut tight enough at the edges. Use the
**whole-specimen** plant mask from **`LM3_Specimen_Segmentation`** (a precise plant-vs-background
binary mask of the entire sheet) to exclude edge pixels that fall outside the true plant material.
**High value:** the leaf products (lamina masks + RGB cutouts) are LM3's headline output, so cleaner
edges directly improve them (and the hole-aware areas).

**Flow:**
1. Train + export **`LM3_Specimen_Segmentation`** — a precise binary mask of ALL plant material on
   the sheet (UNet++ / YOLO26-seg / BiRefNet candidates already set up in that repo).
2. New stage **`SpecimenSegmenter`** runs **right after PlantDetector** (before LeafSegmenter):
   infer the whole-specimen mask in the working/parent frame and **store it** (a per-specimen mask —
   new `specimen_mask` table with a polygon/RLE, or a mask file referenced from the DB).
3. **Refine at leaf-segmentation time:** each instance mask is already in the parent/working frame,
   so its position is known and it shares the specimen mask's coordinate frame. Intersect each
   instance mask with the parent specimen mask and **keep only the parts of the instance mask that
   are also inside the specimen mask**, dropping stray paper/background edge pixels. Apply this to
   **both the `Leaf` AND the `Petiole` instance masks** (petioles are thin, so leftover paper at
   their edges is especially damaging — see #1 petiole width). Recompute the affected geometry
   (area, perimeter, petiole width, etc.) from the refined masks.

**Fallback (if the specimen model isn't precise enough):** run the **`paper_removal`** tool (the one
used to clean the `LM3_Specimen_Segmentation` **training** data) as a second pass on the refined
crops/masks to strip any remaining paper at the edges.

**Touches:** new `modules/specimen_segmenter.py` + an inference wrapper + a `specimen_mask` store;
the intersection/refine step (either inside `leaf_segmenter.persist` or a small post-process stage
right after LeafSegmenter); downstream `leaf_morphology` (areas shift with tighter masks) and every
leaf product (`Original/` + `Oriented/`) inherit the cleaner edges automatically.

**Reference:** the `LM3_Specimen_Segmentation` repo (SAM3 box-prompt mask generation → binary
bg-removal dataset; UNet++ / YOLO26-seg / BiRefNet trainers) and its `paper_removal` training-data
cleaning tool.
