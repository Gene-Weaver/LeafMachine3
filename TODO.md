# LeafMachine3 — TODO / Roadmap

Feature and add-on backlog, in priority order. Each item runs as a pipeline stage or a
post-analysis add-on after the primary pipeline:

```
ingest → ArchivalDetector → PlantDetector → PhenologyDetector → RulerClassifier
→ [RulerConversionFactor — stub] → LeafSegmenter → Morphology → MetricGrounding → Reporter
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

## 1. Petiole width  ⛔ (blocked — needs landmark_measurements + leaf orientation)

**Goal:** measure petiole width (px) per leaf instance and store it, mirroring LM2's petiole
project. Runs as an add-on **after Morphology**.

**Blocked by — must land first, in order:**
1. ✅ **`LM3_Landmark_Detector`** — DONE (alpha, `yolo26x_pose_640`): integrated as the
   `landmark_detector` stage → 31 keypoints per leaf in `leaf_landmark` (working coords),
   self-describing schema in `landmark_schema` / `landmark_skeleton`. (Alpha weights — will be
   retrained, but good enough to build the rest on.)
2. **`landmark_measurements`** (new post-process step; ▶ ready) — derive from the `leaf_landmark`
   keypoints: ordered midvein/petiole **traces**, lamina + midvein **lengths**, **apex/base
   angles**, lobe count, and the **tip→base orientation axis**. Port LM2 `detect_landmarks()` /
   `landmark_processing.py` — a ready port already lives at
   `LM3_Landmark_Detector/landmark_postprocess.py` (`reassemble()`). This also feeds #1a.
3. **Leaf orientation code** — use the tip→base axis from #2 to know which end of the petiole
   meets the blade (LM2 sidestepped this by requiring pre-rotated "Oriented_Masks").

**What / how:** consume the `Leaf` + `Petiole` instance masks per leaf crop. Skeletonize the
petiole, BFS the centerline to find its two ends + skeletal length, then measure the
*perpendicular* thickness near the blade junction (10–20 % down for longer petioles, midpoint
for short ones). Record an attachment flag (petiole within ~20 px of the leaf).

**Improve on LM2:** orient from the landmark axis instead of pre-oriented masks; take the
**median of several perpendicular widths** (LM2 samples one → noisy).

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

**How we'll fill them:** the `LM3_Landmark_Detector` already predicts the **`lamina_tip`** (and
`lamina_base`) — available now in `leaf_landmark` (working coords) for the same crop the rotated
bbox came from. Once `landmark_measurements` (#1 step 2) gives the tip→base axis, apply the same
transform used to orient the leaf to the rotated bbox → the box side along the **tip→base** axis
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
