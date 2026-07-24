# LeafMachine3 — TODO / Roadmap

Feature and add-on backlog, in priority order. Each item runs as a pipeline stage or a
post-analysis add-on after the primary pipeline:

```
ingest → ArchivalDetector → PlantDetector → PhenologyDetector → RulerClassifier
→ [RulerConversionFactor — stub] → LeafSegmenter → Morphology → MetricGrounding → Reporter
```

See `docs/LM3_Plan.html` for the architecture. Legend: **⛔ blocked** · **▶ ready** · **… planned**.

---

## 1. Petiole width  ⛔ (blocked — needs leaf orientation)

**Goal:** measure petiole width (px) per leaf instance and store it, mirroring LM2's petiole
project. Runs as an add-on **after Morphology**.

**Blocked by — must land first, in order:**
1. **`LM3_Landmark_Detector`** — train + export a landmark model (midvein / petiole / apex /
   base keypoints), analogous to LM2's landmark detector.
2. **Leaf orientation code** — use those landmarks (petiole → apex) to define each leaf's
   canonical orientation. This is the hard prerequisite: the petiole-width algorithm needs to
   know which end of the petiole meets the blade (LM2 sidestepped this by requiring pre-rotated
   "Oriented_Masks").

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
