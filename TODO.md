# LeafMachine3 — TODO / Roadmap

Feature and add-on backlog, in priority order. Each item runs as a pipeline stage or a
post-analysis add-on after the primary pipeline:

```
ingest → MPConversionFactor → ArchivalDetector → PlantDetector → SpecimenSegmenter
→ PhenologyDetector → RulerClassifier → RulerConversionFactor → LeafSegmenter → Morphology
→ LandmarkDetector → LandmarkMeasurements → LeafOrientation → PetioleWidth
→ MetricGrounding → Reporter → ECT
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

## 1. Petiole width  ✅ (DONE — `petiole_width` stage)

**Status:** DONE. The `petiole_width` stage (`core/petiole.py`, `modules/petiole_width.py`) measures
each leaf's petiole width = **median** of perpendicular thickness samples of the `Petiole` mask near
the blade junction, along the landmark petiole centerline (`lamina_base` → `petiole_0..4` →
`petiole_tip`) → `leaf_petiole` table (`width_px`, `length_px`, `n_samples`, `touches_leaf`,
`measure_location`, the sample/width segments; `width_cm` reserved for MetricGrounding). The Reporter
draws a purple width band on the summary and a per-leaf `Overlay/Overlay_Petiole/`. **Still TODO:**
1. ▶ **Ground `width_px` → `width_cm`** — now UNBLOCKED (#3 shipped; `specimen.cf_px_per_cm` is
   populated on high-confidence sheets). `MetricGrounding` currently grounds only `leaf_segmentation`
   (`area_cm2`, `perimeter_cm`, `bbox_*_cm`) and never touches `leaf_petiole` → **0/84 `width_cm`**.
   Small change: extend `metric_grounding` to divide `leaf_petiole.width_px` / `length_px` by the CF.
2. ⛔ Run it on the **edge-refined** petiole mask once #5 step 3 lands (it uses the raw mask today).

**Original plan (kept for reference):** measure petiole width (px) per leaf instance, mirroring LM2's
petiole project. Leaf orientation (step 3) is DONE. (We used the landmark petiole centerline instead
of skeletonising the mask; skeletonisation remains a possible fallback if landmarks are unavailable.)

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

## 2. Tabular export  ✅ (DONE as CSV — `reports/Data/`; .xlsx still optional)

**Status:** DONE, as **CSV rather than .xlsx**. `reporting/data_export.py`, called as the
Reporter's last step (`report.data` in `LM3_settings.yaml`), writes nine files into
`reports/Data/`. The read queries live in `core/db.py` (the only module allowed to issue SQL); the
exporter owns column order, units and prose, and a test asserts the two cannot drift apart.

- **`leaf_measurements.csv`** — the master file the request was really about: ONE ROW PER LEAF,
  111 columns, joining `leaf_segmentation` + `leaf_morphology` + `leaf_landmark_measurement` +
  `leaf_petiole` + `bilateral_symmetry` to the specimen and detection metadata that identifies it.
  `leaf_uid` / `crop_file_token` join a row to that leaf's exported images by filename.
- Plus `specimen_summary` (one row per sheet + roll-ups), `detections`, `landmarks`,
  `ruler_conversion_factor`, `ruler_crops`, `run_stages`, `stage_errors`, and a generated
  `data_dictionary.csv` documenting every column of every file.
- **`leaf_ect` is deliberately excluded**: ECT runs AFTER the Reporter (it consumes the Reporter's
  oriented masks), so at export time its rows are absent or left over from the previous run.
- Two schema changes went with it: `leaf_landmark_measurement` gained the five `*_cm` columns and
  MetricGrounding now fills them; `leaf_morphology`'s four never-written cm columns were DROPPED
  (cm for a leaf lives on `leaf_segmentation`, keyed by the same `leaf_id`).

**Still open (optional):** the same row builders can back an `.xlsx` workbook, one sheet per CSV,
via `openpyxl` as a `pyproject` extra. Also unexported: a cm version of
`lamina_area_excl_holes_px` — the px value and the CF are both present, but MetricGrounding does
not ground the hole-corrected area, and the exporter does not compute values the pipeline never
stored.

---

## 3. Ruler CF — pixel ↔ metric  ✅ (DONE — lattice method; MetricGrounding unblocked)

**Status:** DONE, but via a **different (better) approach** than planned below: instead of
bg-removal + tick/block detectors, the shipped stage ports the **tick-lattice** method from
`LM3_Ruler_Segmentation` (`inference/ruler_lattice/`, pure cv2/numpy — no models). It measures every
Ruler crop's periodic tick lattice, names the unit from the `RulerClassifier` class, and fuses all of
a sheet's rulers into ONE CF anchored by the MP-predicted CF.

- **Gated publish:** `specimen.cf_px_per_cm` (working frame) is written **only** at `confidence=high`
  (`status='published'`); a withheld/failed read leaves it NULL so consumers fall back to
  `cf_px_per_cm_predicted_by_mp` rather than trusting a possibly unit-misnamed CF.
- **Tables:** `ruler_CF_lattice` (one row per sheet) + `ruler_CF_lattice_crop` (one per ruler crop,
  including skipped/failed) + the `v_ruler_CF_lattice` view. The old `ruler_cf` table was removed.
- **QC:** `reports/Overlay/Overlay_Ruler_Lattice/` — rebuilt by the Reporter from the stored record
  alone (proved pixel-identical to a live render by the engine's `--selftest`).
- **Verified:** on the 19-image example, 12/19 sheets published a CF → **189/267 leaves now have
  `area_cm2` / `perimeter_cm` / `bbox_*_cm`** (MetricGrounding was previously a no-op).

**Still open (small):** ground **petiole `width_px` → `width_cm`** — MetricGrounding does not touch
`leaf_petiole` yet (0/84 populated). Now unblocked; see #1.

**Reference (kept):** `leafmachine2/machine/utils_ruler.py` (`convert_rulers`); the
`LM3_Ruler_Distance_Groundtruth` tool; `ruler_cf_bench.py` in `LM3_Ruler_Segmentation` compares
against groundtruth.

---

## 4. ECT — Euler Characteristic Transform shape analysis  ✅ (DONE — `ect` stage)

**Status:** DONE. The `ect` stage runs **after the Reporter** (it consumes the Reporter's oriented
leaf-product masks rather than re-deriving them) and computes the ECT per oriented `Leaf_WHOLE` leaf
using the **modern `ect` package natively** (not LM2's older port).

- **Per leaf →** `reports/Leaf_Data/Coordinates/*.h5` with four sections: `image_metadata` (parent
  filename, MP, MP-predicted CF, final CF, image dims, `mask_includes`), `ect_data` (matrix + thetas
  + thresholds), `leaf_outline` (unit-circle normalized by default, optional literal px coords, LM2
  header convention), `leaf_outline_simple` (Douglas-Peucker, LM2 `cutoff=500`).
- **Visuals →** `Leaf_Data/Oriented_Leaf_Radial_ECT/` (polar) + `Oriented_Leaf_ECT/` (Cartesian) +
  `Oriented_Leaf_Radial_ECT_Overlay/` (polar with the traced outline on top), configurable palette.
  All three re-base the direction axis on the leaf tip so they line up with each other and with the
  tip-up oriented masks; the `.h5` matrix stays raw. Credit to **DanChitwood/ect_to_shape_CNN** for
  the radial rendering.
- `include_petiole` / `include_holes` pick the `mask_includes` variant and **override**
  `report.leaf_products` so the Reporter is guaranteed to export the mask ECT needs.
- Parallelized as a **per-leaf fanout on the CPU process pool** (see #8).

**Reference:** `leafmachine2/ect_methods/` (ECT + ridgeline / PGLS analysis scripts).

---

## 5. Leaf Edge Refinement  ◑ (UNBLOCKED — steps 1–2 DONE; **step 3 (the actual refine) is all that's left**)

> **Status update:** the blocker is gone. `LM3_Specimen_Segmentation` is trained + exported and the
> **`SpecimenSegmenter` stage ships** (UNet++ `control` @1024 via ONNX, right after PlantDetector),
> storing a per-specimen mask in `specimen_mask` (+ `Binary_Masks__Specimen` / `RGB_Masks__Specimen`
> exports and an `Overlay_Specimen_Segmentation` QC panel). A `paper_removal` follow-up pass is also
> wired. **What remains is step 3 below:** nothing yet intersects the leaf/petiole instance masks with
> that specimen mask — `leaf_segmenter` does not read `specimen_mask` at all.

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

---

## 6. Pick the production specimen-segmentation model  ▶ (UNet vs YOLO26-seg vs BiRefNet)

All three `LM3_Specimen_Segmentation` models are (or are nearly) trained on the same paperclean
dataset (`-unetMasksFromSam3-paperclean`, 18,904; field project excluded) — **compare them and pick
the production `SpecimenSegmenter`** (the UNet `control` is the currently-wired default —
`models/specimen_segmenter/`).

- **UNet++/EffB7 @1024** (`unet_masksFromSam3_paperclean_control_1024`, val dice 0.943) — shipped
  default; under-segments some faint dried-brown *Quercus* (empty/tiny masks on ~2/19 test sheets).
- **YOLO26x-seg @1280** (`yolo26/.../yolo26x-seg-1280-paperclean`) — training on GPU0.
- **BiRefNet_HR @1024** (`birefnet/runs/birefnet-hr-1024-paperclean`) — fine-tuning on GPU1.
  **BiRefNet looks incredible** even at epoch 135 (just 5 fine-tune epochs): crisp HR lobe
  boundaries, clean removal of the label / ruler / colour-card. → **strong candidate for the new
  production pick.**

  BiRefNet ep135 CPU-inference QC (view these):
  - `LM3_Specimen_Segmentation/birefnet/runs/birefnet-hr-1024-paperclean/qc_images/048ze8d5__CM_2859177966_Sapindaceae_Acer_saccharum__birefnet_ep135_qc.jpg`  (train)
  - `LM3_Specimen_Segmentation/birefnet/runs/birefnet-hr-1024-paperclean/qc_images/048ze8d5__CMN_1804514859_Sapindaceae_Acer_saccharum__birefnet_ep135_qc.jpg`  (test)

**To decide:** once BiRefNet finishes (and the self-heal / self-heal-extLeaf UNet variants), QC all
models on the same specimens (`exp__paperclean_train/qc_infer.py` set + the bad-GT set), compare
edge tightness / thin-structure (petiole) recall / robustness on the faint-dried-leaf hard cases,
then export the winner (ONNX) as the `SpecimenSegmenter`.

**COMPARISON DONE (2026-08-01)** — all three run on 12 specimens (primary QC + Acer + hard cases),
panels in `LM3_Specimen_Segmentation/compare_unet_yolo26_birefnet/`. **No clean single winner — a
3-way tradeoff, each with a distinct failure mode:**
- **BiRefNet** — best mask quality (crispest HR edges; best recall on pale/faded leaves e.g. Quercus
  32% vs UNet 20%; excludes interior paper gaps). Wins ~10/12. **BUT one catastrophic collapse**
  (Macaranga *magnifolia* → 0.5% fg) on an extremely faded/torn leaf.
- **YOLO26-seg** — most robust recall (got that magnifolia at 47.6%; never catastrophic). **BUT
  systematically over-segments** — fills *enclosed interior paper* (Chrysophyllum triangle) → ~+7%.
- **UNet++ control** — clean/tight, rarely over-includes; **weakest recall** (under-segments faded plant).

**Recommendation: ship BiRefNet as `SpecimenSegmenter` for edge quality, WITH a collapse safeguard** —
either (a) fallback to YOLO26 / the LM3 leaf mask when BiRefNet fg ≈ 0 on a plant-bearing sheet, or
(b) run the §7 self-heal-extLeaf variant (the robust-leaf union targets exactly this collapse). If a
single model with no fallback is required: YOLO26 is safest (no catastrophic fail, eat some paper),
UNet if tight paper-exclusion outranks recall. Export the chosen model to ONNX (mirror the UNet
export) and wire it in place of the current UNet `control` default.

**Efficiency (measured this session, 48 GB RTX 6000 Ada) — UNet is by far the lightest, BiRefNet the heaviest:**

| model | params | ckpt | train VRAM | train speed | **GPU inference** |
|---|---|---|---|---|---|
| **UNet++/EffB7** @1024 | ~75M | ~550MB* | ~34GB @bs4 | ~30 min/ep | **76 ms/img (13.2 img/s)** |
| **YOLO26x-seg** @1280 | 70.6M | 136MB | ~38GB @bs16 | ~30 min/ep | **196 ms/img (5.1 img/s)** |
| **BiRefNet-HR/swin-L** @1024 | ~200M | 845MB | 22.6GB @bs2 (OOM @2048) | ~71 min/ep | **197 ms/img (5.1)** — but **CPU = minutes/img** |

<sub>*UNet ckpt inflated by stored EMA. BiRefNet needs torch.compile+bf16 to train; its GPU inference is fine (~200ms) but CPU-only deploy is a non-starter.</sub>

**Production self-training round + cull the degraded specimens (planned).** Instead of self-heal *fixing*
bad masks, this time **remove them**: use the newly-trained UNet++ to regenerate training masks (as with
`-unetMasksFromSam3`), but **cull the poor/degraded specimens** (the extremely faded/torn sheets like the
*Macaranga magnifolia* in the comparison, whose auto-masks are unreliable) from the training set — the
self-heal `worst_gt.json` rankings are a ready-made cull list. Hypothesis: a cleaner (culled) training set
lets the model learn crisper, so it generalizes *better* on the reasonable "Macaranga-like" images rather
than being dragged by unrecoverable sheets. (Then re-QC on the hard set to confirm it didn't just get
brittle.)

---

## 7. Run the self-heal-extLeaf UNet variant  ⏸ (built, DE-QUEUED — run later)

3rd self-heal option that fixes the production self-heal's hard-core collapse-to-empty: at each
self-relabel the target = **model prediction ∪ LM3 `Binary_Masks_Full_Image__Leaf`** (robust leaf
mask = a floor the mask can't drop below). Relabels at **ep 10/20/30**, trains to **34**, 1024/batch4.

**All code is already built** (in `LM3_Specimen_Segmentation/`), just not running:
- `train_unet.py --external-mask-dir <dir>` — unions the external mask into each relabel (done).
- `dataset/make_leaf_masks.py` — runs leafmachine3 (plant-detect + leaf-seg only) on the paperclean
  train images → harvests `Binary_Masks_Full_Image__Leaf` → flat `/data/lm3-specimen-leaf-masks/<stem>.png`.
  Config validated via `--dry-run`. Needs a free GPU (~30 min–2 h).
- `unet/run_selfheal_extleaf.sh <gpu>` — orchestrator: Phase A (leaf masks) → Phase C (train + QC).
- `unet/_autolaunch_extleaf.sh` — waits for a free GPU then runs the orchestrator (was launched, now KILLED).

**To run later:** `bash LM3_Specimen_Segmentation/unet/run_selfheal_extleaf.sh <gpu>` on a free GPU
(or re-launch the `_autolaunch_extleaf.sh` watcher). NOTE Phase A's real leafmachine3 run was never
smoke-tested on a GPU (only dry-run) — the orchestrator aborts before training if it harvests
&lt;1000 masks. Run name will be `unet_masksFromSam3_paperclean_selfheal_extLeaf_1024`.

---

## 8. Performance: CPU parallelism + hardware profiling  ✅ (DONE)

CPU stages all ran on a shared **thread** pool, which gives ~zero speedup for GIL-bound cv2/numpy/
matplotlib work (measured: the ruler lattice *degraded* to 0.72× at 32 threads).

- `PipelineStage.cpu_parallel = "process"` routes a stage to the existing **spawn process pool**;
  `fanout = True` lets `collect_items` emit >1 WorkItem per specimen (the executor then marks a
  specimen done only after its **last** sub-item, so per-image resume stays correct).
- **`ruler_cf`** → process pool: **114s → 8.1s** (~14×) on a 21-sheet run, identical results.
- **`ect`** → **one WorkItem per leaf** + each worker writes its own `.h5`/PNGs and returns only the
  small DB row: **180s → 24s** (~7×). `ect_viz` moved off `pyplot` (global state: thread-unsafe +
  leaks figures) to the OO `Figure`/`FigureCanvasAgg` API.
- **`hardware_setup`** now measures, per machine: the CPU **process-pool knee** (24 workers here, vs
  the blind `cores-2`=62), the **spawn overhead** → a per-stage **`min_pool_items`** threshold below
  which a small batch runs **serially** instead of paying spawn cost, and the **disk-write knee**
  capping the I/O-bound `reporter` (8 workers). GPU `workers_per_gpu` is sized from the free VRAM of
  the **chosen** GPU (`compute.devices`), not `gpus[0]` — sizing against a busy GPU 0 had capped
  every GPU stage at 1 worker.
- **Reporter stays thread-parallel** (deliberately): its cost is `cv2.imwrite` + cv2/numpy, which
  release the GIL, and its `ReportBundle` payload isn't picklable for a spawn pool.

## 9. Run timing report  ✅ (DONE — `timing.enabled`)

`reports/Timing/timing.{csv,html}` — per module: wall time + share, CPU/GPU utilization, **RAM/VRAM
deltas over a pre-run baseline** (so background jobs are excluded), allocated workers, and throughput
in the module's own unit + images/s. The HTML matches the ruler-CF guide style: colored Gantt
timeline, hover tooltips on every column, a "why some modules ran serially" blurb, and a final
**Machine & tuning** section (per-GPU tiles in whole GB, chosen GPU badged).

**This is what caught the GPU regression below** — worth using before/after any perf change.

## 10. ⚠️ GPU: ONNX stages were silently running on CPU  ✅ (FIXED — watch for regressions)

Most "GPU" stages are **ONNX** (archival/plant/landmark detectors, specimen segmenter, ruler
classifier); only `leaf_segmenter` is PyTorch. `torch.cuda.is_available() == True` says **nothing**
about them. Two bugs made ORT fall back to CPU silently:

1. plain **`onnxruntime` (CPU-only) was installed over `onnxruntime-gpu`**, so no CUDA provider was
   offered → fixed by installing only `onnxruntime-gpu` (**1.20.2**; 1.20.1 was yanked from PyPI).
2. **`LD_LIBRARY_PATH` was set in-process** by `ensure_cuda_libpath()`, but the dynamic loader reads
   it only at **process start**, so `libonnxruntime_providers_cuda.so` never loaded → fixed with
   `machine3._exec_with_cuda_libpath()`, which sets it and **re-execs once** before any ORT import.

**Impact:** 19-image example **114s → 75s**; each GPU stage went from ~3 MB VRAM / 0% GPU to
2–7.4 GB / real utilization. **To verify after any env change:** run with `timing.enabled` and check
the **VRAM Δ** column — never trust the provider string alone.

---

## 11. Train UNet++ for leaf **semantic** segmentation  ▶ (crisp edges to help LeafSegmenter)

The specimen comparison (§6) showed UNet++'s clean, tight boundaries; meanwhile the current
`leaf_segmenter` (YOLO26x-seg, **instance** seg) under-segments leaf edges (the whole reason for §5).
**Idea:** train a **UNet++ semantic** leaf segmenter (leaf-vs-background, crisp edges) on the
`LM3_Leaf_Segmentation` masks (semantic union of leaves), then use its mask to **sharpen** the YOLO
leaf instances — YOLO gives the per-leaf **instances** (needed to separate touching leaves), UNet++
gives the crisp **edges**; intersect/snap each YOLO instance to the UNet leaf-semantic mask (same
mechanism as §5 but with a dedicated leaf model instead of the specimen mask). Reuse the
`LM3_Specimen_Segmentation` UNet++ trainer + `paper_removal`/self-training tooling. **Caveat:**
semantic ≠ instance — this refines edges, it does **not** replace YOLO's instance separation.

## 12. Valid-leaf classifier ensemble (LM2-style)  ▶ (quality flag per `Leaf_WHOLE`)

Port LeafMachine2's **valid-leaf classifier ensemble** — a per-`Leaf_WHOLE` quality/validity score that
flags the **highest-quality** leaf masks (complete, unoccluded, cleanly segmented) vs
partial/damaged/overlapping. Add a variable (e.g. `is_valid_leaf` / `leaf_quality` on the leaf row) so
downstream measurement (Morphology, Landmarks, PetioleWidth) can be **restricted to the best leaves**
instead of measuring every detected blob. Reference: LM2's leaf classifier (CNN ensemble). Pairs with
§5/§11 (refine the edges, then rank/keep the good leaves) and with #1a/measurement selection.

## 13. MOMOCs export for Fourier (elliptical Fourier) shape analysis  … (planned)

Emit leaf outlines in a **Momocs**-ready format (R morphometrics package) so users can run
**elliptical Fourier analysis (EFA)** on the leaf shapes. Momocs works on closed-outline coordinate
sets (`Coo`/`Out` objects): export **one ordered boundary (x,y) polyline per leaf** — traced from the
`Leaf_WHOLE` mask contour (single largest closed contour; ideally the **edge-refined / oriented** mask
from §5/§11 so shapes are clean and comparably aligned) — plus a **factor/metadata table keyed by
`leaf_id`** (taxon, sheet, source). Ship as a Reporter output option (per-project: a coords file per
leaf + the factor table, or a combined form `Momocs::import_*` / `Out()` can read directly). Second
shape-analysis path alongside **#4 (ECT)**; reuse the existing `Leaf_Outlines*` outputs / LM2's outline
export if one exists. **Touches:** `reporter` (or a new `export` module); optional R-side helper script.

## 14. STL generator as a leafmachine.org module  ▶ (code exists — expose on the site)

Expose `postprocessing/generate_stl_from_mask.py` (leaf mask → 3D-print STL) as a **module on
leafmachine.org** — a web UI + API endpoint where a user uploads a leaf/mask and downloads a printable
STL. The engine + a `server/postprocess_api.py` already exist; the work is wiring it into the site
(route, upload/download UI, job handling) rather than new geometry.

## 15. Nature-dataset leaf collages + archetype ranking  … (paper figures)

Back on the **Nature-paper dataset**, build collages of: (a) the **best ENTIRE** leaves, (b) the **best
NON-ENTIRE** leaves, and (c) a **LOBED-only** version. Requires running the **leaf archetype ranking**
first: score leaves, then **sample randomly from those scoring > 50%** (a spread of the "best-ish", not
just the single top) so the collage is representative, not cherry-picked. Reuse
`postprocessing/generate_leaf_collage.py`.

## 16. Total leaf area of the Nature dataset vs a familiar area  … (paper impact stat)

Compute the **grand total sum of all leaf area** across every leaf in the Nature dataset (cm² → m²),
then translate it into a **relatable reference** ("≈ N tennis courts / football fields / city blocks")
for the paper's scale/impact framing.

## 17. ECT on low-res sheets is blocky — upscale + smooth experiment  ▶ (improve #4 masks)

Low-resolution sheets yield **blocky masks → jagged ECTs** (#4). Experiment: **upscale the low-res image,
then smooth / interpolate / median-filter** before/after masking so the recovered leaf mask (and its
ECT) is closer to the true smooth outline. Compare ECTs pre/post on low-res examples; pick the
cheapest step (bicubic upscale + light median/Gaussian, or contour smoothing) that de-jags without
eroding real shape detail.

## 18. BiRefNet leaf segmenter module (alt to YOLO26 leaf_segmenter)  ▶ (2 single-class models → rebuild 3-class output)

**Goal:** a new LM3 module that runs the two BiRefNet foreground models as an *alternative* to the current YOLO26x-seg `leaf_segmenter` — for crisp semantic leaf edges where YOLO26 instance-seg under-segments.

**The two BiRefNet models** (each a single-class foreground selector; `LM3_Leaf_Segmentation/birefnet/export/`):
- `birefnet-hr-1024-leaf` = **Leaf ∪ Petiole − Holes** (whole leaf incl. petiole)
- `birefnet-hr-1024-leaf-no-petiole` = **Leaf − Holes** (blade only)

**Reconstruct the 3-class YOLO26 output** (Leaf / Petiole / Hole) from the two masks:
- **Leaf (blade)** = the no-petiole mask
- **Petiole** = leaf mask **−** no-petiole mask (set difference)
- **Holes** = OPEN QUESTION: both models carve holes, so hole regions are 0 in both → not recoverable from this pair alone. Options: keep YOLO26 for the hole class, train a hole model, or drop holes.

**Production runtime = TensorRT ORT-EP + the fp32 ONNX** (`birefnet_leaf_1024_fp32.onnx` run via ORT `TensorrtExecutionProvider` — measured ~99 ms/img, ~3.8 GB VRAM, engine auto-builds + caches). Do NOT use plain ONNX-CUDA (~206 ms, 16.9 GB VRAM). Ship the `.onnx` + the ORT-TRT-EP session config; artifacts in `birefnet/export/<model>/artifacts/`.

**DECISION TEST — do this first:** run all 3 on a selected image set and compare **leaf-edge precision** side-by-side — YOLO26x-seg (3-class) vs BiRefNet-leaf vs BiRefNet-no-petiole (+ the reconstructed petiole). If BiRefNet's crisp edges beat YOLO26's under-segmentation → adopt the BiRefNet **ORT-EP** pair in production.

**Blocked on:** no-petiole training finishing (currently epoch 131/170, ~1.6 days) + its export (autoexport watcher armed). The leaf model is already trained + exported. Related to §11 (UNet-for-leaves — same crisp-edges-vs-YOLO motivation).

## 19. ✅ YOLO26 leaf_segmenter stringing fix — use raster masks.data, not masks.xy  ▶ (largest-CC Leaf/Petiole, keep all Holes)

**✅ DONE (2026-08-11).** Implemented in `leafmachine3/inference/leaf_segmenter.py`: `_to_instances` now reads the raster `results.masks.data` and reduces via new module fn `build_instances` (Leaf/Petiole = largest connected component, Hole = keep all >= min_area_px). `factory.load_segmenter` passes `min_area_px` from `mask_storage.min_area_px` (64). Regression test `tests/test_leaf_segmenter_masks.py` (5 cases: no-string largest-CC, all-holes-kept, multi-detection collapse, ordering, empty). Full suite: 242 passed. Verified on the real model (Quercus_suberoides 792_1691): was a 597-pt strung ring (7 comps / +308px bridge), now a clean 469-pt single-component Leaf polygon.

---

**Bug:** `leafmachine3/inference/leaf_segmenter.py` builds `Instance.polygon` from Ultralytics `results.masks.xy`. `masks.xy` runs `masks2segments(strategy='all')`, which **concatenates a single instance's disconnected mask contours into ONE polygon ring** — so when the leaf mask has islands (a torn fragment, a neighbor leaf poking in) the ring bridges them, drawing thin **"stringing"** when rasterized/used. Proven on `UM_1807393061…Quercus_suberoides…792_1691…`: 1 Leaf instance, raster mask = 7 clean components, polygon = single 597-pt ring, **+308 px of pure string**. NOT a training problem; NOT fixable by capping `max_det` (the islands live inside one instance's mask).

**Fix (Will's rule):** derive masks from the **raster `results.masks.data`**, reduced per class:
- **Leaf, Petiole → single LARGEST connected component** (drops islands + bridges).
- **Hole → keep ALL components** (only sub-~9px speckle removed).

Since `mask_storage.format = polygon_xy`, emit **one polygon per component** via `cv2.findContours(RETR_EXTERNAL)` on the raster (respect `min_area_px=64`) instead of the concatenated ring.

**Where:** `leafmachine3/inference/leaf_segmenter.py` `_to_instances` (currently `polys = masks.xy`). Add a regression test: a multi-component mask must yield N separate polygons with no bridge. The QC harness `LM3_Leaf_Segmentation/birefnet_compare_to_yolo26/mask_utils.py::data_to_classmasks` already implements this — port it.

**Impact:** removes the single biggest visual artifact across the test set; also means part of "BiRefNet looks cleaner" was really YOLO's polygon step, not its mask. Independent of §18.

## 20. Ground the hole-corrected lamina area to cm²  ▶ (the one measurement in `reports/Data` with no cm twin)

**Goal:** give `lamina_area_excl_holes_px` — leaf TISSUE area, holes subtracted — a cm² twin, so the
most biologically meaningful area in `reports/Data/leaf_measurements.csv` is available in real units
without the user doing arithmetic.

**Why it is missing:** MetricGrounding grounds `leaf_segmentation.area_px` (the OUTER silhouette,
holes INCLUDED) → `area_cm2`, and nothing else area-shaped. The hole decomposition
(`lamina_area_excl_holes_px`, `lamina_hole_area_px`, `n_holes`) is computed one stage earlier and
lives on `leaf_morphology`, which has no cm columns. So the export ships
`lamina_area_incl_holes_cm2` but not `lamina_area_excl_holes_cm2`. Nothing is lost — the px value
and `cf_px_per_cm_ruler` are both on every row — but a leaf with holes has no ready-to-use tissue
area, and the exporter deliberately does not compute values the pipeline never stored.

**Where the grounded value should land — pick one:**
- **(a) `leaf_segmentation`** *(recommended)*: add `lamina_area_excl_holes_cm2` and
  `lamina_hole_area_cm2` beside the existing `area_cm2` / `perimeter_cm`. Keeps the single rule
  "a leaf's cm values live on `leaf_segmentation`, keyed by `leaf_id`", which is what the CSV export
  and the data dictionary already tell users. Mildly odd that the px source sits in another table,
  but both are keyed by the same `leaf_id`.
- **(b) `leaf_morphology`**: natural home next to the px twin, but that table's four cm columns were
  just DROPPED for being permanently NULL. Re-adding cm there is only safe if the new ones are
  actually written on every grounded leaf — otherwise the exact confusion returns.

**Touches:** `core/schema.sql` (+ an ALTER migration in `core/db.py`); `_OWNED_LEAF_COLS` so a
`--restart metric_grounding` nulls them; `records.Grounded` + `db.set_leaf_metrics_cm`;
`modules/metric_grounding.py` — it must now read the morphology rows too, so add `"morphology"` to
its `depends_on` (already earlier in `STAGE_ORDER`, so no reordering); the `_LEAF_COLUMNS` spec and
dictionary prose in `reporting/data_export.py`; tests in `test_metric_grounding.py` +
`test_data_export.py`.

**Watch for:** area divides by `cf²`, not `cf`. And `excl + hole == incl` must survive the round
trip to cm, so ground the two components rather than deriving one by subtraction in cm.

## 21. Verify MetricGrounding against the new CSV export  ▶ (hand-check the numbers before anyone publishes them)

**Goal:** independently confirm that every grounded number now leaving LM3 in
`reports/Data/*.csv` is correct. The export made the pipeline's measurements easy to consume, which
means a units or frame error that used to sit unnoticed in SQLite now flows straight into someone's
analysis. Nothing below has been validated against physical ground truth yet.

**Starting artifact:** `examples_out/csv_export_verify/` — a full 22-sheet run (148 leaves, 132 of
them grounded) written with the current code. Its `reports/Data/` is the thing to audit.

**Checks, roughly in order of how badly each would hurt:**

1. **Is the CF itself right?** Everything downstream is a multiple of it. Take a handful of sheets
   from `ruler_conversion_factor.csv` with `status=published, confidence=high`, and measure the
   ruler by hand in the working image: does `cf_px_per_cm` match px-per-cm on the actual ruler?
   Check at least one of each `ruler_unit_type` (`METRIC_MM`, `METRIC_MM_CM`, imperial), since a
   unit-naming error is the failure mode the lattice gate exists to prevent.
2. **Exponents.** Areas must divide by `cf²` and lengths by `cf¹`. Cheap check straight off a CSV
   row: `lamina_area_incl_holes_px / cf² == lamina_area_incl_holes_cm2` and
   `bbox_w_px / cf == bbox_w_cm`, for both a small and a large leaf.
3. **Frame.** The CF and every `_px` value are WORKING-frame. On a sheet with `work_scale < 1`
   (`downsampled = 1`), confirm the cm values are NOT off by `work_scale` or `work_scale²` — i.e.
   nothing applied the scale twice, and nothing grounded against the original dimensions. Compare a
   downsampled sheet's leaf against a same-species non-downsampled one; cm areas should be
   comparable, px areas should not.
4. **The new landmark cm columns** (`lamina_trace_length_cm`, `lamina_extent_cm`,
   `lamina_tip_base_length_cm`, `landmark_leaf_width_cm`, `petiole_trace_length_cm`) are the least
   exercised path — they only started being written with this change. Verify against a ruler on a
   real leaf, and check `lamina_trace_length_cm >= lamina_extent_cm` still holds (arc >= chord).
5. **Absences are real.** Every `_cm` column must be empty exactly when `cf_source = none`, and
   never 0. Asserted on synthetic data in `test_data_export.py`; confirm it on a real run, and
   confirm the 16 ungrounded leaves in the verify run are ungrounded because their sheet's CF was
   genuinely withheld, not because a join dropped them.
6. **No silent MP fallback.** `cf_px_per_cm_predicted_by_mp` is carried in the CSV but must never
   have produced a `_cm` value. Confirm no grounded row exists whose `cf_px_per_cm_ruler` is empty.
7. **CF-free values still make sense.** `leaf_mass_per_area` needs no CF, so it is present even on
   ungrounded rows — check the units really are g/m² and the values are botanically plausible
   (rough range ~20–200 g/m² for most leaves).
8. **Roll-ups.** `specimen_summary.csv` medians and counts are computed from the same rows as
   `leaf_measurements.csv`; spot-check a sheet by hand against its leaf rows.

**Outcome:** either a clean bill of health recorded here, or bugs filed. Worth doing before the
Nature-dataset numbers (§15/§16) are computed from these columns.

## 22. Non-standard leaves (needles, twig-like): reuse the specimen segmenter  ▶ (`use_non_std_leaf_segmenter` toggle — no retraining)

**Idea:** for needle-leaved and twig-like specimens the instance `leaf_segmenter` (YOLO26x-seg,
trained on broad laminas) under-segments badly, but the **SpecimenSegmenter's** whole-sheet
plant-vs-background mask already captures those thin structures well. So offer a manual toggle that
swaps the leaf source: instead of running instance segmentation on each `Leaf_WHOLE` crop, intersect
the existing whole-sheet specimen mask with each leaf box. **Requires no retraining** — both models
already ship.

**Setting:** `modules.leaf_segmenter.use_non_std_leaf_segmenter` (default `false`). When on,
`LeafSegmenter.depends_on` must also include `"specimen_segmenter"`; it still owns
`leaf_segmentation`, so restart/resume semantics are unchanged.

**How:** for each `Leaf_WHOLE` box, crop the `specimen_mask` PNG to the box, contour it, and emit
`leaf_segmentation` rows in parent/working coords exactly as today. The rest of the pipeline is
untouched because it reads the table, not the model.

**The things that will actually bite:**
- **Only ONE class comes out.** The specimen mask is a single binary foreground — there is no
  `Petiole` and no `Hole` in it. So `PetioleWidth` produces nothing, `n_holes` is always 0,
  `lamina_area_excl_holes_px == incl`, and the `LaminaPetiole_*` leaf products are skipped. Decide
  whether that is acceptable for these taxa or whether Petiole/Hole should still come from YOLO26
  (a hybrid: specimen mask for the lamina, instance model for the parts).
- **Conflicts with the largest-connected-component rule (§19).** A needle cluster inside one leaf
  box is legitimately many disconnected components; "Leaf = largest CC" would keep a single needle
  and throw the rest away. This mode needs the opposite rule — keep all components as one instance
  (`num_parts > 1`) — so the reduction has to become mode-aware, not global.
- **Downstream assumptions.** Morphology's rotated bbox, the 31-keypoint landmark pose, the midvein
  trace and bilateral symmetry all assume one broad lamina with a midvein. On needles those are at
  best meaningless and at worst confidently wrong. Consider auto-disabling landmark/bilateral
  stages in this mode, or at least flagging the rows so the CSV export does not present junk
  measurements as real ones.
- **Box quality still gates it.** This changes segmentation, not detection — if `plant_detector`
  does not box the needles in the first place, nothing downstream sees them.

**Test:** pick a conifer / needle-leaved and a twig-like sheet set, run both modes, and compare the
`Leaf_Oriented` masks and `reports/Data/leaf_measurements.csv` side by side.

## 23. User-supplied ruler / unit restrictions — bypass the herbarium-sheet CF anchor  ▶ (large; regression risk)

**Need:** a user who shoots their own images typically uses **one known ruler throughout**. Let
them declare it once, and have the CF stage trust that declaration instead of inferring the unit
type per crop and anchoring to a herbarium-sheet assumption.

**What the pipeline currently assumes, and why it is wrong for this user:**
- `MPConversionFactor` predicts `cf_px_per_cm = 2.6745*MP + 67.52`, a regression **fit on herbarium
  sheets** (708 rows, MP range 14.6–36.2). For copy-stand images at an arbitrary working distance
  that number is simply not about their setup.
- `ruler_lattice/sheet_cf.reconcile_parent` then uses that value as **"the absolute reference"** —
  `anchor_tol=0.25` admissibility gating, a closeness tiebreak between clusters, and harmonic
  disambiguation. A wrong anchor therefore does not just fail to help, it actively steers the
  answer.
- `MIN_FRAME_CM = 20.0` rejects an anchor implying a sheet narrower than 20 cm — another
  sheet-shaped assumption.
- Unit naming comes from the `RulerClassifier` ensemble via `CLASS_SPEC` / `admissible_units`, voted
  per crop, rather than from anything the user knows for certain.

**Three levels, increasing in power — decide which to build:**
1. **Unit restriction only.** User pins the unit system/type (e.g. "metric, mm+cm ruler"), which
   constrains `admissible_units` and kills whole classes of unit-misnaming error. Smallest change,
   keeps every existing safeguard.
2. **Anchor override / disable.** User supplies their own anchor CF, or declares "no anchor" so
   reconciliation falls back to peer clustering alone. Needs care: the anchor exists precisely
   because peer agreement was proven insufficient (see the note at the top of `sheet_cf.py` — a
   whole sheet of crops agreed on a mis-scored lattice and only the anchor caught it).
3. **Fixed known CF.** User measured px/cm once and applies it to the whole project, bypassing
   ruler detection entirely. Trivial to apply, but it must still be recorded with provenance so
   `cf_source` in `reports/Data` says "user_supplied" rather than silently reading as a measurement.

**Non-negotiables:** every path must write honest provenance into `ruler_CF_lattice`
(`cf_source`, `method`, `confidence_reasons_json`) and into the CSV export's `cf_source`, so a
downstream reader can always tell a declared CF from a measured one. And the default path must be
**bit-identical** to today when the feature is unused.

**Why this is flagged large:** it touches the one number every physical measurement in the project
is a multiple of. Needs a regression corpus — the existing herbarium example runs, re-run with the
feature off, compared column-for-column against a stored baseline of `reports/Data/*.csv` — before
any of it ships. Pairs with §21 (verify MetricGrounding), which should be done first so there is a
trusted baseline to regress against.

## 24. YOLO labeler built into the LM3 GUI  ▶ (two modes: revise LM3 output, and annotate training data)

**Need:** an annotation tool inside the LM3 desktop/web GUI, serving two distinct purposes.

**Mode A — revise LM3 output (edits the project DB).** Load a run, draw/adjust/delete boxes and
masks, and write the corrections back to `plant_detection` / `archival_detection` /
`leaf_segmentation`. Then re-run the Reporter so overlays, leaf products, masks and
`reports/Data/*.csv` reflect the corrected geometry.

**Mode B — annotate training data.** Same canvas, but exporting YOLO-format labels for finetuning
or training new model versions, round-tripping with the dataset builders in the `LM3_*` training
areas.

**The hard part is not the canvas — it is making a human edit survive the pipeline:**
- **Re-runs delete edits.** `record_detections` and `record_leaf_instances` are delete-then-insert
  per specimen. Today, re-running `plant_detector` or `leaf_segmenter` would silently wipe every
  correction. Needs an edit provenance/lock concept — e.g. an `edited_by_user` column plus a rule
  that a stage never overwrites a locked row — designed before any UI work starts.
- **An edit must invalidate what was derived FROM it, not just the pictures.** Moving a leaf mask
  changes morphology, landmarks, petiole width, bilateral symmetry and ECT for that leaf. Re-running
  only the Reporter would produce the worst possible result: corrected overlays drawn over stale
  measurements, and a `leaf_measurements.csv` that looks freshly exported while carrying numbers
  from the old mask. There needs to be a per-leaf (or per-specimen) invalidation cascade mirroring
  what `_sync_settings_hash` does for config drift — config drift resets the transitive closure of
  dependent stages; a data edit has to do the same, and currently has no path to.
- **Stage gating.** `is_complete()` skips a stage whose ledger says done, so the invalidation has to
  go through `project_status` / `image_status`, not just mark a row dirty.
- **Coordinate frame.** Everything stored is WORKING frame (`ingest.max_working_dim`). The labeler
  must edit in that frame, or convert explicitly — an edit made against original-resolution pixels
  would be off by `1/work_scale`.
- **Mask representation.** `leaf_segmentation.mask_data` is polygon rings (`polygon_xy`) or COCO
  RLE, with `parent_instance_index` linking Hole/Petiole to their owning Leaf. A mask editor has to
  preserve that parent link and the class trio, not just push pixels.

**Where it goes:** the existing FastAPI + Electron GUI (`leafmachine3/server/`), alongside the
results browser that already lists and serves these artifacts.

**Suggested order:** Mode B first (export-only — it writes no DB rows, so it cannot corrupt a run
and gets the canvas built), then the edit-provenance and invalidation design, then Mode A.
