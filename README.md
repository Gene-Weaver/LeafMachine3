# LeafMachine3

Herbarium specimen image analysis pipeline — the **runtime** package. LeafMachine3 ships
trained, **exported** models (ONNX / OpenVINO) plus a small, sequential manager
(`machine3.py`) that mirrors LeafMachine2's `machine.py`. Each stage saturates all available
hardware internally; the pipeline is resumable and driven by one YAML settings file.

> Status: **early build**. Everything up to segmentation works end-to-end. The
> pixel→metric conversion (`ruler_cf` → `metric_grounding`) is not yet wired — `RulerClassifier`
> assigns a ruler unit type, but the conversion factor step is a disabled stub.

## Pipeline

`ingest → ArchivalDetector → PlantDetector → PhenologyDetector → RulerClassifier →
[RulerConversionFactor — stub] → LeafSegmenter → Morphology → LandmarkDetector →
LandmarkMeasurements → MetricGrounding (no-op until CF) → Reporter`

**LandmarkDetector** (runs on leaf crops) predicts the 31-keypoint `mid15_pet5` pose skeleton
(lamina tip/base, apex/base triples, midvein×15, petiole×5+tip, width×2) with the yolo26x-pose
model. The model is trained on crops with a 10% white border, so the inference wrapper re-adds
that border and maps keypoints back (coordinates as if never padded). Keypoints are stored in
`leaf_landmark` in **working (parent) coords** (plus crop coords), linked to the leaf crop; the
keypoint names + skeleton relationships are seeded, self-describing, into `landmark_schema` /
`landmark_skeleton` (from `core/landmarks.py`).

**LandmarkMeasurements** (post-process, CPU; runs after LandmarkDetector) derives per-leaf
biology from the keypoints into the `leaf_landmark_measurement` table: `lamina_trace_length`
(summed distance along the 15 midvein trace points), `lamina_extent` (straight tip→base),
`leaf_width` (width_left→width_right),
`apex_angle`/`base_angle` with `_type` in `{acute, obtuse, reflex}`, `petiole_trace_length`, and
`lamina_curvature` (arc/extent). It is **occlusion-robust**: keypoints below
`landmark_measurements.min_kpt_conf` are treated as absent and every metric that needs a missing
point returns `NULL` — nothing is fabricated. The angle/type convention (reflex when both arms
point toward the lamina centroid, matching LeafMachine2 `determine_reflex`) is documented and
rendered as a live schematic in
[`modules/experiments/angle_checks.html`](leafmachine3/modules/experiments/angle_checks.html).
See `core/landmark_metrics.py`. (Orientation-aware length/width assignment is still future —
TODO #1a.)

**Morphology** (runs after LeafSegmenter) computes LeafMachine2-style shape metrics per leaf
mask — area, perimeter, centroid, convex hull, convexity/concavity, circularity, aspect ratio —
plus the rotated (minimum) bounding box (rotation angle + `rotated_bbox_dim_max`/`dim_min` =
leaf length/width), stored in the `leaf_morphology` table. The rotated-bbox algorithm is
selectable via `modules.morphology.method`: **`pca`** (area-weighted principal axis — the
default; robust across the broadest range of leaf shapes), `feret` (max-Feret axis + hull
extents, best on clearly elongated leaves), `lm2` (LM2's `fit_min_bbox`), or `minarearect`
(OpenCV `cv2.minAreaRect`; minimizes area, can mis-orient). See
`tests/rotated_bbox_comparision/` for a visual comparison. The
Reporter overlay can draw leaf boxes as the rotated min box (`report.overlay.box_style: rotated`,
default) or the axis-aligned YOLO box (`yolo`).

## Two control surfaces + a hardware profile

- **`LM3_settings.yaml`** — user intent: paths, models, quality knobs, and **all overlay
  colors + display flags** (`report.overlay`). Edit colors/visibility freely; defaults are the
  LeafMachine2 palette.
- **`hardware_settings.yaml`** — machine-tuned performance (workers, batch, tmp, VRAM). Written
  automatically by **LM3_Setup** on first run; rerun with `lm3-setup --force`.
- **project SQLite DB** (`project_status`) — progress / resume.

## Install

```bash
pip install -e ".[gpu,yolo]"      # NVIDIA box (onnxruntime-gpu + ultralytics)
pip install -e ".[cpu,yolo]"      # CPU / DirectML / CoreML box
```

Models are **not** committed. Place exported artifacts under `models/<stage>/` (or symlink the
training-repo exports); paths are set in `LM3_settings.yaml`.

## Run

```bash
machine3 --config LM3_settings.yaml
machine3 --config LM3_settings.yaml --restart leaf_segmenter   # rerun a stage + dependents
lm3-setup --optimize                                           # (re)profile the machine
```

Set `compute.mock: true` to run the whole pipeline with deterministic synthetic models (no
weights or GPU) — used by the test suite to exercise ingest → detect → segment → report.

## Reporter outputs & crop naming

Every output is a folder under `<run>/reports/`, each toggleable in `report` of
`LM3_settings.yaml`:

Each category is a folder with per-output subfolders (harmonized with `Crops/`):

```
reports/
  Overlay/
    Overlay_Summary/             <stem>__Overlay.jpg                   (masks + boxes + landmarks)
    Overlay_Landmarks/           <stem>__LM-leaf__x_y_x_y.jpg          (per leaf: keypoints + measures)
  Crops/RGB__<friendly>/         <stem>__BBOX-<friendly>__x_y_x_y.jpg   (both detectors)
  Binary_Masks/
    Binary_Masks_Full_Image__Leaf/  <stem>__MaskFull-leaf.png          (per specimen)
    Binary_Masks__Leaf/             <stem>__SEG-leaf__x_y_x_y.png       (per leaf crop)
  RGB_Masks/
    RGB_Masks_Full_Image__Leaf/     <stem>__MaskFull-leaf.jpg           (per specimen)
    RGB_Masks__Leaf/                <stem>__SEG-leaf__x_y_x_y.jpg       (per leaf crop)
```

Files are named so they can be reinserted into the parent by filename:
`<stem>__<PREFIX>-<friendly>__x_y_x_y.<ext>`, where `PREFIX` is `BBOX` (detection box),
`SEG` (per-crop mask), `MaskFull` (full-image mask, no coords), or `LM` (per-leaf landmark
overlay). Class → friendly-name
mapping (e.g. `Leaf_WHOLE → leaf`, `Leaf_PARTIAL → leafReject`) and the prefixes live in
`naming` — edit freely. Every output folder toggles independently in `report`.

## Layout

```
leafmachine3/
  machine3.py            pipeline manager
  pipeline.py            STAGE_ORDER + run loop
  core/                  config, db, schema.sql, stage, executor, ingest, project, dirs,
                         landmarks, landmark_metrics, morphometrics, ...
  modules/               the pipeline stages (+ experiments/ scratch schematics)
  inference/             portable exported-model wrappers + mock backends + EP selection
  reporting/             palette.py (config-driven, LM2 colors) + overlay.py (Summary_Image)
  setup/                 LM3_Setup hardware profiler
  server/                optional FastAPI server (lm3 serve)
```

## License

MIT — see [LICENSE](LICENSE).
