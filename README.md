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
LandmarkMeasurements → LeafOrientation → PetioleWidth → MetricGrounding (no-op until CF) → Reporter`

**LandmarkDetector** (runs on leaf crops) predicts the 31-keypoint `mid15_pet5` pose skeleton
(lamina tip/base, apex/base triples, midvein×15, petiole×5+tip, width×2) with the yolo26x-pose
model. The model is trained on crops with a 10% white border, so the inference wrapper re-adds
that border and maps keypoints back (coordinates as if never padded). Keypoints are stored in
`leaf_landmark` in **working (parent) coords** (plus crop coords), linked to the leaf crop; the
keypoint names + skeleton relationships are seeded, self-describing, into `landmark_schema` /
`landmark_skeleton` (from `core/landmarks.py`).

**LandmarkMeasurements** (post-process, CPU; runs after LandmarkDetector) derives per-leaf
biology from the keypoints into the `leaf_landmark_measurement` table: `lamina_trace_length`
(summed distance along the 15 midvein trace points), `lamina_extent` (straight chord between the
first and last midvein point, drawn as the white overlay line), `lamina_curvature` (the **max
midvein bend** in degrees — for each interior midvein point, the angle its two arms make to the
midvein ends; `180 − smallest` such angle, so 0° = straight and larger = more curved, with
`curvature_point` marking that vertex, drawn as the black bend lines beneath the cyan/white),
`lamina_tip_base_length`
(the separate
`lamina_tip`→`lamina_base` distance), `leaf_width` (width_left→width_right),
`apex_angle`/`base_angle` with `_type` in `{acute, obtuse, reflex}`, `petiole_trace_length`, and
`lamina_curvature` (arc/extent). It is **occlusion-robust**: keypoints below
`landmark_measurements.min_kpt_conf` are treated as absent and every metric that needs a missing
point returns `NULL` — nothing is fabricated. The angle/type convention (reflex when both arms
point toward the lamina centroid, matching LeafMachine2 `determine_reflex`) is documented and
rendered as a live schematic in
[`modules/experiments/angle_checks.html`](leafmachine3/modules/experiments/angle_checks.html).
See `core/landmark_metrics.py`. (Orientation-aware length/width assignment is still future —
TODO #1a.)

**LeafOrientation** (post-process, CPU) computes the clockwise rotation that stands each leaf
tip-up / base-down from the keypoints — primary: the `lamina_tip`→`lamina_base` axis; fallback:
PCA of the midvein (≥ `min_midvein` points) with the tip end chosen from the petiole / base / apex
points; else no orientation. The angle + a success flag are stored on the leaf's `leaf_morphology`
row (`oriented_leaf_rotation_angle_degreesCW`, `oriented_leaf_success`). See `core/orientation.py`.

**PetioleWidth** (post-process, CPU) measures each leaf's petiole width from its `Petiole` mask and
the landmark petiole centerline (`lamina_base` → `petiole_0..4` → `petiole_tip`): the **median** of
several perpendicular thickness samples near the blade junction (`core/petiole.py` → `leaf_petiole`
table, with `width_px`, `length_px`, `touches_leaf`, and the sample/width segments). Widths are
pixels until the ruler CF lands. It runs on the raw petiole mask now; TODO #5 will feed it the
edge-refined mask. The Reporter draws the width as a blue band on the summary and a per-leaf
`Overlay/Overlay_Petiole/` — left: masks + sample probes + width band + a lamina-area/petiole-width
panel; right: the petiole in full color (optional configurable background tint, off by default)
blown up (pixelated) with a 1-px blue line at the exact width.

**Leaf products** (Reporter) are the highest-value output: five per-leaf products, each in a
non-oriented **`Original/`** and an upright **`Oriented/`** tree — the Plant_Detector bbox crop,
the lamina mask, the lamina+petiole mask, the **lamina-holes mask** (solid silhouette with holes
filled), the lamina RGB cutout, the lamina+petiole RGB cutout, and the **lamina-holes RGB cutout**
(leaf tissue kept, holes painted `report.leaf_products.hole_rgb_color` = `(10,10,10)` so they're
recoverable by color threshold). Everything except the bbox crop is cropped tight to its mask
("fitted"); cutouts/rotated corners use the `report.leaf_products.background`. Oriented products are
emitted only where LeafOrientation
succeeded; lamina+petiole products are skipped for leaves without a petiole mask. Leaf bbox crops
live here (not in `Crops/`, which now holds only non-leaf classes).

**Morphology** (runs after LeafSegmenter) computes LeafMachine2-style shape metrics per leaf
mask — area, perimeter, centroid, convex hull, convexity/concavity, circularity, aspect ratio —
plus hole-aware lamina areas (`area_px` is the outer Leaf boundary so it already **includes** holes;
`lamina_area_incl_holes_px` = that, `lamina_area_excl_holes_px` = tissue with holes removed,
`lamina_hole_area_px` = Σ hole areas, `n_holes` = hole count) and the rotated (minimum) bounding
box (rotation angle + `rotated_bbox_dim_max`/`dim_min` =
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

## Setup with Verification Steps

Use this for a fresh setup, or to repair a setup that stopped working. Each step checks itself, so
stop at the first one that does not end the way its comment says. The environment comes from
`uv.lock`, so every install gets exactly the package versions this release was tested with.

### 1. Install uv (once per machine)

```bash
# Linux or macOS
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh                        # installs the pinned uv
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.23/install.ps1 | iex"
```

Open a new terminal, then:

```bash
uv --version                       # must print 0.12.23
```

### 2. Get the code

```bash
git clone https://github.com/Gene-Weaver/LeafMachine3.git   # fresh setup
cd LeafMachine3
```

Already have a checkout? Update it instead:

```bash
git checkout -- LM3_settings.yaml  # restores the shipped default settings (copy your edits to LM3_settings.local.yaml first)
git pull --ff-only                 # brings the code and the lock up to date
```

### 3. Build the environment

```bash
uv sync --frozen --extra gpu       # NVIDIA GPU on Linux or Windows; downloads Python 3.11.17 and the locked packages
# uv sync --frozen --extra cpu     # no GPU
# uv sync --frozen --extra macos   # Apple Silicon Mac
uv run lm3 doctor                  # must end with "Result: READY"; otherwise it prints the command that fixes the problem
```

### 4. Install the models

```bash
uv run hf auth login               # only while the model repositories are private: paste a token with access
uv run lm3 models install --yes    # downloads the pinned default models (about 1.7 GB) into models/
uv run lm3 doctor --models         # check 7 should report that every ONNX model runs entirely on the GPU
```

### 5. Run the example

```bash
uv run machine3 --config LM3_settings.yaml   # processes examples/images into runs/demo/; should print no WARNING lines
```

The first run on a new machine tunes a hardware profile before it starts, which adds a few minutes.

### 6. Open the GUI

```bash
uv run lm3 serve                   # then open http://127.0.0.1:8765; Ctrl+C stops the server
```

### Optional: confirm a rebuild did not change results

Keep `runs/demo/` from before the rebuild, then run the example again under a new name and compare:

```bash
uv run machine3 --config LM3_settings.yaml --run-name demo_check
uv run python tools/verification/compare_run_databases.py runs/demo/demo.sqlite runs/demo_check/demo_check.sqlite
                                   # every table prints OK and the exit status is 0 when results are identical
```

### Repairing a broken setup

`machine3` and `lm3 serve` run the first four doctor checks at startup and refuse to start (exit
code 78) when one fails, so a damaged environment shows up immediately.

```bash
uv run lm3 doctor                              # names the first problem and prints the command that fixes it
uv sync --frozen --extra gpu --reinstall       # if unsure: reinstalls every package from the lock
uv run lm3 models verify                       # re-hashes every installed model against the lock
uv run lm3 models install --force --yes        # re-downloads the models if verify reports a mismatch
```

If none of that helps, delete the `.venv` folder and repeat step 3.

## Install

> The uv route in [Setup with Verification Steps](#setup-with-verification-steps) is the supported one. A
> pip-built environment does not match `uv.lock`, so `machine3` and `lm3 serve` refuse to start in it
> (exit 78) unless `LM3_STARTUP_GATE=0` is set. The pip instructions below are kept for reference.

Pinned, reproducible (recommended) — see **[INSTALL.md](INSTALL.md)** for the full guide (extras,
poetry). The runtime needs **no torch and no ultralytics**: every model is an exported end2end ONNX
graph driven by onnxruntime (`leafmachine3/inference/ultra_replacements.py`).

```bash
python -m venv .venv_LM3 && .venv_LM3/bin/pip install -U pip wheel setuptools
.venv_LM3/bin/pip install -r requirements/requirements-gpu.txt     # NVIDIA GPU (default)
.venv_LM3/bin/pip install -r requirements/requirements-cpu.txt     # CPU only
.venv_LM3/bin/pip install -r requirements/requirements-macos.txt   # macOS (MPS + CoreML)
```

Or the flexible extras: `pip install -e ".[gpu]"` (`cpu`/`macos` variants too). The GPU extra pulls the
CUDA 12 runtime libraries as `nvidia-*` wheels; the only system requirement is an NVIDIA driver
≥ 525.60.13 (Linux) / 528.33 (Windows). `onnxruntime-gpu` must stay < 1.21 (1.28+ is a CUDA-13 build).

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
    Overlay_Summary/             <stem>__Overlay.jpg                   (masks + boxes + landmarks + petiole bands)
    Overlay_Landmarks/           <stem>__LM-leaf__x_y_x_y.jpg          (per leaf: keypoints + measures)
    Overlay_Petiole/             <stem>__PET-leaf__x_y_x_y.jpg         (per leaf: petiole width)
    Overlay_Specimen_Segmentation/ <stem>__SpecimenSeg.jpg            (annotated sheet | cutout)
    Overlay_Ruler_Lattice/       <stem>__RulerLattice.png              (ruler-CF QC panel)
  Leaf_Original/                 (7 leaf products, non-oriented)
    Leaf_BBox/                   <stem>__og-BBOX-leaf__x_y_x_y.jpg        (not fitted)
    Lamina_Mask/                 <stem>__og-SEG-lamina__x_y_x_y.png       (fitted; holes removed)
    LaminaPetiole_Mask/          <stem>__og-SEG-laminaPetiole__x_y_x_y.png
    Lamina_Holes_Mask/           <stem>__og-SEG-laminaHoles__x_y_x_y.png  (solid silhouette)
    Lamina_RGB/                  <stem>__og-RGB-lamina__x_y_x_y.jpg
    LaminaPetiole_RGB/           <stem>__og-RGB-laminaPetiole__x_y_x_y.jpg
    Lamina_Holes_RGB/            <stem>__og-RGB-laminaHoles__x_y_x_y.jpg  (holes = (10,10,10))
  Leaf_Oriented/                 (same 7 products rotated tip-up, tagged `or-` instead of `og-`)
  Crops/RGB__<friendly>/         <stem>__BBOX-<friendly>__x_y_x_y.jpg   (NON-leaf classes)
  Specimen_Masks/                (every mask export shares this one parent)
    Binary_Masks_Specimen__Leaf/    <stem>__MaskFull-leaf.png           (per specimen)
    RGB_Masks_Specimen__Leaf/       <stem>__MaskRGBFull-leaf.jpg        (per specimen)
    Binary_Masks__Leaf/             <stem>__SEG-leaf__x_y_x_y.png       (per leaf crop)
    RGB_Masks__Leaf/                <stem>__SEGRGB-leaf__x_y_x_y.jpg    (per leaf crop)
    Binary_Masks_Specimen/          <stem>__MaskFull-specimen.png       (whole specimen)
    RGB_Masks_Specimen/             <stem>__MaskRGBFull-specimen.jpg
    Binary_Masks_Specimen_Inverse/  <stem>__MaskFull-specimenInverse.png     (opt-in)
    RGB_Masks_Specimen_Inverse/     <stem>__MaskRGBFull-specimenInverse.jpg  (opt-in)
  Leaf_Data/
    Bilateral_Symmetry/          <stem>__BSYM-leaf__x_y_x_y.jpg
    Coordinates/                 <stem>__ECT__x_y_x_y.h5
    Oriented_Leaf_ECT/           <stem>__ECT__x_y_x_y.png                    (Cartesian)
    Oriented_Leaf_Radial_ECT/    <stem>__ECT-radial__x_y_x_y.png             (polar)
    Oriented_Leaf_Radial_ECT_Overlay/ <stem>__ECT-radial-overlay__x_y_x_y.png
  Data/                          (the NUMBERS behind every image above -- see below)
    leaf_measurements.csv        ONE ROW PER LEAF: every measurement + its identifying metadata
    specimen_summary.csv         one row per input image, with per-sheet roll-ups
    phenology.csv                one row per sheet, in LeafMachine2's phenology.csv layout
    detections.csv               one row per detection box, both detectors
    landmarks.csv                one row per predicted keypoint (31 per leaf)
    ruler_conversion_factor.csv  one row per sheet: the CF verdict and why
    ruler_crops.csv              one row per candidate ruler crop
    run_stages.csv               one row per pipeline stage
    stage_errors.csv             one row per per-image failure
    data_dictionary.csv          every column above: file, units, meaning
```

### `reports/Data/` — the results as CSV

The Reporter's last step writes the project database out as CSV so a run is analyzable without
opening SQLite. **`leaf_measurements.csv` is the one to start with**: one row per segmented leaf,
carrying every morphology, landmark, petiole and symmetry measurement together with the specimen
and leaf identity needed to trace it back. `crop_file_token` on each row is the filename token of
that leaf's exported images, so a row joins to its pictures by string match; `leaf_uid` adds the
instance index and is stable across re-runs (`leaf_id` is not — it is a project-local row id).

Two things to know before analyzing:

* **Pixels are WORKING-frame.** Every `_px` value is measured on the resized copy the stages
  analyzed. `work_scale` is on every row: original-frame pixels are `value_px / work_scale`.
* **`_cm` columns are empty unless a ruler CF was published.** LM3 grounds only against
  `specimen.cf_px_per_cm`, which the lattice stage publishes for high-confidence sheets only, and
  never against the megapixel estimate. `cf_source` says which case a row is in, and an empty cell
  always means "not measured", never zero.

**`phenology.csv` is LeafMachine2-compatible.** LM2 wrote `Phenology/phenology.csv` by counting
class ids in the plant detector's YOLO label files; LM3 has no label files, so the same file is
rebuilt from `plant_detection`. Its first 14 columns are LM2's, in LM2's order, so an existing LM2
phenology script reads it unchanged; LM3-native columns are appended after them. Two LM2 columns
are **always blank**: `leaflet` and `specimen`, because LM3's detector is trained on
`LM3_Plant_Primary`, which dropped `Specimen` and merged `Leaflet` into `Leaf_WHOLE` — so LM3's
`leaf_whole` also counts what LM2 called `leaflet`. `has_leaves` and `is_fertile` come from the
phenology stage's thresholds rather than LM2's `accept_only_ideal_leaves` /
`minimum_total_reproductive_counts`, and LM3 groups `Bud` with flowers where LM2's `is_fertile`
excluded it.

Toggle the bundle and its individual files under `report.data` in `LM3_settings.yaml`; set
`report.data.format: tsv` for tab-separated output.

Files are named so they can be reinserted into the parent by filename:
`<stem>__<ID>__x_y_x_y.<ext>`, where `ID` is `<PREFIX>-<friendly>` — `BBOX` (detection box),
`SEG` (per-crop mask), `SEGRGB` (its RGB cutout twin), `MaskFull` (full-image mask, no coords),
`MaskRGBFull` (its cutout twin), `LM` (per-leaf landmark overlay), `PET`, or `BSYM`. Class →
friendly-name mapping (e.g. `Leaf_WHOLE → leaf`, `Leaf_PARTIAL → leafReject`) and the prefixes
live in `naming` — edit freely. Every output folder toggles independently in `report`.

**Every `ID` is unique across the whole run with the extension stripped**, so the entire
`reports/` tree can be flattened into one directory without a single file overwriting another.
Two consequences worth knowing: a binary mask and its RGB cutout get different prefixes rather
than relying on `.png` vs `.jpg`, and the leaf products are tagged by tree (`og-` for
`Leaf_Original`, `or-` for `Leaf_Oriented`) because the two render the same leaf at the same box.
The one deliberate exception is `Leaf_Data/Coordinates/*.h5`, which shares the Cartesian ECT
image's `ECT` token — it is data rather than a picture, and its extension separates them.

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
