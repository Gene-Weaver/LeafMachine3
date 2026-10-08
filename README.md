# LeafMachine3

**Automated trait extraction from herbarium specimen images.** LeafMachine3 is a modular suite of
machine learning and computer vision tools that locates, isolates, and measures the plant and
archival components of a digitized specimen: every leaf is segmented, landmarked, oriented, and
measured; every ruler is read for a pixel-to-metric conversion factor; every label, barcode, and
color card is cropped for downstream work. Each run produces annotated overlays, per-leaf cutouts, and a CSV with one row per leaf.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](pyproject.toml)
[![Models on Hugging Face](https://img.shields.io/badge/models-Hugging%20Face-yellow.svg)](https://huggingface.co/phyloforfun)
[![Website](https://img.shields.io/badge/web-leafmachine.org-green.svg)](https://leafmachine.org)

![Three herbarium sheets before and after LeafMachine3: the originals above, the summary overlays below](docs/readme_github/hero_before_after.jpg)

*Three specimens as imaged (top) and the LeafMachine3 summary overlay for each (bottom): leaf
masks, rotated bounding boxes, landmarks, petiole widths, organ and archival detections. All demo
images in this README come from these three sheets.*

---

## Table of Contents

- [About LeafMachine3](#about-leafmachine3)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Install](#install)
  - [Verify](#verify)
  - [Troubleshooting](#troubleshooting)
  - [Docker](#docker)
  - [Updating](#updating)
- [Quick Start](#quick-start)
  - [Run from the GUI](#run-from-the-gui)
  - [Electron desktop app](#electron-desktop-app)
  - [Run from the command line](#run-from-the-command-line)
  - [Where results land](#where-results-land)
- [Workflow](#workflow)
- [In-Depth: the processing modules](#in-depth-the-processing-modules)
  1. [MP Conversion Factor](#1-mp-conversion-factor)
  2. [Archival Detector](#2-archival-detector)
  3. [Plant Detector](#3-plant-detector)
  4. [Specimen Segmenter](#4-specimen-segmenter)
  5. [Phenology Detector](#5-phenology-detector)
  6. [Ruler Classifier](#6-ruler-classifier)
  7. [Ruler Conversion Factor](#7-ruler-conversion-factor)
  8. [Leaf Segmenter](#8-leaf-segmenter)
  9. [Morphology](#9-morphology)
  10. [Landmark Detector](#10-landmark-detector)
  11. [Landmark Measurements](#11-landmark-measurements)
  12. [Leaf Orientation](#12-leaf-orientation)
  13. [Petiole Width](#13-petiole-width)
  14. [Bilateral Symmetry](#14-bilateral-symmetry)
  15. [Metric Grounding](#15-metric-grounding)
  16. [Reporter](#16-reporter)
  17. [Shape (ECT)](#17-shape-ect)
- [Outputs](#outputs)
- [Postprocessing Tools](#postprocessing-tools)
- [Models](#models)
- [Related Projects and Citation](#related-projects-and-citation)
- [License](#license)

---

## About LeafMachine3

LeafMachine3 is the successor to [LeafMachine2](https://github.com/Gene-Weaver/LeafMachine2)
(Weaver & Smith, 2023). It keeps the design that made LeafMachine2 work, breaking a complex
problem into discrete steps the way a person would when extracting data from a specimen, and
rebuilds every step on current models and a runtime that is simpler to install and run.

- **Input:** a folder of herbarium specimen images (JPEG, PNG, TIFF). Field images and other
  digital plant data sets work too; archival detections are simply empty.
- **Output:** a per-specimen summary overlay for quality control, cropped components, per-leaf
  masks and cutouts (as mounted and oriented tip-up), and CSV tables with one row per leaf, per
  specimen, per detection, and per landmark.
- **Pipeline:** seventeen modules run in a fixed order, each one feeding the next. The first two
  detectors place bounding boxes around plant and archival components; everything downstream
  works on those crops. Every module can be switched off individually.

```
MP Conversion Factor → Archival Detector → Plant Detector → Specimen Segmenter → Phenology Detector
→ Ruler Classifier → Ruler Conversion Factor → Leaf Segmenter → Morphology → Landmark Detector
→ Landmark Measurements → Leaf Orientation → Petiole Width → Bilateral Symmetry
→ Metric Grounding → Reporter → Shape (ECT)
```

**What changed from LeafMachine2**

| | LeafMachine2 | LeafMachine3 |
|---|---|---|
| Runtime | PyTorch + Detectron2 + Ultralytics | onnxruntime only; every model is an exported ONNX graph (no torch) |
| Install | conda + CUDA toolkit | `uv sync` from a lock file; CUDA libraries arrive as wheels |
| Models | bundled downloads | versioned on [Hugging Face](https://huggingface.co/phyloforfun), installed with one command |
| Interface | command line + YAML | the same YAML, plus a desktop GUI with live progress and a results browser |
| Leaf landmarks | 9 pseudo-landmarks as fixed boxes | a 31-keypoint pose skeleton (midvein trace, petiole trace, apex, base, width) |
| Leaf segmentation | Mask R-CNN + PointRend | YOLO26 instance segmentation of lamina, petiole, and holes |
| Rulers | 3-network binarization + scanline | unit-type ensemble + a tick-lattice solver, anchored by a resolution prior |
| Resume | per-batch | per-stage ledger in SQLite; rerun any stage and its dependents |

---

## Installation

### Requirements

| | |
|---|---|
| OS | Linux (tested on Ubuntu 22.04), Windows 10/11, macOS 13+ on Apple Silicon |
| Python | none to install: `uv` downloads the pinned 3.11 interpreter itself |
| NVIDIA GPU (recommended) | driver ≥ 525.60 (Linux) / 528.33 (Windows); no CUDA toolkit needed. Check with `nvidia-smi`. |
| CPU-only | supported on every platform; expect runs to be several times slower |
| Apple Silicon | supported; inference runs through the CoreML execution provider, falling back to CPU |
| Disk | About 1.7 GB for default models, plus several GB for the environment (especially GPU dependencies) and space for run outputs |

> **A GPU is optional, not required.** Pick the `cpu` or `macos` variant below and the whole
> pipeline runs unchanged, just slower. The Apple-GPU path has had far less testing than CUDA;
> if `lm3 doctor` reports the CPU provider on a Mac, the run still works.

### Install

Four commands. The environment is built from `uv.lock`, so every install gets exactly the package
versions this release was tested with.

```bash
# 1. uv (once per machine)
curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh                        # Linux / macOS
# powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.23/install.ps1 | iex"   # Windows

# 2. code
git clone https://github.com/Gene-Weaver/LeafMachine3.git
cd LeafMachine3

# 3. environment  (pick ONE line)
uv sync --frozen --extra gpu       # NVIDIA GPU, Linux or Windows
uv sync --frozen --extra cpu       # no GPU, any platform
uv sync --frozen --extra macos     # Apple Silicon

# 4. models (about 1.7 GB, from Hugging Face)
uv run --frozen --no-sync lm3 models install --yes
```

### Verify

```bash
uv --version                               # must print 0.12.23
uv run --frozen --no-sync lm3 doctor                            # must end with "Result: READY"
uv run --frozen --no-sync machine3 --config LM3_settings.yaml   # processes examples/images into runs/demo/
uv run --frozen --no-sync lm3 serve                             # GUI at http://127.0.0.1:8765  (Ctrl+C stops it)
```

`lm3 doctor` checks the interpreter, the lock, the hardware variant, the driver, and that a
real model binds to the accelerator. When something is wrong it prints the command that fixes it.
The first run on a new machine profiles the hardware before it starts, which adds a few minutes.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `machine3` / `lm3 serve` exit with code **78** | the environment does not match `uv.lock` (built with pip, conda, or an old sync) | `uv sync --frozen --extra <gpu\|cpu\|macos> --reinstall` |
| `lm3 doctor` says the **driver is too old** | NVIDIA driver below 525.60 / 528.33 | update the driver, or use `--extra cpu` |
| GPU present but models run on the **CPU** | the CUDA provider failed to load | `uv run --frozen --no-sync lm3 doctor --models` names the missing library; reinstall with `--reinstall` |
| `lm3 models install` **fails to download** | no network, or Hugging Face is rate-limiting anonymous downloads | check the connection and rerun; a free Hugging Face token (`uv run --frozen --no-sync hf auth login`) lifts the anonymous rate limit |
| `lm3 models verify` reports a **hash mismatch** | a partial or stale download | `uv run --frozen --no-sync lm3 models install --force --yes` |
| **Out of GPU memory** | the hardware profile is wrong for this machine | `uv run --frozen --no-sync lm3-setup --force` to re-profile, or lower workers in `hardware_settings.yaml` |
| Only one of several GPUs should be used | | `CUDA_VISIBLE_DEVICES=0 uv run --frozen --no-sync machine3 --config LM3_settings.yaml` |
| **Offline / HPC** compute nodes | no network on the compute node | prefetch once: `lm3 models install --dest "$LM3_MODELS_DIR" --yes`; jobs set the same `LM3_MODELS_DIR` |
| None of the above | | delete `.venv/` and repeat the environment step |

Still stuck? Run `uv run --frozen --no-sync lm3 doctor` and open an issue with its full output.

### Docker

*Coming soon.* A CUDA image with the models pre-installed is planned; the `LM3_MODELS_DIR` mount
described above is the intended way to keep models out of the image.

### Updating

```bash
git checkout -- LM3_settings.yaml   # restores shipped settings; first copy personal settings to LM3_settings.local.yaml
git pull --ff-only
uv sync --frozen --extra gpu        # or cpu / macos
uv run --frozen --no-sync lm3 models install --yes     # picks up any model updates in the lock
```

---

## Quick Start

### Run from the GUI

```bash
uv run --frozen --no-sync lm3 serve
```

![LeafMachine3 Settings tab](docs/readme_github/gui_settings.jpg)

Six tabs, left to right:

| Tab | What it does |
|---|---|
| **Live Status** | per-module progress, worker fleet, throughput and ETA for the active run |
| **Console** | the run's log, streamed |
| **Settings** | every setting in `LM3_settings.yaml`, grouped by module and searchable; the pipeline order is the left rail |
| **Models** | which models are installed, verify them, pick alternates |
| **Results** | browse a finished run: overlays, crops, leaf products, and the CSV tables |
| **Postprocessing** | standalone tools that run on a finished run (STL export, leaf collage) |

Set the **input folder**, name the **project**, press **Start LM3**. The stage bar across the top
mirrors the seventeen modules and fills in as the run advances.

![Live Status tab during a run](docs/readme_github/gui_live_status.jpg)

When the run finishes, the **Results** tab browses everything it wrote, by output folder, with
the run's database alongside; the **Models** tab shows which model revision each module holds
against the release lock; and **Postprocessing** holds the standalone tools.

![Results tab: media browser over a finished run](docs/readme_github/gui_results.jpg)

<details>
<summary>Models and Postprocessing tabs</summary>

![Models tab: every default model installed and matching the lock](docs/readme_github/gui_models.jpg)

![Postprocessing tab: the STL builder and the leaf collage builder](docs/readme_github/gui_postprocess.jpg)

</details>

### Electron desktop app

Install the desktop group with the same hardware extra used for the pipeline (`gpu`, `cpu`, or
`macos`). It includes the pinned Node/npm and uv executables; no separate Node or Python install is
needed. After the sync, `--no-sync` keeps uv from removing the chosen hardware extra or desktop group. For example, on a CPU machine:

```bash
uv sync --frozen --extra cpu --group desktop
uv run --frozen --no-sync lm3-desktop install   # npm ci using the uv-locked Node/npm
uv run --frozen --no-sync lm3-desktop start
```

Desktop targets match the Python lock: Linux x86_64, Windows x64, and Apple Silicon (macOS 13.5+).
Build installers on their target OS and architecture; packaging uses the installed, verified Electron runtime.

For development checks and an unpacked native build:

```bash
uv run --frozen --no-sync lm3-desktop test
uv run --frozen --no-sync lm3-desktop pack
```

Python, Node/npm, and uv are pinned in `uv.lock`. Electron, electron-builder, and their JavaScript
dependencies are integrity-pinned in `app/package-lock.json`; the generated release contracts tie
both locks together, and CI verifies them before packaging. After an intentional dependency change,
update the relevant lock and run `uv run --no-sync python tools/release/write_env_contract.py`.

The packaged desktop shell uses a matching uv checkout for its backend. Set `LM3_ROOT` to that
checkout when launching it, or start that checkout's server with
`uv run --frozen --no-sync lm3 serve` first. Electron verifies the authenticated environment report
before attaching. Legacy virtual environments, Conda, `LM3_PYTHON`, and system Python are unsupported
by the desktop app. Native Windows/macOS packaging is checked in CI; release signing and native
application acceptance remain separate checks.

On Linux, `./launch_gui.sh` starts the same verified desktop command in the background; its
`status`, `stop`, and `restart` commands remain available.

### Run from the command line

```bash
uv run --frozen --no-sync machine3 --config LM3_settings.yaml                            # the whole pipeline
uv run --frozen --no-sync machine3 --config LM3_settings.yaml --run-name my_project      # name the run
uv run --frozen --no-sync machine3 --config LM3_settings.yaml --restart leaf_segmenter   # rerun one stage + everything after it
```

The settings file is the single source of truth: input folders, output folder, run name, and the
`enabled` switch of every module live in `LM3_settings.yaml`. Keep personal paths in
`LM3_settings.local.yaml`, which is ignored by Git. It is a full settings file, selected explicitly:

```bash
cp LM3_settings.yaml LM3_settings.local.yaml   # once, then edit this copy
uv run --frozen --no-sync machine3 --config LM3_settings.local.yaml
```

### Where results land

```
runs/<run_name>/
  <run_name>.sqlite          every result, queryable
  reports/
    Overlay/                 summary overlays + per-leaf landmark / petiole overlays   (start here for QC)
    Leaf_Original/           7 per-leaf products, as mounted
    Leaf_Oriented/           the same 7 products, rotated tip-up
    Crops/                   RGB crops of every non-leaf detection (rulers, labels, fruits, ...)
    Specimen_Masks/          whole-sheet and per-leaf binary masks and RGB cutouts
    Leaf_Data/               bilateral symmetry panels, ECT images and coordinates
    Data/                    the numbers: leaf_measurements.csv is the one to open first
  logs/
```

---

## Workflow

The GUI is organized top-to-bottom the way a run proceeds; this section follows it.

**Project.** Input folders (recursive by default), image extensions, output folder, run name.
Everything a run writes goes to `<output>/<run_name>/`.

**Compute & Hardware.** Which GPUs to use, precision, and the hardware profile. The profile
(`hardware_settings.yaml`) is written automatically the first time LeafMachine3 runs on a machine:
it measures how many workers each stage can hold in VRAM and RAM. Rerun it with
`uv run --frozen --no-sync lm3-setup --force` after a hardware change.

**Image Preparation.** Originals are never modified. Each image is read once, converted to RGB,
and, if its long side exceeds the working limit, downsampled into a working copy that every
module measures on. Every pixel value in the outputs is in that working frame; `work_scale` on
every row converts back.

**Modules.** The seventeen modules in pipeline order, each with an **Enabled** switch and its own
settings. Disabled modules are marked complete without doing work; enabled downstream modules
still run, but may have no inputs. Turn off dependent modules too when you do not need them.

**Reporter.** Which files to write, in which formats, and the overlay palette. Every output folder
toggles independently; the defaults draw everything.

**Resume and rerun.** Progress is a per-stage ledger in the run's SQLite database. Starting the
same run name again picks up where it stopped; `--restart <stage>` discards one stage's results
and everything downstream of it, then reruns them.

---

## In-Depth: the processing modules

One block per module, in runtime pipeline order. ECT runs after Reporter because it consumes
the oriented masks Reporter writes. Each gives what the module does, the model it
runs if it has one, and what it produces. Model metrics are quoted from the Hugging Face model
cards, which hold the full training details.

### 1. MP Conversion Factor

Predicts a pixels-per-centimeter conversion factor from the image's resolution alone. Herbarium
sheets are a standard size, so megapixels alone carry most of the scale information. The value
is a **prior**: the ruler stages publish a measured factor only when it agrees with this estimate
or rests on enough ruler length to outvote it. By default the value itself is never used to
convert measurements; the `use_CF_predicted_by_MP` option of stage 7 makes it the fallback.

| Model | [`lm3_mp_conversion_factor__sqrt_fit`](https://huggingface.co/phyloforfun/lm3_mp_conversion_factor__sqrt_fit) |
|---|---|
| Architecture | one-parameter fit, CF = k·√MP, through the origin |
| Trained on | 708 human ruler measurements, 549 sheets, 48 herbaria |
| Accuracy | R² 0.905; mean absolute error 2.3 %, 95th percentile 7.5 % (in-sample) |

*Output:* `specimen.cf_px_per_cm_predicted_by_mp`. CPU only; runs before any image is decoded.

### 2. Archival Detector

Places bounding boxes around the non-plant components of the sheet: rulers, barcodes, color
cards, labels, maps, envelopes, photos, attached items, and paper weights. The ruler boxes feed
the two ruler stages; every other class is cropped for downstream use (OCR, label transcription,
screening a collection for envelopes or attached items).

![Archival crops: ruler, barcode, label, color card, envelope](docs/readme_github/archival_detector_crops.jpg)

| Model | [`lm3_archival_detector__yolo26x_det_1280`](https://huggingface.co/phyloforfun/lm3_archival_detector__yolo26x_det_1280) (default) · [`yolo26n_det_640`](https://huggingface.co/phyloforfun/lm3_archival_detector__yolo26n_det_640) (lightweight) |
|---|---|
| Architecture | YOLO26x object detection, end-to-end (NMS-free), 1280 px |
| Classes | 9: Ruler, Barcode, Colorcard, Label, Map, Envelope, Photo, Attached Item, Weights |
| Trained on | 8,470 sheets from 67 annotation projects |
| Accuracy | test mAP50 0.949, mAP50-95 0.775 (nano alternate: 0.884 / 0.678) |

*Output:* `archival_detection` table; `Crops/RGB__<class>/` and `detections.csv`.

### 3. Plant Detector

Places bounding boxes around plant organs: whole leaves, partial leaves, single and grouped
fruits, single and grouped flowers, buds, roots, and wood. Whole-leaf boxes (`Leaf_WHOLE`, the
"ideal leaves" of LeafMachine2) are the input to leaf segmentation and landmarking; every class
is counted by the phenology stage and cropped. As in LeafMachine2, leaflets are treated as simple
leaves.

![Plant crops: three leaves, a bud, two fruits](docs/readme_github/plant_detector_crops.jpg)

| Model | [`lm3_plant_detector__yolo26x_det_1280`](https://huggingface.co/phyloforfun/lm3_plant_detector__yolo26x_det_1280) (default) · [`yolo26n_det_640`](https://huggingface.co/phyloforfun/lm3_plant_detector__yolo26n_det_640) (lightweight) |
|---|---|
| Architecture | YOLO26x object detection, end-to-end (NMS-free), 1280 px |
| Classes | 9: Leaf_WHOLE, Leaf_PARTIAL, Seed_Fruit_ONE, Seed_Fruit_MANY, Flower_ONE, Flower_MANY, Bud, Roots, Wood |
| Trained on | 22,785 sheets from 83 annotation projects |
| Accuracy | test mAP50 0.382, mAP50-95 0.233 (nano alternate: 0.237 / 0.135) |

The plant data set is far more heterogeneous than the archival one, so its mAP is lower, as it
was for LeafMachine2. What matters in practice is that a leaf the detector misses is never
segmented or landmarked, so the detector is tuned for recall on whole leaves.

*Output:* `plant_detection` table; `Leaf_Original/Leaf_BBox/` for leaves, `Crops/RGB__<class>/`
for everything else.

### 4. Specimen Segmenter

A whole-sheet plant-versus-background mask, independent of the plant boxes. It removes the paper,
labels, and mounting materials from the sheet to leave only plant material, with an optional
paper-cleanup pass. Used for background removal and for the whole-specimen mask products.

![Specimen segmentation: annotated sheet and the plant-only cutout, three specimens](docs/readme_github/specimen_segmenter.jpg)

| Model | [`lm3_specimen_segmenter__unetpp_effb7_1024`](https://huggingface.co/phyloforfun/lm3_specimen_segmenter__unetpp_effb7_1024) (default) · [`birefnet_hr_swinl_1024`](https://huggingface.co/phyloforfun/lm3_specimen_segmenter__birefnet_hr_swinl_1024) · [`yolo26x_seg_1280`](https://huggingface.co/phyloforfun/lm3_specimen_segmenter__yolo26x_seg_1280) |
|---|---|
| Architecture | UNet++ with an EfficientNet-B7 encoder, 1024 px; alternates are BiRefNet (Swin-L) and YOLO26x-seg |
| Classes | 1: plant |
| Trained on | 18,904 sheets from 75 annotation projects (machine-generated targets, SAM 3 → self-training) |
| Accuracy | test Dice 0.948 (median 0.977), IoU 0.916 against the machine-generated masks |

BiRefNet scores slightly higher at the median and is markedly slower on CPU; YOLO26x-seg keeps
more of the paper enclosed by stems. Faint, dried-brown material on similarly colored paper is
the known weak case for all three.

*Output:* `specimen_mask` table; `Specimen_Masks/*_Specimen/` and `Overlay/Overlay_Specimen_Segmentation/`.

### 5. Phenology Detector

Reads the plant detections to decide whether leaves, flowers, and fruits are present on the sheet.
No model: it applies count thresholds to the Plant Detector's boxes. The output file is laid out
like LeafMachine2's `phenology.csv` so existing phenology scripts read it unchanged; the two
LeafMachine2 columns that LeafMachine3 does not produce (`leaflet`, `specimen`) are left blank.

*Output:* `phenology` table; `has_leaves` / `has_flowers` / `has_fruits` on the specimen;
`Data/phenology.csv`.

### 6. Ruler Classifier

Assigns a unit type to every ruler crop, the first half of measuring the conversion factor.
Each crop is squarified (rotated to landscape, grayscale, tiled into a four-view collage, as in
LeafMachine2) and classified by a three-model ensemble by majority vote. The per-specimen
consensus goes to the next stage.

![Ruler crops from the three specimens](docs/readme_github/ruler_classifier_crops.jpg)

| Model | ensemble: [`yolo26x_cls_224`](https://huggingface.co/phyloforfun/lm3_ruler_classifier_ensemble__yolo26x_cls_224) + [`yolo26n_cls_224`](https://huggingface.co/phyloforfun/lm3_ruler_classifier_ensemble__yolo26n_cls_224) + [`dinov2_frozen_mlp`](https://huggingface.co/phyloforfun/lm3_ruler_classifier_ensemble__dinov2_frozen_mlp) |
|---|---|
| Architecture | YOLO26x and YOLO26n classifiers, and a frozen DINOv2-base with an MLP head; majority vote, YOLO26x breaks ties |
| Classes | 19 unit types (metric cm / mm / mm-cm, imperial 1/8 – 1/32 in, grid, block, mixed, plus `messy`) |
| Trained on | 13,707 labeled ruler crops |
| Accuracy | test accuracy 0.990 / 0.987 / 0.985 (balanced 0.980 / 0.974 / 0.978) for the three members |

*Output:* `ruler_classification` table; `specimen.ruler_class_type`.

### 7. Ruler Conversion Factor

Measures pixels per centimeter from the ruler's tick lattice. Rather than binarizing the ruler
and scanning for marks, as LeafMachine2 did, it fits a regular lattice to the tick pattern of the
unit type the classifier assigned, so hundreds of tick spacings vote on one factor. The factor is
**published** only when the fit is confident and agrees with the resolution prior from stage 1;
otherwise the sheet is left unconverted rather than mis-converted. The QC panel shows the crop,
the lattice fit, and the verdict.

With `modules.ruler_cf.use_CF_predicted_by_MP: true` (off by default), a sheet with no ruler, or
one whose lattice did not pass, gets the stage 1 prediction instead, so it still has cm
measurements. Every output says which kind it is: `specimen.cf_source` and the `cf_source` CSV
column read `measured_from_ruler` or `predicted_from_megapixels`. On the summary overlay, a
predicted factor is labeled in the CF banner, its 1 cm / 1 inch raft sits in the top-left corner
instead of on a ruler, and the exterior 1 cm checkerboard is black and 50% gray instead of black
and white.

![Ruler lattice QC panels for the three specimens](docs/readme_github/ruler_cf_lattice.jpg)

No model; CPU. *Output:* `ruler_CF_lattice` tables; `specimen.cf_px_per_cm` + `cf_source`;
`Overlay/Overlay_Ruler_Lattice/` and `Data/ruler_conversion_factor.csv` (the verdict and why).

### 8. Leaf Segmenter

Turns each whole-leaf crop into instance masks for the lamina, the petiole, and any holes. As in
LeafMachine2, segmentation runs on one leaf crop at a time, so overlapping leaves, mounting tape,
and cluttered backgrounds are handled per leaf and the network only ever sees the leaf it is
asked about. Holes and petioles are attached to their owning leaf.

![Leaf segmentation: detector crop, instance mask, RGB cutout, for one leaf per specimen](docs/readme_github/leaf_segmenter.jpg)

| Model | [`lm3_leaf_segmenter__yolo26x_seg_1024`](https://huggingface.co/phyloforfun/lm3_leaf_segmenter__yolo26x_seg_1024) |
|---|---|
| Architecture | YOLO26x instance segmentation, end-to-end (NMS-free), 1024 px |
| Classes | 3: Leaf, Petiole, Hole |
| Trained on | 16,895 single-leaf crops from 5 annotation projects |
| Accuracy | test mask mAP50 0.726; per class Leaf 0.933, Petiole 0.818, Hole 0.429 |

*Output:* `leaf_segmentation` table; `Specimen_Masks/*__Leaf/` per leaf and per sheet.

### 9. Morphology

Shape metrics for every leaf mask, in the LeafMachine2 vocabulary: area, perimeter, centroid,
convex hull, convexity, concavity, circularity, aspect ratio, and the rotated minimum bounding
box whose long and short sides are the leaf's length and width. Areas are hole-aware: lamina
area including holes, excluding holes, total hole area, and hole count are all reported.

![Hole-aware lamina products: RGB, mask with holes removed, filled silhouette](docs/readme_github/morphology_holes.jpg)

No model; CPU. The rotated-box method is selectable (`pca`, the default, `feret`, `lm2`,
`minarearect`). *Output:* `leaf_morphology` table; the green rotated boxes on the summary overlay.

### 10. Landmark Detector

Predicts a 31-keypoint pose skeleton on every whole-leaf crop: the lamina tip and base, apex and
base triples, 15 points tracing the midvein, 6 tracing the petiole, and the two width points.
LeafMachine2 located 9 pseudo-landmarks as fixed-size boxes; the pose formulation returns every
point with a confidence, so missing or occluded points are known to be missing.

![Landmark overlays for one leaf per specimen](docs/readme_github/landmark_detector.jpg)

| Model | [`lm3_landmark_detector__yolo26x_pose_640`](https://huggingface.co/phyloforfun/lm3_landmark_detector__yolo26x_pose_640) |
|---|---|
| Architecture | YOLO26x pose, end-to-end (NMS-free), 640 px; the `mid15_pet5` skeleton |
| Keypoints | 31 (lamina_tip, apex ×3, midvein ×15, base ×3, lamina_base, petiole ×6, width ×2) |
| Trained on | 15,217 single-leaf crops, 358,054 keypoint instances, 23 annotation projects |
| Accuracy | test box mAP50 0.995; pose mAP50 0.763, mAP50-95 0.433 (OKS) |

*Output:* `leaf_landmark` table (specimen and crop coordinates); `Data/landmarks.csv`.

### 11. Landmark Measurements

Derives per-leaf biology from the keypoints: the midvein trace length and its straight-line
extent, the tip-to-base length, the leaf width, apex and base angles with their type (acute,
obtuse, reflex, following LeafMachine2's convention), the petiole trace length, and the lamina
curvature (the largest midvein bend, 0° = straight). It is occlusion-robust: any metric that needs
a low-confidence keypoint is left `NULL`; nothing is fabricated.

![Landmark overlays with the derived measurements for a second leaf per specimen](docs/readme_github/landmark_measurements.jpg)

No model; CPU. *Output:* `leaf_landmark_measurement` table; the cyan/white/black lines and the
measurement block on `Overlay/Overlay_Landmarks/`.

### 12. Leaf Orientation

Computes the rotation that stands each leaf tip-up and base-down, from the lamina tip → base axis
or, failing that, the midvein's principal axis. Every per-leaf product is then written twice: as
mounted (`Leaf_Original/`) and oriented (`Leaf_Oriented/`). Oriented leaves are what the symmetry
and shape stages consume.

![As-mounted and oriented cutouts, one leaf per specimen](docs/readme_github/leaf_orientation.jpg)

No model; CPU. *Output:* `oriented_leaf_rotation_angle_degreesCW` and `oriented_leaf_success` on
`leaf_morphology`.

### 13. Petiole Width

Measures each petiole's width as the median of several perpendicular thickness samples near the
blade junction, from the petiole mask and the landmark petiole centerline. The overlay shows the
sample probes and the width band, and blows the petiole up to the pixel so the measurement can be
checked by eye.

![Petiole width overlays, one leaf per specimen](docs/readme_github/petiole_width.jpg)

No model; CPU. *Output:* `leaf_petiole` table (`width_px`, `length_px`, `touches_leaf`);
`Overlay/Overlay_Petiole/`.

### 14. Bilateral Symmetry

Scores how closely each oriented leaf mirrors itself across its traced midvein, and rolls that up
with mask and trace quality terms into a single **archetype score**: a ranking of which leaves on
a sheet are the cleanest, most representative examples. Gates veto leaves whose masks or traces
are not trustworthy. The Leaf Collage tool uses this ranking to pick leaves.

![Bilateral symmetry panels](docs/readme_github/bilateral_symmetry.jpg)

No model; CPU. *Output:* `bilateral_symmetry` table; `Leaf_Data/Bilateral_Symmetry/`.

### 15. Metric Grounding

Converts every pixel measurement to real units using the sheet's conversion factor: areas to
cm², lengths to cm. By default that is only a ruler factor the lattice stage published with high
confidence; sheets without one keep their pixel values, and their `_cm` columns are empty rather
than wrong. With `use_CF_predicted_by_MP` on (stage 7), those sheets are grounded with the
resolution estimate instead.

No model; CPU. *Output:* the `_cm` and `_cm2` columns in `Data/leaf_measurements.csv`, with
`cf_source` on each row saying where the factor came from.

### 16. Reporter

Writes every file LeafMachine3 produces: the summary overlay, the per-leaf overlays, crops,
masks, the seven per-leaf products in both trees, and the CSV export of the database. Every
output folder toggles independently, and the overlay palette (LeafMachine2's by default) is fully
configurable.

![The seven leaf products, as mounted and oriented](docs/readme_github/reporter_leaf_products.jpg)

The seven per-leaf products: the detector crop, the lamina mask, the lamina + petiole mask, the
filled lamina silhouette, and RGB cutouts of each (the lamina-holes cutout keeps the tissue and
paints holes a recoverable near-black).

No model. *Output:* everything under `reports/`; see [Outputs](#outputs).

### 17. Shape (ECT)

Computes the Euler Characteristic Transform of every oriented leaf, a topological shape
descriptor that captures the outline from every direction at once and is directly comparable
across leaves and taxa. Three images are written per leaf: the Cartesian ECT, its radial form,
and the radial form with the leaf outline drawn over it, plus the raw coordinates as HDF5.

![ECT: oriented leaf, Cartesian ECT, radial ECT, radial ECT with outline](docs/readme_github/ect.jpg)

No model; CPU. *Output:* `leaf_ect` table; `Leaf_Data/Oriented_Leaf_ECT/`,
`Oriented_Leaf_Radial_ECT/`, `Oriented_Leaf_Radial_ECT_Overlay/`, `Coordinates/*.h5`.

The transform is computed with the [`ect`](https://github.com/MunchLab/ect) Python package
([documentation](https://munchlab.github.io/ect/)). If you use the ECT outputs, please cite:

- Ayub, Y., McGuire-Scullen, S., Percival, S., Weaver, W. N., … Munch, E., & Chitwood, D. H.
  (2026). The Euler Characteristic Transform enables classification of complex plant shapes and
  prediction of leaf venation from blade geometry. *bioRxiv*.
  <https://doi.org/10.64898/2026.04.13.718293>
- Ayub, Y., Munch, E., McGuire Scullen, S., & Chitwood, D. H. (2026). ect: A Python package for
  the Euler Characteristic Transform. *Journal of Open Source Software*, 11(120), 9691.
  <https://doi.org/10.21105/joss.09691>
- Munch, E. (2025). An invitation to the Euler Characteristic Transform. *The American
  Mathematical Monthly*, 132(1), 15–25. <https://doi.org/10.1080/00029890.2024.2409616>

---

## Outputs

### The `reports/` folder

```
reports/
  Overlay/
    Overlay_Summary/               <stem>__Overlay.jpg              masks + boxes + landmarks + petiole bands
    Overlay_Landmarks/             <stem>__LM-leaf__x_y_x_y.jpg     per leaf
    Overlay_Petiole/               <stem>__PET-leaf__x_y_x_y.jpg    per leaf
    Overlay_Specimen_Segmentation/ <stem>__SpecimenSeg.jpg
    Overlay_Ruler_Lattice/         <stem>__RulerLattice.png         ruler QC panel
  Leaf_Original/                   7 products per leaf, as mounted  (og-)
  Leaf_Oriented/                   the same 7, rotated tip-up       (or-)
  Crops/RGB__<class>/              every non-leaf detection
  Specimen_Masks/                  whole-sheet and per-leaf masks, binary and RGB
  Leaf_Data/                       Bilateral_Symmetry/, Oriented_Leaf_ECT/, ..., Coordinates/
  Data/                            CSV export (below)
```

### The CSV tables

| File | One row per |
|---|---|
| `leaf_measurements.csv` | **leaf** — every morphology, landmark, petiole and symmetry measurement, plus the identifiers to trace it back. Start here. |
| `specimen_summary.csv` | input image, with per-sheet roll-ups |
| `phenology.csv` | sheet, in LeafMachine2's layout |
| `detections.csv` | detection box, both detectors |
| `landmarks.csv` | predicted keypoint (31 per leaf) |
| `ruler_conversion_factor.csv` | sheet: the conversion-factor verdict and why |
| `run_stages.csv`, `stage_errors.csv` | pipeline stage; per-image failure |
| `data_dictionary.csv` | every column above: file, units, meaning |

Two things to know before analyzing: every `_px` value is in the **working frame** (divide by
`work_scale` for original-image pixels), and `_cm` columns are **empty unless a ruler conversion
factor was published** for that sheet; an empty cell means "not measured", never zero.

### File naming

Every file is named `<stem>__<ID>__x_y_x_y.<ext>`, where `x_y_x_y` is the detection box in the
working frame, so any crop can be placed back on its parent by filename alone. Every `ID` is
unique across the whole run, so the entire `reports/` tree can be flattened into one folder
without a collision. Class → friendly-name mapping and the prefixes are editable under `naming`.

---

## Postprocessing Tools

Small, standalone tools that run on a finished run. They have their own settings file
(`postprocessing_settings.yaml`) and their own GUI tab.

**3D-printable STL from a mask.** Extrudes any binary mask PNG (a lamina silhouette, a whole
specimen mask) into a flat `.stl` slab of a chosen length and thickness, holes filled, boundary
smoothed. The same tool runs in the browser at [leafmachine.org](https://leafmachine.org) with no
upload.

```bash
uv run --frozen --no-sync python -m leafmachine3.postprocessing.generate_stl_from_mask --config postprocessing_settings.yaml --paths <mask.png>
```

**Leaf collage.** Tiles a run's best leaves, ranked by the bilateral-symmetry archetype score,
into the outline of one primary mask: a leaf built out of leaves. Reads existing Reporter output
only; nothing is re-segmented.

```bash
uv run --frozen --no-sync python -m leafmachine3.postprocessing.generate_leaf_collage --config postprocessing_settings.yaml --run-dir runs/demo --primary-mask <mask.png>
```

---

## Models

Every model is an exported end-to-end ONNX graph run by onnxruntime, published on Hugging Face
with a model card (training data, metrics, license) and a manifest of file hashes. `lm3 models
install` fetches the defaults; alternates install on request and the command prints the settings
lines that select them.

```bash
uv run --frozen --no-sync lm3 models install --list-alternates
uv run --frozen --no-sync lm3 models install --model specimen_segmenter=birefnet_hr_swinl_1024
uv run --frozen --no-sync lm3 models verify
```

| Module | Default | Alternates | License |
|---|---|---|---|
| Archival Detector | [yolo26x_det_1280](https://huggingface.co/phyloforfun/lm3_archival_detector__yolo26x_det_1280) | [yolo26n_det_640](https://huggingface.co/phyloforfun/lm3_archival_detector__yolo26n_det_640) | AGPL-3.0 |
| Plant Detector | [yolo26x_det_1280](https://huggingface.co/phyloforfun/lm3_plant_detector__yolo26x_det_1280) | [yolo26n_det_640](https://huggingface.co/phyloforfun/lm3_plant_detector__yolo26n_det_640) | AGPL-3.0 |
| Specimen Segmenter | [unetpp_effb7_1024](https://huggingface.co/phyloforfun/lm3_specimen_segmenter__unetpp_effb7_1024) | [birefnet_hr_swinl_1024](https://huggingface.co/phyloforfun/lm3_specimen_segmenter__birefnet_hr_swinl_1024), [yolo26x_seg_1280](https://huggingface.co/phyloforfun/lm3_specimen_segmenter__yolo26x_seg_1280) | Apache-2.0 (MIT, AGPL-3.0) |
| Ruler Classifier | ensemble: [yolo26x_cls_224](https://huggingface.co/phyloforfun/lm3_ruler_classifier_ensemble__yolo26x_cls_224), [yolo26n_cls_224](https://huggingface.co/phyloforfun/lm3_ruler_classifier_ensemble__yolo26n_cls_224), [dinov2_frozen_mlp](https://huggingface.co/phyloforfun/lm3_ruler_classifier_ensemble__dinov2_frozen_mlp) | | AGPL-3.0 / Apache-2.0 |
| Leaf Segmenter | [yolo26x_seg_1024](https://huggingface.co/phyloforfun/lm3_leaf_segmenter__yolo26x_seg_1024) | | AGPL-3.0 |
| Landmark Detector | [yolo26x_pose_640](https://huggingface.co/phyloforfun/lm3_landmark_detector__yolo26x_pose_640) | | AGPL-3.0 |
| MP Conversion Factor | [sqrt_fit](https://huggingface.co/phyloforfun/lm3_mp_conversion_factor__sqrt_fit) | | MIT |

Training images and annotations are not distributed. The detector, segmenter, and landmark data
sets were annotated in Labelbox, extending the 494,766-annotation LeafMachine2 effort; the
specimen-segmenter targets are machine-generated.

---

## Related Projects and Citation

- **LeafMachine2** — [github.com/Gene-Weaver/LeafMachine2](https://github.com/Gene-Weaver/LeafMachine2).
  Weaver, W. N., & Smith, S. A. (2023). From leaves to labels: Building modular machine learning
  networks for rapid herbarium specimen analysis with LeafMachine2. *Applications in Plant
  Sciences*, 11(5), e11548. <https://doi.org/10.1002/aps3.11548>
- **VoucherVision** — [github.com/Gene-Weaver/VoucherVision](https://github.com/Gene-Weaver/VoucherVision):
  label transcription with large language models; the Archival Detector's label crops are its input.
- **leafmachine.org** — <https://leafmachine.org>

A LeafMachine3 paper is in preparation. Until it is published, please cite the LeafMachine2 paper
above and link this repository:

```
Weaver, W. N. (2026). LeafMachine3. https://github.com/Gene-Weaver/LeafMachine3
```

---

## License

LeafMachine3 is free software under the **GNU General Public License v3.0**; see
[LICENSE](LICENSE). As an additional term under section 7(b) of that license, copies and
derivative works must keep the author attribution in [NOTICE](NOTICE).

`leafmachine3/inference/ultra_replacements.py` ports code from Ultralytics and remains under the
AGPL-3.0 (see NOTICE). Model weights are not in this repository; each published model's license
is on its Hugging Face model card.
