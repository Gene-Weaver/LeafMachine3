# Installing LeafMachine3

LM3 ships the **runtime** only (exported/trained models + orchestration). Every model is an exported
**end2end ONNX graph**, and the only inference engine is `onnxruntime`
(`leafmachine3/inference/ultra_replacements.py` replaces the Ultralytics predict path). There is
**no torch and no ultralytics** in the runtime environment. Pick the variant for your hardware.

Two ways to install: **pinned requirements files** (exact, reproducible — recommended) or **editable
extras / poetry** (flexible). All three platform variants use the same package versions; only the
`onnxruntime` flavor and, for NVIDIA, the `nvidia-*` CUDA runtime wheels differ.

## The one gotcha: onnxruntime-gpu must be a CUDA-12 build

`onnxruntime-gpu` 1.28+ is a **CUDA-13** build and needs NVIDIA driver **≥ 580**. The GPU
requirements file pins `onnxruntime-gpu==1.20.2` (CUDA 12 / cuDNN 9) plus the exact `nvidia-*`
wheels its CUDA provider loads (cuBLAS, cudart, cuDNN 9, cuFFT, cuRAND, NVRTC). With those, the only
system requirement is a driver ≥ 525.60.13 (Linux) / 528.33 (Windows). Check yours with `nvidia-smi`.

## Pinned requirements (recommended)

```bash
python -m venv .venv_LM3 && .venv_LM3/bin/pip install -U pip wheel setuptools

# NVIDIA GPU (default):
.venv_LM3/bin/pip install -r requirements/requirements-gpu.txt
# CPU only:
.venv_LM3/bin/pip install -r requirements/requirements-cpu.txt
# macOS / Apple Silicon (CoreML):
.venv_LM3/bin/pip install -r requirements/requirements-macos.txt
```

Verify the GPU provider is present (should list `CUDAExecutionProvider`):

```bash
.venv_LM3/bin/python -c "import onnxruntime as o; print(o.__version__, o.get_available_providers())"
```

Tested combo (driver 560 / CUDA 12.6, RTX 6000 Ada, Python 3.11): onnxruntime-gpu 1.20.2,
nvidia-cudnn-cu12 9.1.0.70, nvidia-cublas-cu12 12.4.5.8, numpy 2.4.4, opencv 5.0.0.93, pillow 12.2.0,
scipy 1.17.1.

## Editable extras (pip)

The `[project.optional-dependencies]` in `pyproject.toml` are the flexible equivalent:

```bash
pip install -e ".[gpu]"       # NVIDIA (onnxruntime-gpu + nvidia-* CUDA 12 wheels)
pip install -e ".[cpu]"       # CPU
pip install -e ".[macos]"     # macOS
```

## Poetry

Poetry ≥ 2.0 installs the same PEP 621 project + extras directly:

```bash
poetry install --extras "gpu"     # or "cpu" / "macos"
```

## Models

The runtime needs the exported default models, which are **not in the repository**. They are
published on the Hugging Face Hub (one repo per model, pinned by commit in
`leafmachine3/modelhub/models.lock.yaml`) and installed with one command:

```bash
uv run install_models.py              # or: .venv_LM3/bin/python install_models.py
# equivalent:  lm3 models install     (lm3 models status / verify to inspect)
```

It checks `models/` beside `LM3_settings.yaml` first: files that already match the pinned
version are left alone, missing or outdated ones are downloaded, hash-checked, and swapped in. Any
file it replaces is kept as `<name>.backup` until the whole module is in place, and restored if a
download or check fails (or if LM3 crashes mid-update), so an update never leaves the folder
broken. The GUI has the same button under **Settings > Models from Hugging Face**, and shows it on
the first window whenever a model is missing or a newer one is pinned.

While the model repos are private you must be logged in (`hf auth login`, or `HF_TOKEN`).

**Alternate models.** Published non-default models install only on request, and the command prints
the settings lines that select them:

```bash
lm3 models install --list-alternates
lm3 models install --model specimen_segmenter=yolo26x_seg_1280
```

The specimen segmenter is chosen by name: `modules.specimen_segmenter.model.key` (for example
`unetpp_effb7_1024`, `birefnet_hr_swinl_1024`, `yolo26x_seg_1280`) decides how each sheet is
prepared for the model, so every model gets the resize, padding and normalization it was trained
with. `model.path` must point at that model's file.

**Docker / HPC.** Set `LM3_MODELS_DIR` to a persistent, mounted folder and prefetch on a node with
network access: `lm3 models install --dest "$LM3_MODELS_DIR" --yes`. Compute jobs then run with
the same `LM3_MODELS_DIR` and never touch the network; `lm3 models verify` confirms the folder.

## Run

```bash
machine3 --config LM3_settings.yaml
```

`compute.onnxruntime.ld_library_path: auto` (the default) puts the venv's bundled `nvidia/*/lib`
directories on `LD_LIBRARY_PATH` before workers spawn, so `onnxruntime-gpu` finds CUDA/cuDNN. To run
on a specific GPU, set `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=1 machine3 --config ...`).
