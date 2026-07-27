# Installing LeafMachine3

LM3 ships the **runtime** only (exported/trained models + orchestration). The heavy pieces are
`torch` + `ultralytics` (which drive the YOLO26 `.onnx`/`.pt` detector, segmenter, and pose models)
and `onnxruntime` (ruler ensemble + provider selection). Pick the variant for your hardware.

Two ways to install: **pinned requirements files** (exact, reproducible — recommended) or **editable
extras / poetry** (flexible). All three platform variants use the same package versions; only the
`torch` build and the `onnxruntime` flavour differ.

## The one gotcha: torch CUDA build

`pip install torch` now defaults to a **CUDA-13** wheel, which needs NVIDIA driver **≥ 580**. Many
boxes (e.g. driver 560 = CUDA 12.6) need the **CUDA-12** wheel from the PyTorch index instead. The
GPU requirements file pins `torch==2.6.0+cu124` from `https://download.pytorch.org/whl/cu124`. Check
your driver's max CUDA with `nvidia-smi`; if it's ≥ 13.0 you may use the default wheel.

## Pinned requirements (recommended)

```bash
python -m venv .venv_LM3 && .venv_LM3/bin/pip install -U pip wheel setuptools

# NVIDIA GPU (default):
.venv_LM3/bin/pip install -r requirements/requirements-gpu.txt
# CPU only:
.venv_LM3/bin/pip install -r requirements/requirements-cpu.txt
# macOS / Apple Silicon (MPS + CoreML):
.venv_LM3/bin/pip install -r requirements/requirements-macos.txt
```

Verify the GPU binds (should print `cuda True` and `CUDAExecutionProvider`):

```bash
.venv_LM3/bin/python -c "import torch, onnxruntime as o; print('cuda', torch.cuda.is_available()); print(o.get_available_providers())"
```

Tested combo (driver 560 / CUDA 12.6, RTX 6000 Ada, Python 3.11): torch 2.6.0+cu124, torchvision
0.21.0+cu124, onnxruntime-gpu 1.20.1, ultralytics 8.4.107, numpy 2.4.4, opencv 5.0.0.93, pillow 12.2.0.

## Editable extras (pip)

The `[project.optional-dependencies]` in `pyproject.toml` are the flexible equivalent:

```bash
pip install -e ".[gpu,yolo]"     # NVIDIA  (then reinstall torch from the cu124 index if needed)
pip install -e ".[cpu,yolo]"     # CPU
pip install -e ".[macos,yolo]"   # macOS
```

For NVIDIA with an older driver, force the CUDA-12 torch afterwards:

```bash
pip install --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

## Poetry

Poetry ≥ 2.0 installs the same PEP 621 project + extras directly:

```bash
poetry install --extras "gpu yolo"     # or "cpu yolo" / "macos yolo"
```

For the CUDA-12 torch, add the PyTorch source and pin the wheel (NVIDIA only):

```bash
poetry source add --priority explicit pytorch-cu124 https://download.pytorch.org/whl/cu124
poetry add --source pytorch-cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
```

## Run

```bash
machine3 --config LM3_settings.yaml
```

`compute.onnxruntime.ld_library_path: auto` (the default) puts the venv's bundled `nvidia/*/lib`
directories on `LD_LIBRARY_PATH` before workers spawn, so `onnxruntime-gpu` finds CUDA/cuDNN. To run
on a specific GPU, set `CUDA_VISIBLE_DEVICES` (e.g. `CUDA_VISIBLE_DEVICES=1 machine3 --config ...`).
