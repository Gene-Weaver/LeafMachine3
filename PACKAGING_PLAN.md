# LeafMachine3 Packaging Plan

## Status

- **Scope:** how an inference user obtains a working LM3 environment. This document supersedes
  sections 5 and 11 of `DEPLOYMENT_PLAN.md` (environment families and local installation options).
  Model distribution, the repo split, Electron, and the cluster GUI tunnel stay in `DEPLOYMENT_PLAN.md`.
- **Scope of the dependency set:** everything below describes the **production runtime** that users
  install. It has no torch and no ultralytics (the YOLO26 exports are end2end ONNX graphs driven by
  `leafmachine3/inference/ultra_replacements.py`, merged 2026-10-06). Will's **development
  environment** still carries torch, torchvision and ultralytics (A/B harness, experiments,
  calibration) and is declared as a uv dependency group that no user command installs; see
  section 13. The training stacks stay in their own environments per `DEPLOYMENT_PLAN.md` section 5.
- **Implementation state (2026-10-07):** Phases A and B are DONE on branch `packaging-uv-doctor`,
  which also carries the consolidated runtime / GUI / Electron / data-export work and the specimen
  model keys. Not merged to `master`. The patched ONNX exports are published and pinned (section
  14). Phase C (INSTALL.md, deleting requirements/, uv-based CI) is next.
- **Goal:** zero wiggle room. A user who follows the instructions for their platform gets the exact
  environment the release was tested with, or an explicit failure during installation that names the
  cause. Nothing resolves, upgrades, or falls back on its own.
- **Why this exists:** LeafMachine2 shipped `pip install` instructions with pinned direct dependencies
  and floating transitive ones. Every install day produced a different environment. The support burden
  came from four causes, and each is closed below by construction, not by documentation:

| LM2 failure | Root cause | Closed by |
|---|---|---|
| Works today, breaks next month | transitive deps float | hash-locked universal `uv.lock`, `--frozen` install |
| Wrong Python | user supplied the interpreter | uv downloads the pinned interpreter, system Python is never used |
| torch/CUDA mismatch | index and driver chosen by the user | no torch in the runtime; one `gpu` extra on one index, driver minimum checked by `lm3 doctor` |
| Mixed conda and pip | several tools touching one env | uv is the only supported installer |

## 1. Decisions

These are settled. Later sections implement them.

1. **One Python version.** `3.11` only. `requires-python = ">=3.11,<3.12"`. uv installs it; a system
   Python is never used, even if present (`python-preference = "only-managed"`).
2. **uv is the only native installer.** The pinned-requirements, editable-extras, and Poetry
   sections of `INSTALL.md` are removed. `requirements/*.txt` are deleted; `uv export` can regenerate
   an equivalent view on demand. The uv version is pinned and the install instructions fetch that
   exact version.
3. **One lockfile.** `uv.lock` is committed, resolved universally for Linux x86_64, Windows x86_64,
   and macOS arm64. Every package, transitive included, is pinned by version and sha256.
4. **Hardware is an extra, nothing else is.** Exactly three mutually exclusive extras: `gpu`, `cpu`,
   `macos` (the names LM3 already used). Everything that was optional before (server, postprocessing, ECT, test tooling in `dev`) is
   folded into the base dependency set or `dev`. A user chooses one word and nothing more.
5. **The install is one command per platform plus one check.** `uv sync --frozen --extra <hw>` then
   `uv run lm3 doctor`. The doctor is part of the install, not an optional diagnostic.
6. **Fail at install, not at stage three.** `lm3 doctor` creates a real ONNX Runtime session and runs a
   test model on the requested accelerator. Listing a provider is not proof it loads.
7. **Containers come from the same lock.** The CPU and CUDA images run `uv sync --frozen` against the
   committed `uv.lock`. There is no second dependency specification anywhere.
8. **Apptainer images are published, not converted by users.** CI builds the SIF and pushes it to the
   registry with ORAS. Cluster users `apptainer pull` a finished file.
9. **Every release ships a wheelhouse.** All wheels the lock resolves to, per target, archived with a
   sha256 manifest. An install can complete with no package index reachable.
10. **The lock changes only in a dedicated environment-bump PR** that passes the full matrix plus the
    GPU self-hosted job. Dependabot and Renovate are disabled for Python.
11. **Production is torch-free; dev is not.** The wheel, the three hardware extras, the lockfile's
    user-facing resolution, both images and every wheelhouse contain no torch, torchvision or
    ultralytics. A `[dependency-groups] full` entry carries them for development only
    (`uv sync --extra gpu --group full`); it is locked in the same `uv.lock` but never installed by
    the user instructions, the Dockerfiles, the wheelhouse builder or CI's user-facing legs.
12. **Supported targets** are exactly:

| Target | Extra | Accelerator | Route |
|---|---|---|---|
| Linux x86_64, NVIDIA | `gpu` | CUDA 12.4 wheels | uv, Docker, Apptainer |
| Linux x86_64, no GPU | `cpu` | none | uv, Docker, Apptainer |
| Windows x86_64, NVIDIA | `gpu` | CUDA 12.4 wheels | uv |
| Windows x86_64, no GPU | `cpu` | none | uv |
| macOS 13+ arm64 | `macos` | CoreML EP (ONNX) | uv |

Not supported, and `lm3 doctor` says so by name: Intel macOS, Linux aarch64, Docker on Windows or
macOS, Python other than 3.11, conda environments, AMD or Intel GPUs. Windows DirectML is a future
extra, not part of this plan.

## 2. Project configuration

### 2.1 `pyproject.toml` (implemented)

The authoritative text is `pyproject.toml` on the branch; this section records its properties so the
plan does not carry a second copy that can rot.

- **`requires-python = ">=3.11,<3.12"`**, `.python-version` = `3.11.17`, `[tool.uv] python-preference =
  "only-managed"`: uv downloads the interpreter; a system or conda Python is never used.
- **`required-version = "==0.12.23"`**: a different uv refuses to run against the project.
- **Every direct dependency is an exact pin** at the version the pipeline was verified with (the
  test `tests/test_release_identity.py` enforces it). Base dependencies now include the server
  (fastapi, uvicorn, python-multipart, sse-starlette), the ECT stage, the postprocessing tools and
  `huggingface_hub`, so one `--extra` word is the whole install. `server = []` is kept as an empty
  alias so `.[cpu,dev,server,test]` installs written before the fold (CI) still resolve; it goes with
  the Phase C CI rewrite.
- **Hardware extras `gpu` / `cpu` / `macos`**, mutually exclusive through `[tool.uv] conflicts`.
  `gpu` = `onnxruntime-gpu==1.20.2`, `nvidia-ml-py` (provides the `pynvml` module; the deprecated
  `pynvml` shim is gone) and the six nvidia CUDA 12 wheels, all marked `sys_platform != 'darwin'`.
  `cpu` = `onnxruntime==1.20.1`. `macos` = `onnxruntime==1.20.1; sys_platform == 'darwin'`.
- **`environments` and `required-environments`** both list exactly Linux x86_64, Windows AMD64 and
  macOS arm64. uv resolves only for those and **refuses to lock** if any package lacks a wheel for
  one of them, so a missing wheel is a lock-time error on the lab box, never an install-time surprise
  on a user's machine. Verified in the lock: every production package has wheels for all three; the
  GPU-only packages have none for macOS, by design.
- **Dev group `full`** (Linux only): `torch==2.6.0`, `torchvision==0.21.0` (both from the PyTorch cu124
  index, which serves only this group), `ultralytics==8.4.107`, `onnx==1.22.0`, and
  `opencv-python==5.0.0.93` pinned to the SAME version as `opencv-python-headless` because both install
  the `cv2` directory and a different version would change resize/decode numerics in dev.
- **One lock, one version per package.** A consequence worth knowing: the dev group constrains shared
  transitive packages for production too. Today that pins `sympy` to 1.13.1 (torch's requirement);
  onnxruntime only uses sympy in offline tooling, so it is inert at inference.

### 2.2 Files added at the repo root

```text
.python-version        3.11
uv.lock                committed, generated by `uv lock`, never hand-edited
.dockerignore          mirrors .gitignore plus models/, runs/, logs/, .venv*, app/node_modules
```

### 2.3 `tools/release/versions.env`

One file holds every pinned tool and base-image identity so there is exactly one place to bump them:

```bash
PYTHON_VERSION=3.11.17
UV_VERSION=0.12.23
PYTHON_IMAGE=python:3.11-slim-bookworm@sha256:<digest>
UV_IMAGE=ghcr.io/astral-sh/uv:<x.y.z>@sha256:<digest>
APPTAINER_VERSION=<x.y.z>
CUDA_MIN_DRIVER_LINUX=525.60.13
CUDA_MIN_DRIVER_WINDOWS=528.33
```

`INSTALL.md`, both Dockerfiles, the CI workflow, and `lm3 doctor` read or are generated from this
file. A test (`tests/test_release_identity.py`) asserts the values in `INSTALL.md`, the Dockerfiles,
`pyproject.toml`'s `required-version`, and the doctor's constants agree with it.

## 3. `lm3 doctor`: the install-time gate

`leafmachine3/doctor.py`, registered in `cli.py` as `doctor`. It is the last step of every install
route and the first step of every Slurm job. Exit code 0 means the environment will run the
pipeline on the requested accelerator. Any other exit code names one cause, in one sentence, with the
fix.

### 3.1 Checks, in order, stop at first failure (implemented: `leafmachine3/doctor.py`)

| # | Check | Passes when | Failure message names |
|---|---|---|---|
| 1 | Interpreter | Python 3.11 inside a virtual environment. A different patch or a non-uv interpreter is a **warning** (Will's conda-based dev venv), not a failure | the interpreter found and the `uv sync` command |
| 2 | Hardware variant | exactly one of `onnxruntime-gpu`, `onnxruntime` is installed (`LM3_EXTRA`, when set by an image, must agree) | "both onnxruntime X and onnxruntime-gpu Y are installed" and the one-line repair |
| 3 | Environment contract | every package in `_env_contract.json` for that variant whose marker applies here is installed at exactly the locked version | up to three offenders with found/expected versions, and `uv sync --frozen --extra <variant>` |
| 4 | Platform | (sys.platform, machine) is supported for the variant | the unsupported pair and the supported list |
| 5 | NVIDIA driver (`gpu` only) | `CUDA_VISIBLE_DEVICES` does not hide every GPU; NVML starts; at least one GPU; driver >= 525.60.13 (Linux) / 528.33 (Windows), compared numerically | which of those failed, and the CPU variant as the alternative |
| 6 | Accelerator | in a CHILD process that goes through machine3's own CUDA library-path step, a session on the embedded 1 KB Conv+ReLU probe model binds the variant's provider (CUDA / CPU / CoreML MLProgram) AND runs, agreeing with a CPU reference within 1e-3 | requested vs bound provider, the library onnxruntime could not load (parsed from its stderr), and a hint when `LM3_CUDA_LIBPATH_SET` is set |
| 7 | Models (`--models`) | the model installer's status reports nothing missing or outdated | the actions and `lm3 models install` |
| 8 | Write access | the runtime state dir and the model folder (or their nearest existing parents) are writable | the path |

A development environment (torch / ultralytics present) is a note, or a failure with `--production`.

Check 6 is the one that catches the LM2-class silent CPU fallback. ORT reports
`CUDAExecutionProvider` as available whenever the GPU build is installed; it only fails when the
session is created. This is not hypothetical: on 2026-10-06 the lab venv ran three full pipeline
passes on the CPU provider after a stray `pip install onnxruntime` replaced the GPU binary, with
`get_available_providers()` still listing CUDA on the hardware profile. Nothing in the pipeline
noticed; only stage wall-clock and a 0 MB VRAM column in the timing report gave it away. The doctor creates the session in a subprocess that has gone through the same
LD_LIBRARY_PATH re-exec as `machine3`, so what it tests is what the pipeline will do.

`cpu` and `macos` installs skip check 5; check 6 confirms `CPUExecutionProvider` /
`CoreMLExecutionProvider` binds and runs. There is no
GPU probe and no warning about a GPU that might be present; the user chose `cpu`.

### 3.2 Output

Human-readable by default, one line per check with `ok` or `FAIL`, then a summary block:

```text
LeafMachine3 3.0.0   extra=cuda   python=3.11.x (uv-managed)   uv.lock=sha256:ab12...
GPU: NVIDIA RTX 6000 Ada   driver 560.35.05   onnxruntime-gpu 1.20.2 (CUDAExecutionProvider)   cuDNN 9.1.0.70
Result: READY
```

`--json` emits the same as one object. `lm3 serve` exposes it at `/healthz/doctor` so the GUI can
display the identical verdict.

### 3.3 Where else the checks run

- `machine3` and `lm3 serve` run checks 1 to 4 at startup (cheap, no sessions) and refuse to start on
  failure: `leafmachine3.doctor.startup_gate()` exists and is tested; the two call sites are wired
  once the uncommitted runtime work in `machine3.main` and `server/app.py` is committed. The existing `ensure_hardware_profile` and the fail-loud `make_session` keep covering the
  per-run case.
- The Slurm templates run `lm3 doctor` as their first command so a bad allocation fails in seconds.
- The Docker images use `lm3 doctor --cpu-only-checks` as their `HEALTHCHECK` and the full doctor as
  the documented first command.

### 3.4 `_env_contract.json` (implemented)

Generated by `tools/release/write_env_contract.py`, never edited by hand. It runs `uv export --frozen
--extra <variant>` once per hardware variant, so the contract inherits uv's own resolution and marker
evaluation, and records for each variant every production package with its exact version and
environment marker (69 for `gpu`, 61 for `cpu`, 60 for `macos`), plus the Python patch, uv version and
driver minimums from `versions.env`, and the list of dev-only package names. `--check` exits 1 when
the committed file is stale; `tests/test_release_identity.py` runs it together with `uv lock --check`.

## 4. Native installation (uv)

This is the whole of the new `INSTALL.md` user-facing content, per platform. Nothing else is
documented as supported.

### 4.1 Get uv (pinned version)

Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/<UV_VERSION>/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/<UV_VERSION>/install.ps1 | iex"
```

Then close and reopen the terminal. `uv --version` must print `<UV_VERSION>`. The installer pins the
version in the URL, so a newer uv with different resolution behavior is never picked up by accident.
`required-version` in `pyproject.toml` makes a wrong uv refuse to run.

### 4.2 Get LeafMachine3

```bash
git clone --branch v3.0.0 --depth 1 https://github.com/Gene-Weaver/LeafMachine3.git
cd LeafMachine3
```

Releases are tags. `main` is not an install target.

### 4.3 Install and verify

Pick one line for the hardware. uv downloads Python 3.11, creates `.venv`, and installs exactly the
lock. It never consults a system Python and never resolves.

```bash
uv sync --frozen --extra gpu      # Linux or Windows with an NVIDIA GPU
uv sync --frozen --extra cpu       # Linux or Windows, no GPU
uv sync --frozen --extra macos       # macOS Apple Silicon
uv run lm3 doctor
```

`uv run lm3 models fetch` then populates the model cache (see `DEPLOYMENT_PLAN.md` section 7) and
`uv run lm3 doctor --models` confirms the hashes. From then on:

```bash
uv run lm3 serve                   # browser GUI at http://127.0.0.1:<port>
uv run machine3 --config LM3_settings.yaml
```

No activation step. `uv run` always executes inside `.venv`; a user who types `python` instead of
`uv run` gets their system Python, which does not have LM3, which is a loud failure and not a subtle
one.

### 4.4 Offline install from the wheelhouse

For a machine without index access, or when a release is old enough that the indexes are not trusted:

```bash
# download lm3-wheelhouse-3.0.0-linux-x86_64-cuda.zip and its .sha256 from the release page
sha256sum -c lm3-wheelhouse-3.0.0-linux-x86_64-cuda.zip.sha256
unzip lm3-wheelhouse-3.0.0-linux-x86_64-cuda.zip
UV_PYTHON_INSTALL_MIRROR=file://$PWD/wheelhouse/python uv python install 3.11   # interpreter is in the archive too
uv sync --frozen --extra gpu --offline --no-index --find-links ./wheelhouse
uv run lm3 doctor
```

The lock's hashes are checked against the wheelhouse files, so a tampered or partial archive fails
at sync.

### 4.5 Upgrading

```bash
git fetch --tags && git checkout v3.1.0
uv sync --frozen --extra gpu
uv run lm3 doctor
```

`uv sync` removes packages the new lock no longer lists. There is no in-place `pip install -U` path.

## 5. Container images

### 5.1 Images

```text
ghcr.io/gene-weaver/lm3-runtime-cpu:<version>     also tagged by immutable digest
ghcr.io/gene-weaver/lm3-runtime-cuda:<version>
```

Both are built from `deploy/docker/Dockerfile` with a build arg `LM3_EXTRA={cpu,gpu}`. One
Dockerfile, two targets, same lock as the native install.

```dockerfile
# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE
ARG UV_IMAGE
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS build
ARG LM3_EXTRA
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_PREFERENCE=only-managed UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/lm3
WORKDIR /src
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ${LM3_EXTRA}
COPY leafmachine3 ./leafmachine3
COPY README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ${LM3_EXTRA}

FROM ${PYTHON_IMAGE} AS runtime
ARG LM3_EXTRA
RUN useradd --create-home --uid 1000 lm3
COPY --from=build /opt/lm3 /opt/lm3
ENV PATH=/opt/lm3/bin:$PATH LM3_EXTRA=${LM3_EXTRA} \
    LM3_RUNTIME_DIR=/workspace/runtime LM3_MODEL_CACHE=/workspace/model-cache
USER lm3
WORKDIR /workspace
VOLUME ["/workspace/input", "/workspace/output", "/workspace/runtime", "/workspace/config", "/workspace/model-cache"]
HEALTHCHECK --interval=60s CMD lm3 doctor --quick || exit 1
ENTRYPOINT ["lm3"]
CMD ["--help"]
```

Design points:

- **Base is plain Python, not `nvidia/cuda`.** CUDA 12 runtime, cuBLAS, and cuDNN 9 arrive as pip
  wheels through the lock, identical to the native install. The host contributes only the driver via
  `--gpus all` (NVIDIA Container Toolkit) or Apptainer `--nv`. This keeps one dependency source and
  makes the image several gigabytes smaller than a CUDA devel base.
- **No models in the image.** `/workspace/model-cache` is a mount. Measured on the torch-free venv:
  the `gpu` site-packages are 3.6 GB installed (nvidia wheels 1.9 GB, onnxruntime-gpu 0.8 GB), so
  the image is about 3.5 GB; `cpu` is about 1 GB. Models would double it and tie model versions to
  image versions.
- **No editable install, no source tree.** The runtime stage has `/opt/lm3` and nothing else.
- **Non-root, fixed UID 1000.** Apptainer maps the invoking user anyway; Docker users get sane file
  ownership on the output mount.
- Both images embed `LM3_EXTRA` so the doctor knows which contract to enforce.

### 5.2 Running with Docker (Linux only)

```bash
docker run --rm --gpus all \
  -v "$PWD/input:/workspace/input:ro" -v "$PWD/output:/workspace/output" \
  -v "$HOME/.lm3/runtime:/workspace/runtime" -v "$HOME/.lm3/models:/workspace/model-cache" \
  -v "$PWD/LM3_settings.yaml:/workspace/config/LM3_settings.yaml:ro" \
  ghcr.io/gene-weaver/lm3-runtime-cuda:3.0.0 doctor
```

The first run is `doctor`. If it fails because the host has no NVIDIA Container Toolkit, the message
says so and links the toolkit install page. `machine3 --config /workspace/config/LM3_settings.yaml`
and `serve --host 0.0.0.0 --port 8123` are the other two commands a user needs; `deploy/docker/`
ships a `compose.yaml` that encodes the mounts so the command line above is never typed by hand.

### 5.3 Apptainer (clusters)

CI builds the SIF from the OCI image on the same runner and pushes it with ORAS:

```text
oras://ghcr.io/gene-weaver/lm3-runtime-cuda:3.0.0-sif
oras://ghcr.io/gene-weaver/lm3-runtime-cpu:3.0.0-sif
```

A cluster user never converts an image. They pull a finished file into scratch:

```bash
module load apptainer                                   # site-specific
export APPTAINER_CACHEDIR=/scratch/$USER/apptainer      # home quotas are small
apptainer pull /scratch/$USER/lm3-runtime-cuda-3.0.0.sif oras://ghcr.io/gene-weaver/lm3-runtime-cuda:3.0.0-sif
```

Then prefetch models on a login node with network, since compute nodes often have none:

```bash
apptainer exec --bind /scratch/$USER/lm3-models:/workspace/model-cache \
  /scratch/$USER/lm3-runtime-cuda-3.0.0.sif lm3 models fetch
```

### 5.4 Slurm templates

`deploy/slurm/lm3_gpu.sbatch` and `deploy/slurm/lm3_cpu.sbatch`. The GPU one:

```bash
#!/bin/bash
#SBATCH --job-name=lm3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
set -euo pipefail
SIF=/scratch/$USER/lm3-runtime-cuda-3.0.0.sif
BIND="--bind $INPUT:/workspace/input:ro,$OUTPUT:/workspace/output,$RUNTIME:/workspace/runtime,$MODELS:/workspace/model-cache,$CONFIG:/workspace/config/LM3_settings.yaml:ro"
apptainer exec --nv $BIND "$SIF" lm3 doctor --models      # fail in seconds, not after a 6-hour queue wait
apptainer exec --nv $BIND "$SIF" machine3 --config /workspace/config/LM3_settings.yaml
```

The five bind variables are set at the top of the file with placeholder paths and a comment on each.
The CPU template is identical without `--gres` and `--nv`. The GUI-through-SSH-tunnel route in
`DEPLOYMENT_PLAN.md` section 10 is a third template, `lm3_serve.sbatch`, and is secondary; most
cluster use is headless batch.

### 5.5 What a cluster user needs to know

Documented in `deploy/apptainer/README.md`, kept to one page:

- Apptainer or Singularity 3.8+ (`singularity` works as a drop-in for every command above).
- `--nv` requires the node to have an NVIDIA driver at or above the minimum; the doctor checks it
  inside the container exactly as it does natively.
- Where to put the SIF, cache, and models (scratch, not home).
- The three commands: pull, fetch, sbatch.
- That a native `uv` install into scratch also works and is supported on clusters that allow
  outbound HTTPS from login nodes. The container is preferred because it also freezes glibc and
  OpenCV's system libraries.

## 6. Wheelhouse

### 6.1 Build

`tools/release/build_wheelhouse.py`, run by the release workflow for each target:

| Target | Platform tag | Extra |
|---|---|---|
| linux-x86_64-cuda | manylinux_2_28_x86_64 | cuda |
| linux-x86_64-cpu | manylinux_2_28_x86_64 | cpu |
| windows-x86_64-cuda | win_amd64 | cuda |
| windows-x86_64-cpu | win_amd64 | cpu |
| macos-arm64-mac | macosx_13_0_arm64 | mac |

Steps per target:

1. `uv export --frozen --extra <extra> --no-dev --format requirements-txt --no-emit-project -o req.txt`
   (fully pinned, with hashes, for that extra).
2. `pip download -r req.txt --no-deps --only-binary=:all: --python-version 3.11 --implementation cp
   --platform <tag> --dest wheelhouse/` with the two PyTorch indexes as `--extra-index-url`. Hashes
   from the export are verified by pip.
3. Build the project wheel into the same directory.
4. Add the matching python-build-standalone 3.11 archive that uv would download, so offline hosts
   need nothing from the network.
5. Write `MANIFEST.sha256` and zip.
6. `uv sync --frozen --offline --no-index --find-links wheelhouse/` in a clean container to prove the
   archive is complete; `lm3 doctor --cpu-only-checks` inside it.

### 6.2 Publish

Sizes: roughly 2 GB of wheels per CUDA target (3.6 GB installed), 0.7 GB per CPU target, 0.6 GB for
macOS. GitHub release assets are capped at 2 GB per file, which the CUDA archive sits right at, so the archives are published to the `leafmachine3/lm3-wheelhouse`
Hugging Face dataset repository (the same organization that hosts the models; no per-file size
problem) under `v<version>/<target>/`. The GitHub release page carries each archive's sha256 and
link. The release is not marked published until all five archives and the doctor proof in step 6
exist.

### 6.3 When the wheelhouse is used

It is the fallback, not the daily path. The index install already verifies every hash, so the only
thing the wheelhouse adds is independence from PyPI and download.pytorch.org being reachable and
intact. That is what makes a 2026 release installable in 2029.

## 7. macOS

- Target: Apple Silicon, macOS 13 or newer. The `macos` extra installs the standard `onnxruntime`
  wheel, which includes `CoreMLExecutionProvider`. No torch, no special index, no Rosetta.
- Doctor check 6 creates a CoreML session with `ModelFormat: MLProgram` and runs it.
- With ultralytics gone, the four YOLO models go through LM3's own provider ladder like every other
  model, so CoreML is reachable for all of them from the `.onnx` export alone. Under ultralytics the
  ONNX path offered only CUDA or CPU and CoreML needed a separate `.mlpackage` per model.
- Moving every model to ONNX on the CoreML provider, which is the stated direction, changes nothing
  in this plan: the dependency set already has the provider. It is model-export work tracked in
  `DEPLOYMENT_PLAN.md` section 7 (per-platform artifact selection in `models.lock.yaml`, since the
  leaf and specimen segmenters already have `.mlpackage` exports alongside `.onnx`).
- Docker on macOS is not a route. Docker Desktop cannot reach the GPU, so it would only ever offer
  CPU and would teach users to expect the slow path.

## 8. Windows

- The `gpu` and `cpu` extras work unchanged; cu124 wheels exist for `win_amd64`.
- Minimum driver 528.33 (CUDA 12.4 on Windows). Doctor check 5 uses that constant.
- DLL loading: `onnxruntime-gpu` needs cuDNN 9 and cuBLAS on the loader path. With torch gone the
  DLLs live under the `nvidia/*/bin` package directories; `machine3`'s loader-path step (the
  LD_LIBRARY_PATH re-exec on Linux) gets a Windows branch that calls `os.add_dll_directory` on each
  of them before onnxruntime is imported. The doctor exercises it (check 6). To be confirmed on a
  Windows machine in Phase B; it is the one platform-specific unknown left in this plan.
- Docker Desktop and WSL2 are not a supported route. uv is.
- The Electron backend resolver in `app/main.js` already prefers a `lm3` console script and venv
  prefixes. It gains `<root>/.venv` as a candidate so a uv checkout is found without `VIRTUAL_ENV`.

## 9. CI

`.github/workflows/ci.yml` is rewritten. The pip and `setup-python` matrix goes away.

```yaml
jobs:
  lock-is-current:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { version: "<UV_VERSION>" }
      - run: uv lock --check
      - run: uv run --frozen python tools/release/write_env_contract.py --check

  test:
    needs: lock-is-current
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: ubuntu-latest,  extra: cpu }
          - { os: windows-latest, extra: cpu }
          - { os: macos-14,       extra: macos }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: { version: "<UV_VERSION>" }
      - run: uv sync --frozen --extra ${{ matrix.extra }} --extra dev
      - run: uv run lm3 doctor --no-models
      - run: uv run pytest tests -q -rfE   # existing ratchet and baseline gates unchanged

  gpu:
    needs: lock-is-current
    runs-on: [self-hosted, linux, nvidia]      # the lab workstation
    steps:
      - uses: actions/checkout@v4
      - run: uv sync --frozen --extra gpu --extra dev
      - run: uv run lm3 doctor --models
      - run: uv run pytest tests -q -m "gpu"
      - run: uv run machine3 --config examples/LM3_settings.ci.yaml   # 3-image end-to-end

  images:
    needs: [test, gpu]
    runs-on: ubuntu-latest
    steps:
      - build both images with buildx, args from tools/release/versions.env
      - run lm3-runtime-cpu doctor (CPU) as the image smoke test
      - on a tag: push to ghcr by version, record digests, generate SBOM (syft), build SIFs, oras push
```

The GPU job is the only place `--extra gpu` is exercised end to end, so it is required for merge on
`main`, not advisory. The `images` job builds on every PR and pushes only on tags.

## 10. Release checklist

A release is the tag plus these artifacts, all produced by the `release` workflow from the tag:

1. `uv.lock` unchanged since the last green environment-bump PR (the workflow diffs it against the
   previous tag and fails if it changed outside a PR labeled `env-bump`).
2. `_env_contract.json` matches the lock.
3. Five wheelhouse archives on Hugging Face with sha256 on the release page.
4. Two OCI images on ghcr tagged `<version>` with digests on the release page.
5. Two SIF images on ghcr via ORAS.
6. `models.lock.yaml` revision recorded on the release page.
7. `INSTALL.md` version strings (`<UV_VERSION>`, tag) regenerated from `versions.env`.
8. A clean-account rehearsal: a fresh Linux user, a fresh Windows user, and a fresh macOS user each
   follow `INSTALL.md` verbatim and paste their `lm3 doctor` summary into the release PR. No release
   without three READY summaries.

## 11. Dependency change policy

- Only an `env-bump` PR may modify `pyproject.toml` dependency pins or `uv.lock`. It must state what
  changed and why, pass the full matrix including the GPU job, and update `versions.env` if the uv
  or Python patch moved.
- Dependabot and Renovate are configured to ignore `pyproject.toml` and `uv.lock` (`app/package.json`
  can keep its own policy).
- A security advisory in a pinned package is handled by an `env-bump` PR, not by loosening a pin.
- Driver minimums only ever rise, and only with a CUDA wheel change.

## 12. Implementation phases

### Phase A: lock and contract -- DONE (branch `packaging-uv-doctor`, not merged)

- `pyproject.toml` per section 2.1, `.python-version`, `tools/release/versions.env`, `uv.lock`
  (100 packages, three supported environments), `leafmachine3/_env_contract.json` and its generator.
- **Deferred to Phase C** because they change user-facing docs or CI together: deleting
  `requirements/` (INSTALL.md and README still point at it), the CI rewrite, removing the 3.10 shims.
  Until Phase C, the never-run CI workflow's 3.10 and 3.12 legs would fail against the 3.11-only
  `requires-python`.

**Exit, met 2026-10-06:**
- `uv sync --frozen --extra gpu` builds in 50 s on uv-managed Python 3.11.17 with no torch or
  ultralytics.
- Installed files are byte-identical to the torch-free venv already proven against the GPU baseline:
  7,347 files over 40 shared distributions, 0 hash differences (only `nvidia-ml-py` and the inert
  `sympy` differ in version).
- The pipeline on `examples/images` from the lock-built environment, pinned to GPU 0 with 2 workers
  while both GPUs were busy with training, matches the ultra_rep run from the old venv in **all 14
  tables** (`tools/verification/compare_run_databases.py`).
- `uv sync --frozen --extra gpu --group full` reproduces `.venv_LM3`: every direct pin byte-identical
  (torch 2.6.0+cu124, ultralytics 8.4.107, onnxruntime-gpu, nvidia libs, numpy, OpenCV); 29
  transitive packages resolve to newer patch releases than the months-old pip install, the same
  ones the verified production environment uses.

### Phase B: `lm3 doctor` -- DONE (branch `packaging-uv-doctor`)

- `leafmachine3/doctor.py` (checks 1-8, `--json`, `--quick`, `--models`, `--production`, `--config`),
  `leafmachine3/doctor_probe.py` (the child-process probe with the embedded model, and a placement
  mode), `lm3 doctor` in `cli.py`.
- Startup gate: `machine3` and `lm3 serve` run checks 1-4 before starting and exit 78 with the fix
  on failure; `LM3_STARTUP_GATE=0` bypasses (the test suite pins it off). `GET /healthz/doctor`
  (authenticated; `?full=1` adds the GPU probe).
- Check 3 also verifies every installed file still exists (shared-file clobbering, section 14).
- Check 7 (`--models`) also names any installed ONNX model with ops that have no CUDA kernel.

### Phase C: INSTALL.md and CI

- Rewrite `INSTALL.md` to section 4 only, generated strings from `versions.env`.
- Rewrite `ci.yml` per section 9, register the lab workstation as the self-hosted GPU runner.
- `tests/test_release_identity.py`.

**Exit:** CI is green on all four legs from a push to the GitHub remote.

### Phase D: containers

- `deploy/docker/Dockerfile`, `compose.yaml`, `.dockerignore`.
- `deploy/apptainer/README.md`, `deploy/slurm/*.sbatch`.
- Image build and SIF push in the workflow.

**Exit:** `docker run --gpus all ... doctor` is READY on the lab box; the SIF pulled from ghcr runs
`doctor --models` READY inside a Slurm allocation on Great Lakes with models prefetched on the login
node.

### Phase E: wheelhouse and release workflow

- `tools/release/build_wheelhouse.py`, the five targets, the offline proof.
- Hugging Face dataset repo, release page generator.
- Section 10 checklist encoded as workflow steps.

**Exit:** a tagged release produces every artifact in section 10 with no manual step other than the
three rehearsal summaries.

### Phase F: Electron hookup

- `app/main.js` resolver gains the `.venv` candidate; the packaged app's first-run flow runs the
  section 4 commands and shows the doctor summary before offering to start the server.

**Exit:** a packaged Electron app on a clean machine with uv installed reaches READY without a terminal.

## 13. Developer environment (not shipped)

The ultralytics removal that section 2.1 assumes is done: `leafmachine3/inference/ultra_replacements.py`
replaced `ultralytics.YOLO` for the four end2end ONNX exports (commit 845a08e, 2026-10-06), verified
bitwise against the GPU baseline at the predict level and across all 14 run tables, with the
torch-free venv reproducing the normal venv exactly. The former section 13 ("deferred") is resolved.

What still needs torch and ultralytics is **development on Will's machine**, not the product:

- the ultralytics-parity tests in `tests/test_ultra_replacements.py` (they `importorskip`),
- `tools/verification/ab_ultralytics_vs_ultra_rep.py`,
- experiments under `leafmachine3/modules/experiments/`, calibration work, anything that opens a
  `.pt`.

That is the `full` dependency group. Rules:

- It is locked in `uv.lock` so it is reproducible, but **no user instruction, Dockerfile, wheelhouse
  target or CI user leg ever installs it.** The doctor reports "development environment" when it
  detects torch or ultralytics and the install was not asked for the group.
- The PyTorch index in `pyproject.toml` exists only to serve the group. If the group is ever dropped,
  the index entries go with it.
- Training stacks (SAM3, BiRefNet, exporters) are NOT the `full` group. They keep their own
  environments per `DEPLOYMENT_PLAN.md` section 5.

Two facts the group must carry forward:

- **Division convention.** `ultra_rep` reproduces torch's CUDA reciprocal-multiply for `im /= 255`
  and `/= gain` (see the module docstring). Any future A/B against ultralytics must be run on a CUDA
  device, or the comparison will show 0.02 confidence drift that is ultralytics' CPU/GPU
  inconsistency, not a regression.
- **The check_requirements trap.** On a CPU device, ultralytics' ONNX backend runs
  `pip install onnxruntime` into the active environment, replacing `onnxruntime-gpu`'s binary. This
  happened on 2026-10-06 and silently moved three pipeline runs to the CPU provider. In the `full`
  environment, run ultralytics only with `device="cuda:0"`, or monkeypatch
  `ultralytics.nn.backends.onnx.check_requirements` first.

## 14. Install-test findings, 2026-10-07 (all resolved on the branch unless noted)

Running the documented install from a fresh clone, and a dev <-> production switch, found:

- **Shared-file clobbering.** Uninstalling one package deleted files another still needed while
  its metadata looked clean: the nvidia-* wheels share `nvidia/__init__.py` (the CUDA library lookup
  used `nvidia.__file__` and silently found nothing -> CPU), and ultralytics' `opencv-python` shares
  `cv2/` with production's headless build. Fixed: lookups use `nvidia.__path__`
  (`core/cuda_libs.py`); `opencv-python` is removed by `[tool.uv] override-dependencies`; doctor check
  3 stats every RECORD file.
- **Pillow's decompression-bomb limit** quarantined every sheet over 179 MP as "corrupt". Raised to
  1 GP in one place (`core/imaging.py`); oversized images are now quarantined as `too_large`.
- **Hardware profile fingerprint** used size+mtime, so every re-download looked "changed". Now
  content hashes (cached), over every file a stage loads, naming each difference; legacy entries
  migrate in place. The leaf segmenter's warning is genuine (profile tuned on the .pt): rerun
  `python -m leafmachine3.setup` on an idle GPU.
- **onnxruntime "Memcpy nodes" per worker.** Two causes, both fixed at the source and patched on
  the Hub (2026-10-07; each repo's `manifest.json` has a `patches` record, cards unchanged):
  - YOLO26 exports were opset 19, and onnxruntime 1.20's CUDA provider has no opset-19 Resize kernel,
    so every upsampling layer ran on the CPU. Patched with `tools/modelhub/fix_resize_opset.py`
    (bitwise identical on CPU and CUDA) in archival x/n, plant x/n, landmark, leaf segmenter and the
    YOLO26x-seg specimen alternate. Exporters now pass `opset=18` and refuse a higher opset
    (`LM3_ONNX_OPSET` / `assert_onnx_opset` in each training project).
  - The DINOv2 ruler member's SDPA export computed the attention scale from tensor shapes in all 12
    layers. Re-exported with eager attention (1.4e-5 from the trained model; same argmax); the
    exporter now does this itself and rejects a shape-derived Sqrt.
  - The lock pins the patched revisions. With them, `lm3 doctor --models` reports every ONNX model
    entirely on the GPU, the pipeline matches the previous models in all 14 tables, and a default run
    took 56 s instead of 67 s. LM3's sessions log at ERROR; `LM3_ORT_LOG_SEVERITY` re-enables
    placement logs for debugging.
- **Retired settings** warned once per specimen; now once per run from machine3
  (`core.config.RETIRED_SETTINGS`). The shipped `LM3_settings.yaml` is the clean template and lists
  every live setting.
- **Tooling:** ruff pinned to the lint baseline's 0.12.1; `setuptools` in the test extra; CI on the
  release Python (3.11) only; the known-failing test baseline is empty; the Electron unit tests no
  longer need `npm install`; the conversion-factor fit data installs as optional provenance.
