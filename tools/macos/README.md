# Verifying native CoreML (`coreml_fp32`) on a Mac

Branch `macos-gpu-support`. On a Mac, LM3 now runs its GPU models with Core ML itself, from a
`coreml_fp32` package per model, instead of on the CPU.

**The requirement:** a Mac must produce the same output as the CPU (and as Linux/Windows), to float32
rounding. Faster but different does not pass.

What a `coreml_fp32` package is (made by `tools/modelhub/export_coreml_fp32.py`):

- **float32 weights and arithmetic.** The existing `coreml/` downloads are float16; LM3 never runs those.
- **The CPU's exact input tensor.** There is no CoreML image input: LM3's own preprocessing feeds it.
- **Every input size LM3 uses, in one package.** A YOLO model gets the letterbox it gets everywhere
  (long side `imgsz`, short side any multiple of 32). That is 79 sizes for the detectors, 63 for the
  leaf segmenter and 39 for landmarks, all in one compiled model.
- **No top-k inside the model.** YOLO26 networks stop where their end2end top-k step begins (CoreML got
  the boxes wrong there), and LM3 runs that step in numpy, identically to the ONNX graph.

The six default models get packages: archival, plant, leaf segmenter, landmarks, the UNet++ specimen
segmenter and the YOLO26x ruler member. The YOLO26n and DINOv2 ruler members stay on the CPU. A model
without a package runs on the CPU, as before.

The packages are not on Hugging Face yet. In this test you make them on the Mac from the published
PyTorch checkpoints, with the converter's dependencies in a throwaway uv layer. LM3's own environment,
its lock and the installed ONNX models are not changed. Every converted network is first compared
with the published ONNX model on the CPU, so a package that differs from the ONNX model is never
written.

The earlier runbook on this branch, for onnxruntime's CoreML route, is replaced by this one; do not
run it. That route is still selectable with `LM3_COREML_ROUTE=ort`, but it is not part of this test.

## 1. Check out the branch

From the LeafMachine3 folder you installed from the README:

```bash
git fetch origin
git switch macos-gpu-support
git pull
git log --oneline -1                                  # the commit you are testing
uv sync --frozen --extra macos
uv run --frozen --no-sync lm3 models install --yes    # no-op if the models are current
uv run --frozen --no-sync lm3 models status           # note the models folder it prints
sysctl -n machdep.cpu.brand_string hw.memsize         # chip and RAM (bytes), for the report
```

Below, `MODELS` is that models folder (usually `models` inside the LeafMachine3 folder):

```bash
MODELS=models
CML="--with coremltools==9.0"                                       # what LM3 needs to run packages
CONVERT="$CML --with torch==2.5.1 --with ultralytics==8.4.79 --with segmentation-models-pytorch==0.5.0"
```

`uv run --with ...` puts those packages in a temporary environment layered over LM3's. `.venv` is
not modified.

## 2. Make the six packages

```bash
uv run --frozen --no-sync lm3 models install --formats pytorch,meta --dest ~/lm3-checkpoints --yes
/usr/bin/time -l uv run --frozen --no-sync $CONVERT \
  python tools/modelhub/export_coreml_fp32.py all --models $MODELS --pytorch ~/lm3-checkpoints
```

The first command downloads the pinned checkpoints (about 1.4 GB) into a separate folder. The
second writes `$MODELS/<model>/coreml_fp32/` beside each ONNX model: `model.mlpackage` plus
`model.json`, the contract LM3 reads. For each model it prints:

- one `check` line per tested input size, comparing the traced network with the ONNX model;
- `converted in N s`, then `wrote ...`.

Expect `6 package(s) written`. Report every `check` line, the total time and any error. Deleting
the `coreml_fp32` folders undoes this step.

## 3. Model by model

```bash
uv run --frozen --no-sync $CML python tools/macos/verify_coreml.py --n 23 --json native_1.json
uv run --frozen --no-sync $CML python tools/macos/verify_coreml.py --n 23 --json native_2.json   # again: cached load
```

Each model gets an `[ok  ]` / `[warn]` / `[fail]` line comparing CoreML with the CPU, then a line with:

- **load:** the first run includes compiling the package; the second run loads it from the compile
  cache (`~/Library/Caches/LeafMachine3/coreml`);
- **first call per shape vs later calls:** shows whether Core ML does extra work the first time it
  sees each input size;
- **inputs run on the CPU:** must not appear. It means an input size the package does not list.

Expect `ok` on all 6 and `on CoreML` for all 6. `ok` means within float32 rounding of the CPU, the
same bar CUDA meets on Linux. If any model warns or fails, run that model once more on Core ML's CPU
(also float32). That separates the Apple GPU's arithmetic from the conversion:

```bash
LM3_COREML_UNITS=CPUOnly uv run --frozen --no-sync $CML python tools/macos/verify_coreml.py --n 23 --only <model> --json native_cpuonly.json
```

## 4. Doctor

```bash
uv run --frozen --no-sync $CML lm3 doctor
uv run --frozen --no-sync $CML lm3 doctor --models
```

Report check 7's line.

## 5. Full runs

One after another, not in parallel. `/usr/bin/time -l` reports wall time and peak memory.

```bash
# a. the CPU reference on this commit (CoreML off)
LM3_COREML=0 /usr/bin/time -l uv run --frozen --no-sync machine3 --config LM3_settings.yaml --run-name mac_cpu

# b. native CoreML
/usr/bin/time -l uv run --frozen --no-sync $CML machine3 --config LM3_settings.yaml --run-name mac_native

grep -E "GPU plan|CoreML" runs/mac_native/logs/lm3.log | head -40
uv run --frozen --no-sync python tools/verification/compare_runs_close.py runs/mac_cpu runs/mac_native --json cmp_native.json
```

In the run (b) log, expect:

- `GPU plan: Apple GPU via CoreML, 1 worker` for each GPU stage;
- one `CoreML: <model>/coreml_fp32 loaded in N s` per model;
- no `runs on the CPU` and no `has no input shape` lines.

## Acceptance

`mac_native` vs `mac_cpu`. The inputs are identical, so any difference is Core ML's arithmetic.

| Check | Pass | CUDA vs CPU (Linux) |
|---|---|---|
| packages | 6 written; every ONNX check within 1e-3 | Linux export: YOLO <= 6e-5, UNet++ 1.1e-4 |
| `verify_coreml.py --n 23` | 6/6 `ok`, all on CoreML, no inputs on the CPU | 6/6 |
| `lm3 doctor --models` | check 7 OK | |
| run exit | exit 0, `stage_errors.csv` empty | |
| detections | every box matched (conf >= 0.4: all); IoU min >= 0.99; abs(dconf) max <= 0.03 | all; 0.997; 0.014 |
| leaves | all measured leaves matched; every metric: median <= 0.0005, p95 <= 0.01 | 258/258; <= 0.0001; <= 0.0035 |
| sheets | every count the same (archetypal leaves may differ by 1 on up to 2 sheets) | same; 2 sheets differ by 1 |
| ruler | unit the same on every sheet; px/cm relative change max <= 0.005 | 23/23; 0.0015 |
| specimen mask | area fraction abs diff max <= 0.001 | 0.0003 |
| wall time | (b) well under (a); report both | |

## Report back

- The commit tested, the chip and the RAM.
- Step 2: every `check` line, the total time, any error.
- Both `verify_coreml.py` outputs and their JSON files, plus the CPU-only one if you ran it.
- The doctor `--models` line.
- Wall time and peak memory for runs (a) and (b).
- The `grep` output from the run (b) log.
- The comparison output and `cmp_native.json`.
