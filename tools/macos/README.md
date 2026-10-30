# LeafMachine3 on a Mac: Core ML

On a Mac, LM3 runs its detectors and segmenters on the Apple GPU with Core ML itself. Each model has a
`coreml_fp32` package beside its ONNX file. Everything else runs on the CPU.

**The requirement:** a Mac must produce the same output as the CPU, to float32 rounding. Faster but
different does not pass.

What a `coreml_fp32` package is (made by `tools/modelhub/export_coreml_fp32.py`):

- **float32 weights and arithmetic**, run on CPU+GPU. The Neural Engine computes in float16, so LM3
  never uses it. The older `coreml/` folders on Hugging Face are float16; LM3 never runs those.
- **The CPU's exact input tensor.** LM3's own preprocessing feeds the package; there is no Core ML
  image input.
- **Every input size LM3 uses, in one package.** A YOLO model gets the letterbox it gets everywhere
  (long side `imgsz`, short side any multiple of 32): 79 sizes for the detectors, 63 for the leaf
  segmenter, 39 for landmarks.
- **No top-k inside the model.** YOLO26 networks stop where their end2end top-k step begins, and LM3
  runs that step in numpy, identically to the ONNX graph.
- **Attention rewritten for Apple's GPU compiler.** MPSGraph aborts on Ultralytics' `v @ attn^T`, so
  the exporter writes `(attn @ v^T)^T`, which is bit-identical in torch.

Six default models have packages: archival, plant, leaf segmenter, landmarks, the UNet++ specimen
segmenter and the YOLO26x ruler member. The YOLO26n and DINOv2 ruler members stay on the CPU (no
faster on the GPU). On a Mac, `lm3 models install` downloads ONNX and `coreml_fp32` (3.0 GB in all);
`uv sync --extra macos` installs `coremltools`.

**Memory:** read `/usr/bin/time -l`'s **peak memory footprint** line, not maximum resident set size,
which under-counts Core ML's memory. `time` must wrap Python, not uv, so the commands below put it
after `uv run ...`.

**Shape changes:** Core ML spends about 150-250 ms whenever the input shape changes, so the leaf
segmenter and landmark stages predict each sheet's crops grouped by shape and assemble the rows in the
original order.

## Checking a fresh install

Exactly what a new user does, in a new folder, so the packages come from Hugging Face:

```bash
git clone --branch macos-gpu-support https://github.com/Gene-Weaver/LeafMachine3.git LM3_fresh
cd LM3_fresh
git log --oneline -1                                   # the commit you are testing
uv sync --frozen --extra macos
uv run --frozen --no-sync lm3 models install --yes
uv run --frozen --no-sync lm3 models status            # every stage ok; formats: onnx, json, coreml_fp32
sysctl -n machdep.cpu.brand_string hw.memsize          # chip and RAM (bytes), for the report
```

Then:

```bash
uv run --frozen --no-sync lm3 doctor                   # check 6: "Core ML (coremltools 9.0) ran the probe network ..."
uv run --frozen --no-sync lm3 doctor --models          # check 7: all 6 Core ML models agree with the CPU
uv run --frozen --no-sync /usr/bin/time -l python tools/macos/verify_coreml.py --n 23 --json native.json

# full runs, one after another
LM3_COREML=0 uv run --frozen --no-sync /usr/bin/time -l machine3 --config LM3_settings.yaml --run-name mac_cpu
uv run --frozen --no-sync /usr/bin/time -l machine3 --config LM3_settings.yaml --run-name mac_native
grep -E "GPU plan|CoreML" runs/mac_native/logs/lm3.log | head -40
uv run --frozen --no-sync python tools/verification/compare_runs_close.py runs/mac_cpu runs/mac_native --json cmp_native.json
```

In the run log expect `GPU plan: Apple GPU via CoreML, 1 worker` for each GPU stage, one
`coreml_fp32 loaded` line per model, and no `runs on the CPU` or `has no input shape` lines.

## Acceptance

`mac_native` vs `mac_cpu`. The inputs are identical, so any difference is Core ML's arithmetic.

| Check | Pass | M2 Pro, b724715 |
|---|---|---|
| `lm3 models status` | every stage ok, `coreml_fp32` in the formats | |
| `lm3 doctor` / `--models` | READY; check 7 OK on all 6 | 6/6 |
| `verify_coreml.py --n 23` | 6/6 `ok`, all on CoreML, no inputs on the CPU | 6/6 |
| run exit | exit 0, `stage_errors.csv` empty | |
| detections | every box matched (conf >= 0.4: all); IoU min >= 0.99; abs(dconf) max <= 0.03 | all; 1.000; 0.000 |
| leaves | all measured leaves matched; every metric median <= 0.0005, p95 <= 0.01 | 257/257; 0.0000 |
| sheets | every count the same | 23/23 |
| ruler | unit the same on every sheet; px/cm relative change max <= 0.005 | 23/23; 0.0000 |
| specimen mask | area fraction abs diff max <= 0.001 | 0.0001 |
| wall time | (b) well under (a) | 270 s vs 937 s |

## Making packages for a model release

`tools/modelhub/export_coreml_fp32.py` is the CoreML export for every model published to Hugging Face.
Run it on a Mac, so each package gets a load test there, from the published PyTorch checkpoints, with
the converter's dependencies in a throwaway uv layer (`.venv` is not changed):

```bash
uv run --frozen --no-sync lm3 models install --formats pytorch,meta --dest ~/lm3-checkpoints --yes
uv run --frozen --no-sync --with torch==2.5.1 --with ultralytics==8.4.79 --with segmentation-models-pytorch==0.5.0 \
  /usr/bin/time -l python tools/modelhub/export_coreml_fp32.py all --models models --pytorch ~/lm3-checkpoints
```

Each network is first compared with the published ONNX model, and a package that fails its load test
or disagrees is removed. Copy the `coreml_fp32` folders (in the models layout) to the machine that
publishes, then:

```bash
python tools/modelhub/publish_coreml_fp32.py stage --packages DIR --out STAGE \
  --verified-on "Apple M2 Pro, macOS 26.7.1" --commit <exporter commit>     # review the printed diffs
python tools/modelhub/publish_coreml_fp32.py upload --stage STAGE
python tools/modelhub/build_lock.py                                          # pins the new revisions
```
