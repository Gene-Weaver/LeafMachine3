# Verifying the Apple GPU (CoreML) path

Branch `macos-gpu-support`. On a Mac with the `macos` extra, the GPU stages now run on the Apple GPU
through onnxruntime's CoreML execution provider instead of the CPU.

**The requirement:** a Mac must produce the same output as the CPU (and as Linux/Windows), to
float32 rounding. Faster but different does not pass. These steps check that, model by model and on
full runs, and measure what CoreML saves.

What changed, in short (details in `leafmachine3/inference/coreml.py`):

- **The executor:** on a Mac with CoreML, each GPU stage plans **one CoreML worker** in-process.
  Before, it planned CPU workers.
- **Which models use CoreML:**
  - the 4 YOLO26 models (archival, plant, leaf segmenter, landmarks);
  - the UNet++ specimen segmenter;
  - the YOLO26x ruler member.
  The YOLO26n and DINOv2 ruler members stay on the CPU.
- **The box bug:** the YOLO26 graphs are cut before their end2end top-k tail. CoreML returned
  wrong boxes from that tail. Now CoreML runs the network and numpy runs the tail. On the CPU,
  the cut graph plus numpy tail gives exactly the full graph's output (`tests/test_coreml.py`).
- **Same input as every platform:** every YOLO model gets exactly the tensor it gets on the CPU. The
  image is scaled so its long side is `imgsz`, and the short side is padded only to the next multiple
  of 32. So input shapes vary with each image's aspect ratio, and CoreML compiles one network per
  shape. Padding to a fixed square would avoid that but changes results: on the 23 examples,
  CPU-square vs CPU matched only 237/258 leaves. CoreML therefore never does it.
- **float32 throughout:** the default is the ML Program format on CPU+GPU, which keeps the models'
  float32 arithmetic. The Neural Engine computes only in float16, and the older NeuralNetwork format
  may use float16 on the GPU, so neither is tested here.
- **Varying shapes cost compile time or memory.** Each per-shape session holds a compiled copy of
  the network (about 240 MB of float32 weights for a YOLO26x model). Two ways, both exact:
  - default: one session per shape, 8 kept per model, least recently used dropped. Fewer sessions
    than shapes means recompiling.
  - `LM3_COREML_DYNAMIC=1`: one session per model that accepts every shape. It is untested on
    CoreML, and it may fail to build or run slower.
  Sections 3 and 4 measure both.
- **The options key:** CoreML options are passed as `{"flags": "COREML_FLAG_..."}`. That is the only
  form onnxruntime 1.20.1 reads; named keys like `ModelFormat` or `MLComputeUnits` are silently
  ignored. So the earlier per-model tests that used them ran in the NeuralNetwork format on every
  compute unit, Neural Engine included, which is float16.
- **`lm3 doctor --models`:** on a Mac it now runs each CoreML model against the CPU.
- **The lock:** the `macos` extra now also installs `onnx==1.22.0`, which does the graph cut.
  No model files or Hugging Face revisions changed.

## 1. Check out the branch and sync

From the LeafMachine3 folder you installed from the README:

```bash
git fetch origin
git switch macos-gpu-support
git log --oneline -1                      # note the commit you are testing
uv sync --frozen --extra macos            # adds onnx 1.22.0; nothing else changes
uv run --frozen --no-sync lm3 models install --yes   # no-op if the models are current
sysctl -n machdep.cpu.brand_string hw.memsize        # chip and RAM (bytes), for the report
```

## 2. Doctor

```bash
uv run --frozen --no-sync lm3 doctor             # expect: Result: READY
uv run --frozen --no-sync lm3 doctor --models    # check 7 now runs every CoreML model against the CPU
```

Check 7 should say `all 6 CoreML models agree with the CPU (speed vs CPU: ...)`. Report its exact
line. If it says FAIL or WARN, also run `uv run --frozen --no-sync lm3 doctor --models --json`.

## 3. Model by model

```bash
uv run --frozen --no-sync python tools/macos/verify_coreml.py --n 23 --json coreml_pershape.json
LM3_COREML_DYNAMIC=1 uv run --frozen --no-sync python tools/macos/verify_coreml.py --n 23 --json coreml_dynamic.json
```

Each prints one line per model with:
- whether the confident detections were found;
- box/mask IoU, score difference and keypoint error vs the CPU;
- CPU vs CoreML milliseconds.

It also prints one `CoreML: ... session #N for input HxW ... in S s (K alive, process RSS X GB)`
line per compile. The exit status is 1 if any model fails. Expect `ok` for all 6 in both modes.

From the per-shape log, work out two numbers for the report:
- **compile seconds per session:** the typical `in S s` value;
- **memory per cached session:** how much `process RSS` grows per session while `K alive` climbs
  from 1 to 8.

For the dynamic run, also say how many graph nodes CoreML takes, the same way you counted 607/657
before. If CoreML takes far fewer nodes with dynamic shapes, it falls back to the CPU for the rest.

## 4. Full runs

Run these one after another, not in parallel. `/usr/bin/time -l` reports wall time and peak memory.

```bash
# a. the CPU reference: this commit, CoreML switched off (identical to main's CPU path)
LM3_COREML=0 /usr/bin/time -l uv run --frozen --no-sync machine3 --config LM3_settings.yaml --run-name mac_cpu

# b. CoreML, the default: one session per shape, 8 kept per model
/usr/bin/time -l uv run --frozen --no-sync machine3 --config LM3_settings.yaml --run-name mac_coreml

# c. CoreML, one session per model for every shape (may fail to build: that is a result too)
LM3_COREML_DYNAMIC=1 /usr/bin/time -l uv run --frozen --no-sync machine3 --config LM3_settings.yaml --run-name mac_coreml_dynamic
```

Then a run that keeps every shape's session, so nothing recompiles. On the 23 examples, the leaf
segmenter sees 40 shapes and the landmark model 27, and run (b) rebuilds about 295 + 222 sessions.
Pick the cap from step 3: N × (memory per session) must stay under half the Mac's RAM. Use N = 64
if that fits; otherwise use the largest N that does, and report it.

```bash
# d. CoreML, per-shape sessions, N kept per model
LM3_COREML_MAX_SESSIONS=64 /usr/bin/time -l uv run --frozen --no-sync machine3 --config LM3_settings.yaml --run-name mac_coreml_keep
```

For each CoreML run (b, c, d), report:

```bash
R=runs/mac_coreml           # then runs/mac_coreml_dynamic, runs/mac_coreml_keep
grep -m4 "GPU plan" $R/logs/lm3.log                                              # expect: Apple GPU via CoreML, 1 worker
grep -c "CoreML: .* session #" $R/logs/lm3.log                                   # sessions compiled
grep -o "session #.* in [0-9.]* s" $R/logs/lm3.log | awk '{s+=$(NF-1)} END {print s " s compiling"}'
grep -o "process RSS [0-9.]* GB" $R/logs/lm3.log | sort -t' ' -k3 -n | tail -1    # highest RSS at a compile
grep -E "could not build|cannot run on CoreML|requested CoreML" $R/logs/lm3.log | head
```

## 5. Compare with the CPU

```bash
for R in mac_coreml mac_coreml_dynamic mac_coreml_keep; do
  uv run --frozen --no-sync python tools/verification/compare_runs_close.py runs/mac_cpu runs/$R --label "$R vs CPU" --json cmp_$R.json
done
```

The inputs are identical, so any difference is CoreML's arithmetic. It must stay at float32
rounding level. For scale, CUDA vs CPU on the same 23 sheets is in the right column.

## Acceptance

Each CoreML run vs `mac_cpu`:

| Check | Pass | CUDA vs CPU (Linux) |
|---|---|---|
| `lm3 doctor` | READY; check 7 OK | |
| `verify_coreml.py --n 23` | 6/6 `ok` | |
| run exit | exit 0, `stage_errors.csv` empty, every GPU stage on CoreML | |
| detections | every box matched (conf >= 0.4: all); IoU min >= 0.99; abs(dconf) max <= 0.03 | all; 0.997; 0.014 |
| leaves | all measured leaves matched; every metric: median <= 0.0005, p95 <= 0.01 | 258/258; <= 0.0001; <= 0.0035 |
| sheets | every count the same (archetypal leaves may differ by 1 on up to 2 sheets) | same; 2 sheets differ by 1 |
| ruler | unit the same on every sheet; px/cm relative change max <= 0.005 | 23/23; 0.0015 |
| specimen mask | area fraction abs diff max <= 0.001 | 0.0003 |

A run that passes is a candidate default. The choice between (b), (c) and (d) is wall time and peak
memory.

## Report back

- The commit tested, the chip and the RAM.
- The doctor `--models` line.
- Both `verify_coreml.py` outputs and JSON files, plus compile seconds and memory per session (step 3).
- For the dynamic mode, how many nodes CoreML takes.
- Wall time and peak memory for runs (a) through (d), and the N used for (d).
- For each CoreML run: sessions compiled, seconds compiling, highest RSS, any warnings.
- The three comparison outputs and their JSON.
