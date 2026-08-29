# LeafMachine3 desktop app — installer, launcher, and auto-update plan

**Status:** design approved in outline, implementation not started.
**Goal:** turn the developer-only Electron shell into a signed, 1-click-installable, auto-updating
desktop application for Linux, Windows, and macOS.

The reference implementation for the packaging/release/update pattern is
[`VoucherVisionGO-Editor`](https://github.com/Gene-Weaver/VoucherVisionGO-Editor) — see the appendix.
LM3 follows it closely and diverges only where it must: VVGO is pure JavaScript, while LM3 carries a
~7 GB Python/CUDA backend and ~1.5 GB of model weights.

Sections tagged **TODO(github)** are written and locally testable but cannot be wired up until this
repo has a GitHub remote.

---

## 1. Decisions

| Question | Decision |
|---|---|
| How do users get the Python backend? | Bundle the `uv` binary; a first-run wizard builds a managed venv in a user data dir |
| How do users get the model weights? | A **separate, versioned model pack** installed once — not bundled in the app installer |
| Which hardware is supported? | All three, auto-detected: NVIDIA cu124 (Linux/Windows), macOS MPS + CoreML, CPU fallback everywhere |
| Where does the packaging code live? | `app/` → `desktop/` with internal subdirs; a new top-level `tools/`; dev launchers at `desktop/` root |

### Why `uv` + python-build-standalone, and not the alternatives

| | uv + PBS | PyInstaller freeze | conda/micromamba | require user Python |
|---|---|---|---|---|
| Installer size | **~160 MB** | 6–8 GB | ~200 MB | ~120 MB |
| Preserves `_exec_with_cuda_libpath` | **exactly** | maybe | **breaks it** | exactly |
| Reproduces the tested pin set | **exactly** | exactly | no | user's problem |
| Change GPU flavor after install | **`uv pip sync` another file** | reinstall app | re-solve | manual |
| Upgrade when only LM3 code changed | **2 MB wheel** | full reinstall | full re-solve | `pip install -U` |
| macOS notarization surface | **only Electron is signed** | thousands of `.so`/`.dylib` | same problem | trivial |
| Cross-platform parity | **one code path** | 3 configs | 3 configs | n/a |

The decisive argument against conda — which `docs/LM3_Plan.html` originally recommended — is that
`machine3._exec_with_cuda_libpath()` (`leafmachine3/machine3.py:120`) does
`import nvidia; base.glob("*/lib")`. That layout exists *only* because the `nvidia-*` **pip wheels**
put libraries at `site-packages/nvidia/cublas/lib`. Under conda they live in `$PREFIX/lib`, the glob
returns `[]`, no re-exec happens, `LD_LIBRARY_PATH` stays unset, onnxruntime binds CPU, and
`providers.make_session()` **raises** on any NVIDIA box. Adopting conda would mean rewriting the one
function that makes the GPU work, to gain nothing — and conda-forge cannot reproduce the tested pins
anyway (`ultralytics==8.4.107`, `opencv-python-headless==5.0.0.93`, `ect`, `torch==2.6.0+cu124` from
the PyTorch index are all pip-shaped). It would end up as
`micromamba create python=3.11 && pip install -r requirements-gpu.txt` — i.e. the uv plan with a
second package manager bolted on.

PyInstaller is rated "✗ fragile" for CUDA by our own design doc, and a frozen backend cannot be
repaired or re-flavored without a full app reinstall.

---

## 2. Current state — verified facts

Everything below was read from source, not assumed.

### The shell

`app/` is four files: `main.js` (8 KB), `preload.js`, `package.json`, `package-lock.json`. Its only
dependency is `electron` as a **devDependency**. There is no `build` block, no `appId`, no icon, no
`electron-builder`, no `electron-updater`, and no CI beyond `.github/workflows/ci.yml` (ruff + pytest).

There is **no renderer content in `app/`** — the entire UI is static files served over HTTP by the
Python FastAPI server from `leafmachine3/server/ui/`. That is a deliberate and good design (the same
UI runs in a plain browser, which is how it gets screenshot-tested), but it means the packaged app
cannot render its own installer UI. Hence the separate bootstrap window in Phase 3.

### Blockers for a packaged distribution

1. **`ROOT = path.resolve(__dirname, "..")`** (`main.js:16`). Under asar this becomes
   `…/resources/app.asar/..` = `…/resources/`. It is used for three unrelated things at once — the
   Python path, the uvicorn `cwd`, and the `file:` containment check in `lm3:open-external` — so all
   three break together.
2. **`PYTHON = <ROOT>/.venv_LM3/bin/python`** (`main.js:17`). Assumes a 7 GB sibling dev venv, POSIX
   layout only, with no `Scripts/python.exe` branch and no "not found" diagnostic.
3. **`server/ui/**` is not package data.** `pyproject.toml` declares only
   `leafmachine3 = ["core/schema.sql"]`, there is no `MANIFEST.in`, and `SOURCES.txt` contains zero
   `server/ui` entries. A wheel install ships no UI → `if _ui.is_dir()` is False → no `/` route →
   `loadURL("/")` 404s. This works today **only** because all three requirements files end in `-e .`.
4. **All three requirements files contain `-e .`**, hard-requiring an editable checkout.
5. **Everything resolves against CWD**: `DEFAULT_JOBS_ROOT`, `Path("LM3_settings.yaml")`,
   `hardware_setup.HW_PATH`, and `results_api.run_roots()`'s three `Path.cwd()` pushes. In an
   installed app the install directory is read-only.
6. **Model weights are symlinks into sibling training repos and are gitignored.** There is no
   download code, no manifest, and no gate anywhere in the tree.
7. **Fixed port 8765, and `ensureServer()` attaches to *any* 200 on `/healthz`** — then hands that
   stranger a token it did not generate. Already observed colliding with an unrelated app on this
   machine (see `global_leaf_margin_collage/launch_gui.sh`).
8. **`ELECTRON_RUN_AS_NODE=1` inherited** makes the packaged binary run as plain node: no window, no
   output, no error.
9. **`chrome-sandbox` is not setuid** in this checkout, so `--no-sandbox` is currently mandatory.
10. **`lm3:open-external` permits `file:` only inside `ROOT`.** Real run outputs live wherever
    `project.output.dir` points, so in a packaged app *no user output* would be openable.
11. **`LM3_SETTINGS` vs `LM3_SETTINGS_PATH` split-brain** — the first drives the RUN
    (`metrics_api.default_config_path`), the second drives the SETTINGS RAIL (`settings_api._resolve`).
    Setting one and not the other makes the UI display a different config than the one that executes.
12. **POSIX-only process control**: `start_new_session=True`, `os.killpg`, `os.getpgrp`,
    `prctl(PR_SET_PDEATHSIG)`. On Windows, Stop would orphan the executor's spawn workers and their
    CUDA contexts.

### Corrections to earlier assumptions

- **`hardware_setup._model_hashes()` uses a `size:mtime` signature, not sha256**
  (`hardware_setup.py:656` — `f"sig:{st.st_size}:{int(st.st_mtime)}"`), deliberately, for speed. So
  installing or repairing a weight file flips the hardware fingerprint and triggers a "rerun setup"
  nag. The installer must write `.sha256` sidecars that `_model_hashes` prefers.
- **There is a *third* env-var drift**: `progress_api.py:345` reads `LM3_HARDWARE`,
  `metrics_api.py:135` reads `LM3_HARDWARE_SETTINGS`, and `hardware_setup.HW_PATH` hardcodes
  `Path("hardware_settings.yaml")` — three names for one file.
- **`ensureServer()`'s startup budget is ~102 s, not 30 s**: 60 tries × (500 ms sleep + `ping`'s own
  1200 ms timeout). A missing interpreter hangs for over a minute before a generic error appears.
- **`/healthz` returns a hardcoded `"3.0.0"`** (`app.py:440`) while `pyproject.toml` says `0.1.0`.

### Model inventory — and two curation traps

`du` on `models/` reports 1.3 GB, which is misleading: it does not follow symlinks, and almost all of
that figure is `specimen_segmenter`, the one stage with real files. The shippable set is ~1.5 GB:

| Stage | Artifact | Size |
|---|---|---|
| archival_detector | `model.onnx` | 214 MB |
| plant_detector | `model.onnx` | 214 MB |
| landmark_detector | `model.onnx` | 227 MB |
| leaf_segmenter | `model.pt` | 136 MB |
| specimen_segmenter | `…control_1024.onnx` | 266 MB |
| ruler_classifier | `yolo26x_cls_224/exported/model.onnx` | 109 MB |
| ruler_classifier | `yolo26n_cls_224/exported/model.onnx` | 6.1 MB |
| ruler_classifier | `dinov2_frozen_mlp/exported/model.onnx` | 333 MB |
| ruler_classifier | `splits/label_map.json` | 1.3 KB |
| mp_conversion_factor | `model.json` | 513 B (already committed) |

> **Trap 1 — `specimen_segmenter` holds 6 artifacts; the config selects 1.** `LM3_settings.yaml:66`
> picks only `control_1024.onnx` out of a 1.3 GB directory (two `.onnx`, two `.torchscript.pt`, two
> `.mlpackage`).
>
> **Trap 2 — the ruler member directories total ~6 GB; only `exported/model.onnx` is needed.**
> `models/ruler_classifier` symlinks the whole `LM3_Ruler_Classifier/models` training tree (7.9 GB).
> A naive directory copy would be roughly 13× too large.

The pack builder must therefore resolve artifacts through the config's own `_stage_artifacts`
resolver rather than copying directories.

The ruler ensemble needs exactly four files (`ruler_ensemble.py:44-48,67`) and is **best-effort** — it
degrades to `UNKNOWN` rather than raising — so it can be an optional sub-pack.

---

## 3. Target repository layout

```
LM3/
├─ electron_app_plan.md            ← this document
├─ desktop/                        ← renamed from app/; IS the npm package root
│  ├─ package.json                 electron-builder `build` block; electron-updater in dependencies
│  ├─ main.js                      orchestration only
│  ├─ preload.js  bootstrap-preload.js
│  ├─ src/{paths,config,runtime,models,server,procgroup,updater,security,logging}.js
│  ├─ bootstrap/{index.html,bootstrap.css,bootstrap.js}      first-run + repair wizard
│  ├─ assets/icon.png              ONE 1024² PNG; builder derives .icns/.ico
│  ├─ build-resources/entitlements.mac.plist
│  ├─ resources/                   extraResources SOURCE — generated, mostly gitignored
│  │    runtime-spec.json  models-manifest.json  requirements/  wheels/  uv/<os>-<arch>/
│  ├─ LeafMachine3.command  LeafMachine3.sh  LeafMachine3.bat    ← dev launchers, double-clickable
│  └─ build/                       electron-builder output (gitignored)
├─ tools/
│  ├─ release/{deploy.sh, release.config.json, fetch_uv.sh, prepare_resources.py, .env.signing.example}
│  └─ models/{build_model_pack.py, install_pack.py}
├─ requirements/                   `-e .` REMOVED from all three
├─ leafmachine3/                   nothing packaging-related goes in here
└─ .gitattributes                  *.sh/*.command LF, *.bat CRLF
```

**Why this shape.** `desktop/` is the npm root because electron-builder wants `package.json`,
`node_modules`, `main.js` and the `build` block as siblings — hoisting to the repo root would put
`node_modules/` next to `leafmachine3/` and make `files` globbing fight the Python tree. `tools/`
(not `scripts/`, which reads like npm scripts) keeps every build and release artifact out of the
Python package. Dev launchers sit at `desktop/` root, exactly as VVGO does: a double-clickable
launcher buried in a subdirectory helps nobody. End users never see any of this — they get installers.

---

## 4. Phase 1 — make the Python package location-independent

**Do this first.** It is the highest-risk work (it modifies running code paths) and everything else
depends on it.

- **Delete the trailing `-e .`** from `requirements/requirements-{gpu,cpu,macos}.txt`. They become
  pure third-party pins. `INSTALL.md` keeps `pip install -e .` as an explicit *developer* step.
- **`pyproject.toml`**: `include-package-data = true` plus package-data globs for
  `server/ui/index.html`, `server/ui/settings_meta.json`, `server/ui/css/*.css`, `server/ui/js/*.js`,
  `server/ui/js/tabs/*.js`, `postprocessing/stl_webview/*`; add a matching `MANIFEST.in`.
- **Version unification**: `pyproject` → `3.1.0`; `leafmachine3.__version__` via
  `importlib.metadata.version`; `app.py`'s `FastAPI(version=…)`, the `/healthz` literal, and
  `desktop/package.json` all read or match it.
- **`Config.resolve_model_path()`** in `core/config.py`: a relative path starting `models/` re-roots
  onto `$LM3_MODELS_DIR` when set, else CWD. Sweep `core/validate.py:29,34`,
  `core/config.py:_stage_artifacts:497-501`, and the `modules/*.py` model loads. **Do not rewrite the
  YAML** — it keeps saying `models/archival_detector/model.onnx` and works in both dev and packaged
  mode.
- **Env-var collapse**: one canonical name each with the old name as a deprecated alias —
  `LM3_SETTINGS` (was also `LM3_SETTINGS_PATH`) and `LM3_HARDWARE` (was also
  `LM3_HARDWARE_SETTINGS`). Reuse `metrics_api._find_file()` as the shared resolver.
- **Server additions**: `/healthz` returns the real version plus an `instance_id` echoed from
  `$LM3_INSTANCE_ID`; a new authenticated `GET /v1/runs/roots` returning `results_api.run_roots()`;
  `_model_hashes` prefers a `.sha256` sidecar when one exists.
- **Housekeeping**: scrub the absolute `/datab/…` and `/datac/…` paths from the committed
  `LM3_settings.yaml`; make `settings_api._roots()` (line 907) platform-aware — `LM3_BROWSE_ROOTS`
  plus Windows drive letters, `/Volumes`, `/media`, `/mnt`, `/run/media/$USER` — instead of
  hardcoding `/data,/datac,/scratch`.

**Tests, or blocker #3 silently returns.** `importlib.resources` assertions for
`server/ui/index.html`, `settings_meta.json`, `css/app.css`, `js/tabs/settings.js`; a
`package.json.version == leafmachine3.__version__` assertion; and a CI job that runs
`python -m build --wheel`, installs the wheel into a clean venv, and runs those assertions **from a
different working directory**. That job is the only thing that catches "works because `-e .`".

---

## 5. Phase 2 — Electron restructure

Split `main.js` into the `src/` modules listed in §3.

**Paths.** `path.resolve(__dirname, "..")` appears exactly once, guarded by `!app.isPackaged`, and is
**never** used for the `file:` containment check. Packaged resources come from `process.resourcesPath`.

**Data directory — deliberately not `userData`.** Putting 7 GB under `~/.config` or a roaming
`%APPDATA%` is hostile:

| OS | Path |
|---|---|
| Linux | `${XDG_DATA_HOME:-~/.local/share}/LeafMachine3` |
| macOS | `~/Library/Application Support/LeafMachine3` |
| Windows | `%LOCALAPPDATA%\LeafMachine3` (Local, never Roaming) |

```
<dataDir>/
  runtime/<flavor>/         venv (uv-managed)
  runtime.lock.json         {flavor, python, requirementsSha256, wheelVersion, uvVersion, installedAt}
  runtime.probe.json        post-install verification result
  uv-cache/                 makes a failed install cheap to retry
  models/<modelsVersion>/   weights + .sha256 sidecars
  models.lock.json
  workspace/                ← the server's CWD
    LM3_settings.yaml  hardware_settings.yaml  postprocessing_settings.yaml  presets/  runs/
  logs/                     bootstrap-*.log, server-*.log
```

Only a small `lm3-desktop.json` recording the chosen dataDir lives in `app.getPath('userData')`.

**Python discovery.** `Scripts/python.exe` on win32, `bin/python` elsewhere; resolution order is
`LM3_PYTHON` → `runtime.lock.json` → derived path. `fs.accessSync(python, X_OK)` **before** spawning,
so a missing runtime opens the wizard immediately instead of hanging ~102 s into a generic error.

**Free port and real identity.** `net.createServer().listen(0)` picks the port. Stop attaching to
strangers: pass `LM3_INSTANCE_ID=<nonce>` to the child and only treat a 200 as ours if `/healthz`
echoes it back. Attach-to-existing becomes a dev-only opt-in behind `LM3_ATTACH=1`.

**`ELECTRON_RUN_AS_NODE`, three defenses.** (1) Detect at the top of `main.js` — if `app` is
undefined, print a real diagnostic and `exit 78` instead of failing silently. (2)
`delete childEnv.ELECTRON_RUN_AS_NODE` before every spawn. (3) The launchers `unset` it. While
building the child environment, also strip `PYTHONHOME`, `PYTHONPATH`, `VIRTUAL_ENV`, `CONDA_PREFIX`,
`PIP_*`, `NODE_OPTIONS`, and `LD_PRELOAD` — the classic causes of "it ran the wrong Python".

**Sandbox.** Set `sandbox: true` in `webPreferences` rather than globally passing `--no-sandbox` (the
preload only uses `ipcRenderer`, which works sandboxed). The `deb` target installs setuid
`chrome-sandbox`; `LM3_NO_SANDBOX=1` is a documented escape hatch; `--no-sandbox` stays in the **dev**
launchers only, where the checkout's `chrome-sandbox` genuinely isn't setuid.

**`cwd = <dataDir>/workspace` is the single highest-leverage fix.** That one line fixes
`hardware_setup.HW_PATH`, `app.DEFAULT_JOBS_ROOT`, `read_hardware_settings()`, `settings_api`'s
settings and presets directories, `results_api.run_roots()`'s three `Path.cwd()` pushes,
`progress_api` and `postprocess_api` settings paths, and every relative `Config.resolve_path()`.
Environment threading is belt-and-braces on top: `LM3_SETTINGS`, `LM3_HARDWARE`,
`LM3_POSTPROCESS_SETTINGS`, `LM3_SERVER_JOBS`, `LM3_RUNS_ROOTS`, `LM3_MODELS_DIR`,
`LM3_SERVER_TOKEN`, `LM3_INSTANCE_ID`, and **`LM3_EMBED_TOKEN=0`**.

Setting `LM3_EMBED_TOKEN=0` unconditionally is a **security upgrade**: the shell already passes
`?token=` and `api.js:43` handles it, so no other local process can simply `GET /` to read the secret,
and `app.py`'s loopback-peer + loopback-Host gate never has to fire.

**Process-group kill** (`src/procgroup.js`). POSIX: `detached: true`, then `process.kill(-pid)` with
SIGTERM → grace → SIGKILL. Windows: `taskkill /pid <p> /T /F`. Python side —
`metrics_api.start_run()` gets `creationflags=CREATE_NEW_PROCESS_GROUP` on `nt`, and `_signal_group`
(line 771) an `nt` branch (`CTRL_BREAK_EVENT`, then `taskkill /T /F`). Note that
`executor._set_parent_death_signal()` already returns `False` off Linux, so `reap_orphaned_workers()`
is the only orphan defense on Windows and macOS. A Windows Job Object with `KILL_ON_JOB_CLOSE` is the
only leak-proof answer; log it as future hardening (it needs a native dependency) and ship
`taskkill /T` first.

**Quit path.** `main.js` itself `POST /v1/run/stop`s with the token (5 s timeout) before killing the
server, so Close behaves correctly even if the renderer never ran. Then `killTree(serverPid)`.

**`lm3:open-external` rewrite.** Replace the single `ROOT` with a dynamic root set — dataDir,
workspace, modelsDir, the roots from `GET /v1/runs/roots` (cached 10 s), plus any directory the user
explicitly picked via a native dialog this session. Keep the **structural** per-root test
(`p === root || p.startsWith(root + path.sep)`), the `if (u.host) return false` check, and the
http/https `u.host === self.host` test verbatim. The existing comment naming the exact bypasses it
defeats (`http://127.0.0.1:8765@evil.com/`, `http://127.0.0.1:8765.evil.com/`, `/datac/.../LM3_evil/`)
should be preserved.

**Security controls that must remain untouched**: bearer `secrets.compare_digest`; `?token=` for
SSE and `<img>`; the `yaml_path` suffix allowlist; `_normalize_rel`/`safe_path` with
`_SYMLINK_ESCAPE_OK`; read-only SQLite with `set_authorizer`; the `sandbox allow-scripts` CSP on
run-produced HTML; `setWindowOpenHandler` deny-all; `contextIsolation` / `nodeIntegration: false`.

---

## 6. Phase 3 — runtime bootstrap

`tools/release/fetch_uv.sh` pins a `uv` version plus sha256 and vendors the binary per `(os, arch)`
into `desktop/resources/uv/`.

**Flavor probe** — `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader`, 5 s timeout:

| Flavor | When | Requirements file | Venv size |
|---|---|---|---|
| `gpu-cu124` | linux/win32 + NVIDIA + driver ≥ 550 | `requirements-gpu.txt` | ~7.0 GB |
| `cpu` | linux/win32 without NVIDIA, or user override | `requirements-cpu.txt` | ~2.5 GB |
| `macos` | darwin | `requirements-macos.txt` | ~2.5 GB |

Always overridable, with the detection *reason* shown. Override is mandatory because
`providers.make_session()` **raises** when an NVIDIA GPU is present but ORT bound CPU — a wrong guess
is a hard failure on the user's first real run, not merely a slow one.

> **Windows GPU caveat.** `_exec_with_cuda_libpath()` sets `LD_LIBRARY_PATH`, which Windows ignores;
> the `nvidia-*` wheels put DLLs in `nvidia\*\bin`. Add an `nt` branch calling `os.add_dll_directory()`
> for each `nvidia/*/bin` (no re-exec needed — it takes effect immediately for subsequent
> `LoadLibrary` calls, and spawn workers re-run it). Until that is tested on real Windows + NVIDIA
> hardware, **default Windows to `cpu` and offer `gpu-cu124` as "experimental"**.
>
> Not pursuing DirectML: `onnxruntime-directml` would accelerate the ONNX stages, but
> `leaf_segmenter` is a torch `.pt` driven by ultralytics and torch 2.6 has no DirectML backend. The
> result would be a split-brain pipeline and a fourth dependency matrix.

**Bootstrap sequence**: `uv python install 3.11` → `uv venv` → `uv pip sync <requirements-FLAVOR.txt>`
→ `uv pip install --no-deps <leafmachine3 wheel from resources/wheels>`. Each step is individually
retryable and cheap to restart thanks to the uv cache.

**Verify step** writes `runtime.probe.json`: `sys.version`, `torch.__version__`,
`torch.cuda.is_available()`, `onnxruntime.get_available_providers()`, `leafmachine3.__version__`, and
that `importlib.resources.files("leafmachine3.server")/"ui"/"index.html"` exists. **If the flavor is
`gpu-*` and `CUDAExecutionProvider` is absent, stop here with a loud, specific error** rather than
letting the user discover it via a `RuntimeError` mid-run.

**Wizard window.** Because the main UI is server-served, it cannot render its own installer. Add one
app-owned `file://` window loaded from inside asar (`bootstrap/index.html`), with
`contextIsolation: true`, `sandbox: true`, and its own preload exposing only
`lm3boot.{start, cancel, chooseDir, retry, openLog, copyDiagnostics, onStatus}`.

Pages: data directory (free space per candidate — `metrics_api._free_gb` is the prior art) →
hardware → runtime install (live uv stderr tail, killable) → verify → models (§7) → tune
(`POST /v1/setup`, honoring the 409 single-flight) → done.

**Failure diagnostics.** Every step's full output goes to `<dataDir>/logs/bootstrap-<ts>.log`. On
failure, show the last 40 lines inline plus "Open log" and "Copy diagnostics" (platform, arch, app
version, flavor, `nvidia-smi` output, uv version, failing command, exit code, last 200 log lines).

**Runtime panel**, reachable from the app menu at any time, not only on first run:
Verify · Repair (`uv pip sync --reinstall`) · Reinstall · Change flavor · Open folder · Open log.

**Optional accelerator.** `uv venv --relocatable` produces a venv with no absolute-path shebangs.
Publishing a prebuilt tarball per `(os, arch, flavor)` would let the wizard offer "Download prebuilt
runtime (~3.5 GB, ~4 min)" instead of "Build from wheels (~15–25 min)", with the same lock-file
gating. This is also the offline / air-gapped path: drop the tarball beside the installer.

---

## 7. Phase 4 — the model pack

**The artifact.** `LeafMachine3-Models-<modelsVersion>.lm3pack` — a zip with `manifest.json` at its
root. One cross-platform file, versioned independently of the app.

**Built by** `tools/models/build_model_pack.py`, which resolves each stage's artifact through the
config's own `_stage_artifacts` resolver, dereferences the symlinks, curates the ruler ensemble down
to its four real members, and records `{path, bytes, sha256, stage}` per file. The `stage` field lets
the UI say "RulerClassifier is unavailable" instead of a generic error, and maps 1:1 onto
`validate_ml_artifacts`'s `_MODEL_STAGES`. Optional sub-packs: `ruler` (448 MB — degrades to
`UNKNOWN` if absent) and `coreml` (macOS `.mlpackage` only).

**Four installation routes, all offline-capable today:**

1. Wizard button **"Install model pack…"** → file picker → verify and extract.
2. **Double-click the `.lm3pack`** — electron-builder `fileAssociations` on Windows and macOS opens
   LeafMachine3 and starts the install. This is what makes the whole thing genuinely 2-click.
3. **"Use an existing models folder…"** → point at a directory (dev boxes, lab shares). Hardlinks
   when source and destination share a filesystem, so a lab machine installs instantly.
4. Headless `tools/models/install_pack.py` for HPC and lab provisioning.

**Install semantics.** Extract to `<dataDir>/models/<modelsVersion>/`; verify every file's sha256
against the embedded manifest **before** moving it into place (never rename an unverified file);
write `.sha256` sidecars, which `_model_hashes` then prefers so a repair does not flip the hardware
fingerprint; write `models.lock.json`. `manifest.minAppVersion` lets an older app refuse a newer pack.
Per-launch validation is exists + size + sidecar (milliseconds); a full re-hash is on demand only.
`LM3_MODELS_DIR` points the server at the result; the YAML keeps its relative `models/…` paths.

**TODO(github):** a "Download model pack" button. Recommendation for that day — **app on GitHub
Releases, weights on Hugging Face Hub**. GitHub release assets have a 2 GB per-file cap, no
guaranteed `Range` support, and no versioned tree; HF Hub gives a CDN, real resumability, and content
versioning. The manifest's base-URL split makes it a one-line configuration change either way.

---

## 8. Phase 5 — packaging configuration

`desktop/package.json` `build` block, following VVGO with LM3-specific choices:

- `appId: org.leafmachine.lm3`, `productName: LeafMachine3`,
  `artifactName: LeafMachine3-${version}-${os}-${arch}.${ext}`, `directories.output: build`.
- **macOS: `universal` dmg + zip**, `hardenedRuntime: true`, `gatekeeperAssess: false`, entitlements
  and entitlementsInherit, `notarize: true`. Universal is the *structural* fix for VVGO's known
  `latest-mac.yml` last-writer-wins bug — one build, one manifest, no arch race to lose.
- **Windows: `nsis` + `portable` x64.** `oneClick: false` (LM3 needs a disk-space conversation before
  it downloads 7 GB), `perMachine: false` (per-user → no UAC, which in practice is *closer* to
  one-click than an elevated silent install), `allowToChangeInstallationDirectory: true`, and
  **`deleteAppDataOnUninstall: false`** — uninstall must never delete a 7 GB runtime and 1.5 GB of
  weights.
- **Linux: AppImage + deb.** deb is the only Linux target that installs setuid `chrome-sandbox`, a
  desktop entry, and an icon — the best real 1-click on Ubuntu lab machines. Skipping rpm (needs
  `rpmbuild` on the release host, for users we do not have) and tar.gz (adds nothing over AppImage).
- `extraResources`: `requirements/`, `wheels/`, `runtime-spec.json`, `models-manifest.json`. The `uv`
  binary is copied in by a **`beforePack` hook** for the current target only — shipping all four
  platform binaries would waste ~140 MB, and `${os}`/`${arch}` macros inside `extraResources.from`
  are unreliable across builder versions.
- `asar: true`, `asarUnpack: []` — everything executable lives in extraResources, which is outside
  asar by definition.
- `publish: [{provider: "github", owner: "__TODO_OWNER__", repo: "__TODO_REPO__"}]` — **TODO(github)**.

**Icon.** No LeafMachine icon exists anywhere in the tree. Generate a placeholder 1024² `icon.png`
(a simple leaf mark on the UI's `#101012`) and replace it before any public release —
electron-builder derives `.icns` and `.ico` from that one file, so the swap is a one-file change. Do
not hand-maintain a per-size icon ladder.

**Dev launchers** at `desktop/` root, mirroring VVGO: `.command` / `.sh` (`cd "$(dirname "$0")"`,
`unset ELECTRON_RUN_AS_NODE`, `./node_modules/.bin/electron . --no-sandbox`) and `.bat` with forced
CRLF. Add `.gitattributes` with `*.sh|*.command text eol=lf` and `*.bat text eol=crlf`.

---

## 9. Phase 6 — auto-update

**Shell updates mirror VVGO exactly.** `electron-updater` in **dependencies** (not devDependencies —
it must be inside the asar); `autoDownload = false`; `autoInstallOnAppQuit = true`; the check fires in
`app.whenReady()` behind `if (app.isPackaged)` after a 5 s delay; six `autoUpdater.on(...)` handlers
(`checking-for-update`, `update-available`, `update-not-available`, `download-progress`,
`update-downloaded`, `error`) collapse onto one `update-status` IPC channel
`{status, version, percent, message}`. Windows portable builds cannot self-update, so
`isPortableWindows()` (`process.env.PORTABLE_EXECUTABLE_DIR != null`) queries
`api.github.com/repos/…/releases/latest` directly with a 10 s `AbortController` and emits
`'available-manual'`. IPC: `get-update-info` / `check-for-update` / `download-update` /
`install-update`; `installDate` and `lastUpdateCheck` persisted to `userData`.

**UI surfaces.** The server UI already has `<div class="toasts" id="toasts">`, so: a 10 s
auto-dismiss toast, plus an "Updates" block in an About panel, both driven by
`window.lm3desktop.onUpdateStatus(cb)` and `getUpdateInfo()` added to `preload.js`. No splash screen
is needed — the bootstrap wizard is the splash.

### LM3's extra dimension: three independent version axes

The app bundle is the **sole source of truth** for what the runtime and models should be:

1. `appVersion` — `package.json`, driven by electron-updater.
2. `runtimeSpec` — `{flavorPolicy, pythonVersion, requirementsSha256, wheelVersion}` from
   `resources/runtime-spec.json`, generated at build time.
3. `modelsVersion` — from `resources/models-manifest.json`.

Every launch compares those against `dataDir/runtime.lock.json` and `models.lock.json`:

| Condition | Action |
|---|---|
| no lock file | FIRST_RUN wizard |
| detected flavor ≠ locked flavor, not user-pinned | offer a flavor change (non-blocking) |
| `requirementsSha256` differs | RUNTIME_UPDATE — `uv pip sync` applies the minimal delta (minutes) |
| `wheelVersion` differs | WHEEL_UPDATE — force-reinstall the 2 MB wheel (seconds) |
| `modelsVersion` differs | prompt for the new model pack |
| otherwise | launch |

This is **content-addressed, not channel-addressed**, which is why stripping `-e .` matters: with it
present, every LM3 commit would look like a requirements change. Net effect — an LM3 code-only
release is a 3-second wheel swap; a torch bump is a real `uv pip sync`; an Electron/Chromium bump
touches neither the runtime nor the models. Record `{appVersion, wheelVersion, modelsVersion, flavor}`
into each run's SQLite for provenance.

**Testable today with zero GitHub involvement**: electron-updater reads `dev-app-update.yml` in dev
and supports the `generic` provider. Point it at `http://127.0.0.1:8080/` serving a directory of fake
releases and the entire flow — `available-manual` included — can be exercised locally.

---

## 10. Phase 7 — release script

`tools/release/deploy.sh`, following VVGO's shape:

1. Read the version from `desktop/package.json`.
2. `prepare_resources.py` — build the wheel, copy requirements, fetch uv, write `runtime-spec.json`.
3. Three parallel builds (`--linux --x64`, `--win --x64`, `--mac --universal`), each to its own
   logfile, surfaced through VVGO's
   `building|packaging|signing|artifact|notariz|stapl|error|fail` grep filter — a deliberate design
   so a silently-skipped notarization can never hide.
4. macOS second notarization pass: `xcrun notarytool submit --wait` then `xcrun stapler staple` and
   `stapler validate` on each DMG, because electron-builder's `notarize: true` only notarizes the
   `.app` *inside* the DMG, not the container.
5. **TODO(github)** — `gh release delete`/recreate with `#Label` suffixes, uploading the artifacts
   **plus `latest.yml`, `latest-mac.yml`, and `latest-linux.yml`** (electron-updater fetches those;
   easy to forget).

Differences from VVGO: the `gh release` step sits behind `--release` and **defaults off**; a preflight
fails loudly with "set owner/repo in tools/release/release.config.json" until a remote exists; and
any webpage-copy destination is a config value, not a hardcoded `/Users/willwe/…` path.

Commit `.env.signing.example` with `APPLE_ID`, `APPLE_APP_SPECIFIC_PASSWORD`, and `APPLE_TEAM_ID`
(VVGO references this file but never added it); keep `.env.signing` gitignored. Windows ships
unsigned — with the resulting SmartScreen warning documented in the README — until an EV certificate
exists.

---

## 11. What is blocked on GitHub

Everything below is written and locally testable; only the final wiring waits.

| Item | Status |
|---|---|
| `publish.owner` / `publish.repo` | `__TODO_OWNER__` / `__TODO_REPO__`; `pyproject.toml` already points at `Gene-Weaver/LeafMachine3` |
| `gh release create` in `deploy.sh` | behind `--release`; preflight errors until configured |
| electron-updater against GitHub | testable now via the `generic` provider + a local static server |
| Portable-Windows `available-manual` REST call | URL comes from `release.config.json` |
| Model pack **download** button | file-picker / folder / hardlink routes all work offline today |
| `.github/workflows/release.yml` | optional stub, tag-triggered, `--skip-release` |
| Code signing certificates | independent of GitHub; blocked on certs |

---

## 12. Verification

1. **Wheel isolation** (catches the `-e .` regression): `python -m build --wheel`, install into a
   clean venv, then from `/tmp` assert the UI resources import and `machine3 --help` works. Add as CI.
2. **Existing suite green**: `.venv_LM3/bin/python -m pytest` (56 tests).
3. **Dev-mode parity**: `cd desktop && ./LeafMachine3.sh` must still attach to the existing
   `.venv_LM3` and behave exactly as today, via the `LM3_PYTHON` override.
4. **Cold bootstrap on Linux**: move `~/.local/share/LeafMachine3` aside, launch, walk the wizard,
   and confirm the venv builds, the probe reports `CUDAExecutionProvider`, the model pack installs and
   verifies, `POST /v1/setup` writes `hardware_settings.yaml` into `workspace/`, and a real run
   completes. Compare against the known 19-image / ~75 s baseline; the VRAM Δ column in
   `reports/Timing/timing.csv` is the reliable tell that the GPU actually bound.
5. **Packaged smoke on Linux**: `npm run pack`, then run `build/linux-unpacked/leafmachine3` —
   exercises asar path resolution, `resourcesPath`, and the free-port/identity logic outside dev mode.
6. **AppImage + deb**: install the deb on a clean-ish user and confirm `chrome-sandbox` is setuid and
   the app runs *without* `--no-sandbox`.
7. **Updater flow**: serve a fake release directory over `http://127.0.0.1:8080`, point
   `dev-app-update.yml` at it, and exercise checking → available → download-progress → downloaded →
   install.
8. **Three-axis gate**: hand-edit `runtime.lock.json` to a stale `requirementsSha256` and confirm
   RUNTIME_UPDATE fires; bump only `wheelVersion` and confirm the fast path; bump `modelsVersion` and
   confirm the model-pack prompt.
9. **Security regressions**: `curl http://127.0.0.1:<port>/` must NOT contain a token (because
   `LM3_EMBED_TOKEN=0`); `GET /v1/settings?yaml_path=/etc/passwd` still fails; `lm3:open-external`
   still refuses `file:///etc/passwd` and `http://127.0.0.1:<port>@evil.com/` but now *accepts* a real
   run output living outside the install directory.
10. **Windows and macOS** cannot be verified from the Linux dev box. Build and smoke-test on those
    hosts before any release, and treat Windows GPU as experimental until `os.add_dll_directory` has
    been tested on real NVIDIA hardware.

---

## 13. Sequencing

Phases 1 and 2 carry the real risk — they modify running code paths. Phases 3–7 are almost purely
additive. **Land 1 and 2 with tests green before writing a line of the installer**, or the
bootstrapper and the path refactor end up being debugged simultaneously.

Phase 0 is this document plus repository hygiene: add `.gitattributes`, and move or gitignore the
five `LM3_settings.bak.*.yaml`, `LM3_settings.orig.yaml`, `todo.html`/`todo.json`, and `.coverage`
sitting at the repo root — they are the first thing a new contributor sees.

---

## Appendix — the VoucherVisionGO-Editor pattern

Recorded here because LM3 deliberately copies it and it is not documented anywhere else.

- **One `package.json`** with the electron-builder config inline as a `build` block. No
  `electron-builder.yml`.
- **Exactly two build inputs**: `assets/icon.png` (a single 1024 PNG — builder derives `.icns` and
  `.ico` into `build/.icon-icns/` and `build/.icon-ico/`) and
  `build-resources/entitlements.mac.plist` (four keys: `allow-jit`,
  `allow-unsigned-executable-memory`, `disable-library-validation`,
  `files.user-selected.read-write`).
- **No `afterSign` / `afterPack` / `beforeBuild` hooks anywhere.** Notarization is
  electron-builder's `notarize: true` plus a manual second pass in `deploy.sh`.
- **No CI.** No `.github/` directory at all. The whole pipeline is one local `deploy.sh` run from the
  author's Mac, which cross-builds all targets.
- **No `GH_TOKEN`.** electron-builder is never invoked with `--publish`; it only uses the `publish`
  block to emit `app-update.yml` and `latest*.yml`. Upload happens through `gh release create`, which
  uses the developer's local `gh auth login`.
- **`#Label` suffixes** on `gh release create` asset paths set the human-readable label on the
  Releases page ("macOS (Apple Silicon)", "Windows Installer (64-bit, auto-update)").
- **Idempotent re-release**: delete the existing release, the local tag, and the remote tag before
  creating, so re-running the same version just works.
- **Known bug to avoid**: the two parallel macOS builds both write `latest-mac.yml` and the last
  writer wins, so arm64 auto-update is broken. LM3 avoids this structurally by building `universal`.
- **`.gitattributes`** forces `*.sh` / `*.command` to LF and `*.bat` to CRLF — that is what keeps the
  dev launchers working cross-platform.

> **Unrelated security note for that repo**: its live `.env.signing` sits in plaintext inside a
> Dropbox folder with a real Apple ID and app-specific password. It is gitignored, but Dropbox sync is
> not a secret store — worth rotating and moving to a keychain.
