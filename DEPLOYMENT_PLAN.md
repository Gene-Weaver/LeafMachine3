# LeafMachine3 Packaging and Deployment Plan

## Status

- **Decision:** Split the current work into two source repositories:
  - **LM3**: the clean, user-facing LeafMachine3 product and deliverable.
  - **LM3-Training**: the source code used to train, evaluate, and export LM3 models.
- **Model distribution:** Publish selected trained models on Hugging Face.
- **Training data:** Keep images, labels, derived datasets, and private dataset metadata outside public repositories and container images.
- **Container strategy:** Build several focused OCI images rather than one monolithic image.
- **Cluster strategy:** Build OCI images with Docker/BuildKit, then run them on university clusters with Apptainer and the site's scheduler, such as Slurm.
- **GUI strategy:** Package Electron separately as a thin desktop client. The LM3 backend is the authoritative owner of runtime state and also serves the browser GUI.

Environment packaging (uv lockfile, `lm3 doctor`, container and wheelhouse builds) is specified in
`PACKAGING_PLAN.md`, which supersedes sections 5 and 11 below.

This plan complements `UNIFIED_RUNTIME_IMPLEMENTATION_PLAN.md`. Runtime unification should be completed before deployment behavior is treated as stable.

## 1. Objectives

The deployment design must:

1. Keep LM3 straightforward to install and run for users who do not need training code.
2. Support CLI, Python, browser, and Electron access to the same LM3 runtime.
3. Allow a CLI-started run to be discovered and monitored from the GUI.
4. Support a single user running LM3 with allocated university-cluster resources.
5. Reproduce training environments without publishing private training data.
6. Avoid storing virtual environments, caches, checkpoints, models, datasets, or generated outputs in ordinary Git repositories.
7. Keep incompatible ML dependency stacks isolated from one another.
8. Make every released LM3 model traceable to source, environment, and training metadata.

## 2. Artifact boundaries

Each artifact class should have one authoritative home.

| Artifact | Authoritative location | Included | Excluded |
|---|---|---|---|
| LM3 product source | LM3 Git repository | Python package, CLI, runtime service, web GUI, Electron source, deployment definitions, tests, documentation | Training code, datasets, model binaries, local environments, run outputs |
| Training source | LM3-Training Git repository | Trainers, evaluators, exporters, data preparation code, schemas, configs, shared libraries, tests | Private images and labels, checkpoints, caches, W&B artifacts, generated exports |
| Released models | Hugging Face model repositories | Selected weights, portable inference exports, metadata, model cards | Training datasets and unselected checkpoints |
| Runtime environments | OCI image registry | Locked runtime and training dependencies | Model caches, private data, user settings, credentials |
| Private datasets | University/project storage | Images, labels, generated datasets, private manifests | Publicly distributed source |
| Training outputs | Private project or object storage | Checkpoints, logs, W&B runs, evaluation results, intermediate exports | Ordinary Git history |
| Runtime state | User/project runtime directory | Run registry, logs, state database, job metadata | Source repository and container layers |

Git packages source. OCI images package executable environments. Hugging Face packages released models. Private storage packages data and generated research artifacts.

## 3. Repository layout

### 3.1 LM3 product repository

The LM3 repository should be the only repository required by an inference user.

Suggested structure:

```text
LeafMachine3/
├── src/leafmachine3/
│   ├── cli/
│   ├── runtime/
│   ├── server/
│   ├── workflows/
│   └── models/
├── web/
├── desktop/
├── containers/
├── deploy/
│   ├── docker/
│   ├── apptainer/
│   └── slurm/
├── tests/
├── docs/
├── models.lock.yaml
├── pyproject.toml
├── uv.lock
├── .gitignore
└── .dockerignore
```

The repository should contain:

- Headless Python and CLI interfaces.
- One long-lived backend/runtime service.
- Browser UI assets and APIs.
- Electron source and platform packaging configuration.
- Model resolution and verification code.
- CPU and NVIDIA runtime container definitions.
- Single-user cluster launch and tunnel documentation.
- Small synthetic or freely redistributable test fixtures.

### 3.2 LM3-Training repository

Keep the different training projects in one source monorepo unless an individual project later needs independent governance, licensing, or release ownership.

Suggested structure:

```text
LeafMachine3-Training/
├── packages/
│   └── lm3_training_common/
├── projects/
│   ├── archival_detector/
│   ├── label_classifier/
│   ├── landmark_detector/
│   ├── leaf_segmentation/
│   ├── plant_detector/
│   ├── ruler_classifier/
│   ├── ruler_distance_groundtruth/
│   ├── ruler_segmentation/
│   └── specimen_segmentation/
├── environments/
│   ├── yolo/
│   ├── sam3/
│   ├── birefnet/
│   ├── onnx_openvino/
│   ├── tensorrt/
│   └── data_tools/
├── containers/
├── tests/
├── docs/
├── .gitignore
└── .dockerignore
```

Each incompatible environment should have its own `pyproject.toml` and lockfile. Do not force all training projects into one shared dependency solution merely because they occupy one Git repository.

### 3.3 Why two repositories

The split provides:

- A small and stable LM3 deliverable.
- Faster installation and CI for inference users.
- Independent release cadences for product and research code.
- A clear boundary between supported runtime code and experimental training workflows.
- Easier licensing, documentation, security review, and issue tracking.
- No requirement for ordinary users to clone or understand the training stack.

Avoid Git submodules initially. The LM3-Training monorepo can share internal code directly, and released models create a clean interface between training and LM3.

## 4. Source migration and Git hygiene

The existing training directories contain a large amount of generated state. Their current disk usage should not be interpreted as the necessary Git repository size.

Do not initialize a repository over the entire working tree and run `git add .`. Instead:

1. Create a clean, empty LM3-Training repository directory.
2. Install `.gitignore` and `.dockerignore` rules before copying source.
3. Copy an explicit allowlist of reviewed source, config, documentation, schema, and test files.
4. Review every included file for credentials, private paths, internal hostnames, and dataset records.
5. Use `git status` and a staged-file audit before the first commit.
6. Run secret scanning before publishing.

At minimum, exclude:

```gitignore
.venv*/
**/.venv*/
**/.cache/
**/__pycache__/
**/.pytest_cache/
**/wandb/
**/runs/
**/logs/
**/checkpoints/
**/weights/
**/exports/
**/exported/
**/data/
**/datasets/
**/*.pt
**/*.pth
**/*.ckpt
**/*.onnx
**/*.engine
**/*.plan
**/*.ort
**/*.db
**/*.db-shm
**/*.db-wal
PRIVATE.yaml
.env
```

The final ignore rules must be reviewed against legitimate source files. Do not broadly ignore formats such as all YAML, JSON, CSV, or text files because those formats may contain necessary configuration, schemas, or small test fixtures.

No `.venv` directory should be copied, committed, or distributed. Virtual environments are reconstructed locally from lockfiles or provided by containers.

## 5. Dependency and environment strategy

LM3 currently spans dependency families that should remain isolated. In particular, YOLO, SAM3, BiRefNet, inference, and model exporters may require different NumPy, PyTorch, CUDA, ONNX Runtime, and framework versions.

Recommended environment families:

| Environment | Purpose |
|---|---|
| `lm3-runtime-cpu` | Portable CPU inference, CLI, backend, and web GUI |
| `lm3-runtime-cuda` | NVIDIA GPU inference, CLI, backend, and web GUI |
| `lm3-train-yolo` | Compatible Ultralytics-based training projects |
| `lm3-train-sam3` | SAM3 training with its required dependency constraints |
| `lm3-train-birefnet` | BiRefNet/specimen segmentation training |
| `lm3-export-onnx-openvino` | Portable inference export and validation |
| `lm3-export-tensorrt` | NVIDIA-specific optimization and engine generation |
| `lm3-data-tools` | Labelbox synchronization, aggregation, conversion, and validation |
| Native macOS export environment | CoreML conversion, which should not be treated as a Linux-container task |

For each environment:

- Declare supported Python versions.
- Maintain a direct dependency declaration and an exact lockfile.
- Pin container base images by immutable digest for releases.
- Record the CUDA and framework compatibility range.
- Include a fast environment smoke test.
- Build and test the container in CI.
- Publish the final image by both semantic tag and immutable digest.

Use independent lock roots where dependency constraints conflict. A top-level task runner can provide consistent commands without combining all packages into a single lockfile.

## 6. Container strategy

### 6.1 Containers are execution packages

Containers should package code and dependencies, but not mutable application state.

Do not bake the following into normal images:

- Released model binaries.
- Hugging Face caches.
- Private datasets.
- User settings.
- LM3 run state or logs.
- Labelbox or Hugging Face credentials.
- Training checkpoints and W&B artifacts.

Mount these at runtime instead.

### 6.2 Build separate final images

Use common, reusable build stages where this reduces duplication, but publish distinct final images. A representative naming scheme is:

```text
ghcr.io/<organization>/lm3-runtime-cpu:<version>
ghcr.io/<organization>/lm3-runtime-cuda:<version>
ghcr.io/<organization>/lm3-train-yolo:<version>
ghcr.io/<organization>/lm3-train-sam3:<version>
ghcr.io/<organization>/lm3-train-birefnet:<version>
ghcr.io/<organization>/lm3-export-tensorrt:<version>
ghcr.io/<organization>/lm3-data-tools:<version>
```

Use multi-stage Dockerfiles so compilers, package caches, test dependencies, and build-only tools do not remain in runtime images.

### 6.3 Runtime mounts

A normal LM3 container should receive explicit mounts for:

```text
/workspace/input       read-only input data when possible
/workspace/output      writable workflow outputs
/workspace/runtime     runtime registry, logs, and state database
/workspace/config      selected LM3 settings
/workspace/model-cache persistent Hugging Face/model cache
```

The exact host paths are deployment-specific and must not be embedded in application logic.

## 7. Hugging Face model distribution

### 7.1 Repository organization

Prefer one Hugging Face model repository per independently versioned model family, grouped under one organization or collection. For example:

```text
leafmachine3/lm3-plant-detector
leafmachine3/lm3-landmark-detector
leafmachine3/lm3-archival-detector
leafmachine3/lm3-specimen-segmentation
leafmachine3/lm3-ruler-classifier
```

This allows model families to evolve independently while the LM3 product pins a tested combination.

### 7.2 Model lock manifest

LM3 should include a release-controlled `models.lock.yaml` containing at least:

```yaml
schema_version: 1
models:
  plant_detector:
    repo_id: leafmachine3/lm3-plant-detector
    revision: FULL_HUGGING_FACE_COMMIT_HASH
    file: inference/model.onnx
    sha256: EXPECTED_FILE_HASH
    backend: onnxruntime
    lm3_compatibility: ">=3.0,<4.0"
```

Requirements:

- Pin the full immutable Hugging Face commit, not `main`.
- Verify file hashes after download.
- Download only the files required by the chosen backend.
- Store the cache outside the source tree.
- Record the resolved model revision in each LM3 run manifest.
- Provide clear errors for missing, incompatible, or corrupt model artifacts.

### 7.3 Connected and offline installations

Connected installations can download pinned artifacts into a persistent cache on first use.

For clusters with restricted compute-node networking, provide a preflight workflow resembling:

```text
lm3 models prefetch --release <version> --cache <persistent-path>
lm3 models verify --release <version> --cache <persistent-path>
```

The batch job then mounts the verified cache and runs without network access. An optional release-specific offline model bundle can be provided for fully air-gapped installations, but it should remain separate from the base runtime image.

### 7.4 TensorRT

Treat ONNX or another portable representation as the primary distributable when possible. TensorRT engines can be sensitive to TensorRT version, platform, and GPU architecture. Prefer generating and caching an engine on the destination system, keyed by:

- Source model revision and hash.
- TensorRT version.
- CUDA/runtime version.
- GPU architecture or compute capability.
- Precision and optimization settings.

Do not silently reuse an incompatible engine.

## 8. Private training data and reproducibility

LM3-Training will publish training code, not the training datasets.

The public repository should include:

- Expected dataset directory layouts.
- Label and metadata schemas.
- Data validation commands.
- Conversion, splitting, and preprocessing code.
- Configuration templates containing placeholder paths.
- A tiny synthetic or redistributable fixture for tests.
- Documentation explaining how a user supplies their own dataset.
- A dataset fingerprinting utility.

It should not include:

- Images or labels from the private corpus.
- Dataset exports or archives.
- Local databases containing records or paths.
- QC images derived from private data.
- Labelbox credentials or project identifiers that should remain private.
- Absolute paths tied to `/datac/Labelbox_Dump` or another developer machine.

Every important training run should produce a run manifest recording:

```text
training Git commit
container image digest
environment lock version
base-model repository and revision
private dataset identifier and fingerprint
sample counts and split fingerprints
hyperparameters
random seed
hardware description
evaluation results
promoted Hugging Face repository and revision
```

The dataset identifier and fingerprint provide provenance without exposing dataset contents. Model cards should state that training data are not distributed and document relevant limitations, intended use, and evaluation information.

## 9. Unified CLI, server, browser, and Electron architecture

### 9.1 Backend authority

The Python backend must be the sole authority for:

- Active and historical runs.
- Project and settings selection.
- Workflow state transitions.
- Process and scheduler job identity.
- Progress, logs, errors, and terminal state.
- Model versions used by a run.

The CLI, browser GUI, and Electron client must invoke or observe this same backend. They must not maintain independent definitions of an active project or active run.

### 9.2 Settings

The GUI may edit supported LM3 settings files through the backend, with validation and atomic writes. It must not create a parallel GUI-only settings authority.

Each run should record:

- The settings file path used.
- A snapshot or canonical serialization of effective settings.
- A settings content hash.
- CLI overrides and their resolved values.

This makes a run understandable even if the original settings file changes later.

### 9.3 Electron

Electron should be packaged as a normal platform-specific desktop application and should not contain the Python ML environments.

Electron responsibilities:

- Display the shared web interface.
- Discover or launch a local LM3 backend when explicitly requested.
- Connect to an existing backend URL.
- Enforce a single Electron process per desktop user as a convenience.
- Restore the most recent valid backend connection.

Electron must not be relied upon to keep a run alive or define which run is active.

### 9.4 Browser access

The same web GUI should work directly in a browser. This is the primary GUI route for cluster deployments and also provides a debugging route independent of Electron.

## 10. Single-user cluster deployment

### 10.1 Execution model

Use OCI images as the build and registry format. On clusters that do not permit Docker daemons, pull or convert the selected OCI image to an Apptainer SIF image.

A representative workflow is:

```text
Developer/CI builds OCI image
        ↓
Image is pushed to a registry
        ↓
User runs `apptainer pull` on the cluster
        ↓
User submits an LM3 Slurm job
        ↓
LM3 backend starts inside the allocation
        ↓
User opens an SSH tunnel
        ↓
Browser or Electron connects to localhost
```

### 10.2 Cluster launch behavior

The deployment utilities should:

1. Select or validate a persistent project runtime directory.
2. Select a verified model cache.
3. Submit a scheduler job requesting the necessary CPU, memory, time, and GPUs.
4. Start the LM3 backend inside the allocation.
5. Bind the backend to `127.0.0.1` or another explicitly configured secure interface.
6. Write connection metadata into the job-scoped runtime record.
7. Print the exact SSH tunnel command needed by the user.
8. Preserve backend state and logs when the browser or Electron disconnects.
9. Reconcile the runtime state with the scheduler job when the user reconnects.

Conceptual tunnel:

```bash
ssh -N -L 8123:<allocated-node>:8123 <user>@<cluster-login-host>
```

The user then opens:

```text
http://localhost:8123
```

The actual command may require a login-node jump or site-specific proxy. Document this per supported cluster rather than embedding one institution's topology into LM3.

### 10.3 Cluster security boundary

For the initial single-user deployment:

- Bind to localhost by default.
- Use SSH tunneling rather than exposing an unauthenticated public port.
- Do not put Hugging Face, Labelbox, or registry tokens in images.
- Pass secrets at runtime using approved files, environment injection, or the scheduler's secret mechanism.
- Use restrictive permissions on runtime files and logs.
- Treat the user account and SSH authentication as the initial access-control boundary.

Multiuser tenancy, shared services, and public web exposure are explicitly out of scope for this phase.

## 11. Local installation options

Support more than one installation mode without creating different application semantics.

### Desktop-first installation

- Install the packaged Electron application.
- Install or start the matching LM3 backend using a supported local mechanism.
- Electron connects to the local backend.
- Appropriate for ordinary workstation users.

### Container-first installation

- Pull the CPU or CUDA LM3 runtime image.
- Mount settings, input, output, runtime, and model-cache directories.
- Open the browser GUI or connect Electron to the exposed local port.
- Appropriate for Linux workstations and reproducible deployments.

### Native developer installation

- Clone the relevant source repository.
- Reconstruct an environment from the committed lockfile.
- Run the same backend and clients used by packaged deployments.
- Appropriate for development and debugging, not the primary end-user installation route.

## 12. CI, releases, and provenance

### 12.1 LM3 CI

For each supported runtime target:

- Lint and unit-test Python code.
- Test CLI and server entry points.
- Test settings validation and runtime-state reconciliation.
- Build web assets.
- Build CPU and CUDA container images.
- Run container smoke tests.
- Verify the model lock manifest.
- Build Electron installers on the corresponding operating systems.
- Generate an SBOM for release containers.

### 12.2 LM3-Training CI

Use path-scoped jobs so changes to one trainer do not rebuild every training image unnecessarily.

- Validate all configuration files.
- Run dataset-schema tests using synthetic fixtures.
- Build changed environment targets.
- Import each training package in its own environment.
- Run a tiny CPU or reduced-resource smoke test where feasible.
- Validate exporters independently.
- Scan staged source for unexpectedly large files and secrets.

Full GPU training should not run on every pull request. Schedule selected GPU integration tests separately or run them before a model promotion.

### 12.3 Release linkage

An LM3 release should identify:

- LM3 Git tag and commit.
- Runtime container tags and immutable digests.
- Electron installer versions.
- `models.lock.yaml` revision.
- Supported Python, CUDA, GPU, and platform ranges.
- Database/runtime-state schema version.
- Upgrade and rollback notes.

A promoted model should identify:

- LM3-Training Git commit.
- Training container digest.
- Private dataset fingerprint.
- Evaluation record.
- Expected input/output contract.
- Compatible LM3 versions.

## 13. Implementation phases

### Phase 1: Inventory and clean source extraction

- Classify the current LM3 and training files as source, configuration, model, private data, cache, environment, or output.
- Create a reviewed source allowlist for each training project.
- Identify reusable shared training utilities.
- Audit credentials and hard-coded paths.
- Produce initial `.gitignore` and `.dockerignore` policies.

**Exit condition:** The proposed source for both repositories is known, and no private or generated artifact is accidentally included.

### Phase 2: Create the repository split

- Keep/refine the LM3 product repository.
- Create the clean LM3-Training source monorepo.
- Move shared training utilities into an explicit package.
- Preserve legally required notices and third-party licenses.
- Add repository-specific README, contribution, and security documentation.

**Exit condition:** A user can clone either repository without downloading datasets, models, environments, or generated outputs.

### Phase 3: Normalize configuration and paths

- Replace machine-specific paths with typed configuration or CLI parameters.
- Document environment variables and directory contracts.
- Provide safe example configuration without credentials.
- Validate missing input, output, cache, and runtime paths clearly.

**Exit condition:** Source code can run outside `/datac/Labelbox_Dump` without edits.

### Phase 4: Lock independent environments

- Define the supported environment families.
- Add a `pyproject.toml` and exact lockfile for each.
- Document why incompatible stacks are separate.
- Add import and basic execution smoke tests.

**Exit condition:** Each environment can be reconstructed from source on a clean machine.

### Phase 5: Implement unified runtime behavior

- Complete the work in `UNIFIED_RUNTIME_IMPLEMENTATION_PLAN.md`.
- Ensure CLI-started and GUI-started runs use the same runtime registry.
- Make the backend authoritative for active-run state.
- Remove redundant GUI-only control paths after migration tests pass.

**Exit condition:** CLI, browser, and Electron consistently observe the same runs and settings.

### Phase 6: Introduce Hugging Face model management

- Establish the Hugging Face organization and repository naming rules.
- Select portable release artifacts for each model family.
- Add model cards and provenance metadata.
- Implement `models.lock.yaml`, download, cache, verification, and offline prefetch behavior.
- Record resolved model identities in run manifests.

**Exit condition:** A clean LM3 installation can obtain and verify all models without bundling them into Git or the base image.

### Phase 7: Build focused OCI images

- Build CPU and CUDA LM3 runtime images.
- Build independent training/export images.
- Add non-root users, minimal runtime layers, health checks, and SBOM generation.
- Pin release bases and publish immutable digests.

**Exit condition:** A clean Docker/OCI host can run the supported LM3 and training smoke tests using only documented mounts.

### Phase 8: Package Electron

- Make Electron a thin client of the backend.
- Implement local connection discovery and explicit remote URL configuration.
- Add a desktop single-instance lock.
- Build and sign platform installers as appropriate.

**Exit condition:** Closing or reopening Electron does not alter backend run ownership, and Electron can reconnect to local or tunneled cluster backends.

### Phase 9: Validate Apptainer and Slurm deployment

- Pull the released OCI runtime into Apptainer.
- Validate NVIDIA GPU passthrough.
- Add example Slurm scripts with configurable resources and mounts.
- Add model-prefetch and offline validation steps.
- Validate the SSH-tunneled browser GUI.
- Test reconnecting after closing the browser or Electron.

**Exit condition:** One cluster user can submit, monitor, reconnect to, and complete an LM3 run without Docker privileges or public network exposure.

### Phase 10: Documentation and release rehearsal

- Write quick starts for local CPU, local NVIDIA, cluster inference, and cluster training.
- Document supported environments and known incompatibilities.
- Rehearse installation from a clean account with no developer caches.
- Test failure cases: missing model, corrupt cache, job cancellation, node failure, stale runtime record, and incompatible GPU.

**Exit condition:** A new user can install and run LM3 using only published documentation and artifacts.

## 14. Explicit non-goals for the initial release

- A shared multiuser hosted LM3 service.
- Publicly exposed cluster web services.
- Bundling every trainer into the LM3 runtime image.
- Publishing private datasets.
- Uploading all historical checkpoints or training outputs.
- Supporting every model export framework in one environment.
- Treating Electron as the process supervisor or runtime authority.
- Guaranteeing that one TensorRT engine works across all GPUs and TensorRT versions.

## 15. Recommended first implementation milestone

The first practical milestone should demonstrate the architecture end to end with one representative model:

1. Extract the plant-detector training source into LM3-Training.
2. Lock and containerize its training environment.
3. Publish one selected inference model to Hugging Face.
4. Add that artifact to the LM3 model lock manifest.
5. Run LM3 locally from the CUDA runtime container.
6. Run the same OCI image with Apptainer in a Slurm allocation.
7. Start the run from the CLI and observe it through the tunneled browser GUI.
8. Reconnect with Electron and confirm that it shows the same backend-owned run.

This vertical slice will reveal packaging, runtime, model, scheduler, and GUI integration problems before every training project is migrated.

## 16. Final target architecture

```text
                        SOURCE AND BUILD

       LM3 Git repository                 LM3-Training Git repository
               |                                      |
               | CI                                   | CI / training
               v                                      v
      Runtime OCI images                    Training OCI images
               |                                      |
               |                                      v
               |                           Private datasets and outputs
               |                                      |
               |                               selected models
               |                                      v
               +----------------------- Hugging Face model repos
               |                                      |
               +------------------+-------------------+
                                  |
                                  v
                         DEPLOYED LM3 BACKEND
                         /        |         \
                        /         |          \
                     CLI      Browser GUI   Electron
                                  |
                    local Docker or cluster Apptainer/Slurm
```

The central architectural rule is that every interface controls or observes the same LM3 backend, while source, environments, models, data, and runtime state remain separate, versioned artifact classes.
