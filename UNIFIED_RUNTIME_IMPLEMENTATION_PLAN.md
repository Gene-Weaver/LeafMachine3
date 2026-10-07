# Unified LM3 runtime implementation plan

Date: 2026-08-28  
Status: proposed; no implementation performed

## 1. Outcome and governing invariants

The implementation should leave LM3 with two deliberately separate sources of truth:

- **Runtime truth:** what is executing now, published by the executing LM3 process through one shared runtime lease/record.
- **Settings truth:** the canonical YAML edited by the GUI and used as intent for the next GUI-started run.

The following invariants are acceptance requirements, not aspirations:

1. At most one LM3 pipeline or hardware-tuning activity can hold a given runtime deployment lease at a time. On a desktop the deployment is the user's local LM3 installation; on a cluster it is one scheduler allocation/job.
2. Every entry point reaches the same lease code: `machine3` CLI, `python -m leafmachine3`, direct `machine3()` calls, the GUI launcher, and legacy server jobs.
3. `GET /v1/runtime` and the GUI obtain active identity from the lease record, never from mutable YAML or a filesystem recency guess.
4. An active record contains the exact config path and effective project identity used at launch.
5. Status, logs, results, and postprocessing follow the active record's DB/run path while a run is active.
6. Editing settings during a run never changes which run the GUI is displaying. The UI labels those edits as applying to the next run.
7. Closing Electron never stops an LM3 run. Stopping a run is a separate, explicit action.
8. Electron stops a server only when it can prove that this Electron process spawned that exact server instance.
9. A second Electron launch focuses the existing window and exits before starting or attaching to a server.
10. No process is signaled solely because a PID appeared in a JSON file or unauthenticated `/healthz` response.

The coordination boundary is deliberately **one single-user LM3 deployment**. On a workstation that means one OS user on one host. On a cluster it means one user's scheduler allocation/container, identified by `LM3_DEPLOYMENT_ID` and given its own node-local `LM3_RUNTIME_DIR`. True multi-user service deployment, cross-user arbitration, and a central scheduler-submission portal are outside this change.

## 2. Target architecture

```text
CLI / Python API ─┐
                  ├─> machine3() ─> RuntimeLease ─> active.json + OS-held lock
GUI / HTTP start ─┘                         │
                                            ├─> immutable launch manifest in run dir
                                            └─> project SQLite ledger

Electron ─> LM3 server ─> GET /v1/runtime ─> RuntimeRegistry reader
                    │
                    ├─> status/logs/results use active.db_path/run_dir
                    └─> settings UI uses canonical next-run settings path
```

The file lock determines exclusivity and liveness. JSON describes the lock holder; JSON never grants authority by itself.

## 2.1 Supported single-user cluster profile

The first cluster target is one user installing LM3 in their own cluster-accessible resources and running one LM3 deployment inside a scheduler allocation. It does not require a central service or administrator-operated multi-user portal.

```text
user submits Slurm job
        │
        └─> one compute-node allocation
              └─> LM3 container / environment
                    ├─> lm3-serve (web UI + observer API)
                    └─> machine3   (CLI pipeline + runtime-lease owner)

user browser ── SSH tunnel ──> compute-node lm3-serve URL
```

Required behavior:

- The server and pipeline run in the same allocation and see the same node-local runtime directory and active project DB.
- The pipeline may still be launched from the CLI or a batch script. It registers itself automatically, so the server does not have to launch it.
- The browser GUI shows the CLI run, stage progress, logs, and results.
- Start is disabled while the CLI run owns the deployment lease.
- The initial cluster version treats a CLI run as observer-only: Stop is performed through the terminal or scheduler (`scancel` for Slurm). Cooperative GUI stopping remains optional Phase 9.
- Closing the browser, Electron, an SSH tunnel, or the observer server does not stop the pipeline.
- The installation may be a Docker/OCI image, an Apptainer/Singularity image derived from it, or a conventional virtual environment. The runtime contract is the same.

Use a job-scoped namespace rather than a shared-home user lock:

```bash
export LM3_DEPLOYMENT_ID="slurm-${SLURM_JOB_ID}"
export LM3_RUNTIME_DIR="${SLURM_TMPDIR}/lm3-runtime"
```

If `SLURM_TMPDIR` is unavailable, the job wrapper must create a private directory on another node-local scratch filesystem. Do not fall back to a shared home directory for the active lock or registry: two legitimate allocations for the same user must not block one another.

LM3's active SQLite database also needs same-host-safe storage. SQLite WAL mode is not supported across machines on a network filesystem because the WAL index uses shared memory ([SQLite WAL documentation](https://www.sqlite.org/wal.html)). The reference cluster deployment should therefore keep the active DB on node-local scratch and copy/checkpoint it into persistent project storage at completion. This requires a supported separation between persistent artifact output and active state/DB storage; it must not be implemented as an undocumented wrapper hack.

Remote access for this profile uses a unique per-job port and an SSH tunnel. The server binds to loopback whenever the cluster allows SSH/ProxyJump access to the compute node. The job log prints the node, port, deployment ID, and connection command, and the user visits a local URL such as `http://127.0.0.1:8765/`. A remotely attached Electron shell is optional; the ordinary browser is the primary cluster client.

Open OnDemand and reverse-proxy support are useful future refinements for this same single-user-per-allocation model, but are not required for the first cluster deliverable. A persistent multi-user LM3 portal is explicitly out of scope.

## 3. Core contracts

### 3.1 Runtime directory

Add `leafmachine3/core/runtime.py` and resolve its storage directory in this order:

1. explicit `runtime_dir` argument;
2. `LM3_RUNTIME_DIR`;
3. for a scheduler job, a private directory below node-local job scratch;
4. platform user-runtime location for a desktop/local deployment;
5. a documented per-user cache fallback only for a desktop/local deployment.

Keep it independent of the checkout and current working directory so separate LM3 installations and arbitrary CLI working directories still coordinate. Create the directory with user-only permissions where the platform supports them.

`LM3_DEPLOYMENT_ID` namespaces records and diagnostics. Local desktop launches default to a stable local deployment ID; Slurm reference scripts set `slurm-${SLURM_JOB_ID}`. When scheduler metadata is present but no safe node-local runtime directory can be resolved, fail before running rather than silently using shared home storage.

Files:

- `activity.lock` — stable lock inode; never delete it during normal operation.
- `active.json` — atomic description of the current lock holder.
- `last.json` — final record for the most recently completed, stopped, failed, or abandoned activity.

Use a small standard-library cross-platform lock adapter (`fcntl.flock` on POSIX and a Windows locking implementation). Do not infer exclusivity from file existence. Write JSON through temp-file + `fsync` + `os.replace`; validate schema and bound field sizes on read.

### 3.2 Runtime record schema

Version the contract from the first release. A proposed record is:

```json
{
  "schema_version": 1,
  "run_id": "uuid",
  "activity": "pipeline",
  "state": "starting",
  "launcher": "cli",
  "pid": 1234,
  "process_started_at": 1787932800.25,
  "started_at": "2026-08-28T12:00:00Z",
  "updated_at": "2026-08-28T12:00:01Z",
  "deployment": {
    "id": "slurm-482193",
    "scheduler": "slurm",
    "job_id": "482193",
    "step_id": "0",
    "node": "gpu042",
    "container_id": null
  },
  "config": {
    "path": "/abs/path/settings.yaml",
    "sha256": "..."
  },
  "project": {
    "run_name": "acer_rubrum",
    "input_dirs": ["/abs/input"],
    "output_dir": "/abs/output",
    "run_dir": "/abs/output/acer_rubrum",
    "db_path": "/abs/output/acer_rubrum/acer_rubrum.sqlite",
    "log_path": "/abs/output/acer_rubrum/logs/lm3.log"
  },
  "overrides": {
    "run_name": "acer_rubrum",
    "input_dir": "/abs/input",
    "output_dir": "/abs/output",
    "restart": []
  },
  "control": {
    "mode": "managed-child",
    "owner_instance_id": "server-uuid"
  },
  "error": null,
  "finished_at": null,
  "returncode": null
}
```

Rules:

- Do not store bearer tokens, environment dumps, or other secrets.
- `run_id` identifies one invocation even when it resumes the same project DB.
- `deployment.id` identifies the local desktop installation or scheduler allocation within which exclusivity applies.
- Scheduler/node/container fields are descriptive. They never authorize PID signaling across a host or container boundary.
- `config.path` is the source YAML; `config.sha256` fingerprints the bytes actually loaded.
- Project paths are absolute and resolved before publication.
- `launcher` is descriptive (`cli`, `python`, `server`, `legacy-job`), not an authorization decision.
- `control` is capability metadata. A server may stop a run only when it also holds a live local child handle matching `run_id`, PID, and process creation time. Metadata alone is insufficient.

### 3.3 Registry lifecycle

`RuntimeLease` should support these transitions:

```text
lock acquired → starting → running → done
                            ├──────→ error
                            ├──────→ stopped
                            └──────→ interrupted

lock released without finalization → abandoned (classified by the next reader)
```

Acquisition must happen after cheap config loading/validation has resolved project identity, but before hardware profiling, orphan-worker cleanup, directory creation, database writes, model loading, or GPU work. If acquisition fails, raise a typed `RuntimeBusyError` containing the sanitized current record and exit the CLI with a distinct nonzero code.

Hold the lock file descriptor for the entire pipeline. In `finally`, atomically write the terminal record to `last.json`, remove or replace `active.json`, then release the lock. A reader finding `active.json` while the lock is acquirable classifies the record as stale/abandoned under the cleanup lock; it must never signal the recorded PID.

### 3.4 Immutable launch manifest

After `build_dirs()` resolves the final temp/run paths, write `<run>/logs/run_manifest.json` atomically. It should contain:

- runtime schema and `run_id`;
- source config absolute path and SHA-256;
- the complete effective config after defaults and CLI/programmatic overrides;
- all explicit overrides;
- resolved project paths;
- LM3/Python/platform versions and launcher type;
- start timestamp.

Update only terminal fields at completion. This replaces the batch's need to generate a full YAML merely for provenance while retaining stronger provenance than the generated YAML currently provides.

## 4. Phase-by-phase implementation

### Phase 0 — Freeze contracts and add characterization tests

Goal: capture current behavior before refactoring.

Actions:

- Add endpoint-contract tests for `/v1/run/active`, `/v1/run/start`, `/v1/run/stop`, `/v1/status`, `/v1/runs`, and `/v1/settings`.
- Add characterization tests proving the current settings-path split, configured-project precedence, and CLI invisibility. Mark them as expected-to-change rather than encoding the bugs permanently.
- Inventory external consumers of legacy `/v1/jobs`; the static UI defines helpers for it but uses `/v1/run/start` for normal runs.
- Record a clean CLI mock-pipeline baseline so runtime wrapping can be shown not to alter pipeline results or resume behavior.

Files: new `tests/test_runtime_registry.py`, `tests/test_runtime_api.py`, and additions to `tests/test_settings_ui.py`, `tests/test_config.py`, and pipeline integration tests.

Exit gate: tests demonstrate the contradictions reliably without requiring CUDA or a real model.

### Phase 1 — Canonicalize configuration and workspace paths

Goal: every subsystem resolves the same settings and hardware files.

Add `leafmachine3/core/paths.py` with a `RuntimePaths`/resolver API for:

- canonical next-run settings;
- hardware settings;
- postprocessing settings;
- server jobs directory;
- runtime registry directory;
- configured run roots.

Canonical names:

- `LM3_SETTINGS`
- `LM3_HARDWARE`
- `LM3_POSTPROCESS_SETTINGS`
- `LM3_SERVER_JOBS`
- `LM3_RUNS_ROOTS`
- `LM3_RUNTIME_DIR`
- `LM3_DEPLOYMENT_ID`

Migration behavior for one release:

- If only `LM3_SETTINGS_PATH` is set, honor it and emit a deprecation warning.
- If `LM3_SETTINGS` and `LM3_SETTINGS_PATH` resolve to the same file, use it and warn once.
- If they resolve differently, fail server startup with a precise split-brain error rather than choosing silently.
- Apply the same rule to `LM3_HARDWARE_SETTINGS` versus `LM3_HARDWARE`.

Replace local path logic in:

- `leafmachine3/server/settings_api.py`
- `leafmachine3/server/metrics_api.py`
- `leafmachine3/server/progress_api.py`
- `leafmachine3/server/postprocess_api.py`
- `leafmachine3/server/results_api.py`
- `leafmachine3/server/app.py` / `JobManager._base_settings()`

Expose resolved paths in server startup logs and `GET /healthz` diagnostics. Preserve explicit `yaml_path` support for deliberate file editing, but the default must always come from the shared resolver.

Exit gate: with any supported CWD, every subsystem reports the same canonical settings path; conflicting legacy variables stop startup.

### Phase 2 — Build the runtime registry and wrap `machine3()`

Goal: make all execution entry points visible and mutually exclusive without server participation.

Actions:

- Implement `RuntimeLease`, `RuntimeRecord`, `RuntimeBusyError`, atomic record IO, and the platform lock adapter in `leafmachine3/core/runtime.py`.
- Add a serializable `Config.to_dict()` and a pure project-path resolver so expected run/DB paths can be published before `build_dirs()` writes anything.
- Extend `machine3()` with keyword-only `run_name`, `launcher`, and optional runtime context arguments.
- Extend `_cli_overrides()` and the CLI parser with `--run-name`.
- Load and validate config, calculate absolute project identity, acquire the lease, then continue into hardware/profile/directory/model work.
- Update the record from `starting` to `running` when the DB and log paths are ready.
- Finalize the record and manifest on normal return, exception, and `KeyboardInterrupt`.
- Ensure a busy second invocation does not call `reap_orphaned_workers()`, touch output directories, initialize CUDA, or mutate a project DB.

Direct Python callers receive the same behavior automatically because the lease wraps the public `machine3()` function, not just `main()`.

Exit gate: a helper subprocess holding a lease makes CLI, direct Python, server start, and legacy jobs all refuse a second run with the same active identity.

### Phase 3 — Replace the server's private run truth

Goal: make server run APIs a view/controller over the shared registry.

Refactor `metrics_api.py`:

- Replace `_RUN`, `_ADOPT_TRIED`, and `active_run.json` adoption as authoritative state.
- Retain only a small `_ManagedChild` object for a child this server actually spawned: `Popen`, `run_id`, PID creation time, log handle, and server instance ID.
- Add `GET /v1/runtime` returning `{active, last, next_run_settings, server, diagnostics}`.
- Keep `GET /v1/run/active` temporarily as a compatibility projection of `/v1/runtime`.
- Make `POST /v1/run/start` pass the canonical config path explicitly and wait for either the child's matching registry record or child exit. Return 409 with the actual active record when the lease is busy.
- Derive `can_stop` only by matching the registry record to `_ManagedChild`. After a server restart or for a CLI run, report `can_stop: false` and a reason.
- Make `POST /v1/run/stop` reject observer-only runs with 409/403; never synthesize a process group from registry JSON.
- When this server stops its own child, verify `run_id`, PID, and process creation time before signaling. Preserve the current process-group escalation only for that verified child.

Do not call this behavior “adoption.” A restarted server observes the live run but does not acquire control of it.

Legacy `JobManager`:

- Route pipeline execution through the same `machine3()` lease immediately.
- Deprecate `/v1/jobs` if there is no real external consumer.
- If uploads remain valuable, make `/v1/jobs` a staging API that ultimately invokes the same `/v1/run/start` service rather than maintaining its own queue/run state.
- Remove duplicate status/results logic only after compatibility tests or a documented API break.

Exit gate: a CLI run appears in `/v1/runtime` within one polling interval; Start is rejected; Stop says the run is externally controlled; a GUI-started run remains stoppable only by its launching server instance.

### Phase 4 — Make active runtime the status authority

Goal: eliminate YAML/discovery guesses for live status while retaining historical browsing.

Change `progress_api.resolve_run()` precedence to:

1. explicit `db=` or explicit historical `run=` selector;
2. active runtime registry record;
3. explicitly selected server/UI historical run;
4. canonical next-run project from settings, for an idle prepared project;
5. filesystem discovery, only as legacy/history fallback.

Specific changes:

- Replace `_BOUND`, `_BOUND_STATE`, `bind_run()`, and metrics-to-progress binding with a registry reader.
- Build the active `RunRef` directly from `active.project.db_path` and `run_dir`.
- Use the runtime `run_id` in snapshot cache keys so two invocations of the same project cannot alias.
- Make logs dynamically follow the active `run_id` in “follow active” mode.
- On completion, follow `last.json` rather than jumping to the most recently modified unrelated run.
- Keep explicit `?run=`/`?db=` for historical inspection.
- Remove the duplicate progress-router `GET /v1/runs`; retain the richer `results_api` route as the only run listing.
- Make `results_api.run_roots()` use the shared settings resolver and include the active/last runtime output roots explicitly.
- Make postprocessing default to the selected run/runtime run path, not re-derive it from the next-run YAML.

Filesystem discovery remains useful for older runs and pre-registry processes, but it must be labeled `source: discovered` and must never enable Start/Stop decisions.

Exit gate: changing `project.run_name` during an active run does not move status, logs, results, or postprocessing away from the active DB.

### Phase 5 — Fix Electron instance and server ownership

Goal: make the desktop shell a safe client of a separately identifiable server.

Single instance:

- Call `app.requestSingleInstanceLock()` before `whenReady()`, IPC registration with side effects, or `ensureServer()`.
- If acquisition fails, quit immediately.
- Handle `second-instance` by restoring, showing, and focusing the existing window.

Server identity:

- Give each server a UUID `instance_id` at startup.
- Have `/healthz` return `service: "leafmachine3"`, a protocol version, `instance_id`, PID, and ownership mode.
- When Electron spawns a server, pass a random expected `LM3_INSTANCE_ID` and retain the actual child handle.
- Treat the server as owned only if the health response matches service, protocol, expected instance ID, PID, and live child handle.
- If another LM3 server is present, attach as a client and set `serverOwned=false`.
- If an unrelated service answers on the port, fail explicitly rather than attaching.

Shutdown:

- Delete the policy that attached servers are killed on Electron quit.
- On quit, close the window and stop only a verified owned server. Never stop a pipeline as part of Close.
- For an owned server, request graceful shutdown with its token, then signal only the retained child/process group after identity revalidation.
- `LM3_KEEP_SERVER` may remain as an owned-server lifecycle preference, but it is no longer an ownership safety switch.

Connection authentication needs an explicit attached-server path. Recommended behavior:

- Owned server: use the token Electron generated and passed to the child.
- Attached server: use its normal loopback bootstrap/connection descriptor; do not send a newly generated, mismatched token.
- If token bootstrap is disabled, show a connection error or token prompt rather than spawning over or killing the server.

Exit gate: closing an Electron window attached to `lm3 serve` leaves that server and any run alive; a second Electron launch creates no second server/window.

### Phase 6 — Replace the renderer's active-project state model

Goal: display “running now,” “viewing,” and “next run” as separate concepts.

Replace top-bar state with:

- `runtime.active` — authoritative current activity;
- `runtime.last` — last terminal invocation;
- `view.mode` — `follow-active` or `historical`;
- `view.runRef` — DB/run selected for status/results;
- `settings` — canonical next-run settings;
- `busy` — a request currently being submitted/stopped, not runtime truth.

Remove or rewrite:

- `openFresh()`;
- sticky `state.newRun` suppression;
- blanking the visible run name without writing YAML;
- logic that decides run activity from a status snapshot;
- Close/New run paths that stop a job automatically.

UI behavior:

- A persistent “Running now” chip/card displays run name, origin, config path, PID, run path, and control status from `/v1/runtime`.
- The primary settings strip is labeled “Next run settings.” If its config differs from the active config, show both paths and a clear informational banner.
- Start is disabled whenever any runtime lease is active, regardless of launcher.
- Stop is enabled only when `can_stop` is true. Otherwise show “Started externally; stop it from its owner” with launch metadata.
- “New run” becomes “Prepare next run”; it clears a draft only and never hides or stops active runtime status.
- Add a run selector using the existing results run list and explicit status `run`/`db` parameters.
- Default view is “Follow active.” Selecting history pauses follow mode; a visible action returns to the active run.
- Close only closes the desktop application.

The full Settings tab and top-bar quick fields continue writing the same canonical YAML through one Settings API. Keep the existing validation, atomic writes, backups, and mtime conflict checks.

Exit gate: opening the GUI during any CLI run immediately shows that run while the editable fields continue to show, and clearly label, next-run settings.

### Phase 7 — Simplify the batch workflow and preserve provenance

Goal: remove per-species config mutation as a control mechanism.

Update `run_global_greening.sh` to call:

```text
python -m leafmachine3 \
  --config LM3_settings_global_greening.yaml \
  --run-name <species> \
  --input <species-dir> \
  --output <output-root>
```

Replace generated `_configs/<species>.yaml` with the pipeline-generated immutable `run_manifest.json`. If downstream tools require YAML, optionally export an effective YAML snapshot beside the JSON, but generate it inside the shared Python launch path rather than with shell `sed`.

The sequential batch naturally releases and reacquires the runtime lease between species. The GUI in follow-active mode switches by `run_id` as each species begins. During the short inter-species gap it should show the just-completed `last` record rather than a random discovered run.

Exit gate: the batch uses one template, every species has complete launch provenance, and the GUI follows species transitions without a watcher YAML.

### Phase 8 — Ship the single-user cluster reference deployment

Goal: make the unified runtime usable by one researcher inside their own Slurm allocation/container, with browser monitoring through an SSH tunnel.

Deliverables:

- An OCI/Docker image definition and documented Apptainer-compatible execution path, or the corresponding packaging artifact selected by the deployment work.
- A reference `deploy/slurm/lm3.sbatch` wrapper that:
  - requests configurable GPUs, CPUs, memory, and wall time;
  - creates a private node-local runtime/state directory;
  - sets `LM3_DEPLOYMENT_ID=slurm-${SLURM_JOB_ID}`;
  - bind-mounts persistent input/output/settings and node-local scratch into the container;
  - chooses a unique job port;
  - starts `lm3-serve` and the CLI pipeline in the same allocation;
  - prints safe connection/tunnel instructions;
  - preserves pipeline exit status;
  - stops only the observer server during wrapper cleanup;
  - checkpoints/copies the active project DB and manifest to persistent storage on normal completion and handled termination.
- A supported `project.output.state_dir` (final name to be settled with the path resolver) so the active SQLite DB can live on node-local storage while reports/artifacts use persistent project storage.
- A connection descriptor such as `<persistent-run>/logs/lm3-connection.json` containing no bearer token, only deployment ID, scheduler job ID, node, port, run ID, and timestamps.
- CLI documentation for submitting, finding the compute node/port, creating an SSH/ProxyJump tunnel, opening the browser, reconnecting after a tunnel drop, monitoring, and stopping through Slurm.
- Container health/probe commands that verify LM3, CUDA/ONNX providers, mounted settings, writable persistent output, writable node-local state, and runtime namespace before the expensive pipeline begins.

Representative flow:

```text
sbatch deploy/slurm/lm3.sbatch --run-name acer_rubrum ...
  → job 482193 starts on gpu042
  → log prints an ssh -J ... -L 8765:127.0.0.1:<job-port> command
  → user opens http://127.0.0.1:8765/
  → GUI reports “Running via CLI · Slurm 482193 · gpu042”
  → tunnel/browser may disconnect without affecting LM3
  → job completion persists the DB, manifest, logs, and outputs
```

Security requirements:

- Prefer compute-node loopback binding plus SSH tunneling.
- If a site's routing requires binding the compute interface, require a random per-job token, a site firewall/proxy boundary, strict origin/host validation, and encrypted browser transport through the tunnel/proxy. Never advertise the raw compute-node URL as publicly safe.
- Do not put bearer tokens in the connection descriptor, scheduler listing, or normal command-line process list.
- The browser-facing server may read only the paths mounted/allowed for this deployment.

Cluster acceptance tests:

- two jobs belonging to the same Unix user but using different `SLURM_JOB_ID` namespaces can run concurrently without sharing registry state;
- two LM3 starts inside one allocation contend on the same lease and only one proceeds;
- the server started before or after the CLI process discovers the same active run;
- dropping and recreating the SSH tunnel does not change runtime state;
- stopping the observer server does not stop the pipeline;
- `scancel` yields an interrupted/abandoned record that is reconciled safely on resume;
- active DB reads occur from the same compute host and node-local state path;
- the terminal DB copy opens cleanly from persistent storage after WAL checkpoint/finalization;
- no CUDA or full-dataset test is required for ordinary CI; a Slurm-like environment-variable and mock-pipeline test covers the namespace/wrapper contract, with a real-cluster smoke test as the release gate.

Exit gate: a user can install the container/environment in their own resources, submit a GPU job, follow a CLI-started run in the browser through the printed tunnel command, disconnect/reconnect safely, and recover persistent run outputs without administrator-operated LM3 infrastructure.

### Phase 9 — Optional cooperative control of externally started runs

This phase is optional and should not delay correct observation. Do not implement external stopping by signaling recorded PIDs/process groups.

If stopping a CLI-started run from the GUI is required, add a cooperative same-user control channel owned by the LM3 process:

- registry advertises a per-run local socket/named-pipe endpoint and nonce-derived capability;
- owner validates the request and changes its own state to `stopping`;
- a cancellation token reaches the pipeline loop and executors;
- feeders stop assigning work, workers finish/checkpoint current items, and the owner terminates only worker processes it created after a grace period;
- final state is `stopped`, not `abandoned`.

This needs explicit cancellation points in `RunContext`, `run_pipeline()`, and the worker executors. Until that work is complete, external runs remain safely observable and Start remains globally blocked.

Exit gate: a cooperative stop of a CLI run checkpoints cleanly without signaling the terminal's shell process group or leaving CUDA workers.

## 5. API migration contract

Add:

- `GET /v1/runtime` — canonical runtime, last-run, next-settings, and diagnostics payload.

Keep temporarily:

- `GET /v1/run/active` — compatibility projection, with deprecation metadata/header.
- `POST /v1/run/start` — same route, now registry-backed.
- `POST /v1/run/stop` — same route, with strict `can_stop` enforcement.

Consolidate:

- one `GET /v1/runs`, owned by `results_api`;
- one settings resolver for all settings routes;
- one project reference shape shared by runtime, progress, results, and postprocessing.

Do not overload one field called `active` to mean both “the process is live” and “the historical run currently selected in the UI.” Use `runtime.active` and `view.selected` separately.

## 6. Failure and race handling

The implementation is incomplete unless these cases are designed and tested:

- **Two simultaneous CLI starts:** one acquires the lock; the other exits before any output/GPU mutation.
- **CLI versus GUI start race:** child lease acquisition decides; the losing API returns the winner's identity.
- **Process dies with SIGKILL:** OS releases lock; next reader classifies stale JSON as abandoned; next run may start.
- **PID reuse:** process creation time and lock state prevent a stale record from authorizing control.
- **Malformed/truncated active JSON:** report diagnostics, use lock state for exclusivity, and never guess a PID to signal.
- **Registry directory unavailable:** fail before starting work with a precise configuration error; do not silently run unregistered.
- **Server restarts during a run:** new server observes registry and active DB but reports `can_stop=false` unless cooperative control exists.
- **Electron crashes:** owned server watchdog may stop the server; the separate pipeline lease/run survives and remains observable after restart.
- **Settings file edited or deleted mid-run:** runtime stays pinned to its launch manifest; next-run editor reports its own file problem independently.
- **Active DB not created yet:** runtime state is `starting`; GUI shows launch identity without falling back to another run.
- **Batch transition:** `last.run_id` is shown until the next active record appears.
- **Old pre-registry run:** optional discovery can display it as unverified history/live-looking activity, but cannot block or authorize process control. During migration, label this limitation explicitly.

## 7. Test matrix

### Unit tests

- canonical/legacy settings path precedence and conflict errors;
- POSIX and Windows lock adapter behavior;
- atomic record writes and schema validation;
- lock contention using real subprocesses;
- stale/abandoned classification after forced subprocess exit;
- record redaction and field-size validation;
- `--run-name` precedence over YAML;
- manifest source hash/effective config correctness;
- runtime-to-`RunRef` mapping;
- `can_stop` identity matching and PID-reuse rejection.

### API integration tests

- externally held test lease appears in `/v1/runtime` and disables start;
- GUI-managed mock child registers, streams status, and can be stopped;
- server restart observes but does not control the child;
- active runtime outranks a conflicting YAML project;
- changing YAML during a run does not move status/log/results;
- explicit historical selector outranks follow-active only for that request;
- one `/v1/runs` route with stable IDs;
- all modules report the same settings path;
- legacy env conflicts fail loudly.

### Electron/renderer tests

- second instance focuses the first;
- unrelated port occupant produces an error;
- attached manual LM3 server is never killed on Close;
- owned server is stopped only after identity verification;
- Close never calls `/v1/run/stop`;
- external runtime is shown immediately;
- next-run settings remain visible and distinct;
- active-to-next-batch-run transitions update the UI by `run_id`;
- history selection and “Follow active” behave independently.

### End-to-end gates

Run with compute mocks where possible:

1. CLI first, GUI second.
2. GUI first, browser second.
3. Direct Python call first, API observer second.
4. Crash GUI and server while pipeline stays active, then reopen.
5. Attempt a second run from every entry point.
6. Kill the run hard, verify abandoned classification, then resume.
7. Run a three-item mock batch with one shared YAML and `--run-name` overrides.
8. Verify project DB/output content matches the pre-refactor mock baseline.
9. Simulate two Slurm job namespaces for one Unix user and verify they do not contend, while two starts within one namespace do.

## 8. Safe rollout and pull-request sequence

Recommended reviewable sequence:

1. **PR 1:** characterization tests and shared path resolver.
2. **PR 2:** runtime registry, lock tests, `--run-name`, and launch manifest.
3. **PR 3:** registry-backed server runtime/run APIs; legacy JobManager routed through the lease.
4. **PR 4:** progress/results/postprocessing consolidation.
5. **PR 5:** Electron single-instance and server ownership.
6. **PR 6:** renderer state/UI redesign and historical selector.
7. **PR 7:** Global Greening batch simplification and removal of watcher workaround.
8. **PR 8:** single-user Slurm/container reference deployment and node-local active-state support.
9. **PR 9, optional:** cooperative external stop.

Do not deploy or merge execution-path changes into the checkout while a long batch is using that checkout. Each Global Greening species launches a fresh Python interpreter, so changing source midway could make later species run different runtime semantics than earlier ones. Merge after the batch is stopped/completed, or run the batch from an immutable worktree/environment.

For one transition release:

- honor legacy environment aliases with warnings;
- preserve `/v1/run/active`;
- retain filesystem discovery for history and pre-registry visibility;
- retain legacy `/v1/jobs` only if an identified consumer needs it;
- remove `LM3_settings_gg_watch.yaml` only after follow-active integration passes.

## 9. Definition of done

The unified runtime work is complete when a CLI-started mock or real LM3 run can be followed by opening Electron, and all of the following agree without manual configuration:

- runtime run name and invocation ID;
- source/effective settings identity;
- run directory and SQLite ledger;
- active stage and progress;
- logs, results, and postprocessing target;
- Start disabled because the shared lease is occupied;
- Stop capability accurately reflects ownership;
- next-run YAML visibly remains a separate editable concern;
- closing/reopening the GUI has no effect on the run;
- a second GUI instance never exists;
- a single-user cluster job can expose the same GUI safely through an SSH tunnel without requiring a multi-user service.

No watcher YAML, client-only hidden project, filesystem recency guess, or server-private adoption record may participate in declaring the active runtime.
