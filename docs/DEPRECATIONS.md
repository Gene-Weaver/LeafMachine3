# LM3 deprecation register

Every compatibility surface LM3 still serves, the release that removes it, and why it exists at
all. A deprecation with no named owner never happens, so nothing may be marked deprecated here
without a removal release beside it.

**Current release:** the `VERSION` file at the repo root (read by `pyproject.toml` at build time and back as `leafmachine3.__version__`; bumped every commit by `tools/release/bump_version.py`). This is
the release that introduces `GET /v1/runtime`, the runtime registry and the launch handshake, so it
is the first release in which a client can migrate off the projections below.

**Transition release:** `3.1.0`. Everything in the *Scheduled for removal* table is deleted there —
one release of overlap, no more. The release name lives in code as
`leafmachine3.server.metrics_api.COMPAT_REMOVAL_RELEASE`; it is a constant rather than a comment so
that this document, the `Warning` / `X-LM3-Removed-In` headers a client actually receives, and
`tests/test_compat_cleanup.py` cannot drift apart. Changing the schedule means changing that
constant, and the test then requires this file to agree.

## LM3_RUNTIME_V2 — the runtime cutover

`LM3_RUNTIME_V2` gated the Steps 3-7 runtime wiring while it landed. **It now defaults to ON**, and
the variable survives only as an escape hatch.

| Value | Behavior |
|---|---|
| unset, or any truthy value | the runtime registry: leases, records, the launch handshake, registry-backed status. **This is the default.** |
| `0` / `false` / `no` | the pre-Step-3 path. `GET /v1/runtime` 404s, `/v1/run/active` reports `idle` while a CLI run is live, and two roots can run concurrently. |

**Scheduled for removal in `3.1.0`**, with the projections below. After that there is one runtime.

### The one hard rule before enabling it

From plan section 6, and it is an operational rule rather than something code can enforce:

> **Runtime v2 must not be enabled while a pre-registry run or a resumable batch is still executing
> from the old environment.**

A legacy LM3 process holds no lease, so a runtime-v2 run started beside it sees a free deployment
and starts — two pipelines on the same GPUs, which is the exact failure the lease exists to prevent.
Filesystem discovery may *display* such a run as unverified history; it can neither block a start
nor authorize a stop, which is precisely why the human rule is required.

Concretely, before enabling on a machine that has been running LM3:

1. Let any in-flight `machine3` run finish, or stop it.
2. Let any `run_global_greening.sh` batch reach a species boundary, or stop it between species.
3. Confirm nothing is live: `ps -eo pid,cmd | grep -E 'machine3|leafmachine3'` and
   `nvidia-smi --query-compute-apps=pid,process_name --format=csv`.
4. Only then start the server, the GUI, or the batch again.

Resuming a batch afterwards is safe and needs no special handling: the skip check reads each
species' own SQLite, so anything that did not finish is picked up. A species refused because the
deployment was busy exits **75** and the wrapper records it as `retryable(busy)` rather than a
failure.

### Pinning the old path

`LM3_RUNTIME_V2=0` restores pre-Step-3 behavior for one process. It exists for debugging and for a
site that hits a blocker mid-upgrade; it is not a supported long-term configuration, and it goes
away in `3.1.0`. If you find yourself needing it, the bug that forced you there is worth filing.

## How a deprecation reaches a client

An HTTP client never reads a docstring, so every deprecated HTTP surface answers with the headers
built by `metrics_api.deprecation_headers()`:

| Header | Value | Meaning |
|---|---|---|
| `Deprecation` | `true` | RFC 8594 / draft-ietf-httpapi-deprecation-header |
| `Link` | `</v1/runtime>; rel="successor-version"` | discover the replacement without a changelog |
| `Warning` | `299 - "... removed in LM3 3.1.0; use ..."` | human-readable, surfaced by most clients |
| `X-LM3-Removed-In` | `3.1.0` | machine-readable; the one a migration script should read |
| `X-LM3-Deprecated-Fields` | e.g. `db_path` | the ROUTE is fine, a field in its body is going |

`Sunset` is deliberately **not** sent. RFC 8594 requires an HTTP-date, and this project ships
releases, not dates; inventing one would be a promise the server cannot keep.

The headers ride on the response, never in the body. `GET /v1/run/active`'s key set is pinned
key-for-key (`tests/_contract_helpers.RUN_RECORD_SPEC`), so a `deprecated: true` field would break
the very clients the projection exists to serve.

---

## Scheduled for removal in 3.1.0

### 1. `GET /v1/run/active`

* **Replaced by** `GET /v1/runtime`.
* **Why deprecated.** It is a projection of one server's private `_Run`, and it cannot express
  `run_id`, `activity`, the bounded child tree, record classification, schema compatibility, or
  `can_stop`. Under `LM3_RUNTIME_V2` it projects the registry so an un-migrated client sees a CLI
  run instead of `idle` (plan §5), but every question a control surface actually needs answered is
  only answerable on `/v1/runtime`.
* **Why kept for one release.** It is the only run-state route an older desktop shell knows, and
  the top bar falls back to it when `/v1/runtime` answers 404.
* **Owner:** `leafmachine3/server/metrics_api.py`, route `run_active`.

### 2. The `db_path` projection

* **Replaced by** `active_db_path` (plan §2.10's storage role).
* **Where it appears.**
  * `metrics_api._Run.record()["db_path"]` — the body of `GET /v1/run/active`,
    `POST /v1/run/start` and `POST /v1/run/stop`. The two `POST` routes are **kept** by §5, so
    they announce the field with `X-LM3-Deprecated-Fields: db_path` rather than
    `Deprecation: true` alone; a client must not read "a field is going" as "stop calling this".
  * `leafmachine3/core/runtime/config_io.db_path()` — a pure projection that reads
    `active_db_path` off the object it is handed and raises a `DeprecationWarning`.
  * `progress_api.RunRef.db_path` / `as_dict()["db_path"]` and `results_api`'s run-reference
    `db_path` — both already **resolve** the §2.10 roles (`active_db_path` while live,
    `archived_db_path` after finalization) and only publish the legacy field name.
* **Why deprecated.** `db_path = run_dir/<run_name>.sqlite` is false on a staged (cluster) run,
  where the active ledger is node-local scratch and the archive is a versioned file named by a
  pointer (§2.10). One name cannot mean both.
* **Why not removed sooner.** Removing a key from a compatibility projection breaks the clients the
  projection exists for. It goes when the route goes.

### 3. The `adopted` field of the run record

* **Replaced by** `runtime.active.can_stop` (+ `can_stop_reason`) on `GET /v1/runtime`.
* Frozen at `false` since §2.5 made control authority handle ownership alone — there is no
  adoption any more. The field survives only because the record's key set is pinned; it is removed
  with the route.

### 4. `LM3_SETTINGS_PATH` → `LM3_SETTINGS`

* Honored alone with a once-per-process warning; agreeing with `LM3_SETTINGS` warns and is used;
  **disagreeing is a startup error naming both files** (`paths.LegacyEnvConflictError`) — a split
  brain that stops being fatal after the first resolution is a split brain that ships.
* **Owner:** `leafmachine3/core/paths.py` (`LEGACY_ENV_ALIASES`, `_resolve_legacy_env`).

### 5. `LM3_HARDWARE_SETTINGS` → `LM3_HARDWARE`

* Same mechanism, same schedule, same conflict rule.

### 6. Legacy profile / settings locations adopted by copy

* **Replaced by** the deployment config directory (`<user-config>/lm3/<deployment>/`), resolved by
  `paths.hardware_profile_path()` / `paths.postprocessing_settings_path()`.
* A `hardware_settings.yaml` beside the settings file, and a checkout-level
  `postprocessing_settings.yaml`, are copied once into the deployment config directory with a
  warning that says the legacy location "will stop being consulted in a future release". That
  release is `3.1.0`.
* **Owner:** `leafmachine3/core/paths.py`.

### 7. Internal: the host-hook run binding

* `progress_api._BOUND`, `_BOUND_STATE`, `bind_run()`, `clear_run()`, `bind_job_source()`,
  `_from_job_source()` and their only production caller, `metrics_api._bind_status()`.
* **Replaced by** `metrics_api.active_run_ref()` / `progress_api.active_runtime_ref()` — the
  registry read — with `invalidate_runtime_cache()` / `invalidate_run_caches()` as the cache-drop
  hook `bind_run()` used to provide.
* Not a wire surface; listed because the two halves must be deleted in **one** change. Deleting the
  callee alone leaves `_bind_status`'s bare `except` swallowing an `AttributeError`, and the status
  stream silently stops following runs.

---

## Under audit — no removal scheduled

### `POST /v1/jobs`, `GET /v1/jobs/{jid}`, `/events`, `/results`

Plan §4 Step 7: *"Removal is a decision about external users, not in-tree call sites."* This is the
audit, not a schedule. **Nothing here is deprecated yet and no code deletes these routes.**

**In-tree evidence (verified by `tests/test_compat_cleanup.py`, which re-derives it rather than
trusting this paragraph):**

| Source | Finding |
|---|---|
| `leafmachine3/server/ui/js/api.js` | `getJob`, `getJobResults`, `streamJobEvents` are **defined and called nowhere** (plan Appendix B, C2). Helper definitions are not consumers. |
| the rest of `ui/js/` | no tab constructs a `/v1/jobs` URL; the Run button uses `POST /v1/run/start`. |
| `app/main.js`, `app/preload.js` | no reference. |
| `leafmachine3/**/*.py` | no caller; the only mentions are the route definitions and comments. |
| `tests/` | no test posts to `/v1/jobs`. `tests/test_server_handshake.py` calls `app._run_job_as_subprocess()` directly, to prove the server is not a lease holder — the worker, not the route. |
| `run_global_greening.sh`, `README.md` | no reference. |
| `docs/LM3_Plan.html` | describes the routes in the ORIGINAL design. A design document is not a consumer. |

**What the routes still hold alone:** multipart **upload**. `POST /v1/run/start` runs a settings
file that already exists on the server's filesystem; there is no other way to hand the server
images from a client. That is the only reason to keep them.

**The unknowable half:** whether an operator's script calls them. A source grep cannot answer that,
so the server now records the evidence instead of guessing. The first call to each legacy route in
a process logs a warning naming the User-Agent
(`leafmachine3.server.app.note_legacy_jobs_use`) — once per route per process, because `/events` is
an SSE stream and `/{jid}` is polled, and a warning per call would bury the log it is meant to
inform. The query string is never logged: on the SSE routes it can carry `?token=`.

**Recommendation.** Ship 3.0.0 with the instrumentation. If no such warning is reported against
3.0.0, deprecate all four routes in 3.1.0 for removal in 3.2.0 — *unless* the upload capability is
to be kept, in which case re-implement `POST /v1/jobs` as a thin staging endpoint that writes the
uploads and then calls the same `start_run()` service, and retire only the parallel queue and job
state. Do not delete them in 3.1.0: that would be a removal one release after the first
announcement anybody could have seen, which is exactly what the one-transition-release rule
forbids.

### `LM3_settings_gg_watch.yaml`

**Retained deliberately. Not deprecated. No migration code deletes it, now or in 3.1.0.**

It is a manual operational artifact: a settings file with an empty `project.run_name`, which is how
an operator forces `progress_api._from_settings()` to decline so the GUI follows whatever run is
actually going. Plan §4 Step 7 keeps it "until follow-active is validated", and follow-active
(`resolve_run()` tier 2, the active runtime record) has shipped but has not yet been validated in
anger against a real Global Greening batch. It is referenced by no source file — it is a file a
human points the app at — so there is nothing to remove from code, and a migration that deleted a
user's YAML would be destroying an operator's workaround before proving the replacement.

Revisit after one full Global Greening batch has been run against `GET /v1/runtime` follow-active.

---

## Retained, not deprecated — read this before "cleaning them up"

| Surface | Status |
|---|---|
| `POST /v1/run/start`, `POST /v1/run/stop` | **Kept** (§5). Handshake-backed; 409 on busy; strict `can_stop`. Only the `db_path` field in their response body is deprecated. |
| `pid` on `/healthz` and in the run record | **Kept, diagnostic only.** §2.5: a PID never authorizes control. `/v1/runtime` says so in the payload (`server.pid_is_diagnostic_only`). It is not deprecated — it is demoted. |
| `metrics_api._state_path()` | Kept as a read-and-**unlink** helper. Nothing writes `active_run.json` any more; the helper exists so a file left by an older build is found and deleted rather than read. |

## Already removed

| Surface | Removed in | Note |
|---|---|---|
| `metrics_api._adopt()`, `_pid_alive()`, `_Run.create_time`, `_Run.persist()` | 3.0.0 | §2.5: control authority is handle ownership alone. |
| `active_run.json` (the server's PID re-adoption file) | 3.0.0 | Purged on first request if an older build left one. |
| The duplicate `GET /v1/runs` in `progress_api` | 3.0.0 | `results_api` owns the one route (§5). `?limit=` moved to `GET /v1/runs/-/selector`. |

---

## Naming rules this register enforces (§5)

* **One project-reference shape.** `run_id`, `run_name`, `run_dir`, `active_db_path`, `activity`,
  `state` — shared by runtime, progress, results and postprocessing. New surfaces use the §2.10
  storage-role names, never `db_path`. `metrics_api.active_run_ref()` is the canonical spelling and
  is pinned as such.
  * Not yet converged: `progress_api.RunRef.as_dict()` and `results_api.run_ref()` both publish
    `db_path` for the resolved role (correct value, retired name), and `results_api`'s reference
    carries `id` and `live` that the others do not. They are read-only projections, so the
    divergence is cosmetic today — but a fourth surface must copy `active_run_ref()`, not either of
    them. Converge on the next change to those two files, and drop the `db_path` key there with
    the 3.1.0 removal.
* **`active` means the process is live. It never means "the run selected in the UI."** Those are
  `runtime.active` and `view.selected`. A payload may carry both, but not one field doing both
  jobs: `GET /v1/runs/-/selector` publishes `active` (live only) beside `selected_default`, and the
  renderer keeps `state.selected` separate from the runtime record.
  * Known exception, not yet fixed: `progress_api.list_runs()` sets `entry["active"]` per row to
    mean "this is the run being followed". That envelope is no longer reachable over HTTP (its
    route was removed in 3.0.0) and the function survives only as the Status tab's idle summary.
    Rename it `followed` when that function is next touched.
