# Adversarial review — Unified LM3 runtime implementation plan

Reviewed: `UNIFIED_RUNTIME_IMPLEMENTATION_PLAN.md` (dated 2026-08-28, 627 lines)
Review date: 2026-08-28
Method: every claim the plan makes about existing code was checked against the tree. Findings
carry `file:line` evidence. Nothing below is inferred from the plan's own prose.

## Verdict

The plan's diagnosis is sound and most of its claims about the current code check out. The
architecture — an OS-held lock for exclusivity, JSON as description rather than authority, and a
strict separation between "running now" and "next run settings" — is the right shape, and the
refusal to authorize PID signaling from JSON is correct.

It is **not ready to implement as written**. Four issues will break working behavior on contact,
and none of them is mentioned anywhere in the document. The most serious is that the lease, placed
exactly where the plan says to place it, deadlocks VRAM calibration against itself and does so
*silently*.

Counts: 4 blocking, 6 significant gaps, 6 factual corrections, 2 internal inconsistencies.

---

## 1. Blocking

### B1. The lease deadlocks VRAM calibration, and the failure is silent

Invariant 1 makes "hardware-tuning activity" a lease holder. §3.3 places acquisition before
"hardware profiling". But hardware profiling *is* a nested LM3 pipeline run:

- `machine3()` calls `ensure_hardware_profile(cfg)` at `leafmachine3/machine3.py:61` — after the
  plan's acquisition point.
- `ensure_hardware_profile` (`leafmachine3/setup/hardware_setup.py:92`) calls `run_setup`, which
  reaches `calibrate_gpu_stages` at `leafmachine3/setup/hardware_setup.py:311`.
- `calibrate_gpu_stages` runs a real one-worker LM3 pass **in a subprocess**:
  `cmd = [sys.executable, "-m", "leafmachine3.machine3", ...]` at
  `leafmachine3/setup/calibrate.py:153-159`.

So the tuning parent holds the lease and the pipeline child is refused. The child exits nonzero
and `calibrate.py:163-166` treats that as:

```python
log.warning("calibration run exited %d -- keeping heuristic estimates\n%s", ...)
return False
```

A **warning**, not an error. Calibration therefore degrades silently back to the heuristic
per-worker VRAM estimates — the ones this repo measured at 3.3x–9.2x too high, which is what
capped worker counts and cost ~22% wall clock before calibration existed. A user clicking
"Calibrate VRAM" in the Machine panel would see it "succeed" and get worse sizing than before.

Severity is bounded but real: the auto-trigger path (`ensure_hardware_profile` →
`run_setup(cfg, optimize=True)`) leaves `calibrate=False` (`hardware_setup.py:127-130`), so only
explicit `--calibrate` and the GUI button hit it. That is still the entire calibration feature.

**Required:** a lease-inheritance concept. The record already carries
`control.owner_instance_id`; add the analogous "this child runs under an already-held lease"
token (env-passed, validated against the live lock holder) and have `calibrate._run_pipeline`
propagate it. Alternatively exempt `activity: "calibration"` children explicitly. Either way the
plan must name `calibrate.py`, and Phase 2's exit gate — which celebrates "a helper subprocess
holding a lease makes CLI, direct Python, server start ... all refuse a second run" — must
exclude this topology, because that is precisely the calibration topology.

### B2. Concurrent multi-GPU runs are removed without acknowledgment

The plan treats one-run-per-deployment as self-evidently correct. Today, concurrent LM3 runs are
a supported, tested scenario:

- `Config.resolve_gpus()` (`leafmachine3/core/config.py:402-417`) accepts an explicit ordinal
  list, so pinning run A to GPU 0 and run B to GPU 1 is a first-class config.
- `tests/test_executor.py:367 test_reaper_never_touches_a_concurrent_lm3_runs_workers` exists
  specifically to pin that the orphan reaper is safe under concurrent runs. That test's premise
  becomes unreachable under the plan.

On a two-card workstation this is a capability regression, and the plan offers no desktop escape
hatch — `LM3_DEPLOYMENT_ID` is introduced for cluster namespacing only (§3.1), never as a
deliberate desktop override.

**Required:** state the removal explicitly and decide. Either document `LM3_DEPLOYMENT_ID` as the
supported way to run two independent deployments on one host, or scope the lease per-GPU-set
rather than per-deployment. Then say what happens to the concurrency test.

### B3. The Global Greening batch loses the lease between species, and one lost race fails the rest

§4 Phase 7 states, approvingly: "The sequential batch naturally releases and reacquires the
runtime lease between species." Combined with Phase 6's "Start is disabled whenever any runtime
lease is active", that means **Start is enabled during every inter-species gap**.

If a user starts a GUI run in that window:

1. The GUI run takes the lease.
2. The next species gets `RuntimeBusyError` and the "distinct nonzero code" of §3.3.
3. `run_global_greening.sh` treats any nonzero exit as a per-species failure, records it, and
   **continues to the next species** — the loop's `else` branch is deliberately "keep going: one
   bad species should not cost the other twenty."
4. Every remaining species fails the same way, in seconds.

A 13-species multi-day batch is destroyed by one click. The plan presents the gap as benign and
never mentions the interaction.

**Required:** one of — hold a single batch-scoped lease across the whole batch; or make the busy
exit code distinguishable and teach the script to wait and retry rather than count it as a
species failure; or have the GUI refuse Start while a batch-scoped record is present. Whichever
is chosen needs a test (see §5).

### B4. The test suite would acquire a real, machine-wide lease

`tests/test_pipeline_mock.py` calls `machine3()` directly at lines 46, 335, 346, 359 and 360.
Once the lease wraps the public `machine3()` function — which §4 Phase 2 explicitly requires, "not
just `main()`" — running `pytest` acquires the user's real deployment lease five-plus times.

Consequences: the suite fails whenever a real run is active, and worse, running the suite
**blocks a production run from starting**. The plan's test matrix (§7) mentions `LM3_RUNTIME_DIR`
only in the cluster context and never mandates test isolation.

**Required:** `tests/conftest.py` must set `LM3_RUNTIME_DIR` and `LM3_DEPLOYMENT_ID` to a
per-session `tmp_path` before any test imports `machine3`, and the plan should say so as a Phase 2
deliverable rather than leaving it to be discovered.

---

## 2. Significant gaps

### G1. Postprocessing and the standalone tools sit outside the lease
Invariant 1 covers "pipeline or hardware-tuning activity" only. `postprocess_api.py` and the
standalone post-run tools driven by `postprocessing_settings.yaml` can use the GPU and write into
a run directory. Nothing stops one from running against a run that a live pipeline is writing.
Decide explicitly whether they are lease holders, lease-exempt readers, or blocked while a lease
is held.

### G2. Follow-active will display the calibration run as if it were the user's run
`calibrate.py:142-144` sets `project.run_name = CALIBRATION_RUN_NAME` and
`run_mode.overwrite = True`. Under the plan that child publishes an active record. Phase 6's
follow-active UI has no filter on activity type, so the GUI would switch to the calibration
project mid-tuning. The record schema already has an `activity` field (§3.2) — no phase uses it.
Specify that the UI follows `activity: "pipeline"` and reports other activities as a distinct
"tuning in progress" state.

### G3. `tmp_dir` cannot be published at lease time — say so, or the resolver will drift
Phase 2 wants "a pure project-path resolver so expected run/DB paths can be published before
`build_dirs()` writes anything". Most of that is safe: `build_dirs` derives
`root = out / run_name` (`leafmachine3/core/dirs.py:42-44`) and `db = root/f"{run}.sqlite"`
(`dirs.py:63`) with no uniquification. But `tmp` resolves from the bound `hardware_settings.tmp_dir`
when `tmp_dir: auto` (`dirs.py:50-54`), and the profile is bound by `ensure_hardware_profile()`
*after* the acquisition point. Bound the resolver's contract to `run_dir` / `db_path` / `log_path`
explicitly. Left unstated, someone will add tmp to the resolver and reintroduce the exact
two-sources-of-truth bug this plan exists to eliminate.

### G4. No reader rule for a higher `schema_version`
§3.2 versions the record "from the first release" but §6 never says what a reader does with a
record written by a newer LM3. On the cluster profile — a container image and a host virtualenv
sharing a runtime directory — skew is likely. Define it: an unknown-higher version is treated as
opaque-but-live, exclusivity still comes from the lock, control is refused, and the UI says the
record came from a newer LM3.

### G5. Electron's single-instance lock is keyed on app name, not deployment
`app/package.json` declares `"name": "lm3-desktop"` with no `appId`/`build` section, so
`requestSingleInstanceLock()` keys off a userData directory derived from that name. Two LM3
checkouts on one machine share the lock, and invariant 9 would make the second checkout's launch
focus the *first* checkout's window. That directly collides with §8's own advice to "run the batch
from an immutable worktree/environment". Key the lock on the deployment (port / runtime dir /
`LM3_DEPLOYMENT_ID`), and scope the Definition-of-Done item "a second GUI instance never exists"
to one deployment.

### G6. §3.1 and §8 give opposite advice about multiple checkouts
§3.1 deliberately keeps the runtime directory "independent of the checkout and current working
directory so separate LM3 installations still coordinate". §8 recommends running a long batch
"from an immutable worktree/environment" so that merging does not change semantics mid-batch.
Under §3.1 that worktree and the development checkout contend on one lease, so following §8's
advice means the developer cannot run anything locally for the batch's duration. Pick one and
say which.

---

## 3. Factual corrections

### F1. The PID-reuse guard already exists; the real defect is narrower
Invariant 10 and §6's "PID reuse" bullet read as though nothing guards this today. In fact
`_pid_alive(pid, create_time)` (`leafmachine3/server/metrics_api.py:440-450`) rejects a reused PID
by comparing process creation time within 1.0 s, and `_adopt()` calls it before adopting
(`metrics_api.py:478`).

The actual unguarded step is different and worth stating precisely, because it changes what has to
be fixed: adoption derives a **process group** from that PID —
`run.pgid = os.getpgid(pid)` (`metrics_api.py:494`) — and `_signal_group` later `killpg`s it
(`metrics_api.py:804`). Group membership can change after the creation-time check passes, and the
guard says nothing about *authority*: adoption grants stop control over a process this server
never spawned. Fixing "PID reuse" is not the point; fixing "control without provenance" is.

### F2. The settings-path split is 4-way and the fallbacks differ, not just the variable names
Phase 1 frames this as `LM3_SETTINGS` vs `LM3_SETTINGS_PATH`. Verified reality:

| Module | Variable | Fallback when unset |
|---|---|---|
| `server/settings_api.py:41,110` | `LM3_SETTINGS_PATH` | `./LM3_settings.yaml` (CWD) |
| `server/metrics_api.py:139` | `LM3_SETTINGS` | `_search_roots()` scan (`metrics_api.py:127-130`) |
| `server/postprocess_api.py:234` | `LM3_SETTINGS` | `LM3_settings.yaml` (CWD) |
| `server/progress_api.py:335` | `LM3_SETTINGS` | `LM3_settings.yaml` (CWD) |
| `server/metrics_api.py:135` | `LM3_HARDWARE_SETTINGS` | `_search_roots()` scan |
| `server/progress_api.py:345` | `LM3_HARDWARE` | `hardware_settings.yaml` (CWD) |

Two of these also disagree on *shape* of failure: `_find_file` returns `None` when the override is
not a file (`metrics_api.py:124-126`) rather than falling back. Unifying the variable names
without unifying the fallbacks leaves the same split-brain reachable from a different CWD. Phase 1's
exit gate ("with any supported CWD, every subsystem reports the same canonical settings path")
is the right gate; the action list under it should name the fallback unification too.

### F3. `LM3_settings_gg_watch.yaml` is referenced by no source file
§8 lists "remove `LM3_settings_gg_watch.yaml` only after follow-active integration passes" as a
migration constraint. Nothing under `leafmachine3/`, `app/`, `tests/`, or the shell scripts
references it. It is a hand-used config file whose removal is gated by nothing but habit. Treating
it as a coupled migration step overstates the work.

### F4. `/v1/jobs` has in-tree UI consumers, so "no real external consumer" is the wrong test
Phase 0 says to "inventory external consumers"; Phase 3 says deprecate "if there is no real
external consumer". The shipped UI already defines `getJob`, `getJobResults`, and a job-events SSE
subscription (`leafmachine3/server/ui/js/api.js:360-367`), and the server serves
`POST /v1/jobs`, `GET /v1/jobs/{jid}`, `/events`, `/results`
(`leafmachine3/server/app.py:478-525`). `api.js:348` documents the split itself. The deletion gate
should be "the upload/Files path in the shipped UI no longer needs it", which is an in-tree check,
not an external-consumer survey.

### F5. `/healthz` returns `pid` *for the express purpose* the plan forbids
`leafmachine3/server/app.py:532-538` returns `pid`, with the comment: "`pid` is what lets a client
that ATTACHED to an already-running server still stop it: with no process handle of its own, that
pid is its only escalation path". Phase 5 adds fields to `/healthz` but never says to remove or
re-purpose `pid`, while simultaneously outlawing the behavior it exists for. State explicitly that
`pid` becomes verification-only (matched against a retained child handle) and that attached
clients deliberately lose their escalation path — that is a behavior removal someone will
otherwise "fix" back.

### F6. Electron's token reuse only works if the user exported it
Phase 5's concern about attached-server authentication is correct and worth keeping. Evidence:
`app/main.js:25` is `const TOKEN = process.env.LM3_SERVER_TOKEN || crypto.randomBytes(16)...`,
and `main.js:277` loads `?token=${TOKEN}`. The comment claims reuse "keeps attaching to an
already-running server working", but that only holds when the user has exported
`LM3_SERVER_TOKEN` into both processes. Otherwise Electron attaches with a freshly minted,
mismatched token — exactly the case the plan wants to handle.

---

## 4. Internal inconsistencies

### I1. Invariant 5 is contradicted by Phase 4
Invariant 5: "Status, logs, results, and postprocessing follow the active record's DB/run path
while a run is active" — presented as an acceptance requirement, not an aspiration. Phase 4's own
precedence list puts "explicit `db=` or explicit historical `run=` selector" *above* the active
record, and Phase 6 says "Selecting history pauses follow mode". Restate invariant 5 as holding
"in follow-active mode, which is the default", or the acceptance test contradicts the design.

### I2. Nothing states that the server is not a lease holder
§2.1 requires `lm3-serve` and the CLI pipeline to run in the same allocation, while invariant 1
allows one lease per deployment. It is inferable from §3.3 (the lease wraps `machine3()`) that the
server never takes one, but for a document whose whole purpose is removing ambiguity about who
owns what, say it in one line.

---

## 5. Test-matrix gaps

§7 is thorough on the lock mechanics and thin on the interactions above. Missing:

- calibration under a held lease — that the nested `machine3` child succeeds, and that a
  calibration failure is loud rather than a warning (B1);
- `conftest` runtime isolation — a test proving the suite never touches the default runtime
  directory (B4);
- the batch inter-species race, and whichever fix is chosen for it (B3);
- a shell-level assertion that the busy exit code is distinguishable from an ordinary pipeline
  failure — B3's fix depends on it, and §3.3 only promises "a distinct nonzero code";
- concurrent runs on disjoint GPU sets: either they still work, or the removal is pinned (B2);
- a record written with a higher `schema_version` (G4);
- two Electron launches from two checkouts (G5);
- postprocessing started against a run with a live lease (G1).

---

## 6. Sequencing

### S1. PR2 and PR3 leave a window where server run-state and the lease disagree
PR2 wraps `machine3()`; PR3 makes the server registry-backed. Between them, `POST /v1/run/start`
spawns a child that can now die with `RuntimeBusyError`, while the server's `_RUN` / `_ADOPT_TRIED`
private state (`metrics_api.py:427-429`) still believes it owns run truth. PR2 must at least
surface the busy exit code in the start response, or PR2 and PR3 should land together.

### S2. The plan's own merge warning is currently live
§8's "do not merge execution-path changes into the checkout while a long batch is using that
checkout" is correct and applies right now: the Global Greening batch is stopped partway through
`cornus_sericea` and will resume from this checkout.

### S3. Phase 7's replacement command drops behavior the script has gained
`run_global_greening.sh` was reworked on 2026-08-28 to skip species whose pipeline already ran to
completion, reading `project_status` from each species' SQLite. Phase 7's proposed invocation
shows only `--config/--run-name/--input/--output`. The `--run-name` change is compatible with the
resume probe (it reads the DB, not `_configs/`), so retiring `_configs/<species>.yaml` is safe —
but a verbatim replacement of the script body would silently drop the skip/resume logic. Phase 7
should say it is changing only how the run name is passed.

---

## 7. Smaller notes

- **§3.1 fallback 5** ("a documented per-user cache fallback") can land on an NFS-mounted home in
  a lab setup, where `flock` semantics are unreliable. The plan already forbids shared home for
  the cluster case; extend the caution — or a filesystem-type check — to the desktop fallback.
- **§3.2** shows `error: null` / `finished_at: null` / `returncode: null` in an active record.
  State that readers must tolerate absent keys, not just null ones; it is the cheap half of G4.
- **Phase 5** passes `LM3_INSTANCE_ID` by environment. That is readable via `/proc/<pid>/environ`
  for the same user, which is fine — it is a nonce for identity matching, not a secret — but it is
  worth one sentence so nobody later mistakes it for one and moves it somewhere "safer".
- **§5** keeps `/v1/run/active` "temporarily" and §8 preserves it "for one transition release",
  but no phase owns its removal and §9 does not mention it. Assign the deletion to a phase.
- The plan is well served by its §6 failure list; the cases it does cover are the right ones and
  are covered precisely. The gaps above are all *interaction* cases rather than mechanism cases,
  which is a reasonable blind spot for a design document and a bad one for an implementation plan.

---

## 8. Recommended changes before implementation

1. Resolve B1 with an explicit lease-inheritance design; make calibration failure loud.
2. Decide B2 in writing, with a documented desktop override, and say what happens to
   `test_reaper_never_touches_a_concurrent_lm3_runs_workers`.
3. Resolve B3 by choosing a batch-lease strategy, and add the shell-visible busy exit code.
4. Add the `conftest` isolation requirement to Phase 2 (B4).
5. Add the missing tests in §5 above to §7's matrix.
6. Fold the F1–F6 corrections into the phases that depend on them — F2 in particular changes the
   size of Phase 1.
7. Restate invariant 5 (I1) so the acceptance criteria are self-consistent.
