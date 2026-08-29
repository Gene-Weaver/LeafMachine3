# LM3 GUI / CLI control investigation

Date: 2026-08-26  
Scope: read-only investigation; no application code or settings were changed, and no process-control requests or signals were sent.

## Executive summary

The odd behavior is architectural, not a single display bug. There are currently several independent notions of "the active project":

1. the YAML file shown and edited by the Settings UI;
2. the YAML file used by GUI-launched runs;
3. the run record persisted only for processes launched by the GUI server;
4. the run inferred by the status subsystem from a configured path or a filesystem scan;
5. a client-only `newRun` state that can intentionally hide server status.

These sources can disagree silently. A CLI run is not registered in the GUI run manager, so `/v1/run/active` normally reports idle even while LM3 is running. The status subsystem may nevertheless find a live SQLite ledger, meaning the progress display and the Start/Stop controls can disagree about whether a run exists. For the Global Greening batch, the batch script changes run identity on every species by creating a per-species YAML, while the server normally reads the fixed checkout `LM3_settings.yaml`; this makes correct automatic following accidental and configuration-dependent.

There is also no Electron single-instance lock. Multiple desktop processes can attach to the same server. More seriously, the default Electron quit path deliberately terminates a server it merely attached to. Although the server shutdown endpoint says a separately launched LM3 job survives server shutdown, the renderer's Close and New run actions can call `/v1/run/stop` when they believe a managed run is active. Ownership and observation are therefore mixed together.

## Current live batch observation

The active Global Greening project at inspection time was `chamaenerion_angustifolium`:

- project DB: `/datab/Global_Greening/GBIF/LM3/chamaenerion_angustifolium/chamaenerion_angustifolium.sqlite`
- active stage: `plant_detector`
- ledger progress observed: 9,746 / 11,994
- DB mtime observed: 2026-08-26 11:29:54 EDT
- WAL mtime observed: 2026-08-26 11:29:56 EDT

The advancing WAL establishes that the run was live without signaling or attaching a debugger to it. Host process enumeration was unavailable inside the investigation sandbox's PID namespace, so no claim is made about the host PID tree.

## Findings

### Critical: GUI-launched-run state is not global LM3 runtime state

`metrics_api.py` owns an in-memory `_RUN` and persists it to `runs/_server_jobs/active_run.json`. That record is created only in `start_run()`, the GUI server's launcher. `_adopt()` can only re-adopt a process for which that file was previously written. It does not discover arbitrary `python -m leafmachine3` processes.

Consequences:

- A CLI or shell-script run can be using the GPUs while `/v1/run/active` says `idle`.
- The GUI Start button is gated by that run record, so it can remain enabled during an unrelated CLI run.
- The status subsystem may independently discover the CLI ledger and show activity, producing a contradictory UI: live stage data alongside idle/enabled run controls.
- Stop semantics apply only to the run manager's `_RUN`; there is no principled global arbitration preventing two LM3 runs.

The Global Greening shell script is exactly this external-launch case. It invokes `python -m leafmachine3` directly for each species and never informs the server run manager.

### Critical: settings resolution is split between two environment variables

The Settings API resolves `explicit argument > LM3_SETTINGS_PATH > ./LM3_settings.yaml`. The run launcher and progress subsystem resolve `LM3_SETTINGS`, and postprocessing does the same. Therefore the UI can edit one YAML while Start, status, and postprocessing use another.

Current relevant defaults when the Electron server starts from the checkout root:

- Settings tab: `/datac/Labelbox_Dump/LM3/LM3_settings.yaml`
- GUI Start default: `/datac/Labelbox_Dump/LM3/LM3_settings.yaml`
- status configured-project default: `/datac/Labelbox_Dump/LM3/LM3_settings.yaml`
- active batch template: `/datac/Labelbox_Dump/LM3/LM3_settings_global_greening.yaml`
- active species config: `/datab/Global_Greening/GBIF/LM3/_configs/chamaenerion_angustifolium.yaml`

They coincide only when both environment variables are unset and the server CWD is the checkout. They do not coincide with the batch's per-species config.

### High: configured project outranks live-run discovery

`progress_api.resolve_run()` prioritizes a bound run, then the project derived from the server's settings, and only then filesystem discovery. `_from_settings()` returns `<output.dir>/<run_name>` even if it does not exist. This deliberately prevents discovery from wandering to an unrelated run, but it also prevents a GUI opened during a CLI batch from following the actually live project whenever the server's settings contain any nonempty run name.

At inspection time `LM3_settings.yaml` points to `liquidambar_examples6`, while the live CLI batch project is `chamaenerion_angustifolium`. With default server environment, status resolution is consequently pinned to the Liquidambar path rather than the live batch.

`LM3_settings_gg_watch.yaml` works around this by making `project.run_name` empty, which forces `_from_settings()` to return `None` and allows discovery. That is a second control scheme, and it demonstrates the underlying ambiguity rather than resolving it.

### High: the renderer adds another independent active-project state

On every GUI open, `openFresh()` asks `/v1/run/active`. Because a CLI run is invisible to that endpoint, it calls `resetForNewRun()`. That function:

- sets client-only `state.newRun = true`;
- clears the client run/snapshot;
- blanks the visible project-name field without writing the YAML;
- ignores status frames until a frame looks both running and non-stale.

This is the direct source of much of the "active project" oddness. The visible project field, run chip, YAML value, `/v1/run/active`, and status stream can all name different things. The code comments explicitly describe this as a sticky client-side reset intended to suppress server discovery.

### High: Electron is not single-instance

There is no call to Electron's `app.requestSingleInstanceLock()` and no `second-instance` handler. Each launch can create another window/process, and all instances target the same fixed host/port by default. Optimistic YAML mtime checks reduce simultaneous-save damage but do not establish one authoritative controller.

### High: attached-server ownership is unsafe

`ensureServer()` treats any 200 response from `/healthz` on the configured port as the LM3 server and records the returned PID. It does not verify an instance nonce or executable identity. By default, quitting Electron calls `/v1/shutdown` and escalates through SIGTERM and SIGKILL against that PID even if Electron did not launch it. `LM3_KEEP_SERVER=1` is an opt-out, not safe ownership detection.

The server's `/v1/shutdown` endpoint intentionally does not stop a separately launched LM3 subprocess. That reduces damage for GUI-managed runs that outlive a server restart, but it does not justify Electron taking ownership of any attached server. It can still terminate a manually started server/debug session and disrupt all browser clients.

### High: UI Close/New run combine observation with destructive control

The renderer's Close action checks the status snapshot, calls `/v1/run/stop` when it appears running, waits, then asks Electron to quit. New run can also stop the active managed job. This makes a monitoring UI destructive by default. A GUI opened merely to observe a headless run should not acquire stop/kill authority implicitly.

The present split can behave inconsistently in either direction:

- CLI run: status may show running but run manager is idle; Close may or may not decide it is running based on the snapshot.
- adopted GUI run: run manager exposes a PID/process group and Stop can terminate it.
- manually started server: Electron quit terminates the server even though it did not create it.

### Medium: batch identity is encoded by generated YAML files

The CLI offers `--input` and `--output` overrides but no `--run-name`. `run_global_greening.sh` therefore uses `sed` to create one full YAML per species solely to change `project.run_name`. This is defensible provenance, but it means the GUI cannot follow the batch by watching one settings file: the effective config path changes per species and the shared template itself remains on an older species name.

### Medium: settings writes are mostly well-defended, but the control surface is duplicated

The Settings API validates before writing, uses atomic replacement, creates backups, and supports an `if_mtime` conflict check. Those are good safeguards. The issue is not primarily write integrity; it is that top-bar fields, the full Settings tab, presets, CLI overrides, generated per-species YAMLs, environment-selected YAML paths, status binding, filesystem discovery, and client-only reset state all participate in selecting or describing a run.

## Desired invariant

The clean target is:

> One machine-wide LM3 runtime registry is authoritative for execution; one canonical settings path is authoritative for editable intent; every GUI/browser/CLI client observes the same runtime registry; control authority is explicit and distinct from observation.

Settings should describe what the next run will do. Runtime state should describe what is running now. The current project should never be inferred independently from mutable settings by several modules once a run exists.

## Recommended design direction (no implementation performed)

1. **Add Electron single-instance enforcement.** Acquire `app.requestSingleInstanceLock()` before server/window creation; a second launch should focus the existing window and exit.
2. **Separate server ownership from server attachment.** Only stop a server whose unique instance ID and owner relationship prove Electron spawned it. Closing an observer window must not stop an attached/manual server or LM3 run.
3. **Create a machine-wide runtime registry.** Every LM3 entry point (CLI, Python API, GUI launcher, batch script) should acquire/register the same machine-level run lease and publish PID, process-group/session identity, create time, config path, run directory, DB path, start time, and owner/control capability. The registry must validate liveness and PID reuse. This is the source for `/v1/run/active`.
4. **Make CLI participate automatically.** Registration belongs inside the shared Python execution path, not in individual wrappers, so `python -m leafmachine3`, console scripts, direct Python calls, and GUI launches all appear identically.
5. **Use one settings environment variable and one resolver.** Replace the `LM3_SETTINGS_PATH`/`LM3_SETTINGS` split with a single canonical resolved path shared by Settings, run control, status, and postprocessing. Return/log it in health/runtime metadata and warn on legacy conflicts during migration.
6. **Pass project identity explicitly once running.** The runtime record's DB/run directory should drive status, console, results, and postprocessing. Do not re-derive an active run from the editable settings file.
7. **Make observation and control separate capabilities.** A GUI attaching to a CLI run should show it immediately. Stop should be available only under a clearly defined policy; Close should close the GUI, not implicitly stop the run.
8. **Remove the client-only active-project fiction.** `newRun` may remain as an editor workflow, but it must not suppress authoritative runtime state. Present "Running now" and "Next-run settings" as separate concepts when they differ.
9. **Add `--run-name` (or a general explicit project reference).** The batch can then use one settings template plus explicit input/output/name overrides, while optionally archiving the fully resolved effective config as provenance.
10. **Keep a run selector for historical inspection.** The backend already accepts explicit run/DB selection in status routes. Wire that through the UI, while reserving an unambiguous "Follow active run" mode for the runtime registry.

## Suggested implementation order and tests

1. Canonical settings resolver and conflict diagnostics.
2. Shared runtime registration/lease in the Python execution core, with CLI-first tests.
3. Status/run API consolidation around that registry.
4. Electron single-instance and safe ownership rules.
5. UI separation of active runtime from next-run settings; remove sticky suppression.
6. Batch simplification and explicit run naming.

Essential integration scenarios:

- Start through CLI, then open GUI: GUI immediately shows exact config, run name, PID, DB, stage, and progress.
- Start through direct Python call, then open GUI: same result.
- Start through GUI, then open browser UI: both show identical runtime state.
- Launch Electron twice: second invocation focuses the first; no second controller/server.
- Attach Electron to a manually started server, then close Electron: server and run remain alive.
- Edit next-run YAML during an active run: active display remains pinned to its immutable launch record and clearly labels settings as applying to the next run.
- Attempt a second LM3 run from any entry point: the common lease refuses it with the identity of the existing run.
- Crash/restart the server or GUI: the live CLI/LM3 run is rediscovered from the registry without filesystem guessing.
- Stale registry/PID reuse: rejected using process create time and verified command/runtime identity.

## Existing project notes

The repository already records much of the same diagnosis in TODO item 25 and portions of `electron_app_plan.md`. Those notes are directionally correct. The key addition from this investigation is that the renderer's `openFresh()`/`state.newRun` behavior and Electron's attached-server shutdown policy materially compound the settings/run-resolution split and should be treated as first-class design problems, not UI polish.
