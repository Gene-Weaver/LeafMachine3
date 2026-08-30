# Unified runtime — Step 1/2/5a closeout

Author: closeout pass, 2026-08-29
Baseline commit: `5ddfee4` ("Snapshot before the unified-runtime refactor")
Plan: `UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`, revision 13, 2269 lines, 60 gates in §8
Evidence: `.step2_evidence/{raw_findings.json,reconciliation.md,baseline_5ddfee4.txt,user_config_before.txt}`

**Working-tree snapshot for every code claim below: 2026-08-29 10:48–10:51 local.**
This checkout is being modified by another process while this document was written. Files changed
under me mid-pass — `leafmachine3/core/{config,dirs,paths}.py` and `leafmachine3/machine3.py` at
10:48:01–10:48:17, `tests/test_relative_path_contract.py` created at 10:50:26. Every "fix status"
cell is therefore a reading at that instant, not a durable claim. The Phase 8 verification pass
re-measures.

This report merges the three final gate agents' finding sets (`raw_findings.json` → `gate_sets`:
5 regression + 23 conformance + 9 completeness = **37 findings**, assigned `F-001`…`F-037` in that
order) and applies the seven verified corrections in `.step2_evidence/reconciliation.md`. The
earlier 12 auditor sets (`all_sets`, including the 40-finding merged set) are used for
cross-reference and concurrence only.

---

## 1. Environment facts (reconciliation R6, R7)

> **Sanitized.** Machine keys, deployment-key hashes, usernames and absolute local paths in this
> document are templated (`<machine-key-A>`, `<user-config>`, `<home>`, …). They are derived host
> identity, not findings, and nothing here depends on their literal values. Raw, unredacted
> verification material was kept out of version control — see §12.


| Fact | Measurement | Source |
|---|---|---|
| No LM3 run was live at any point during this closeout | `ps -eo pid,ppid,etime,stat,cmd \| grep -iE 'global_greening\|machine3\|leafmachine3'` → no match; `nvidia-smi --query-compute-apps` → no compute apps | R6 |
| Nothing here stopped, signaled, adopted, or resumed any run | no process control was exercised by this pass | R6 |
| Two hardware profiles in the real user config are **byte-identical** | md5 `0a690a86bdbd6386f839146795c2d550` for both; both 3605 bytes | R7, `user_config_before.txt` |
| They differ only in the machine-key component of the **filename** | `hardware_settings.<machine-key-A>.yaml` vs `hardware_settings.<machine-key-B>.yaml`, because the machine-key derivation in `paths.py` changed between the two writes | R7 |
| This is **not** a GPU-visibility difference | both carry an empty `gpus:` list | R7 |
| Both are left in place, untouched, **deliberately** | outside the repo, outside any agent's ownership; removal is the user's decision | R7 |

`~/.config/lm3` was neither read-modified nor deleted by this pass. No test suite was run by this
pass.

---

## 2. Platform qualification

This section is normative and now also lives in the plan at §1.1 (`plan:233-262`, added in
revision 13).

| Platform | Status | Validated on |
|---|---|---|
| **Linux desktop** | **qualification target now** — §8 gates must actually pass here before Step 3 proceeds | this host, Linux CI |
| **Linux cluster** | **qualification target now** | Slurm allocation, Linux CI |
| **Windows** | **implemented but not yet natively validated** | a real Windows machine, later |
| **macOS** | **implemented but not yet natively validated** — packaging, signing, notarization, permissions, and execution, all of it | a real Mac, later |

Three rules, non-negotiable:

1. **Deferred native testing does not block Linux Step 3.** Windows and macOS qualification is a
   later release gate, not a prerequisite for the Linux path. No finding in this report that is
   Windows- or macOS-only is marked as blocking Step 3.
2. **Windows and macOS are "implemented but not yet natively validated", never "passed".** The 29
   Windows adapter tests execute on Linux against the injectable `FakeWin32` seam. That is a test of
   our *model* of the Windows object manager, not of the object manager. Gates 27–34 are not closed.
3. **An unexecuted CI workflow is not evidence.** `.github/workflows/ci.yml` now defines a matrixed
   `platform-primitives` job for `windows-latest` and `macos-latest`. `git remote -v` in this
   checkout returns **nothing**, so that job has never executed and cannot execute from here. The
   file is retained deliberately — it runs the moment the repository has a remote — but it must
   never be cited as though it had run. The job's own comment says so (`ci.yml:40-43`).

Corollary for this report: eight findings are marked `unverified` **solely** because settling them
requires an execution this host cannot perform — a real Windows kernel, a real Mac, an executed CI
run, a built wheel, or a JavaScript implementation that does not yet exist. They are listed in §7.

---

## 3. Reconciliation R1–R7, applied

Each correction below overrides any auditor wording that contradicts it. The right-hand column names
the findings whose text this pass rewrote.

### R1 — CI was **not** absent at baseline

```
git ls-tree -r 5ddfee4 --name-only | grep github   ->  .github/workflows/ci.yml
git status --short .github                          ->   M .github/workflows/ci.yml
git diff --stat 5ddfee4 -- .github/workflows/ci.yml ->  1 file changed, 65 insertions(+)
git remote -v                                       ->  (empty)
```

The baseline already shipped a Linux CI workflow (`test`, ubuntu-latest, Python 3.10/3.11/3.12). The
Step-1/2 work **added** a matrixed windows+macos `platform-primitives` job to it — 65 added lines,
zero deletions. Any phrasing of the form "there is no Windows/macOS CI job" or "there is no
`.github/workflows`" is **factually wrong about the baseline** and is restated throughout this
document as:

> the added cross-platform job **has never executed, because this checkout has no git remote**.

That is a real limitation, and a different one. (The orchestrating session also once reported the
directory absent; that was an artifact of running `ls` from `/datac/Labelbox_Dump` instead of the
repo root. Recorded.)
Affects **F-035**, and the CI half of **F-015** and **F-016**.

### R2 — `/healthz` does **not** perform legacy adoption

`leafmachine3/core/paths.py:1405` (re-read at snapshot):
```
    record("hardware_profile", lambda: hardware_profile_path(env=env, adopt_legacy=False))
```
The health-diagnostics path explicitly opts **out** of adoption. Any claim that `/healthz` triggers
the legacy copy is wrong. Affects **F-001** (its summary named `/healthz`) and the adopt half of
**F-019**.

### R3 — the **status** path is the adoption-capable caller, repeatedly

```
leafmachine3/server/progress_api.py:372-375  ->  app.canonical_hardware_path()   # every status frame
leafmachine3/server/metrics_api.py:120-123   ->  app.canonical_hardware_path()
leafmachine3/server/app.py:145-150           ->  paths.hardware_profile_path(env=env, settings_file=beside)
leafmachine3/core/paths.py:1185              ->  adopt_legacy: bool = True
leafmachine3/core/paths.py:1205              ->  if target.is_file() or not adopt_legacy or settings_file is None: return target
```
The defect is real; only the attribution was wrong. Note the shape precisely: the copy is skipped
once the target exists, so this is a **first-call write**, not a per-frame write. The **per-frame**
cost — machine-key derivation, and with it an NVML init/shutdown when pynvml is present — is a
separate and still-real defect, because `paths.py` contains no `functools.lru_cache` anywhere
(`grep -n "lru_cache\|functools" leafmachine3/core/paths.py` → no matches).
Affects **F-001** and **F-031** (merged), and **F-019**.

### R4 — gate 13 is **not** missing from the plan

`plan:1813`: *"13. The raw bearer token appears in no log, and the HTML bootstrap is served
`no-store`."* The requirement exists and always did. What was missing was an **owning bullet in the
§4 implementation sequence** — and as of revision 13 that too now exists: `plan:1626-1631` assigns
gate 13 explicitly to Step 5b, and `plan:168` records it as amendment L4. So the correct statement
is: *the requirement existed; it was unassigned in the sequence; it has since been assigned to Step
5b and is still unimplemented in code.* Affects **F-011** and **F-034**.

### R5 — the global-greening configuration uses **absolute** paths

```
LM3_settings_global_greening.yaml:5   dirs: [/datab/Global_Greening/GBIF/images/acer_negundo]
LM3_settings_global_greening.yaml:79  models_dir: /datac/Labelbox_Dump/LM3/models/ruler_classifier
```
The relative-path defect **cannot** affect that job. No finding in this report describes global
greening as at risk. Every relative-path finding is scoped to: relative configurations generally,
the `examples/*.yaml` set, the seeded built-in default (`output.dir: runs`), and relative CLI or
public-function overrides. Affects **F-007**, **F-008**, **F-029**, **F-037**.

### R6, R7 — see §1 above.

---

## 4. Duplicate and contradiction resolution

### 4.1 Findings that are the same defect

| Cluster | IDs | Concurring gate agents | Resolution |
|---|---|---|---|
| `build_dirs()` resolves `output.dir` against the CWD | **F-007** (primary), **F-029** | conformance + completeness; also 3 of the earlier auditors, who dropped it as "Step 3 owns the wiring" | One defect. F-029 carries the better evidence: a measured single-process divergence. Tracked as F-007. |
| Postprocessing CLIs never reach the canonical resolver | **F-009** (primary), **F-030** | conformance + completeness; also the postprocessing fix agent's own hand-off | One defect, identical evidence and identical two-line fix. Tracked as F-009. |
| `resolve_port()` has no production caller | **F-006** (primary), **F-033** | conformance + completeness | One defect. Severity resolved to **high** for the defect (an implemented rule nothing enforces), but F-033 is right that gate 41 is a §8 pre-enablement gate, not a Step 1/2 exit-gate item — so it is **not** blocking. |
| Raw bearer token logged at WARNING | **F-011** (primary), **F-034** | conformance + completeness; also Appendix A row D4 | One defect. F-011's "gate 13 has no §4 step bullet at all" is **refuted by R4 and by plan revision 13**. F-034's timing argument survives and is the reason to pull it forward. |
| Legacy hardware adopt is a write inside path resolution | **F-001**, **F-031** (primary mechanism) | regression + completeness | One defect, two attributions. **F-031 is correct** (status path); **F-001's `/healthz` attribution is wrong** (R2). Tracked jointly; see §5 F-001. |
| Packaging: gate 57 unverified / wheel ships no GUI / no packaging config | **F-016**, **F-017**, **F-036** | conformance + completeness | Three overlapping statements of two distinct defects: (a) the **Python wheel** omits the GUI and over-includes `leafmachine3_backup` — F-017; (b) the **Electron** side has a pin nothing checks and no build configuration — F-016. F-036 restates both and is tracked against them. |

Net: **37 reported → 33 distinct defects.**

### 4.2 Contradictions, resolved by reading the code in this pass

| # | Contradiction | Resolution |
|---|---|---|
| C1 | F-001: "`/healthz` … creates a file in `~/.config/lm3` on its first call". F-031: the **status frame** does. | **F-031 right, F-001 wrong.** `paths.py:1405` passes `adopt_legacy=False`; `app.py:150` (reached from `progress_api.py:375` and `metrics_api.py:123`) does not. F-001's own *measured* probe called `hardware_profile_path()` and `canonical_hardware_path()` **directly** from a `python -c`, which is the adoption-capable resolver — so its measurement is sound and its route description is not. |
| C2 | F-016/F-036: "a tree-wide grep finds no electron-builder, electron-forge or @electron/packager" and "app/package.json has only … `devDependencies:{electron:43.4.1}`". | **Partially refuted, and stale.** At snapshot, `app/package.json:13` is `"electron-builder": "26.15.3"` and `app/build/` exists — and is **empty**. `git diff 5ddfee4 -- app/package.json` shows both the pin and the builder dependency were added. There is still **no `build` block, no `appId`, no icon, and no electron-builder configuration file**, so the substance of the finding stands and its premise does not. |
| C3 | F-024 cites `leafmachine3/core/paths.py:700` for `filesystem_type`. | **Line citation wrong, defect right.** `filesystem_type` is defined at `paths.py:612-630`; the `return None` for every non-Linux, non-Windows platform is `paths.py:630`, and its docstring at `:621-622` states the gap deliberately. Corrected citation used below. |
| C4 | F-028 says the plan's cfg=`None` crash citation (`hardware_setup.py:145`) is wrong and the first dereference is `:157 → _choose_tmp_dir:553`. | **Both are now stale.** Re-derived at snapshot: `run_setup` is at `hardware_setup.py:164`; `hardware_profile_path(cfg)` at `:185` takes `cfg: Any = None` and is None-tolerant; `_fingerprint(cfg)` at `:186` ignores `cfg` entirely (`:648-662`); `_probe_bound_provider(cfg)` at `:197` swallows the `AttributeError` in a bare `except Exception`. **The first unguarded dereference is `_choose_tmp_dir(cfg)` at `:198`, which does `cfg.project` at `:600`.** The durable fix for the plan is to cite by **symbol**, not by line — `hardware_setup.py` has been renumbered twice during this workflow. |
| C5 | F-007/F-029 call the divergence blocking; the earlier merged auditor set **dropped** it as out of Step-2 scope. | **The gate agents were right.** The plan itself now agrees: revision 13 adds §3.5 and a **Step 3 entry prerequisite** (`plan:1559-1563`) stating the record would otherwise name a directory the run never creates. |
| C6 | F-002 cites `paths.py:462` for the exception swallow. | Harmless drift: `:462` is `def read_gpu_uuids`; the swallow is `paths.py:480-482`. Defect confirmed. |

---

## 5. The 37 findings

> **Superseded facts, flagged rather than silently rewritten.** This report is an audit record, so
> the findings below still read as they were measured at the 10:48–10:57 snapshot. Three statements
> in them are no longer true of the tree, and the next implementer must not take them as
> specification:
>
> | Reads | Now |
> |---|---|
> | `hardware_profile_path()` takes `adopt_legacy` / `settings_file` and can adopt | **Both parameters and the embedded adopt are removed.** Migration is reachable only through `migrate_legacy_hardware_profile()`. Pinned by `test_postprocessing_cli_agreement.py::test_the_hardware_resolver_exposes_no_migration_parameter`. |
> | F-001 / F-031 open — a read path performs the legacy adopt | **Fixed.** Resolution is pure and memoized; adoption happens once, at controlled startup. `test_read_paths_are_pure.py`. |
> | F-009 has 7 tests | **16**, including two that execute each CLI's real `main()` and an 8-process adopt race. |
> | F-017 open; "no wheel was built" | **Closed.** A wheel is now built, INSTALLED into a throwaway venv and probed from an unrelated CWD (`tests/test_version_identity.py`, and a CI job). It ships the browser UI, `settings_meta.json`, `schema.sql`, the calibration images and all 33 `stl_webview` files; `leafmachine3_backup` is excluded. |
> | F-010's account of the available commands | **Partially superseded.** There is now a canonical `lm3` command with a `serve` subcommand (`leafmachine3/cli.py`), so an installed wheel has a documented way to start the server. F-010's substance — a packaged Electron app resolves `ROOT` to `<install>/resources` and so cannot SPAWN a server without `LM3_PYTHON` — is unchanged and still owned by Step 5b. |
>
> Ownership reassignments (§ below) and the Phase 8 evidence are current.


> **Currency note.** The fix-status column below was written against the 10:48–10:57 snapshot.
> Phases 3–6 of this pass then fixed **F-007, F-008, F-011 and F-031**, and a follow-up pass fixed
> **F-009**; §Phase 8 measured all five, and those rows have been brought current. Ownership was
> also reassigned for **F-006** (→ Step 5b, server startup), **F-010** (→ Step 5b) and **F-012**
> (→ Step 3, first task), and **F-010 is now confirmed** rather than unverified. Every other row still reads as of the snapshot, and
> `open` there means open.


### 5.1 Index

Severity is **as assessed after reconciliation**, not as filed. `Step` is the §4 step that owns the
fix; **"unassigned"** means no §4 bullet claims it, which is itself a plan defect and is called out.
`Blocks` = blocks Step 3 **on Linux**.

| ID | Sev | File:line | Status | Step | Blocks | Fix status | Regression test |
|---|---|---|---|---|---|---|---|
| F-001 | medium | `leafmachine3/core/paths.py:1205-1220` | confirmed | unassigned | no | open | none for the caller split |
| F-002 | medium | `leafmachine3/core/paths.py:480-482` | confirmed | unassigned | no | open | none |
| F-003 | informational (was low) | `~/.config/lm3/default-<hash8>/` | confirmed | none (operator) | no | deliberately not fixed (R7) | n/a |
| F-004 | low | `tests/test_runtime_lease.py:692-694` | confirmed | unassigned (Step 1 hygiene) | no | open | n/a — it is the test |
| F-005 | medium (was low) | `tests/conftest.py:331-342` | confirmed | unassigned (Step 1 hygiene) | no | open | none |
| F-006 | high | `leafmachine3/core/paths.py:382`, `server/app.py:893` | confirmed | **5b** (server startup) | no | open | `tests/test_paths.py::test_named_deployment_without_port_is_a_startup_error` covers the function, **nothing** covers a call site |
| F-007 | blocking | `leafmachine3/core/dirs.py:53` → now `:58` | confirmed | **3** (entry prerequisite, `plan:1559-1563`) | **yes** | **FIXED and VERIFIED** (Phase 3; §Phase 8) | `tests/test_relative_path_contract.py::test_build_dirs_and_the_early_resolver_agree` (created 10:50:26) |
| F-008 | high | `leafmachine3/core/config.py:412` → now `:411-438` | confirmed | **3** (§3.5) | **yes** | **FIXED and VERIFIED** (Phase 3; §Phase 8) | `tests/test_relative_path_contract.py::test_relative_input_and_model_paths_resolve_against_the_settings_file` |
| F-009 | high | `postprocessing/generate_stl_from_mask.py:254`; `generate_leaf_collage.py:1391` | confirmed | unassigned (§3.1 resolver scope) | no | **FIXED and VERIFIED** (`test_postprocessing_cli_agreement.py`) | `test_postprocessing_cli_agreement.py` (16 tests) |
| F-010 | high | `app/main.js:162-183` | **confirmed** — measured in a packaged AppImage (Phase 7) | **5b** | no | open | packaged-layout probe (Phase 7); no automated test yet |
| F-011 | high | `leafmachine3/server/app.py:447`, `:857` | confirmed | **5b** (assigned in revision 13, `plan:1626-1631`) | no | **FIXED and VERIFIED** (Phase 6; `test_token_hygiene.py`) | none |
| F-012 | high | `leafmachine3/core/runtime/_types.py:238-240` | confirmed | **3 — first task** | no | open | none |
| F-013 | medium | `leafmachine3/core/runtime/_types.py:507-517` | confirmed | 3 | no | open | none |
| F-014 | medium | `leafmachine3/core/runtime/records.py:216-239`; `pyproject.toml:16-21` | confirmed (Linux); **unverified** off-Linux | 3 + packaging | no | open | none |
| F-015 | medium | `.github/workflows/ci.yml:22`, `:58` | confirmed | unassigned (CI) | no | open | `tests/test_ci_workflow.py::test_the_platform_job_installs_only_the_cpu_and_dev_extras` — **currently pins the defect** |
| F-016 | medium | `.github/workflows/ci.yml:44`; `app/package.json` | **unverified**, premise partially refuted (C2) | 5a | no | partially addressed | `tests/test_ci_workflow.py` has no Node/Electron case |
| F-017 | medium | `pyproject.toml:43-52` | **unverified** (no wheel built) | unassigned (5a/7 packaging) | no | open | none |
| F-018 | medium | `leafmachine3/core/paths.py:936`, `:959` | confirmed | unassigned (proposed: 4 or 6) | no | open | `tests/test_paths.py::test_workspace_pointer_roundtrip_and_exact_schema` covers the primitive, not the GUI action |
| F-019 | medium | `leafmachine3/server/app.py:713-723`; `paths.py:1401` | confirmed | 5b (`/healthz` body) | no | open | none |
| F-020 | medium | `tests/test_executor.py:367`; `core/executor.py:368` | confirmed | 3 | no | open | none (the named test is byte-unchanged) |
| F-021 | medium | repo root — no release-notes file | confirmed | unassigned (§6 demands it; 7 is the natural host) | no | open | n/a |
| F-022 | medium | `leafmachine3/core/runtime/lease.py:347-364`, `:641` | **unverified** (Windows semantics) | 3 | no | open | none for a merged handoff |
| F-023 | medium | `tests/golden/deployment_key_vectors.json` | **unverified** (no JS exists) | 5b (not yet listed there) | no | open | `tests/test_paths.py:145` covers the Python side only |
| F-024 | medium | `leafmachine3/core/paths.py:612-630` (**not** `:700`) | confirmed (code); **unverified** natively | unassigned (gate 42) | no | open | none for Darwin |
| F-025 | low | `leafmachine3/core/runtime/records.py:1197`, `:1260` | confirmed | 3 or 4 | no | open | none |
| F-026 | low | `tests/test_runtime_grant.py:58-64` | confirmed | unassigned (Step 2 cleanup) | no | open | n/a |
| F-027 | low | `tests/_contract_helpers.py:238-255` | confirmed | unassigned (Step 1 hygiene) | no | open | none |
| F-028 | low | `plan:1909` (Appendix A), `plan:2042` | confirmed as a plan defect; **the proposed correction is also stale** (C4) | 3 (gate 45) | no | open | n/a |
| F-029 | — | duplicate of **F-007** | confirmed | 3 | yes | see F-007 | see F-007 |
| F-030 | — | duplicate of **F-009** | confirmed | unassigned | no | see F-009 | see F-009 |
| F-031 | high | `leafmachine3/server/progress_api.py:372-375` | confirmed — **this is the correct mechanism for F-001** | unassigned | no | **FIXED and VERIFIED** (Phase 5; `test_read_paths_are_pure.py`) | none |
| F-032 | medium | `tests/conftest.py:220-226` | confirmed | 1 / gate 46 | no | open | `tests/test_runtime_isolation.py::test_the_default_runtime_directory_is_never_touched` — inherits the weakness |
| F-033 | — | duplicate of **F-006** | confirmed | unassigned | no | see F-006 | see F-006 |
| F-034 | — | duplicate of **F-011** | confirmed | 5b | no | see F-011 | see F-011 |
| F-035 | medium | `.github/workflows/ci.yml:36-43`; `git remote -v` | confirmed (the no-remote fact) | unassigned (2 exit gate / 7) | **no** (§1.1 rule 1) | open by design | `tests/test_ci_workflow.py` (10 cases) pins the file's *shape*, never its execution |
| F-036 | — | restates **F-016** + **F-017** | see those | 5a / packaging | no | see those | see those |
| F-037 | low | `leafmachine3/machine3.py:186` | confirmed | 3 | no | open | none |

---

### 5.2 Detail

Each entry: **Defect** (what actually fails) · **Evidence** · **Reconciliation** where one applies ·
**Remaining risk**. Severity, status, step, blocking and fix status are in the index above and are
not repeated.

---

**F-001 · `leafmachine3/core/paths.py:1205-1220` · medium · confirmed · unassigned · does not block**

*Defect.* Asking "where does this deployment's hardware profile live?" **writes a file** into the
user's real configuration directory. `hardware_profile_path()` defaults `adopt_legacy=True`, and on
the first call for a deployment it copies a `hardware_settings.yaml` sitting beside the resolved
settings file into `<user-config>/lm3/<deployment>/hardware_settings.<machine-key>.yaml`. A
resolver is supposed to be a pure query; this one has a side effect on the filesystem, so any
read-only caller that does not explicitly opt out mutates the user's config on first use.

*Evidence.*
```
paths.py:1185   adopt_legacy: bool = True
paths.py:1205   if target.is_file() or not adopt_legacy or settings_file is None: return target
paths.py:1207   legacy = Path(settings_file).parent / LEGACY_HARDWARE_FILENAME
paths.py:1212-1213   _mkdir_private(target.parent); _write_atomic(target, legacy.read_text(...))
```
Reproduced by the regression agent with a direct call and no `LM3_*` set, which emitted
`adopted the legacy hardware profile /datac/Labelbox_Dump/LM3/hardware_settings.yaml into
<user-config>/lm3/default-<hash8>/hardware_settings.<machine-key-A>.yaml` and left a new
3605-byte 0600 file behind. Independently reproduced with `HOME` redirected to a scratch directory.

*Reconciliation (R2).* The finding's summary named the unauthenticated `GET /healthz` as the
triggering caller. **That is wrong.** `paths.py:1405` passes `adopt_legacy=False` on the diagnostics
path. The measurement stands — it called the resolver **directly** — but the route named does not.
The real production route is F-031. Severity is **kept** at medium: the defect is the side effect in
the resolver, not the identity of one caller.

*Remaining risk.* Until resolution and adoption are split, any future read-only caller re-acquires
the same hazard silently, and Step 3 multiplies the callers. The copy is idempotent (`:1205` returns
early once the target exists), so this is a first-call write, not per-frame — do not conflate it with
the per-frame machine-key cost, which is a separate defect (F-019, F-031).

---

**F-002 · `leafmachine3/core/paths.py:480-482` · medium · confirmed · unassigned · does not block**

*Defect.* The machine key silently changes when an **optional** dependency becomes importable.
`read_gpu_uuids()` folds `ModuleNotFoundError: pynvml` into the same empty tuple it returns for a
genuine CPU-only node, and the empty tuple serializes as `gpu-uuids: none` in the key input. So
`pip install pynvml` on an otherwise unchanged machine relocates the hardware profile to a different
filename and silently re-tunes from scratch, orphaning the tuned one. The key is a function of the
environment, not of the hardware.

*Evidence.*
```
paths.py:462   def read_gpu_uuids() -> tuple[str, ...]:
paths.py:480-482   except Exception:  # noqa: BLE001 - no NVML, no driver, no GPUs: all mean "none"
                       log.debug("no NVML GPU UUIDs available", exc_info=True); return ()
```
Measured on this host: `nvidia-smi --query-gpu=uuid` reports two real GPUs, while
`python -c "import pynvml"` → `ModuleNotFoundError` (re-confirmed at snapshot). Injecting the two
real UUIDs yields machine key `<machine-key-with-gpu-uuids>`; the live key is `<machine-key-A>`. `pynvml` is
declared only in the `gpu` extra (`pyproject.toml:28`), which is not installed here.

*Remaining risk.* This is the mechanism most likely to produce the F-003 symptom again. Gate 11's
corollary — one machine must not silently acquire two profiles — is not enforceable while a key
input can be `none` for two different reasons. The honest fix is a third token (`gpu-uuids:
unavailable`) so the key records which input it actually had.

---

**F-003 · `~/.config/lm3/default-<hash8>/` · informational (downgraded from low) · confirmed · no owner · does not block**

*Defect.* The user's real config directory holds two hardware profiles for one machine and one
deployment.

*Evidence.* Both files md5 `0a690a86bdbd6386f839146795c2d550`, both 3605 bytes, mtimes 2026-08-28
22:23 and 2026-08-29 00:33 (`user_config_before.txt`). The regression agent could not reproduce
`<machine-key-B>` from eight serialization variants of the current inputs.

*Reconciliation (R7).* R7 settles the cause: the two names differ **only** in the machine-key
component of the filename, because the derivation in `paths.py` changed between the two writes.
Both carry an empty `gpus:` list, so this is **not** a GPU-visibility divergence. Both are left in
place deliberately. The finding's framing — a stale artifact of an earlier revision — is right;
its implied "the current algorithm is unstable" reading is not. **Downgraded to informational: no
code change, no action by any agent.**

*Remaining risk.* None to the code. One operator decision: whether to delete the stale file before
Step 3 begins writing real deployment records beside it. Nothing in the repo may do that.

---

**F-004 · `tests/test_runtime_lease.py:692-694` · low · confirmed · unassigned · does not block**

*Defect.* One lease test allocates a temp directory with `tempfile.mkdtemp()` instead of taking the
`tmp_path` fixture, and never removes it. Every full-suite run therefore leaks a directory
containing a live-looking `activity.lock` and `active.json` — which is precisely the residue Step 1
and gate 46 exist to eliminate.

*Evidence.* Re-read at snapshot, in
`test_busy_error_carries_the_sanitized_winner_when_a_record_exists`:
```
692:    import tempfile
694:    root = Path(tempfile.mkdtemp())
```
No cleanup on any path; the function's only `try/finally` covers `owner.release()`. Attributed by
re-running the suite with a private `TMPDIR` and finding exactly one un-owned residue —
`<TMPDIR>/tmpkdu0q45f/runtime/pytest-lease/{active.json,activity.lock}` — with zero leaked
`lm3-pytest-*` sandboxes, so conftest's own teardown is correct and this directory is the test's.

*Remaining risk.* Cosmetic on a developer box; on a shared CI runner with a small `TMPDIR` it
accumulates. Every other test in the file already uses fixtures, so this is a one-line correction.

---

**F-005 · `tests/conftest.py:331-342` · medium (upgraded from low) · confirmed · unassigned · does not block**

*Defect.* `fresh_out_dir()` `rmtree`s a **process-global, repo-relative** directory,
`<repo>/examples_out/<name>`. Two suite runs in one checkout destroy each other's outputs. This is
not hypothetical: two agents were running the suite concurrently in this checkout during the gate
measurement.

*Evidence.*
```
conftest.py:331   EXAMPLES_OUT = _REPO_ROOT / "examples_out"
conftest.py:334-341   def fresh_out_dir(name): ... d = EXAMPLES_OUT / name
                        if d.exists(): shutil.rmtree(d, ignore_errors=True)
```
`ps -eo pid,etimes,cmd` during the measurement showed a second live
`timeout 1200 python -m pytest tests/ -q -p no:recording` in the same checkout. The conftest runtime
sandbox is per-session; `examples_out` is not.

*Upgrade rationale.* Filed low. Raised to medium because it is the **same shared-directory race that
would corrupt the §8 gate-60 mock-pipeline output-inventory baseline**, and moving the output root
later moves that baseline. It must be settled *before* gate 60's baseline is captured, not after.

*Remaining risk.* The three gate runs produced byte-identical results (11 failed / 912 passed / 6
skipped, same failure set), so it did not perturb the reported numbers. The hazard survives.

---

**F-006 · `leafmachine3/core/paths.py:382`; `leafmachine3/server/app.py:893` · high · confirmed · unassigned · does not block**

*Defect.* Gate 41 — "a named non-default deployment without `LM3_PORT` fails at startup"
(`plan:1841`) — cannot fire. `resolve_port()` is implemented, exported and unit-tested, and is
called by **nothing**. `serve()` takes `port: int = 8765` for every deployment, so a named
deployment started without `LM3_PORT` silently collides with the default deployment's port instead
of erroring. Two further startup preconditions have the same shape: `check_runtime_filesystem` is
reached only through a `RuntimeLease` construction that no production code performs, and §6's
"registry directory unavailable: fail before starting work" has no typed preflight at all — an
unwritable deployment directory surfaces as a bare `OSError` from inside `PosixLeaseAdapter.acquire`.

*Evidence.*
```
$ grep -rn "resolve_port" --include=*.py leafmachine3/
leafmachine3/core/paths.py:382:def resolve_port(...)      <- definition
leafmachine3/core/paths.py:1429: "raw_deployment_id", "resolve_port",   <- __all__
(no other hit)
leafmachine3/server/app.py:893   def serve(host="127.0.0.1", port: int = 8765, ...)
```
The docstring concedes it (`paths.py:385-386`): *"Deliberately NOT called during path resolution:
pytest gives every worker a named deployment."* That reasoning is correct and is not the defect; the
missing half is a **startup** call site.

*Plan defect.* Gate 41 exists in §8 but **no §4 step bullet claims it** — Step 1 is scoped to path
unification, Step 2 is explicitly not wired to production, Step 5b is Electron identity. Name an
owner (Step 3 for CLI entry points, Step 5b for the server).

*Remaining risk.* Two named deployments on one host silently share port 8765; the second `bind()`
fails with a generic `OSError` instead of the plan's precise startup error, or worse, attaches to the
wrong server. Not a Step 3 blocker because Step 3 does not start servers.

---

**F-007 (with F-029) · `leafmachine3/core/dirs.py:53` · blocking · confirmed · Step 3 entry prerequisite · BLOCKS Step 3 · fixed in-tree at 10:48:01, unverified**

*Defect.* The single most important finding in the set. `build_dirs()` resolved a relative
`project.output.dir` against the **process CWD**, while `core/runtime/config_io.resolve_run_paths()`
and every migrated server module resolve the same value against the **settings file**. The same
config, in the same process, produced two different run directories. Because Step 3 publishes
`config_io`'s answer into `active.json` while `machine3()` creates `build_dirs()`'s directories, every
relative-output run would have recorded a directory the run never creates — silently, and on the
default path, since the shipped default `output.dir` was the relative string `runs`.

*Evidence (as filed; measured in one process by the completeness agent).* `cwd=<scratch>/decoycwd`,
config at `<scratch>/cfgdir/LM3_settings.yaml`, default `project.output.dir: runs`:
```
config_io run_dir           : <scratch>/cfgdir/runs/demo
config_io active_db         : <scratch>/cfgdir/runs/demo/demo.sqlite
build_dirs would use        : runs/demo   -> <scratch>/decoycwd/runs/demo
Config.resolve_path('runs') : <scratch>/decoycwd/runs
```
Static evidence at the time of filing:
```
dirs.py:53     out = Path(str(cfg.project.output.dir))
dirs.py:70     d.mkdir(parents=True, exist_ok=True)
config.py:218  "output": {"dir": "runs", "tmp_dir": "auto"},
machine3.py:63 dirs = build_dirs(cfg)
```
`metrics_api.start_run` papered over the split by forcing `cwd = cfg_path.parent` for the child; a
CLI run started from a shell got no such correction.

*Reconciliation (R5).* This does **not** put the global-greening job at risk — that config uses
absolute paths (`LM3_settings_global_greening.yaml:5`, `:79`). Scope is relative configs, the
`examples/` YAMLs, the seeded `output.dir: runs` default, and relative CLI overrides.

*State at snapshot (moving).* The fix landed **during this closeout**, at 10:48:01:
```
dirs.py:55-58   # THE SEAM (plan section 3.5) ...
                out = paths.resolve_project_output_dir(getattr(cfg, "source_path", None),
                                                       cfg.project.output.dir)
dirs.py:70-72   tmp = Path(cfg.resolve_path(tmp_cfg)) / run / "_tmp_original"
paths.py:1299-1327  resolve_project_output_dir(): absolute passes through; "auto" -> default_output_dir();
                    relative joins Path(settings_file).resolve().parent; PathsError when no settings file
paths.py:1281-1296  default_output_dir(): <checkout>/runs in a dev checkout, else
                    <user-data>/lm3/<deployment>/runs -- never the user-config directory
config.py:218   "output": {"dir": "auto", "tmp_dir": "auto"},
machine3.py:_cli_overrides  relative --input/--output absolutized against the caller's CWD before the merge
```
and `tests/test_relative_path_contract.py` was created at 10:50:26 with 14 cases including
`test_build_dirs_and_the_early_resolver_agree` and `test_the_absolute_global_greening_config_is_unaffected`.
The plan gained §3.5 (`plan:1411-1497`), amendments L1–L3 (`plan:165-167`), and a Step 3 entry
prerequisite (`plan:1559-1563`).

*Remaining risk.* **The fix is unverified — no test run was performed by this pass.** And one half of
§3.5 has *not* landed: `examples/*.yaml` are unmigrated (see §8, N-1), so the newly-correct rule 1
now points twelve example configs at directories that do not exist. Do not treat F-007 as closed
until Phase 8 reports the contract tests green **and** the example migration lands.

---

**F-008 · `leafmachine3/core/config.py:412` · high · confirmed · Step 3 (§3.5) · BLOCKS Step 3 · fixed in-tree at 10:48:01, unverified**

*Defect.* `Config.resolve_path()` — the highest-traffic path helper in the tree, through which every
model file, `models_dir` and input directory passes — joined `Path.cwd()`. Step 1 made this newly
*inconsistent*: `metrics_api.start_run` now spawns the child with `cwd = cfg_path.parent`
unconditionally, so the Settings tab validated a relative model path against the **server's** CWD
while the run resolved it against the **config's** directory. That disagreement did not exist before
Step 1.

*Evidence (as filed).*
```
config.py:408   """Absolutize ``p``: expand ``~``; absolute paths pass through; else join CWD."""
config.py:412   return str(Path.cwd() / path)
```
Callers: `inference/factory.py:46,64`, `server/settings_api.py:577-585`. At the time of filing,
`grep -rn "Path.cwd()" leafmachine3/` returned only this line, `postprocess_api.py:1796` (a reported
diagnostic field), and a comment in `results_api.py:486`.

*State at snapshot.* Fixed. `config.py:411-438` now resolves against `self.source_path` and **raises**
rather than guessing when there is none:
```
config.py:431-437  source = getattr(self, "source_path", None)
                   if not source: raise ValueError("... LM3 never joins a configured path onto the
                   current working directory (plan section 3.5 rule 6) ...")
config.py:438      return str(Path(source).resolve().parent / path)
```
The two surviving `Path.cwd()` reads in the package are `postprocess_api.py:1796` (a reported field,
steers nothing) and a comment.

*Remaining risk.* Unverified. The raise-on-no-source-path branch is a **behavior change** for any
caller constructing a `Config` without `Config.load()` — that is the failure mode Phase 8 should watch
for in the full suite.

---

**F-009 (with F-030) · `leafmachine3/postprocessing/generate_stl_from_mask.py:254`, `generate_leaf_collage.py:1391` · high · confirmed · unassigned · does not block**

*Defect.* `postprocessing/config.py` was rewired to the canonical resolver, but both standalone
postprocessing CLIs still default `--config` to the CWD-relative string
`"postprocessing_settings.yaml"`. `load_settings()` reaches the resolver **only** when `path is
None`, and argparse always supplies the string — so the canonical branch is dead code from either
CLI. Consequence today: the Postprocess tab reads
`<user-config>/lm3/<deployment>/postprocessing.yaml` while the CLIs read `./postprocessing_settings.yaml`,
and the user's real `/datac/Labelbox_Dump/LM3/postprocessing_settings.yaml` is now invisible to the
GUI — with no legacy-adopt step for §3.1 row 3 and no release note anywhere.

*Evidence (re-read at snapshot; both files untouched since 2026-07-30 and 2026-08-14).*
```
generate_stl_from_mask.py:254  ap.add_argument("--config", default="postprocessing_settings.yaml", ...)
generate_leaf_collage.py:1391  ap.add_argument("--config", default="postprocessing_settings.yaml", ...)
postprocessing/config.py:23    if path is None:      <- resolver branch, unreachable from either CLI
postprocessing/config.py:35-36 else: p = Path(path)
```
The same defect **was** fixed for `lm3-setup`: `hardware_setup.py:1005` now defaults `--config` to
`None`. Two agents concurred (conformance and completeness), and the postprocessing fix agent filed
it as an explicit hand-off because the CLI files were outside its ownership. Nobody picked it up.

*Plan defect.* §3.1's resolver-scope bullet names "the standalone hardware-setup **and postprocessing
CLIs**", but no §4 step bullet owns the postprocessing half. Also unimplemented: row 3's "on miss:
packaged defaults" cell — `load_settings` returns `{}` and each tool falls back to its own module
constants.

*Remaining risk.* Two config sources for one feature, and a silent migration of the user's existing
file. This is the strongest candidate for the release-notes gap in F-021.

---

**F-010 · `app/main.js:162-183` · high · UNVERIFIED · Step 5b (not yet listed there) · does not block**

*Defect.* Electron spawns the server as `uvicorn leafmachine3.server.app:create_app --factory` with
`cwd = <checkout>` and **none** of `LM3_SETTINGS`, `LM3_DEPLOYMENT_ID` or `LM3_RUNTIME_DIR`. Only
`lm3 serve` resolves with `seed=True` and exports `LM3_SETTINGS`. So under Electron, settings
resolution survives on §3.1 row 5 — dev-checkout detection — alone. In an installed wheel row 5 is
gone, `canonical_settings_path()` names a file that does not exist, and the first-run hardware-setup
path reaches `app.py:762`, `cfg = Config.load(...) if job.cfg_path.is_file() else None`, handing
`run_setup(None)` — the `AttributeError` gate 45 exists to remove. Compounding it, `pyproject`
`[project.scripts]` declares only `machine3` and `lm3-setup`: **there is no `lm3` console script**, so
`lm3 serve` is not a spellable command.

*Evidence (re-read at snapshot).*
```
app/main.js:162-183   spawn(PYTHON, ["-m","uvicorn","leafmachine3.server.app:create_app","--factory", ...],
                        { cwd: ROOT, env: { ...process.env, LM3_SERVER_TOKEN: TOKEN,
                          ...(KEEP_SERVER ? {} : { LM3_OWNER_PID: String(process.pid) }) }, ... })
$ grep -rn "LM3_SETTINGS|LM3_DEPLOYMENT_ID|LM3_RUNTIME_DIR" app/*.js  -> only app/main.js:21 LM3_PORT
pyproject.toml:36-38  [project.scripts] machine3 = ...; lm3-setup = ...     (no `lm3`)
```
Partial mitigation landed: `create_app()` now seeds row 1 once per process at `app.py:613-618`, with
a comment naming `app/main.js:162-165` as the reason. That closes the *seeding* half.

*Why unverified.* The failure mode is "an installed, non-checkout wheel". No wheel was built and no
packaged install was launched by this pass or by any auditor. The static reading is sound; the
consequence is not measured. Per §1.1 rule 2, it is labeled **implemented-but-not-validated**, not
confirmed.

*Plan defect.* §3.1 says "`lm3 serve` and Electron always pass the canonical settings path
explicitly". §2.11/Step 5b list Electron's duties and never repeat this one; Step 5a is scoped to the
pin. The clause has **no owner in any step**.

*Remaining risk.* Every GUI-side gate (53–56, 59) is untestable from a wheel until this and F-017 are
both fixed.

---

**F-011 (with F-034) · `leafmachine3/server/app.py:447`, `:857` · high · confirmed · Step 5b · does not block**

*Defect.* The raw bearer token is logged in plaintext at WARNING level on every first server start,
and the token-bearing HTML bootstrap is served `Cache-Control: no-cache`, which permits storage.
This is a live log-exposure today, not a future concern: `serve()` mints and logs the token before
`uvicorn.run`.

*Evidence (re-read at snapshot).*
```
app.py:443-447   token = secrets.token_urlsafe(24); os.environ[_TOKEN_ENV] = token
                 log.warning("LM3 server token (set %s to override): %s", _TOKEN_ENV, token)
app.py:915       _server_token()   # ensure a token is minted + logged before start
app.py:851       # comment explaining that "no-cache" still stores
app.py:857       response.headers.setdefault("Cache-Control", "no-cache")
$ grep -n "no-store" leafmachine3/server/app.py   -> no matches
```
Both are already recorded in the plan's Appendix A (row D4, `plan:91`).

*Reconciliation (R4).* F-011's summary — "Gate 13 has no §4 step bullet at all" — is **refuted**.
Gate 13 exists at `plan:1813` and always did; and as of revision 13 the plan assigns it explicitly to
Step 5b at `plan:1626-1631`, with amendment L4 at `plan:168` recording exactly this. The correct
statement is: the requirement existed, it was unassigned in the sequence, it is now assigned, and it
is still unimplemented in code. **Severity kept at high** — the reconciliation corrects the plan
claim, not the exposure.

*Remaining risk.* F-034's timing argument is the operative one and survives intact: **Step 3 adds
batch and Slurm-facing logs**, so the window in which this secret lands in retained job output opens
at Step 3, not at Step 5b. The log-line change is one line and does not depend on any Step 5b
artifact. Recommend pulling it forward.

---

**F-012 · `leafmachine3/core/runtime/_types.py:238-240` · high · confirmed · Step 3 · does not block**

*Defect.* §3.3's state machine is decorative. `STATE_TRANSITIONS` is dead code — nothing in
`records.py` or `lease.py` consults it, and no test asserts an illegal transition is refused — so a
root can publish `done` and then `running`. `RecordStore.write_active(record)` validates a record in
isolation and never compares it against the previous state on disk. There is also **no record
builder**: `RecordStore` only writes a fully-formed frozen `RuntimeRecord`, so Step 3 must
hand-assemble 15 common fields plus blocks at roughly six call sites (machine3, hardware_setup root,
calibration child, batch root, batch item, server launch) with nothing enforcing consistency between
them.

*Evidence.*
```
$ grep -rn "STATE_TRANSITIONS" --include=*.py leafmachine3/ tests/
leafmachine3/core/runtime/_types.py:238   <- definition
leafmachine3/core/runtime/__init__.py:97, :290  <- re-export + __all__
leafmachine3/core/runtime/_types.py:998         <- __all__
(no consumer, no test)
```

*Remaining risk.* This is Step 3 work, not a Step 3 blocker — but it is the item most likely to
produce a silently inconsistent registry if Step 3 is written call-site by call-site. The two
constructor helpers and one `RecordStore.transition()` should land before the sixth call site, not
after.

---

**F-013 · `leafmachine3/core/runtime/_types.py:507-517` · medium · confirmed · Step 3 · does not block**

*Defect.* `DeploymentInfo` can be validated but not **produced**. Nothing in the tree detects the
scheduler name, `job_id`, `step_id`, `node` or `container_id`. And the plan's own example
contradicts the implementation's comment about which value `id` carries.

*Evidence.*
```
_types.py:512   id: str    # the canonical deployment key
_types.py:513-517  scheduler / job_id / step_id / node / container_id, all Optional, all descriptive
records.py:519, :390   decode/encode only
$ grep -rn "SLURM_JOB_ID|scheduler" leafmachine3/core/runtime/*.py  -> no detector
```
The only scheduler probe in the tree is the module-private `paths._scheduler_job_id`
(`paths.py:749-750`, over `SLURM_JOB_ID`/`SLURM_JOBID`/`PBS_JOBID`/`LSB_JOBID`), which returns a job
id only — no scheduler name, no step id, no container id. `paths.read_node_name` (`:457-459`) covers
`node`.

*The unpinned contract.* §3.2's example shows `"id": "slurm-482193"`, which is the **raw**
`LM3_DEPLOYMENT_ID` from §3.1's Slurm example; its canonical key would be `slurm-482193-<hash8>`.
§2.11 requires clients to verify the deployment key from `/healthz`, so which of the two lands in
the record is load-bearing and the plan states both.

*Remaining risk.* If Step 3 picks the raw value and `/healthz` reports the canonical one, every
client-side deployment check fails against a record that looks correct. Settle `deployment.id` as the
canonical key and fix §3.2's example JSON before Step 3 writes a record.

---

**F-014 · `leafmachine3/core/runtime/records.py:216-239`; `pyproject.toml:16-21` · medium · confirmed on Linux, UNVERIFIED off-Linux · Step 3 + packaging · does not block**

*Defect.* `process_start_time()` returns `0.0` on every platform except Linux unless `psutil` is
installed, and `psutil` is declared in **no** dependency list. On a clean Windows or macOS install
every record therefore carries `process_started_at: 0.0`, validation accepts it, and §2.5's PID-reuse
defense degrades silently to PID-only matching — the exact failure §2.5 exists to prevent.

*Evidence (re-read at snapshot).*
```
records.py:225-233  try: import psutil ... except Exception: pass
records.py:234-239  if sys.platform.startswith("linux"): return _linux_process_start_time(target)
                    ... return 0.0
records.py:722      if record.process_started_at < 0: raise RecordSchemaError(...)   # 0.0 is accepted
$ grep -n psutil pyproject.toml   -> no match
pyproject.toml:16-21  dependencies = [numpy, opencv-python-headless, pillow, pyyaml]
```
`pip install -e ".[cpu,dev]"` — exactly what the new windows-latest/macos-latest CI job installs
(`ci.yml:58`) — pulls no `psutil`. `metrics_api._pid_alive` already imports it, so it is a de-facto
dependency.

*Why partly unverified.* On Linux the `/proc` fallback works and this is a non-issue; that half is
confirmed by reading. The degraded-match consequence exists only on Windows and macOS and has never
been executed there. Per §1.1 rule 2, that half is **not validated**.

*Remaining risk.* The docstring at `:222` already says a caller comparing two unknowns must treat the
match as failed. **Nothing enforces it.** Whichever Step 3 call site implements the five-way
`can_stop` match will get this wrong by default.

---

**F-015 · `.github/workflows/ci.yml:22`, `:58` · medium · confirmed · unassigned · does not block**

*Defect.* Neither CI job installs the `server` extra, so every FastAPI-dependent test
`importorskip`s in CI and the job stays green. Step 1's entire server-side exit gate is invisible to
CI on all three platforms — **including** in the new platform job, which explicitly names
`tests/test_settings_path_unification.py` among the files it runs and therefore reads as coverage it
does not have.

*Evidence (re-read at snapshot).*
```
ci.yml:22   pip install -e ".[cpu,dev]"     (test job)
ci.yml:58   pip install -e ".[cpu,dev]"     (platform-primitives job)
pyproject.toml:33   server = ["fastapi>=0.110", "uvicorn>=0.29", "python-multipart>=0.0.9", "sse-starlette>=2.0"]
```
`tests/test_settings_path_unification.py` guards its server assertions with
`pytest.importorskip("fastapi")` at `:412`, `:432`, `:482`, `:513`. Same for `tests/test_api_contract.py`.

*The test that pins the defect.* `tests/test_ci_workflow.py:166`,
`test_the_platform_job_installs_only_the_cpu_and_dev_extras`, currently **asserts the wrong
behavior**. Fixing the workflow requires changing that test in the same commit.

*Remaining risk.* An `importorskip` cannot make a CI leg fail. The platform job needs the same
skip-detection guard it already applies to the two real-Win32 node ids (`ci.yml:76-91`), or a green
leg means nothing for the server tests it claims to run.

---

**F-016 (restated by F-036) · `.github/workflows/ci.yml:44`; `app/package.json` · medium · UNVERIFIED, premise partially refuted · Step 5a · does not block**

*Defect.* Gate 57 — "Electron runs an exactly pinned, patched release on a currently supported major
line" — is verified by **nothing automated**. CI has no Node step, so neither the pin, the lockfile,
nor the binary is checked on any push. The 33.4.11 → 43.4.1 jump (roughly 20 Chromium majors under
`leafmachine3/server/ui`) rests on one agent's manual xvfb probe, and the pin can rot off the
supported line (43 EOL 2027-01-05) with nothing noticing.

*Evidence (re-read at snapshot).*
```
$ grep -ni "node|npm|electron" .github/workflows/ci.yml   -> (none)
app/package.json:12-14   "devDependencies": { "electron": "43.4.1", "electron-builder": "26.15.3" }
$ ls app/build   -> (empty directory)
$ grep -n '"build"' app/package.json   -> (no build block)
$ grep -n requestSingleInstanceLock app/main.js  -> (absent; Step 5b owns it)
```

*Contradiction C2, applied.* The finding as filed says the repository has "no electron-builder,
electron-forge or @electron/packager" and that `app/package.json` carries only
`devDependencies:{electron:43.4.1}`. **Both statements are stale**: `electron-builder` 26.15.3 is a
declared devDependency and installed under `app/node_modules/`, and an (empty) `app/build/` exists.
The substance survives unchanged: there is still **no `build` block, no `appId`, no icon, and no
electron-builder configuration file**, so there is still nothing to package and Step 5a's
"smoke-test packaging" clause is still unsatisfiable as written.

*Why unverified.* No `npm` command was run by this pass, and no packaging was attempted. The Electron
pin's correctness — 43.4.1 exact, above the CVE-2026-34776 fixed versions, on a supported major — was
checked by an earlier auditor and is not re-established here.

*Remaining risk.* Amend Step 5a's wording: as written, its exit condition cannot be met by this
repository. Either add the packaging configuration or state that packaging is deferred and record
5a as "install-and-launch verified only", which is what `app/ELECTRON_UPGRADE.md` already says.

---

**F-017 (restated by F-036) · `pyproject.toml:43-52` · medium · UNVERIFIED · unassigned · does not block**

*Defect.* Only the calibration images were actually packaged. The wheel still omits the **entire
GUI** — `leafmachine3/server/ui`, including `settings_meta.json` — and `postprocessing/stl_webview`,
and `packages.find` additionally ships the stray top-level `leafmachine3_backup` package.
`app.py:625` mounts the UI behind `if _ui.is_dir():`, so from an installed wheel the GUI is simply
absent, silently.

*Evidence (re-read at snapshot).*
```
pyproject.toml:43-44   [tool.setuptools.packages.find] include = ["leafmachine3*"]
pyproject.toml:46-52   [tool.setuptools.package-data]
                       leafmachine3 = ["core/schema.sql"]
                       "leafmachine3.setup.calibration_images" = ["*.jpg", "README.md"]
$ ls -d leafmachine3_backup          -> leafmachine3_backup      (matched by "leafmachine3*")
$ ls leafmachine3/server/ui          -> css  index.html  js  settings_meta.json
$ ls leafmachine3/server/ui/__init__.py -> No such file or directory   (packages.find skips it)
$ ls leafmachine3/postprocessing/stl_webview -> assets  _build.py  generate_3d_file.html  README.md  src
```

*Why unverified.* No wheel was built and no throwaway venv install was performed, here or by any
auditor. The §7 row this covers — "calibration images resolve from an installed wheel/container
image, not only a source checkout" — was verified once **by hand** and by nothing repeatable.

*Remaining risk.* Every GUI-side gate (53–56, 59) is untestable from a wheel. The missing repeatable
test is specific and cheap: build a wheel into a tmp dir, install into a throwaway venv, and from an
unrelated CWD assert `calibration_images_dir()` resolves into the package, `settings_meta.json` is
present, and the UI mount point exists.

---

**F-018 · `leafmachine3/core/paths.py:936`, `:959` · medium · confirmed · unassigned · does not block**

*Defect.* `write_workspace_pointer` / `clear_workspace_pointer` have no production caller, and no §4
step bullet implements the GUI action that would call them — so §3.1 precedence **row 3 is
unreachable in practice**. The pointer can only ever be created by hand.

*Evidence (re-read at snapshot).*
```
$ grep -rn "write_workspace_pointer|clear_workspace_pointer" --include=*.py leafmachine3/
leafmachine3/core/paths.py:936   <- definition (:959 for clear)
(every other hit is in tests/test_paths.py)
$ grep -n workspace leafmachine3/server/settings_api.py
:117 (a docstring naming the pointer in the precedence chain), :637 (a comment)   <- no endpoint
```
§3.1's own table names the sole writer ("the server's Settings API") and the trigger ("the explicit
GUI 'choose settings file' action only"). §4 Step 1 says only "Replace local path logic in
settings_api.py …"; Steps 4, 5b, 6 and 7 never mention the pointer. Gate 15 covers only the **read**
side — a pointer whose target is gone must fail visibly — which is implemented and tested
(`tests/test_paths.py::test_pointer_with_a_missing_target_fails_visibly`).

*Remaining risk.* A fully-specified, fully-tested precedence row that no user can reach. Low
operational risk, but it means §3.1's row 3 is currently unfalsifiable in production.

---

**F-019 · `leafmachine3/server/app.py:713-723`; `leafmachine3/core/paths.py:1401` · medium · confirmed · Step 5b · does not block**

*Defect.* Two things, one line apart. (a) The **unauthenticated** `/healthz` publishes
`machine_key` — a host fingerprint — and the full absolute resolved-path map, including the user's
home-directory layout. That is more than §5b's specified body (service, protocol version, instance
ID, deployment key, ownership mode), and §2.11 requires redaction from `/healthz` diagnostics.
(b) `machine_key` is **recomputed on every request**, and the same uncached chain runs on every
status frame, so with `pynvml` present each health poll and each frame performs an
`nvmlInit()`/`nvmlShutdown()` cycle inside the server while workers hold CUDA contexts.

*Evidence (re-read at snapshot).*
```
app.py:713-723   @app.get("/healthz")   # the file documents this route as deliberately unauthenticated
                 return {"status": "ok", "version": "3.0.0", "provider": ..., "pid": ...,
                         "paths": path_diagnostics()}
paths.py:1399-1408   describe_resolved_paths records deployment_id, deployment_key, machine_key,
                     runtime_base, runtime_dir, settings, hardware_profile, postprocess_settings,
                     server_jobs_root, dev_checkout
paths.py:1401    record("machine_key", machine_key)
$ grep -n "lru_cache|functools" leafmachine3/core/paths.py   -> no matches   (nothing is memoized)
paths.py:467,478   pynvml.nvmlInit() ... pynvml.nvmlShutdown()
progress_api.py:365-375   docstring: "this is read on every status frame"
```

*Reconciliation (R2).* The finding does **not** claim adoption on `/healthz` and must not be read
that way: `paths.py:1405` passes `adopt_legacy=False`. The `/healthz` problems are **disclosure** and
**cost**, not mutation. The mutation is F-001/F-031, on the status path.

*Remaining risk.* Both fixes are small and independent: memoize `machine_key()`/
`detect_machine_identity()` per process (all three inputs are fixed for the process lifetime), and
decide the `/healthz` body explicitly in Step 5b — keep `deployment_key` (§2.11 needs clients to
verify it), move `machine_key` and the absolute path map behind the token or reduce them to
basenames. Note that `pynvml` is absent on this host, so the NVML cost is currently latent here and
would appear on any machine with the `gpu` extra installed.

---

**F-020 · `tests/test_executor.py:367`; `leafmachine3/core/executor.py:368` · medium · confirmed · Step 3 · does not block**

*Defect.* §2.1 requires
`test_reaper_never_touches_a_concurrent_lm3_runs_workers` to be "reframed" for the deployment model —
the reaper must not touch workers belonging to **another live deployment**. It was never reframed:
the named test is byte-unchanged from baseline, and the reaper has no notion of a deployment at all.
Nothing pins that reaping happens only **after** the lease is acquired, which is §1 invariant 14 ("a
losing launch performs no orphan reaping").

*Evidence (re-read at snapshot).*
```
$ git status --short   ->  tests/test_executor.py is NOT listed (byte-unchanged from 5ddfee4)
$ git diff --stat 5ddfee4 -- leafmachine3/machine3.py   (at the time of filing: no diff)
executor.reap_orphaned_workers (executor.py:368) selects on uid, interpreter path, and whether the
parent command line looks like LM3
$ grep -n "LM3_DEPLOYMENT_ID|deployment" leafmachine3/core/executor.py  -> nothing
```
The test's docstring still reads "The whole safety story: a live LM3 parent means its children are
NOT orphans." §3.3 orders acquisition before orphan-worker cleanup, but `machine3.py` had no lease
wiring, so there was no ordering to test.

*Remaining risk.* Two named deployments on one host, one reaping the other's workers. The fix has a
prerequisite nobody has stated: the reaper cannot see a deployment unless the deployment id is put
into the **worker's** environment or command line. That belongs on Step 3's bullet list and appears
in no step today.

---

**F-021 · repo root — no release-notes file · medium · confirmed · unassigned · does not block**

*Defect.* The one clause in the plan that explicitly demands a document *outside* the plan produced
no document. §6: *"Hard rule: runtime v2 must not be enabled while a pre-registry run or a resumable
batch is still executing from the old environment. This belongs in the release notes and the
deployment instructions, not only here."* There are no release notes. Meanwhile Step 1 has already
shipped four user-visible behavior changes that nothing tells anyone about.

*Evidence (re-read at snapshot).*
```
$ ls *.md
DEPLOYMENT_PLAN.md  electron_app_plan.md  GUI_LM3_CONTROL_INVESTIGATION.md  INSTALL.md
README.md  TODO.md  UNIFIED_RUNTIME_IMPLEMENTATION_PLAN{,_CONSENSUS,_REVIEW}.md
(no RELEASE_NOTES.md, no CHANGELOG)
```
Unannounced behavior changes already shipped:
1. the hardware profile moved to `<user-config>/lm3/<deployment>/`, adopted by copy, one release only;
2. `postprocess_api`'s `allowed_roots` no longer include the server's CWD — anyone running the server
   from a directory of runs must now set `LM3_RUNS_ROOTS` or `LM3_POSTPROCESS_ROOTS`;
3. GUI-started runs now execute with `cwd = <config dir>`, moving where every relative model/input
   path resolves;
4. `LM3_SETTINGS_PATH` and `LM3_HARDWARE_SETTINGS` are deprecated with a one-release window.

Add to that list, from this closeout: the postprocessing settings relocation (F-009) and — as of
10:48 today — the §3.5 relative-path rule change and the `output.dir` default becoming `auto`.

*Remaining risk.* The cutover rule is an **operational safety rule**, not documentation polish. §7's
release-facing rows cannot be closed without it.

---

**F-022 · `leafmachine3/core/runtime/lease.py:347-364`, `:641` · medium · UNVERIFIED · Step 3 · does not block**

*Defect.* `ChildHandoff.popen_kwargs` is a silent trap laid directly in Step 3's path. On Windows a
`STARTUPINFOEX` handle allowlist is **exhaustive**, and a second `startupinfo=` argument **replaces**
the first — so a launcher that also passes the §2.4 status-pipe handle drops the lease with no error
and no test failing. On POSIX the same shape applies to `pass_fds`.

*Evidence (re-read at snapshot).*
```
lease.py:347-349  """`popen_kwargs` describes the lease and nothing else. A launcher that also passes
                     the section 2.4 status pipe MERGES its descriptor into this `pass_fds` tuple rather
                     than replacing it -- passing two separate `pass_fds` would silently drop one."""
lease.py:364      popen_kwargs={"pass_fds": (fd,)}
lease.py:641      popen_kwargs=_win32.handle_list_popen_kwargs([duplicate])
_win32.py:71      def handle_list_popen_kwargs(handles: Sequence[int], ...)   # ONE sequence
```
Both `_handoff` docstrings state the merge requirement. **Nothing enforces it.** `ChildHandoff`
exposes `lease_fd`/`lease_handle` (`_types.py:727-732`) so a correct merge is possible — but the
ready-made `popen_kwargs` is the obvious thing to pass and becomes wrong the moment a status pipe
exists, which is exactly what Step 3 adds.

*Why unverified.* The Windows half rests on Win32 `STARTUPINFOEX` semantics that have never executed
on a real kernel from this repository. The 29 Windows adapter tests run against `FakeWin32` on Linux.
Per §1.1 rule 2 this is a model, not the object manager.

*Remaining risk.* The failure is **silent**: the child runs, holds no lease, and the deployment looks
free. Add `merge_popen_kwargs(*, handoffs=(), extra_fds=(), extra_handles=())` in one place and make
`popen_kwargs` private or explicitly lease-only, **before** Step 3 writes the launcher.

---

**F-023 · `tests/golden/deployment_key_vectors.json` · medium · UNVERIFIED · Step 5b (not yet listed there) · does not block**

*Defect.* §2.1 requires the deployment key to be "derived by one function with **golden test vectors
shared between Python and Electron**, because the two must agree byte-for-byte", and gate 40's second
half requires the canonical keys to "match Electron's golden vectors". The vectors exist and are
consumed by **exactly one** implementation. Nothing obliges Step 5b's JavaScript to read them, and
there is no harness of any kind for the second half.

*Evidence (re-read at snapshot).*
```
$ grep -rn "deployment_key_vectors" . (excluding node_modules)
tests/test_paths.py:145   <- the only consumer
$ ls app/*.js   ->  app/main.js  app/preload.js      (no deployment.js, no key derivation)
```
The vectors deliberately preserve the literal formula's leading-dash artifact (`..` → `-5ec1f7e7`,
`ß` → `-cd3a7e92`) — exactly the class of detail a from-scratch JS implementation will "tidy" with a
`toLowerCase()` or a `normalize()`.

*Why unverified.* There is no JavaScript implementation to test. This is a gap, not a defect in
shipped code.

*Remaining risk.* A mismatch produces a **directory divergence before any test could catch it** —
Electron and Python would key the same deployment differently, which is the precise failure §2.1 was
written to prevent. Gate 40 must not be counted as passing: its POSIX/Windows non-contention half is
covered (`tests/test_paths.py:136`, `tests/test_runtime_lease_windows.py:389`); its Electron-agreement
half has no test and cannot until 5b lands.

---

**F-024 · `leafmachine3/core/paths.py:612-630` (**not** `:700`) · medium · confirmed in code, UNVERIFIED natively · unassigned · does not block**

*Defect.* Network-filesystem detection is Linux + Windows only. On macOS `filesystem_type()` returns
`None`, `is_network_filesystem(None)` is `False`, and `check_runtime_filesystem` therefore **permits**
it — so gate 42 ("a runtime directory on a network filesystem is refused without the explicit
override") passes vacuously on the platform the new CI job now nominally runs. An AFP/SMB/NFS home
directory would be accepted as a lease location.

*Evidence (re-read at snapshot; citation corrected per C3).*
```
paths.py:612-630   def filesystem_type(...):
paths.py:621-622     """macOS is deliberately ``None`` today: there is no dependency-free ``statfs``
                        binding, and a guessed answer would either refuse a working local disk or
                        bless an NFS mount."""
paths.py:626-630     linux -> _linux_fs_type; win -> _windows_fs_type; return None
paths.py:633-642   is_network_filesystem(None) -> False
paths.py:645-658   check_runtime_filesystem: if not is_network_filesystem(fs_type): return fs_type
```
The implementer disclosed the gap and recorded it as a follow-up rather than faking it; no fix agent
acted. The macos-latest leg runs `tests/test_paths.py`, which asserts the Linux and Windows behaviors
only.

*Why partly unverified.* The code path is confirmed by reading. The consequence — a real Mac with a
network-mounted home silently accepting a lease location — has never been executed. §1.1 defers macOS
native validation, so this is explicitly a deferred-platform item, **not** a Linux Step 3 blocker.

*Remaining risk.* The gap is currently **invisible at startup**. Minimum acceptable interim: have
`check_runtime_filesystem` log a named warning on an unknown filesystem type rather than passing
silently, so it surfaces at startup instead of when an NFS lock fails.

---

**F-025 · `leafmachine3/core/runtime/records.py:1197`, `:1260` · low · confirmed · Step 3 or 4 · does not block**

*Defect.* The registry has **two independent answers** to "when is a grant prunable".
`records.prune_children` reimplements grant expiry instead of delegating to `grant.prune_grants`.

*Evidence (re-read at snapshot).*
```
records.py:1197   def _grant_is_expired(path: Path, now: float) -> bool:   # parses the grant file itself
records.py:1260   if _grant_is_expired(path, now):
grant.py:614      def prune_grants(deployment_dir, clock=..., retention=...)
$ grep -n "prune_grants" leafmachine3/core/runtime/records.py   -> no match
```
`grant.py` has its own expiry parsing, its own treatment of unreadable grants (removed, since they can
never be redeemed), and its own consumed-marker retention. The two can drift on the exclusive-vs-
inclusive expiry boundary that `grant.py` deliberately tested at 1059.999 / 1060.0.

*Remaining risk.* Low today (neither is wired to production). It becomes a live inconsistency the
moment Step 3 starts pruning, and the symptom — a grant one module considers live and the other
deletes — is hard to attribute after the fact.

---

**F-026 · `tests/test_runtime_grant.py:58-64` · low · confirmed · unassigned · does not block**

*Defect.* The `_StandInRecords` bridge that stood in for `records.py` while it was being written
concurrently is still in the file. It is inert today, but a bare `except Exception` around the import
means a future import error in `records.py` **silently re-arms it**, converting a hard failure into a
green run against a fake.

*Evidence (re-read at snapshot).*
```
test_runtime_grant.py:58-64  try: from leafmachine3.core.runtime import records as _real_records
                                  _RECORDS_AVAILABLE = True
                             except Exception: _real_records = None; _RECORDS_AVAILABLE = False
test_runtime_grant.py:67     class _StandInRecords:
                             @pytest.fixture(autouse=True) def _records_bridge(monkeypatch): ...
```
Its author explicitly listed deletion as the completion step for that hand-off.

*Remaining risk.* 63 tests that the Step 2 exit gate claims exercise the real
`atomic_write_json`/`read_json_file` could, under one import error, exercise a stub instead — and
report green.

---

**F-027 · `tests/_contract_helpers.py:238-255` · low · confirmed · unassigned · does not block**

*Defect.* The gate-46 isolation depends on no test module ever adding `LM3_RUNTIME_DIR`,
`LM3_DEPLOYMENT_ID` or an `XDG_*` name to its `delenv` list. The sibling file carries a comment
saying so; this one does not, so it currently omits the dangerous names **by luck rather than by
rule**. A future "for completeness" addition would evaporate the isolation with no failing assertion.

*Evidence (re-read at snapshot).*
```
test_characterization_paths.py:50-53   "``LM3_RUNTIME_DIR`` and ``LM3_DEPLOYMENT_ID`` are deliberately
   NOT in this list: ``tests/conftest.py`` sets them at import time to keep the suite out of the real
   deployment, and clearing them here would undo it."
_contract_helpers.py:238-255   the parallel _PATH_ENV tuple, with no such note
```

*Remaining risk.* An invariant that is remembered rather than checked. The enforceable version is
cheap: a case in `tests/test_runtime_isolation.py` that greps every `tests/*.py` for a `delenv` of
those names and fails.

---

**F-028 · `UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md:1909` (and `:2042`) · low · confirmed as a plan defect; the proposed correction is itself stale · Step 3 (gate 45) · does not block**

*Defect.* Appendix A and §2.13 cite the wrong first dereference for the `cfg=None` crash that gate 45
exists to remove, so Step 3 would implement gate 45's "precise error message" at the wrong place.

*Evidence (re-read at snapshot).*
```
plan:1909   | **`run_setup` dereferences cfg unconditionally — no `None` guard exists** |
              `hardware_setup.py:145`, `:162`, `:662` |
plan:2042   `app.py:577` passes `None` and `run_setup` dereferences it at `hardware_setup.py:145`/`:162`
```
The Step-1 implementer reported the correction — *"the FIRST dereference is `hardware_setup.py:157`
→ `_choose_tmp_dir` → `:553` (`cfg.project`), not `:145` and not `:162`"* — and no agent updated the
plan. That correction is the only recon-verified plan correction from this workflow that landed
nowhere.

*Resolution (C4) — re-derived first-hand at snapshot, because `hardware_setup.py` has been renumbered
again since the correction was filed:*
```
hardware_setup.py:164   def run_setup(cfg, *, optimize=True, quick=False, force=False, ...)
hardware_setup.py:185     hw_path = hardware_profile_path(cfg)      # def at :48, `cfg: Any = None` -> None-tolerant
hardware_setup.py:186     fingerprint = _fingerprint(cfg)           # def at :648 -- ignores cfg entirely
hardware_setup.py:197     provider = _probe_bound_provider(cfg)     # def at :568 -- bare `except Exception` swallows it
hardware_setup.py:198     tmp_dir = ... _choose_tmp_dir(cfg, min_free_gb=50)
hardware_setup.py:600       tmp_cfg = str(_cfg_get(_cfg_get(cfg.project, "output"), "tmp_dir", "auto"))   <- FIRST UNGUARDED DEREFERENCE
```
So **both** the plan's `:145` and the implementer's `:157/:553` are now wrong. The durable fix is to
cite by **symbol** — `run_setup` → `_choose_tmp_dir` → `cfg.project` — rather than by line. This file
has been renumbered twice during this workflow and will be again.

*Remaining risk.* Small but real: gate 45's whole point is a precise error at the right place, and
the plan currently points three different readers at three different wrong lines.

---

**F-029 · duplicate of F-007.** Filed independently by the completeness agent against
`leafmachine3/core/dirs.py:53` at severity blocking, with the measured three-way divergence
(`config_io` vs `build_dirs` vs `Config.resolve_path`) reproduced in one process from a decoy CWD, and
with the correct conclusion that it must land **before** Step 3 wires the record. Its evidence is the
stronger of the two and is quoted under F-007. Both agents concurred; three of the earlier auditors
also reported it and the merge agent **dropped** it as out of Step-2 scope — see contradiction C5,
where the gate agents are upheld.

---

**F-030 · duplicate of F-009.** Filed independently by the completeness agent against the same two
argparse lines, adding the observation that the two module docstrings
(`generate_stl_from_mask.py:8-11`, `generate_leaf_collage.py:21-24`) still print
`--config postprocessing_settings.yaml` as the recommended invocation, and proposing the specific
test: both parsers leave `--config` at `None`, and a decoy `postprocessing_settings.yaml` in the CWD is
invisible. Two gate agents plus the postprocessing fix agent concurred.

---

**F-031 · `leafmachine3/server/progress_api.py:372-375` · high · confirmed · unassigned · does not block**

*Defect.* **This is the correct mechanism behind F-001.** The §3.1 row-2 legacy adopt is a filesystem
write performed as a side effect of path resolution, and that resolution runs on the **status path** —
`progress_api._hardware()` → `app.canonical_hardware_path()` → `paths.hardware_profile_path()` with
`adopt_legacy` defaulting to `True`. §1 invariant 13 says the server "observes"; here the observer
writes into the user's configuration directory. The same call also re-derives the machine key on every
frame, and `paths.py` memoizes nothing.

*Evidence (re-read at snapshot).*
```
progress_api.py:365-375   """... this is read on every status frame."""
                          return _read_yaml(canonical_hardware_path())   # resolution NOT cached
metrics_api.py:120-123    path = canonical_hardware_path()
app.py:145-150            return paths.hardware_profile_path(env=env, settings_file=beside)
paths.py:1185             adopt_legacy: bool = True
paths.py:1208-1213        if legacy.is_file(): _mkdir_private(target.parent); _write_atomic(target, ...)
```
Reproduced with `HOME` redirected to a scratch directory; the adoption log line names both paths. On
the real box it produced the two byte-identical copies of the checkout's `hardware_settings.yaml`
described in R7.

*Reconciliation (R2, R3).* R3 confirms this route exactly and adds the shape correction that must be
carried: because `paths.py:1205` returns early once the target exists, the **write** is a first-call
event, not a per-frame event. The **per-frame** defect is the machine-key derivation (and NVML cycle
where pynvml is installed), which is real and separate. Do not describe this as a 2 Hz write.

*Remaining risk.* The fix has a template already accepted in this tree: `create_app()` does exactly
this split for the row-1 seed at `app.py:613-618` — adopt once at startup, and have every
per-request/per-frame reader pass `adopt_legacy=False`. The hardware chain was simply not covered by
that finding.

---

**F-032 · `tests/conftest.py:220-226` · medium · confirmed · Step 1 / gate 46 · does not block**

*Defect.* The gate-46 guard is weaker than its docstring claims. It snapshots `(exists,
st_mtime_ns)` of the **top-level** default locations only. A directory's mtime moves when an entry is
created directly inside it — **not** when a file is added to an already-existing child. So a write
into an existing `~/.config/lm3/<deployment>/` is invisible to the guard. On any developer box that has
ever run LM3, that subdirectory always exists, which is precisely the case the guard is written for.

*Evidence (re-read at snapshot).*
```
conftest.py:220-226   def _stat_signature(path): """(exists, mtime_ns). A directory's mtime moves when
                        an entry is created inside it, so this catches "a test created
                        <default>/lm3/<deployment>" as well as outright creation."""
test_runtime_isolation.py:115-127   iterates DEFAULT_LOCATIONS with _stat_signature, nothing deeper
```
Disproof on this host: a file was added to the existing deployment directory at 2026-08-29 00:33 and
the **snapshotted parent kept its 2026-08-28 22:23 mtime**:
```
drwxrwxr-x  ~/.config/lm3                                2026-08-28 22:23:00   <- snapshotted, unchanged
drwx------  ~/.config/lm3/default-<hash8>               2026-08-29 00:33:18
-rw-------  .../hardware_settings.86c99...yaml           2026-08-29 00:33:18   <- added after the guard existed
```

*Not a live leak.* The completeness agent verified independently that the current suite writes nothing:
a full `pytest tests/ -q -p no:recording` with `HOME` redirected to a scratch directory wrote nothing
at all under that HOME, and neither did `-n 4` nor the scoped nine-file set. The two stale files came
from un-isolated ad-hoc `python -c` invocations (F-001's mechanism), not from pytest.

*Remaining risk.* This is a **guard-strength** defect: gate 46 is currently satisfied in behavior and
under-proven in mechanism. Correct the docstring at minimum — it overstates what one `stat` proves —
and make the signature recursive over the LM3-owned trees with a file-count cap.

---

**F-033 · duplicate of F-006.** Filed independently by the completeness agent against
`leafmachine3/server/app.py:893` with the same `grep -rn "resolve_port"` evidence. It adds the correct
scoping judgment used in this report: gate 41 is a §8 pre-enablement gate rather than a Step 1/2/5a
exit-gate item, so the missing call site does **not** by itself fail those exit gates — but the plan
has an implemented rule that nothing enforces. It also correctly credits the `paths` author for
deliberately keeping `resolve_port` out of path resolution.

---

**F-034 · duplicate of F-011.** Filed independently by the completeness agent against
`leafmachine3/server/app.py:447` with the same quoted `_server_token()` body. It contributes the
timing argument this report adopts: §4 assigns the `/healthz` body and the connection-descriptor flow
to Step 5b, **but Step 3 adds batch and Slurm-facing logs**, so the window in which the secret lands in
retained job output opens at Step 3. The one-line log change depends on no Step 5b artifact and should
be pulled forward.

---

**F-035 · `.github/workflows/ci.yml:36-43` · medium · confirmed (the no-remote fact) · unassigned · does NOT block Step 3 on Linux**

*Defect.* The windows/macOS `platform-primitives` job is well constructed but **has never executed and
cannot execute from this checkout, which has no git remote.** Separately, the Linux `test` job it sits
beside runs the whole suite, which is red at baseline (11 failures, 8 of them
`ModuleNotFoundError: No module named 'ect'` and a NumPy < 2.0 gap), so even once a remote exists the CI signal for Step 2 is
buried in unrelated noise.

*Evidence (re-read at snapshot).*
```
ci.yml:40-43   # NOT YET DEMONSTRATED: this checkout has no git remote, so nothing under
               # .github/workflows/ executes from it. Step 2 may not claim its "all three platforms"
               # exit gate, or §8 gate 27, on the strength of this file alone
$ git remote -v          ->  (empty)
ci.yml:24-26   - name: Test (mock pipeline — no models/GPU required)
                 run: pytest -q
$ python -c "import ect"  ->  ModuleNotFoundError: No module named 'ect'
```

*Reconciliation (R1) — the correction that matters most in this set.* This finding must **not** be read
as "there is no cross-platform CI". The baseline shipped `.github/workflows/ci.yml`
(`git ls-tree -r 5ddfee4 --name-only | grep github`), and the Step-1/2 work **added** the matrixed
windows+macos job to it — `git diff --stat 5ddfee4 -- .github/workflows/ci.yml` reports
**1 file changed, 65 insertions(+), 0 deletions**. The accurate statement is the one used throughout
this document: *the added cross-platform job has never executed, because this checkout has no git
remote.*

*Why it does not block.* §1.1 rule 1: deferred native testing does not block Linux Step 3. §1.1 rule 3
is the operative constraint instead — this workflow is **not evidence** and must never be cited as
though it had run. The nine-file selection the job runs was verified to pass on this Linux host (477
passed, 3 skipped), which establishes that the file list is self-consistent and nothing more.

*Remaining risk.* Step 2's "all three platforms" exit gate and §8 gate 27 are **open**, indefinitely,
by design. Ten cases in `tests/test_ci_workflow.py` pin the workflow file's *shape* — including that
the Windows leg names the two real Win32 node ids and fails if they skip — which is the right
mitigation for an unrunnable job, but it is a test of the YAML, not of Windows.

---

**F-036 · restates F-016 and F-017.** Filed by the completeness agent against `pyproject.toml:46`,
arguing correctly that Step 5a's "smoke-test packaging" clause is unsatisfiable on **both** sides: no
Electron packaging exists to test, and the Python wheel ships no GUI for a packaged shell to load. Its
concrete package-data prescription is the one this report endorses — add
`"leafmachine3.server" = ["ui/**/*"]` and `"leafmachine3.postprocessing" = ["stl_webview/**/*"]`,
narrow `packages.find` to `["leafmachine3", "leafmachine3.*"]` so `leafmachine3_backup` stops shipping,
and add the missing `lm3` console script. Its claim that `app/package.json` carries no packaging
dependency is **stale** (contradiction C2).

---

**F-037 · `leafmachine3/machine3.py:186` · low · confirmed · Step 3 · does not block**

*Defect.* `machine3`'s CLI makes `--config` **required**, so §3.1's row-1 chain (`LM3_SETTINGS` →
workspace pointer → deployment file → dev checkout → seed) is unreachable from the primary execution
entry point. `machine3.py` imports `leafmachine3.core.paths` nowhere.

*Evidence (re-read at snapshot, after the 10:48:17 edit to this file).*
```
machine3.py:186   parser.add_argument("--config", required=True, help="path to LM3_settings.yaml")
machine3.py:58    cfg = Config.load(cfg_path, overrides=_cli_overrides(input_dir, output_dir, restart))
```
`machine3.py` is **not** on §4 Step 1's replace list, so this is not a Step 1 exit-gate failure — but
§1 invariant 4 ("every execution entry point reaches the same lease code") and §9's CLI-then-GUI
agreement make it a Step 3 concern.

*Remaining risk.* The fix has a trap worth stating: route `args.config` through
`paths.settings_path(args.config)` so an omitted `--config` runs row 1 and an explicitly named missing
file is row 1's hard error — the shape `lm3-setup` already adopted (`hardware_setup.py:1005`,
`default=None`). **Do not drop `required=True` without that resolver call**, or the CLI gains a
CWD-shaped default, which is the bug §3.1 exists to kill.

---


### Ownership reassignments made at closeout

| Finding | Was | Now | Why |
|---|---|---|---|
| **F-006** — `resolve_port()` has no callers; `serve()` hardcodes 8765, so gate 41 cannot fire | unassigned (gate 41 had no §4 bullet) | **Step 5b, server startup** | The port rule is a *startup* decision, and 5b already owns `/healthz`, the deployment key and the connection descriptor — the other three places the port is asserted. Fixing it anywhere else would split one decision across two steps. Deliberately **not** implemented at closeout: the expected behavior on an already-bound port is the user's call. |
| **F-010** — a packaged app's `ROOT` resolves to `<install>/resources`, so the default `LM3_PYTHON` does not exist | unverified, owner unlisted | **Step 5b**, **confirmed** | Phase 7 measured it inside a running packaged AppImage (`CWD=/tmp/.mount_.../resources`). 5b owns server discovery, ownership and attachment, which is exactly the decision this is: a packaged app can **attach** to a running server but cannot **spawn** one. |
| **F-012** — `STATE_TRANSITIONS` is dead code; nothing refuses an illegal transition | Step 3 (unordered) | **Step 3, first task** | Step 3 is what starts *publishing* states (`starting` → `running` → terminal) from `machine3()`, the batch and the handshake. Enforcement has to exist before the first writer, or the record builders get written against an unenforced state machine and every later caller inherits the gap. |

## 6. Verification provenance — why some findings are `unverified`

The finding pipeline for this workflow had two stages. Seven auditors produced 55 raw findings; a
merge agent folded them into 40 reported findings and performed an **adversarial verify pass** over
them — re-running `tests/test_api_contract.py` to reproduce two blocking regressions, reading
`lease.py:270-310`, `records.py:305-330`/`:1620-1670`, `paths.py:185-245`/`:400-430`/`:955-1000`/
`:1210-1235`, and `grant.py:440-470` line by line, executing a set-difference of `lease.__all__` and
`config_io.__all__` against `runtime.__all__`, and grep-confirming eight further claims.

**32 of those 40 went through that pass. 8 did not.** The merge agent stated its own gaps: it did not
re-run the full suite, did not verify the Windows-only claims against a real Windows host, did not
independently fetch Electron's breaking-changes document, did not reproduce the timing-dependent
measurements (it confirmed the code paths instead), and did not inspect `node_modules` or run any npm
command.

The three **gate** agents whose 37 findings this report covers ran later, against a tree in which the
post-audit fixes had already landed. Rather than inherit that accounting, this pass re-verified every
one of the 37 first-hand at the 10:48–10:51 snapshot — reading the cited code, re-deriving the line
numbers, and correcting four citations (C2, C3, C4, C6). **Twenty-nine are marked `confirmed` on that
basis.** None is marked `refuted` outright; three have a refuted *premise* with surviving substance
(F-001's route, F-011's "no step owns gate 13", F-016's "no packaging dependency"), and one is
downgraded to informational because R7 falsifies its framing (F-003).

**The eight `unverified` findings, and what evidence would settle each:**

| ID | Why it cannot be confirmed from here | Evidence that would settle it |
|---|---|---|
| F-010 | The failure mode is an installed, non-checkout wheel launched under Electron. No wheel was built; no packaged install was launched. | Build a wheel, install into a throwaway venv, launch `app/main.js` against it, and observe whether `POST /v1/setup` returns a precise "no resolvable config: `<path>`" error or an `AttributeError`. |
| F-014 (off-Linux half) | `process_start_time()` returning `0.0` matters only where the `/proc` fallback is absent. Linux is confirmed; Windows and macOS have never run it. | Run `tests/test_runtime_records.py` on a real Windows and a real Mac with `psutil` absent, and assert the `can_stop` five-way match fails rather than succeeds on two unknown creation times. |
| F-016 | Rests on npm state and on packaging behavior; no npm command was run by this pass, and there is nothing to package. | `npm ci` in `app/`, `npm ls electron` asserting the exact version, `node_modules/electron/dist/version`, with `ELECTRON_RUN_AS_NODE` and `ELECTRON_NO_ATTACH_CONSOLE` scrubbed — plus an actual packaging attempt once a build configuration exists. |
| F-017 | Wheel contents were read from `pyproject.toml`, never from a built wheel. | Build the wheel into a tmp dir, install into a throwaway venv, and from an unrelated CWD assert `calibration_images_dir()` resolves into the package, `settings_meta.json` is present, the UI mount point exists, and `leafmachine3_backup` is absent. |
| F-022 (Windows half) | `STARTUPINFOEX` allowlist exhaustiveness and second-`startupinfo=` replacement are Win32 kernel semantics. The 29 adapter tests execute against `FakeWin32` on Linux. | A real Windows run of a child that receives **both** a valid lease reference and a working status pipe, plus a negative test that the lease is not silently absent. |
| F-023 | There is no JavaScript key derivation in the repository to compare against. | Ship `app/deployment.js` plus a Node test that recomputes slug and canonical for all 23 vectors in `tests/golden/deployment_key_vectors.json`, and run it in CI. |
| F-024 (macOS half) | `filesystem_type()` returning `None` on Darwin is confirmed by reading; that a real Mac with a network-mounted home is then accepted has never been executed. | A real macOS run against an AFP/SMB/NFS-backed home, asserting `check_runtime_filesystem` refuses without `LM3_ALLOW_NETWORK_RUNTIME`. |
| F-035 (the job's correctness) | The no-remote fact is confirmed; whether the job passes is not, and cannot be from here. | Push the branch to the upstream named by `pyproject`'s `Homepage` and obtain a green `platform-primitives / windows-latest` and `/ macos-latest` run. |

Per §1.1 rule 2, none of these may be reported as "passed" at any point. Windows and macOS status
remains **"implemented but not yet natively validated"**.

---

## 7. Findings that must NOT be implemented until they carry exact evidence

Everything in this section is **explicitly blocked from implementation**. Acting on an
under-specified finding produces a change nobody can review and a test nobody can write, and in this
codebase the likely outcome is a second incompatible rule beside the first — which is precisely how
F-007 and F-008 came to exist.

**7.1 Anything phrased only as a "port issue."** Several intermediate auditor notes reduce to the
word "port" with no failing behavior attached. `resolve_port()`, `LM3_PORT`, the Electron `PORT`
constant (`app/main.js:21`), `serve(port=8765)` and gate 41 are five different things, and a
finding that does not say which one it means is not actionable.

*Required before any port change is implemented:* (a) the exact entry point and file:line;
(b) the environment — is `LM3_DEPLOYMENT_ID` set, and to what; (c) the observed behavior, verbatim,
including the exit code and the error text; (d) the expected behavior quoted from §2.1, §3.1 or the
relevant §8 gate; (e) whether the report is about **binding** a port, **discovering** an occupant, or
**refusing to start**. The one port item in this report that meets that bar is **F-006**, and it is
stated narrowly: `resolve_port()` has zero callers, so gate 41 cannot fire. Nothing broader than that
is authorized.

**7.2 "The reaper should know about deployments" (beyond F-020's exact scope).** F-020 is actionable
only as far as its evidence goes: the named test is byte-unchanged and `executor.py` contains no
deployment token. The *design* — how a reaper learns another deployment's identity — is unspecified,
and it has a hard prerequisite nobody has written down (the deployment id must reach the worker's
environment or command line first). Do not implement a heuristic. Required first: a stated mechanism
for worker-side deployment attribution, and a test that a worker of **another live deployment**
survives.

**7.3 "`/healthz` exposes too much."** F-019 is actionable on the two mechanical halves — memoize
`machine_key()`, and stop recomputing the path map per request. The **policy** half ("move it behind
the token" vs "reduce to basenames") is a Step 5b decision that changes a client contract §2.11
depends on. Required first: the decided `/healthz` body, written into §4 Step 5b. Do not narrow the
body on a reviewer's judgment; `deployment_key` in particular must stay, because §2.11 requires
clients to verify it.

**7.4 "Delete the stale hardware profile."** R7 settles the facts and explicitly leaves both files in
place. No agent may delete, move, or rewrite anything under `~/.config/lm3`. Required first: the
user's decision.

**7.5 "Add packaging."** F-016/F-017/F-036 establish that Step 5a's exit condition is unsatisfiable
as written. That is a **plan amendment**, not a coding task. Required first: a decision recorded in
§4 Step 5a on whether packaging is in scope for this workflow at all. The Python `package-data` fix
(F-017) is separable, small, and may proceed on its own evidence.

**7.6 Any finding whose only support is "the docstring says a caller must."** Three findings in this
set rest on a docstring-stated contract with no enforcement (F-014's unknown-creation-time rule,
F-022's merge requirement, F-025's expiry delegation). Each is real, and each is implementable only
once the *enforcing* API is named. Required before implementation: the function signature that will
enforce it, so the contract lives in code rather than in prose.

---

## 8. Defects observed during this closeout that are not among the 37

Clearly labeled: these were found by this pass while reconciling, not by any gate agent.

**N-1 · `examples/*.yaml` semantic break, sequencing hazard · observed open 10:48:42, closed in-tree 10:48:52 · unverified**

When the §3.5 rule-1 change landed in `config.py`/`dirs.py`/`paths.py` at 10:48:01, every
`examples/*.yaml` still carried repository-root-relative values — five with `dir: examples_out`, all
twelve with `models_dir: models/ruler_classifier`. Under the newly-correct rule 1 those resolve
against `examples/`, i.e. `examples/examples_out` and `examples/models/ruler_classifier`, **neither of
which exists** (`ls -d examples/models` → No such file or directory; `models` and `examples/images`
both exist at the repository root). For roughly 51 seconds the twelve example configs pointed at
nothing.

The migration landed at 10:48:52 — all twelve files now show ` M` in `git status`, `dirs: [images]`,
`dir: ../examples_out`, `models_dir: ../models/ruler_classifier`, and
`grep "models_dir: models/\|dir: examples_out\|examples/images" examples/*.yaml` returns nothing. The
covering test `tests/test_relative_path_contract.py::test_migrated_example_configs_still_point_at_their_repository_files`
was created at 10:50:26.

*Why it is recorded anyway.* The window is closed here, but the **ordering hazard is real for anyone
who pulls a partial state**: the semantic change and the data migration are two separate edits, and
plan amendment L3 (`plan:167`) requires them to ship together. They must be one commit.

**N-2 · plan line citations across the finding set are stale by roughly 110 lines**

The consensus plan grew from 2128 to 2269 lines in revision 13 (`git diff --stat 5ddfee4` → 141
insertions, 1 deletion). Every finding that cites a plan line number — "§8 gate 44 (line 1700)",
"§4 Step 1 exit gate (line 1417)", "§2.11 (lines 921-926)" — now points at the wrong line. Section and
gate **numbers** are stable and correct; line numbers are not. Current anchors, re-derived:
§1.1 at `plan:233`, §3.5 at `plan:1411`, §4 Step 1 at `:1503`, Step 2 at `:1539`, Step 3 at `:1557`
(entry prerequisite `:1559-1563`), Step 4 at `:1595`, Step 5 at `:1617` (gate 13 bullet `:1626-1631`),
Step 6 at `:1640`, Step 7 at `:1650`, Step 8 at `:1660`, §8 at `:1797` (60 numbered gates), gate 13 at
`:1813`, gate 41 at `:1841`, gate 44 at `:1844`, gate 45 at `:1845`, gate 46 at `:1846`, gate 57 at
`:1857`, Appendix A's cfg-`None` row at `:1909`.

**N-3 · `hardware_setup.py` has been renumbered twice; three documents now cite three different wrong lines**

See contradiction C4 and F-028. Cite by symbol.

---

## 9. Step 3 entry criteria

State as of the Phase 8 verification run (§ above), which post-dates the 10:48–10:57 snapshot the
finding table was written against. Criteria 2, 3, 4, 7 and 8 were "pending verification" at that
snapshot and have since been **measured**; criteria 5 and 6 were "NOT MET" and were **fixed** in
Phase 5 of this pass. Each row below cites the measurement, not an intention.

| # | Criterion | State | Basis |
|---|---|---|---|
| 1 | The relative-path contract is written into the consensus plan | **met** | §3.5 at `plan:1411-1497`: six exhaustive rules, the `auto` default decision, the example-migration requirement, and the testable `build_dirs == resolve_run_paths` invariant. Amendments L1–L3 at `plan:165-167`. Step 3 entry prerequisite at `plan:1559-1563`. |
| 2 | `build_dirs()` and the early runtime resolver produce identical paths | **met** | `dirs.py` resolves `output.dir` AND `tmp_dir` through `paths.resolve_project_output_dir()` — the same function `config_io.resolve_run_paths()` calls, so they agree by construction rather than by coincidence. `test_build_dirs_and_the_early_resolver_agree` is parametrized over relative, nested-relative and absolute `output.dir` and asserts **all three** halves: `root == run_dir`, `db_path == active_db_path`, and `logs == log_path.parent`. 27/27 in `test_relative_path_contract.py`. |
| 3 | Relative config behavior is independent of launcher CWD | **met** | Measured, not argued: `test_the_same_settings_file_resolves_identically_from_two_different_cwds` loads one settings file from three different CWDs (two temp dirs and the repo root) and asserts a single distinct result. `test_resolution_does_not_depend_on_the_cwd_once_the_config_is_loaded` loads once, chdirs away, and re-resolves five paths to identical values. The rule-6 raise is pinned by `test_a_relative_path_with_no_settings_file_refuses_rather_than_guessing`. |
| 4 | Example config semantics preserved or intentionally migrated | **met** | All twelve `examples/*.yaml` migrated to settings-relative values. Preservation was **proven mechanically at migration time**: every path-valued field was resolved under the old rule (repo root) and the new rule (settings-file dir) and compared — 0 mismatches, 0 missing targets. `test_migrated_example_configs_still_point_at_their_repository_files` is parametrized per file (12 cases) and runs from a non-repo CWD, asserting each model path lands under `<repo>/models` and each migrated input is `<repo>/examples/images`. |
| 5 | Path/status/health reads perform no filesystem migration | **met — fixed in Phase 5** | `hardware_profile_path()` now defaults `adopt_legacy=False` and is documented as pure; the adopt moved to `paths.migrate_legacy_hardware_profile()`, called **once** from `create_app` and from `hardware_setup`'s two controlled entry points. `app.canonical_hardware_path()` no longer resolves the settings file at all and is memoized. Evidence: `test_repeated_read_requests_create_and_modify_nothing` fires 100 read requests across 5 endpoints and asserts the entire sandbox tree is byte-identical by size+mtime; `test_reads_never_adopt_a_legacy_hardware_profile` plants a legacy profile and proves 50 reads do not copy it. |
| 6 | The machine/NVML probe is not repeated in polling loops | **met — fixed in Phase 5** | `machine_key()` memoizes the no-argument case in `_MACHINE_KEY_CACHE`; an explicit identity or any detect kwarg still bypasses it, and `reset_machine_key_cache()` exists for tests. `test_the_machine_probe_runs_once_not_once_per_poll` counts calls to `detect_machine_identity` across 60 read requests and asserts **≤ 1**. The memo is not allowed to hide a real change: the hardware-path cache is keyed on deployment key + config root + `LM3_HARDWARE`, and `test_the_hardware_path_memo_still_notices_a_deliberate_change` proves an `LM3_HARDWARE` change is seen immediately. |
| 7 | Focused Linux tests pass | **met** | 601 passed, 3 skipped across the 17 focused files (§ Phase 8 step 2), including every file this criterion originally named. |
| 8 | The full suite introduces no new failing test node IDs against the persisted baseline | **met** | `11 failed, 966 passed, 6 skipped`. Classified by **set difference on node ids** against `.step2_evidence/baseline_5ddfee4.txt`, not by counting: new failing node IDs empty, newly-passing empty, same 11 node IDs. One new failure did appear on the first run and was fixed; it is recorded in Phase 8 rather than quietly dropped. |
| 9 | The global-greening run and the real user config are untouched | **met** | R6: no `machine3`/`leafmachine3`/`global_greening` process and no `nvidia-smi` compute app at any point. R7: both `~/.config/lm3` profiles byte-identical, left in place deliberately. R5: `LM3_settings_global_greening.yaml:5,79` are absolute, so the §3.5 change cannot move that job — and `test_the_absolute_global_greening_config_is_unaffected` exists to pin it (unrun). This pass wrote exactly one file, `UNIFIED_RUNTIME_STEP2_CLOSEOUT.md`. |
| 10 | This report complete | **met** | 37 findings, IDs `F-001`–`F-037`, all seven reconciliations applied and visible, duplicates merged (§4.1), six contradictions resolved by first-hand code reading (§4.2), eight `unverified` findings named with the evidence each needs (§6), non-actionable findings blocked (§7). |

**Blocking summary.** Exactly one finding was ever assessed as blocking Step 3 on Linux — **F-007**
(with F-029), the `build_dirs`/`config_io` divergence — and the plan now agrees, having made it a
written Step 3 entry prerequisite. **It is fixed and verified**: the invariant is asserted over
relative, nested-relative and absolute output dirs, from three different CWDs, against the real
`Config.load()` and the real `build_dirs()`.

Criteria 5 and 6, which this report first recorded as NOT MET, were fixed in Phase 5 of the same
pass and are now measured. F-011 (the raw bearer token at WARNING) is likewise fixed: gate 13 is
assigned to Step 5b in the plan and its behavior is pinned by `tests/test_token_hygiene.py`,
including a static backstop that fails if a future edit passes the token to a log call.

**All ten criteria are met.** Step 3 may open on Linux. Windows and macOS remain
*implemented but not yet natively validated* per §1.1 and are not gates on this step.

---

## Phase 8 — verification evidence

Run by the orchestrating session on 2026-08-29, against the working tree carrying the Phase 3–6
work described in §10 below. **`HEAD` remains at `5ddfee4`** — nothing in this closeout was
committed; every change described here is uncommitted working-tree state. Interpreter `<conda-prefix>/bin/python` (3.11.5). Every number here was measured,
not reported by an agent.

### Order of execution, as prescribed

**1. Focused new path-integration tests**

```
$ python -m pytest tests/test_relative_path_contract.py -q -p no:recording
27 passed in 0.46s
```

**2. Existing path / setup / runtime-config / lease / settings-unification tests**

```
$ python -m pytest -q -p no:recording \
    tests/test_relative_path_contract.py tests/test_read_paths_are_pure.py \
    tests/test_token_hygiene.py tests/test_paths.py tests/test_setup_paths.py \
    tests/test_settings_path_unification.py tests/test_runtime_config_io.py \
    tests/test_runtime_records.py tests/test_runtime_grant.py tests/test_runtime_lease.py \
    tests/test_runtime_lease_windows.py tests/test_runtime_isolation.py \
    tests/test_runtime_init.py tests/test_api_contract.py tests/test_dirs.py \
    tests/test_config.py tests/test_postprocessing_config.py
601 passed, 3 skipped in 17.33s
```

**3. Ruff on changed files, no `|| true`**

```
$ ruff check leafmachine3/core/paths.py leafmachine3/core/dirs.py leafmachine3/machine3.py \
             leafmachine3/server/app.py tests/test_relative_path_contract.py \
             tests/test_read_paths_are_pure.py tests/test_token_hygiene.py
All checks passed!
```

Three findings remain in two files I edited and are **pre-existing**, proven by running ruff against
the same files at `5ddfee4` through `git show` and comparing counts:

| File | findings at `5ddfee4` | findings now |
|---|---|---|
| `leafmachine3/core/config.py` | 1 (`F401 typing.Iterable`) | 1 — same |
| `leafmachine3/setup/hardware_setup.py` | 2 (`F401 shutil`, `F841 ctx_spawn`) | 2 — same |

**4. Linux runtime primitive suite**

```
$ python -m pytest -q -p no:recording tests/test_runtime_lease.py \
    tests/test_runtime_lease_windows.py tests/test_runtime_records.py \
    tests/test_runtime_grant.py tests/test_runtime_config_io.py tests/test_runtime_init.py
293 passed, 2 skipped in 8.03s
```

The two skips are the only genuinely-off-Windows assertions, named explicitly with `-rs`:
`test_runtime_lease_windows.py:741` ("a real STARTUPINFO only exists on Windows") and `:747`
("requires the real Win32 object manager"). **29 Windows-adapter tests execute on this Linux host**
against the injectable fake. Per §1.1 rule 2 that is a test of our *model* of the Win32 object
manager and is recorded as **implemented but not yet natively validated**, never as passed.

### Pre-full-suite isolation verification

Probed from inside a pytest session and recorded at `.step2_evidence/isolation_probe.json`.

| Check | Result |
|---|---|
| `LM3_RUNTIME_DIR` | `/tmp/lm3-pytest-<session>/runtime` — sandboxed |
| `LM3_DEPLOYMENT_ID` | `pytest-lm3-pytest-<session>-main` — per session, per xdist worker |
| `LM3_SETTINGS`, `LM3_SETTINGS_PATH`, `LM3_HARDWARE_SETTINGS`, `LM3_RUNS_ROOTS` | cleared |
| `LM3_HARDWARE`, `LM3_POSTPROCESS_SETTINGS`, `LM3_SERVER_JOBS` | sandboxed paths |
| `LM3_PORT` | `18765` — sandboxed |
| `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME`, `XDG_RUNTIME_DIR` | sandboxed |
| `XDG_DATA_HOME` | **was NOT isolated; fixed during this pass** — see below |
| `MPLCONFIGDIR` | real `~/.cache/matplotlib`, the deliberate documented read-only performance carve-out |
| **Subprocess inheritance** | a child `python -c` reports the same sandboxed `LM3_RUNTIME_DIR`, `LM3_DEPLOYMENT_ID`, `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME` — isolation is inherited, not merely set |
| **Production lease unreachable** | production key `default-<hash8>` → `<user-cache>/lm3/runtime/default-<hash8>`; test key `pytest-<session>-<hash8>` → `/tmp/lm3-pytest-…/runtime/…`. Different keys, different directories, and **the production runtime directory does not exist** — tests cannot see or acquire the production deployment lease. |
| GPU calibration / real inference | none run |

**A gap this probe found, in work done earlier in this same pass.** Adding
`paths.user_data_dir()` for the §3.5 `output.dir: auto` default introduced a dependency on
`XDG_DATA_HOME`, which `tests/conftest.py` did not isolate — it pointed at the developer's real
`<user-data> (the real XDG_DATA_HOME)`. On this host `dev_checkout_root()` is non-null so `auto`
resolves to `<checkout>/runs` and no test reached it, but a test exercising the installed branch
would have written RUN OUTPUT into the developer's real data directory. `XDG_DATA_HOME` is now
sandboxed and added to the guard's literal floor (`tests/conftest.py`).

### Full suite

```
$ python -m pytest tests/ -q -p no:recording
11 failed, 966 passed, 6 skipped in 56.48s
```

Classified against the **persisted** baseline `.step2_evidence/baseline_5ddfee4.txt`
(11 failed / 321 passed / 3 skipped at `5ddfee4`) by set difference on node ids, not by counting:

```
$ comm -23 now_failing.txt baseline_failing.txt     # NEW failures
(empty)
$ comm -13 now_failing.txt baseline_failing.txt     # FIXED since baseline
(empty)
counts: now=11 baseline=11
```

**Zero new failing test node IDs. Zero node IDs newly passing. +645 passing tests.**
Stated as node IDs deliberately: a count comparison would hide a swap in which one test
starts failing as another starts passing. The 11 failing node ids are exactly the
persisted baseline set:

| Node id | Pre-existing cause |
|---|---|
| `tests/test_bilateral_symmetry.py` ×5 | **NumPy**: `AttributeError: module 'numpy' has no attribute 'trapezoid'`. `np.trapezoid` arrived in NumPy 2.0; this interpreter has **1.26.4**. |
| `tests/test_pipeline_mock.py` ×3 | **`ect`**: `ModuleNotFoundError: No module named 'ect'` |
| `tests/test_working_frame.py::test_reporter_outputs_are_working_sized_and_survive_a_deleted_original` | **`ect`**, same cause |
| `tests/test_settings_ui.py` ×2 | **Settings UI**: the pre-existing `report.data` rail-group / `settings_meta.json` assertions |

**5 NumPy · 4 `ect` · 2 Settings UI.** An earlier revision of this report, and the orchestrating
session, stated "8 of 11 trace to `ect`". That was wrong: the five `test_bilateral_symmetry`
failures are a NumPy version gap, not a missing module.

**And "environment gaps, not code defects" is itself an overstatement — withdrawn.** These are
**three categories of pre-existing failures in this environment**. Every one fails identically at
`5ddfee4`, so none is a consequence of this work; but each has an unresolved question behind it and
none was diagnosed to a root cause or fixed here:

- **NumPy**: `pyproject.toml` declares `numpy>=1.24` while production code calls `np.trapezoid`,
  which needs 2.0. A clean install anywhere in the declared range reproduces the failure — that is a
  **packaging defect**, not a property of this machine.
- **`ect`**: the module is **enabled by default**, yet `ect` appears in no declared runtime
  dependency and no optional extra. A default-on stage with an undeclared import is a real gap.
- **Settings UI**: `settings_meta.json` documents `report.data.*` settings that no longer exist.
  That reads as a **code/test contract discrepancy**, not an environment property at all.

Recorded so a later reader does not inherit the tidier, wrong version.

The 3 new skips over baseline are the 2 real-Win32 guards above plus one Windows-registry guard in
`tests/test_paths.py:288`.

### The real user configuration, snapshotted around the full suite

```
$ diff -u .step2_evidence/user_config_pre_fullsuite.txt \
          .step2_evidence/user_config_post_fullsuite.txt
UNCHANGED
```

Covering `~/.config/lm3` (with md5 of both profiles), `~/.local/state/lm3`, `~/.cache/lm3`, and the
real `XDG_DATA_HOME` LM3 directory. Nothing under `~/.config/lm3` was created, modified, or deleted.
The two byte-identical legacy profiles (R7) are still in place, untouched, as instructed.

### One new failure was found and fixed during verification — recorded, not hidden

The first full-suite run was **12 failed**, not 11. The extra failure was
`tests/test_token_hygiene.py::test_a_generated_token_is_never_written_to_a_log`, written earlier in
this same pass. It passed in isolation and failed in the full suite.

The cause matters more than the failure. `leafmachine3/core/logging_setup.py:19,34` removes the root
handlers and sets `root.propagate = False`, so once any earlier test has started a pipeline,
`caplog` captures **nothing at all** — and the security assertion `assert token not in captured`
against an empty capture passes **vacuously**. The test was therefore not merely order-dependent; in
the passing direction it was proving nothing. Both log-capturing tests now attach a private handler
to the `leafmachine3` logger and force propagation for their duration, and both assert
`assert captured, "captured nothing -- the assertion below would pass vacuously"` before making any
security claim.


---

## 10. What this closeout pass changed

Phases 2–6 of the closeout, all on Linux, all verified in §Phase 8.

### Phase 2 — the plan (revision 12 → 13)

`UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_CONSENSUS.md`, 2,129 → 2,269 lines:

| Addition | Where |
|---|---|
| §1.1 platform qualification policy | `plan:233` |
| §3.5 the six-rule relative-path contract, the `auto` default decision, the example-migration requirement, the testable invariant | `plan:1411` |
| Revision 13 amendments L1–L5 | `plan:158` |
| Step 3 entry prerequisite | Step 3 heading |
| Gate 13 assigned to Step 5b | Step 5 bullet list |

### Phase 3 — one path-resolution seam

| File | Change |
|---|---|
| `leafmachine3/core/config.py` | `Config.resolve_path()` resolves against `source_path`, not `Path.cwd()`; raises under rule 6 rather than inventing a base; `normpath` so `..` collapses without following symlinks. Built-in `output.dir` default `runs` → `auto`. |
| `leafmachine3/core/paths.py` | `user_data_dir()`, `default_output_dir()`, `AUTO_OUTPUT_DIR`; `resolve_project_output_dir()` handles `auto` and normalizes. |
| `leafmachine3/core/dirs.py` | `build_dirs()` resolves `output.dir` **and** `tmp_dir` through `paths.resolve_project_output_dir()` — the same function the §2.7 resolver uses. |
| `leafmachine3/machine3.py` | `_cli_overrides()` absolutizes relative `--input`/`--output` against the caller's CWD **before** the merge, while the provenance still exists (rules 2–3). |
| `examples/*.yaml` ×12 | migrated to settings-relative values, effective locations proven unchanged. |

`Config.resolve_path()` is the seam because every path consumer already funnels through it:
`core/validate.py:29,34`, `core/ingest.py:164`, `inference/factory.py:46,64,146`,
`setup/hardware_setup.py:602,607,803`, `server/settings_api.py:577-585`, `config.py:540,543`.

#### `machine3.py` was modified deliberately, and that changes what the Step 2 gate can claim

Step 2's exit gate is "the primitives pass in isolation on all three platforms. **No production
entry point has changed.**" §Phase 8 verified that gate at the time and it held: `machine3.py` was
byte-identical to `5ddfee4`.

It is no longer. `_cli_overrides()` now absolutizes relative `--input`/`--output` against the
caller's CWD. This is intentional and is **not** the thing the gate was protecting against:

- The gate exists so that the **runtime primitives** (`core/runtime/**` — lease, records, grants)
  cannot leak into an execution path before Step 3 wires them in deliberately. That property is
  intact and re-verified: no production module imports `leafmachine3.core.runtime`, and the only
  tree-wide mentions outside `core/runtime/` and `tests/` are two comments.
- The `machine3.py` edit is **§3.5 path-contract work**, not runtime-v2 work. It imports nothing
  new; it normalizes two override strings to absolute paths before `Config.load()` merges them.
  Rules 2 and 3 require it to happen at that boundary, because after the deep-merge nothing can
  distinguish a CLI-supplied `output.dir` from a YAML-supplied one, and the two take different bases.
- It was explicitly authorized: "CLI/public-function overrides must be normalized before provenance
  is lost during configuration merging", alongside "do not wire runtime-v2 into `machine3()` yet".

So the gate should now be read in two halves: **"no runtime-v2 wiring in a production entry point"
holds, unchanged and verified**; "`machine3.py` is byte-identical" does not, by design. A reviewer
checking the gate mechanically with `git diff --stat 5ddfee4 -- leafmachine3/machine3.py` will see a
16-line diff and should read this paragraph rather than treat it as a violation.

### Phase 4 — `tests/test_relative_path_contract.py` (249 lines, 27 tests)

Covers all twelve required cases. Drives the real `Config.load()` and the real `build_dirs()`, from
a CWD that is never the settings directory.

### Phase 6b (review follow-up) — F-009 and an atomic adopt

`--config` on both standalone postprocessing CLIs now defaults to `None`, so the §3.1 row-3 resolver
actually runs; the loader's `None` branch had been dead code for them. Migration of a checkout-level
`postprocessing_settings.yaml` is `paths.migrate_legacy_postprocessing_settings()` — copy, never
move, located from `dev_checkout_root()` and never the CWD.

**The adopt was racy, and both adopts were.** `if not target.exists(): _write_atomic(target, ...)`
is check-then-act, and `_write_atomic` ends in `os.replace`, which overwrites unconditionally. A
second process — or a user save from the GUI — landing in that window would be silently overwritten.
This matters *now* rather than after Step 5b, because until the lease is wired in there is nothing
preventing a CLI and the server from adopting concurrently.

`paths._install_if_absent()` replaces it: content is written and fsynced to a temp sibling, then
published with `os.link`, which fails `FileExistsError` when the destination exists and is atomic on
POSIX and Windows NTFS. Where hardlinks are unsupported it **skips with a warning** rather than
writing the destination directly — `O_CREAT|O_EXCL` would settle the create decision race-free, but
the file becomes visible at creation, so a concurrent reader could parse truncated YAML as the
deployment's settings. Migration is best-effort and the caller degrades to packaged defaults, so
skipping loudly is strictly safer than publishing a partial file. **Both**
`migrate_legacy_postprocessing_settings()` and
`migrate_legacy_hardware_profile()` use it — the hardware adopt had the identical defect and fixing
only the reported one would have left a matched pair half-repaired.

Proven, not asserted: `test_concurrent_adopts_produce_exactly_one_winner_and_no_clobber` drives
**8 real spawned processes** through the adopt released simultaneously on a barrier — 8 exit-0,
exactly 1 winner, legacy content intact, no temp files left. A negative control confirmed the old
sequence clobbers a user save under the same interleaving while the new one refuses.

**One migration path, not two.** `hardware_profile_path()` had kept `adopt_legacy` /
`settings_file` parameters and an embedded copy of the old check-then-`_write_atomic` sequence. No
caller enabled it, but a disabled-by-default adopt still contradicted the function's stated purity
contract and left the identical race reachable through its public API. Both parameters and the whole
block are **removed**; migration is now reachable only through
`migrate_legacy_hardware_profile()`. Pinned by a test that asserts the signature exposes no adoption
parameter and that the function body contains no writer.

`tests/test_postprocessing_cli_agreement.py` also now **executes each CLI's real `main()`** with only
`run()` stubbed, from a CWD holding a decoy settings file, and asserts the settings dict `main()`
assembled came from the canonical file. Reading source text proves the default changed; it does not
prove the parser and `main()` reach the resolver.

### Phase 5 — reads are pure, probes are memoized

| File | Change |
|---|---|
| `leafmachine3/core/paths.py` | `hardware_profile_path()` pure (`adopt_legacy` now defaults `False`); adoption extracted to `migrate_legacy_hardware_profile()`; `machine_key()` memoized with `reset_machine_key_cache()`. |
| `leafmachine3/server/app.py` | `canonical_hardware_path()` pure + memoized on (deployment key, config root, `LM3_HARDWARE`); `reset_path_caches()`; `create_app` runs the migration exactly once. |
| `leafmachine3/setup/hardware_setup.py` | resolution pure; `migrate_legacy_profile()` called from the two controlled entry points. |
| `tests/conftest.py` | `XDG_DATA_HOME` isolated (see Phase 8). |
| `tests/test_read_paths_are_pure.py` | 154 lines, 5 tests. |

Three existing tests asserted the old "resolution adopts" contract and were updated **deliberately**
to assert the new one, each now also asserting that resolution writes nothing:
`test_paths.py::test_legacy_hardware_profile_is_COPIED_not_moved`,
`test_setup_paths.py::test_a_beside_the_settings_profile_is_copied_into_the_canonical_path`,
`test_settings_path_unification.py::test_a_profile_beside_the_settings_file_is_adopted_by_copy`.

### Phase 6 — gate 13

`_server_token()` never logs the raw secret; it logs where the descriptor will be and which variable
overrides it. `redact_token()` added for messages and diagnostics. The bootstrap document is served
`Cache-Control: no-store`; static assets keep `no-cache` revalidation.
`tests/test_token_hygiene.py` (148 lines, 6 tests) includes a static backstop that fails if a future
edit passes the token to a log call.

### Evidence: what is committed and what is not

Raw verification material lives at `.step2_evidence/` and is **git-ignored deliberately**. It
carries the machine key, GPU UUIDs, the node name, absolute home paths and full pytest output —
derived host identity that is worthless as a finding and undesirable in history.

What IS committed is the sanitized equivalent:

| Committed | Contents |
|---|---|
| `docs/verification/STEP2_BASELINE_NODE_IDS.txt` | the authoritative 11 pre-refactor failing node IDs, the exact command, and the three distinct causes |
| `docs/verification/STEP2_VERIFICATION_SUMMARY.md` | the six verification steps and their results, the ruff pre-existing table, the isolation checks, and the untouched-state result |

The committed node-ID list was diffed against the raw evidence and matches exactly. This document
is likewise sanitized: machine keys, deployment-key hashes, usernames and absolute local paths are
templated (`<machine-key-A>`, `<user-config>`, `<home>`).

Also **not** committed and deliberately left alone: the two pre-existing legacy hardware profiles
under `<user-config>/lm3/`. They are byte-identical to each other and outside the repository.

---

## 11. Phase 7 — Electron packaging (Linux built; Windows/macOS defined only)

Recorded here because §1.1 rule 3 forbids treating a definition as evidence. Detail lives in
`app/ELECTRON_PACKAGING.md`; the human test plan is `app/NATIVE_ACCEPTANCE_CHECKLIST.md`.

**Pin unchanged and verified:** `app/package.json` still carries the bare string `43.4.1`, and
`requestSingleInstanceLock` is still absent from `app/main.js` (0 occurrences) — Step 5b was not
started. `electron-builder` added at exact `26.15.3`.

### Electron status, stated precisely

Four separate claims, because they are routinely collapsed into one and they are not the same:

| Claim | Status |
|---|---|
| The Linux package **works for attachment** | **yes, measured.** The shipped AppImage was executed under `Xvfb` against a stand-in server: it attached, loaded the UI with its token, and tore down cleanly on `SIGTERM` with no orphan and no held port. |
| The Linux package **can spawn an LM3 server** | **no.** In a packaged app `ROOT = path.resolve(__dirname, "..")` is `<install>/resources`, so the default `LM3_PYTHON` does not exist. It can attach to a running server; it cannot start one **unless `LM3_PYTHON` is set**. (F-010, Step 5b.) |
| Windows and macOS | **implemented but not yet natively validated.** Config schema-validated with a negative control, and `--dir` builds produce a `.app` and a `.exe` on Linux. Nothing was built, run, signed, or notarized on either platform; no Windows machine and no Mac was involved. |
| `requestSingleInstanceLock()` | **not present, and that is correct.** It remains **Step 5b** work. §2.11 requires the Electron upgrade to land first precisely because adding that call on the old version would have introduced CVE-2026-34776. Verified absent from `app/main.js` (0 occurrences). |

### Executed on Linux

`npm run dist:linux` (2m17s, zero warnings) produced AppImage 121.8 MB, tar.gz 115.3 MB, deb
95.1 MB, from `electron=43.4.1 chrome=150.0.7871.224 node=24.18.1`. `dist/` (645 MB) was deleted
after the sha256s were recorded; `/app/dist/` and `/app/.cache/` are now git-ignored.

The shipped AppImage was **smoke-tested by execution** under `Xvfb`, against a stand-in server: it
attached, loaded the UI with its token, and on `SIGTERM` completed an orderly teardown leaving no
orphan process and no held port.

Two real packaging defects were found and fixed rather than worked around: the deb target failed
outright on a missing `homepage`, and `desktopName` was unset. A `desktop.entry` block was found to
be dead config — those keys are owned by `productName`/`description`/`linux.category` — and was cut.

### NOT executed — Windows and macOS

| What was done | What it proves | What it does not |
|---|---|---|
| Schema validation against electron-builder's own `scheme.json`, with a **negative control** that correctly rejects misspelled keys | every win/mac/nsis/portable/dmg key is real, correctly spelled and type-correct | nothing about behavior |
| Cross-platform `--dir` build on Linux | the config is consumable; a `.app` with a readable `Info.plist` and a `LeafMachine3.exe` with a rewritten version resource and 7-size `.ico` are produced | nothing about execution |
| Real installer targets attempted | they failed on **host toolchains** (`wine ENOENT`, `sips ENOENT`), not on configuration | nothing about the installers |

**Nothing was built, run, signed, or notarized on Windows or macOS. No Windows machine and no Mac
was involved.** Status stays *implemented but not yet natively validated* (§1.1 rule 2).

### One measured finding that matters for shipping

In a packaged app `main.js`'s `ROOT = path.resolve(__dirname, "..")` resolves to
`<install>/resources`, so the default `LM3_PYTHON` (`ROOT/.venv_LM3/bin/python`) does not exist —
measured, `CWD=/tmp/.mount_.../resources`. **A packaged app can attach to a running LM3 server but
cannot spawn one unless `LM3_PYTHON` is set.** `main.js` was outside that pass's ownership, so this
is documented, not fixed. Related: its failure path uses a modal `showErrorBox`, so a failed
packaged launch never exits on a headless host.

The smoke test's clean teardown demonstrates the **current** kill-the-attached-server policy — which
is precisely what Step 5b's exit gate deletes. The checklist marks that section failing-by-design
today and tags every row **NOW** / **5b** / **PKG** so nobody tests behavior that does not exist yet.

### A `.gitignore` trap found in passing

The repo's pre-existing Python rule `build/` is unanchored and was silently swallowing `app/build/`
(confirmed with `git check-ignore -v`) — the icons and macOS entitlements would never have been
committed. A `!/app/build/` negation fixes it; re-verified after the change.
