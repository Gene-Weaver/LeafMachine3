# Step 1/2 closeout — verification summary

Sanitized. Machine keys, GPU UUIDs, node name, usernames and absolute local paths are deliberately
absent or templated; nothing below depends on their literal values. The raw material this summarizes
(`.step2_evidence/`) is **git-ignored on purpose** — it carries the machine key, GPU UUIDs, the node
name, absolute home paths and full pytest output.

Measured 2026-08-29 by the orchestrating session under CPython 3.11.5. **`HEAD` remained at
`5ddfee4` throughout**; every change verified here was uncommitted working-tree state.

## Command

```
python -m pytest tests/ -q -p no:recording
```

`-p no:recording` is mandatory in this environment: pytest-recording and pytest-vcr conflict and
pytest hard-errors at startup without it.

## Results, in the order run

| # | Scope | Result |
|---|---|---|
| 1 | Focused postprocessing + path tests | **221 passed, 1 skipped** |
| 1b | CI gate negative controls (`tests/test_ci_gates.py`) | **36 passed** |
| 1c | Python 3.10 gate/version tests | **38 passed, 2 skipped** |
| 2 | Focused Step 1/2 suite (23 files) | **652 passed, 3 skipped** |
| 3 | Lint ratchet via `tools/ci/check_lint_ratchet.py`, no `\|\| true` | **79 findings, none new** (baseline 79) |
| 4 | Full Linux suite | **11 failed, 1007 passed, 6 skipped** — 1018 executed, floor 980 |
| 5 | New failing node IDs vs baseline | **none** |
| 6 | Real user config / production runtime | **unchanged** |

## Failing node IDs

Exactly the 11 in `STEP2_BASELINE_NODE_IDS.txt`, classified by **set difference on node IDs**, not
by count — a count comparison would hide a swap in which one test starts failing as another starts
passing:

```
comm -23 now_failing baseline_failing   ->  (empty)   new failing node IDs
comm -13 now_failing baseline_failing   ->  (empty)   newly passing node IDs
now=11  baseline=11
```

**Zero new failing test node IDs. +686 passing tests** over the 321-passing baseline.
Gated by `tools/ci/check_test_baseline.py`, the same checker CI runs.

Three categories of pre-existing failures **in this environment**. They are legitimately
pre-existing — every one fails identically at `5ddfee4` — but "environmental, not code defects" was
an overstatement, and each category has an unresolved question behind it:

- **5 × NumPy.** `AttributeError: module 'numpy' has no attribute 'trapezoid'`. `np.trapezoid`
  arrived in NumPy 2.0 and this interpreter has 1.26.4 — but `pyproject.toml` declares
  `numpy>=1.24` while production code calls `np.trapezoid`. The floor and the code disagree, so a
  clean install inside the declared range reproduces this. That is a packaging defect, not just a
  local gap.
- **4 × `ect`.** `ModuleNotFoundError: No module named 'ect'` — and `ect` appears in no declared
  runtime dependency or optional extra, while the ECT module is **enabled by default**. A default-on
  stage whose import is undeclared is a real gap, whatever the local interpreter has installed.
- **2 × Settings UI.** `report.data` rail-group and `settings_meta.json` assertions. These look like
  a genuine code/test contract discrepancy — metadata describing settings that no longer exist —
  rather than anything about this machine.

None of them is caused by, or affected by, the Step 1/2 work; all three predate it and are recorded
here so they are not mistaken for consequences of it. None has been diagnosed to a root cause or
fixed by this pass.

## Ruff

All 4 remaining findings are pre-existing, proven by running ruff against the same files at
`5ddfee4` via `git show` and comparing counts:

| File | at `5ddfee4` | now |
|---|---|---|
| `leafmachine3/core/config.py` | 1 | 1 |
| `leafmachine3/setup/hardware_setup.py` | 2 | 2 |
| `leafmachine3/postprocessing/generate_leaf_collage.py` | 1 | 1 |

Every file authored by this work is ruff-clean.

## Test isolation

Verified from inside a pytest session rather than assumed:

- `LM3_RUNTIME_DIR`, `LM3_DEPLOYMENT_ID`, `LM3_HARDWARE`, `LM3_POSTPROCESS_SETTINGS`,
  `LM3_SERVER_JOBS`, `LM3_PORT` → sandboxed; `LM3_SETTINGS`, `LM3_SETTINGS_PATH`,
  `LM3_HARDWARE_SETTINGS`, `LM3_RUNS_ROOTS` → cleared.
- `XDG_CONFIG_HOME`, `XDG_STATE_HOME`, `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `XDG_RUNTIME_DIR` →
  sandboxed. `XDG_DATA_HOME` was **not** isolated until this pass; adding the
  `project.output.dir: auto` default made run output depend on it.
- **A child process inherits the sandbox** — verified by reading the environment back out of a
  spawned interpreter, not by trusting that it was set.
- **The production deployment lease is unreachable from tests**: the production and test deployment
  keys differ, and the production runtime directory does not exist.
- No GPU calibration and no real inference were run.

## Untouched state

The real user configuration was snapshotted (paths, sizes, mtimes and md5 sums) immediately before
and after the full suite:

- `<user-config>/lm3` — **unchanged**. Two pre-existing legacy hardware profiles remain in place,
  byte-identical to each other (1 distinct md5), deliberately not deleted.
- `<user-state>/lm3`, `<user-cache>/lm3`, `<user-data>/lm3` — unchanged.
- The production runtime directory was **never created**.
- No `machine3`, `leafmachine3` or `global_greening` process was running at any point, and
  `nvidia-smi` reported no compute apps. Nothing was stopped, signaled or adopted.

## Platform scope

Linux desktop and Linux cluster are the qualification target. The 29 Windows lease-adapter tests
that execute here run against an injectable fake — a test of the *model* of the Win32 object
manager, not the object manager. Windows and macOS are **implemented but not yet natively
validated**. The cross-platform CI job is defined but **has never executed** (this checkout has no
git remote), and an unexecuted workflow is not evidence.
