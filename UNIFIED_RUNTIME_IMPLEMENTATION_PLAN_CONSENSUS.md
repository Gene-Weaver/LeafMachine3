# Unified LM3 runtime — consensus implementation plan

Date: 2026-08-28
Revision: 15 (amended after the batch-ownership reversal and the control-authority simplification)
Status: **implementation-ready; every protocol below is stated, not deferred**
Supersedes: `UNIFIED_RUNTIME_IMPLEMENTATION_PLAN.md` (proposal)
Inputs reconciled: the original proposal, a code-verified adversarial review
(`UNIFIED_RUNTIME_IMPLEMENTATION_PLAN_REVIEW.md`), and twelve rounds of independent assessment.

## 0. What this document is

The original proposal's architecture survived review intact. What did not survive was the
assumption that a single flat lease could be dropped into `machine3()` without breaking three
things that work today: VRAM calibration, the Global Greening batch, and the test suite.

Revision 1 replaced the flat lease with an activity hierarchy. Revision 2 closed the holes in that
hierarchy — a subactivity could **outlive its root lease**, and the server's Start endpoint
**cannot report a busy loser at all** in its current shape. Revision 3 removes the remaining
deferrals: root finalization while a child is alive, who writes which record, and the settings
precedence chains that the original GUI/settings split actually turns on.

**A note on what "exact" means here.** Anywhere this document previously said "X or Y", it now says
X. Those either/ors were the last places two implementers could have built different systems.

Every code claim below was verified against the tree. Claims that were wrong in the earlier review
are corrected in **Appendix B**; the corrections change scope, so they are part of the plan.

### Retained core principles

1. An OS-held lock establishes exclusivity and liveness.
2. JSON describes the lock holder. It never grants control authority.
3. Runtime truth and next-run settings are separate concerns and are labeled as such in the UI.
4. CLI, Python API, browser, and Electron all observe the same backend-owned runtime record.
5. No process is signaled because a PID appeared in a file or an unauthenticated response.

### Revision 2 amendments

| # | Issue | Resolution |
|---|---|---|
| A1 | A subactivity could outlive the root lease and let a second root start | Children **inherit the lease reference**; §2.2 |
| A2 | `/v1/run/start` returns after `Popen`, so a busy child can never yield 409 | Explicit **launch handshake** over a status pipe; §2.4 |
| A3 | Child-record storage left as an either/or | Exact file layout + **activity-specific schemas**; §3.2 |
| A4 | Path unification missed setup/calibration; calibration bootstrap is reentrant | Resolver scope widened; **provisional profile** for the child; §2.2, §3.1 |
| A5 | Steps 3 and 5 shipped separately would reintroduce the batch gap | Merged into one shipping unit; **superseded by revision 14**, which accepts the gap and removes the batch orchestrator (§2.3) |
| A6 | Cluster `state_dir` contradicts `db_path = run_dir/<name>.sqlite` | Four explicit path roles + **periodic backup and SIGTERM checkpoint**; §2.10 |
| A7 | A valid LM3 server for the *wrong deployment* was treated as "unrelated" | `/healthz` carries the deployment key; one auth flow chosen; §2.9 |
| A8 | `tmp_path` is function-scoped; xdist workers would share a deployment | Isolation at conftest import; §4 Step 1 |

### Revision 3 amendments

| # | Issue | Resolution |
|---|---|---|
| B1 | Root finalization could remove `active.json` while a child still held the lock | Root joins children first; record survives a hard kill; §3.3 |
| B2 | No record-writer ownership; the shared lock cannot serialize JSON writes | Root owns `active.json`, child owns its own file; §3.2 |
| B3 | Capability had nothing to validate against | One-use **grant record**; §2.2 |
| B4 | Settings precedence was deferred to implementation | Normative table for every path; §3.1 |
| B5 | Console handling was still an either/or | Control pipe + server-private request log; §2.4 |
| B6 | Local descriptor holds a token, cluster descriptor must not | Two artifacts: `connection.private.json` / `connection.public.json`; §2.12 |
| B7 | Backup could corrupt the last good archive; "resume without loss" overclaimed | Atomic generations, stated RPO; §2.10 |
| B8 | Deployment key used raw env text as a path component; port rule ambiguous | Canonical slug+hash, explicit port rule, NFS refused; §2.1, §3.1 |
| B9 | `hardware_setup` with no config is a live crash | Setup requires a resolved config; §2.13 |

### Revision 4 amendments

Revision 3's new material introduced four cross-contract problems of its own, plus two
self-contradictions. All are corrected here.

| # | Issue | Resolution |
|---|---|---|
| C1 | A machine-scoped hardware profile is shared mutable state across deployments | Profile is **deployment-scoped**; §3.1 |
| C2 | GUI hardware setup runs in-process, so Stop would kill the server | Setup runs as a controllable **subprocess**; §2.13 |
| C3 | Postprocessing was both "beside settings" and "deployment-scoped"; workspace pointer undefined | Deployment-scoped; pointer fully specified; §3.1 |
| C4 | Browser auth vanished after the transition release | HTML-meta bootstrap is **permanent** for loopback/tunneled browsers; §2.11 |
| C5 | Grant consumption needed two writers on one file | Atomic **claim-by-rename**; §2.2 |
| C6 | Abandoned cleanup writes `last.json`, violating single-writer | Explicit **recovery-writer** role; §3.2 |
| C7 | Handshake timeout left the child running | Timeout **terminates** the child before responding; §2.4 |
| C8 | `O_CREAT\|O_EXCL` cannot rotate on server restart | Temp sibling + atomic replace; §2.12 |
| C9 | `casefold`/NFKC is not reproducible in JavaScript | Hash-of-raw authoritative, ASCII-only cosmetic slug; §2.1 |
| C10 | "Next safe point" checkpoint can miss the scheduler grace window | Dedicated checkpoint thread + versioned snapshots; §2.10 |

### Revision 5 errata

Architecture accepted at revision 4. These are narrow safety and protocol details, not design
changes.

| # | Issue | Resolution |
|---|---|---|
| D1 | Two conflicting archive contracts in one section | Versioned pointer used throughout; §2.10 |
| D2 | The handshake is POSIX-only (`pass_fds`) | Windows transport named; §2.4 |
| D3 | `<machine-key>` was used but never defined | Defined, and includes host identity; §3.1 |
| D4 | The server logs the raw bearer token | Never log it; `no-store` on the bootstrap; §2.11 |
| D5 | A grant was claimed before it was validated | Validate → claim → revalidate; §2.2 |
| D6 | Setup progress transport left as an either/or | Append-only JSONL event log; §2.13 |
| D7 | Stale workspace pointer behavior undefined | Explicit failure state, never a silent fallback; §3.1 |
| D8 | `children/` and consumed grants grow forever | Documented retention on finalization/recovery; §3.2 |
| D9 | Postprocessing lock could land on NFS | Lives in the local runtime registry; §2.8 |

### Revision 6 errata

| # | Issue | Resolution |
|---|---|---|
| E1 | Archive contract still read two ways on desktop | Explicit **in-place vs staged** modes; §2.10 |
| E2 | Windows handshake still offered two transports | `STARTUPINFOEX` handle allowlist chosen; §2.4 |
| E3 | Machine key collides across containers sharing an image machine ID | Canonical input with node name **always** included; §3.1 |
| E4 | "Atomically replace" does not survive node failure | Explicit fsync sequence; §2.10 |
| E5 | Invariant 6 demanded project identity every activity has | Qualified "where applicable"; §1 |

### Revision 7 errata

| # | Issue | Resolution |
|---|---|---|
| F1 | Checkpoint referenced "steps 1-5" after the procedure grew to 8 | Runs through step 7; pruning follows; §2.10 |
| F2 | Grant step 1 forward-referenced the step-4 descriptor check | Descriptor validated in step 1, before any rename; §2.2 |
| F3 | A staged run had no defined state before its first snapshot | `archive_status: pending/ready/n/a`; §2.10, §3.2 |
| F4 | Periodic, signal, and final backups could race each other | One checkpoint coordinator/mutex; §2.10 |

### Revision 8 errata

| # | Issue | Resolution |
|---|---|---|
| G1 | Coalescing a final/signal checkpoint could drop writes | Only periodic requests may coalesce; §2.10 |
| G2 | A terminal archive failure was unrepresentable | `archive_status: stale` / `failed`; §2.10 |
| G3 | `n-a` vs `n/a` in the revision-7 summary | Standardized on `n/a` |

### Revision 9 amendments

| # | Issue | Resolution |
|---|---|---|
| H1 | **Windows `LockFileEx` locks do not transfer to children** — the subactivity lifetime mechanism was POSIX-only | Named kernel event carries lifetime; file lock keeps cross-session exclusion; §2.2 |
| H2 | Gate 14 contradicted the new `failed` archive state | Qualified by `archive_status`; §8 |
| H3 | `<generation>` undefined; a resumed run could reuse a filename | `<run_id>.<sequence>` with exclusive creation; §2.10 |

### Revision 10 amendments

| # | Issue | Resolution |
|---|---|---|
| I1 | `Local\` left the cross-session lease broken, and the `SeCreateGlobalPrivilege` claim was wrong | Authoritative `Global\` SID-scoped event; §2.2 |
| I2 | Windows handle validation had no mechanism | `OpenEventW` + `CompareObjectHandles`; §2.2 |
| I3 | Two summary lines still said "locked file description" | Reworded to "lease reference" |

### Revision 11 errata

| # | Issue | Resolution |
|---|---|---|
| J1 | A busy contender leaked the event handle and pinned the deployment | `CloseHandle` before raising `RuntimeBusyError`; §2.2 |
| J2 | Only the POSIX descriptor's inheritance was disarmed in the child | `SetHandleInformation(..., 0)` + clear `LM3_LEASE_EVENT_HANDLE`; §2.2 |
| J3 | An "optional" Windows `LockFileEx` with undefined failure behavior | Removed; `activity.lock` is passive on Windows; §2.2 |
| J4 | Deferred Windows floor, undefined SID hash, one stale phrase | Normative Win10/2016; `sha256`(string SID)[:16]; reworded; §2.2 |

### Revision 12 amendments

| # | Issue | Resolution |
|---|---|---|
| K1 | **The root's inheritable event handle leaked to its own executor workers** | Non-inheritable owner handle + per-launch `DuplicateHandle`; §2.2 |
| K2 | Windows section still framed as "event plus file lock" / two primitives | Rewritten as one primitive; §2.2 |
| K3 | Finalization still said "lock descriptor" universally | Platform-neutral "lease reference"; §3.3 |

### Revision 13 amendments

Raised by implementing Steps 1, 2 and 5a and auditing the result. Two are defects the
implementation exposed in the plan itself; two are policy the plan never stated.

| # | Issue | Resolution |
|---|---|---|
| L1 | **The plan never said what a relative path in a config resolves against.** Step 1 unified settings resolution while `Config.resolve_path()` kept joining the CWD, so `build_dirs()` and `resolve_run_paths()` now return two different run directories for one config in one process — and the shipped default `output.dir` is the relative string `runs` | Six-rule relative-path contract, one seam, and a testable invariant; §3.5 |
| L2 | The built-in `output.dir: runs` default would, under §3.1's seeded settings location, put first-run results inside a hidden **configuration** directory | Default becomes `auto`, resolving to `<checkout>/runs` in a dev checkout and `<user-data>/lm3/<deployment>/runs` when installed; §3.5 |
| L3 | `examples/*.yaml` silently depend on being launched from the repository root | Migrated to settings-relative values that preserve their effective locations, with a test; §3.5 |
| L4 | Gate 13 (token redaction, `no-store`) was stated in §2.11 and §8 but **owned by no step**, so it was implemented nowhere | Assigned explicitly to Step 5b; §4 |
| L5 | No stated policy on which platforms are qualified when, so Linux-only evidence risked being reported as cross-platform validation | Linux is the qualification target now; Windows/macOS are "implemented but not yet natively validated"; §1.1 |

### Revision 15 amendment

| # | Issue | Resolution |
|---|---|---|
| N1 | **§2.5's five-way `can_stop` match added two checks that were redundant and one that was vacuous.** A server holding a `Popen` is the child's parent, so PID reuse cannot occur; and creation time comes from `process_start_time()`, which returns `0.0` without the optional `psutil` — the normal state on Windows and macOS — so the comparison degrades to "always matches" there | Control authority is **handle ownership alone**: a retained live child handle for this `run_id`, launched by this `instance_id`. Matches the rule §2.11 already used for Electron. The `process_start_time` entry task is withdrawn; §2.5 |

### Revision 14 amendment

| # | Issue | Resolution |
|---|---|---|
| M1 | **The batch was made a lease-holding root activity to close an inter-species gap that does not cost what §2.3 assumed.** `run_global_greening.sh` is 226 lines that call LM3 once per species and skip the ones already complete. Making it a `batch` root pulled the grant/capability/inheritance machinery onto a convenience wrapper | The batch stays a shell script; each species takes an ordinary `pipeline` root lease. `batch` and `batch_item_pipeline` are removed; §2.3 records why |

### What changed from the original proposal

| Area | Proposal | Consensus |
|---|---|---|
| Lease granularity | one flat lease per deployment | **root activity** lease + inherited **subactivity** lease reference |
| Calibration | unaddressed (would deadlock) | inherited lease reference + capability + provisional profile |
| Batch | lease released between species | **unchanged, deliberately**: each species takes an ordinary `pipeline` lease; a lost gap costs one re-runnable species (§2.3) |
| `LM3_DEPLOYMENT_ID` | descriptive namespacing | **load-bearing**: keys registry dir, `/healthz`, Electron identity |
| Concurrency | silently removed | preserved via explicitly named deployments |
| Server Start | "wait for record or exit" (unimplementable as written) | explicit status-pipe handshake before responding |
| Tests | unaddressed | mandatory runtime-namespace isolation, xdist-safe |
| Electron | add `requestSingleInstanceLock()` | **upgrade first** (CVE-2026-34776), pin a supported major |
| Invariant 5 | absolute | qualified to follow-active mode |
| Postprocessing | unscoped | explicit read/write + resource policy |
| Schema skew | unspecified | defined fail-safe behavior |
| Cluster DB | "state_dir" mentioned late | four named path roles in the core model |

---

## 1. Governing invariants

Acceptance requirements, not aspirations. Each maps to a gate in §8.

1. At most one **root activity** holds a given deployment's lease at a time. A root activity is a
   `pipeline` or a `hardware_setup`.
2. A **subactivity** (`calibration_pipeline`) executes under its parent's
   root lease by **inheriting the parent's lease reference** — on POSIX the locked open file
   description, on Windows a handle to the lease event, because Windows file locks do not transfer
   to children at all (§2.2). It never acquires, replaces, or unlocks it. The deployment stays occupied for as long as the root *or any live subactivity*
   holds that lease reference — so killing the parent does not free the lease while a child runs.
3. Ordinary executor worker processes never inherit the lease reference — neither the POSIX lock
   descriptor nor the Windows lease-event handle.
4. Every execution entry point reaches the same lease code: `machine3` CLI, `python -m leafmachine3`,
   direct `machine3()` calls, the GUI launcher, hardware setup, and legacy server jobs.
5. `GET /v1/runtime` and the GUI obtain active identity from the lease record, never from mutable
   YAML or a filesystem recency guess.
6. An active record contains the exact config path used at launch, and — **where the activity has
   one** — the effective project identity. A `hardware_setup` root deliberately carries no
   project (§3.2).
7. **In follow-active mode**, which is the default, status, logs, results, and postprocessing
   targets follow the active record's paths. An explicit `run=`/`db=` selector deliberately
   overrides this for that client request and visibly pauses follow-active.
8. Editing settings during a run never changes which run the GUI is displaying. The UI labels those
   edits as applying to the next run.
9. Closing Electron never stops an LM3 run. Stopping a run is a separate, explicit action.
10. Electron stops a server only when it can prove this Electron process spawned that exact server
    instance **for this deployment**.
11. A second Electron launch **for the same deployment** focuses the existing window and exits.
    Deliberately distinct deployments may each have a window.
12. No process is signaled solely because a PID appeared in a JSON file or an unauthenticated
    `/healthz` response.
13. The server process itself is **never** a lease holder, without exception. It observes, and it
    launches children that acquire leases — including hardware setup, which runs as its own
    subprocess so that Stop can terminate it without terminating the server (§2.13).
14. A losing (busy) launch performs no **execution-owned** mutation: no run directory, no project
    DB, no CUDA initialization, no orphan reaping. Server-private staging is permitted and is
    defined in §2.4.

The coordination boundary is **one named LM3 deployment**. By default that is one OS user on one
host. `LM3_DEPLOYMENT_ID` splits it deliberately. Multi-user service deployment and cross-user
arbitration remain out of scope.

### 1.1 Platform qualification policy

The invariants above are platform-neutral. What is *qualified*, and when, is not — and conflating
"implemented" with "validated" is how a project ships a Windows lease nobody ever ran.

**The qualification target now is Linux: Linux desktop and Linux cluster.** Those two are what the
gates in §8 must actually pass against before Step 3 proceeds and before release.

**Windows and macOS code and packaging configurations must still be implemented correctly** — the
Windows lease adapter of §2.2, the `STARTUPINFOEX` handshake of §2.4, and the Electron packaging
definitions of §2.11 are all in scope and are all expected to be right. What is **deferred** is
their *native execution testing*:

| Platform | Status while deferred | Validated later on |
|---|---|---|
| Linux desktop / cluster | **qualification target** — gates must pass here now | this host and Linux CI |
| Windows | **implemented but not yet natively validated** | a real Windows machine |
| macOS | **implemented but not yet natively validated** — packaging, signing, notarization, permissions, and execution all | a real Mac |

Three rules follow, and they are not negotiable:

1. **Deferred native testing does not block Linux Step 3 work.** Windows and macOS qualification is
   a later release gate, not a prerequisite for the Linux path.
2. **Windows and macOS are labeled "implemented but not yet natively validated", never "passed".**
   A gate proven only against the injectable Win32 fake of §2.2 is a test of our *model* of the
   object manager, not of the object manager. Say so, every time, in every report.
3. **An unexecuted CI workflow is not evidence.** The cross-platform job definitions are retained
   deliberately so they run the moment the repository has a remote — but a workflow that has never
   run proves nothing, and must never be cited as though it had.

---

## 2. Resolved design decisions

Each subsection settles something two reviewers previously read two ways. These are decisions, not
options.

### 2.1 Deployment identity is load-bearing

`LM3_DEPLOYMENT_ID` is the **deployment key**. It must resolve to:

- a distinct runtime registry directory (lock, `active.json`, `last.json`, `children/`);
- a distinct Electron user-data / single-instance identity;
- a distinct default server port, or an explicitly configured one;
- a value reported by `/healthz` and verified by every client.

Default desktop deployments coordinate. Advanced users who want two independent runs opt in:

```bash
LM3_DEPLOYMENT_ID=gpu0  LM3_PORT=8765  # compute.devices: [0]
LM3_DEPLOYMENT_ID=gpu1  LM3_PORT=8766  # compute.devices: [1]
```

**Why deployments and not per-GPU locks:** per-GPU locks would require the runtime API and the GUI
to represent several simultaneous active projects, pulling scheduler semantics into scope. A named
deployment is one active project, one GUI, one lease — the same mental model, instantiated twice.

**The raw environment value is never used as a path component.** A value containing `/`, `..`,
Unicode look-alikes, or excessive length would escape or destabilize the namespace. The canonical
key is:

```
canonical = ascii_slug(raw)[:32] + "-" + sha256(raw.encode("utf-8")).hexdigest()[:8]

ascii_slug(raw):
    for each character: if it is ASCII A-Z -> lowercase it (add 0x20);
                        else if it is ASCII a-z or 0-9 -> keep;
                        else -> "-"          # every non-ASCII byte included
    collapse runs of "-", strip leading/trailing "-"
```

**The authority is the hash, not the slug.** `str.casefold()` and NFKC normalization are *not*
reproducible across Python and JavaScript — JS has `toLowerCase()`, which differs from casefold
(`ß` casefolds to `ss` but lowercases to `ß`), and normalization tables vary by engine version. A
key that disagrees between the two produces a directory mismatch *before* any golden test could
catch it. So the slug is deliberately ASCII-only, mechanical, and cosmetic; the `sha256` of the raw
UTF-8 bytes is what actually separates namespaces, and that is byte-identical everywhere.

Defaults, stated so there is no ambiguity:

- Unset `LM3_DEPLOYMENT_ID` means the raw literal **`"default"`**. Nothing else.
- Explicitly setting `LM3_DEPLOYMENT_ID=default` is therefore **identical** to leaving it unset,
  including owning port 8765.
- An empty or whitespace-only value is rejected at startup rather than silently becoming `default`.

The key is derived by one function with **golden test vectors shared between Python and Electron**,
because the two must agree byte-for-byte or Electron will single-instance-lock against a different
deployment than the server it talks to.

**Port rule, stated exactly:** the default deployment uses 8765. Any explicitly named additional
deployment **must** set `LM3_PORT`; starting a named non-default deployment without one is a
startup error, not a silent collision on 8765. `/healthz` deployment verification (§2.11) remains
the final defense, because a user can still point two deployments at one port by hand.

Consequences to document:

- Two checkouts under the default deployment coordinate. That is intentional.
- An immutable batch worktree and a development checkout may run concurrently **only** when given
  different deployment IDs, ports, output paths, and GPU sets.
- `tests/test_executor.py::test_reaper_never_touches_a_concurrent_lm3_runs_workers` stays valid,
  reframed: the reaper must not touch workers belonging to **another live deployment**.

### 2.2 Root activities and inherited subactivities

```
root: pipeline | hardware_setup
  └── subactivity: calibration_pipeline
```

#### Lock lifetime — the mechanism, not just the rule

Revision 1 said a child "validates that the root lock is held live". That is insufficient: if the
parent is killed after the child validates, the OS releases the parent's lock while the child keeps
running, and a new root acquires an apparently free lease. Two LM3 runs then overlap — the exact
failure the lease exists to prevent.

The child therefore **inherits the lease reference itself**. On POSIX that is the lock:

- POSIX: `flock` locks belong to the *open file description*, not the file descriptor. A child that
  inherits the descriptor shares that description, so the lock is released only when the last
  descriptor referring to it closes. The parent opens `activity.lock`, acquires `LOCK_EX|LOCK_NB`,
  and passes the descriptor to approved subactivity spawns with
  `subprocess.Popen(..., pass_fds=(lock_fd,))`.
- The descriptor number and the capability travel in the child's environment
  (`LM3_LEASE_FD`, `LM3_LEASE_CAPABILITY`).
- The child validates, then immediately calls `os.set_inheritable(lock_fd, False)` and clears both
  environment variables **before** it spawns any executor worker. Workers must never hold the
  deployment open.
- Python descriptors are non-inheritable by default (PEP 446) and `subprocess` closes everything
  not in `pass_fds`, so the default posture is already correct. The requirement is to never mark
  the descriptor globally inheritable, and to prove it with a test rather than trust it.
- **Windows cannot use this mechanism at all**, and it is not a matter of a different API spelling.
  `LockFileEx` locks belong to the *process*, not to a handle: Microsoft states that when a file
  handle is inherited by a child, "the child process is not granted access to the locked region",
  and that when a process terminates holding a lock, "the locks are unlocked by the operating
  system". Killing the root would therefore free the Windows lease while its subactivity kept
  running — exactly the violation this section exists to prevent. (The OS unlock is also *delayed*
  and resource-dependent, so a dead process's lock can linger, which is the opposite failure.)
  Windows gets a different lifetime primitive; see below.

#### The Windows lease: one named global kernel event

`LockFileEx` cannot serve, for the reason above. The Windows lease is therefore a **single**
primitive — a named kernel event in the `Global\` namespace — which provides both properties at
once: machine-wide exclusion, and a lifetime that outlives the creating process while any child
still holds a handle.

```
Global\lm3-lease-<user-SID-hash>-<canonical-deployment-key>
```

A named event rather than a named mutex, deliberately: a mutex carries thread affinity and
abandonment semantics we do not want. Nobody *acquires* the event; its **existence** is the lease,
and `CreateEventW` reporting `ERROR_ALREADY_EXISTS` is the atomic test-and-create. The object lives
until its **last handle** closes, and handles are inheritable — a close analogue of POSIX
open-file-description lifetime.

`Global\`, not `Local\`. `SeCreateGlobalPrivilege` gates only **file-mapping and symbolic-link**
objects in the global namespace; events do not require it, and Microsoft's own namespace
documentation demonstrates `CreateEventW(NULL, FALSE, FALSE, L"Global\CSAPP")`. A `Local\` event is
scoped to one logon session, which would leave the dead-root-with-live-child window unguarded
against a contender in another session of the same user — a documented hole is still a hole, and
invariant 2 is unqualified.

The name embeds a hash of the user's SID so that two different users on one machine get distinct
leases (the global namespace is machine-wide). That hash is defined, not left to taste:
`sha256` of the SID in its canonical **`ConvertSidToStringSidW`** form (`S-1-5-21-…`), encoded UTF-8,
truncated to **16 hex characters** — the same bounded-hash convention as the deployment key (§2.1)
and the machine key (§3.1). The event is created with a `SECURITY_ATTRIBUTES` ACL granting the
current user only.

**If the global event cannot be created, that is a startup error.** There is no silent fall back to
`Local\`, which would weaken the invariant precisely when the environment is unusual.

**Windows does not lock `activity.lock` at all.** The file remains as a passive compatibility
artifact so the registry directory has the same shape on every platform, but no `LockFileEx` is
taken and nothing consults it. Revision 10 kept it as an "optional auxiliary" with undefined failure
behavior, which is just a second mechanism waiting to be mistaken for a source of truth.

**Root acquisition** is one step:

1. `CreateEventW` the canonical global name with a **non-inheritable** handle and a
   current-user-only ACL. If
   `GetLastError() == ERROR_ALREADY_EXISTS`, the deployment is occupied — by a live root, or by a
   subactivity whose root has already died.
   **`CloseHandle` that returned handle immediately, before raising `RuntimeBusyError`.**
   `CreateEventW` returns a valid handle to the *existing* object on `ERROR_ALREADY_EXISTS`, and the
   object lives until its last handle closes — so a contender that keeps it would pin the deployment
   occupied after the real run exits. That is not hypothetical for a direct Python caller who
   catches `RuntimeBusyError` and carries on in the same process.

**ACL detail that will otherwise bite:** when `CreateEventW` opens an *existing* named event it
requests `EVENT_ALL_ACCESS`, so the current-user-only DACL must grant `EVENT_ALL_ACCESS` to that
user or our own next process fails to open its own lease.

**The root's own handle is never inheritable, and this is the whole ballgame.** If the root created
an inheritable handle and simply held it, every ordinary executor worker it spawned with
`bInheritHandles=TRUE` would inherit the lease — and disarming inheritance *inside an approved
child* does nothing about that, because the workers are spawned by the **root**, from the root's own
handle. One long-lived worker would then hold the deployment occupied after the root and its
subactivities had all exited. POSIX is safe here by construction (descriptors are non-inheritable by
default under PEP 446, and `pass_fds` opts in per spawn); Windows needs the duplication dance
explicitly.

**Subactivity launch:**

1. the root keeps its **non-inheritable owner handle** untouched;
2. for this launch only, `DuplicateHandle` it into a temporary **inheritable** copy;
3. pass only that duplicate through the `STARTUPINFOEX` handle allowlist the status pipe already
   uses (§2.4), as `LM3_LEASE_EVENT_HANDLE`;
4. `CloseHandle` the parent's duplicate immediately after `CreateProcess` — **on every path,
   including every failure path**, or a failed spawn leaks a lease reference that nothing will ever
   release;
5. the child validates its copy and makes it non-inheritable, as specified below;
6. **serialize the spawn window.** Between steps 2 and 4 an inheritable lease handle exists in the
   root, so a *concurrent* `CreateProcess` from another thread — an executor worker starting at that
   moment — could inherit it. Subactivity launches and worker launches take one process-creation
   mutex so that window is never open during an unrelated spawn.

**Handle validation, concretely.** "Verify it is the right event" needs a mechanism, not an
intention. The child:

1. opens the canonical global name with `OpenEventW`;
2. calls **`CompareObjectHandles`** (`handleapi.h`) on that handle and the inherited one — the API
   exists specifically to decide whether two handles refer to the same underlying kernel object;
3. rejects and exits on mismatch, and closes the comparison handle either way.

**`CompareObjectHandles` requires Windows 10 / Windows Server 2016, and that is LM3's normative
Windows floor** — not a value to confirm later. Anything older is unsupported for the runtime lease,
which is not a meaningful constraint in 2026.

**After validating, the child immediately:**

1. calls `SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)` so the lease handle is **not**
   inherited any further — without this an ordinary executor worker would inherit it and hold the
   deployment open, which invariant 3 forbids;
2. clears `LM3_LEASE_EVENT_HANDLE` from its environment, alongside `LM3_LEASE_CAPABILITY` and the
   POSIX `LM3_LEASE_FD`.

That yields the required behavior with no session caveat: kill the root, and the event object
persists because the child still holds a handle, so the next root — **in any logon session of that
user** — sees `ERROR_ALREADY_EXISTS` and stays out. When the last child exits, the last handle
closes, the object is destroyed, and the deployment is free.

A lease-guardian process holding `LockFileEx` until the whole tree exits was the alternative. It was
rejected: it adds a process to supervise, monitor, and clean up, and it relocates the lifetime
problem rather than solving it.

**Mandatory test, on both platforms:** kill the root parent while a subactivity child is still
running, and prove a second root activity remains blocked until that child exits. On Windows this is
a first-class CI job, not an extrapolation from the POSIX result — the two implementations share no
mechanism.

#### Capability validation — against a one-use grant record

"Validates the capability value" is meaningless without saying what it is validated *against*.
Before launching a child the root writes a **grant record** to
`children/<child_run_id>.grant.json`:

```json
{
  "schema_version": 1,
  "child_run_id": "uuid",
  "parent_run_id": "uuid",
  "deployment_id": "canonical-key",
  "purpose": "calibration_pipeline",
  "capability_sha256": "...",
  "issued_at": 1787932800.25,
  "expires_at": 1787932860.25,
  "consumed": false
}
```

**Consumption is a rename, not a field update.** A `consumed` flag would need the child to write a
file the root owns — a second writer on one file, which §3.2 forbids and which the shared lock
cannot serialize. Instead the child claims the grant by renaming it:

```
children/<id>.grant.json   →   children/<id>.grant.consumed.json
```

Exactly one claimant can win that rename; a replay finds no source file and fails closed. This
keeps one-writer-per-file without introducing a second lock. (The `"consumed"` field stays in the
schema as a readable marker written by the claimant *into the renamed file*, not as the mechanism.)

Order of operations in the child. **Validation comes first**: claiming before checking would let an
accidental or mis-ordered invalid child consume the legitimate child's grant and lock it out of a
run it was entitled to.

1. **Validate everything, including the lease reference, before touching the grant file.**
   - `sha256(raw) == capability_sha256`, deployment ID, parent `run_id`, declared purpose,
     `expires_at`;
   - **the inherited lease reference**, not merely its presence. On POSIX: `fstat` the descriptor,
     confirm it names the same inode as the deployment's stable `activity.lock`, re-assert the lock
     non-blockingly on that inherited open file description, and reject any descriptor that turns
     out to be an ordinary independently opened handle while the parent's lock is held. On Windows:
     `OpenEventW` the canonical global lease name and `CompareObjectHandles` it against the
     inherited handle, rejecting on mismatch (§2.2). Each platform needs its own check and its own
     test.
2. Claim by atomic rename. On failure — already consumed, or absent — exit immediately. The rename
   still decides the single winner; validation merely stops the wrong process from racing for it.
3. Re-read the claimed file and revalidate expiry and identity, since the claim itself can be
   delayed.
4. Stop the lease reference propagating any further, **before** loading config or doing any other
   expensive work, so a later fork cannot carry it: on POSIX `os.set_inheritable(lock_fd, False)`,
   on Windows `SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)`. Then clear
   `LM3_LEASE_CAPABILITY`, `LM3_LEASE_FD`, **and `LM3_LEASE_EVENT_HANDLE`** from the environment.

No step may forward-reference a later one. A child that has not finished step 1 — descriptor
included — must not rename anything, or an invalid child could consume the legitimate child's grant
and lock it out of a run it was entitled to.

The raw capability value travels only in the child's environment (`LM3_LEASE_CAPABILITY`). Any
mismatch at any step fails closed.

**Scope statement, so nobody over-trusts this:** the grant is *coordination* protection — it stops
an accidental or mis-ordered invocation of the internal child path from bypassing the lease. It is
**not** a security boundary against hostile code running as the same user, which can read the
environment and the runtime directory anyway. Do not add features that assume otherwise.

Root `active.json` stays authoritative. Child detail is stored per §3.2.

The UI therefore shows a tree and never mistakes `_lm3_calibration` for the user's project:

```
Hardware setup
└── calibration pipeline

Global Greening (a shell loop; each species is an ordinary root)
└── (no subactivity — see §2.3)
```

#### Calibration must not bootstrap itself recursively

`run_setup` writes `hardware_settings.yaml` only *after* `_apply_vram_measurements` returns
(`hardware_setup.py:176`), and calibration runs inside that call (`hardware_setup.py:305-315`).
So the calibration child starts with no profile on disk, `ensure_hardware_profile` sees
`not HW_PATH.exists()` (`hardware_setup.py:100-105`), and runs a **second full setup sweep inside
the child**. It is not infinite — the nested `run_setup` defaults to `calibrate=False` — but it is
reentrant, slow, and it perturbs the very VRAM measurement being taken.

Fix: the parent writes a **provisional profile** to a temporary path and points the child at it
(`LM3_HARDWARE=<provisional>`), so the child binds a realistic profile and its
`ensure_hardware_profile` is a no-op. A "profile under construction" sentinel that short-circuits
auto-setup in a `calibration_pipeline` child is an acceptable alternative, but the provisional
profile is preferred because it also makes calibration measure under the configuration the real run
will use.

#### Calibration failure must be loud

`calibrate.py:163-166` currently downgrades a nonzero child exit to
`log.warning(... "keeping heuristic estimates")` and returns `False`. When calibration was
explicitly requested (`--calibrate`, or the GUI Calibrate VRAM button), a child failure must make
the CLI exit nonzero and put the GUI setup job into `error`. Silent fallback to heuristic estimates
is acceptable only when calibration was not requested, or when the user explicitly selects an
allow-fallback option.

### 2.3 The batch is NOT a lease-holding root — reversed in revision 14

Revisions 1-13 required a Python batch orchestrator holding one root lease for the whole batch, so
that Start could never be enabled in an inter-species gap. **That is reversed.** The reasoning is
recorded rather than deleted, because the reversal turns on a cost estimate that was never checked.

**What the batch actually is.** `run_global_greening.sh` is 226 lines. For each species it reads
`project_status` from that species' SQLite, skips the ones already complete, runs
`python -m leafmachine3 --config … --input … --output …`, appends a row to a summary TSV, and
continues past failures. It is sequential LM3 invocations plus a skip check — a convenience wrapper,
not an execution engine.

**What the batch lease was buying.** Without it, no lease is held between species, so a GUI Start can
win that gap and the next species exits 75. The consequence is one species skipped, recorded as a
failure row in the summary, and picked up by the next run — because `species_state()` marks it
incomplete and the script is idempotent by construction. That is visible and self-healing, not data
loss.

**What it was costing.** A `batch` root activity, a `batch_item_pipeline` subactivity, a batch record
schema, an orchestrator to write and test, and the grant/capability/lease-inheritance machinery
exercised on the convenience path — to prevent a failure mode that costs one re-run.

**Decision.** The batch stays a shell script. Each species takes an ordinary `pipeline` root lease,
exactly like any other run. Lease inheritance is therefore exercised by **calibration alone**, which
genuinely requires it: a calibration child must run under its parent's lease because the parent is
mid-`run_setup` and holds the deployment.

Two consequences to implement rather than assume:

- **The batch must distinguish a busy exit from a real failure.** Exit 75 means "another root holds
  the deployment", which is a retry, not a defect in that species. The summary TSV records it
  distinctly so a human reading the run can tell "re-run this" from "this species is broken".
- **The GUI shows the current species as an ordinary `pipeline` run**, with no batch tree and no
  "between items" state. There is nothing to show between species because nothing is running.

**The normative model, in full.** For each species the wrapper: (1) decides from that species' own
SQLite whether it is already complete, (2) invokes ordinary `machine3`, which takes a normal
`pipeline` root lease, (3) lets the GUI show that species as an ordinary active run, (4) waits for
LM3 to exit and release the lease, (5) records the result in the summary TSV and moves on. LM3 owns
the safety and state of one invocation; the wrapper owns sequencing, and nothing else.

If a future batch genuinely needs gap-free exclusivity — a shared cluster where losing a species is
expensive rather than annoying — the mechanism to reach for is a batch-scoped root lease exactly as
revisions 1-13 described, and this section is the record of how it was specified.

**Stop is no longer a batch operation, and that is the boundary.** The GUI observes and can stop the
*currently running species*, because that is an ordinary run. Stopping the whole sequence means
interrupting the wrapper — Ctrl-C, or cancelling its Slurm job. One consequence the wrapper must
honor: if an individual species is stopped from the GUI, the wrapper must NOT advance as though it
succeeded; it records failed/retryable, and the DB-driven skip check picks it up next time.

### 2.4 Launch handshake and the staging boundary

`POST /v1/run/start` today creates the run's log directory at `metrics_api.py:716`, opens the
console log at `:730`, spawns at `:745`, and **returns at `:787` without waiting**. A child that
exits 75 therefore cannot influence the HTTP response — it has already been sent. "Return 409 when
busy" is unimplementable without a handshake, and "the busy loser creates no directories" is false
because the *server* created one before it knew the outcome.

**Protocol.** The server creates a status channel and passes it to the child. The transport is
platform-specific and both halves are normative — `pass_fds` does not exist on Windows, so naming
only the POSIX side would leave the Windows implementation to be invented:

| Platform | Transport |
|---|---|
| POSIX | anonymous pipe; write end passed via `pass_fds`, its number in `LM3_STATUS_FD` |
| Windows | inheritable anonymous-pipe handle passed through a `STARTUPINFOEX` handle allowlist, its value in `LM3_STATUS_HANDLE` |

The inherited-handle allowlist is chosen over a private named pipe deliberately: it mirrors the
POSIX side almost exactly, whereas a named pipe would drag in its own addressing, authentication,
ACL, cleanup, and replay rules — a second protocol to specify and test for no gain. Offering both
would have left the critical choice to implementation, which is the very thing this section exists
to prevent.

Timeout, maximum message size, EOF handling, child termination, and record reconciliation are
**identical on both platforms**. The handshake and process-tree Stop tests run on Windows CI, not
only the lock-adapter tests. Then:

1. Child loads and validates config, resolves project identity, and attempts the lease.
2. Child writes exactly one JSON line and flushes:
   - `{"status":"acquired","run_id":...,"project":{...}}`, then continues into execution; or
   - `{"status":"busy","active":{...sanitized winner record...}}`, then exits **75**.
3. Server reads with a bounded timeout. `acquired` → 200 with the runtime identity; `busy` → 409
   with the winner's record.
   **On timeout, malformed status, oversized status, or EOF the server must not simply return 500
   and walk away** — a slow child could acquire the lease and run on after the API reported failure,
   leaving a run nobody launched. The server instead: (i) terminates the retained process group /
   job object, (ii) waits and escalates if it ignores the first signal, (iii) reconciles any runtime
   record the child may already have published, and (iv) returns 500 only once no managed child
   remains. Regression test: a child that *does* acquire the lease but deliberately delays its
   status line past the timeout.
4. **Only after `acquired`** may execution-owned directories be created.

**Console handling — one design, because "or" here means dropped output or a deadlock.** Output
destinations are fixed when `Popen` runs, so this cannot be decided later:

- The status handshake uses a **dedicated small control pipe**, carrying exactly one JSON line.
- The child's `stdout`/`stderr` go **to a server-private request log under the jobs root** from the
  moment of `Popen` — permitted staging under the boundary below. They are never wired to a pipe
  the server might not be draining.
- After `acquired`, LM3 creates its normal application log under the run directory. The
  server-private console log stays with the managed launch and is **not** moved across filesystems.

That separation is what makes the handshake safe: a child can emit unbounded startup chatter before
it sends its status line without ever filling a pipe. Test exactly that case.

The lease attempt happens before hardware profiling and model loading, so the handshake timeout is
bounded by config load, not by a cold torch import. Size it accordingly (order of seconds, not the
240 s Electron uses for server boot) and make it configurable.

**Staging boundary.** `JobManager.create` (`app.py:71-78`) stages uploads and writes a job-local
`LM3_settings.yaml` before execution. That is legitimate and must keep working. The rule:

- **Permitted for a losing launch:** writes under the server-private jobs root, namespaced per job
  ID — uploaded inputs, a generated settings file, a server-side request log.
- **Prohibited for a losing launch:** anything under `<output.dir>/<run_name>/`, the project DB,
  the run's `logs/`, CUDA initialization, and orphan reaping.

Invariant 14 is stated in exactly those terms so it is testable.

### 2.5 Control authority is separate from observation

Observation comes from the registry. Control requires provenance. **A server may signal a run only
if it launched that run and still holds the handle**:

- a **retained live child handle** (`Popen` on POSIX, the job object on Windows) —
- for the runtime `run_id` the record names, launched by this server `instance_id`.

Anything else is **observer-only**: `can_stop: false`, with a reason the UI shows. A restarted
server observes and cannot signal. This is not "adoption" and must not be called that.

**Never reconstruct control authority from a PID in a JSON file.** That is the whole rule, and the
handle is what enforces it.

#### Why this replaced a five-way match — revision 15

Revisions 1-14 required the handle *plus* a matching PID *plus* a matching process creation time.
Those two extra checks were redundant and actively hazardous:

- **Redundant.** A process holding a `Popen` is the child's PARENT. The OS cannot recycle that PID
  until the child is reaped, and `proc.poll()` already answers "is it still alive". PID substitution
  cannot occur while the handle is held, so comparing the PID defends a case that does not exist.
  The checks would only matter for signaling a process the server did **not** launch — which this
  section forbids outright.
- **Hazardous.** Creation time comes from `records.process_start_time()`, which returns `0.0` when
  the optional `psutil` is absent — the normal state on Windows and macOS. A match test then reads
  `0.0 == 0.0` as agreement and silently degrades to "always matches" on two of three platforms. A
  check that looks rigorous and is vacuous is worse than no check, because it is trusted.
- **Inconsistent.** §2.11 already states this simpler rule for Electron→server ownership: "Electron
  shuts down only a server represented by its retained child handle with a matching instance ID and
  deployment key." Two ownership rules for one question is how they drift apart.

`process_started_at` **stays in the record** (§3.2) as descriptive provenance — it is genuinely
useful in diagnostics. It simply stops being a control input.

What this policy costs, stated plainly: after a server restart, a still-running run is observable
but not stoppable from the GUI. Stop it from the terminal that owns it, or by killing it. The
previous design did not actually offer more — it refused cross-restart control too.

`/healthz` identifies service, protocol compatibility, **and deployment key**. Its `pid` field must
be **removed or explicitly demoted to diagnostic-only**, because `app.py:532-538` currently
documents that field as the mechanism by which an attached client kills a server it does not own.
That behavior is being deliberately deleted, so leaving the field undocumented invites its return.

### 2.6 UI behavior is activity-aware

Default mode is follow-active.

| Root activity | GUI behavior |
|---|---|
| `pipeline` | follow its project DB and logs |
| `hardware_setup` | show a tuning state in the Machine panel; **do not** switch project history to `_lm3_calibration` |
| unknown/future | disable Start, refuse control, show a compatibility warning |

Historical selection pauses follow-active explicitly, with a visible action to return. Editing YAML
affects only the next run. Close never stops a pipeline.

### 2.7 The early path resolver has a bounded contract

Before directory creation, publish only paths that are deterministic from config:

- `run_dir` — `output.dir / run_name` (`dirs.py:42-44`)
- `active_db_path` — `run_dir / f"{run_name}.sqlite"` on desktop (`dirs.py:63`); see §2.10 for cluster
- `log_path` — under `run_dir / "logs"` (`dirs.py:56`)

**Do not publish the final `tmp_dir` before `build_dirs()` succeeds.** `_ensure_tmp`
(`dirs.py:67-76`) falls back to `<root>/_tmp_original` on `OSError`, so a configured scratch path
is not knowable in advance. The record must not carry `tmp_dir` at `starting`.

Fix-it: the `build_dirs` docstring (`dirs.py:39-41`) still claims `tmp` comes from the tuned
`hardware_settings.tmp_dir` when `tmp_dir: auto`. The code has not done that since the change
described at `dirs.py:46-49`, and `bind_hardware` (`config.py:447-456`) only attaches the profile
object. That stale docstring already misled one reviewer; correct it alongside the resolver.

### 2.8 Postprocessing concurrency policy

- A postprocessor may run against a completed historical run.
- It is **refused** when its target is the currently active pipeline's run directory.
- CPU-only tools may run against a different completed run while a pipeline is active.
- Every tool declares `read_only` / `read_write` and `cpu` / `gpu` resource metadata.
- Two `read_write` tools targeting the same completed run are serialized by a per-run advisory lock
  that lives in the **local deployment runtime registry**, keyed by a hash of the resolved
  `artifact_dir` — never as a lock file inside the output directory itself. Placing it beside the
  artifacts would put an advisory lock on exactly the network storage §3.1 declares unreliable for
  locking. Cross-host concurrent postprocessing is therefore out of scope for the single-user cluster
  profile, and is stated as such rather than left to look supported.
- Any future GPU postprocessor must participate in explicit GPU/resource coordination.
- Standalone CLI tools use the same guard as the HTTP API, not a parallel one.

Scope note: the shipped postprocessors (`leafmachine3/postprocessing/`) import no torch or
onnxruntime and are CPU-only today. The concrete present-day risk is a report/DB read-write race
against a live run. The `gpu` metadata field exists so the first GPU postprocessor cannot be added
without confronting the question.

### 2.9 Schema skew fails safe

When a reader encounters a record with a higher `schema_version`:

- the OS lock still decides whether the deployment is occupied;
- if held → report `active: true, compatible: false`;
- disable Start and all control actions;
- do not interpret unknown fields;
- display "This runtime was created by a newer LM3";
- if the lock is free → preserve/quarantine the stale record and permit a new root activity under
  the cleanup lock.

Readers tolerate **missing** optional keys as well as explicit `null`.

### 2.10 Cluster storage model, defined in the core

The core model cannot assume `db_path = run_dir / <run_name>.sqlite` while the cluster profile puts
the active SQLite on node-local scratch. Define four roles now, used on every platform:

| Role | Desktop | Cluster |
|---|---|---|
| `artifact_dir` | `run_dir` | persistent project storage |
| `active_state_dir` | `run_dir` | node-local scratch |
| `active_db_path` | `run_dir/<run>.sqlite` | `active_state_dir/<run>.sqlite` |
| `archive_pointer_path` | `null` | `artifact_dir/archive.current.json` |
| `archived_db_path` | **equal to `active_db_path`** | the versioned file **named by that pointer** |

**There are exactly two archive modes, and the mode is explicit in the record.**

| Mode | When | Behavior |
|---|---|---|
| **in-place** | `artifact_dir == active_state_dir` (desktop, and any single-filesystem run) | `archive_pointer_path` is `null`; `archived_db_path` equals `active_db_path`; **no copying, no generations, no pointer** |
| **staged** | `artifact_dir != active_state_dir` (the cluster profile) | versioned snapshots and a **mandatory** pointer |

Revision 5 declared archives "never a fixed filename", which was true for the cluster and wrong for
the desktop — where it would have introduced pointer indirection and background copying into a run
that has nothing to stage. Modes keep today's desktop behavior byte-for-byte while giving the
cluster its recovery guarantees. `db_path` survives as a deprecated projection of `active_db_path`
for one transition release.

SQLite WAL is not supported across machines on a network filesystem, which is why the active DB
must be node-local. **Completion-only checkpointing is insufficient** — Slurm preemption and
wall-time limits kill jobs mid-run.

**In staged mode, backups are generational and resolved through a pointer.** Overwriting a stable
archive path in place means an allocation that dies mid-copy destroys the last good archive, so no
staged path ever does `os.replace(..., artifact_dir/<run>.sqlite)`. In staged mode `archived_db_path`
is whatever the pointer currently names.

Atomic rename prevents a *torn read*, but by itself it does not survive node failure — the rename
and the pointer can both still be in page cache. The procedure therefore fsyncs at every step that
must outlive the node:

1. create `archive.<generation>.sqlite.tmp` with **exclusive creation** (`O_EXCL` / `CREATE_NEW`),
   where `<generation>` is `<run_id>.<sequence>` — the run's UUID plus a monotonic counter. Keying on
   `run_id` rather than `run_name` is what stops a *resumed* invocation of the same project from
   reusing a filename and overwriting an earlier run's archive; exclusive creation turns any
   residual collision into an error instead of silent data loss;
2. run the SQLite online backup into it;
3. `PRAGMA integrity_check` it, then `fsync` the file;
4. rename it to `archive.<generation>.sqlite`, then **`fsync` the parent directory**;
5. write the new pointer to an exclusive temporary file and `fsync` **that file**;
6. `os.replace` it onto `archive.current.json`, then **`fsync` the parent directory**;
7. update the runtime record's resolved `archived_db_path` and archive timestamp;
8. prune older generations **only after** the pointer is durable — that is, only after step 6's
   directory fsync returns.

A reader always resolves through the pointer, so it sees either the old generation or the new one
and never a half-rotated pair. In-place mode runs none of this.

**Signal handling:** the handler only sets a checkpoint-requested event — it never performs a SQLite
backup inside the handler. But "the main loop observes it at its next safe point" is not good enough
either: a single stage can run for minutes, and Slurm's grace window is typically 120 seconds, so
the checkpoint would arrive after the job was already gone. Instead a **dedicated checkpoint thread
with its own read-only SQLite connection** waits on that event and runs the **complete backup
transaction through step 7** immediately, concurrently with whatever stage is executing. Pruning
(step 8) follows afterward and is not part of the transaction: the snapshot is durable and
authoritative once step 6's directory fsync returns, and step 7 records it. Finalize the root record
only after the final archive has succeeded, or after the failure has been recorded in the record.
Test a simulated stage that runs longer than the scheduler grace period.

**All three backup triggers go through one coordinator.** Periodic, signal-triggered, and final
backups must serialize on a single checkpoint mutex owned by the checkpoint thread. Left
unsynchronized they can interleave and race pointer publication, generation pruning, and the
`active.json` update — producing a pointer naming a generation another trigger has just pruned. No
request ever starts a second concurrent transaction.

**But coalescing is only safe for periodic requests.** A snapshot that began at T₀ reflects the
database as of T₀. Merging a signal or final request that arrived at T₁ > T₀ into that in-flight
backup would silently drop every write between T₀ and T₁ — the archive would look successful while
missing exactly the work the checkpoint was meant to save. The rules are therefore asymmetric:

| Request arriving mid-backup | Behavior |
|---|---|
| **periodic** | may coalesce into the running snapshot — one interval of drift is already within the stated RPO |
| **signal-triggered** (grace signal, `SIGTERM`) | **must queue exactly one follow-up**; never merged into a snapshot that began earlier |
| **final** (run finalization) | **must queue exactly one follow-up**, and finalization waits for a snapshot that *began after database writes stopped* — not merely for the in-flight one to finish |

That last clause is the one that makes finalization meaningful: a final archive is only complete if
its backup started after the last write, so waiting on an earlier in-flight snapshot is not
sufficient no matter how recently it began.

**A staged run has no archive until its first checkpoint commits.** Requiring `archived_db_path` to
name "the current generation" is unsatisfiable in the window between launch and the first snapshot,
and treating the absent pointer as an error would make every staged run start in a failure state.
The record therefore carries an explicit `archive_status`:

| `archive_status` | `archived_db_path` | Missing/unreadable pointer means |
|---|---|---|
| `pending` — staged run, no snapshot committed yet | `null` | **expected**; not an error |
| `ready` — at least one snapshot committed | the generation the pointer names | **an error** |
| `n/a` — in-place mode | equal to `active_db_path` | n/a; there is no pointer |
| `stale` — terminal; an earlier snapshot committed but the final checkpoint failed | the last good generation | n/a; run has ended |
| `failed` — terminal; no snapshot ever committed | `null` | n/a; run has ended |

`archive_status` moves `pending` → `ready` only when step 6's directory fsync returns, so it never
advertises a snapshot that is not durable.

**Pointer failure modes are part of the contract.** While `ready`, a missing, malformed, or stale
`archive.current.json` must produce a precise "archive pointer unreadable" state rather than a
guessed filename. Test all three, plus the `pending` window.

**A terminal archive failure must be representable.** Finalization is permitted once an archive
failure has been *recorded*, but "`last.json` always carries a resolved archive path" cannot hold if
the very first snapshot is also the final one and it fails. `archive_status` therefore gains a
terminal value, and the two cases are distinguished:

| Situation at finalization | `archive_status` | `archived_db_path` in `last.json` |
|---|---|---|
| an earlier snapshot committed; the final checkpoint failed | `stale` | the **last good** resolved generation, plus `archive_error` describing the failed final checkpoint |
| no snapshot ever succeeded | `failed` | `null`, with `archive_error` — the GUI reports that **recovery data is unavailable** for this run rather than offering a broken link |

In both cases `last.json` must **never** name a node-local scratch path after finalization: that
storage is gone with the allocation, and a historical GUI following it would produce a confusing
"database missing" instead of an honest "this run has no archive". A resolved path in `last.json` is
required whenever one exists; `null` is permitted only under `failed`.

**Honest recovery-point objective.** "Resume without loss" is true only after a clean shutdown or a
grace-signal checkpoint. A hard node failure loses everything since the last periodic backup. State
it plainly in the docs: **at most one backup interval of ledger progress**. Test killing the process
midway through a backup and prove the previous archive still opens.

**GUI database resolution:**

- while the run is active → `active_db_path`;
- after terminal finalization → `archived_db_path`;
- `last.json` must **never** leave the historical GUI pointing at deleted node-local scratch.

### 2.11 Electron: upgrade first, then lock, then attach

The installed Electron **was 33.4.11** when this section was written. Step 5a has since
landed: it is now pinned exactly at **43.4.1** (`app/package.json`, `app/package-lock.json`),
above the CVE fix on every affected line and on a currently supported major. Step 5b has also now
landed in the implementation tree: `requestSingleInstanceLock()` is deployment-scoped, attachment
uses the authenticated connection descriptor, and closing an attached shell signals nothing. The
analysis below is retained because it records why the upgrade had to precede the lock.

**CVE-2026-34776 / GHSA-3c8v-cfp5-9885** (published April 2026, CVSS 5.3): on macOS and Linux, apps
calling `app.requestSingleInstanceLock()` are vulnerable to an out-of-bounds heap read when parsing
a crafted second-instance message; leaked memory can reach the `second-instance` handler. Limited
to processes running as the same user. Windows unaffected. **No app-side workaround** — patched in
38.8.6, 39.8.1, 40.8.1, 41.0.0.

This host is Linux and the current version is below every patched line, so the original Phase 5
would have *introduced* the vulnerability by adding the API call that triggers it.

**Version policy.** 38.8.6 is the *CVE floor*, not the *support floor*. Electron supports only the
latest three stable majors on an 8-week cadence, so by August 2026 the 38.x line is already outside
the support window. The release gate is: **exactly pin a patched release from a currently supported
major line**, verified against the Electron release schedule at implementation time — not a
`^`-range, and not 38.x by default. Upgrade and smoke-test packaging **as its own change**; a
five-plus-major jump inside a behavior PR conflates two failure modes.

**Then:**

- derive Electron's user-data / single-instance identity from the canonical deployment key, using
  the golden vectors of §2.1;
- call `requestSingleInstanceLock()` before `whenReady()`, before IPC registration with side
  effects, and before `ensureServer()`;
- a second launch for the same deployment focuses the existing window and quits; a different
  deployment may hold its own lock and window;
- `additionalData` may tell the first instance what was requested; it does not create a
  differently scoped lock;
- **verify the deployment key from `/healthz`.** A valid LM3 server belonging to a *different*
  deployment answering the configured port is its own named error ("wrong deployment on this
  port"), distinct from "an unrelated service is on this port". Neither may be attached to.

**Authentication — two permanent flows, one per client kind.** These are not alternatives to pick
between at implementation time; each client kind has exactly one supported path.

| Client | Flow | Status |
|---|---|---|
| Electron / local process | reads `connection.private.json` (§2.12) for its deployment | permanent |
| Browser (loopback or SSH-tunneled) | same-origin HTML token bootstrap: the server injects `<meta name="lm3-token">` into `index.html` for a loopback client with a loopback `Host` header | **permanent, not transitional** |

Revision 3 said the HTML-meta bootstrap was retained "only during the transition release". That
would have broken the cluster deliverable outright: `connection.public.json` deliberately carries no
token, `connection.private.json` is node-local on the compute node, and a user opening the tunneled
URL would have had no way to authenticate at all. The same-origin bootstrap already gates on
loopback client *and* loopback `Host` (`app.py:654-657`), which is exactly the tunneled case, so it
is the right permanent mechanism rather than a legacy one.

An interactive token prompt remains the explicit fallback when neither applies. A future
`lm3 connect` helper that reads the private descriptor over SSH and issues a one-time browser
bootstrap code is a possible addition, not a prerequisite.

**The token must never be logged or cached.** `_server_token()` currently prints the generated
secret at warning level — `log.warning("LM3 server token (set %s to override): %s", ...)` at
`app.py:289`. On a cluster that lands the bearer token in Slurm job output and retained container
logs, where it long outlives the allocation. Required:

- never log the raw token, at any level;
- log the **path to `connection.private.json`** instead, so a user can still find it;
- redact tokens from error messages, tracebacks, and `/healthz` diagnostics;
- serve the token-bearing HTML bootstrap with **`Cache-Control: no-store`**, not `no-cache`. The
  codebase already documents the distinction at `app.py:661-672` — `no-cache` still stores the
  response and merely revalidates it — and today that route sets `no-cache` (`app.py:672`), so a
  token-bearing page is written to the browser cache;
- test that the token appears in no server, setup, batch, or Slurm log.

Electron never invents a token and assumes an existing server accepts it, and never escalates to PID
signaling.

Electron shuts down only a server represented by its retained child handle with a matching instance
ID and deployment key.

### 2.12 Two connection descriptors, not one

Revision 2 contradicted itself: §2.11 put the bearer token in the connection descriptor, while the
cluster step required a descriptor *without* bearer tokens for persistent storage. Both are right
about their own context, so there are two artifacts with different names and different homes.

| | `connection.private.json` | `connection.public.json` |
|---|---|---|
| Location | deployment runtime dir (node-local, user-only) | may be written to persistent cluster storage |
| Contains | token, instance ID, deployment key, host, port, PID | deployment key, scheduler job ID, node, port, run ID, timestamps, tunnel instructions |
| Never contains | — | **any token or secret** |
| Read by | Electron, local clients | the user, for building an SSH tunnel |

**Creation, not correction — and it must rotate.** Exclusive-create at the final path works only on
the very first server launch; a restarted server would find the old file and be unable to publish
its new token and instance ID. The procedure is therefore:

1. create a **unique temporary sibling** in the same directory, user-only from its first byte —
   `os.open(tmp, O_CREAT|O_EXCL|O_WRONLY, 0o600)`, never written broadly and `chmod`ed afterward,
   which leaves a readable window;
2. write and `fsync` it;
3. `os.replace` it onto `connection.private.json`;
4. `fsync` the directory where the platform supports it;
5. on shutdown, delete the file **only if its instance ID still matches the stopping server**, so a
   restarted server's descriptor is never removed by its predecessor's late cleanup.

On Windows, create the temporary file with an ACL granting the current user only; POSIX mode bits
are not a substitute there. Test token rotation across a restart, and a stale descriptor left by a
crashed server.

Neither descriptor is ever merged into a runtime record (§3.2 forbids secrets in records, and the
public descriptor is a deployment artifact rather than run state).

### 2.13 Hardware setup requires a resolved config

The revision-2 schema allowed a `hardware_setup` root with no config. That is not merely permissive,
it describes a live crash: `app.py:577` passes `cfg = ... if job.cfg_path.is_file() else None`, and
`run_setup` dereferences the config unconditionally — `_fingerprint(cfg)` at `hardware_setup.py:145`
and `_chosen_gpu_indices(cfg)` at `:162`, which reads `cfg.compute` at `:662`. There is no
`cfg is None` guard anywhere in `hardware_setup.py`, so the GUI's Optimize Hardware action with a
missing config file raises `AttributeError`, caught at `app.py:583` and surfaced as an opaque job
error.

**Decision: optimized or calibrated setup requires a valid canonical config.** Model hashes, enabled
stages, `compute.devices`, and the output scratch location all come from configuration, so a profile
built without one would describe a machine the user is not going to run. The server resolves the
canonical settings path (§3.1) instead of passing `None`; if it cannot, the setup job fails with a
precise message naming the missing path rather than an `AttributeError`. `config` is therefore
required on the `hardware_setup` record.

#### Setup must run as a subprocess, not in the server

GUI hardware setup currently executes **inside the server process**: `app.py:574-581` defines an
`async def _run()` that calls `run_in_threadpool(run_setup, cfg, ...)`. There is no `Popen`, no
child, and no process group of its own.

That is irreconcilable with §3.3, which requires Stop to terminate the retained root process group
so a calibration tree dies as one unit. If the setup root *is* the server, "terminate the root
process group" means killing the server — along with every unrelated request it is serving.

**Decision: GUI hardware setup runs as a dedicated `lm3-setup` subprocess.**

- The subprocess acquires the `hardware_setup` root lease; the server retains its `Popen` handle and
  process group.
- Calibration runs inside that same process group and inherits the lease descriptor exactly as
  §2.2 describes.
- Setup progress reaches the UI through a **server-private append-only JSONL event log**, chosen over
  a live channel because it survives server and UI disconnects, cannot fill a pipe and stall the
  subprocess, and can be replayed on reconnect.
- Stop terminates the setup/calibration tree and leaves the server running.

This is also what lets invariant 13 drop its exception: with setup outside the server, **no** code
path makes the server a lease holder, and the rule becomes absolute rather than "absolute except
one case" — which is the kind of exception that erodes.

---

## 3. Core contracts

### 3.1 Runtime directory, deployment key, and resolver scope

Add `leafmachine3/core/runtime.py` and `leafmachine3/core/paths.py`.

**`LM3_RUNTIME_DIR` names the BASE directory.** The deployment directory is
`<base>/<deployment_id>`. This is stated normatively because both readings are plausible and they
differ in behavior: the base reading lets one node-local scratch path host several deployments,
which the Slurm example (`LM3_RUNTIME_DIR=${SLURM_TMPDIR}/lm3-runtime`,
`LM3_DEPLOYMENT_ID=slurm-${SLURM_JOB_ID}`) depends on.

Resolution order for the base directory:

1. explicit `runtime_dir` argument;
2. `LM3_RUNTIME_DIR`;
3. for a scheduler job, a private directory below node-local job scratch;
4. platform user-runtime location for a desktop/local deployment;
5. a documented per-user cache fallback, desktop only.

Keep it independent of the checkout and CWD. Create with user-only permissions where supported.
When scheduler metadata is present but no safe node-local directory can be resolved, **fail before
running** rather than silently using shared home storage.

**Network filesystems are refused, not merely documented.** `flock` semantics on NFS and similar
network mounts are unreliable, and an unreliable lock is worse than no lock because it looks like
one. Resolution detects a network-backed filesystem for the chosen base directory and **fails at
startup** with the detected filesystem type and the path. The single override is
`LM3_ALLOW_NETWORK_RUNTIME=1`, which logs a prominent warning naming the risk. This applies to the
desktop cache fallback as well as the cluster path.

Files per deployment directory:

- `activity.lock` — stable lock inode; never deleted during normal operation.
- `active.json` — bounded description of the current root lock holder.
- `children/<child_run_id>.json` — full subactivity lifecycle records.
- `last.json` — final record of the most recent **root** activity.
- `connection.private.json` — server connection descriptor with the token, user-only (§2.12).

Use a small cross-platform lease adapter: `fcntl.flock` on POSIX, and on Windows a **named global
kernel event** that carries both exclusion and lifetime across the activity tree. On Windows
`activity.lock` is a passive artifact only — never locked, never consulted (§2.2; the primitives are
not interchangeable, and the event is the sole authority there). Never infer exclusivity from file
existence. Write JSON via temp file + `fsync` +
`os.replace`; validate schema and bound field sizes on read.

**Resolver scope.** The canonical resolver must cover more than the server modules. It must also
own:

- `hardware_setup.HW_PATH`, currently the bare relative `Path("hardware_settings.yaml")`
  (`hardware_setup.py:36`) — CWD-dependent;
- `calibrate.DEFAULT_IMAGE_DIR`, currently the bare relative `Path("examples/images")`
  (`calibrate.py:41`) — CWD-dependent, and a packaged resource;
- the standalone hardware-setup and postprocessing CLIs;
- every other packaged-resource path.

Canonical variable names: `LM3_SETTINGS`, `LM3_HARDWARE`, `LM3_POSTPROCESS_SETTINGS`,
`LM3_SERVER_JOBS`, `LM3_RUNS_ROOTS`, `LM3_RUNTIME_DIR`, `LM3_DEPLOYMENT_ID`.

#### Normative precedence table

These chains are the fix for the original GUI/settings split, so they belong here rather than in
implementation. **No path falls back to the current working directory.** That single rule is what
makes "the GUI and the CLI disagree depending on where you launched them" impossible.

| Path | Precedence (first match wins) | On miss |
|---|---|---|
| next-run settings | 1. explicit argument / `--config` · 2. `LM3_SETTINGS` · 3. deployment workspace pointer · 4. `<user-config>/lm3/<deployment>/LM3_settings.yaml` · 5. *dev checkout only:* `<checkout root>/LM3_settings.yaml` | seed 4 from the packaged template, log the path loudly, continue. An explicit `--config` naming a missing file is always a hard error. |
| hardware profile | 1. `LM3_HARDWARE` · 2. `<user-config>/lm3/<canonical-deployment>/hardware_settings.<machine-key>.yaml` · 3. legacy adopt (one release, with a warning): **copy** a beside-the-settings profile into path 2 and use the copy | run setup once, as today |
| postprocessing settings | 1. `LM3_POSTPROCESS_SETTINGS` · 2. `<user-config>/lm3/<canonical-deployment>/postprocessing.yaml` | packaged defaults |
| server jobs root | 1. `LM3_SERVER_JOBS` · 2. `<user-state>/lm3/<deployment>/jobs` | create |
| runs roots (history) | 1. `LM3_RUNS_ROOTS` · 2. active/last runtime output roots · 3. `project.output.dir` from the resolved settings | empty list |
| runtime registry base | the five-step order above | startup error |
| calibration images | packaged resource via `importlib.resources` | packaging error |

Decisions embedded in that table, called out because each was previously ambiguous:

- **The hardware profile is deployment-scoped, not machine-scoped.** Revision 3 said machine-scoped
  on the reasoning that a profile describes the box. That is only half true, and the half that is
  false makes it shared mutable state: `run_setup` sizes GPU stages against the *selected*
  `compute.devices` subset (`hardware_setup.py:160-164`), instantiates only the stages the supplied
  config enables, and builds a staleness fingerprint that embeds config-derived model hashes
  (`_fingerprint` → `model_hashes=_model_hashes(cfg)` at `hardware_setup.py:616`, resolved through
  "the config's own stage-artifact resolution"). Two named deployments hold *separate* leases, so
  nothing stops them tuning concurrently — one pinned to GPU 0 and one to GPU 1 — while writing the
  same file and carrying each other's measurements forward. The deployment lock does not protect a
  path outside the deployment.

  The profile therefore lives at
  `<user-config>/lm3/<canonical-deployment>/hardware_settings.<machine-key>.yaml`.

  **`<machine-key>` must include host identity, not just hardware model — and node name is not a
  fallback, it is always included.** A university cluster routinely has many nodes with identical
  CPU/GPU models sharing one networked user-configuration filesystem while holding independent
  node-local runtime leases. Worse, a *container image* ships one baked `/etc/machine-id` to every
  node that runs it, so "trusted machine identity, else hostname" collapses to a single value across
  the whole allocation — and on CPU-only nodes there are no GPU UUIDs to break the tie either. Every
  node would then share one profile.

  The key is `sha256` truncated to 16 hex characters over a **canonical serialized input**, with
  every field always present and explicit markers for missing values:

  ```
  machine-key/v1
  os-machine-id: <value> | none        # /etc/machine-id, IOPlatformUUID, MachineGuid
  node: <SLURM_NODENAME or hostname>   # ALWAYS included, never merely a fallback
  gpu-uuids: <sorted NVML UUIDs, comma-separated> | none
  ```

  Because `node:` is unconditional, two containerized nodes sharing an image machine ID still derive
  different keys. Golden vectors pin the serialization, and a test covers exactly that duplicated-
  image-machine-id case on CPU-only nodes.

  Driver version, ORT/provider versions, selected devices, enabled stages, and model fingerprints
  stay in the profile's *contents* as staleness fields. They must not enter the machine key, or
  every config change would orphan a profile instead of invalidating it.

  The file's contents record the selected device set, enabled stages, model fingerprint,
  driver/runtime versions, and tuning inputs. A globally
  shared cache remains possible later, but it would need its own machine-wide lock and
  content-addressing by device/config tuning fingerprint — unnecessary complexity now.
- **`--config` affects only the settings path** for that invocation. Hardware (deployment-scoped,
  machine-keyed) and postprocessing (deployment-scoped) paths are unmoved unless separately
  overridden.
- **Missing settings seed from the packaged template** rather than failing, so a first-run install
  works; but an explicit `--config` pointing at nothing is a hard error, because the user stated an
  intent that cannot be satisfied.
- **A development checkout is detected from `leafmachine3.__file__`, not the CWD** — the package's
  parent contains the repo marker and is not under `site-packages`/`dist-packages`. That preserves
  `cd LM3 && python -m leafmachine3.server` ergonomics without reintroducing CWD dependence.
- **The deployment workspace pointer is a defined artifact, not a vague third source.** Since the
  original defect was redundant project-selection mechanisms, an undefined pointer would recreate
  it. Exactly:

  | Property | Value |
  |---|---|
  | Path | `<user-config>/lm3/<canonical-deployment>/workspace.json` |
  | Schema | `{"schema_version": 1, "settings_path": "/abs/path/LM3_settings.yaml"}` — nothing else |
  | Sole writer | the server's Settings API |
  | Update rule | temp sibling + `fsync` + `os.replace` |
  | Changed by | the explicit GUI "choose settings file" action only |
  | Never | declares, implies, or influences the **active run**; it selects next-run settings and nothing more |
  | If its target is missing | fail with a precise "selected settings file is missing" state naming the path, and let the user re-choose or clear the pointer |

  That last row matters more than it looks: silently falling through to the default settings file
  would mean the next run quietly uses a configuration the user did not choose — the exact class of
  surprise this plan exists to remove.

- **`lm3 serve` and Electron always pass the canonical settings path explicitly** rather than
  relying on any fallback.
- **Calibration images are a packaged resource.** `calibrate.py:41`'s bare relative
  `examples/images` becomes `importlib.resources`, so calibration cannot depend on the CWD.

See Appendix A for the eight paths that currently disagree.

### 3.2 Record schemas

Storage layout, settled:

| File | Contents |
|---|---|
| `active.json` | bounded **root** record, plus `current_child` and `last_child` **summaries** |
| `children/<child_run_id>.json` | full child lifecycle record |
| `last.json` | **roots only** |
| each run's own SQLite, plus the wrapper's summary TSV | durable per-run history (never the registry) |
| `GET /v1/runtime` | merges the bounded active tree for presentation |

A child summary is `{run_id, activity, state, started_at, run_name, run_dir}` — bounded so
`active.json` stays bounded however many children a root launches over its life.

#### Writer ownership — one writer per file

Parent and child share the **same open file description**, so the activity lock does **not**
serialize their JSON writes: two atomic replacements can still clobber each other. Exclusivity and
mutual exclusion are different problems, and the lease only solves the first. Ownership is therefore
assigned statically:

| File | Sole writer |
|---|---|
| `active.json` (incl. `current_child`, `last_child`) | the **root** process |
| `children/<child_run_id>.json` | that **child** |
| `children/<child_run_id>.grant.json` | the **root** writes it; the child *claims* it by atomic rename (§2.2), never by editing it |
| `last.json` | the **root**, at finalization |

Sequence:

1. Root writes `current_child` into `active.json` **before** launching the child.
2. Child writes only its own record for its whole life.
3. The runtime API follows the root's `current_child.run_id` and merges the latest child record.
4. On normal child completion the root moves `current_child` → `last_child`.
5. If the root dies, the child may finish **its own** record but must never modify or remove the
   root record.

**Retention.** `children/` records and consumed grant files would otherwise accumulate forever. On
root finalization, and again in any recovery transaction, the writer prunes them under these rules:
preserve anything referenced by `active.json` or `last.json`; preserve durable batch history, which
lives in each run's own DB and the wrapper's TSV rather than the registry; and remove expired grants and unreferenced
child records beyond a documented retention limit (default: the last 50 children per deployment).

**One documented exception: the recovery writer.** After a hard kill, the stale root record must be
finalized by somebody, and the root is gone. A process that (a) successfully acquires the now-free
activity lock, (b) re-reads the stale `active.json` and confirms it is genuinely abandoned, and
(c) completes the cleanup transaction, **temporarily becomes the recovery writer** for `last.json`
and `active.json` — before it begins its own root activity. That role is bounded by holding the
activity lock, so it cannot race a live root. It is the only case in which a non-root process writes
either file.

If a future feature genuinely needs two concurrent writers on one file, add a separate
`metadata.lock`. The activity lock cannot be reused for it, and neither the grant claim nor the
recovery writer needs it.

**Common fields** (all records):

```json
{
  "schema_version": 1,
  "run_id": "uuid",
  "activity": "pipeline",
  "activity_role": "root",
  "parent_run_id": null,
  "state": "starting",
  "launcher": "cli",
  "pid": 1234,
  "process_started_at": 1787932800.25,
  "started_at": "2026-08-28T12:00:00Z",
  "updated_at": "2026-08-28T12:00:01Z",
  "deployment": {
    "id": "slurm-482193", "scheduler": "slurm", "job_id": "482193",
    "step_id": "0", "node": "gpu042", "container_id": null
  },
  "error": null,
  "finished_at": null,
  "returncode": null
}
```

**Activity-specific requirements.** Revision 1 required `config` and `project` on every root, which
does not fit hardware setup. (It did not fit `batch` either; revision 14 removed that activity
outright, §2.3.) Corrected:

| Activity | Role | Required | Notes |
|---|---|---|---|
| `pipeline` | root | `config{path,sha256}`, `project{...}` | the ordinary case |
| `hardware_setup` | root | `config{path,sha256}`, `hardware{destination_path}` | config **required**; see §2.13 |
| `calibration_pipeline` | child | `parent_run_id`, `project{...}` | `run_name` is `_lm3_calibration` |

The `project` block, where required:

```json
"project": {
  "run_name": "acer_rubrum",
  "input_dirs": ["/abs/input"],
  "artifact_dir": "/abs/output/acer_rubrum",
  "active_state_dir": "/abs/output/acer_rubrum",
  "active_db_path": "/abs/output/acer_rubrum/acer_rubrum.sqlite",
  "archive_mode": "in-place",
  "archive_status": "n/a",
  "archive_pointer_path": null,
  "archived_db_path": "/abs/output/acer_rubrum/acer_rubrum.sqlite",
  "run_dir": "/abs/output/acer_rubrum",
  "log_path": "/abs/output/acer_rubrum/logs/lm3.log"
}
```

That example is a desktop run, hence `archive_mode: "in-place"`, a null pointer, and
`archive_status: "n/a"`. A staged (cluster) run carries `"archive_mode": "staged"`, a non-null
`archive_pointer_path`, and — only once `archive_status` is `ready` — an `archived_db_path` naming
the current generation. Before its first committed snapshot it is `archive_status: "pending"` with
`archived_db_path: null` (§2.10).

Rules:

- No bearer tokens, environment dumps, or other secrets. The connection descriptor is a separate
  0600 file and is never merged into a record.
- `run_id` identifies one invocation even when it resumes the same project DB.
- `activity_role` ∈ `root | child`. Only `root` records correspond to a lock acquisition; a child
  record describes a process holding an **inherited** description.
- `parent_run_id` is required for children and must match a live root.
- Scheduler/node/container fields are descriptive; they never authorize signaling across a host or
  container boundary.
- `config.sha256` fingerprints the bytes actually loaded.
- Paths are absolute and resolved. `tmp_dir` is deliberately absent (§2.7).
- `launcher` (`cli`, `python`, `server`, `legacy-job`) is descriptive, never an authorization
  decision. Revision 14 removed `batch` from this enum too: a process started by the shell wrapper
  is simply a CLI-launched pipeline, and LM3 has no reason to know that a project-specific shell
  loop happens to be its parent.
- `control` (`{mode, owner_instance_id}`) is capability metadata. A server may stop a run only
  under §2.5's handle-ownership rule.

### 3.3 Lifecycle

```
lock acquired → starting → running → done
                            ├──────→ error
                            ├──────→ stopped
                            └──────→ interrupted
lock released without finalization → abandoned (classified by the next reader)
```

Acquire **after** cheap config loading/validation has resolved project identity, and **before**
hardware profiling, orphan-worker cleanup, directory creation, database writes, model loading, or
GPU work. In `machine3()` that is between `cfg.validate()` and `ensure_hardware_profile(cfg)`
(`machine3.py:58-61`).

On failure raise a typed `RuntimeBusyError` carrying the sanitized current record; report it over
the status pipe when one is present (§2.4); exit the CLI with code **75**.

Hold the lease reference for the entire root activity — the locked open file description on POSIX,
the owner event handle on Windows — passing it (POSIX) or an inheritable duplicate of it (Windows)
only to approved subactivities (§2.2).

#### Finalization must not outrun the children

The inherited lease reference keeps the deployment occupied after a parent dies — which is correct — but
it creates a second hazard: if the root finalizes normally while a child is still alive, the child
still holds the lease reference while the root has already removed `active.json`. The deployment is then
**occupied but unidentifiable**: the API cannot say what is running, and the GUI shows idle while
the GPUs are pinned.

The contract:

- A root **must terminate or join every subactivity before it finalizes**, on the normal path and
  on the interrupted path (`SIGTERM`, `KeyboardInterrupt`) alike.
- `active.json` is **never removed while any approved child is alive**.
- Server Stop targets the retained **root process group** (POSIX) or **job object** (Windows), so a
  calibration tree stops as one unit rather than orphaning its child.
- If the root is hard-killed, the child keeps the lease and the **stale root record stays in place**
  — it is the only description of what is running.
- Once that last child exits and the lease becomes acquirable, the next reader classifies the root
  as `abandoned` under the cleanup lock.

Only then, in `finally`, atomically write the terminal record to `last.json`, remove or replace
`active.json`, and release the lease reference. A reader finding `active.json` while the lease is
acquirable
classifies the record as stale/abandoned under the cleanup lock, and must never signal the recorded
PID.

Three distinct tests, because they fail differently: hard-killed parent with a surviving child;
graceful `SIGTERM`/`KeyboardInterrupt` with a live child; and a server Stop of a
calibration tree.

### 3.4 Immutable launch manifest

After `build_dirs()` resolves the final paths, atomically write `<run>/logs/run_manifest.json`
containing: runtime schema and `run_id`; `parent_run_id` when a child; source config absolute path
and SHA-256; the complete effective config after defaults and overrides; all explicit overrides;
resolved project paths including the four storage roles and the now-settled `tmp_dir`;
LM3/Python/platform versions and launcher; start timestamp. Update only terminal fields at
completion.

On the cluster, the manifest is written to `artifact_dir` so it survives the allocation, even when
`active_state_dir` is node-local.

### 3.5 The relative-path contract

Revision 12 specified where the *runtime's own* files live but never said what a **relative path
written in a configuration** resolves against. That omission was not cosmetic: Step 1 unified the
settings-path resolution while `Config.resolve_path()` kept joining `Path.cwd()`, so
`build_dirs()` and `runtime.config_io.resolve_run_paths()` began returning **two different run
directories for one config in one process** — and the shipped default `output.dir` is the relative
string `runs`, so this is the default path, not an edge case. This section closes it.

**The contract.** Six rules, and they are exhaustive:

| # | Where the relative path appears | Resolves against |
|---|---|---|
| 1 | any path-valued field **written in a YAML settings file** | the **directory containing that settings file** |
| 2 | a relative **CLI argument** (`--config`, `--input`, `--output`, …) | the **caller's invocation CWD**, absolutized at the CLI boundary |
| 3 | a relative argument passed directly to `machine3()` (`input_dir`, `output_dir`, …) | the **caller's CWD**, normalized immediately on entry |
| 4 | a canonical environment path variable | **must already be absolute**; a relative value is an error |
| 5 | workspace pointer, runtime record, connection descriptor, launch manifest | **absolute only**, never relative |
| 6 | anywhere else | nothing — **no subsystem may join a configured path onto its own CWD** |

Rule 6 is the load-bearing one. Rules 1-3 decide the base *once*, at the boundary where the
provenance is still known; rule 6 forbids every later subsystem from deciding it again. A relative
value that survives past configuration load is a defect, not a deferred decision.

**Rules 2 and 3 must be applied before the merge.** `_cli_overrides()` layers CLI values into the
merged mapping, after which nothing can tell a CLI-supplied `output.dir` from a YAML-supplied one.
Absolutize at the boundary or the provenance — and therefore the correct base — is gone.

**Scope: every path-valued field, not just `project.output.dir`.** The contract governs at least
`project.input.dirs`, `project.output.dir`, `project.output.tmp_dir`, any active-state or
cluster-state path, every `modules.*.model.path`, every `models_dir`, and every path consumed by
validation, setup, inference, results, or postprocessing. In the current tree all of these already
funnel through `Config.resolve_path()` (`config.py:407-412`), which is therefore **the one seam**:
`core/validate.py:29,34`, `core/ingest.py:164`, `inference/factory.py:46,64,146`,
`setup/hardware_setup.py:602,607,803`, `server/settings_api.py:577-585`, and `config.py:540,543`
all call it. `Config` already records `source_path` (`config.py:324`), so rule 1 is implementable
there without threading a new argument through any caller.

**Required invariant, testable and CWD-independent:**

```
build_dirs(cfg).root    == resolve_run_paths(cfg).run_dir
build_dirs(cfg).db_path == resolve_run_paths(cfg).active_db_path
```

Two implementations that merely happen to agree under today's tests do not satisfy this. Either one
resolver serves both, or one normalized effective configuration feeds both.

#### The built-in `output.dir` default

`builtin_defaults()` ships `output.dir: "runs"` (`config.py:218`). Under rule 1 that default would
resolve beside the settings file — and §3.1 puts the seeded settings file at
`<user-config>/lm3/<deployment>/LM3_settings.yaml`, so a first run would write its results **into a
hidden configuration directory**. That is unacceptable, and it is an artifact of a *default* being
treated as if the user had written it.

**Decision: the built-in default becomes `auto`**, matching the existing `tmp_dir: auto` idiom, and
`auto` resolves to an intentional, documented location:

| Deployment | `output.dir: auto` resolves to |
|---|---|
| development checkout (detected from `leafmachine3.__file__` per §3.1, never the CWD) | `<checkout root>/runs` |
| installed package / container | `<user-data>/lm3/<canonical-deployment>/runs` |

`<user-data>` is `$XDG_DATA_HOME` or `~/.local/share` on Linux, `~/Library/Application Support` on
macOS, `%LOCALAPPDATA%` on Windows. It is **never** the user-config directory. The dev-checkout row
preserves today's `<repo>/runs` behavior byte for byte, so no existing checkout moves.

An *explicit* relative `output.dir` written by a user still follows rule 1. Only the absent-key
default is `auto`.

#### Example configurations must be migrated, not left to luck

`examples/*.yaml` currently depend on being launched from the repository root: five configs carry
`input.dirs: [examples/images]` and `output.dir: examples_out`, and **all twelve** carry
`models/...` model paths and `models_dir: models/ruler_classifier`. Under rule 1 those become
`examples/examples/images` and `examples/models/...` — silently wrong.

Migrate them so their **effective locations are unchanged**: relative to `examples/`, the values
become `images`, `../examples_out`, and `../models/...`. A test must assert that each migrated
example resolves to the same repository file it resolved to before, and must run from a CWD that is
not the repository root.

The absolute-path configurations (`LM3_settings_global_greening.yaml`, and the seven `examples/`
configs that already use `/datac/...`) are unaffected by rule 1 and must be proven unchanged.

---

## 4. Implementation sequence

Each step must leave the tree working.

### Step 1 — Characterization, isolation, and path unification

- Endpoint-contract tests for `/v1/run/active`, `/v1/run/start`, `/v1/run/stop`, `/v1/status`,
  `/v1/runs`, `/v1/settings`.
- Characterization tests for the current settings-path split, configured-project precedence, and
  CLI invisibility, marked expected-to-change.
- A deterministic CLI mock-pipeline baseline: DB rows, resume behavior, output inventory.
- **Test isolation, one method.** `tmp_path` is function-scoped, and a session fixture using
  `tmp_path_factory` does not run before test modules are imported — so neither can satisfy "before
  any test imports `machine3`". Set the environment in `conftest` at **import time** (or
  `pytest_configure`), and tear it down at session end. Export `LM3_RUNTIME_DIR` and
  `LM3_DEPLOYMENT_ID`, giving **each pytest-xdist worker its own deployment ID** derived from
  `PYTEST_XDIST_WORKER`. Add a test asserting the default runtime directory is never touched. This
  is not optional: `tests/test_pipeline_mock.py` calls `machine3()` at lines 46, 335, 346, 359 and
  360, and `tests/conftest.py` currently has no environment isolation, so without it the suite
  would take the user's real lease and could block a production run.
- Add `paths.py` with the resolver, covering the **full scope in §3.1** — including
  `hardware_setup.HW_PATH` and `calibrate.DEFAULT_IMAGE_DIR`, which are bare relative paths today.
- Define `LM3_RUNTIME_DIR` as a base directory, normatively, with a test.
- **Unify fallbacks, not just names.** The split is 6-way across 4 modules with 3 different
  fallbacks (Appendix A). Renaming variables while leaving three fallbacks reachable from three
  CWDs preserves the bug.
- Migration for one release: honor `LM3_SETTINGS_PATH` alone with a deprecation warning; if it and
  `LM3_SETTINGS` resolve to the same file, use it and warn once; if they resolve differently, fail
  server startup with a precise split-brain error. Same for `LM3_HARDWARE_SETTINGS` vs
  `LM3_HARDWARE`.
- Replace local path logic in `settings_api.py`, `metrics_api.py`, `progress_api.py`,
  `postprocess_api.py`, `results_api.py`, `app.py`/`JobManager._base_settings()`, `hardware_setup.py`,
  and `calibrate.py`.
- Fix the stale `build_dirs` docstring (§2.7).
- Expose resolved paths in startup logs and `/healthz` diagnostics.

Exit gate: from any supported CWD every subsystem reports the same canonical settings path;
conflicting legacy variables stop startup; pytest provably cannot see the default runtime
directory, including under xdist.

### Step 2 — Runtime primitives only (not wired to production)

- `RuntimeLease`, `RuntimeRecord`, `RuntimeBusyError`, atomic record IO, POSIX and Windows lease
  adapters, deployment-key derivation, capability mint/validate, inherited-descriptor plumbing,
  compatibility reader.
- `Config.to_dict()` and the bounded pure path resolver of §2.7, emitting the four storage roles.
- Tests: POSIX and Windows lock behavior; atomic writes and schema validation per activity type;
  contention using real subprocesses; stale/abandoned classification; **descriptor inheritance —
  approved child inherits, executor workers do not**; **parent killed while an inherited child
  lives keeps the deployment occupied**; **graceful root finalization refuses to run while a child
  is alive**; grant-record single use, expiry, and forgery rejection; writer-ownership races (root
  and child writing concurrently must not clobber); higher `schema_version` handling; record
  redaction and field bounds; deployment-key canonicalization plus golden vectors shared with
  Electron; network-filesystem refusal and its override.

Exit gate, stated in the terms §1.1 and §3.5 now make available — the original wording contradicted
both, and a gate nobody can honestly pass is worse than no gate:

- **Linux primitives pass**, in isolation, on the qualification target.
- **The Windows adapter's behavior passes against the injectable Win32 fake** — ordering, handle
  lifetime, the close-on-every-failure-path rule, the serialized spawn window, the
  `CompareObjectHandles` rejection. This is a test of our *model* of the object manager, not of the
  object manager.
- **Native Windows and macOS validation is deferred** (§1.1) and does not gate this step. Neither
  may be reported as "passed".
- **No runtime-v2 wiring has entered a production execution path**: nothing outside
  `leafmachine3/core/runtime/` and `tests/` imports it. This is the property the gate exists to
  protect, and it is checkable.

The gate deliberately no longer says "no production entry point has changed". `machine3.py` changed
under §3.5 — `_cli_overrides()` absolutizes relative CLI overrides against the caller's CWD before
the merge, because after the merge the provenance that decides the base is gone. That is
path-contract work, imports nothing new, and is unrelated to the runtime primitives this step
delivers. Byte-identity was a proxy for "no runtime-v2 wiring"; the real property is stated directly
above, so the proxy is retired rather than quietly violated.

### Step 3 — Execution integration

**Entry tasks, ordered.** Step 2 deliberately shipped primitives that are defined but not yet
enforced. Each becomes wrong in a different way once Step 3 starts publishing records, so they are
listed here as explicit first work rather than left to be rediscovered during integration:

1. **Enforce `STATE_TRANSITIONS` before the first writer exists.** It is defined in `_types.py` and
   consulted by nothing. Step 3 is what begins publishing `starting → running → terminal` from
   `machine3()`, hardware setup and the handshake; if enforcement lands after the record builders, every
   builder is written against an unenforced machine and inherits the gap. A root must not be able to
   publish `done` and then `running`.
2. **One canonical `DeploymentInfo` builder.** Nothing today guarantees `DeploymentInfo.id` receives
   the *canonical* deployment key rather than the raw `LM3_DEPLOYMENT_ID`. Two call sites that
   differ produce two registry directories for one deployment, which is the failure the key exists
   to prevent. Build it in one place and give it no other constructor.
3. **One subprocess-argument merging helper for every child launch.** The lease reference (POSIX
   `pass_fds`, Windows `STARTUPINFOEX` allowlist) and the launch status pipe (§2.4) both need to
   inject arguments into the same `Popen` / `CreateProcess` call. Supplied independently, the second
   silently discards the first's handle and the child starts without a lease it believes it has.
   One helper composes them, and it is the only thing allowed to.
**Two tasks moved out of this list.** They were listed here in revision 13 and do not belong:
ordinary pipeline execution neither opens a server port nor authorizes Stop from a registry record,
so neither should block wrapping `machine3()`.

- **`process_start_time()` returning `0.0` must mean UNKNOWN, never a match** → **withdrawn in
  revision 15, not merely moved.** It was a real hazard only because §2.5 compared creation times;
  now that control authority is handle ownership alone, nothing compares them and the `0.0` case
  cannot mislead anything. `process_started_at` remains in the record as descriptive provenance.
  Deleting the check is safer than keeping one that silently degrades to "always matches" wherever
  `psutil` is absent.
- **Wire `resolve_port()`** → **Step 5b**, with server and Electron startup. Gate 41 (a named
  non-default deployment without `LM3_PORT` fails at startup) is a startup decision, and 5b already
  owns `/healthz`, the deployment key and the connection descriptor.

Either may be implemented early; neither gates Step 3.

**Entry prerequisite (§3.5).** Step 3 wires the runtime record's paths in while `machine3()`
still creates `build_dirs()`'s paths. Until `build_dirs(cfg).root == resolve_run_paths(cfg).run_dir`
holds independently of the process CWD, the record would name a directory the run never creates.
The §3.5 contract must land, with its regression tests, before this step opens.

Revisions 1-13 merged this with batch ownership into one shipping unit, because splitting them left
a window in which the shell batch took and released a lease per species. Revision 14 removed the
batch orchestrator entirely (§2.3), so that window is now the accepted behavior and the merge has no
purpose. Step 3 is execution integration alone.

Initially keep the integration behind `LM3_RUNTIME_V2`, defaulting off until its exit gate passes —
the flag is about landing safely rather than about waiting for the batch. That gate has now passed;
the cutover status below is normative for the current default.

- Wrap `machine3()` (the public function, not just `main()`), so direct Python callers inherit it.
- Wrap standalone and GUI hardware setup as a root `hardware_setup` activity.
- Add inherited subactivities for `ensure_hardware_profile` auto-setup and for calibration; make
  `calibrate._run_pipeline` pass the lease reference and capability, and give the child a **provisional
  profile** (§2.2).
- Make requested-calibration failure loud.
- Implement the **launch handshake** (§2.4) in `POST /v1/run/start`: control pipe for status, child
  stdout/stderr to the server-private request log, run-directory creation only after `acquired`.
- Implement root finalization ordering (§3.3): join children before finalizing; Stop targets the
  root process group / job object.
- Resolve the canonical config for hardware setup instead of passing `None`, and move GUI setup out
  of the server into an `lm3-setup` subprocess with a retained handle (§2.13).
- Enforce the staging boundary and make invariant 14 testable.
- Extend `_cli_overrides()` and the CLI parser with `--run-name`.
- Publish `starting` at acquisition; move to `running` when the DB and log paths exist; finalize on
  normal return, exception, and `KeyboardInterrupt`.
- Reduce the server's private state to a `_ManagedChild` control handle only.
- Teach `run_global_greening.sh` to distinguish exit **75** (another root holds the deployment — a
  retry) from a genuine species failure, and record the two differently in the summary TSV (§2.3).
  The script otherwise stays as it is: it is a convenience wrapper, not an execution engine.

Exit gate: a helper subprocess holding a root lease makes CLI, direct Python, server start, and
legacy jobs all refuse a second **root** activity — while calibration under a parent lease still
succeeds; a busy start returns 409 with the winner's identity; and a species launched by
`run_global_greening.sh` against a busy deployment exits 75 and is recorded as retryable rather than
failed.

**Cutover status.** This exit gate is implemented and regression-pinned, including execution of the
real shell wrapper (gates 47/48) and two concurrent deployment processes carrying disjoint explicit
GPU plans (gate 43). `LM3_RUNTIME_V2` therefore defaults ON; explicit `0` retains the transition
path until its scheduled 3.1.0 removal. A broken runtime import fails visibly rather than silently
selecting that unsafe compatibility path.

### Step 4 — Registry-backed server and status

- Add `GET /v1/runtime` returning `{active, last, next_run_settings, server, diagnostics}` with the
  bounded activity tree.
- Keep `GET /v1/run/active` as a compatibility projection with deprecation metadata.
- `can_stop` derives only from §2.5: a retained live child handle for this `run_id`, launched by
  this `instance_id`. No PID comparison, no creation-time comparison.
- `POST /v1/run/stop` rejects observer-only runs with 409/403 and never synthesizes a process group
  from registry JSON.
- Change `progress_api.resolve_run()` precedence to: explicit `db=`/`run=`; active runtime record;
  explicitly selected historical run; canonical next-run project for an idle prepared project;
  filesystem discovery last, labeled `source: discovered` and never able to authorize control.
- Replace `_BOUND`, `_BOUND_STATE`, `bind_run()`, and `_from_job_source` with a registry reader.
- **Delete the re-adoption machinery** that handle ownership (§2.5) makes dead: `_adopt()`
  (`metrics_api.py`, ~45 lines), `_pid_alive()` and its `psutil` / `os.kill(pid, 0)` fallback
  (~19 lines), and `_Run.persist()`'s `create_time` plus the on-disk state file whose only stated
  purpose is "so a RESTARTED server can re-adopt a still-running LM3". A restarted server is an
  observer; there is nothing for it to re-adopt. Reduce the server's private state to the
  `_ManagedChild` control handle §4 Step 3 already calls for.
- Use `run_id` in snapshot cache keys; follow the active `run_id` for logs; on completion follow
  `last.json` rather than the most recently modified unrelated run.
- Remove the duplicate progress-router `GET /v1/runs`, keeping the `results_api` route, and delete
  the registration-order workaround at `app.py:613-615` that exists only because of the duplication.
- `results_api.run_roots()` uses the shared resolver and includes active/last runtime output roots.
- Implement §2.9 unknown-schema behavior and §2.8 postprocessing guards.

Exit gate: changing `project.run_name` during an active run does not move status, logs, results, or
postprocessing away from the active DB.

### Step 5 — Electron prerequisite, then ownership

- **5a:** pin and upgrade Electron to a patched release on a currently supported major line, and
  smoke-test packaging, as its own change (§2.11).
- **5b:** deployment-scoped `requestSingleInstanceLock()`; server `instance_id`; `/healthz`
  returning service, protocol version, instance ID, **deployment key**, and ownership mode;
  expected-instance-ID handshake for spawned servers; the connection-descriptor auth flow;
  attach-as-client for a matching deployment; named distinct errors for "wrong deployment on this
  port" and "unrelated service on this port".
- **Own gate 13 here.** §2.11 already requires it and §8 gate 13 already states it, but no step
  in this sequence claimed it, so it was implemented nowhere: never log the raw bearer token at
  any level; log the path to `connection.private.json` instead; redact tokens from error
  messages, tracebacks and `/healthz` diagnostics; serve the token-bearing HTML bootstrap with
  `Cache-Control: no-store` (token-free static assets may keep revalidation caching); and test
  that the token appears in no server, setup, batch, or Slurm log.
- Delete the policy that attached servers are killed on quit. Remove PID-derived shutdown
  authority. `LM3_KEEP_SERVER` may survive as an owned-server lifecycle preference but is no longer
  an ownership safety switch.

Exit gate: closing an Electron window attached to `lm3 serve` leaves that server and any run alive;
a second launch for the same deployment creates no second server or window; a second launch for a
different deployment does.

### Step 6 — Renderer redesign

Replace top-bar state with `runtime.active`, `runtime.last`, `view.mode`, `view.runRef`, `settings`,
`busy`. Remove `openFresh()`, sticky `state.newRun` suppression, run-name blanking without writing
YAML, deriving run activity from a status snapshot, and Close/New paths that stop a job. Implement
§2.6. "New run" becomes "Prepare next run". Add a run selector over the existing results run list.

Exit gate: opening the GUI during any CLI run immediately shows that run, while editable fields
continue to show — and clearly label — next-run settings.

### Step 7 — Compatibility cleanup

- Retain `/v1/run/active` for one declared transition release, with a **named release owning its
  removal**.
- Audit actual external `/v1/jobs` consumers. Helper *definitions* in `api.js` are not consumers
  (Appendix B, C2). Removal is a decision about external users, not in-tree call sites.
- Keep `LM3_settings_gg_watch.yaml` as a manual operational artifact until follow-active is
  validated; **no migration code deletes it**.
- Retire the deprecated `db_path` projection in favor of `active_db_path`.

### Step 8 — Single-user cluster reference deployment

Proceed with the original cluster plan — node-local runtime directory, OCI/Apptainer image,
`deploy/slurm/lm3.sbatch`, `connection.public.json` (§2.12, token-free) beside the persistent run,
loopback binding plus SSH tunnel, container health probes — now built on the §2.10 storage model
rather than an ad-hoc `state_dir`. Deliver generational online backup, grace-signal checkpointing,
the stated RPO, and a tested restore/resume procedure.

---

## 5. API migration contract

Add `GET /v1/runtime` — canonical runtime, bounded activity tree, last-run, next-settings,
diagnostics.

Keep temporarily: `GET /v1/run/active` (compatibility projection, deprecation header);
`POST /v1/run/start` (handshake-backed, 409 on busy); `POST /v1/run/stop` (strict `can_stop`: handle ownership only).

Consolidate: one `GET /v1/runs` owned by `results_api`; one settings resolver for all settings
routes; one project-reference shape shared by runtime, progress, results, and postprocessing.

Do not overload a single field named `active` to mean both "the process is live" and "the
historical run selected in the UI". Use `runtime.active` and `view.selected`.

---

## 6. Failure and race handling

- **Two simultaneous root starts:** one acquires; the other exits 75 before any execution-owned mutation.
- **CLI vs GUI start race:** child lease acquisition decides; the losing API returns 409 with the winner's identity via the handshake.
- **GUI start during an inter-species gap:** the GUI wins; the next species exits 75 and is
  recorded as retryable. Re-running the batch picks it up, because the skip check is by DB state
  (§2.3). Deliberately accepted rather than prevented.
- **Root parent killed while a subactivity runs:** the inherited description keeps the deployment occupied until the child exits, and the stale root record stays in place as the only description of what is running; no second root may start. Once the child exits the root is classified `abandoned`.
- **Root finalizes gracefully while a child is alive:** cannot happen — the root joins or terminates its children first, so the deployment is never occupied-but-unidentifiable.
- **Root and child write records concurrently:** impossible by ownership (§3.2); the shared lock deliberately is not relied on to serialize writes.
- **Replayed or expired capability:** the grant is single-use with an expiry; a second use fails closed.
- **Child emits heavy startup output before the handshake:** stdout/stderr go to a file, not a pipe, so it cannot deadlock the handshake.
- **Backup interrupted mid-write:** the previous archive generation is intact; the temporary sibling is discarded.
- **Named deployment started without `LM3_PORT`:** startup error, not a silent collision on 8765.
- **Runtime directory on a network filesystem:** startup error unless explicitly overridden.
- **Hardware setup with no resolvable config:** precise error naming the missing path, not an `AttributeError`.
- **Stop pressed during GUI hardware setup:** the setup/calibration subprocess tree dies; the server keeps serving.
- **Two named deployments tune concurrently:** each writes its own deployment-scoped profile; neither reads or overwrites the other's.
- **Handshake times out after the child won the lease:** the server terminates the child, reconciles any record it published, and only then reports failure — no unlaunched run survives.
- **Grant replayed:** the claim rename finds no source file and fails closed.
- **Inherited descriptor is not really the lock:** `fstat` inode comparison plus a non-blocking re-assert rejects it.
- **Server restarts and must publish a new token:** temp sibling + atomic replace rotates the private descriptor; the predecessor's late cleanup cannot delete it.
- **Node killed between archive generations:** the atomic pointer still names a complete snapshot.
- **Grace signal arrives during a minutes-long stage:** the dedicated checkpoint thread archives immediately rather than waiting for the stage boundary.
- **Calibration under a parent lease:** succeeds via inherited lease reference + capability; an unauthorized child path fails closed.
- **First-ever calibration with no profile on disk:** the child binds the provisional profile and does not re-enter setup.
- **Requested calibration fails:** loud, nonzero, no silent heuristic fallback.
- **SIGKILL of a lone root:** OS releases the lock; the next reader classifies stale JSON as abandoned; never signals the PID.
- **Lock descriptor leaked into an executor worker:** prevented by non-inheritable defaults plus explicit clearing; covered by a regression test.
- **PID reuse:** process creation time plus lock state prevent a stale record from authorizing control.
- **Malformed/truncated JSON:** report diagnostics, use lock state for exclusivity, never guess a PID.
- **Higher `schema_version`:** §2.9.
- **Registry directory unavailable:** fail before starting work; never run unregistered.
- **Server restarts during a run:** observes, `can_stop=false`.
- **Electron crashes:** an owned-server watchdog may stop the server; the pipeline lease survives and stays observable.
- **Wrong LM3 deployment answering the configured port:** named error; never attached to, never signaled.
- **Settings edited or deleted mid-run:** runtime stays pinned to its launch manifest.
- **Active DB not yet created:** state is `starting`; the GUI shows launch identity without falling back.
- **Batch transition:** `last_child` is shown until the next child record appears.
- **Postprocess against the active run:** refused (§2.8). Two `read_write` tools on one completed run: serialized.
- **Slurm preemption / wall-time kill:** grace-signal checkpoint plus periodic backup bound the loss; resume from `archived_db_path`.
- **Pre-registry run during cutover:** a legacy LM3 process holds no lease, so a runtime-v2 run could start beside it. **Hard rule: runtime v2 must not be enabled while a pre-registry run or a resumable batch is still executing from the old environment.** This belongs in the release notes and the deployment instructions, not only here. Discovery may *display* such a run as unverified history; it can neither block nor authorize control, which is precisely why the operational rule is required.

---

## 7. Test matrix

**Unit.** Every row of the §3.1 precedence table, including the no-CWD-fallback rule and dev-checkout
detection from `__file__`; legacy variable conflict errors; `LM3_RUNTIME_DIR` base-vs-final
semantics; deployment-key canonicalization (`/`, `..`, Unicode, over-length, empty) plus golden
vectors shared with Electron; network-filesystem refusal and override; named-deployment-without-port
error; POSIX and Windows lock adapters; atomic writes and per-activity schema validation;
subprocess lock contention; stale/abandoned classification; descriptor inheritance (approved child
yes, executor worker no); **root killed while inherited child lives**; **graceful finalization
blocked by a live child**; grant claim-by-rename (single winner, replay fails closed), post-claim expiry
re-check, and forgery rejection; inherited-descriptor identity (`fstat` inode match, non-blocking
re-assert, rejection of an independently opened handle) on POSIX **and** Windows; writer-ownership
concurrency; recovery-writer cleanup after a hard kill; `--run-name` precedence over YAML; manifest hash and effective-config correctness;
runtime→`RunRef` mapping; `can_stop` handle-ownership match, and a run this server did not
launch reporting `can_stop: false` with a reason; unknown
`schema_version` reader behavior; `connection.private.json` created 0600 from first write, rotated across a restart, and not deleted
by a predecessor's late cleanup; deployment-key ASCII-slug determinism against the JavaScript
implementation; `LM3_DEPLOYMENT_ID` unset ≡ `default`; workspace-pointer schema and atomic update.

**API integration.** Handshake returns 200 on acquire and 409 on busy, with the winner's record; a
child that floods stdout before its status line still completes the handshake; a busy start creates
no execution-owned directory but may stage privately; hardware setup with no resolvable config
fails with a precise message; server Stop of a calibration tree stops the child too; **Stop during GUI
hardware setup kills the setup subprocess and leaves the server serving**; a handshake that times
out after the child acquired the lease leaves no surviving run; a tunneled loopback browser session
authenticates through the same-origin meta bootstrap and is served `Cache-Control: no-store`;
a missing, malformed, or stale archive pointer produces a precise error rather than a guessed path;
a desktop run stays in-place with a null pointer and copies nothing;
a resumed run of the same project cannot reuse an earlier generation filename;
a losing Windows contender releases its event handle so the deployment frees when the winner exits;
a Windows worker outliving the root and its subactivity does not keep the deployment occupied;
a staged run before its first snapshot reports `archive_status: pending` rather than an error;
concurrent periodic, signal, and final backup requests serialize instead of racing; a final request
arriving mid-backup queues rather than coalescing, so the archive includes writes made after that
backup began; a run whose only snapshot failed finalizes as `archive_status: failed` with a null
path and the GUI says recovery data is unavailable;
a workspace pointer whose target has been deleted fails visibly instead of falling back; an externally held test
lease appears in `/v1/runtime` and disables Start; a GUI-managed mock child registers, streams
status, and can be stopped; a restarted server observes but cannot control; active runtime outranks
a conflicting YAML project; changing YAML mid-run moves nothing; an explicit historical selector
overrides follow-active for that request only; one `/v1/runs` route; all modules report one settings
path; legacy env conflicts fail loudly; postprocess against the active run is refused while a
different completed run is allowed; two `read_write` postprocessors on one completed run serialize.

**Electron/renderer.** Second instance of the same deployment focuses the first; two named
deployments each get a window; an unrelated port occupant errors; **a valid LM3 server for a
different deployment on the port errors distinctly**; an attached manual server is never killed on
Close; an owned server is stopped only after identity verification; Close never calls
`/v1/run/stop`; external runtime shows immediately; next-run settings stay visible and distinct;
run transitions update by `run_id`; history selection and Follow-active behave independently.

**Platform CI.** Lock adapter and Electron single-instance behavior on Linux, macOS, and Windows.

**End-to-end (mocked compute).** CLI first / GUI second; GUI first / browser second; direct Python
first / API observer second; crash GUI and server while the pipeline continues, then reopen; a
second root start attempted from every entry point; hard-kill then abandoned classification then
resume; **first-ever calibration with no hardware profile, launched from an arbitrary CWD**; a
a three-species mock batch loop in which each species is an ordinary root, including one species
launched against a busy deployment that exits 75 and is recorded as retryable (§2.3); DB/output content matches the pre-refactor baseline; two Slurm job
namespaces for one user do not contend while two starts inside one namespace do; **node-local DB
snapshot, simulated preemption, restore and resume**; **process killed midway through a backup and
the previous archive generation still opens**; **a grace signal delivered during a stage longer than
the scheduler grace period still produces an archive**; **two named deployments tune hardware
concurrently without touching each other's profile**; **calibration images resolve from an installed
wheel/container image, not only a source checkout**; **the bearer token appears in no server,
setup, batch, or Slurm log**; **the launch handshake and process-tree Stop pass on Windows CI**.

---

## 8. Mandatory regression gates

The new runtime is not enabled by default until **all** of these pass:

1. Calibration succeeds under a parent lease.
2. A requested calibration failure is visible and exits nonzero.
3. First-ever calibration with no profile does not re-enter setup inside the child.
4. Killing a root parent while an inherited child runs keeps a second root blocked until the child exits, and the root record survives as the description of what is running.
5. A root cannot finalize gracefully while an approved child is alive; `active.json` is never removed under a live child.
6. Server Stop of a calibration tree stops the child, not just the root.
7. An executor worker never holds the lease reference — the POSIX lock descriptor or the Windows lease-event handle.
8. A grant is single-use by rename and expires; a replayed or forged capability fails closed.
9. A bogus inherited lease reference is rejected: on POSIX a descriptor that is not the real lock inode, on Windows a handle that `CompareObjectHandles` shows is not the canonical lease event.
10. Stop during GUI hardware setup kills the setup subprocess and leaves the server serving.
11. Two named deployments tune hardware concurrently; neither reads nor overwrites the other's profile.
12. Nodes with identical hardware but different host identity derive different machine keys.
13. The raw bearer token appears in no log, and the HTML bootstrap is served `no-store`.
14. A missing, malformed, or stale archive pointer fails precisely; `last.json` carries a resolved archive path whenever `archive_status` is `ready` or `stale`, and `null` only under `failed`.
15. A workspace pointer whose target is gone fails visibly rather than falling back to defaults.
16. An invalid child cannot consume a valid child's grant.
17. The launch handshake and process-tree Stop pass on Windows CI, not only the lock adapter.
18. A desktop run resolves `archive_mode: in-place`, a null pointer, and performs no archive copying.
19. A staged run survives node failure after the pointer fsync: the pointed-to generation opens cleanly.
20. Two containerized nodes sharing one image machine ID derive different machine keys, including CPU-only nodes.
21. A staged run before its first snapshot reports `pending` with a null archive path, not a failure.
22. A child that fails descriptor validation never renames the grant, leaving it claimable by the legitimate child.
23. Overlapping backup triggers serialize through one coordinator; no pointer ever names a pruned generation.
24. A final or signal checkpoint arriving mid-backup queues instead of coalescing, and finalization waits for a snapshot begun after writes stopped.
25. A run whose only snapshot failed finalizes as `failed` with a null archive path; one with an earlier good snapshot finalizes as `stale` naming it.
26. `last.json` never names a node-local scratch path after finalization.
27. **On Windows**, killing the root while a subactivity runs leaves the deployment occupied until that child exits — verified against the Windows adapter itself, not inferred from the POSIX result.
28. **On Windows, that exclusion holds across logon sessions**: a contender in a second session of the same user is refused while the orphaned child lives.
29. On Windows, the lease event is created in `Global\` with a current-user-only ACL, is destroyed when the last handle in the activity tree closes, and a failure to create it is a startup error rather than a `Local\` fallback.
30. Two different Windows users on one machine get distinct leases (SID-hashed name) and do not block each other.
31. A Windows contender that loses acquisition closes its event handle: after the winner exits, the deployment is free even though the loser's process is still alive.
32. A Windows executor worker does not inherit the lease event handle, and `LM3_LEASE_EVENT_HANDLE` is absent from its environment.
33. **A Windows worker spawned by the ROOT does not inherit the lease**: with an ordinary worker still alive after the root and its approved child have exited, the event disappears and another root acquires successfully.
34. A failed subactivity spawn on Windows closes the parent's inheritable duplicate; no lease reference is leaked on any error path.
35. A resumed invocation of the same project cannot reuse an archive generation filename; exclusive creation turns a collision into an error.
36. A busy start returns 409 with the winner's identity, not a late failure.
37. A busy loser creates no execution-owned directory, DB, or CUDA context; private staging still works.
38. A child flooding stdout before its status line does not stall the handshake.
39. Default deployments contend.
40. Explicitly different deployment IDs do not contend, and their canonical keys match Electron's golden vectors.
41. A named non-default deployment without `LM3_PORT` fails at startup.
42. A runtime directory on a network filesystem is refused without the explicit override.
43. Two deployments pinned to disjoint GPUs run safely side by side.
44. No resolved path falls back to the current working directory.
45. Hardware setup without a resolvable config fails with a precise message, not an `AttributeError`.
46. Pytest can neither observe nor block a real run, including under xdist.
47. A three-species mock batch runs each species as an ordinary root; a species launched against a busy deployment exits 75 and is recorded as RETRYABLE, distinctly from a species failure.
48. The batch script preserves skip, resume, failure continuation, logs, and summary TSV — unchanged by this refactor, and pinned so the shell wrapper cannot regress while everything around it moves.
49. Postprocessing against the active run is refused; against a different completed run it works.
50. A higher-version runtime record blocks Start without granting control.
51. A killed lone owner releases the lock and is classified abandoned.
52. CLI-first, Python-first, GUI-first, and server-restart scenarios all agree.
53. Two Electron launches for one deployment produce one window; two deployments produce two.
54. A valid LM3 server for a different deployment on the port is refused with a distinct error.
55. Closing attached Electron kills neither the server nor the pipeline.
56. `connection.private.json` is user-only from its first write on every platform.
57. Electron runs an exactly pinned, patched release on a currently supported major line.
58. Node-local active DB survives simulated preemption and resumes from the archived snapshot; a
    backup killed midway leaves the previous generation intact.
59. The GUI reads `active_db_path` while running and `archived_db_path` after finalization, never a
    deleted scratch path.
60. The deterministic mock pipeline's DB rows, resume behavior, and output inventory match the
    pre-refactor baseline.

---

## 9. Definition of done

A CLI-started mock or real LM3 run can be followed by opening Electron, and all of the following
agree without manual configuration: runtime run name and invocation ID; source/effective settings
identity; run directory and SQLite ledger; active stage and progress; logs, results, and
postprocessing target; Start disabled because the deployment lease is occupied; Stop capability
accurately reflecting ownership; next-run YAML visibly a separate editable concern; closing and
reopening the GUI having no effect on the run; no second GUI instance for one deployment; a
single-user cluster job exposing the same GUI safely through an SSH tunnel and surviving a
preemption with a resumable archived DB.

No watcher YAML, client-only hidden project, filesystem recency guess, or server-private adoption
record may participate in declaring the active runtime.

---

## Appendix A — Verified code state

Every reference checked against the tree on 2026-08-28.

| Fact | Evidence |
|---|---|
| Lease insertion point is between validate and hardware profiling | `machine3.py:58-61` |
| Hardware profiling runs *inside* `machine3()` | `machine3.py:61`, `hardware_setup.py:92` |
| Auto-setup triggers when the profile is absent | `hardware_setup.py:100-105` |
| Calibration spawns a nested LM3 subprocess | `calibrate.py:153-159` via `hardware_setup.py:305-315` |
| Nested-run failure is only a warning | `calibrate.py:163-166` |
| Profile is written only *after* calibration returns | `hardware_setup.py:176` |
| Auto-setup does not calibrate by default | `hardware_setup.py:124-130` (`calibrate: bool = False`) |
| Calibration run name | `calibrate.py:143`; `CALIBRATION_RUN_NAME = "_lm3_calibration"` at `calibrate.py:43` |
| `HW_PATH` is a bare relative path | `hardware_setup.py:36` |
| Calibration images are a bare relative path | `calibrate.py:41` |
| Explicit GPU subsets are supported | `config.py:402-417` |
| Concurrent-run reaper safety is pinned by test | `tests/test_executor.py:367` |
| Batch continues past a nonzero species exit | `run_global_greening.sh:207-210` |
| Mock e2e calls `machine3()` directly | `tests/test_pipeline_mock.py:46,335,346,359,360` |
| `conftest` has no runtime isolation | `tests/conftest.py` (no env handling) |
| **Start creates the log dir before spawning** | `metrics_api.py:716` |
| **Start opens the console log before spawning** | `metrics_api.py:730` |
| **Start spawns and returns without waiting** | `metrics_api.py:745`, returns at `:787` |
| Legacy jobs stage uploads + settings before execution | `app.py:71-78` |
| **GUI hardware setup passes `None` when the config file is absent** | `app.py:577` |
| **`run_setup` dereferences cfg unconditionally — no `None` guard exists** | `hardware_setup.py:145`, `:162`, `:662` |
| **GUI hardware setup runs in-process, with no child or process group** | `app.py:574-581` (`run_in_threadpool(run_setup, ...)`) |
| **Setup sizes GPU stages against the selected `compute.devices` subset** | `hardware_setup.py:160-164` |
| **The staleness fingerprint embeds config-derived model hashes** | `_fingerprint` → `model_hashes=_model_hashes(cfg)` at `hardware_setup.py:616`, resolved at `:630-635` |
| **The server logs the raw bearer token at warning level** | `app.py:289` |
| **The UI route sets `no-cache`, which still stores the response** | `app.py:672`, with the distinction documented at `:661-672` |
| Run paths are deterministic | `dirs.py:42-44`, `:56`, `:63` |
| tmp can fall back during creation | `dirs.py:67-76` |
| `build_dirs` docstring is stale | `dirs.py:39-41` vs code at `dirs.py:46-54` |
| `bind_hardware` does not rewrite `tmp_dir` | `config.py:447-456` |
| PID-reuse guard already exists | `metrics_api.py:440-450`, called at `:478` — **no longer load-bearing**: revision 15 makes control authority handle ownership alone, and `_pid_alive`/`_adopt` are deleted in Step 4 |
| Adoption derives a process group from a recorded PID | `metrics_api.py:494`, signaled at `:804` |
| Server private run state | `metrics_api.py:427-429`, `_adopt()` at `:459` |
| Run spawned in its own session | `metrics_api.py:751` |
| `/healthz` returns `pid` expressly for killing | `app.py:532-538` |
| `/v1/runs` duplication has an order workaround | `app.py:613-615` |
| `/v1/jobs` routes | `app.py:478-525` |
| Token precedence: meta, then URL, then storage | `api.js:33-43` |
| Token embedding is refused off-loopback | `app.py:654-657` |
| Electron invents a token when unset | `main.js:25`, used at `main.js:277` |
| Electron kills attached servers | `main.js:36`, `main.js:86` |
| No `requestSingleInstanceLock` anywhere | absent from `app/main.js` |
| Installed Electron | **43.4.1** exactly (`app/package.json`); was 33.4.11 when Appendix A was first recorded, upgraded by Step 5a |
| Postprocessors are CPU-only today | `leafmachine3/postprocessing/` — no torch/onnxruntime imports |

**Settings resolution split — Step 1 must unify the fallbacks, not just the names:**

| Module | Variable | Fallback when unset |
|---|---|---|
| `settings_api.py:41,110` | `LM3_SETTINGS_PATH` | `./LM3_settings.yaml` (CWD) |
| `metrics_api.py:139` | `LM3_SETTINGS` | `_search_roots()` scan (`metrics_api.py:127-130`) |
| `postprocess_api.py:234` | `LM3_SETTINGS` | `LM3_settings.yaml` (CWD) |
| `progress_api.py:335` | `LM3_SETTINGS` | `LM3_settings.yaml` (CWD) |
| `metrics_api.py:135` | `LM3_HARDWARE_SETTINGS` | `_search_roots()` scan |
| `progress_api.py:345` | `LM3_HARDWARE` | `hardware_settings.yaml` (CWD) |
| `hardware_setup.py:36` | *(none)* | bare relative `hardware_settings.yaml` (CWD) |
| `calibrate.py:41` | *(none)* | bare relative `examples/images` (CWD) |

All eight are replaced by the normative table in §3.1, in which no path falls back to the CWD.

`_find_file` also differs in failure shape: it returns `None` when the override is not a file
(`metrics_api.py:124-126`) rather than falling through.

## Appendix B — Corrections to the first adversarial review

Recorded because each changes scope.

- **C1 — the `tmp_dir` rationale was wrong; the conclusion stands.** The review claimed the bound
  hardware profile drives `tmp` for `tmp_dir: auto`. It does not: `dirs.py:50-54` resolves `auto`
  to `<root>/_tmp_original`, and `bind_hardware` (`config.py:447-456`) only attaches the profile.
  The review was misled by a **stale docstring** at `dirs.py:39-41`. The conclusion — do not
  prepublish `tmp_dir` — survives on the `_ensure_tmp` fallback (`dirs.py:67-76`).
- **C2 — `/v1/jobs` has no in-tree consumer.** `getJob`, `getJobResults`, and `streamJobEvents` are
  *defined* at `api.js:360-367` and called nowhere. The one apparent link — the comment at
  `api.js:337` saying postprocessing's long jobs "answer with {job_id} — poll getJob()" — is stale:
  `postprocess_api.py` has no `job_id` return path. Removal is gated on an **external** consumer
  audit, not on in-tree call sites.
- **C3 — postprocessing GPU contention was speculative.** No shipped postprocessor imports torch or
  onnxruntime. The real present-day risk is a read-write race on run directories and reports.
- **C4 — the attached-server token mismatch is narrower than stated.** `api.js:33-43` prefers the
  server-injected `<meta name="lm3-token">` over Electron's `?token=` (and persists it), so page
  API calls against an attached server usually work. It breaks when embedding is refused
  (`app.py:654-657`) and for Electron's own out-of-band `POST /v1/shutdown`, which uses its invented
  token and then escalates to PID signaling.
- **C5 — the PID-reuse guard already exists** (SUPERSEDED by revision 15: control authority is
  handle ownership, so nothing compares PIDs and `_pid_alive` is deleted). `_pid_alive`
  (`metrics_api.py:440-450`) compares
  process creation time. The genuine defect is granting stop authority to a server that did not
  spawn the process, then signaling a process group derived from that PID (`:494`, `:804`).
- **C6 — CVE-2026-34776 was missed.** Adding `requestSingleInstanceLock()` on the then-installed Electron 33.4.11 on
  Linux introduces a known out-of-bounds read with no app-side workaround. Now a hard prerequisite
  (§2.11, gate 21).

## Appendix C — Amendments in revision 2

- **A1 — a subactivity could outlive the root lease.** Revision 1 had the child *validate* the root
  lock rather than hold it, so killing the parent freed the lease while the child ran and a second
  root could start. Fixed by inheriting the lease reference — the locked open file description on
  POSIX, the lease-event handle on Windows (§2.2, and see revision 9/10) — with a mandatory
  kill-the-parent test.
- **A2 — the busy-loser 409 was unimplementable.** `/v1/run/start` returns at `metrics_api.py:787`,
  right after `Popen` at `:745`, having already created the log directory at `:716`. Fixed with an
  explicit status-pipe handshake and a stated staging boundary (§2.4).
- **A3 — child-record storage was an either/or.** Settled into an exact file layout with bounded
  summaries, plus activity-specific required fields — revision 1 wrongly required `project` on
  every root, which does not fit `batch` or `hardware_setup` (§3.2).
- **A4 — resolver scope and calibration bootstrap.** `hardware_setup.HW_PATH` and
  `calibrate.DEFAULT_IMAGE_DIR` are bare relative paths and were outside the stated scope. The
  calibration child also re-enters setup because the profile is written only after calibration
  returns (`hardware_setup.py:176`); fixed with a provisional profile (§2.2, §3.1).
- **A5 — Steps 3 and 5 must ship together** (SUPERSEDED by revision 14, §2.3), else the batch gap returns for a release; plus a hard
  cutover rule for pre-registry runs (§4, §6).
- **A6 — cluster storage contradicted the core model.** Four named path roles, periodic online
  backup, and grace-signal checkpointing (§2.10).
- **A7 — deployment identity in `/healthz`**, a single attached-server auth flow, and an Electron
  version policy that pins a *supported* major rather than the CVE floor (§2.11).
- **A8 — test isolation was not implementable as written.** `tmp_path` is function-scoped; use
  conftest-import-time setup, and per-xdist-worker deployment IDs (§4 Step 1).

## Appendix D — Amendments in revision 3

Revision 2 was structurally right but still deferred five contracts and left four either/ors. Both
categories are the kind of gap that lets two implementers build different systems, so all nine are
now decided.

- **B1 — root finalization could strand the deployment.** The inherited lock meant a root finalizing
  normally while a child lived would remove `active.json` while the child still held the lock:
  occupied, but with nothing describing what was running. Roots now join or terminate children
  before finalizing, `active.json` survives a hard kill until the last child exits, and Stop targets
  the root process group / job object (§3.3).
- **B2 — nobody owned the record files.** Parent and child share one open file description, so the
  activity lock does not serialize their writes — exclusivity and mutual exclusion are different
  problems. Static single-writer ownership per file, with a separate `metadata.lock` if that ever
  stops being enough (§3.2).
- **B3 — the capability had nothing to validate against.** Now a one-use grant record with a hash,
  purpose, expiry, and consumed flag — and an explicit statement that it is coordination protection,
  not a security boundary against same-user code (§2.2).
- **B4 — settings precedence was deferred to implementation.** It is the fix for the original split,
  so it is now a normative table with one governing rule: **no path falls back to the CWD**. The
  embedded decisions — machine-scoped hardware profile, `--config` moving only the settings path,
  template seeding on first run, dev-checkout detection from `__file__` — were each previously
  ambiguous (§3.1).
- **B5 — console handling was an either/or.** Output destinations are fixed at `Popen`, so this
  could not be decided later: control pipe for status, child stdout/stderr to the server-private
  request log, application log after acquisition (§2.4).
- **B6 — the two descriptors contradicted each other.** `connection.private.json` (token, user-only,
  node-local) and `connection.public.json` (token-free, safe for persistent storage), created
  restricted from the first byte rather than `chmod`ed afterward (§2.12).
- **B7 — backups could destroy the last good archive.** Generational: temp sibling, integrity check,
  atomic replace, one retained generation; signal handlers set a flag rather than running a backup;
  and an honest RPO of one backup interval on hard node failure (§2.10).
- **B8 — deployment keys and ports were loose.** Canonical slug+hash so raw environment text is
  never a path component, an explicit `LM3_PORT` requirement for named deployments, and network
  filesystems refused rather than merely documented (§2.1, §3.1).
- **B9 — `hardware_setup` without a config is a live crash**, not just a permissive schema:
  `app.py:577` passes `None` and `run_setup` dereferences it at `hardware_setup.py:145`/`:162` with
  no guard. Setup now requires a resolved config, which fixes the crash and makes the profile
  describe the machine the user will actually run (§2.13).

## Appendix E — Amendments in revision 4

Revision 3's new material introduced problems of its own. Four were cross-contract; two were
self-contradictions written into the same document.

- **C1 — a machine-scoped hardware profile is shared mutable state.** Revision 3 moved the profile
  to a machine-wide path on the reasoning that it "describes the box". Half of that is false: setup
  sizes stages against the selected `compute.devices` subset (`hardware_setup.py:160-164`),
  instantiates only config-enabled stages, and fingerprints config-derived model hashes (`:616`).
  Two named deployments hold separate leases, so nothing prevents them tuning concurrently into one
  file — the deployment lock does not protect a path outside the deployment. Now deployment-scoped
  and machine-keyed (§3.1).
- **C2 — GUI hardware setup cannot honor the Stop contract.** It runs in the server's threadpool
  (`app.py:574-581`), so "terminate the root process group" would kill the server. Setup now runs as
  an `lm3-setup` subprocess — which also lets invariant 13 drop its "except when the server holds a
  hardware lease" exception (§2.13).
- **C3 — two self-contradictions in the settings section.** Postprocessing settings were listed as
  living beside the resolved settings *and* described as deployment-scoped eleven lines later; and
  the "deployment workspace pointer" was used in the precedence chain without ever being defined —
  in a plan whose original defect was redundant project-selection mechanisms. Both fixed (§3.1).
- **C4 — browser authentication vanished.** Revision 3 kept the HTML-meta bootstrap "only during the
  transition release", while the public descriptor carries no token and the private one is
  node-local. A tunneled cluster user would have had no way to authenticate at all. The same-origin
  bootstrap is now permanent — it already gates on loopback client and loopback `Host`
  (`app.py:654-657`), which is exactly the tunneled case (§2.11).
- **C5 — grant consumption needed two writers on one file**, which §3.2 forbids and the shared lock
  cannot serialize. Replaced with atomic claim-by-rename, plus real descriptor validation rather
  than a presence check (§2.2).
- **C6 — abandoned cleanup was an unstated writer exception.** Named explicitly as the bounded
  recovery-writer role (§3.2).
- **C7 — a timed-out child was left running** and could acquire the lease after the API reported
  failure. The server now terminates it before responding (§2.4).
- **C8 — the private descriptor could not rotate.** Exclusive-create at the final path succeeds only
  once; a restarted server could never publish a new token. Temp sibling plus atomic replace, and
  shutdown deletes only a descriptor whose instance ID still matches (§2.12).
- **C9 — the deployment-key algorithm was not reproducible across languages.** `str.casefold()` and
  NFKC have no faithful JavaScript equivalent, so Python and Electron could derive different
  directories before any golden test ran. The hash of the raw UTF-8 bytes is now authoritative and
  the slug is mechanical ASCII (§2.1).
- **C10 — the checkpoint could miss the grace window.** "Next safe point in the main loop" can be
  minutes away inside one stage, against a typical 120-second Slurm grace. A dedicated checkpoint
  thread with its own read connection, plus versioned snapshots behind an atomic pointer so
  mid-rotation crashes are defined (§2.10).


## Appendix F — Errata in revision 5

Architecture was accepted at revision 4; these are narrow safety and protocol details. Two were
defects introduced by revision 4 itself.

- **D1 — two conflicting archive contracts in one section.** Revision 4 added versioned snapshots
  behind an atomic pointer but left the earlier table and procedure replacing a stable
  `artifact_dir/<run>.sqlite`. The pointer design is now used throughout; there is no
  `os.replace` onto a fixed archive filename anywhere (§2.10).
- **D2 — the handshake was specified for POSIX only.** `pass_fds` does not exist on Windows, so the
  Windows transport is now named explicitly, with identical timeout/size/EOF/termination semantics
  and Windows CI coverage (§2.4).
- **D3 — `<machine-key>` was used but never defined.** It must include host identity: a cluster
  routinely has many nodes with identical GPU models sharing one networked config filesystem, and a
  model-name key would collide across all of them (§3.1).
- **D4 — the server logs the raw bearer token** (`app.py:289`), which on a cluster persists in Slurm
  output and container logs. Also, the token-bearing HTML is served `no-cache`, which still stores
  it; `no-store` is required. The codebase already documents that distinction at `app.py:661-672`
  (§2.11).
- **D5 — the grant was claimed before it was validated**, so an accidental invalid child could
  consume the legitimate child's grant and lock it out. Validate → claim → revalidate (§2.2).
- **D6 — setup progress transport was still an either/or.** Append-only JSONL event log: it survives
  disconnects, cannot stall the subprocess by filling a pipe, and replays on reconnect (§2.13).
- **D7 — a stale workspace pointer had no defined behavior.** Failing visibly beats falling back to
  the default settings file, which would silently run a configuration the user did not choose
  (§3.1).
- **D8 — the registry grew without bound.** Documented retention for `children/` and consumed grants
  at finalization and recovery (§3.2).
- **D9 — the postprocessing lock could land on NFS**, immediately after §3.1 declares network-backed
  locking unreliable. It lives in the local deployment runtime registry, keyed by a hash of the
  resolved artifact directory (§2.8).


## Appendix G — Errata in revision 6

- **E1 — the archive contract still read two ways.** Revision 5 declared archives "never a fixed
  filename", which is right for the cluster and wrong for the desktop, where it would have added
  pointer indirection and background copying to a run with nothing to stage — and the desktop table
  row and the `project` example both still showed the fixed path. Now two explicit modes: in-place
  (null pointer, `archived_db_path == active_db_path`, no copying) and staged (versioned snapshots,
  mandatory pointer), with `archive_mode` recorded (§2.10, §3.2).
- **E2 — the Windows handshake still offered a choice**, which is exactly the deferral D2 claimed to
  close. `STARTUPINFOEX` inherited-handle allowlist chosen, with the reason recorded: it mirrors the
  POSIX side, while a named pipe would add addressing, authentication, ACL, cleanup, and replay
  rules (§2.4).
- **E3 — the machine key was not container-safe.** "Trusted machine identity, else hostname"
  collapses to one value when a container image ships a baked `/etc/machine-id` to every node, and
  CPU-only nodes have no GPU UUIDs to break the tie. Node name is now unconditional, over a
  canonical serialization with explicit missing-value markers (§3.1).
- **E4 — atomic replace is not durability.** Rename prevents a torn read but leaves both the rename
  and the pointer in page cache; an exact fsync sequence now covers the generation file, its
  directory, the pointer temp file, and the pointer's directory, with pruning only after (§2.10).
- **E5 — invariant 6 contradicted the activity schemas** by demanding project identity from `batch`
  and `hardware_setup` roots, which deliberately have none. Qualified to "where applicable" (§1).


## Appendix H — Errata in revision 7

- **F1 — a stale cross-reference.** The checkpoint thread was still told to run "steps 1-5" from
  when the procedure had five steps; revision 5's durability sequence grew it to eight, and step 5
  now only writes the *temporary* pointer. It runs the complete transaction through step 7; pruning
  (step 8) follows and is not part of it (§2.10).
- **F2 — the grant order still forward-referenced itself.** Step 1 said to perform "the descriptor
  check in step 4", but step 4 ran *after* the claim — so a child with a bogus descriptor could still
  consume the legitimate child's grant, which is exactly the failure D5 was written to prevent.
  Descriptor validation moved into step 1, with an explicit rule that no step may forward-reference
  a later one (§2.2).
- **F3 — the pre-first-snapshot window was undefined.** A staged record was required to name "the
  current generation" when none exists yet, while a missing pointer was classified as an error —
  so every staged run would have begun in a failure state. Added `archive_status`
  (`pending` / `ready` / `n/a`), with `pending` making an absent pointer expected rather than
  broken (§2.10, §3.2).
- **F4 — three backup triggers, no coordinator.** Periodic, signal-triggered, and final backups
  could interleave and race pointer publication against pruning, letting the pointer name a
  generation another trigger had just removed. All three now serialize on one checkpoint mutex, with
  in-flight requests coalesced rather than started concurrently (§2.10).


## Appendix I — Errata in revision 8

- **G1 — coalescing was specified too permissively.** "Coalesced into the running one, or queued"
  is safe for a periodic request but not for a signal or final one: a snapshot that began at T₀
  reflects the database at T₀, so merging a final request that arrived at T₁ > T₀ would silently
  drop every write in between while reporting success. Periodic requests may coalesce; signal and
  final requests queue exactly one follow-up; and finalization waits for a snapshot that *began
  after writes stopped*, not merely for the in-flight one to end (§2.10).
- **G2 — a terminal archive failure had no representation.** Finalization was permitted once a
  failure was recorded, while `last.json` was simultaneously required to carry a resolved archive
  path — impossible when the first snapshot is also the final one and it fails. Added terminal
  `archive_status` values: `stale` (an earlier snapshot stands, final checkpoint failed) and
  `failed` (nothing ever committed, null path, GUI reports recovery data unavailable). Neither may
  name node-local scratch after finalization (§2.10).
- **G3 — `n-a` in the revision-7 summary** where the normative schema says `n/a`. Standardized.


## Appendix J — Amendments in revision 9

- **H1 — the subactivity lifetime mechanism was POSIX-only, and I had labelled the gap a
  portability detail.** Revision 2 established that children inherit the *locked open file
  description*, which is a `flock` property; every revision since said Windows "needs the equivalent
  design (`LockFileEx` on an inheritable `HANDLE`)". There is no equivalent: Microsoft states that
  an inheriting child "is not granted access to the locked region", and that a terminating process's
  locks are released by the OS. On Windows, killing the root would have freed the lease while its
  subactivity ran on — the precise violation invariant 2 exists to prevent, hidden behind a sentence
  that read like a to-do. Windows now uses a named kernel event for lifetime (its existence is the
  lease; it survives until the last inherited handle closes) alongside `LockFileEx` for
  cross-session exclusion, with the `Local\` namespace limitation stated rather than left to be
  discovered (§2.2).
- **H2 — gate 14 still demanded a resolved archive path unconditionally**, contradicting the
  `failed` state revision 8 had just introduced. Qualified by `archive_status` (§8).
- **H3 — `<generation>` was never defined.** Keyed on `run_id` plus a monotonic sequence, created
  exclusively, so a resumed invocation of the same project cannot reuse a filename and overwrite an
  earlier run's archive (§2.10).


## Appendix K — Amendments in revision 10

- **I1 — documenting a hole does not close it.** Revision 9 used a `Local\` event and then admitted
  that after a root died, a child could not block a contender in another logon session — which
  contradicts invariant 2, stated without qualification. The justification was also wrong on the
  facts: `SeCreateGlobalPrivilege` gates only **file-mapping and symbolic-link** objects in the
  global namespace, not events, and Microsoft's namespace documentation shows
  `CreateEventW(NULL, FALSE, FALSE, L"Global\CSAPP")` directly. The lease is now an authoritative
  `Global\` event named with a hash of the user's SID plus the deployment key, current-user-only
  ACL, and a hard startup error if it cannot be created — no silent `Local\` fallback. `LockFileEx`
  drops to an auxiliary on-disk artifact and is explicitly not a second source of truth (§2.2).
- **I2 — "verify the handle" was an intention, not a mechanism.** The child now `OpenEventW`s the
  canonical name and calls `CompareObjectHandles` (`handleapi.h`) against the inherited handle,
  which is the API Microsoft documents for deciding whether two handles name the same kernel object.
  That sets a Windows 10 / Server 2016 support floor, which is not a constraint in 2026 (§2.2).
- **I3 — two summary lines still described the POSIX mechanism as universal** after revision 9
  scoped the normative text. Reworded to "lease reference".


## Appendix L — Errata in revision 11

Four handle-lifecycle details in the new Windows lease.

- **J1 — a busy contender leaked the lease.** `CreateEventW` returns a *valid handle to the existing
  object* on `ERROR_ALREADY_EXISTS`, and the object lives until its last handle closes. A contender
  that raised `RuntimeBusyError` without closing it would keep the deployment occupied after the
  real run exited — and a direct Python caller that catches the exception and continues in-process
  makes that a live scenario, not a theoretical one. `CloseHandle` before raising (§2.2).
  While confirming this, a second trap surfaced and is now documented: opening an *existing* named
  event requests `EVENT_ALL_ACCESS`, so the current-user-only DACL must grant it or LM3's own next
  process cannot open its own lease.
- **J2 — only the POSIX side disarmed inheritance.** The child cleared `LM3_LEASE_FD` and called
  `os.set_inheritable(..., False)`, with no Windows counterpart — so an ordinary executor worker
  could inherit the lease event and hold the deployment open, violating invariant 3. Added
  `SetHandleInformation(handle, HANDLE_FLAG_INHERIT, 0)` and clearing `LM3_LEASE_EVENT_HANDLE`
  (§2.2).
- **J3 — an "optional" `LockFileEx` is a second mechanism waiting to be mistaken for truth.** With
  the global event authoritative, Windows takes no file lock at all; `activity.lock` remains only as
  a passive artifact so the registry directory has one shape everywhere (§2.2).
- **J4 — three specification loose ends.** Windows 10 / Server 2016 is now the *normative* floor
  rather than something to confirm later; the SID hash is `sha256` of the canonical
  `ConvertSidToStringSidW` form truncated to 16 hex, matching the deployment- and machine-key
  convention; and one remaining "holds that description" became "holds that lease reference".


## Appendix M — Amendments in revision 12

- **K1 — the root's own workers could inherit the lease, and disarming it in the child did nothing
  about that.** Revision 10 had the root create an *inheritable* event handle and hold it for the
  whole run. Revision 11 then carefully made the handle non-inheritable **inside an approved
  child** — but ordinary executor workers are spawned by the **root**, from the root's own handle,
  so the fix never touched them. One long-lived worker would hold the deployment occupied after the
  root and every subactivity had exited: the lease outliving everything meant to hold it, which is
  the same class of failure as J1 arriving by a different route. The root now keeps a
  **non-inheritable owner handle** and makes a temporary inheritable `DuplicateHandle` per approved
  launch, closed on every path including failure, with the brief inheritable window serialized
  against concurrent process creation. POSIX was never exposed to this — PEP 446 makes descriptors
  non-inheritable by default and `pass_fds` opts in per spawn — which is precisely why the Windows
  adapter needed it spelled out rather than mirrored (§2.2).
- **K2 — the Windows section still read "a named kernel event, plus the file lock"** and framed the
  design as one predicate over two primitives, contradicting J3's removal of `LockFileEx` two
  revisions earlier. Rewritten as the single primitive it now is (§2.2).
- **K3 — finalization still said "lock descriptor" and "inherited descriptor" universally**, after
  revisions 9 and 10 had scoped that language everywhere else. Now "lease reference" (§3.3).
