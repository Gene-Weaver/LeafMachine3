# Native acceptance checklist — Windows and macOS

For a **human, on real hardware.** Nothing in this file has been run. Every item below is
unvalidated; this is the script for the release gate, not a record of one.

Companion document: `ELECTRON_PACKAGING.md` (what the packaging config is, and what was and was not
executed on the Linux build host).

## Ground rules

1. **This checklist does not block Linux Step 3.** Per the consensus plan §1, Windows and macOS
   qualification is a later release gate, not a prerequisite for the Linux path. Do not hold Step 3
   for anything here.
2. **Until an item is ticked on real hardware, the correct wording is "implemented but not yet
   natively validated" — never "passed."** A result from a fake, a cross-build, or a CI job that has
   not run is not evidence.
3. **Check the gate column before testing anything.** Several items describe behavior that does not
   exist yet. Testing them today produces a "failure" that is just the plan not having happened.

## Gate legend

| Gate | Meaning |
|---|---|
| **NOW** | Testable against the current `main.js`. |
| **5b** | Requires plan Step 5b (deployment-scoped single-instance lock, `/healthz` identity + deployment key + ownership mode, connection-descriptor auth, named distinct port errors, removal of the kill-attached-server policy). **Not implemented — do not test yet.** |
| **PKG** | Requires a real signed/notarized installer to exist. Nothing on this Linux host can produce one. |

---

## 0. Prerequisites before anyone starts

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | PKG | A Windows machine builds `npm run dist:win` | `dist/LeafMachine3-3.0.0-x64-setup.exe` and `…-x64-portable.exe` are produced. **Never done: the Linux host fails this target on missing `wine`.** |
| ☐ | PKG | A Mac builds `npm run dist:mac` with `APPLE_ID` + `APPLE_APP_SPECIFIC_PASSWORD` + `APPLE_TEAM_ID` (or the API-key trio) set | `dist/LeafMachine3-3.0.0-{x64,arm64}.dmg` and `.zip`, signed and notarized. **Never done: no Mac was involved.** |
| ☐ | PKG | Windows code signing is configured | A `signtoolOptions` or `azureSignOptions` block exists in `electron-builder.yml`. **There is none today** — installers will be unsigned. |
| ☐ | NOW | An LM3 Python environment exists on the test machine and its interpreter path is known | Needed for every "owned server" item — see the `LM3_PYTHON` note in §7. |

---

## 1. Installation and first launch

| ✔ | Gate | Platform | Item | Expected result |
|---|---|---|---|---|
| ☐ | PKG | Win | Run the NSIS installer on a machine with no prior LM3 | Wizard appears (not one-click), install directory is changeable, per-user install needs no admin elevation. |
| ☐ | PKG | Win | Installer completes | Desktop shortcut and Start-menu entry named **LeafMachine3**, both with the leaf icon (not the generic Electron icon). |
| ☐ | PKG | Win | Launch from the Start menu | Window titled "LeafMachine3", leaf icon in the taskbar, dark `#101012` background with no white flash. |
| ☐ | PKG | Win | Run the `portable` .exe on a machine where installing is forbidden | Runs with no installation and no admin rights. |
| ☐ | PKG | mac | Open the `.dmg` | Window shows the app on the left and an `/Applications` alias on the right; drag-to-install works. |
| ☐ | PKG | mac | Launch from `/Applications` | App opens; Dock icon is the leaf icon; menu bar reads "LeafMachine3". |
| ☐ | PKG | mac | Both architectures | Repeat on Apple silicon **and** on an Intel Mac. `LSMinimumSystemVersion` is 12.0, so also confirm the oldest macOS you intend to support. |
| ☐ | NOW | both | First launch with no LM3 server running | See §7 — with default settings the packaged app **cannot** start a server and will show "Cannot start LeafMachine3". That is current expected behavior, not a bug in the installer. |

## 2. Single-instance behavior — **Step 5b, NOT IMPLEMENTED**

> `main.js` does **not** call `app.requestSingleInstanceLock()` today. That call is Step 5b's, and it
> was deliberately left out of the Electron 43 upgrade and out of packaging. **Do not test this
> section before Step 5b lands** — launching twice today will open two windows, and that is the
> current design, not a regression.

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | 5b | Launch the app twice for the **same** deployment | Exactly one window and one server. The second launch focuses the existing window and exits. |
| ☐ | 5b | Launch twice for **different** `LM3_DEPLOYMENT_ID` values | Two windows and two servers. Deployments do not share a lock. |
| ☐ | 5b | Confirm `userData` is deployment-scoped | `app.setPath("userData", …)` is set **before** `requestSingleInstanceLock()`. Chromium's `SingletonLock` / `SingletonSocket` / `SingletonCookie` live inside `userData`; locking first gives every deployment one shared lock. Verify the files appear under the per-deployment directory. |
| ☐ | 5b | Win: kill the first instance with Task Manager, then relaunch | The stale lock is reclaimed; the app starts normally rather than refusing forever. |
| ☐ | 5b | mac: relaunch from the Dock while running | Existing window is focused; no second instance. |
| ☐ | 5b | mac: `open -n` (force new instance) | Refused or focuses the existing window — not a second server on the same port. |

## 3. Attach to an existing LM3 server

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | NOW | Start `lm3 serve` in a terminal, then launch the GUI | The GUI attaches to that server (log line `server attached at http://127.0.0.1:<port>`) and does **not** spawn a second one. Only one server process exists. |
| ☐ | NOW | The attached window shows the real LM3 UI | The UI loads, authenticated, with no login prompt — the token is passed in the URL. |
| ☐ | NOW | Start a long LM3 run from the terminal server, then launch the GUI | The GUI attaches and shows the already-running run. |
| ☐ | 5b | `/healthz` identity handshake | `/healthz` reports service, protocol version, instance ID, deployment key and ownership mode; the GUI verifies them before attaching. |
| ☐ | 5b | Attaching uses the connection descriptor, not a PID | The GUI reads `connection.private.json`; no bearer token appears in any log at any level (plan gate 13). |

## 4. Closing an attached GUI must not kill the server or the run — **Step 5b**

> **Current behavior is the opposite, deliberately.** Today `main.js` escalates
> `POST /v1/shutdown` → `SIGTERM` → `SIGKILL` against *any* server it is bound to, including one it
> merely attached to. That was measured on the packaged Linux artifact (see `ELECTRON_PACKAGING.md`
> §4.3). Step 5b's exit gate deletes this policy. **Do not test this section before Step 5b lands —
> it will fail by design.**

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | 5b | Attach to `lm3 serve`, then close the GUI window | The terminal server is still running and still answering `/healthz`. |
| ☐ | 5b | Attach during a live run, then close the GUI | The run continues to completion. Outputs are complete; no checkpoint/resume was triggered. |
| ☐ | 5b | Reopen the GUI afterwards | It reattaches to the same server and shows the same run. |
| ☐ | 5b | Win: close via the window ✕, via Task Manager "End task", and via a machine sign-out | All three leave the attached server alive. |
| ☐ | 5b | mac: ⌘Q, red-dot close, and Force Quit | All three leave the attached server alive. |
| ☐ | 5b | PID-derived shutdown authority is gone | Nothing in `main.js` signals a server it did not spawn. |

## 5. Owned-server shutdown

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | NOW | Launch the GUI so it **spawns** the server, then quit | The server exits; the port is free; `netstat`/`lsof` shows no listener; no orphaned Python process. |
| ☐ | NOW | Quit while a run is active | The UI's Close button confirms first ("Close LeafMachine3?"), stops and checkpoints the job, then quits. Reopening and pressing Start LM3 resumes it. |
| ☐ | NOW | `LM3_KEEP_SERVER=1`, then quit | The spawned server is left running on purpose. |
| ☐ | NOW | Kill the GUI process hard (Task Manager / Force Quit / `kill -9`) | The owned server notices via its `LM3_OWNER_PID` watchdog and exits by itself within a bounded time. Nothing is left holding the port. |
| ☐ | NOW | Immediately relaunch after each of the above | The new launch starts a *fresh* server rather than silently adopting a stale one running old code. |

## 6. Two distinct port errors — **Step 5b**

> Today `main.js` produces one generic message ("the LM3 server exited … Is something else already
> on http://host:port?"). Step 5b requires two **named, distinguishable** errors.

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | 5b | Occupy the port with a **non-LM3** service (e.g. `python -m http.server 8765`), then launch | Error names *"unrelated service on this port"*. It must not claim the port holder is an LM3 server, and must not attach. |
| ☐ | 5b | Occupy the port with an LM3 server of a **different deployment key**, then launch | Error names *"wrong deployment on this port"* — a different message and a different error identity from the one above. |
| ☐ | 5b | The two messages are actually different | Compare the literal strings side by side. A single message with a substituted noun is not two errors. |
| ☐ | 5b | Neither message leaks a bearer token | Plan gate 13: tokens redacted from error messages, tracebacks and `/healthz` diagnostics. |

## 7. Path handling — spaces and non-ASCII

Both platforms routinely produce these paths and both have historically broken on them.

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | PKG | Win | Install to a path with a space, e.g. `C:\Program Files\LeafMachine3` and `C:\Users\Ana María\AppData\Local\Programs\LeafMachine3` | Installs and launches. |
| ☐ | PKG | Win: log in as a user with a **non-ASCII** name (`Ana María`, `王小明`) and install | Installs and launches; `%LOCALAPPDATA%` and `userData` resolve correctly; no mojibake in paths shown in the UI. |
| ☐ | NOW | Win: point `LM3_PYTHON` at an interpreter under a path with spaces | The server spawns. `main.js` passes the interpreter as `spawn`'s first argument (no shell), so this should hold — confirm it does. |
| ☐ | NOW | Win: choose an input image directory with spaces, non-ASCII characters, and a trailing backslash | The run reads every image; output paths are correct. |
| ☐ | NOW | Win: a path longer than 260 characters | Either it works, or it fails with a clear message. It must not silently truncate or half-write. |
| ☐ | PKG | mac: run as a user with a non-ASCII short name and a space in the home directory | App launches; `~/Library/Application Support/LeafMachine3` is created correctly. |
| ☐ | NOW | mac: input directory with non-ASCII characters, including **decomposed** Unicode (NFD — macOS's native form for filenames, e.g. `é` as `e` + U+0301) | Filenames round-trip. Output names match input names. This is the classic macOS bug. |
| ☐ | NOW | both: an input directory on a network mount / external volume | Reads and writes work, or fail with a clear message. |

## 8. Windows process-tree handling

`main.js`'s shutdown path is POSIX-only: `process.kill(-pid, sig)` to signal a process *group*, plus
`SIGTERM`/`SIGKILL` escalation. **Windows has no process groups and no signals**, and
`detached: true` means something different there. A CUDA-loaded LM3 server spawns children; if the
tree is not killed as a unit, those children survive and hold the GPU.

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | NOW | Launch the GUI so it spawns a server, start a run (so the server has children), then quit normally | The server **and every descendant** exit. Check Task Manager's tree view, and confirm GPU memory is released (`nvidia-smi`). |
| ☐ | NOW | Repeat, but Force-quit the GUI from Task Manager | Same: no surviving Python, no held GPU memory, no held port. |
| ☐ | NOW | Confirm the mechanism, not just the outcome | Windows requires a **Job Object** with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (or `taskkill /T /F /PID`). Verify which one is in force; a passing test that works by luck of timing is not a pass. |
| ☐ | NOW | `process.kill(-pid, …)` is not silently doing nothing | On Windows a negative PID is not a group. Confirm the Windows path does not fall through to a no-op that leaves the escalation ladder with no rungs. |
| ☐ | NOW | Sign out of Windows while a run is active | No orphaned `python.exe` survives into the next session. |
| ☐ | 5b | Same tree tests after the attached-server policy change | An **attached** server's tree is left entirely alone; an **owned** server's tree is still killed as a unit. |

## 9. macOS permissions, signing, notarization, Gatekeeper

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | PKG | `codesign --verify --deep --strict --verbose=2 LeafMachine3.app` | `valid on disk`, `satisfies its Designated Requirement`. |
| ☐ | PKG | `codesign -d --entitlements - LeafMachine3.app` | Shows the entitlements from `build/entitlements.mac.plist`; the four helper apps show `build/entitlements.mac.inherit.plist`. |
| ☐ | PKG | Hardened runtime is on | `codesign -d -vvv` shows `flags=…(runtime)`. |
| ☐ | PKG | Notarization succeeded | `xcrun notarytool history` shows `status: Accepted` for this build. If it was rejected, read the log — a rejection is a *fail*, not a warning. |
| ☐ | PKG | Ticket stapled | `xcrun stapler validate LeafMachine3.app` → `The validate action worked!` |
| ☐ | PKG | **Quarantine / Gatekeeper on a clean machine** | Download the dmg over HTTPS onto a Mac that has never seen this app (so `com.apple.quarantine` is set) and open it. It opens with **no** "cannot be opened because the developer cannot be verified" dialog and **no** right-click→Open workaround. |
| ☐ | PKG | `spctl --assess --type execute -vv LeafMachine3.app` | `accepted`, `source=Notarized Developer ID`. |
| ☐ | PKG | Confirm quarantine was actually present | `xattr -p com.apple.quarantine` on the downloaded dmg before opening. Testing a file copied over the LAN, or one built locally, tests nothing. |
| ☐ | PKG | The hardened runtime can still spawn Python | With `LM3_PYTHON` set to a Homebrew/conda/venv interpreter, the signed app starts an LM3 server and the run completes. This is what `disable-library-validation` + `allow-dyld-environment-variables` are for, and it is the single most likely entitlement failure. |
| ☐ | PKG | Local network permission | On macOS 13+, first launch prompts to allow local network access, and the prompt shows the `NSLocalNetworkUsageDescription` text. **Deny it** and confirm the app reports a clear error rather than hanging on `/healthz`. |
| ☐ | NOW | Files & Folders permission | Point a run at `~/Documents` or `~/Desktop`; macOS prompts once; after granting, the run reads every image. After denying, the failure message is clear. |
| ☐ | NOW | Full Disk Access is **not** required | The app works without it. If it turns out to be required, that is a finding to fix, not to document. |
| ☐ | PKG | Rosetta | The x64 build runs under Rosetta 2 on Apple silicon, and the arm64 build runs natively. Confirm which one the dmg actually installed (`file` on the binary). |

## 10. Upgrade and uninstall

| ✔ | Gate | Item | Expected result |
|---|---|---|---|
| ☐ | PKG | Win: install 3.0.0, create settings/runs, then install a later build over it | Upgrade succeeds without a manual uninstall. **Settings and run history survive** (`nsis.deleteAppDataOnUninstall` is `false`). Shortcuts point at the new version. |
| ☐ | PKG | Win: no stale processes block the upgrade | If the app is running, the installer either closes it cleanly or refuses with a clear message — it must not corrupt the installation. |
| ☐ | PKG | Win: uninstall from Settings → Apps | Program files and shortcuts are removed. **User data under `%APPDATA%` is intentionally kept**; confirm that is still what you want at release time. |
| ☐ | PKG | Win: install per-user, then confirm no admin prompt ever appeared | `perMachine: false` — a lab user without admin rights can install. |
| ☐ | PKG | mac: replace `/Applications/LeafMachine3.app` with a newer build | Launches; settings under `~/Library/Application Support/LeafMachine3` survive. |
| ☐ | PKG | mac: uninstall by dragging to Trash | App is gone. `~/Library/Application Support/LeafMachine3` and `~/Library/Caches/org.leafmachine.lm3` remain — document where they are so a user can clear them. |
| ☐ | PKG | mac: the bundle identifier never changes across versions | `org.leafmachine.lm3`. Changing it makes every upgrade look like a different app to macOS. |
| ☐ | NOW | Linux: `sudo apt install ./LeafMachine3-3.0.0-linux-amd64.deb`, then remove | Installs to `/opt/LeafMachine3`, desktop entry appears. **The dpkg package name is `lm3-desktop`, not `leafmachine3`** — remove with `sudo apt remove lm3-desktop`. |
| ☐ | NOW | Auto-update | Not configured (`publish: null`). If auto-update is ever wanted, it is new work and needs its own gate. |

---

## Reporting

When this checklist is run, report per item: **passed on <OS version, hardware>**, **failed (with
the observed behavior)**, or **not testable yet (gate <n>)**. Do not aggregate to "Windows passed" or
"macOS passed" — the plan requires the per-platform status line to stay "implemented but not yet
natively validated" until every non-gated row above is ticked on real hardware.
