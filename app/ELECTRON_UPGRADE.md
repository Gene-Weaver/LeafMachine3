# Electron upgrade — 33.4.11 → 43.4.1

Step 5a of the unified-runtime plan (§2.11, §4 Step 5a). This is deliberately its own change: it
moves the Electron runtime and nothing else. No `requestSingleInstanceLock()`, no deployment-key
identity, no `/healthz` verification, no ownership change — all of that is Step 5b.

| | |
|---|---|
| Old | `"electron": "^33.2.0"` → resolved `33.4.11` |
| New | `"electron": "43.4.1"` — exact, no `^`, no `~` |
| Checked against the live release data on | **2026-08-28** |

## Why this had to land before Step 5b

**CVE-2026-34776 / GHSA-3c8v-cfp5-9885** — "Out-of-bounds read in second-instance IPC on macOS and
Linux". CVSS 3.1 **5.3** (`AV:L/AC:H/PR:L/UI:N/S:U/C:H/I:N/A:L`), CWE-125, published 2026-04-03.
Verified directly against `https://api.github.com/advisories/GHSA-3c8v-cfp5-9885` on 2026-08-28:

| Affected range | First patched |
|---|---|
| `< 38.8.6` | 38.8.6 |
| `>= 39.0.0-alpha.1, < 39.8.1` | 39.8.1 |
| `>= 40.0.0-alpha.1, < 40.8.1` | 40.8.1 |
| `>= 41.0.0-alpha.1, < 41.0.0` | 41.0.0 |

The bug is only reachable in an app that calls `app.requestSingleInstanceLock()`. This shell does
not call it today, so 33.4.11 was not *exploitable* here — but Step 5b's whole job is to add that
call. Adding it on 33.4.11 would have *introduced* the vulnerability. Hence: upgrade first, lock
second. 43.4.1 is above every affected range.

## Why 43.4.1 specifically

The gate is "a patched release from a **currently supported** major line, verified against the
release schedule at implementation time." 38.8.6 is the CVE floor, not the support floor.

Support policy (`docs/latest/tutorial/electron-timelines.md`, fetched 2026-08-28): *"The latest
three stable major versions are supported by the Electron team"* and *"We only support the latest
minor release for each stable release series"* — so a security fix lands on `43.4.x`, never on an
older 43 minor.

Live data from `https://releases.electronjs.org/schedule.json` and the npm dist-tags
(`https://registry.npmjs.org/-/package/electron/dist-tags`), both fetched **2026-08-28**:

| Major | Latest stable | Stable date | EOL | Status |
|---|---|---|---|---|
| 44 | 44.0.0 (npm `latest`) | 2026-08-25 | 2027-03-02 | supported, **3 days old, zero patch releases** |
| **43** | **43.4.1** | 2026-06-30 | **2027-01-05** | **supported — chosen** |
| 42 | 42.10.1 | 2026-05-05 | **2026-10-20** | supported, but ~7 weeks of runway left |
| 41 | 41.10.7 | 2026-03-10 | 2026-08-25 | **EOL as of three days ago** |
| 40 / 39 / 38 | 40.10.6 / 39.8.10 / 38.8.6 | — | — | EOL |

So the eligible set is exactly `{42.10.1, 43.4.1, 44.0.0}`, and three of the four versions the
advisory names as "patched" (38.8.6, 39.8.1, 40.8.1) sit on dead lines. 41.0.0's line died on
2026-08-25.

43.4.1 is the middle of the three, and it is the one that trades best:

- **42.10.1** goes EOL 2026-10-20, roughly seven weeks out. Pinning it would mean the security
  posture this step exists to establish expires before the refactor ships.
- **44.0.0** is a `.0` released three days ago with no patch releases behind it, and it additionally
  drops macOS 12, Windows ia32 and Linux armv7l, statically links ANGLE, and removes the renderer
  `clipboard` module. None of that breaks *this* app, but it is the least-baked option.
- **43.4.1** is a mature patch release on a supported line with EOL 2027-01-05 — about four months
  of security coverage, enough to carry Steps 5b through 9 without a second forced bump.

Re-pin before **2027-01-05**.

## API changes required: none

Every `## Planned Breaking API Changes` section from 34.0 through 44.0 was read in full
(`docs/latest/breaking-changes.md`, fetched 2026-08-28) and cross-checked against the complete
Electron surface these two files touch:

`app.whenReady/quit/exit`, `app.on("activate"|"before-quit"|"window-all-closed"|"quit")`,
`BrowserWindow` (`width/height/minWidth/minHeight/backgroundColor/title/show/webPreferences`),
`BrowserWindow.getAllWindows`, `once("ready-to-show")`, `show/hide/isDestroyed/reload/loadURL`,
`webContents.setWindowOpenHandler`, `webContents.toggleDevTools`, `Menu.setApplicationMenu` /
`buildFromTemplate` with roles `quit|resetZoom|zoomIn|zoomOut|togglefullscreen`,
`shell.openExternal`, `shell.openPath`, `dialog.showErrorBox`, `dialog.showMessageBox`,
`ipcMain.handle`, `contextBridge.exposeInMainWorld`, `ipcRenderer.invoke`, and
`process.versions.{electron,chrome,node}`.

**Nothing in that list changed between 33.x and 43.x**, so `main.js` and `preload.js` are byte-for-byte
unchanged. The near misses, recorded so the next reader does not have to re-derive them:

| Change | Why it does not apply |
|---|---|
| 43.0 rounded corners on Linux; 43.0 WCO native title-bar layout | Both are **frameless-window** behaviors. This window is framed (no `frame: false`, no `titleBarOverlay`). |
| 43.0 dialog `defaultPath` defaults to Downloads; 43.0 `showHiddenFiles` removed on Linux | Only affect `showOpen/SaveDialog`. This app uses only `showMessageBox` and `showErrorBox`. |
| 39.0 `window.open` popups always resizable | `setWindowOpenHandler` returns `{ action: "deny" }` for every URL, so no popup is ever created. |
| 40.0 renderer `clipboard` deprecated; 44.0 removed | `preload.js` exposes no clipboard surface. |
| 38.0 `ELECTRON_OZONE_PLATFORM_HINT` / `ORIGINAL_XDG_CURRENT_DESKTOP` removed | Neither is read anywhere in `app/`. |
| 36.0 `app.commandLine` lowercasing; 37.0 `ProtocolResponse.session`; 35.0 `WebRequestFilter.urls`; 42.0 `Notification`/OSR/`clearStorageData`; 44.0 `select-client-certificate`, `app.isUnityRunning` | None of these APIs is used. |

The changes that **do** land are install mechanics and platform behavior, not code:

1. **42.0 removed the `postinstall` binary download.** `npm install` / `npm ci` no longer place a
   binary in `node_modules/electron/dist/`. It arrives on the first `npx electron`, or explicitly
   via `npx install-electron --no`. `ELECTRON_SKIP_BINARY_DOWNLOAD` is gone; cross-platform
   selection moved from npm config flags to `ELECTRON_INSTALL_PLATFORM` / `ELECTRON_INSTALL_ARCH`.
   Any CI or air-gapped install recipe for `app/` has to account for this.
2. **`@electron/get` 2.0.3 → 5.1.0**, now ESM-only, and it dropped `global-agent`/`got` for
   `undici`. `GLOBAL_AGENT_HTTPS_PROXY` no longer steers the binary download; use `ELECTRON_MIRROR`
   / `ELECTRON_CUSTOM_DIR` on a proxied node. The lockfile shrank by ~700 lines for this reason.
3. **38.0 made `--ozone-platform` default to `auto`**, so on a Wayland session Electron now runs as
   a native Wayland client rather than under XWayland. `--ozone-platform=x11` forces the old path.
4. Runtime substrate: Node **20.18.3 → 24.18.1** in the main process, Chromium **130 → 150** in the
   renderer. `main.js` is CommonJS and uses only `child_process`, `http`, `path`, `crypto` and
   WHATWG `new URL()`, none of which Node 24 changed.

## What was verified on this host

Ubuntu 22.04.5, x86_64, glibc 2.35, Node v24.18.1, npm 11.16.0, `DISPLAY=:1`,
`XDG_SESSION_TYPE=x11`, Xvfb present.

| Check | Result |
|---|---|
| `npm install` (network reachable) | OK — lockfile regenerated by npm, never hand-edited |
| `npm ls electron` | `electron@43.4.1` |
| `package.json` pin | `"electron": "43.4.1"` — no range operator |
| `package-lock.json` | root `devDependencies.electron == "43.4.1"`; `node_modules/electron` resolves 43.4.1 |
| `npm audit` | 0 vulnerabilities |
| `npx install-electron --no` | binary downloaded; `node_modules/electron/dist/version` == `43.4.1` |
| `electron --version` (env scrubbed) | `v43.4.1` |
| `node --check main.js`, `node --check preload.js` | both OK |
| Main-process boot under `xvfb-run` | exit 0; reported `electron=43.4.1 chrome=150.0.7871.224 node=24.18.1` |
| Hidden `BrowserWindow` on `about:blank` with the **real** `preload.js` and `main.js`'s exact `webPreferences`, under `xvfb-run` | exit 0; `ready-to-show` fired; `window.lm3desktop` exposed `isElectron`, `versions`, and both IPC functions |

Both window probes ran twice, once with `--no-sandbox` and once without. **Both passed.**
`node_modules/electron/dist/chrome-sandbox` is mode `0755` and not setuid root, but this kernel
allows unprivileged user namespaces, so Chromium uses the namespace sandbox and does not abort.
`--no-sandbox` (the existing `npm run start:nosandbox`) remains the safe default for hosts that
restrict user namespaces.

**The `ELECTRON_RUN_AS_NODE` trap, measured live.** This environment exports
`ELECTRON_RUN_AS_NODE=1` and `ELECTRON_NO_ATTACH_CONSOLE=1`. With them set,
`./node_modules/.bin/electron --version` prints `v24.18.1` — the bundled *Node* version — and exits
0, which looks like a plausible green and means nothing. Every check above was run under
`env -u ELECTRON_RUN_AS_NODE -u ELECTRON_NO_ATTACH_CONSOLE`. Any future smoke script must do the
same.

## What was NOT verified, and cannot be from here

- **Packaging.** There is nothing to package. `app/package.json` has no `build` block, no `appId`
  and no icon; there is no electron-builder, electron-forge or `@electron/packager` anywhere in the
  repo, and `.github/workflows/ci.yml` has no Node step at all. What is verified above is
  **install + launch**, not packaging. Adding a packager is separate work.
- **The real LM3 window.** No probe started the LM3 server or loaded the LM3 UI — deliberately. The
  Chromium 130 → 150 renderer jump is a real regression surface for
  `leafmachine3/server/ui/`, and it is untested by this change.
- **Wayland.** This host is X11-only, so the 38.0 ozone default-to-`auto` path cannot be exercised
  here. A user on GNOME/Wayland gets a native Wayland client for the first time.
- **Real GPU rendering.** Both probes ran with `--disable-gpu` under Xvfb (SwiftShader). Compositor
  interaction, the menu bar as a desktop environment actually draws it, and hardware acceleration
  are all unverified.
- **macOS and Windows.** Nothing here was run off Linux: 44.0's macOS 12 drop is moot at 43, but
  42.0's `UNNotification` code-signing requirement, 40.0's dSYM `tar.xz`, and Windows behavior in
  general are untested.
- **`requestSingleInstanceLock()` itself.** Not called by this app and deliberately not added here.
  Step 5b must verify it returns `true` on a first instance and that a second instance is refused —
  and must set the deployment-scoped `app.setPath("userData", …)` **before** requesting the lock,
  because Chromium's `ProcessSingleton` files (`SingletonLock`, `SingletonSocket`,
  `SingletonCookie`) are created inside `userData`. Requesting the lock first would give every
  deployment one shared lock.

---

## Addendum (2026-08-29) — packaging now exists

The "**Packaging.** There is nothing to package." entry above is out of date as of 2026-08-29.
`app/electron-builder.yml` (electron-builder **26.15.3**, pinned exactly) now defines Linux, Windows
and macOS targets, and `npm run pack` / `dist:linux` / `dist:win` / `dist:mac` / `validate:config`
drive them. See **`app/ELECTRON_PACKAGING.md`**.

What that change does and does not alter for this document:

- **The Electron pin is untouched.** Still `"electron": "43.4.1"`, exact, no range operator. Still
  re-pin before **2027-01-05**. The packaged Linux artifacts were verified to carry
  `electron=43.4.1 chrome=150.0.7871.224 node=24.18.1`.
- **No application code changed.** `main.js` and `preload.js` are still byte-for-byte unchanged, and
  `requestSingleInstanceLock()` is still deliberately absent — Step 5b still owns it, and the
  "What was NOT verified" note about it still stands verbatim.
- **The `ELECTRON_RUN_AS_NODE` trap was re-measured today and still bites**: unscrubbed,
  `./node_modules/.bin/electron --version` prints `v24.18.1`; scrubbed, `v43.4.1`. Every packaging
  command was run under `env -u ELECTRON_RUN_AS_NODE -u ELECTRON_NO_ATTACH_CONSOLE`.
- **Linux packaging is now verified; the other two are still not.** Three Linux artifacts were built
  and the AppImage was launched, attached to a stand-in LM3 server, and shut down cleanly with no
  orphans. Windows and macOS were schema-validated and cross-built to `--dir` only; no installer was
  produced, nothing was signed, nothing was notarized, and no Windows machine or Mac was involved.
- **"The real LM3 window" is still unverified.** The packaging smoke test used a stand-in server on
  purpose, so the Chromium 130 → 150 renderer jump remains untested against `leafmachine3/server/ui/`.

One packaging-relevant fact about `main.js` was measured while doing this, and is recorded here
because it is a runtime property of the shell rather than of the config: in a packaged app,
`ROOT = path.resolve(__dirname, "..")` resolves to `<install>/resources`, so the default
`LM3_PYTHON` (`ROOT/.venv_LM3/bin/python`) does not exist. A packaged app can **attach** to a running
LM3 server but cannot **spawn** one without `LM3_PYTHON` being set. Details and the measurement are
in `ELECTRON_PACKAGING.md` §2.
