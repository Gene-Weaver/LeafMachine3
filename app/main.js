/* LeafMachine3 desktop shell.
 *
 * Deliberately thin: the entire interface is the LM3 server's own web UI, so this process only
 * (1) makes sure a server is listening, (2) opens a window on it, and (3) shuts that server down
 * again on quit. Keeping the UI server-side means the same interface works in a plain browser,
 * which is also how it gets tested.
 *
 * (3) is not best-effort. A server left running holds the port and keeps answering /healthz, so
 * the NEXT launch attaches to it instead of starting one -- silently running however old that
 * process's code is -- and quitting that window used to leave it behind again. Shutdown therefore
 * escalates until the port is actually quiet, and covers a server we attached to as well as one we
 * spawned. Set LM3_KEEP_SERVER=1 to opt out (see below).
 */
const { app, BrowserWindow, Menu, shell, dialog, ipcMain } = require("electron");
const { spawn } = require("child_process");
const http = require("http");
const path = require("path");
const crypto = require("crypto");

const HOST = process.env.LM3_HOST || "127.0.0.1";
const PORT = parseInt(process.env.LM3_PORT || "8765", 10);
const ROOT = path.resolve(__dirname, "..");                 // the LM3 checkout
const PYTHON = process.env.LM3_PYTHON || path.join(ROOT, ".venv_LM3", "bin", "python");
// Reuse an existing token so attaching to an already-running server keeps working.
const TOKEN = process.env.LM3_SERVER_TOKEN || crypto.randomBytes(16).toString("hex");
// Quitting stops the server -- including one this shell merely attached to, because that is the
// case that produces orphans. Opt out when you are running a server by hand (browser testing, a
// debugger, `lm3 serve` in a terminal) and want the window to close without taking it with you.
const KEEP_SERVER = /^(1|true|yes|on)$/i.test(String(process.env.LM3_KEEP_SERVER || ""));
// A cold LM3 server imports torch + CUDA, which on a loaded machine is minutes, not seconds (a warm
// start is ~1s). Too small a budget here is itself an orphan factory: ensureServer gives up, the
// app quits, and the server it started finishes booting into an empty room.
const STARTUP_TIMEOUT_S = Math.max(5, parseInt(process.env.LM3_START_TIMEOUT_S || "240", 10) || 240);

let serverProc = null;      // only set when WE spawned it
let serverPid = null;       // set for a spawned server AND (from /healthz) one we attached to
let teardown = null;        // the in-flight shutdown promise, once quitting has begun
let win = null;

const base = () => `http://${HOST}:${PORT}`;

/** True when a server answers /healthz. Also records its pid, which is what makes a server we
 *  attached to killable -- without it, shutdown has nothing to escalate to. */
function ping(timeoutMs = 1200) {
  return new Promise((resolve) => {
    const req = http.get(`${base()}/healthz`, { timeout: timeoutMs }, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (d) => { if (body.length < 4096) body += d; });
      res.on("end", () => {
        if (res.statusCode === 200 && !serverPid) {
          try {
            const pid = JSON.parse(body).pid;
            if (Number.isInteger(pid) && pid > 1) serverPid = pid;
          } catch (_) { /* an older server without `pid`: signaling just is not available */ }
        }
        resolve(res.statusCode === 200);
      });
    });
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
  });
}

/** True while the server process exists. Signal 0 tests without touching it. */
function serverAlive() {
  if (!serverPid) return false;
  try {
    process.kill(serverPid, 0);
    return true;
  } catch (err) {
    return err.code === "EPERM";                            // exists, not ours to signal
  }
}

/** Signal the server's whole process group, falling back to the bare pid.
 *
 * The group is what reaches anything the server spawned for itself. A negative pid can only ever
 * name a group whose leader IS that pid, so when the server is not a group leader (someone started
 * it by hand in a shell job) this cannot stray into an unrelated group -- it gets ESRCH and falls
 * through to the pid. Returns false when there is nothing left alive to signal.
 */
function signalServer(sig) {
  if (!serverPid) return false;
  try {
    process.kill(-serverPid, sig);
    return true;
  } catch (err) {
    if (err.code === "ESRCH") {
      try {
        process.kill(serverPid, sig);
        return true;
      } catch (_) {
        return false;
      }
    }
    return false;
  }
}

/** POST /v1/shutdown -- the only lever that works on a server we did not spawn. */
function requestShutdown(timeoutMs = 4000) {
  return new Promise((resolve) => {
    const req = http.request(
      {
        host: HOST, port: PORT, path: "/v1/shutdown", method: "POST", timeout: timeoutMs,
        headers: { Authorization: `Bearer ${TOKEN}`, "Content-Length": "0" },
      },
      (res) => { res.resume(); resolve(res.statusCode === 200); },
    );
    req.on("error", () => resolve(false));
    req.on("timeout", () => { req.destroy(); resolve(false); });
    req.end();
  });
}

/** Wait for the server to be neither answering nor running. */
function serverGone({ tries = 24, delayMs = 250 } = {}) {
  return waitFor(async () => !(await ping(400)) && !serverAlive(), { tries, delayMs });
}

/** Stop the LM3 server this window is bound to, escalating until the port is quiet.
 *
 * Each step alone has a hole, which is why there are three:
 *   1. POST /v1/shutdown -- the only step available for a server we ATTACHED to, and the only one
 *      that lets the server exit from the inside on its own terms.
 *   2. SIGTERM the group -- for a server too wedged to answer HTTP at all.
 *   3. SIGKILL the group -- for uvicorn parked in graceful shutdown waiting on the SSE streams
 *      this UI holds open. That wait is unbounded, and it is how these servers used to survive
 *      a SIGTERM and outlive the app.
 */
async function shutdownServer() {
  if (KEEP_SERVER) return "left running (LM3_KEEP_SERVER)";

  const responding = await ping(600);                       // also fills in serverPid, if we attached
  if (responding) {
    await requestShutdown();
    if (await serverGone()) return "stopped";
  }
  if (!serverPid) return responding ? "unreachable" : "already down";
  if (!serverAlive()) return "already down";

  signalServer("SIGTERM");
  if (await serverGone({ tries: 40, delayMs: 250 })) return "terminated";

  signalServer("SIGKILL");
  if (await serverGone()) return "killed";
  return "STILL RUNNING";
}

async function waitFor(predicate, { tries = 60, delayMs = 500 } = {}) {
  for (let i = 0; i < tries; i += 1) {
    if (await predicate(i)) return true;
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return false;
}

/** Start the LM3 server ourselves, unless one is already listening on the port. */
async function ensureServer() {
  if (await ping()) return "attached";                      // someone already runs it
  serverProc = spawn(
    PYTHON,
    ["-m", "uvicorn", "leafmachine3.server.app:create_app", "--factory",
      "--host", HOST, "--port", String(PORT)],
    {
      cwd: ROOT,
      // LM3_OWNER_PID arms the server's own watchdog: if this process is SIGKILLed or crashes,
      // no shutdown request is ever sent, and the server exits on noticing we are gone. Under
      // LM3_KEEP_SERVER the variable is withheld, because the opt-out has to reach BOTH halves of
      // the mechanism -- arming it here would have the watchdog kill the very server this process
      // just promised to leave running.
      env: {
        ...process.env,
        LM3_SERVER_TOKEN: TOKEN,
        ...(KEEP_SERVER ? {} : { LM3_OWNER_PID: String(process.pid) }),
      },
      stdio: ["ignore", "pipe", "pipe"],
      // Its own process group, so one signal reaches the server and everything it spawned -- and
      // so a signal aimed at that group can never travel back up into this process.
      detached: true,
    },
  );
  serverPid = serverProc.pid;
  let starting = true;
  let exited = null;                                        // set if it dies before it ever serves
  const tail = [];                                          // last few lines, for the error dialog
  const note = (chunk) => {
    for (const line of String(chunk).split("\n")) {
      if (line.trim()) tail.push(line.trim());
    }
    while (tail.length > 8) tail.shift();
  };
  serverProc.stdout.on("data", (d) => { note(d); process.stdout.write(`[lm3] ${d}`); });
  serverProc.stderr.on("data", (d) => { note(d); process.stderr.write(`[lm3] ${d}`); });
  serverProc.on("exit", (code) => {
    serverProc = null;
    exited = code;
    // While starting, the failure is reported by the throw below (with the log tail, which is what
    // actually says WHY) -- two dialogs for one failure would just be noise.
    if (code && code !== 0 && !app.isQuiting && !starting) {
      dialog.showErrorBox("LeafMachine3 server stopped", `The LM3 server exited with code ${code}.`);
    }
  });

  const tries = Math.ceil((STARTUP_TIMEOUT_S * 1000) / 500);
  const ok = await waitFor(async (i) => {
    if (exited !== null) return true;                       // stop waiting -- diagnosed below
    if (i && i % 20 === 0) console.log(`[lm3-desktop] still waiting for the server (${i / 2}s)…`);
    return ping();
  }, { tries, delayMs: 500 });
  starting = false;

  // A server that died on the way up is the common failure (the port is taken, the venv is wrong),
  // and it fails in milliseconds -- so say so immediately instead of sitting out the whole budget.
  if (exited !== null && !(await ping(600))) {
    const why = tail.length ? `\n\n${tail.join("\n")}` : "";
    throw new Error(`the LM3 server exited (code ${exited}) before it began serving.`
      + ` Is something else already on ${base()}?${why}`);
  }
  if (!ok) {
    // Never leave the half-started server behind. It would finish booting seconds later, hold the
    // port, and be adopted by the next launch -- the orphan this whole file is careful about.
    signalServer("SIGTERM");
    setTimeout(() => signalServer("SIGKILL"), 2000).unref();
    throw new Error(`the LM3 server did not come up on ${base()} within ${STARTUP_TIMEOUT_S}s`);
  }
  return "spawned";
}

function buildMenu() {
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {
      label: "LeafMachine3",
      submenu: [
        { label: "Reload", accelerator: "CmdOrCtrl+R", click: () => win && win.reload() },
        { label: "Toggle Developer Tools", accelerator: "CmdOrCtrl+Shift+I",
          click: () => win && win.webContents.toggleDevTools() },
        { type: "separator" },
        { label: "Open in browser", click: () => shell.openExternal(base()) },
        { type: "separator" },
        { role: "quit" },
      ],
    },
    { label: "View", submenu: [{ role: "resetZoom" }, { role: "zoomIn" }, { role: "zoomOut" },
      { type: "separator" }, { role: "togglefullscreen" }] },
  ]));
}

async function createWindow() {
  win = new BrowserWindow({
    width: 1680,
    height: 1020,
    minWidth: 1100,
    minHeight: 700,
    backgroundColor: "#101012",                             // matches the UI so there is no white flash
    title: "LeafMachine3",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.once("ready-to-show", () => win.show());
  win.webContents.setWindowOpenHandler(({ url }) => {       // external links go to the real browser
    // Only web URLs are handed to the OS. shell.openExternal() will launch a
    // registered handler for ANY scheme it is given (file:, smb:, ms-msdt:, …),
    // so the scheme is checked before the URL leaves this process.
    if (isWebUrl(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  // Hand the token over in the URL rather than relying on the server embedding it in the HTML.
  // api.js accepts ?token= and stashes it, so the desktop app authenticates even when the server
  // is run with LM3_EMBED_TOKEN=0 (which stops any other local process from simply GETting "/"
  // to read the secret).
  await win.loadURL(`${base()}/?token=${encodeURIComponent(TOKEN)}`);
}

/** http/https only — everything else is refused rather than handed to the OS. */
function isWebUrl(url) {
  try {
    const p = new URL(String(url)).protocol;
    return p === "http:" || p === "https:";
  } catch {
    return false;                                           // not parseable as a URL at all
  }
}

// Let the renderer hand a produced file (overlay, timing report, STL) to the OS. Only this server's
// own origin and file paths INSIDE the LM3 checkout are allowed.
//
// Both checks are structural, never string prefixes: "http://127.0.0.1:8765@evil.com/" and
// "http://127.0.0.1:8765.evil.com/" both start with base(), and "/datac/.../LM3_evil/run.sh"
// starts with ROOT — so a prefix test would open all three.
ipcMain.handle("lm3:open-external", async (_evt, url) => {
  let u;
  try {
    u = new URL(String(url || ""));
  } catch {
    return false;
  }

  if (u.protocol === "http:" || u.protocol === "https:") {
    const self = new URL(base());
    // `host` carries the port, so this compares 127.0.0.1:8765 as one unit.
    if (u.protocol === self.protocol && u.host === self.host) return shell.openExternal(u.href);
    return false;
  }

  if (u.protocol === "file:") {
    if (u.host) return false;                               // file://evil.com/... is not local
    const p = path.resolve(decodeURIComponent(u.pathname));
    // The separator is what makes this a containment test instead of a prefix test.
    if (p === ROOT || p.startsWith(ROOT + path.sep)) return shell.openPath(p);
  }

  return false;
});

// The UI's Close button: it has already asked the server to stop any running LM3 job (and waited),
// so by the time this fires there is nothing to interrupt. Quitting also stops the server itself --
// see the "before-quit" handler below.
ipcMain.handle("lm3:quit", async (_evt, opts) => {
  // Second, native confirmation. The renderer already asked (and stopped any running job), but
  // Close sits next to the live indicator and quitting is not undoable, so it is worth one
  // deliberate click in a dialog the page cannot dismiss for you.
  const running = !!(opts && opts.running);
  const { response } = await dialog.showMessageBox(win, {
    type: "question",
    buttons: ["Cancel", "Close LeafMachine3"],
    defaultId: 0,                 // Enter cancels
    cancelId: 0,
    title: "Close LeafMachine3",
    message: "Close LeafMachine3?",
    detail: running
      ? "The running LM3 job has been stopped and checkpointed. It is resumable -- reopening and "
        + "pressing Start LM3 continues it from where it stopped."
      : "No LM3 job is running.",
  });
  if (response !== 1) return false;
  app.isQuiting = true;
  app.quit();
  return true;
});

app.whenReady().then(async () => {
  buildMenu();
  try {
    const how = await ensureServer();
    console.log(`[lm3-desktop] server ${how} at ${base()}`);
    await createWindow();
  } catch (err) {
    dialog.showErrorBox("Cannot start LeafMachine3", String(err && err.message ? err.message : err));
    app.quit();
  }
  app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
});

// Closing the app means the server goes too -- it is part of the app, not a bystander that
// happens to be listening. This has to run in "before-quit" rather than "quit": "quit" fires
// synchronously as the process is already on its way out, so anything asynchronous started there
// (waiting for a port to fall quiet, escalating a signal) is a race that is always lost -- which
// is how the old SIGTERM-and-hope handler let servers survive.
app.on("before-quit", (event) => {
  app.isQuiting = true;
  event.preventDefault();                                   // the teardown below exits for us
  if (teardown) return;                                     // ...and a second Quit must not pre-empt it
  if (win && !win.isDestroyed()) { try { win.hide(); } catch (_) {} }
  teardown = shutdownServer()
    .then((how) => console.log(`[lm3-desktop] server ${how}`))
    .catch((err) => console.error("[lm3-desktop] server shutdown failed:", err))
    .finally(() => app.exit(0));                            // exit(), not quit(): skips before-quit
});

app.on("window-all-closed", () => app.quit());

// Ctrl-C (or a SIGTERM from whatever launched us) should tear down exactly like Quit does.
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, () => app.quit());

app.on("quit", () => {
  // Last resort: something exited us without going through before-quit. Nothing asynchronous can
  // be relied on here, so this is the one blunt signal we can still land.
  if (!teardown && !KEEP_SERVER) signalServer("SIGKILL");
});
