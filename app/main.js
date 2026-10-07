/* LeafMachine3 desktop shell.
 *
 * Deliberately thin: the entire interface is the LM3 server's own web UI, so this process only
 * (1) proves which deployment it belongs to, (2) makes sure a server for THAT deployment is
 * listening, and (3) opens a window on it. Keeping the UI server-side means the same interface
 * works in a plain browser, which is also how it gets tested.
 *
 * What this file no longer does, and why (plan sections 2.5, 2.11, invariants 9/10/12):
 *
 *   * it does not scrape `pid` out of an unauthenticated /healthz response and signal it. A PID in
 *     a response body authorizes nothing. Shutdown authority is the RETAINED CHILD HANDLE of a
 *     server this process spawned, whose instance ID and deployment key still match -- and nothing
 *     else. A server we merely attached to is left running when the window closes;
 *   * it does not invent a bearer token and hope an existing server accepts it. The token comes
 *     from this deployment's connection.private.json (section 2.12), or -- for a server we are
 *     about to spawn -- is minted BY us and handed to that child;
 *   * it does not treat "something answered 200 on the port" as "my server is up". A valid LM3
 *     server belonging to a DIFFERENT deployment and an unrelated service are two distinct, named
 *     refusals, and neither may be attached to.
 *
 * Structure note: every module-level side effect lives in boot(), which runs only under Electron.
 * That is what lets app/test/*.test.js require this file in plain Node and exercise the deployment
 * key, the port rule, the /healthz classification and the shutdown decision without a GUI -- and
 * it is also what section 2.11 asks for, since requestSingleInstanceLock() must run before IPC
 * registration with side effects and before any server is contacted.
 */
const http = require("http");
const path = require("path");
const os = require("os");
const fs = require("fs");
const crypto = require("crypto");

const RUNNING_IN_ELECTRON = Boolean(process.versions && process.versions.electron);

// --------------------------------------------------------------------------------------------- //
// Section 2.1 -- deployment identity
// --------------------------------------------------------------------------------------------- //
const DEFAULT_DEPLOYMENT_ID = "default";
const DEFAULT_PORT = 8765;
const DEPLOYMENT_SLUG_LENGTH = 32;
const DEPLOYMENT_HASH_LENGTH = 8;
const APP_DIRNAME = "lm3";
const CONNECTION_PRIVATE_FILENAME = "connection.private.json";
const SERVICE_NAME = "leafmachine3";

class DeploymentIdentityError extends Error {}
class DeploymentPortError extends Error {}
class BackendLaunchError extends Error {}

// --------------------------------------------------------------------------------------------- //
// Python/backend discovery -- pure and bounded (no probe process, no import, no network)
// --------------------------------------------------------------------------------------------- //
function isRunnableFile(file, { platform = process.platform, fsImpl = fs } = {}) {
  if (!file) return false;
  try {
    if (!fsImpl.statSync(file).isFile()) return false;
    // Windows decides executability from the extension. POSIX must reject a readable but
    // non-executable script now, rather than surfacing spawn EACCES after a token has been minted.
    if (platform !== "win32") fsImpl.accessSync(file, fs.constants.X_OK);
    return true;
  } catch (_) {
    return false;
  }
}

function pythonCandidates(prefix, platform = process.platform) {
  if (!prefix) return [];
  const root = path.resolve(prefix);
  return platform === "win32"
    ? [path.join(root, "Scripts", "python.exe"), path.join(root, "python.exe")]
    : [path.join(root, "bin", "python3"), path.join(root, "bin", "python")];
}

function findOnPath(names, env = process.env, platform = process.platform, fsImpl = fs) {
  const raw = String(env.PATH || "");
  if (!raw) return null;
  const separator = platform === "win32" ? ";" : path.delimiter;
  for (const directory of raw.split(separator).filter(Boolean)) {
    for (const name of names) {
      const candidate = path.join(directory, name);
      if (isRunnableFile(candidate, { platform, fsImpl })) return candidate;
    }
  }
  return null;
}

/** Resolve the installed LM3 backend without executing candidates.
 *
 * This deliberately returns a launch SPEC instead of "the Python path": a wheel install already
 * supplies the canonical `lm3 serve` command, while a developer checkout and an activated venv
 * naturally supply Python. No `spawnSync --version`, import probe, PATH shell, or unbounded child
 * is used here; a wrong candidate fails through the existing bounded server-start handshake.
 */
function resolveBackendLaunch({ env = process.env, root = path.resolve(__dirname, ".."),
                                packaged = false, platform = process.platform,
                                fsImpl = fs } = {}) {
  const explicit = String(env.LM3_PYTHON || "").trim();
  if (explicit) {
    const command = path.resolve(expandHome(explicit, env));
    if (!isRunnableFile(command, { platform, fsImpl })) {
      throw new BackendLaunchError(
        `LM3_PYTHON names ${command}, but that file is not a runnable Python interpreter.`);
    }
    return {
      command,
      argsPrefix: ["-m", "uvicorn", "leafmachine3.server.app:create_app", "--factory"],
      cwd: root,
      source: "LM3_PYTHON",
    };
  }

  const prefixes = [];
  const addPrefix = (value, source) => {
    const clean = String(value || "").trim();
    if (clean) prefixes.push([expandHome(clean, env), source]);
  };
  addPrefix(env.VIRTUAL_ENV, "VIRTUAL_ENV");
  addPrefix(env.CONDA_PREFIX, "CONDA_PREFIX");
  // In a source checkout this is the normal path. In a package it is considered only when
  // LM3_ROOT was explicitly supplied; <install>/resources is known not to contain the venv.
  if (!packaged || String(env.LM3_ROOT || "").trim()) addPrefix(root, "LM3_ROOT checkout");

  const seen = new Set();
  for (const [prefix, source] of prefixes) {
    for (const command of pythonCandidates(prefix, platform)) {
      if (seen.has(command)) continue;
      seen.add(command);
      if (!isRunnableFile(command, { platform, fsImpl })) continue;
      return {
        command,
        argsPrefix: ["-m", "uvicorn", "leafmachine3.server.app:create_app", "--factory"],
        cwd: source === "LM3_ROOT checkout" ? root : os.homedir(),
        source,
      };
    }
  }

  // A normal installed wheel exposes `lm3`. On Windows pip creates lm3.exe; deliberately ignore
  // .cmd/.bat so spawning never needs a shell (and therefore never reparses user-controlled text).
  const consoleNames = platform === "win32" ? ["lm3.exe"] : ["lm3"];
  const console = findOnPath(consoleNames, env, platform, fsImpl);
  if (console) {
    return { command: console, argsPrefix: ["serve"], cwd: os.homedir(), source: "PATH" };
  }
  return null;
}

/** Section 2.1's slug, character by character. Deliberately NOT toLowerCase() and NOT NFKC.
 *
 * `String.prototype.toLowerCase()` and Unicode normalization are the two things that do NOT agree
 * between Python and JavaScript (`ß` casefolds to `ss` in Python but lowercases to `ß` here, and
 * normalization tables move with engine versions). A key that disagrees between the two produces a
 * runtime-directory mismatch BEFORE any golden test could catch it -- the shell would single-
 * instance-lock against one deployment and talk to another. So the slug is mechanical ASCII and
 * purely cosmetic; the sha256 of the raw UTF-8 bytes is what actually separates namespaces.
 *
 * `for...of` iterates CODE POINTS, which is what makes an astral-plane character (an emoji) count
 * as one replacement here and in Python's `for ch in raw`, rather than two UTF-16 halves.
 */
function asciiSlug(raw) {
  const mapped = [];
  for (const ch of String(raw)) {
    const code = ch.codePointAt(0);
    if (code >= 0x41 && code <= 0x5a) mapped.push(String.fromCharCode(code + 0x20));   // A-Z
    else if ((code >= 0x61 && code <= 0x7a) || (code >= 0x30 && code <= 0x39)) mapped.push(ch);
    else mapped.push("-");                                   // everything else, non-ASCII included
  }
  const collapsed = [];
  for (const ch of mapped) {
    if (ch === "-" && collapsed[collapsed.length - 1] === "-") continue;
    collapsed.push(ch);
  }
  return collapsed.join("").replace(/^-+/, "").replace(/-+$/, "");
}

/** `ascii_slug(raw)[:32] + "-" + sha256(raw utf-8)[:8]`, literally.
 *
 * The separator survives an empty slug on purpose (`"//"` -> `"-732c4e97"`). Python keeps it, so
 * "tidying up the leading dash" here would desynchronize the two implementations -- which is the
 * one failure mode the golden vectors exist to prevent.
 */
function canonicalDeploymentKey(raw) {
  const digest = crypto.createHash("sha256")
    .update(Buffer.from(String(raw), "utf8"))
    .digest("hex")
    .slice(0, DEPLOYMENT_HASH_LENGTH);
  return `${asciiSlug(raw).slice(0, DEPLOYMENT_SLUG_LENGTH)}-${digest}`;
}

/** The raw LM3_DEPLOYMENT_ID. Unset means the literal "default". Nothing else. */
function rawDeploymentId(env = process.env) {
  const raw = env.LM3_DEPLOYMENT_ID;
  if (raw === undefined || raw === null) return DEFAULT_DEPLOYMENT_ID;
  if (String(raw).trim() === "") {
    throw new DeploymentIdentityError(
      "LM3_DEPLOYMENT_ID is empty or whitespace-only. Unset it for the default deployment, "
      + "or give it a non-empty name.");
  }
  return String(raw);
}

function deploymentKeyFor(env = process.env) {
  return canonicalDeploymentKey(rawDeploymentId(env));
}

function isDefaultDeployment(env = process.env) {
  return rawDeploymentId(env) === DEFAULT_DEPLOYMENT_ID;
}

/** Section 2.1's port rule: 8765 for the default deployment, an EXPLICIT LM3_PORT for any other.
 *
 * Defaulting a named deployment to 8765 is the silent collision this rule exists to prevent -- two
 * "independent" deployments would share one server and one lease without ever saying so.
 */
function resolvePort(env = process.env) {
  const raw = (env.LM3_PORT || "").toString().trim();
  if (raw) {
    const port = Number(raw);
    if (!Number.isInteger(port)) throw new DeploymentPortError(`LM3_PORT is not an integer: ${raw}`);
    if (port < 1 || port > 65535) throw new DeploymentPortError(`LM3_PORT is out of range: ${port}`);
    return port;
  }
  if (isDefaultDeployment(env)) return DEFAULT_PORT;
  throw new DeploymentPortError(
    `deployment ${JSON.stringify(rawDeploymentId(env))} is not the default deployment, so it must `
    + `set LM3_PORT explicitly. Starting it without one would silently collide with the default `
    + `deployment on port ${DEFAULT_PORT}.`);
}

// --------------------------------------------------------------------------------------------- //
// Section 2.12 -- where this deployment's private connection descriptor lives
// --------------------------------------------------------------------------------------------- //
function expandHome(p, env = process.env) {
  const value = String(p);
  if (value === "~") return env.HOME || os.homedir();
  if (value.startsWith(`~${path.sep}`) || value.startsWith("~/")) {
    return path.join(env.HOME || os.homedir(), value.slice(2));
  }
  return value;
}

function isDir(p) {
  try { return fs.statSync(p).isDirectory(); } catch { return false; }
}

/** Mirror of `paths.runtime_base_dir` for the three steps a DESKTOP shell can be in.
 *
 * The scheduler branch (SLURM_TMPDIR and friends) is deliberately not reimplemented: an Electron
 * window inside a batch allocation is not a supported configuration, and a second, drifting copy of
 * that rule is worse than not having it. `LM3_RUNTIME_DIR` covers the case anyway, and
 * `/healthz.paths.runtime_dir` is cross-checked once a server has proved its deployment key.
 */
function runtimeBaseDir(env = process.env) {
  const explicit = (env.LM3_RUNTIME_DIR || "").toString().trim();
  if (explicit) return path.resolve(expandHome(explicit, env));

  if (process.platform === "linux") {
    const xdg = (env.XDG_RUNTIME_DIR || "").toString().trim();
    if (xdg && isDir(xdg)) return path.join(xdg, APP_DIRNAME);
  } else if (process.platform === "darwin") {
    const tmp = (env.TMPDIR || "").toString().trim();
    if (tmp && isDir(tmp)) return path.join(tmp, APP_DIRNAME);
  } else if (process.platform === "win32") {
    const local = (env.LOCALAPPDATA || "").toString().trim();
    if (local) return path.join(local, APP_DIRNAME, "runtime");
  }
  return path.join(userCacheDir(env), APP_DIRNAME, "runtime");
}

function userCacheDir(env = process.env) {
  const home = env.HOME || os.homedir();
  if (process.platform === "win32") {
    return (env.LOCALAPPDATA || "").toString().trim() || path.join(home, "AppData", "Local");
  }
  if (process.platform === "darwin") return path.join(home, "Library", "Caches");
  return (env.XDG_CACHE_HOME || "").toString().trim() || path.join(home, ".cache");
}

function deploymentRuntimeDir(env = process.env) {
  return path.join(runtimeBaseDir(env), deploymentKeyFor(env));
}

function connectionPrivatePath(env = process.env) {
  return path.join(deploymentRuntimeDir(env), CONNECTION_PRIVATE_FILENAME);
}

/** Section 2.1: "a distinct Electron user-data / single-instance identity".
 *
 * Electron keeps its single-instance lock under userData, so scoping userData by the CANONICAL
 * deployment key is the whole mechanism: two launches of one deployment collapse into one window,
 * while two deliberately distinct deployments each hold their own lock, their own window, and
 * their own profile. Derived from the key rather than the raw id because the raw value may contain
 * separators, traversal, or non-ASCII look-alikes that must never reach a path component.
 */
function userDataDirFor(appDataRoot, env = process.env) {
  return path.join(String(appDataRoot), "lm3-desktop", deploymentKeyFor(env));
}

/** Read a connection descriptor. Returns null for absent, unreadable or non-object content. */
function readConnectionDescriptor(file) {
  try {
    const payload = JSON.parse(fs.readFileSync(file, "utf8"));
    return payload && typeof payload === "object" && !Array.isArray(payload) ? payload : null;
  } catch {
    return null;
  }
}

// --------------------------------------------------------------------------------------------- //
// Section 2.11 -- what is on the port, and whether we may attach to it
// --------------------------------------------------------------------------------------------- //
const PORT_STATE = {
  IDLE: "idle",                             // nothing answered
  OURS: "ours",                             // a valid LM3 server for THIS deployment
  WRONG_DEPLOYMENT: "wrong-deployment",     // a valid LM3 server for a DIFFERENT deployment
  UNRELATED: "unrelated",                   // something else entirely
  UNIDENTIFIED: "unidentified",             // an LM3 server too old to name its deployment
  WRONG_INSTANCE: "wrong-instance",         // right deployment, not the instance we spawned
};

class WrongDeploymentOnPortError extends Error {
  constructor(message) { super(message); this.name = "WrongDeploymentOnPortError"; this.kind = PORT_STATE.WRONG_DEPLOYMENT; }
}
class UnrelatedServiceOnPortError extends Error {
  constructor(message) { super(message); this.name = "UnrelatedServiceOnPortError"; this.kind = PORT_STATE.UNRELATED; }
}
class UnidentifiedServerOnPortError extends Error {
  constructor(message) { super(message); this.name = "UnidentifiedServerOnPortError"; this.kind = PORT_STATE.UNIDENTIFIED; }
}
class WrongInstanceOnPortError extends Error {
  constructor(message) { super(message); this.name = "WrongInstanceOnPortError"; this.kind = PORT_STATE.WRONG_INSTANCE; }
}
class MissingCredentialsError extends Error {
  constructor(message) { super(message); this.name = "MissingCredentialsError"; }
}

/** Decide what is on the port from one /healthz body. Pure -- no I/O, no globals.
 *
 * `expectedInstanceId` is section 2.11's handshake for a server WE spawned: the shell mints the ID,
 * passes it down as LM3_INSTANCE_ID, and refuses anything on the port that answers with a different
 * one. Without it, a shell that loses a startup race cannot tell its own server from one that
 * already owned the port -- both answer 200.
 */
function classifyHealth(body, { deploymentKey, expectedInstanceId = null } = {}) {
  if (!body || typeof body !== "object" || Array.isArray(body)) return PORT_STATE.UNRELATED;
  if (body.service !== SERVICE_NAME) return PORT_STATE.UNRELATED;
  const key = typeof body.deployment_key === "string" ? body.deployment_key : "";
  if (!key) return PORT_STATE.UNIDENTIFIED;
  if (key !== deploymentKey) return PORT_STATE.WRONG_DEPLOYMENT;
  if (expectedInstanceId && body.instance_id !== expectedInstanceId) return PORT_STATE.WRONG_INSTANCE;
  return PORT_STATE.OURS;
}

/** Turn a refusal into the NAMED error section 2.11 requires. Never called for OURS/IDLE. */
function portConflictError(state, { url, body = null, expectedInstanceId = null } = {}) {
  const where = url || "the configured port";
  switch (state) {
    case PORT_STATE.WRONG_DEPLOYMENT:
      return new WrongDeploymentOnPortError(
        `a valid LM3 server for a DIFFERENT deployment is on ${where}: it reports deployment `
        + `${JSON.stringify((body && body.deployment_key) || "")}`
        + `${body && body.deployment_id ? ` (${body.deployment_id})` : ""}. `
        + `Give this deployment its own LM3_PORT, or launch the other one instead. `
        + `Attaching would put two deployments on one server.`);
    case PORT_STATE.UNIDENTIFIED:
      return new UnidentifiedServerOnPortError(
        `an LM3 server on ${where} does not report a deployment key, so it cannot be verified. `
        + `It is older than this desktop app; restart it from this checkout, or free the port.`);
    case PORT_STATE.WRONG_INSTANCE:
      return new WrongInstanceOnPortError(
        `another LM3 server instance for this deployment answered ${where} `
        + `(expected instance ${expectedInstanceId}, got ${JSON.stringify((body && body.instance_id) || "")}). `
        + `The server this app started is not the one holding the port.`);
    default:
      return new UnrelatedServiceOnPortError(
        `an unrelated service is on ${where}; it is not an LM3 server. `
        + `Free the port, or set LM3_PORT to one this deployment can own.`);
  }
}

// --------------------------------------------------------------------------------------------- //
// Section 2.11 / invariants 9-10 -- what this process is allowed to shut down
// --------------------------------------------------------------------------------------------- //
/** The shutdown decision, stated as data so it can be tested without a GUI or a server.
 *
 * "Electron shuts down only a server represented by its retained child handle with a matching
 * instance ID and deployment key" (section 2.11). Every other outcome is "leave it running" -- and
 * that includes the case this file used to treat as the important one, an attached server, which
 * used to be escalated to SIGKILL on quit.
 */
function shutdownPlan({ hasHandle = false, keepServer = false, spawnedInstanceId = null,
                        health = null, deploymentKey = null } = {}) {
  if (!hasHandle) {
    return { act: false, reason: "attached", detail: "left running (this app did not start it)" };
  }
  if (keepServer) {
    return { act: false, reason: "keep-server", detail: "left running (LM3_KEEP_SERVER)" };
  }
  if (health) {
    if (deploymentKey && health.deployment_key && health.deployment_key !== deploymentKey) {
      return { act: false, reason: "deployment-mismatch",
               detail: "left running (it answers for a different deployment than we started)" };
    }
    if (spawnedInstanceId && health.instance_id && health.instance_id !== spawnedInstanceId) {
      return { act: false, reason: "instance-mismatch",
               detail: "left running (a different server instance holds the port)" };
    }
  }
  return { act: true, reason: "owned", detail: "stopping the server this app started" };
}

// --------------------------------------------------------------------------------------------- //
// Everything below runs only under Electron
// --------------------------------------------------------------------------------------------- //
function boot() {
  const { app, BrowserWindow, Menu, shell, dialog, ipcMain } = require("electron");
  const { spawn } = require("child_process");

  const HOST = process.env.LM3_HOST || "127.0.0.1";
  // The checkout when there is one. A packaged app resolves this to <install>/resources, but the
  // backend resolver does not mistake that for a checkout unless LM3_ROOT was explicitly supplied.
  const ROOT = path.resolve((process.env.LM3_ROOT || "").trim() || path.join(__dirname, ".."));
  let backend = null;
  let backendError = null;
  try {
    backend = resolveBackendLaunch({
      env: process.env, root: ROOT, packaged: Boolean(app.isPackaged), platform: process.platform,
    });
  } catch (err) {
    backendError = err;
  }
  // A lifecycle preference, no longer an ownership safety switch (Step 5b). It says "do not stop
  // the server I started when I quit"; it has nothing to say about a server we merely attached to,
  // because that one is never stopped in the first place.
  const KEEP_SERVER = /^(1|true|yes|on)$/i.test(String(process.env.LM3_KEEP_SERVER || ""));
  // A cold LM3 server imports torch + CUDA, which on a loaded machine is minutes, not seconds (a
  // warm start is ~1s). Too small a budget here is itself an orphan factory: ensureServer gives up,
  // the app quits, and the server it started finishes booting into an empty room.
  const STARTUP_TIMEOUT_S = Math.max(5, parseInt(process.env.LM3_START_TIMEOUT_S || "240", 10) || 240);

  let DEPLOYMENT_KEY = null;
  let PORT = null;
  let identityError = null;
  try {
    DEPLOYMENT_KEY = deploymentKeyFor(process.env);
    PORT = resolvePort(process.env);
  } catch (err) {
    identityError = err;
  }

  let serverProc = null;              // the RETAINED CHILD HANDLE: the whole of our control authority
  let spawnedInstanceId = null;       // what we told that child to call itself
  let lastHealth = null;              // the most recent /healthz body, for diagnostics only
  let token = null;
  let teardown = null;                // the in-flight shutdown promise, once quitting has begun
  let win = null;

  const base = () => `http://${HOST}:${PORT}`;

  // ----------------------------------------------------------------------------------------- //
  // Identity first: user data, then the single-instance lock, then IPC, then any server.
  // ----------------------------------------------------------------------------------------- //
  if (identityError) {
    // Nothing about this app can be correct without a deployment identity, and a dialog needs a
    // ready app -- so fail after ready, with the real message, rather than half-starting.
    app.whenReady().then(() => {
      dialog.showErrorBox("Cannot start LeafMachine3", String(identityError.message || identityError));
      app.exit(2);
    });
    return;
  }

  // Section 2.1: "a distinct Electron user-data / single-instance identity". Electron's
  // single-instance lock lives under userData, so scoping userData by the canonical deployment key
  // is what makes two deliberately distinct deployments able to hold two locks and two windows
  // while two launches of the SAME deployment collapse into one.
  app.setPath("userData", userDataDirFor(app.getPath("appData"), process.env));

  // BEFORE whenReady(), before IPC registration with side effects, before ensureServer().
  // additionalData tells the first instance what was requested; it does not create a differently
  // scoped lock (section 2.11).
  const gotLock = app.requestSingleInstanceLock({
    deploymentKey: DEPLOYMENT_KEY,
    deploymentId: rawDeploymentId(process.env),
    argv: process.argv.slice(1),
  });
  if (!gotLock) {
    console.log(`[lm3-desktop] another window already owns deployment ${DEPLOYMENT_KEY}; exiting`);
    app.quit();
    return;
  }

  app.on("second-instance", (_event, _argv, _cwd, additionalData) => {
    const which = (additionalData && additionalData.deploymentKey) || DEPLOYMENT_KEY;
    console.log(`[lm3-desktop] second launch for deployment ${which}: focusing the existing window`);
    if (!win || win.isDestroyed()) return;
    if (win.isMinimized()) win.restore();
    win.show();
    win.focus();
  });

  // ----------------------------------------------------------------------------------------- //
  // HTTP
  // ----------------------------------------------------------------------------------------- //
  /** GET /healthz. Resolves the parsed body, or null when nothing usable answered.
   *
   * It records NOTHING that could be used to signal a process. The `pid` this used to scrape is
   * exactly the "PID appeared in an unauthenticated response" that invariant 12 forbids acting on.
   */
  function health(timeoutMs = 1200) {
    return new Promise((resolve) => {
      const req = http.get(`${base()}/healthz`, { timeout: timeoutMs }, (res) => {
        let body = "";
        res.setEncoding("utf8");
        res.on("data", (d) => { if (body.length < 65536) body += d; });
        res.on("end", () => {
          if (res.statusCode !== 200) return resolve(null);
          try {
            const parsed = JSON.parse(body);
            lastHealth = parsed && typeof parsed === "object" ? parsed : null;
            resolve(lastHealth === null ? {} : parsed);
          } catch {
            resolve({});                       // 200 with a body we cannot parse: not an LM3 server
          }
        });
      });
      req.on("error", () => resolve(null));
      req.on("timeout", () => { req.destroy(); resolve(null); });
    });
  }

  const portQuiet = async (timeoutMs = 400) => (await health(timeoutMs)) === null;

  function requestShutdown(timeoutMs = 4000) {
    return new Promise((resolve) => {
      const headers = { "Content-Length": "0" };
      if (token) headers.Authorization = `Bearer ${token}`;
      const req = http.request(
        { host: HOST, port: PORT, path: "/v1/shutdown", method: "POST", timeout: timeoutMs, headers },
        (res) => { res.resume(); resolve(res.statusCode === 200); },
      );
      req.on("error", () => resolve(false));
      req.on("timeout", () => { req.destroy(); resolve(false); });
      req.end();
    });
  }

  async function waitFor(predicate, { tries = 60, delayMs = 500 } = {}) {
    for (let i = 0; i < tries; i += 1) {
      if (await predicate(i)) return true;
      await new Promise((r) => setTimeout(r, delayMs));
    }
    return false;
  }

  // ----------------------------------------------------------------------------------------- //
  // Credentials (section 2.12)
  // ----------------------------------------------------------------------------------------- //
  /** The bearer token for THIS deployment, or null.
   *
   * LM3_SERVER_TOKEN first, because an operator who set it means it. Otherwise the deployment's
   * connection.private.json. There is deliberately no "mint one and hope": a random token is only
   * ever correct for a server this process is about to start, and using one against an existing
   * server produced a window whose every /v1 call 401'd.
   *
   * `hint` is a runtime directory reported by a server that has ALREADY proved its deployment key,
   * used only when the locally computed path holds nothing -- which is how a shell in an unusual
   * runtime-directory configuration still finds the descriptor without this file reimplementing the
   * whole resolver.
   */
  function loadToken({ hint = null } = {}) {
    const explicit = (process.env.LM3_SERVER_TOKEN || "").trim();
    if (explicit) return explicit;
    const candidates = [connectionPrivatePath(process.env)];
    if (hint) candidates.push(path.join(hint, CONNECTION_PRIVATE_FILENAME));
    for (const file of candidates) {
      const descriptor = readConnectionDescriptor(file);
      if (!descriptor) continue;
      if (descriptor.deployment_key && descriptor.deployment_key !== DEPLOYMENT_KEY) continue;
      if (typeof descriptor.token === "string" && descriptor.token) return descriptor.token;
    }
    return null;
  }

  // ----------------------------------------------------------------------------------------- //
  // Shutdown -- owned servers only
  // ----------------------------------------------------------------------------------------- //
  /** Signal the process group of the server WE spawned. Never called without a live handle. */
  function signalOwnServer(sig) {
    if (!serverProc || serverProc.exitCode !== null || !serverProc.pid) return false;
    try {
      process.kill(-serverProc.pid, sig);     // detached: the child leads its own group
      return true;
    } catch (err) {
      if (err.code !== "ESRCH") return false;
      try { process.kill(serverProc.pid, sig); return true; } catch { return false; }
    }
  }

  /** Stop the server this app started, and only that one. Returns a human-readable outcome.
   *
   * Closing a window NEVER stops an LM3 run (invariant 9) and never stops a server this process did
   * not spawn (invariant 10). Both used to happen here: quitting escalated SIGTERM/SIGKILL at a PID
   * read out of /healthz, which is how closing a window could kill `lm3 serve` in someone's
   * terminal -- and the pipeline it was watching with it.
   */
  async function shutdownServer() {
    const plan = shutdownPlan({
      hasHandle: Boolean(serverProc), keepServer: KEEP_SERVER,
      spawnedInstanceId, health: lastHealth, deploymentKey: DEPLOYMENT_KEY,
    });
    if (!plan.act) return plan.detail;

    // Ask first: the server exits from the inside on its own terms, and a run it launched is a
    // separate session that deliberately survives (section 2.4's start_new_session).
    if (await requestShutdown()) {
      if (await waitFor(portQuiet, { tries: 24, delayMs: 250 })) return "stopped";
    }
    signalOwnServer("SIGTERM");
    if (await waitFor(portQuiet, { tries: 40, delayMs: 250 })) return "terminated";
    // SIGKILL for uvicorn parked in graceful shutdown waiting on the SSE streams this UI holds
    // open. That wait is unbounded, and it is how these servers used to outlive the app.
    signalOwnServer("SIGKILL");
    if (await waitFor(portQuiet, { tries: 24, delayMs: 250 })) return "killed";
    return "STILL RUNNING";
  }

  // ----------------------------------------------------------------------------------------- //
  // Start or attach
  // ----------------------------------------------------------------------------------------- //
  async function ensureServer() {
    const existing = await health();
    if (existing !== null) {
      const state = classifyHealth(existing, { deploymentKey: DEPLOYMENT_KEY });
      if (state !== PORT_STATE.OURS) throw portConflictError(state, { url: base(), body: existing });
      token = loadToken({ hint: existing.paths && existing.paths.runtime_dir });
      if (!token) {
        throw new MissingCredentialsError(
          `an LM3 server for this deployment is running on ${base()}, but its connection `
          + `descriptor could not be read (${connectionPrivatePath(process.env)}). `
          + `Set LM3_SERVER_TOKEN, or restart that server so it republishes one.`);
      }
      return "attached";
    }

    // Nothing is listening, so we start one -- and only now may we mint a token, because the
    // server that will accept it is the child we are about to create.
    if (backendError) throw backendError;
    if (!backend) {
      throw new BackendLaunchError(
        "No installed LM3 backend was found. Install the LM3 wheel so `lm3` is on PATH, activate "
        + "its virtual/Conda environment before launching, set LM3_PYTHON, set LM3_ROOT for a "
        + "checkout, or start `lm3 serve` first and reopen the GUI.");
    }
    token = loadToken() || crypto.randomBytes(24).toString("hex");
    spawnedInstanceId = crypto.randomBytes(16).toString("hex");

    serverProc = spawn(
      backend.command,
      [...backend.argsPrefix, "--host", HOST, "--port", String(PORT)],
      {
        cwd: backend.cwd,
        env: {
          ...process.env,
          LM3_SERVER_TOKEN: token,
          // THE CUTOVER. Electron is the GUI, and the whole point of the unified runtime is that
          // opening the GUI during a CLI run shows that run. A server started without this flag
          // gets the pre-Step-3 blindness -- /v1/run/active reports idle while a pipeline is
          // live -- so a GUI that spawns its own server must ask for the new behavior explicitly.
          // An operator who has deliberately set LM3_RUNTIME_V2 (to 0, to pin the old path while
          // debugging) keeps their value: this only supplies a default.
          LM3_RUNTIME_V2: (process.env.LM3_RUNTIME_V2 || "1"),
          // The expected-instance-ID handshake (section 2.11): the child publishes this on
          // /healthz, and we refuse to adopt anything on the port that answers differently.
          LM3_INSTANCE_ID: spawnedInstanceId,
          // So the child's connection.private.json names the address it is actually on.
          LM3_BIND_HOST: HOST,
          LM3_BIND_PORT: String(PORT),
          // LM3_OWNER_PID arms the server's own watchdog: if this process is SIGKILLed or crashes,
          // no shutdown request is ever sent, and the server exits on noticing we are gone. Under
          // LM3_KEEP_SERVER the variable is withheld, because the preference has to reach BOTH
          // halves -- arming it would have the watchdog stop the very server we promised to leave.
          ...(KEEP_SERVER ? {} : { LM3_OWNER_PID: String(process.pid) }),
        },
        stdio: ["ignore", "pipe", "pipe"],
        // Its own process group, so one signal reaches the server and everything it spawned -- and
        // so a signal aimed at that group can never travel back up into this process.
        detached: true,
      },
    );

    let starting = true;
    let exited = null;                                      // set if it dies before it ever serves
    const tail = [];                                        // last few lines, for the error dialog
    const note = (chunk) => {
      for (const line of String(chunk).split("\n")) if (line.trim()) tail.push(line.trim());
      while (tail.length > 8) tail.shift();
    };
    serverProc.stdout.on("data", (d) => { note(d); process.stdout.write(`[lm3] ${d}`); });
    serverProc.stderr.on("data", (d) => { note(d); process.stderr.write(`[lm3] ${d}`); });
    serverProc.on("exit", (code) => {
      serverProc = null;                                    // the handle is the authority; it is gone
      exited = code;
      if (code && code !== 0 && !app.isQuiting && !starting) {
        dialog.showErrorBox("LeafMachine3 server stopped", `The LM3 server exited with code ${code}.`);
      }
    });

    const tries = Math.ceil((STARTUP_TIMEOUT_S * 1000) / 500);
    const ok = await waitFor(async (i) => {
      if (exited !== null) return true;                     // stop waiting -- diagnosed below
      if (i && i % 20 === 0) console.log(`[lm3-desktop] still waiting for the server (${i / 2}s)…`);
      return (await health()) !== null;
    }, { tries, delayMs: 500 });
    starting = false;

    if (exited !== null && (await health(600)) === null) {
      const why = tail.length ? `\n\n${tail.join("\n")}` : "";
      throw new Error(`the LM3 server exited (code ${exited}) before it began serving.`
        + ` Is something else already on ${base()}?${why}`);
    }
    if (!ok) {
      // Never leave the half-started server behind: it would finish booting seconds later and hold
      // the port. This is OUR child, so signaling it is exactly the authority we do have.
      signalOwnServer("SIGTERM");
      setTimeout(() => signalOwnServer("SIGKILL"), 2000).unref();
      throw new Error(`the LM3 server did not come up on ${base()} within ${STARTUP_TIMEOUT_S}s`);
    }

    // The handshake: whatever is answering the port must be the instance we just created.
    const body = await health(2000);
    const state = classifyHealth(body, {
      deploymentKey: DEPLOYMENT_KEY, expectedInstanceId: spawnedInstanceId,
    });
    if (state !== PORT_STATE.OURS) {
      throw portConflictError(state, { url: base(), body, expectedInstanceId: spawnedInstanceId });
    }
    return `spawned via ${backend.source}`;
  }

  // ----------------------------------------------------------------------------------------- //
  // Window and menu
  // ----------------------------------------------------------------------------------------- //
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

  /** http/https only -- everything else is refused rather than handed to the OS. */
  function isWebUrl(url) {
    try {
      const p = new URL(String(url)).protocol;
      return p === "http:" || p === "https:";
    } catch {
      return false;
    }
  }

  async function createWindow() {
    win = new BrowserWindow({
      width: 1680,
      height: 1020,
      minWidth: 1100,
      minHeight: 700,
      backgroundColor: "#101012",                           // matches the UI so there is no white flash
      title: rawDeploymentId(process.env) === DEFAULT_DEPLOYMENT_ID
        ? "LeafMachine3" : `LeafMachine3 -- ${rawDeploymentId(process.env)}`,
      show: false,
      webPreferences: {
        preload: path.join(__dirname, "preload.js"),
        contextIsolation: true,
        nodeIntegration: false,
      },
    });
    win.once("ready-to-show", () => win.show());
    win.webContents.setWindowOpenHandler(({ url }) => {      // external links go to the real browser
      // Only web URLs are handed to the OS. shell.openExternal() will launch a registered handler
      // for ANY scheme it is given (file:, smb:, ms-msdt:, …), so the scheme is checked before the
      // URL leaves this process.
      if (isWebUrl(url)) shell.openExternal(url);
      return { action: "deny" };
    });
    // Hand the token over in the URL rather than relying on the server embedding it in the HTML.
    // api.js accepts ?token= and stashes it, so the desktop app authenticates even when the server
    // is run with LM3_EMBED_TOKEN=0 (which stops any other local process from simply GETting "/"
    // to read the secret). The token comes from this deployment's descriptor, never from thin air.
    await win.loadURL(`${base()}/?token=${encodeURIComponent(token || "")}`);
  }

  // ----------------------------------------------------------------------------------------- //
  // IPC -- registered AFTER the single-instance lock (section 2.11)
  // ----------------------------------------------------------------------------------------- //
  // Let the renderer hand a produced file (overlay, timing report, STL) to the OS. Only this
  // server's own origin and file paths INSIDE the LM3 checkout are allowed.
  //
  // Both checks are structural, never string prefixes: "http://127.0.0.1:8765@evil.com/" and
  // "http://127.0.0.1:8765.evil.com/" both start with base(), and "/datac/.../LM3_evil/run.sh"
  // starts with ROOT -- so a prefix test would open all three.
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
      if (u.host) return false;                             // file://evil.com/... is not local
      const p = path.resolve(decodeURIComponent(u.pathname));
      // The separator is what makes this a containment test instead of a prefix test.
      if (p === ROOT || p.startsWith(ROOT + path.sep)) return shell.openPath(p);
    }

    return false;
  });

  /** Read-only facts the UI may show: which deployment this window belongs to, and who owns the
   *  server. `serverOwned` is descriptive; the renderer has no control authority of its own. */
  ipcMain.handle("lm3:identity", async () => ({
    deploymentId: rawDeploymentId(process.env),
    deploymentKey: DEPLOYMENT_KEY,
    port: PORT,
    baseUrl: base(),
    serverOwned: Boolean(serverProc),
    serverInstanceId: (lastHealth && lastHealth.instance_id) || null,
  }));

  // The UI's Close button. Closing the window does NOT stop a running LM3 job (invariant 9) and
  // does not stop a server this app merely attached to (invariant 10). Stopping a run is a
  // separate, explicit action that lives in the UI.
  ipcMain.handle("lm3:quit", async (_evt, opts) => {
    const running = !!(opts && opts.running);
    const stopsServer = shutdownPlan({
      hasHandle: Boolean(serverProc), keepServer: KEEP_SERVER,
      spawnedInstanceId, health: lastHealth, deploymentKey: DEPLOYMENT_KEY,
    }).act;
    const { response } = await dialog.showMessageBox(win, {
      type: "question",
      buttons: ["Cancel", "Close LeafMachine3"],
      defaultId: 0,                 // Enter cancels
      cancelId: 0,
      title: "Close LeafMachine3",
      message: "Close LeafMachine3?",
      detail: [
        running
          ? "The LM3 job that is running KEEPS RUNNING. Closing this window does not stop it -- "
            + "reopen the app to watch it again, or press Stop first if you want it to end."
          : "No LM3 job is running.",
        stopsServer
          ? "The LM3 server this app started will stop."
          : "The LM3 server keeps running; this app did not start it.",
      ].join("\n\n"),
    });
    if (response !== 1) return false;
    app.isQuiting = true;
    app.quit();
    return true;
  });

  // ----------------------------------------------------------------------------------------- //
  // Lifecycle
  // ----------------------------------------------------------------------------------------- //
  app.whenReady().then(async () => {
    buildMenu();
    try {
      const how = await ensureServer();
      console.log(`[lm3-desktop] server ${how} at ${base()} (deployment ${DEPLOYMENT_KEY})`);
      await createWindow();
    } catch (err) {
      const name = err && err.name && err.name !== "Error" ? `${err.name}: ` : "";
      dialog.showErrorBox("Cannot start LeafMachine3",
        `${name}${String(err && err.message ? err.message : err)}`);
      app.quit();
    }
    app.on("activate", () => { if (BrowserWindow.getAllWindows().length === 0) createWindow(); });
  });

  // Quitting stops the server we STARTED, and nothing else. This has to run in "before-quit"
  // rather than "quit": "quit" fires synchronously as the process is already on its way out, so
  // anything asynchronous started there is a race that is always lost.
  app.on("before-quit", (event) => {
    app.isQuiting = true;
    event.preventDefault();                                 // the teardown below exits for us
    if (teardown) return;                                   // ...and a second Quit must not pre-empt it
    if (win && !win.isDestroyed()) { try { win.hide(); } catch (_) {} }
    teardown = shutdownServer()
      .then((how) => console.log(`[lm3-desktop] server ${how}`))
      .catch((err) => console.error("[lm3-desktop] server shutdown failed:", err))
      .finally(() => app.exit(0));                          // exit(), not quit(): skips before-quit
  });

  app.on("window-all-closed", () => app.quit());

  // Ctrl-C (or a SIGTERM from whatever launched us) should tear down exactly like Quit does.
  for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(sig, () => app.quit());

  app.on("quit", () => {
    // Last resort for a path that skipped before-quit. It signals ONLY a live child handle -- the
    // blunt SIGKILL that used to fire here reached attached servers too, which is invariant 10's
    // counterexample and the reason closing a window could kill someone's `lm3 serve`.
    if (!teardown && !KEEP_SERVER && serverProc) signalOwnServer("SIGKILL");
  });
}

module.exports = {
  // section 2.1
  asciiSlug, canonicalDeploymentKey, rawDeploymentId, deploymentKeyFor, isDefaultDeployment,
  resolvePort, DeploymentIdentityError, DeploymentPortError,
  DEFAULT_DEPLOYMENT_ID, DEFAULT_PORT, SERVICE_NAME,
  // packaged backend discovery
  isRunnableFile, pythonCandidates, findOnPath, resolveBackendLaunch, BackendLaunchError,
  // section 2.12
  runtimeBaseDir, userCacheDir, deploymentRuntimeDir, connectionPrivatePath,
  readConnectionDescriptor, userDataDirFor, CONNECTION_PRIVATE_FILENAME,
  // section 2.11
  PORT_STATE, classifyHealth, portConflictError, shutdownPlan,
  WrongDeploymentOnPortError, UnrelatedServiceOnPortError, UnidentifiedServerOnPortError,
  WrongInstanceOnPortError, MissingCredentialsError,
  // Exported for app/test only. boot() is where every module-level side effect lives, so a test
  // that wants to observe the ORDER of "lock, then IPC, then contact a server" has to be able to
  // run it against an injected Electron. It is never called from here except under Electron.
  _boot: boot,
};

if (RUNNING_IN_ELECTRON) boot();
