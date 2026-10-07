/* Step 5b: identity, attachment and ownership (plan sections 2.11, 2.12; invariants 9, 10, 12).
 *
 * No GUI is launched and no LM3 server is started. `electron` is replaced by the double in
 * ./fake_electron.js, and the thing on the port is a throwaway `http.createServer` bound to
 * 127.0.0.1 on an OS-assigned port -- never 8765, never uvicorn, torn down in the same test. That
 * is what makes "an unrelated service is on this port" testable at all: the case is defined by what
 * answers, so something has to answer.
 */
const test = require("node:test");
const assert = require("node:assert");
const http = require("node:http");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const main = require("../main.js");
const { installFakeElectron, makeFakeElectron } = require("./fake_electron.js");

const APP_DIR = path.join(__dirname, "..");
process.setMaxListeners(0);        // boot() installs SIGINT/SIGTERM/SIGHUP handlers on every call

// --------------------------------------------------------------------------------------------- //
// Helpers
// --------------------------------------------------------------------------------------------- //
/** A stub HTTP service. `healthBody === null` means "answer 404", i.e. not an LM3 server at all. */
function startStub({ healthBody }) {
  const seen = [];
  const server = http.createServer((req, res) => {
    seen.push(`${req.method} ${req.url}`);
    if (req.url === "/healthz" && healthBody !== null) {
      const body = typeof healthBody === "string" ? healthBody : JSON.stringify(healthBody);
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(body);
      return;
    }
    if (req.url === "/healthz") { res.writeHead(404); res.end("not here"); return; }
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end("{}");
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve({
      port: server.address().port, seen, close: () => new Promise((r) => server.close(r)),
    }));
  });
}

function healthBodyFor({ deploymentKey, instanceId = "abcdef0123456789", runtimeDir = "" }) {
  return {
    service: "leafmachine3", status: "ok", version: "3.0.0", protocol_version: 1,
    instance_id: instanceId, deployment_id: "default", deployment_key: deploymentKey,
    ownership_mode: "independent", provider: "CPUExecutionProvider",
    pid: 4242, pid_is_diagnostic_only: true, paths: { runtime_dir: runtimeDir },
  };
}

/** Boot main.js against the fake Electron with a scratch environment. Returns the recorder. */
async function boot(envPatch, fakeOpts = {}) {
  const saved = { ...process.env };
  const fake = makeFakeElectron(fakeOpts);
  const uninstall = installFakeElectron(APP_DIR, fake.electron);
  // Start from a scrubbed environment. This file is also run FROM pytest, whose conftest gives the
  // session its own LM3_DEPLOYMENT_ID and LM3_RUNTIME_DIR -- inheriting those would silently make
  // every "same deployment" assertion here compare two different keys.
  for (const key of Object.keys(process.env)) if (key.startsWith("LM3_")) delete process.env[key];
  Object.assign(process.env, envPatch);
  try {
    main._boot();
    return { fake, restore: () => { uninstall(); for (const k of Object.keys(process.env)) delete process.env[k]; Object.assign(process.env, saved); } };
  } catch (err) {
    uninstall();
    for (const k of Object.keys(process.env)) delete process.env[k];
    Object.assign(process.env, saved);
    throw err;
  }
}

function scratchRuntime(deploymentId, descriptor) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-shell-test-"));
  const key = main.canonicalDeploymentKey(deploymentId);
  const dir = path.join(root, key);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  if (descriptor) {
    fs.writeFileSync(path.join(dir, main.CONNECTION_PRIVATE_FILENAME),
      JSON.stringify(descriptor), { mode: 0o600 });
  }
  return { root, dir, key };
}

const settle = () => new Promise((r) => setTimeout(r, 60));

function executable(file) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, "#!/bin/sh\nexit 0\n", { mode: 0o755 });
  fs.chmodSync(file, 0o755);
  return file;
}

// --------------------------------------------------------------------------------------------- //
// Packaged backend discovery -- no child process is ever used as a probe
// --------------------------------------------------------------------------------------------- //
test("an explicit LM3_PYTHON wins and a bad explicit path fails instead of falling through", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-backend-root-"));
  const python = executable(path.join(root, "chosen-python"));
  const spec = main.resolveBackendLaunch({
    env: { LM3_PYTHON: python, PATH: "" }, root, packaged: true,
  });
  assert.strictEqual(spec.command, python);
  assert.strictEqual(spec.source, "LM3_PYTHON");
  assert.deepStrictEqual(spec.argsPrefix,
    ["-m", "uvicorn", "leafmachine3.server.app:create_app", "--factory"]);

  assert.throws(() => main.resolveBackendLaunch({
    env: { LM3_PYTHON: path.join(root, "missing"), PATH: path.dirname(python) },
    root, packaged: true,
  }), main.BackendLaunchError);
});

test("an activated virtual environment is usable by a packaged app", () => {
  const prefix = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-active-venv-"));
  const python = executable(path.join(prefix, "bin", "python3"));
  const spec = main.resolveBackendLaunch({
    env: { VIRTUAL_ENV: prefix, PATH: "" }, root: "/not/a/checkout", packaged: true,
    platform: "linux",
  });
  assert.strictEqual(spec.command, python);
  assert.strictEqual(spec.source, "VIRTUAL_ENV");
});

test("a checkout launch finds the environment created by uv sync", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-uv-checkout-"));
  const python = executable(path.join(root, ".venv", "bin", "python3"));
  const spec = main.resolveBackendLaunch({ env: { PATH: "" }, root, platform: "linux" });
  assert.strictEqual(spec.command, python);
  assert.strictEqual(spec.cwd, root);
});

test("a packaged Windows shell can use an explicit checkout's uv environment", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-win-uv-checkout-"));
  const python = executable(path.join(root, ".venv", "Scripts", "python.exe"));
  const spec = main.resolveBackendLaunch({
    env: { LM3_ROOT: root, PATH: "" }, root, packaged: true, platform: "win32",
  });
  assert.strictEqual(spec.command, python);
  assert.strictEqual(spec.cwd, root);
});

test("a packaged app finds the installed lm3 console command on PATH without probing it", () => {
  const bin = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-console-bin-"));
  const command = executable(path.join(bin, "lm3"));
  const spec = main.resolveBackendLaunch({
    env: { PATH: bin }, root: "/packaged/resources", packaged: true, platform: "linux",
  });
  assert.strictEqual(spec.command, command);
  assert.strictEqual(spec.source, "PATH");
  assert.deepStrictEqual(spec.argsPrefix, ["serve"]);
});

test("Windows discovery uses lm3.exe and never requires a shell", () => {
  const bin = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-win-console-"));
  const command = executable(path.join(bin, "lm3.exe"));
  const spec = main.resolveBackendLaunch({
    env: { PATH: `${bin};C:\\Windows\\System32` }, root: "C:\\packaged\\resources",
    packaged: true, platform: "win32",
  });
  assert.strictEqual(spec.command, command);
  assert.strictEqual(spec.source, "PATH");
  assert.deepStrictEqual(spec.argsPrefix, ["serve"]);
  assert.ok(!Object.prototype.hasOwnProperty.call(spec, "shell"));
});

test("packaged discovery returns null promptly when no supported backend is installed", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "lm3-no-backend-"));
  const started = Date.now();
  const spec = main.resolveBackendLaunch({
    env: { PATH: "" }, root, packaged: true, platform: "linux",
  });
  assert.strictEqual(spec, null);
  assert.ok(Date.now() - started < 100, "resolution is filesystem-only, not a hanging probe child");
});

// --------------------------------------------------------------------------------------------- //
// classifyHealth -- the two distinct named errors, plus the instance handshake
// --------------------------------------------------------------------------------------------- //
test("a matching LM3 server for this deployment is the only attachable state", () => {
  const key = main.canonicalDeploymentKey("default");
  assert.strictEqual(
    main.classifyHealth(healthBodyFor({ deploymentKey: key }), { deploymentKey: key }),
    main.PORT_STATE.OURS);
});

test("a valid LM3 server for a DIFFERENT deployment is its own named error", () => {
  const mine = main.canonicalDeploymentKey("gpu0");
  const theirs = main.canonicalDeploymentKey("gpu1");
  const body = healthBodyFor({ deploymentKey: theirs });
  const state = main.classifyHealth(body, { deploymentKey: mine });
  assert.strictEqual(state, main.PORT_STATE.WRONG_DEPLOYMENT);

  const err = main.portConflictError(state, { url: "http://127.0.0.1:8766", body });
  assert.ok(err instanceof main.WrongDeploymentOnPortError);
  assert.strictEqual(err.name, "WrongDeploymentOnPortError");
  assert.ok(err.message.includes(theirs), err.message);
  // The two refusals must not be the same class, or a caller cannot tell them apart.
  assert.ok(!(err instanceof main.UnrelatedServiceOnPortError));
});

test("an unrelated service on the port is a DIFFERENT named error", () => {
  const mine = main.canonicalDeploymentKey("default");
  for (const body of [{}, { service: "prometheus" }, [], null, "nope", { status: "ok" }]) {
    assert.strictEqual(main.classifyHealth(body, { deploymentKey: mine }),
      main.PORT_STATE.UNRELATED, JSON.stringify(body));
  }
  const err = main.portConflictError(main.PORT_STATE.UNRELATED, { url: "http://127.0.0.1:8765" });
  assert.ok(err instanceof main.UnrelatedServiceOnPortError);
  assert.strictEqual(err.name, "UnrelatedServiceOnPortError");
  assert.ok(!(err instanceof main.WrongDeploymentOnPortError));
});

test("an LM3 server that cannot name its deployment is refused, not assumed to be ours", () => {
  const mine = main.canonicalDeploymentKey("default");
  const body = { service: "leafmachine3", status: "ok" };
  assert.strictEqual(main.classifyHealth(body, { deploymentKey: mine }), main.PORT_STATE.UNIDENTIFIED);
  assert.ok(main.portConflictError(main.PORT_STATE.UNIDENTIFIED, {}) instanceof
    main.UnidentifiedServerOnPortError);
});

test("the expected-instance-ID handshake refuses a server we did not spawn", () => {
  const key = main.canonicalDeploymentKey("default");
  const body = healthBodyFor({ deploymentKey: key, instanceId: "someone-elses" });
  assert.strictEqual(
    main.classifyHealth(body, { deploymentKey: key, expectedInstanceId: "ours-0000" }),
    main.PORT_STATE.WRONG_INSTANCE);
  // ...and imposes nothing when we did not spawn anything, which is the attach case.
  assert.strictEqual(main.classifyHealth(body, { deploymentKey: key }), main.PORT_STATE.OURS);
});

// --------------------------------------------------------------------------------------------- //
// shutdownPlan -- invariants 9 and 10
// --------------------------------------------------------------------------------------------- //
test("an attached server is never stopped, whatever else is true", () => {
  const key = main.canonicalDeploymentKey("default");
  const plan = main.shutdownPlan({ hasHandle: false, deploymentKey: key,
                                   health: healthBodyFor({ deploymentKey: key }) });
  assert.strictEqual(plan.act, false);
  assert.strictEqual(plan.reason, "attached");
});

test("a server this app spawned is stopped; LM3_KEEP_SERVER leaves it running", () => {
  assert.strictEqual(main.shutdownPlan({ hasHandle: true }).act, true);
  const kept = main.shutdownPlan({ hasHandle: true, keepServer: true });
  assert.strictEqual(kept.act, false);
  assert.strictEqual(kept.reason, "keep-server");
});

test("a handle whose server now answers for another deployment or instance is left alone", () => {
  const mine = main.canonicalDeploymentKey("gpu0");
  const drifted = main.shutdownPlan({
    hasHandle: true, deploymentKey: mine,
    health: healthBodyFor({ deploymentKey: main.canonicalDeploymentKey("gpu1") }),
  });
  assert.strictEqual(drifted.act, false);
  assert.strictEqual(drifted.reason, "deployment-mismatch");

  const wrongInstance = main.shutdownPlan({
    hasHandle: true, deploymentKey: mine, spawnedInstanceId: "ours",
    health: healthBodyFor({ deploymentKey: mine, instanceId: "theirs" }),
  });
  assert.strictEqual(wrongInstance.act, false);
  assert.strictEqual(wrongInstance.reason, "instance-mismatch");
});

// --------------------------------------------------------------------------------------------- //
// Single-instance identity
// --------------------------------------------------------------------------------------------- //
test("two deployments get two distinct user-data / single-instance identities", () => {
  const a = main.userDataDirFor("/appdata", { LM3_DEPLOYMENT_ID: "gpu0" });
  const b = main.userDataDirFor("/appdata", { LM3_DEPLOYMENT_ID: "gpu1" });
  assert.notStrictEqual(a, b);
  // Same deployment, spelled two ways that mean the same thing, is ONE identity.
  assert.strictEqual(main.userDataDirFor("/appdata", {}),
                     main.userDataDirFor("/appdata", { LM3_DEPLOYMENT_ID: "default" }));
  // The raw id never reaches the path; the canonical key does.
  const nasty = main.userDataDirFor("/appdata", { LM3_DEPLOYMENT_ID: "../../etc" });
  assert.ok(!nasty.includes(".."), nasty);
  assert.ok(nasty.endsWith(main.canonicalDeploymentKey("../../etc")));
});

test("the lock is requested before IPC registration and before whenReady", async () => {
  const scratch = scratchRuntime("default", null);
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: "8765" });
  try {
    const order = booted.fake.rec.order;
    assert.strictEqual(order[0], "requestSingleInstanceLock",
      `the lock must come first, got ${JSON.stringify(order)}`);
    const firstIpc = order.findIndex((o) => o.startsWith("ipc:"));
    const ready = order.indexOf("whenReady");
    assert.ok(firstIpc > 0, "IPC handlers were never registered");
    assert.ok(ready > firstIpc, "whenReady must come after IPC registration");
    // userData is scoped to the deployment BEFORE the lock, because the lock lives under it.
    assert.deepStrictEqual(booted.fake.rec.setPath[0][0], "userData");
    assert.ok(booted.fake.rec.setPath[0][1].endsWith(main.canonicalDeploymentKey("default")));
    // additionalData names the deployment; it does not create a differently scoped lock.
    assert.strictEqual(booted.fake.rec.lockData.deploymentKey, main.canonicalDeploymentKey("default"));
  } finally {
    booted.restore();
  }
});

test("a second launch for the same deployment quits without a window, IPC, or a server", async () => {
  const scratch = scratchRuntime("default", null);
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: "8765" }, { lock: false });
  try {
    const rec = booted.fake.rec;
    assert.strictEqual(rec.lockCalls, 1);
    assert.strictEqual(rec.quitCalls, 1, "a losing second instance must quit");
    assert.deepStrictEqual(Object.keys(rec.handlers), [], "no IPC handler may be registered");
    assert.deepStrictEqual(rec.windows, [], "no second window");
    assert.ok(!rec.order.includes("whenReady"), "it must not wait for ready and start a server");
  } finally {
    booted.restore();
  }
});

test("a second launch focuses the first instance's window", async () => {
  const key = main.canonicalDeploymentKey("default");
  const stub = await startStub({ healthBody: healthBodyFor({ deploymentKey: key }) });
  const scratch = scratchRuntime("default", { deployment_key: key, token: "tok-1", instance_id: "i1" });
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "" });
  try {
    await booted.fake.ready();
    await settle();
    const win = booted.fake.rec.windows[0];
    assert.ok(win, `a window should exist; errors=${JSON.stringify(booted.fake.rec.errorBoxes)}`);
    win.minimized = true;
    win.focused = false;
    const handler = booted.fake.rec.listeners["second-instance"][0];
    handler({}, [], "/", { deploymentKey: key });
    assert.strictEqual(win.minimized, false, "a minimized window is restored");
    assert.strictEqual(win.focused, true, "the existing window is focused");
    assert.strictEqual(booted.fake.rec.windows.length, 1, "no second window is created");
  } finally {
    booted.restore();
    await stub.close();
  }
});

// --------------------------------------------------------------------------------------------- //
// Credentials and attachment (section 2.12)
// --------------------------------------------------------------------------------------------- //
test("attaching reads the token from connection.private.json, never invents one", async () => {
  const key = main.canonicalDeploymentKey("default");
  const stub = await startStub({ healthBody: healthBodyFor({ deploymentKey: key }) });
  const scratch = scratchRuntime("default", {
    schema_version: 1, service: "leafmachine3", deployment_key: key,
    instance_id: "i-attach", token: "the-real-token", host: "127.0.0.1", port: stub.port,
  });
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "" });
  try {
    await booted.fake.ready();
    await settle();
    assert.deepStrictEqual(booted.fake.rec.errorBoxes, []);
    const win = booted.fake.rec.windows[0];
    assert.ok(win.loadedURL.includes("token=the-real-token"), win.loadedURL);
    assert.ok(stub.seen.includes("GET /healthz"));
  } finally {
    booted.restore();
    await stub.close();
  }
});

test("a descriptor for another deployment is ignored rather than used", async () => {
  const mine = main.canonicalDeploymentKey("default");
  const stub = await startStub({ healthBody: healthBodyFor({ deploymentKey: mine }) });
  // The file sits at OUR path but claims a different deployment: refuse rather than send its token.
  const scratch = scratchRuntime("default", {
    deployment_key: main.canonicalDeploymentKey("gpu1"), token: "not-ours",
  });
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "" });
  try {
    await booted.fake.ready();
    await settle();
    assert.strictEqual(booted.fake.rec.windows.length, 0, "no window without credentials");
    const [[title, body]] = booted.fake.rec.errorBoxes;
    assert.strictEqual(title, "Cannot start LeafMachine3");
    assert.ok(body.startsWith("MissingCredentialsError:"), body);
    assert.ok(!body.includes("not-ours"), "the foreign token must never be echoed");
  } finally {
    booted.restore();
    await stub.close();
  }
});

test("a wrong-deployment server on the port is refused by name, and not attached to", async () => {
  const theirs = main.canonicalDeploymentKey("gpu1");
  const stub = await startStub({ healthBody: healthBodyFor({ deploymentKey: theirs }) });
  const scratch = scratchRuntime("default", { deployment_key: main.canonicalDeploymentKey("default"),
                                              token: "mine" });
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "" });
  try {
    await booted.fake.ready();
    await settle();
    assert.strictEqual(booted.fake.rec.windows.length, 0);
    const [[, body]] = booted.fake.rec.errorBoxes;
    assert.ok(body.startsWith("WrongDeploymentOnPortError:"), body);
    assert.ok(body.includes(theirs), body);
    // Refused means refused: no authenticated request was ever made.
    assert.deepStrictEqual(stub.seen.filter((s) => !s.endsWith("/healthz")), []);
  } finally {
    booted.restore();
    await stub.close();
  }
});

test("an unrelated service on the port is refused by its own name", async () => {
  const stub = await startStub({ healthBody: null });      // 404 on /healthz: not an LM3 server
  const scratch = scratchRuntime("default", null);
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "", LM3_PYTHON: "/nonexistent/python" });
  try {
    await booted.fake.ready();
    await settle();
    // Nothing answered /healthz with 200, so the shell treats the port as free and tries to SPAWN;
    // the interpreter preflight is what stops it. The "unrelated service" refusal below is the
    // case where something DOES answer 200 with a body that is not ours.
    const [[, body]] = booted.fake.rec.errorBoxes;
    assert.ok(body.includes("/nonexistent/python"), body);
  } finally {
    booted.restore();
    await stub.close();
  }

  const chatty = await startStub({ healthBody: { service: "grafana", status: "ok" } });
  const scratch2 = scratchRuntime("default", null);
  const booted2 = await boot({ LM3_RUNTIME_DIR: scratch2.root, LM3_PORT: String(chatty.port),
                               LM3_SERVER_TOKEN: "" });
  try {
    await booted2.fake.ready();
    await settle();
    const [[, body]] = booted2.fake.rec.errorBoxes;
    assert.ok(body.startsWith("UnrelatedServiceOnPortError:"), body);
    assert.strictEqual(booted2.fake.rec.windows.length, 0);
  } finally {
    booted2.restore();
    await chatty.close();
  }
});

// --------------------------------------------------------------------------------------------- //
// Close (invariants 9 and 10)
// --------------------------------------------------------------------------------------------- //
test("Close never stops the run and never stops a server this app attached to", async () => {
  const key = main.canonicalDeploymentKey("default");
  const stub = await startStub({ healthBody: healthBodyFor({ deploymentKey: key }) });
  const scratch = scratchRuntime("default", { deployment_key: key, token: "t", instance_id: "i" });
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "" });
  try {
    await booted.fake.ready();
    await settle();
    assert.strictEqual(booted.fake.rec.windows.length, 1, "attached, with a window");

    const before = stub.seen.length;
    const quit = booted.fake.rec.handlers["lm3:quit"];
    assert.strictEqual(await quit({}, { running: true }), true);
    assert.strictEqual(booted.fake.rec.quitCalls, 1);

    // ...and the teardown that quitting triggers.
    const beforeQuit = booted.fake.rec.listeners["before-quit"][0];
    beforeQuit({ preventDefault: () => {} });
    await settle();

    const afterwards = stub.seen.slice(before);
    assert.deepStrictEqual(afterwards.filter((s) => s.includes("/v1/run/stop")), [],
      "closing the window must never stop a run (invariant 9)");
    assert.deepStrictEqual(afterwards.filter((s) => s.includes("/v1/shutdown")), [],
      "closing the window must never stop a server we attached to (invariant 10)");
    assert.deepStrictEqual(booted.fake.rec.exitCodes, [0], "the app itself still exits");
  } finally {
    booted.restore();
    await stub.close();
  }
});

test("the Close dialog says the run keeps running and who owns the server", async () => {
  const key = main.canonicalDeploymentKey("default");
  const stub = await startStub({ healthBody: healthBodyFor({ deploymentKey: key }) });
  const scratch = scratchRuntime("default", { deployment_key: key, token: "t" });
  let detail = null;
  const fakeOpts = { messageBoxResponse: 0 };               // Cancel: nothing must happen
  const booted = await boot({ LM3_RUNTIME_DIR: scratch.root, LM3_PORT: String(stub.port),
                              LM3_SERVER_TOKEN: "" }, fakeOpts);
  try {
    booted.fake.electron.dialog.showMessageBox = async (_win, opts) => {
      detail = opts.detail; return { response: 0 };
    };
    await booted.fake.ready();
    await settle();
    const quit = booted.fake.rec.handlers["lm3:quit"];
    assert.strictEqual(await quit({}, { running: true }), false, "Cancel must not quit");
    assert.strictEqual(booted.fake.rec.quitCalls, 0);
    assert.ok(/KEEPS RUNNING/.test(detail), detail);
    assert.ok(/did not start it/.test(detail), detail);
  } finally {
    booted.restore();
    await stub.close();
  }
});
