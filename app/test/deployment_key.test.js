/* The Python <-> JavaScript deployment-key agreement (plan section 2.1).
 *
 * This is the one test in the tree whose failure has no other symptom. A key that disagrees between
 * `leafmachine3.core.paths.canonical_deployment_key` and this shell sends the two halves of the
 * desktop app to two different runtime directories and two different single-instance locks: the
 * window would lock against one deployment and talk to the server of another, and every later test
 * would still pass. So the golden vectors are consumed here BYTE FOR BYTE, from the same file
 * `tests/test_paths.py` reads, rather than from a copy.
 */
const test = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const main = require("../main.js");

const VECTORS_FILE = path.join(__dirname, "..", "..", "tests", "golden", "deployment_key_vectors.json");
const GOLDEN = JSON.parse(fs.readFileSync(VECTORS_FILE, "utf8"));

test("the golden vector file is the one Python pins, and is not empty", () => {
  assert.strictEqual(GOLDEN.schema_version, 1);
  assert.ok(GOLDEN.vectors.length >= 20, "the vector set shrank -- that is a coverage loss");
});

for (const vec of GOLDEN.vectors) {
  // `raw: null` means LM3_DEPLOYMENT_ID is unset, which is the raw literal "default". Nothing else.
  const raw = vec.raw === null ? vec.effective_raw : vec.raw;

  test(`ascii_slug agrees with Python for ${vec.name}`, () => {
    assert.strictEqual(main.asciiSlug(raw), vec.slug);
  });

  test(`canonical_deployment_key agrees with Python for ${vec.name}`, () => {
    assert.strictEqual(main.canonicalDeploymentKey(raw), vec.canonical);
  });
}

test("every canonical key in the vector set is distinct", () => {
  // Truncation must never merge two different raw values -- the hash is what guarantees it, and a
  // collision here would be a silent namespace merge rather than an error.
  const keys = GOLDEN.vectors.map((v) => main.canonicalDeploymentKey(v.effective_raw));
  const byKey = new Map();
  for (let i = 0; i < keys.length; i += 1) {
    const seen = byKey.get(keys[i]);
    if (seen !== undefined && GOLDEN.vectors[seen].effective_raw !== GOLDEN.vectors[i].effective_raw) {
      assert.fail(`${GOLDEN.vectors[seen].name} and ${GOLDEN.vectors[i].name} share ${keys[i]}`);
    }
    byKey.set(keys[i], i);
  }
});

test("the slug is never used as a path component on its own", () => {
  // A raw value that slugs to empty still yields a usable, distinct directory name.
  for (const raw of ["..", "///", "실험실", "ß"]) {
    const key = main.canonicalDeploymentKey(raw);
    assert.ok(!key.includes("/") && !key.includes("\\"), key);
    assert.notStrictEqual(key, "");
    assert.notStrictEqual(key, ".");
    assert.notStrictEqual(key, "..");
  }
});

test("unset, explicit 'default', and a named deployment resolve as section 2.1 states", () => {
  assert.strictEqual(main.rawDeploymentId({}), "default");
  assert.strictEqual(main.deploymentKeyFor({}), main.deploymentKeyFor({ LM3_DEPLOYMENT_ID: "default" }));
  assert.ok(main.isDefaultDeployment({}));
  assert.ok(!main.isDefaultDeployment({ LM3_DEPLOYMENT_ID: "gpu1" }));
});

test("an empty or whitespace-only deployment id is refused, never silently 'default'", () => {
  for (const value of ["", "   ", "\t"]) {
    assert.throws(() => main.rawDeploymentId({ LM3_DEPLOYMENT_ID: value }),
      main.DeploymentIdentityError, `LM3_DEPLOYMENT_ID=${JSON.stringify(value)} must be refused`);
  }
});

test("the port rule: 8765 for the default deployment, explicit for any named one", () => {
  assert.strictEqual(main.resolvePort({}), 8765);
  assert.strictEqual(main.resolvePort({ LM3_DEPLOYMENT_ID: "default" }), 8765);
  assert.strictEqual(main.resolvePort({ LM3_DEPLOYMENT_ID: "gpu1", LM3_PORT: "8766" }), 8766);
  // Gate 41: a named deployment with no port is a startup error, not a silent collision on 8765.
  assert.throws(() => main.resolvePort({ LM3_DEPLOYMENT_ID: "gpu1" }), main.DeploymentPortError);
  assert.throws(() => main.resolvePort({ LM3_PORT: "not-a-port" }), main.DeploymentPortError);
  assert.throws(() => main.resolvePort({ LM3_PORT: "70000" }), main.DeploymentPortError);
});
