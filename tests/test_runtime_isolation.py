"""Gate 46: pytest can neither observe nor block a real run, including under xdist.

The suite's runtime isolation lives in ``tests/conftest.py``'s module body (plan section 4 Step 1,
lines 1381-1394). These tests are its enforcement: they check that the isolation is actually in
effect, that the deployment id is unique per xdist worker AND per session, and -- the load-bearing
one -- that the DEFAULT runtime directory, the one a non-isolated LM3 would have used, was neither
created nor written during the session.

The default-location assertions are deliberately built on the environment as it looked *before*
``conftest`` rewrote it (``conftest.real_environment()``), so they compare the developer's real
locations against a snapshot taken before collection. A test that merely re-resolved the *current*
environment would be a tautology: it would assert that the sandbox is the sandbox.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from leafmachine3.core import paths as P
from tests import conftest as C

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _under(path: Path, root: Path) -> bool:
    return path.resolve().is_relative_to(root.resolve())


# --------------------------------------------------------------------------------------------- #
# the environment is isolated
# --------------------------------------------------------------------------------------------- #

def test_the_sandbox_exists_and_is_not_the_checkout_or_the_home_directory():
    sandbox = C.LM3_PYTEST_SANDBOX
    assert sandbox.is_dir()
    assert not _under(sandbox, _REPO_ROOT)
    assert not _under(sandbox, Path.home())


@pytest.mark.parametrize(
    "name",
    ["LM3_RUNTIME_DIR", "LM3_SERVER_JOBS", "XDG_CONFIG_HOME", "XDG_STATE_HOME",
     "XDG_CACHE_HOME", "XDG_RUNTIME_DIR", "LM3_HARDWARE", "LM3_POSTPROCESS_SETTINGS"],
)
def test_every_redirected_variable_points_inside_the_sandbox(name: str):
    value = os.environ.get(name)
    assert value, f"{name} is not set -- the isolation did not run"
    assert _under(Path(value), C.LM3_PYTEST_SANDBOX), f"{name}={value} escapes the sandbox"


@pytest.mark.parametrize(
    "name",
    ["LM3_SETTINGS", "LM3_SETTINGS_PATH", "LM3_HARDWARE_SETTINGS", "LM3_RUNS_ROOTS",
     "LM3_STATUS_ROOTS", "LM3_POSTPROCESS_ROOTS", "LM3_ALLOW_NETWORK_RUNTIME"],
)
def test_every_cleared_variable_is_unset(name: str):
    # Cleared rather than pointed at a file: naming one would move the settings resolver off its
    # "unset" precedence row, which several characterization tests pin. Isolation for these comes
    # from the XDG rewrite, which keeps every remaining fallback inside the sandbox.
    assert name not in os.environ


def test_the_developers_hardware_profile_is_unreachable():
    """The hardware profile is the one config file the tree WRITES, so no variable may name the
    developer's. Its sandbox stand-in is allowed to exist -- setup writing one there is the
    isolation working, not failing."""
    named = Path(os.environ["LM3_HARDWARE"])
    assert named != _REPO_ROOT / "hardware_settings.yaml"
    assert named.parent == C.LM3_PYTEST_SANDBOX
    # The legacy alias stays UNSET: setting it would put every session on the one-release
    # deprecation path and warn once per process, which is itself under test elsewhere.
    assert "LM3_HARDWARE_SETTINGS" not in os.environ


def test_the_resolver_places_this_deployments_runtime_directory_in_the_sandbox():
    resolved = P.deployment_runtime_dir(env=os.environ, check_filesystem=False)
    assert _under(resolved, C.LM3_PYTEST_SANDBOX)
    assert resolved.name == P.deployment_key(os.environ)


def test_the_isolation_actually_changes_the_answer():
    """Not a tautology check on the sandbox: the real environment resolves somewhere else entirely."""
    real = C.real_environment()
    isolated = P.deployment_runtime_dir(env=os.environ, check_filesystem=False)
    try:
        default = P.runtime_base_dir(env=real, create=False, check_filesystem=False)
    except P.PathsError as exc:                     # e.g. a scheduler job with no node-local scratch
        pytest.skip(f"the real environment has no resolvable runtime base: {exc}")
    assert not _under(default, C.LM3_PYTEST_SANDBOX)
    assert not _under(isolated, default)


# --------------------------------------------------------------------------------------------- #
# THE guard: the default runtime directory is never touched
# --------------------------------------------------------------------------------------------- #

def test_the_snapshot_of_default_locations_is_real():
    """Guard the guard: an empty or sandbox-shaped snapshot would make the next test vacuous."""
    snapshot = C.DEFAULT_LOCATIONS
    assert snapshot, (
        "pytest_configure did not snapshot the default locations -- conftest was imported twice "
        "under two module names, or the hook did not run"
    )
    assert "resolver:runtime-base" in snapshot, C.DEFAULT_LOCATION_ERRORS
    for label, (path, _sig) in snapshot.items():
        assert not _under(path, C.LM3_PYTEST_SANDBOX), f"{label} ({path}) is inside the sandbox"


def test_the_default_runtime_directory_is_never_touched():
    """Gate 46. Every LM3-owned default location is byte-for-byte as it was before collection.

    ``exists`` catches creation; the directory's ``st_mtime_ns`` catches an entry being created
    inside one that already existed (a deployment directory under an LM3 the developer really
    uses). Both were captured in ``pytest_configure``, before any test module was imported.
    """
    drifted = []
    for label, (path, before) in C.DEFAULT_LOCATIONS.items():
        after = C._stat_signature(path)
        if after != before:
            drifted.append(f"{label}: {path} was {before}, is now {after}")
    assert not drifted, "a test touched the default LM3 runtime state:\n  " + "\n  ".join(drifted)


def test_the_checkout_configuration_files_were_not_written():
    """The other half of gate 46: no test may rewrite the developer's own settings or profile.

    ``hardware_setup`` writes its profile with an atomic rename, so a write always moves the
    mtime; reading a file does not. An unchanged signature is therefore a real "was not written".
    """
    drifted = []
    for label, (path, before) in C.CHECKOUT_FILES.items():
        after = C._stat_signature(path)
        if after != before:
            drifted.append(f"{label}: {path} was {before}, is now {after}")
    assert not drifted, "a test wrote into the checkout:\n  " + "\n  ".join(drifted)


def test_the_touch_detector_actually_detects_a_touch(tmp_path: Path):
    """Negative control for the two guards above.

    They are only as good as the signature they compare, so prove it moves for both shapes of
    violation: a default directory being created, and a deployment directory appearing inside one
    that already existed.
    """
    target = tmp_path / "lm3"
    absent = C._stat_signature(target)
    assert absent == (False, None)

    target.mkdir()
    created = C._stat_signature(target)
    assert created != absent

    time.sleep(0.02)                       # cheap insurance against a coarse-granularity filesystem
    (target / "default-37a8eec1").mkdir()
    assert C._stat_signature(target) != created


def test_the_default_deployment_is_not_this_sessions_deployment():
    """A real run uses the ``default`` deployment; the suite must never be able to contend with it."""
    assert P.is_default_deployment(os.environ) is False
    real_key = P.deployment_key(C.real_environment())
    assert P.deployment_key(os.environ) != real_key


# --------------------------------------------------------------------------------------------- #
# deployment-id derivation
# --------------------------------------------------------------------------------------------- #

def test_worker_id_is_read_from_the_xdist_variable():
    assert C.lm3_worker_id({"PYTEST_XDIST_WORKER": "gw3"}) == "gw3"
    assert C.lm3_worker_id({}) == "main"
    assert C.lm3_worker_id({"PYTEST_XDIST_WORKER": ""}) == "main"


def test_deployment_ids_differ_per_worker_and_per_session():
    a0 = C.lm3_deployment_id("lm3-pytest-aaaa1111", "gw0")
    a1 = C.lm3_deployment_id("lm3-pytest-aaaa1111", "gw1")
    b0 = C.lm3_deployment_id("lm3-pytest-bbbb2222", "gw0")
    assert a0 != a1, "two workers of one session would share a deployment"
    assert a0 != b0, "two sessions would share a deployment"
    keys = {P.deployment_key({"LM3_DEPLOYMENT_ID": i}) for i in (a0, a1, b0)}
    assert len(keys) == 3, "the canonical keys collapsed -- the runtime directories would collide"


def test_a_derived_deployment_id_survives_canonicalization_intact():
    """The id is already an ASCII slug, so section 2.1's canonical key stays human-readable."""
    ident = C.lm3_deployment_id("lm3-pytest-Aa_09", "gw11")
    assert P.ascii_slug(ident) == ident
    assert P.canonical_deployment_key(ident).startswith(ident[:P.DEPLOYMENT_SLUG_LENGTH])


def test_the_live_deployment_id_is_the_derived_one():
    expected = C.lm3_deployment_id(C.LM3_PYTEST_SANDBOX.name, C.lm3_worker_id())
    assert os.environ["LM3_DEPLOYMENT_ID"] == expected


def test_a_named_deployment_states_its_port():
    """Gate 41: a non-default deployment without LM3_PORT is a startup error, and this one is named."""
    assert P.resolve_port(os.environ) == int(os.environ["LM3_PORT"])


# --------------------------------------------------------------------------------------------- #
# xdist
# --------------------------------------------------------------------------------------------- #

_DUMP_ENV = "LM3_PYTEST_ID_DUMP"


def test_dump_deployment_id_for_the_xdist_probe():
    """Doubles as the payload of the nested ``-n 2`` run below; a plain assertion otherwise."""
    deployment = os.environ["LM3_DEPLOYMENT_ID"]
    assert deployment.startswith("pytest-")
    dump = os.environ.get(_DUMP_ENV)
    if dump:
        record = {"worker": C.lm3_worker_id(), "deployment": deployment,
                  "runtime": os.environ["LM3_RUNTIME_DIR"]}
        (Path(dump) / f"{C.lm3_worker_id()}.json").write_text(json.dumps(record), encoding="utf-8")


def test_xdist_workers_get_distinct_deployment_ids(tmp_path: Path):
    """Runs this file's dump test on two real xdist workers and compares what they resolved.

    ``--dist=each`` is what makes one selected test run on BOTH workers; the default scheduler
    would hand a single test to a single worker and prove nothing.
    """
    pytest.importorskip("xdist", reason="pytest-xdist is not installed")
    dump = tmp_path / "ids"
    dump.mkdir()
    env = dict(os.environ)
    env[_DUMP_ENV] = str(dump)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:recording", "-n", "2", "--dist=each",
         f"{Path(__file__).name}::test_dump_deployment_id_for_the_xdist_probe"],
        cwd=str(_REPO_ROOT / "tests"), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, f"nested xdist run failed:\n{proc.stdout}\n{proc.stderr}"
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(dump.glob("*.json"))]
    assert len(records) == 2, f"expected one record per worker, got {records}"
    assert {r["worker"] for r in records} == {"gw0", "gw1"}
    deployments = {r["deployment"] for r in records}
    assert len(deployments) == 2, f"the workers shared a deployment id: {deployments}"
    # They share ONE session, so they share the sandbox and the session token -- only the worker
    # suffix may differ. That is what keeps a single teardown able to remove the whole tree.
    assert {r["runtime"] for r in records} == {os.environ["LM3_RUNTIME_DIR"]}
    assert all(d.startswith(f"pytest-{C._slugify(C.LM3_PYTEST_SANDBOX.name)}-") for d in deployments)
