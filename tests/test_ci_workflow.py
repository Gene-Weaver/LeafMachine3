"""The CI matrix itself is a plan deliverable, so it gets a test.

Section 4 Step 2's exit gate is "the primitives pass in isolation **on all three platforms**", and
section 2.2 (the kill-the-root-while-a-child-lives test) says in as many words: "On Windows this is
a first-class CI job, not an extrapolation from the POSIX result -- the two implementations share
no mechanism." Section 7's "Platform CI" row names Linux, macOS and Windows, and gate 27 requires
the Windows exclusion be "verified against the Windows adapter itself, not inferred from the POSIX
result".

That is a claim about *where the suite executes*, which no Python module can assert about itself --
the only artifact that can carry it is the workflow file. So these tests read
``.github/workflows/ci.yml`` as data and defend four things a future edit could quietly undo:

1. the Linux job is not weakened to buy the new platforms;
2. a windows-latest **and** a macos-latest leg exist and actually run the Step 2 primitives;
3. that leg is scoped to the primitive files (the exit gate says "in isolation"; a whole-suite run
   on a fresh runner drowns the platform signal in the unrelated ``No module named 'ect'`` family)
   and does not carry ``-p no:recording``, which only a developer interpreter with both
   pytest-recording and pytest-vcr installed needs -- ``.[cpu,dev]`` installs neither;
4. the two tests that touch the genuine Win32 object manager are named explicitly on the Windows
   leg *and* the step fails if they SKIP rather than run. pytest exits 0 on a skip, so without that
   guard the job would be green while proving exactly nothing about ``CreateEventW`` /
   ``ERROR_ALREADY_EXISTS`` / ``CompareObjectHandles`` / ``DuplicateHandle`` -- the only semantics
   the Linux ``FakeWin32`` harness cannot establish.

Point 4's mirror image matters too, and is asserted here: the FakeWin32 harness must NOT be gated
behind Windows. Its Linux-executing tests remain the primary coverage of the adapter's *logic*; the
Windows job is purely additive, covering only the real kernel semantics.

NOT A PROOF OF EXECUTION: this checkout has no git remote, so nothing under .github/workflows/ runs
from it. These tests prove the workflow *says* the right thing; only a green run on the upstream
named by pyproject's Homepage discharges Step 2's exit gate or gate 27.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CI_YML = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_WINDOWS_LEASE_TESTS = _REPO_ROOT / "tests" / "test_runtime_lease_windows.py"
_WINDOWS_HANDSHAKE_TESTS = _REPO_ROOT / "tests" / "test_server_handshake.py"

# What the platform job must cover. Step 2's primitives and the path resolver they sit on, PLUS
# Step 3's handshake: gate 17 is "the launch handshake and process-tree Stop pass on Windows CI,
# not only the lock adapter", and the handshake is the piece with a genuinely different Windows
# implementation (STARTUPINFOEX handle allowlist instead of pass_fds, a job object instead of a
# process group). A list of Step 2 files alone leaves gate 17 unmet however green it runs.
#
# "In isolation" still binds: named files only, never a bare `tests/` target, because a whole-suite
# run on a fresh runner drowns the platform signal in optional-dependency noise.
_STEP_2_PRIMITIVE_TESTS = (
    "tests/test_runtime_lease.py",
    "tests/test_runtime_lease_windows.py",
    "tests/test_runtime_records.py",
    "tests/test_runtime_grant.py",
    "tests/test_runtime_config_io.py",
    "tests/test_runtime_isolation.py",
    "tests/test_paths.py",
    "tests/test_setup_paths.py",
    "tests/test_settings_path_unification.py",
)

#: Step 3's platform-sensitive surface. Separate from the tuple above so a reader can see which
#: gate each group answers.
_STEP_3_PLATFORM_TESTS = (
    "tests/test_runtime_launch.py",
    "tests/test_runtime_execution.py",
    "tests/test_server_handshake.py",
)

# The checks that cannot run anywhere but a real Windows kernel. Gates 17/27 live or die on these.
_REAL_WIN32_TESTS = (
    (_WINDOWS_LEASE_TESTS, "test_the_real_allowlist_is_a_startupinfo"),
    (_WINDOWS_LEASE_TESTS, "test_the_real_surface_acquires_and_releases"),
    (_WINDOWS_HANDSHAKE_TESTS, "test_the_real_windows_job_object_stops_root_and_grandchild"),
)


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    """The parsed workflow. Loaded once; every test here is read-only."""
    assert _CI_YML.is_file(), f"{_CI_YML} is the entire CI surface of this repo and must exist"
    data = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and isinstance(data.get("jobs"), dict)
    return data


def _job(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    jobs = workflow["jobs"]
    assert name in jobs, f"ci.yml lost its `{name}` job; jobs are {sorted(jobs)}"
    return jobs[name]


def _steps_running(job: dict[str, Any], needle: str) -> list[dict[str, Any]]:
    """Every step whose shell command mentions ``needle``."""
    return [s for s in job.get("steps", []) if needle in str(s.get("run", ""))]


def _flat(run: str) -> str:
    """Collapse a YAML block scalar's line breaks and continuations into one comparable line."""
    return " ".join(run.replace("\\\n", " ").split())


# --- the Linux job must not be traded away for the new ones -------------------------------------


def _release_python_minor() -> str:
    for line in (_REPO_ROOT / "tools" / "release" / "versions.env").read_text().splitlines():
        if line.startswith("PYTHON_VERSION="):
            return ".".join(line.split("=", 1)[1].strip().split(".")[:2])
    raise AssertionError("tools/release/versions.env has no PYTHON_VERSION")


def test_the_linux_job_still_runs_the_whole_suite_on_the_release_python(workflow: dict[str, Any]) -> None:
    """Adding platforms must be additive. The finding that prompted this was explicit that
    ``test: runs-on: ubuntu-latest`` stays exactly as it was -- the platform legs cover the kernel
    semantics a fake cannot model, not the suite.

    It used to pin a 3.10/3.11/3.12 matrix. requires-python now admits exactly one minor (the
    uv-managed interpreter in tools/release/versions.env), so the matrix is derived from that file:
    a version bump cannot leave CI testing an interpreter the package refuses to install on."""
    job = _job(workflow, "test")
    assert job["runs-on"] == "ubuntu-latest"
    assert job["strategy"]["matrix"]["python-version"] == [_release_python_minor()]
    assert _steps_running(job, "pytest"), "the Linux job no longer runs pytest"


def test_every_ci_python_is_the_release_python(workflow: dict[str, Any]) -> None:
    want = _release_python_minor()
    seen = re.findall(r'python-version:\s*"?\[?"?([0-9.]+)', _CI_YML.read_text(encoding="utf-8"))
    assert seen and all(v == want for v in seen), f"CI uses {seen}; requires-python admits only {want}"


# --- section 7 "Platform CI": Linux, macOS and Windows ------------------------------------------


def _platform_job(workflow: dict[str, Any]) -> dict[str, Any]:
    """The job whose matrix carries the non-Linux runners, found by shape rather than by name so a
    rename does not silently turn this whole module into a no-op."""
    for job in workflow["jobs"].values():
        oses = job.get("strategy", {}).get("matrix", {}).get("os", [])
        if any(str(o).startswith("windows") for o in oses):
            return job
    raise AssertionError(
        "no job in ci.yml has a windows runner in its matrix; section 4 Step 2's exit gate is "
        "'the primitives pass in isolation on all three platforms'"
    )


def test_ci_covers_windows_and_macos_as_well_as_linux(workflow: dict[str, Any]) -> None:
    """Section 7's 'Platform CI' row names all three by name, and section 2.2 upgrades the Windows
    half from a nice-to-have to 'a first-class CI job'."""
    job = _platform_job(workflow)
    oses = job["strategy"]["matrix"]["os"]
    assert "windows-latest" in oses
    assert "macos-latest" in oses
    assert job["runs-on"] == "${{ matrix.os }}"
    # One platform failing must not hide the other's result -- they share no mechanism.
    assert job["strategy"].get("fail-fast") is False


def test_the_platform_job_runs_the_step_2_primitives_and_the_step_3_handshake_in_isolation(
    workflow: dict[str, Any],
) -> None:
    """'In isolation' is the exit gate's own word: named files, never a bare ``tests/`` target,
    because a whole-suite run on a fresh runner drowns the platform signal in optional-dependency
    noise (the ``No module named 'ect'`` family).

    The Step 3 files are here because gate 17 asks for the handshake and process-tree Stop on
    Windows CI specifically, and those have a different Windows implementation from the POSIX one --
    which is the entire reason the plan refuses to let a Linux result stand in for them.
    """
    job = _platform_job(workflow)
    candidates = _steps_running(job, "pytest")
    assert candidates, "the platform job runs no pytest at all"
    primitive_steps = [s for s in candidates if "tests/test_runtime_lease.py" in str(s["run"])]
    assert len(primitive_steps) == 1, "expected exactly one Step 2 primitives step"
    command = _flat(str(primitive_steps[0]["run"]))

    for path in _STEP_2_PRIMITIVE_TESTS:
        assert path in command, f"the platform job does not run {path}"
    for path in _STEP_3_PLATFORM_TESTS:
        assert path in command, (
            f"the platform job does not run {path}; gate 17 asks for the launch handshake on "
            f"Windows CI, not only the lock adapter")
    # No bare directory target: that is what makes the run 'isolated' rather than a whole suite.
    assert not re.search(r"(?<![\w/])tests/?(?:\s|$)", command), (
        "the platform job targets all of tests/; the exit gate asks for the primitives in isolation"
    )


def test_the_platform_job_does_not_carry_the_local_only_recording_flag(
    workflow: dict[str, Any],
) -> None:
    """``-p no:recording`` exists only because pytest-recording and pytest-vcr collide in some
    developer interpreters. ``.[cpu,dev]`` installs neither, and the flag would hard-error on a
    runner that has no such plugin to disable."""
    job = _platform_job(workflow)
    for step in job.get("steps", []):
        assert "no:recording" not in str(step.get("run", ""))


def test_the_platform_job_installs_no_heavy_extras_but_does_install_the_server(
    workflow: dict[str, Any]
) -> None:
    """No GPU or ML extras on a hosted runner -- but the ``server`` extra is NOT optional.

    This asserted ``".[cpu,dev]"`` literally. The intent behind that was "nothing heavy": the runtime
    primitives are pure Python, and pulling torch or onnxruntime-gpu onto a hosted runner makes the
    job slow and flaky for reasons unrelated to the lease adapter. That intent is intact.

    The literal was wrong, though. This job runs ``tests/test_settings_path_unification.py``, which
    imports the server modules, and ``server`` is a separate extra in ``pyproject.toml`` -- so on a
    genuinely clean runner the job failed at COLLECTION for want of FastAPI. ``server`` is
    fastapi + uvicorn + python-multipart + sse-starlette: pure Python, no GPU, no ML stack, so it
    costs the runner nothing that the original intent was protecting against.
    """
    job = _platform_job(workflow)
    installs = _steps_running(job, "pip install")
    assert installs, "the platform job never installs the package"
    flat = " ".join(_flat(str(s["run"])) for s in installs)
    for needed in ("cpu", "dev", "server", "test"):
        assert needed in flat, (
            f"the platform job must install [cpu,dev,server,test]; {needed!r} is missing from: {flat}")
    for heavy in ("gpu", "yolo", "macos"):
        assert f",{heavy}" not in flat and f"[{heavy}" not in flat, (
            f"the platform job must not pull the {heavy!r} extra onto a hosted runner")


# --- gate 27: the real Win32 surface must EXECUTE, not skip -------------------------------------


def test_the_windows_leg_names_the_real_win32_tests(workflow: dict[str, Any]) -> None:
    """A fake cannot establish event semantics or that a Job Object kills a real descendant."""
    job = _platform_job(workflow)
    guards = [
        s
        for s in job.get("steps", [])
        if "windows" in str(s.get("if", "")) and "pytest" in str(s.get("run", ""))
    ]
    assert len(guards) == 1, "expected exactly one windows-only step asserting the real Win32 surface"
    command = _flat(str(guards[0]["run"]))
    for source, name in _REAL_WIN32_TESTS:
        node = f"tests/{source.name}::{name}"
        assert node in command, f"the Windows guard does not run {node}"


def test_the_windows_leg_fails_if_the_real_win32_tests_skip(workflow: dict[str, Any]) -> None:
    """pytest exits 0 on a skip, so naming the node ids is necessary but not sufficient: without an
    explicit check the job stays green while the tests quietly skip and prove nothing. The step
    must therefore inspect the outcome, not just the exit code."""
    job = _platform_job(workflow)
    (guard,) = [
        s
        for s in job.get("steps", [])
        if "windows" in str(s.get("if", "")) and "pytest" in str(s.get("run", ""))
    ]
    run = str(guard["run"])
    assert "skipped" in run, (
        "the windows guard step does not check for skips; a skipif that silently skips would leave "
        "gate 27 unproven with a green tick"
    )
    assert "3 passed" in run, "the windows guard step does not assert all native tests actually ran"
    # The grep guards are only load-bearing under a shell that stops on error and honors pipefail.
    assert guard.get("shell") == "bash"


def test_the_named_real_win32_tests_still_exist_under_those_names(workflow: dict[str, Any]) -> None:
    """Node ids in YAML are unchecked strings: rename a test and the Windows leg would collect
    nothing (pytest exit 4) or, worse, drift out of sync with what the plan thinks is covered."""
    for path, name in _REAL_WIN32_TESTS:
        source = path.read_text(encoding="utf-8")
        assert re.search(rf"^def {re.escape(name)}\(", source, re.MULTILINE), (
            f"{name} no longer exists in {path.name}; ci.yml still names it"
        )


# --- the fake harness stays the primary coverage, ungated ---------------------------------------


def test_the_fake_win32_harness_is_not_gated_behind_windows() -> None:
    """Explicitly out of scope for the platform job: the FakeWin32 tests must keep executing on
    Linux. They are the primary coverage of the adapter's *logic* and satisfy the repo rule that
    Windows paths be unit-testable off-Windows. Only the two real-surface tests may be skipif'd, so
    the count of platform gates in that file is exactly two."""
    source = _WINDOWS_LEASE_TESTS.read_text(encoding="utf-8")
    gates = re.findall(r'^@pytest\.mark\.skipif\(sys\.platform != "win32"', source, re.MULTILINE)
    assert len(gates) == 2, (
        f"expected exactly 2 Windows-only gates in {_WINDOWS_LEASE_TESTS.name}, found {len(gates)}; "
        "the fake-driven tests must keep running on Linux"
    )
    # And no module-level gate that would skip the whole file off-Windows.
    assert "pytestmark" not in source


def test_every_primitive_test_file_named_by_ci_exists(workflow: dict[str, Any]) -> None:
    """A path typo in the workflow would make the platform job collect fewer files than the exit
    gate covers, and pytest would not complain about the ones it never heard of."""
    for path in _STEP_2_PRIMITIVE_TESTS:
        assert (_REPO_ROOT / path).is_file(), f"ci.yml names {path}, which does not exist"
