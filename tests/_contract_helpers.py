"""Shared helpers for the HTTP endpoint-contract tests (``tests/test_api_contract.py``).

Deliberately NOT a conftest: the unified-runtime plan's §4 Step 1 gives ``tests/conftest.py`` a
different job (process-wide runtime isolation set at import time), and these helpers must not
compete with it. The leading underscore keeps pytest from collecting this module.

Nothing here touches the network, a GPU, a model, or the user's real runtime state. Every seam a
server route reaches for -- the CWD, ``LM3_SETTINGS``/``LM3_SETTINGS_PATH``, the jobs root, the
hardware profile, the run-root scanners -- is redirected into a pytest ``tmp_path`` by
:func:`isolate_server_paths`, and the one route that would fork a real pipeline is stubbed at the
``subprocess`` seam by :func:`install_fake_spawn`.
"""
from __future__ import annotations

import contextlib
import logging

import os
import threading
from pathlib import Path
from typing import Any, Iterable, Iterator

import pytest

# --------------------------------------------------------------------------- #
# JSON shape assertions
# --------------------------------------------------------------------------- #
# A contract test that checks two keys cannot catch a Step 4 regression, so every response is
# pinned as an EXACT key set plus a per-key type. The aliases below spell out the nullable cases,
# because ``Optional[...]`` is what the current handlers actually emit and a refactor that turns a
# null into a missing key is precisely the breakage worth failing on.
NONE = type(None)
BOOL = (bool,)
INT = (int,)
NUM = (int, float)
STR = (str,)
LIST = (list,)
DICT = (dict,)
OPT_STR = (str, NONE)
OPT_INT = (int, NONE)
OPT_NUM = (int, float, NONE)
OPT_BOOL = (bool, NONE)
OPT_DICT = (dict, NONE)


def assert_json_shape(payload: Any, spec: dict[str, tuple], *, where: str) -> None:
    """Assert ``payload`` is a dict whose key set is EXACTLY ``spec`` and whose values match it.

    ``where`` names the endpoint so a failure reads as a contract breach rather than a KeyError.
    """
    assert isinstance(payload, dict), f"{where}: expected a JSON object, got {type(payload).__name__}"
    missing = sorted(set(spec) - set(payload))
    extra = sorted(set(payload) - set(spec))
    assert not missing and not extra, (
        f"{where}: response key set drifted -- missing={missing} unexpected={extra}"
    )
    for key, allowed in spec.items():
        value = payload[key]
        # bool is a subclass of int, so an int-typed field must not silently accept True.
        if bool not in allowed and isinstance(value, bool):
            raise AssertionError(f"{where}: {key!r} is a bool but the contract says {allowed}")
        assert isinstance(value, allowed), (
            f"{where}: {key!r} is {type(value).__name__}, contract says "
            f"{tuple(t.__name__ for t in allowed)}"
        )


def assert_every_item_shape(rows: Iterable[Any], spec: dict[str, tuple], *, where: str) -> None:
    for i, row in enumerate(rows):
        assert_json_shape(row, spec, where=f"{where}[{i}]")


def flattened_app_routes(app: Any) -> list[Any]:
    """Return concrete routes across FastAPI's eager and lazy router representations.

    FastAPI before 0.141 copied included APIRouter routes directly into ``app.routes``. Newer
    releases retain an ``_IncludedRouter`` wrapper and expose the source router through
    ``original_router``. The application behaves the same in both cases, but tests that audit
    duplicate registrations must inspect the concrete source routes instead of mistaking a lazy
    wrapper for an absent endpoint.
    """
    concrete: list[Any] = []
    pending = list(getattr(app, "routes", ()))
    while pending:
        route = pending.pop(0)
        original = getattr(route, "original_router", None)
        if original is not None:
            pending[0:0] = list(getattr(original, "routes", ()))
            continue
        concrete.append(route)
    return concrete


# --------------------------------------------------------------------------- #
# Response contracts (one place, so the "preserved" and "changes" tests agree)
# --------------------------------------------------------------------------- #
#: ``GET /v1/run/active`` / the body of ``POST /v1/run/start`` and ``/v1/run/stop``.
#: ``metrics_api._Run.record()`` and ``metrics_api._IDLE_RECORD`` must stay key-for-key identical --
#: the module docstring promises "the key set never changes", and the UI relies on it.
RUN_RECORD_SPEC: dict[str, tuple] = {
    "active": BOOL,
    "state": STR,
    "run_name": OPT_STR,
    "pid": OPT_INT,
    "pgid": OPT_INT,
    "started_at": OPT_NUM,
    "started_iso": OPT_STR,
    "finished_at": OPT_NUM,
    "finished_iso": OPT_STR,
    "elapsed_s": OPT_NUM,
    "returncode": OPT_INT,
    "stopped_by_user": BOOL,
    "adopted": BOOL,
    "error": OPT_STR,
    "config_path": OPT_STR,
    "cwd": OPT_STR,
    "output_dir": OPT_STR,
    "run_dir": OPT_STR,
    "db_path": OPT_STR,
    "log_path": OPT_STR,
    "console_log": OPT_STR,
    "tmp_dir": OPT_STR,
    "input_dirs": LIST,
    "restart": LIST,
    "argv": LIST,
}

#: ``GET /v1/settings`` -- ``settings_api.read_settings``.
SETTINGS_SPEC: dict[str, tuple] = {
    "yaml_path": STR,
    "dir": STR,
    "exists": BOOL,
    "mtime": OPT_NUM,
    "size": OPT_INT,
    "values": DICT,
    "defaults": DICT,
    "effective": DICT,
    "text": OPT_STR,
    "text_truncated": BOOL,
    "readonly": BOOL,
    "error": OPT_STR,
    "backups": LIST,
}

#: ``GET /v1/status`` -- ``progress_api._empty_status()`` and ``_build_status()`` must agree, so
#: the front end never has to test for a key's existence (progress_api._empty_status docstring).
STATUS_SPEC: dict[str, tuple] = {
    "ready": BOOL,
    "t": NUM,
    # RunRef.as_dict()
    "run_name": OPT_STR,
    "run_path": OPT_STR,
    "db_path": OPT_STR,
    "log_path": OPT_STR,
    "job_id": OPT_STR,
    "source": OPT_STR,
    "state": STR,
    "stale": BOOL,
    "stale_for_s": OPT_NUM,
    "started_at": OPT_STR,
    "started_ts": OPT_NUM,
    "finished_at": OPT_STR,
    "elapsed_s": OPT_NUM,
    "eta_s": OPT_NUM,
    "calibration": OPT_NUM,
    "images_total": INT,
    "images_done": INT,
    "modules": LIST,
    "active": OPT_DICT,
    "next": OPT_DICT,
    "workers": LIST,
    "workers_meta": DICT,
    "totals": DICT,
    "recent": LIST,
    "provenance": DICT,
}

#: ``GET /v1/runs`` as served by ``results_api`` (the router that currently wins -- see the
#: duplicate-route tests).
RUNS_ENVELOPE_SPEC: dict[str, tuple] = {
    "t": NUM,
    "n": INT,
    "roots": LIST,
    "runs": LIST,
}

#: One entry of ``GET /v1/runs`` -> ``runs[]`` -- ``results_api.Run.summary()``: its own fields
#: plus the ``_run_state`` roll-up merged in.
RUN_SUMMARY_SPEC: dict[str, tuple] = {
    "id": STR,
    "name": STR,
    "path": STR,
    "root": STR,
    "rel": STR,
    "job_id": OPT_STR,
    "db": OPT_STR,
    "db_path": OPT_STR,
    "has_db": BOOL,
    "has_reports": BOOL,
    "mtime": OPT_NUM,
    "indexed": BOOL,
    "n_files": OPT_INT,
    "bytes": OPT_INT,
    # _run_state()
    "state": STR,
    "modules_total": OPT_INT,
    "modules_done": OPT_INT,
    "modules_skipped": OPT_INT,
    "modules_error": OPT_INT,
    "n_images": OPT_INT,
    "n_images_done": OPT_INT,
    "started_at": OPT_STR,
    "finished_at": OPT_STR,
}

#: ``GET /healthz`` -- unauthenticated by design (it is the Electron shell's readiness probe).
#: ``paths`` is the plan section 4, Step 1 diagnostics block ("Expose resolved paths in startup logs
#: and ``/healthz`` diagnostics"): it is what makes the Step 1 exit gate -- every subsystem agreeing
#: on one canonical settings path from any CWD -- checkable from outside the process. It is
#: non-secret (resolved paths and deployment identity only, never a token), so an unauthenticated
#: probe may carry it; do NOT "fix" this spec by deleting the key. ``DICT``, not ``OPT_DICT``:
#: ``app.path_diagnostics()`` returns a dict on every branch, including its ``PathsError`` one.
HEALTHZ_SPEC: dict[str, tuple] = {
    "status": STR,
    "version": STR,
    "provider": STR,
    "pid": INT,
    "paths": DICT,
}


# --------------------------------------------------------------------------- #
# Path isolation
# --------------------------------------------------------------------------- #
#: A minimal but VALID config: ``Config.validate()`` checks structural coherence only (no artifact
#: existence), so ``compute.mock`` plus one input dir is enough for ``POST /v1/run/start`` to get
#: past validation without a model, a GPU, or an image on disk.
_MINIMAL_SETTINGS = """\
version: 3
project:
  run_name: {run_name}
  input:
    dirs:
      - {input_dir}
    recursive: true
  output:
    dir: {output_dir}
    tmp_dir: {tmp_dir}
  run_mode:
    overwrite: false
    restart: []
  logging:
    level: WARNING
    to_file: false
    to_console: false
compute:
  devices: cpu
  mock: true
  precision: fp32
"""

#: Every environment variable that steers a server path today. They are cleared wholesale and then
#: re-pointed at the sandbox, so a variable set in the developer's shell cannot leak a real
#: directory into a contract assertion. The SIX-way settings/hardware split (§3.1, Appendix A) is
#: exactly why this list has to name both spellings of each.
_PATH_ENV = (
    "LM3_SETTINGS",             # metrics_api.default_config_path
    "LM3_SETTINGS_PATH",        # settings_api.settings_path
    "LM3_SETTINGS_META",
    "LM3_HARDWARE",             # progress_api._hardware
    "LM3_HARDWARE_SETTINGS",    # metrics_api.hardware_path
    "LM3_SERVER_JOBS",
    "LM3_RUNS_ROOTS",
    "LM3_STATUS_ROOTS",
    "LM3_MACHINE3_BIN",
    "LM3_SERVER_TOKEN",
    "LM3_OWNER_PID",
    "LM3_EMBED_TOKEN",
)

TEST_TOKEN = "contract-test-token"


class Sandbox:
    """The tmp-dir view of the world a contract test runs against."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.cwd = root / "cwd"
        self.settings_path = self.cwd / "LM3_settings.yaml"
        self.hardware_path = self.cwd / "hardware_settings.yaml"
        self.jobs_root = root / "server_jobs"
        self.output_dir = root / "out"
        self.input_dir = root / "in"
        self.tmp_dir = root / "scratch"
        self.run_name = "contract_run"

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def write_settings(self, *, run_name: str | None = None) -> Path:
        self.run_name = run_name or self.run_name
        self.settings_path.write_text(
            _MINIMAL_SETTINGS.format(
                run_name=self.run_name,
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                tmp_dir=self.tmp_dir,
            ),
            encoding="utf-8",
        )
        return self.settings_path


def isolate_server_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    """Redirect every server path seam into ``tmp_path`` and reset the module-level caches.

    The CWD move matters as much as the environment: ``results_api.run_roots()`` unconditionally
    appends ``Path.cwd()``, ``cwd()/runs`` and ``cwd()/examples_out``, and
    ``metrics_api._search_roots()`` falls back to the checkout root -- so a test left in the repo
    would index the developer's real runs and assert against them.
    """
    box = Sandbox(tmp_path)
    for d in (box.cwd, box.jobs_root, box.output_dir, box.input_dir):
        d.mkdir(parents=True, exist_ok=True)
    box.write_settings()
    # A hardware profile beside the config keeps start_run's cwd choice inside the sandbox instead
    # of falling through to the real checkout root (metrics_api.start_run's `cwd = next(...)`).
    box.hardware_path.write_text("version: 1\nmodules: {}\n", encoding="utf-8")

    monkeypatch.chdir(box.cwd)
    for name in _PATH_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LM3_SERVER_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("LM3_SETTINGS", str(box.settings_path))
    monkeypatch.setenv("LM3_SETTINGS_PATH", str(box.settings_path))
    monkeypatch.setenv("LM3_HARDWARE", str(box.hardware_path))
    monkeypatch.setenv("LM3_HARDWARE_SETTINGS", str(box.hardware_path))
    monkeypatch.setenv("LM3_SERVER_JOBS", str(box.jobs_root))
    monkeypatch.setenv("LM3_EMBED_TOKEN", "0")
    # Never let _machine3_argv() resolve a real console script: start_run is always stubbed, but
    # the argv it records is asserted on and must not depend on what happens to be installed.
    monkeypatch.setenv("LM3_MACHINE3_BIN", str(tmp_path / "fake-machine3"))
    return box


def reset_server_module_state() -> None:
    """Drop every process-wide cache and binding the server modules keep.

    These are module globals, not per-app state: ``metrics_api._RUN`` is the active-run singleton,
    ``progress_api._BOUND`` is what ``start_run`` pins, and three TTL caches would happily serve one
    test's run listing to the next.
    """
    from leafmachine3.server import metrics_api, progress_api, results_api

    metrics_api._RUN = None
    metrics_api._ADOPT_TRIED = True          # never adopt a record from the developer's real run
    progress_api._BOUND = None
    progress_api._BOUND_STATE = None
    progress_api._JOB_SOURCE = None
    progress_api._SNAPSHOT_CACHE.invalidate()
    progress_api._RUNS_CACHE.invalidate()
    progress_api._SCAN_CACHE.invalidate()
    progress_api._YAML_CACHE.clear()
    results_api._EXTRA_ROOTS.clear()
    results_api._RUNS_CACHE.invalidate()
    results_api._MEDIA_CACHE.clear()


# --------------------------------------------------------------------------- #
# The spawn seam
# --------------------------------------------------------------------------- #
class FakeChild:
    """Stand-in for the ``Popen`` handle ``metrics_api.start_run`` retains.

    It stays "alive" (``poll() is None``) until :meth:`release` is called, because that is the only
    way ``/v1/run/start``'s 409-on-busy branch and ``/v1/run/stop``'s happy path are reachable
    without a real process. ``wait()`` blocks the reaper thread exactly as the real one does.
    """

    #: Well outside ``/proc/sys/kernel/pid_max`` on Linux, so nothing real can ever wear it.
    FAKE_PID = 2 ** 31 - 7

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = list(argv)
        self.kwargs = dict(kwargs)
        self.pid = self.FAKE_PID
        self.returncode: int | None = None
        self._done = threading.Event()

    def poll(self) -> int | None:
        return self.returncode if self._done.is_set() else None

    def wait(self, timeout: float | None = None) -> int:
        self._done.wait(timeout)
        return self.returncode if self.returncode is not None else 0

    def release(self, returncode: int = 0) -> None:
        self.returncode = returncode
        self._done.set()


class _FakeSubprocess:
    """The tiny slice of :mod:`subprocess` that ``metrics_api.start_run`` uses."""

    STDOUT = -2
    DEVNULL = -3

    def __init__(self) -> None:
        self.calls: list[FakeChild] = []

    def Popen(self, argv: list[str], **kwargs: Any) -> FakeChild:   # noqa: N802 - mirrors the API
        child = FakeChild(argv, **kwargs)
        self.calls.append(child)
        return child


def install_fake_spawn(monkeypatch: pytest.MonkeyPatch) -> _FakeSubprocess:
    """Replace the ``subprocess`` module object metrics_api resolves at call time.

    Swapping the module ATTRIBUTE rather than ``subprocess.Popen`` itself keeps the real
    :mod:`subprocess` untouched for everything else in the interpreter, and makes the seam the
    launch handshake of §2.4 will later replace explicit and greppable.
    """
    from leafmachine3.server import metrics_api

    fake = _FakeSubprocess()
    monkeypatch.setattr(metrics_api, "subprocess", fake)
    # The sampler is a process-wide daemon thread that reads /proc (and probes VRAM) on first use.
    # A contract test must not start it, and neither call has anything to do with the HTTP shape.
    monkeypatch.setattr(metrics_api.metrics, "reset_baseline", lambda: {}, raising=True)
    monkeypatch.setattr(metrics_api.metrics, "set_worker_root", lambda pid: None, raising=True)
    return fake


def drain_reapers(fake: _FakeSubprocess, timeout: float = 5.0) -> None:
    """Let every stubbed child exit and its reaper thread finish, before monkeypatch unwinds.

    ``metrics_api._reap`` runs on a daemon thread that calls back into the patched metrics module;
    letting it outlive the patch would have it touch the real sampler after the test is over.
    """
    for child in fake.calls:
        child.release()
    for thread in threading.enumerate():
        if thread.name == "lm3-run-reaper":
            thread.join(timeout)


def make_run_dir(root: Path, name: str) -> Path:
    """A directory both run scanners accept: ``<dir>/<dir.name>.sqlite`` (``core/dirs.py``).

    The ledger is a zero-byte file on purpose -- ``_run_state`` / ``_read_ledger`` both degrade to
    their blank roll-up, which is the shape a contract test wants to pin without depending on the
    project schema.
    """
    run_dir = root / name
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / f"{name}.sqlite").write_bytes(b"")
    return run_dir


def bearer(token: str = TEST_TOKEN) -> dict[str, str]:
    """The ONE way a client authenticates today: an ``Authorization: Bearer <secret>`` header."""
    return {"Authorization": f"Bearer {token}"}


def repo_relative(path: Path) -> str:
    """Small readability helper for failure messages."""
    return os.path.relpath(str(path))


__all__ = [
    "Sandbox", "TEST_TOKEN", "FakeChild",
    "assert_json_shape", "assert_every_item_shape", "bearer", "drain_reapers",
    "install_fake_spawn", "isolate_server_paths", "make_run_dir", "repo_relative",
    "reset_server_module_state",
    "HEALTHZ_SPEC", "RUNS_ENVELOPE_SPEC", "RUN_RECORD_SPEC", "RUN_SUMMARY_SPEC",
    "SETTINGS_SPEC", "STATUS_SPEC",
    "BOOL", "DICT", "INT", "LIST", "NONE", "NUM", "OPT_BOOL", "OPT_DICT", "OPT_INT",
    "OPT_NUM", "OPT_STR", "STR",
]


@contextlib.contextmanager
def capture_lm3_logs() -> Iterator[list[str]]:
    """Capture ``leafmachine3.*`` log records regardless of global logging state.

    NOT ``caplog``. ``leafmachine3.core.logging_setup`` removes the root handlers and sets
    ``root.propagate = False`` (logging_setup.py:19,34), so once any test has started a pipeline,
    caplog captures nothing at all -- and a "the token is not in the captured logs" assertion
    against an EMPTY capture passes vacuously. That is a silently useless security test, which is
    worse than a failing one. This attaches a handler to the ``leafmachine3`` logger itself and
    forces propagation for the duration, then restores both.
    """
    logger = logging.getLogger("leafmachine3")
    messages: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = _Collect(level=logging.DEBUG)
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    try:
        yield messages
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
