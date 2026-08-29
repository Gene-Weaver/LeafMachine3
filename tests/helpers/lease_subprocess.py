"""A tiny CLI driven by ``tests/test_runtime_lease.py`` to exercise the lease with REAL processes.

Plan section 2.2 makes claims that only real processes can settle: a ``flock`` belongs to the open
file description so an inherited descriptor keeps the lease alive after the root is killed
(invariant 2, gate 4); an ordinary executor worker spawned without ``pass_fds`` inherits nothing
(invariant 3, gate 7); a killed lone owner releases the lock and leaves ``active.json`` behind to be
classified as abandoned (gate 51). Simulating those in-process would be simulating the very thing
under test.

Every mode is driven by environment variables rather than a parsed command line, so the test can
add one without touching the argument grammar:

``LM3_TEST_DIR``            the deployment runtime directory (a pytest ``tmp_path``)
``LM3_TEST_KEY``            the canonical deployment key
``LM3_TEST_READY``          a file this process touches once it has reached its steady state
``LM3_TEST_STOP``           a file whose appearance tells this process to exit
``LM3_TEST_REPORT``         where this process writes its JSON report
``LM3_TEST_CHILD_READY``    (root-with-child) the child's ready file
``LM3_TEST_CHILD_REPORT``   (root-with-child) the child's report file
``LM3_TEST_WORKER_REPORT``  (inherited-child) the worker's report file

Exit codes: 0 on success, ``EXIT_CODE_BUSY`` (75) when the lease was occupied -- the code plan
section 3.3 assigns to a busy start -- and 1 for an unexpected failure.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# The helper is executed as a script by ``sys.executable``, so it puts the checkout on the path
# itself rather than depending on how pytest was invoked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from leafmachine3.core.runtime import lease as lease_mod          # noqa: E402 - after the path fix
from leafmachine3.core.runtime._types import (                    # noqa: E402 - after the path fix
    ENV_LEASE_CAPABILITY,
    ENV_LEASE_EVENT_HANDLE,
    ENV_LEASE_FD,
    EXIT_CODE_BUSY,
    RuntimeBusyError,
)

#: Nothing in these modes waits forever. A wedged helper would otherwise turn one failing assertion
#: into a hung suite, and the tests all finish in well under a second.
WAIT_TIMEOUT_S = 60.0
POLL_S = 0.01


def _path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} must be set for this helper mode")
    return Path(value)


def _report(payload: dict[str, object]) -> None:
    target = os.environ.get("LM3_TEST_REPORT")
    if target:
        Path(target).write_text(json.dumps(payload), encoding="utf-8")


def _touch(name: str) -> None:
    value = os.environ.get(name)
    if value:
        Path(value).write_text("ready", encoding="utf-8")


def _wait_for_stop() -> bool:
    """Poll for the stop file. Returns False if the timeout won instead."""
    stop = os.environ.get("LM3_TEST_STOP")
    if not stop:
        return True
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    target = Path(stop)
    while time.monotonic() < deadline:
        if target.exists():
            return True
        time.sleep(POLL_S)
    return False


def _wait_for_file(path: Path) -> bool:
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(POLL_S)
    return False


def _deployment_dir() -> Path:
    return _path("LM3_TEST_DIR")


def _deployment_key() -> str:
    return os.environ.get("LM3_TEST_KEY", "default")


def _lease() -> lease_mod.RuntimeLease:
    return lease_mod.RuntimeLease(
        deployment_dir=_deployment_dir(), deployment_key=_deployment_key(), env=os.environ
    )


def _fd_identity(fd: int) -> dict[str, object]:
    """What a process can say about a descriptor number without assuming it is ours."""
    try:
        info = os.fstat(fd)
    except OSError as exc:
        return {"open": False, "errno": exc.errno}
    return {"open": True, "dev": info.st_dev, "ino": info.st_ino}


def _lease_env_visible() -> dict[str, str | None]:
    return {name: os.environ.get(name) for name in (ENV_LEASE_FD, ENV_LEASE_EVENT_HANDLE, ENV_LEASE_CAPABILITY)}


# --------------------------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------------------------- #

def mode_root_hold() -> int:
    """Acquire the lease as a lone root and hold it until told to stop (or until killed)."""
    lease = _lease()
    try:
        lease.acquire()
    except RuntimeBusyError as busy:
        _report({"result": "busy", "detail": str(busy)})
        return EXIT_CODE_BUSY
    _report({"result": "acquired", "pid": os.getpid(), "fd": getattr(lease.adapter, "fd", None)})
    _touch("LM3_TEST_READY")
    _wait_for_stop()
    lease.release()
    return 0


def mode_contend() -> int:
    """One attempt at the lease; report which way it went. The whole two-process contention test."""
    lease = _lease()
    try:
        lease.acquire()
    except RuntimeBusyError as busy:
        _report({"result": "busy", "detail": str(busy), "deployment_key": busy.deployment_key})
        return EXIT_CODE_BUSY
    _report({"result": "acquired", "pid": os.getpid()})
    _touch("LM3_TEST_READY")
    _wait_for_stop()
    lease.release()
    return 0


def mode_root_with_child() -> int:
    """Acquire, hand the reference to ONE approved subactivity, then wait to be killed.

    This is the gate 4 fixture: the test SIGKILLs this process while the child is still alive and
    then proves a second root is still refused.
    """
    lease = _lease()
    try:
        lease.acquire()
    except RuntimeBusyError as busy:
        _report({"result": "busy", "detail": str(busy)})
        return EXIT_CODE_BUSY
    child_env = dict(os.environ)
    child_env["LM3_TEST_READY"] = os.environ["LM3_TEST_CHILD_READY"]
    child_env["LM3_TEST_REPORT"] = os.environ["LM3_TEST_CHILD_REPORT"]
    with lease.child_handoff("capability-for-the-test") as handoff:
        child_env.update(handoff.env)
        child = subprocess.Popen(                                     # noqa: S603 - fixed argv
            [sys.executable, str(Path(__file__).resolve()), "inherited-child"],
            env=child_env,
            **dict(handoff.popen_kwargs),
        )
    _report({"result": "acquired", "pid": os.getpid(), "child_pid": child.pid,
             "fd": getattr(lease.adapter, "fd", None)})
    _touch("LM3_TEST_READY")
    _wait_for_stop()
    child.wait(timeout=WAIT_TIMEOUT_S)
    lease.release()
    return 0


def mode_root_with_worker() -> int:
    """Acquire, then spawn an ORDINARY worker -- no handoff, no ``pass_fds``.

    Invariant 3 is about workers spawned by the ROOT, from the root's own reference, which is the
    case a child-side disarm can do nothing about. The worker's report is the evidence.
    """
    lease = _lease()
    try:
        lease.acquire()
    except RuntimeBusyError as busy:
        _report({"result": "busy", "detail": str(busy)})
        return EXIT_CODE_BUSY
    fd = getattr(lease.adapter, "fd", None)
    worker_report = _path("LM3_TEST_WORKER_REPORT")
    worker_env = dict(os.environ)
    worker_env["LM3_TEST_REPORT"] = str(worker_report)
    worker_env["LM3_TEST_FD"] = str(fd)
    worker_env.pop("LM3_TEST_STOP", None)
    subprocess.run(                                                   # noqa: S603 - fixed argv
        [sys.executable, str(Path(__file__).resolve()), "worker"],
        env=worker_env, check=False, timeout=WAIT_TIMEOUT_S,
    )
    try:
        worker_result = json.loads(worker_report.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        worker_result = None
    _report({
        "result": "acquired",
        "pid": os.getpid(),
        "fd": fd,
        "owner_fd_inheritable": os.get_inheritable(fd) if fd is not None else None,
        "worker": worker_result,
    })
    _touch("LM3_TEST_READY")
    _wait_for_stop()
    lease.release()
    return 0


def mode_inherited_child() -> int:
    """The approved subactivity: inherit, validate, disarm, then spawn an ordinary worker.

    The worker is spawned WITHOUT the handoff, which is what an executor worker is, and its report
    is what gate 7 asserts against.
    """
    lease = lease_mod.inherit_lease(
        deployment_dir=_deployment_dir(), deployment_key=_deployment_key(), env=os.environ
    )
    fd = getattr(lease.adapter, "fd", None)
    try:
        lease.validate_inherited()
    except Exception as exc:                                          # noqa: BLE001 - reported, then exit
        _report({"result": "rejected", "error": type(exc).__name__, "detail": str(exc)})
        return 1
    lease.disarm_and_clear(os.environ)
    worker_report = os.environ.get("LM3_TEST_WORKER_REPORT")
    worker_result: dict[str, object] | None = None
    if worker_report:
        worker_env = dict(os.environ)
        worker_env["LM3_TEST_REPORT"] = worker_report
        worker_env["LM3_TEST_FD"] = str(fd)
        worker_env.pop("LM3_TEST_STOP", None)      # the worker never waits; it reports and exits
        subprocess.run(                                               # noqa: S603 - fixed argv
            [sys.executable, str(Path(__file__).resolve()), "worker"],
            env=worker_env, check=False, timeout=WAIT_TIMEOUT_S,
        )
        try:
            worker_result = json.loads(Path(worker_report).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            worker_result = None
    _report({
        "result": "validated",
        "pid": os.getpid(),
        "fd": fd,
        "fd_identity": _fd_identity(fd) if fd is not None else None,
        "inheritable": os.get_inheritable(fd) if fd is not None else None,
        "lease_env_after_disarm": _lease_env_visible(),
        "worker": worker_result,
    })
    _touch("LM3_TEST_READY")
    _wait_for_stop()
    return 0


def mode_worker() -> int:
    """An ordinary executor worker: it must see no lease reference at all (invariant 3, gate 7)."""
    raw_fd = os.environ.get("LM3_TEST_FD")
    fd = int(raw_fd) if raw_fd and raw_fd.isdigit() else None
    _report({
        "result": "worker",
        "pid": os.getpid(),
        "fd": fd,
        "fd_identity": _fd_identity(fd) if fd is not None else None,
        "lease_env": _lease_env_visible(),
    })
    return 0


def mode_bogus_child() -> int:
    """A child whose ``LM3_LEASE_FD`` is an ordinary independently opened handle on the lock file.

    Gate 9: the inode matches, so only the non-blocking re-assert can tell the difference. It must
    be rejected while the parent's lock is held.
    """
    lock_path = _deployment_dir() / "activity.lock"
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    os.environ[ENV_LEASE_FD] = str(fd)
    lease = lease_mod.inherit_lease(
        deployment_dir=_deployment_dir(), deployment_key=_deployment_key(), env=os.environ
    )
    try:
        lease.validate_inherited()
    except Exception as exc:                                          # noqa: BLE001 - the expected path
        _report({"result": "rejected", "error": type(exc).__name__, "detail": str(exc)})
        return 0
    _report({"result": "accepted"})
    return 1


_MODES = {
    "root-hold": mode_root_hold,
    "root-with-child": mode_root_with_child,
    "root-with-worker": mode_root_with_worker,
    "inherited-child": mode_inherited_child,
    "worker": mode_worker,
    "contend": mode_contend,
    "bogus-child": mode_bogus_child,
}


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in _MODES:
        raise SystemExit(f"usage: {argv[0]} {{{'|'.join(sorted(_MODES))}}}")
    return _MODES[argv[1]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
