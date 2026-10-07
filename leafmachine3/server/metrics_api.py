"""leafmachine3.server.metrics_api -- HTTP surface for the machine monitor + LM3 run control.

Two jobs, one router:

1. **Machine monitor.** Publishes the ring buffer kept by :mod:`leafmachine3.server.metrics` --
   a current point, the rolling window that fills the plots on first paint, and an SSE stream that
   keeps them live. Nothing here samples hardware; every read is a ring-buffer copy (microseconds),
   so a browser at 2 Hz costs the same as a browser at 0.1 Hz.

2. **Run control.** Starts / stops / reports the LM3 run behind the app's Run button.

   THE RUN IS A SUBPROCESS, AND THAT IS NOT NEGOTIABLE. ``leafmachine3.machine3.main`` opens with
   :func:`~leafmachine3.machine3._exec_with_cuda_libpath`, which puts the venv's bundled
   ``nvidia-*`` libs on ``LD_LIBRARY_PATH`` and RE-EXECS the interpreter, because the dynamic loader
   reads that variable only at PROCESS START. onnxruntime-gpu dlopen's
   ``libonnxruntime_providers_cuda.so`` (which links libcublasLt / cuDNN 9 from those wheels); if the
   variable is not already set when the process starts, ORT silently falls back to
   ``CPUExecutionProvider`` and every ONNX module runs on the CPU -- no error, just a run that is
   ~40x slower. Calling :func:`leafmachine3.machine3.machine3` in-process cannot fix that (machine3()
   deliberately does NOT re-exec: it would restart the server instead of LM3). So the app launches
   the CLI, in its own session, and reads its progress from the project SQLite ledger.

Public Python surface (other server modules import these rather than re-deriving them):
    hardware_profile()      -> parsed + annotated hardware_settings.yaml
    active()                -> the most recent run record (active or finished)
    is_active()             -> bool
    active_db_path()        -> Path | None    (the project SQLite the progress module tails)
    active_log_path()       -> Path | None    (<run>/logs/lm3.log)
    active_console_path()   -> Path | None    (stdout+stderr capture, catches pre-logging crashes)
    start_run(...) / stop_run(...)            -> raise RunError (carries .status) on refusal
    router(dependencies=[Depends(require_token)])  -> the APIRouter the integrator mounts

MOUNTING: this router SUPERSEDES :func:`leafmachine3.server.metrics.router` -- both serve
``/v1/metrics``, and mounting both would leave the first-registered one shadowing the other. Mount
this one only.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from leafmachine3.core.paths import PathsError, resolve_project_output_dir
from leafmachine3.core.runtime._types import (
    DEFAULT_HANDSHAKE_TIMEOUT_S,
    ENV_STATUS_FD,
    ENV_STATUS_HANDLE,
    HTTP_STATUS_BUSY,
    MAX_HANDSHAKE_BYTES,
    HandshakeStatus,
)
from leafmachine3.server import metrics

log = logging.getLogger("leafmachine3.server.metrics_api")

# Display names only. Both files are RESOLVED by leafmachine3.core.paths (plan section 3.1 rows
# 1-2) via the leafmachine3.server.app helpers -- never by scanning the CWD and the checkout for a
# file with one of these names, which is what made this module disagree with settings_api about
# which config the app was on.
HW_FILENAME = "hardware_settings.yaml"
CFG_FILENAME = "LM3_settings.yaml"

# Grace between SIGTERM and SIGKILL when stopping a run. LM3 checkpoints every image into the
# project DB and reclaims 'running' stages on the next start, so a hard stop costs at most the
# images currently in flight -- 10 s is plenty for the executor's pools to unwind on their own.
DEFAULT_STOP_GRACE_S = 10.0

# Mirrors leafmachine3.pipeline.STAGE_ORDER (key, display name, device kind, cpu parallelism).
# Duplicated on purpose: importing pipeline.STAGE_ORDER would drag in every stage module (torch,
# cv2, ultralytics) just to label a settings panel. leafmachine3.core.config.CANONICAL_STAGE_KEYS is
# the cheap cross-check, and _stage_rows() logs if the two ever drift.
_STAGE_META: tuple[tuple[str, str, str, str], ...] = (
    ("mp_conversion_factor", "MP Conversion Factor", "cpu", "thread"),
    ("archival_detector", "Archival Detector", "gpu", ""),
    ("plant_detector", "Plant Detector", "gpu", ""),
    ("specimen_segmenter", "Specimen Segmenter", "gpu", ""),
    ("phenology_detector", "Phenology Detector", "cpu", "thread"),
    ("ruler_classifier", "Ruler Classifier", "gpu", ""),
    ("ruler_cf", "Ruler Conversion Factor", "cpu", "process"),
    ("leaf_segmenter", "Leaf Segmenter", "gpu", ""),
    ("morphology", "Morphology", "cpu", "thread"),
    ("landmark_detector", "Landmark Detector", "gpu", ""),
    ("landmark_measurements", "Landmark Measurements", "cpu", "thread"),
    ("leaf_orientation", "Leaf Orientation", "cpu", "thread"),
    ("petiole_width", "Petiole Width", "cpu", "thread"),
    ("metric_grounding", "Metric Grounding", "cpu", "thread"),
    ("reporter", "Reporter", "cpu", "thread"),
    ("ect", "ECT", "cpu", "process"),
)


class RunError(RuntimeError):
    """A refusal the HTTP layer turns into a status code (409 busy, 400 bad request, ...).

    ``payload`` carries the structured body a refusal needs in order to be ACTIONABLE -- section
    2.4's "``busy`` -> 409 with the winner's record". It stays ``None`` everywhere else, so every
    pre-existing refusal keeps its plain-string ``detail`` and the published API contract does not
    move just because the flag exists.
    """

    def __init__(self, message: str, status: int = 400, *,
                 payload: Optional[dict] = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


# --------------------------------------------------------------------------- #
# The launch handshake, the staging boundary, and the retained control handle
# (plan sections 2.4, 2.5 and 2.13; section 4, Step 3)
# --------------------------------------------------------------------------- #
# EVERYTHING in this section is inert only when ``LM3_RUNTIME_V2=0``. With that explicit fallback
# ``start_run`` takes exactly the path it took before Step 3 -- same mkdir, same console log, same
# Popen keywords -- the compatibility path is useful only if it remains untouched.

#: Overrides :data:`DEFAULT_HANDSHAKE_TIMEOUT_S`. Section 2.4 requires the timeout to be bounded
#: AND configurable; a deployment on a very slow filesystem may legitimately need longer than the
#: default to parse a config, and nothing about that should require a code change.
ENV_HANDSHAKE_TIMEOUT = "LM3_HANDSHAKE_TIMEOUT_S"

#: Where a managed launch keeps its server-private request log (section 2.4: "permitted for a
#: losing launch -- writes under the server-private jobs root, namespaced per job ID").
LAUNCHES_DIRNAME = "launches"

#: THIS server process's identity, minted once, never persisted. Section 2.5 as amended by plan
#: revision 15 makes control authority "a retained live child handle for this ``run_id``, launched
#: by this server ``instance_id``" -- and the instance id is what makes "launched by THIS server"
#: expressible to a client that only ever sees JSON. A restart mints a new one, which is precisely
#: the point: the new process is an OBSERVER of anything the old one launched.
#:
#: Deliberately in-memory. The moment it is written to a file, a restarted server can read it back
#: and claim authority over a child whose handle died with its predecessor -- invariant 12 wearing
#: a different hat.
SERVER_INSTANCE_ID = uuid.uuid4().hex

#: The wire version of ``GET /v1/runtime``. Section 2.9 is about RECORD schema skew; this is the
#: separate question "does this client understand this server's runtime view", which section 2.11
#: gives ``/healthz`` for the server as a whole and step 5b owns.
RUNTIME_PROTOCOL_VERSION = 1

#: How long a registry read is reused before the lock is probed again. The runtime view is polled
#: (the top bar every 4 s, the SSE stream on every run-state edge), and each read is an ``open`` +
#: non-blocking ``flock`` + ``close`` plus a JSON parse. Half a second is far below any human-visible
#: latency and bounds the probe rate no matter how many clients attach.
RUNTIME_CACHE_TTL_S = 0.5

#: The release that DELETES every compatibility surface this module still serves (section 4 Step 7:
#: "a named release owning its removal" -- a deprecation with no owner never happens). It is a
#: constant, not a comment, because ``docs/DEPRECATIONS.md`` and the ``Warning`` headers below must
#: name the same release, and ``tests/test_compat_cleanup.py`` asserts they do.
#:
#: The transition release is the one currently in ``pyproject.toml`` (``leafmachine3.__version__``):
#: it is the release that introduces ``GET /v1/runtime``, so it is the first release in which a
#: client can migrate. One release of overlap, then the projections go.
COMPAT_REMOVAL_RELEASE = "3.1.0"

#: The successor every deprecated run-control surface points at. One string so a header, a
#: docstring and the register cannot drift.
COMPAT_SUCCESSOR_ROUTE = "/v1/runtime"

def deprecation_headers(*, what: str, replacement: str,
                        successor: Optional[str] = COMPAT_SUCCESSOR_ROUTE,
                        fields: Optional[str] = None) -> dict:
    """The headers announcing that ``what`` goes away in :data:`COMPAT_REMOVAL_RELEASE`.

    A client only ever sees HTTP, so this is the only place a deprecation can actually reach one.
    ``Deprecation`` and the ``rel="successor-version"`` link are the RFC 8594 /
    draft-ietf-httpapi-deprecation-header spellings.

    ``Sunset`` is deliberately NOT sent: RFC 8594 requires an HTTP-date, and this project ships
    releases, not dates -- inventing one would be a promise the server cannot keep. The release
    rides in ``Warning`` (human-readable, and already surfaced by most HTTP clients) and in the
    LM3-specific ``X-LM3-Removed-In``, which is the machine-readable one a migration script reads.

    Pass ``fields=`` when the ROUTE survives and only a field inside its body is going
    (``db_path``); omit it when the whole route is going.
    """
    headers = {
        "Deprecation": "true",
        "X-LM3-Removed-In": COMPAT_REMOVAL_RELEASE,
        "Warning": (f'299 - "{what} is deprecated and will be removed in LM3 '
                    f'{COMPAT_REMOVAL_RELEASE}; use {replacement}. '
                    f'See docs/DEPRECATIONS.md."'),
    }
    if successor:
        headers["Link"] = f'<{successor}>; rel="successor-version"'
    if fields:
        # The route is NOT deprecated -- a field in its body is. Spelled separately so a client
        # cannot read "Deprecation: true" as "stop calling this endpoint".
        headers["X-LM3-Deprecated-Fields"] = fields
    return headers



def runtime_v2() -> bool:
    """THE reader of ``LM3_RUNTIME_V2`` for the whole server (section 4, Step 3).

    ``leafmachine3.server.app`` imports this one rather than growing its own: the flag decides
    whether the Step 3 wiring is live, and two readers is how half a feature ships.

    One helper rather than an ``os.environ`` check per call site: a flag spelled in four places is
    four places that can default differently, and four places to miss when the flag is finally
    removed. A missing or broken runtime package is allowed to propagate: now that the registry is
    the safety default, silently falling back would re-enable concurrent roots. The supported old
    path is explicit ``LM3_RUNTIME_V2=0``, never an import accident.
    """
    from leafmachine3.core.runtime.execution import runtime_v2_enabled

    return bool(runtime_v2_enabled())


def _handshake_timeout_s() -> float:
    """How long the server waits for the child's one status line.

    Order of seconds by design (section 2.4): the child attempts the lease immediately after
    ``cfg.validate()``, before hardware profiling and before any model import, so this waits on a
    YAML parse -- not on a cold torch import, and nothing like the 240 s Electron allows for server
    boot. Measured: ``import leafmachine3.machine3`` is 0.2 s and pulls in neither torch nor
    onnxruntime.
    """
    raw = os.environ.get(ENV_HANDSHAKE_TIMEOUT, "").strip()
    if raw:
        try:
            value = float(raw)
        except ValueError:
            log.warning("%s=%r is not a number; using %.1fs", ENV_HANDSHAKE_TIMEOUT, raw,
                        DEFAULT_HANDSHAKE_TIMEOUT_S)
        else:
            if value > 0:
                return value
            log.warning("%s=%r must be positive; using %.1fs", ENV_HANDSHAKE_TIMEOUT, raw,
                        DEFAULT_HANDSHAKE_TIMEOUT_S)
    return float(DEFAULT_HANDSHAKE_TIMEOUT_S)


# --------------------------------------------------------------------------- #
# Process isolation: ONE platform-aware source, and the Windows job object
# (sections 2.4 "child termination ... identical on both platforms", 3.3, 2.13)
# --------------------------------------------------------------------------- #
#: Win32 numbers the job object needs. ``_winapi`` exports none of them, and a bare ``0x2000``
#: sitting in a kill path is unreviewable.
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOBOBJECT_EXTENDED_LIMIT_INFORMATION = 9              # a JOBOBJECTINFOCLASS member
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


def _is_windows_name(name: str) -> bool:
    """Accepts either spelling: ``os.name`` says ``nt``, ``sys.platform`` says ``win32``."""
    return name == "nt" or name.startswith("win")


def process_isolation_kwargs() -> dict:
    """THE source of the "give this child its own tree" ``Popen`` keywords. Both launch sites use it.

    POSIX: ``start_new_session=True``, so one ``killpg`` reaches the executor's spawn workers and
    any subactivity that inherited the lease. Windows: ``CREATE_NEW_PROCESS_GROUP`` -- because
    ``start_new_session`` is SILENTLY DISCARDED there. CPython's Windows ``_execute_child`` takes it
    as ``unused_start_new_session`` and there is no guard in ``Popen.__init__``, so hardcoding it
    (which both managed launches did) left every Windows child in the SERVER's own process group,
    with no group and no job object of its own -- and section 2.4's "child termination is identical
    on both platforms" degraded to "terminate exactly one process" over there.

    Delegated to :func:`leafmachine3.setup.hardware_setup.setup_popen_kwargs` rather than
    re-spelled: that function already had the ``nt`` branch and its own docstring already promised
    "the server builds this argv, spawns it with ``setup_popen_kwargs``". Two spellings of one
    platform rule is how the two launch sites drifted apart in the first place.

    These are ordinary keywords, not inheritance allowlists, so they merge through
    ``compose_launch`` without touching section 2.4's ``pass_fds`` / handle allowlist -- and on
    Windows CPython ORs ``EXTENDED_STARTUPINFO_PRESENT`` into ``creationflags`` itself whenever
    ``startupinfo.lpAttributeList`` is set, so the two cannot clobber each other.

    A process group is only half of it on Windows: ``CREATE_NEW_PROCESS_GROUP`` merely enables
    ``GenerateConsoleCtrlEvent``, while ``TerminateProcess`` still reaches one process. The other
    half is the job object :class:`_ManagedChild` assigns -- see :class:`_JobObject`.
    """
    from leafmachine3.setup.hardware_setup import setup_popen_kwargs

    return dict(setup_popen_kwargs())


def _build_win32_job_surface() -> Any:
    """The real ``ctypes`` binding behind :class:`_JobObject`. Constructed ONLY on Windows.

    ``ctypes.WinDLL`` and ``ctypes.wintypes`` do not exist off Windows, so both are reached only
    from inside this function, which nothing calls unless the platform check has already passed --
    the same discipline :mod:`leafmachine3.core.runtime._win32` uses for the lease adapter, and the
    reason this module still imports cleanly on Linux.

    ``restype`` is set on every handle-returning call on purpose: ctypes defaults to ``c_int``, and
    a 64-bit HANDLE truncated to 32 bits is a handle that fails later, somewhere else.
    """
    import ctypes                                       # noqa: PLC0415 - portable, unlike WinDLL
    from ctypes import wintypes                         # noqa: PLC0415 - Windows-only attribute

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)   # type: ignore[attr-defined]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                 wintypes.DWORD]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),             # ULONG_PTR
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class RealWin32JobSurface:
        """Exactly the five calls :class:`_JobObject` makes, and nothing else."""

        def create_job(self) -> int:
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
            return int(handle)

        def set_kill_on_close(self, job: int) -> None:
            info = ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                    job, _JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
                    ctypes.byref(info), ctypes.sizeof(info)):
                raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

        def assign_process(self, job: int, process_handle: int) -> None:
            if not kernel32.AssignProcessToJobObject(job, process_handle):
                raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")

        def terminate_job(self, job: int, exit_code: int) -> None:
            if not kernel32.TerminateJobObject(job, int(exit_code)):
                raise OSError(ctypes.get_last_error(), "TerminateJobObject failed")

        def open_process(self, pid: int) -> int:
            handle = kernel32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, int(pid))
            if not handle:
                raise OSError(ctypes.get_last_error(), "OpenProcess failed")
            return int(handle)

        def close_handle(self, handle: int) -> None:
            kernel32.CloseHandle(handle)

    return RealWin32JobSurface()


class _JobObject:
    """One managed child's Win32 job object -- the Windows spelling of a POSIX process group.

    Section 2.4 requires child termination to be IDENTICAL on both platforms, and section 3.3 says
    Stop "targets the retained root process group (POSIX) or job object (Windows), so a calibration
    tree stops as one unit rather than orphaning its child". ``CREATE_NEW_PROCESS_GROUP`` does not
    deliver that on its own -- it only enables ``GenerateConsoleCtrlEvent``, while
    ``TerminateProcess`` still reaches exactly one process, so ``calibrate._run_pipeline``'s
    grandchild (which deliberately starts no session of its own) and every executor spawn worker
    would survive a Stop. ``TerminateJobObject`` is what makes the tree one unit.

    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` is deliberate rather than incidental: the tree dies with
    the handle. A POSIX orphan is still reachable by ``killpg`` from any terminal, Windows offers no
    such tool, and section 2.5 forbids a restarted server from ever signaling a run it did not
    launch -- so an orphaned Windows tree would be precisely the "run nobody launched" section 2.4
    exists to prevent, with no way to end it.

    IMPLEMENTED BUT NOT YET NATIVELY VALIDATED (section 1.1, rule 2). Every line here is exercised
    on Linux against an injected fake surface, which is a test of our MODEL of the object manager,
    not of the object manager.
    """

    def __init__(self, surface: Any) -> None:
        self._surface = surface
        self.handle: Optional[int] = int(surface.create_job())
        try:
            surface.set_kill_on_close(self.handle)
        except BaseException:
            # The handle exists but the object is not usable: close it HERE, because the caller
            # has no reference to a half-built object and would otherwise leak a kernel handle.
            self.close()
            raise

    def assign(self, process_handle: int) -> None:
        self._surface.assign_process(self.handle, int(process_handle))

    def terminate(self, exit_code: int = 1) -> bool:
        """Kill the whole tree in one call. ``False`` means the caller should fall back."""
        if self.handle is None:
            return False
        try:
            self._surface.terminate_job(self.handle, int(exit_code))
            return True
        except Exception:                              # noqa: BLE001 - already gone, or denied
            log.debug("TerminateJobObject failed", exc_info=True)
            return False

    def close(self) -> None:
        """Release the handle. With KILL_ON_JOB_CLOSE this is also the tree's last rites."""
        handle, self.handle = self.handle, None
        if handle is None:
            return
        with contextlib.suppress(Exception):
            self._surface.close_handle(handle)


def _attach_job_object(proc: Any, *, platform_name: Optional[str] = None,
                       surface: Any = None) -> Optional[_JobObject]:
    """Put ``proc`` in a fresh job object on Windows. ``None`` everywhere else, and on failure.

    POSIX returns ``None`` because the session created by ``start_new_session`` already IS the tree
    and ``killpg`` already reaches all of it. ``surface`` is injectable so the Windows path runs on
    Linux -- the platform half that is never run is the platform half that is wrong.

    A failure here is logged loudly and does not refuse the launch: a run that starts with a
    degraded Stop is better than no run, and the log line is what tells an operator which one they
    have. The assignment happens immediately after ``Popen`` returns, so a child that spawns its own
    children before it is assigned is a theoretical race -- LM3's children come from a config load
    and a model import later, well past this point.
    """
    name = platform_name if platform_name is not None else os.name
    if surface is None:
        if not _is_windows_name(name):
            return None
        surface = _build_win32_job_surface()

    job: Optional[_JobObject] = None
    borrowed: Optional[int] = None
    try:
        job = _JobObject(surface)
        handle = getattr(proc, "_handle", None)         # subprocess.Popen's Windows process handle
        if handle is None:
            borrowed = int(surface.open_process(int(getattr(proc, "pid", 0) or 0)))
            handle = borrowed
        job.assign(int(handle))
        return job
    except Exception as exc:                           # noqa: BLE001 - never refuse a launch on it
        log.error("could not put the managed child in a job object (%s); Stop will reach only the "
                  "direct child on this platform", exc, exc_info=True)
        if job is not None:
            job.close()
        return None
    finally:
        if borrowed is not None:
            with contextlib.suppress(Exception):
                surface.close_handle(borrowed)


class _ManagedChild:
    """A live child THIS server launched -- and the ONLY thing that authorizes signaling it.

    Plan section 2.5 as of revision 15: control authority is handle ownership ALONE. No PID
    comparison, no process-creation-time comparison. Holding the ``Popen`` makes this process the
    child's parent, so the OS cannot recycle its PID before we reap it and ``poll()`` is already a
    complete answer to "is it still alive"; a PID read back out of a JSON file authorizes nothing,
    which is the whole of invariant 12.

    One class for both managed children -- the pipeline run (section 2.4) and the ``lm3-setup``
    subprocess (section 2.13) -- because "terminate the tree, escalate, then report" is the same
    problem twice and two copies of it drift.
    """

    def __init__(self, proc: Any, *, kind: str, console_path: Optional[Path] = None,
                 log_fh: Any = None, run_id: str = "", platform_name: Optional[str] = None,
                 job_surface: Any = None) -> None:
        self.proc = proc
        self.kind = str(kind)                          # "pipeline" | "hardware_setup"
        self.run_id = str(run_id or "")
        # Stamped, not looked up: the handle and the instance that owns it are born together, so a
        # later reader cannot mistake a handle inherited across a module reload for one this
        # process launched. Section 2.5's "launched by this server instance_id", made checkable.
        self.instance_id = SERVER_INSTANCE_ID
        self.console_path = Path(console_path) if console_path else None
        self.log_fh = log_fh
        self.pid = int(getattr(proc, "pid", 0) or 0)
        self.started_at = time.time()
        # start_new_session=True gives the child its own group, so one killpg reaches the executor's
        # spawn workers AND, for setup, the calibration child that inherited the lease.
        try:
            self.pgid: Optional[int] = os.getpgid(self.pid) if hasattr(os, "getpgid") else None
        except OSError:
            self.pgid = self.pid or None
        # The Windows half of the SAME idea, because section 2.4 says child termination is
        # "identical on both platforms": there is no pgid over there (no os.getpgid, and
        # start_new_session is discarded by CPython's Windows _execute_child), so the tree is a job
        # object instead. ``None`` on POSIX, where the session already is the tree.
        self.job = _attach_job_object(proc, platform_name=platform_name, surface=job_surface)

    # -- observation --------------------------------------------------------- #
    @property
    def alive(self) -> bool:
        try:
            return self.proc.poll() is None
        except Exception:                              # noqa: BLE001 - a stub handle, or already reaped
            return False

    @property
    def returncode(self) -> Optional[int]:
        try:
            return self.proc.poll()
        except Exception:                              # noqa: BLE001
            return None

    # -- control ------------------------------------------------------------- #
    def signal_group(self, sig: int) -> bool:
        """Signal the whole tree. Never signals our own group, and never a bare recorded PID."""
        pgid = self.pgid
        if pgid and hasattr(os, "killpg") and pgid != os.getpgrp():
            try:
                os.killpg(pgid, sig)
                return True
            except ProcessLookupError:
                return False
            except OSError:
                log.debug("killpg(%s, %s) failed; falling back to the handle", pgid, sig, exc_info=True)
        # No process group -- i.e. Windows, or getpgid failed. The retained handle (and the job
        # object built from it) is still ours, and it is the only authority section 2.5 recognizes.
        job = self.job
        if job is not None:
            # TerminateJobObject reaches the WHOLE tree in one call, which is what section 3.3 asks
            # of Stop and what proc.terminate() alone could never do: TerminateProcess reaches one
            # process, so a calibration grandchild or an executor spawn worker would survive it.
            # It is not graceful -- but neither was TerminateProcess, so nothing is lost by using
            # it for SIGTERM as well, and the escalation ladder in terminate() keeps its shape.
            if job.terminate(exit_code=int(sig) or 1):
                return True
            log.debug("TerminateJobObject did not take for the managed %s child; falling back to "
                      "the process handle", self.kind)
        try:
            if sig == getattr(signal, "SIGKILL", None):
                self.proc.kill()
            elif sig == getattr(signal, "SIGTERM", None):
                self.proc.terminate()
            else:
                os.kill(self.pid, sig)
            return True
        except Exception:                              # noqa: BLE001 - already gone
            return False

    def terminate(self, *, grace_s: float = DEFAULT_STOP_GRACE_S,
                  hard_s: float = 5.0) -> Optional[int]:
        """SIGTERM the tree, wait, then SIGKILL it -- section 2.4's "waits and escalates".

        Returns the exit status once no managed child remains, or ``None`` if it outlived even the
        SIGKILL wait (which the caller must report rather than paper over: section 2.4 returns 500
        "only once no managed child remains").
        """
        if not self.alive:
            return self.returncode
        self.signal_group(getattr(signal, "SIGTERM", 15))
        deadline = time.time() + max(0.0, float(grace_s))
        while time.time() < deadline and self.alive:
            time.sleep(0.05)
        if self.alive:
            log.warning("managed %s child pid=%s ignored SIGTERM after %.1fs -- sending SIGKILL",
                        self.kind, self.pid, grace_s)
            self.signal_group(getattr(signal, "SIGKILL", 9))
            hard_deadline = time.time() + max(0.0, float(hard_s))
            while time.time() < hard_deadline and self.alive:
                time.sleep(0.05)
        if not self.alive:
            self.release_job()
        return None if self.alive else self.returncode

    def release_job(self) -> None:
        """Drop the job handle. Idempotent, and safe to call from the reaper or from terminate().

        Deliberately NOT called while the tree is still wanted: KILL_ON_JOB_CLOSE means releasing
        the handle kills whatever is left in the job, which is exactly right once the managed child
        has exited (any surviving grandchild is an orphan section 2.4 does not tolerate) and exactly
        wrong before that.
        """
        job, self.job = self.job, None
        if job is not None:
            job.close()

    def close_log(self, note: str = "") -> None:
        if self.log_fh is None:
            return
        try:
            if note:
                self.log_fh.write(note if note.endswith("\n") else note + "\n")
            self.log_fh.flush()
            self.log_fh.close()
        except Exception:                              # noqa: BLE001 - bookkeeping only
            pass
        self.log_fh = None


# --------------------------------------------------------------------------- #
# The status channel (the SERVER half of section 2.4; the child half is
# leafmachine3.core.runtime.execution.StatusChannel)
# --------------------------------------------------------------------------- #
def _posix_status_pipe() -> tuple:
    """An anonymous pipe whose WRITE end the child inherits by number (``LM3_STATUS_FD``)."""
    read_fd, write_fd = os.pipe()
    # subprocess clears CLOEXEC for everything in pass_fds, but say it anyway: the value the child
    # is told to write to must be inheritable no matter which launcher ends up carrying it.
    os.set_inheritable(write_fd, True)
    state = {"open": True}

    def close_child_end() -> None:
        if state["open"]:
            state["open"] = False
            with contextlib.suppress(OSError):
                os.close(write_fd)

    return read_fd, write_fd, {"pass_fds": (write_fd,)}, close_child_end


def _windows_status_pipe() -> tuple:
    """The same pipe on Windows: an INHERITABLE handle passed through the STARTUPINFOEX allowlist.

    ``pass_fds`` does not exist there, so the transport is normative in section 2.4's own table
    rather than left to implementation. ``_winapi.CreatePipe`` hands back non-inheritable handles,
    so the write end is duplicated with ``bInheritHandle=True`` and the original closed -- the
    duplicate is the value that goes into ``LM3_STATUS_HANDLE`` and into the handle allowlist.
    """
    import _winapi                                     # noqa: PLC0415 - Windows-only, at call time
    import msvcrt                                      # noqa: PLC0415

    from leafmachine3.core.runtime import _win32

    read_handle, write_handle = _winapi.CreatePipe(None, 0)
    try:
        inheritable = _winapi.DuplicateHandle(
            _winapi.GetCurrentProcess(), write_handle, _winapi.GetCurrentProcess(),
            0, True, _winapi.DUPLICATE_SAME_ACCESS)
    finally:
        _winapi.CloseHandle(write_handle)
    read_fd = msvcrt.open_osfhandle(read_handle, os.O_RDONLY)
    state = {"open": True}

    def close_child_end() -> None:
        if state["open"]:
            state["open"] = False
            with contextlib.suppress(OSError):
                _winapi.CloseHandle(inheritable)

    return read_fd, inheritable, _win32.handle_list_popen_kwargs([inheritable]), close_child_end


class _Handshake:
    """One outcome of the section 2.4 read: what the child said, or why it said nothing."""

    __slots__ = ("outcome", "payload", "detail")

    #: Everything that is not ``acquired`` or ``busy`` shares one consequence -- terminate the
    #: retained tree, reconcile, 500 -- so they are spelled out separately only for the message.
    FAILURES = ("timeout", "eof", "malformed", "oversized")

    def __init__(self, outcome: str, payload: Optional[dict] = None, detail: str = "") -> None:
        self.outcome = outcome
        self.payload = payload
        self.detail = detail

    @property
    def ok(self) -> bool:
        return self.outcome in ("acquired", "busy")

    def __repr__(self) -> str:                         # pragma: no cover - diagnostics only
        return f"_Handshake({self.outcome!r}, detail={self.detail!r})"


class _LaunchStatusPipe:
    """The server end of the launch handshake: create it, contribute it, read ONE line, close it.

    It is a dedicated control pipe and NOT the child's stdout, for the reason section 2.4 gives:
    stdout goes to a server-private request log from the moment of ``Popen``, so a child may emit
    unbounded startup chatter before its status line without ever filling a pipe the server is not
    draining. This pipe carries exactly one bounded JSON line and is then closed.

    ``factory`` is injectable so the Windows transport is exercised on Linux -- the platform half
    that is never run is the platform half that is wrong.
    """

    def __init__(self, *, platform_name: Optional[str] = None, factory: Any = None) -> None:
        name = platform_name if platform_name is not None else sys.platform
        self.windows = name.startswith("win")
        make = factory or (_windows_status_pipe if self.windows else _posix_status_pipe)
        self.read_fd, self.child_value, self._popen_kwargs, self._close_child = make()
        self.env_name = ENV_STATUS_HANDLE if self.windows else ENV_STATUS_FD
        self._read_open = True
        self._fd_lock = threading.Lock()               # so pump and close cannot both close read_fd
        self._thread: Optional[threading.Thread] = None

    # -- launch side --------------------------------------------------------- #
    def contribution(self) -> Any:
        """What this pipe needs added to the child's launch.

        Returned as a ``LaunchContribution`` rather than as loose keywords because the lease handoff
        wants the SAME exhaustive keyword (``pass_fds`` / the Windows ``handle_list``) and the
        second one supplied silently replaces the first. ``compose_launch`` is the only thing
        allowed to merge them (Step 3 entry task 3).
        """
        from leafmachine3.core.runtime.launch import LaunchContribution

        return LaunchContribution(
            name="launch-status-pipe",
            env={self.env_name: str(self.child_value)},
            popen_kwargs=dict(self._popen_kwargs),
            close=self.close_child_end,
        )

    def close_child_end(self) -> None:
        """Drop OUR copy of the write end. Until this runs, EOF can never arrive."""
        self._close_child()

    # -- read side ----------------------------------------------------------- #
    def read(self, timeout_s: float) -> _Handshake:
        """Read one JSON line, bounded in both time and bytes.

        A reader THREAD rather than ``select``: ``select`` does not work on pipes on Windows, and
        section 2.4 requires the timeout, the size cap, the EOF handling and the termination path to
        be identical on both platforms.

        Normally the thread ends promptly: on the timeout branch the caller terminates the child,
        which closes the last write end, which is the EOF that ends it. But the module's own 500
        branch admits the case where that does not happen ("the child ... survived SIGKILL"), and a
        grandchild that inherited the write end can outlive the escalation -- so the thread OWNS
        ``read_fd`` from here on and closes it itself. :meth:`close` never frees the number out from
        under it, because a freed number is reused, and a zombie reader parked on a recycled
        descriptor eats bytes belonging to whatever the server opened next.
        """
        self.close_child_end()
        box: dict[str, Any] = {}

        def pump() -> None:
            buf = bytearray()
            try:
                while True:
                    chunk = os.read(self.read_fd, 4096)
                    if not chunk:
                        box["eof"] = True
                        break
                    buf.extend(chunk)
                    if b"\n" in buf:
                        break
                    if len(buf) > MAX_HANDSHAKE_BYTES:
                        box["oversized"] = True
                        break
            except OSError as exc:
                box["error"] = str(exc)
            box["data"] = bytes(buf)
            # Whoever leaves the read loop closes the descriptor, and exactly one of the two ever
            # does (``_close_read_fd`` is guarded). That is what lets close() walk away from a
            # thread still blocked in os.read() without freeing the number it is blocked on.
            self._close_read_fd()

        thread = threading.Thread(target=pump, name="lm3-launch-handshake", daemon=True)
        self._thread = thread
        thread.start()
        thread.join(max(0.05, float(timeout_s)))
        if thread.is_alive():
            return _Handshake("timeout", None,
                              f"the child sent no status line within {float(timeout_s):.1f}s")
        return self._parse(box)

    @staticmethod
    def _parse(box: dict) -> _Handshake:
        if box.get("oversized"):
            return _Handshake("oversized", None,
                              f"the status line exceeded {MAX_HANDSHAKE_BYTES} bytes")
        data = box.get("data") or b""
        line = data.split(b"\n", 1)[0].strip()
        if not line:
            reason = box.get("error") or "the child closed the status channel without writing"
            return _Handshake("eof", None, str(reason))
        if len(line) > MAX_HANDSHAKE_BYTES:
            return _Handshake("oversized", None,
                              f"the status line exceeded {MAX_HANDSHAKE_BYTES} bytes")
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            return _Handshake("malformed", None, f"the status line was not JSON: {exc}")
        if not isinstance(payload, dict):
            return _Handshake("malformed", None, "the status line was not a JSON object")
        status = payload.get("status")
        if status == HandshakeStatus.ACQUIRED.value:
            return _Handshake("acquired", payload)
        if status == HandshakeStatus.BUSY.value:
            return _Handshake("busy", payload)
        return _Handshake("malformed", payload, f"unknown handshake status {status!r}")

    def _close_read_fd(self) -> None:
        """Close the read end at most once, whichever of pump/close gets here first."""
        with self._fd_lock:
            if not self._read_open:
                return
            self._read_open = False
        with contextlib.suppress(OSError):
            os.close(self.read_fd)

    def close(self, *, join_s: float = 5.0) -> None:
        """Close both ends -- but NEVER out from under a blocked reader thread.

        If the pump is still parked in ``os.read`` after the join (a grandchild inherited the write
        end and survived the escalation), the descriptor stays open and the pump keeps ownership of
        it: it closes it when EOF finally arrives. One descriptor per stuck launch, reclaimed if the
        survivor is ever killed, is the honest cost. Closing it here instead would free the NUMBER
        while a thread is blocked on it -- ``close()`` does not wake a blocked ``read()`` on Linux --
        and the next descriptor the server opens gets that number and that zombie reader.
        """
        self.close_child_end()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, float(join_s)))
            if thread.is_alive():
                log.warning("the launch status reader is still blocked; leaving fd %s to it rather "
                            "than freeing a number it is reading from", self.read_fd)
                return
        self._close_read_fd()


# --------------------------------------------------------------------------- #
# The staging boundary (section 2.4; invariant 14 stated in testable terms)
# --------------------------------------------------------------------------- #
def execution_owned_paths(run_dir: Any, run_name: str) -> tuple:
    """Everything a LOSING launch is forbidden to create, named once.

    Section 2.4's prohibited list, verbatim: "anything under ``<output.dir>/<run_name>/``, the
    project DB, the run's ``logs/``". The permitted list -- uploads, the generated settings file
    and the server-side request log, all under the server-private jobs root -- is what
    :class:`~leafmachine3.server.app.JobManager` already writes and what the managed launch's
    console log joins.

    It is a function, and public, so the invariant-14 test and the server agree on what the
    invariant MEANS instead of each keeping its own list.
    """
    root = Path(run_dir)
    return (root, root / "logs", root / "logs" / "lm3.log", root / "logs" / "console.log",
            root / f"{run_name}.sqlite", root / "_tmp_original")


def _staging_snapshot(prohibited: tuple) -> set:
    """Which prohibited paths existed BEFORE the launch (a resumed run legitimately has some)."""
    return {p for p in prohibited if p.exists()}


def _check_staging_boundary(prohibited: tuple, before: set, why: str) -> list:
    """Log, loudly, anything the server created that a losing launch may not create.

    The check is the enforcement: with the flag on the server creates none of these, so a
    non-empty result means a regression put a ``mkdir`` back on the pre-handshake path.
    """
    created = [p for p in prohibited if p.exists() and p not in before]
    if created:
        log.error("invariant 14 violated: a %s launch created execution-owned paths %s",
                  why, [str(p) for p in created])
    return created

# --------------------------------------------------------------------------- #
# Path resolution -- one canonical chain, shared with every other server module
# --------------------------------------------------------------------------- #
# ``_repo_root`` / ``_search_roots`` / ``_find_file`` are GONE. They implemented a fourth
# resolution rule (CWD first, then the checkout root, with a set-but-missing override returning
# None instead of falling through), which is exactly the "three fallbacks reachable from three
# CWDs" plan section 4 Step 1 says a rename would preserve. Under an installed wheel the checkout
# root was site-packages itself, so the scan could also read -- and _state_path could write --
# inside a library directory.
def hardware_path() -> Optional[Path]:
    """The hardware profile, or ``None`` when it has not been written yet.

    Section 3.1 row 2: ``LM3_HARDWARE`` (honoring the deprecated ``LM3_HARDWARE_SETTINGS`` for one
    release), else ``<user-config>/lm3/<deployment>/hardware_settings.<machine-key>.yaml``, with a
    one-release adopt of a profile sitting beside the resolved settings file. ``None`` is still the
    miss shape because ``hardware_profile()`` renders a "run LM3_Setup" panel from it.
    """
    from leafmachine3.server.app import canonical_hardware_path

    try:
        path = canonical_hardware_path()
    except PathsError as exc:
        log.warning("cannot resolve the hardware profile: %s", exc)
        return None
    return path if path.is_file() else None


def default_config_path() -> Optional[Path]:
    """The canonical next-run settings file, or ``None`` when nothing resolves to a real file.

    Section 3.1 row 1. ``start_run`` turns ``None`` into a 400, so resolution never seeds here:
    creating a settings file as a side effect of pressing Start would be a surprise, and
    ``lm3 serve`` has already seeded one at startup if the install needed it.
    """
    from leafmachine3.server.app import canonical_settings_path

    try:
        path = canonical_settings_path()
    except PathsError as exc:
        log.warning("cannot resolve the LM3 settings file: %s", exc)
        return None
    return path if path.is_file() else None


def _free_gb(path: Path) -> Optional[float]:
    """Free space on the filesystem holding ``path`` (walking up to the nearest existing parent)."""
    probe = path
    for _ in range(6):
        if probe.exists():
            try:
                return round(shutil.disk_usage(probe).free / (1024.0 ** 3), 1)
            except OSError:
                return None
        if probe.parent == probe:
            break
        probe = probe.parent
    return None


def _iso(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


# --------------------------------------------------------------------------- #
# Hardware profile
# --------------------------------------------------------------------------- #
def _stage_rows(stages: dict, n_gpus: int, io_workers: Optional[int]) -> list[dict]:
    """One row per canonical module, annotated with the PLANNED worker count.

    The planned count is the number the status tab draws per-worker bars for: GPU modules run
    ``workers_per_gpu`` spawn workers on EACH visible GPU, CPU modules run ``workers`` (threads or
    spawn processes depending on ``cpu_parallel``). A module missing from the profile was disabled
    when LM3_Setup last ran -- it is still listed, marked ``tuned: false``, so the panel shows the
    full pipeline instead of a hole.
    """
    known = {key for key, _, _, _ in _STAGE_META}
    extra = [k for k in stages if k not in known]
    if extra:                                          # profile written by a newer LM3 than this UI
        log.debug("hardware profile has unknown stage keys: %s", extra)

    rows: list[dict] = []
    order = 0
    for key, name, device, parallel in _STAGE_META:
        cfg = stages.get(key) or {}
        order += 1
        if device == "gpu":
            per_gpu = cfg.get("workers_per_gpu")
            planned = int(per_gpu) * max(1, n_gpus) if per_gpu else None
        else:
            workers = cfg.get("workers", io_workers)
            planned = int(workers) if workers else None
        rows.append({
            "key": key,
            "name": name,
            "order": order,
            "device": device,
            "cpu_parallel": str(cfg.get("cpu_parallel", parallel)) or None,
            "tuned": key in stages,
            "planned_workers": planned,
            "workers": cfg.get("workers"),
            "workers_per_gpu": cfg.get("workers_per_gpu"),
            "batch": cfg.get("batch"),
            "queue_size": cfg.get("queue_size"),
            "peak_vram_mb": cfg.get("peak_vram_mb"),
            "io_bound": bool(cfg.get("io_bound", False)),
            "disk_write_mbps": cfg.get("disk_write_mbps"),
            "min_pool_items": cfg.get("min_pool_items"),
            "spawn_overhead_s": cfg.get("spawn_overhead_s"),
        })
    for key in extra:
        cfg = stages.get(key) or {}
        order += 1
        rows.append({"key": key, "name": key.replace("_", " ").title(), "order": order,
                     "device": "cpu", "cpu_parallel": cfg.get("cpu_parallel"), "tuned": True,
                     "planned_workers": cfg.get("workers"), "workers": cfg.get("workers"),
                     "workers_per_gpu": cfg.get("workers_per_gpu"), "batch": cfg.get("batch"),
                     "queue_size": cfg.get("queue_size"), "peak_vram_mb": cfg.get("peak_vram_mb"),
                     "io_bound": bool(cfg.get("io_bound", False)),
                     "disk_write_mbps": cfg.get("disk_write_mbps"),
                     "min_pool_items": cfg.get("min_pool_items"),
                     "spawn_overhead_s": cfg.get("spawn_overhead_s")})
    return rows


def _drift(fingerprint: dict) -> dict:
    """Compare the profile's fingerprint against the LIVE machine.

    hardware_setup does the authoritative check (it also hashes the model exports), but that needs a
    loaded Config. This is the cheap version -- core count, RAM, GPU names, driver -- enough for the
    app to say "your profile predates this hardware, rerun LM3_Setup" without touching disk.
    """
    reasons: list[str] = []
    try:
        machine = metrics.describe_machine()
    except Exception:                                  # noqa: BLE001 - a hint must never 500
        return {"checked": False, "stale": False, "reasons": []}

    cores = machine.get("cpu", {}).get("cores_logical")
    want_cores = fingerprint.get("cpu_cores")
    if cores and want_cores and int(cores) != int(want_cores):
        reasons.append(f"CPU cores {want_cores} -> {cores}")

    ram_mb = machine.get("memory", {}).get("ram_total_mb")
    want_ram = fingerprint.get("ram_gb")
    if ram_mb and want_ram and abs(ram_mb / 1024.0 - float(want_ram)) > 2.0:
        reasons.append(f"RAM {want_ram} GB -> {round(ram_mb / 1024.0)} GB")

    live_gpus = [g.get("name") for g in machine.get("gpus", [])]
    want_gpus = [g[0] if isinstance(g, (list, tuple)) and g else None
                 for g in (fingerprint.get("gpus") or [])]
    if live_gpus != want_gpus:
        reasons.append(f"GPUs {want_gpus or 'none'} -> {live_gpus or 'none'}")

    driver, want_driver = machine.get("gpu_driver"), fingerprint.get("driver")
    if driver and want_driver and str(driver) != str(want_driver):
        reasons.append(f"driver {want_driver} -> {driver}")

    return {"checked": True, "stale": bool(reasons), "reasons": reasons}


def hardware_profile() -> dict:
    """``hardware_settings.yaml`` parsed, annotated, and safe to render.

    ``raw`` is the file verbatim (so the app can show it as YAML); everything above it is the
    derived view the tuning panel actually draws: per-module planned workers, GPU sizing, the
    measured spawn overhead and the disk-write knee.
    """
    path = hardware_path()
    if path is None:
        return {"available": False, "path": None, "reason": f"no {HW_FILENAME} found -- run LM3_Setup",
                "raw": {}, "stages": _stage_rows({}, 0, None), "gpus": [], "n_gpus": 0,
                "tuning": {}, "fingerprint": {}, "drift": {"checked": False, "stale": False,
                                                           "reasons": []}}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError) as exc:
        return {"available": False, "path": str(path), "reason": f"could not read {path.name}: {exc}",
                "raw": {}, "stages": _stage_rows({}, 0, None), "gpus": [], "n_gpus": 0,
                "tuning": {}, "fingerprint": {}, "drift": {"checked": False, "stale": False,
                                                           "reasons": []}}

    gpus = list(raw.get("gpus") or [])
    stages = dict(raw.get("stages") or {})
    io_workers = raw.get("io_workers")
    tmp_dir = raw.get("tmp_dir")
    rows = _stage_rows(stages, len(gpus), io_workers)

    # The two measured constants worth surfacing on their own: the spawn-pool startup cost (which
    # sets each process module's min_pool_items) and the disk-write knee (which caps the Reporter).
    spawn = next((r["spawn_overhead_s"] for r in rows if r.get("spawn_overhead_s")), None)
    reporter = next((r for r in rows if r["key"] == "reporter"), {})
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = None

    return {
        "available": True,
        "path": str(path),
        "mtime": mtime,
        "age_s": round(time.time() - mtime, 1) if mtime else None,
        "generated_at": raw.get("generated_at"),
        "lm3_version": raw.get("lm3_version"),
        "provider": raw.get("provider"),
        "precision": raw.get("precision"),
        "cpu_cores": raw.get("cpu_cores"),
        "ram_gb": raw.get("ram_gb"),
        "io_workers": io_workers,
        "tmp_dir": tmp_dir,
        "tmp_free_gb": _free_gb(Path(str(tmp_dir))) if tmp_dir else None,
        "gpus": gpus,
        "n_gpus": len(gpus),
        "stages": rows,
        "tuning": {
            "io_workers": io_workers,
            "spawn_overhead_s": spawn,
            "disk_write_mbps": reporter.get("disk_write_mbps"),
            "disk_writers": reporter.get("workers"),
            "tmp_dir": tmp_dir,
            "provider": raw.get("provider"),
            "precision": raw.get("precision"),
            "n_gpus": len(gpus),
            "gpu_workers": sum(r["planned_workers"] or 0 for r in rows if r["device"] == "gpu"),
        },
        "fingerprint": dict(raw.get("fingerprint") or {}),
        "drift": _drift(dict(raw.get("fingerprint") or {})),
        "raw": raw,
    }


# --------------------------------------------------------------------------- #
# Run control
# --------------------------------------------------------------------------- #
class _Run:
    """One LM3 process THIS server launched, and everything the app needs to describe it.

    Step 4 removed the "or adopted" half. Section 2.5, revision 15: control authority is a retained
    live child handle and nothing else, so a run this process did not launch has no ``_Run`` at all
    -- it is observed through the registry (:func:`runtime_view`) and reported ``can_stop: false``.
    The server's private state is now exactly this object and the handle it holds.
    """

    def __init__(self, **fields: Any) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_fh: Any = None
        self.pid: int = 0
        self.pgid: Optional[int] = None
        # Frozen at False. Adoption is gone (section 2.5); the field survives only because
        # ``GET /v1/run/active`` is a compatibility projection whose key set is pinned until
        # section 4 Step 7 retires the route.
        self.adopted = False
        # The section 2.5 control handle. Set only on the runtime-v2 path, where it -- and nothing
        # else -- is what authorizes signaling this run.
        self.child: Optional[_ManagedChild] = None
        self.run_id = ""                               # the runtime record id, from the handshake
        self.state = "running"                         # running | stopping | done | error
        self.returncode: Optional[int] = None
        self.stopped_by_user = False
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.run_name = ""
        self.config_path = ""
        self.cwd = ""
        self.output_dir = ""
        self.run_dir = ""
        self.db_path = ""
        self.log_path = ""
        self.console_log = ""
        self.tmp_dir = ""
        self.input_dirs: list[str] = []
        self.restart: list[str] = []
        self.argv: list[str] = []
        for key, value in fields.items():
            setattr(self, key, value)

    @property
    def alive(self) -> bool:
        """Is the process we launched still running? Answered from the HANDLE, never from a PID.

        There is no third branch any more. A ``_Run`` without a handle would describe a process
        this server does not own, and probing its PID -- with ``psutil`` or ``os.kill(pid, 0)`` --
        is exactly the reconstruction invariant 12 forbids. ``False`` is the honest answer.
        """
        if self.child is not None:                     # a retained handle is a complete answer
            return self.child.alive
        if self.proc is not None:                      # the pre-flag path: the Popen IS the handle
            return self.proc.poll() is None
        return False

    def record(self) -> dict:
        """The wire shape of ``GET /v1/run/active`` (also used for the SSE ``run`` frame).

        Two keys here are deprecated and both are removed in :data:`COMPAT_REMOVAL_RELEASE`, with
        the route that carries them (section 4 Step 7): ``db_path``, a projection of the section
        2.10 ``active_db_path`` storage role, and ``adopted``, frozen at ``False`` since section
        2.5 made control authority handle ownership alone. Neither is removed EARLIER than the
        route, because the key set is pinned key-for-key (``tests/_contract_helpers.py``) and
        dropping a key from a compatibility projection breaks the clients it exists to serve.
        """
        active = self.state in ("running", "stopping")
        end = self.finished_at or time.time()
        return {
            "active": active,
            "state": self.state,
            "run_name": self.run_name,
            "pid": self.pid,
            "pgid": self.pgid,
            "started_at": self.started_at,
            "started_iso": _iso(self.started_at),
            "finished_at": self.finished_at,
            "finished_iso": _iso(self.finished_at),
            "elapsed_s": round(end - self.started_at, 1) if self.started_at else None,
            "returncode": self.returncode,
            "stopped_by_user": self.stopped_by_user,
            "adopted": self.adopted,
            "error": self.error,
            "config_path": self.config_path,
            "cwd": self.cwd,
            "output_dir": self.output_dir,
            "run_dir": self.run_dir,
            "db_path": self.db_path,
            "log_path": self.log_path,
            "console_log": self.console_log,
            "tmp_dir": self.tmp_dir,
            "input_dirs": list(self.input_dirs),
            "restart": list(self.restart),
            "argv": list(self.argv),
        }


_IDLE_RECORD: dict = {
    "active": False, "state": "idle", "run_name": None, "pid": None, "pgid": None,
    "started_at": None, "started_iso": None, "finished_at": None, "finished_iso": None,
    "elapsed_s": None, "returncode": None, "stopped_by_user": False, "adopted": False,
    "error": None, "config_path": None, "cwd": None, "output_dir": None, "run_dir": None,
    "db_path": None, "log_path": None, "console_log": None, "tmp_dir": None,
    "input_dirs": [], "restart": [], "argv": [],
}

_RUN: Optional[_Run] = None
_RUN_LOCK = threading.RLock()
#: Set once the legacy adoption record has been looked for and removed. See
#: :func:`_purge_legacy_state_file`.
_LEGACY_STATE_PURGED = False

#: A managed launch that has been admitted but has not answered yet -- the section 2.4 handshake is
#: in flight. It exists so ``_RUN_LOCK`` does not have to be: the handshake is a bounded read of up
#: to ``LM3_HANDSHAKE_TIMEOUT_S`` (30 s by default) followed, on the failure branch, by a SIGTERM
#: grace and a SIGKILL wait, and holding the lock across all of that froze every other run-control
#: and status request -- including the un-threadpooled ``GET /v1/run/active``, which blocks the ASGI
#: event loop and with it the whole server. ``stop_run`` already had this shape (lock, signal,
#: unlock, wait, relock); the launch path now matches it.
#:
#: Set ONLY on the runtime-v2 path. With the flag off it stays ``None`` and every reader below takes
#: exactly the branch it took before Step 3.
_LAUNCHING: Optional[dict] = None


def _state_path() -> Path:
    """Where the pre-Step-4 adoption record USED to live. Nothing writes it any more.

    The file existed so a restarted server could re-attach to a still-running LM3 by reading a PID
    out of JSON. Section 2.5 (revision 15) deletes that: control authority is a retained live child
    handle, a restarted server is an observer, and a PID in a file authorizes nothing. The path is
    still resolved here -- by the SAME helper ``JobManager`` uses, section 3.1 row 4 -- because a
    file left behind by an older build has to be found in order to be removed
    (:func:`_purge_legacy_state_file`), and because "which jobs root is this server on?" is a
    question every module must answer identically.
    """
    from leafmachine3.server.app import server_jobs_root

    return server_jobs_root() / "active_run.json"


def _purge_legacy_state_file() -> None:
    """Delete a pre-Step-4 ``active_run.json``, once per process.

    Not tidiness. That file is a PID plus a process-creation time, and its only reader was the
    re-adoption path that section 2.5 removes; leaving it on disk is an invitation to write that
    reader again ("there is already a record, why not use it?"). Removing it makes the absence of
    the mechanism visible on the filesystem as well as in the code.
    """
    global _LEGACY_STATE_PURGED
    if _LEGACY_STATE_PURGED:
        return
    _LEGACY_STATE_PURGED = True
    try:
        path = _state_path()
    except Exception:                                  # noqa: BLE001 - a broken env must still serve
        return
    try:
        if path.is_file():
            path.unlink()
            log.info("removed the legacy adoption record %s: a restarted server is an observer "
                     "(plan section 2.5)", path)
    except OSError:                                    # bookkeeping only -- never fail a request
        log.debug("could not remove the legacy adoption record %s", path, exc_info=True)


def _bind_status(run: "_Run") -> None:
    """Tell the status side which run to describe.

    Without it, progress_api has to GUESS which ledger to describe -- from the configured
    <output dir>/<project name>, or by scanning the disk -- and a run whose output lands outside
    the configured output dir (an example, a one-off --output) is invisible to both. The status
    stream, the stage bar AND the console all resolve through it, so the console can sit on a
    dead ledger reporting "waiting for log" while a run is going.

    IT IS NO LONGER THE AUTHORITY. Step 4 makes the ACTIVE RUNTIME RECORD the second tier of
    ``progress_api.resolve_run()`` -- above anything bound from here -- so the run holding the
    deployment describes itself, whoever launched it. What survives here is the tier below that:
    the run THIS server launched, named explicitly. It is the only thing that still works when the
    registry is unreadable, and the only thing at all when the flag is off. Section 4 Step 7 can
    drop it along with ``/v1/run/active``.

    :func:`active_run_ref` is the registry-backed replacement other modules read.

    No ``state`` is passed on purpose: that would override the ledger's own reading for as long
    as the binding lasts, freezing the run at "running" after it finished. The ledger already
    knows how it ended.

    The binding is NOT cleared when the run ends -- "the run you just did, and how it finished"
    is exactly what the app should still be showing. It lives only in this process, so the next
    server start goes back to the registry (or, failing that, to the configured project).
    """
    try:
        from leafmachine3.server import progress_api

        progress_api.bind_run(run.db_path, run_name=run.run_name)
    except Exception:                                  # noqa: BLE001 - never break a launch over this
        log.debug("could not bind the status stream to run %r", run.run_name, exc_info=True)


def _current() -> Optional[_Run]:
    """The run THIS server launched, or ``None``.

    It no longer adopts anything. A run started elsewhere is reported by :func:`runtime_view` and
    projected into :func:`active`, with ``can_stop: false`` -- observed, never claimed.
    """
    with _RUN_LOCK:
        _purge_legacy_state_file()
        return _RUN



def _idle_record() -> dict:
    idle = dict(_IDLE_RECORD)                          # fresh lists: callers must not alias the template
    idle["input_dirs"], idle["restart"], idle["argv"] = [], [], []
    return idle


def _launching_record() -> Optional[dict]:
    """The wire record for a launch whose handshake has not answered yet, or ``None``.

    Same key set as every other run record -- the docstring of :meth:`_Run.record` promises it never
    changes -- with section 3.3's own first lifecycle state, ``starting``. Answering "idle" while a
    child is being launched would be a lie the UI acts on (it re-enables Start), and answering
    nothing at all is what made holding ``_RUN_LOCK`` across the handshake look acceptable.
    """
    with _RUN_LOCK:
        pending = dict(_LAUNCHING) if _LAUNCHING is not None else None
    if pending is None:
        return None
    record = _idle_record()
    record.update({
        "active": True,
        "state": "starting",
        "run_name": pending.get("run_name"),
        "config_path": pending.get("config_path"),
        "cwd": pending.get("cwd"),
        "output_dir": pending.get("output_dir"),
        "run_dir": pending.get("run_dir"),
        "started_at": pending.get("started_at"),
        "started_iso": _iso(pending.get("started_at")),
        "argv": list(pending.get("argv") or []),
    })
    return record


# --------------------------------------------------------------------------- #
# The registry reader -- GET /v1/runtime (plan sections 2.5, 2.9, 3.2 and 5)
# --------------------------------------------------------------------------- #
# Observation comes from the registry; control comes from a handle (section 2.5). Everything below
# this line is the OBSERVATION half: it opens no lease, signals nothing, and writes nothing. The
# only control input it produces is ``can_stop``, and that is derived from ``_ManagedChild`` --
# never from a PID, a process creation time, or anything else read off disk.

#: ``(monotonic deadline, snapshot, diagnostics)`` -- see :data:`RUNTIME_CACHE_TTL_S`.
_RUNTIME_CACHE: Optional[tuple] = None
_RUNTIME_CACHE_LOCK = threading.Lock()


def invalidate_runtime_cache() -> None:
    """Drop every cached read behind the runtime view: the registry snapshot and the settings parse.

    Called whenever this server changes something the registry will report (a launch, a stop), so
    the next reader sees the new state immediately instead of up to :data:`RUNTIME_CACHE_TTL_S`
    later. It is also the hook progress_api's snapshot caches need when they stop being invalidated
    by ``bind_run``.

    The settings memo (:func:`_load_next_run_config`) is already self-invalidating on the file's
    mtime, so dropping it here is belt and braces -- it costs one re-parse at a run-state edge and
    removes any chance of a process-wide cache outliving the world it was read from.
    """
    global _RUNTIME_CACHE
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE = None
    with _SETTINGS_CACHE_LOCK:
        _SETTINGS_CACHE.clear()


def deployment_dir() -> Path:
    """This deployment's runtime directory -- ``<runtime base>/<canonical deployment key>``.

    Resolved through :mod:`leafmachine3.core.paths` (section 3.1) with ``create=False``: reading the
    registry must never bring a deployment into existence, because "is anything running here?" is a
    question an idle server asks constantly and the answer must not be a side effect.

    ``check_filesystem=False`` for the same reason. Section 3.1's network-filesystem refusal is a
    STARTUP policy aimed at whoever is about to take a lock on an unreliable one; an observer that
    refuses to look is strictly worse than one that looks and reports, and it would turn a
    misconfigured runtime base into a server that cannot even say what is running.
    """
    from leafmachine3.core import paths as core_paths

    return core_paths.deployment_runtime_dir(create=False, check_filesystem=False)


def _read_registry() -> tuple:
    """``(snapshot, diagnostics)``. Never raises: a broken environment must still be REPORTABLE."""
    diagnostics: dict = {"deployment_dir": None, "deployment_key": None, "readable": False,
                         "error": None}
    try:
        from leafmachine3.core import paths as core_paths
        from leafmachine3.core.runtime import records as runtime_records
    except Exception as exc:                           # noqa: BLE001 - a base install must serve
        diagnostics["error"] = f"the unified runtime is unavailable: {exc}"
        return None, diagnostics
    try:
        directory = deployment_dir()
        diagnostics["deployment_dir"] = str(directory)
        diagnostics["deployment_key"] = core_paths.deployment_key()
    except Exception as exc:                           # noqa: BLE001 - PathsError, RuntimeDirectoryError
        diagnostics["error"] = str(exc)
        return None, diagnostics
    try:
        snapshot = runtime_records.read_runtime(directory)
    except Exception as exc:                           # noqa: BLE001 - never take the server down
        diagnostics["error"] = f"could not read the runtime registry: {exc}"
        log.debug("runtime registry read failed", exc_info=True)
        return None, diagnostics
    diagnostics["readable"] = True
    return snapshot, diagnostics


def runtime_snapshot(*, refresh: bool = False) -> tuple:
    """``(RuntimeSnapshot | None, diagnostics)`` for this deployment, TTL-cached.

    The snapshot's ``active`` flag comes from the OS lock and never from the JSON, so a held lease
    whose record is missing, corrupt or from a newer build still reads as occupied (section 2.9).
    """
    global _RUNTIME_CACHE
    now = time.monotonic()
    with _RUNTIME_CACHE_LOCK:
        cached = _RUNTIME_CACHE
        if not refresh and cached is not None and cached[0] > now:
            return cached[1], cached[2]
    snapshot, diagnostics = _read_registry()
    with _RUNTIME_CACHE_LOCK:
        _RUNTIME_CACHE = (time.monotonic() + RUNTIME_CACHE_TTL_S, snapshot, diagnostics)
    return snapshot, diagnostics


def _read_last_record() -> Optional[Any]:
    """The most recent terminal ROOT record (``last.json``), sanitized, or ``None``.

    Read with the same public helpers ``RecordStore.read_last`` uses rather than by constructing a
    store: a store is scoped to a writer identity, and this process is never a writer of either
    file (section 3.2's writer-ownership table).
    """
    try:
        from leafmachine3.core.paths import LAST_RECORD_FILENAME
        from leafmachine3.core.runtime.records import (
            read_json_file,
            record_from_dict,
            sanitize_record,
        )
    except Exception:                                  # noqa: BLE001
        return None
    try:
        path = deployment_dir() / LAST_RECORD_FILENAME
        payload, error = read_json_file(path)
        if payload is None or error is not None:
            return None
        return sanitize_record(record_from_dict(payload, path=path))
    except Exception:                                  # noqa: BLE001 - malformed / newer schema
        log.debug("could not read the last runtime record", exc_info=True)
        return None


def _record_payload(record: Any) -> Optional[dict]:
    try:
        from leafmachine3.core.runtime.records import record_to_dict

        return dict(record_to_dict(record))
    except Exception:                                  # noqa: BLE001
        log.debug("could not serialize a runtime record", exc_info=True)
        return None


def _child_view(summary: Any, children: tuple) -> Optional[dict]:
    """One BOUNDED child entry: section 3.2's six-field summary, merged with that child's record.

    Bounded is the requirement, not an optimization. ``active.json`` carries at most a
    ``current_child`` and a ``last_child`` however many children a root launches over its life, and
    the runtime view must inherit that property -- "the bounded activity tree ... never a growing
    list" -- so this returns one object and the caller calls it at most twice.
    """
    if summary is None:
        return None
    view = {
        "run_id": summary.run_id,
        "activity": summary.activity.value,
        "state": summary.state.value,
        "started_at": summary.started_at,
        "run_name": summary.run_name,
        "run_dir": summary.run_dir,
        # Filled from the child's own record below when it is readable. The summary is the ROOT's
        # writing and can lag: section 3.2 has the root publish ``current_child`` BEFORE the launch,
        # after which the child alone writes its own file.
        "updated_at": None, "finished_at": None, "returncode": None, "error": None,
        "record_available": False,
    }
    for child in children:
        if child.run_id != summary.run_id:
            continue
        view.update({
            "state": child.state.value,
            "updated_at": child.updated_at,
            "finished_at": child.finished_at,
            "returncode": child.returncode,
            "error": child.error,
            "record_available": True,
        })
        break
    return view


def _can_stop(run_id: Optional[str], *, compatible: bool) -> tuple:
    """``(can_stop, reason)`` -- section 2.5 as amended by plan revision 15, and nothing else.

    The whole test is: do we hold a retained LIVE child handle, for THIS ``run_id``, launched by
    THIS server instance? There is deliberately no PID comparison and no process-creation-time
    comparison. Revision 15 calls both "redundant and actively hazardous": holding the ``Popen``
    makes us the child's parent, so its PID cannot be recycled before we reap it, while
    ``process_start_time()`` returns ``0.0`` without ``psutil`` and a ``0.0 == 0.0`` match reads as
    agreement on two of three platforms.

    ``reason`` is written for a person to read in the GUI, because "you cannot stop this" without a
    reason is the failure mode section 2.6 is trying to remove.
    """
    if not compatible:
        return False, ("this runtime was created by a newer LM3; control actions are disabled "
                       "until you upgrade (plan section 2.9)")
    with _RUN_LOCK:
        run = _RUN
    if run is None:
        return False, ("this server did not launch this run and holds no handle for it -- it is "
                       "observing. Stop it from the terminal or window that started it.")
    if run.state not in ("running", "stopping"):
        return False, "the run this server launched has already finished"
    handle = run.child
    if handle is None:
        # The pre-flag path: the retained ``Popen`` IS the handle, and there is no runtime record
        # to match a run_id against, so ownership is the whole answer.
        if run.proc is None:
            return False, "this server holds no handle for the active run"
        return (run.proc.poll() is None), (
            None if run.proc.poll() is None else "the run this server launched has already exited")
    if handle.instance_id != SERVER_INSTANCE_ID:
        return False, "the handle for this run belongs to a different server instance"
    if run_id and handle.run_id and handle.run_id != run_id:
        return False, (f"this server holds a handle for run {handle.run_id}, not for the run that "
                       f"currently holds the deployment ({run_id})")
    if not handle.alive:
        return False, "the run this server launched has already exited"
    return True, None


def active_run_ref() -> Optional[dict]:
    """THE registry-backed answer to "which project ledger is live?", or ``None``.

    This is the replacement for the private ``bind_run`` / ``_BOUND`` binding: progress, results and
    postprocessing all need one project reference (section 5, "one project-reference shape shared by
    runtime, progress, results, and postprocessing"), and taking it from the registry is what makes
    a CLI-started run visible to the GUI without the server having launched anything.

    ``None`` for an idle deployment, for an incompatible record (section 2.9 forbids interpreting
    it), and -- deliberately -- for a ``hardware_setup`` root, which carries no ``project`` block at
    all: section 2.6 says a tuning activity must show a Machine-panel state and must NOT move
    project history to ``_lm3_calibration``.
    """
    if not runtime_v2():
        return None
    snapshot, _diag = runtime_snapshot()
    if snapshot is None or not snapshot.compatible:
        return None
    record = snapshot.record
    if record is None or record.project is None:
        return None
    project = record.project
    return {
        "run_id": record.run_id,
        "activity": record.activity.value,
        "state": record.state.value,
        "run_name": project.run_name,
        "run_dir": project.run_dir,
        "artifact_dir": project.artifact_dir,
        "active_db_path": project.active_db_path,
        "log_path": project.log_path,
        "source": "runtime",
    }


#: ``str(settings path) -> ((st_mtime, st_size), Config | None, error | None)``. One entry, replaced
#: in place: only the canonical settings file is ever read here, so this cannot grow.
_SETTINGS_CACHE: dict = {}
_SETTINGS_CACHE_LOCK = threading.Lock()


def _load_next_run_config(path: Path) -> tuple:
    """``(Config | None, error message | None)`` for ``path``, re-parsing only when it changes.

    ``Config.load`` is a ``yaml.safe_load`` of an ~9 KB document through PyYAML's pure-Python
    loader: measured at ~17 ms, which was essentially the entire cost of ``GET /v1/runtime``.
    :data:`RUNTIME_CACHE_TTL_S` does not cover it -- that TTL bounds the registry read and the lock
    probe, and says so; the settings load sits outside it in :func:`runtime_view`.

    Keyed on the FILE (mtime AND size, so a same-timestamp rewrite of a different length is still
    noticed), never on a clock, because section 2.6 promises "editing YAML affects only the next
    run" and the corollary is that an edit must show up on the very next read -- including one made
    by ``PUT /v1/settings`` a millisecond ago. Same shape as ``progress_api._read_yaml``, which
    solved this for the 2 Hz status frame.

    Only the PARSE is memoized. ``resolve_project_output_dir`` stays live in the caller because it
    reads the environment as well as the file, and a cache keyed on the file alone must never pin an
    answer that the environment has since moved.
    """
    try:
        stat = path.stat()
    except OSError as exc:                             # vanished between resolution and read
        return None, str(exc)
    key = (stat.st_mtime, stat.st_size)
    with _SETTINGS_CACHE_LOCK:
        cached = _SETTINGS_CACHE.get(str(path))
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    error: Optional[str] = None
    cfg = None
    try:
        from leafmachine3.core.config import Config

        cfg = Config.load(path)
    except Exception as exc:                           # noqa: BLE001 - a bad YAML is reportable
        cfg, error = None, str(exc)
    with _SETTINGS_CACHE_LOCK:
        # A failed parse is cached too, and for the same reason: a half-written YAML under a polling
        # GUI would otherwise pay the full parse-and-raise on every frame. The mtime key is what
        # makes that safe -- fixing the file changes the key.
        _SETTINGS_CACHE.clear()
        _SETTINGS_CACHE[str(path)] = (key, cfg, error)
    return cfg, error


def next_run_settings() -> dict:
    """What the NEXT run would use -- never a description of the run in progress.

    Section 2.6: "Editing YAML affects only the next run", and section 3.4 pins a running job to its
    launch manifest. The GUI has no way to say that today, which is why the field is named for what
    it is and carries ``applies_to`` in the payload rather than leaving the label to a renderer.
    """
    out: dict = {"applies_to": "next run", "settings_path": None, "exists": False,
                 "run_name": None, "output_dir": None, "run_dir": None, "input_dirs": [],
                 "error": None}
    path = default_config_path()
    if path is None:
        out["error"] = f"no {CFG_FILENAME} resolves to a readable file"
        return out
    out["settings_path"] = str(path)
    out["exists"] = True
    cfg, error = _load_next_run_config(path)
    if cfg is None:
        out["error"] = error
        return out
    try:
        run_name = str(cfg.project.run_name)
        out["run_name"] = run_name
        out_dir = resolve_project_output_dir(path, str(cfg.project.output.dir))
        if out_dir is not None:
            out["output_dir"] = str(out_dir)
            out["run_dir"] = str(out_dir / run_name)
        out["input_dirs"] = [str(d) for d in (cfg.project.input.dirs or [])]
    except Exception as exc:                           # noqa: BLE001 - an unresolvable path is
        out["error"] = str(exc)                        # reportable, never a 500 on an observer
    return out


def _server_view() -> dict:
    """Identity and capability of THIS server process. No token, ever (section 3.2 / gate 13)."""
    try:
        from leafmachine3.core.runtime._types import SCHEMA_VERSION
    except Exception:                                  # noqa: BLE001
        SCHEMA_VERSION = None                          # noqa: N806 - mirrors the imported name
    return {
        "instance_id": SERVER_INSTANCE_ID,
        "protocol_version": RUNTIME_PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
        "runtime_v2": runtime_v2(),
        "pid": os.getpid(),
        # Diagnostic only. Section 2.5: a PID never authorizes control, here or on /healthz.
        "pid_is_diagnostic_only": True,
    }


def _handle_view() -> Optional[dict]:
    """What this server is holding, if anything -- the visible half of "control needs a handle"."""
    with _RUN_LOCK:
        run = _RUN
    if run is None:
        return None
    handle = run.child
    return {
        "kind": handle.kind if handle is not None else "pipeline",
        "run_id": (handle.run_id if handle is not None else run.run_id) or None,
        "instance_id": handle.instance_id if handle is not None else SERVER_INSTANCE_ID,
        "pid": run.pid or None,
        "alive": run.alive,
        "state": run.state,
    }


def runtime_view() -> dict:
    """``GET /v1/runtime`` -- the canonical runtime, bounded, with last-run and next-run settings.

    Shape is section 4 Step 4 verbatim: ``{active, last, next_run_settings, server, diagnostics}``.

    ``active`` is ``None`` only when the deployment lock is FREE. When it is held the object is
    always present, even if the record is missing, unreadable or from a newer build -- section 2.9:
    "the OS lock still decides whether the deployment is occupied". In the incompatible case
    ``record`` is ``null`` (nothing unknown is interpreted), ``compatible`` is false and
    ``can_stop`` is false, which is what tells the GUI to disable Start and every control action.
    """
    snapshot, diagnostics = runtime_snapshot()
    diagnostics = dict(diagnostics)
    diagnostics["handle"] = _handle_view()
    active_view: Optional[dict] = None

    if snapshot is not None:
        diagnostics["lease_occupied"] = bool(snapshot.active)
        diagnostics["classification"] = snapshot.classification.value
        diagnostics["record_path"] = str(snapshot.path) if snapshot.path else None
        if snapshot.message:
            diagnostics["message"] = snapshot.message
        record = snapshot.record
        if snapshot.active or record is not None:
            can_stop, reason = _can_stop(record.run_id if record is not None else None,
                                         compatible=snapshot.compatible)
            active_view = {
                "occupied": bool(snapshot.active),
                "compatible": bool(snapshot.compatible),
                "classification": snapshot.classification.value,
                "schema_version": snapshot.schema_version,
                "message": snapshot.message,
                "run_id": record.run_id if record is not None else None,
                "activity": record.activity.value if record is not None else None,
                "state": record.state.value if record is not None else "unknown",
                # Never a partially-interpreted payload: either the record parsed against THIS
                # build's schema or the field is null (section 2.9, "do not interpret unknown
                # fields"). The raw bytes stay on disk; ``record_path`` says where.
                "record": _record_payload(record) if record is not None else None,
                "current_child": _child_view(
                    record.current_child if record is not None else None,
                    snapshot.children),
                "last_child": _child_view(
                    record.last_child if record is not None else None, snapshot.children),
                "can_stop": can_stop,
                "can_stop_reason": reason,
            }
    else:
        diagnostics.setdefault("lease_occupied", None)
        diagnostics.setdefault("classification", None)

    last_record = _read_last_record()
    return {
        "active": active_view,
        "last": _record_payload(last_record) if last_record is not None else None,
        "next_run_settings": next_run_settings(),
        "server": _server_view(),
        "diagnostics": diagnostics,
    }


def _registry_projection() -> Optional[dict]:
    """The active runtime record, projected into ``GET /v1/run/active``'s legacy key set.

    This is what "keep ``/v1/run/active`` as a compatibility projection" (section 5) means for a run
    this server did not launch: an old client polling the legacy route sees the CLI run rather than
    "idle", with the SAME key set -- ``metrics_api._Run.record``'s docstring promises that never
    changes, and section 4 Step 7 owns the route's removal.

    ``None`` when there is nothing to project, when the record is from a newer build (section 2.9
    forbids interpreting it, and inventing legacy fields out of it would be exactly that), and when
    the lock says the deployment is FREE however loudly the JSON says "running".
    """
    snapshot, _diag = runtime_snapshot()
    if snapshot is None or not snapshot.compatible:
        return None
    try:
        from leafmachine3.core.runtime import RecordClassification
    except Exception:                                  # noqa: BLE001 - a base install must still
        return None                                    # answer the legacy route, just with "idle"
    # Section 2.9: "the OS lock still decides whether the deployment is occupied" -- never the JSON.
    # ABANDONED is precisely where the two disagree: the writer was SIGKILLed or OOM-killed, the
    # kernel dropped the lease, and ``active.json`` still says "running" because nothing rewrites it
    # until the next acquisition -- an unbounded window (section 3.3, "lock released without
    # finalization -> abandoned"). Projecting that as ``active: true`` wedges the top bar for the
    # whole window (Start stays disabled, Stop answers 409 because ``_refuse_observer_stop`` sees a
    # free lock), masks THIS server's own terminal record inside ``active()``, and publishes a
    # ``pid`` beside it -- the field section 3.3 says a reader "must never signal". The record stays
    # fully describable on ``GET /v1/runtime`` (``occupied: false, classification: "abandoned"``);
    # only the legacy ``active: true`` claim is withdrawn. ``progress_api._from_runtime`` and
    # ``results_api.runtime_view`` already gate on LIVE; this was the last reader that did not.
    if snapshot.classification is not RecordClassification.LIVE:
        return None
    record = snapshot.record
    # LIVE already implies a non-terminal state (``records.classify_record``), so this is redundant
    # today. It stays because it is the check that names the legacy vocabulary this projection is
    # allowed to emit, and a future state added to the lifecycle must be admitted deliberately.
    if record is None or record.state.value not in ("starting", "running", "stopping"):
        return None
    out = _idle_record()
    project = record.project
    # The legacy record's ``started_at`` is a unix epoch float; the registry's is section 3.2's
    # ISO-8601-with-Z. Parsed as UTC explicitly -- ``time.mktime`` would read it as local time and
    # be wrong by the offset, and wrong twice a year by the DST rule on top of that.
    started: Optional[float] = None
    try:
        started = (datetime.strptime(record.started_at, "%Y-%m-%dT%H:%M:%SZ")
                   .replace(tzinfo=timezone.utc).timestamp())
    except (TypeError, ValueError):                    # a record from a build with another format
        started = None
    out.update({
        "active": True,
        # "starting" is section 3.3's own first lifecycle state and the legacy route has always
        # been able to say it (see _launching_record), so no new vocabulary reaches old clients.
        "state": "running" if record.state.value == "running" else record.state.value,
        "run_name": project.run_name if project is not None else None,
        "pid": record.pid or None,
        "started_at": started,
        "started_iso": record.started_at,
        "elapsed_s": round(time.time() - started, 1) if started else None,
        "config_path": record.config.path if record.config is not None else None,
        "run_dir": project.run_dir if project is not None else None,
        "output_dir": (str(Path(project.run_dir).parent)
                       if project is not None and project.run_dir else None),
        "db_path": project.active_db_path if project is not None else None,
        "log_path": project.log_path if project is not None else None,
        "input_dirs": list(project.input_dirs) if project is not None else [],
    })
    return out


def active() -> dict:
    """The most recent run record -- ``active`` says whether it is still going.

    The key set never changes: before the first run every field is null and ``state`` is "idle";
    after a run ends the record STAYS (with ``active: false``) so the app can reveal the results of
    the run that just finished without asking a second question.
    """
    run = _current()
    if run is not None and run.state in ("running", "stopping"):
        return run.record()
    # Ordered so the published record wins the moment it exists: the launch sentinel is cleared
    # just AFTER ``_RUN`` is installed, and a reader landing in that overlap must see the run, not
    # a "starting" record for a launch that has already finished starting.
    launching = _launching_record()
    if launching is not None:
        return launching
    if runtime_v2():
        # Section 5: this route becomes a compatibility PROJECTION. A run started from the CLI (or
        # by any other root) holds the deployment and is the honest answer to "what is active?",
        # even though this server launched nothing -- and it is what makes step 6's exit gate
        # ("opening the GUI during any CLI run immediately shows that run") reachable at all.
        # It outranks our own FINISHED record for the same reason: a live run beats a dead one.
        projected = _registry_projection()
        if projected is not None:
            return projected
    if run is not None:
        return run.record()
    return _idle_record()


def is_active() -> bool:
    """Is ANY root activity holding this deployment -- ours or somebody else's?

    Under the flag the lock is the authority (section 2.9: "the OS lock still decides whether the
    deployment is occupied"), so this is true for a CLI run too. That is the point: the question
    callers ask it is "may I start?", and the answer must not depend on who launched the incumbent.
    """
    if _launching_record() is not None:                # admitted, still handshaking: not idle
        return True
    run = _current()
    if run is not None and run.state in ("running", "stopping") and run.alive:
        return True
    if runtime_v2():
        snapshot, _diag = runtime_snapshot()
        if snapshot is not None and snapshot.active:
            return True
    return False


#: Section 2.10 STORAGE ROLE -> the ``_Run`` attribute holding the same path, for the roles whose
#: legacy attribute is spelled differently. ``db_path`` is the deprecated projection of
#: ``active_db_path`` and this mapping is the last place inside this module that still spells it:
#: every reader below asks for the ROLE, and the legacy attribute survives only because
#: ``_Run.record()``'s key set is frozen until :data:`COMPAT_REMOVAL_RELEASE` retires the route.
#: Roles not listed here use their own name as the attribute name.
_RUN_ATTR_FOR_ROLE = {"active_db_path": "db_path"}


def _path_or_none(role: str) -> Optional[Path]:
    """One of the active run's section 2.10 storage roles, registry first, private record second.

    The registry describes whichever root holds the deployment; ``_RUN`` describes only what this
    process launched. Preferring the registry is what stops the status stream, the log tail and the
    console from sitting on the last run THIS server started while a different one is going.

    ``role`` is a storage-role name (``active_db_path``, ``log_path``, ``run_dir``), which is also
    the key :func:`active_run_ref` publishes. ``console_log`` is deliberately NOT one: it is the
    SERVER's request log, so a run this server did not launch simply has none (section 2.4 puts it
    under the jobs root, not the run dir) and it falls through to the private record every time.
    """
    ref = active_run_ref()
    if ref is not None and ref.get(role):
        return Path(str(ref[role]))
    run = _current()
    if run is None:
        return None
    raw = getattr(run, _RUN_ATTR_FOR_ROLE.get(role, role), "")
    return Path(raw) if raw else None


def active_db_path() -> Optional[Path]:
    """The project SQLite of the current/last run (what the progress module tails).

    Named for the section 2.10 role, not for the deprecated ``db_path`` projection, and it resolves
    through the role too -- so the day :data:`COMPAT_REMOVAL_RELEASE` deletes the projection, this
    function does not move.
    """
    return _path_or_none("active_db_path")


def active_log_path() -> Optional[Path]:
    """``<run>/logs/lm3.log`` -- LM3's own log file, the reliable live log source."""
    return _path_or_none("log_path")


def active_console_path() -> Optional[Path]:
    """The captured stdout+stderr. Holds tracebacks that die BEFORE logging is configured."""
    return _path_or_none("console_log")


def _reap(run: _Run) -> None:
    """Wait for the run to end, then record how it ended and hand the workers table back."""
    # ``wait()`` on the handle, never a PID poll. Every ``_Run`` is a child of this process, so
    # this branch is the only one there is (Step 4 removed the adopted-run polling branch along
    # with ``_pid_alive``).
    if run.proc is not None:
        try:
            run.returncode = run.proc.wait()
        except Exception:                              # noqa: BLE001
            run.returncode = None

    with _RUN_LOCK:
        run.finished_at = time.time()
        if run.stopped_by_user:
            run.state = "done"
        elif run.returncode in (0, None):
            run.state = "done"
        else:
            run.state = "error"
            run.error = f"machine3 exited with code {run.returncode}"
        if run.log_fh is not None:
            try:                                       # the child is gone, so appending is safe
                run.log_fh.write(
                    f"\n[LM3 app] run finished at {_iso(run.finished_at)} "
                    f"(exit code {run.returncode}, "
                    f"{round(run.finished_at - run.started_at, 1)} s)\n")
                run.log_fh.flush()
                run.log_fh.close()
            except Exception:                          # noqa: BLE001
                pass
            run.log_fh = None
    if run.child is not None:
        # KILL_ON_JOB_CLOSE: on Windows this is also what reaps anything still in the job after the
        # root exited. On POSIX there is no job and this is a no-op.
        run.child.release_job()
    metrics.set_worker_root(os.getpid())               # stop scanning a pid that no longer exists
    invalidate_runtime_cache()                         # the lease is free; do not serve it as held
    log.info("LM3 run %s finished: state=%s rc=%s", run.run_name, run.state, run.returncode)


def _machine3_argv() -> list[str]:
    """The command that launches LM3.

    Prefer the console script from THIS venv (``<venv>/bin/machine3``) so the run uses the same
    interpreter and the same installed leafmachine3 as the server. ``python -m
    leafmachine3.machine3`` is the fallback; both route through ``machine3.main()``, which is the
    part that matters -- see this module's docstring for why an in-process call is wrong.
    """
    override = os.environ.get("LM3_MACHINE3_BIN")
    if override:
        return [override]
    exe = Path(sys.executable).with_name("machine3")
    if exe.is_file() and os.access(exe, os.X_OK):
        return [str(exe)]
    found = shutil.which("machine3")
    if found:
        return [found]
    return [sys.executable, "-m", "leafmachine3.machine3"]


def _normalize_restart(restart: Any) -> list[str]:
    """``None`` / ``""`` -> resume; ``"all"`` -> full rebuild; a key or list of keys -> those."""
    if restart is None or restart == "" or restart is False:
        return []
    if isinstance(restart, str):
        return ["all"] if restart.strip().lower() == "all" else [restart.strip()]
    if isinstance(restart, (list, tuple)):
        keys = [str(k).strip() for k in restart if str(k).strip()]
        return ["all"] if any(k.lower() == "all" for k in keys) else keys
    raise RunError(f"restart must be a stage key, a list of keys, or 'all' (got {restart!r})")


def start_run(config_path: Optional[str] = None, input_dir: Optional[str] = None,
              output_dir: Optional[str] = None, restart: Any = None) -> dict:
    """Launch LM3 as a subprocess and return its run record.

    ``input_dir`` / ``output_dir`` map to ``machine3 --input/--output`` (the same overrides the CLI
    takes). Everything else -- run name, temp-file location, module toggles -- lives in the YAML;
    save it through the settings module first, then start.
    """
    global _RUN, _LAUNCHING
    from leafmachine3.core.config import CANONICAL_STAGE_KEYS, Config

    # ``_RUN_LOCK`` covers the admission decision and the config resolution -- both cheap -- and,
    # on the runtime-v2 path, is RELEASED before the section 2.4 handshake. See ``_LAUNCHING``.
    with _RUN_LOCK:
        current = _current()
        if current is not None and current.state in ("running", "stopping"):
            if current.alive:
                raise RunError(
                    f"a run is already active (pid {current.pid}, run '{current.run_name}') -- "
                    "stop it before starting another", status=409)
            _reap_stale(current)
        if _LAUNCHING is not None:
            # Two starts racing: the first one is past admission and inside its handshake. The
            # loser gets the same 409 it would have got had the winner finished, and -- this is the
            # point -- it gets it from a lock nobody is holding for 30 s.
            raise RunError("a run is already being launched (run "
                           f"'{_LAUNCHING.get('run_name')}') -- wait for it to answer", status=409)

        if config_path:
            cfg_path = Path(str(config_path)).expanduser()
            if not cfg_path.is_absolute():
                # Precedence row 1 is "explicit argument", not "explicit argument joined onto
                # whatever directory the server was launched from" (section 3.1: no path falls
                # back to the CWD). Refuse rather than guess.
                raise RunError(f"config_path must be absolute; got {config_path!r}", status=400)
        else:
            cfg_path = default_config_path()
        if cfg_path is None:
            raise RunError(f"no {CFG_FILENAME} found -- pass config_path", status=400)
        cfg_path = cfg_path.resolve()
        if not cfg_path.is_file():
            raise RunError(f"config file not found: {cfg_path}", status=400)

        keys = _normalize_restart(restart)
        bad = [k for k in keys if k != "all" and k not in CANONICAL_STAGE_KEYS]
        if bad:
            raise RunError(f"unknown restart stage key(s): {bad}; valid keys are "
                           f"{list(CANONICAL_STAGE_KEYS)} or 'all'", status=400)

        # Resolve the run exactly the way machine3 will, so the record points at the real files
        # before the process has created any of them. Same override tree as machine3._cli_overrides.
        overrides: dict = {"project": {}}
        if input_dir:
            overrides["project"]["input"] = {"dirs": [str(input_dir)]}
        if output_dir:
            overrides["project"]["output"] = {"dir": str(output_dir)}
        if keys:
            overrides["project"]["run_mode"] = {"restart": "all" if keys == ["all"] else keys}
        try:
            cfg = Config.load(cfg_path, overrides=overrides)
            cfg.validate()                             # fail here with a 400, not in the child
        except (ValueError, FileNotFoundError) as exc:
            raise RunError(str(exc), status=400) from exc

        # The child runs in the CONFIG's own directory, always. The old election ("the first of
        # [config dir, server CWD, checkout root] that contains hardware_settings.yaml") existed
        # only because hardware_setup.HW_PATH was CWD-relative; now that the profile is
        # deployment-scoped (section 3.1 row 2) that probe can only ever fall through, and a run
        # whose relative model paths resolved against the server's launch directory is precisely
        # the GUI/CLI disagreement this step removes. The child inherits LM3_DEPLOYMENT_ID and
        # LM3_RUNTIME_DIR through ``env`` below, so it resolves the same deployment we did.
        cwd = cfg_path.parent

        run_name = str(cfg.project.run_name)
        # Relative output dirs hang off the settings FILE, which is also how progress_api's
        # _from_settings and core.paths.resolve_project_output_dir read them -- one rule, so the
        # status stream describes the directory the child actually writes into.
        out_dir = resolve_project_output_dir(cfg_path, str(cfg.project.output.dir))
        if out_dir is None:                            # settings with no output dir at all
            raise RunError("project.output.dir is empty in the settings file", status=400)
        run_dir = out_dir / run_name
        logs_dir = run_dir / "logs"
        tmp_cfg = str(getattr(cfg.project.output, "tmp_dir", "auto") or "auto")
        tmp_dir = (run_dir / "_tmp_original" if tmp_cfg.lower() == "auto"
                   else Path(tmp_cfg).expanduser() / run_name / "_tmp_original")
        argv = _machine3_argv() + ["--config", str(cfg_path)]
        if input_dir:
            argv += ["--input", str(input_dir)]
        if output_dir:
            argv += ["--output", str(output_dir)]
        for key in keys:
            argv += ["--restart", key]

        if runtime_v2():
            # Section 2.4: <run_dir>/logs and its console.log are EXECUTION-owned, so this path
            # creates neither. The child creates them once it holds the lease, which is what makes
            # "the busy loser creates no directories" true rather than merely intended.
            #
            # The lock ENDS with this assignment. Everything section 2.4 asks for next -- Popen, the
            # bounded read, the terminate/escalate ladder, the record reconciliation -- happens with
            # ``_RUN_LOCK`` free, because it is up to ~45 s of work and ``_current()`` (so
            # ``GET /v1/run/active``, ``GET /v1/run/console``, the SSE stream and ``POST
            # /v1/run/stop``) takes the same lock.
            _LAUNCHING = {"run_name": run_name, "config_path": str(cfg_path), "cwd": str(cwd),
                          "output_dir": str(out_dir), "run_dir": str(run_dir),
                          "argv": list(argv), "started_at": time.time()}
        else:
            try:
                logs_dir.mkdir(parents=True, exist_ok=True)   # build_dirs is exist_ok, so this is safe
            except OSError as exc:
                raise RunError(f"cannot create the run log directory {logs_dir}: {exc}", status=400) from exc

            console = logs_dir / "console.log"
            try:
                fh = console.open("a", encoding="utf-8", errors="replace")
                fh.write(f"\n[LM3 app] {_iso(time.time())} launching: {' '.join(argv)}\n"
                         f"[LM3 app] cwd={cwd}\n")
                fh.flush()
            except OSError as exc:
                raise RunError(f"cannot open the console log {console}: {exc}", status=400) from exc

            env = os.environ.copy()
            # MUST NOT be inherited: machine3.main() skips the LD_LIBRARY_PATH re-exec when it is set,
            # which is exactly the failure mode this whole subprocess dance exists to avoid.
            env.pop("LM3_CUDA_LIBPATH_SET", None)
            env.pop("LM3_SERVER_TOKEN", None)              # the run has no business holding the secret
            env["PYTHONUNBUFFERED"] = "1"                  # so console.log tails live, not in 4 KB blocks

            try:
                proc = subprocess.Popen(
                    argv, cwd=str(cwd), env=env,
                    stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    close_fds=True,
                    # Own session/process group: one killpg then reaches the executor's spawn workers
                    # too, and the run survives a server restart instead of dying with it.
                    start_new_session=True,
                )
            except OSError as exc:
                fh.close()
                raise RunError(f"could not launch {argv[0]}: {exc}", status=500) from exc

            run = _Run(run_name=run_name, config_path=str(cfg_path), cwd=str(cwd),
                       output_dir=str(out_dir), run_dir=str(run_dir),
                       db_path=str(run_dir / f"{run_name}.sqlite"),
                       log_path=str(logs_dir / "lm3.log"), console_log=str(console),
                       tmp_dir=str(tmp_dir),
                       input_dirs=[str(d) for d in (cfg.project.input.dirs or [])],
                       restart=keys, argv=argv)
            run.proc = proc
            run.log_fh = fh
            run.pid = proc.pid
            try:
                run.pgid = os.getpgid(proc.pid)
            except OSError:
                run.pgid = proc.pid
            # No ``psutil`` creation-time probe any more: it existed only to guard a PID read back
            # out of ``active_run.json``, and section 2.5 (revision 15) calls that check "redundant
            # and actively hazardous" -- ``process_start_time`` returns 0.0 without psutil, so the
            # comparison silently degrades to "always matches" on two of three platforms.

            _RUN = run
            _bind_status(run)
            # Every ram_delta_mb / vram_delta_mb from here reads as "what THIS run added", and the
            # worker table follows the run's spawn children instead of the server's (metrics gotcha #1).
            metrics.reset_baseline()
            metrics.set_worker_root(proc.pid)
            threading.Thread(target=_reap, args=(run,), name="lm3-run-reaper", daemon=True).start()
            log.info("started LM3 run '%s' pid=%s cwd=%s", run_name, proc.pid, cwd)
            return run.record()

    # -- runtime v2 only, and deliberately OUTSIDE the lock ------------------------------------ #
    try:
        return _start_run_v2(
            cfg_path=cfg_path, cwd=cwd, run_name=run_name, out_dir=out_dir, run_dir=run_dir,
            logs_dir=logs_dir, tmp_dir=tmp_dir, argv=argv, keys=keys,
            input_dirs=[str(d) for d in (cfg.project.input.dirs or [])])
    finally:
        # EVERY path, including BaseException: a sentinel left standing would refuse every later
        # start with a 409 about a launch that is over. ``_start_run_v2`` publishes ``_RUN`` before
        # it returns, so on the success path the record is already visible when this clears.
        with _RUN_LOCK:
            _LAUNCHING = None


def _launch_dir(run_name: str) -> Path:
    """A per-launch directory under the SERVER-PRIVATE jobs root (section 2.4, permitted staging).

    Namespaced per launch id rather than per run name: two launches of the same project must not
    append to one another's request log, and a launch that LOSES must still have somewhere of its
    own to have written to.
    """
    from leafmachine3.server.app import server_jobs_root

    launch_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    directory = server_jobs_root(create=True) / LAUNCHES_DIRNAME / launch_id
    directory.mkdir(parents=True, exist_ok=True)
    try:
        (directory / "launch.json").write_text(
            json.dumps({"launch_id": launch_id, "run_name": run_name, "t": time.time()}),
            encoding="utf-8")
    except OSError:                                    # diagnostics only
        log.debug("could not write %s/launch.json", directory, exc_info=True)
    return directory


def _reconcile_after_failed_handshake(run_name: str) -> Optional[str]:
    """Section 2.4 step 3 (iii): "reconciles any runtime record the child may already have published".

    A child that acquired the lease and then failed to answer may already have written
    ``active.json``. Once it is gone that record describes nothing, and leaving it there makes the
    deployment look permanently occupied to every later reader.

    The reconciliation runs under ``lease.cleanup_lease`` -- which is not a second lock but the same
    activity lease, taken for the sole purpose of classifying (section 3.3). Acquiring it is proof
    the child is really gone; failing to acquire it is proof something is still holding the
    deployment, and then there is nothing abandoned to clean up and we say so. The server never
    becomes an execution lease holder this way (invariant 13): it holds the lock only inside this
    call, publishes no record of its own, and signals no PID out of any record.
    """
    try:
        from leafmachine3.core.runtime.lease import cleanup_lease
        from leafmachine3.core.runtime.records import recover_abandoned
    except Exception:                                  # noqa: BLE001
        return None
    try:
        with cleanup_lease() as lease:
            if lease is None:
                return ("the deployment is still occupied -- another root holds the lease, so "
                        "nothing was reconciled")
            recovered = recover_abandoned(Path(lease.deployment_dir), lease=lease)
    except Exception as exc:                           # noqa: BLE001 - never mask the launch failure
        log.warning("could not reconcile the runtime record after a failed handshake for %r: %s",
                    run_name, exc, exc_info=True)
        return f"record reconciliation failed: {exc}"
    if recovered is None:
        return None
    log.info("reconciled an abandoned runtime record (run_id=%s) after a failed handshake",
             recovered.run_id)
    return f"an abandoned record for run_id {recovered.run_id} was finalized"


def _start_run_v2(*, cfg_path: Path, cwd: Path, run_name: str, out_dir: Path, run_dir: Path,
                  logs_dir: Path, tmp_dir: Path, argv: list, keys: list,
                  input_dirs: list) -> dict:
    """``POST /v1/run/start`` with the section 2.4 launch handshake.

    The caller does NOT hold ``_RUN_LOCK`` -- it holds the ``_LAUNCHING`` sentinel instead, which is
    what keeps a second start out while leaving ``_current()`` (and therefore every status route and
    the SSE stream) answerable throughout a read that is bounded at 30 s and a failure branch that
    adds a 10 s SIGTERM grace and a 5 s SIGKILL wait on top. This function takes the lock only to
    publish ``_RUN``.

    The shape of the whole thing, in order, because each step exists to make the next one honest:

    1. A server-private launch directory and request log -- the ONLY thing written before the child
       answers, and it is inside the jobs root, which section 2.4 permits even to a losing launch.
    2. A dedicated status pipe, composed with any other launch contribution through
       ``compose_launch``: ``pass_fds`` and the Windows handle allowlist are exhaustive, so a
       second, independently-passed one would silently drop the first.
    3. ``Popen`` with stdout/stderr on the request log from the very first byte -- never on a pipe
       the server might not drain.
    4. ONE bounded read. ``acquired`` -> 200 with the identity; ``busy`` -> 409 with the winner;
       anything else -> terminate the retained tree, escalate, reconcile, and only then 500.
    5. No execution-owned directory is created here at ALL. The child creates them, after it has
       acquired -- which is what makes "the busy loser creates no directories" true rather than
       merely intended.
    """
    global _RUN

    prohibited = execution_owned_paths(run_dir, run_name)
    pre_existing = _staging_snapshot(prohibited)

    launch_dir = _launch_dir(run_name)
    console = launch_dir / "console.log"
    try:
        fh = console.open("a", encoding="utf-8", errors="replace")
        fh.write(f"\n[LM3 app] {_iso(time.time())} launching: {' '.join(argv)}\n"
                 f"[LM3 app] cwd={cwd}\n")
        fh.flush()
    except OSError as exc:
        raise RunError(f"cannot open the launch request log {console}: {exc}", status=500) from exc

    # A FILTERED copy, never os.environ wholesale: a raw copy hands the child our own
    # LM3_STATUS_FD -- a descriptor NUMBER that means something else over there -- and any lease
    # variables that must come from a handoff or not at all.
    try:
        from leafmachine3.core.runtime.execution import child_base_env
        from leafmachine3.core.runtime.launch import composed_launch
    except Exception as exc:                           # noqa: BLE001
        fh.close()
        raise RunError(f"the unified runtime is unavailable: {exc}", status=500) from exc

    env = child_base_env(extra={"PYTHONUNBUFFERED": "1"})
    # MUST NOT be inherited: machine3.main() skips the LD_LIBRARY_PATH re-exec when it is set,
    # which is exactly the failure mode this whole subprocess dance exists to avoid.
    env.pop("LM3_CUDA_LIBPATH_SET", None)
    env.pop("LM3_SERVER_TOKEN", None)                  # the run has no business holding the secret

    pipe = _LaunchStatusPipe()
    child: Optional[_ManagedChild] = None
    try:
        try:
            with composed_launch(
                [pipe.contribution()],
                base_env=env,
                base_kwargs={
                    "cwd": str(cwd),
                    "stdout": fh, "stderr": subprocess.STDOUT, "stdin": subprocess.DEVNULL,
                    # Own session (POSIX) / own process group (Windows), from the ONE helper both
                    # managed launches share: one killpg reaches the executor's spawn workers and
                    # any subactivity that inherited the lease (section 3.3, "Stop targets the
                    # retained root process group"), and the hardcoded start_new_session this
                    # replaced was a silent no-op on Windows. _ManagedChild adds the job object
                    # that makes the Windows tree kill real.
                    **process_isolation_kwargs(),
                },
            ) as composed:
                proc = subprocess.Popen(argv, env=composed.env, **composed.popen_kwargs)
        except OSError as exc:
            raise RunError(f"could not launch {argv[0]}: {exc}", status=500) from exc

        child = _ManagedChild(proc, kind="pipeline", console_path=console, log_fh=fh)
        handshake = pipe.read(_handshake_timeout_s())

        if handshake.outcome == "busy":
            winner = (handshake.payload or {}).get("active")
            # The loser exits 75 on its own; joining it here is what lets the 409 promise that no
            # second process is running and that nothing under the run directory was touched.
            child.terminate(grace_s=2.0)
            child.close_log("[LM3 app] refused: another root activity holds this deployment\n")
            _check_staging_boundary(prohibited, pre_existing, "busy")
            message = ("another root activity already holds this deployment -- "
                       f"start refused for run '{run_name}'")
            raise RunError(message, status=HTTP_STATUS_BUSY,
                           payload={"reason": "busy", "active": winner,
                                    "request_log": str(console)})

        if handshake.outcome != "acquired":
            # Section 2.4: do NOT just return 500 and walk away. A slow child could acquire the
            # lease and run on after the API reported failure, leaving a run nobody launched.
            log.error("launch handshake failed (%s) for run %r: %s",
                      handshake.outcome, run_name, handshake.detail)
            leftover = child.terminate(grace_s=DEFAULT_STOP_GRACE_S)
            still_alive = child.alive
            note = _reconcile_after_failed_handshake(run_name)
            child.close_log(f"[LM3 app] launch handshake {handshake.outcome}: {handshake.detail}\n")
            _check_staging_boundary(prohibited, pre_existing, "failed-handshake")
            if still_alive:
                # "returns 500 only once no managed child remains" -- so say plainly that one does.
                raise RunError(
                    f"launch handshake {handshake.outcome} ({handshake.detail}) and the child "
                    f"(pid {child.pid}) survived SIGKILL; the deployment may still be occupied. "
                    f"Request log: {console}", status=500)
            detail = f"launch handshake {handshake.outcome}: {handshake.detail}"
            if leftover is not None:
                detail += f"; the child was terminated (exit {leftover})"
            if note:
                detail += f"; {note}"
            raise RunError(f"{detail}. Request log: {console}", status=500)

        run_id = str((handshake.payload or {}).get("run_id") or "")
        child.run_id = run_id
    except BaseException:
        if child is not None and child.alive:
            child.terminate(grace_s=2.0)
        if child is not None:
            child.close_log()
        else:
            with contextlib.suppress(Exception):
                fh.close()
        raise
    finally:
        pipe.close()

    run = _Run(run_name=run_name, config_path=str(cfg_path), cwd=str(cwd),
               output_dir=str(out_dir), run_dir=str(run_dir),
               db_path=str(run_dir / f"{run_name}.sqlite"),
               log_path=str(logs_dir / "lm3.log"), console_log=str(console),
               tmp_dir=str(tmp_dir), input_dirs=list(input_dirs),
               restart=list(keys), argv=list(argv))
    run.proc = child.proc
    run.child = child                                  # the control handle -- section 2.5
    run.log_fh = fh
    run.pid = child.pid
    run.pgid = child.pgid
    run.run_id = run_id

    with _RUN_LOCK:                                    # the ONLY place this path takes the lock
        _RUN = run
    # The child published ``active.json`` before it answered the handshake, so anything this server
    # read a moment ago is already stale. Drop it BEFORE binding the status stream, which resolves
    # the ledger to follow out of exactly that record.
    invalidate_runtime_cache()
    _bind_status(run)
    metrics.reset_baseline()
    metrics.set_worker_root(child.pid)
    threading.Thread(target=_reap, args=(run,), name="lm3-run-reaper", daemon=True).start()
    log.info("started LM3 run '%s' pid=%s run_id=%s cwd=%s", run_name, child.pid, run_id or "?", cwd)

    record = run.record()
    # Section 2.4 step 3: "acquired -> 200 with the runtime identity". Added HERE rather than in
    # _Run.record() so GET /v1/run/active keeps the key set its own docstring promises never
    # changes; the registry-backed GET /v1/runtime is Step 4's.
    record["run_id"] = run_id
    return record


def _reap_stale(run: _Run) -> None:
    """Finalize a record whose process vanished without the reaper thread noticing."""
    run.state = "done" if run.returncode in (0, None) else "error"
    run.finished_at = run.finished_at or time.time()


def _signal_group(run: _Run, sig: int) -> bool:
    """Signal the whole run: its process group first (that reaches the spawn workers).

    Reachable ONLY for a run this process launched. ``run.child`` (the flag-on path) and
    ``run.proc`` (the pre-flag path) are both retained live handles; ``run.pgid`` is derived from
    one of them at launch, by ``os.getpgid`` on our own child, and never read back out of a file.
    A ``_Run`` with neither handle cannot exist any more -- Step 4 deleted the only thing that
    built one (``_adopt``) -- and if one ever did, this refuses rather than signaling a PID whose
    provenance it cannot state (section 2.5, invariant 12).
    """
    if run.child is not None:
        # Section 2.5: when we hold a live handle for this run, THAT is the authority.
        return run.child.signal_group(sig)
    if run.proc is None:
        log.error("refusing to signal run %r: this server holds no handle for it", run.run_name)
        return False
    pgid = run.pgid
    # Refuse to signal our OWN group -- if start_new_session somehow failed, killpg here would take
    # the server down with the run.
    if pgid and hasattr(os, "killpg") and pgid != os.getpgrp():
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False
        except OSError:
            log.debug("killpg(%s, %s) failed; falling back to the pid", pgid, sig, exc_info=True)
    try:
        os.kill(run.pid, sig)
        return True
    except OSError:
        return False


def _observed_active() -> Optional[dict]:
    """The winner's record for a refusal body -- sanitized by ``read_runtime``, never interpreted."""
    snapshot, _diag = runtime_snapshot()
    if snapshot is None or not snapshot.active:
        return None
    if not snapshot.compatible:
        return {"compatible": False, "schema_version": snapshot.schema_version,
                "message": snapshot.message}
    return _record_payload(snapshot.record) if snapshot.record is not None else None


def _refuse_observer_stop() -> None:
    """Raise the section 2.5 refusal when the deployment is held by a run we do not own.

    Separated from ``stop_run``'s "nothing is running" branch on purpose: the two are different
    facts and a client that cannot tell them apart will retry the wrong one. 409, not 403, because
    this is a state conflict -- the same request succeeds from the process that owns the run.
    """
    if not runtime_v2():
        return
    snapshot, _diag = runtime_snapshot(refresh=True)
    if snapshot is None or not snapshot.active:
        return
    _can, reason = _can_stop(
        snapshot.record.run_id if snapshot.record is not None else None,
        compatible=snapshot.compatible)
    raise RunError(
        "this deployment is held by a run this server did not launch, so it cannot be stopped "
        f"from here -- {reason}", status=409,
        payload={"reason": "observer-only", "can_stop": False, "active": _observed_active()})


def stop_run(grace_s: float = DEFAULT_STOP_GRACE_S) -> dict:
    """Stop the active run: SIGTERM, then SIGKILL after ``grace_s``.

    LM3 is resumable -- every image is checkpointed in the project ledger and ``reclaim_running()``
    turns interrupted stages back to 'pending' on the next start -- so a stopped run restarts from
    where it was, minus the images that were mid-flight.

    OBSERVER-ONLY RUNS ARE REFUSED (section 2.5). If the deployment is held by a root this server
    did not launch -- a CLI run, or one launched by a previous instance of this server -- there is
    no handle, and the refusal is a 409 that says so and names the winner. What this function will
    NOT do, ever, is rebuild a process group out of a registry record and signal it: that is
    invariant 12 verbatim, and the machinery that used to do it (``_adopt`` and ``_pid_alive``) was
    deleted rather than merely bypassed.
    """
    with _RUN_LOCK:
        run = _current()
        if run is None or run.state not in ("running", "stopping"):
            if _LAUNCHING is not None:
                # Section 2.5: control needs a retained handle, and during the handshake the launch
                # thread is the only holder of one. Saying so beats "no active run to stop", which
                # would be false.
                raise RunError("a run is being launched and has not answered the handshake yet -- "
                               "there is no managed child to stop", status=409)
            _refuse_observer_stop()
            raise RunError("no active run to stop", status=409)
        if not run.alive:
            _reap_stale(run)
            return run.record()
        can_stop, reason = _can_stop(run.run_id or None, compatible=True)
        if not can_stop:
            raise RunError(reason or "this server may not stop that run", status=409,
                           payload={"reason": "observer-only", "can_stop": False,
                                    "active": _observed_active()})
        run.state = "stopping"
        run.stopped_by_user = True
        _signal_group(run, signal.SIGTERM)
        log.info("SIGTERM sent to LM3 run '%s' (pgid %s)", run.run_name, run.pgid)

    deadline = time.time() + max(0.0, float(grace_s))
    while time.time() < deadline:
        if not run.alive:
            break
        time.sleep(0.2)

    if run.alive:                                      # ignored SIGTERM (a wedged CUDA call, say)
        log.warning("LM3 run '%s' ignored SIGTERM after %.1fs -- sending SIGKILL", run.run_name, grace_s)
        _signal_group(run, signal.SIGKILL)
        hard_deadline = time.time() + 5.0
        while time.time() < hard_deadline and run.alive:
            time.sleep(0.1)

    # The reaper thread sets the terminal state; give it a moment so the response is not stale.
    for _ in range(20):
        if run.state not in ("running", "stopping"):
            break
        time.sleep(0.1)
    invalidate_runtime_cache()
    with _RUN_LOCK:
        if run.state == "stopping" and not run.alive:
            _reap_stale(run)
        return run.record()


def console_tail(offset: int = -65536, max_bytes: int = 262144) -> dict:
    """Byte-range read of the captured stdout+stderr.

    A negative ``offset`` means "the last N bytes" (the first call), and ``next_offset`` feeds the
    next poll, so the console view appends instead of refetching. This is the ONLY place a crash
    before ``start_logging()`` shows up -- config errors, a missing model export, an OOM kill.
    """
    path = active_console_path()
    if path is None or not path.is_file():
        return {"path": str(path) if path else None, "size": 0, "offset": 0,
                "next_offset": 0, "text": "", "eof": True}
    try:
        size = path.stat().st_size
        start = max(0, size + offset) if offset < 0 else min(int(offset), size)
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(max(0, int(max_bytes)))
        return {"path": str(path), "size": size, "offset": start,
                "next_offset": start + len(chunk),
                "text": chunk.decode("utf-8", "replace"),
                "eof": start + len(chunk) >= size}
    except OSError as exc:
        return {"path": str(path), "size": 0, "offset": 0, "next_offset": 0,
                "text": f"[LM3 app] could not read the console log: {exc}", "eof": True}


# --------------------------------------------------------------------------- #
# SSE
# --------------------------------------------------------------------------- #
def _frame(kind: str, **payload: Any) -> str:
    """One typed SSE envelope -- the panel switches on ``type`` (same contract as metrics.py)."""
    return "data: " + json.dumps({"type": kind, **payload}) + "\n\n"


def stream_frames(interval_s: Optional[float] = None, *, workers_every: int = 4,
                  max_seconds: float = 86400.0) -> Iterator[str]:
    """Synchronous form of the metrics stream (tests, non-async callers).

    Frames: ``hello`` (machine + the whole rolling window + the run record) once, then ``metrics``
    per new sample, ``workers`` every ``workers_every`` metrics frames, and ``run`` whenever the
    run's state or pid changes.
    """
    sampler = metrics.get_sampler()
    period = float(interval_s or sampler.interval)
    yield _frame("hello", machine=sampler.describe_machine(), history=sampler.history(), run=active())
    deadline = time.time() + max_seconds
    last_seq, tick, last_run = -1, 0, _run_key()
    while time.time() < deadline:
        point = sampler.snapshot()
        if point.get("seq") != last_seq:
            last_seq = point.get("seq")
            yield _frame("metrics", snapshot=point)
            tick += 1
            if workers_every and tick % workers_every == 0:
                yield _frame("workers", workers=sampler.workers())
        key = _run_key()
        if key != last_run:
            last_run = key
            yield _frame("run", run=active())
        time.sleep(period)


def _run_key() -> tuple:
    """Cheap change-detector for the run record (state + identity), so `run` frames are rare."""
    run = _current()
    if run is not None and run.state in ("running", "stopping"):
        return (run.pid, run.state, run.returncode)
    if _launching_record() is not None:                # the SSE stream shows "starting" too
        return (0, "starting", None)
    return (run.pid, run.state, run.returncode) if run is not None else (0, "idle", None)


# --------------------------------------------------------------------------- #
# FastAPI router (the integrator mounts this; app.py is not edited here)
# --------------------------------------------------------------------------- #
def _expected_token() -> Optional[str]:
    """The server's Bearer secret. ``app._server_token`` mints it into the environment at
    startup, so the env is the cheap read; the import is only the cold-start fallback."""
    token = os.environ.get("LM3_SERVER_TOKEN")
    if token:
        return token
    try:
        from leafmachine3.server.app import _server_token

        return _server_token()
    except Exception:  # noqa: BLE001 - app.py optional / not yet initialized
        return None


def router(dependencies: Optional[list] = None) -> Any:
    """Build the ``/v1`` router for metrics + run control.

    ``fastapi`` is imported lazily, exactly like ``leafmachine3.server.app.create_app``, so a base
    install without the ``server`` extra can still import this module. Pass the app's auth
    dependency through ``dependencies`` (``[Depends(require_token)]``).

    ``dependencies`` is applied PER ROUTE rather than to the router, because ``/metrics/stream``
    cannot use a header-only guard: ``EventSource`` is unable to set an ``Authorization`` header,
    so it accepts ``?token=`` as well and checks it itself against the same secret -- exactly the
    arrangement ``progress_api.router`` and ``postprocess_api.router`` already use for their SSE
    routes. Loopback-only binding is what keeps a secret in a URL acceptable.
    """
    import asyncio

    from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Response
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import StreamingResponse

    guards = list(dependencies or [])
    api = APIRouter(prefix="/v1", tags=["metrics"])

    # ``from __future__ import annotations`` (top of this file) stringifies every annotation, and
    # FastAPI resolves those strings against the handler's MODULE globals -- where a name imported
    # inside router() does not exist. Publishing it is what lets the lazy import (needed so a base
    # install without the server extra can still import this module) coexist with
    # ``response: Response`` on the deprecated-projection route below.
    globals().setdefault("Response", Response)

    async def stream_token(
        token: Optional[str] = Query(default=None, description="Bearer secret, for EventSource"),
        authorization: str = Header(default=""),
    ) -> None:
        """Auth for the SSE route: ``?token=`` OR the Authorization header."""
        if not guards:                                 # mounted without auth -> nothing to check
            return
        expected = _expected_token()
        if not expected:
            return
        if token and secrets.compare_digest(str(token), expected):
            return
        if authorization and secrets.compare_digest(authorization, f"Bearer {expected}"):
            return
        raise HTTPException(status_code=401, detail="invalid or missing token")

    def _fail(exc: RunError) -> Any:
        if exc.payload is not None:
            # Section 2.4: a 409 has to name the WINNER, so this refusal carries a body rather
            # than a sentence. Refusals without a payload keep their plain-string detail.
            return HTTPException(status_code=exc.status,
                                 detail={"message": str(exc), **exc.payload})
        return HTTPException(status_code=exc.status, detail=str(exc))

    # -- metrics ------------------------------------------------------------ #
    @api.get("/metrics", dependencies=guards)
    async def get_metrics(workers: bool = False) -> dict:
        """Current point + static machine facts. ``seq`` is monotonic -- use it to skip redraws."""
        point = metrics.snapshot()
        out = {"snapshot": point, "machine": metrics.describe_machine(),
               "t": point.get("t"), "seq": point.get("seq"), "ready": point.get("ready", False)}
        if workers:
            out["workers"] = metrics.workers()
        return out

    @api.get("/metrics/machine", dependencies=guards)
    async def get_machine() -> dict:
        return metrics.describe_machine()

    @api.get("/metrics/history", dependencies=guards)
    async def get_history(since: Optional[float] = None, max_points: Optional[int] = None) -> dict:
        """The rolling window, columnar. Pass ``since=t_last`` to append instead of refetching."""
        return metrics.history(since=since, max_points=max_points)

    @api.get("/metrics/workers", dependencies=guards)
    async def get_workers() -> dict:
        return metrics.workers()

    @api.get("/metrics/stream", dependencies=[Depends(stream_token)])
    async def metrics_stream(interval: Optional[float] = None, workers_every: int = 4,
                             token: Optional[str] = None) -> Any:
        """SSE at the sampler's rate (~2 Hz). One connection fills the plots AND keeps them live.

        ``token`` is for ``EventSource``, which cannot set an Authorization header. The
        ``stream_token`` dependency above is what checks it -- the app's own ``require_token``
        reads the HEADER only, so guarding this route with it would 401 every EventSource.
        """
        sampler = metrics.get_sampler()
        period = max(0.1, float(interval or sampler.interval))

        async def gen() -> Any:
            # Every read is a ring-buffer copy, so this streams without ever blocking the loop.
            #
            # STOPPING ON DISCONNECT: Starlette drives this generator and awaits ``send`` for each
            # frame; once the client is gone that send raises (ClientDisconnect / CancelledError at
            # shutdown), the exception lands here at the ``yield``, and the ``finally`` runs. The
            # heartbeat below is what bounds the detection latency -- a stream that yielded nothing
            # would never learn the socket had closed, so we always write SOMETHING every 15 s even
            # if the sampler has stalled.
            last_hb = time.monotonic()
            try:
                yield _frame("hello", machine=sampler.describe_machine(),
                             history=sampler.history(), run=active())
                last_seq, tick, last_run = -1, 0, _run_key()
                while True:
                    point = sampler.snapshot()
                    if point.get("seq") != last_seq:
                        last_seq = point.get("seq")
                        yield _frame("metrics", snapshot=point)
                        tick += 1
                        if workers_every and tick % workers_every == 0:
                            yield _frame("workers", workers=sampler.workers())
                    key = _run_key()
                    if key != last_run:
                        last_run = key
                        yield _frame("run", run=active())
                    now = time.monotonic()
                    if now - last_hb > 15.0:
                        last_hb = now
                        yield ": hb\n\n"               # comment frame: keeps idle proxies honest
                    await asyncio.sleep(period)
            finally:
                # Reached on a client disconnect, on shutdown cancellation, and on a clean return.
                # Nothing to release (the sampler is shared and stays running) -- this is the proof
                # the generator really did stop rather than leaking a task per reload.
                log.debug("metrics stream closed")

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                          "X-Accel-Buffering": "no"})

    # -- hardware ----------------------------------------------------------- #
    @api.get("/hardware/profile", dependencies=guards)
    async def get_hardware_profile() -> dict:
        """The tuned ``hardware_settings.yaml``: per-module workers, GPU sizing, spawn overhead,
        disk knee -- plus a cheap staleness check against the live machine."""
        return await run_in_threadpool(hardware_profile)

    # -- run control -------------------------------------------------------- #
    # ``/run/start`` and ``/run/stop`` are KEPT by section 5 -- only the ``db_path`` field in the
    # record they return is deprecated (section 2.10: a projection of the ``active_db_path``
    # storage role). Announced with ``X-LM3-Deprecated-Fields`` rather than ``Deprecation: true``
    # alone, so a client cannot misread "a field is going" as "stop calling this endpoint".
    #: ``successor=None``: there is no successor ROUTE here, only a successor FIELD. Sending a
    #: ``Link: rel="successor-version"`` would tell a client to stop calling an endpoint section 5
    #: explicitly keeps.
    _RECORD_FIELD_HEADERS = deprecation_headers(
        what="the db_path field of the run record",
        replacement="active_db_path (GET /v1/runtime -> active.record.project.active_db_path)",
        successor=None,
        fields="db_path",
    )

    @api.post("/run/start", dependencies=guards)
    async def run_start(payload: Optional[dict] = Body(default=None),
                        response: Response = None) -> dict:
        body = payload or {}
        unknown = set(body) - {"config_path", "input_dir", "output_dir", "restart"}
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown field(s): {sorted(unknown)}")
        if response is not None and runtime_v2():
            response.headers.update(_RECORD_FIELD_HEADERS)
        try:
            return await run_in_threadpool(
                start_run,
                config_path=body.get("config_path"),
                input_dir=body.get("input_dir"),
                output_dir=body.get("output_dir"),
                restart=body.get("restart"),
            )
        except RunError as exc:
            raise _fail(exc) from exc

    @api.post("/run/stop", dependencies=guards)
    async def run_stop(payload: Optional[dict] = Body(default=None),
                       response: Response = None) -> dict:
        grace = float((payload or {}).get("grace_s", DEFAULT_STOP_GRACE_S))
        if response is not None and runtime_v2():
            response.headers.update(_RECORD_FIELD_HEADERS)
        try:
            return await run_in_threadpool(stop_run, grace)
        except RunError as exc:
            raise _fail(exc) from exc

    # -- the canonical runtime view (section 4 Step 4, section 5) ----------- #
    # Behind the flag, and the flag is read PER REQUEST rather than here: whether a route EXISTS
    # must not depend on what the environment looked like at the instant the app object was built.
    # Every other reader in this module (``active``, ``_can_stop``, ``start_run``) is dynamic, and a
    # single static reader among them is how a server ends up serving the v2 record from
    # ``/v1/run/active`` while ``/v1/runtime`` 404s -- one flag, one reader, one answer.
    #
    # Off, the answer is 404 and not an empty 200. Those are different claims: 404 says "this build
    # cannot tell you what is running"; an empty 200 says "nothing is running" -- which would be a
    # lie while a CLI run is in fact going, and a renderer must be able to tell them apart.
    @api.get("/runtime", dependencies=guards)
    async def get_runtime() -> dict:
        """``{active, last, next_run_settings, server, diagnostics}`` -- :func:`runtime_view`.

        Off the loop: it probes the deployment lock, parses two small JSON files, and reads the
        canonical settings file. The lock probe and the JSON are microseconds on a local disk; the
        settings YAML is NOT -- ~9 KB through PyYAML's pure-Python loader measures ~17 ms, which is
        why :func:`_load_next_run_config` memoizes that parse on the file's mtime and size. None of
        the three is bounded on a network filesystem.
        """
        if not runtime_v2():
            raise HTTPException(status_code=404, detail="Not Found")
        return await run_in_threadpool(runtime_view)

    @api.get("/run/active", dependencies=guards)
    async def run_active(response: Response = None) -> dict:
        """The compatibility projection of :func:`runtime_view` (section 5). Superseded.

        Off the event loop, like every sibling route: ``active()`` takes ``_RUN_LOCK``, and a
        threading lock awaited from a coroutine blocks the LOOP THREAD, not just this request --
        measured, a 3 s hold starved every other route and the SSE heartbeats for the whole 3 s.
        Narrowing the launch critical section (see ``_LAUNCHING``) is the fix; this is the defense
        in depth that keeps any future hold off the loop.

        The deprecation metadata rides in HEADERS, never in the body: the body's key set is pinned
        (``_Run.record``'s docstring promises it never changes, and ``tests/_contract_helpers.py``
        asserts it key-for-key), so a ``deprecated: true`` field would break the very clients the
        projection exists to keep working. Headers are gated on the flag for the same reason
        everything else in this step is -- with ``LM3_RUNTIME_V2`` off there is no ``/v1/runtime``
        worth migrating to yet.
        """
        if response is not None and runtime_v2():
            # Step 4 announced the deprecation with no removal date, because section 4 Step 7 owns
            # naming the release. Step 7 has now named it: COMPAT_REMOVAL_RELEASE, in
            # docs/DEPRECATIONS.md and in X-LM3-Removed-In below. The whole ROUTE goes -- with it
            # the ``db_path`` and ``adopted`` fields, which is why neither gets its own schedule.
            response.headers.update(deprecation_headers(
                what="GET /v1/run/active",
                replacement=("GET /v1/runtime, which can express run_id, activity, the bounded "
                             "child tree and can_stop"),
            ))
        return await run_in_threadpool(active)

    @api.get("/run/console", dependencies=guards)
    async def run_console(offset: int = -65536, max_bytes: int = 262144) -> dict:
        return await run_in_threadpool(console_tail, offset, max_bytes)

    return api


__all__ = ["RunError", "router", "hardware_profile", "hardware_path", "default_config_path",
           "active", "is_active", "active_db_path", "active_log_path", "active_console_path",
           "start_run", "stop_run", "console_tail", "stream_frames",
           # Step 3: the flag reader, the launch handshake and the staging boundary that
           # leafmachine3.server.app reuses -- ONE reader of LM3_RUNTIME_V2 for the whole server.
           "runtime_v2", "execution_owned_paths", "process_isolation_kwargs",
           # Step 7: the named removal release and the header set that announces it. One constant
           # so docs/DEPRECATIONS.md, the Warning header and the tests cannot drift apart.
           "COMPAT_REMOVAL_RELEASE", "COMPAT_SUCCESSOR_ROUTE", "deprecation_headers",
           # Step 4: the registry reader. ``active_run_ref`` is the one project reference every
           # other server module should resolve through (section 5), and
           # ``invalidate_runtime_cache`` is the hook that replaces ``bind_run``'s cache drop.
           "SERVER_INSTANCE_ID", "runtime_view", "runtime_snapshot", "active_run_ref",
           "next_run_settings", "deployment_dir", "invalidate_runtime_cache"]
