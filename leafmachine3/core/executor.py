"""leafmachine3.core.executor -- per-stage GPU/CPU saturation with per-image checkpointing.

Model in one paragraph: stages are SEQUENTIAL (a hard barrier between them). WITHIN a
stage a pool of persistent WORKER PROCESSES (``spawn`` start-method, so each owns its own
CUDA context / ONNX Runtime session, is GIL-free and crash-isolated) PULLS ``WorkItem``s
from a BOUNDED queue as workers finish -- not ``pool.map`` -- so one slow 12k-px sheet never
head-of-line-stalls the rest and memory stays flat via backpressure. Workers are PURE
(``build_model`` + ``infer``, never touch the DB). The PARENT is the SINGLE DB WRITER: as
each result lands it runs ``stage.persist(...)`` AND ``mark_image_done`` in ONE transaction,
so an abrupt kill leaves exactly the finished items marked ``done`` -> a clean, additive resume.

For CPU-only stages, mock runs, or machines with no GPU, ``run`` takes a simpler in-process
path (serial, or a thread pool for pure-Python CPU stages) that still checkpoints through the
same single-writer ``persist`` + ``mark_image_done`` transaction. This keeps the whole thing
correct and picklable-safe for the mock smoke test while retaining the process-pool path for
real GPU inference.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import subprocess
import signal
import sys
import threading
from collections import Counter, deque
from typing import Any, Iterable, Optional

from leafmachine3.core.device import Device
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.executor")

_CTX = mp.get_context("spawn")
_DEFAULT_LIVENESS_TIMEOUT_S = 120.0
_DEFAULT_PER_WORKER_VRAM_MB = 8000.0
# How long a worker waits for the warm-load gate before loading anyway. The gate is a throughput
# optimization, not a correctness barrier, and its permits can LEAK: a worker SIGKILLed inside
# build_model (wedged CUDA context, OOM killer) never releases the POSIX semaphore it holds, so
# after `concurrent_warm_loads` such deaths every later replacement would block on it forever.
# Loading un-gated after the timeout is strictly better than never loading at all.
_WARM_LOAD_GATE_TIMEOUT_S = 300.0
# Two independent limits on device-fault recovery, doing two different jobs.
#
# PER ITEM (`compute.max_item_requeues`) is what guarantees the stage TERMINATES: a requeued item
# never advances `seen`, so one that faults on every attempt would loop forever. After this many
# hand-backs the item is failed instead. Scaled by a specimen's sub-item count at the call site --
# a fanout stage has many WorkItems per specimen_id and WorkItem carries no identity of its own.
#
# PER STAGE (`compute.max_device_recoveries`) is only a spawn-storm rail, since every requeue also
# costs a process spawn. It is deliberately sized off the ITEM count, not the worker count: faults
# arrive per image, so a worker-scaled ceiling gets spent early in a long run and then starts
# failing images that one retry would have fixed.
_DEFAULT_MAX_ITEM_REQUEUES = 3
_MIN_RECOVERY_BUDGET = 64
_FATAL_DEVICE_MARKERS = (
    "out of memory",
    "cuda error",
    "cuda runtime",
    "cublas",
    "cudnn",
    "device-side assert",
    "cudaerror",
    "cuda_error",
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class StageError(RuntimeError):
    """A stage failed hard (raised under ``fail_fast`` so the run aborts)."""


class FatalDeviceError(RuntimeError):
    """A worker hit an unrecoverable device fault (CUDA OOM / wedged context).

    The worker exits and the parent re-queues the offending item and spawns a
    replacement worker so the pool size is preserved.
    """


# --------------------------------------------------------------------------- #
# Cross-process STOP sentinel (identity-stable across spawn via __reduce__)
# --------------------------------------------------------------------------- #
def _resolve_stop() -> "_Stop":
    return _STOP


class _Stop:
    """A picklable sentinel that always unpickles to the receiver's module global."""

    def __reduce__(self):  # noqa: D401 - pickling hook
        return (_resolve_stop, ())


_STOP = _Stop()


# --------------------------------------------------------------------------- #
# Small tolerant getters (config Sections, dicts, dataclasses all supported)
# --------------------------------------------------------------------------- #
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a mapping, Section, or object, tolerating any of them."""
    if obj is None:
        return default
    for accessor in (lambda: obj[key], lambda: getattr(obj, key), lambda: obj.get(key)):
        try:
            value = accessor()
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _free_vram_mb(index: int) -> Optional[float]:
    """Free VRAM on GPU ``index`` in MiB via NVML, falling back to ``nvidia-smi``."""
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return float(info.free) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i",
                str(index),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def _is_fatal_device_error(exc: BaseException) -> bool:
    """Heuristically classify an inference exception as an unrecoverable device fault."""
    if isinstance(exc, FatalDeviceError):
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in message for marker in _FATAL_DEVICE_MARKERS)


# --------------------------------------------------------------------------- #
# Device planning
# --------------------------------------------------------------------------- #
class DeviceManager:
    """Decide how many worker slots a stage gets and on which devices.

    Preference order for GPU stages:
      1. reuse LM3_Setup's tuned ``workers_per_gpu`` from the hardware profile (fast path);
      2. otherwise size by free VRAM against the ``compute.vram`` budget.
    CPU stages (and mock / no-GPU machines) get a flat worker count from the hardware
    profile or ``cfg.io_workers()``.
    """

    def __init__(self, cfg: Any, stage: PipelineStage) -> None:
        self.cfg = cfg
        self.stage = stage

    # -- public ------------------------------------------------------------- #
    def plan(self) -> list[Device]:
        if self._is_cpu_only():
            return [Device("cpu")] * max(1, self._cpu_worker_count())

        gpus = self._resolve_gpus()
        if not gpus:
            return [Device("cpu")] * max(1, self._cpu_worker_count())

        # Size every GPU independently, from its OWN live free VRAM ("building blocks"): a card
        # already holding another job takes fewer workers than an idle one, so the counts are
        # deliberately uneven. The hardware profile supplies the per-worker COST; free VRAM at
        # run time -- not a number frozen at profile time -- decides how many of them fit.
        return self._vram_plan(gpus)

    @staticmethod
    def ensure_cuda_libpath() -> None:
        """Put the venv's bundled NVIDIA ``.so`` dirs on ``LD_LIBRARY_PATH``.

        onnxruntime-gpu silently drops to CPU if those libs are not on the loader path
        at process start. Spawn children inherit this process's environment, so exporting
        it here (before any worker starts) is enough -- no re-exec required.
        """
        from leafmachine3.core.cuda_libs import nvidia_lib_dirs

        libdirs = nvidia_lib_dirs()                  # via nvidia.__path__: survives a deleted __init__.py
        if not libdirs:
            return
        current = os.environ.get("LD_LIBRARY_PATH", "")
        parts = current.split(os.pathsep) if current else []
        changed = False
        for directory in libdirs:
            if directory not in parts:
                parts.insert(0, directory)
                changed = True
        if changed:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

    # -- internals ---------------------------------------------------------- #
    def _is_cpu_only(self) -> bool:
        if self.stage.device_kind == "cpu":
            return True
        if bool(_get(self.cfg, "compute", None) and _get(self.cfg.compute, "mock", False)):
            return True
        devices = _get(_get(self.cfg, "compute", None), "devices", "auto")
        return isinstance(devices, str) and devices.lower() == "cpu"

    def _resolve_gpus(self) -> list[int]:
        resolver = getattr(self.cfg, "resolve_gpus", None)
        if callable(resolver):
            try:
                return list(resolver() or [])
            except Exception:
                return []
        return []

    def _cpu_worker_count(self) -> int:
        hw = self._hw_stage()
        workers = _get(hw, "workers", None)
        if workers:
            return int(workers)
        io_workers = getattr(self.cfg, "io_workers", None)
        if callable(io_workers):
            try:
                return int(io_workers())
            except Exception:
                pass
        return max(1, (os.cpu_count() or 2))

    def _hw_stage(self) -> Any:
        hardware = _get(self.cfg, "hardware", None)
        if hardware is None:
            return None
        getter = getattr(hardware, "stage", None)
        if callable(getter):
            try:
                return getter(self.stage.key)
            except Exception:
                return None
        stages = _get(hardware, "stages", None)
        if stages is not None:
            return _get(stages, self.stage.key, None)
        return None

    def _vram_plan(self, gpus: list[int]) -> list[Device]:
        vram = _get(_get(self.cfg, "compute", None), "vram", {}) or {}
        safety = float(_get(vram, "safety_fraction", 0.90) or 0.90)
        reserve = float(_get(vram, "reserve_mb", 1024) or 0.0)
        max_per_gpu = int(_get(vram, "max_workers_per_gpu", 4) or 4)
        per_worker = self._per_worker_vram_mb(vram)

        # A profile-frozen count is only the fallback for when the live VRAM probe fails.
        fallback = max(1, min(max_per_gpu, int(_get(self._hw_stage(), "workers_per_gpu", 1) or 1)))

        plan: list[Device] = []
        for gpu in gpus:
            free = _free_vram_mb(gpu)
            if free and per_worker > 0:
                # 0 is a legitimate answer: a card too full for even one worker gets SKIPPED
                # rather than force-fed one, which is precisely how a busy GPU OOMs.
                budget = free * safety - reserve
                count = max(0, min(int(budget // per_worker), max_per_gpu))
                if count == 0:
                    log.warning(
                        "%s: GPU %d has %.0f MB free -> %.0f MB usable (x%.2f safety, -%.0f reserved), "
                        "under the %.0f MB one worker needs; skipping it",
                        self.stage.key, gpu, free, budget, safety, reserve, per_worker,
                    )
            else:
                count = fallback                       # VRAM unreadable -> trust the profile
            plan.extend([Device("cuda", gpu)] * count)

        if not plan:
            # Every GPU looked too full. Falling back to CPU here would be orders of magnitude
            # slower and deeply confusing, so put one worker on the roomiest card and let it try.
            roomiest = max(gpus, key=lambda g: _free_vram_mb(g) or 0.0)
            log.warning("%s: no GPU fits a %.0f MB worker -- forcing 1 worker onto GPU %d",
                        self.stage.key, per_worker, roomiest)
            plan = [Device("cuda", roomiest)]

        log.info("%s: GPU plan %s @ %.0f MB/worker",
                 self.stage.key,
                 {g: sum(1 for d in plan if d.index == g) for g in sorted(set(gpus))},
                 per_worker)
        return plan

    def _per_worker_vram_mb(self, vram: Any) -> float:
        """Cost of ONE worker of this stage in MiB, with OOM headroom folded in.

        Order of preference: test hook -> explicit config -> MEASURED (calibration run) ->
        heuristic per-worker estimate -> a conservative constant. Note this deliberately does
        NOT read ``peak_vram_mb``: that field is the total across *all* workers, and reading it
        as a per-worker cost under-counted the fit by the worker count itself.
        """
        probe = getattr(self.cfg, "probe_vram_mb", None)
        if callable(probe):
            try:
                value = probe(self.stage)
                if value:
                    return float(value)
            except Exception:
                pass
        configured = _get(vram, "per_worker_mb", "auto")
        if isinstance(configured, (int, float)) and configured > 0:
            return float(configured)                   # explicit override wins, verbatim

        headroom = float(_get(vram, "headroom_factor", 1.15) or 1.15)
        hw = self._hw_stage()
        measured = _get(hw, "vram_per_worker_mb", None)          # from the calibration run
        if measured:
            return float(measured) * headroom
        estimated = _get(hw, "est_vram_per_worker_mb", None)     # heuristic, still per worker
        if estimated:
            return float(estimated) * headroom
        return _DEFAULT_PER_WORKER_VRAM_MB


# --------------------------------------------------------------------------- #
# Worker process body (must be top-level for spawn picklability)
# --------------------------------------------------------------------------- #
def _set_parent_death_signal() -> bool:
    """Ask the kernel to SIGKILL this process the moment its parent dies (Linux ``PR_SET_PDEATHSIG``).

    ``daemon=True`` is NOT sufficient: multiprocessing implements it in an ``atexit`` hook, so it
    only runs on an orderly interpreter shutdown. A ``SIGKILL``ed (or hard-crashed, or OOM-killed)
    parent runs no Python at all, and its GPU workers are reparented to init and keep their CUDA
    contexts **forever** -- 11 strays once held 6.8 GB on this machine. This is kernel-enforced, so
    it is the only thing that closes that hole.

    Caveat that dictates where this may be called from: pdeathsig fires when the creating THREAD
    exits, not the creating process, so it is only safe when workers are spawned from the main
    thread (see ``_pdeathsig_safe``).
    """
    if sys.platform != "linux":
        return False
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        return libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) == 0
    except Exception:  # noqa: BLE001 - no libc / no prctl -> fall back to the startup sweep
        return False



def reap_orphaned_workers(dry_run: bool = False) -> list[tuple[int, float]]:
    """Kill LM3 worker processes that outlived their parent, and report ``[(pid, vram_mb)]``.

    Second line of defense behind :func:`_set_parent_death_signal`, covering the cases it cannot:
    workers spawned before this fix shipped, workers spawned off a non-main thread, a lost arming
    race, and non-Linux hosts. An orphan can never be handed work again -- its queues died with
    its parent -- so it is pure leaked VRAM.

    Deliberately narrow, because this kills processes: a candidate must be OUR uid, running OUR
    interpreter, be a ``multiprocessing.spawn`` child, and have a parent that is NOT a live
    LeafMachine3 process. That last test is what makes a CONCURRENT LM3 run safe -- its workers
    have a living LM3 parent, so they are never touched.
    """
    try:
        import psutil
    except Exception:  # noqa: BLE001 - no psutil -> pdeathsig is the only defense
        log.debug("psutil unavailable; skipping the orphaned-worker sweep")
        return []

    me = os.getpid()
    my_uid = os.getuid() if hasattr(os, "getuid") else None
    my_exe = os.path.realpath(sys.executable)
    vram_by_pid = _worker_vram_by_pid()
    reaped: list[tuple[int, float]] = []

    for proc in psutil.process_iter(["pid", "ppid", "cmdline", "uids", "exe"]):
        try:
            info = proc.info
            pid = info["pid"]
            if pid == me:
                continue
            cmdline = info.get("cmdline") or []
            if not any("multiprocessing.spawn" in part for part in cmdline):
                continue
            if my_uid is not None and (info.get("uids") is None or info["uids"].real != my_uid):
                continue
            exe = info.get("exe")
            if not exe or os.path.realpath(exe) != my_exe:      # a different venv is not ours
                continue
            if _parent_is_live_lm3(psutil, info.get("ppid")):
                continue                                        # adopted by a running LM3 -> leave it

            mb = vram_by_pid.get(pid, 0.0)
            if dry_run:
                reaped.append((pid, mb))
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()                                     # it was holding VRAM; insist
            reaped.append((pid, mb))
        except Exception:  # noqa: BLE001 - process vanished mid-scan, or is not inspectable
            continue

    if reaped:
        total = sum(mb for _, mb in reaped)
        log.warning("reaped %d orphaned LM3 worker(s)%s, freeing %.0f MB of VRAM: %s",
                    len(reaped), " (dry run)" if dry_run else "", total,
                    ", ".join(str(p) for p, _ in reaped))
    return reaped


def _parent_is_live_lm3(psutil: Any, ppid: Optional[int]) -> bool:
    """True when ``ppid`` is a running LeafMachine3 process (so its children are NOT orphans).

    Reparenting is not simply "ppid == 1": on a systemd host, orphans are adopted by
    ``systemd --user``, which is very much alive. So this asks what the parent actually IS.
    """
    if not ppid or ppid <= 1:
        return False
    try:
        parent = psutil.Process(ppid)
        blob = " ".join(parent.cmdline())
    except Exception:  # noqa: BLE001 - parent gone or unreadable -> treat the child as orphaned
        return False
    return ("leafmachine3" in blob) or ("machine3" in blob)


def _worker_vram_by_pid() -> dict[int, float]:
    """``{pid: MiB}`` for processes currently holding GPU memory (best effort, for logging)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return {}
        found = {}
        for line in out.stdout.strip().splitlines():
            pid, _, mem = line.partition(",")
            found[int(pid.strip())] = float(mem.strip())
        return found
    except Exception:  # noqa: BLE001 - no nvidia-smi / unparsable -> sizes are cosmetic anyway
        return {}


def _pdeathsig_safe(stage_key: str = "") -> bool:
    """Whether ``PR_SET_PDEATHSIG`` may be armed for workers spawned from HERE.

    The kernel delivers pdeathsig when the creating THREAD exits, not when the parent process
    does. Arming it for workers spawned off a short-lived helper thread would therefore kill
    the pool the moment that thread returned -- mid-run, with work in flight. Only the main
    thread is guaranteed to outlive the pool, so that is the only place it is armed; anywhere
    else falls back to :func:`reap_orphaned_workers` at the next startup.
    """
    if threading.current_thread() is threading.main_thread():
        return True
    log.warning(
        "%s: workers spawned off thread %r, so parent-death SIGKILL is not armed for them; "
        "a hard kill of this process would strand them until the next LM3 startup sweep",
        stage_key or "stage", threading.current_thread().name,
    )
    return False


def _worker(
    device: Device,
    stage: PipelineStage,
    task_q: "mp.Queue",
    result_q: "mp.Queue",
    load_gate: Any,
    parent_pid: int = 0,
    arm_pdeathsig: bool = False,
) -> None:
    """Persistent worker: pin the GPU, warm-load ONCE (staggered behind ``load_gate`` so N
    workers do not warm-load simultaneously and OOM), then pull-and-infer until ``_STOP``."""
    if arm_pdeathsig:
        _set_parent_death_signal()
    # Race: if the parent died between spawning us and our prctl call, the signal we just armed
    # will never fire. Re-check lineage now -- a changed ppid means we are already orphaned, and
    # an orphan can never be handed work, so leaving would only strand VRAM.
    if parent_pid and os.getppid() != parent_pid:
        return
    if device.kind == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)  # before any torch/ort import
    # Bounded acquire, not `with load_gate`: a permit held by a worker that was SIGKILLed inside
    # build_model is never returned, and enough of those would wedge every replacement here for
    # good. See _WARM_LOAD_GATE_TIMEOUT_S.
    held = False
    try:
        held = bool(load_gate.acquire(timeout=_WARM_LOAD_GATE_TIMEOUT_S))
        if not held:
            log.warning("%s: warm-load gate timed out after %.0fs; loading un-gated",
                        stage.key, _WARM_LOAD_GATE_TIMEOUT_S)
        model = stage.build_model(device)
    except Exception as exc:  # cannot even load the model on this device
        result_q.put(("fatal_start", None, repr(exc)))
        return
    finally:
        if held:
            load_gate.release()

    while True:
        item = task_q.get()
        if item is _STOP:
            break
        try:
            payload = stage.infer(item, model)
            result_q.put(("ok", item, payload))
        except Exception as exc:  # isolate per-image failures; keep the pool alive
            if _is_fatal_device_error(exc):
                result_q.put(("requeue", item, repr(exc)))
                break
            result_q.put(("err", item, repr(exc)))


# --------------------------------------------------------------------------- #
# The executor
# --------------------------------------------------------------------------- #
class StageExecutor:
    """Runs one :class:`PipelineStage` to completion with per-image checkpointing."""

    def __init__(self, cfg: Any, stage: PipelineStage) -> None:
        self.cfg = cfg
        self.stage = stage
        self._devices: list[Device] = []
        self._gate: Any = None
        self._remaining: Counter = Counter()   # specimen_id -> outstanding (sub-)items; single-writer
        self._retry: deque = deque()           # requeued items; collector appends, FEEDER puts
        self._finished = threading.Event()     # set when the collector stops; releases the feeder
        self._recoveries = 0                   # requeues + respawns spent, against the stage rail
        self._requeues: Counter = Counter()    # specimen_id -> hand-backs, against the item cap
        self._subitems: Counter = Counter()    # specimen_id -> sub-items, snapshot before dispatch

    # -- entry point -------------------------------------------------------- #
    def run(self, project: Any, items: Iterable[WorkItem]) -> None:
        DeviceManager.ensure_cuda_libpath()
        items = list(items)

        done = set(project.db.done_ids(self.stage.key))          # snapshot BEFORE dispatch
        todo = [it for it in items if it.specimen_id not in done]
        # For a fanout stage (>1 WorkItem per specimen) mark a specimen done only after its LAST
        # sub-item checkpoints. For a normal stage every count is 1 -> mark done on its one item.
        self._remaining = Counter(it.specimen_id for it in todo)

        # No-work seeding: eligible specimens with NO WorkItem are marked done(no_work=1)
        # so per-image depends_on gating downstream never stalls waiting on them.
        have = {it.specimen_id for it in items}
        for sid in project.db.eligible_specimens(self.stage.key, self.stage.depends_on):
            if sid not in have and sid not in done:
                project.db.mark_image_done(sid, self.stage.key, no_work=True)

        if not todo:
            return

        devices = DeviceManager(self.cfg, self.stage).plan()
        devices = self._apply_worker_tiers(devices, len(todo))
        todo = self._longest_first(todo)
        if self._use_inprocess(devices, len(todo)):
            self._run_inprocess(project, todo, devices)
        else:
            self._run_pool(project, todo, devices)

    # -- scheduling --------------------------------------------------------- #
    def _apply_worker_tiers(self, devices: list[Device], n_items: int) -> list[Device]:
        """Cap the plan by the stage's STATIC worker policy, if it declares one.

        Ordering matters: this runs AFTER the VRAM planner, so the tier can only ever take
        workers away, never add them. That is what "validate the machine can handle it first"
        means in practice -- ask for the tier, but hand back the most the card can actually
        hold, so a 16-worker tier on a busy GPU quietly becomes however many fit.

        Trimming is round-robin across GPUs so a multi-GPU plan keeps its spread instead of
        piling the survivors onto whichever card happened to be listed first.
        """
        tiers = getattr(self.stage, "worker_tiers", ()) or ()
        if not tiers or not devices:
            return devices

        target = next((w for limit, w in tiers if limit is None or n_items < limit), tiers[-1][1])
        target = max(1, int(target))
        if target >= len(devices):
            log.info("%s: %d items -> tier allows %d worker(s); machine fits %d, using all %d",
                     self.stage.key, n_items, target, len(devices), len(devices))
            return devices

        by_gpu: dict[Any, list[Device]] = {}
        for d in devices:
            by_gpu.setdefault(getattr(d, "index", 0), []).append(d)
        order = sorted(by_gpu, key=lambda i: (-len(by_gpu[i]), i))     # roomiest card first
        trimmed: list[Device] = []
        while len(trimmed) < target:
            moved = False
            for idx in order:
                if by_gpu[idx] and len(trimmed) < target:
                    trimmed.append(by_gpu[idx].pop())
                    moved = True
            if not moved:
                break
        log.info("%s: %d items -> static tier caps this at %d worker(s) (VRAM would allow %d) %s",
                 self.stage.key, n_items, len(trimmed), len(devices),
                 {g: sum(1 for d in trimmed if getattr(d, "index", 0) == g)
                  for g in sorted({getattr(d, "index", 0) for d in trimmed})})
        return trimmed

    def _longest_first(self, todo: list[WorkItem]) -> list[WorkItem]:
        """Order work longest-first so the stage does not end waiting on one huge item.

        Workers PULL from a shared queue, so assignment is already dynamic and self-balancing.
        What a dynamic queue cannot fix is a big item dispatched LATE: once it starts, every
        other worker drains the queue and idles until that one finishes. Starting the biggest
        items first bounds that tail -- the classic LPT result, makespan <= (4/3 - 1/3m) x optimal.

        Item cost is taken as ``len(item.payload)`` (a leaf_segmenter item is one sheet carrying
        all of its leaf crops, and it segments them one at a time), overridable per stage via a
        ``work_weight(item)`` method.

        Measured on 114 herbarium sheets (median 12 leaf crops per sheet, mean 43, max 300):
        dispatching in specimen_id order costs 1.57x the ideal makespan at 16 workers, and LPT
        brings that to 1.00x -- roughly 36% off leaf_segmenter's wall time. The skew scales with
        worker count (only 1.10x at 4 workers), so this matters far more now that measured VRAM
        sizing hands these stages 16 workers instead of 4.
        """
        if len(todo) < 2:
            return todo
        if not bool(_get(_get(self.cfg, "compute", None), "longest_first", True)):
            return todo                                   # opt-out: restores raw specimen_id order
        weigh = getattr(self.stage, "work_weight", None)

        def cost(item: WorkItem) -> float:
            if callable(weigh):
                try:
                    return float(weigh(item))
                except Exception:  # noqa: BLE001 - a bad weight must never fail the stage
                    return 0.0
            payload = getattr(item, "payload", None)
            try:
                return float(len(payload))                    # sized payload -> number of sub-units
            except TypeError:
                return 0.0                                    # unsized (fanout items) -> no reorder

        weights = [cost(it) for it in todo]
        if len(set(weights)) < 2:                             # uniform cost -> ordering is a no-op
            return todo
        ordered = [it for _, it in sorted(zip(weights, todo), key=lambda p: -p[0])]
        log.info("%s: longest-first over %d items (max=%.0f, mean=%.1f sub-units/item)",
                 self.stage.key, len(todo), max(weights), sum(weights) / len(weights))
        return ordered

    # -- path selection ----------------------------------------------------- #
    def _use_inprocess(self, devices: list[Device], n_items: int) -> bool:
        """Prefer the simple, fork-free thread path for mock, CPU stages, or GPU-less plans.

        Exception: a compute-heavy CPU stage that opts into ``cpu_parallel="process"`` runs on the
        spawn process pool (``_run_pool``) so it bypasses the GIL -- but ONLY when there is real
        parallelism to be had (>1 planned worker), we are not mocking, AND the batch is big enough
        that the pool's spawn cost is worth paying (``n_items >= min_pool_items``). A tiny batch
        runs serially in-process instead."""
        if bool(_get(_get(self.cfg, "compute", None), "mock", False)):
            return True
        if self.stage.device_kind == "cpu":
            if (getattr(self.stage, "cpu_parallel", "thread") == "process"
                    and len(devices) > 1 and n_items >= self._min_pool_items()):
                return False                             # -> spawn process pool
            return True                                  # thread pool, or serial for a tiny process batch
        return all(d.kind == "cpu" for d in devices)

    def _min_pool_items(self) -> int:
        """Fewest WorkItems that justify spinning up the spawn process pool (spawn cost vs work).
        Measured by hardware_setup (``stages.<key>.min_pool_items`` / ``process_pool_min_items``)."""
        hw = _get(self.cfg, "hardware", None)
        st = None
        getter = getattr(hw, "stage", None) if hw is not None else None
        if callable(getter):
            try:
                st = getter(self.stage.key)
            except Exception:  # noqa: BLE001
                st = None
        m = _get(st, "min_pool_items", None)
        if m:
            return int(m)
        return int(_get(hw, "process_pool_min_items", 8) or 8)

    def _record_exec(self, mode: str, workers: int, n_items: int, serial_reason: Optional[str] = None) -> None:
        """Stash how this stage actually ran (+ why, if serial) for the timing report to read off the stage."""
        self.stage._exec_info = {
            "mode": mode, "workers": int(workers), "n_items": int(n_items),
            "min_pool_items": self._min_pool_items()
            if getattr(self.stage, "cpu_parallel", "thread") == "process" else None,
            "serial_reason": serial_reason,
        }

    # -- in-process path ---------------------------------------------------- #
    def _run_inprocess(self, project: Any, todo: list[WorkItem], devices: list[Device]) -> None:
        device = devices[0] if devices else Device("cpu")
        model = self.stage.build_model(device)

        # Pure-Python CPU stages (model is None) are thread-safe to run concurrently; the parent
        # thread stays the single writer. GPU/model stages run serially in-process. A process-pool
        # stage that lands here fell back for a small batch -> run it SERIALLY (never thread it: its
        # work is GIL-bound / thread-unsafe, which is exactly why it wanted processes).
        process_fallback = (self.stage.device_kind == "cpu"
                            and getattr(self.stage, "cpu_parallel", "thread") == "process")
        n_threads = 1 if process_fallback else (len(devices) if (model is None and len(devices) > 1) else 1)
        reason = None
        if n_threads <= 1:
            if process_fallback:
                reason = (f"batch of {len(todo)} < the {self._min_pool_items()}-item threshold, so the "
                          "spawn process pool would cost more than it saves — ran serially instead")
            elif model is not None:
                reason = "warm-loads a model and its per-item work is light, so it runs serially in-process"
            else:
                reason = "only one CPU worker was allocated"
        self._record_exec("serial" if n_threads <= 1 else "thread", n_threads, len(todo), reason)
        if n_threads <= 1:
            for item in todo:
                status, payload = self._safe_infer(item, model)
                self._checkpoint(project, item, status, payload)
            return

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = {pool.submit(self._safe_infer, item, model): item for item in todo}
            for future in as_completed(futures):
                item = futures[future]
                status, payload = future.result()
                self._checkpoint(project, item, status, payload)

    def _safe_infer(self, item: WorkItem, model: Any) -> tuple[str, Any]:
        try:
            return "ok", self.stage.infer(item, model)
        except Exception as exc:  # pragma: no cover - defensive per-item isolation
            return "err", repr(exc)

    def _warm_load_concurrency(self, n_workers: int) -> int:
        """How many workers may warm-load their model at the same time.

        This gate used to be ``Semaphore(1)``, from when per-worker VRAM was an unmeasured guess
        and simultaneous loads were a plausible OOM. Strict serialization costs ``n * load_time``
        before the pool reaches full speed, which quietly caps short modules: measured on 120
        sheets, ``ruler_classifier`` needed ~1.2s of actual inference across 16 workers but spent
        ~16s ramping, making it 3.5x SLOWER at 16 workers than at 4.

        It is safe to relax now because the allocator has already proven all ``n`` workers fit in
        this card's free VRAM at steady state; only the transient load spike is unaccounted for,
        so this stays bounded rather than unlimited. Tune with
        ``compute.vram.concurrent_warm_loads`` (0 or negative -> unbounded).
        """
        vram = _get(_get(self.cfg, "compute", None), "vram", {}) or {}
        configured = int(_get(vram, "concurrent_warm_loads", 4) or 4)
        if configured <= 0:
            return max(1, n_workers)
        return max(1, min(n_workers, configured))

    # -- process-pool path -------------------------------------------------- #
    def _run_pool(self, project: Any, todo: list[WorkItem], devices: list[Device]) -> None:
        self._devices = devices
        n = min(len(devices), len(todo))                          # don't spawn 6 workers for 1 image
        self._record_exec("process" if self.stage.device_kind == "cpu" else "gpu", n, len(todo))
        task_q: "mp.Queue" = _CTX.Queue(maxsize=2 * n)
        result_q: "mp.Queue" = _CTX.Queue(maxsize=4 * n)
        self._gate = _CTX.Semaphore(self._warm_load_concurrency(n))
        self._retry.clear()
        self._finished.clear()
        self._recoveries = 0
        self._requeues.clear()
        self._subitems = Counter(self._remaining)   # BEFORE dispatch starts decrementing it

        workers = [
            _CTX.Process(
                target=_worker,
                args=(devices[i], self.stage, task_q, result_q, self._gate,
                      os.getpid(), _pdeathsig_safe(self.stage.key)),
                name=f"{self.stage.key}-w{i}",
                daemon=True,
            )
            for i in range(n)
        ]
        for worker in workers:
            worker.start()

        feeder = threading.Thread(
            target=self._feed, args=(task_q, todo, n), name=f"{self.stage.key}-feed", daemon=True
        )
        feeder.start()

        try:
            self._collect(project, task_q, result_q, len(todo), workers)
        finally:
            # Order matters: releasing the feeder FIRST lets it hand out the STOPs, so workers
            # leave their pull loop on their own and _shutdown's join succeeds instead of having
            # to escalate to SIGTERM/SIGKILL.
            self._finished.set()
            feeder.join(timeout=15)
            self._shutdown(workers, task_q, result_q)

    def _feed(self, task_q: "mp.Queue", todo: list[WorkItem], n: int) -> None:
        """Sole producer for ``task_q`` -- the initial work AND every requeued item.

        The collector deliberately never puts here. It is the only consumer of ``result_q``, so a
        collector blocked on a full ``task_q`` stops draining results, the workers then block
        filling ``result_q``, and with nobody left pulling tasks neither queue can ever drain
        again. That cycle hung a 29-worker archival_detector pool for 39 hours on the Global
        Greening batch after a burst of CUDA OOM faults. This thread may block on ``put`` freely:
        it consumes nothing, so it cannot be half of that cycle.
        """
        for item in todo:
            while not self._finished.is_set():
                self._put_retries(task_q)                         # retries ride along with new work
                try:
                    task_q.put(item, timeout=0.5)                 # bounded -> stays shutdown-aware
                    break
                except queue.Full:
                    continue                                      # backpressure; workers are busy
            if self._finished.is_set():
                return

        # The STOPs go in LAST, and only once the collector has retired everything. A retry handed
        # back after them would sit BEHIND them in the queue: workers would take the STOPs, exit,
        # and nobody would be left to run it.
        while not self._finished.is_set():
            if not self._put_retries(task_q):
                self._finished.wait(0.25)
        self._put_retries(task_q)
        for _ in range(n):
            try:
                task_q.put(_STOP, timeout=5)
            except queue.Full:                                    # _shutdown escalates from here
                return

    def _put_retries(self, task_q: "mp.Queue") -> bool:
        """Move buffered retries onto ``task_q``; True if any moved. Feeder thread only."""
        moved = False
        while self._retry:
            try:
                task_q.put(self._retry[0], timeout=0.25)
            except queue.Full:
                break
            self._retry.popleft()                                 # sole consumer -> safe after put
            moved = True
        return moved

    def _recovery_budget(self, n_items: int) -> int:
        """Stage-wide spawn-storm rail. See the notes on _MIN_RECOVERY_BUDGET."""
        configured = _get(_get(self.cfg, "compute", None), "max_device_recoveries", None)
        if configured is not None:
            return max(0, int(configured))
        return max(_MIN_RECOVERY_BUDGET, n_items // 4)

    def _requeue_allowance(self, sid: int) -> int:
        """How many times one specimen may be handed back before it is failed.

        Scaled by that specimen's sub-item count so a fanout stage -- where 30 leaves share one
        specimen_id -- is not starved by a cap meant for a single sheet.
        """
        configured = _get(_get(self.cfg, "compute", None), "max_item_requeues", None)
        per_item = max(1, int(configured if configured is not None else _DEFAULT_MAX_ITEM_REQUEUES))
        return per_item * max(1, int(self._subitems.get(sid, 1)))

    def _collect(
        self,
        project: Any,
        task_q: "mp.Queue",
        result_q: "mp.Queue",
        expected: int,
        workers: list[Any],
    ) -> None:
        timeout = float(_get(_get(self.cfg, "compute", None), "liveness_timeout_s",
                             _DEFAULT_LIVENESS_TIMEOUT_S))
        budget = self._recovery_budget(expected)
        seen = 0
        while seen < expected:
            try:
                status, item, payload = result_q.get(timeout=timeout)
            except queue.Empty:
                if not any(w.is_alive() for w in workers):        # LIVENESS: dead pool, no hang
                    log.error(
                        "%s: all workers died with %d/%d items outstanding",
                        self.stage.key, expected - seen, expected,
                    )
                    break
                continue

            if status == "fatal_start":                           # a worker could not warm-load
                log.error("%s: worker failed to start: %s", self.stage.key, payload)
                # A warm-load that lost a VRAM race is usually transient, and without a
                # replacement the pool bleeds a worker per fault until nothing is left pulling.
                if self._recoveries < budget:
                    self._recoveries += 1
                    self._respawn(workers, task_q, result_q)
                elif not any(w.is_alive() for w in workers):
                    log.error("%s: recovery budget (%d) spent and no workers left; giving up "
                              "with %d/%d items outstanding",
                              self.stage.key, budget, expected - seen, expected)
                    break
                continue

            if status == "requeue":                               # worker hit a fatal device error
                sid = item.specimen_id
                allowance = self._requeue_allowance(sid)
                spent = self._recoveries >= budget
                if self._requeues[sid] >= allowance or spent:
                    # Retire it as an error so `seen` advances -- a requeued item never does, so
                    # retrying past this point could not terminate the loop.
                    log.error("%s: %s reached; failing %s instead of retrying: %s", self.stage.key,
                              f"stage recovery budget ({budget})" if spent
                              else f"item retry limit ({allowance})", sid, payload)
                    seen += 1
                    self._checkpoint(project, item, "err", payload)
                    if not spent:                                 # keep the pool staffed
                        self._recoveries += 1
                        self._respawn(workers, task_q, result_q)
                    continue
                self._requeues[sid] += 1
                self._recoveries += 1
                log.warning("%s: requeuing %s after device fault (attempt %d/%d): %s",
                            self.stage.key, sid, self._requeues[sid], allowance, payload)
                # Hand it to the FEEDER, never put it here -- see _feed for why the collector
                # must never become a producer on task_q.
                self._retry.append(item)
                self._respawn(workers, task_q, result_q)          # keep pool size; item retried
                continue

            seen += 1
            self._checkpoint(project, item, status, payload)

    def _respawn(self, workers: list[Any], task_q: "mp.Queue", result_q: "mp.Queue") -> None:
        """Start a replacement worker after a fatal device fault so the pool stays full."""
        if not self._devices:
            return
        device = self._devices[len(workers) % len(self._devices)]
        replacement = _CTX.Process(
            target=_worker,
            args=(device, self.stage, task_q, result_q, self._gate,
                  os.getpid(), _pdeathsig_safe(self.stage.key)),
            name=f"{self.stage.key}-w{len(workers)}",
            daemon=True,
        )
        replacement.start()
        workers.append(replacement)

    def _shutdown(self, workers: list[Any], task_q: "mp.Queue", result_q: "mp.Queue") -> None:
        for worker in workers:
            worker.join(timeout=30)
            if not worker.is_alive():
                continue
            worker.terminate()                                    # SIGTERM: no zombie GPU workers
            worker.join(timeout=10)
            if worker.is_alive():
                # A worker wedged inside a CUDA call does not act on SIGTERM, and leaving it
                # alive strands its context. SIGKILL is the only thing it cannot ignore.
                log.warning("%s: worker pid=%s ignored SIGTERM; sending SIGKILL to free its VRAM",
                            self.stage.key, worker.pid)
                worker.kill()
                worker.join(timeout=10)
        for q in (task_q, result_q):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

    # -- shared single-writer checkpoint ------------------------------------ #
    def _checkpoint(self, project: Any, item: WorkItem, status: str, payload: Any) -> None:
        """Persist one result AND mark the image done/error in ONE atomic transaction.

        For a fanout stage a specimen has several sub-items: each persists, but the specimen is
        marked done only after its LAST sub-item (``_remaining`` hits 0). A single failed sub-item
        is logged and counted (a bad leaf must not error the whole sheet), whereas a normal stage
        marks the specimen errored on failure as before."""
        sid = item.specimen_id
        fanout = getattr(self.stage, "fanout", False)
        with project.db.transaction():
            if status == "ok":
                self.stage.persist(project, item, payload)
            elif fanout:
                log.warning("%s: sub-item failed on specimen %s: %s", self.stage.key, sid, payload)
            else:
                project.db.mark_image_error(sid, self.stage.key, payload)
            if status == "ok" or fanout:                          # count this (sub-)item as retired
                self._remaining[sid] -= 1
                if self._remaining[sid] <= 0:
                    project.db.mark_image_done(sid, self.stage.key)
        if status == "err" and bool(_get(self.cfg, "fail_fast", False)):
            raise StageError(f"{self.stage.key} failed on specimen {item.specimen_id}: {payload}")
