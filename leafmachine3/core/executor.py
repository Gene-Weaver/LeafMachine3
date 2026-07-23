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
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

from leafmachine3.core.device import Device
from leafmachine3.core.stage import PipelineStage, WorkItem

log = logging.getLogger("leafmachine3.executor")

_CTX = mp.get_context("spawn")
_DEFAULT_LIVENESS_TIMEOUT_S = 120.0
_DEFAULT_PER_WORKER_VRAM_MB = 8000.0
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

        hw = self._hw_stage()
        if hw is not None and not bool(_get(self.cfg, "force_reprobe", False)):
            per_gpu = int(_get(hw, "workers_per_gpu", 1) or 1)
            return [Device("cuda", g) for g in gpus for _ in range(max(1, per_gpu))]

        return self._vram_plan(gpus)

    @staticmethod
    def ensure_cuda_libpath() -> None:
        """Put the venv's bundled NVIDIA ``.so`` dirs on ``LD_LIBRARY_PATH``.

        onnxruntime-gpu silently drops to CPU if those libs are not on the loader path
        at process start. Spawn children inherit this process's environment, so exporting
        it here (before any worker starts) is enough -- no re-exec required.
        """
        try:
            import nvidia  # type: ignore
        except Exception:
            return
        try:
            base = Path(nvidia.__file__).resolve().parent
        except Exception:
            return
        libdirs = [str(p) for p in base.glob("*/lib") if p.is_dir()]
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

        plan: list[Device] = []
        for gpu in gpus:
            free = _free_vram_mb(gpu)
            if free and per_worker > 0:
                fit = int((free * safety - reserve) // per_worker)
            else:
                fit = max_per_gpu
            count = max(1, min(fit, max_per_gpu))
            plan.extend([Device("cuda", gpu)] * count)
        return plan or [Device("cpu")]

    def _per_worker_vram_mb(self, vram: Any) -> float:
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
            return float(configured)
        peak = _get(self._hw_stage(), "peak_vram_mb", None)
        if peak:
            return float(peak)
        return _DEFAULT_PER_WORKER_VRAM_MB


# --------------------------------------------------------------------------- #
# Worker process body (must be top-level for spawn picklability)
# --------------------------------------------------------------------------- #
def _worker(
    device: Device,
    stage: PipelineStage,
    task_q: "mp.Queue",
    result_q: "mp.Queue",
    load_gate: Any,
) -> None:
    """Persistent worker: pin the GPU, warm-load ONCE (staggered behind ``load_gate`` so N
    workers do not warm-load simultaneously and OOM), then pull-and-infer until ``_STOP``."""
    if device.kind == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device.index)  # before any torch/ort import
    try:
        with load_gate:
            model = stage.build_model(device)
    except Exception as exc:  # cannot even load the model on this device
        result_q.put(("fatal_start", None, repr(exc)))
        return

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

    # -- entry point -------------------------------------------------------- #
    def run(self, project: Any, items: Iterable[WorkItem]) -> None:
        DeviceManager.ensure_cuda_libpath()
        items = list(items)

        done = set(project.db.done_ids(self.stage.key))          # snapshot BEFORE dispatch
        todo = [it for it in items if it.specimen_id not in done]

        # No-work seeding: eligible specimens with NO WorkItem are marked done(no_work=1)
        # so per-image depends_on gating downstream never stalls waiting on them.
        have = {it.specimen_id for it in items}
        for sid in project.db.eligible_specimens(self.stage.key, self.stage.depends_on):
            if sid not in have and sid not in done:
                project.db.mark_image_done(sid, self.stage.key, no_work=True)

        if not todo:
            return

        devices = DeviceManager(self.cfg, self.stage).plan()
        if self._use_inprocess(devices):
            self._run_inprocess(project, todo, devices)
        else:
            self._run_pool(project, todo, devices)

    # -- path selection ----------------------------------------------------- #
    def _use_inprocess(self, devices: list[Device]) -> bool:
        """Prefer the simple, fork-free path for mock, CPU stages, or GPU-less plans."""
        if bool(_get(_get(self.cfg, "compute", None), "mock", False)):
            return True
        if self.stage.device_kind == "cpu":
            return True
        return all(d.kind == "cpu" for d in devices)

    # -- in-process path ---------------------------------------------------- #
    def _run_inprocess(self, project: Any, todo: list[WorkItem], devices: list[Device]) -> None:
        device = devices[0] if devices else Device("cpu")
        model = self.stage.build_model(device)

        # Pure-Python CPU stages (model is None) are thread-safe to run concurrently; the
        # parent thread stays the single writer. GPU/model stages run serially in-process.
        n_threads = len(devices) if (model is None and len(devices) > 1) else 1
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

    # -- process-pool path -------------------------------------------------- #
    def _run_pool(self, project: Any, todo: list[WorkItem], devices: list[Device]) -> None:
        self._devices = devices
        n = min(len(devices), len(todo))                          # don't spawn 6 workers for 1 image
        task_q: "mp.Queue" = _CTX.Queue(maxsize=2 * n)
        result_q: "mp.Queue" = _CTX.Queue(maxsize=4 * n)
        self._gate = _CTX.Semaphore(1)                            # stagger warm-loads

        workers = [
            _CTX.Process(
                target=_worker,
                args=(devices[i], self.stage, task_q, result_q, self._gate),
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
            self._shutdown(workers, task_q, result_q)
            feeder.join(timeout=5)

    def _feed(self, task_q: "mp.Queue", todo: list[WorkItem], n: int) -> None:
        for item in todo:
            task_q.put(item)                                      # blocks at maxsize -> backpressure
        for _ in range(n):
            task_q.put(_STOP)

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
                if not any(w.is_alive() for w in workers):
                    break
                continue

            if status == "requeue":                               # worker hit a fatal device error
                log.warning("%s: requeuing %s after device fault: %s",
                            self.stage.key, item.specimen_id, payload)
                task_q.put(item)
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
            args=(device, self.stage, task_q, result_q, self._gate),
            name=f"{self.stage.key}-w{len(workers)}",
            daemon=True,
        )
        replacement.start()
        workers.append(replacement)

    def _shutdown(self, workers: list[Any], task_q: "mp.Queue", result_q: "mp.Queue") -> None:
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.terminate()                                # no zombie GPU workers
        for q in (task_q, result_q):
            try:
                q.cancel_join_thread()
                q.close()
            except Exception:
                pass

    # -- shared single-writer checkpoint ------------------------------------ #
    def _checkpoint(self, project: Any, item: WorkItem, status: str, payload: Any) -> None:
        """Persist one result AND mark the image done/error in ONE atomic transaction."""
        with project.db.transaction():
            if status == "ok":
                self.stage.persist(project, item, payload)
                project.db.mark_image_done(item.specimen_id, self.stage.key)
            else:
                project.db.mark_image_error(item.specimen_id, self.stage.key, payload)
        if status == "err" and bool(_get(self.cfg, "fail_fast", False)):
            raise StageError(f"{self.stage.key} failed on specimen {item.specimen_id}: {payload}")
