"""leafmachine3.server.metrics -- machine performance sampling for the LM3 app.

The bottom panel of the LM3 app is a live machine monitor (CPU, RAM, GPU, VRAM, disk, net).
This module is the only thing that talks to the hardware for it: ONE background sampler thread
keeps a bounded ROLLING WINDOW (~5 minutes at 2 Hz) of every counter, and the HTTP layer just
serves the ring buffer. Nothing here blocks, and nothing here is per-request expensive -- a
browser polling at 2 Hz costs the same as a browser polling at 0.1 Hz.

Reuse, not reinvention: the sampling conventions come straight from
``leafmachine3.setup.timing.UtilizationSampler`` -- the same NVML acquisition idiom, the same
``psutil.cpu_percent`` priming, the same daemon-thread + ``Event`` stop, and above all the same
BASELINE-DELTA convention (RAM/VRAM already resident before an LM3 run started are subtracted, so
a figure reads as "what THIS run added"). :class:`MetricsSampler` is deliberately duck-type
compatible with ``UtilizationSampler`` -- it exposes ``.samples``, ``.baseline_ram_mb``,
``.baseline_vram_mb`` and ``.window_stats(t0, t1)`` with identical semantics -- so the timing
report's helpers can read this sampler directly instead of a second thread sampling the same
counters. (Caveat: this sampler's window is bounded, so ``window_stats`` over a module that
finished more than ``window_s`` ago returns what is still in the ring; the timing REPORT keeps its
own unbounded sampler for that reason.)

Dependency-light by design: ``psutil`` and ``pynvml`` are each behind ``try/except``. A missing
psutil blanks the CPU/RAM/disk/net fields, a missing (or driverless) NVML blanks the GPU list --
neither raises, and the shape of every payload stays identical so the app never has to branch.

Public surface (module-level shims delegate to the lazily started singleton):
    describe_machine()  -> static facts: CPU model, core counts, RAM, GPU names/VRAM, limits
    snapshot()          -> the newest sample (adds the per-core array, which is NOT retained)
    history(...)        -> the rolling window, columnar, for initial plot fill / incremental polls
    workers()           -> live LM3 worker PROCESSES (per-worker CPU / RSS / VRAM)
    sse_frames(...)     -> a typed Server-Sent-Events generator (hello + metrics + workers)
    router(...)         -> an optional FastAPI APIRouter the integrator can mount
    stop()              -> shut the sampler thread down
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import platform
import socket
import threading
import time
from collections import deque
from typing import Any, Iterator, Optional

log = logging.getLogger("leafmachine3.server.metrics")

# Defaults: 2 Hz over a 5-minute window == 600 retained samples. Both are env-overridable so a
# slow machine can drop to 1 Hz without a code change.
DEFAULT_INTERVAL_S = float(os.environ.get("LM3_METRICS_INTERVAL_S", "0.5"))
DEFAULT_WINDOW_S = float(os.environ.get("LM3_METRICS_WINDOW_S", "300"))

_MB = 1024.0 * 1024.0
_NVML_TEMPERATURE_GPU = 0
_NVML_CLOCK_SM = 1


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _r(value: Any, ndigits: int = 1) -> Optional[float]:
    """Round for the wire, preserving ``None`` (a null means "this machine cannot report it")."""
    if value is None:
        return None
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return None


def _txt(value: Any) -> Optional[str]:
    """NVML returns ``bytes`` on older nvidia-ml-py builds and ``str`` on newer ones."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _import_psutil() -> Any:
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:  # noqa: BLE001 - optional dependency
        return None


def _import_pynvml() -> Any:
    """Import pynvml with its deprecation FutureWarning muted (the shim is what ships here)."""
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml  # type: ignore

        return pynvml
    except Exception:  # noqa: BLE001 - no GPU tooling on this machine
        return None


def _cpu_model() -> Optional[str]:
    """The marketing CPU name. ``platform.processor()`` is just "x86_64" on Linux, so read
    /proc/cpuinfo first and fall back to whatever the platform will admit to."""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or None


# --------------------------------------------------------------------------- #
# The sampler
# --------------------------------------------------------------------------- #
class MetricsSampler:
    """Background thread sampling machine utilization into a bounded ring buffer.

    Thread-safe: every mutation of the ring / latest-per-core / worker table happens under
    ``self._lock``, and every public reader copies out under the same lock. Memory is bounded by
    construction -- the ring is a ``deque(maxlen=...)`` and the ONLY unbounded-looking field
    (per-core CPU%, which is 64 floats on this box) is kept for the LATEST sample only, because
    the core heatmap is instantaneous and 600x64 floats would be pure waste.
    """

    def __init__(self, interval_s: float = DEFAULT_INTERVAL_S,
                 window_s: float = DEFAULT_WINDOW_S) -> None:
        self.interval = max(0.1, float(interval_s))
        self.window_s = max(self.interval * 4, float(window_s))
        self.capacity = max(8, int(round(self.window_s / self.interval)))

        self._lock = threading.RLock()
        self._ring: deque[dict] = deque(maxlen=self.capacity)
        self._per_core: list[float] = []
        self._seq = 0
        self._started_at = 0.0
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._psutil = _import_psutil()
        self._nvml: Any = None
        self._handles: list[tuple[int, Any]] = []      # (nvml index, handle)
        self._gpu_static: list[dict] = []
        self._machine: Optional[dict] = None

        # rate counters (bytes/ms deltas need the previous read)
        self._prev_t: float = 0.0
        self._prev_disk: Any = None
        self._prev_net: Any = None

        # baseline-delta convention, identical to setup.timing.UtilizationSampler
        self.baseline_ram_mb = 0.0
        self.baseline_vram_mb = 0.0
        self.baseline_at = 0.0

        # LM3 worker processes (spawn children of the process that runs the modules)
        self._worker_root_pid = os.getpid()
        self._proc_cache: dict[int, Any] = {}          # pid -> psutil.Process (cpu_percent deltas)
        self._workers: dict = {"t": 0.0, "root_pid": self._worker_root_pid, "n_workers": 0,
                               "workers": [], "host": None}
        self._worker_every = max(1, int(round(1.0 / self.interval)))   # rescan children ~1 Hz
        self._ticks = 0

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> "MetricsSampler":
        """Initialize the backends, capture the baseline, and start the sampling thread."""
        if self._thread is not None and self._thread.is_alive():
            return self

        if self._psutil is not None:
            try:
                self._psutil.cpu_percent(interval=None)              # prime the CPU% deltas
                self._psutil.cpu_percent(interval=None, percpu=True)
            except Exception:  # noqa: BLE001
                pass

        self._nvml = _import_pynvml()
        if self._nvml is not None:
            try:
                self._nvml.nvmlInit()
                count = int(self._nvml.nvmlDeviceGetCount())
                self._handles = [(i, self._nvml.nvmlDeviceGetHandleByIndex(i)) for i in range(count)]
            except Exception:  # noqa: BLE001 - no driver / no GPU -> CPU-only monitoring
                self._nvml, self._handles = None, []
        self._gpu_static = self._read_gpu_static()

        self._started_at = time.time()
        self.reset_baseline()
        self._prime_rates()
        self._tick()                                                 # one sample immediately
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._loop, name="lm3-metrics", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop the sampling thread (idempotent; safe to call from atexit)."""
        self._stop_evt.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def reset_baseline(self) -> dict:
        """Re-capture the RAM/VRAM baseline. Call this when an LM3 run STARTS so every
        ``*_delta_mb`` reads as memory that run added, not memory the desktop was already using."""
        self.baseline_ram_mb = self._read_ram_used_mb()
        self.baseline_vram_mb = self._read_vram_used_mb()
        self.baseline_at = time.time()
        return {"ram_used_mb": _r(self.baseline_ram_mb), "vram_used_mb": _r(self.baseline_vram_mb),
                "captured_at": self.baseline_at}

    def set_worker_root(self, pid: int) -> None:
        """Point worker discovery at another process.

        Default is *this* process, which is correct for the server (it runs the modules in a
        thread of its own process, so the executor's spawn workers are its children). Set this
        when the modules run somewhere else (e.g. a CLI run the app merely observes)."""
        with self._lock:
            self._worker_root_pid = int(pid)
            self._proc_cache.clear()

    # -- sampling loop ------------------------------------------------------ #
    def _loop(self) -> None:
        next_t = time.monotonic() + self.interval
        while True:
            delay = next_t - time.monotonic()
            if self._stop_evt.wait(delay if delay > 0 else 0):
                return
            next_t += self.interval
            if next_t < time.monotonic():          # fell behind under load -> resync, never spiral
                next_t = time.monotonic() + self.interval
            try:
                self._tick()
            except Exception:  # noqa: BLE001 - a monitor must never take the app down
                log.debug("metrics tick failed", exc_info=True)

    def _tick(self) -> None:
        now = time.time()
        cpu_pct, per_core, freq_mhz, load = self._read_cpu()
        ram_used, ram_total, ram_pct, swap_used, swap_total, swap_pct = self._read_mem()
        gpus = self._read_gpus()
        disk, net = self._read_io(now)

        gpu_used = sum(g["mem_used_mb"] for g in gpus if g.get("mem_used_mb") is not None)
        gpu_total = sum(g["mem_total_mb"] for g in gpus if g.get("mem_total_mb") is not None)
        gpu_utils = [g["util_pct"] for g in gpus if g.get("util_pct") is not None]

        record = {
            "t": round(now, 3),
            "cpu_pct": _r(cpu_pct),
            "cpu_freq_mhz": _r(freq_mhz, 0),
            "load_avg": load,
            "ram_used_mb": _r(ram_used, 0),
            "ram_total_mb": _r(ram_total, 0),
            "ram_pct": _r(ram_pct),
            "ram_delta_mb": _r(max(0.0, ram_used - self.baseline_ram_mb) if ram_used is not None else None, 0),
            "swap_used_mb": _r(swap_used, 0),
            "swap_total_mb": _r(swap_total, 0),
            "swap_pct": _r(swap_pct),
            "gpus": gpus,
            "gpu_util_pct": _r(sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None,
            "gpu_mem_used_mb": _r(gpu_used, 0) if gpus else None,
            "gpu_mem_total_mb": _r(gpu_total, 0) if gpus else None,
            "gpu_mem_pct": _r(100.0 * gpu_used / gpu_total) if gpu_total else None,
            "vram_delta_mb": _r(max(0.0, gpu_used - self.baseline_vram_mb), 0) if gpus else None,
            "disk_read_mbps": _r(disk[0], 2),
            "disk_write_mbps": _r(disk[1], 2),
            "disk_busy_pct": _r(disk[2]),
            "net_recv_mbps": _r(net[0], 3),
            "net_sent_mbps": _r(net[1], 3),
            "n_procs": self._read_n_procs(),
        }

        with self._lock:
            self._seq += 1
            record["seq"] = self._seq
            record["uptime_s"] = round(now - self._started_at, 1)
            self._ring.append(record)
            self._per_core = per_core
            self._ticks += 1
            due = (self._ticks % self._worker_every) == 0

        if due:
            try:
                self._refresh_workers(now)
            except Exception:  # noqa: BLE001
                log.debug("worker refresh failed", exc_info=True)

    # -- readers ------------------------------------------------------------ #
    def _read_cpu(self) -> tuple[Optional[float], list[float], Optional[float], Optional[list]]:
        if self._psutil is None:
            return None, [], None, None
        try:
            total = float(self._psutil.cpu_percent(interval=None))
        except Exception:  # noqa: BLE001
            total = None
        try:
            per_core = [round(float(v), 1) for v in self._psutil.cpu_percent(interval=None, percpu=True)]
        except Exception:  # noqa: BLE001
            per_core = []
        try:
            freq = self._psutil.cpu_freq()
            freq_mhz = float(freq.current) if freq else None
        except Exception:  # noqa: BLE001 - cpu_freq is unavailable in many containers
            freq_mhz = None
        try:
            load = [round(float(v), 2) for v in self._psutil.getloadavg()]
        except Exception:  # noqa: BLE001 - not on every platform
            load = None
        return total, per_core, freq_mhz, load

    def _read_ram_used_mb(self) -> float:
        if self._psutil is None:
            return 0.0
        try:
            return float(self._psutil.virtual_memory().used) / _MB
        except Exception:  # noqa: BLE001
            return 0.0

    def _read_vram_used_mb(self) -> float:
        total = 0.0
        for _, handle in self._handles:
            try:
                total += float(self._nvml.nvmlDeviceGetMemoryInfo(handle).used) / _MB
            except Exception:  # noqa: BLE001
                pass
        return total

    def _read_mem(self) -> tuple:
        if self._psutil is None:
            return (None,) * 6
        try:
            vm = self._psutil.virtual_memory()
            ram_used, ram_total, ram_pct = float(vm.used) / _MB, float(vm.total) / _MB, float(vm.percent)
        except Exception:  # noqa: BLE001
            ram_used = ram_total = ram_pct = None
        try:
            sw = self._psutil.swap_memory()
            swap_used, swap_total, swap_pct = float(sw.used) / _MB, float(sw.total) / _MB, float(sw.percent)
        except Exception:  # noqa: BLE001
            swap_used = swap_total = swap_pct = None
        return ram_used, ram_total, ram_pct, swap_used, swap_total, swap_pct

    def _read_gpus(self) -> list[dict]:
        """One dict per GPU. EVERY NVML query is individually guarded: a card that will not report
        fan speed still reports utilization, and the field just comes back null."""
        out: list[dict] = []
        for index, handle in self._handles:
            row: dict[str, Any] = {"index": index, "util_pct": None, "mem_util_pct": None,
                                   "mem_used_mb": None, "mem_total_mb": None, "mem_pct": None,
                                   "temp_c": None, "power_w": None, "power_limit_w": None,
                                   "sm_clock_mhz": None, "fan_pct": None}
            try:
                rates = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                row["util_pct"] = _r(rates.gpu)
                row["mem_util_pct"] = _r(rates.memory)
            except Exception:  # noqa: BLE001
                pass
            try:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                used, total = float(mem.used) / _MB, float(mem.total) / _MB
                row["mem_used_mb"] = _r(used, 0)
                row["mem_total_mb"] = _r(total, 0)
                row["mem_pct"] = _r(100.0 * used / total) if total else None
            except Exception:  # noqa: BLE001
                pass
            try:
                row["temp_c"] = _r(self._nvml.nvmlDeviceGetTemperature(handle, _NVML_TEMPERATURE_GPU), 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                row["power_w"] = _r(float(self._nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0)
            except Exception:  # noqa: BLE001
                pass
            try:
                row["power_limit_w"] = _r(
                    float(self._nvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000.0, 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                row["sm_clock_mhz"] = _r(self._nvml.nvmlDeviceGetClockInfo(handle, _NVML_CLOCK_SM), 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                row["fan_pct"] = _r(self._nvml.nvmlDeviceGetFanSpeed(handle), 0)
            except Exception:  # noqa: BLE001 - passively cooled / datacenter cards have no fan
                pass
            out.append(row)
        return out

    def _prime_rates(self) -> None:
        self._prev_t = time.time()
        if self._psutil is None:
            return
        try:
            self._prev_disk = self._psutil.disk_io_counters()
        except Exception:  # noqa: BLE001
            self._prev_disk = None
        try:
            self._prev_net = self._psutil.net_io_counters()
        except Exception:  # noqa: BLE001
            self._prev_net = None

    def _read_io(self, now: float) -> tuple[tuple, tuple]:
        """Disk (read MB/s, write MB/s, busy %) and net (recv MB/s, sent MB/s) as counter deltas."""
        blank = ((None, None, None), (None, None))
        if self._psutil is None:
            return blank
        dt = now - self._prev_t
        if dt <= 0:
            return blank
        self._prev_t = now

        read = write = busy = None
        try:
            disk = self._psutil.disk_io_counters()
            if disk is not None and self._prev_disk is not None:
                read = max(0.0, (disk.read_bytes - self._prev_disk.read_bytes) / _MB / dt)
                write = max(0.0, (disk.write_bytes - self._prev_disk.write_bytes) / _MB / dt)
                prev_busy = getattr(self._prev_disk, "busy_time", None)
                now_busy = getattr(disk, "busy_time", None)
                if prev_busy is not None and now_busy is not None:
                    # busy_time is milliseconds summed over devices, so it can exceed wall time on
                    # a multi-disk box -- clamp to 100% rather than report a nonsense number.
                    busy = min(100.0, max(0.0, (now_busy - prev_busy) / (dt * 1000.0) * 100.0))
            self._prev_disk = disk if disk is not None else self._prev_disk
        except Exception:  # noqa: BLE001
            pass

        recv = sent = None
        try:
            net = self._psutil.net_io_counters()
            if net is not None and self._prev_net is not None:
                recv = max(0.0, (net.bytes_recv - self._prev_net.bytes_recv) / _MB / dt)
                sent = max(0.0, (net.bytes_sent - self._prev_net.bytes_sent) / _MB / dt)
            self._prev_net = net if net is not None else self._prev_net
        except Exception:  # noqa: BLE001
            pass

        return (read, write, busy), (recv, sent)

    def _read_n_procs(self) -> Optional[int]:
        if self._psutil is None:
            return None
        try:
            return len(self._psutil.pids())
        except Exception:  # noqa: BLE001
            return None

    # -- LM3 worker processes ----------------------------------------------- #
    def _gpu_proc_memory(self) -> dict[int, tuple[float, int]]:
        """``pid -> (VRAM MB, GPU index holding the most of it)`` from NVML's compute-process list."""
        out: dict[int, tuple[float, int]] = {}
        for index, handle in self._handles:
            try:
                procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:  # noqa: BLE001
                continue
            for proc in procs:
                used = getattr(proc, "usedGpuMemory", None)
                if not used:
                    continue
                mb = float(used) / _MB
                pid = int(proc.pid)
                prev = out.get(pid)
                if prev is None:
                    out[pid] = (mb, index)
                else:
                    out[pid] = (prev[0] + mb, index if mb > prev[0] else prev[1])
        return out

    def _refresh_workers(self, now: float) -> None:
        """Rebuild the per-worker table.

        WHAT COUNTS AS A WORKER: LM3's executor runs GPU stages and ``cpu_parallel="process"``
        stages on a spawn pool, so each worker is a real child PROCESS and is visible here.
        Thread-pool stages have NO children -- their workers are threads of this process, which is
        why ``host.n_threads`` is reported too: that is the only signal a threaded module gives.
        The multiprocessing resource tracker is a child but not a worker, so it is filtered out.
        """
        if self._psutil is None:
            return
        try:
            root = self._psutil.Process(self._worker_root_pid)
        except Exception:  # noqa: BLE001 - root vanished
            with self._lock:
                self._workers = {"t": now, "root_pid": self._worker_root_pid, "n_workers": 0,
                                 "workers": [], "host": None}
            return

        try:
            children = root.children(recursive=True)
        except Exception:  # noqa: BLE001
            children = []

        live = {child.pid for child in children}
        for pid in list(self._proc_cache):
            if pid not in live:
                self._proc_cache.pop(pid, None)

        gpu_mem = self._gpu_proc_memory()
        rows: list[dict] = []
        for child in children:
            pid = child.pid
            proc = self._proc_cache.get(pid)
            if proc is None:
                proc = child
                self._proc_cache[pid] = proc
                try:
                    proc.cpu_percent(interval=None)      # prime; first read is always 0.0
                except Exception:  # noqa: BLE001
                    pass
            try:
                cmdline = " ".join(proc.cmdline())
                if "resource_tracker" in cmdline:        # bookkeeping child, not a worker
                    continue
                info = {
                    "pid": pid,
                    "cpu_pct": _r(proc.cpu_percent(interval=None)),
                    "rss_mb": _r(proc.memory_info().rss / _MB, 0),
                    "n_threads": proc.num_threads(),
                    "status": proc.status(),
                    "started_at": round(proc.create_time(), 3),
                    "age_s": round(now - proc.create_time(), 1),
                    "spawned": "multiprocessing.spawn" in cmdline,
                }
            except Exception:  # noqa: BLE001 - raced with process exit
                continue
            vram = gpu_mem.get(pid)
            info["gpu_index"] = vram[1] if vram else None
            info["vram_mb"] = _r(vram[0], 0) if vram else None
            rows.append(info)

        # Positional labels ordered by start time: the executor names workers "<stage>-w<i>" in
        # spawn order but that name never reaches the OS, so w0..wN here is start-order, which is
        # the same order for a pool that starts together. It is a display label, not an identity.
        rows.sort(key=lambda r: (r["started_at"], r["pid"]))
        for i, row in enumerate(rows):
            row["label"] = f"w{i}"

        host = None
        try:
            host = {
                "pid": root.pid,
                "cpu_pct": _r(root.cpu_percent(interval=None)),
                "rss_mb": _r(root.memory_info().rss / _MB, 0),
                "n_threads": root.num_threads(),
            }
        except Exception:  # noqa: BLE001
            pass

        with self._lock:
            self._workers = {"t": round(now, 3), "root_pid": self._worker_root_pid,
                             "n_workers": len(rows), "workers": rows, "host": host}

    # -- static machine description ----------------------------------------- #
    def _read_gpu_static(self) -> list[dict]:
        out: list[dict] = []
        for index, handle in self._handles:
            row: dict[str, Any] = {"index": index, "name": None, "vram_total_mb": None,
                                   "uuid": None, "compute_capability": None, "power_limit_w": None}
            try:
                row["name"] = _txt(self._nvml.nvmlDeviceGetName(handle))
            except Exception:  # noqa: BLE001
                pass
            try:
                row["vram_total_mb"] = _r(float(self._nvml.nvmlDeviceGetMemoryInfo(handle).total) / _MB, 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                row["uuid"] = _txt(self._nvml.nvmlDeviceGetUUID(handle))
            except Exception:  # noqa: BLE001
                pass
            try:
                major, minor = self._nvml.nvmlDeviceGetCudaComputeCapability(handle)
                row["compute_capability"] = f"{major}.{minor}"
            except Exception:  # noqa: BLE001
                pass
            try:
                row["power_limit_w"] = _r(
                    float(self._nvml.nvmlDeviceGetEnforcedPowerLimit(handle)) / 1000.0, 0)
            except Exception:  # noqa: BLE001
                pass
            out.append(row)
        return out

    def describe_machine(self) -> dict:
        """Static hardware facts. Computed once and cached -- none of this changes at runtime."""
        if self._machine is not None:
            machine = dict(self._machine)
            machine["sampler"] = self._sampler_info()
            machine["baseline"] = {"ram_used_mb": _r(self.baseline_ram_mb, 0),
                                   "vram_used_mb": _r(self.baseline_vram_mb, 0),
                                   "captured_at": self.baseline_at}
            return machine

        psu = self._psutil
        cores_logical = cores_physical = None
        freq_min = freq_max = None
        ram_total = swap_total = None
        boot_time = None
        driver = None
        if psu is not None:
            try:
                cores_logical = psu.cpu_count(logical=True)
                cores_physical = psu.cpu_count(logical=False)
            except Exception:  # noqa: BLE001
                pass
            try:
                freq = psu.cpu_freq()
                if freq:
                    freq_min, freq_max = _r(freq.min, 0), _r(freq.max, 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                ram_total = _r(psu.virtual_memory().total / _MB, 0)
                swap_total = _r(psu.swap_memory().total / _MB, 0)
            except Exception:  # noqa: BLE001
                pass
            try:
                boot_time = round(psu.boot_time(), 0)
            except Exception:  # noqa: BLE001
                pass
        if self._nvml is not None:
            try:
                driver = _txt(self._nvml.nvmlSystemGetDriverVersion())
            except Exception:  # noqa: BLE001
                pass

        self._machine = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "boot_time": boot_time,
            "cpu": {
                "model": _cpu_model(),
                "arch": platform.machine(),
                "cores_physical": cores_physical,
                "cores_logical": cores_logical,
                "freq_min_mhz": freq_min,
                "freq_max_mhz": freq_max,
            },
            "memory": {"ram_total_mb": ram_total, "swap_total_mb": swap_total},
            "gpus": self._gpu_static,
            "n_gpus": len(self._gpu_static),
            "gpu_driver": driver,
            "available": {"psutil": psu is not None, "nvml": self._nvml is not None},
        }
        machine = dict(self._machine)
        machine["sampler"] = self._sampler_info()
        machine["baseline"] = {"ram_used_mb": _r(self.baseline_ram_mb, 0),
                               "vram_used_mb": _r(self.baseline_vram_mb, 0),
                               "captured_at": self.baseline_at}
        return machine

    def _sampler_info(self) -> dict:
        with self._lock:
            n = len(self._ring)
        return {"interval_s": self.interval, "window_s": self.window_s,
                "capacity": self.capacity, "n_samples": n, "running": self.running,
                "started_at": self._started_at}

    # -- public reads ------------------------------------------------------- #
    def snapshot(self) -> dict:
        """The newest sample, plus the per-core array (retained for the latest sample only)."""
        with self._lock:
            if not self._ring:
                return {"t": time.time(), "seq": 0, "ready": False, "cpu_per_core_pct": [], "gpus": []}
            latest = dict(self._ring[-1])
            latest["cpu_per_core_pct"] = list(self._per_core)
        latest["ready"] = True
        return latest

    def history(self, since: Optional[float] = None, max_points: Optional[int] = None) -> dict:
        """The rolling window in COLUMNAR form -- parallel arrays, one value per sample.

        ``since`` (unix seconds) returns only newer samples, so the app can fill its plots once and
        then poll incrementally. ``max_points`` stride-downsamples for a narrow canvas; the newest
        sample is always the last element so a live plot never lags its own tail.
        Per-core CPU% is deliberately absent -- the core heatmap is instantaneous, see snapshot().
        """
        with self._lock:
            rows = list(self._ring)
            gpu_indices = [g["index"] for g in self._gpu_static]

        if since is not None:
            rows = [r for r in rows if r["t"] > since]
        if max_points and len(rows) > max_points > 0:
            stride = (len(rows) + max_points - 1) // max_points
            kept = rows[::stride]
            if kept and rows and kept[-1] is not rows[-1]:
                kept.append(rows[-1])
            rows = kept

        scalar_keys = ("cpu_pct", "cpu_freq_mhz", "ram_used_mb", "ram_pct", "ram_delta_mb",
                       "swap_used_mb", "swap_pct", "gpu_util_pct", "gpu_mem_used_mb",
                       "gpu_mem_pct", "vram_delta_mb", "disk_read_mbps", "disk_write_mbps",
                       "disk_busy_pct", "net_recv_mbps", "net_sent_mbps")
        gpu_keys = ("util_pct", "mem_pct", "mem_used_mb", "temp_c", "power_w", "sm_clock_mhz")

        series = {key: [r.get(key) for r in rows] for key in scalar_keys}
        gpus = []
        for pos, index in enumerate(gpu_indices):
            entry: dict[str, Any] = {"index": index}
            for key in gpu_keys:
                entry[key] = [
                    (r["gpus"][pos].get(key) if pos < len(r.get("gpus") or []) else None)
                    for r in rows
                ]
            gpus.append(entry)

        return {
            "t": [r["t"] for r in rows],
            "seq": [r["seq"] for r in rows],
            "n": len(rows),
            "interval_s": self.interval,
            "window_s": self.window_s,
            "t_first": rows[0]["t"] if rows else None,
            "t_last": rows[-1]["t"] if rows else None,
            "series": series,
            "gpus": gpus,
        }

    def workers(self) -> dict:
        """Live LM3 worker PROCESSES (see :meth:`_refresh_workers` for what qualifies)."""
        with self._lock:
            table = self._workers
            return {"t": table["t"], "root_pid": table["root_pid"],
                    "n_workers": table["n_workers"],
                    "workers": [dict(w) for w in table["workers"]],
                    "host": dict(table["host"]) if table["host"] else None}

    # -- setup.timing.UtilizationSampler compatibility ---------------------- #
    @property
    def samples(self) -> list[tuple[float, float, float, float, float]]:
        """``(t, cpu%, gpu%, vram_mb, ram_mb)`` tuples -- the exact tuple shape
        ``setup.timing.UtilizationSampler.samples`` produces, so the timing helpers can read this
        sampler instead of running a second thread over the same counters."""
        with self._lock:
            rows = list(self._ring)
        return [(r["t"], r["cpu_pct"] or 0.0, r["gpu_util_pct"] or 0.0,
                 r["gpu_mem_used_mb"] or 0.0, r["ram_used_mb"] or 0.0) for r in rows]

    def window_stats(self, t0: float, t1: float) -> dict:
        """CPU/GPU utilization + RAM/VRAM deltas-over-baseline across ``[t0, t1]``.

        Same keys and same baseline-delta convention as ``UtilizationSampler.window_stats``. Note
        the window here is bounded: a module that finished longer ago than ``window_s`` has aged
        out of the ring and its stats are gone."""
        rows = [s for s in self.samples if t0 <= s[0] <= t1] or [s for s in self.samples if t0 <= s[0]]
        if not rows:
            return {"cpu_mean": None, "cpu_peak": None, "gpu_mean": None, "gpu_peak": None,
                    "ram_avg": None, "ram_max": None, "vram_avg": None, "vram_max": None}
        cpu = [r[1] for r in rows]
        gpu = [r[2] for r in rows]
        vram_d = [max(0.0, r[3] - self.baseline_vram_mb) for r in rows]
        ram_d = [max(0.0, r[4] - self.baseline_ram_mb) for r in rows]
        return {"cpu_mean": sum(cpu) / len(cpu), "cpu_peak": max(cpu),
                "gpu_mean": sum(gpu) / len(gpu), "gpu_peak": max(gpu),
                "ram_avg": sum(ram_d) / len(ram_d), "ram_max": max(ram_d),
                "vram_avg": sum(vram_d) / len(vram_d), "vram_max": max(vram_d)}


# --------------------------------------------------------------------------- #
# Module-level singleton
# --------------------------------------------------------------------------- #
_SAMPLER: Optional[MetricsSampler] = None
_SAMPLER_LOCK = threading.Lock()


def get_sampler() -> MetricsSampler:
    """The process-wide sampler, started on first use."""
    global _SAMPLER
    with _SAMPLER_LOCK:
        if _SAMPLER is None:
            _SAMPLER = MetricsSampler().start()
            atexit.register(_SAMPLER.stop)
        elif not _SAMPLER.running:
            _SAMPLER.start()
        return _SAMPLER


def describe_machine() -> dict:
    return get_sampler().describe_machine()


def snapshot() -> dict:
    return get_sampler().snapshot()


def history(since: Optional[float] = None, max_points: Optional[int] = None) -> dict:
    return get_sampler().history(since=since, max_points=max_points)


def workers() -> dict:
    return get_sampler().workers()


def reset_baseline() -> dict:
    return get_sampler().reset_baseline()


def set_worker_root(pid: int) -> None:
    get_sampler().set_worker_root(pid)


def stop() -> None:
    """Stop the singleton sampler (server shutdown)."""
    global _SAMPLER
    with _SAMPLER_LOCK:
        if _SAMPLER is not None:
            _SAMPLER.stop()


# --------------------------------------------------------------------------- #
# Server-Sent Events
# --------------------------------------------------------------------------- #
def _frame(kind: str, **payload: Any) -> str:
    """One SSE ``data:`` frame. Every frame is a typed envelope so the panel can switch on
    ``type`` instead of guessing which fields a frame happens to carry."""
    return "data: " + json.dumps({"type": kind, **payload}) + "\n\n"


def hello_frame() -> str:
    """The first frame of a stream: machine description + the whole rolling window, so a single
    connection fills the plots and then keeps them live (no separate priming request)."""
    sampler = get_sampler()
    return _frame("hello", machine=sampler.describe_machine(), history=sampler.history())


def sse_frames(interval_s: Optional[float] = None, *, workers_every: int = 4,
               max_seconds: float = 86400.0) -> Iterator[str]:
    """Blocking generator of typed SSE frames: ``hello``, then a ``metrics`` frame per new sample
    and a ``workers`` frame every ``workers_every`` metrics frames.

    This is the synchronous form (handy for tests and non-async callers); the FastAPI route runs
    the same sequence on the event loop with ``asyncio.sleep`` so it never blocks it."""
    sampler = get_sampler()
    period = float(interval_s or sampler.interval)
    yield hello_frame()
    deadline = time.time() + max_seconds
    last_seq, tick = -1, 0
    while time.time() < deadline:
        point = sampler.snapshot()
        if point.get("seq") != last_seq:
            last_seq = point.get("seq")
            yield _frame("metrics", snapshot=point)
            tick += 1
            if workers_every and tick % workers_every == 0:
                yield _frame("workers", workers=sampler.workers())
        time.sleep(period)


# --------------------------------------------------------------------------- #
# Optional FastAPI router (the integrator mounts this; app.py is not edited here)
# --------------------------------------------------------------------------- #
def router(dependencies: Optional[list] = None) -> Any:
    """Build the ``/v1/metrics`` APIRouter.

    ``fastapi`` is imported lazily, exactly like ``leafmachine3.server.app.create_app``, so a base
    install without the ``server`` extra can still import this module. Pass the app's auth
    dependency through ``dependencies`` (``[Depends(require_token)]``) to protect the routes."""
    import asyncio

    from fastapi import APIRouter
    from fastapi.responses import StreamingResponse

    api = APIRouter(prefix="/v1/metrics", tags=["metrics"], dependencies=dependencies or [])

    @api.get("")
    async def get_snapshot() -> dict:
        return snapshot()

    @api.get("/machine")
    async def get_machine() -> dict:
        return describe_machine()

    @api.get("/history")
    async def get_history(since: Optional[float] = None, max_points: Optional[int] = None) -> dict:
        return history(since=since, max_points=max_points)

    @api.get("/workers")
    async def get_workers() -> dict:
        return workers()

    @api.get("/stream")
    async def stream(interval: Optional[float] = None, workers_every: int = 4) -> "StreamingResponse":
        sampler = get_sampler()
        period = float(interval or sampler.interval)

        async def gen() -> Any:
            # Same frame sequence as sse_frames, but paced with asyncio.sleep: the reads are all
            # ring-buffer copies (microseconds), so this streams without ever blocking the loop.
            yield hello_frame()
            last_seq, tick = -1, 0
            while True:
                point = sampler.snapshot()
                if point.get("seq") != last_seq:
                    last_seq = point.get("seq")
                    yield _frame("metrics", snapshot=point)
                    tick += 1
                    if workers_every and tick % workers_every == 0:
                        yield _frame("workers", workers=sampler.workers())
                await asyncio.sleep(period)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return api


__all__ = ["MetricsSampler", "get_sampler", "describe_machine", "snapshot", "history",
           "workers", "reset_baseline", "set_worker_root", "stop", "sse_frames", "router"]
