"""LM3 run-timing profiler: time each module, sample compute utilization, and emit a CSV + a
guide-styled HTML timeline into ``reports/Timing/``.

Enabled per run with ``timing.enabled: true`` (LM3_settings.yaml). It is deliberately coarse -- one
row per pipeline MODULE (they run sequentially, so module granularity is what matters) -- plus, when
a stage reports them, a few in-module component timings (e.g. the Reporter's per-overlay seconds).

Three things per module:
  * wall time + share of the run,
  * compute utilization: mean/peak system CPU% and, for GPU stages, the run GPU%
    (sampled by a background thread and correlated to each module's wall-clock window),
  * throughput in the module's natural unit (images, ruler crops, or leaves) AND images/s.

Dependency-light: psutil / pynvml are optional; missing either just blanks that column.
"""
from __future__ import annotations

import csv
import html
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("leafmachine3.timing")


# --------------------------------------------------------------------------- #
# Utilization sampler
# --------------------------------------------------------------------------- #
class UtilizationSampler:
    """Background thread sampling system CPU% and (optional) GPU% every ``interval`` seconds."""

    def __init__(self, interval: float = 0.25, gpu_indices: Optional[list[int]] = None) -> None:
        self.interval = max(0.05, float(interval))
        self.gpu_indices = gpu_indices
        # per sample: (t, cpu%, gpu%, gpu_mem_used_mb, ram_used_mb, own_vram_mb)
        self.samples: list[tuple[float, float, float, float, float, float]] = []
        self.n_cores = os.cpu_count() or 1
        # baselines captured BEFORE the pipeline runs, so per-module deltas exclude background jobs
        self.baseline_ram_mb = 0.0
        self.baseline_vram_mb = 0.0
        self._pynvml = None
        self._handles: list = []
        self._pid_cache: set = set()
        self._pid_cache_t = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _read_ram_mb(self) -> float:
        try:
            import psutil
            return float(psutil.virtual_memory().used) / 1e6
        except Exception:  # noqa: BLE001
            return 0.0

    def _read_vram_mb(self) -> float:
        tot = 0.0
        for h in self._handles:
            try:
                tot += float(self._pynvml.nvmlDeviceGetMemoryInfo(h).used) / 1e6
            except Exception:  # noqa: BLE001
                pass
        return tot

    def _read_own_vram_mb(self) -> float:
        """VRAM charged to THIS process tree, via NVML per-process accounting.

        Device-total usage minus a baseline is only honest when nothing else on the card
        moves; a shared GPU breaks that assumption. Summing the compute processes that are
        us or our descendants gives a figure that is ours no matter what else is resident --
        which is what worker sizing has to divide free VRAM by.
        """
        if self._pynvml is None:
            return 0.0
        pids = self._own_pids()
        tot = 0.0
        for h in self._handles:
            try:
                procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(h)
            except Exception:  # noqa: BLE001
                continue
            for pr in procs:
                if pr.pid in pids and getattr(pr, "usedGpuMemory", None):
                    tot += float(pr.usedGpuMemory) / 1e6
        return tot

    def _own_pids(self) -> set:
        """PIDs of this process and its descendants (worker pools are spawned children)."""
        now = time.time()
        if now - self._pid_cache_t < 1.0 and self._pid_cache:
            return self._pid_cache
        pids = {os.getpid()}
        try:
            import psutil
            for child in psutil.Process().children(recursive=True):
                pids.add(child.pid)
        except Exception:  # noqa: BLE001 - no psutil -> own pid only
            pass
        self._pid_cache, self._pid_cache_t = pids, now
        return pids

    def _read_gpu_util(self) -> float:
        utils = []
        for h in self._handles:
            try:
                utils.append(float(self._pynvml.nvmlDeviceGetUtilizationRates(h).gpu))
            except Exception:  # noqa: BLE001
                pass
        return (sum(utils) / len(utils)) if utils else 0.0

    def start(self) -> None:
        try:
            import psutil
            psutil.cpu_percent(interval=None)                         # prime the CPU% delta baseline
        except Exception:  # noqa: BLE001
            pass
        try:
            import pynvml
            pynvml.nvmlInit()
            n = pynvml.nvmlDeviceGetCount()
            idx = self.gpu_indices if self.gpu_indices else list(range(n))
            self._pynvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in idx if 0 <= i < n]
        except Exception:  # noqa: BLE001 - no GPU / no pynvml -> CPU-only timing
            self._pynvml, self._handles = None, []
        # baseline snapshot (background RAM + whatever already sits on the run's GPU(s))
        self.baseline_ram_mb = self._read_ram_mb()
        self.baseline_vram_mb = self._read_vram_mb()
        self._thread = threading.Thread(target=self._loop, name="lm3-util-sampler", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            import psutil
        except Exception:  # noqa: BLE001
            psutil = None
        while not self._stop.wait(self.interval):
            cpu = float(psutil.cpu_percent(interval=None)) if psutil else 0.0
            self.samples.append((time.time(), cpu, self._read_gpu_util(),
                                 self._read_vram_mb(), self._read_ram_mb(),
                                 self._read_own_vram_mb()))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def window_stats(self, t0: float, t1: float) -> dict:
        """CPU/GPU utilization + RAM/VRAM DELTAS-over-baseline aggregated over samples in ``[t0, t1]``.

        A module faster than one sample interval lands between samples. Falling back to "every
        sample after t0" would credit it with the peak of everything that ran AFTERWARDS -- which
        made short modules report a neighbour's VRAM verbatim -- so the fallback is the single
        nearest sample instead.
        """
        rows = [s for s in self.samples if t0 <= s[0] <= t1]
        if not rows and self.samples:
            rows = [min(self.samples, key=lambda s: min(abs(s[0] - t0), abs(s[0] - t1)))]
        if not rows:
            return {"cpu_mean": None, "cpu_peak": None, "gpu_mean": None, "gpu_peak": None,
                    "ram_avg": None, "ram_max": None, "vram_avg": None, "vram_max": None,
                    "vram_own_max": None}
        cpu = [r[1] for r in rows]
        gpu = [r[2] for r in rows]
        # deltas over the pre-run baseline, floored at 0 (a module never "frees" below baseline usefully)
        vram_d = [max(0.0, r[3] - self.baseline_vram_mb) for r in rows]
        ram_d = [max(0.0, r[4] - self.baseline_ram_mb) for r in rows]
        return {
            "cpu_mean": sum(cpu) / len(cpu), "cpu_peak": max(cpu),
            "gpu_mean": (sum(gpu) / len(gpu)), "gpu_peak": max(gpu),
            "ram_avg": sum(ram_d) / len(ram_d), "ram_max": max(ram_d),
            "vram_avg": sum(vram_d) / len(vram_d), "vram_max": max(vram_d),
            # NOT a delta: VRAM actually charged to our processes, so a busy card cannot skew it
            "vram_own_max": max((r[5] for r in rows), default=0.0),
        }


# --------------------------------------------------------------------------- #
# Per-module metadata: throughput unit + the DB count that feeds it
# --------------------------------------------------------------------------- #
# unit -> which count in _counts() the module's throughput is measured against
_STAGE_UNIT: dict[str, str] = {
    "mp_conversion_factor": "images", "archival_detector": "images", "plant_detector": "images",
    "specimen_segmenter": "images", "phenology_detector": "images", "ruler_classifier": "ruler_crops",
    "ruler_cf": "ruler_crops", "leaf_segmenter": "images", "morphology": "leaves",
    "landmark_detector": "leaves", "landmark_measurements": "leaves", "leaf_orientation": "leaves",
    "petiole_width": "leaves", "metric_grounding": "leaves", "reporter": "images", "ect": "leaf_ect",
    "bilateral_symmetry": "leaves", "momocs": "leaf_momocs",
}
_UNIT_LABEL = {"images": "image", "ruler_crops": "ruler crop", "leaves": "leaf", "leaf_ect": "leaf",
               "leaf_momocs": "leaf"}


def _counts(db: Any) -> dict[str, int]:
    def n(sql: str) -> int:
        try:
            return int(list(db.conn.execute(sql))[0][0])
        except Exception:  # noqa: BLE001
            return 0
    return {
        "images": n("SELECT COUNT(*) FROM specimen"),
        "ruler_crops": n("SELECT COUNT(*) FROM ruler_classification"),
        "leaves": n("SELECT COUNT(*) FROM leaf_segmentation"),
        "leaf_ect": n("SELECT COUNT(*) FROM leaf_ect"),
        "leaf_momocs": n("SELECT COUNT(*) FROM leaf_momocs"),
    }


_MODE_LABEL = {"gpu": "gpu", "process": "cpu·process", "thread": "cpu·thread", "serial": "cpu·serial"}


def _stage_exec(stage: Any, cfg: Any) -> tuple[str, int, Optional[str]]:
    """(execution kind, workers ACTUALLY allocated, serial-reason note) for this run.

    Prefers what the executor recorded on the stage (``_exec_info``); falls back to the profile."""
    info = getattr(stage, "_exec_info", None) if stage is not None else None
    if info:
        mode = info.get("mode", "serial")
        workers = int(info.get("workers", 1) or 1)
        note = info.get("serial_reason") if mode == "serial" else None
        return _MODE_LABEL.get(mode, mode), workers, note
    # fallback: no per-run record -> derive intent from config/profile
    dk = getattr(stage, "device_kind", "cpu")
    hw = _hw_stage(cfg, stage.key)
    if dk == "cuda":
        return "gpu", int(_get(hw, "workers_per_gpu", 1) or 1), None
    par = getattr(stage, "cpu_parallel", "thread")
    w = int(_get(hw, "workers", 1) or 1)
    return ("cpu·process" if par == "process" else "cpu·thread"), w, None


def _hw_stage(cfg: Any, key: str) -> Any:
    hw = getattr(cfg, "hardware", None)
    if hw is None:
        return None
    getter = getattr(hw, "stage", None)
    try:
        return getter(key) if callable(getter) else None
    except Exception:  # noqa: BLE001
        return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    try:
        if hasattr(obj, "get"):
            v = obj.get(key, default)
        elif hasattr(obj, key):
            v = getattr(obj, key)
        else:
            v = obj[key]
    except Exception:  # noqa: BLE001
        return default
    return default if v is None else v


def collect_rows(stages: list, timer: Any, sampler: UtilizationSampler, db: Any, cfg: Any) -> list[dict]:
    """One dict per timed module: time, share, utilization, throughput."""
    counts = _counts(db)
    total = sum(timer.times.values()) or 1.0
    rows: list[dict] = []
    order = {s.key: i for i, s in enumerate(stages)}
    for key, secs in sorted(timer.times.items(), key=lambda kv: order.get(kv[0], 999)):
        stage = next((s for s in stages if s.key == key), None)
        win = timer.windows.get(key, (0.0, 0.0))
        util = sampler.window_stats(*win) if sampler else {}
        unit_key = _STAGE_UNIT.get(key, "images")
        n_items = counts.get(unit_key, 0)
        n_images = counts.get("images", 0)
        kind, workers, serial_note = _stage_exec(stage, cfg) if stage else ("cpu", 1, None)
        rows.append({
            "stage": key, "seconds": secs, "share": 100.0 * secs / total,
            "exec": kind, "workers": workers, "serial_note": serial_note,
            "cpu_mean": util.get("cpu_mean"), "cpu_peak": util.get("cpu_peak"),
            "gpu_mean": util.get("gpu_mean"), "gpu_peak": util.get("gpu_peak"),
            "ram_avg": util.get("ram_avg"), "ram_max": util.get("ram_max"),
            "vram_avg": util.get("vram_avg"), "vram_max": util.get("vram_max"),
            "vram_own_max": util.get("vram_own_max"),
            "unit": _UNIT_LABEL.get(unit_key, "image"), "n_items": n_items,
            "unit_per_s": (n_items / secs if secs > 0 else 0.0),
            "images_per_s": (n_images / secs if secs > 0 else 0.0),
            "subsections": timer.subsections.get(key, {}),
        })
    return rows


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["stage", "exec", "workers", "seconds", "share_pct", "cpu_util_mean_pct",
            "cpu_util_peak_pct", "ram_delta_avg_mb", "ram_delta_max_mb", "gpu_util_mean_pct",
            "gpu_util_peak_pct", "vram_delta_avg_mb", "vram_delta_max_mb", "vram_own_max_mb",
            "unit", "n_items", "throughput_per_unit_s", "throughput_images_s", "note"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["stage"], r["exec"], r["workers"], f"{r['seconds']:.3f}", f"{r['share']:.1f}",
                _f(r["cpu_mean"], 1), _f(r["cpu_peak"], 1), _f(r.get("ram_avg"), 0), _f(r.get("ram_max"), 0),
                _f(r["gpu_mean"], 1), _f(r["gpu_peak"], 1), _f(r.get("vram_avg"), 0), _f(r.get("vram_max"), 0),
                _f(r.get("vram_own_max"), 0),
                r["unit"], r["n_items"], f"{r['unit_per_s']:.3f}", f"{r['images_per_s']:.3f}",
                r.get("serial_note") or "",
            ])
        # component sub-timings (e.g. Reporter). These are CPU-time SUMMED across the module's worker
        # threads, so they total more than its wall time; share_pct is each component's slice of that.
        for r in rows:
            subs = r.get("subsections") or {}
            subtotal = sum(subs.values()) or 1.0
            for name, secs in subs.items():
                w.writerow([f"{r['stage']}::{name}", "sub", "", f"{secs:.3f}",
                            f"{100.0 * secs / subtotal:.1f}"] + [""] * 12 + ["CPU-time share"])


def _f(v: Any, nd: int) -> str:
    return "" if v is None else f"{v:.{nd}f}"


def _mb_fmt(mb: Any) -> str:
    """A memory amount in MB as a compact string: GB (1 dp) at >=1 GB, else whole MB."""
    if mb is None:
        return "-"
    mb = float(mb)
    if mb < 1:
        return "0"
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def _hardware_summary(cfg: Any) -> Optional[dict]:
    """The tuned values hardware_setup wrote (machine + per-stage sizing), for the report's context
    section. Returns None when no profile is bound."""
    hw = getattr(cfg, "hardware", None)
    if hw is None:
        return None
    stages = _get(hw, "stages", {}) or {}
    items = list(stages.items()) if hasattr(stages, "items") else []
    proc = next((v for _, v in items if _get(v, "cpu_parallel", None) == "process"), None)
    io = next((v for _, v in items if _get(v, "io_bound", None)), None)
    dev = _get(_get(cfg, "compute", None), "devices", "auto")
    chosen: set = set()
    if isinstance(dev, (list, tuple)):
        for d in dev:
            try:
                chosen.add(int(d))
            except (TypeError, ValueError):
                pass
    gpus = _get(hw, "gpus", []) or []
    gpu_list = [{
        "index": _get(g, "index"), "name": _get(g, "name", "GPU"),
        "free_gb": round((_get(g, "free_vram_mb", 0) or 0) / 1024),
        "total_gb": round((_get(g, "total_vram_mb", 0) or 0) / 1024),
        "used": (_get(g, "index") in chosen) if chosen else None,
    } for g in gpus]
    return {
        "cpu_cores": _get(hw, "cpu_cores"), "ram_gb": _get(hw, "ram_gb"), "gpu_list": gpu_list,
        "provider": _get(hw, "provider"), "precision": _get(hw, "precision"),
        "io_workers": _get(hw, "io_workers"),
        "proc_workers": _get(proc, "workers"), "spawn_s": _get(proc, "spawn_overhead_s"),
        "disk_mbps": _get(io, "disk_write_mbps"), "generated_at": _get(hw, "generated_at"),
    }


def write_timing_reports(stages: list, timer: Any, sampler: UtilizationSampler, project: Any,
                         out_dir: Path, cfg: Any, *, run_name: str = "run") -> Path:
    """Write ``timing.csv`` + ``timing.html`` into ``out_dir`` and return the HTML path."""
    rows = collect_rows(stages, timer, sampler, project.db, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "timing.csv")
    html_path = out_dir / "timing.html"
    html_path.write_text(_render_html(rows, timer, sampler, run_name, _hardware_summary(cfg)),
                         encoding="utf-8")
    log.info("timing report -> %s", html_path)
    return html_path


# --------------------------------------------------------------------------- #
# HTML (style matched to LM3_Ruler_Segmentation/guide/how_CF_works.html)
# --------------------------------------------------------------------------- #
_EXEC_COLOR = {"gpu": "var(--acc2)", "cpu·process": "var(--acc3)", "cpu·thread": "var(--acc)",
               "cpu·serial": "var(--dim)"}


def _render_html(rows: list[dict], timer: Any, sampler: UtilizationSampler, run_name: str,
                 hardware: Optional[dict] = None) -> str:
    total = sum(r["seconds"] for r in rows) or 1.0
    n_images = next((r["n_items"] for r in rows if r["stage"] == "reporter"), 0) \
        or next((r["n_items"] for r in rows if r["unit"] == "image"), 0)
    slowest = max(rows, key=lambda r: r["seconds"]) if rows else None
    cpu_all = [s[1] for s in (sampler.samples if sampler else [])]
    gpu_all = [s[2] for s in (sampler.samples if sampler else [])]
    peak_ram = max((r.get("ram_max") or 0 for r in rows), default=0)
    peak_vram = max((r.get("vram_max") or 0 for r in rows), default=0)
    stats = [
        ("total wall", f"{total:.1f}s"),
        ("images", str(n_images)),
        ("modules timed", str(len(rows))),
        ("slowest", f"{slowest['stage']} · {slowest['seconds']:.1f}s" if slowest else "-"),
        ("peak CPU", f"{max(cpu_all):.0f}%" if cpu_all else "-"),
        ("peak GPU", f"{max(gpu_all):.0f}%" if gpu_all else "-"),
        ("peak RAM Δ", _mb_fmt(peak_ram)),
        ("peak VRAM Δ", _mb_fmt(peak_vram)),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div></div>'
        for l, v in stats)

    # --- horizontal proportional timeline (the sequential module sections) ---
    segs = []
    for r in rows:
        w = 100.0 * r["seconds"] / total
        color = _EXEC_COLOR.get(r["exec"], "var(--acc)")
        label = f'{r["stage"]}<span>{r["seconds"]:.1f}s</span>' if w > 6 else ""
        segs.append(
            f'<div class="seg" style="width:{w:.3f}%;--c:{color}" '
            f'title="{html.escape(r["stage"])} — {r["seconds"]:.2f}s ({r["share"]:.1f}%)">{label}</div>')
    gantt = f'<div class="gantt">{"".join(segs)}</div>'
    legend = ('<div class="legend">'
              '<span><i style="background:var(--acc2)"></i>GPU</span>'
              '<span><i style="background:var(--acc3)"></i>CPU · process pool</span>'
              '<span><i style="background:var(--acc)"></i>CPU · thread pool</span></div>')

    # --- why any module ran serially (one worker) ---
    serial_mods = [r for r in rows if r["exec"] == "cpu·serial"]
    pool_note = ""
    if serial_mods:
        items = "".join(
            f'<li><b>{html.escape(r["stage"])}</b> — {html.escape(r["serial_note"] or "ran serially")}</li>'
            for r in serial_mods)
        pool_note = (
            '<div class="note"><b>Why some modules ran serially (one worker):</b>'
            f'<ul>{items}</ul>'
            'A module runs serially only when parallelizing would not help: a process-pool stage whose '
            'batch is below its measured min-items threshold (spawn cost &gt; gain, threshold = spawn '
            'overhead ÷ per-item cost, from <span class="mono">hardware_setup</span>), or a stage that '
            'warm-loads a model and does only light per-item work. Everything else runs on a thread pool '
            '(I/O / GIL-releasing work) or a spawn process pool (GIL-bound work).</div>')

    # --- detail table ---
    head = [
        ("module", "the pipeline stage — modules run sequentially"),
        ("exec", "how the stage ran: gpu · cpu·process (spawn pool) · cpu·thread (thread pool) · "
                 "cpu·serial (in-process, one worker)"),
        ("workers", "processes / threads actually allocated to this stage this run"),
        ("time", "wall-clock seconds the stage took"),
        ("share", "this stage's share of total run wall time"),
        ("CPU mean/peak", "system CPU utilization during the stage (100% = all logical cores), mean / peak"),
        ("RAM Δ avg/max", "system RAM used ABOVE the pre-run baseline during the stage (avg / max), so "
                          "background processes are excluded"),
        ("GPU", "mean GPU utilization over the run's GPU(s) during the stage"),
        ("VRAM Δ avg/max", "GPU memory used ABOVE the pre-run baseline on the run's GPU(s) during the "
                           "stage (avg / max)"),
        ("throughput", "items processed per second in the stage's natural unit (image / ruler crop / leaf)"),
        ("images/s", "parent images processed per second"),
    ]
    thead = "".join(f'<th title="{html.escape(t, quote=True)}">{h}</th>' for h, t in head)

    def _mem_cell(avg, mx):
        if avg is None and mx is None:
            return "-"
        return f'{_mb_fmt(avg)} <span class="dim">/ {_mb_fmt(mx)}</span>'

    trows = []
    for r in rows:
        bar = f'<div class="mini"><span style="width:{r["share"]:.1f}%;background:{_EXEC_COLOR.get(r["exec"], "var(--acc)")}"></span></div>'
        cpu = f'{_f(r["cpu_mean"], 0)}% <span class="dim">/ {_f(r["cpu_peak"], 0)}%</span>' if r["cpu_mean"] is not None else "-"
        gpu = f'{_f(r["gpu_mean"], 0)}%' if (r["gpu_mean"] or 0) > 1 else "-"
        ram_c = _mem_cell(r.get("ram_avg"), r.get("ram_max"))
        vram_c = _mem_cell(r.get("vram_avg"), r.get("vram_max")) if (r.get("vram_max") or 0) > 1 else "-"
        exec_cell = html.escape(r["exec"])
        if r.get("serial_note"):
            exec_cell += f' <span class="serialflag" title="{html.escape(r["serial_note"], quote=True)}">serial</span>'
        trows.append(
            "<tr>"
            f'<td class="mono">{html.escape(r["stage"])}</td>'
            f'<td class="mono dim">{exec_cell}</td>'
            f'<td class="num">{r["workers"]}</td>'
            f'<td class="num">{r["seconds"]:.2f}</td>'
            f'<td class="num">{r["share"]:.1f}%{bar}</td>'
            f'<td class="num">{cpu}</td><td class="num">{ram_c}</td>'
            f'<td class="num">{gpu}</td><td class="num">{vram_c}</td>'
            f'<td class="num">{r["unit_per_s"]:.2f} <span class="dim">{html.escape(r["unit"])}/s</span></td>'
            f'<td class="num">{r["images_per_s"]:.2f}</td>'
            "</tr>")
        subs = r.get("subsections") or {}
        subtotal = sum(subs.values()) or 1.0
        for name, secs in subs.items():
            trows.append(
                '<tr class="sub">'
                f'<td class="mono dim">&nbsp;&nbsp;↳ {html.escape(name)}</td>'
                '<td class="dim" style="font-size:11px">CPU-time</td><td></td>'
                f'<td class="num dim">{secs:.2f}</td>'
                f'<td class="num dim">{100.0 * secs / subtotal:.1f}%</td>'
                '<td></td><td></td><td></td><td></td><td></td><td></td></tr>')
    table = (f"<div class='tblwrap'><table><thead><tr>{thead}</tr></thead>"
             f"<tbody>{''.join(trows)}</tbody></table></div>")

    # --- machine & tuning context (what hardware_setup measured) ---
    hw_html = ""
    if hardware:
        hstats = [
            (hardware.get("cpu_cores"), "logical cores"),
            (f'{hardware["ram_gb"]} GB' if hardware.get("ram_gb") else None, "RAM"),
            (hardware.get("provider"), "ONNX provider"),
            (hardware.get("precision"), "precision"),
            (hardware.get("io_workers"), "io workers (thread stages)"),
            (hardware.get("proc_workers"), "process-pool workers (CPU knee)"),
            (f'{hardware["spawn_s"]}s' if hardware.get("spawn_s") is not None else None, "spawn overhead"),
            (f'{hardware["disk_mbps"]} MB/s' if hardware.get("disk_mbps") else None, "disk write (reporter cap)"),
        ]
        cards = "".join(
            f'<div class="stat"><div class="v">{html.escape(str(v))}</div>'
            f'<div class="l">{html.escape(l)}</div></div>'
            for v, l in hstats if v not in (None, "None", ""))
        # each GPU as its own tile with free / total VRAM in whole GB + a fill bar
        gtiles = ""
        for g in hardware.get("gpu_list", []):
            free, total = g.get("free_gb") or 0, g.get("total_gb") or 0
            pct = (100.0 * free / total) if total else 0.0
            used = g.get("used")
            cls = "gputile used" if used else "gputile"
            badge = '<span class="ubadge">used this run</span>' if used else (
                '<span class="ubadge idle">idle</span>' if used is False else "")
            gtiles += (
                f'<div class="{cls}">'
                f'<div class="gname">GPU {html.escape(str(g.get("index")))} · '
                f'{html.escape(str(g.get("name")))}{badge}</div>'
                f'<div class="gvram"><b>{free}</b> / {total} GB free</div>'
                f'<div class="gbar"><span style="width:{pct:.0f}%"></span></div>'
                '</div>')
        gpu_block = f'<h3 class="gpuh">GPUs</h3><div class="gpugrid">{gtiles}</div>' if gtiles else ""
        gen = hardware.get("generated_at")
        hw_html = (
            '<h2>Machine &amp; tuning</h2>'
            '<p class="hint">What the LM3 profiler (<span class="mono">hardware_setup</span>) measured on '
            'this machine and used to size every module' + (f' — profiled {html.escape(str(gen))}' if gen else '')
            + '.</p>'
            f'<div class="grid g3">{cards}</div>'
            + gpu_block +
            '<p class="hint">GPU stages run a spawn worker pool per GPU, sized like building blocks: '
            'each card&rsquo;s <b>own live free VRAM</b> divided by one worker&rsquo;s measured cost, so a card '
            'already holding another job takes fewer workers than an idle one and the counts are '
            'deliberately uneven. Per-worker cost is <b>measured</b> by '
            '<span class="mono">--calibrate</span> (a real one-worker run over example images, sampling '
            'NVML per-process VRAM); without it LM3 falls back to a built-in estimate that runs high and '
            'therefore under-fills the card. CPU stages: light ones use a thread pool at <span class="mono">io '
            'workers</span>; the compute-heavy <span class="mono">ruler_cf</span>/<span class="mono">ect</span> '
            'use a spawn <b>process pool</b> sized to the measured CPU knee (threads are GIL-bound for them); '
            'the disk-write-bound <span class="mono">reporter</span> is capped at the disk-write knee. Re-tune '
            'with <span class="mono">python -m leafmachine3.setup --force --calibrate</span>.</p>')

    b_ram = getattr(sampler, "baseline_ram_mb", 0.0) if sampler else 0.0
    b_vram = getattr(sampler, "baseline_vram_mb", 0.0) if sampler else 0.0
    baseline_note = (
        '<p class="hint">RAM&nbsp;Δ / VRAM&nbsp;Δ are the increase <b>over the pre-run baseline</b> '
        f'(RAM {html.escape(_mb_fmt(b_ram))}, VRAM {html.escape(_mb_fmt(b_vram))}) captured before the '
        'pipeline started, so processes already resident (incl. other GPUs&rsquo; jobs) are excluded — a '
        'module&rsquo;s figure is what IT added.</p>')

    return _HTML_SHELL.format(run=html.escape(run_name), stats=stat_html, gantt=gantt,
                              legend=legend, pool_note=pool_note, table=table,
                              baseline_note=baseline_note, hardware=hw_html)


_HTML_SHELL = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LM3 — run timing · {run}</title>
<style>
:root{{--bg:#101012;--panel:#191a1d;--panel2:#1f2024;--ink:#e8e8ea;--mute:#9ca3af;--dim:#6f757f;
 --line:#2a2b30;--acc:#fb923c;--acc2:#38bdf8;--acc3:#4ade80;--warn:#fbbf24;--bad:#f87171}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
 font:16px/1.66 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Helvetica,Arial,sans-serif;
 -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1180px;margin:0 auto;padding:48px 26px 90px}}
.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
header{{border-bottom:2px solid var(--line);padding-bottom:22px;margin-bottom:8px}}
.kicker{{font-size:11.5px;letter-spacing:.16em;text-transform:uppercase;color:var(--acc);font-weight:700}}
h1{{font-size:2.2rem;line-height:1.1;margin:.3em 0 .3em;letter-spacing:-.022em}}
.sub{{color:var(--mute);font-size:1.02rem;max-width:80ch;margin:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:26px 0 8px}}
.grid.g3{{grid-template-columns:repeat(3,minmax(0,1fr))}}
.stat{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:13px 15px}}
.stat .v{{font-size:1.34rem;font-weight:650;letter-spacing:-.02em;color:var(--acc2);
 font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
.stat .l{{font-size:12px;color:var(--mute);margin-top:6px}}
h2{{font-size:1.16rem;margin:44px 0 4px;letter-spacing:-.01em}}
.hint{{color:var(--dim);font-size:13px;margin:0 0 16px}}
.gantt{{display:flex;width:100%;height:52px;border-radius:8px;overflow:hidden;border:1px solid var(--line);
 background:var(--panel)}}
.gantt .seg{{position:relative;height:100%;border-right:1px solid var(--bg);background:var(--c);
 opacity:.86;display:flex;align-items:center;justify-content:center;overflow:hidden;
 font-size:11px;font-weight:650;color:#0b0b0d;white-space:nowrap;padding:0 4px;min-width:2px}}
.gantt .seg span{{margin-left:6px;font-weight:500;opacity:.8;font-family:ui-monospace,monospace}}
.gantt .seg:hover{{opacity:1}}
.legend{{display:flex;gap:18px;margin:12px 0 0;font-size:12.5px;color:var(--mute)}}
.legend i{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.tblwrap{{overflow-x:auto;margin:20px 0;border:1px solid var(--line);border-radius:9px;background:var(--panel)}}
table{{border-collapse:collapse;width:100%;font-size:13px;min-width:1000px}}
thead th{{text-align:right;padding:11px 13px;color:var(--mute);font-weight:600;font-size:11.5px;
 letter-spacing:.05em;text-transform:uppercase;border-bottom:1px solid var(--line);white-space:nowrap;
 cursor:help;border-bottom-style:dotted;border-bottom-width:1px}}
thead th:hover{{color:var(--ink)}}
thead th:first-child,thead th:nth-child(2){{text-align:left}}
.serialflag{{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:4px;font-size:10px;
 letter-spacing:.04em;text-transform:uppercase;color:var(--warn);border:1px solid var(--warn);cursor:help}}
.note{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
 border-radius:8px;padding:12px 16px;margin:18px 0 4px;font-size:13.4px;color:var(--mute);line-height:1.6;max-width:96ch}}
.note b{{color:var(--ink);font-weight:640}}
.note ul{{margin:.5em 0 .7em;padding-left:20px}}
.note li{{margin:.28em 0}}
.note li b{{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace;font-size:.94em}}
.gpuh{{font-size:.86rem;color:var(--mute);letter-spacing:.09em;text-transform:uppercase;margin:20px 0 8px;font-weight:650}}
.gpugrid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:0 0 8px}}
.gputile{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--dim);
 border-radius:9px;padding:13px 15px;opacity:.72}}
.gputile.used{{border-left-color:var(--acc3);opacity:1;box-shadow:0 0 0 1px var(--acc3) inset}}
.gname{{font-size:12.5px;color:var(--mute);margin-bottom:8px;line-height:1.35}}
.ubadge{{margin-left:8px;padding:1px 6px;border-radius:4px;font-size:9.5px;letter-spacing:.05em;
 text-transform:uppercase;color:var(--acc3);border:1px solid var(--acc3);white-space:nowrap}}
.ubadge.idle{{color:var(--dim);border-color:var(--line)}}
.gvram{{font-size:1.18rem;font-variant-numeric:tabular-nums;color:var(--ink);
 font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
.gvram b{{color:var(--acc3);font-weight:650}}
.gbar{{height:6px;border-radius:4px;background:var(--panel2);margin-top:9px;overflow:hidden}}
.gbar span{{display:block;height:100%;background:linear-gradient(90deg,var(--acc3),var(--acc2));border-radius:4px}}
tbody td{{padding:9px 13px;border-bottom:1px solid var(--line);vertical-align:middle}}
tbody tr:last-child td{{border-bottom:none}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;
 font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace;white-space:nowrap}}
td.mono{{font-family:ui-monospace,SFMono-Regular,Menlo,"DejaVu Sans Mono",monospace}}
.dim{{color:var(--dim)}}
tr.sub td{{padding:5px 13px;font-size:12px;background:#141519}}
.mini{{height:5px;border-radius:3px;background:var(--panel2);margin-top:5px;overflow:hidden}}
.mini span{{display:block;height:100%;border-radius:3px}}
tbody tr:hover td{{background:#1c1d21}}
footer{{margin-top:40px;color:var(--dim);font-size:12px;border-top:1px solid var(--line);padding-top:16px}}
</style></head><body><div class="wrap">
<header><div class="kicker">LeafMachine3 · run timing</div>
<h1>{run}</h1>
<p class="sub">Per-module wall time, compute utilization, and throughput for this pipeline run. Modules run
sequentially; each bar below is one module, its width proportional to its share of the run.</p></header>
<div class="grid g3">{stats}</div>
<h2>Timeline</h2><p class="hint">Hover a segment for its exact time. Color = how the module parallelizes.</p>
{gantt}{legend}
{pool_note}
<h2>Per-module detail</h2><p class="hint">Hover a column heading for what it means.</p>
{table}
{baseline_note}
{hardware}
<footer>Generated by <span class="mono">leafmachine3.setup.timing</span> · CPU% is system-wide (100% = all logical cores); GPU% is the mean over the run's GPU(s). Throughput = items processed ÷ module wall time. Indented <span class="mono">↳</span> rows are in-module components timed as CPU-time <b>summed across the module's worker threads</b> (so they total more than its wall time); their % is each component's share of that.</footer>
</div></body></html>"""
