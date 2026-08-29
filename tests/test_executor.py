"""Executor routing + fanout-completion unit tests (the process path itself needs real spawn, so
the mock e2e + a real run cover it; here we pin the pure decision logic)."""
from __future__ import annotations

import os
import queue
import sys
import threading
import types

from collections import Counter

from leafmachine3.core.device import Device
from leafmachine3.core.executor import StageExecutor


def _stage(device_kind="cpu", cpu_parallel="thread", fanout=False):
    return types.SimpleNamespace(key="x", device_kind=device_kind,
                                 cpu_parallel=cpu_parallel, fanout=fanout)


def _cfg(mock=False):
    return types.SimpleNamespace(compute=types.SimpleNamespace(mock=mock))


def _ex(stage, cfg):
    return StageExecutor(cfg, stage)


def test_cpu_process_stage_routes_to_pool_when_parallel():
    ex = _ex(_stage(device_kind="cpu", cpu_parallel="process"), _cfg())
    assert ex._use_inprocess([Device("cpu")] * 4, 100) is False    # >1 worker + big batch -> spawn pool
    assert ex._use_inprocess([Device("cpu")], 100) is True         # 1 worker -> no point spawning


def test_cpu_process_stage_runs_serial_for_small_batch():
    """A tiny batch (< min_pool_items, default 8) skips the spawn pool and runs in-process/serial."""
    ex = _ex(_stage(device_kind="cpu", cpu_parallel="process"), _cfg())
    assert ex._use_inprocess([Device("cpu")] * 8, 3) is True       # spawn cost > gain -> serial
    assert ex._use_inprocess([Device("cpu")] * 8, 8) is False      # at the threshold -> pool


def test_cpu_thread_stage_stays_inprocess():
    ex = _ex(_stage(device_kind="cpu", cpu_parallel="thread"), _cfg())
    assert ex._use_inprocess([Device("cpu")] * 8, 100) is True


def test_mock_always_inprocess_even_for_process_stage():
    ex = _ex(_stage(device_kind="cpu", cpu_parallel="process"), _cfg(mock=True))
    assert ex._use_inprocess([Device("cpu")] * 8, 100) is True


def test_gpu_stage_routing_unchanged():
    ex = _ex(_stage(device_kind="cuda"), _cfg())
    assert ex._use_inprocess([Device("cuda", 0), Device("cuda", 0)], 100) is False
    assert ex._use_inprocess([Device("cpu")], 100) is True         # GPU-less plan -> in-process


def test_fanout_marks_specimen_done_only_after_last_subitem():
    """The per-specimen ``_remaining`` counter (set in run()) must defer mark_image_done until a
    specimen's last sub-item checkpoints, so a fanout stage keeps per-image resume correct."""
    from collections import Counter

    ex = _ex(_stage(cpu_parallel="process", fanout=True), _cfg())
    marked: list[int] = []

    class _DB:
        def transaction(self):
            import contextlib
            return contextlib.nullcontext()

        def mark_image_done(self, sid, key):
            marked.append(sid)

        def mark_image_error(self, *a):
            raise AssertionError("should not error")

    project = types.SimpleNamespace(db=_DB())
    ex.stage = types.SimpleNamespace(key="ect", cpu_parallel="process", fanout=True,
                                     persist=lambda *a: None)
    ex._remaining = Counter({7: 3})                                # specimen 7 has 3 leaves

    def item(sid):
        return types.SimpleNamespace(specimen_id=sid)

    ex._checkpoint(project, item(7), "ok", {"row": 1})
    ex._checkpoint(project, item(7), "ok", {"row": 2})
    assert marked == []                                            # not done until the last leaf
    ex._checkpoint(project, item(7), "ok", {"row": 3})
    assert marked == [7]                                           # done exactly once, on the 3rd


# --------------------------------------------------------------------------- #
# Per-GPU "building block" worker sizing
#
# Every GPU is sized from its OWN live free VRAM divided by one worker's measured cost, so
# the counts are deliberately uneven: a card already holding another job takes fewer workers
# than an idle one. These pin that behaviour, plus the two ways it used to go wrong (reading
# the all-workers total as a per-worker cost, and force-feeding a worker onto a full card).
# --------------------------------------------------------------------------- #
import pytest

from leafmachine3.core.executor import DeviceManager


def _gpu_cfg(*, free_by_gpu, hw_stage=None, vram=None):
    """A cfg whose GPU list and per-GPU free VRAM are both fully controlled by the test."""
    compute = types.SimpleNamespace(
        mock=False,
        devices=sorted(free_by_gpu),
        vram={"safety_fraction": 1.0, "reserve_mb": 0, "per_worker_mb": "auto",
              "max_workers_per_gpu": 16, **(vram or {})},
    )
    hardware = types.SimpleNamespace(stage=lambda key: hw_stage)
    return types.SimpleNamespace(compute=compute, hardware=hardware)


def _plan(monkeypatch, cfg, free_by_gpu):
    monkeypatch.setattr("leafmachine3.core.executor._free_vram_mb",
                        lambda idx: float(free_by_gpu[idx]))
    dm = DeviceManager(cfg, _stage(device_kind="cuda"))
    monkeypatch.setattr(dm, "_resolve_gpus", lambda: sorted(free_by_gpu))
    return dm.plan()


def _counts(plan):
    return {g: sum(1 for d in plan if d.index == g) for g in {d.index for d in plan}}


def test_workers_per_gpu_is_uneven_and_follows_each_cards_free_vram(monkeypatch):
    """GPU 0 holding another job must take fewer workers than an idle GPU 1."""
    free = {0: 5_000.0, 1: 48_000.0}
    hw = {"vram_per_worker_mb": 2_000.0, "workers_per_gpu": 4}
    plan = _plan(monkeypatch, _gpu_cfg(free_by_gpu=free, hw_stage=hw), free)

    counts = _counts(plan)
    # headroom_factor defaults to 1.15 -> 2300 MB/worker
    assert counts[0] == 2          # 5000 / 2300
    assert counts[1] == 16         # 48000/2300 = 20, clamped by max_workers_per_gpu
    assert counts[0] != counts[1], "per-GPU counts must be free-VRAM driven, not one flat number"


def test_full_gpu_is_skipped_rather_than_given_a_worker(monkeypatch):
    """A card too full for even one worker must be left alone -- forcing one onto it is how it OOMs."""
    free = {0: 500.0, 1: 20_000.0}
    hw = {"vram_per_worker_mb": 4_000.0}
    plan = _plan(monkeypatch, _gpu_cfg(free_by_gpu=free, hw_stage=hw), free)

    assert all(d.index == 1 for d in plan), "a full GPU must not be scheduled"
    assert len(plan) == 4          # 20000 / 4600


def test_all_gpus_full_still_yields_one_worker_on_the_roomiest(monkeypatch):
    """Falling back to CPU here would be orders of magnitude slower, so try the roomiest card."""
    free = {0: 100.0, 1: 900.0}
    hw = {"vram_per_worker_mb": 8_000.0}
    plan = _plan(monkeypatch, _gpu_cfg(free_by_gpu=free, hw_stage=hw), free)

    assert [d.index for d in plan] == [1]
    assert plan[0].kind == "cuda"


def test_measured_vram_is_preferred_over_the_heuristic_estimate(monkeypatch):
    """A calibrated figure must win; the estimate is only the fallback."""
    free = {0: 24_000.0}
    hw = {"vram_per_worker_mb": 2_000.0, "est_vram_per_worker_mb": 20_000.0}
    plan = _plan(monkeypatch, _gpu_cfg(free_by_gpu=free, hw_stage=hw), free)

    assert len(plan) == 10         # 24000 / 2300, not 24000 / 23000 == 1


def test_peak_vram_total_is_never_read_as_a_per_worker_cost(monkeypatch):
    """peak_vram_mb is the total across ALL workers; dividing free VRAM by it under-fits badly."""
    free = {0: 40_000.0}
    hw = {"peak_vram_mb": 36_400.0, "vram_per_worker_mb": 2_500.0}
    plan = _plan(monkeypatch, _gpu_cfg(free_by_gpu=free, hw_stage=hw), free)

    assert len(plan) > 1, "reading the all-workers total per worker collapses the plan to 1"


def test_explicit_per_worker_override_wins_verbatim(monkeypatch):
    """An explicit compute.vram.per_worker_mb is taken as-is, with no headroom multiplier."""
    free = {0: 10_000.0}
    hw = {"vram_per_worker_mb": 500.0}
    cfg = _gpu_cfg(free_by_gpu=free, hw_stage=hw, vram={"per_worker_mb": 5_000})
    plan = _plan(monkeypatch, cfg, free)

    assert len(plan) == 2          # 10000 / 5000 exactly, no 1.15x applied


def test_real_config_does_not_short_circuit_worker_sizing(monkeypatch):
    """Regression: a REAL Config must size from measured VRAM, not the all-workers total.

    ``Config.probe_vram_mb`` is a bound method, so it is always the first branch the executor
    consults -- it is not an absent test hook. It used to return the profile's ``peak_vram_mb``
    (the total across every worker), which pinned every GPU to exactly one worker no matter how
    much VRAM was free. The SimpleNamespace configs above cannot catch that, because they have
    no such method at all.
    """
    from leafmachine3.core.config import Config

    cfg = Config({"compute": {"mock": False, "devices": [0],
                              "vram": {"safety_fraction": 1.0, "reserve_mb": 0,
                                       "per_worker_mb": "auto", "max_workers_per_gpu": 16,
                                       "headroom_factor": 1.0}}})
    cfg.bind_hardware({"stages": {"x": {"batch": 16, "workers_per_gpu": 15,
                                        "peak_vram_mb": 41400,        # 2760 * 15 workers
                                        "vram_per_worker_mb": 2760.0,
                                        "vram_measured": True}}})

    monkeypatch.setattr("leafmachine3.core.executor._free_vram_mb", lambda idx: 27_600.0)
    dm = DeviceManager(cfg, _stage(device_kind="cuda"))
    monkeypatch.setattr(dm, "_resolve_gpus", lambda: [0])

    assert dm._per_worker_vram_mb(cfg.compute["vram"]) == pytest.approx(2760.0)
    assert len(dm.plan()) == 10, "27600 MB / 2760 MB per worker == 10, not 1"


def test_explicit_per_worker_override_still_reaches_the_executor():
    """The one thing probe_vram_mb should still answer: an explicit user override."""
    from leafmachine3.core.config import Config

    cfg = Config({"compute": {"vram": {"per_worker_mb": 3000}}})
    assert cfg.probe_vram_mb(_stage(device_kind="cuda")) == 3000

    cfg_auto = Config({"compute": {"vram": {"per_worker_mb": "auto"}}})
    assert cfg_auto.probe_vram_mb(_stage(device_kind="cuda")) is None


# --------------------------------------------------------------------------- #
# Longest-processing-time-first dispatch
#
# Workers pull from a shared queue, so assignment self-balances; the failure mode is a BIG
# item dispatched late, which leaves every other worker idle while it finishes. These pin the
# ordering and the makespan property it buys.
# --------------------------------------------------------------------------- #
from leafmachine3.core.stage import WorkItem


def _items(sizes):
    return [WorkItem(i, list(range(n))) for i, n in enumerate(sizes)]


def _makespan(items, m):
    """Greedy pull-queue simulation: each worker takes the next item when it frees up."""
    end = [0.0] * m
    for it in items:
        i = min(range(m), key=lambda k: end[k])
        end[i] += len(it.payload)
    return max(end)


def test_longest_first_orders_biggest_payload_first():
    ex = _ex(_stage(device_kind="cuda"), _cfg())
    ordered = ex._longest_first(_items([3, 300, 1, 40]))
    assert [len(i.payload) for i in ordered] == [300, 40, 3, 1]


def test_longest_first_cuts_the_idle_tail():
    """Skew like the real corpus: mostly small sheets, a few ~250-crop monsters arriving late.

    Sized so no single item exceeds a worker's ideal share -- otherwise the makespan is pinned
    by that one item and no ordering can help, which says nothing about the scheduler.
    """
    m = 8
    sizes = [8] * 80 + [40] * 15 + [250] * 5          # big ones LAST, as specimen_id order can
    todo = _items(sizes)
    ex = _ex(_stage(device_kind="cuda"), _cfg())

    naive = _makespan(todo, m)
    lpt = _makespan(ex._longest_first(todo), m)
    lower_bound = max(sum(sizes) / m, max(sizes))      # no schedule can beat either
    assert lpt < naive, f"longest-first must shorten the tail ({lpt} vs {naive})"
    assert lpt <= (4 / 3 - 1 / (3 * m)) * lower_bound, f"LPT bound violated: {lpt}"


def test_longest_first_is_a_noop_for_uniform_work():
    """Fanout stages emit one sub-item each; reordering them would only churn."""
    ex = _ex(_stage(device_kind="cpu"), _cfg())
    todo = _items([1, 1, 1, 1])
    assert ex._longest_first(todo) is todo


def test_longest_first_respects_a_stage_supplied_weight():
    stage = _stage(device_kind="cuda")
    stage.work_weight = lambda item: 100 - item.specimen_id      # inverse of payload length
    ex = _ex(stage, _cfg())
    ordered = ex._longest_first(_items([1, 2, 3]))
    assert [i.specimen_id for i in ordered] == [0, 1, 2]


def test_longest_first_survives_a_broken_weight_and_unsized_payloads():
    """A bad weight or an unsized payload must degrade to 'no reorder', never raise."""
    stage = _stage(device_kind="cuda")
    stage.work_weight = lambda item: 1 / 0
    ex = _ex(stage, _cfg())
    todo = _items([5, 1])
    assert ex._longest_first(todo) is todo                        # all weights 0 -> uniform

    ex2 = _ex(_stage(device_kind="cuda"), _cfg())
    unsized = [WorkItem(0, object()), WorkItem(1, object())]
    assert ex2._longest_first(unsized) is unsized


# --------------------------------------------------------------------------- #
# Orphaned-worker reaping
#
# daemon=True is an atexit hook, so a SIGKILLed parent leaves its GPU workers alive holding
# CUDA contexts forever. PR_SET_PDEATHSIG closes that (kernel-enforced); reap_orphaned_workers
# is the sweep for everything pdeathsig cannot cover. The must-not-regress property is that a
# CONCURRENT LM3 run's workers are never mistaken for orphans.
# --------------------------------------------------------------------------- #
from leafmachine3.core import executor as _ex_mod


class _FakeProc:
    def __init__(self, pid, ppid, cmdline, uid, exe):
        self.info = {"pid": pid, "ppid": ppid, "cmdline": cmdline,
                     "uids": types.SimpleNamespace(real=uid), "exe": exe}
        self.terminated = False

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def _fake_psutil(procs, parents):
    """A psutil stand-in: `procs` are scanned, `parents` maps pid -> cmdline list."""
    class _NoSuch(Exception):
        pass

    class _Timeout(Exception):
        pass

    class _P:
        def __init__(self, pid):
            if pid not in parents:
                raise _NoSuch(pid)
            self._c = parents[pid]

        def cmdline(self):
            return self._c

    return types.SimpleNamespace(process_iter=lambda attrs=None: list(procs),
                                 Process=_P, NoSuchProcess=_NoSuch, TimeoutExpired=_Timeout)


def _install_fake(monkeypatch, procs, parents):
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(procs, parents))
    monkeypatch.setattr(_ex_mod, "_worker_vram_by_pid", lambda: {})


def test_reaper_kills_a_worker_whose_parent_died(monkeypatch):
    exe = os.path.realpath(sys.executable)
    orphan = _FakeProc(4242, 1, ["python", "-c", "from multiprocessing.spawn import spawn_main"],
                       os.getuid(), exe)
    _install_fake(monkeypatch, [orphan], parents={})           # parent no longer exists
    assert [p for p, _ in _ex_mod.reap_orphaned_workers()] == [4242]
    assert orphan.terminated


def test_reaper_never_touches_a_concurrent_lm3_runs_workers(monkeypatch):
    """The whole safety story: a live LM3 parent means its children are NOT orphans."""
    exe = os.path.realpath(sys.executable)
    sibling = _FakeProc(5555, 999, ["python", "-c", "from multiprocessing.spawn import spawn_main"],
                        os.getuid(), exe)
    _install_fake(monkeypatch, [sibling],
                  parents={999: ["python", "-m", "leafmachine3.machine3", "--config", "x.yaml"]})
    assert _ex_mod.reap_orphaned_workers() == []
    assert not sibling.terminated


def test_reaper_ignores_reparenting_to_systemd_user_when_parent_is_lm3(monkeypatch):
    """Orphan detection must ask WHAT the parent is, not just whether ppid == 1.

    On a systemd host orphans are adopted by `systemd --user`, which is very much alive, so a
    naive `ppid == 1` test finds nothing at all.
    """
    exe = os.path.realpath(sys.executable)
    orphan = _FakeProc(6001, 710075, ["python", "-c", "from multiprocessing.spawn import spawn_main"],
                       os.getuid(), exe)
    _install_fake(monkeypatch, [orphan],
                  parents={710075: ["/lib/systemd/systemd", "--user"]})
    assert [p for p, _ in _ex_mod.reap_orphaned_workers()] == [6001]


def test_reaper_skips_other_users_and_other_venvs(monkeypatch):
    """It kills processes, so it must stay inside our uid and our interpreter."""
    exe = os.path.realpath(sys.executable)
    spawn_cmd = ["python", "-c", "from multiprocessing.spawn import spawn_main"]
    other_user = _FakeProc(7001, 1, spawn_cmd, os.getuid() + 1, exe)
    other_venv = _FakeProc(7002, 1, spawn_cmd, os.getuid(), "/somewhere/else/bin/python")
    not_a_worker = _FakeProc(7003, 1, ["python", "train.py"], os.getuid(), exe)
    _install_fake(monkeypatch, [other_user, other_venv, not_a_worker], parents={})
    assert _ex_mod.reap_orphaned_workers() == []
    assert not any(p.terminated for p in (other_user, other_venv, not_a_worker))


def test_reaper_dry_run_reports_without_killing(monkeypatch):
    exe = os.path.realpath(sys.executable)
    orphan = _FakeProc(8001, 1, ["python", "-c", "from multiprocessing.spawn import spawn_main"],
                       os.getuid(), exe)
    _install_fake(monkeypatch, [orphan], parents={})
    assert [p for p, _ in _ex_mod.reap_orphaned_workers(dry_run=True)] == [8001]
    assert not orphan.terminated, "dry run must not kill anything"


def test_pdeathsig_is_only_armed_from_the_main_thread():
    """Armed off a helper thread it would fire when THAT thread exits, killing the pool mid-run."""
    assert _ex_mod._pdeathsig_safe("s") is True
    out = []
    t = threading.Thread(target=lambda: out.append(_ex_mod._pdeathsig_safe("s")))
    t.start()
    t.join()
    assert out == [False]


def test_longest_first_can_be_disabled(monkeypatch):
    """compute.longest_first: false restores raw DB order (the A/B knob for benchmarking)."""
    cfg = types.SimpleNamespace(
        compute=types.SimpleNamespace(mock=False, longest_first=False))
    ex = StageExecutor(cfg, _stage(device_kind="cuda"))
    todo = _items([3, 300, 1])
    assert ex._longest_first(todo) is todo


# --------------------------------------------------------------------------- #
# Static worker tiers (ruler_classifier)
#
# VRAM is the wrong sizing input for a cheap, short stage: 16 workers FIT but never pay for
# themselves. The tier table caps by batch size, and always AFTER the VRAM planner, so it can
# only take workers away -- never hand out more than the machine can hold.
# --------------------------------------------------------------------------- #
def _tiered_stage(tiers=((100, 2), (1000, 4), (None, 16))):
    s = _stage(device_kind="cuda")
    s.worker_tiers = tiers
    return s


def _cuda(n, index=1):
    return [Device("cuda", index) for _ in range(n)]


@pytest.mark.parametrize("n_items,expected", [
    (1, 2), (99, 2),          # < 100  -> 2
    (100, 4), (999, 4),       # < 1000 -> 4
    (1000, 16), (5000, 16),   # >= 1000 -> up to 16
])
def test_worker_tiers_pick_the_band_for_the_batch_size(n_items, expected):
    ex = _ex(_tiered_stage(), _cfg())
    assert len(ex._apply_worker_tiers(_cuda(16), n_items)) == expected


def test_worker_tier_never_exceeds_what_the_machine_can_hold():
    """The whole point: ask for the tier, get the most the GPU actually fits."""
    ex = _ex(_tiered_stage(), _cfg())
    assert len(ex._apply_worker_tiers(_cuda(3), 5000)) == 3      # tier 16, VRAM fits 3
    assert len(ex._apply_worker_tiers(_cuda(1), 500)) == 1       # tier 4,  VRAM fits 1


def test_worker_tier_trim_keeps_the_multi_gpu_spread():
    """Trimming must not pile every survivor onto one card."""
    ex = _ex(_tiered_stage(), _cfg())
    plan = _cuda(2, index=0) + _cuda(14, index=1)
    kept = ex._apply_worker_tiers(plan, 500)                     # tier -> 4
    assert len(kept) == 4
    assert {d.index for d in kept} == {0, 1}, "both GPUs should survive the trim"


def test_stages_without_tiers_are_untouched():
    """Every other module keeps pure free-VRAM sizing."""
    ex = _ex(_stage(device_kind="cuda"), _cfg())
    plan = _cuda(16)
    assert ex._apply_worker_tiers(plan, 5) is plan


def test_ruler_classifier_declares_the_agreed_policy():
    from leafmachine3.modules.ruler_classifier import RulerClassifier
    assert RulerClassifier.worker_tiers == ((100, 2), (1000, 4), (None, 16))


# --------------------------------------------------------------------------- #
# Device-fault recovery must not deadlock
#
# A burst of CUDA OOM faults used to hang the whole stage. Both queues are bounded, and the
# requeue branch made the COLLECTOR -- the only consumer of result_q -- a producer on task_q:
# collector blocks putting -> stops draining result_q -> workers block putting results -> nobody
# pulls tasks -> neither queue can drain. Observed on the Global Greening batch: a 29-worker
# archival_detector pool wedged for 39 hours holding ~94 GB of VRAM, with no error and no exit.
# These pin the invariants that make that cycle impossible.
# --------------------------------------------------------------------------- #
from leafmachine3.core.executor import _STOP, _worker


class _ExplodingTaskQ:
    """A task queue that fails the test if the collector ever produces on it."""

    def put(self, *a, **k):
        raise AssertionError("the collector must never put on task_q -- that is the deadlock")

    def put_nowait(self, *a, **k):
        self.put()


class _ScriptedResultQ:
    def __init__(self, events):
        self._events = list(events)

    def get(self, timeout=None):
        if not self._events:
            raise queue.Empty
        return self._events.pop(0)


def _recording_project():
    errors: list[tuple] = []
    done: list[int] = []

    class _DB:
        def transaction(self):
            import contextlib
            return contextlib.nullcontext()

        def mark_image_done(self, sid, key, **kw):
            done.append(sid)

        def mark_image_error(self, sid, key, msg):
            errors.append((sid, msg))

    return types.SimpleNamespace(db=_DB()), errors, done


def _collect_ex(max_recoveries=None, max_item_requeues=None):
    compute = types.SimpleNamespace(mock=False, liveness_timeout_s=0.01)
    if max_recoveries is not None:
        compute.max_device_recoveries = max_recoveries
    if max_item_requeues is not None:
        compute.max_item_requeues = max_item_requeues
    ex = _ex(_stage(device_kind="cuda"), types.SimpleNamespace(compute=compute))
    ex.stage = types.SimpleNamespace(key="archival_detector", fanout=False,
                                     persist=lambda *a: None)
    ex._devices = []                                               # makes _respawn a no-op
    ex._remaining = Counter()
    return ex


def test_collector_never_produces_on_the_task_queue():
    """THE regression: a requeue must be buffered for the feeder, never put by the collector."""
    ex = _collect_ex()
    project, _, _ = _recording_project()
    item = types.SimpleNamespace(specimen_id=42)
    result_q = _ScriptedResultQ([("requeue", item, "CUDA out of memory"),
                                 ("ok", item, {"row": 1})])
    workers = [types.SimpleNamespace(is_alive=lambda: True, pid=1)]

    ex._collect(project, _ExplodingTaskQ(), result_q, 1, workers)

    assert list(ex._retry) == [item]        # handed to the feeder instead of put directly
    assert ex._recoveries == 1


def test_feeder_hands_back_retries_before_the_stop_sentinels():
    """A retry queued after the STOPs would sit behind them: workers take the STOPs, exit, and
    the item is never run. The feeder must therefore hold the STOPs until the collector is done."""
    ex = _collect_ex()
    task_q: queue.Queue = queue.Queue(maxsize=8)
    retry = types.SimpleNamespace(specimen_id=99)

    feeder = threading.Thread(target=ex._feed, args=(task_q, ["a", "b"], 2), daemon=True)
    feeder.start()

    for expected in ("a", "b"):
        assert task_q.get(timeout=5) == expected

    ex._retry.append(retry)                                        # fault lands after the feed
    assert task_q.get(timeout=5) is retry                          # ... and goes out immediately

    ex._finished.set()                                             # collector says everything retired
    assert task_q.get(timeout=5) is _STOP                          # only NOW do the STOPs appear
    assert task_q.get(timeout=5) is _STOP
    feeder.join(timeout=5)
    assert not feeder.is_alive()


def test_item_retry_limit_retires_an_item_that_faults_every_time():
    """A requeued item never advances ``seen``, so an item that always faults could not terminate
    the collect loop. Past its allowance it is retired as an error instead of retried forever."""
    ex = _collect_ex(max_item_requeues=2)
    project, errors, _ = _recording_project()
    item = types.SimpleNamespace(specimen_id=7)
    result_q = _ScriptedResultQ([("requeue", item, "CUDA out of memory")] * 6)
    workers = [types.SimpleNamespace(is_alive=lambda: True, pid=1)]

    ex._collect(project, _ExplodingTaskQ(), result_q, 1, workers)  # must RETURN, not spin

    assert list(ex._retry) == [item, item]                         # retried exactly twice
    assert errors == [(7, "CUDA out of memory")]                   # then failed, so `seen` advances


def test_retry_allowance_scales_with_a_fanout_specimens_subitems():
    """A flat per-specimen cap would starve a fanout stage: 30 leaves share one specimen_id and
    WorkItem has no identity of its own, so each leaf's fault would burn the same allowance."""
    ex = _collect_ex(max_item_requeues=3)
    ex._subitems = Counter({5: 1, 9: 30})                          # sheet vs 30-leaf specimen
    assert ex._requeue_allowance(5) == 3
    assert ex._requeue_allowance(9) == 90
    assert ex._requeue_allowance(404) == 3                         # unknown specimen -> flat cap


def test_stage_recovery_rail_is_sized_off_items_not_workers():
    """Faults arrive per image. A worker-scaled ceiling gets spent early in a long run and then
    starts failing images that one retry would have fixed -- which is what a 4-worker, 200-item
    fault-injection run actually did before this was rescaled."""
    ex = _collect_ex()
    assert ex._recovery_budget(20) == 64                           # floor for small batches
    assert ex._recovery_budget(8464) == 2116                       # scales with the workload
    assert _collect_ex(max_recoveries=5)._recovery_budget(8464) == 5   # explicit config wins


def test_worker_loads_anyway_when_the_warm_load_gate_never_frees():
    """A worker SIGKILLed inside build_model never returns its gate permit. Enough of those and
    every replacement would block on the gate forever, so the acquire is bounded."""
    loaded: list[str] = []
    released: list[int] = []

    class _LeakedGate:
        def acquire(self, timeout=None):
            return False                                           # permit is gone for good
        def release(self):
            released.append(1)

    class _TaskQ:
        def __init__(self):
            self._items = [types.SimpleNamespace(specimen_id=1), _STOP]
        def get(self):
            return self._items.pop(0)

    results: list[tuple] = []
    stage = types.SimpleNamespace(
        key="archival_detector",
        build_model=lambda device: loaded.append("model") or "model",
        infer=lambda item, model: {"row": item.specimen_id},
    )
    _worker(Device("cpu"), stage, _TaskQ(),
            types.SimpleNamespace(put=results.append), _LeakedGate())

    assert loaded == ["model"]                                     # loaded despite the dead gate
    assert released == []                                          # never release what we don't hold
    assert results == [("ok", results[0][1], {"row": 1})]
