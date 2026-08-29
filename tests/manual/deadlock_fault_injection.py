"""Drive StageExecutor._run_pool through a REAL spawn pool while injecting CUDA-OOM device
faults, and assert the stage completes.

Not collected by pytest (the suite keeps real spawn out of unit tests; see the note at the top of
tests/test_executor.py). Run it by hand after touching the feeder/collector:

    .venv_LM3/bin/python tests/manual/deadlock_fault_injection.py /some/scratch/markers

It reproduces the conditions that hung the Global Greening batch for 39 hours -- small bounded
queues, a slow single-writer collector, and a burst of faults. Against the pre-fix executor it
deadlocks on the FIRST requeue and never returns; against the fixed one it finishes 200/200 in
about 4s with 20 faults recovered and 0 errors.
"""
from __future__ import annotations
import contextlib, os, pathlib, sys, time, types
from collections import Counter

from leafmachine3.core.device import Device
from leafmachine3.core.executor import StageExecutor
from leafmachine3.core.stage import WorkItem

MARKERS = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lm3_fault_markers")
N_ITEMS, N_WORKERS, FAULT_EVERY = 200, 4, 10


class FaultyStage:
    """Picklable (spawn needs that). Faults ONCE per selected specimen, then succeeds."""
    key = "archival_detector"
    device_kind = "cuda"
    cpu_parallel = "process"
    fanout = False
    depends_on = ()

    def build_model(self, device):
        return "model"

    def infer(self, item, model):
        sid = item.specimen_id
        if sid % FAULT_EVERY == 0:
            marker = MARKERS / f"{sid}"
            if not marker.exists():
                marker.write_text("faulted")
                raise RuntimeError("CUDA error: out of memory")   # classified as a device fault
        return {"sid": sid}

    def persist(self, project, item, payload):
        pass


class _DB:
    def __init__(self):
        self.done, self.errors = [], []
    def transaction(self):
        return contextlib.nullcontext()
    def mark_image_done(self, sid, key, **kw):
        time.sleep(0.02)                      # slow single writer -> result_q backs up
        self.done.append(sid)
    def mark_image_error(self, sid, key, msg):
        self.errors.append((sid, msg))


def main():
    MARKERS.mkdir(parents=True, exist_ok=True)
    for f in MARKERS.iterdir():
        f.unlink()

    cfg = types.SimpleNamespace(compute=types.SimpleNamespace(
        mock=False, liveness_timeout_s=10.0, vram={"concurrent_warm_loads": 2}))
    ex = StageExecutor(cfg, FaultyStage())
    db = _DB()
    project = types.SimpleNamespace(db=db)

    todo = [WorkItem(specimen_id=i, payload={"i": i}) for i in range(1, N_ITEMS + 1)]
    ex._remaining = Counter(it.specimen_id for it in todo)
    devices = [Device("cpu") for _ in range(N_WORKERS)]

    t0 = time.time()
    ex._run_pool(project, todo, devices)
    elapsed = time.time() - t0

    faults = len(list(MARKERS.iterdir()))
    print(f"elapsed      : {elapsed:.1f}s")
    print(f"faults fired : {faults}")
    print(f"done         : {len(db.done)}/{N_ITEMS}")
    print(f"errors       : {len(db.errors)}")
    print(f"recoveries   : {getattr(ex, '_recoveries', 'n/a')}")
    ok = len(db.done) == N_ITEMS and faults >= N_ITEMS // FAULT_EVERY
    print("RESULT       :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
