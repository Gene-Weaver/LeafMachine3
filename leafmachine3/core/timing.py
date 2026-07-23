"""Per-stage wall-clock timing report."""
from __future__ import annotations

import contextlib
import time


class TimeReport:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[key] = self.times.get(key, 0.0) + (time.perf_counter() - t0)

    def log_report(self, log) -> None:
        if not self.times:
            return
        log.info("time report:")
        for k, v in self.times.items():
            log.info("  %-24s %7.1fs", k, v)
        log.info("  %-24s %7.1fs", "TOTAL", sum(self.times.values()))
