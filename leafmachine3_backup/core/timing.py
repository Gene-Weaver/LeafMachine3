"""Per-stage wall-clock timing report."""
from __future__ import annotations

import contextlib
import time


class TimeReport:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.windows: dict[str, tuple[float, float]] = {}   # key -> (start, end) wall-clock (time.time)
        self.subsections: dict[str, dict[str, float]] = {}  # key -> {component: seconds} (optional detail)

    @contextlib.contextmanager
    def stage(self, key: str):
        t0 = time.perf_counter()
        w0 = time.time()
        try:
            yield
        finally:
            self.times[key] = self.times.get(key, 0.0) + (time.perf_counter() - t0)
            self.windows[key] = (w0, time.time())

    def add_subsections(self, key: str, parts: dict[str, float]) -> None:
        """Merge a stage's fine-grained component timings (e.g. Reporter per-overlay seconds)."""
        if not parts:
            return
        acc = self.subsections.setdefault(key, {})
        for name, secs in parts.items():
            acc[name] = acc.get(name, 0.0) + float(secs)

    def log_report(self, log) -> None:
        if not self.times:
            return
        log.info("time report:")
        for k, v in self.times.items():
            log.info("  %-24s %7.1fs", k, v)
        log.info("  %-24s %7.1fs", "TOTAL", sum(self.times.values()))
