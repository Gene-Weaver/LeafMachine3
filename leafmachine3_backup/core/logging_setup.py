"""Console + file logging for a run."""
from __future__ import annotations

import logging
import sys


def start_logging(dirs, cfg) -> logging.Logger:
    lvl_name = "INFO"
    try:
        lvl_name = str(cfg.project.logging.level)
    except Exception:
        pass
    level = getattr(logging, lvl_name.upper(), logging.INFO)

    root = logging.getLogger("leafmachine3")
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")

    if _flag(cfg, "to_console", True):
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    if _flag(cfg, "to_file", True):
        try:
            fh = logging.FileHandler(dirs.logs / "lm3.log")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception:
            pass
    root.propagate = False
    return root


def _flag(cfg, name, default):
    try:
        v = getattr(cfg.project.logging, name)
        return default if v is None else bool(v)
    except Exception:
        return default
