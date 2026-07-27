"""Output directory tree for one LeafMachine3 run."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dirs:
    root: Path        # <output.dir>/<run_name>
    working: Path     # working-copy symlinks
    tmp: Path         # converted/resized copies
    crops: Path       # _crops: detection-crop images (WORKING — ruler/segmenter/landmark inference reads these)
    reports: Path     # Reporter outputs
    logs: Path
    db_path: Path


def build_dirs(cfg) -> Dirs:
    """Create the run's directory tree and return the resolved paths.

    tmp uses the machine-tuned ``hardware_settings.tmp_dir`` when the YAML tmp_dir is
    ``auto`` and a profile is bound; otherwise ``<root>/_tmp``.
    """
    out = Path(str(cfg.project.output.dir))
    run = str(cfg.project.run_name)
    root = out / run

    tmp_cfg = str(getattr(cfg.project.output, "tmp_dir", "auto"))
    hw = getattr(cfg, "hardware", None)
    hw_tmp = getattr(hw, "tmp_dir", None) if hw is not None else None
    tmp = (Path(hw_tmp) / run) if (tmp_cfg == "auto" and hw_tmp) else (root / "_tmp")

    working, crops, reports, logs = root / "_working", root / "_crops", root / "reports", root / "logs"
    for d in (root, working, crops, reports, logs):
        d.mkdir(parents=True, exist_ok=True)
    tmp = _ensure_tmp(tmp, root)
    return Dirs(root, working, tmp, crops, reports, logs, root / f"{run}.sqlite")


def _ensure_tmp(tmp: Path, root: Path) -> Path:
    """Create ``tmp``; if that is not permitted (e.g. an unwritable tuned scratch dir),
    fall back to ``<root>/_tmp`` so a run never dies on the tmp location."""
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
    except OSError:
        fallback = root / "_tmp"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
