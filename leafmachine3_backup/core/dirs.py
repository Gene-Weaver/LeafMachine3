"""Output directory tree for one LeafMachine3 run.

``_working`` vs ``_tmp_original`` -- they are an INDEX and the BYTES it may point at, not duplicates:

  _working/<stem>.jpg    one symlink per input, the uniform handle every stage opens. It points
                         either STRAIGHT AT the untouched original (already RGB JPEG and within
                         max_working_dim) or at that image's normalized copy. Stages never have to
                         know which -- that is the whole point of the indirection. Symlinks only, so
                         the dir costs kilobytes.
  _tmp_original/         the actual normalized copies ingest had to write (RGB conversion and/or
                         downscale to ingest.max_working_dim) -- typically most of the set, and the
                         only heavy part. Kept separate so ``project.output.tmp_dir`` can place these
                         bytes on a fast/large scratch disk while the run output lives elsewhere;
                         with tmp_dir 'auto' they simply sit in the run dir beside _working.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Dirs:
    root: Path        # <output.dir>/<run_name>
    working: Path     # _working: one symlink per input -> the original OR its _tmp_original copy (see module docstring)
    tmp: Path         # _tmp_original: ingest's converted/downscaled copies (the bytes _working may point at)
    crops: Path       # _crops: detection-crop images (WORKING — ruler/segmenter/landmark inference reads these)
    masks: Path       # _specimen_masks: SpecimenSegmenter mask PNGs (WORKING — Reporter reads these)
    ruler_squarify: Path   # _ruler_squarify: RulerClassifier four-tile collages (reused by the lattice CF + its QC)
    ruler_cf_lattice: Path # _ruler_cf_lattice: lattice CF rot/tick rasters (Reporter reads these to redraw QC)
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

    # Ingest's normalized copies (converted / downscaled originals) live in a working dir named
    # ``_tmp_original`` INSIDE the run output. An explicit ``project.output.tmp_dir`` (non-auto) still
    # wins for scratch-on-another-disk setups, but 'auto' no longer scatters ``*_tmp.jpg`` into the run
    # root (which is what happened when the profiler's tuned tmp_dir resolved to the output dir).
    tmp_cfg = str(getattr(cfg.project.output, "tmp_dir", "auto"))
    if tmp_cfg and tmp_cfg.lower() != "auto":
        tmp = Path(tmp_cfg) / run / "_tmp_original"
    else:
        tmp = root / "_tmp_original"

    working, crops, reports, logs = root / "_working", root / "_crops", root / "reports", root / "logs"
    masks = root / "_specimen_masks"
    ruler_squarify = root / "_ruler_squarify"
    ruler_cf_lattice = root / "_ruler_cf_lattice"
    for d in (root, working, crops, masks, ruler_squarify, ruler_cf_lattice, reports, logs):
        d.mkdir(parents=True, exist_ok=True)
    tmp = _ensure_tmp(tmp, root)
    return Dirs(root, working, tmp, crops, masks, ruler_squarify, ruler_cf_lattice,
                reports, logs, root / f"{run}.sqlite")


def _ensure_tmp(tmp: Path, root: Path) -> Path:
    """Create ``tmp``; if that is not permitted (e.g. an unwritable configured scratch dir),
    fall back to ``<root>/_tmp_original`` so a run never dies on the tmp location."""
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp
    except OSError:
        fallback = root / "_tmp_original"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
