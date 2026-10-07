"""leafmachine3.server.results_api -- browse a finished run's media + inspect its project DB.

Two jobs, one module, because they answer the same question from two sides: "what did this run
actually produce?"

  MEDIA   A run writes thousands of files into ``<run>/reports/`` under a fixed two-level layout
          (``<Category>/<Sub_Category>/<stem>__<LABEL>__x_y_x_y.<ext>``). We index that tree once,
          group it by directory, and hand the UI COUNTS plus a PAGINATED file list -- a 2,500-file
          run must never be shipped to the browser in one response, and a grid of 5,000 px sheets
          must never be rendered from the originals. Hence ``/media`` (counts), ``/media/list``
          (pages), ``/file`` (bytes) and ``/thumb`` (a cached, downscaled JPEG).

  DB      Every run carries a SQLite ledger (``<run>/<run>.sqlite``, or ``run.sqlite`` for a job
          submitted to the server). We expose EVERY table and view with full column metadata and a
          paginated, filterable, sortable row reader -- opened read-only, WAL-safe, so a live run
          can be inspected while it writes.

SECURITY -- the two rules this module exists to enforce:

  1. PATH TRAVERSAL. Every ``path`` parameter is a run-root-relative POSIX path. It is normalized
     LEXICALLY first (``..`` segments collapsed) and rejected if it escapes the root, then checked
     again after ``realpath``. The one deliberate exception is ``_working/`` and ``_tmp_original/``,
     which LM3 fills with SYMLINKS to the untouched originals -- those legitimately resolve outside
     the run dir, so the escape is allowed there and NOWHERE else. See :func:`safe_path`.

  2. SQL INJECTION. Table, view and column names cannot be parameterized in SQLite, so every one is
     WHITELISTED against the live schema (exact string match against ``sqlite_master`` /
     ``PRAGMA table_info``) before it is quoted into a statement. Every VALUE is bound. The optional
     free-text SELECT box additionally runs behind ``mode=ro`` + ``PRAGMA query_only`` + a
     ``set_authorizer`` hook that permits only SELECT/READ, so ``ATTACH``/``PRAGMA``/DDL cannot even
     be compiled. See :func:`_readonly_connect` and :func:`run_query`.

``fastapi`` is imported lazily inside :func:`router`, exactly like ``server.app.create_app`` and
``server.metrics.router``, so this module imports cleanly on a base install. ``PIL``/``cv2`` are
optional too -- without either, thumbnails 501 and everything else still works.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
import secrets
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Iterator, Optional

from leafmachine3.core import paths
from leafmachine3.core.paths import PathsError

log = logging.getLogger("leafmachine3.server.results")

# --------------------------------------------------------------------------- #
# Tunables (env-overridable so a slow disk / huge run can be adjusted without a code change)
# --------------------------------------------------------------------------- #
RUNS_CACHE_TTL_S = float(os.environ.get("LM3_RUNS_CACHE_TTL_S", "5"))
MEDIA_CACHE_TTL_S = float(os.environ.get("LM3_MEDIA_CACHE_TTL_S", "8"))
SCAN_MAX_DEPTH = int(os.environ.get("LM3_RUNS_SCAN_DEPTH", "5"))
QUERY_TIMEOUT_S = float(os.environ.get("LM3_DB_QUERY_TIMEOUT_S", "10"))

#: Version of the run-REFERENCE shape (:func:`run_ref`) that ``GET /v1/runs/-/selector`` returns.
#: Bumped only when a field changes meaning; new optional fields do not bump it.
RUN_REF_SCHEMA_VERSION = 1

MEDIA_LIST_DEFAULT = 200
MEDIA_LIST_MAX = 2000
TABLE_ROWS_DEFAULT = 100
TABLE_ROWS_MAX = 5000
QUERY_ROWS_DEFAULT = 500
QUERY_ROWS_MAX = 5000
SAMPLES_DEFAULT = 4
SAMPLES_MAX = 32

# Thumbnails snap UP to one of these widths. A fixed ladder (rather than an arbitrary ``w``) is what
# keeps <run>/_thumbs/ bounded -- otherwise a resizing grid would bake a new cache tier per pixel.
THUMB_WIDTHS: tuple[int, ...] = (96, 128, 192, 256, 384, 512, 768, 1024)
THUMB_QUALITY = 82

# Run-dir children that are LM3 machinery, not a browsable media category. ``_thumbs`` is ours.
_RUN_INTERNAL = frozenset({"logs", "_thumbs"})
# Working dirs: real, occasionally useful intermediates. Grouped separately from reports/.
_WORKING_DIRS: tuple[str, ...] = (
    "_crops", "_specimen_masks", "_ruler_squarify", "_ruler_cf_lattice", "_working",
)
# Directories the run SCANNER must never descend into (heavy, and never contain a nested run).
_SCAN_PRUNE = frozenset({
    "reports", "logs", "_thumbs", "_crops", "_working", "_tmp", "_tmp_original",
    "_specimen_masks", "_ruler_squarify", "_ruler_cf_lattice",
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "models", "site-packages",
})
# The ONLY run-relative prefixes whose symlinks may resolve outside the run dir. LM3 builds
# ``_working/`` as a symlink farm pointing at the untouched originals (see core/dirs.py), so a
# blanket "realpath must stay inside the root" rule would 403 every original image.
_SYMLINK_ESCAPE_OK = frozenset({"_working", "_tmp_original", "_tmp"})

_IMAGE_EXTS = frozenset({"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp", "gif"})
_HTML_EXTS = frozenset({"html", "htm"})
_TABLE_EXTS = frozenset({"csv", "tsv"})
_DATA_EXTS = frozenset({"h5", "hdf5", "npy", "npz", "json", "yaml", "yml", "parquet", "pkl", "pt"})
_TEXT_EXTS = frozenset({"txt", "log", "md", "cfg", "ini"})

_EXTRA_MIME = {
    "h5": "application/x-hdf5", "hdf5": "application/x-hdf5",
    "npy": "application/octet-stream", "npz": "application/octet-stream",
    "yaml": "text/yaml", "yml": "text/yaml", "log": "text/plain",
    "sqlite": "application/vnd.sqlite3", "db": "application/vnd.sqlite3",
}

# One-line descriptions of the categories LM3 actually writes. Matched longest-prefix-first against
# the category name so ``Crops/RGB__label`` inherits the ``Crops`` blurb.
_CATEGORY_BLURB: dict[str, str] = {
    "Overlay/Overlay_Summary": "Full sheet with masks, boxes, landmarks and the conversion-factor banner drawn on top.",
    "Overlay/Overlay_Landmarks": "One image per leaf: keypoints, skeleton and the derived measurement panel.",
    "Overlay/Overlay_Petiole": "One image per leaf: the petiole width band, its samples and a measurement panel.",
    "Overlay/Overlay_Specimen_Segmentation": "Two panels per sheet: the annotated sheet beside the background-removed cutout.",
    "Overlay/Overlay_Ruler_Lattice": "Ruler conversion-factor QC: squarify tile, deskewed strip, tick overlays and per-unit combs.",
    "Overlay": "Annotated renders drawn over the source imagery.",
    "Specimen_Masks/Binary_Masks_Specimen_Inverse":
        "Inverse of the whole-specimen mask: white everywhere the sheet is not plant.",
    "Specimen_Masks/RGB_Masks_Specimen_Inverse":
        "The sheet with the plant lifted off it, painted over with report.masks.inverse_fill.",
    "Specimen_Masks/Binary_Masks_Specimen": "Whole-specimen binary mask on the full sheet frame.",
    "Specimen_Masks/RGB_Masks_Specimen": "Whole-specimen RGB cutout on the full sheet frame.",
    "Specimen_Masks": "Binary and RGB masks together: whole-specimen cutouts plus per-class full-sheet and per-crop masks.",
    "Leaf_Data/Bilateral_Symmetry": "Per-leaf symmetry QC: oriented halves, straightened view and mirrored overlap.",
    "Crops": "Raw RGB bounding-box crops of the non-leaf detector classes.",
    "Leaf_Original": "Per-leaf products in the leaf's original orientation (bbox, lamina masks, cutouts).",
    "Leaf_Oriented": "Per-leaf products rotated tip-up (bbox, lamina masks, cutouts).",
    "Leaf_Data/Coordinates": "ECT coordinate archives (.h5): image metadata, ECT matrix and the traced contour.",
    "Leaf_Data/Oriented_Leaf_Radial_ECT": "Polar Euler Characteristic Transform render, one per leaf.",
    "Leaf_Data/Oriented_Leaf_ECT": "Cartesian Euler Characteristic Transform render, one per leaf.",
    "Leaf_Data/Oriented_Leaf_Radial_ECT_Overlay":
        "Polar Euler Characteristic Transform with the traced leaf outline drawn on top, tip up in both.",
    "Leaf_Data": "Shape-analysis products from the ECT module.",
    "Collage": "Leaf collages from the postprocessing tab: the run's high-scoring leaves tiled "
               "into the shape of one primary mask, each PNG paired with a .json of its leaves.",
    "Data": "The run's measurements as CSV: one row per leaf in leaf_measurements.csv, plus "
            "per-specimen, detection, landmark and ruler-CF tables and a data dictionary.",
    "Timing": "Per-module run timing: timing.csv and the timing.html report.",
    "_crops": "Working detection crops the ruler, segmenter and landmark modules read.",
    "_specimen_masks": "Working whole-specimen mask PNGs the Reporter reads.",
    "_ruler_squarify": "Working four-tile ruler collages built for the ruler classifier.",
    "_ruler_cf_lattice": "Working lattice rasters (deskewed strips and tick maps) kept for QC redraws.",
    "_working": "One symlink per input image -- the uniform handle every module opens.",
}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class ResultsError(Exception):
    """A failure with an HTTP status attached.

    Raised by the pure-Python layer so the helpers stay importable (and unit-testable) without
    fastapi; :func:`router` translates these into ``HTTPException``.
    """

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = int(status)
        self.detail = str(detail)


def _bad(detail: str) -> ResultsError:
    return ResultsError(400, detail)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _ext_of(name: str) -> str:
    return Path(name).suffix.lstrip(".").lower()


def file_kind(name: str) -> str:
    """Coarse type used to pick a renderer in the UI: image / html / table / data / text / other."""
    ext = _ext_of(name)
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _HTML_EXTS:
        return "html"
    if ext in _TABLE_EXTS:
        return "table"
    if ext in _DATA_EXTS:
        return "data"
    if ext in _TEXT_EXTS:
        return "text"
    return "other"


def media_type_of(name: str) -> str:
    ext = _ext_of(name)
    if ext in _EXTRA_MIME:
        return _EXTRA_MIME[ext]
    guessed, _enc = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


# The working dirs are named for the filesystem, not for a reader. Spell them out rather than
# title-casing (which would give "Ruler Cf Lattice").
_WORKING_LABELS = {
    "_crops": "Detection crops", "_specimen_masks": "Specimen masks",
    "_ruler_squarify": "Ruler squarify tiles", "_ruler_cf_lattice": "Ruler lattice rasters",
    "_working": "Working images", "_tmp_original": "Normalized originals",
    "rot": "Deskewed strips", "ticks": "Tick maps",
}


def _humanize(segment: str) -> str:
    """``Binary_Masks_Specimen__Leaf`` -> ``Binary Masks Specimen - Leaf``."""
    if segment in _WORKING_LABELS:
        return _WORKING_LABELS[segment]
    parts = [p.replace("_", " ").strip() for p in segment.split("__") if p.strip()]
    return " - ".join(parts) if parts else segment


def category_label(name: str) -> str:
    """Friendly label for a category path, disambiguating siblings by their parent folder.

    ``Overlay/Overlay_Summary`` -> ``Overlay Summary`` (the leaf already names its parent), but
    ``Leaf_Original/Lamina_Mask`` -> ``Leaf Original - Lamina Mask`` so it cannot be confused with
    the identically named folder under ``Leaf_Oriented``.
    """
    segments = [s for s in PurePosixPath(name).parts if s]
    if not segments:
        return name
    leaf = _humanize(segments[-1])
    if len(segments) == 1:
        return leaf
    parent = _humanize(segments[-2])
    if leaf.lower().startswith(parent.lower()):
        return leaf
    return f"{parent} - {leaf}"


def category_blurb(name: str) -> Optional[str]:
    """Longest-prefix blurb for a category (exact match wins, then each parent)."""
    probe = name
    while probe:
        hit = _CATEGORY_BLURB.get(probe)
        if hit:
            return hit
        parent = str(PurePosixPath(probe).parent)
        probe = "" if parent in (".", "/", probe) else parent
    return None


def _sample_spread(items: list, n: int) -> list:
    """Pick ``n`` items spread evenly across ``items`` (not just the first n).

    A preview of four consecutive files from one specimen says nothing about a 2,000-file category;
    an even spread shows the range.
    """
    if n <= 0 or not items:
        return []
    if len(items) <= n:
        return list(items)
    step = (len(items) - 1) / float(n - 1) if n > 1 else 0.0
    return [items[int(round(i * step))] for i in range(n)]


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


# --------------------------------------------------------------------------- #
# Path safety
# --------------------------------------------------------------------------- #
def _normalize_rel(root: Path, raw: str) -> str:
    """Reduce a caller-supplied path to a clean run-relative POSIX path, or raise 403.

    Lexical FIRST: ``..`` is collapsed and any residual escape is refused before the filesystem is
    touched at all. An absolute path is accepted only when it is literally inside the run root
    (``report_manifest.path`` stores absolutes, so the UI can hand one straight back).
    """
    if raw is None:
        raise _bad("path is required")
    text = str(raw).strip().replace("\\", "/")
    if not text:
        raise _bad("path is required")
    if "\x00" in text:
        raise ResultsError(403, "path rejected: NUL byte")

    if text.startswith("/"):
        try:
            rel = os.path.relpath(os.path.normpath(text), str(root))
        except ValueError:                                   # different drives (Windows)
            raise ResultsError(403, "path escapes the run directory")
        text = rel.replace(os.sep, "/")

    parts: list[str] = []
    for seg in text.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if not parts:
                raise ResultsError(403, "path escapes the run directory")
            parts.pop()
            continue
        parts.append(seg)
    if not parts:
        raise _bad("path resolves to the run directory itself")
    return "/".join(parts)


def safe_path(root: Path, raw: str, *, must_exist: bool = True) -> tuple[Path, str]:
    """Resolve ``raw`` inside ``root``. Returns ``(absolute_path, clean_relative_path)``.

    Raises :class:`ResultsError` 403 on any traversal attempt and 404 when the file is missing.
    See the module docstring for why ``_working``/``_tmp_original`` may point outside the root.
    """
    root = Path(root).resolve()
    rel = _normalize_rel(root, raw)
    full = root / rel

    real = Path(os.path.realpath(str(full)))
    if real != root and root not in real.parents:
        # The lexical check already passed, so we only get here via a SYMLINK. Permit it solely for
        # LM3's own symlink farms; anything else is a planted link and is refused.
        if rel.split("/", 1)[0] not in _SYMLINK_ESCAPE_OK:
            raise ResultsError(403, "path escapes the run directory")

    if must_exist:
        if not full.exists():
            raise ResultsError(404, f"no such file: {rel}")
        if not full.is_file():
            raise ResultsError(400, f"not a file: {rel}")
    return full, rel


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """One LM3 run directory: the unit everything in this module is scoped to."""

    id: str
    name: str
    path: Path
    root: Path                       # the search root it was found under
    db_path: Optional[Path] = None
    job_id: Optional[str] = None

    @property
    def reports_dir(self) -> Path:
        return self.path / "reports"

    def summary(self) -> dict:
        st = _run_state(self.db_path)
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            mtime = None
        cached = _MEDIA_CACHE.get(str(self.path))
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "root": str(self.root),
            "rel": _rel_or_name(self.path, self.root),
            "job_id": self.job_id,
            "db": self.db_path.name if self.db_path else None,
            "db_path": str(self.db_path) if self.db_path else None,
            "has_db": bool(self.db_path and self.db_path.is_file()),
            "has_reports": self.reports_dir.is_dir(),
            "mtime": round(mtime, 3) if mtime else None,
            "indexed": cached is not None,
            "n_files": cached.n_files if cached else None,
            "bytes": cached.bytes if cached else None,
            **st,
        }


def _rel_or_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace(os.sep, "/")
    except ValueError:
        return path.name


def _run_id(path: Path) -> str:
    """Readable + collision-free: the folder name plus 8 hex of its absolute path.

    Two runs can share a name (``ACAULE/Cusco`` and ``BUKASOVII/Cusco``), so the id has to carry
    something path-derived; keeping the name in front means the URL still says what it points at.
    """
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", path.name).strip("-") or "run"
    return f"{slug}-{digest}"


def _looks_like_run(path: Path) -> Optional[Path]:
    """Return the run's DB path if ``path`` is an LM3 run directory, else ``None``.

    Canonical signature is ``<dir>/<dir.name>.sqlite`` (``core/dirs.py``); a job submitted to the
    server uses ``run.sqlite``. A run whose DB was moved away is still browsable, so a
    ``reports/`` + (``logs/`` or ``_working/``) pair also counts.
    """
    named = path / f"{path.name}.sqlite"
    if named.is_file():
        return named
    job_db = path / "run.sqlite"
    if job_db.is_file():
        return job_db
    if (path / "reports").is_dir() and ((path / "logs").is_dir() or (path / "_working").is_dir()):
        loose = sorted(path.glob("*.sqlite"))
        return loose[0] if loose else None
    return None


def _iter_run_dirs(root: Path, max_depth: int) -> Iterator[Path]:
    """Breadth-first walk that PRUNES at every run it finds (a run never contains another run)."""
    frontier = [(root, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        try:
            children = sorted(p for p in current.iterdir() if p.is_dir())
        except (OSError, PermissionError):
            continue
        for child in children:
            if child.name in _SCAN_PRUNE or child.name.startswith("."):
                continue
            if _looks_like_run(child) is not None:
                yield child
                continue                                     # prune: do not descend into a run
            if depth + 1 < max_depth:
                frontier.append((child, depth + 1))


_EXTRA_ROOTS: list[Path] = []


def add_run_root(path: str | os.PathLike) -> Path:
    """Register an extra directory to scan for runs (the integrator's hook).

    Useful when the user points ``project.output.dir`` somewhere unusual mid-session; the standard
    roots are re-read from ``LM3_settings.yaml`` on every scan, so this is only for the odd case.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir() and resolved not in _EXTRA_ROOTS:
        _EXTRA_ROOTS.append(resolved)
        _RUNS_CACHE.invalidate()
    return resolved


def run_roots() -> list[Path]:
    """Every directory searched for runs, de-duplicated, in priority order."""
    roots: list[Path] = []

    def _push(candidate: Any) -> None:
        if not candidate:
            return
        try:
            resolved = Path(str(candidate)).expanduser().resolve()
        except (OSError, ValueError):
            return
        if resolved.is_dir() and resolved not in roots:
            roots.append(resolved)

    for extra in _EXTRA_ROOTS:
        _push(extra)

    # Section 3.1 row 5, in order and resolved by the canonical resolver: LM3_RUNS_ROOTS, then the
    # ACTIVE and LAST runtime output roots, then the configured project.output.dir (relative values
    # hang off the settings FILE). Read straight off disk each time so saving the Settings tab --
    # or starting a run from the CLI -- immediately changes what the Results tab can see.
    try:
        for root in paths.runs_roots(env=_env(), runtime_roots=runtime_roots(),
                                     settings_file=_settings_file(),
                                     settings_output_dir=_settings_output_dir()):
            _push(root)
    except PathsError as exc:                                # a broken env must not empty the tab
        log.debug("could not resolve the run-history roots (%s)", exc)

    # Runs submitted through POST /v1/jobs. Imported lazily: server.app must be free to import this
    # module at the top of create_app without a circular import.
    try:
        from leafmachine3.server.app import server_jobs_root

        _push(server_jobs_root())
    except Exception:                                        # noqa: BLE001 - app.py is optional here
        pass

    # ``Path.cwd()``, ``cwd/runs`` and ``cwd/examples_out`` used to be pushed here unconditionally.
    # They are gone (section 3.1: "No path falls back to the current working directory"): with the
    # variable unset they were the ONLY roots, so the Results tab listed whatever happened to sit
    # beside the launcher -- and listed nothing at all when the server was started from elsewhere,
    # for the same configuration. An empty list is the honest answer; LM3_RUNS_ROOTS is the knob.
    return roots


def _env() -> Any:
    """The environment with the deprecated aliases folded in (see ``app.server_env``)."""
    from leafmachine3.server.app import server_env

    return server_env()


def _settings_file() -> Optional[Path]:
    """The canonical settings file (section 3.1 row 1), or ``None`` if it cannot be resolved."""
    try:
        from leafmachine3.server.app import canonical_settings_path

        return canonical_settings_path()
    except Exception as exc:                                 # noqa: BLE001 - never fail discovery
        log.debug("could not resolve the settings file (%s)", exc)
        return None


def _settings_output_dir() -> Optional[str]:
    """``project.output.dir`` verbatim from the canonical settings file (may be relative)."""
    path = _settings_file()
    if path is None or not path.is_file():
        return None
    try:
        import yaml

        with path.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        return ((cfg.get("project") or {}).get("output") or {}).get("dir")
    except Exception as exc:                                 # noqa: BLE001 - never fail discovery
        log.debug("could not read project.output.dir (%s)", exc)
        return None


# --------------------------------------------------------------------------- #
# The runtime registry, read as an OBSERVER (sections 3.1 row 5, 2.5, 2.9)
# --------------------------------------------------------------------------- #
# Everything here READS ``<deployment runtime dir>/active.json`` and ``last.json``. It never writes
# a record, never acquires or probes for the purpose of acquiring, and never signals anything:
# control authority is a retained live child handle (section 2.5) and belongs to ``metrics_api``.
# What the Results tab needs from the registry is exactly two facts -- where the run happening RIGHT
# NOW writes (so it appears in the history list while it runs, whoever launched it), and which row
# of that list it is (so the renderer can offer "follow the active run" instead of guessing from
# mtime). Both are read-only, and both are gated on ``LM3_RUNTIME_V2`` because a registry that
# nothing writes yet must not change what this tab shows.


def _env_quiet() -> Any:
    """:func:`_env`, but a broken/absent ``server.app`` degrades to the raw environment.

    Discovery must never raise: an unreadable environment should cost the runtime roots, not the
    whole Results tab.
    """
    try:
        return _env()
    except Exception as exc:                                 # noqa: BLE001 - never fail discovery
        log.debug("could not fold the server environment (%s)", exc)
        return os.environ


def _runtime_v2() -> bool:
    """THE feature-flag read for this module.

    ``execution.runtime_v2_enabled`` is the one helper (Step 3); this module does not grow its own
    ``os.environ`` check. Imported lazily to avoid a module-import cycle. A broken runtime is not
    translated to ``False``: doing so would make Results disagree with Start about which runtime is
    active, and would silently restore filesystem guessing under the safety-default cutover.
    """
    from leafmachine3.core.runtime.execution import runtime_v2_enabled

    return bool(runtime_v2_enabled(_env_quiet()))


def _deployment_dir() -> Optional[Path]:
    """``<runtime base>/<canonical deployment key>``, or ``None`` when it cannot be resolved.

    ``check_filesystem=False``: refusing a network-backed runtime directory (section 3.1) is the
    ACQUISITION path's job, and it fails there loudly. A reader that refused as well would empty
    the Results tab for a deployment that is running perfectly well behind
    ``LM3_ALLOW_NETWORK_RUNTIME=1``.
    """
    try:
        return paths.deployment_runtime_dir(env=_env_quiet(), create=False, check_filesystem=False)
    except Exception as exc:                                 # noqa: BLE001 - never fail discovery
        log.debug("could not resolve the deployment runtime directory (%s)", exc)
        return None


@dataclass(frozen=True)
class RuntimeView:
    """What the registry says, bounded to what a HISTORY view is allowed to conclude.

    ``record`` is whatever ``active.json`` holds, whatever its classification; ``active`` is that
    same record only when it is ``LIVE`` (the lease is held and the record is non-terminal). The two
    differ for an ABANDONED record -- a run whose writer died -- and the difference matters: its
    output directory still belongs in the history roots, but it is not a live run and the renderer
    must not offer to follow or stop it.
    """

    enabled: bool                                # LM3_RUNTIME_V2
    deployment_dir: Optional[Path] = None
    record: Any = None                           # RuntimeRecord | None -- active.json, as published
    active: Any = None                           # RuntimeRecord | None -- only when LIVE
    last: Any = None                             # RuntimeRecord | None -- last.json
    lease_held: bool = False
    compatible: bool = True
    classification: Optional[str] = None
    message: Optional[str] = None


def runtime_view(*, lease_probe: Optional[Callable[[], bool]] = None) -> RuntimeView:
    """Read the deployment registry. Never raises, never writes, never acquires.

    ``lease_probe`` is injectable for the caller that already knows the answer (``metrics_api``
    holds the child handle) and for tests; the default asks the platform lease adapter, which opens
    its own descriptor, takes a SHARED lock and lets go.

    Child records are deliberately NOT read (``include_children=False``): a ``calibration_pipeline``
    child writes into a scratch directory under the run name ``_lm3_calibration``, and section 2.2
    is explicit that the UI must never mistake it for the user's project.
    """
    if not _runtime_v2():
        return RuntimeView(enabled=False)

    deployment_dir = _deployment_dir()
    if deployment_dir is None or not deployment_dir.is_dir():
        return RuntimeView(enabled=True, deployment_dir=deployment_dir)

    try:
        from leafmachine3.core.runtime import records as _records
        from leafmachine3.core.runtime._types import RecordClassification

        snapshot = _records.read_runtime(
            deployment_dir, lease_probe=lease_probe, include_children=False,
        )
    except Exception as exc:                                 # noqa: BLE001 - never fail discovery
        log.debug("could not read the runtime registry at %s (%s)", deployment_dir, exc)
        return RuntimeView(enabled=True, deployment_dir=deployment_dir)

    record = snapshot.record
    live = snapshot.compatible and snapshot.classification is RecordClassification.LIVE
    return RuntimeView(
        enabled=True,
        deployment_dir=deployment_dir,
        record=record,
        active=record if live else None,
        last=_read_last_record(deployment_dir),
        lease_held=bool(snapshot.active),
        compatible=bool(snapshot.compatible),
        classification=snapshot.classification.value if snapshot.classification else None,
        message=snapshot.message,
    )


def _read_last_record(deployment_dir: Path) -> Any:
    """``last.json``, sanitized, or ``None`` when it is absent, malformed or from a newer build.

    ``RecordStore.read_last()`` is the writer-scoped twin of this: a store is constructed with the
    ``run_id`` whose writes it is allowed to make, and a server observing somebody else's
    deployment has no such id. Same three steps, same sanitize, no fabricated ownership.
    """
    try:
        from leafmachine3.core.runtime import records as _records

        payload, error = _records.read_json_file(deployment_dir / paths.LAST_RECORD_FILENAME)
        if payload is None or error is not None:
            return None
        return _records.sanitize_record(_records.record_from_dict(payload))
    except Exception as exc:                                 # noqa: BLE001 - never fail discovery
        log.debug("could not read last.json in %s (%s)", deployment_dir, exc)
        return None


def _output_root(run_dir: Any) -> Optional[Path]:
    """The directory a run's own directory sits IN -- i.e. a ``project.output.dir``-shaped root.

    ``None`` for anything that is not an absolute path with a real parent. The filesystem root is
    refused explicitly: a record naming ``/run`` would otherwise hand the scanner ``/`` and a
    five-level walk of the whole machine.
    """
    if not run_dir:
        return None
    try:
        path = Path(str(run_dir))
    except (OSError, ValueError):
        return None
    if not path.is_absolute():                               # records store absolute paths
        return None
    parent = path.parent
    if parent == path or str(parent) == path.anchor:
        return None
    return parent


def runtime_roots(*, lease_probe: Optional[Callable[[], bool]] = None) -> list[Path]:
    """Section 3.1 row 5, slot 2: the output roots named by the ACTIVE and LAST root records.

    Three paths per record, because section 2.10 gives a run three storage roles and they are the
    same directory only in in-place mode: ``run_dir`` (browsable), ``artifact_dir`` (persistent
    project storage) and ``active_state_dir`` (node-local scratch, where a STAGED run's live
    ledger actually is). Their PARENTS are the roots; the scanner finds the run inside.

    ``record``, not ``active``: a run whose writer died is not live, but its output is still
    history and the tab that lists finished runs should keep finding it.

    A ``hardware_setup`` root contributes nothing -- it has no project block at all (invariant 6).
    """
    view = runtime_view(lease_probe=lease_probe)
    roots: list[Path] = []
    for record in (view.record, view.last):
        project = getattr(record, "project", None)
        if project is None:
            continue
        for candidate in (project.run_dir, project.artifact_dir, project.active_state_dir):
            root = _output_root(candidate)
            if root is not None and root not in roots:
                roots.append(root)
    return roots


# --------------------------------------------------------------------------- #
# Run references -- the one shape the renderer selects a run BY
# --------------------------------------------------------------------------- #

def _ref(
    *,
    source: str,
    run_name: str,
    run_dir: str,
    ref_id: Optional[str] = None,                            # the wire field is "id"
    run_id: Optional[str] = None,
    db_path: Optional[str] = None,
    activity: Optional[str] = None,
    state: Optional[str] = None,
    live: bool = False,
) -> dict:
    return {
        "schema_version": RUN_REF_SCHEMA_VERSION,
        "source": source,
        "id": ref_id,
        "run_id": run_id,
        "run_name": run_name,
        "run_dir": run_dir,
        "db_path": db_path,
        "activity": activity,
        "state": state,
        "live": live,
    }


def run_ref(run: Run, *, source: str = "history") -> dict:
    """The reference shape for a run discovered on disk. ``id`` is what every ``/v1/runs/{run}``
    route keys off, so a selector row is directly usable as a link target."""
    return _ref(
        source=source,
        ref_id=run.id,
        run_name=run.name,
        run_dir=str(run.path),
        db_path=str(run.db_path) if run.db_path else None,
    )


def _record_ref(record: Any, *, source: str, live: bool, by_path: dict[str, Run]) -> Optional[dict]:
    """The reference shape for a runtime record, joined to the discovered run when there is one.

    ``id`` is ``None`` when the run directory is not (yet) discoverable -- a run that acquired the
    lease seconds ago has a record before it has a ledger. That is the honest answer: the renderer
    can name and follow the run without being handed a ``/v1/runs/{id}`` link that would 404.
    """
    project = getattr(record, "project", None)
    if record is None or project is None:                    # hardware_setup: no project, no ref
        return None
    if live:
        db_path = project.active_db_path                     # the ledger the run is writing NOW
    else:
        db_path = project.archived_db_path or project.active_db_path
    match = by_path.get(_key(project.run_dir)) or by_path.get(_key(project.active_state_dir))
    return _ref(
        source=source,
        ref_id=match.id if match is not None else None,
        run_id=record.run_id,
        run_name=project.run_name,
        run_dir=project.run_dir,
        db_path=db_path,
        activity=record.activity.value,
        state=record.state.value,
        live=live,
    )


def _key(path: Any) -> str:
    """A comparison key for two spellings of one directory (symlinked scratch, ``..``, ``~``)."""
    try:
        return str(Path(str(path)).expanduser().resolve())
    except (OSError, ValueError):
        return str(path)


def run_selector(
    *,
    refresh: bool = False,
    limit: int = 0,
    lease_probe: Optional[Callable[[], bool]] = None,
) -> dict:
    """Payload for ``GET /v1/runs/-/selector`` -- the history list plus who is live.

    This exists because ``GET /v1/runs`` has a frozen key set (its envelope and row shapes are
    pinned as a preserved contract) and the Step 6 renderer needs three things that are not in it:
    which row is the run happening right now, which row is the most recent finished run, and a
    stable reference shape it can hand to the other tabs as ``view.runRef``.

    ``runtime`` is the honesty block: with the flag off, or the registry unreadable, or a record
    written by a newer LM3 (section 2.9), ``active: null`` means UNKNOWN, not idle -- and the
    renderer must say so rather than render an idle machine.
    """
    view = runtime_view(lease_probe=lease_probe)
    runs = discover_runs(refresh=refresh)
    by_path = {_key(run.path): run for run in runs}

    active_ref = _record_ref(view.active, source="runtime-active", live=True, by_path=by_path)
    last_ref = _record_ref(view.last, source="runtime-last", live=False, by_path=by_path)

    marked: dict[str, dict] = {}
    for ref in (active_ref, last_ref):                       # active wins a tie: it is the live one
        if ref is not None and ref["id"] and ref["id"] not in marked:
            marked[ref["id"]] = ref

    rows = [marked.get(run.id) or run_ref(run) for run in runs]
    if limit and limit > 0:
        rows = rows[:limit]

    default = None
    for ref in (active_ref, last_ref):
        if ref is not None and ref["id"]:
            default = ref["id"]
            break
    if default is None and rows:
        default = rows[0]["id"]

    return {
        "t": round(time.time(), 3),
        "schema_version": RUN_REF_SCHEMA_VERSION,
        "n": len(rows),
        "roots": [str(p) for p in (_RUNS_CACHE.roots or run_roots())],
        "runs": rows,
        "active": active_ref,
        "last": last_ref,
        "follow": "active" if active_ref else ("last" if last_ref else None),
        "selected_default": default,
        "runtime": {
            "enabled": view.enabled,
            "lease_held": view.lease_held,
            "compatible": view.compatible,
            "classification": view.classification,
            "message": view.message,
        },
    }


@dataclass
class _RunsCache:
    built_at: float = 0.0
    runs: dict[str, Run] = field(default_factory=dict)
    roots: list[Path] = field(default_factory=list)

    def invalidate(self) -> None:
        self.built_at = 0.0


_RUNS_CACHE = _RunsCache()


def discover_runs(*, refresh: bool = False) -> list[Run]:
    """All runs visible under :func:`run_roots`, newest first. Cached for ``RUNS_CACHE_TTL_S``."""
    now = time.time()
    if not refresh and _RUNS_CACHE.runs and (now - _RUNS_CACHE.built_at) < RUNS_CACHE_TTL_S:
        return _sorted_runs(_RUNS_CACHE.runs.values())

    roots = run_roots()
    found: dict[str, Run] = {}
    jobs_root = _jobs_root()
    for root in roots:
        # A root can itself BE a run (`--jobs-root` pointed at one run, or an output dir with a
        # single run inside it that the user opened directly).
        candidates: Iterable[Path] = ([root] if _looks_like_run(root) else []) or _iter_run_dirs(root, SCAN_MAX_DEPTH)
        for path in candidates:
            key = str(path)
            if key in {str(r.path) for r in found.values()}:
                continue
            db = _looks_like_run(path)
            job_id = None
            if jobs_root is not None and path.name == "run" and path.parent.parent == jobs_root:
                job_id = path.parent.name
            run = Run(id=_run_id(path), name=path.name, path=path, root=root, db_path=db, job_id=job_id)
            found[run.id] = run

    _RUNS_CACHE.runs = found
    _RUNS_CACHE.roots = roots
    _RUNS_CACHE.built_at = now
    return _sorted_runs(found.values())


def _jobs_root() -> Optional[Path]:
    try:
        from leafmachine3.server.app import server_jobs_root

        return server_jobs_root().expanduser().resolve()
    except Exception:                                        # noqa: BLE001
        return None


def _sorted_runs(runs: Iterable[Run]) -> list[Run]:
    def key(run: Run) -> float:
        try:
            return run.path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(runs, key=key, reverse=True)


def resolve_run(run_id: str) -> Run:
    """Look a run up by id, by its 8-hex suffix, or by folder name when that is unambiguous.

    A run that finished seconds ago must be reachable immediately, so a miss forces one rescan
    before giving up.
    """
    if not run_id or not str(run_id).strip():
        raise _bad("run id is required")
    wanted = str(run_id).strip()

    for refresh in (False, True):
        runs = discover_runs(refresh=refresh)
        by_id = {r.id: r for r in runs}
        if wanted in by_id:
            return by_id[wanted]
        suffix = [r for r in runs if r.id.endswith("-" + wanted)]
        if len(suffix) == 1:
            return suffix[0]
        by_name = [r for r in runs if r.name == wanted]
        if len(by_name) == 1:
            return by_name[0]
        if len(by_name) > 1 and refresh:
            names = ", ".join(sorted(r.id for r in by_name))
            raise _bad(f"run name '{wanted}' is ambiguous; use one of: {names}")
    raise ResultsError(404, f"unknown run: {wanted}")


def _run_state(db_path: Optional[Path]) -> dict:
    """Roll ``project_status`` up into a one-word state plus module/image counters.

    A DISABLED module is recorded ``done`` with ``started_at == finished_at``
    (``mark_stage_complete_no_work``) -- counted as 'skipped' here so a run that only executed three
    modules does not read as sixteen modules complete.
    """
    blank = {
        "state": "unknown", "modules_total": None, "modules_done": None, "modules_skipped": None,
        "modules_error": None, "n_images": None, "n_images_done": None,
        "started_at": None, "finished_at": None,
    }
    if not db_path or not Path(db_path).is_file():
        return blank
    conn = _readonly_connect(db_path, quiet=True)
    if conn is None:
        return blank
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT stage_key, state, started_at, finished_at FROM project_status ORDER BY stage_order")]
        n_images = _scalar(conn, "SELECT COUNT(*) FROM specimen") or 0
        n_done = _scalar(conn, "SELECT COUNT(*) FROM image_status "
                               "WHERE stage_key='reporter' AND state='done'") or 0
    except sqlite3.Error:
        return blank
    finally:
        conn.close()

    if not rows:
        return blank
    skipped = sum(1 for r in rows if r["state"] == "done" and r["started_at"]
                  and r["started_at"] == r["finished_at"])
    done = sum(1 for r in rows if r["state"] == "done") - skipped
    errored = sum(1 for r in rows if r["state"] == "error")
    running = any(r["state"] == "running" for r in rows)
    pending = any(r["state"] == "pending" for r in rows)

    if running:
        state = "running"
    elif errored:
        state = "error"
    elif pending:
        state = "partial"
    else:
        state = "done"

    starts = [r["started_at"] for r in rows if r["started_at"]]
    ends = [r["finished_at"] for r in rows if r["finished_at"]]
    return {
        "state": state,
        "modules_total": len(rows),
        "modules_done": done,
        "modules_skipped": skipped,
        "modules_error": errored,
        "n_images": n_images,
        "n_images_done": n_done,
        "started_at": min(starts) if starts else None,
        "finished_at": max(ends) if ends and not (running or pending) else None,
    }


def list_runs(*, refresh: bool = False) -> dict:
    """Payload for ``GET /v1/runs`` -- the ONE run listing (section 4 Step 4).

    The envelope is a frozen contract: exactly ``t``/``n``/``roots``/``runs``, with ``n`` equal to
    ``len(runs)``. Nothing is added here for the Step 6 renderer -- "which of these rows is the run
    happening right now" is answered by ``/v1/runs/-/selector``, which also takes the ``?limit=``
    the deleted progress-router duplicate honored. (Deliberately NOT added to this route: the
    duplicate's own contract test uses "ignores ?limit=" as the fingerprint of which router
    answered, so honoring it here would break that test for a reason unrelated to the deletion.)
    """
    runs = discover_runs(refresh=refresh)
    return {
        "t": round(time.time(), 3),
        "n": len(runs),
        "roots": [str(p) for p in (_RUNS_CACHE.roots or run_roots())],
        "runs": [r.summary() for r in runs],
    }


# --------------------------------------------------------------------------- #
# Media index
# --------------------------------------------------------------------------- #
@dataclass
class _MediaFile:
    name: str
    rel: str
    bytes: int
    mtime: float


@dataclass
class _Category:
    name: str                       # e.g. "Overlay/Overlay_Summary"
    group: str                      # "reports" | "working"
    rel: str                        # run-relative dir, e.g. "reports/Overlay/Overlay_Summary"
    files: list = field(default_factory=list)
    bytes: int = 0

    def kind(self) -> str:
        """The category's dominant file kind -- what the UI should render it as."""
        counts: dict[str, int] = {}
        for f in self.files:
            k = file_kind(f.name)
            counts[k] = counts.get(k, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0] if counts else "other"

    def exts(self) -> dict:
        counts: dict[str, int] = {}
        for f in self.files:
            counts[_ext_of(f.name) or "-"] = counts.get(_ext_of(f.name) or "-", 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


@dataclass
class _MediaIndex:
    built_at: float
    categories: dict                # name -> _Category
    n_files: int
    bytes: int
    signature: tuple


# Bounded LRU. An index holds one record per file in the run (name, rel path, size, mtime), so a
# 2,500-file run is cheap but browsing dozens of runs is not -- without a cap this dict is a slow
# leak that only ever grows. Insertion order is the recency order (see media_index).
_MEDIA_CACHE: "OrderedDict[str, _MediaIndex]" = OrderedDict()
_MEDIA_CACHE_MAX = int(os.environ.get("LM3_MEDIA_CACHE_RUNS", "8"))
_MEDIA_CACHE_LOCK = threading.Lock()


def _tree_signature(run: Run) -> tuple:
    """Cheap change detector: mtimes of the run root and its top-level output dirs.

    A directory's mtime changes when entries are added or removed, so this catches a Reporter that
    opened a new category folder without walking thousands of files.
    """
    probes = [run.path, run.reports_dir] + [run.path / d for d in _WORKING_DIRS]
    out: list[float] = []
    for p in probes:
        try:
            out.append(round(p.stat().st_mtime, 3))
        except OSError:
            out.append(-1.0)
    return tuple(out)


def _walk_category(root: Path, base: Path, group: str, prefix: str,
                   categories: dict) -> tuple[int, int]:
    """Index every file under ``base`` into per-directory categories. Returns (n_files, bytes)."""
    n_files = 0
    total = 0
    if not base.is_dir():
        return 0, 0
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _RUN_INTERNAL and not d.startswith("."))
        filenames = [f for f in filenames if not f.startswith(".")]
        if not filenames:
            continue
        here = Path(dirpath)
        rel_dir = os.path.relpath(str(here), str(root)).replace(os.sep, "/")
        rel_from_base = os.path.relpath(str(here), str(base)).replace(os.sep, "/")
        name = prefix if rel_from_base == "." else (f"{prefix}/{rel_from_base}" if prefix else rel_from_base)
        cat = categories.get(name)
        if cat is None:
            cat = _Category(name=name, group=group, rel=rel_dir)
            categories[name] = cat
        for fname in sorted(filenames):
            fpath = here / fname
            try:
                st = fpath.stat()                            # follows symlinks: _working/ needs it
            except OSError:
                continue
            cat.files.append(_MediaFile(fname, f"{rel_dir}/{fname}", int(st.st_size), round(st.st_mtime, 3)))
            cat.bytes += int(st.st_size)
            total += int(st.st_size)
            n_files += 1
    return n_files, total


def media_index(run: Run, *, refresh: bool = False) -> _MediaIndex:
    """Build (or reuse) the run's file index.

    Cached per run for ``MEDIA_CACHE_TTL_S``, in an LRU bounded to ``_MEDIA_CACHE_MAX`` runs so
    browsing a directory full of old runs cannot grow the server's memory without limit.
    """
    key = str(run.path)
    now = time.time()
    signature = _tree_signature(run)
    with _MEDIA_CACHE_LOCK:
        cached = _MEDIA_CACHE.get(key)
        if cached is not None and not refresh:
            fresh = (now - cached.built_at) < MEDIA_CACHE_TTL_S
            if fresh or cached.signature == signature:
                _MEDIA_CACHE.move_to_end(key)            # touched -> most recently used
                return cached

    categories: dict = {}
    n_files, total = _walk_category(run.path, run.reports_dir, "reports", "", categories)
    for wd in _WORKING_DIRS:
        wn, wb = _walk_category(run.path, run.path / wd, "working", wd, categories)
        n_files += wn
        total += wb

    index = _MediaIndex(built_at=now, categories=categories, n_files=n_files, bytes=total,
                        signature=signature)
    with _MEDIA_CACHE_LOCK:
        _MEDIA_CACHE[key] = index
        _MEDIA_CACHE.move_to_end(key)
        while len(_MEDIA_CACHE) > max(1, _MEDIA_CACHE_MAX):
            _MEDIA_CACHE.popitem(last=False)             # drop the least recently used
    return index


def media_categories(run: Run, *, refresh: bool = False, samples: int = SAMPLES_DEFAULT,
                     include_working: bool = True) -> dict:
    """Payload for ``GET /v1/runs/{run}/media``."""
    samples = _clamp(samples, 0, SAMPLES_MAX)
    index = media_index(run, refresh=refresh)

    out: list[dict] = []
    for name in sorted(index.categories):
        cat = index.categories[name]
        if not cat.files:
            continue
        if cat.group == "working" and not include_working:
            continue
        kind = cat.kind()
        out.append({
            "name": cat.name,
            "label": category_label(cat.name),
            "blurb": category_blurb(cat.name),
            "group": cat.group,
            "rel": cat.rel,
            "count": len(cat.files),
            "bytes": cat.bytes,
            "kind": kind,
            "exts": cat.exts(),
            "thumbable": kind == "image",
            "newest": max((f.mtime for f in cat.files), default=None),
            "sample_paths": [f.rel for f in _sample_spread(cat.files, samples)],
        })

    groups: dict[str, dict] = {}
    for cat in out:
        g = groups.setdefault(cat["group"], {"id": cat["group"], "n_files": 0, "n_categories": 0, "bytes": 0})
        g["n_files"] += cat["count"]
        g["bytes"] += cat["bytes"]
        g["n_categories"] += 1
    labels = {"reports": "Reports", "working": "Working files"}
    for gid, g in groups.items():
        g["label"] = labels.get(gid, gid.title())

    return {
        "run": run.summary(),
        "t": round(time.time(), 3),
        "indexed_at": round(index.built_at, 3),
        "n_files": index.n_files,
        "bytes": index.bytes,
        "n_categories": len(out),
        "groups": [groups[g] for g in ("reports", "working") if g in groups],
        "categories": out,
    }


def media_list(run: Run, *, category: Optional[str] = None, offset: int = 0,
               limit: int = MEDIA_LIST_DEFAULT, sort: str = "name", desc: bool = False,
               q: Optional[str] = None, refresh: bool = False) -> dict:
    """Payload for ``GET /v1/runs/{run}/media/list`` -- one page of files."""
    index = media_index(run, refresh=refresh)
    limit = _clamp(limit, 1, MEDIA_LIST_MAX)
    offset = max(0, int(offset or 0))

    if category:
        cat = index.categories.get(category)
        if cat is None:
            raise ResultsError(404, f"unknown category: {category}")
        files = list(cat.files)
        kind = cat.kind()
        group = cat.group
    else:                                                    # no category -> the whole run, flat
        files = [f for name in sorted(index.categories) for f in index.categories[name].files]
        kind = "mixed"
        group = None

    if q:
        needle = str(q).strip().lower()
        files = [f for f in files if needle in f.name.lower()]

    keys = {"name": lambda f: f.name.lower(), "mtime": lambda f: f.mtime, "bytes": lambda f: f.bytes}
    if sort not in keys:
        raise _bad(f"sort must be one of {sorted(keys)}")
    files.sort(key=keys[sort], reverse=bool(desc))

    page = files[offset:offset + limit]
    return {
        "run": run.id,
        "category": category,
        "label": category_label(category) if category else None,
        "group": group,
        "kind": kind,
        "offset": offset,
        "limit": limit,
        "n_total": len(files),
        "n_returned": len(page),
        "sort": sort,
        "desc": bool(desc),
        "q": q or None,
        "files": [{
            "name": f.name,
            "stem": Path(f.name).stem,
            "path": f.rel,
            "ext": _ext_of(f.name),
            "kind": file_kind(f.name),
            "thumbable": file_kind(f.name) == "image",
            "bytes": f.bytes,
            "mtime": f.mtime,
        } for f in page],
    }


# --------------------------------------------------------------------------- #
# Thumbnails
# --------------------------------------------------------------------------- #
def _snap_width(width: int) -> int:
    """Snap up to the nearest cache tier so ``_thumbs/`` cannot grow one tier per requested pixel."""
    want = _clamp(width, THUMB_WIDTHS[0], THUMB_WIDTHS[-1])
    for w in THUMB_WIDTHS:
        if want <= w:
            return w
    return THUMB_WIDTHS[-1]


def _thumb_root(run: Run) -> Path:
    """``<run>/_thumbs``, or a system-temp mirror when the run dir is read-only.

    A run copied off an archive share is a legitimate thing to browse; it must not fail to render
    just because we cannot write beside it.
    """
    primary = run.path / "_thumbs"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        probe = primary / ".writable"
        probe.touch()
        probe.unlink()
        return primary
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "lm3_thumbs" / _run_id(run.path)
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _thumb_path(run: Run, rel: str, width: int) -> Path:
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()
    return _thumb_root(run) / f"w{width}" / digest[:2] / f"{digest[2:18]}.jpg"


def _render_thumb(src: Path, dst: Path, width: int) -> None:
    """Write a downscaled JPEG of ``src`` to ``dst`` (atomically). Prefers PIL, falls back to cv2.

    The temp file name comes from ``mkstemp``, NOT from the pid: a grid fires dozens of parallel
    requests and Starlette serves them from a threadpool, so several threads of the SAME process can
    be rendering into this directory at once. A shared temp name would have them clobber each other
    and ``os.replace`` would fail (or, worse, publish a half-written JPEG).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=dst.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        if not _render_thumb_pil(src, tmp, width) and not _render_thumb_cv2(src, tmp, width):
            raise ResultsError(501, "no image backend available (install Pillow or opencv-python)")
        os.replace(str(tmp), str(dst))                       # atomic: never serve a half-written JPEG
    finally:
        try:
            tmp.unlink()                                     # no-op once the replace consumed it
        except OSError:
            pass


def _render_thumb_pil(src: Path, dst: Path, width: int) -> bool:
    try:
        from PIL import ImageOps

        from leafmachine3.core.imaging import configure_pillow

        Image = configure_pillow()
    except Exception:                                        # noqa: BLE001 - optional dependency
        return False
    try:
        with Image.open(str(src)) as im:
            # draft() lets libjpeg decode a 5,000 px sheet at 1/8 scale in the DCT domain -- the
            # single biggest win in this file; a full decode per tile makes the grid unusable.
            try:
                im.draft("RGB", (width, width))
            except Exception:                                # noqa: BLE001 - non-JPEG has no draft
                pass
            im = ImageOps.exif_transpose(im) or im
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            w, h = im.size
            if w > width:
                im = im.resize((width, max(1, round(h * width / float(w)))), Image.LANCZOS)
            if im.mode != "RGB":
                im = im.convert("RGB")
            im.save(str(dst), "JPEG", quality=THUMB_QUALITY, optimize=True)
        return True
    except Exception as exc:                                 # noqa: BLE001
        log.debug("PIL thumbnail failed for %s (%s)", src, exc)
        return False


def _render_thumb_cv2(src: Path, dst: Path, width: int) -> bool:
    try:
        import cv2
    except Exception:                                        # noqa: BLE001 - optional dependency
        return False
    try:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            return False
        h, w = img.shape[:2]
        if w > width:
            img = cv2.resize(img, (width, max(1, round(h * width / float(w)))), interpolation=cv2.INTER_AREA)
        return bool(cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), THUMB_QUALITY]))
    except Exception as exc:                                 # noqa: BLE001
        log.debug("cv2 thumbnail failed for %s (%s)", src, exc)
        return False


_THUMB_LOCKS: dict[str, Any] = {}
_THUMB_LOCKS_GUARD = threading.Lock()


def _thumb_lock(key: str) -> Any:
    """One lock per (run, file, width) so a grid does not render the same thumbnail N times.

    The browser opens ~6 connections per host and the tiles arrive in bursts, so without this the
    first paint of a folder does duplicate JPEG decodes of 5,000 px sheets.
    """
    with _THUMB_LOCKS_GUARD:
        lock = _THUMB_LOCKS.get(key)
        if lock is None:
            if len(_THUMB_LOCKS) > 1024:                     # bounded: drop the ones nobody holds
                for k, held in list(_THUMB_LOCKS.items()):
                    if not held.locked():
                        _THUMB_LOCKS.pop(k, None)
            lock = _THUMB_LOCKS[key] = threading.Lock()
        return lock


def _thumb_is_current(dst: Path, src: Path) -> bool:
    try:
        return dst.is_file() and dst.stat().st_mtime >= src.stat().st_mtime
    except OSError:
        return False


def thumbnail(run: Run, path: str, width: int = 256) -> tuple[Path, int]:
    """Return ``(cached_jpeg_path, snapped_width)``, rendering it if stale or missing."""
    src, rel = safe_path(run.path, path)
    if file_kind(src.name) != "image":
        raise ResultsError(415, f"not an image: {rel}")
    width = _snap_width(width)
    dst = _thumb_path(run, rel, width)
    if _thumb_is_current(dst, src):
        return dst, width
    with _thumb_lock(f"{run.path}|{rel}|{width}"):
        if _thumb_is_current(dst, src):                      # another thread rendered it while we waited
            return dst, width
        _render_thumb(src, dst, width)
    return dst, width


def clear_thumbs(run: Run) -> dict:
    """Delete the run's thumbnail cache (exposed so the UI can offer a 'rebuild previews')."""
    root = run.path / "_thumbs"
    removed = 0
    if root.is_dir():
        removed = sum(1 for _ in root.rglob("*") if _.is_file())
        shutil.rmtree(root, ignore_errors=True)
    return {"run": run.id, "removed": removed, "path": str(root)}


# --------------------------------------------------------------------------- #
# SQLite inspection
# --------------------------------------------------------------------------- #
_DENY_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"})

# sqlite3 authorizer action codes we allow. Everything else (ATTACH 24, PRAGMA 19, every DDL/DML
# code) falls through to DENY, so a hostile statement fails at COMPILE time, not at execution.
_SQLITE_SELECT = 21
_SQLITE_READ = 20
_SQLITE_FUNCTION = 31
_SQLITE_RECURSIVE = 33
_SQLITE_DENY = 1


def _readonly_connect(db_path: Path, *, quiet: bool = False,
                      authorize: bool = False) -> Optional[sqlite3.Connection]:
    """Open the project ledger read-only.

    ``mode=ro`` is a URI flag, so this also works while LM3 is mid-run: the DB is in WAL mode and a
    reader never blocks the writer. ``query_only`` is belt-and-braces; the authorizer (used only for
    the free-text SELECT box) is the actual guarantee that nothing but a read can be compiled.

    The ``immutable=1`` retry is for archived runs. Reading a WAL database requires SQLite to create
    a ``-shm`` lock file beside it, so a run copied onto a read-only share fails the plain open with
    "attempt to write a readonly database". ``immutable=1`` drops locking and reads the main file
    directly -- which is only correct because we reach it exclusively when the directory is NOT
    writable, and a directory nothing can write to cannot be hosting a live run. (Trade-off: if such
    a copy carries a ``-wal`` file, transactions still parked in it are not visible.)
    """
    path = Path(db_path)
    if not path.is_file():
        return None
    last: Optional[Exception] = None
    for uri in (f"file:{path}?mode=ro", f"file:{path}?mode=ro&immutable=1"):
        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=5.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only = ON")           # must run BEFORE the authorizer
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()   # force a real read now
            if authorize:
                conn.set_authorizer(_authorizer)
            return conn
        except sqlite3.Error as exc:
            last = exc
            if conn is not None:
                conn.close()
    if not quiet:
        log.warning("cannot open %s read-only (%s)", path, last)
    return None


def _authorizer(action: int, arg1: Optional[str], arg2: Optional[str],
                dbname: Optional[str], source: Optional[str]) -> int:
    if action in (_SQLITE_SELECT, _SQLITE_READ, _SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == _SQLITE_FUNCTION:
        return _SQLITE_DENY if str(arg2 or "").lower() in _DENY_FUNCTIONS else sqlite3.SQLITE_OK
    return _SQLITE_DENY


def _open_db(run: Run, *, authorize: bool = False) -> sqlite3.Connection:
    if not run.db_path:
        raise ResultsError(404, f"run '{run.id}' has no project database")
    conn = _readonly_connect(run.db_path, authorize=authorize)
    if conn is None:
        raise ResultsError(404, f"project database not readable: {run.db_path}")
    return conn


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> Optional[int]:
    try:
        row = conn.execute(sql, params).fetchone()
    except sqlite3.Error:
        return None
    if row is None or row[0] is None:
        return None
    return int(row[0])


def _jsonable(value: Any) -> Any:
    """Make a SQLite value JSON-safe. BLOBs become a size marker, never raw bytes."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<blob {len(bytes(value))} B>"
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None                                          # NaN/Inf are not valid JSON
    return value


def _qi(identifier: str) -> str:
    """Quote a SQL identifier. Belt-and-braces only: every identifier reaching this function has
    already been whitelisted against the live schema, but doubling embedded quotes means even a
    table someone named ``weird"name`` produces valid SQL rather than a syntax error we would then
    have to reason about."""
    return '"' + str(identifier).replace('"', '""') + '"'


def _schema_objects(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    rows = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
        "AND name NOT LIKE 'sqlite_%' ORDER BY type DESC, name")
    return [(r["name"], r["type"]) for r in rows]


def _columns_of(conn: sqlite3.Connection, name: str) -> list[dict]:
    """Column metadata for a table OR view. ``name`` must already be whitelisted."""
    out: list[dict] = []
    for row in conn.execute(f"PRAGMA table_info({_qi(name)})"):
        out.append({
            "name": row["name"],
            "type": (row["type"] or "").upper() or None,
            "pk": bool(row["pk"]),
            "notnull": bool(row["notnull"]),
            "default": row["dflt_value"],
        })
    return out


def db_tables(run: Run) -> dict:
    """Payload for ``GET /v1/runs/{run}/db/tables`` -- every table and view, fully described."""
    conn = _open_db(run)
    try:
        objects = _schema_objects(conn)
        tables: list[dict] = []
        for name, kind in objects:
            columns = _columns_of(conn, name)
            n_rows = _scalar(conn, f"SELECT COUNT(*) FROM {_qi(name)}")
            tables.append({
                "name": name,
                "type": kind,
                "n_rows": n_rows,
                "n_columns": len(columns),
                "pk": [c["name"] for c in columns if c["pk"]],
                "columns": columns,
            })
        page_size = _scalar(conn, "PRAGMA page_size")
        page_count = _scalar(conn, "PRAGMA page_count")
    finally:
        conn.close()

    try:
        size = run.db_path.stat().st_size if run.db_path else None
    except OSError:
        size = None
    return {
        "run": run.id,
        "db_path": str(run.db_path) if run.db_path else None,
        "bytes": size,
        "page_size": page_size,
        "page_count": page_count,
        "n_tables": sum(1 for t in tables if t["type"] == "table"),
        "n_views": sum(1 for t in tables if t["type"] == "view"),
        "tables": tables,
    }


# ---- free-text filter -> parameterized SQL --------------------------------- #
_TERM_RE = re.compile(r'"([^"]*)"|(\S+)')
_OPS = ("!=", ">=", "<=", ":", "=", ">", "<")


def _like_escape(text: str) -> str:
    """Escape LIKE wildcards so a literal ``%`` in a filename is not a match-anything."""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _build_where(where: str, columns: list[str]) -> tuple[str, list]:
    """Compile the free-text filter box into a safe WHERE clause.

    Grammar (whitespace-separated terms, ANDed; ``"quoted phrases"`` keep their spaces):
        ``foo``          any column contains foo
        ``col:foo``      that column contains foo
        ``col=foo``      that column equals foo (numeric compare when both sides are numeric)
        ``col!=foo`` / ``col>N`` / ``col<N`` / ``col>=N`` / ``col<=N``
        ``col=NULL``     that column IS NULL   (``col!=NULL`` -> IS NOT NULL)

    Column names are matched against the live schema and refused otherwise; every value is bound.
    """
    clauses: list[str] = []
    params: list = []
    lower = {c.lower(): c for c in columns}

    for match in _TERM_RE.finditer(where or ""):
        term = match.group(1) if match.group(1) is not None else match.group(2)
        if not term:
            continue

        column = op = value = None
        for candidate in _OPS:
            head, sep, tail = term.partition(candidate)
            if sep and head.lower() in lower:
                column, op, value = lower[head.lower()], candidate, tail
                break

        if column is None:                                   # bare term: OR across every column
            needle = f"%{_like_escape(term)}%"
            ors = " OR ".join(f"CAST({_qi(c)} AS TEXT) LIKE ? ESCAPE '\\'" for c in columns)
            clauses.append(f"({ors})")
            params.extend([needle] * len(columns))
            continue

        if value.upper() == "NULL":
            clauses.append(f'{_qi(column)} IS {"NOT NULL" if op == "!=" else "NULL"}')
            continue

        if op == ":":
            clauses.append(f"CAST({_qi(column)} AS TEXT) LIKE ? ESCAPE '\\'")
            params.append(f"%{_like_escape(value)}%")
            continue

        number = _as_number(value)
        if number is None:
            # A non-numeric comparison is a text comparison; CAST keeps it well-defined for INTEGER
            # columns holding text (SQLite is dynamically typed, so that happens).
            clauses.append(f"CAST({_qi(column)} AS TEXT) {op} ?")
            params.append(value)
        else:
            clauses.append(f"{_qi(column)} {op} ?")
            params.append(number)

    return (" AND ".join(clauses) if clauses else ""), params


def _as_number(text: str) -> Optional[float | int]:
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return None


def _has_rowid(conn: sqlite3.Connection, name: str) -> bool:
    """Views and WITHOUT ROWID tables have no rowid; probe once rather than guess."""
    try:
        conn.execute(f"SELECT rowid FROM {_qi(name)} LIMIT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def db_table(run: Run, table: str, *, offset: int = 0, limit: int = TABLE_ROWS_DEFAULT,
             order_by: Optional[str] = None, desc: bool = False,
             where: Optional[str] = None) -> dict:
    """Payload for ``GET /v1/runs/{run}/db/table/{name}`` -- one page of rows.

    ``table`` and ``order_by`` are whitelisted against the live schema before they are quoted in;
    every filter value is bound.
    """
    conn = _open_db(run)
    try:
        objects = dict(_schema_objects(conn))
        if table not in objects:
            raise ResultsError(404, f"unknown table or view: {table}")
        kind = objects[table]
        columns = _columns_of(conn, table)
        names = [c["name"] for c in columns]
        if not names:
            raise ResultsError(404, f"'{table}' exposes no columns")

        limit = _clamp(limit, 1, TABLE_ROWS_MAX)
        offset = max(0, int(offset or 0))

        if order_by:
            if order_by not in names:
                raise _bad(f"order_by must be one of the columns of '{table}'")
            order_sql = f' ORDER BY {_qi(order_by)} {"DESC" if desc else "ASC"}'
        elif _has_rowid(conn, table):
            # Stable pagination needs a deterministic order; rowid is the natural insertion order.
            order_sql = f' ORDER BY rowid {"DESC" if desc else "ASC"}'
        else:
            order_sql = ""

        where_sql, params = _build_where(where or "", names)
        filter_sql = f" WHERE {where_sql}" if where_sql else ""

        n_total = _scalar(conn, f"SELECT COUNT(*) FROM {_qi(table)}{filter_sql}", tuple(params))
        n_rows_all = n_total if not filter_sql else _scalar(conn, f"SELECT COUNT(*) FROM {_qi(table)}")

        sql = f"SELECT * FROM {_qi(table)}{filter_sql}{order_sql} LIMIT ? OFFSET ?"
        try:
            cursor = conn.execute(sql, tuple(params) + (limit, offset))
            rows = [[_jsonable(v) for v in row] for row in cursor.fetchall()]
            out_columns = [d[0] for d in cursor.description]
        except sqlite3.Error as exc:
            raise _bad(f"query failed: {exc}")
    finally:
        conn.close()

    type_by_name = {c["name"]: c["type"] for c in columns}
    return {
        "run": run.id,
        "table": table,
        "type": kind,
        "columns": out_columns,
        "column_types": [type_by_name.get(c) for c in out_columns],
        "pk": [c["name"] for c in columns if c["pk"]],
        "rows": rows,
        "n_total": n_total or 0,
        "n_rows_table": n_rows_all or 0,
        "offset": offset,
        "limit": limit,
        "order_by": order_by,
        "desc": bool(desc),
        "where": where or None,
    }


_SELECT_HEAD = re.compile(r"^\s*(?:WITH\b|SELECT\b|VALUES\b)", re.IGNORECASE)


def run_query(run: Run, sql: str, *, limit: int = QUERY_ROWS_DEFAULT) -> dict:
    """Payload for ``GET/POST /v1/runs/{run}/db/query`` -- one ad-hoc read-only SELECT.

    Three independent gates, because one is never enough for a SQL box:
      1. TEXT   a single statement that starts with SELECT / WITH / VALUES;
      2. HANDLE the connection is ``mode=ro`` + ``PRAGMA query_only``;
      3. COMPILE an authorizer that permits only SELECT/READ, so ATTACH, PRAGMA, DDL and DML are
         rejected before the statement is even prepared.
    A progress handler aborts anything that outstays ``QUERY_TIMEOUT_S``.
    """
    text = (sql or "").strip()
    if not text:
        raise _bad("sql is required")
    body = text[:-1].strip() if text.endswith(";") else text
    if ";" in body:
        raise _bad("only a single statement is allowed")
    if not _SELECT_HEAD.match(body):
        raise _bad("only SELECT queries are allowed")
    if not sqlite3.complete_statement(body + ";"):
        raise _bad("incomplete SQL statement")

    limit = _clamp(limit, 1, QUERY_ROWS_MAX)
    conn = _open_db(run, authorize=True)
    deadline = time.monotonic() + QUERY_TIMEOUT_S
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
    started = time.perf_counter()
    try:
        cursor = conn.execute(body)
        # limit + 1 so we can honestly report whether more rows existed.
        fetched = cursor.fetchmany(limit + 1)
        columns = [d[0] for d in cursor.description] if cursor.description else []
        truncated = len(fetched) > limit
        rows = [[_jsonable(v) for v in row] for row in fetched[:limit]]
    except sqlite3.DatabaseError as exc:
        message = str(exc)
        if "interrupted" in message.lower():
            raise ResultsError(408, f"query exceeded {QUERY_TIMEOUT_S:g}s and was cancelled")
        if "not authorized" in message.lower():
            raise ResultsError(403, "statement not permitted on a read-only connection")
        raise _bad(f"query failed: {message}")
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()

    return {
        "run": run.id,
        "sql": body,
        "columns": columns,
        "rows": rows,
        "n_rows": len(rows),
        "truncated": truncated,
        "limit": limit,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
    }


# --------------------------------------------------------------------------- #
# FastAPI router
# --------------------------------------------------------------------------- #
def _expected_token() -> Optional[str]:
    """The Bearer secret the app is using, or None when the server never minted one."""
    token = os.environ.get("LM3_SERVER_TOKEN")
    if token:
        return token
    try:
        from leafmachine3.server.app import _server_token

        return _server_token()
    except Exception:  # noqa: BLE001 - app.py optional / not yet initialized
        return None


def router(dependencies: Optional[list] = None) -> Any:
    """Build the ``/v1/runs`` APIRouter.

    ``fastapi`` is imported lazily, exactly like ``server.app.create_app`` and
    ``server.metrics.router``, so this module still imports on a base install. Pass the app's auth
    dependency through ``dependencies`` (``[Depends(require_token)]``).

    Routes are plain ``def`` (not ``async def``): every one of them does blocking disk or SQLite
    work, and Starlette runs sync endpoints in its threadpool -- an ``async def`` here would stall
    the event loop that is simultaneously streaming metrics and logs.

    ``dependencies`` is applied PER ROUTE, not to the router, because the four byte-serving routes
    are consumed by ``<img src>`` / ``<a href>`` / the lightbox, which -- exactly like
    ``EventSource`` -- cannot set an ``Authorization`` header. ``api.js`` therefore puts the secret
    on the query string (``mediaUrl()``), so those routes need a guard that accepts ``?token=`` as
    well as the header. Guarding them with the app's header-only ``require_token`` 401s every
    thumbnail and leaves the Results media browser showing nothing but placeholders. This mirrors
    what ``metrics_api.router`` and ``postprocess_api.router`` already do for their SSE routes.
    """
    from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
    from fastapi.responses import FileResponse

    deps = list(dependencies or [])
    api = APIRouter(prefix="/v1/runs", tags=["results"])

    async def media_token(
        token: Optional[str] = Query(default=None, description="Bearer secret, for <img src>"),
        authorization: str = Header(default=""),
    ) -> None:
        """Auth for the byte-serving routes: ``?token=`` OR the Authorization header."""
        if not deps:                                   # mounted without auth -> nothing to check
            return
        expected = _expected_token()
        if not expected:
            return
        if token and secrets.compare_digest(str(token), expected):
            return
        if authorization and secrets.compare_digest(authorization, f"Bearer {expected}"):
            return
        raise HTTPException(status_code=401, detail="invalid or missing token")

    media_deps = [Depends(media_token)]

    def _run(run_id: str) -> Run:
        try:
            return resolve_run(run_id)
        except ResultsError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail)

    def _guard(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ResultsError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail)

    # ---- runs ------------------------------------------------------------- #
    @api.get("", dependencies=deps)
    def get_runs(refresh: bool = False) -> dict:
        return list_runs(refresh=refresh)

    # The literal is TWO segments, so it cannot be shadowed by -- and cannot shadow -- ``/{run}``,
    # which matches one. A run directory may legally be named anything, so a one-segment literal
    # such as ``/_selector`` would quietly make a run of that name unreachable by name.
    @api.get("/-/selector", dependencies=deps)
    def get_run_selector(refresh: bool = False, limit: int = 0) -> dict:
        return run_selector(refresh=refresh, limit=limit)

    @api.get("/{run}", dependencies=deps)
    def get_run(run: str) -> dict:
        return _run(run).summary()

    # ---- media ------------------------------------------------------------ #
    @api.get("/{run}/media", dependencies=deps)
    def get_media(run: str, refresh: bool = False, samples: int = SAMPLES_DEFAULT,
                  include_working: bool = True) -> dict:
        target = _run(run)
        return _guard(media_categories, target, refresh=refresh, samples=samples,
                      include_working=include_working)

    @api.get("/{run}/media/list", dependencies=deps)
    def get_media_list(
        run: str,
        category: Optional[str] = None,
        offset: int = 0,
        limit: int = MEDIA_LIST_DEFAULT,
        sort: str = Query("name", pattern="^(name|mtime|bytes)$"),
        desc: bool = False,
        q: Optional[str] = None,
        refresh: bool = False,
    ) -> dict:
        target = _run(run)
        return _guard(media_list, target, category=category, offset=offset, limit=limit,
                      sort=sort, desc=desc, q=q, refresh=refresh)

    # NOTE: the byte-serving routes declare ``response_class=FileResponse`` and carry NO return
    # annotation. This module uses ``from __future__ import annotations``, so a ``-> FileResponse``
    # hint would reach FastAPI as the unresolvable string "FileResponse" (the class is a local
    # import inside this function) and blow up OpenAPI schema generation for the WHOLE app.
    def _serve_file(run: str, path: str, download: bool):
        target = _run(run)
        full, rel = _guard(safe_path, target.path, path)
        headers = {
            "Cache-Control": "private, max-age=60",
            "X-Content-Type-Options": "nosniff",
        }
        if file_kind(full.name) == "html":
            # Run-produced HTML (timing.html) is same-origin with the app, so a sandbox CSP gives it
            # an opaque origin: it can still run its own inline scripts, but cannot read the
            # localStorage that holds the server token. The UI should ALSO iframe it sandboxed.
            headers["Content-Security-Policy"] = "sandbox allow-scripts"
        if download:
            headers["Content-Disposition"] = f'attachment; filename="{full.name}"'
        return FileResponse(str(full), media_type=media_type_of(full.name), headers=headers,
                            filename=None)

    @api.get("/{run}/file", response_class=FileResponse, dependencies=media_deps)
    def get_file(run: str, path: str, download: bool = False):
        return _serve_file(run, path, download)

    @api.get("/{run}/media/file", response_class=FileResponse, dependencies=media_deps)
    def get_media_file(run: str, path: str, download: bool = False):
        """Alias of ``/file`` -- the shipped ``api.js`` builds ``mediaUrl()`` against this path."""
        return _serve_file(run, path, download)

    def _serve_thumb(run: str, path: str, w: int):
        target = _run(run)
        thumb, width = _guard(thumbnail, target, path, w)
        return FileResponse(str(thumb), media_type="image/jpeg", headers={
            # Content-addressed by (path, width, source mtime) -> safe to cache hard in the browser.
            "Cache-Control": "private, max-age=86400",
            "X-Thumb-Width": str(width),
        })

    @api.get("/{run}/thumb", response_class=FileResponse, dependencies=media_deps)
    def get_thumb(run: str, path: str, w: int = 256):
        return _serve_thumb(run, path, w)

    @api.get("/{run}/media/thumb", response_class=FileResponse, dependencies=media_deps)
    def get_media_thumb(run: str, path: str, w: int = 256):
        """Alias of ``/thumb``, mirroring the ``/media/file`` alias."""
        return _serve_thumb(run, path, w)

    @api.delete("/{run}/thumbs", dependencies=deps)
    def delete_thumbs(run: str) -> dict:
        return _guard(clear_thumbs, _run(run))

    # ---- database --------------------------------------------------------- #
    @api.get("/{run}/db/tables", dependencies=deps)
    def get_db_tables(run: str) -> dict:
        return _guard(db_tables, _run(run))

    @api.get("/{run}/db/table/{name}", dependencies=deps)
    def get_db_table(
        run: str,
        name: str,
        offset: int = 0,
        limit: int = TABLE_ROWS_DEFAULT,
        order_by: Optional[str] = None,
        desc: bool = False,
        where: Optional[str] = None,
    ) -> dict:
        return _guard(db_table, _run(run), name, offset=offset, limit=limit,
                      order_by=order_by, desc=desc, where=where)

    @api.get("/{run}/db/query", dependencies=deps)
    def get_db_query(run: str, sql: str, limit: int = QUERY_ROWS_DEFAULT) -> dict:
        return _guard(run_query, _run(run), sql, limit=limit)

    @api.post("/{run}/db/query", dependencies=deps)
    def post_db_query(run: str, payload: dict = Body(...)) -> dict:
        """POST twin of the SELECT box -- a long query does not belong in a URL."""
        return _guard(run_query, _run(run), str(payload.get("sql", "")),
                      limit=int(payload.get("limit", QUERY_ROWS_DEFAULT)))

    return api


__all__ = [
    "Run", "ResultsError",
    "RuntimeView",
    "router", "add_run_root", "run_roots", "discover_runs", "resolve_run", "list_runs",
    "runtime_view", "runtime_roots", "run_ref", "run_selector",
    "media_index", "media_categories", "media_list", "safe_path",
    "thumbnail", "clear_thumbs", "file_kind", "media_type_of", "category_label", "category_blurb",
    "db_tables", "db_table", "run_query",
]
