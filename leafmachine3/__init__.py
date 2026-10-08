"""LeafMachine3 package.

``__version__`` is the ONE authoritative LM3 version at runtime. It is read from the installed
distribution metadata, whose source is ``version`` in ``pyproject.toml``.

This exists because the tree carried three disagreeing versions: ``pyproject.toml`` said 0.1.0
while the server, ``/healthz``, the hardware profile and ``app/package.json`` all said 3.0.0. The
runtime launch manifest reads installed metadata, so it would have recorded 0.1.0 for a run whose
``/healthz`` advertised 3.0.0 -- and plan Step 5b adds protocol and instance verification that
compares exactly these values across a process boundary. One source, resolved once.
"""
from __future__ import annotations

from pathlib import Path

#: The VERSION file at the checkout root (beside uv.lock) is the one place the LM3 version is
#: written. ``pyproject.toml`` reads it at build time (``[tool.setuptools.dynamic]``), so installed
#: metadata and this fallback are the same number; the fallback only matters for a bare checkout.
_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def _read_version_file() -> str | None:
    try:
        text = _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


_FALLBACK_VERSION = _read_version_file() or "0.0.0+unknown"


def _resolve_version() -> str:
    # In a checkout the VERSION file wins: an editable install freezes its metadata at `uv sync`
    # time, so after a bump the metadata lags the file until the next sync. An installed wheel has
    # no VERSION file beside the package and reads its own metadata, which was built FROM the file.
    from_file = _read_version_file()
    if from_file:
        return from_file
    try:
        from importlib.metadata import PackageNotFoundError, version  # noqa: PLC0415
    except ImportError:                                   # pragma: no cover - Python < 3.8 only
        return _FALLBACK_VERSION
    try:
        return version("leafmachine3")
    except PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = _resolve_version()

__all__ = ["__version__"]
