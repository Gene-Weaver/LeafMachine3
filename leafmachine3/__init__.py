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

#: Fallback used only when running from a checkout with no installed distribution metadata. Keep it
#: equal to ``pyproject.toml``'s ``version``; ``tests/test_version_identity.py`` enforces that.
_FALLBACK_VERSION = "3.0.0"


def _resolve_version() -> str:
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
