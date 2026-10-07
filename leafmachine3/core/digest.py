"""Content digests of model files, computed once and cached by (path, size, mtime).

The hardware profile records which models it was tuned for, so a run can tell when a model really
changed. It used to record size + mtime. A model re-downloaded from the Hub (byte-identical, new
mtime) therefore looked "changed" on every fresh install, and the warning cried wolf; a different
model of the same size would have slipped through. Content is the right identity, and hashing ~1.7
GB of models on every run is the wrong price, so the sha256 is cached against the file's stat
signature: the first run after an install pays a few seconds, later runs pay a stat() per file.

The cache is advisory. Any read/write problem falls back to hashing; a stale entry cannot be
served because the key includes size and mtime_ns of the file as it is now.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger("leafmachine3.digest")

_CHUNK = 8 * 1024 * 1024


def cache_path() -> Path:
    from leafmachine3.core import paths  # noqa: PLC0415 - avoid an import cycle at module load

    return paths.user_cache_dir() / "lm3" / "file_digests.json"


def _key(path: Path) -> tuple[str, str]:
    real = Path(os.path.realpath(path))
    st = real.stat()
    return str(real), f"{st.st_size}:{st.st_mtime_ns}"


def _read_cache(cache: Path) -> dict:
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_cache(cache: Path, data: dict) -> None:
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
        os.replace(tmp, cache)
    except OSError as exc:                          # read-only home, full disk: hashing still worked
        log.debug("could not update the digest cache %s: %s", cache, exc)


def sha256_file(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def file_digests(files: Iterable[os.PathLike | str], cache: Optional[Path] = None) -> dict[str, str]:
    """``{str(path): sha256}`` for every existing file in ``files`` (missing ones are skipped)."""
    cache = cache if cache is not None else cache_path()
    data = _read_cache(cache)
    out: dict[str, str] = {}
    dirty = False
    for f in files:
        p = Path(f)
        try:
            real, sig = _key(p)
        except OSError:
            continue
        hit = data.get(real)
        if isinstance(hit, dict) and hit.get("sig") == sig and isinstance(hit.get("sha256"), str):
            out[str(f)] = hit["sha256"]
            continue
        digest = sha256_file(real)
        data[real] = {"sig": sig, "sha256": digest}
        out[str(f)] = digest
        dirty = True
    if dirty:
        _write_cache(cache, data)
    return out


def combined_digest(files: Iterable[os.PathLike | str], cache: Optional[Path] = None) -> Optional[str]:
    """One location-independent identity for a SET of files: sha256 over their sorted digests.

    ``None`` when none of the files exist. Sorting the digests (not the paths) makes the identity
    depend on content only, so the same models in another folder or another checkout match.
    """
    digests = sorted(file_digests(files, cache).values())
    if not digests:
        return None
    return hashlib.sha256("\n".join(digests).encode("ascii")).hexdigest()
