"""leafmachine3.core.ingest -- image ingestion & working-copy normalization.

``machine3()`` calls :class:`ImageIngestor` right after the DB is built and BEFORE
stage 1. It guarantees the immutability contract: the ORIGINAL images are strictly
READ-ONLY and the pipeline only ever operates on a working-copy SYMLINK.

Why symlink-to-tmp rather than copy-everything: the vast majority of herbarium inputs
are already usable, so we symlink them in place and only materialize a real ``.jpg`` in
``_tmp`` for the small fraction that need conversion (non-JPEG) or downscaling (long
side above ``ingest.max_working_dim``). This keeps disk usage minimal, preserves
provenance (the symlink target is exactly what the pipeline read), and leaves exactly
ONE downstream code path (every specimen has a ``working_path`` that is a readable JPEG
or symlink to one).

The ingestor is idempotent and resumable: a re-run skips originals whose ``(size,
mtime)`` signature is unchanged and whose working symlink still resolves. Files that are
not decodable images are quarantined with a marker under ``INVALID/`` -- the original is
never moved or modified.
"""
from __future__ import annotations

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from leafmachine3.core.dirs import build_dirs
from leafmachine3.core.imaging import MAX_IMAGE_PIXELS, configure_pillow
from leafmachine3.core.records import SpecimenRecord

#: PIL.Image with LM3's decompression-bomb limit (core/imaging.py): herbarium sheets are huge.
Image = configure_pillow()

log = logging.getLogger(__name__)

#: Extensions that are already in the target working format (no conversion needed).
JPEG_EXTS = {".jpg", ".jpeg"}
#: Every raster format we know how to decode & normalize.
SUPPORTED = JPEG_EXTS | {".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass(frozen=True)
class _Meta:
    """Decoded probe result for one image."""

    w: int
    h: int
    mode: str
    fmt: str

    @property
    def wh(self) -> tuple[int, int]:
        return self.w, self.h


class ImageIngestor:
    """Scan input dirs, normalize where needed, and register specimens in the DB.

    Parameters
    ----------
    cfg:
        The resolved :class:`~leafmachine3.core.config.Config`. Reads
        ``project.input`` (dirs / recursive / image_extensions), ``ingest`` and
        ``io_workers()``.
    db:
        The open :class:`~leafmachine3.core.db.ProjectDB` (single writer). The
        ingestor calls ``ingest_signatures``, ``upsert_specimen`` and, on restart,
        ``clear_specimens``.
    """

    def __init__(self, cfg, db) -> None:
        self.cfg = cfg
        self.db = db
        self.dirs = build_dirs(cfg)                     # idempotent; gives working/_tmp/root
        self.working = self.dirs.working
        self.tmp = self.dirs.tmp
        self.invalid = self.dirs.root / "INVALID"
        self.max_dim = int(cfg.ingest.max_working_dim)  # long-side cap for the working copy
        self.q = int(cfg.ingest.jpg_quality)            # ~100

    # ------------------------------------------------------------------ public
    def run(self, restart: bool = False) -> None:
        """Ingest every input image. Idempotent + resumable.

        A re-run skips originals whose ``(size, mtime)`` are unchanged and whose
        working symlink still resolves; changed / new / broken ones are re-ingested.

        Parameters
        ----------
        restart:
            When true, wipe the working set (symlinks + normalized ``_tmp`` copies)
            and clear the specimen table before re-scanning from scratch.
        """
        if restart:
            self._clean_working_set()
            self.db.clear_specimens()

        originals = self._scan()
        prior = self.db.ingest_signatures()             # {original_path: (size, mtime)}
        log.info("ingest | %d candidate image(s) under input dirs", len(originals))

        n_workers = max(1, int(self.cfg.io_workers()))
        with ThreadPoolExecutor(n_workers) as pool:     # ingest is I/O-bound
            futs = {pool.submit(self._one, p, prior.get(str(p))): p for p in originals}
            for fut in as_completed(futs):
                try:
                    fut.result()                        # each _one upserts + checkpoints itself
                except Exception:                       # pragma: no cover - defensive
                    log.exception("ingest failed for %s", futs[fut])

    # ----------------------------------------------------------------- per-file
    def _one(self, original: Path, prior: Optional[tuple[int, float]]) -> None:
        """Ingest a single original image (skip if unchanged, else normalize + register)."""
        st = original.stat()
        sig = (st.st_size, st.st_mtime)
        if prior is not None and tuple(prior) == sig and self._link_ok(original):
            return                                      # unchanged -> resume skip

        try:
            meta = self._probe(original)                # verify() then read w/h/mode/format
        except Image.DecompressionBombError as exc:     # larger than 2 x MAX_IMAGE_PIXELS
            self._quarantine(original, "too_large",
                             f"{exc} (LM3 decodes up to {MAX_IMAGE_PIXELS:,} px; see core/imaging.py)")
            return
        except Exception:
            self._quarantine(original, "corrupt")
            return

        needs_norm = original.suffix.lower() not in JPEG_EXTS or max(meta.w, meta.h) > self.max_dim
        if needs_norm:
            src = self._normalize(original, meta)        # RGB jpg into _tmp, downscaled to max_dim
            work_w, work_h = self._probe(src).wh
        else:
            src, (work_w, work_h) = original, (meta.w, meta.h)

        link = self._link_into_working_set(src, original.stem)
        long_orig = max(meta.w, meta.h) or 1
        self.db.upsert_specimen(SpecimenRecord(
            image_name=original.name,
            image_stem=link.stem,
            original_path=str(original),
            working_path=str(link),
            width=work_w,
            height=work_h,
            original_width=meta.w,
            original_height=meta.h,
            work_scale=max(work_w, work_h) / long_orig,
            orig_size_bytes=sig[0],
            orig_mtime=sig[1],
            # Two DIFFERENT facts, deliberately kept apart. `normalized` means a copy had to be
            # written -- true for an RGB/JPEG conversion alone. `downsampled` means pixels were
            # actually thrown away. A small TIFF is the case that separates them: normalized, but
            # not downsampled. Everything downstream measures in the working frame, so
            # `downsampled` is the flag that says whether that frame differs from what came in.
            normalized=(src is not original),
            downsampled=(work_w != meta.w or work_h != meta.h),
        ))

    # ------------------------------------------------------------------ helpers
    def _scan(self) -> list[Path]:
        """Recurse the configured input dirs and collect image-extension files."""
        inp = self.cfg.project.input
        exts = {str(e).lower() for e in inp.get("image_extensions", sorted(SUPPORTED))}
        recursive = bool(inp.get("recursive", True))
        found: list[Path] = []
        seen: set[str] = set()
        for raw in inp.get("dirs", []):
            root = Path(self.cfg.resolve_path(str(raw)))
            if not root.exists():
                log.warning("ingest | input dir does not exist: %s", root)
                continue
            it = root.rglob("*") if recursive else root.glob("*")
            for p in it:
                if not p.is_file() or p.suffix.lower() not in exts:
                    continue
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                found.append(p)
        return sorted(found)

    @staticmethod
    def _probe(path: Path) -> _Meta:
        """Decode header + integrity-check ``path`` and return its dimensions/mode/format.

        ``Image.verify()`` consumes the file object, so we reopen to read the size.
        """
        with Image.open(path) as im:
            im.verify()                                 # raises on truncated/corrupt data
        with Image.open(path) as im:
            w, h = im.size
            return _Meta(int(w), int(h), im.mode, im.format or "")

    def _normalize(self, original: Path, meta: _Meta) -> Path:
        """Write an RGB JPEG copy of ``original`` into ``_tmp`` (downscaled to max_dim)."""
        self.tmp.mkdir(parents=True, exist_ok=True)
        dst = self.tmp / f"{original.stem}_{self._short_hash(original)}_tmp.jpg"
        with Image.open(original) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")                  # collapse L / RGBA / P / CMYK to 3-channel
            long_side = max(im.size)
            if long_side > self.max_dim:
                scale = self.max_dim / float(long_side)
                new_size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
                im = im.resize(new_size, Image.LANCZOS)
            im.save(dst, format="JPEG", quality=self.q)
        return dst

    def _link_into_working_set(self, src: Path, stem: str) -> Path:
        """Create a symlink to ``src`` in the working set; hash-suffix on stem collision."""
        self.working.mkdir(parents=True, exist_ok=True)
        target = src.resolve()
        link = self.working / f"{stem}{src.suffix.lower()}"
        if link.is_symlink() or link.exists():
            try:
                if link.is_symlink() and link.resolve() == target:
                    return link                         # already the right link
            except OSError:
                pass                                    # broken symlink -> replace below
            link = self.working / f"{stem}_{self._short_hash(src)}{src.suffix.lower()}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
        return link

    def _link_ok(self, original: Path) -> bool:
        """Best-effort resume check: a resolving working symlink for ``original`` exists."""
        for name in (f"{original.stem}{original.suffix.lower()}", f"{original.stem}.jpg"):
            link = self.working / name
            if link.is_symlink() and link.exists():     # exists() is False for a broken link
                return True
        return False

    def _quarantine(self, original: Path, reason: str, detail: str = "") -> None:
        """Record a non-decodable input under ``INVALID/`` WITHOUT touching the original."""
        self.invalid.mkdir(parents=True, exist_ok=True)
        marker = self.invalid / f"{original.stem}_{self._short_hash(original)}.{reason}.txt"
        marker.write_text(f"{reason}\t{original}" + (f"\t{detail}" if detail else "") + "\n", encoding="utf-8")
        log.warning("ingest | quarantined (%s): %s%s", reason, original, f" -- {detail}" if detail else "")

    def _clean_working_set(self) -> None:
        """Remove working-set symlinks and normalized ``_tmp`` copies (restart)."""
        for link in list(self.working.glob("*")):
            if link.is_symlink() or link.is_file():
                link.unlink(missing_ok=True)
        for f in list(self.tmp.glob("*_tmp.jpg")):
            f.unlink(missing_ok=True)

    @staticmethod
    def _short_hash(path: Path) -> str:
        """Stable 8-char digest of an absolute path (collision-suffix / tmp naming)."""
        return hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
