"""LeafMachine2 "squarify" ruler preprocessing -- PORTED VERBATIM for inference.

WHY THIS EXISTS
---------------
All three ruler-classifier ensemble members were TRAINED on pre-squarified inputs:
``LM3_Ruler_Classifier/models/common/paths.py`` points training at
``datasets/LM3_RulerCrops-h-squarify/images`` (1440x1440, aspect ratio exactly 1.00).
Serving raw ruler strips instead is a train/serve mismatch: a 755x96 strip put through
``resize_shortest(256) -> center_crop(224)`` keeps only ~11% of the ruler length, so the
model sees a generic patch of ticks and the members disagree three ways.

This file is a verbatim copy of ``LM3_Ruler_Classifier/squarify.py`` (CLI stripped),
which is pixel-identical to the LM2 oracle
``LM3_Ruler_Classifier/reference_lm2_squarify.py`` under ``augment=False`` -- pinned by
``LM3_Ruler_Classifier/test_squarify_equivalence.py``. Keep the two in sync; if the
training-side file changes, re-copy it rather than editing this one.

Note ``_make_img_hor`` is LM2's own step (rotate CCW when portrait), so the "-h" in the
dataset name is produced INSIDE squarify -- no separate rotation is needed at inference.

Depends only on numpy + cv2 + stdlib.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import cv2
import numpy as np

__all__ = ["RulerSquarifier"]

METHODS = ("tile_four", "stack", "quartiles", "nine", "maxheight")


class RulerSquarifier:
    """Versatile, pixel-exact LeafMachine2 ruler squarifier.

    Parameters
    ----------
    sz : int
        Target square edge for a single squarify variant (LM2's ``sz``, default 720).
        For ``method='tile_four'`` the final collage is ``2*sz`` on a side.
    method : str
        One of ``'tile_four', 'stack', 'quartiles', 'nine', 'maxheight'``.
    augment : bool
        When False (default) the transform is fully deterministic and matches the
        oracle's no-flip branch (no randomness is consumed). When True, random
        180-degree flips are applied exactly where -- and in the order -- LM2 applies
        them, driven by ``rng`` for reproducibility.
    quality : int
        JPEG quality used when saving (cv2 ``IMWRITE_JPEG_QUALITY``).
    rng : None | int | random.Random | np.random.Generator
        Source of randomness for ``augment=True``. Ignored when ``augment=False``.
        ``None`` falls back to the global ``random`` module (matching LM2's
        module-level ``random.random()``).
    """

    METHODS = METHODS

    def __init__(self, sz=720, method="tile_four", augment=False, quality=95, rng=None):
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        self.sz = int(sz)
        self.method = method
        self.augment = bool(augment)
        self.quality = int(quality)
        self._rng = self._coerce_rng(rng)

    # ------------------------------------------------------------------ #
    # RNG helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _coerce_rng(rng):
        """Normalise the user-supplied rng into a zero-arg callable -> float [0,1).

        Accepts: None (use the global ``random`` module, like LM2), an int seed,
        a ``random.Random``, a ``np.random.Generator``, or any callable / object
        exposing ``.random()``.
        """
        if rng is None:
            return None  # fall back to the global random module (LM2 behaviour)
        if isinstance(rng, (int, np.integer)):
            return random.Random(int(rng)).random
        if isinstance(rng, np.random.Generator):
            return lambda: float(rng.random())
        if isinstance(rng, random.Random):
            return rng.random
        if hasattr(rng, "random"):
            return rng.random
        if callable(rng):
            return rng
        raise TypeError(
            "rng must be None, an int seed, random.Random, np.random.Generator, "
            f"or a callable; got {type(rng)!r}"
        )

    def _flip_draw(self):
        """Return a float in [0,1); LM2 flips when this is < 0.5."""
        if self._rng is None:
            return random.random()
        return float(self._rng())

    def _maybe_flip180(self, img):
        """Apply LM2's ``if random.random() < 0.5: rotate 180`` only when augmenting.

        When ``augment`` is False this is a no-op AND consumes no randomness, so the
        output is bit-identical to the oracle's no-flip branch.
        """
        if self.augment and self._flip_draw() < 0.5:
            return cv2.rotate(img, cv2.ROTATE_180)
        return img

    # ------------------------------------------------------------------ #
    # Input normalization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_gray_bgr(img):
        """Normalize ANY input to a 3-channel BGR image whose three channels are an
        identical grayscale plane.

        Applied as the very first step of ``_apply`` -- BEFORE orientation/squarify --
        so RGB (or BGRA) inputs are reduced to the grayscale the ruler crops and the
        classifier expect, regardless of how the image was loaded.

        For an already-grayscale input (a single channel, or 3 equal channels as
        ``cv2.imread`` yields for our grayscale crops) this is VALUE-identity:
        OpenCV's BGR2GRAY weights sum to exactly 2**14 in fixed point, so equal
        channels map back to themselves. The 3-channel output is kept so the verbatim
        LM2 pixel ops (which allocate ``[..., 3]`` buffers) run unchanged and the
        squarify result stays pixel-identical to the grayscale pipeline.
        """
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3 and img.shape[2] == 1:
            gray = img[:, :, 0]
        elif img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"unsupported image shape for grayscale conversion: {img.shape}")
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    # Verbatim LM2 pixel primitives (copied from reference_lm2_squarify.py).
    # np.zeros + fill-255 is kept exactly as LM2 writes it.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_img_hor(img):
        """LM2 make_img_hor: rotate CCW if portrait. Returns the SAME array object
        when already landscape -- callers only READ it (the stacking funcs write into
        fresh buffers), so one horizontalized copy is safely shared across variants."""
        try:
            h, w, c = img.shape
        except Exception:
            h, w = img.shape
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    @staticmethod
    def _create_white_bg(img, squarifyRatio, h, w):
        w_plus = w
        imgBG = np.zeros([h, w_plus, 3], dtype=np.uint8)
        imgBG[:] = 255
        imgBG[: img.shape[0], : img.shape[1], :] = img
        return imgBG

    @staticmethod
    def _stack_image(img, squarifyRatio, h, w_plus, showImg):
        wChunk = int(w_plus / squarifyRatio)
        hTotal = int(h * squarifyRatio)
        imgBG = np.zeros([hTotal, wChunk, 3], dtype=np.uint8)
        imgBG[:] = 255

        wStart = 0
        wEnd = wChunk
        for i in range(1, squarifyRatio + 1):
            wStartImg = (wChunk * i) - wChunk
            wEndImg = wChunk * i
            hStart = (i * h) - h
            hEnd = i * h
            imgBG[hStart:hEnd, wStart:wEnd] = img[:, wStartImg:wEndImg]
        return imgBG

    @staticmethod
    def _stack_image_quartile(img, q_increment, h, w, showImg):
        imgBG = np.zeros([h * 2, h * 2, 3], dtype=np.uint8)
        imgBG[:] = 255

        increment = 0
        for row in range(0, 2):
            for col in range(0, 2):
                ONE = row * h
                TWO = (row * h) + h
                THREE = col * h
                FOUR = (col * h) + h

                one = q_increment * increment
                two = (q_increment * increment) + h

                if (increment < 3) and (two < w):
                    imgBG[ONE:TWO, THREE:FOUR] = img[:, one:two]
                else:
                    imgBG[ONE:TWO, THREE:FOUR] = img[:, w - h : w]
                increment += 1
        return imgBG

    @staticmethod
    def _stack_image_nine(img, q_increment, h, w, showImg):
        imgBG = np.zeros([h * 3, h * 3, 3], dtype=np.uint8)
        imgBG[:] = 255

        increment = 0
        for row in range(0, 3):
            for col in range(0, 3):
                ONE = row * h
                TWO = (row * h) + h
                THREE = col * h
                FOUR = (col * h) + h

                one = q_increment * increment
                two = (q_increment * increment) + h

                if (increment < 8) and (two < w):
                    imgBG[ONE:TWO, THREE:FOUR] = img[:, one:two]
                else:
                    imgBG[ONE:TWO, THREE:FOUR] = img[:, w - h : w]
                increment += 1
        return imgBG

    @staticmethod
    def _calc_squarify_ratio(img):
        doStack = False
        h, w, c = img.shape

        ratio = w / h
        ratio_plus = math.ceil(ratio)
        w_plus = ratio_plus * h

        ratio_go = w / h
        if ratio_go > 4:
            doStack = True

        squarifyRatio = 0
        if doStack:
            for i in range(1, ratio_plus):
                if (i * h) < (w_plus / i):
                    continue
                else:
                    squarifyRatio = i - 1
                    break
            while (w % squarifyRatio) != 0:
                w += 1
        return doStack, squarifyRatio, w, h

    @staticmethod
    def _calc_squarify(img, cuts):
        h, w, c = img.shape
        q_increment = int(np.floor(w / cuts))
        return q_increment, w, h

    # ------------------------------------------------------------------ #
    # LM2 squarify variants. Each accepts an optional pre-horizontalized
    # image (`img_hor`); when None it horizontalizes itself, so standalone
    # calls stay correct. The trailing random 180-flip is routed through
    # _maybe_flip180 (no-op + no randomness consumed when augment is False).
    # ------------------------------------------------------------------ #
    def _squarify(self, imgSquarify, img_hor=None):
        """LM2 ``squarify(img, False, True, sz)`` -> sz x sz."""
        imgSquarify = self._make_img_hor(imgSquarify) if img_hor is None else img_hor
        doStack, squarifyRatio, w_plus, h = self._calc_squarify_ratio(imgSquarify)

        if doStack:
            imgBG = self._create_white_bg(imgSquarify, squarifyRatio, h, w_plus)
            imgSquarify = self._stack_image(imgBG, squarifyRatio, h, w_plus, False)

        # makeSquare is always True for our pinned API.
        dim = (self.sz, self.sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation=cv2.INTER_AREA)

        return self._maybe_flip180(imgSquarify)

    def _squarify_maxheight(self, img, h, w):
        """LM2 ``squarify_maxheight(img, h, w, False)`` -> h x w.

        NB: this variant rotates CW (not CCW) when portrait, so it cannot reuse the
        shared landscape image used by the other three. It also flips BEFORE resizing
        in LM2, so the flip is applied to the pre-resize image to stay pixel-exact.
        """
        if img.shape[0] > img.shape[1]:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        img = self._maybe_flip180(img)  # LM2 flips here, before resize

        resized = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_NEAREST)
        return resized

    def _squarify_quartiles(self, imgSquarify, img_hor=None):
        """LM2 ``squarify_quartiles(img, False, True, sz, doFlip=False)`` -> sz x sz.

        ``doFlip`` is always False for our pinned API (the tile_four caller passes
        ``doFlip=showImg`` which is False), so the pre-stack ROTATE_180 is skipped."""
        imgSquarify = self._make_img_hor(imgSquarify) if img_hor is None else img_hor
        # doFlip == False -> skip the unconditional pre-stack ROTATE_180

        q_increment, w, h = self._calc_squarify(imgSquarify, 4)
        imgSquarify = self._stack_image_quartile(imgSquarify, q_increment, h, w, False)

        dim = (self.sz, self.sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation=cv2.INTER_AREA)

        return self._maybe_flip180(imgSquarify)

    def _squarify_nine(self, imgSquarify, img_hor=None):
        """LM2 ``squarify_nine(img, False, True, sz)`` -> sz x sz."""
        imgSquarify = self._make_img_hor(imgSquarify) if img_hor is None else img_hor

        q_increment, w, h = self._calc_squarify(imgSquarify, 9)
        imgSquarify = self._stack_image_nine(imgSquarify, q_increment, h, w, False)

        dim = (self.sz, self.sz)
        imgSquarify = cv2.resize(imgSquarify, dim, interpolation=cv2.INTER_AREA)

        return self._maybe_flip180(imgSquarify)

    def _tile_four(self, img):
        """LM2 ``squarify_tile_four_versions(img, False, True, sz)`` -> 2*sz square.

        Efficiency: stack/quartiles/nine all begin with the SAME CCW-horizontalized
        image (make_img_hor), and none of the downstream ops mutate it in place, so
        we compute it once and reuse. maxheight needs a CW rotation, so it keeps its
        own dedicated path on the original image.

        Flip-draw order matches the oracle exactly: squarify, maxheight, quartiles,
        nine (one Bernoulli draw each when augment is True)."""
        sz = self.sz
        h = int(sz * 2)
        w = int(sz * 2)
        h2 = int(h / 2)
        w2 = int(w / 2)  # noqa: F841  (kept to mirror LM2 exactly)

        img_hor = self._make_img_hor(img)  # computed once, reused 3x

        sq1 = self._squarify(img, img_hor=img_hor)
        sq2 = self._squarify_maxheight(img, h / 2, w / 2)
        sq3 = self._squarify_quartiles(img, img_hor=img_hor)
        sq4 = self._squarify_nine(img, img_hor=img_hor)

        imgBG = np.zeros([h, w, 3], dtype=np.uint8)
        imgBG[:] = 255

        imgBG[0:h2, 0:h2, :] = sq1
        imgBG[:h2, h2:w, :] = sq2
        imgBG[h2:w, :h2, :] = sq3
        imgBG[h2:w, h2:w, :] = sq4

        return imgBG

    # ------------------------------------------------------------------ #
    # Core compute dispatch
    # ------------------------------------------------------------------ #
    def _apply(self, img):
        """Run the configured method on an image; return a 3-channel BGR ndarray.

        The image is first normalized to grayscale (then back to 3-channel) BEFORE any
        orientation check, so RGB/BGRA inputs are reduced to grayscale; this is
        value-identity for already-grayscale inputs (see ``_to_gray_bgr``)."""
        img = self._to_gray_bgr(img)
        m = self.method
        if m == "tile_four":
            return self._tile_four(img)
        if m == "maxheight":
            # Oracle pin: squarify_maxheight(img, sz, sz, False) -> sz x sz.
            return self._squarify_maxheight(img, self.sz, self.sz)
        # stack / quartiles / nine share the CCW-horizontalized image.
        hor = self._make_img_hor(img)
        if m == "stack":
            return self._squarify(img, img_hor=hor)
        if m == "quartiles":
            return self._squarify_quartiles(img, img_hor=hor)
        if m == "nine":
            return self._squarify_nine(img, img_hor=hor)
        raise ValueError(f"unknown method {m!r}")  # pragma: no cover

    # ------------------------------------------------------------------ #
    # Public: single-image transform
    # ------------------------------------------------------------------ #
    def transform(self, image, save_to=None):
        """Squarify one image; ALWAYS return the resulting ndarray.

        Parameters
        ----------
        image : str | os.PathLike | np.ndarray
            A path to read with ``cv2.imread``, or an already-decoded ndarray. May be
            color (3-channel BGR / 4-channel BGRA) or grayscale (single channel or 3
            equal channels); it is converted to grayscale before anything else (RGB
            rulers are normalized to the grayscale the classifier expects).
        save_to : None | str | os.PathLike
            If None, nothing is written (pure inference path).
            If a directory (existing, or a string ending with ``os.sep``), the output
            is written as ``<save_to>/<original filename>`` -- which requires ``image``
            to be a path; otherwise a clear error is raised. Otherwise ``save_to`` is
            treated as a full output file path. Saving never alters the returned array.
        """
        src_path = None
        if isinstance(image, (str, os.PathLike)):
            src_path = os.fspath(image)
            img = cv2.imread(src_path)
            if img is None:
                raise ValueError(f"cv2.imread failed to load image: {src_path!r}")
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise TypeError(
                "image must be a path (str/os.PathLike) or an np.ndarray, "
                f"got {type(image).__name__}"
            )

        out = self._apply(img)

        if save_to is not None:
            self._write(out, save_to, src_path)

        return out

    def _resolve_out_path(self, save_to, src_path):
        """Resolve ``save_to`` into a concrete output file path."""
        save_to = os.fspath(save_to)
        is_dir = os.path.isdir(save_to) or save_to.endswith(os.sep)
        if is_dir:
            if src_path is None:
                raise ValueError(
                    "save_to is a directory but image was passed as an ndarray; "
                    "there is no original filename to use. Pass a full output file "
                    "path as save_to instead."
                )
            os.makedirs(save_to, exist_ok=True)
            return os.path.join(save_to, os.path.basename(src_path))
        parent = os.path.dirname(save_to)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return save_to

    def _write(self, out, save_to, src_path):
        """Write ``out`` to disk, honouring JPEG quality for jpg/jpeg outputs."""
        out_path = self._resolve_out_path(save_to, src_path)
        ext = os.path.splitext(out_path)[1].lower()
        if ext in (".jpg", ".jpeg"):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        else:
            params = []
        if not cv2.imwrite(out_path, out, params):
            raise IOError(f"cv2.imwrite failed for {out_path!r}")
        return out_path

    # ------------------------------------------------------------------ #
    # Public: directory batch build
    # ------------------------------------------------------------------ #
    def process_dir(self, src_dir, dest_dir, workers=16, limit=None):
        """Squarify every ``*.jpg`` in ``src_dir`` into ``dest_dir`` (filenames kept).

        Parallelized with a thread pool (cv2/numpy release the GIL during the heavy
        decode/resize work, so threads give real throughput without pickling cost).
        Per-image failures are caught, counted, and skipped -- the batch never aborts.
        Returns a stats dict.
        """
        src_dir = os.fspath(src_dir)
        dest_dir = os.fspath(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

        paths = sorted(glob(os.path.join(src_dir, "*.jpg")))
        if limit is not None:
            paths = paths[: int(limit)]

        total = len(paths)
        written = 0
        failed = 0
        errors = []

        def _one(p):
            self.transform(p, save_to=dest_dir)
            return p

        workers = max(1, int(workers))
        if total == 0:
            pass
        elif workers == 1:
            # Serial fast-path: avoid pool overhead for tiny / single-worker jobs.
            for p in paths:
                try:
                    _one(p)
                    written += 1
                except Exception as e:  # noqa: BLE001 -- never abort the batch
                    failed += 1
                    errors.append((os.path.basename(p), repr(e)))
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_one, p): p for p in paths}
                for fut in as_completed(futs):
                    p = futs[fut]
                    try:
                        fut.result()
                        written += 1
                    except Exception as e:  # noqa: BLE001 -- never abort the batch
                        failed += 1
                        errors.append((os.path.basename(p), repr(e)))

        return {
            "total": total,
            "written": written,
            "failed": failed,
            "src_dir": src_dir,
            "dest_dir": dest_dir,
            "method": self.method,
            "sz": self.sz,
            "augment": self.augment,
            "errors": errors,
        }


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _derive_dest(src_dir):
    """Derive a sibling data-project dest from a source images dir.

    ``data/<slug>/images`` -> ``data/<slug>-h-squarify/images``. If the path does not
    end in an ``images`` leaf, fall back to appending '-h-squarify' to the dir name.
    """
    src_dir = os.path.normpath(os.fspath(src_dir))
    parent, leaf = os.path.split(src_dir)
    if leaf == "images":
        proj_parent, slug = os.path.split(parent)
        return os.path.join(proj_parent, f"{slug}-h-squarify", "images")
    return f"{src_dir}-h-squarify"


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="LeafMachine2 'squarify' ruler preprocessing (pixel-exact)."
    )
    p.add_argument("--src", required=True, help="Source directory of *.jpg crops.")
    p.add_argument(
        "--dest",
        default=None,
        help="Destination dir. If omitted, derives data/<slug>-h-squarify/images.",
    )
    p.add_argument("--sz", type=int, default=720, help="Per-variant square edge (default 720).")
    p.add_argument(
        "--method",
        default="tile_four",
        choices=list(METHODS),
        help="Squarify method (default tile_four).",
    )
    p.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95).")
    p.add_argument("--workers", type=int, default=16, help="Thread workers (default 16).")
    p.add_argument(
        "--augment",
        action="store_true",
        help="Enable LM2's random 180-deg flips (default off = deterministic).",
    )
    p.add_argument("--limit", type=int, default=None, help="Process at most N images.")
    return p
