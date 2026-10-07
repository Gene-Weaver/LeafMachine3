"""Run directory tree: where ingest's normalized copies (_tmp_original) land.

``project.output.tmp_dir`` is the user knob for relocating ONLY those bulky derived inputs (the
RGB-converted / downscaled copies) onto a scratch disk; everything else -- including the _working
symlink index that points at them -- always stays in the run output dir.
"""
from __future__ import annotations

import types
from pathlib import Path

from leafmachine3.core.dirs import build_dirs


def _cfg(out: str, tmp: str):
    return types.SimpleNamespace(
        project=types.SimpleNamespace(
            run_name="r1", output=types.SimpleNamespace(dir=out, tmp_dir=tmp)))


def test_tmp_dir_auto_lands_in_the_run_dir(tmp_path: Path) -> None:
    d = build_dirs(_cfg(str(tmp_path / "runs"), "auto"))
    assert d.tmp == tmp_path / "runs" / "r1" / "_tmp_original"
    assert d.tmp.is_dir()
    # the normalized copies must NOT scatter into the run root (they used to, as <root>/*_tmp.jpg)
    assert d.tmp.parent == d.root and d.tmp != d.root


def test_tmp_dir_can_be_relocated_to_scratch(tmp_path: Path) -> None:
    d = build_dirs(_cfg(str(tmp_path / "runs"), str(tmp_path / "scratch")))
    assert d.tmp == tmp_path / "scratch" / "r1" / "_tmp_original"
    assert d.tmp.is_dir()
    # only the heavy copies move; the run tree (incl. the _working index that symlinks INTO them) stays
    assert d.root == tmp_path / "runs" / "r1"
    assert d.working == d.root / "_working" and d.reports == d.root / "reports"


def test_working_and_tmp_are_distinct_dirs(tmp_path: Path) -> None:
    d = build_dirs(_cfg(str(tmp_path / "runs"), "auto"))
    assert d.working != d.tmp                       # index vs the bytes it points at
    assert d.working.is_dir() and d.tmp.is_dir()


def test_unwritable_tmp_dir_falls_back_to_the_run_dir(tmp_path: Path) -> None:
    """A bad scratch path must never kill a run -- ingest just writes into the run dir instead."""
    d = build_dirs(_cfg(str(tmp_path / "runs"), "/proc/lm3_cannot_write_here"))
    assert d.tmp == d.root / "_tmp_original"
    assert d.tmp.is_dir()
