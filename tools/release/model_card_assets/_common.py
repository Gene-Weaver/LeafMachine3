"""Shared helpers for the model-card asset renderers in this folder."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]                      # the LM3 checkout
sys.path.insert(0, str(ROOT))
from leafmachine3.core.config import Config  # noqa: E402
from leafmachine3.core.db import ProjectDB  # noqa: E402
from leafmachine3.core.device import Device  # noqa: E402


def load_config(settings: str, stage: str, model_path: str) -> Config:
    """LM3_settings.yaml with one stage's model path pointed at the model being rendered."""
    return Config.load(settings, overrides={"modules": {stage: {"model": {"path": str(Path(model_path).resolve())}}}})


def device(spec: str) -> Device:
    """'cuda:N' -> GPU N (pinned the way LM3 workers pin theirs); 'cpu' -> CPU."""
    if spec.startswith("cuda"):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", spec.split(":")[1] if ":" in spec else "0")
        return Device("cuda", 0)
    return Device("cpu", 0)


def run_dir(source: str) -> Path:
    """A provenance 'source_run' ('LM3/examples_out/<run>') or a report path inside a run -> the run dir."""
    p = Path(source)
    if p.parts and p.parts[0] == "LM3":
        p = ROOT.joinpath(*p.parts[1:])
    while p != p.parent and not (p / f"{p.name}.sqlite").exists():
        p = p.parent
    if not (p / f"{p.name}.sqlite").exists():
        raise SystemExit(f"no LM3 run database found for {source}")
    return p


class Runs:
    """Read-only access to example runs' databases, opened from temporary copies so the runs are never touched."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lm3_card_assets_"))
        self.dbs: dict[Path, ProjectDB] = {}

    def db(self, source: str) -> ProjectDB:
        run = run_dir(source)
        if run not in self.dbs:
            src = run / f"{run.name}.sqlite"
            dst = self.tmp / run.name
            dst.mkdir()
            for suffix in ("", "-wal", "-shm"):
                if Path(f"{src}{suffix}").exists():
                    shutil.copy2(f"{src}{suffix}", dst / f"{src.name}{suffix}")
            db = ProjectDB(dst / src.name)
            db.connect()
            self.dbs[run] = db
        return self.dbs[run]

    @staticmethod
    def specimen(db: ProjectDB, stem: str) -> tuple[int, str]:
        rows = db._query("SELECT specimen_id, working_path FROM specimen WHERE image_stem = ?", (stem,))
        if not rows:
            raise SystemExit(f"specimen {stem} not found in {db.path}")
        wp = Path(str(rows[0]["working_path"]))
        if not wp.is_absolute():                                  # older runs store it relative to the checkout
            wp = ROOT / wp
        return int(rows[0]["specimen_id"]), str(wp)

    def close(self):
        for db in self.dbs.values():
            db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


def published(docs: Path, prov: dict, name: str, render_all: bool) -> bool:
    """Re-render only what the card shows: a file already in docs/ or in the provenance 'kept' list.
    Candidates that were pruned are skipped unless --all (then pick again by hand)."""
    return render_all or name in set(prov.get("kept") or []) or (docs / name).exists()
