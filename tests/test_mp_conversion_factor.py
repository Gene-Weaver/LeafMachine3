"""Tests for the MPConversionFactor stage: linear predictor, model.json load, fit reproducibility."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from leafmachine3.core.db import ProjectDB
from leafmachine3.core.records import SpecimenRecord
from leafmachine3.inference.mp_conversion_factor import MpConversionFactorModel, load_model
from leafmachine3.modules.mp_conversion_factor import MPConversionFactor

#: Where `lm3 models install` put the model: $LM3_MODELS_DIR if set, else the checkout's models/ (the
#: same rule as leafmachine3.modelhub.installer.models_root for a checkout).
_MODEL_DIR = (Path(os.environ["LM3_MODELS_DIR"]).expanduser() if os.environ.get("LM3_MODELS_DIR")
              else Path(__file__).resolve().parents[1] / "models") / "mp_conversion_factor"
_FALLBACK_SLOPE = 2.6744940855608585
_FALLBACK_INTERCEPT = 67.51862180698473


def test_predict_cf_linear_and_rounding() -> None:
    m = MpConversionFactorModel(slope=2.0, intercept=10.0)
    assert m.predict_cf(2000, 1000) == 14.0          # mp = 2.0 -> 2*2 + 10
    assert m.predict_cf(1000, 1000) == 12.0          # mp = 1.0
    m2 = MpConversionFactorModel(slope=1.0 / 3.0, intercept=0.0)
    assert m2.predict_cf(1000, 1000) == 0.33         # mp = 1.0 -> 0.3333.. rounded to 2 dp


def test_megapixels_rounds_to_4dp() -> None:
    m = MpConversionFactorModel()
    assert m.megapixels(1944, 2592) == 5.0388     # 5.038848 -> 4 dp
    assert m.megapixels(4000, 3000) == 12.0
    assert m.megapixels(0, 1000) is None and m.megapixels(None, None) is None


def test_predict_cf_degenerate_returns_none() -> None:
    m = MpConversionFactorModel()
    assert m.predict_cf(0, 1000) is None
    assert m.predict_cf(1000, 0) is None
    assert m.predict_cf(None, None) is None


def test_load_model_json() -> None:
    m = load_model(str(_MODEL_DIR / "model.json"))
    assert m.source == "model.json"
    assert abs(m.slope - _FALLBACK_SLOPE) < 1e-6
    assert abs(m.intercept - _FALLBACK_INTERCEPT) < 1e-6


def test_load_model_missing_falls_back() -> None:
    m = load_model("/no/such/path/model.json")
    assert m.source == "fallback"
    assert m.slope == _FALLBACK_SLOPE and m.intercept == _FALLBACK_INTERCEPT


def test_fit_is_reproducible() -> None:
    """Re-fitting fit_data.csv (numpy OLS) reproduces the shipped model.json coefficients.

    fit_data.csv is training PROVENANCE, published in the model's Hugging Face repo and installed by
    `lm3 models install` as an optional file (the stage itself never reads it). An install made
    before the lock listed it has model.json but not the fit data; that is a valid install, so this
    skips with the command that fetches it rather than failing.
    """
    if not (_MODEL_DIR / "fit_data.csv").is_file():
        pytest.skip(f"no {_MODEL_DIR / 'fit_data.csv'}; fetch it with: "
                    "lm3 models install --actions mp_conversion_factor --force")
    mp, cf = [], []
    with open(_MODEL_DIR / "fit_data.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            mp.append(float(r["mp"]))
            cf.append(float(r["cf"]))
    x = np.sqrt(np.asarray(mp))
    k = float((x @ np.asarray(cf)) / (x @ x))
    model = json.loads((_MODEL_DIR / "model.json").read_text())
    assert model["model"] == "sqrt" and model["frame"] == "working"
    assert model["n"] == len(mp)
    assert abs(k - model["k"]) < 1e-9
    # The form is not an assumption imposed on the data: fitting the exponent FREE lands on ~0.5,
    # which is the physical prediction. If this drifts, the shipped form no longer matches the fit.
    b_free = float(np.polyfit(np.log(mp), np.log(cf), 1)[0])
    assert abs(b_free - 0.5) < 0.02, f"free exponent {b_free:.4f} is no longer ~0.5"


def test_stage_writes_specimen_column(tmp_path) -> None:
    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    sid = db.upsert_specimen(SpecimenRecord(
        image_name="a.jpg", image_stem="a", original_path="/o/a.jpg", working_path="/w/a.jpg",
        width=1600, height=1200, original_width=4000, original_height=3000,
    ))
    project = SimpleNamespace(db=db)
    stage = MPConversionFactor(None)
    items = stage.collect_items(project)
    assert len(items) == 1 and items[0].payload == (4000, 3000)   # ORIGINAL dims, not working
    model = MpConversionFactorModel(slope=2.0, intercept=10.0)
    for it in items:
        stage.persist(project, it, stage.infer(it, model))
    row = db.get_specimen(sid)
    # mp = 4000*3000/1e6 = 12.0 -> 2*12 + 10 = 34.0
    assert row["original_mp"] == 12.0
    assert row["cf_px_per_cm_predicted_by_mp"] == 34.0


def test_sqrt_form_reads_working_dims_and_is_scale_equivariant(tmp_path) -> None:
    """The sqrt form is evaluated on the WORKING dims, and that is not a shortcut -- it is exact.

    ``cf = k*sqrt(MP)`` is homogeneous in the linear resize factor, so predicting on the original
    and scaling by work_scale gives the same number as predicting on the working image directly.
    The linear form is not homogeneous (its intercept does not scale), which is why it must keep
    reading the originals. If this equality ever breaks, the lattice's anchor silently moves frame.
    """
    from leafmachine3.inference.mp_conversion_factor import MpConversionFactorModel

    lin, sq = MpConversionFactorModel(), MpConversionFactorModel(form="sqrt")
    assert (lin.frame, sq.frame) == ("original", "working")

    ow, oh = 4000, 5000                       # original
    ws = 0.5
    ww, wh = int(ow * ws), int(oh * ws)        # working
    # abs=0.01 is exactly predict_cf's documented 2-dp rounding (60.625 -> 60.62) and nothing more;
    # the underlying identity k*sqrt(MP_orig)*ws == k*sqrt(MP_work) is exact.
    assert sq.predict_cf(ww, wh) == pytest.approx(sq.predict_cf(ow, oh) * ws, abs=0.01)
    assert sq.k * math.sqrt(ww * wh / 1e6) == pytest.approx(
        sq.k * math.sqrt(ow * oh / 1e6) * ws, rel=1e-12)          # unrounded: exact
    # the linear form cannot be moved this way at all -- the gap is what the intercept costs
    lin_via, lin_only = lin.predict_cf(ow, oh) * ws, lin.predict_cf(ww, wh)
    assert abs(lin_only / lin_via - 1) > 0.15

    db = ProjectDB.open_or_create(tmp_path / "p.sqlite")
    db.upsert_specimen(SpecimenRecord(
        image_name="a.jpg", image_stem="a", original_path="/o/a.jpg", working_path="/w/a.jpg",
        width=ww, height=wh, original_width=ow, original_height=oh, work_scale=ws))
    project = SimpleNamespace(db=db)

    import leafmachine3.modules.mp_conversion_factor as mod

    stage = mod.MPConversionFactor(None)
    stage.build_model = lambda device: sq                  # force the sqrt form
    assert stage.collect_items(project)[0].payload == (ww, wh)      # WORKING dims
    stage.build_model = lambda device: lin
    assert stage.collect_items(project)[0].payload == (ow, oh)      # ORIGINAL dims
