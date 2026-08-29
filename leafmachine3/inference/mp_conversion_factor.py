"""Megapixel -> pixel/cm conversion-factor predictor (MPConversionFactor stage backend).

A one-feature linear model ``cf_px_per_cm = slope * megapixels + intercept`` where
``megapixels = original_width * original_height / 1e6``. The fitted coefficients live in the
artifact ``models/mp_conversion_factor/model.json`` (produced by ``fit_mp_cf.py`` from the shipped
``fit_data.csv``); this loader reads them, falling back to the baked-in fit if the file is
unreadable so the stage never hard-fails. Dependency-light: json + math only.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("leafmachine3.inference.mp_conversion_factor")

# Baked-in OLS fit (models/mp_conversion_factor/fit_mp_cf.py on fit_data.csv: N=708, R^2=0.91,
# MP range 14.6-36.2). model.json is the canonical artifact; these are the fallback if it is missing.
_FALLBACK_SLOPE = 2.6744940855608585
_FALLBACK_INTERCEPT = 67.51862180698473


#: Square-root refit on the SAME 708 sheets: ``cf = k * sqrt(MP)``. Fit through the origin because
#: a zero-pixel image has a zero conversion factor; the free-exponent fit on this data lands at
#: 0.4904, and the free-intercept fit lands at 0.637 px/cm -- i.e. the data asks for this form.
#: See leafmachine3/modules/experiments/MP_range for the derivation and the evaluation.
_FALLBACK_SQRT_K = 27.1121


@dataclass
class MpConversionFactorModel:
    """Resolution->CF predictor. ``predict_cf`` returns px/cm rounded to 2 dp, or ``None`` for a
    degenerate (non-positive) image size.

    Two forms, selected by ``form``:

    ``linear``  ``cf = slope*MP + intercept`` -- the original fit. Accurate inside its 14.6-36.2 MP
        training band and wrong by up to 193% outside it, because the intercept survives as MP -> 0
        where the true CF goes to zero.
    ``sqrt``    ``cf = k*sqrt(MP)`` -- the physically correct form. A sheet of fixed physical size
        imaged at 4x the pixel count has 2x the px/cm, so CF goes as the square root of MP.

    The forms are NOT interchangeable about which frame they may be evaluated in. ``sqrt`` is
    homogeneous in the linear resize factor, so ``k*sqrt(MP_orig) * work_scale == k*sqrt(MP_work)``
    exactly -- the prediction can be made straight from the working image. ``linear`` cannot: scaling
    ``a*MP + b`` by work_scale scales ``b`` too, and evaluating at ``MP_work`` does not.
    """
    slope: float = _FALLBACK_SLOPE
    intercept: float = _FALLBACK_INTERCEPT
    source: str = "fallback"
    form: str = "linear"
    k: float = _FALLBACK_SQRT_K

    def megapixels(self, width, height) -> Optional[float]:
        """Image megapixels ``width*height/1e6`` rounded to 4 dp, or ``None`` for a degenerate size."""
        try:
            w, h = float(width), float(height)
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        return round(w * h / 1e6, 4)

    def predict_cf(self, width, height) -> Optional[float]:
        """Predicted px/cm (2 dp) from the 4dp megapixels, so it reproduces from the stored MP."""
        mp = self.megapixels(width, height)
        if mp is None:
            return None
        if self.form == "sqrt":
            return round(self.k * math.sqrt(mp), 2)
        return round(self.slope * mp + self.intercept, 2)

    def formula_symbolic(self) -> str:
        """The model's equation with its CONSTANTS but MP left as a symbol, e.g.
        ``cf = 27.1121 x sqrt(MP)``. Distinct from :meth:`formula_text`, which substitutes one
        image's megapixels -- this is the model, that is the evaluation."""
        if self.form == "sqrt":
            return f"cf = {self.k:.4f} x sqrt(MP)"
        return f"cf = {self.slope:.4f} x MP + {self.intercept:.2f}"

    def formula_text(self, mp) -> Optional[str]:
        """The prediction written out with this image's numbers in it, e.g.
        ``cf = 27.1121 x sqrt(6.8352 MP) = 70.88 px/cm``.

        Rendered here rather than at the display site so the string can never describe a different
        model than the one that produced the number -- it is built from the same coefficients.
        """
        try:
            mp = float(mp)
        except (TypeError, ValueError):
            return None
        if mp <= 0:
            return None
        if self.form == "sqrt":
            return (f"cf = {self.k:.4f} x sqrt({mp:.4f} MP) = "
                    f"{round(self.k * math.sqrt(mp), 2):.2f} px/cm")
        return (f"cf = {self.slope:.4f} x {mp:.4f} MP + {self.intercept:.2f} = "
                f"{round(self.slope * mp + self.intercept, 2):.2f} px/cm")

    @property
    def frame(self) -> str:
        """Which image frame this form's prediction is expressed in.

        ``sqrt`` is evaluated on the WORKING dims and already lands in the working frame, so the
        lattice must NOT rescale it. ``linear`` is evaluated on the ORIGINAL dims and still needs
        the ``* work_scale`` step. Publishing this as a property keeps the two call sites -- the
        stage that computes it and the lattice that consumes it -- from disagreeing.
        """
        return "working" if self.form == "sqrt" else "original"


def load_model(path) -> MpConversionFactorModel:
    """Load the fitted coefficients from ``model.json``; fall back to the baked fit on any error."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return MpConversionFactorModel(
            slope=float(d.get("slope", _FALLBACK_SLOPE)),
            intercept=float(d.get("intercept", _FALLBACK_INTERCEPT)),
            source=os.path.basename(str(path)),
            form=str(d.get("model", "linear")).lower(),
            k=float(d.get("k", _FALLBACK_SQRT_K)),
        )
    except Exception as exc:  # noqa: BLE001 - never crash the stage on a missing/garbled artifact
        log.warning("MP conversion-factor model.json unreadable (%s); using baked fallback fit", exc)
        return MpConversionFactorModel()
