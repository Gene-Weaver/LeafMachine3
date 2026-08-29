"""Bilateral-symmetry experiment: how symmetric is a leaf lamina, and does that flag clean masks?

Not a pipeline stage (yet) -- an offline experiment that reads an already-computed run and writes
``modules/experiments/bilateral_symmetry.html``. Run it with::

    .venv_LM3/bin/python -m leafmachine3.modules.experiments.bilateral_symmetry.run \
        --run examples_out/testing_up_to_specimen_seg

Layout:
    geometry.py  oriented-mask frame: silhouette + landmarks, exactly co-registered
    axes.py      chord and midvein axes; the curvilinear (s, u) frame; strip profiles
    metrics.py   scalar symmetry metrics (Shi 2018, Wang 2018, and signed/weighted variants)
    shape.py     whole-shape measures on the straightened halves (Dice/IoU, Hausdorff, moments, lobes)
    quality.py   mask-quality diagnostics + the composite "archetypal leaf" score
    figures.py   matplotlib figure builders (base64 data URIs)
    report.py    the HTML report (style matched to setup/timing.py)
    run.py       the driver
"""
