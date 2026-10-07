"""LeafMachine3 inference package — backend factory + mock/real implementations.

Stages obtain models exclusively through the factory loaders re-exported here, which
dispatch to the deterministic mock backends when ``compute.mock`` is set and to the real
Ultralytics / onnxruntime backends otherwise.
"""
from leafmachine3.inference.factory import (
    load_detector,
    load_landmark_pose,
    load_mp_conversion_factor,
    load_ruler_ensemble,
    load_segmenter,
    load_specimen_segmenter,
)

__all__ = [
    "load_detector", "load_segmenter", "load_ruler_ensemble", "load_landmark_pose",
    "load_specimen_segmenter", "load_mp_conversion_factor",
]
