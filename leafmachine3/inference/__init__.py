"""LeafMachine3 inference package — backend factory + mock/real implementations.

Stages obtain models exclusively through the factory loaders re-exported here, which
dispatch to the deterministic mock backends when ``compute.mock`` is set and to the real
Ultralytics / onnxruntime backends otherwise.
"""
from leafmachine3.inference.factory import (
    load_detector,
    load_ruler_ensemble,
    load_segmenter,
)

__all__ = ["load_detector", "load_segmenter", "load_ruler_ensemble"]
