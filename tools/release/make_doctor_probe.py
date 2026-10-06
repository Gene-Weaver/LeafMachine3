#!/usr/bin/env python3
"""Regenerate the tiny ONNX model `lm3 doctor` runs on the accelerator (leafmachine3/doctor_probe.py).

A single 3x3 Conv (3 -> 8 channels) + ReLU on a 1x3x32x32 input. A Conv, not an Add, on purpose: on
CUDA it is what loads cuDNN's convolution engines at RUN time, which is where a missing or mismatched
cuDNN sublibrary actually fails (the provider can bind and still die on the first conv). Weights are
deterministic so the CPU reference is reproducible. IR 8 / opset 17 so every onnxruntime LM3 has
pinned can load it.

Needs the `onnx` package (development group `full`):

    uv run --group full python tools/release/make_doctor_probe.py   # prints the base64 to paste
"""
import base64

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

w = (np.arange(8 * 3 * 3 * 3, dtype=np.float32).reshape(8, 3, 3, 3) % 7 - 3) / 10.0
b = np.linspace(-0.2, 0.2, 8, dtype=np.float32)
graph = helper.make_graph(
    [helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[1, 1, 1, 1]),
     helper.make_node("Relu", ["c"], ["y"])],
    "lm3_doctor_probe",
    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32])],
    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 32, 32])],
    initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="lm3-doctor")
model.ir_version = 8
onnx.checker.check_model(model)
data = model.SerializeToString()
print(f"# {len(data)} bytes")
print(base64.b64encode(data).decode())
