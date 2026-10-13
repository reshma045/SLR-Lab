"""ONNX export, PyTorch/ONNX parity check and models/meta.json."""
from __future__ import annotations

import json

import numpy as np
import torch

from src.common.config import (FEATURE_DIM, IDLE_INDEX, MODEL_LEN, MODELS, NATIVE_FPS, WINDOW,
                               class_names)
from src.train.model import JSLTransformer


def export_onnx(model: JSLTransformer, path=None) -> float:
    """Export to ONNX (opset 14, dynamic batch) and return max |torch - onnx| on random input."""
    import onnxruntime as ort

    path = path or MODELS / "jsl_model.onnx"
    model.eval()
    dummy = torch.randn(2, MODEL_LEN, FEATURE_DIM)
    kwargs = dict(input_names=["input"], output_names=["logits"], opset_version=14,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:                                   # older torch without the dynamo flag
        torch.onnx.export(model, (dummy,), str(path), **kwargs)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    diff = 0.0
    for b in (1, 3, 8):
        x = np.random.default_rng(b).normal(0, 0.1, (b, MODEL_LEN, FEATURE_DIM)).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        out = sess.run(["logits"], {"input": x})[0]
        diff = max(diff, float(np.abs(ref - out).max()))
    return diff


def write_meta(report: dict) -> None:
    """Save models/meta.json describing classes, temporal setup and feature layout."""
    names = class_names()
    meta = {
        "classes": names, "idle_index": IDLE_INDEX, "window": WINDOW, "model_len": MODEL_LEN,
        "native_fps": NATIVE_FPS,
        "input": {"name": "input", "shape": ["B", MODEL_LEN, FEATURE_DIM], "dtype": "float32"},
        "output": {"name": "logits", "shape": ["B", len(names)]},
        "feature_layout": {
            "dim": FEATURE_DIM,
            "slots": {"left": [0, 63], "right": [63, 126]},
            "per_hand": "21 MediaPipe keypoints x (x, y, z), each minus the wrist keypoint",
            "missing_hand": "zeros", "mirrored_input": False,
            "temporal": f"last {WINDOW} frames (front zero-padded) linearly resampled to {MODEL_LEN}",
        },
        "val_metrics_optimistic": {k: report[k] for k in ("val_top1", "val_top3")},
    }
    with open(MODELS / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
