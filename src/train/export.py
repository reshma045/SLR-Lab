"""ONNX export, PyTorch/ONNX parity check and models/meta.json."""
from __future__ import annotations

import json

import numpy as np
import torch

from src.common.config import (FEATURE_DIM, IDLE_INDEX, MODEL_IN, MODEL_LEN, MODELS, NATIVE_FPS, WINDOW,
                               class_names)
from src.train.model import JSLNet


def export_onnx(model: JSLNet, path=None) -> float:
    """Export to ONNX (outputs `logits` and `embedding`, dynamic batch); return max |torch - onnx|."""
    import onnxruntime as ort

    path = path or MODELS / "jsl_model.onnx"
    model.eval()
    model.late_on = False
    dummy = torch.randn(2, MODEL_LEN, MODEL_IN)
    kwargs = dict(input_names=["input"], output_names=["logits", "embedding"], opset_version=17,
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}})
    try:
        torch.onnx.export(model, (dummy,), str(path), dynamo=False, **kwargs)
    except TypeError:                                   # older torch without the dynamo flag
        torch.onnx.export(model, (dummy,), str(path), **kwargs)

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    diff = 0.0
    for b in (1, 3, 8):
        x = np.random.default_rng(b).normal(0, 0.5, (b, MODEL_LEN, MODEL_IN)).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x))
        out = sess.run(["logits", "embedding"], {"input": x})
        diff = max(diff, float(np.abs(ref[0].numpy() - out[0]).max()), float(np.abs(ref[1].numpy() - out[1]).max()))
    return diff


def write_meta(report: dict) -> None:
    """Save models/meta.json describing classes, temporal setup and feature layout."""
    names = class_names()
    meta = {
        "classes": names, "idle_index": IDLE_INDEX, "window": WINDOW, "model_len": MODEL_LEN,
        "native_fps": NATIVE_FPS,
        "input": {"name": "input", "shape": ["B", MODEL_LEN, MODEL_IN], "dtype": "float32"},
        "outputs": {"logits": ["B", len(names)], "embedding": ["B", "2*dim"]},
        "feature_layout": {
            "version": 3, "per_frame_dim": FEATURE_DIM, "model_in": MODEL_IN,
            "blocks": "see src/common/config.py: hand shapes, wrist/elbow positions, nose, lips, mouth, eyes, "
                      "brows, head roll, fingertips-to-face, presence; + velocity over 1 and 2 steps",
            "raw": "src/common/landmarks.py (MediaPipe hands + pose + face mesh subset)",
            "aspect_corrected": True, "mirrored_input": False,
            "temporal": f"last {WINDOW} frames at 30 fps (time-based) linearly resampled to {MODEL_LEN}",
        },
        "metrics": {k: report.get(k) for k in ("synthetic_val_top1", "real_val_top1", "user_recordings")},
    }
    with open(MODELS / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
