"""Model forward pass and ONNX I/O contract."""
import numpy as np
import pytest
import torch

from src.common.config import FEATURE_DIM, MODEL_LEN, MODELS, NUM_CLASSES
from src.train.model import JSLTransformer

ONNX = MODELS / "jsl_model.onnx"


def test_forward_shape():
    m = JSLTransformer().eval()
    with torch.no_grad():
        assert m(torch.zeros(4, MODEL_LEN, FEATURE_DIM)).shape == (4, NUM_CLASSES)


@pytest.mark.skipif(not ONNX.exists(), reason="run python run_training.py first")
def test_onnx_io_shape():
    import onnxruntime as ort
    s = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    (inp,), (out,) = s.get_inputs(), s.get_outputs()
    assert inp.name == "input" and inp.shape[1:] == [MODEL_LEN, FEATURE_DIM]
    assert out.name == "logits"
    for b in (1, 5):
        y = s.run(["logits"], {"input": np.zeros((b, MODEL_LEN, FEATURE_DIM), np.float32)})[0]
        assert y.shape == (b, NUM_CLASSES) and np.isfinite(y).all()


@pytest.mark.skipif(not (MODELS / "best.pt").exists() or not ONNX.exists(), reason="no trained model")
def test_onnx_matches_torch():
    import onnxruntime as ort
    m = JSLTransformer().eval()
    m.load_state_dict(torch.load(MODELS / "best.pt", map_location="cpu")["state_dict"])
    x = np.random.default_rng(1).normal(0, 0.1, (3, MODEL_LEN, FEATURE_DIM)).astype(np.float32)
    s = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy()
    assert np.abs(ref - s.run(["logits"], {"input": x})[0]).max() < 1e-4
