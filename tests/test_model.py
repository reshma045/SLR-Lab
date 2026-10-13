"""Model forward pass, ONNX I/O contract and parity."""
import numpy as np
import pytest
import torch

from src.common.config import MODEL_IN, MODEL_LEN, MODELS, NUM_CLASSES
from src.train.model import AWP, JSLNet

ONNX = MODELS / "jsl_model.onnx"


def _onnx_ok() -> bool:
    if not ONNX.exists():
        return False
    import onnxruntime as ort
    return ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"]).get_inputs()[0].shape[-1] == MODEL_IN


def test_forward_shapes_and_late_dropout():
    m = JSLNet().eval()
    with torch.no_grad():
        logits, emb = m(torch.zeros(4, MODEL_LEN, MODEL_IN))
    assert logits.shape == (4, NUM_CLASSES) and emb.shape == (4, 384)
    m.train()
    m.late_on = True
    assert m(torch.randn(2, MODEL_LEN, MODEL_IN))[0].shape == (2, NUM_CLASSES)


def test_awp_restores_weights():
    m = JSLNet()
    loss = m(torch.randn(3, MODEL_LEN, MODEL_IN))[0].sum()
    loss.backward()
    before = {k: v.clone() for k, v in m.state_dict().items()}
    awp = AWP(m)
    awp.perturb()
    assert any(not torch.equal(before[k], v) for k, v in m.state_dict().items())
    awp.restore()
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


@pytest.mark.skipif(not _onnx_ok(), reason="run python run_training.py first")
def test_onnx_io():
    import onnxruntime as ort
    s = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    assert [o.name for o in s.get_outputs()] == ["logits", "embedding"]
    for b in (1, 5):
        lg, em = s.run(["logits", "embedding"], {"input": np.zeros((b, MODEL_LEN, MODEL_IN), np.float32)})
        assert lg.shape == (b, NUM_CLASSES) and em.shape[0] == b and np.isfinite(lg).all()


@pytest.mark.skipif(not (_onnx_ok() and (MODELS / "best.pt").exists()), reason="no trained model")
def test_onnx_matches_torch():
    import onnxruntime as ort
    m = JSLNet().eval()
    m.load_state_dict(torch.load(MODELS / "best.pt", map_location="cpu")["state_dict"])
    x = np.random.default_rng(1).normal(0, 0.5, (3, MODEL_LEN, MODEL_IN)).astype(np.float32)
    s = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = m(torch.from_numpy(x))[0].numpy()
    assert np.abs(ref - s.run(["logits"], {"input": x})[0]).max() < 1e-3
