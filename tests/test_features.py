"""Feature layout and resampling."""
from types import SimpleNamespace

import numpy as np

from src.common.config import FEATURE_DIM, MODEL_LEN, WINDOW
from src.common.features import frame_landmarks, hands_present, make_hands, resample, results_to_vector


def _fake_results(label: str, offset: float):
    pts = [SimpleNamespace(x=offset + i * 0.01, y=0.5 + i * 0.02, z=-i * 0.001) for i in range(21)]
    lm = SimpleNamespace(landmark=pts)
    info = SimpleNamespace(classification=[SimpleNamespace(label=label)])
    return lm, info


def test_layout_left_then_right_wrist_relative():
    l_lm, l_info = _fake_results("Left", 0.2)
    r_lm, r_info = _fake_results("Right", 0.7)
    res = SimpleNamespace(multi_hand_landmarks=[r_lm, l_lm], multi_handedness=[r_info, l_info])
    v = results_to_vector(res)
    assert v.shape == (FEATURE_DIM,) and v.dtype == np.float32
    assert np.allclose(v[0:3], 0) and np.allclose(v[63:66], 0)          # wrist is the origin
    assert np.isclose(v[3], 0.01) and np.isclose(v[4], 0.02)            # keypoint 1 of left slot
    assert hands_present(v) == (True, True)


def test_missing_hand_is_zero():
    lm, info = _fake_results("Right", 0.4)
    v = results_to_vector(SimpleNamespace(multi_hand_landmarks=[lm], multi_handedness=[info]))
    assert np.all(v[:63] == 0) and np.abs(v[63:]).sum() > 0
    assert hands_present(v) == (False, True)


def test_black_frame_gives_zero_vector():
    with make_hands() as hands:
        v = frame_landmarks(np.zeros((480, 640, 3), np.uint8), hands)
    assert v.shape == (FEATURE_DIM,) and not v.any()


def test_resample_lengths_and_endpoints(random_seq):
    for n in (5, 60, 80):
        out = resample(random_seq[:n], MODEL_LEN)
        assert out.shape == (MODEL_LEN, FEATURE_DIM)
        assert np.allclose(out[0], random_seq[0]) and np.allclose(out[-1], random_seq[n - 1])
    assert resample(np.zeros((WINDOW, FEATURE_DIM)), MODEL_LEN).shape == (MODEL_LEN, FEATURE_DIM)
    assert resample(random_seq[:1], 10).shape == (10, FEATURE_DIM)
