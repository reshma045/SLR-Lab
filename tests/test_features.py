"""Feature layout v2: scale-free hand shapes, body-relative positions, signer-left/right slots."""
from types import SimpleNamespace

import numpy as np

from src.common.config import FEATURE_DIM, MODEL_LEN, NOSE, POS_L, POS_R, PRESENT, SHAPE_L, SHAPE_R
from src.common.features import (FeatureExtractor, assign_slots, build_vector, hands_present,
                                 pose_present, resample)


def _hand(label: str, wrist_x: float, scale: float = 0.1):
    """Fake MediaPipe hand: keypoint i at wrist + (i, 2i) * scale/10; middle MCP (9) sets palm size."""
    pts = [SimpleNamespace(x=wrist_x + i * scale / 10, y=0.6 + 2 * i * scale / 10, z=0.0) for i in range(21)]
    return SimpleNamespace(landmark=pts), SimpleNamespace(classification=[SimpleNamespace(label=label)])


def _results(*hands):
    return SimpleNamespace(multi_hand_landmarks=[h[0] for h in hands], multi_handedness=[h[1] for h in hands])


BODY = {"center": np.array([0.5, 0.4], np.float32), "width": 0.3, "nose": np.array([0.5, 0.2], np.float32),
        "wrists": (np.array([0.7, 0.6], np.float32), np.array([0.3, 0.6], np.float32))}


def test_layout_and_scale_invariance():
    near = build_vector(_results(_hand("Left", 0.3, scale=0.3)), BODY, 1.0)    # big hand (close to camera)
    far = build_vector(_results(_hand("Left", 0.3, scale=0.1)), BODY, 1.0)     # small hand
    assert near.shape == (FEATURE_DIM,) and near.dtype == np.float32
    assert np.allclose(near[SHAPE_R], far[SHAPE_R], atol=1e-5)                  # shape is scale-free
    rel = near[SHAPE_R].reshape(21, 3)
    assert np.allclose(rel[0], 0) and np.isclose(np.linalg.norm(rel[9, :2]), 1.0)   # wrist origin, palm = 1
    assert near[PRESENT].tolist() == [0.0, 1.0]
    assert np.allclose(near[POS_R][:2], (np.array([0.3, 0.6]) - BODY["center"]) / BODY["width"])
    assert np.allclose(near[NOSE], (BODY["nose"] - BODY["center"]) / BODY["width"])
    assert not near[SHAPE_L].any() and not near[POS_L].any()


def test_slots_follow_pose_wrists_not_labels():
    # both hands labelled "Left" by MediaPipe; the pose wrists decide which is which
    v = build_vector(_results(_hand("Left", 0.31), _hand("Left", 0.69)), BODY, 1.0)
    assert hands_present(v) == (True, True)
    assert np.isclose(v[POS_L][0], (0.69 - 0.5) / 0.3) and np.isclose(v[POS_R][0], (0.31 - 0.5) / 0.3)


def test_label_fallback_without_pose():
    # raw (unmirrored) frame: MediaPipe "Left" is the signer's right hand
    assert assign_slots([np.zeros((21, 3))], [1], None) == [1]
    v = build_vector(_results(_hand("Right", 0.6)), None, 1.0)
    assert hands_present(v) == (True, False) and not pose_present(v)
    assert not v[POS_L].any()                                                  # no body -> no positions


def test_black_frame_gives_zero_vector():
    with FeatureExtractor() as fx:
        v, _, _ = fx.process(np.zeros((480, 640, 3), np.uint8))
    assert v.shape == (FEATURE_DIM,) and not v.any()


def test_resample_lengths_and_endpoints(random_seq):
    for n in (5, 60, 80):
        out = resample(random_seq[:n], MODEL_LEN)
        assert out.shape == (MODEL_LEN, FEATURE_DIM)
        assert np.allclose(out[0], random_seq[0]) and np.allclose(out[-1], random_seq[n - 1])
    assert resample(random_seq[:1], 10).shape == (10, FEATURE_DIM)
