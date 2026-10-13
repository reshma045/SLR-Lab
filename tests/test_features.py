"""Raw landmarks -> feature layout v3: scale-free hands, body-relative positions, lips, face, motion."""
import numpy as np

from src.common.config import (ELBOWS, EYES, FEATURE_DIM, LIPS, MODEL_IN, MODEL_LEN, MOUTH, POS_L, POS_R,
                               PRESENT, ROLL, SHAPE_L, SHAPE_R, TIPS)
from src.common.features import (MIRROR_PERM, MIRROR_SIGN, add_motion, frame_features, resample,
                                 sequence_features)
from src.common.landmarks import (FACE_POS, R_FACE, R_FACE_P, R_HAND_L, R_HAND_P, R_HAND_R, R_POSE,
                                  R_POSE_P, RAW_DIM, LandmarkExtractor, assign_slots, resample_raw)


def _raw(hand_scale=0.1, left=True, right=True, face=True, pose=True, mouth_open=0.02):
    raw = np.zeros(RAW_DIM, np.float32)
    if pose:
        p = np.zeros((33, 4), np.float32)
        p[:, 3] = 1
        p[11, :2], p[12, :2] = (0.8, 0.5), (0.5, 0.5)              # signer's left shoulder on image right
        p[13, :2], p[14, :2] = (0.85, 0.7), (0.45, 0.7)
        p[0, :2] = (0.65, 0.3)
        raw[R_POSE] = p.ravel()
        raw[R_POSE_P] = 1
    for k, (sl, wx) in enumerate(((R_HAND_L, 0.8), (R_HAND_R, 0.5))):
        if (left, right)[k]:
            pts = np.array([(wx + i * hand_scale / 10, 0.6 + 2 * i * hand_scale / 10, 0) for i in range(21)], np.float32)
            raw[sl] = pts.ravel()
            raw[R_HAND_P.start + k] = 1
    if face:
        f = np.zeros((len(FACE_POS), 3), np.float32)
        def put(i, x, y): f[FACE_POS[i], :2] = (x, y)               # noqa: E306
        for i in FACE_POS:
            put(i, 0.65, 0.3)
        put(234, 0.55, 0.3); put(454, 0.75, 0.3)                    # face width 0.2
        put(61, 0.62, 0.36); put(291, 0.68, 0.36)                   # mouth width 0.06
        put(13, 0.65, 0.36 - mouth_open / 2); put(14, 0.65, 0.36 + mouth_open / 2)
        put(33, 0.6, 0.28); put(263, 0.7, 0.28)
        raw[R_FACE] = f.ravel()
        raw[R_FACE_P] = 1
    return raw


def test_shapes_are_scale_free_and_positions_body_relative():
    near, far = frame_features(_raw(hand_scale=0.3)), frame_features(_raw(hand_scale=0.1))
    assert near.shape == (FEATURE_DIM,) and near.dtype == np.float32
    assert np.allclose(near[SHAPE_L], far[SHAPE_L], atol=1e-5)          # close or far: same shape
    rel = near[SHAPE_L].reshape(21, 3)
    assert np.allclose(rel[0], 0) and np.isclose(np.linalg.norm(rel[9, :2]), 1.0)
    centre, sw = np.array([0.65, 0.5]), 0.3
    assert np.allclose(near[POS_L][:2], (np.array([0.8, 0.6]) - centre) / sw)
    assert np.allclose(near[ELBOWS][:2], (np.array([0.85, 0.7]) - centre) / sw)
    assert near[PRESENT].tolist() == [1, 1, 1, 1]


def test_face_features():
    closed, opened = frame_features(_raw(mouth_open=0.0)), frame_features(_raw(mouth_open=0.03))
    assert np.isclose(opened[MOUTH][0], 0.5) and np.isclose(closed[MOUTH][0], 0.0)   # opening / width
    assert np.isclose(opened[MOUTH][1], 0.3)                                          # mouth / face width
    assert np.allclose(opened[ROLL], [1, 0])                                          # level head
    assert np.abs(opened[LIPS]).max() <= 1.0 + 1e-5                                   # mouth-normalised


def test_missing_parts_are_zero():
    v = frame_features(_raw(left=False, face=False, pose=False))
    assert v[PRESENT].tolist() == [0, 1, 0, 0]
    assert not v[SHAPE_L].any() and not v[POS_R].any() and not v[LIPS].any() and not v[TIPS].any()
    assert v[SHAPE_R].any() and not v[EYES].any()


def test_mirror_is_an_exact_involution():
    assert (MIRROR_PERM[MIRROR_PERM] == np.arange(FEATURE_DIM)).all()
    assert (MIRROR_SIGN * MIRROR_SIGN[MIRROR_PERM] == 1).all()


def test_slot_assignment():
    wrists = (np.array([0.8, 0.6]), np.array([0.5, 0.6]))
    a, b = np.zeros((21, 3)), np.zeros((21, 3))
    a[0, :2], b[0, :2] = (0.51, 0.6), (0.79, 0.6)
    assert assign_slots([a, b], [1, 1], wrists) == [1, 0]                 # pose wrists win over labels
    assert assign_slots([a], [1], None) == [1]                            # raw-frame label fallback


def test_resample_raw_never_blends_missing_parts():
    r0, r1 = _raw(right=False), _raw()
    out = resample_raw([0.0, 0.1], [r0, r1], fps=30)
    assert len(out) == 4
    mid = out[1]                                                          # nearest frame, not a 1/3 blend
    assert mid[R_HAND_P.start + 1] == 0 and not mid[R_HAND_R].any()
    assert np.allclose(out[:, R_HAND_L], r0[R_HAND_L])                   # present in both -> interpolated (equal)


def test_motion_and_model_input_shapes():
    x = np.cumsum(np.ones((MODEL_LEN, FEATURE_DIM), np.float32), 0)
    m = add_motion(x)
    assert m.shape == (MODEL_LEN, MODEL_IN)
    assert np.allclose(m[:-1, FEATURE_DIM:2 * FEATURE_DIM], 1) and np.allclose(m[-1, FEATURE_DIM:], 0)
    assert np.allclose(m[:-2, 2 * FEATURE_DIM:], 2)


def test_black_frame_gives_zero_features():
    with LandmarkExtractor() as lx:
        raw, _ = lx.process(np.zeros((480, 640, 3), np.uint8))
    assert raw.shape == (RAW_DIM,)
    assert not sequence_features(raw[None]).any()


def test_resample_lengths(random_seq):
    for n in (5, 60, 80):
        out = resample(random_seq[:n], MODEL_LEN)
        assert out.shape == (MODEL_LEN, FEATURE_DIM)
        assert np.allclose(out[0], random_seq[0]) and np.allclose(out[-1], random_seq[n - 1])
