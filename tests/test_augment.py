"""Augmentation shapes and invariants for layout v2."""
import numpy as np
import pytest

from src.common.config import (DATA, FEATURE_DIM, IDLE_INDEX, MODEL_LEN, NOSE, POS_L, POS_R, PRESENT,
                               SHAPE_L, SHAPE_R, WINDOW)
from src.train import augment as A


def test_mirror_twice_is_identity(random_seq):
    assert np.array_equal(A.mirror(A.mirror(random_seq)), random_seq)


def test_mirror_swaps_blocks_and_negates_x(random_seq):
    m = A.mirror(random_seq)
    assert np.allclose(m[:, SHAPE_R.start], -random_seq[:, SHAPE_L.start])     # x of keypoint 0
    assert np.allclose(m[:, SHAPE_R.start + 1], random_seq[:, SHAPE_L.start + 1])  # y unchanged
    assert np.allclose(m[:, POS_L.start], -random_seq[:, POS_R.start])
    assert np.allclose(m[:, POS_L.start + 2], random_seq[:, POS_R.start + 2])  # palm size unchanged
    assert np.allclose(m[:, NOSE.start], -random_seq[:, NOSE.start])
    assert np.array_equal(m[:, PRESENT], random_seq[:, PRESENT][:, ::-1])


def test_shapes(random_seq, rng):
    assert A.speed_warp(random_seq, 2.0).shape == (40, FEATURE_DIM)
    assert A.speed_warp(random_seq, 0.5).shape == (160, FEATURE_DIM)
    for fn in (lambda s: A.add_noise(s, rng), lambda s: A.scale(s, 1.1, 1.02), lambda s: A.shift(s, .1, -.1),
               lambda s: A.rotate(s, 7), lambda s: A.hand_dropout(s, rng), A.pose_dropout):
        out = fn(random_seq)
        assert out.shape == random_seq.shape and out.dtype == np.float32


def test_missing_hands_stay_zero(random_seq, rng):
    for out in (A.add_noise(random_seq, rng), A.rotate(random_seq, 10), A.scale(random_seq, 0.9),
                A.shift(random_seq, 0.2, 0.2)):
        assert not out[:10, SHAPE_L].any() and not out[:10, POS_L].any()
        assert not out[70:, SHAPE_R].any() and not out[70:, POS_R].any()
        assert np.array_equal(out[:, PRESENT], random_seq[:, PRESENT])


def test_rotation_preserves_hand_lengths(random_seq):
    r = A.rotate(random_seq, 9.0)
    n0 = np.hypot(random_seq[:, A.SHAPE_X], random_seq[:, A.SHAPE_Y])
    n1 = np.hypot(r[:, A.SHAPE_X], r[:, A.SHAPE_Y])
    assert np.allclose(n0, n1, atol=1e-5)


def test_hand_dropout_clears_presence(rng):
    seq = np.ones((50, FEATURE_DIM), np.float32)
    out = A.hand_dropout(seq, np.random.default_rng(3), max_frames=10)
    dropped = out[:, PRESENT.start] == 0
    assert dropped.any() and not out[dropped][:, SHAPE_L].any()


def test_place_in_window_and_model_input(random_seq, rng):
    w = A.place_in_window(random_seq, np.zeros((WINDOW, FEATURE_DIM), np.float32), rng)
    assert w.shape == (WINDOW, FEATURE_DIM)
    assert A.to_model_input(w).shape == (MODEL_LEN, FEATURE_DIM)


@pytest.mark.skipif(not (DATA / "landmarks_full" / "00.npy").exists(), reason="landmarks not extracted")
def test_sampler_epoch(rng):
    from src.train.dataset import WindowSampler, load_signs
    train, _ = load_signs(None)
    x, y = WindowSampler(train, per_class=2).epoch(rng)
    assert x.shape == (len(y), MODEL_LEN, FEATURE_DIM) and np.isfinite(x).all()
    assert set(np.unique(y)) == set(range(IDLE_INDEX + 1))
