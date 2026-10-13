"""Augmentation shapes and invariants."""
import numpy as np
import pytest

from src.common.config import DATA, FEATURE_DIM, IDLE_INDEX, MODEL_LEN, WINDOW
from src.train import augment as A


def test_mirror_twice_is_identity(random_seq):
    assert np.array_equal(A.mirror(A.mirror(random_seq)), random_seq)


def test_mirror_swaps_slots_and_negates_x(random_seq):
    m = A.mirror(random_seq)
    assert np.allclose(m[:, 63], -random_seq[:, 0])        # right-slot x <- -left-slot x
    assert np.allclose(m[:, 64], random_seq[:, 1])         # y unchanged
    assert np.allclose(m[:, 2], random_seq[:, 65])         # z unchanged


def test_shapes(random_seq, rng):
    assert A.speed_warp(random_seq, 2.0).shape == (40, FEATURE_DIM)
    assert A.speed_warp(random_seq, 0.5).shape == (160, FEATURE_DIM)
    for fn in (lambda s: A.add_noise(s, rng), lambda s: A.scale(s, 1.1),
               lambda s: A.rotate(s, 7), lambda s: A.hand_dropout(s, rng)):
        out = fn(random_seq)
        assert out.shape == random_seq.shape and out.dtype == np.float32


def test_noise_and_rotation_keep_missing_hands_zero(random_seq, rng):
    for out in (A.add_noise(random_seq, rng), A.rotate(random_seq, 10), A.scale(random_seq, 0.9)):
        assert not out[:10, :63].any() and not out[70:, 63:].any()


def test_rotation_preserves_lengths(random_seq):
    r = A.rotate(random_seq, 9.0)
    n0 = np.hypot(random_seq[:, A.X_IDX], random_seq[:, A.Y_IDX])
    n1 = np.hypot(r[:, A.X_IDX], r[:, A.Y_IDX])
    assert np.allclose(n0, n1, atol=1e-6)


def test_place_in_window_and_model_input(random_seq, rng):
    pad = np.zeros((WINDOW, FEATURE_DIM), np.float32)
    w = A.place_in_window(random_seq, pad, rng)
    assert w.shape == (WINDOW, FEATURE_DIM)
    assert A.to_model_input(w).shape == (MODEL_LEN, FEATURE_DIM)


@pytest.mark.skipif(not (DATA / "landmarks_full" / "00.npy").exists(), reason="landmarks not extracted")
def test_sampler_epoch(rng):
    from src.train.dataset import WindowSampler, load_signs
    x, y = WindowSampler(load_signs(), per_class=2).epoch(rng)
    assert x.shape == (len(y), MODEL_LEN, FEATURE_DIM) and np.isfinite(x).all()
    assert set(np.unique(y)) == set(range(IDLE_INDEX + 1))
