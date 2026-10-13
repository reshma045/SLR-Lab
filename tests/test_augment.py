"""Augmentation invariants for layout v3."""
import numpy as np
import pytest

from src.common.config import (DATA, FEATURE_DIM, IDLE_INDEX, LIPS, MODEL_IN, MODEL_LEN, NOSE, POS_L, POS_R,
                               PRESENT, ROLL, SHAPE_L, SHAPE_R, TIPS, WINDOW)
from src.common.features import BLOCKS
from src.train import augment as A


def _missing_zero(seq):
    pres = seq[:, PRESENT] > 0.5
    return all(not seq[np.ix_(~pres[:, need].all(1), idx)].any() for idx, need in BLOCKS)


def test_mirror_twice_is_identity(random_seq):
    assert np.allclose(A.mirror(A.mirror(random_seq)), random_seq)


def test_mirror_swaps_hands_and_flips_x(random_seq):
    m = A.mirror(random_seq)
    assert np.allclose(m[:, SHAPE_R.start], -random_seq[:, SHAPE_L.start])
    assert np.allclose(m[:, SHAPE_R.start + 1], random_seq[:, SHAPE_L.start + 1])
    assert np.allclose(m[:, POS_L.start], -random_seq[:, POS_R.start])
    assert np.allclose(m[:, NOSE.start], -random_seq[:, NOSE.start])
    assert np.allclose(m[:, LIPS.start], -random_seq[:, LIPS.start + 20])      # contour point 0 <-> 10, x flipped
    assert np.allclose(m[:, ROLL.start + 1], -random_seq[:, ROLL.start + 1])
    assert np.allclose(m[:, TIPS.start], -random_seq[:, TIPS.start + 4])
    assert np.array_equal(m[:, PRESENT], random_seq[:, PRESENT][:, [1, 0, 2, 3]])
    assert _missing_zero(m)


def test_augmentations_keep_shapes_and_missing_parts(random_seq, rng):
    outs = [A.affine(random_seq, 12, 0.1, 1.1, (0.1, -0.1)), A.add_noise(random_seq, rng),
            A.hand_dropout(random_seq, rng), A.temporal_mask(random_seq, rng),
            A.drop_part(random_seq, np.arange(80), A.FACE_PART, 2)]
    for out in outs:
        assert out.shape == random_seq.shape and out.dtype == np.float32 and _missing_zero(out)
    assert np.array_equal(A.affine(random_seq, 5)[:, PRESENT], random_seq[:, PRESENT])
    assert A.speed_warp(random_seq, 2.0).shape == (40, FEATURE_DIM)


def test_rotation_preserves_hand_lengths(random_seq):
    r = A.affine(random_seq, 9.0)
    x, y = A.ROT_X[:42], A.ROT_Y[:42]                                      # the 42 hand keypoints
    assert np.allclose(np.hypot(random_seq[:, x], random_seq[:, y]), np.hypot(r[:, x], r[:, y]), atol=1e-5)


def test_model_input(random_seq, rng):
    w = A.place_in_window(random_seq, np.zeros((WINDOW, FEATURE_DIM), np.float32), rng, WINDOW)
    assert A.to_model_input(w).shape == (MODEL_LEN, MODEL_IN)


@pytest.mark.skipif(not (DATA / "landmarks_raw" / "00.npy").exists(), reason="landmarks not extracted")
def test_sampler_epoch(rng):
    from src.train.dataset import WindowSampler, load_signs
    train, _ = load_signs(None)
    x, y = WindowSampler(train, per_class=2).epoch(rng)
    assert x.shape == (len(y), MODEL_LEN, MODEL_IN) and np.isfinite(x).all()
    assert set(np.unique(y)) == set(range(IDLE_INDEX + 1))
