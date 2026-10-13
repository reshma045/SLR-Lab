"""On-the-fly augmentations for (T, 136) feature sequences (layout v2, see src/common/features.py).

Every function is pure numpy and takes an explicit RNG so validation can use held-out seeds.
Shapes are in palm-length units, positions in shoulder-width units.
"""
from __future__ import annotations

import numpy as np

from src.common.config import (FEATURE_DIM, MODEL_LEN, NOSE, POS_L, POS_R, PRESENT, SHAPE_L,
                               SHAPE_R, WINDOW)
from src.common.features import resample

SHAPE_X = np.r_[np.arange(SHAPE_L.start, SHAPE_R.stop, 3)]       # x of every hand keypoint
SHAPE_Y = SHAPE_X + 1
POS_X = np.array([POS_L.start, POS_R.start, NOSE.start])          # x of wrists and nose
POS_Y = POS_X + 1
SLOT = [(SHAPE_L, POS_L, PRESENT.start), (SHAPE_R, POS_R, PRESENT.start + 1)]


def present_mask(seq: np.ndarray) -> np.ndarray:
    """(T, 2) bool: is the left / right hand present in each frame."""
    return seq[:, PRESENT] > 0.5


def pose_mask(seq: np.ndarray) -> np.ndarray:
    """(T,) bool: were the shoulders found in each frame."""
    return np.abs(seq[:, NOSE]).sum(1) > 0


def mirror(seq: np.ndarray) -> np.ndarray:
    """Swap left/right blocks and negate x (a different-handed signer). mirror(mirror(s)) == s."""
    out = seq.copy()
    for a, b in ((SHAPE_L, SHAPE_R), (POS_L, POS_R)):
        out[:, a], out[:, b] = seq[:, b], seq[:, a]
    out[:, PRESENT] = seq[:, PRESENT][:, ::-1]
    out[:, SHAPE_X] *= -1
    out[:, POS_X] *= -1
    return out


def speed_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Play the sequence `factor` times faster (factor>1 -> shorter)."""
    return resample(seq, max(2, int(round(len(seq) / factor))))


def _restore_missing(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Keep absent hands / missing pose exactly zero after a numeric augmentation."""
    hm, pm = present_mask(src), pose_mask(src)
    for k, (shape, pos, _) in enumerate(SLOT):
        out[~hm[:, k], shape] = 0
        out[~(hm[:, k] & pm), pos] = 0
    out[~pm, NOSE] = 0
    out[:, PRESENT] = src[:, PRESENT]
    return out.astype(np.float32)


def add_noise(seq: np.ndarray, rng: np.random.Generator, shape_sigma: float = 0.03,
              pos_sigma: float = 0.015) -> np.ndarray:
    """Gaussian jitter on present hands / found pose only."""
    out = seq.copy()
    out[:, SHAPE_L.start:SHAPE_R.stop] += rng.normal(0, shape_sigma, (len(seq), 126))
    out[:, POS_L.start:NOSE.stop] += rng.normal(0, pos_sigma, (len(seq), 8))
    return _restore_missing(seq, out)


def scale(seq: np.ndarray, body: float, hand: float = 1.0) -> np.ndarray:
    """Body-proportion jitter on positions, small residual jitter on hand shapes."""
    out = seq.copy()
    out[:, POS_X] *= body
    out[:, POS_Y] *= body
    out[:, SHAPE_L.start:SHAPE_R.stop] *= hand
    return _restore_missing(seq, out)


def shift(seq: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Translate wrist and nose positions (different shoulder-centre estimate / posture)."""
    out = seq.copy()
    out[:, POS_X] += dx
    out[:, POS_Y] += dy
    return _restore_missing(seq, out)


def rotate(seq: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate (x, y) of hand shapes and body positions by `degrees` (camera roll)."""
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    out = seq.copy()
    for xi, yi in ((SHAPE_X, SHAPE_Y), (POS_X, POS_Y)):
        x, y = seq[:, xi], seq[:, yi]
        out[:, xi] = c * x - s * y
        out[:, yi] = s * x + c * y
    return _restore_missing(seq, out)


def hand_dropout(seq: np.ndarray, rng: np.random.Generator, max_frames: int = 4) -> np.ndarray:
    """Remove a hand in a few random frames to mimic MediaPipe misses."""
    out = seq.copy()
    for shape, pos, pres in SLOT:
        n = int(rng.integers(0, max_frames + 1))
        if n:
            idx = rng.choice(len(seq), size=min(n, len(seq)), replace=False)
            out[idx, shape] = 0
            out[idx, pos] = 0
            out[idx, pres] = 0
    return out


def pose_dropout(seq: np.ndarray) -> np.ndarray:
    """Drop all body-relative information (shoulders not visible)."""
    out = seq.copy()
    out[:, POS_L.start:NOSE.stop] = 0
    return out


def place_in_window(sign: np.ndarray, pad: np.ndarray, rng: np.random.Generator,
                    window: int = WINDOW, min_cover: float = 0.8) -> np.ndarray:
    """Put `sign` at a random offset inside a `window`-frame buffer pre-filled with `pad`."""
    out = pad[:window].copy()
    n = len(sign)
    need = int(np.ceil(min_cover * min(n, window)))
    start = int(rng.integers(need - n, window - need + 1))
    s0, s1 = max(0, start), min(window, start + n)
    out[s0:s1] = sign[s0 - start:s1 - start]
    return out


def to_model_input(window_seq: np.ndarray) -> np.ndarray:
    """Resample a (WINDOW, D) buffer to (MODEL_LEN, D) - identical to the live path at 30 fps."""
    assert window_seq.shape[1] == FEATURE_DIM
    return resample(window_seq, MODEL_LEN)
