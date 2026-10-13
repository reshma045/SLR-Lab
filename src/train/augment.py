"""On-the-fly augmentations for (T, FEATURE_DIM) feature sequences (layout v3).

Pure numpy with explicit RNGs (validation uses held-out seeds). Geometry comes from the tables in
src/common/features.py so every block (hands, body, lips, face) is transformed consistently.
Several ideas follow the 1st-place Kaggle ISLR solution: speed resampling, mirror, affine
(scale / shear / rotation / shift), temporal masking and random part dropout.
"""
from __future__ import annotations

import numpy as np

from src.common.config import FEATURE_DIM, MODEL_LEN, PRESENT
from src.common.features import (BLOCKS, BODY_X, BODY_Y, MIRROR_PERM, MIRROR_SIGN, ROT_X, ROT_Y,
                                 add_motion, resample)

HAND_PARTS = [(0, [b for b, need in BLOCKS if 0 in need]), (1, [b for b, need in BLOCKS if 1 in need])]
FACE_PART = [b for b, need in BLOCKS if 2 in need]
BODY_PART = [b for b, need in BLOCKS if 3 in need]


def present(seq: np.ndarray) -> np.ndarray:
    """(T, 4) bool presence of left hand, right hand, face, body."""
    return seq[:, PRESENT] > 0.5


def restore_missing(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Keep every block that was absent in `src` exactly zero after a numeric augmentation."""
    pres = present(src)
    for idx, need in BLOCKS:
        ok = pres[:, need].all(1)
        out[np.ix_(~ok, idx)] = 0
    out[:, PRESENT] = src[:, PRESENT]
    return out.astype(np.float32)


def mirror(seq: np.ndarray) -> np.ndarray:
    """Left-right mirror (a left-handed signer). mirror(mirror(s)) == s."""
    return (seq[:, MIRROR_PERM] * MIRROR_SIGN).astype(np.float32)


def speed_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Play the sequence `factor` times faster (factor > 1 -> shorter)."""
    return resample(seq, max(2, int(round(len(seq) / factor))))


def affine(seq: np.ndarray, degrees: float = 0.0, shear: float = 0.0, body_scale: float = 1.0,
           shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Camera roll + shear on every (x, y) pair; body-proportion scale and shift on body positions."""
    out = seq.copy()
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    x, y = seq[:, ROT_X], seq[:, ROT_Y]
    x2 = x + shear * y
    out[:, ROT_X] = c * x2 - s * y
    out[:, ROT_Y] = s * x2 + c * y
    out[:, BODY_X] = out[:, BODY_X] * body_scale + shift[0]
    out[:, BODY_Y] = out[:, BODY_Y] * body_scale + shift[1]
    return restore_missing(seq, out)


def add_noise(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.02) -> np.ndarray:
    """Gaussian jitter on visible parts only (presence flags untouched)."""
    out = seq + rng.normal(0, sigma, seq.shape).astype(np.float32)
    return restore_missing(seq, out)


def drop_part(seq: np.ndarray, frames: np.ndarray, blocks: list[np.ndarray], flag: int) -> np.ndarray:
    """Remove one body part (hand / face / body) in the given frames, as if not detected."""
    out = seq.copy()
    for idx in blocks:
        out[np.ix_(frames, idx)] = 0
    out[frames, PRESENT.start + flag] = 0
    return out


def hand_dropout(seq: np.ndarray, rng: np.random.Generator, max_frames: int = 6) -> np.ndarray:
    """Remove a hand in a few random frames to mimic MediaPipe misses."""
    out = seq
    for flag, blocks in HAND_PARTS:
        n = int(rng.integers(0, max_frames + 1))
        if n:
            out = drop_part(out, rng.choice(len(seq), size=min(n, len(seq)), replace=False), blocks, flag)
    return out


def temporal_mask(seq: np.ndarray, rng: np.random.Generator, frac: tuple[float, float] = (0.05, 0.2)) -> np.ndarray:
    """Hold the previous frame over a short random span (dropped/frozen camera frames)."""
    out = seq.copy()
    n = int(len(seq) * rng.uniform(*frac))
    if n > 1:
        a = int(rng.integers(1, max(2, len(seq) - n)))
        out[a:a + n] = out[a - 1]
    return out


def place_in_window(sign: np.ndarray, pad: np.ndarray, rng: np.random.Generator, window: int,
                    min_cover: float = 0.8) -> np.ndarray:
    """Put `sign` at a random offset inside a `window`-frame buffer pre-filled with `pad`."""
    out = pad[:window].copy()
    n = len(sign)
    need = int(np.ceil(min_cover * min(n, window)))
    start = int(rng.integers(need - n, window - need + 1))
    s0, s1 = max(0, start), min(window, start + n)
    out[s0:s1] = sign[s0 - start:s1 - start]
    return out


def to_model_input(window_seq: np.ndarray) -> np.ndarray:
    """(WINDOW, F) buffer -> (MODEL_LEN, 3F): resample exactly like the live path, add motion."""
    assert window_seq.shape[1] == FEATURE_DIM
    return add_motion(resample(window_seq, MODEL_LEN))
