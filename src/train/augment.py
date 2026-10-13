"""On-the-fly augmentations for (T, 126) landmark sequences.

Every function is pure numpy and takes an explicit RNG so validation can use held-out seeds.
"""
from __future__ import annotations

import numpy as np

from src.common.config import HAND_DIM, MODEL_LEN, WINDOW
from src.common.features import resample

X_IDX = np.arange(0, 2 * HAND_DIM, 3)          # x coordinate of every keypoint, both hands
Y_IDX = X_IDX + 1


def present_mask(seq: np.ndarray) -> np.ndarray:
    """(T, 2) bool: is the left / right hand slot non-zero in each frame."""
    return np.stack([np.abs(seq[:, :HAND_DIM]).sum(1) > 0,
                     np.abs(seq[:, HAND_DIM:]).sum(1) > 0], axis=1)


def mirror(seq: np.ndarray) -> np.ndarray:
    """Swap hand slots and negate x (a different-handed signer). mirror(mirror(s)) == s."""
    out = np.concatenate([seq[:, HAND_DIM:], seq[:, :HAND_DIM]], axis=1).copy()
    out[:, X_IDX] *= -1
    return out


def speed_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    """Play the sequence `factor` times faster (factor>1 -> shorter)."""
    return resample(seq, max(2, int(round(len(seq) / factor))))


def _keep_zeros(fn):
    def wrapped(seq: np.ndarray, *a, **k) -> np.ndarray:
        m = present_mask(seq)
        out = fn(seq, *a, **k)
        out[:, :HAND_DIM][~m[:, 0]] = 0
        out[:, HAND_DIM:][~m[:, 1]] = 0
        return out.astype(np.float32)
    return wrapped


@_keep_zeros
def add_noise(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.005) -> np.ndarray:
    """Gaussian jitter on present hands only."""
    return seq + rng.normal(0, sigma, seq.shape).astype(np.float32)


def scale(seq: np.ndarray, factor: float) -> np.ndarray:
    """Uniform scale of wrist-relative coordinates (zeros stay zeros)."""
    return (seq * factor).astype(np.float32)


def rotate(seq: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate (x, y) of every keypoint about its wrist by `degrees`."""
    t = np.deg2rad(degrees)
    c, s = np.cos(t), np.sin(t)
    out = seq.copy()
    x, y = seq[:, X_IDX], seq[:, Y_IDX]
    out[:, X_IDX] = c * x - s * y
    out[:, Y_IDX] = s * x + c * y
    return out.astype(np.float32)


def hand_dropout(seq: np.ndarray, rng: np.random.Generator, max_frames: int = 4) -> np.ndarray:
    """Zero a few random frames per hand slot to mimic MediaPipe misses."""
    out = seq.copy()
    for slot in range(2):
        n = rng.integers(0, max_frames + 1)
        if n:
            idx = rng.choice(len(seq), size=min(n, len(seq)), replace=False)
            out[idx, slot * HAND_DIM:(slot + 1) * HAND_DIM] = 0
    return out


def place_in_window(sign: np.ndarray, pad: np.ndarray, rng: np.random.Generator,
                    window: int = WINDOW, min_cover: float = 0.8) -> np.ndarray:
    """Put `sign` at a random offset inside a `window`-frame buffer pre-filled with `pad`.

    The offset is chosen so at least `min_cover` of the sign (and >= min_cover of the
    window when the sign is longer than the window) stays visible — mimicking a rolling
    live buffer that holds most of the sign plus surrounding rest frames.
    """
    out = pad[:window].copy()
    n = len(sign)
    need = int(np.ceil(min_cover * min(n, window)))
    lo, hi = need - n, window - need             # start offset range (may be negative)
    start = int(rng.integers(lo, hi + 1))
    s0, s1 = max(0, start), min(window, start + n)
    out[s0:s1] = sign[s0 - start:s1 - start]
    return out


def to_model_input(window_seq: np.ndarray) -> np.ndarray:
    """Resample a (WINDOW, 126) buffer to (MODEL_LEN, 126) — identical to the live path."""
    return resample(window_seq, MODEL_LEN)
