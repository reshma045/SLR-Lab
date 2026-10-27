"""Synthetic signed sentences by stitching real isolated-sign motion (PoseStitch-SLT idea).

For a sentence's sign sequence: take each sign's active part from a randomly chosen recording
(reference video, its simulated webcam crops, or the user's takes), speed-warp it, join neighbours
with a short interpolated transition, add rest at both ends, then augment the whole sequence.
"""
from __future__ import annotations

import numpy as np

from src.common.config import FEATURE_DIM, MODEL_IN, class_names
from src.common.features import add_motion, resample
from src.train.dataset import SignSeq, augment
from src.train import augment as A

STEP = 2              # 30 fps landmark timeline -> 15 steps / s model timeline
MAX_STEPS = 256       # ~17 s


class Stitcher:
    def __init__(self, signs: list[SignSeq]) -> None:
        names = class_names()
        self.pool: dict[str, list[SignSeq]] = {}
        for s in signs:
            self.pool.setdefault(names[s.label], []).append(s)

    def stitch(self, sign_names: list[str], rng: np.random.Generator, aug: bool = True) -> np.ndarray:
        """(T, FEATURE_DIM) 30 fps feature sequence of the signed sentence."""
        parts: list[np.ndarray] = []
        last = None
        for k, name in enumerate(sign_names):
            options = self.pool[name]
            s = options[rng.integers(len(options))]
            m = 3
            core = s.seq[max(0, s.on - m):min(len(s.seq), s.off + m)]
            core = A.speed_warp(core, float(rng.uniform(0.8, 1.25))) if aug else core
            if k == 0:
                n = int(rng.integers(8, 25))
                parts.append(s.seq[max(0, s.on - m - n):max(0, s.on - m)] if s.on - m > 0 else
                             np.zeros((n, FEATURE_DIM), np.float32))
            elif last is not None:
                n = int(rng.integers(3, 10))
                a = np.linspace(0, 1, n + 2)[1:-1, None]
                parts.append((last[-1] * (1 - a) + core[0] * a).astype(np.float32))
            parts.append(core)
            last = core
            if k == len(sign_names) - 1:
                n = int(rng.integers(8, 25))
                tail = s.seq[min(len(s.seq), s.off + m):min(len(s.seq), s.off + m + n)]
                parts.append(tail if len(tail) else np.zeros((n, FEATURE_DIM), np.float32))
        seq = np.concatenate([p for p in parts if len(p)]).astype(np.float32)
        return augment(seq, rng) if aug else seq


def to_input(seq30: np.ndarray) -> np.ndarray:
    """30 fps features -> (steps, MODEL_IN) model input at 15 steps/s with velocity features."""
    n = max(2, min(MAX_STEPS, int(round(len(seq30) / STEP))))
    x = add_motion(resample(seq30, n))
    assert x.shape[1] == MODEL_IN
    return x


def batch(xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length inputs -> (B, T, D) array and (B, T) bool padding mask (True = pad)."""
    T = max(len(x) for x in xs)
    out = np.zeros((len(xs), T, MODEL_IN), np.float32)
    mask = np.ones((len(xs), T), bool)
    for i, x in enumerate(xs):
        out[i, :len(x)] = x
        mask[i, :len(x)] = False
    return out, mask
