"""Sample generation that mirrors the live path: rolling 96-frame window -> resample to 60.

Each training sample is built from a sign's full native-fps sequence (rest -> sign -> rest):
speed-warp it, slide a 96-frame window over it so that most of the active sign ("core") is
inside, fill anything outside the recording with zeros / rest frames, augment, resample to 60.
Idle samples come from empty frames, the signer's rest pose, and partial signs (the first
<=50% or last <=30% of a sign) so the model waits until a sign is mostly in the buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.config import DATA, FEATURE_DIM, IDLE_INDEX, NUM_SIGNS, WINDOW
from src.common.features import resample
from src.train import augment as A

CORE_THRESHOLD = 0.12      # distance from rest pose that counts as "signing"
TRAIN_SPEEDS = [(0.70, 0.85), (0.90, 1.20), (1.25, 1.40)]
VAL_SPEEDS = [(0.85, 0.90), (1.20, 1.25)]          # held out from training


@dataclass
class SignSeq:
    """One reference recording with its active core [on, off) and rest bank."""
    label: int
    seq: np.ndarray
    on: int
    off: int

    @property
    def rest(self) -> np.ndarray:
        return np.concatenate([self.seq[: max(self.on, 8)], self.seq[max(self.off, len(self.seq) - 8):]])


def find_core(seq: np.ndarray, margin: int = 3) -> tuple[int, int]:
    """Return [on, off) of the active part: frames that differ from the start/end rest poses."""
    k = np.ones(5) / 5
    d0 = np.convolve(np.linalg.norm(seq - seq[:8].mean(0), axis=1), k, "same")
    d1 = np.convolve(np.linalg.norm(seq - seq[-8:].mean(0), axis=1), k, "same")
    on = int(np.argmax(d0 > CORE_THRESHOLD)) if (d0 > CORE_THRESHOLD).any() else 0
    off = len(seq) - int(np.argmax(d1[::-1] > CORE_THRESHOLD)) if (d1 > CORE_THRESHOLD).any() else len(seq)
    on, off = max(0, on - margin), min(len(seq), off + margin)
    if off - on < 30:                                  # degenerate -> fall back to middle 60%
        on, off = int(len(seq) * 0.2), int(len(seq) * 0.8)
    return on, off


def load_signs() -> list[SignSeq]:
    """Load data/landmarks_full/XX.npy for all signs."""
    out = []
    for i in range(NUM_SIGNS):
        seq = np.load(DATA / "landmarks_full" / f"{i:02d}.npy").astype(np.float32)
        out.append(SignSeq(i, seq, *find_core(seq)))
    return out


def _speed(rng: np.random.Generator, bands: list[tuple[float, float]]) -> float:
    lo, hi = bands[rng.integers(len(bands))]
    return float(rng.uniform(lo, hi))


def _augment(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        win = A.mirror(win)
    win = A.scale(win, rng.uniform(0.85, 1.15))
    win = A.rotate(win, rng.uniform(-10, 10))
    win = A.hand_dropout(win, rng, max_frames=6)
    return A.add_noise(win, rng, sigma=0.005)


def _outside_fill(s: SignSeq, n: int, rng: np.random.Generator) -> np.ndarray:
    """Frames for buffer positions before/after the recording: zeros or a held rest pose."""
    if rng.random() < 0.5:
        return np.zeros((n, FEATURE_DIM), np.float32)
    return np.repeat(s.rest[rng.integers(len(s.rest))][None], n, axis=0)


def _warped(s: SignSeq, rng: np.random.Generator, bands) -> tuple[np.ndarray, int, int]:
    """Speed-warped sequence and its core [on, off); sometimes rest frames become no-hands."""
    f = _speed(rng, bands)
    seq = A.speed_warp(s.seq, f)
    on = min(int(round(s.on / f)), len(seq) - 2)
    off = min(max(on + 2, int(round(s.off / f))), len(seq))
    if rng.random() < 0.3:                              # user's hands out of view around the sign
        seq = seq.copy()
        seq[:on] = 0
        seq[off:] = 0
    return seq, on, off


def _cut(s: SignSeq, seq: np.ndarray, w: int, rng: np.random.Generator) -> np.ndarray:
    """Buffer = seq[w : w+WINDOW]; positions outside the recording get zeros / held rest."""
    pre = min(WINDOW, max(0, -w))
    body = seq[max(0, w):max(0, min(len(seq), w + WINDOW))]
    post = WINDOW - pre - len(body)
    parts = [_outside_fill(s, pre, rng)] if pre else []
    parts += [body] + ([_outside_fill(s, post, rng)] if post else [])
    return np.concatenate(parts).astype(np.float32)


def sign_window(s: SignSeq, rng: np.random.Generator, bands, min_cover: float = 0.8) -> np.ndarray:
    """A (WINDOW, 126) live-like buffer containing >= min_cover of the sign's core."""
    seq, on, off = _warped(s, rng, bands)
    need = int(np.ceil(min_cover * min(off - on, WINDOW)))
    # window start w (seq coords) with overlap([w, w+WINDOW), [on, off)) >= need
    return _cut(s, seq, int(rng.integers(on + need - WINDOW, off - need + 1)), rng)


def fragment_window(s: SignSeq, rng: np.random.Generator, bands) -> np.ndarray:
    """Idle: the buffer holds only the opening (<=50%) or the tail end (<=30%) of a sign,
    positioned exactly where the live buffer would have it (newest / oldest frames)."""
    seq, on, off = _warped(s, rng, bands)
    if rng.random() < 0.65:                             # sign just started: newest frames
        n = int((off - on) * rng.uniform(0.05, 0.5))
        return _cut(s, seq, on + n - WINDOW, rng)
    n = int((off - on) * rng.uniform(0.0, 0.3))         # sign almost scrolled out: oldest frames
    return _cut(s, seq, off - n, rng)


def idle_window(signs: list[SignSeq], rng: np.random.Generator, bands=TRAIN_SPEEDS) -> np.ndarray:
    """A (WINDOW, 126) buffer with no complete sign in it."""
    r = rng.random()
    s = signs[rng.integers(len(signs))]
    if r < 0.2:                                         # nobody / hands out of view
        win = np.zeros((WINDOW, FEATURE_DIM), np.float32)
    elif r < 0.45:                                      # resting, hands clasped, stretched
        rest = s.rest
        a = rng.integers(0, max(1, len(rest) - 10))
        win = resample(rest[a:a + rng.integers(10, len(rest) - a + 1)], WINDOW)
    elif r < 0.6:                                       # hands come into / leave view at rest
        win = np.zeros((WINDOW, FEATURE_DIM), np.float32)
        k = int(rng.integers(10, WINDOW - 10))
        part = resample(s.rest, WINDOW - k)
        if rng.random() < 0.5:
            win[k:] = part
        else:
            win[:WINDOW - k] = part
    else:
        return fragment_window(s, rng, bands)
    return win


class WindowSampler:
    """Generates (x, y) batches on the fly; `val=True` uses held-out speed bands."""

    def __init__(self, signs: list[SignSeq], per_class: int = 96, idle_ratio: float = 3.0,
                 val: bool = False) -> None:
        self.signs, self.per_class, self.idle_n = signs, per_class, int(per_class * idle_ratio)
        self.bands = VAL_SPEEDS if val else TRAIN_SPEEDS

    def epoch(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One epoch of freshly augmented samples, shape (N, 60, 126), labels (N,)."""
        xs, ys = [], []
        for s in self.signs:
            for _ in range(self.per_class):
                xs.append(A.to_model_input(_augment(sign_window(s, rng, self.bands), rng)))
                ys.append(s.label)
        for _ in range(self.idle_n):
            xs.append(A.to_model_input(_augment(idle_window(self.signs, rng, self.bands), rng)))
            ys.append(IDLE_INDEX)
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)
