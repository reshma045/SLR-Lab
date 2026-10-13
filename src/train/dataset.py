"""Sample generation that mirrors the live path: last 3.2 s (96 frames @30 fps) -> resample to 60.

Each training sample is built from a full 30 fps sequence (rest -> sign -> rest): speed-warp it,
slide a 96-frame window over it so that most of the active sign ("core") is inside, fill anything
outside the recording with zeros / rest frames, augment, resample to 60. Idle samples come from
empty frames, rest poses, and partial signs (first <=50% / last <=30% of a sign).

Sequences come from the 25 reference videos and, if present, the user's own recordings
(data/user/<profile>/<sign id>/*.npy, recorded from the dashboard). When the user has >= 3
recordings of a sign, the newest one is held out as a *real* validation example.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.common.config import (DATA, FEATURE_DIM, IDLE_INDEX, NUM_SIGNS, POS_L, POS_R, PRESENT,
                               USER_DATA, WINDOW)
from src.common.features import resample, sequence_features
from src.common.landmarks import RAW_DIM
from src.train import augment as A

CORE_THRESHOLD = 0.6       # motion-signal distance from the rest pose that counts as "signing"
TRAIN_SPEEDS = [(0.70, 0.85), (0.90, 1.20), (1.25, 1.40)]
VAL_SPEEDS = [(0.85, 0.90), (1.20, 1.25)]          # held out from training
USER_SHARE = 0.7           # fraction of a sign's samples drawn from user recordings (if any)
MIN_FOR_HOLDOUT = 3


@dataclass
class SignSeq:
    """One recording with its active core [on, off)."""
    label: int
    seq: np.ndarray
    on: int
    off: int
    source: str = "ref"                    # "ref" or "user"
    name: str = field(default="")

    @property
    def rest(self) -> np.ndarray:
        return np.concatenate([self.seq[: max(self.on, 8)], self.seq[max(self.off, len(self.seq) - 8):]])


def motion_signal(seq: np.ndarray) -> np.ndarray:
    """Scale-free per-frame descriptor used only to locate the active part of a recording."""
    return np.concatenate([seq[:, POS_L], seq[:, POS_R], seq[:, PRESENT] * 0.5, seq[:, :126] * 0.03], axis=1)


def find_core(seq: np.ndarray, margin: int = 3) -> tuple[int, int]:
    """Return [on, off) of the active part: frames that differ from the start/end rest poses."""
    m = motion_signal(seq)
    k = np.ones(5) / 5
    d0 = np.convolve(np.linalg.norm(m - m[:8].mean(0), axis=1), k, "same")
    d1 = np.convolve(np.linalg.norm(m - m[-8:].mean(0), axis=1), k, "same")
    on = int(np.argmax(d0 > CORE_THRESHOLD)) if (d0 > CORE_THRESHOLD).any() else 0
    off = len(seq) - int(np.argmax(d1[::-1] > CORE_THRESHOLD)) if (d1 > CORE_THRESHOLD).any() else len(seq)
    on, off = max(0, on - margin), min(len(seq), off + margin)
    if off - on < 20:                                  # degenerate -> fall back to middle 60%
        on, off = int(len(seq) * 0.2), int(len(seq) * 0.8)
    return on, off


def _make(label: int, raw: np.ndarray, source: str, name: str) -> SignSeq:
    seq = sequence_features(raw)
    return SignSeq(label, seq, *find_core(seq), source=source, name=name)


def _load_raw(path) -> np.ndarray | None:
    """Raw landmark file, or None for files from an older format."""
    a = np.load(path)
    return a if a.ndim == 2 and a.shape[1] == RAW_DIM else None


def load_signs(profile: str | None = "default") -> tuple[list[SignSeq], list[SignSeq]]:
    """(training sequences, held-out user validation sequences)."""
    train = [_make(i, np.load(DATA / "landmarks_raw" / f"{i:02d}.npy"), "ref", f"ref/{i:02d}")
             for i in range(NUM_SIGNS)]
    aug = DATA / "landmarks_raw_aug"                   # simulated close-webcam crops of the references
    for d in sorted(aug.iterdir()) if aug.exists() else []:
        train += [_make(i, np.load(d / f"{i:02d}.npy"), "ref", f"{d.name}/{i:02d}")
                  for i in range(NUM_SIGNS) if (d / f"{i:02d}.npy").exists()]
    val: list[SignSeq] = []
    base = USER_DATA / profile if profile else None
    if base and base.exists():
        for i in range(NUM_SIGNS):
            files = sorted((base / f"{i:02d}").glob("*.npy"))
            raws = [(f, _load_raw(f)) for f in files]
            seqs = [_make(i, r, "user", f"{profile}/{i:02d}/{f.stem}") for f, r in raws if r is not None]
            seqs = [s for s in seqs if len(s.seq) >= 30 and s.seq[:, PRESENT].any()]
            if len(seqs) >= MIN_FOR_HOLDOUT:
                val.append(seqs.pop())                 # newest recording = real validation example
            train += seqs
    return train, val


def _speed(rng: np.random.Generator, bands: list[tuple[float, float]]) -> float:
    lo, hi = bands[rng.integers(len(bands))]
    return float(rng.uniform(lo, hi))


def augment(win: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Mirror, affine (roll / shear / body scale / shift), part dropout, temporal mask, noise."""
    if rng.random() < 0.5:
        win = A.mirror(win)
    if rng.random() < 0.8:
        win = A.affine(win, rng.uniform(-15, 15), rng.uniform(-0.1, 0.1), rng.uniform(0.85, 1.15),
                       (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)))
    win = A.hand_dropout(win, rng, max_frames=6)
    every = np.arange(len(win))
    if rng.random() < 0.1:                              # face not found (turned away, mask, lighting)
        win = A.drop_part(win, every, A.FACE_PART, 2)
    if rng.random() < 0.06:                             # shoulders not visible
        win = A.drop_part(win, every, A.BODY_PART, 3)
    if rng.random() < 0.3:
        win = A.temporal_mask(win, rng)
    return A.add_noise(win, rng)


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
    if rng.random() < 0.3:                              # hands out of view around the sign
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
    """A (WINDOW, D) live-like buffer containing >= min_cover of the sign's core."""
    seq, on, off = _warped(s, rng, bands)
    need = int(np.ceil(min_cover * min(off - on, WINDOW)))
    return _cut(s, seq, int(rng.integers(on + need - WINDOW, off - need + 1)), rng)


def fragment_window(s: SignSeq, rng: np.random.Generator, bands) -> np.ndarray:
    """Idle: only the opening (<=50%) or tail end (<=30%) of a sign, where the live buffer has it."""
    seq, on, off = _warped(s, rng, bands)
    if rng.random() < 0.65:
        n = int((off - on) * rng.uniform(0.05, 0.5))
        return _cut(s, seq, on + n - WINDOW, rng)
    n = int((off - on) * rng.uniform(0.0, 0.3))
    return _cut(s, seq, off - n, rng)


def idle_window(signs: list[SignSeq], rng: np.random.Generator, bands=TRAIN_SPEEDS) -> np.ndarray:
    """A (WINDOW, D) buffer with no complete sign in it."""
    r = rng.random()
    s = signs[rng.integers(len(signs))]
    if r < 0.2:
        return np.zeros((WINDOW, FEATURE_DIM), np.float32)
    if r < 0.45:                                        # resting, stretched
        rest = s.rest
        a = int(rng.integers(0, max(1, len(rest) - 10)))
        return resample(rest[a:a + int(rng.integers(10, max(11, len(rest) - a + 1)))], WINDOW)
    if r < 0.6:                                         # hands come into / leave view at rest
        win = np.zeros((WINDOW, FEATURE_DIM), np.float32)
        k = int(rng.integers(10, WINDOW - 10))
        part = resample(s.rest, WINDOW - k)
        if rng.random() < 0.5:
            win[k:] = part
        else:
            win[:WINDOW - k] = part
        return win
    return fragment_window(s, rng, bands)


class WindowSampler:
    """Generates (x, y) epochs on the fly; `val=True` uses held-out speed bands."""

    def __init__(self, signs: list[SignSeq], per_class: int = 96, idle_ratio: float = 3.0,
                 val: bool = False) -> None:
        self.signs, self.per_class, self.idle_n = signs, per_class, int(per_class * idle_ratio)
        self.bands = VAL_SPEEDS if val else TRAIN_SPEEDS
        self.by_class: dict[int, tuple[list[SignSeq], list[SignSeq]]] = {}
        for s in signs:
            ref, user = self.by_class.setdefault(s.label, ([], []))
            (user if s.source == "user" else ref).append(s)

    def _pick(self, label: int, rng: np.random.Generator) -> SignSeq:
        ref, user = self.by_class[label]
        pool = user if user and (not ref or rng.random() < USER_SHARE) else ref
        return pool[rng.integers(len(pool))]

    def epoch(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """One epoch of freshly augmented samples, shape (N, 60, D), labels (N,)."""
        xs, ys = [], []
        for label in sorted(self.by_class):
            for _ in range(self.per_class):
                xs.append(A.to_model_input(augment(sign_window(self._pick(label, rng), rng, self.bands), rng)))
                ys.append(label)
        for _ in range(self.idle_n):
            xs.append(A.to_model_input(augment(idle_window(self.signs, rng, self.bands), rng)))
            ys.append(IDLE_INDEX)
        return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def real_val_set(val: list[SignSeq], rng: np.random.Generator, per_seq: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Windows from held-out user recordings at natural speed (no augmentation): the honest metric."""
    xs, ys = [], []
    for s in val:
        for _ in range(per_seq):
            xs.append(A.to_model_input(sign_window(s, rng, [(1.0, 1.0)])))
            ys.append(s.label)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64)


def center_window(s: SignSeq) -> np.ndarray:
    """Deterministic (WINDOW, F) live-like buffer with the sign's core centred (natural speed)."""
    core = s.off - s.on
    w = s.on + core // 2 - WINDOW // 2 if core < WINDOW else s.on + (core - WINDOW) // 2
    pre = min(WINDOW, max(0, -w))
    body = s.seq[max(0, w):max(0, min(len(s.seq), w + WINDOW))]
    post = WINDOW - pre - len(body)
    parts = ([np.zeros((pre, FEATURE_DIM), np.float32)] if pre else []) + [body]
    parts += [np.zeros((post, FEATURE_DIM), np.float32)] if post else []
    return np.concatenate(parts).astype(np.float32)
