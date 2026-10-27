"""Live gloss-free translation: segment continuous signing by hand motion, translate each segment.

A segment starts when a hand moves (wrist speed above a threshold) for a moment, and ends when the
hands are still or out of view for PAUSE_S. The segment's frames are put on the 30 fps timeline and
fed to the gloss-free model, which outputs a Japanese sentence directly (no sign labels in between).
"""
from __future__ import annotations

import threading
from collections import deque

import numpy as np

from src.common.config import FEATURE_DIM, NATIVE_FPS, POS_L, POS_R, PRESENT
from src.glossfree.corpus import CharVocab, build
from src.glossfree.stitch import batch, to_input

PAUSE_S = 1.2          # stillness that ends a sentence
START_S = 0.3          # sustained motion that starts one
MIN_S, MAX_S = 1.0, 15.0
SPEED = 0.35           # shoulder widths per second
PAD_S = 0.6            # context kept before the first motion / after the last

_lock = threading.Lock()
_cache: dict = {}


def load_model():
    """(model, vocab, ja->en) shared by all clients, or None if not trained. Reloads if retrained."""
    from src.glossfree.train import CKPT
    if not CKPT.exists():
        return None
    mtime = CKPT.stat().st_mtime
    with _lock:
        if _cache.get("mtime") != mtime:
            import torch
            from src.glossfree.model import GlossFreeSLT
            ck = torch.load(CKPT, map_location="cpu")
            vocab = CharVocab([])
            vocab.chars = ck["chars"]
            vocab.index = {c: i for i, c in enumerate(vocab.chars)}
            m = GlossFreeSLT(len(vocab.chars))
            m.load_state_dict(ck["state_dict"])
            m.eval()
            _cache.update(mtime=mtime, model=m, vocab=vocab, en={e["ja"]: e["en"] for e in build()})
        return _cache["model"], _cache["vocab"], _cache["en"]


def translate(times: list[float], vecs: list[np.ndarray]) -> tuple[str, str | None] | None:
    loaded = load_model()
    if loaded is None:
        return None
    import torch
    model, vocab, en = loaded
    t = np.asarray(times)
    v = np.stack(vecs)
    grid = np.arange(t[0], t[-1] + 1e-9, 1 / NATIVE_FPS)
    seq = np.stack([np.interp(grid, t, v[:, k]) for k in range(FEATURE_DIM)], 1).astype(np.float32)
    x, pad = batch([to_input(seq)])
    with torch.no_grad():
        ids = model.greedy(torch.from_numpy(x), torch.from_numpy(pad), vocab.BOS, vocab.EOS)[0]
    ja = vocab.decode(ids)
    return ja, en.get(ja)


class GlossFreeLive:
    """Per-client segmenter. Call push() for every frame's feature vector."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.buf: deque[tuple[float, np.ndarray]] = deque()
        self.prev: tuple[float, np.ndarray] | None = None
        self.moving_since: float | None = None
        self.start: float | None = None
        self.last_active = 0.0

    def _speed(self, t: float, v: np.ndarray) -> float:
        if self.prev is None:
            return 0.0
        pt, pv = self.prev
        dt = max(t - pt, 1e-3)
        sp = 0.0
        for k, sl in enumerate((POS_L, POS_R)):
            if v[PRESENT.start + k] > 0.5 and pv[PRESENT.start + k] > 0.5:
                sp = max(sp, float(np.linalg.norm(v[sl][:2] - pv[sl][:2])) / dt)
        return sp

    def push(self, t: float, v: np.ndarray) -> dict:
        """Returns {"state": "idle" | "signing"} or, when a sentence finishes, {"state": "done", ...}."""
        self.buf.append((t, v))
        while self.buf and t - self.buf[0][0] > MAX_S + 2 * PAD_S:
            self.buf.popleft()
        active = self._speed(t, v) > SPEED
        self.prev = (t, v)
        if self.start is None:
            if active:
                self.moving_since = self.moving_since if self.moving_since is not None else t
                if t - self.moving_since >= START_S:
                    self.start, self.last_active = self.moving_since - PAD_S, t
            else:
                self.moving_since = None
            return {"state": "idle"}
        if active:
            self.last_active = t
        hands = v[PRESENT.start] > 0.5 or v[PRESENT.start + 1] > 0.5
        ended = t - self.last_active > PAUSE_S or (not hands and t - self.last_active > PAUSE_S / 2)
        if not ended and t - self.start < MAX_S:
            return {"state": "signing", "elapsed": round(t - self.start, 1)}
        seg = [(ft, fv) for ft, fv in self.buf if self.start <= ft <= self.last_active + PAD_S]
        dur = self.last_active - self.start
        self.start, self.moving_since = None, None
        if dur < MIN_S or len(seg) < 10:
            return {"state": "idle"}
        res = translate([s[0] for s in seg], [s[1] for s in seg])
        if res is None:
            return {"state": "unavailable"}
        return {"state": "done", "japanese": res[0], "english": res[1], "duration": round(dur, 1)}
