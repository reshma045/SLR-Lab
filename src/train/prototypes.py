"""Few-shot "personal examples": the model's embedding of each of the user's takes.

Live, the current window's embedding is compared (cosine) with these, and the result is blended
with the classifier (see JSLPredictor). New takes are added instantly after recording - no
retraining needed - and the whole set is rebuilt after every training run.
"""
from __future__ import annotations

import numpy as np

from src.common.config import MODELS, NUM_SIGNS, USER_DATA
from src.common.landmarks import RAW_DIM
from src.train import augment as A
from src.train.dataset import _make, center_window

PROTO_FILE = MODELS / "prototypes.npz"


def embed(session, raw: np.ndarray, label: int) -> np.ndarray | None:
    """L2-normalised embedding of one take (its sign centred in a live-like window)."""
    if raw.ndim != 2 or raw.shape[1] != RAW_DIM or len(raw) < 10:
        return None
    s = _make(label, raw, "user", "")
    x = A.to_model_input(center_window(s))[None]
    e = session.run(["embedding"], {"input": x})[0][0]
    return (e / (np.linalg.norm(e) + 1e-8)).astype(np.float32)


def load() -> tuple[np.ndarray, np.ndarray] | None:
    """(labels (N,), vectors (N, E)) or None."""
    if not PROTO_FILE.exists():
        return None
    d = np.load(PROTO_FILE)
    return (d["labels"], d["vecs"]) if len(d["labels"]) else None


def _save(labels: list[int], vecs: list[np.ndarray]) -> None:
    MODELS.mkdir(exist_ok=True)
    tmp = PROTO_FILE.with_name("prototypes.tmp.npz")
    np.savez(tmp, labels=np.asarray(labels, np.int64),
             vecs=np.stack(vecs).astype(np.float32) if vecs else np.zeros((0, 1), np.float32))
    tmp.replace(PROTO_FILE)


def rebuild(session, profile: str) -> int:
    """Embed every take of `profile` with the current model. Returns the number of prototypes."""
    labels, vecs = [], []
    base = USER_DATA / profile
    for i in range(NUM_SIGNS):
        for f in sorted((base / f"{i:02d}").glob("*.npy")) if base.exists() else []:
            e = embed(session, np.load(f), i)
            if e is not None:
                labels.append(i)
                vecs.append(e)
    _save(labels, vecs)
    return len(labels)


def add(session, raw: np.ndarray, label: int) -> bool:
    """Append one new take's embedding (called right after recording)."""
    e = embed(session, raw, label)
    if e is None:
        return False
    cur = load()
    labels = list(cur[0]) if cur else []
    vecs = list(cur[1]) if cur else []
    if vecs and vecs[0].shape != e.shape:          # model changed since the file was built
        labels, vecs = [], []
    _save(labels + [label], vecs + [e])
    return True
