"""Make the repo root importable and expose shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import FEATURE_DIM, PRESENT  # noqa: E402
from src.common.features import BLOCKS  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def random_seq(rng) -> np.ndarray:
    """(80, FEATURE_DIM) fake features: left hand missing early, right hand missing late, face gap."""
    seq = rng.normal(0, 0.5, (80, FEATURE_DIM)).astype(np.float32)
    pres = np.ones((80, 4), np.float32)
    pres[:10, 0] = 0
    pres[70:, 1] = 0
    pres[30:35, 2] = 0
    seq[:, PRESENT] = pres
    for idx, need in BLOCKS:                      # absent parts are exactly zero, like real data
        ok = (pres[:, need] > 0.5).all(1)
        seq[np.ix_(~ok, idx)] = 0
    return seq
