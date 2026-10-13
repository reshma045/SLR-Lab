"""Make the repo root importable and expose shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.common.config import FEATURE_DIM, NOSE, POS_L, POS_R, PRESENT, SHAPE_L, SHAPE_R  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def random_seq(rng) -> np.ndarray:
    """An (80, 136) fake feature sequence: left hand missing early, right hand missing late."""
    seq = rng.normal(0, 0.5, (80, FEATURE_DIM)).astype(np.float32)
    seq[:, PRESENT] = 1.0
    seq[:10, SHAPE_L] = 0
    seq[:10, POS_L] = 0
    seq[:10, PRESENT.start] = 0
    seq[70:, SHAPE_R] = 0
    seq[70:, POS_R] = 0
    seq[70:, PRESENT.start + 1] = 0
    seq[:, NOSE] = rng.normal(0, 0.1, (80, 2)) + np.array([0, -0.7])
    return seq
