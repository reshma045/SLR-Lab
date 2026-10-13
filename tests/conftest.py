"""Make the repo root importable and expose shared fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def random_seq(rng) -> np.ndarray:
    """A (80, 126) fake landmark sequence with a few missing-hand frames."""
    seq = rng.normal(0, 0.05, (80, 126)).astype(np.float32)
    seq[:10, :63] = 0
    seq[70:, 63:] = 0
    return seq
