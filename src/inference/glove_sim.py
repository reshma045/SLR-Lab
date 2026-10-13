"""Glove sensor sources for the UI's sensor panel.

IMPORTANT: `SimulatedGlove` is a *simulation* for the demo UI. Its output is never fed to the
classifier. To connect a real tactile glove, implement `GloveSource.read()` with the same
return shape and pass it to `JSLPredictor(glove=...)`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.common.config import HAND_DIM

N_TAXELS = 50          # 5 x 10 pressure grid per hand
N_IMU = 6              # accel xyz + gyro xyz
TIPS = [4, 8, 12, 16, 20]


class GloveSource(ABC):
    """Interface for a pair of sensor gloves."""

    @abstractmethod
    def read(self, landmarks: np.ndarray | None = None) -> dict:
        """Return {"left": {"pressure": [50 floats 0..1], "imu": [6 floats]}, "right": {...}}.

        `landmarks` (the current 126-d feature vector) is optional context; real gloves ignore it.
        """


class SimulatedGlove(GloveSource):
    """Smooth random-walk pressure/IMU, loosely driven by detected hand openness and motion."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.pressure = {h: self.rng.uniform(0.05, 0.2, N_TAXELS) for h in ("left", "right")}
        self.imu = {h: np.zeros(N_IMU) for h in ("left", "right")}
        self.prev = {h: None for h in ("left", "right")}

    def _hand(self, name: str, pts: np.ndarray | None) -> dict:
        target = np.full(N_TAXELS, 0.08)
        motion = np.zeros(3)
        if pts is not None:
            spread = float(np.linalg.norm(pts[TIPS, :2], axis=1).mean())   # ~0.05 fist .. 0.2 open
            grip = float(np.clip(1.0 - (spread - 0.05) / 0.15, 0.0, 1.0))
            grid = np.zeros((5, 10))
            grid[:, :6] = grip                       # finger pads press when the hand closes
            grid[:, 6:] = 0.3 + 0.5 * grip           # palm region
            target = grid.ravel() * self.rng.uniform(0.7, 1.0, N_TAXELS)
            if self.prev[name] is not None:
                motion = (pts[TIPS].mean(0) - self.prev[name][TIPS].mean(0)) * 40
            self.prev[name] = pts
        else:
            self.prev[name] = None
        p = self.pressure[name]
        p += 0.25 * (target - p) + self.rng.normal(0, 0.02, N_TAXELS)
        self.pressure[name] = np.clip(p, 0, 1)
        imu_target = np.concatenate([motion, np.array([motion[1], -motion[0], motion[2]]) * 0.5])
        self.imu[name] = 0.7 * self.imu[name] + 0.3 * imu_target + self.rng.normal(0, 0.03, N_IMU)
        return {"pressure": [round(float(v), 3) for v in self.pressure[name]],
                "imu": [round(float(v), 3) for v in self.imu[name]]}

    def read(self, landmarks: np.ndarray | None = None) -> dict:
        out = {}
        for slot, name in enumerate(("left", "right")):
            pts = None
            if landmarks is not None:
                h = landmarks[slot * HAND_DIM:(slot + 1) * HAND_DIM]
                if np.abs(h).sum() > 0:
                    pts = h.reshape(21, 3)
            out[name] = self._hand(name, pts)
        out["simulated"] = True
        return out
