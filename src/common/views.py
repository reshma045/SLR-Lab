"""Simulated camera framings of the full-body reference videos (pixel-level augmentation/testing)."""
from __future__ import annotations

import cv2
import numpy as np


def webcam_view(frame: np.ndarray, zoom: float, shift_x: float = 0.0) -> np.ndarray:
    """Crop a 4:3 upper-body window and resize to 640x480, like sitting close to a laptop webcam.

    zoom 1 = whole frame height; 1.8 ~ typical laptop distance. shift_x moves the crop sideways
    (fraction of the crop width) to mimic a signer who is not centred.
    """
    h, w = frame.shape[:2]
    ch = int(h / zoom)
    cw = min(w, int(ch * 4 / 3))
    cx, cy = int(w / 2 + shift_x * cw), int(h * 0.36)
    x0 = max(0, min(w - cw, cx - cw // 2))
    y0 = max(0, min(h - ch, cy - ch // 2))
    return cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_LINEAR)
