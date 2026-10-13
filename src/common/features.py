"""Hand-landmark feature extraction shared by offline extraction and live inference.

Layout (must never diverge between training and inference):
    126 = Left-hand slot [0:63] + Right-hand slot [63:126]
    each slot = 21 keypoints x (x, y, z), each minus the wrist keypoint
    missing hand -> zeros
Frames are never mirrored before MediaPipe; mirroring for display is CSS-only.
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.common.config import FEATURE_DIM, HAND_DIM


def make_hands(det_conf: float = 0.5, track_conf: float = 0.5, static: bool = False):
    """Create a MediaPipe Hands instance (max 2 hands)."""
    return mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2,
                                    min_detection_confidence=det_conf,
                                    min_tracking_confidence=track_conf)


def results_to_vector(results) -> np.ndarray:
    """Convert MediaPipe Hands results into the 126-d wrist-relative feature vector."""
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    if results.multi_hand_landmarks:
        for lm, info in zip(results.multi_hand_landmarks, results.multi_handedness):
            slot = 0 if info.classification[0].label == "Left" else 1
            pts = np.array([(p.x, p.y, p.z) for p in lm.landmark], dtype=np.float32)
            pts -= pts[0]  # wrist-relative
            out[slot * HAND_DIM:(slot + 1) * HAND_DIM] = pts.flatten()
    return out


def process_bgr(frame_bgr: np.ndarray, hands):
    """Run MediaPipe on a BGR frame; return (feature vector, raw results)."""
    results = hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return results_to_vector(results), results


def frame_landmarks(frame_bgr: np.ndarray, hands) -> np.ndarray:
    """Feature vector for one BGR frame."""
    return process_bgr(frame_bgr, hands)[0]


def hands_present(vec: np.ndarray) -> tuple[bool, bool]:
    """(left present, right present) for a 126-d vector."""
    return bool(np.abs(vec[:HAND_DIM]).sum() > 0), bool(np.abs(vec[HAND_DIM:]).sum() > 0)


def resample(seq: np.ndarray, target: int) -> np.ndarray:
    """Linearly interpolate a (T, D) sequence to (target, D)."""
    t = len(seq)
    if t == target:
        return seq.astype(np.float32)
    if t == 1:
        return np.repeat(seq, target, axis=0).astype(np.float32)
    idx = np.linspace(0, t - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    a = (idx - lo)[:, None]
    return (seq[lo] * (1 - a) + seq[hi] * a).astype(np.float32)
