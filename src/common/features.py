"""Hand + body feature extraction shared by offline extraction and live inference.

Layout v2 (136 floats per frame; slices in src/common/config.py):
    [0:63]    signer's LEFT hand shape   - 21 keypoints (x, y, z), wrist-relative, divided by palm length
    [63:126]  signer's RIGHT hand shape
    [126:129] left wrist (x, y) relative to the shoulder centre / shoulder width, palm length / shoulder width
    [129:132] right wrist, same
    [132:134] nose (x, y) relative to the shoulder centre / shoulder width
    [134:136] left / right hand present (1.0 / 0.0)

Why: v1 used raw wrist-relative coordinates, so a hand close to a laptop webcam looked ~3x bigger
than in the full-body reference videos, and the model could not tell where the hands were relative
to the face/body. Shapes are now scale-free and positions are body-relative.

All x coordinates are multiplied by the frame aspect ratio first, so 720x480 references and 640x480
webcams share one isotropic coordinate system. Frames are never mirrored before MediaPipe; "left" is
the signer's own left hand (assigned via the nearest pose wrist, else MediaPipe's handedness label).
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from src.common.config import FEATURE_DIM, NOSE, POS_L, POS_R, PRESENT, SHAPE_L, SHAPE_R

_P = mp.solutions.pose.PoseLandmark
MIDDLE_MCP = 9


class FeatureExtractor:
    """MediaPipe Hands + Pose -> 136-d frame vectors. One instance per video / live client."""

    def __init__(self, det_conf: float = 0.5, track_conf: float = 0.5, static: bool = False,
                 pose_every: int = 1) -> None:
        self.hands = mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2,
                                              min_detection_confidence=det_conf,
                                              min_tracking_confidence=track_conf)
        self.pose = mp.solutions.pose.Pose(static_image_mode=static, model_complexity=0,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_every = max(1, pose_every)
        self.n = 0
        self.body: dict | None = None        # last shoulder centre / width / wrists / nose

    def close(self) -> None:
        self.hands.close()
        self.pose.close()

    def __enter__(self) -> "FeatureExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray):
        """Return (136-d vector, hands results, pose results or None if pose was skipped)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hres = self.hands.process(rgb)
        pres = None
        if self.n % self.pose_every == 0:
            pres = self.pose.process(rgb)
            self.body = body_reference(pres, w / h)
        self.n += 1
        return build_vector(hres, self.body, w / h), hres, pres


def body_reference(pose_results, aspect: float) -> dict | None:
    """Shoulder centre/width, wrists and nose in aspect-corrected coordinates, or None."""
    if pose_results is None or not pose_results.pose_landmarks:
        return None
    lm = pose_results.pose_landmarks.landmark

    def pt(i):
        return np.array([lm[i].x * aspect, lm[i].y], dtype=np.float32)

    ls, rs = pt(_P.LEFT_SHOULDER), pt(_P.RIGHT_SHOULDER)
    width = float(np.linalg.norm(ls - rs))
    if width < 1e-3 or min(lm[_P.LEFT_SHOULDER].visibility, lm[_P.RIGHT_SHOULDER].visibility) < 0.3:
        return None
    return {"center": (ls + rs) / 2, "width": width, "nose": pt(_P.NOSE),
            "wrists": (pt(_P.LEFT_WRIST), pt(_P.RIGHT_WRIST))}


def build_vector(hand_results, body: dict | None, aspect: float) -> np.ndarray:
    """Assemble the 136-d feature vector from MediaPipe results."""
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    hands = []
    if hand_results is not None and hand_results.multi_hand_landmarks:
        for lm, info in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            pts = np.array([(p.x * aspect, p.y, p.z * aspect) for p in lm.landmark], dtype=np.float32)
            # MediaPipe assumes a mirrored selfie image: on raw frames its "Left" is the signer's right.
            hands.append((pts, 1 if info.classification[0].label == "Left" else 0))
    slots = assign_slots([h[0] for h in hands], [h[1] for h in hands], body)
    for (pts, _), slot in zip(hands, slots):
        if slot is None:
            continue
        rel = pts - pts[0]
        palm = float(np.linalg.norm(rel[MIDDLE_MCP, :2])) or 1e-3
        out[(SHAPE_L, SHAPE_R)[slot]] = (rel / palm).ravel()
        out[PRESENT.start + slot] = 1.0
        if body is not None:
            pos = (pts[0, :2] - body["center"]) / body["width"]
            out[(POS_L, POS_R)[slot]] = [pos[0], pos[1], palm / body["width"]]
    if body is not None:
        out[NOSE] = (body["nose"] - body["center"]) / body["width"]
    return out


def assign_slots(hand_pts: list[np.ndarray], labels: list[int], body: dict | None) -> list[int | None]:
    """Signer-left = 0, signer-right = 1. Prefer the nearest pose wrist; fall back to labels/x order."""
    if not hand_pts:
        return []
    if body is not None:
        wl, wr = body["wrists"]
        cost = [[float(np.linalg.norm(p[0, :2] - w)) for w in (wl, wr)] for p in hand_pts]
        if len(hand_pts) == 1:
            return [int(np.argmin(cost[0]))]
        straight = cost[0][0] + cost[1][1]
        swapped = cost[0][1] + cost[1][0]
        return [0, 1] if straight <= swapped else [1, 0]
    if len(hand_pts) == 2 and labels[0] == labels[1]:
        # the signer's left hand appears on the image's right (larger x) on raw, unmirrored frames
        return [0, 1] if hand_pts[0][0, 0] > hand_pts[1][0, 0] else [1, 0]
    return list(labels[:2]) + [None] * (len(hand_pts) - 2)


def hands_present(vec: np.ndarray) -> tuple[bool, bool]:
    """(left present, right present) for a feature vector."""
    return bool(vec[PRESENT.start] > 0.5), bool(vec[PRESENT.start + 1] > 0.5)


def pose_present(vec: np.ndarray) -> bool:
    """True if the shoulders were found (nose/positions are body-relative)."""
    return bool(np.abs(vec[NOSE]).sum() > 0)


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
