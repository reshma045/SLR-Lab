"""Raw landmark capture: MediaPipe Hands + Pose + Face Mesh -> one flat float vector per frame.

Raw frames are what recordings store, so features can be redesigned later without re-recording.
Coordinates: x multiplied by the frame aspect ratio (isotropic), y in [0, 1], z as MediaPipe gives
(hands/face z also scaled by aspect). Frames are never mirrored. "Left" = the signer's own left.

Raw layout (RAW_DIM floats):
    [0:63]     left hand 21 x (x, y, z)        [63:126]  right hand
    [126:128]  left / right hand present
    [128:260]  pose 33 x (x, y, z, visibility)
    [260:659]  face: FACE_KEEP points x (x, y, z)
    659 face present, 660 pose present, 661 aspect ratio
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

LIPS_OUTER_UP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LIPS_OUTER_LO = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER_UP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
LIPS_INNER_LO = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LIP_CONTOURS = [LIPS_OUTER_UP, LIPS_OUTER_LO, LIPS_INNER_UP, LIPS_INNER_LO]   # each symmetric: i <-> 10-i
_fm = mp.solutions.face_mesh_connections


def _pts(edges) -> list[int]:
    return sorted({a for e in edges for a in e})


FACE_KEEP = sorted(set(_pts(_fm.FACEMESH_LIPS) + _pts(_fm.FACEMESH_LEFT_EYE) + _pts(_fm.FACEMESH_RIGHT_EYE)
                       + _pts(_fm.FACEMESH_LEFT_EYEBROW) + _pts(_fm.FACEMESH_RIGHT_EYEBROW)
                       + _pts(_fm.FACEMESH_FACE_OVAL) + [1, 2, 4, 98, 327]))
FACE_POS = {idx: i for i, idx in enumerate(FACE_KEEP)}

R_HAND_L, R_HAND_R, R_HAND_P = slice(0, 63), slice(63, 126), slice(126, 128)
R_POSE = slice(128, 128 + 33 * 4)
R_FACE = slice(R_POSE.stop, R_POSE.stop + 3 * len(FACE_KEEP))
R_FACE_P, R_POSE_P, R_ASPECT = R_FACE.stop, R_FACE.stop + 1, R_FACE.stop + 2
RAW_DIM = R_FACE.stop + 3
P_LSH, P_RSH, P_LEL, P_REL, P_LWR, P_RWR, P_NOSE = 11, 12, 13, 14, 15, 16, 0


class LandmarkExtractor:
    """One instance per video / live client (MediaPipe graphs keep tracking state)."""

    def __init__(self, det_conf: float = 0.3, track_conf: float = 0.3, static: bool = False,
                 pose_every: int = 1, face: bool = True) -> None:
        self.hands = mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2,
                                              min_detection_confidence=det_conf,
                                              min_tracking_confidence=track_conf)
        self.pose = mp.solutions.pose.Pose(static_image_mode=static, model_complexity=0,
                                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face = mp.solutions.face_mesh.FaceMesh(static_image_mode=static, max_num_faces=1,
                                                    refine_landmarks=False, min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5) if face else None
        self.pose_every, self.n = max(1, pose_every), 0
        self.last_pose: tuple[np.ndarray, bool, object] = (np.zeros(132, np.float32), False, None)

    def close(self) -> None:
        self.hands.close()
        self.pose.close()
        if self.face:
            self.face.close()

    def __enter__(self) -> "LandmarkExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def process(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return (raw vector, {"hands", "pose", "face"} MediaPipe results for drawing)."""
        h, w = frame_bgr.shape[:2]
        aspect = w / h
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw = np.zeros(RAW_DIM, np.float32)
        raw[R_ASPECT] = aspect
        if self.n % self.pose_every == 0:
            pres = self.pose.process(rgb)
            if pres.pose_landmarks:
                p = np.array([(l.x * aspect, l.y, l.z * aspect, l.visibility)
                              for l in pres.pose_landmarks.landmark], np.float32).ravel()
                self.last_pose = (p, True, pres)
            else:
                self.last_pose = (np.zeros(132, np.float32), False, pres)
        self.n += 1
        pose_vec, pose_ok, pres = self.last_pose
        raw[R_POSE] = pose_vec
        raw[R_POSE_P] = float(pose_ok)
        hres = self.hands.process(rgb)
        hands, labels = [], []
        if hres.multi_hand_landmarks:
            for lm, info in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                hands.append(np.array([(p.x * aspect, p.y, p.z * aspect) for p in lm.landmark], np.float32))
                # MediaPipe assumes a mirrored selfie: on raw frames its "Left" is the signer's right
                labels.append(1 if info.classification[0].label == "Left" else 0)
        wrists = None
        if pose_ok:
            pv = pose_vec.reshape(33, 4)
            wrists = (pv[P_LWR, :2], pv[P_RWR, :2])
        for pts, slot in zip(hands, assign_slots(hands, labels, wrists)):
            if slot is not None:
                raw[(R_HAND_L, R_HAND_R)[slot]] = pts.ravel()
                raw[R_HAND_P.start + slot] = 1.0
        fres = None
        if self.face is not None:
            fres = self.face.process(rgb)
            if fres.multi_face_landmarks:
                lm = fres.multi_face_landmarks[0].landmark
                raw[R_FACE] = np.array([(lm[i].x * aspect, lm[i].y, lm[i].z * aspect) for i in FACE_KEEP],
                                       np.float32).ravel()
                raw[R_FACE_P] = 1.0
        return raw, {"hands": hres, "pose": pres, "face": fres}


def assign_slots(hand_pts: list[np.ndarray], labels: list[int], wrists) -> list[int | None]:
    """Signer-left = 0, signer-right = 1. Prefer the nearest pose wrists; fall back to labels/x order."""
    if not hand_pts:
        return []
    if wrists is not None:
        cost = [[float(np.linalg.norm(p[0, :2] - w)) for w in wrists] for p in hand_pts]
        if len(hand_pts) == 1:
            return [int(np.argmin(cost[0]))]
        straight, swapped = cost[0][0] + cost[1][1], cost[0][1] + cost[1][0]
        return [0, 1] if straight <= swapped else [1, 0]
    if len(hand_pts) == 2 and labels[0] == labels[1]:
        # the signer's left hand appears on the image's right (larger x) on raw, unmirrored frames
        return [0, 1] if hand_pts[0][0, 0] > hand_pts[1][0, 0] else [1, 0]
    return list(labels[:2]) + [None] * (len(hand_pts) - 2)


def resample_raw(times: list[float], raws: list[np.ndarray], fps: float = 30.0) -> np.ndarray:
    """Timestamped raw frames -> uniform `fps` grid. Interpolates a body part only when it is
    present in both neighbouring frames; otherwise takes the nearest frame (no ghost blending)."""
    t = np.asarray(times, np.float64)
    v = np.stack(raws).astype(np.float32)
    grid = np.arange(t[0], t[-1] + 1e-9, 1 / fps)
    j = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    a = np.clip((grid - t[j]) / np.maximum(t[j + 1] - t[j], 1e-9), 0, 1)
    lo, hi = v[j], v[j + 1]
    near = np.where((a < 0.5)[:, None], lo, hi)
    out = near.copy()
    lerp = lo * (1 - a[:, None]) + hi * a[:, None]
    for sl, flag in ((R_HAND_L, R_HAND_P.start), (R_HAND_R, R_HAND_P.start + 1), (R_POSE, R_POSE_P),
                     (R_FACE, R_FACE_P)):
        both = (lo[:, flag] > 0.5) & (hi[:, flag] > 0.5)
        out[both, sl] = lerp[both, sl]
    return out
