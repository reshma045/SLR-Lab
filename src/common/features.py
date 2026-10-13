"""Per-frame features (layout v3) computed from raw landmarks - shared by training and live inference.

Hands:  scale-free shapes (wrist-relative / palm length), wrist + elbow positions relative to the
        shoulders (in shoulder widths), fingertip positions relative to the nose (contact with face).
Face:   4 lip contours normalised to the mouth (mouthing / lip movement), mouth opening, eye
        opening, eyebrow raise, head tilt.
Motion: velocity over 1 and 2 steps is appended on the 60-step model input (`add_motion`).

Mirroring (left-handed signer augmentation) is a fixed index permutation + sign flip
(`MIRROR_PERM`, `MIRROR_SIGN`) so it is exact and testable.
"""
from __future__ import annotations

import numpy as np

from src.common.config import (BROWS, ELBOWS, EYES, FEATURE_DIM, LIPS, MOUTH, NOSE, POS_L, POS_R,
                               PRESENT, ROLL, SHAPE_L, SHAPE_R, TIPS)
from src.common.landmarks import (FACE_KEEP, FACE_POS, LIP_CONTOURS, P_LEL, P_LSH, P_NOSE, P_REL,
                                  P_RSH, R_FACE, R_FACE_P, R_HAND_L, R_HAND_P, R_HAND_R, R_POSE,
                                  R_POSE_P)

MIDDLE_MCP = 9
LIP_IDX = [FACE_POS[i] for c in LIP_CONTOURS for i in c]          # 44 points


def _norm(v: np.ndarray) -> np.ndarray:
    return np.linalg.norm(v, axis=-1)


def sequence_features(raw: np.ndarray) -> np.ndarray:
    """(T, RAW_DIM) raw landmarks -> (T, FEATURE_DIM) features (vectorised over frames)."""
    raw = np.atleast_2d(raw).astype(np.float32)
    T = len(raw)
    out = np.zeros((T, FEATURE_DIM), np.float32)
    hand_ok = raw[:, R_HAND_P] > 0.5
    pose = raw[:, R_POSE].reshape(T, 33, 4)
    ls, rs = pose[:, P_LSH, :2], pose[:, P_RSH, :2]
    centre, sw = (ls + rs) / 2, _norm(ls - rs)
    body_ok = (raw[:, R_POSE_P] > 0.5) & (sw > 1e-3) & (pose[:, P_LSH, 3] > 0.3) & (pose[:, P_RSH, 3] > 0.3)
    sw_safe = np.where(body_ok, sw, 1.0)[:, None]
    face = raw[:, R_FACE].reshape(T, len(FACE_KEEP), 3)[..., :2]
    F = lambda i: face[:, FACE_POS[i]]                               # noqa: E731
    face_w = _norm(F(234) - F(454))
    face_ok = (raw[:, R_FACE_P] > 0.5) & (face_w > 1e-3)
    fw = np.where(face_ok, face_w, 1.0)[:, None]
    nose_face = F(1)

    tips = []
    for k, (rsl, shape_sl, pos_sl) in enumerate(((R_HAND_L, SHAPE_L, POS_L), (R_HAND_R, SHAPE_R, POS_R))):
        pts = raw[:, rsl].reshape(T, 21, 3)
        rel = pts - pts[:, :1]
        palm = np.maximum(_norm(rel[:, MIDDLE_MCP, :2]), 1e-3)
        ok = hand_ok[:, k]
        out[:, shape_sl] = np.where(ok[:, None], (rel / palm[:, None, None]).reshape(T, 63), 0)
        pos = np.concatenate([(pts[:, 0, :2] - centre) / sw_safe, (palm / sw_safe[:, 0])[:, None]], 1)
        out[:, pos_sl] = np.where((ok & body_ok)[:, None], pos, 0)
        t = np.concatenate([(pts[:, 4, :2] - nose_face) / fw, (pts[:, 8, :2] - nose_face) / fw], 1)
        tips.append(np.where((ok & face_ok)[:, None], t, 0))
    out[:, TIPS] = np.concatenate(tips, 1)

    elb = np.concatenate([(pose[:, P_LEL, :2] - centre) / sw_safe, (pose[:, P_REL, :2] - centre) / sw_safe], 1)
    out[:, ELBOWS] = np.where(body_ok[:, None], elb, 0)
    out[:, NOSE] = np.where(body_ok[:, None], (pose[:, P_NOSE, :2] - centre) / sw_safe, 0)

    mc = (F(61) + F(291)) / 2
    mw = np.maximum(_norm(F(61) - F(291)), 1e-4)
    lips = (face[:, LIP_IDX] - mc[:, None]) / mw[:, None, None]
    face_feats = [
        (LIPS, lips.reshape(T, -1)),
        (MOUTH, np.stack([_norm(F(13) - F(14)) / mw, mw / fw[:, 0]], 1)),
        (EYES, np.stack([_norm(F(386) - F(374)), _norm(F(159) - F(145))], 1) / fw),
        (BROWS, np.stack([_norm(F(334) - F(386)), _norm(F(105) - F(159))], 1) / fw),
        (ROLL, (F(263) - F(33)) / np.maximum(_norm(F(263) - F(33)), 1e-4)[:, None]),
    ]
    for sl, val in face_feats:
        out[:, sl] = np.where(face_ok[:, None], val, 0)
    out[:, PRESENT] = np.stack([hand_ok[:, 0], hand_ok[:, 1], face_ok, body_ok], 1).astype(np.float32)
    return out


def frame_features(raw: np.ndarray) -> np.ndarray:
    """One raw frame -> one feature vector."""
    return sequence_features(raw[None])[0]


def add_motion(x: np.ndarray) -> np.ndarray:
    """(..., T, F) -> (..., T, 3F): features, x[t+1]-x[t], x[t+2]-x[t] (zero-padded at the end)."""
    d1 = np.zeros_like(x)
    d2 = np.zeros_like(x)
    d1[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
    d2[..., :-2, :] = x[..., 2:, :] - x[..., :-2, :]
    return np.concatenate([x, d1, d2], axis=-1).astype(np.float32)


def hands_present(vec: np.ndarray) -> tuple[bool, bool]:
    return bool(vec[PRESENT.start] > 0.5), bool(vec[PRESENT.start + 1] > 0.5)


def face_present(vec: np.ndarray) -> bool:
    return bool(vec[PRESENT.start + 2] > 0.5)


def pose_present(vec: np.ndarray) -> bool:
    return bool(vec[PRESENT.start + 3] > 0.5)


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


# ---------------------------------------------------------------- geometry tables for augmentation
def _r(sl: slice) -> np.ndarray:
    return np.arange(sl.start, sl.stop)


def _build_mirror() -> tuple[np.ndarray, np.ndarray]:
    perm = np.arange(FEATURE_DIM)
    sign = np.ones(FEATURE_DIM, np.float32)
    perm[_r(SHAPE_L)], perm[_r(SHAPE_R)] = _r(SHAPE_R), _r(SHAPE_L)
    perm[_r(POS_L)], perm[_r(POS_R)] = _r(POS_R), _r(POS_L)
    e = _r(ELBOWS)
    perm[e] = e[[2, 3, 0, 1]]
    for sl in (EYES, BROWS):                                        # (left, right) -> (right, left)
        r = _r(sl)
        perm[r] = r[::-1]
    t = _r(TIPS)
    perm[t] = t[[4, 5, 6, 7, 0, 1, 2, 3]]
    p = _r(PRESENT)
    perm[p] = p[[1, 0, 2, 3]]
    lip = _r(LIPS).reshape(4, 11, 2)
    perm[lip.ravel()] = lip[:, ::-1, :].ravel()                     # point i <-> 10 - i in each contour
    # x coordinates flip sign (applied to the destination index)
    xs = np.r_[_r(SHAPE_L)[::3], _r(SHAPE_R)[::3], POS_L.start, POS_R.start, e[[0, 2]], NOSE.start,
               lip[..., 0].ravel(), t[::2]]
    sign[xs] = -1
    sign[ROLL.start + 1] = -1                                       # tilt vector: (dx, dy) -> (dx, -dy)
    return perm, sign


MIRROR_PERM, MIRROR_SIGN = _build_mirror()
# (x, y) index pairs that rotate with a camera roll / body positions that scale & shift
_shape_x = np.r_[_r(SHAPE_L)[::3], _r(SHAPE_R)[::3]]
ROT_X = np.r_[_shape_x, POS_L.start, POS_R.start, _r(ELBOWS)[::2], NOSE.start, _r(LIPS)[::2],
              ROLL.start, _r(TIPS)[::2]]
ROT_Y = ROT_X + 1
BODY_X = np.r_[POS_L.start, POS_R.start, _r(ELBOWS)[::2], NOSE.start]
BODY_Y = BODY_X + 1
# which presence flags each block needs: (indices, [flag offsets that must all be 1])
BLOCKS = [(_r(SHAPE_L), [0]), (_r(SHAPE_R), [1]), (_r(POS_L), [0, 3]), (_r(POS_R), [1, 3]),
          (_r(ELBOWS), [3]), (_r(NOSE), [3]), (_r(LIPS), [2]), (_r(MOUTH), [2]), (_r(EYES), [2]),
          (_r(BROWS), [2]), (_r(ROLL), [2]), (_r(TIPS)[:4], [0, 2]), (_r(TIPS)[4:], [1, 2])]
