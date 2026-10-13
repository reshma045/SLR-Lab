"""The user's own sign recordings: data/user/<profile>/<sign id>/<timestamp>.npy (+ .json meta).

Takes store RAW landmarks (hands + pose + face, src/common/landmarks.py), resampled onto the 30 fps
timeline used in training, so features can change in future versions without re-recording.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from src.common.config import NATIVE_FPS, NUM_SIGNS, USER_DATA
from src.common.landmarks import R_FACE_P, R_HAND_P, R_POSE_P, resample_raw

MIN_HAND_RATE = 0.25       # a take must show hands in at least this share of frames
MIN_POSE_RATE = 0.5        # ... and shoulders in at least this share


def _profile_dir(profile: str) -> Path:
    return USER_DATA / (re.sub(r"[^A-Za-z0-9_-]", "", profile) or "default")


def quality(raw: np.ndarray) -> dict:
    """Share of frames with hands / shoulders / face and an ok flag with a human-readable reason."""
    hands = float((raw[:, R_HAND_P].sum(1) > 0).mean())
    pose = float(raw[:, R_POSE_P].mean())
    face = float(raw[:, R_FACE_P].mean())
    reason = ""
    if hands < MIN_HAND_RATE:
        reason = "Hands were not visible enough - keep both hands in the frame while signing."
    elif pose < MIN_POSE_RATE:
        reason = "Shoulders not visible - sit back so your head and shoulders are in view."
    return {"hand_rate": round(hands, 3), "pose_rate": round(pose, 3), "face_rate": round(face, 3),
            "ok": not reason, "reason": reason}


def save(profile: str, sign_id: int, times: list[float], raws: list[np.ndarray]) -> dict:
    """Validate and store one take of raw frames. Returns {saved, name, path, quality, frames, count}."""
    if not 0 <= sign_id < NUM_SIGNS or len(times) < 10:
        return {"saved": False, "quality": {"ok": False, "reason": "Too few frames captured."}}
    raw = resample_raw(times, raws, NATIVE_FPS)
    q = quality(raw)
    if not q["ok"]:
        return {"saved": False, "quality": q, "frames": len(raw)}
    d = _profile_dir(profile) / f"{sign_id:02d}"
    d.mkdir(parents=True, exist_ok=True)
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    np.save(d / f"{name}.npy", raw)
    meta = {"sign_id": sign_id, "frames": len(raw), "duration_s": round(len(raw) / NATIVE_FPS, 2),
            "captured_fps": round(len(times) / max(times[-1] - times[0], 1e-6), 1), "format": "raw-v3", **q,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (d / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"saved": True, "name": name, "path": str(d / f"{name}.npy"), "quality": q, "frames": len(raw),
            "count": count(profile, sign_id)}


def count(profile: str, sign_id: int) -> int:
    d = _profile_dir(profile) / f"{sign_id:02d}"
    return len(list(d.glob("*.npy"))) if d.exists() else 0


def counts(profile: str) -> list[int]:
    return [count(profile, i) for i in range(NUM_SIGNS)]


def listing(profile: str, sign_id: int) -> list[dict]:
    d = _profile_dir(profile) / f"{sign_id:02d}"
    out = []
    for f in sorted(d.glob("*.npy")) if d.exists() else []:
        meta = {}
        if f.with_suffix(".json").exists():
            meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        out.append({"name": f.stem, **meta})
    return out


def delete(profile: str, sign_id: int, name: str) -> bool:
    if not re.fullmatch(r"[0-9-]+", name):
        return False
    d = _profile_dir(profile) / f"{sign_id:02d}"
    removed = False
    for ext in (".npy", ".json"):
        p = d / f"{name}{ext}"
        if p.exists():
            p.unlink()
            removed = True
    return removed


def delete_all(profile: str) -> None:
    shutil.rmtree(_profile_dir(profile), ignore_errors=True)


def profiles() -> list[str]:
    return sorted(p.name for p in USER_DATA.iterdir() if p.is_dir()) if USER_DATA.exists() else []
