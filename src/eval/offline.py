"""Offline evaluation: recorded data pushed frame by frame through the exact live decision path.

Evaluation sets
  ref_30fps   reference videos, 30 fps                    (training signer - optimistic)
  ref_15fps   reference videos as a 15 fps camera          (frame-rate robustness)
  ref_zoom18  reference videos as a close laptop webcam    (framing robustness, never trained on)
  user_takes  your newest take per sign, held out from training (HONEST: your own signing)

For each sign the prediction is the first sign the system emits (None if it emits nothing), exactly
like the live app. Results go to models/eval_report.json; progress to models/eval_status.json.
"""
from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable

import numpy as np

from src.common.config import MODELS, NATIVE_FPS, USER_DATA, load_label_map
from src.common.features import sequence_features
from src.common.landmarks import RAW_DIM
from src.common.views import webcam_view
from src.eval.metrics import classification_report
from src.inference.predictor import JSLPredictor

LEAD, TAIL = 30, 45
REPORT = MODELS / "eval_report.json"
STATUS = MODELS / "eval_status.json"
SETS = {
    "ref_30fps": {"title": "Reference videos · 30 fps", "fps": 30.0, "zoom": 1.0,
                  "note": "Same signer as training: optimistic"},
    "ref_15fps": {"title": "Reference videos · 15 fps camera", "fps": 15.0, "zoom": 1.0,
                  "note": "Frame-rate robustness"},
    "ref_zoom18": {"title": "Reference videos · close laptop webcam", "fps": 30.0, "zoom": 1.8,
                   "note": "Framing robustness; this crop is never used in training"},
    "user_takes": {"title": "Your held-out takes", "note": "Honest: your newest take of each sign, never trained on"},
}


def _status(**kw) -> None:
    MODELS.mkdir(exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def _first(emitted: list) -> tuple[str | None, float | None, float | None]:
    return (emitted[0][0], emitted[0][1], emitted[0][2]) if emitted else (None, None, None)


def replay_videos(fps: float, zoom: float, progress: Callable[[int, int, str], None] | None = None) -> list[dict]:
    """Reference videos through MediaPipe + predictor. Returns one row per sign."""
    from scripts.extract_landmarks import TMP, is_blue_frame, raw_videos, read_frames

    videos = raw_videos()
    label_map = load_label_map()
    pred = JSLPredictor()
    rows = []
    try:
        for k, sid in enumerate(sorted(label_map)):
            label = label_map[sid]
            if progress:
                progress(k, len(label_map), label)
            if label not in videos:
                continue
            frames = [f for f in read_frames(videos[label]) if is_blue_frame(f)]
            if zoom != 1.0:
                frames = [webcam_view(f, zoom) for f in frames]
            seq = [frames[0]] * LEAD + frames + [frames[-1]] * TAIL
            pred.reset()
            emitted = []
            keep = [i for i in range(len(seq)) if int(i * fps / NATIVE_FPS) != int((i - 1) * fps / NATIVE_FPS)]
            for i in keep:
                p = pred.process_frame(seq[i], t=i / NATIVE_FPS, encode=False, want_glove=False)
                if p["emitted"]:
                    emitted.append((p["emitted"], i / NATIVE_FPS, p["prediction"]["confidence"]))
            sign, t, conf = _first(emitted)
            end = (LEAD + len(frames)) / NATIVE_FPS
            rows.append({"target": label, "predicted": sign, "confidence": conf,
                         "delay_s": None if t is None else round(t - end, 2), "n_emitted": len(emitted)})
    finally:
        pred.close()
        shutil.rmtree(TMP, ignore_errors=True)
    return rows


def replay_takes(profile: str, progress: Callable[[int, int, str], None] | None = None) -> list[dict]:
    """Held-out user takes (newest per sign, only signs with >= 3 takes) through the predictor's
    decision path (features are already extracted, so MediaPipe is skipped)."""
    label_map = load_label_map()
    base = USER_DATA / profile
    items = []
    for sid, label in sorted(label_map.items()):
        files = sorted((base / sid).glob("*.npy")) if base.exists() else []
        if len(files) >= 3:
            raw = np.load(files[-1])
            if raw.ndim == 2 and raw.shape[1] == RAW_DIM:
                items.append((label, raw))
    if not items:
        return []
    pred = JSLPredictor()
    rows = []
    try:
        for k, (label, raw) in enumerate(items):
            if progress:
                progress(k, len(items), label)
            feats = sequence_features(raw)
            seq = np.concatenate([np.repeat(feats[:1], LEAD, 0), feats, np.repeat(feats[-1:], TAIL, 0)])
            pred.reset()
            emitted = []
            for i, v in enumerate(seq):
                e = pred.push_vector(v, t=i / NATIVE_FPS)
                if e:
                    emitted.append((e, i / NATIVE_FPS, float(pred.ema.max())))
            sign, t, conf = _first(emitted)
            end = (LEAD + len(feats)) / NATIVE_FPS
            rows.append({"target": label, "predicted": sign, "confidence": conf,
                         "delay_s": None if t is None else round(t - end, 2), "n_emitted": len(emitted)})
    finally:
        pred.close()
    return rows


def evaluate(profile: str = "default", sets: list[str] | None = None) -> dict:
    """Run the chosen evaluation sets and save models/eval_report.json."""
    sets = sets or list(SETS)
    onnx = MODELS / "jsl_model.onnx"
    report = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": profile,
              "model_trained": time.strftime("%Y-%m-%d %H:%M", time.localtime(onnx.stat().st_mtime)) if onnx.exists() else None,
              "sets": {}}
    for j, name in enumerate(sets):
        meta = SETS[name]

        def progress(k: int, total: int, label: str, j=j, name=name) -> None:
            _status(state="running", set=name, set_index=j, sets=len(sets), item=k, items=total, label=label)

        rows = replay_takes(profile, progress) if name == "user_takes" else \
            replay_videos(meta["fps"], meta["zoom"], progress)
        if not rows:
            report["sets"][name] = {**meta, "rows": [], "report": None,
                                    "skipped": "Record at least 3 takes of a sign to get held-out takes"}
            continue
        rep = classification_report([r["target"] for r in rows], [r["predicted"] for r in rows])
        delays = [r["delay_s"] for r in rows if r["delay_s"] is not None and r["predicted"] == r["target"]]
        rep["mean_delay_s"] = float(np.mean(delays)) if delays else None
        report["sets"][name] = {**meta, "rows": rows, "report": rep}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _status(state="done", finished=report["created"])
    return report
