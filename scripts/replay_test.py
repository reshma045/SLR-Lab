"""Replay test: push each reference video through JSLPredictor frame by frame (the live path).

For each sign: reset the predictor, feed LEAD copies of the first signing frame (the user was
resting before signing), then every signing frame, then TAIL copies of the last frame (the user
rests afterwards), at 30 fps timestamps. MediaPipe, the 96-frame buffer, EMA and the decision
logic all run exactly as in the server.

    python scripts/replay_test.py            # uses the zip / mp4s in data/raw/
    python scripts/replay_test.py --fps 15   # simulate a slower camera/browser (drops frames)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scripts.extract_landmarks import TMP, collect_videos, is_blue_frame, read_frames  # noqa: E402
from src.common.config import DATA, MODELS, NATIVE_FPS, load_label_map  # noqa: E402
from src.inference.predictor import JSLPredictor  # noqa: E402

LEAD, TAIL = 30, 45
TARGET = 22


def replay(fps: float = NATIVE_FPS) -> dict:
    """Run the replay over all 25 videos; return a report dict."""
    import shutil

    zips = sorted((DATA / "raw").glob("*.zip"))
    videos = collect_videos(zips[0] if zips else None, None if zips else DATA / "raw")
    label_map = load_label_map()
    pred = JSLPredictor()
    if pred.mode != "model":
        sys.exit("models/jsl_model.onnx not found - run python run_training.py first")
    rows, strict, lenient = [], 0, 0
    try:
        for sid in sorted(label_map):
            label = label_map[sid]
            frames = [f for f in read_frames(videos[label]) if is_blue_frame(f)]
            seq = [frames[0]] * LEAD + frames + [frames[-1]] * TAIL
            pred.reset()
            emitted = []
            keep = [i for i in range(len(seq)) if int(i * fps / NATIVE_FPS) != int((i - 1) * fps / NATIVE_FPS)]
            for i in keep:                     # frames that a `fps` camera would have delivered
                f = seq[i]
                p = pred.process_frame(f, t=i / NATIVE_FPS, encode=False)
                if p["emitted"]:
                    emitted.append((p["emitted"], round(i / NATIVE_FPS, 2), p["prediction"]["confidence"]))
            names = [e[0] for e in emitted]
            ok_strict = names == [label]
            ok_lenient = bool(names) and names[0] == label
            strict += ok_strict
            lenient += ok_lenient
            rows.append({"id": sid, "label": label, "emitted": emitted,
                         "correct": ok_lenient, "only_correct": ok_strict})
            mark = "OK " if ok_lenient else "MISS"
            extra = "" if ok_strict or not ok_lenient else "  (+ later extra emissions)"
            print(f"[{mark}] {sid} {label:<8} -> {names or '(nothing emitted)'}{extra}", flush=True)
    finally:
        pred.close()
        shutil.rmtree(TMP, ignore_errors=True)
    report = {"fps": fps, "correct_first_emission": lenient, "only_correct_emission": strict,
              "total": len(rows), "target": TARGET, "rows": rows}
    name = "replay_report.json" if fps == NATIVE_FPS else f"replay_report_{fps:g}fps.json"
    with open(MODELS / name, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReplay @ {fps:g} fps: {lenient}/{len(rows)} signs correctly emitted first "
          f"({strict}/{len(rows)} with no other emission). Target >= {TARGET}.")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=NATIVE_FPS)
    r = replay(ap.parse_args().fps)
    sys.exit(0 if r["correct_first_emission"] >= TARGET else 1)
