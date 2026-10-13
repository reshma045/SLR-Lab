"""Extract raw MediaPipe landmarks (hands + pose + face) from the JSL reference videos.

Usage:
    python scripts/extract_landmarks.py                 # auto-finds zip / mp4s in data/raw
    python scripts/extract_landmarks.py --zip path.zip
    python scripts/extract_landmarks.py --videos-dir path/to/mp4s
    python scripts/extract_landmarks.py --zooms 1.3,1.6,2.1   # + simulated close-webcam crops

Outputs (relative to repo root):
    data/landmarks_raw/XX.npy              (T, RAW_DIM) float32, native 30 fps, signing frames only
    data/landmarks_raw_aug/z<zoom>/XX.npy  same frames as a close laptop-webcam view (--zooms)
    data/extraction_report.json            per-sign frame counts + hand / pose / face detection rates

Each video = title card (pink) -> signer on blue chroma-key. Only blue-background frames are kept.
Raw layout: src/common/landmarks.py; features are computed from raw at training time.
Frames are NOT mirrored.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common.config import DATA, load_label_map, nfc  # noqa: E402
from src.common.landmarks import R_FACE_P, R_HAND_P, R_POSE_P, LandmarkExtractor  # noqa: E402
from src.common.views import webcam_view  # noqa: E402

TMP = DATA / "_tmp"
BLUE_THRESHOLD = 0.35
DET_CONF = 0.3  # tuned: 0.3 recovers overlapping-hand frames (ゴルフ 85->88%, ほとんど 93->97%)


def decode_zip_name(raw: str) -> str:
    """Handle '#U3042'-style escapes and mis-decoded zip names (cp437 -> utf-8/cp932)."""
    if "#U" in raw:
        raw = re.sub(r"#U([0-9a-fA-F]{4,6})", lambda m: chr(int(m.group(1), 16)), raw)
        return nfc(raw)
    try:
        raw.encode("ascii")
        return nfc(raw)
    except UnicodeEncodeError:
        pass
    for enc in ("utf-8", "cp932"):
        try:
            return nfc(raw.encode("cp437").decode(enc))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return nfc(raw)


def collect_videos(zip_path: Path | None, videos_dir: Path | None) -> dict[str, Path]:
    """Return {NFC label: ascii-named temp copy of the mp4}."""
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    found: dict[str, Path] = {}
    n = 0

    def add(label: str, src_bytes_reader) -> None:
        nonlocal n
        dst = TMP / f"v{n:03d}.mp4"
        with open(dst, "wb") as out:
            src_bytes_reader(out)
        found[label] = dst
        n += 1

    if zip_path:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                base = info.filename.replace("\\", "/").split("/")[-1]
                if info.is_dir() or "__MACOSX" in info.filename or base.startswith("."):
                    continue
                if not base.lower().endswith(".mp4"):
                    continue
                label = decode_zip_name(base)[:-4]
                add(label, lambda out, i=info: shutil.copyfileobj(zf.open(i), out))
    else:
        for p in sorted(videos_dir.glob("*.mp4")):
            label = nfc(p.stem)
            add(label, lambda out, p=p: shutil.copyfileobj(open(p, "rb"), out))
    return found


def is_blue_frame(frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([90, 80, 80]), np.array([130, 255, 255]))
    return mask.mean() / 255.0 > BLUE_THRESHOLD


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--videos-dir", type=Path)
    ap.add_argument("--zooms", default="", help="also extract webcam-like crops, e.g. 1.3,1.6,2.1 "
                    "-> data/landmarks_raw_aug/z<zoom>/XX.npy (extra training data)")
    args = ap.parse_args()
    zooms = [float(z) for z in args.zooms.split(",") if z.strip()]

    zip_path, vdir = args.zip, args.videos_dir
    if not zip_path and not vdir:
        zips = sorted((DATA / "raw").glob("*.zip"))
        if zips:
            zip_path = zips[0]
        elif list((DATA / "raw").glob("*.mp4")):
            vdir = DATA / "raw"
        else:
            sys.exit("No zip or mp4 files found in data/raw/ (use --zip or --videos-dir)")

    label_map = load_label_map()
    label_to_id = {v: k for k, v in label_map.items()}

    videos = collect_videos(zip_path, vdir)
    unknown = [l for l in videos if l not in label_to_id]
    missing = [l for l in label_to_id if l not in videos]
    if unknown:
        print(f"WARNING: videos not in label_map.json (skipped): {unknown}")
    if missing:
        print(f"WARNING: labels with no video: {missing}")

    out_dir = DATA / "landmarks_raw"
    out_dir.mkdir(exist_ok=True)
    report = {}

    def rates(raw: np.ndarray) -> tuple[float, float, float]:
        return (float((raw[:, R_HAND_P].sum(1) > 0).mean()), float(raw[:, R_POSE_P].mean()),
                float(raw[:, R_FACE_P].mean()))

    for label, path in sorted(videos.items(), key=lambda kv: label_to_id.get(kv[0], "99")):
        if label not in label_to_id:
            continue
        sid = label_to_id[label]
        frames = read_frames(path)
        if not frames:
            print(f"[{sid}] {label}: could not decode video (install ffmpeg / check codec)")
            continue
        signing = [f for f in frames if is_blue_frame(f)]
        note = ""
        if len(signing) < 30:
            signing = frames[len(frames) // 3:]
            note = " (blue-detect failed, used last 2/3 of video)"
        # fresh trackers per video so tracking state never leaks between videos
        with LandmarkExtractor(det_conf=DET_CONF, track_conf=DET_CONF) as lx:
            raw = np.stack([lx.process(f)[0] for f in signing])
        np.save(out_dir / f"{sid}.npy", raw)
        hand, pose, face = rates(raw)
        for z in zooms:                                    # same frames, simulated close-webcam framing
            zd = DATA / "landmarks_raw_aug" / f"z{z:g}"
            zd.mkdir(parents=True, exist_ok=True)
            with LandmarkExtractor(det_conf=DET_CONF, track_conf=DET_CONF) as lz:
                zraw = np.stack([lz.process(webcam_view(f, z))[0] for f in signing])
            np.save(zd / f"{sid}.npy", zraw)
            note += f" | z{z:g} hands {rates(zraw)[0]:.0%} face {rates(zraw)[2]:.0%}"
        report[sid] = {"label": label, "total_frames": len(frames), "signing_frames": len(signing),
                       "hand_detection_rate": round(hand, 3), "pose_detection_rate": round(pose, 3),
                       "face_detection_rate": round(face, 3)}
        flag = "" if hand >= 0.8 else "  <-- LOW DETECTION"
        print(f"[{sid}] {label}: {len(signing)}/{len(frames)} frames, hands {hand:.0%}, pose {pose:.0%}, "
              f"face {face:.0%}{note}{flag}", flush=True)

    json.dump(report, open(DATA / "extraction_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nDone: {len(report)}/{len(label_map)} signs extracted -> {out_dir}")


if __name__ == "__main__":
    main()
