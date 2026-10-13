"""Extract MediaPipe hand landmarks from the JSL reference videos.

Usage:
    python scripts/extract_landmarks.py                 # auto-finds zip / mp4s in data/raw
    python scripts/extract_landmarks.py --zip path.zip
    python scripts/extract_landmarks.py --videos-dir path/to/mp4s

Outputs (relative to repo root):
    data/landmarks_full/XX.npy   (T, 126) float32, native fps, signing frames only
    data/landmarks/XX.npy        (60, 126) float32, whole sign resampled to 60 frames
    data/extraction_report.json  per-sign frame counts + hand detection rate

Each video = title card (pink) -> signer on blue chroma-key. Only blue-background
frames are kept. 126 = 2 hands x 21 keypoints x (x,y,z), wrist-relative.
Slots: Left hand (MediaPipe handedness) = 0..62, Right = 63..125. Frames are NOT mirrored.
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
from src.common.config import DATA, MODEL_LEN, load_label_map, nfc  # noqa: E402
from src.common.features import frame_landmarks, make_hands, resample  # noqa: E402

TMP = DATA / "_tmp"
TARGET_FRAMES = MODEL_LEN
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
    args = ap.parse_args()

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

    (DATA / "landmarks").mkdir(exist_ok=True)
    (DATA / "landmarks_full").mkdir(exist_ok=True)
    report = {}

    for label, path in sorted(videos.items(), key=lambda kv: label_to_id.get(kv[0], "99")):
        if label not in label_to_id:
            continue
        # fresh tracker per video so tracking state never leaks between videos
        with make_hands(det_conf=DET_CONF, track_conf=DET_CONF) as hands:
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
            seq = np.stack([frame_landmarks(f, hands) for f in signing])
            detected = int((np.abs(seq).sum(axis=1) > 0).sum())
            rate = detected / len(seq)
            np.save(DATA / "landmarks_full" / f"{sid}.npy", seq)
            np.save(DATA / "landmarks" / f"{sid}.npy", resample(seq, TARGET_FRAMES))
            report[sid] = {"label": label, "total_frames": len(frames),
                           "signing_frames": len(signing), "hand_detection_rate": round(rate, 3)}
            flag = "" if rate >= 0.8 else "  <-- LOW DETECTION"
            print(f"[{sid}] {label}: {len(signing)}/{len(frames)} signing frames, "
                  f"hands in {rate:.0%}{note}{flag}")

    json.dump(report, open(DATA / "extraction_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    shutil.rmtree(TMP, ignore_errors=True)
    print(f"\nDone: {len(report)}/{len(label_map)} signs extracted -> data/landmarks/, data/landmarks_full/")


if __name__ == "__main__":
    main()
