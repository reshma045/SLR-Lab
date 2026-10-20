"""Import extra sign videos (e.g. KoSign, JSL-DC, more signers) as additional training examples.

    python scripts/import_videos.py --source kosign --dir D:/KoSign/front_mp4 --map kosign_map.csv
    python scripts/import_videos.py --source signer2 --dir videos/            # file name = sign name
    python scripts/import_videos.py --source kosign --dir ... --map ... --add-new   # also add new words

The CSV map has a header and columns `file,sign[,english]` (file name relative to --dir, sign in
Japanese exactly as in data/label_map.json). Without --map, each file's name (without extension) is
the sign. Videos may have any resolution / frame rate: frames are downscaled to <= 1280 px wide and
landmarks are resampled to 30 fps. Output: data/landmarks_raw_extra/<source>/<sign id>/<file>.npy,
which training picks up automatically (run python run_training.py afterwards).

Use only data you are licensed to use for training (KoSign: research use under the NII IDR terms).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.common.config import DATA, NATIVE_FPS, nfc  # noqa: E402
from src.common.landmarks import R_FACE_P, R_HAND_P, R_POSE_P, LandmarkExtractor, resample_raw  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm"}
EXTRA = DATA / "landmarks_raw_extra"


def read_mapping(map_path: Path | None, vdir: Path) -> list[tuple[Path, str, str]]:
    """[(video path, sign, english)] from the CSV map or from file names."""
    if map_path:
        rows = []
        with open(map_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                rows.append((vdir / r["file"], nfc(r["sign"].strip()), (r.get("english") or "").strip()))
        return rows
    return [(p, nfc(p.stem), "") for p in sorted(vdir.rglob("*")) if p.suffix.lower() in VIDEO_EXT]


def extract(path: Path) -> np.ndarray | None:
    """Raw landmarks of a whole video at 30 fps (None if unreadable). Uses an ASCII temp copy so
    Japanese file names work with OpenCV on Windows."""
    tmp = DATA / "_tmp_import" / f"clip{path.suffix.lower()}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(path.read_bytes())
    cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS) or NATIVE_FPS
    times, raws = [], []
    with LandmarkExtractor() as lx:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > 1280:
                frame = cv2.resize(frame, (1280, int(h * 1280 / w)), interpolation=cv2.INTER_AREA)
            raws.append(lx.process(frame)[0])
            times.append(i / fps)
            i += 1
    cap.release()
    tmp.unlink(missing_ok=True)
    return resample_raw(times, raws, NATIVE_FPS) if len(raws) >= 10 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="name for this dataset/signer, e.g. kosign")
    ap.add_argument("--dir", required=True, type=Path, help="folder with the videos")
    ap.add_argument("--map", type=Path, help="CSV with columns file,sign[,english]")
    ap.add_argument("--add-new", action="store_true", help="add signs not yet in data/label_map.json")
    args = ap.parse_args()

    label_map = json.loads((DATA / "label_map.json").read_text(encoding="utf-8"))
    english = json.loads((DATA / "signs_english.json").read_text(encoding="utf-8"))
    sign_to_id = {nfc(v): k for k, v in label_map.items()}
    rows = read_mapping(args.map, args.dir)
    added, imported, skipped = [], 0, []
    for path, sign, eng in rows:
        if sign not in sign_to_id:
            if not args.add_new:
                skipped.append(sign)
                continue
            sid = f"{len(label_map):02d}"
            label_map[sid], sign_to_id[sign] = sign, sid
            english.setdefault(sign, eng or sign)
            added.append(sign)
        sid = sign_to_id[sign]
        raw = extract(path)
        if raw is None:
            print(f"  could not read {path.name}")
            continue
        out = EXTRA / args.source / sid
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{path.stem}.npy", raw)
        imported += 1
        hand = float((raw[:, R_HAND_P].sum(1) > 0).mean())
        print(f"[{sid}] {sign}: {len(raw)} frames, hands {hand:.0%}, pose {raw[:, R_POSE_P].mean():.0%}, "
              f"face {raw[:, R_FACE_P].mean():.0%}", flush=True)
    if added:
        (DATA / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
        (DATA / "signs_english.json").write_text(json.dumps(english, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Added {len(added)} new signs to the vocabulary (now {len(label_map)}).")
    if skipped:
        print(f"Skipped {len(skipped)} videos of signs not in the vocabulary (use --add-new): "
              f"{sorted(set(skipped))[:20]}")
    print(f"Imported {imported} videos into {EXTRA / args.source}. Now run: python run_training.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
