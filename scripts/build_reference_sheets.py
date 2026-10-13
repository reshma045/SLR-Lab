"""Build the labelled reference keyframe sheets Claude compares live clips against.

    python scripts/build_reference_sheets.py     # needs the videos in data/raw/ and extracted landmarks

Writes data/reference_sheets/XX.jpg (6 keyframes of each sign's active part, with hand skeletons).
These contain the reference signer's image, so they are git-ignored like the videos.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from scripts.extract_landmarks import TMP, collect_videos, is_blue_frame, read_frames  # noqa: E402
from src.common.config import DATA, load_label_map  # noqa: E402
from src.common.features import make_hands, process_bgr  # noqa: E402
from src.inference.claude_verifier import REF_DIR, SHEET_FRAMES, make_sheet  # noqa: E402
from src.inference.predictor import annotate  # noqa: E402
from src.train.dataset import find_core  # noqa: E402


def crop_signer(img: np.ndarray) -> np.ndarray:
    """The signer stands centred in a wide 720x480 frame: keep the middle ~2/3 (about square)."""
    w = img.shape[1]
    return img[:, int(w * 0.17):int(w * 0.83)]


def main() -> None:
    zips = sorted((DATA / "raw").glob("*.zip"))
    videos = collect_videos(zips[0] if zips else None, None if zips else DATA / "raw")
    REF_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with make_hands(0.3, 0.3, static=True) as hands:
            for sid, label in sorted(load_label_map().items()):
                frames = [f for f in read_frames(videos[label]) if is_blue_frame(f)]
                on, off = find_core(np.load(DATA / "landmarks_full" / f"{sid}.npy"))
                span = frames[on + 4:max(on + 10, off - 4)]       # active part, without rest pose
                idx = np.linspace(0, len(span) - 1, SHEET_FRAMES).round().astype(int)
                picked = [crop_signer(annotate(span[i], process_bgr(span[i], hands)[1])) for i in idx]
                (REF_DIR / f"{sid}.jpg").write_bytes(make_sheet(picked))
                print(f"[{sid}] {label}: frames {on}-{off}")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print(f"Wrote {REF_DIR}")


if __name__ == "__main__":
    main()
