"""Project-wide paths and constants shared by extraction, training and inference."""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
FRONTEND = ROOT / "frontend"

USER_DATA = DATA / "user"           # the user's own recordings (git-ignored)
SETTINGS_FILE = DATA / "settings.json"

# Feature layout v2 (see src/common/features.py):
HAND_DIM = 63              # one hand's shape: 21 keypoints x (x, y, z), wrist-relative / palm length
SHAPE_L = slice(0, 63)     # signer's LEFT hand shape
SHAPE_R = slice(63, 126)   # signer's RIGHT hand shape
POS_L = slice(126, 129)    # left wrist (x, y) and palm size, relative to shoulders / shoulder width
POS_R = slice(129, 132)    # right wrist ...
NOSE = slice(132, 134)     # nose (x, y) relative to shoulders / shoulder width
PRESENT = slice(134, 136)  # 1.0 if left / right hand detected
FEATURE_DIM = 136
WINDOW = 96                # live rolling buffer length (~3.2 s at 30 fps)
MODEL_LEN = 60             # frames fed to the model after resampling
NATIVE_FPS = 30.0
NUM_SIGNS = 25
IDLE_INDEX = 25            # extra "no sign" class
NUM_CLASSES = NUM_SIGNS + 1
IDLE_LABEL = "—"


def nfc(s: str) -> str:
    """Unicode NFC normalisation (Japanese file names come in NFD from macOS zips)."""
    return unicodedata.normalize("NFC", s)


def load_label_map() -> dict[str, str]:
    """Return {"00": "〜から", ...} from data/label_map.json (NFC-normalised)."""
    with open(DATA / "label_map.json", encoding="utf-8") as f:
        return {k: nfc(v) for k, v in json.load(f).items()}


def load_english() -> dict[str, str]:
    """Return {Japanese sign: English gloss}."""
    with open(DATA / "signs_english.json", encoding="utf-8") as f:
        return {nfc(k): v for k, v in json.load(f).items()}


def class_names() -> list[str]:
    """Class index -> display name; index 25 is the idle class."""
    lm = load_label_map()
    return [lm[f"{i:02d}"] for i in range(NUM_SIGNS)] + [IDLE_LABEL]
