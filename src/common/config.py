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

# Per-frame feature layout v3 (computed from raw landmarks, see src/common/features.py):
HAND_DIM = 63              # one hand's shape: 21 keypoints x (x, y, z), wrist-relative / palm length
SHAPE_L = slice(0, 63)     # signer's LEFT hand shape
SHAPE_R = slice(63, 126)   # signer's RIGHT hand shape
POS_L = slice(126, 129)    # left wrist (x, y) rel. shoulder centre / shoulder width, + palm / shoulder width
POS_R = slice(129, 132)    # right wrist ...
ELBOWS = slice(132, 136)   # left elbow (x, y), right elbow (x, y) rel. shoulders
NOSE = slice(136, 138)     # nose (x, y) rel. shoulders
LIPS = slice(138, 226)     # 4 lip contours x 11 points x (x, y), rel. mouth centre / mouth width
MOUTH = slice(226, 228)    # mouth opening / mouth width, mouth width / face width
EYES = slice(228, 230)     # left, right eye opening / face width
BROWS = slice(230, 232)    # left, right eyebrow height above the eye / face width
ROLL = slice(232, 234)     # unit vector right eye -> left eye (head tilt)
TIPS = slice(234, 242)     # L thumb, L index, R thumb, R index tip (x, y) rel. nose / face width
PRESENT = slice(242, 246)  # left hand, right hand, face, body (shoulders) detected
FEATURE_DIM = 246
MODEL_IN = FEATURE_DIM * 3  # + velocity over 1 and 2 steps (added on the 60-step model input)
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
