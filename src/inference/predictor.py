"""Live JSL predictor: MediaPipe features -> rolling buffer -> ONNX -> smoothed decision."""
from __future__ import annotations

import base64
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.common.config import (FEATURE_DIM, IDLE_INDEX, MODEL_LEN, MODELS, NUM_SIGNS, WINDOW,
                               class_names, load_english)
from src.common.features import hands_present, make_hands, process_bgr, resample
from src.inference.glove_sim import GloveSource, SimulatedGlove

MIN_FILL = 48            # start predicting once the buffer holds this many frames
PREDICT_EVERY = 3        # run the model every Nth frame
EMA_ALPHA = 0.4
CONF_THRESHOLD = 0.60
STREAK_NEEDED = 5        # consecutive predictions with the same top-1
COOLDOWN_S = 1.5
DEMO_PERIOD_S = 2.5

_draw = mp.solutions.drawing_utils
_styles = mp.solutions.drawing_styles
_conns = mp.solutions.hands.HAND_CONNECTIONS


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    e = np.exp(x - x.max())
    return e / e.sum()


class JSLPredictor:
    """Stateful per-client predictor. Call `process_frame` for every incoming camera frame."""

    def __init__(self, model_path: Path | None = None, glove: GloveSource | None = None,
                 det_conf: float = 0.5, track_conf: float = 0.5) -> None:
        model_path = Path(model_path or MODELS / "jsl_model.onnx")
        self.names = class_names()
        self.english = load_english()
        self.session = None
        if model_path.exists():
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self.session = ort.InferenceSession(str(model_path), opts,
                                                providers=["CPUExecutionProvider"])
        self.mode = "model" if self.session else "demo"
        self.hands = make_hands(det_conf, track_conf)
        self.glove = glove or SimulatedGlove()
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear buffer and decision state (e.g. between replay videos)."""
        self.buffer: deque[np.ndarray] = deque(maxlen=WINDOW)
        self.frame_idx = 0
        self.ema: np.ndarray | None = None
        self.streak_cls, self.streak_n = -1, 0
        self.blocked_cls = -1
        self.last_emit_t = -1e9
        self.last_frame_t: float | None = None
        self.fps = 0.0
        self.demo_t0: float | None = None

    # ------------------------------------------------------------------ core
    def model_input(self) -> np.ndarray:
        """Front-pad the buffer with zeros to WINDOW frames and resample to (1, MODEL_LEN, 126)."""
        buf = np.zeros((WINDOW, FEATURE_DIM), dtype=np.float32)
        if self.buffer:
            buf[WINDOW - len(self.buffer):] = np.stack(self.buffer)
        return resample(buf, MODEL_LEN)[None]

    def _infer(self) -> np.ndarray:
        logits = self.session.run(["logits"], {"input": self.model_input()})[0][0]
        return softmax(logits)

    def _demo_probs(self, t: float) -> np.ndarray:
        if self.demo_t0 is None:
            self.demo_t0 = t
        k = int((t - self.demo_t0) // DEMO_PERIOD_S)
        phase = ((t - self.demo_t0) % DEMO_PERIOD_S) / DEMO_PERIOD_S
        p = np.full(len(self.names), 0.01)
        cls = k % NUM_SIGNS
        p[cls] = 0.45 + 0.45 * np.sin(np.pi * phase)
        p[(cls + 1) % NUM_SIGNS] = 0.12
        p[(cls + 7) % NUM_SIGNS] = 0.07
        p[IDLE_INDEX] = 0.3 * (1 - np.sin(np.pi * phase))
        return p / p.sum()

    def push_vector(self, vec: np.ndarray, t: float | None = None) -> str | None:
        """Add one 126-d frame vector; update probabilities/decision. Returns emitted sign or None."""
        t = time.monotonic() if t is None else t
        self.buffer.append(vec.astype(np.float32))
        self.frame_idx += 1
        if self.mode == "demo":
            probs = self._demo_probs(t)
        elif len(self.buffer) >= MIN_FILL and self.frame_idx % PREDICT_EVERY == 0:
            probs = self._infer()
        else:
            return None
        self.ema = probs if self.ema is None else EMA_ALPHA * probs + (1 - EMA_ALPHA) * self.ema
        return self._decide(t)

    def _decide(self, t: float) -> str | None:
        top = int(self.ema.argmax())
        conf = float(self.ema[top])
        if top == self.streak_cls:
            self.streak_n += 1
        else:
            self.streak_cls, self.streak_n = top, 1
        if top != self.blocked_cls:          # idle or a different sign re-arms the last one
            self.blocked_cls = -1
        if (top != IDLE_INDEX and conf >= CONF_THRESHOLD and self.streak_n >= STREAK_NEEDED
                and top != self.blocked_cls and t - self.last_emit_t >= COOLDOWN_S):
            self.blocked_cls, self.last_emit_t = top, t
            return self.names[top]
        return None

    # ------------------------------------------------------------------ payload
    def _entry(self, idx: int) -> dict:
        name = self.names[idx]
        return {"sign": name, "english": self.english.get(name, ""),
                "confidence": round(float(self.ema[idx]), 4)}

    def current(self) -> tuple[dict, list[dict]]:
        """Current smoothed prediction (idle -> sign None) and top-3 non-idle signs."""
        if self.ema is None:
            return {"sign": None, "english": None, "confidence": 0.0, "is_confident": False}, []
        top = int(self.ema.argmax())
        top3 = [self._entry(int(i)) for i in np.argsort(-self.ema[:NUM_SIGNS])[:3]]
        if top == IDLE_INDEX:
            pred = {"sign": None, "english": None,
                    "confidence": round(float(self.ema[top]), 4), "is_confident": False}
        else:
            pred = {**self._entry(top), "is_confident": float(self.ema[top]) >= CONF_THRESHOLD}
        return pred, top3

    def process_frame(self, frame_bgr: np.ndarray, t: float | None = None,
                      encode: bool = True) -> dict:
        """Full live step for one BGR frame -> UI payload."""
        t = time.monotonic() if t is None else t
        if self.last_frame_t is not None and t > self.last_frame_t:
            inst = 1.0 / (t - self.last_frame_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self.last_frame_t = t
        vec, results = process_bgr(frame_bgr, self.hands)
        emitted = self.push_vector(vec, t)
        pred, top3 = self.current()
        left, right = hands_present(vec)
        payload = {
            "prediction": pred, "top3": top3, "emitted": emitted,
            "left_hand": left, "right_hand": right,
            "buffer_fill": round(len(self.buffer) / WINDOW, 3),
            "fps_server": round(self.fps, 1), "mode": self.mode,
            "glove": self.glove.read(vec if (left or right) else None),
        }
        if encode:
            payload["frame"] = self.encode(frame_bgr, results)
        return payload

    @staticmethod
    def encode(frame_bgr: np.ndarray, results) -> str:
        """Draw hand skeletons, downscale to <=640 px wide, return base64 JPEG."""
        img = frame_bgr.copy()
        if results is not None and results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                _draw.draw_landmarks(img, lm, _conns, _styles.get_default_hand_landmarks_style(),
                                     _styles.get_default_hand_connections_style())
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
