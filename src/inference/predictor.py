"""Live JSL predictor: MediaPipe features -> rolling buffer -> ONNX -> smoothed decision."""
from __future__ import annotations

import base64
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.common.config import (FEATURE_DIM, IDLE_INDEX, MODEL_LEN, MODELS, NATIVE_FPS, NUM_SIGNS,
                               WINDOW, class_names, load_english)
from src.common.features import hands_present, make_hands, process_bgr
from src.inference.glove_sim import GloveSource, SimulatedGlove

# The buffer is time-based so any camera frame rate maps onto the 30 fps timeline used in training:
# the last WINDOW/30 s (3.2 s) are interpolated at MODEL_LEN evenly spaced instants.
SPAN_S = (WINDOW - 1) / NATIVE_FPS
MIN_FILL = 48            # start predicting once the buffer spans this many 30 fps frames (1.6 s)
PREDICT_EVERY = 3        # run the model every 3/30 s (every 3rd frame at 30 fps)
EMA_ALPHA = 0.4
CONF_THRESHOLD = 0.60
STREAK_NEEDED = 5        # consecutive predictions with the same top-1
COOLDOWN_S = 1.5
DEMO_PERIOD_S = 2.5

_draw = mp.solutions.drawing_utils
_styles = mp.solutions.drawing_styles
_conns = mp.solutions.hands.HAND_CONNECTIONS


def annotate(frame_bgr: np.ndarray, results) -> np.ndarray:
    """Copy of the frame with MediaPipe hand skeletons drawn."""
    img = frame_bgr.copy()
    if results is not None and results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            _draw.draw_landmarks(img, lm, _conns, _styles.get_default_hand_landmarks_style(),
                                 _styles.get_default_hand_connections_style())
    return img


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
        self.buffer: deque[np.ndarray] = deque()          # per-frame 126-d vectors ...
        self.times: deque[float] = deque()                  # ... and their timestamps (s)
        self.last_pred_t = -1e9
        self.clip: deque[tuple[np.ndarray, bool]] = deque(maxlen=WINDOW)  # (annotated 320 px frame, hands?)
        self.frame_idx = 0
        self.ema: np.ndarray | None = None
        self.streak_cls, self.streak_n = -1, 0
        self.blocked_cls = -1
        self.last_emit_t = -1e9
        self.last_frame_t: float | None = None
        self.fps = 0.0
        self.demo_t0: float | None = None

    # ------------------------------------------------------------------ core
    def span(self) -> float:
        """Seconds of history in the buffer."""
        return self.times[-1] - self.times[0] if self.times else 0.0

    def model_input(self) -> np.ndarray:
        """Interpolate the last 3.2 s at MODEL_LEN instants -> (1, MODEL_LEN, 126).

        Instants before the first buffered frame are zeros (front padding). At exactly 30 fps this
        equals "last 96 frames, zero-padded, linearly resampled to 60" used in training.
        """
        out = np.zeros((MODEL_LEN, FEATURE_DIM), dtype=np.float32)
        if not self.buffer:
            return out[None]
        ts = np.array([self.times[0] - 1 / NATIVE_FPS, *self.times])
        vs = np.vstack([np.zeros((1, FEATURE_DIM), np.float32), np.stack(self.buffer)])
        targets = ts[-1] - (WINDOW - 1 - np.linspace(0, WINDOW - 1, MODEL_LEN)) / NATIVE_FPS
        ok = targets >= ts[0]
        tv = targets[ok]
        j = np.clip(np.searchsorted(ts, tv, side="right") - 1, 0, len(ts) - 2)
        a = np.clip((tv - ts[j]) / np.maximum(ts[j + 1] - ts[j], 1e-9), 0, 1)[:, None]
        out[ok] = vs[j] * (1 - a) + vs[j + 1] * a
        return out[None]

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
        self.times.append(t)
        while self.times and t - self.times[0] > SPAN_S + 2 / NATIVE_FPS:
            self.times.popleft()
            self.buffer.popleft()
        self.frame_idx += 1
        if self.mode == "demo":
            probs = self._demo_probs(t)
        elif (self.span() >= (MIN_FILL - 1) / NATIVE_FPS - 1e-6
              and t - self.last_pred_t >= PREDICT_EVERY / NATIVE_FPS - 1e-6):
            self.last_pred_t = t
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
            "buffer_fill": round(min(1.0, self.span() / SPAN_S), 3),
            "fps_server": round(self.fps, 1), "mode": self.mode,
            "glove": self.glove.read(vec if (left or right) else None),
        }
        img = annotate(frame_bgr, results)
        self.last_annotated = img                       # for local display (OpenCV demo)
        h, w = img.shape[:2]
        self.clip.append((cv2.resize(img, (320, int(h * 320 / w)), interpolation=cv2.INTER_AREA), left or right))
        if encode:
            payload["frame"] = self.encode(img)
        return payload

    def clip_frames(self) -> list[np.ndarray]:
        """Buffered annotated frames (oldest first), trimmed to the span where hands are visible."""
        has = [h for _, h in self.clip]
        if not any(has):
            return [f for f, _ in self.clip]
        a = has.index(True)
        b = len(has) - has[::-1].index(True)
        return [f for f, _ in list(self.clip)[a:b]]

    @staticmethod
    def encode(img: np.ndarray) -> str:
        """Downscale an (annotated) frame to <=640 px wide and return base64 JPEG."""
        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()
