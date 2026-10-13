"""Live JSL predictor: MediaPipe hands + pose -> time-based buffer -> ONNX -> smoothed decision."""
from __future__ import annotations

import base64
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.common.config import (FEATURE_DIM, IDLE_INDEX, MODEL_LEN, MODELS, NATIVE_FPS, NUM_SIGNS,
                               WINDOW, class_names, load_english)
from src.common.features import FeatureExtractor, hands_present, pose_present
from src.inference.glove_sim import GloveSource, SimulatedGlove

# The buffer is time-based so any camera frame rate maps onto the 30 fps timeline used in training:
# the last WINDOW/30 s (3.2 s) are interpolated at MODEL_LEN evenly spaced instants.
SPAN_S = (WINDOW - 1) / NATIVE_FPS
MIN_FILL = 48            # start predicting once the buffer spans this many 30 fps frames (1.6 s)
PREDICT_EVERY = 3        # run the model every 3/30 s
DEMO_PERIOD_S = 2.5
RELOAD_CHECK_S = 2.0

_draw = mp.solutions.drawing_utils
_styles = mp.solutions.drawing_styles
_conns = mp.solutions.hands.HAND_CONNECTIONS
_POSE = mp.solutions.pose.PoseLandmark


@dataclass
class DecisionParams:
    """User-tunable decision settings (Settings page)."""
    conf_threshold: float = 0.60    # minimum smoothed probability to show/emit a sign
    streak: int = 5                 # consecutive predictions with the same top-1 before emitting
    cooldown_s: float = 1.5         # minimum time between emissions
    ema_alpha: float = 0.4          # smoothing: weight of the newest prediction


def annotate(frame_bgr: np.ndarray, hand_results, body: dict | None = None, aspect: float = 1.0) -> np.ndarray:
    """Copy of the frame with hand skeletons and (if known) the shoulder line drawn."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    if body is not None:
        c, half = body["center"], body["width"] / 2
        p1 = (int((c[0] - half) / aspect * w), int(c[1] * h))
        p2 = (int((c[0] + half) / aspect * w), int(c[1] * h))
        cv2.line(img, p1, p2, (246, 130, 59), 2)
        nose = body["nose"]
        cv2.circle(img, (int(nose[0] / aspect * w), int(nose[1] * h)), 4, (246, 130, 59), -1)
    if hand_results is not None and hand_results.multi_hand_landmarks:
        for lm in hand_results.multi_hand_landmarks:
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
                 det_conf: float = 0.3, track_conf: float = 0.3, pose_every: int = 2,
                 params: DecisionParams | None = None) -> None:
        self.model_path = Path(model_path or MODELS / "jsl_model.onnx")
        self.names = class_names()
        self.english = load_english()
        self.params = params or DecisionParams()
        self.session, self.model_mtime, self._next_check = None, 0.0, 0.0
        self._load_model()
        self.fx = FeatureExtractor(det_conf, track_conf, pose_every=pose_every)
        self.glove = glove or SimulatedGlove()
        self.last_vec = np.zeros(FEATURE_DIM, np.float32)
        self.last_annotated: np.ndarray | None = None
        self.reset()

    # ------------------------------------------------------------------ model
    def _load_model(self) -> None:
        if not self.model_path.exists():
            self.session, self.mode = None, "demo"
            return
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        session = ort.InferenceSession(str(self.model_path), opts, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape[-1] != FEATURE_DIM:      # stale model from an older layout
            self.session, self.mode = None, "demo"
            return
        self.session, self.mode = session, "model"
        self.model_mtime = self.model_path.stat().st_mtime

    def maybe_reload(self, now: float) -> bool:
        """Hot-reload the ONNX model if it was retrained. Returns True if reloaded."""
        if now < self._next_check:
            return False
        self._next_check = now + RELOAD_CHECK_S
        mtime = self.model_path.stat().st_mtime if self.model_path.exists() else 0.0
        if mtime and mtime != self.model_mtime:
            self._load_model()
            self.ema = None
            return True
        return False

    def set_params(self, **kw) -> None:
        """Update decision settings (unknown keys ignored)."""
        for k, v in kw.items():
            if hasattr(self.params, k) and v is not None:
                setattr(self.params, k, type(getattr(self.params, k))(v))

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """Clear buffer and decision state (e.g. between replay videos)."""
        self.buffer: deque[np.ndarray] = deque()          # per-frame feature vectors ...
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

    def span(self) -> float:
        """Seconds of history in the buffer."""
        return self.times[-1] - self.times[0] if self.times else 0.0

    # ------------------------------------------------------------------ core
    def model_input(self) -> np.ndarray:
        """Interpolate the last 3.2 s at MODEL_LEN instants -> (1, MODEL_LEN, D).

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
        return softmax(self.session.run(["logits"], {"input": self.model_input()})[0][0])

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
        """Add one frame vector; update probabilities/decision. Returns emitted sign or None."""
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
        a = self.params.ema_alpha
        self.ema = probs if self.ema is None else a * probs + (1 - a) * self.ema
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
        p = self.params
        if (top != IDLE_INDEX and conf >= p.conf_threshold and self.streak_n >= p.streak
                and top != self.blocked_cls and t - self.last_emit_t >= p.cooldown_s):
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
            pred = {**self._entry(top),
                    "is_confident": float(self.ema[top]) >= self.params.conf_threshold}
        return pred, top3

    def process_frame(self, frame_bgr: np.ndarray, t: float | None = None, encode: bool = True,
                      want_glove: bool = True) -> dict:
        """Full live step for one BGR frame -> UI payload."""
        t = time.monotonic() if t is None else t
        reloaded = self.maybe_reload(time.monotonic())
        if self.last_frame_t is not None and t > self.last_frame_t:
            inst = 1.0 / (t - self.last_frame_t)
            self.fps = inst if self.fps == 0 else 0.9 * self.fps + 0.1 * inst
        self.last_frame_t = t
        vec, hres, _ = self.fx.process(frame_bgr)
        self.last_vec = vec
        emitted = self.push_vector(vec, t)
        pred, top3 = self.current()
        left, right = hands_present(vec)
        payload = {
            "prediction": pred, "top3": top3, "emitted": emitted,
            "left_hand": left, "right_hand": right, "pose": pose_present(vec),
            "buffer_fill": round(min(1.0, self.span() / SPAN_S), 3),
            "fps_server": round(self.fps, 1), "mode": self.mode, "model_reloaded": reloaded,
            "params": asdict(self.params),
        }
        if want_glove:
            payload["glove"] = self.glove.read(None)
        h, w = frame_bgr.shape[:2]
        img = annotate(frame_bgr, hres, self.fx.body, w / h)
        self.last_annotated = img
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
        self.fx.close()
