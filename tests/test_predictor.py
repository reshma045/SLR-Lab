"""JSLPredictor payloads and decision logic."""
import shutil
import zipfile

import cv2
import numpy as np
import pytest

from src.common.config import DATA, FEATURE_DIM, IDLE_INDEX, NUM_CLASSES
from src.inference import predictor as P
from src.inference.predictor import JSLPredictor

KEYS = {"frame", "prediction", "top3", "emitted", "left_hand", "right_hand", "buffer_fill",
        "fps_server", "mode", "glove"}


def _check_payload(p: dict) -> None:
    assert KEYS <= set(p)
    assert set(p["prediction"]) >= {"sign", "english", "confidence", "is_confident"}
    assert isinstance(p["left_hand"], bool) and isinstance(p["right_hand"], bool)
    assert 0 <= p["buffer_fill"] <= 1 and p["mode"] in ("model", "demo")
    assert isinstance(p["frame"], str) and len(p["frame"]) > 100
    for side in ("left", "right"):
        assert len(p["glove"][side]["pressure"]) == 50 and len(p["glove"][side]["imu"]) == 6
    assert p["glove"]["simulated"] is True
    for t in p["top3"]:
        assert {"sign", "english", "confidence"} <= set(t)


def test_black_frame_payload():
    pred = JSLPredictor()
    try:
        black = np.zeros((480, 640, 3), np.uint8)
        for i in range(60):
            p = pred.process_frame(black, t=i / 30)
        _check_payload(p)
        assert not p["left_hand"] and not p["right_hand"]
        if pred.mode == "model":
            assert p["prediction"]["sign"] is None          # no hands -> idle, nothing shown
            assert p["emitted"] is None
    finally:
        pred.close()


def _reference_frame(tmp_path):
    zips = sorted((DATA / "raw").glob("*.zip"))
    if not zips:
        pytest.skip("reference video zip not in data/raw/")
    with zipfile.ZipFile(zips[0]) as zf:
        info = next(i for i in zf.infolist() if i.filename.lower().endswith(".mp4"))
        dst = tmp_path / "ref.mp4"
        with zf.open(info) as src, open(dst, "wb") as out:
            shutil.copyfileobj(src, out)
    cap = cv2.VideoCapture(str(dst))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


def test_reference_frame_payload(tmp_path):
    frame = _reference_frame(tmp_path)
    pred = JSLPredictor()
    try:
        p = pred.process_frame(frame, t=0.0)
        _check_payload(p)
        assert p["left_hand"] or p["right_hand"]
    finally:
        pred.close()


class _FakeSession:
    def __init__(self):
        self.cls = IDLE_INDEX

    def run(self, _names, _feeds):
        logits = np.zeros((1, NUM_CLASSES), np.float32)
        logits[0, self.cls] = 8.0
        emb = np.zeros((1, 8), np.float32)
        emb[0, 0] = 1.0
        return [logits, emb]


def test_decision_logic_streak_cooldown_rearm():
    pred = JSLPredictor()
    pred.session, pred.mode = _FakeSession(), "model"
    vec = np.ones(FEATURE_DIM, np.float32)
    t = [0.0]

    def feed(n):
        out = []
        for _ in range(n):
            t[0] += 1 / 30
            e = pred.push_vector(vec, t[0])
            if e:
                out.append(e)
        return out

    try:
        pred.session.cls = 3
        assert feed(P.MIN_FILL - 1) == []                   # not enough buffer yet
        first = feed(P.PREDICT_EVERY * pred.params.streak + 3)
        assert first == [pred.names[3]]                     # emitted once after the streak
        assert feed(90) == []                               # held sign never re-emits
        pred.session.cls = IDLE_INDEX
        assert feed(30) == []
        pred.session.cls = 3
        assert feed(60) == [pred.names[3]]                  # idle re-armed it
    finally:
        pred.close()


def test_time_based_buffer_matches_frame_based_at_30fps():
    from src.common.config import MODEL_LEN, WINDOW
    from src.common.features import resample
    pred = JSLPredictor()
    try:
        v = np.random.default_rng(0).normal(size=(130, FEATURE_DIM)).astype(np.float32)
        for n in (20, 96, 130):
            pred.reset()
            for i in range(n):
                pred.push_vector(v[i], t=i / 30)
            buf = np.zeros((WINDOW, FEATURE_DIM), np.float32)
            last = v[max(0, n - WINDOW):n]
            buf[WINDOW - len(last):] = last
            assert np.allclose(pred.window_features(), resample(buf, MODEL_LEN), atol=1e-6)
    finally:
        pred.close()


def test_low_fps_covers_same_time_span():
    """At 10 fps the model input must still describe the last 3.2 s, not the last 9.6 s."""
    pred = JSLPredictor()
    try:
        for i in range(0, 300, 3):                      # 10 fps for 10 s; value = timestamp
            pred.push_vector(np.full(FEATURE_DIM, i / 30, np.float32), t=i / 30)
        x = pred.window_features()[:, 0]
        assert abs(x[-1] - 297 / 30) < 1e-5 and abs(x[0] - (297 - 95) / 30) < 1e-5
        assert pred.span() <= 96 / 30 + 0.2
    finally:
        pred.close()


def test_personal_examples_shift_probabilities():
    """A prototype that matches the live embedding pulls probability towards its sign."""
    pred = JSLPredictor()
    try:
        fake = _FakeSession()
        fake.cls = 3
        pred.session, pred.mode = fake, "model"
        pred.protos = None
        for i in range(60):
            pred.push_vector(np.ones(FEATURE_DIM, np.float32), t=i / 30)
        base = pred._infer()
        vecs = np.zeros((2, 8), np.float32)
        vecs[0, 0] = 1.0                       # matches the fake embedding -> sign 7
        vecs[1, 1] = 1.0                       # orthogonal -> sign 9
        pred.protos = (np.array([7, 9]), vecs)
        blended = pred._infer()
        assert blended[7] > base[7] + 0.1 and blended[7] > blended[9]
        assert np.isclose(blended.sum(), 1.0) and np.isclose(blended[IDLE_INDEX], base[IDLE_INDEX], atol=1e-6)
    finally:
        pred.close()
