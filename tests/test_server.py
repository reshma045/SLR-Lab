"""HTTP + WebSocket contract of the FastAPI server."""
import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from src.inference.server import app, decode_frame


def _jpeg_b64(img) -> str:
    return base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def test_status_and_index():
    c = TestClient(app)
    s = c.get("/status").json()
    assert s["status"] == "ok" and s["num_signs"] == 25 and len(s["signs"]) == 25
    r = c.get("/")
    assert r.status_code == 200 and "SIMULATED SENSOR FEED" in r.text and "/static/camera.js" in r.text


def test_decode_frame_rejects_garbage():
    assert decode_frame("not base64!!") is None
    assert decode_frame(base64.b64encode(b"junk").decode()) is None
    assert decode_frame(_jpeg_b64(np.zeros((8, 8, 3), np.uint8))).shape == (8, 8, 3)


def test_websocket_frame_and_bad_frame():
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "frame", "data": "garbage"})
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"type": "frame", "data": _jpeg_b64(np.zeros((480, 640, 3), np.uint8))})
        p = ws.receive_json()
        assert p["type"] == "result" and "prediction" in p and "pose" in p and p["frame"]
        assert p["claude"]["status"] in ("off", "ready")
