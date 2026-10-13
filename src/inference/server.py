"""FastAPI server: serves the UI and runs the live predictor over a WebSocket."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.common.config import FRONTEND, MODELS, MODEL_LEN, NUM_SIGNS, WINDOW, class_names, load_english
from src.inference.predictor import JSLPredictor

log = logging.getLogger("jsl.server")
app = FastAPI(title="JSL Real-Time Recognition")


def decode_frame(b64: str) -> np.ndarray | None:
    """base64 JPEG (optionally a data: URL) -> BGR image, or None if undecodable."""
    try:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    except Exception:  # noqa: BLE001 - any garbage input is just a bad frame
        return None


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page UI."""
    return FileResponse(FRONTEND / "index.html", media_type="text/html; charset=utf-8")


@app.get("/status")
def status() -> dict:
    """Health / configuration info."""
    english = load_english()
    model = (MODELS / "jsl_model.onnx").exists()
    return {"status": "ok", "mode": "model" if model else "demo", "model_loaded": model,
            "window": WINDOW, "model_len": MODEL_LEN, "num_signs": NUM_SIGNS,
            "signs": [{"sign": s, "english": english.get(s, "")} for s in class_names()[:NUM_SIGNS]],
            "glove": "simulated"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Client sends {"type":"frame","data":"<b64 jpeg>"}; server replies with the payload.

    Only the most recent unprocessed frame is kept, so a slow server drops frames instead of lagging.
    """
    await ws.accept()
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)          # MediaPipe graph is single-threaded per client
    predictor = await loop.run_in_executor(pool, JSLPredictor)
    latest: dict[str, str | None] = {"data": None}
    ready = asyncio.Event()

    def handle(data: str) -> dict:
        frame = decode_frame(data)
        if frame is None:
            return {"type": "error", "error": "bad frame", "mode": predictor.mode}
        try:
            return {"type": "result", **predictor.process_frame(frame)}
        except Exception as exc:  # noqa: BLE001 - never let one frame kill the session
            log.exception("frame processing failed")
            return {"type": "error", "error": str(exc), "mode": predictor.mode}

    async def receiver() -> None:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "frame" and isinstance(msg.get("data"), str):
                latest["data"] = msg["data"]           # overwrite: drop stale frames
                ready.set()
            elif msg.get("type") == "reset":
                await loop.run_in_executor(pool, predictor.reset)

    async def worker() -> None:
        while True:
            await ready.wait()
            ready.clear()
            data, latest["data"] = latest["data"], None
            if data is not None:
                await ws.send_json(await loop.run_in_executor(pool, handle, data))

    tasks = [asyncio.create_task(receiver()), asyncio.create_task(worker())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, RuntimeError)):
                log.warning("websocket closed: %r", exc)
    finally:
        for t in tasks:
            t.cancel()
        await loop.run_in_executor(pool, predictor.close)
        pool.shutdown(wait=False)
