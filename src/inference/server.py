"""FastAPI server: dashboard pages, REST API and the live recognition WebSocket."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.app import api, recordings, settings, store
from src.common.config import FRONTEND, MODEL_IN, MODELS, MODEL_LEN, NUM_SIGNS, WINDOW, class_names, load_english
from src.inference.claude_verifier import ClaudeVerifier
from src.inference.predictor import JSLPredictor
from src.train import prototypes

log = logging.getLogger("jsl.server")
app = FastAPI(title="JSL Real-Time Recognition")
verifier = ClaudeVerifier(settings.load()["claude"]["model"])
api.verifier = verifier
app.include_router(api.router)
app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
PAGES = {"": "index.html", "record": "record.html", "test": "test.html", "history": "history.html",
         "settings": "settings.html"}


class SettingsCache:
    """Re-read settings at most once a second (they can change from the Settings page)."""

    def __init__(self) -> None:
        self.value, self.t = settings.load(), time.monotonic()

    def get(self) -> dict:
        if time.monotonic() - self.t > 1.0:
            self.value, self.t = settings.load(), time.monotonic()
        return self.value


def decode_frame(b64: str) -> np.ndarray | None:
    """base64 JPEG (optionally a data: URL) -> BGR image, or None if undecodable."""
    try:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    except Exception:  # noqa: BLE001 - any garbage input is just a bad frame
        return None


def model_mode() -> str:
    """'model' if a compatible ONNX model exists, else 'demo'."""
    onnx = MODELS / "jsl_model.onnx"
    if not onnx.exists():
        return "demo"
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    return "model" if sess.get_inputs()[0].shape[-1] == MODEL_IN else "demo"


@app.get("/status")
def status() -> dict:
    """Health / configuration info."""
    english = load_english()
    onnx = MODELS / "jsl_model.onnx"
    mode = model_mode()
    return {"status": "ok", "mode": mode, "model_loaded": mode == "model",
            "model_updated": time.strftime("%Y-%m-%d %H:%M", time.localtime(onnx.stat().st_mtime)) if onnx.exists() else None,
            "window": WINDOW, "model_len": MODEL_LEN, "num_signs": NUM_SIGNS,
            "signs": [{"sign": s, "english": english.get(s, "")} for s in class_names()[:NUM_SIGNS]],
            "glove": "simulated", "claude": verifier.status()}


@app.get("/", include_in_schema=False)
@app.get("/{page}", include_in_schema=False)
def page(page: str = "") -> FileResponse:
    """Serve a dashboard page."""
    if page not in PAGES:
        raise HTTPException(404)
    return FileResponse(FRONTEND / PAGES[page], media_type="text/html; charset=utf-8")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    """Client messages: frame / hello{page} / ask_claude / record_start{sign_id} / record_cancel / reset.

    Only the most recent unprocessed frame is kept, so a slow server drops frames instead of lagging.
    """
    await ws.accept()
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=1)          # MediaPipe graphs are single-threaded per client
    claude_pool = ThreadPoolExecutor(max_workers=1)   # network-bound; never blocks the live loop
    cfg = SettingsCache()
    predictor = await loop.run_in_executor(pool, JSLPredictor)
    session = uuid.uuid4().hex[:8]
    state: dict = {"page": "live", "latest": None, "rec": None}
    claude: dict = {"status": "ready" if verifier.available else "off"}
    ready = asyncio.Event()

    def start_claude(trigger: str, det_id: int | None = None) -> None:
        """Snapshot the clip and ask Claude in the background (runs on the predictor thread)."""
        c = cfg.get()["claude"]
        if not (verifier.available and c["enabled"]) or claude.get("status") == "checking":
            return
        frames, top3 = predictor.clip_frames(), predictor.current()[1]
        claude.clear()
        claude.update(status="checking", trigger=trigger, model=c["model"])

        def done(fut) -> None:
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {"status": "error", "error": str(exc)}
            claude.clear()
            claude.update(res, trigger=trigger)
            if det_id is not None:
                store.set_claude(det_id, res, c["model"])
        claude_pool.submit(verifier.verify, frames, top3, c["model"]).add_done_callback(done)

    def handle_record(t: float) -> dict | None:
        rec = state["rec"]
        if rec is None:
            return None
        rec["times"].append(t)
        rec["raws"].append(predictor.last_raw.copy())
        elapsed = t - rec["t0"]
        if elapsed < rec["duration"]:
            return {"state": "recording", "sign_id": rec["sign_id"], "elapsed": round(elapsed, 2),
                    "duration": rec["duration"]}
        state["rec"] = None
        res = recordings.save(cfg.get()["profile"], rec["sign_id"], rec["times"], rec["raws"])
        if res["saved"] and predictor.session is not None:     # instant personalisation, no retrain
            res["personal_example"] = prototypes.add(predictor.session, np.load(res["path"]), rec["sign_id"])
        res.pop("path", None)
        return {"state": "saved" if res["saved"] else "rejected", "sign_id": rec["sign_id"], **res}

    def handle(data: str) -> dict:
        frame = decode_frame(data)
        if frame is None:
            return {"type": "error", "error": "bad frame", "mode": predictor.mode}
        s = cfg.get()
        try:
            predictor.set_params(**s["recognition"])
            t = time.monotonic()
            payload = predictor.process_frame(frame, t=t, want_glove=s["display"]["show_glove"])
            rec = handle_record(t)
            em = payload["emitted"]
            if em and state["page"] == "live":
                conf = payload["prediction"]["confidence"]
                det_id = store.add_detection(session, em, predictor.english.get(em, ""), conf)
                payload["detection_id"] = det_id
                mode = s["claude"]["mode"]
                if mode == "auto" or (mode == "uncertain" and conf < s["claude"]["uncertain_below"]):
                    start_claude(em, det_id)
            claude_view = dict(claude)
            if claude_view.get("status") in ("off", "ready"):
                claude_view["status"] = "ready" if verifier.available and s["claude"]["enabled"] else "off"
            return {"type": "result", **payload, "claude": claude_view, "recording": rec}
        except Exception as exc:  # noqa: BLE001 - never let one frame kill the session
            log.exception("frame processing failed")
            return {"type": "error", "error": str(exc), "mode": predictor.mode}

    async def receiver() -> None:
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "frame" and isinstance(msg.get("data"), str):
                state["latest"] = msg["data"]          # overwrite: drop stale frames
                ready.set()
            elif kind == "hello":
                state["page"] = str(msg.get("page", "live"))
            elif kind == "reset":
                await loop.run_in_executor(pool, predictor.reset)
            elif kind == "ask_claude":
                await loop.run_in_executor(pool, start_claude, "manual")
            elif kind == "record_start":
                sid = int(msg.get("sign_id", -1))
                if 0 <= sid < NUM_SIGNS:
                    dur = float(msg.get("duration") or cfg.get()["record"]["duration_s"])
                    state["rec"] = {"sign_id": sid, "t0": time.monotonic(), "duration": dur,
                                    "times": [], "raws": []}
            elif kind == "record_cancel":
                state["rec"] = None

    async def worker() -> None:
        while True:
            await ready.wait()
            ready.clear()
            data, state["latest"] = state["latest"], None
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
        claude_pool.shutdown(wait=False, cancel_futures=True)
