"""End-to-end check: start the server, hit /status, send a WebSocket frame, load the UI in
headless Chromium with a fake camera, collect console errors and save screenshots.

    python scripts/ui_check.py          # screenshots -> docs/ui_1080p.png, docs/ui_1280.png
"""
from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
OUT = ROOT / "docs"


def wait_ready(timeout: float = 60) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(URL + "/status", timeout=2) as r:
                return json.loads(r.read().decode("utf-8"))
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("server did not start")


async def ws_check() -> dict:
    import cv2
    import numpy as np
    import websockets

    img = np.full((480, 640, 3), 40, np.uint8)
    data = base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()
    async with websockets.connect(f"ws://127.0.0.1:{PORT}/ws", max_size=None) as ws:
        await ws.send(json.dumps({"type": "frame", "data": data}))
        return json.loads(await asyncio.wait_for(ws.recv(), 30))


def browser_check() -> list[str]:
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-fake-device-for-media-stream",
                                          "--use-fake-ui-for-media-stream"])
        for name, (w, h) in {"ui_1080p": (1920, 960), "ui_1280": (1280, 720)}.items():
            page = browser.new_page(viewport={"width": w, "height": h})
            page.on("console", lambda m: m.type == "error" and errors.append(f"console: {m.text}"))
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
            page.goto(URL)
            page.wait_for_timeout(7000)
            scroll = page.evaluate("[document.documentElement.scrollHeight, innerHeight,"
                                   " document.documentElement.scrollWidth, innerWidth]")
            if scroll[0] > scroll[1] or scroll[2] > scroll[3]:
                errors.append(f"{name}: page overflows viewport {scroll}")
            ws_state = page.inner_text("#wsTxt")
            print(f"{name}: ws={ws_state} mode={page.inner_text('#modeTxt')} fps={page.inner_text('#fpsTxt')} "
                  f"buffer={page.inner_text('#bufTxt')}")
            if ws_state != "connected":
                errors.append(f"{name}: websocket not connected ({ws_state})")
            page.screenshot(path=str(OUT / f"{name}.png"))
            page.click("#helpBtn")
            page.wait_for_timeout(300)
            if name == "ui_1080p":
                page.screenshot(path=str(OUT / "ui_help.png"))
            page.close()
        browser.close()
    return errors


def main() -> int:
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.inference.server:app",
                               "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
                              cwd=ROOT)
    try:
        status = wait_ready()
        print("/status:", {k: v for k, v in status.items() if k != "signs"}, f"signs={len(status['signs'])}")
        payload = asyncio.run(ws_check())
        keys = sorted(k for k in payload if k != "frame")
        print("WS payload keys:", keys, "| frame bytes:", len(payload.get("frame", "")))
        assert payload["type"] == "result" and {"prediction", "top3", "glove", "buffer_fill"} <= set(payload)
        errors = browser_check()
        print("Browser errors:", errors or "none")
        print(f"Screenshots: {OUT / 'ui_1080p.png'}, {OUT / 'ui_1280.png'}, {OUT / 'ui_help.png'}")
        return 1 if errors else 0
    finally:
        server.terminate()
        server.wait(10)


if __name__ == "__main__":
    sys.exit(main())
