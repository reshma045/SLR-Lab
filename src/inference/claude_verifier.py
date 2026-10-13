"""Optional second opinion from Claude (vision) for a recognised sign.

The local OpenCV + MediaPipe + Transformer pipeline stays the real-time classifier. When it emits
a sign (or the user presses "Ask Claude"), a 6-keyframe sheet of the live clip is sent to Claude
together with labelled keyframe sheets of the 25 reference videos (prompt-cached), and Claude picks
the matching sign from the closed list.

Needs ANTHROPIC_API_KEY in the environment or in a git-ignored `.env` file at the repo root.
Live webcam frames are sent to the Anthropic API only when a key is configured.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from src.common.config import DATA, ROOT, class_names, load_english

MODEL = "claude-opus-5-5"
REF_DIR = DATA / "reference_sheets"
SHEET_FRAMES = 6
CELL_H = 240                       # cell width follows each source's aspect ratio


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=value lines); never overrides variables already set."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def make_sheet(frames: list[np.ndarray], n: int = SHEET_FRAMES) -> bytes:
    """n evenly spaced frames -> 3x2 numbered grid JPEG (time runs left->right, top->bottom)."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    h, w = frames[0].shape[:2]
    size = (int(round(CELL_H * w / h)), CELL_H)
    cells = []
    for k, i in enumerate(idx):
        c = cv2.resize(frames[i], size, interpolation=cv2.INTER_AREA)
        cv2.rectangle(c, (0, 0), (26, 22), (0, 0, 0), -1)
        cv2.putText(c, str(k + 1), (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cells.append(c)
    cols = 3
    rows = [np.hstack(cells[r * cols:(r + 1) * cols]) for r in range(len(cells) // cols)]
    return cv2.imencode(".jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()


def _image_block(jpeg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(jpeg).decode("ascii")}}


class ClaudeVerifier:
    """Thread-safe-enough (stateless per call) wrapper around the Messages API."""

    def __init__(self) -> None:
        load_dotenv()
        self.names = class_names()[:-1]
        self.english = load_english()
        self.available = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        self.client = None
        if self.available:
            import anthropic
            self.client = anthropic.Anthropic(max_retries=1, timeout=60.0)
        self.references = self._reference_blocks()

    def _reference_blocks(self) -> list[dict]:
        blocks: list[dict] = []
        for i, name in enumerate(self.names):
            p = REF_DIR / f"{i:02d}.jpg"
            if p.exists():
                blocks.append({"type": "text", "text": f"Reference sign: {name} ({self.english.get(name, '')})"})
                blocks.append(_image_block(p.read_bytes()))
        if blocks:
            blocks[-1]["cache_control"] = {"type": "ephemeral"}   # cache the stable reference prefix
        return blocks

    def status(self) -> dict:
        return {"available": self.available, "model": MODEL, "references": len(self.references) // 2}

    def verify(self, frames: list[np.ndarray], local_top3: list[dict] | None = None) -> dict:
        """Ask Claude which of the 25 signs the clip shows. Returns a JSON-serialisable dict."""
        if not self.available:
            return {"status": "unavailable", "error": "No ANTHROPIC_API_KEY configured"}
        if len(frames) < SHEET_FRAMES:
            return {"status": "error", "error": "Not enough frames buffered yet"}
        import anthropic

        hint = ", ".join(f"{t['sign']} {t['confidence']:.0%}" for t in (local_top3 or []))
        intro = ("You are verifying an isolated Japanese Sign Language (JSL) sign recognised from a webcam. "
                 "Each image is a 3x2 grid of 6 keyframes in time order (1-6), with MediaPipe hand skeletons "
                 "drawn. The person faces the camera and the frames are NOT mirrored, exactly like the "
                 "references. Compare hand shapes, positions relative to the body and face, and motion.")
        question = (f"Which of the 25 reference signs does this live clip show? Local model top-3: {hint or 'n/a'}. "
                    "Choose \"none\" if no sign is being made or none of the references matches. "
                    "Judge from the images; the local model can be wrong.")
        schema = {"type": "object", "additionalProperties": False,
                  "required": ["sign", "confidence", "reason"],
                  "properties": {"sign": {"type": "string", "enum": self.names + ["none"]},
                                 "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                                 "reason": {"type": "string"}}}
        content = [{"type": "text", "text": intro}, *self.references,
                   {"type": "text", "text": "Live clip:"}, _image_block(make_sheet(frames)),
                   {"type": "text", "text": question}]
        t0 = time.time()
        try:
            resp = self.client.beta.messages.create(
                model=MODEL, max_tokens=4000,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": content}])
        except anthropic.AuthenticationError:
            return {"status": "error", "error": "Invalid ANTHROPIC_API_KEY"}
        except anthropic.RateLimitError:
            return {"status": "error", "error": "Rate limited - try again shortly"}
        except anthropic.APIStatusError as e:
            return {"status": "error", "error": f"API error {e.status_code}"}
        except anthropic.APIConnectionError:
            return {"status": "error", "error": "Network error"}
        latency = round(time.time() - t0, 2)
        if resp.stop_reason == "refusal":
            return {"status": "error", "error": "Request declined", "latency_s": latency}
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {"status": "error", "error": "Unparseable response", "latency_s": latency}
        sign = None if data["sign"] == "none" else data["sign"]
        return {"status": "done", "sign": sign, "english": self.english.get(sign, "") if sign else "",
                "confidence": data["confidence"], "reason": data["reason"], "latency_s": latency,
                "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}
