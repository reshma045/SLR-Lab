"""One-click entry point: start the JSL recognition server on http://127.0.0.1:8000."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from src.common.config import MODELS  # noqa: E402

HOST, PORT = "127.0.0.1", 8000

if __name__ == "__main__":
    mode = "MODEL" if (MODELS / "jsl_model.onnx").exists() else "DEMO (no models/jsl_model.onnx - run python run_training.py)"
    print(f"JSL recognition server - mode: {mode}")
    print(f"Open http://{HOST}:{PORT} in your browser (Ctrl+C to stop)")
    uvicorn.run("src.inference.server:app", host=HOST, port=PORT, log_level="warning")
