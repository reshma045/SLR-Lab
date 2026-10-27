"""Run scripts/evaluate.py as a background process from the dashboard."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading

from src.common.config import MODELS, ROOT
from src.eval.offline import REPORT, STATUS

LOG = MODELS / "eval.log"
_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def running() -> bool:
    return _proc is not None and _proc.poll() is None


def start(profile: str, sets: list[str] | None = None) -> dict:
    global _proc
    with _lock:
        if running():
            return {"started": False, "error": "An evaluation is already running"}
        MODELS.mkdir(exist_ok=True)
        STATUS.write_text(json.dumps({"state": "starting"}), encoding="utf-8")
        cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--profile", profile]
        if sets:
            cmd += ["--sets", *sets]
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        _proc = subprocess.Popen(cmd, cwd=ROOT, stdout=open(LOG, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                                 creationflags=flags, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return {"started": True}


def status() -> dict:
    st = {}
    if STATUS.exists():
        try:
            st = json.loads(STATUS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st = {}
    if st.get("state") in ("starting", "running") and not running():
        st["state"] = "failed" if _proc is not None else "interrupted"
    return {**st, "running": running()}


def report() -> dict | None:
    return json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else None
