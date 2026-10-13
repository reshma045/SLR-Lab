"""Run `run_training.py` as a background process from the dashboard and report its progress."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading

from src.common.config import MODELS, ROOT

LOG = MODELS / "train.log"
STATUS = MODELS / "train_status.json"
_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def running() -> bool:
    return _proc is not None and _proc.poll() is None


def start(profile: str, epochs: int = 200) -> dict:
    """Start training unless one is already running."""
    global _proc
    with _lock:
        if running():
            return {"started": False, "error": "Training is already running"}
        MODELS.mkdir(exist_ok=True)
        STATUS.write_text(json.dumps({"state": "starting", "epoch": 0, "max_epochs": epochs, "history": []}),
                          encoding="utf-8")
        log = open(LOG, "w", encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        _proc = subprocess.Popen([sys.executable, str(ROOT / "run_training.py"), "--profile", profile,
                                  "--epochs", str(epochs)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=flags, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        return {"started": True}


def stop() -> bool:
    if running():
        _proc.terminate()
        STATUS.write_text(json.dumps({"state": "cancelled"}), encoding="utf-8")
        return True
    return False


def status() -> dict:
    """Training status file + whether the process is alive + the last log lines."""
    st = {}
    if STATUS.exists():
        try:
            st = json.loads(STATUS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            st = {}
    if st.get("state") in ("starting", "training", "exporting") and not running():
        st["state"] = "failed" if _proc is not None and _proc.returncode else st.get("state")
        if _proc is None:                      # server restarted mid-training
            st["state"] = "interrupted"
    tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-12:] if LOG.exists() else []
    report = {}
    if (MODELS / "train_report.json").exists():
        report = json.loads((MODELS / "train_report.json").read_text(encoding="utf-8"))
    return {**st, "running": running(), "log_tail": tail, "last_report": report,
            "has_previous": (MODELS / "previous" / "jsl_model.onnx").exists()}


def rollback() -> bool:
    """Restore the model saved before the last training run."""
    prev = MODELS / "previous"
    if running() or not (prev / "jsl_model.onnx").exists():
        return False
    for f in prev.iterdir():
        shutil.copy2(f, MODELS / f.name)
    return True
