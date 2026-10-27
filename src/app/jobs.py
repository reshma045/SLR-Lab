"""Generic background job (one script at a time) for dashboard-triggered long tasks."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

from src.common.config import ROOT


class Job:
    def __init__(self, script: str, log: Path) -> None:
        self.script, self.log = script, log
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, *args: str) -> dict:
        with self.lock:
            if self.running():
                return {"started": False, "error": "Already running"}
            self.log.parent.mkdir(exist_ok=True)
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            self.proc = subprocess.Popen([sys.executable, str(ROOT / self.script), *args], cwd=ROOT,
                                         stdout=open(self.log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                                         creationflags=flags, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            return {"started": True}
