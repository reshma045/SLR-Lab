"""SQLite store (data/app.db) for the detection history and self-test results."""
from __future__ import annotations

import csv
import io
import sqlite3
import threading
import time

from src.common.config import DATA

DB = DATA / "app.db"
_lock = threading.Lock()
SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session TEXT, sign TEXT, english TEXT,
  confidence REAL, claude_status TEXT, claude_sign TEXT, claude_confidence TEXT,
  claude_reason TEXT, claude_model TEXT, claude_latency REAL);
CREATE TABLE IF NOT EXISTS tests (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run TEXT, profile TEXT, target TEXT,
  predicted TEXT, correct INTEGER, latency_s REAL, confidence REAL, model_mtime REAL);
"""


def _conn() -> sqlite3.Connection:
    DATA.mkdir(exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def add_detection(session: str, sign: str, english: str, confidence: float) -> int:
    with _lock, _conn() as c:
        cur = c.execute("INSERT INTO detections (ts, session, sign, english, confidence) VALUES (?,?,?,?,?)",
                        (time.time(), session, sign, english, confidence))
        return int(cur.lastrowid)


def set_claude(det_id: int, res: dict, model: str) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE detections SET claude_status=?, claude_sign=?, claude_confidence=?, claude_reason=?,"
                  " claude_model=?, claude_latency=? WHERE id=?",
                  (res.get("status"), res.get("sign") or ("none" if res.get("status") == "done" else None),
                   res.get("confidence"), res.get("reason") or res.get("error"), model,
                   res.get("latency_s"), det_id))


def detections(limit: int = 500, q: str = "") -> list[dict]:
    with _lock, _conn() as c:
        sql, args = "SELECT * FROM detections", []
        if q:
            sql += " WHERE sign LIKE ? OR english LIKE ? OR claude_sign LIKE ?"
            args = [f"%{q}%"] * 3
        rows = c.execute(sql + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) for r in rows]


def detection_stats() -> dict:
    with _lock, _conn() as c:
        total = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        today = c.execute("SELECT COUNT(*) FROM detections WHERE ts >= ?", (time.time() - 86400,)).fetchone()[0]
        avg = c.execute("SELECT AVG(confidence) FROM detections").fetchone()[0]
        checked = c.execute("SELECT COUNT(*), SUM(claude_sign = sign) FROM detections WHERE claude_status='done'").fetchone()
        top = c.execute("SELECT sign, COUNT(*) n FROM detections GROUP BY sign ORDER BY n DESC LIMIT 25").fetchall()
        return {"total": total, "last_24h": today, "avg_confidence": avg or 0,
                "claude_checked": checked[0], "claude_agreed": checked[1] or 0,
                "by_sign": [dict(r) for r in top]}


def clear_detections() -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM detections")


def add_test(run: str, profile: str, target: str, predicted: str | None, latency: float | None,
             confidence: float | None, model_mtime: float) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT INTO tests (ts, run, profile, target, predicted, correct, latency_s, confidence, model_mtime)"
                  " VALUES (?,?,?,?,?,?,?,?,?)",
                  (time.time(), run, profile, target, predicted, int(predicted == target), latency,
                   confidence, model_mtime))


def test_rows(run: str | None = None) -> list[dict]:
    with _lock, _conn() as c:
        if run:
            rows = c.execute("SELECT * FROM tests WHERE run=? ORDER BY id", (run,)).fetchall()
        else:
            rows = c.execute("SELECT * FROM tests ORDER BY id").fetchall()
        return [dict(r) for r in rows]


def test_runs() -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT run, MIN(ts) started, COUNT(*) n, SUM(correct) correct, AVG(latency_s) latency,"
                         " profile FROM tests GROUP BY run ORDER BY started DESC").fetchall()
        return [dict(r) for r in rows]


def clear_tests() -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM tests")


def to_csv(rows: list[dict]) -> str:
    """Rows -> CSV text with a UTF-8 BOM so Excel shows Japanese correctly."""
    if not rows:
        return "﻿"
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        r = dict(r)
        if r.get("ts"):
            r["ts"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"]))
        w.writerow(r)
    return "﻿" + buf.getvalue()
