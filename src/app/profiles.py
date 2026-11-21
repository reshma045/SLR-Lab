"""Signer profile metadata (data/profiles.json, git-ignored): hearing status and JSL experience,
used to break evaluation results down by group (e.g. Deaf vs hearing signers)."""
from __future__ import annotations

import json
import re
import threading

from src.common.config import DATA

FILE = DATA / "profiles.json"
HEARING = {"deaf": "Deaf", "hoh": "Hard of hearing", "hearing": "Hearing", "unspecified": "Not specified"}
JSL = {"native": "Native signer", "fluent": "Fluent", "learner": "Learner", "none": "No JSL experience",
       "unspecified": "Not specified"}
_lock = threading.Lock()


def _clean(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "", name)[:32] or "default"


def load_all() -> dict:
    with _lock:
        if FILE.exists():
            try:
                return json.loads(FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}


def get(name: str) -> dict:
    p = load_all().get(_clean(name), {})
    return {"hearing": p.get("hearing", "unspecified"), "jsl": p.get("jsl", "unspecified")}


def update(name: str, patch: dict) -> dict:
    name = _clean(name)
    with _lock:
        data = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}
        cur = data.get(name, {})
        if patch.get("hearing") in HEARING:
            cur["hearing"] = patch["hearing"]
        if patch.get("jsl") in JSL:
            cur["jsl"] = patch["jsl"]
        data[name] = cur
        FILE.parent.mkdir(parents=True, exist_ok=True)
        FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return get(name)
