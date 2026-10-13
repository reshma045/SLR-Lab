"""Persistent app settings (data/settings.json) and the Claude API key (git-ignored .env)."""
from __future__ import annotations

import copy
import json
import os
import re
import threading

from src.common.config import ROOT, SETTINGS_FILE

ENV_FILE = ROOT / ".env"
CLAUDE_MODELS = {
    "claude-haiku-5-5": "Claude Haiku 5.5 - fastest, cheapest (recommended for real time)",
    "claude-sonnet-5-5": "Claude Sonnet 5.5 - balanced",
    "claude-opus-5-5": "Claude Opus 5.5 - most capable, slowest",
}
DEFAULTS: dict = {
    "profile": "default",
    "recognition": {"conf_threshold": 0.60, "streak": 5, "cooldown_s": 1.5, "ema_alpha": 0.4},
    "camera": {"send_fps": 18, "jpeg_quality": 0.7, "mirror": True},
    "display": {"show_glove": False, "show_top3": True, "theme": "dark", "history_len": 8},
    "claude": {"enabled": True, "mode": "auto", "model": "claude-haiku-5-5", "uncertain_below": 0.8},
    "record": {"duration_s": 4.0, "countdown_s": 3, "target_per_sign": 5},
    "test": {"timeout_s": 7.0, "rounds": 25, "show_reference": False},
}
LIMITS = {  # (min, max) for numeric fields
    "recognition.conf_threshold": (0.2, 0.99), "recognition.streak": (1, 20),
    "recognition.cooldown_s": (0.0, 10.0), "recognition.ema_alpha": (0.05, 1.0),
    "camera.send_fps": (5, 30), "camera.jpeg_quality": (0.3, 0.95),
    "display.history_len": (3, 20), "claude.uncertain_below": (0.3, 1.0),
    "record.duration_s": (2.0, 8.0), "record.countdown_s": (0, 10), "record.target_per_sign": (1, 50),
    "test.timeout_s": (3.0, 20.0), "test.rounds": (1, 200),
}
_lock = threading.Lock()


def _merge(base: dict, new: dict, prefix: str = "") -> dict:
    """Deep-merge `new` into a copy of `base`, keeping only known keys and coercing/clamping types."""
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if k not in base:
            continue
        key = f"{prefix}{k}"
        if isinstance(base[k], dict):
            out[k] = _merge(base[k], v if isinstance(v, dict) else {}, key + ".")
            continue
        try:
            if isinstance(base[k], bool):
                v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
            elif isinstance(base[k], (int, float)):
                v = type(base[k])(float(v))
                lo, hi = LIMITS.get(key, (None, None))
                if lo is not None:
                    v = type(base[k])(min(max(v, lo), hi))
            else:
                v = str(v)
        except (TypeError, ValueError):
            continue
        out[k] = v
    if out.get("claude", {}).get("model") not in CLAUDE_MODELS and "claude" in out:
        out["claude"]["model"] = DEFAULTS["claude"]["model"]
    if "claude" in out and out["claude"].get("mode") not in ("auto", "uncertain", "manual"):
        out["claude"]["mode"] = "auto"
    if "profile" in out:
        out["profile"] = re.sub(r"[^A-Za-z0-9_-]", "", out["profile"])[:32] or "default"
    return out


def load() -> dict:
    """Current settings (defaults merged with the saved file)."""
    with _lock:
        saved = {}
        if SETTINGS_FILE.exists():
            try:
                saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                saved = {}
        return _merge(DEFAULTS, saved)


def update(patch: dict) -> dict:
    """Merge a partial update, save and return the full settings."""
    merged = _merge(load(), patch)
    with _lock:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged


# ---------------------------------------------------------------- API key (.env)
def get_api_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY", "")


def masked_key() -> str:
    k = get_api_key()
    return f"{k[:7]}…{k[-4:]}" if len(k) > 12 else ("set" if k else "")


def set_api_key(key: str) -> None:
    """Store the key in .env (git-ignored) and the current process environment. Empty removes it."""
    key = key.strip()
    lines = ENV_FILE.read_text(encoding="utf-8").splitlines() if ENV_FILE.exists() else []
    lines = [ln for ln in lines if not ln.strip().startswith("ANTHROPIC_API_KEY=")]
    if key:
        lines.append(f"ANTHROPIC_API_KEY={key}")
        os.environ["ANTHROPIC_API_KEY"] = key
    else:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    ENV_FILE.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
