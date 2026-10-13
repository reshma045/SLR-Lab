"""Dashboard backend: settings validation, API key storage, recordings, history/test store, API."""
import numpy as np
import pytest

from src.app import recordings, settings, store
from src.common.config import FEATURE_DIM, NOSE, PRESENT


@pytest.fixture
def tmp_app(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(settings, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(recordings, "USER_DATA", tmp_path / "user")
    monkeypatch.setattr(store, "DB", tmp_path / "app.db")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return tmp_path


def test_settings_merge_clamp_and_validate(tmp_app):
    s = settings.update({"recognition": {"conf_threshold": 5, "streak": "7"}, "claude": {"model": "gpt-4"},
                         "unknown": 1, "profile": "Ana Lab!"})
    assert s["recognition"]["conf_threshold"] == 0.99 and s["recognition"]["streak"] == 7
    assert s["claude"]["model"] == "claude-haiku-5-5"            # unknown model -> default (Haiku 5.5)
    assert "unknown" not in s and s["profile"] == "AnaLab"
    assert settings.load() == s


def test_api_key_is_stored_in_env_file_and_masked(tmp_app):
    settings.set_api_key("sk-ant-test-1234567890abcd")
    assert "ANTHROPIC_API_KEY=sk-ant-test-1234567890abcd" in (tmp_app / ".env").read_text()
    assert settings.masked_key().startswith("sk-ant-") and "1234567890" not in settings.masked_key()
    settings.set_api_key("")
    assert settings.masked_key() == "" and "ANTHROPIC" not in (tmp_app / ".env").read_text()


def _take(n=60, hands=True, pose=True, fps=15.0):
    times = [i / fps for i in range(n)]
    vecs = []
    for _ in times:
        v = np.zeros(FEATURE_DIM, np.float32)
        if hands:
            v[PRESENT.start] = 1
            v[:63] = 0.5
        if pose:
            v[NOSE] = [0, -0.7]
        vecs.append(v)
    return times, vecs


def test_recording_resampled_to_30fps_and_quality_gate(tmp_app):
    res = recordings.save("default", 3, *_take())
    assert res["saved"] and res["count"] == 1
    seq = np.load(next((tmp_app / "user" / "default" / "03").glob("*.npy")))
    assert seq.shape[1] == FEATURE_DIM and abs(len(seq) - 4 * 30) <= 2           # 4 s at 30 fps
    assert not recordings.save("default", 3, *_take(hands=False))["saved"]
    assert not recordings.save("default", 3, *_take(pose=False))["saved"]
    assert recordings.counts("default")[3] == 1
    name = recordings.listing("default", 3)[0]["name"]
    assert recordings.delete("default", 3, name) and recordings.count("default", 3) == 0
    assert not recordings.delete("default", 3, "../../evil")


def test_store_history_and_tests(tmp_app):
    det = store.add_detection("s1", "ある", "To exist", 0.9)
    store.set_claude(det, {"status": "done", "sign": "ある", "confidence": "high", "latency_s": 1.2}, "claude-haiku-5-5")
    st = store.detection_stats()
    assert st["total"] == 1 and st["claude_agreed"] == 1
    store.add_test("r1", "default", "ある", "ある", 1.0, 0.9, 0)
    store.add_test("r1", "default", "いる", None, None, None, 0)
    runs = store.test_runs()
    assert runs[0]["n"] == 2 and runs[0]["correct"] == 1
    assert store.to_csv(store.test_rows()).startswith("﻿")


def test_api_endpoints(tmp_app):
    from fastapi.testclient import TestClient
    from src.inference.server import app
    c = TestClient(app)
    assert c.get("/api/settings").json()["settings"]["claude"]["model"] == "claude-haiku-5-5"
    r = c.post("/api/settings", json={"display": {"theme": "light"}}).json()
    assert r["settings"]["display"]["theme"] == "light"
    assert c.post("/api/claude/key", json={"key": "not-a-key"}).status_code == 400
    assert len(c.get("/api/signs").json()["signs"]) == 25
    assert c.post("/api/tests", json={"run": "x", "target": "ある", "predicted": "ある", "latency_s": 1}).json()["ok"]
    assert c.get("/api/tests").json()["accuracy"] == 1.0
    assert c.get("/api/history.csv").status_code == 200
    for page in ("/", "/record", "/test", "/history", "/settings"):
        assert c.get(page).status_code == 200
