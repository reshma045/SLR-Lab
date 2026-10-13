"""Claude verifier: works without a key, sheet layout, never calls the API in tests."""
import cv2
import numpy as np

from src.inference import claude_verifier as C


def test_sheet_is_3x2_grid_keeping_aspect():
    frames = [np.full((240, 320, 3), i, np.uint8) for i in range(40)]
    img = cv2.imdecode(np.frombuffer(C.make_sheet(frames), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (2 * C.CELL_H, 3 * 320, 3)


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(C, "load_dotenv", lambda *a, **k: None)
    v = C.ClaudeVerifier()
    assert v.status()["available"] is False
    assert v.verify([np.zeros((240, 320, 3), np.uint8)] * 10)["status"] == "unavailable"


def test_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nJSL_TEST_A='x'\nJSL_TEST_B=y\n", encoding="utf-8")
    monkeypatch.setenv("JSL_TEST_B", "keep")
    monkeypatch.delenv("JSL_TEST_A", raising=False)
    C.load_dotenv(env)
    import os
    assert os.environ["JSL_TEST_A"] == "x" and os.environ["JSL_TEST_B"] == "keep"
