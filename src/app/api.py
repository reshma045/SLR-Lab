"""REST API for the dashboard pages (settings, recordings, training, history, self-tests)."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, Response

from src.app import recordings, settings, store, trainer
from src.common.config import MODELS, NUM_SIGNS, class_names, load_english
from src.inference.claude_verifier import REF_DIR

router = APIRouter(prefix="/api")
verifier = None          # set by server.py (shared ClaudeVerifier)


def _signs() -> list[dict]:
    english = load_english()
    return [{"id": i, "sign": s, "english": english.get(s, ""),
             "reference": (REF_DIR / f"{i:02d}.jpg").exists()}
            for i, s in enumerate(class_names()[:NUM_SIGNS])]


# ------------------------------------------------------------------ settings
@router.get("/settings")
def get_settings() -> dict:
    return {"settings": settings.load(), "claude_models": settings.CLAUDE_MODELS,
            "claude_key": settings.masked_key(), "profiles": recordings.profiles()}


@router.post("/settings")
def post_settings(patch: dict = Body(...)) -> dict:
    s = settings.update(patch)
    if verifier is not None:
        verifier.model = s["claude"]["model"]
    return get_settings()


@router.post("/claude/key")
def set_key(body: dict = Body(...)) -> dict:
    key = str(body.get("key", "")).strip()
    if key and not key.startswith("sk-ant-"):
        raise HTTPException(400, "That does not look like an Anthropic API key (should start with sk-ant-)")
    settings.set_api_key(key)
    return {"claude_key": settings.masked_key()}


@router.post("/claude/test")
def test_claude(body: dict = Body(default={})) -> dict:
    if verifier is None:
        return {"ok": False, "error": "Verifier not initialised"}
    return verifier.test(body.get("model") or settings.load()["claude"]["model"])


# ------------------------------------------------------------------ signs & references
@router.get("/signs")
def signs() -> dict:
    prof = settings.load()["profile"]
    counts = recordings.counts(prof)
    return {"signs": [{**s, "recordings": counts[s["id"]]} for s in _signs()], "profile": prof}


@router.get("/reference/{sid}.jpg")
def reference(sid: int) -> FileResponse:
    p = REF_DIR / f"{sid:02d}.jpg"
    if not 0 <= sid < NUM_SIGNS or not p.exists():
        raise HTTPException(404, "No reference sheet (run scripts/build_reference_sheets.py)")
    return FileResponse(p, media_type="image/jpeg")


# ------------------------------------------------------------------ recordings
@router.get("/recordings/{sid}")
def list_recordings(sid: int) -> dict:
    return {"recordings": recordings.listing(settings.load()["profile"], sid)}


@router.delete("/recordings/{sid}/{name}")
def delete_recording(sid: int, name: str) -> dict:
    if not recordings.delete(settings.load()["profile"], sid, name):
        raise HTTPException(404, "Recording not found")
    return {"ok": True, "count": recordings.count(settings.load()["profile"], sid)}


@router.delete("/recordings")
def delete_all_recordings() -> dict:
    recordings.delete_all(settings.load()["profile"])
    return {"ok": True}


# ------------------------------------------------------------------ training
@router.post("/train")
def start_training(body: dict = Body(default={})) -> dict:
    epochs = int(body.get("epochs", 200))
    return trainer.start(settings.load()["profile"], max(5, min(epochs, 400)))


@router.get("/train/status")
def training_status() -> dict:
    return trainer.status()


@router.post("/train/stop")
def stop_training() -> dict:
    return {"stopped": trainer.stop()}


@router.post("/train/rollback")
def rollback() -> dict:
    if not trainer.rollback():
        raise HTTPException(409, "Nothing to roll back to (or training is running)")
    return {"ok": True}


@router.get("/train/confusion.png")
def confusion() -> FileResponse:
    p = MODELS / "confusion.png"
    if not p.exists():
        raise HTTPException(404, "No confusion matrix yet")
    return FileResponse(p, media_type="image/png", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ history
@router.get("/history")
def history(limit: int = 500, q: str = "") -> dict:
    return {"rows": store.detections(min(limit, 5000), q), "stats": store.detection_stats()}


@router.get("/history.csv")
def history_csv() -> Response:
    return Response(store.to_csv(store.detections(100000)), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=jsl_history.csv"})


@router.delete("/history")
def clear_history() -> dict:
    store.clear_detections()
    return {"ok": True}


# ------------------------------------------------------------------ self-tests
@router.post("/tests")
def add_test(body: dict = Body(...)) -> dict:
    onnx = MODELS / "jsl_model.onnx"
    store.add_test(str(body.get("run", "")), settings.load()["profile"], str(body.get("target", "")),
                   body.get("predicted"), body.get("latency_s"), body.get("confidence"),
                   onnx.stat().st_mtime if onnx.exists() else 0.0)
    return {"ok": True}


@router.get("/tests")
def tests(run: str | None = None) -> dict:
    rows = store.test_rows(run)
    names = class_names()[:NUM_SIGNS]
    per_sign = {n: {"n": 0, "correct": 0} for n in names}
    confusion: dict[str, dict[str, int]] = {}
    for r in rows:
        ps = per_sign.setdefault(r["target"], {"n": 0, "correct": 0})
        ps["n"] += 1
        ps["correct"] += r["correct"]
        pred = r["predicted"] or "(none)"
        confusion.setdefault(r["target"], {}).setdefault(pred, 0)
        confusion[r["target"]][pred] += 1
    n = len(rows)
    return {"runs": store.test_runs(), "n": n, "accuracy": (sum(r["correct"] for r in rows) / n) if n else None,
            "per_sign": per_sign, "confusion": confusion, "rows": rows[-200:]}


@router.get("/tests.csv")
def tests_csv() -> Response:
    return Response(store.to_csv(store.test_rows()), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=jsl_selftests.csv"})


@router.delete("/tests")
def clear_tests() -> dict:
    store.clear_tests()
    return {"ok": True}
