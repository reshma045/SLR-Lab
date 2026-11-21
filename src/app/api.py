"""REST API for the dashboard pages (settings, recordings, training, history, self-tests)."""
from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import FileResponse, Response

from src.app import evaluator, recordings, settings, store, trainer
from src.common.config import MODELS, NUM_SIGNS, class_names, load_english
from src.inference.claude_verifier import REF_DIR

router = APIRouter(prefix="/api")
verifier = None          # set by server.py (shared ClaudeVerifier)


def rebuild_personal_examples() -> int:
    """Re-embed the active profile's takes with the current model (after deletes / profile switch)."""
    onnx = MODELS / "jsl_model.onnx"
    if not onnx.exists():
        return 0
    import onnxruntime as ort
    from src.train import prototypes
    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    try:
        return prototypes.rebuild(sess, settings.load()["profile"])
    except Exception:  # noqa: BLE001 - an incompatible old model must not break the API
        return 0


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
    old_profile = settings.load()["profile"]
    s = settings.update(patch)
    if s["profile"] != old_profile:
        rebuild_personal_examples()
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
    rebuild_personal_examples()
    return {"ok": True, "count": recordings.count(settings.load()["profile"], sid)}


@router.delete("/recordings")
def delete_all_recordings() -> dict:
    recordings.delete_all(settings.load()["profile"])
    rebuild_personal_examples()
    return {"ok": True}


# ------------------------------------------------------------------ training
@router.post("/train")
def start_training(body: dict = Body(default={})) -> dict:
    epochs = int(body.get("epochs", 120))
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


# ------------------------------------------------------------------ sentence mode
@router.post("/sentence")
def make_sentence(body: dict = Body(...)) -> dict:
    """Recognised sign sequence -> natural Japanese + English (Claude if configured)."""
    glosses = [str(g) for g in body.get("glosses", [])][:30]
    if verifier is None:
        raise HTTPException(503, "Verifier not initialised")
    s = settings.load()["claude"]
    if s["enabled"]:
        res = verifier.sentence(glosses, s["model"])
    else:
        english = [verifier.english.get(g, "") for g in glosses]
        res = {"status": "fallback", "japanese": "・".join(glosses), "english": " / ".join(e for e in english if e),
               "note": "Claude is disabled in Settings - showing the sign sequence only"}
    if glosses:
        store.add_sentence(glosses, res.get("japanese", ""), res.get("english", ""), res.get("model"))
    return res


@router.get("/sentences")
def list_sentences() -> dict:
    return {"rows": store.sentences()}


# ------------------------------------------------------------------ evaluation metrics
def _realtime(run: str | None = None) -> dict:
    """Accuracy / precision / recall / F1 of live Self-test attempts (webcam, real time)."""
    import numpy as np
    from src.eval.metrics import classification_report
    rows = store.test_rows(run)
    if not rows:
        return {"report": None, "n": 0}
    rep = classification_report([r["target"] for r in rows], [r["predicted"] for r in rows])
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None and r["correct"]]
    rep["mean_latency_s"] = float(np.mean(lat)) if lat else None
    rep["median_latency_s"] = float(np.median(lat)) if lat else None
    runs = []
    for r in store.test_runs():
        rr = store.test_rows(r["run"])
        rp = classification_report([x["target"] for x in rr], [x["predicted"] for x in rr])
        runs.append({"run": r["run"], "started": r["started"], "n": rp["n"], "accuracy": rp["accuracy"],
                     "macro_precision": rp["macro_precision"], "macro_recall": rp["macro_recall"],
                     "macro_f1": rp["macro_f1"]})
    return {"report": rep, "n": rep["n"], "runs": runs}


@router.get("/metrics/realtime")
def realtime_metrics(run: str | None = None) -> dict:
    return _realtime(run)


@router.get("/metrics/offline")
def offline_metrics() -> dict:
    return {"report": evaluator.report(), "status": evaluator.status()}


@router.post("/metrics/offline/run")
def run_offline(body: dict = Body(default={})) -> dict:
    from src.eval.offline import SETS
    sets = [s for s in body.get("sets", []) if s in SETS] or None
    return evaluator.start(settings.load()["profile"], sets)


@router.get("/metrics.csv")
def metrics_csv() -> Response:
    """One row per evaluation set (offline sets + real-time self-tests) with the headline metrics."""
    rows = []
    rep = evaluator.report()
    for name, s in ((rep or {}).get("sets") or {}).items():
        if s.get("report"):
            r = s["report"]
            rows.append({"mode": "offline", "set": s["title"], "n": r["n"], "accuracy": round(r["accuracy"], 4),
                         "macro_precision": round(r["macro_precision"], 4), "macro_recall": round(r["macro_recall"], 4),
                         "macro_f1": round(r["macro_f1"], 4), "weighted_f1": round(r["weighted_f1"], 4),
                         "no_sign_rate": round(r["none_rate"], 4), "note": s.get("note", "")})
    rt = _realtime()
    if rt["report"]:
        r = rt["report"]
        rows.append({"mode": "real-time", "set": "Self-test (live webcam)", "n": r["n"], "accuracy": round(r["accuracy"], 4),
                     "macro_precision": round(r["macro_precision"], 4), "macro_recall": round(r["macro_recall"], 4),
                     "macro_f1": round(r["macro_f1"], 4), "weighted_f1": round(r["weighted_f1"], 4),
                     "no_sign_rate": round(r["none_rate"], 4), "note": "Your live signing"})
    import csv
    import io
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=jsl_metrics.csv"})


# ------------------------------------------------------------------ gloss-free translation
from src.app.jobs import Job  # noqa: E402

glossfree_job = Job("scripts/train_glossfree.py", MODELS / "glossfree.log")


@router.get("/glossfree/status")
def glossfree_status() -> dict:
    import json
    from src.glossfree.train import REPORT, STATUS
    st = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    if st.get("state") in ("training", "evaluating") and not glossfree_job.running() and glossfree_job.proc is not None:
        st["state"] = "failed"
    if not REPORT.exists():
        return {"trained": False, "status": st, "running": glossfree_job.running(),
                "hint": "Train it on the Evaluation page or run python scripts/train_glossfree.py"}
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    return {"trained": True, "status": st, "running": glossfree_job.running(), "metrics": rep["metrics"],
            "description": rep["description"], "trained_at": rep["trained"], "test_items": rep["test_items"],
            "examples": rep["examples"][::3][:21]}


@router.post("/glossfree/train")
def glossfree_train() -> dict:
    return glossfree_job.start("--profile", settings.load()["profile"])


# ------------------------------------------------------------------ learning curves, signer groups
from src.app import profiles  # noqa: E402


@router.get("/metrics/curves")
def learning_curves() -> dict:
    from src.eval.curves import backfill_from_logs, curves
    backfill_from_logs()
    return curves()


@router.get("/metrics/curves.png")
def learning_curves_png() -> FileResponse:
    from src.eval.curves import backfill_from_logs, plot
    backfill_from_logs()
    return FileResponse(plot(MODELS / "learning_curves.png"), media_type="image/png",
                        headers={"Cache-Control": "no-store"})


@router.get("/profiles")
def list_profiles() -> dict:
    names = sorted(set(recordings.profiles()) | set(profiles.load_all()) | {settings.load()["profile"]})
    return {"active": settings.load()["profile"], "hearing_options": profiles.HEARING, "jsl_options": profiles.JSL,
            "profiles": {n: profiles.get(n) for n in names}}


@router.post("/profiles/{name}")
def update_profile(name: str, body: dict = Body(...)) -> dict:
    return profiles.update(name, body)


def _breakdown() -> dict:
    """Self-test (real-time) metrics per hearing group and per signer profile."""
    import numpy as np
    from src.eval.metrics import classification_report
    rows = store.test_rows()
    by_group: dict[str, list[dict]] = {}
    by_profile: dict[str, list[dict]] = {}
    for r in rows:
        prof = r["profile"] or "default"
        by_profile.setdefault(prof, []).append(r)
        by_group.setdefault(profiles.get(prof)["hearing"], []).append(r)

    def summarise(rs: list[dict]) -> dict:
        rep = classification_report([x["target"] for x in rs], [x["predicted"] for x in rs])
        lat = [x["latency_s"] for x in rs if x["latency_s"] is not None and x["correct"]]
        return {"n": rep["n"], "accuracy": rep["accuracy"], "macro_precision": rep["macro_precision"],
                "macro_recall": rep["macro_recall"], "macro_f1": rep["macro_f1"], "none_rate": rep["none_rate"],
                "mean_latency_s": float(np.mean(lat)) if lat else None}

    groups = []
    for key, label in profiles.HEARING.items():
        rs = by_group.get(key, [])
        signers = sorted({x["profile"] or "default" for x in rs})
        groups.append({"group": key, "label": label, "signers": len(signers),
                       **(summarise(rs) if rs else {"n": 0})})
    per_profile = [{"profile": p, **profiles.get(p), **summarise(rs)} for p, rs in sorted(by_profile.items())]
    return {"groups": groups, "profiles": per_profile, "total_tests": len(rows)}


@router.get("/metrics/breakdown")
def breakdown() -> dict:
    return _breakdown()


@router.get("/metrics/breakdown.csv")
def breakdown_csv() -> Response:
    import csv
    import io
    b = _breakdown()
    buf = io.StringIO()
    cols = ["level", "name", "hearing", "jsl", "signers", "n", "accuracy", "macro_precision", "macro_recall",
            "macro_f1", "none_rate", "mean_latency_s"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for g in b["groups"]:
        w.writerow({"level": "group", "name": g["label"], "hearing": g["group"], **g})
    for p in b["profiles"]:
        w.writerow({"level": "signer", "name": p["profile"], "signers": 1, **p})
    return Response("\ufeff" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": "attachment; filename=jsl_deaf_hearing_breakdown.csv"})
