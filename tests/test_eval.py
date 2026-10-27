"""Evaluation metrics, gloss-free corpus / model / segmenter, and the metrics API."""
import numpy as np
import pytest
import torch

from src.common.config import DATA, FEATURE_DIM, MODEL_IN, POS_R, PRESENT, load_label_map
from src.eval.metrics import classification_report
from src.glossfree.corpus import CharVocab, build, split
from src.glossfree.metrics import bleu4, chrf, exact, wer


def test_classification_report_matches_sklearn():
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    yt = ["a", "a", "b", "b", "c", "c", "c", "d"]
    yp = ["a", "b", "b", None, "c", "a", "c", "e"]
    r = classification_report(yt, yp)
    yp2 = [p or "(none)" for p in yp]
    P, R, F, _ = precision_recall_fscore_support(yt, yp2, labels=["a", "b", "c", "d"], average="macro", zero_division=0)
    assert np.isclose(r["accuracy"], accuracy_score(yt, yp2))
    assert np.allclose([r["macro_precision"], r["macro_recall"], r["macro_f1"]], [P, R, F])
    assert r["none_rate"] == 1 / 8 and r["per_class"]["e"]["fp"] == 1 and r["confusion"]["b"]["(none)"] == 1


def test_translation_metrics():
    refs = ["今日は雨です。", "コーヒーが嫌いです。"]
    assert bleu4(refs, refs) == 100 and chrf(refs, refs) == 100 and exact(refs, refs) == 1
    hyp = ["今日は雪です。", "ビールが嫌いです。"]
    assert 0 < bleu4(hyp, refs) < 100 and 0 < chrf(hyp, refs) < 100 and exact(hyp, refs) == 0
    assert wer([["a", "b"]], [["a", "c", "b"]]) == pytest.approx(1 / 3)


def test_corpus_integrity():
    entries = build()
    vocab = set(load_label_map().values())
    assert all(set(e["signs"]) <= vocab for e in entries)
    assert len({tuple(e["signs"]) for e in entries}) == len(entries)            # one sentence per sign sequence
    train, test = split(entries)
    train_signs = {s for e in train for s in e["signs"]}
    assert test and all(set(e["signs"]) <= train_signs for e in test)          # test = unseen combinations only
    assert not {e["ja"] for e in train} & {e["ja"] for e in test}
    v = CharVocab([e["ja"] for e in entries])
    assert v.decode(v.encode(entries[0]["ja"])[1:]) == entries[0]["ja"]


def test_glossfree_model_shapes():
    from src.glossfree.model import GlossFreeSLT
    from src.glossfree.stitch import batch
    m = GlossFreeSLT(30).eval()
    x, pad = batch([np.zeros((40, MODEL_IN), np.float32), np.zeros((25, MODEL_IN), np.float32)])
    logits = m(torch.from_numpy(x), torch.from_numpy(pad), torch.ones(2, 7, dtype=torch.long))
    assert logits.shape == (2, 7, 30)
    out = m.greedy(torch.from_numpy(x), torch.from_numpy(pad), 1, 2, max_len=5)
    assert len(out) == 2 and len(out[0]) <= 5


@pytest.mark.skipif(not (DATA / "landmarks_raw" / "00.npy").exists(), reason="landmarks not extracted")
def test_stitching():
    from src.glossfree.stitch import Stitcher, to_input
    from src.train.dataset import load_signs
    train, _ = load_signs(None)
    st = Stitcher(train)
    seq = st.stitch(["今日", "雨"], np.random.default_rng(0))
    assert seq.shape[1] == FEATURE_DIM and len(seq) > 60
    assert to_input(seq).shape[1] == MODEL_IN


def test_live_segmenter_detects_a_sentence():
    from src.glossfree import live
    seg = live.GlossFreeLive()
    v = np.zeros(FEATURE_DIM, np.float32)
    v[PRESENT.start + 1] = 1
    states = []
    for i in range(150):                                  # 2 s of moving right hand, then 2 s still
        t = i / 30
        w = v.copy()
        w[POS_R.start] = 0.6 * np.sin(4 * t) if t < 2 else 0.0
        states.append(seg.push(t, w)["state"])
    assert "signing" in states
    assert states[-1] in ("done", "unavailable", "idle") and ("done" in states or "unavailable" in states)


def test_metrics_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.app import store
    from src.inference.server import app
    monkeypatch.setattr(store, "DB", tmp_path / "app.db")
    c = TestClient(app)
    assert c.get("/api/metrics/realtime").json()["report"] is None
    for t, p in (("今日", "今日"), ("雨", None), ("犬", "雨")):
        c.post("/api/tests", json={"run": "r", "target": t, "predicted": p, "latency_s": 1.0})
    r = c.get("/api/metrics/realtime").json()["report"]
    assert r["n"] == 3 and np.isclose(r["accuracy"], 1 / 3) and r["none_rate"] == pytest.approx(1 / 3)
    assert c.get("/api/metrics/offline").status_code == 200
    assert c.get("/api/metrics.csv").status_code == 200
    assert "trained" in c.get("/api/glossfree/status").json()
    assert c.get("/eval").status_code == 200
