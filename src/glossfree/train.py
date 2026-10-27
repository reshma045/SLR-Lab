"""Train + evaluate the gloss-free translation model; compare with the gloss-based pipeline.

Test set: sentences never seen in training (new word combinations), each signed 3 times with fresh
stitching. Reported: BLEU-4, chrF, exact match (gloss-free), plus seen-sentence exact match, and for
the gloss-based pipeline (isolated-sign classifier + decision logic) the sign error rate (WER) and
how often it recovers the exact sign sequence.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from src.common.config import MODELS, NATIVE_FPS
from src.glossfree.corpus import CharVocab, build, split
from src.glossfree.metrics import bleu4, chrf, exact, wer
from src.glossfree.model import GlossFreeSLT
from src.glossfree.stitch import Stitcher, batch, to_input
from src.train.dataset import load_signs

CKPT = MODELS / "glossfree.pt"
REPORT = MODELS / "glossfree_report.json"
STATUS = MODELS / "glossfree_status.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED, TEST_SEED = 11, 4242
EPOCHS, PER_EPOCH, BATCH = 60, 1536, 32
REPEATS = 3


def _status(**kw) -> None:
    MODELS.mkdir(exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def _targets(vocab: CharVocab, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.encode(t) for t in texts]
    L = max(len(i) for i in ids)
    arr = np.full((len(ids), L), vocab.PAD, np.int64)
    for k, i in enumerate(ids):
        arr[k, :len(i)] = i
    t = torch.from_numpy(arr)
    return t[:, :-1], t[:, 1:]


def translate(model: GlossFreeSLT, vocab: CharVocab, inputs: list[np.ndarray]) -> list[str]:
    model.eval()
    out = []
    for i in range(0, len(inputs), 32):
        x, pad = batch(inputs[i:i + 32])
        ids = model.greedy(torch.from_numpy(x).to(DEVICE), torch.from_numpy(pad).to(DEVICE), vocab.BOS, vocab.EOS)
        out += [vocab.decode(r) for r in ids]
    return out


def gloss_based(seqs30: list[np.ndarray]) -> list[list[str]]:
    """Run the isolated-sign recogniser's live decision path over each stitched sentence."""
    from src.inference.predictor import JSLPredictor
    pred = JSLPredictor()
    out = []
    try:
        for seq in seqs30:
            pred.reset()
            emitted = []
            padded = np.concatenate([np.repeat(seq[:1], 30, 0), seq, np.repeat(seq[-1:], 45, 0)])
            for i, v in enumerate(padded):
                e = pred.push_vector(v, t=i / NATIVE_FPS)
                if e:
                    emitted.append(e)
            out.append(emitted)
    finally:
        pred.close()
    return out


def train_glossfree(profile: str | None = "default", epochs: int = EPOCHS) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    entries = build()
    train_set, test_set = split(entries)
    vocab = CharVocab([e["ja"] for e in entries])
    signs, real = load_signs(profile)
    stitcher = Stitcher(signs + real)
    print(f"Gloss-free: {len(train_set)} training sentences, {len(test_set)} unseen test sentences, "
          f"{len(vocab.chars)} characters, device {DEVICE}")

    trng = np.random.default_rng(TEST_SEED)
    test_seq = [(e, stitcher.stitch(e["signs"], trng, aug=False)) for e in test_set for _ in range(REPEATS)]
    seen = [train_set[i] for i in trng.choice(len(train_set), size=min(30, len(train_set)), replace=False)]
    seen_seq = [(e, stitcher.stitch(e["signs"], trng, aug=False)) for e in seen]

    model = GlossFreeSLT(len(vocab.chars)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=0.05)
    steps = epochs * (PER_EPOCH // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-4, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.PAD, label_smoothing=0.1)
    t0 = time.time()
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for _ in range(PER_EPOCH // BATCH):
            es = [train_set[i] for i in rng.integers(len(train_set), size=BATCH)]
            x, pad = batch([to_input(stitcher.stitch(e["signs"], rng)) for e in es])
            tin, tout = _targets(vocab, [e["ja"] for e in es])
            logits = model(torch.from_numpy(x).to(DEVICE), torch.from_numpy(pad).to(DEVICE), tin.to(DEVICE))
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tout.to(DEVICE).reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        row = {"epoch": ep, "loss": round(tot / (PER_EPOCH // BATCH), 4)}
        if ep % 10 == 0 or ep == epochs:
            hyp = translate(model, vocab, [to_input(s) for _, s in seen_seq])
            row["seen_exact"] = round(exact(hyp, [e["ja"] for e, _ in seen_seq]), 3)
        history.append(row)
        _status(state="training", epoch=ep, max_epochs=epochs, history=history, elapsed=round(time.time() - t0, 1))
        print(f"epoch {ep:3d}  loss {row['loss']:.4f}" + (f"  seen-sentence exact {row['seen_exact']:.2f}" if "seen_exact" in row else "")
              + f"  [{time.time() - t0:.0f}s]", flush=True)

    _status(state="evaluating", epoch=epochs, max_epochs=epochs, history=history)
    refs = [e["ja"] for e, _ in test_seq]
    hyps = translate(model, vocab, [to_input(s) for _, s in test_seq])
    seen_h = translate(model, vocab, [to_input(s) for _, s in seen_seq])
    gb = gloss_based([s for _, s in test_seq])
    true_signs = [e["signs"] for e, _ in test_seq]
    metrics = {"bleu4": bleu4(hyps, refs), "chrf": chrf(hyps, refs), "exact": exact(hyps, refs),
               "seen_exact": exact(seen_h, [e["ja"] for e, _ in seen_seq]),
               "gloss_based_wer": wer(gb, true_signs),
               "gloss_based_exact_signs": float(np.mean([g == t for g, t in zip(gb, true_signs)]))}
    examples = [{"signs": e["signs"], "reference": e["ja"], "english": e["en"], "gloss_free": h, "gloss_based_signs": g}
                for (e, _), h, g in zip(test_seq, hyps, gb)]
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "chars": vocab.chars,
                "dim": 192}, CKPT)
    report = {"trained": time.strftime("%Y-%m-%d %H:%M:%S"), "profile": profile, "device": str(DEVICE),
              "epochs": epochs, "train_seconds": round(time.time() - t0, 1),
              "train_sentences": len(train_set), "test_sentences": len(test_set), "test_items": len(test_seq),
              "metrics": metrics, "examples": examples,
              "description": ("Gloss-free model trained only on (motion, Japanese sentence) pairs synthesised by stitching "
                              "real recordings of the 50 signs (PoseStitch-SLT idea). Test = sentences never seen in "
                              "training. Same signer as training, and outputs are limited to this sentence space.")}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _status(state="done", metrics=metrics)
    return report
