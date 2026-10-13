"""Training loop, evaluation and reporting for the JSL transformer.

Two validation sets:
  * synthetic - held-out augmentation seeds / speed bands of the training recordings (OPTIMISTIC)
  * real      - the newest recording per sign from the user's own data, never trained on (HONEST);
                only exists once the user has >= 3 recordings of a sign.
Progress is written to models/train_status.json so the dashboard can show it live.
"""
from __future__ import annotations

import json
import shutil
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from src.common.config import MODELS, NUM_CLASSES, NUM_SIGNS, class_names
from src.train.dataset import WindowSampler, load_signs, real_val_set
from src.train.model import JSLTransformer

SEED = 7
VAL_SEED = 12345
BATCH = 32
MAX_EPOCHS = 200
PATIENCE = 20
STATUS = MODELS / "train_status.json"

HONESTY = ("NOTE: 'synthetic val' re-uses the training recordings with held-out augmentation, so it is "
           "OPTIMISTIC.\n      'real val' (your newest recording of each sign, never trained on) is the "
           "honest number; record >= 3 takes per sign in the dashboard to get it.")


def write_status(**kw) -> None:
    """Atomically update models/train_status.json for the dashboard."""
    MODELS.mkdir(exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    """Loss, top-1/top-3 accuracy, per-class F1 and predictions on a fixed set."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(torch.from_numpy(x[i:i + 256])) for i in range(0, len(x), 256)])
    yt = torch.from_numpy(y)
    top3 = logits.topk(3, dim=1).indices
    pred = top3[:, 0].numpy()
    return {"loss": nn.functional.cross_entropy(logits, yt).item(),
            "top1": float((pred == y).mean()),
            "top3": float((top3 == yt[:, None]).any(1).float().mean()),
            "f1": f1_score(y, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0),
            "pred": pred}


def save_confusion(y: np.ndarray, pred: np.ndarray, names: list[str], title: str) -> None:
    """Write models/confusion.png (row-normalised)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fam in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "Hiragino Sans"):
        if any(fam in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fam
            break
    cm = confusion_matrix(y, pred, labels=list(range(NUM_CLASSES))).astype(float)
    cm /= np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    labels = names[:-1] + ["idle"]
    ax.set_xticks(range(NUM_CLASSES), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(NUM_CLASSES), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(MODELS / "confusion.png", dpi=110)
    plt.close(fig)


def _backup_previous() -> None:
    """Keep the previous model so a bad retrain can be rolled back from the dashboard."""
    prev = MODELS / "previous"
    for name in ("jsl_model.onnx", "meta.json", "train_report.json", "confusion.png"):
        if (MODELS / name).exists():
            prev.mkdir(exist_ok=True)
            shutil.copy2(MODELS / name, prev / name)


def train(profile: str | None = "default", max_epochs: int = MAX_EPOCHS) -> tuple[JSLTransformer, dict]:
    """Train with early stopping; save models/best.pt; return (best model, report)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    MODELS.mkdir(exist_ok=True)
    names = class_names()
    signs, real = load_signs(profile)
    n_user = sum(s.source == "user" for s in signs) + len(real)
    n_ref = sum(s.source == "ref" for s in signs)
    print(f"Training recordings: {n_ref} reference (incl. simulated webcam crops) + {n_user} of yours "
          f"(profile '{profile}'); real-val recordings held out: {len(real)}")
    xv, yv = WindowSampler(signs, per_class=40, val=True).epoch(np.random.default_rng(VAL_SEED))
    xr, yr = real_val_set(real, np.random.default_rng(VAL_SEED)) if real else (None, None)
    print(HONESTY)

    model = JSLTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    sampler = WindowSampler(signs, per_class=96)
    best, best_loss, bad, epoch = None, float("inf"), 0, 0
    t0 = time.time()
    history = []
    for epoch in range(1, max_epochs + 1):
        x, y = sampler.epoch(rng)
        perm = rng.permutation(len(x))
        model.train()
        tot = 0.0
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(torch.from_numpy(x[idx])), torch.from_numpy(y[idx]))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        m = evaluate(model, xv, yv)
        r = evaluate(model, xr, yr) if xr is not None else None
        sched.step(m["loss"])
        improved = m["loss"] < best_loss - 1e-4
        if improved:
            best_loss, bad = m["loss"], 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best, "epoch": epoch}, MODELS / "best.pt")
        else:
            bad += 1
        row = {"epoch": epoch, "train_loss": round(tot / len(x), 4), "val_top1": round(m["top1"], 4),
               "real_top1": round(r["top1"], 4) if r else None}
        history.append(row)
        write_status(state="training", epoch=epoch, max_epochs=max_epochs, history=history[-200:],
                     elapsed=round(time.time() - t0, 1))
        print(f"epoch {epoch:3d}  train_loss {row['train_loss']:.4f}  val_loss {m['loss']:.4f}  "
              f"synthetic_top1 {m['top1']:.3f}" + (f"  REAL_top1 {r['top1']:.3f}" if r else "")
              + f"  lr {opt.param_groups[0]['lr']:.1e}{'  *' if improved else ''}  [{time.time() - t0:.0f}s]",
              flush=True)
        if bad >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val-loss improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best)
    m = evaluate(model, xv, yv)
    r = evaluate(model, xr, yr) if xr is not None else None
    shown, ys, title = (r, yr, "REAL validation: your held-out recordings") if r else \
        (m, yv, "Synthetic validation (held-out augmentation; optimistic)")
    _backup_previous()
    save_confusion(ys, shown["pred"], names, title)
    report = {"synthetic_val_top1": m["top1"], "synthetic_val_top3": m["top3"],
              "real_val_top1": r["top1"] if r else None, "real_val_top3": r["top3"] if r else None,
              "real_val_recordings": len(real), "user_recordings": n_user, "profile": profile,
              "per_class_f1": {("idle" if i == NUM_CLASSES - 1 else names[i]): round(float(f), 4)
                               for i, f in enumerate(shown["f1"])},
              "per_class_f1_source": "real" if r else "synthetic",
              "epochs_trained": epoch, "train_seconds": round(time.time() - t0, 1),
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(MODELS / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== Best model ===")
    print(f"synthetic val: top-1 {m['top1']:.4f}  top-3 {m['top3']:.4f}   (optimistic)")
    if r:
        print(f"REAL val     : top-1 {r['top1']:.4f}  top-3 {r['top3']:.4f}   ({len(real)} held-out recordings)")
    print(HONESTY)
    return model, report
