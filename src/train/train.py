"""Training loop, held-out evaluation and reporting for the JSL transformer."""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from src.common.config import MODELS, NUM_CLASSES, class_names
from src.train.dataset import WindowSampler, load_signs
from src.train.model import JSLTransformer

SEED = 7
VAL_SEED = 12345           # validation samples use a seed never used for training
BATCH = 32
MAX_EPOCHS = 200
PATIENCE = 20

HONESTY = ("NOTE: validation uses held-out augmentation seeds and speed bands of the SAME 25 "
           "recordings (one signer, one example per sign).\n      Val accuracy is therefore "
           "OPTIMISTIC - the real test is live webcam use by other people.")


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    """Loss, top-1/top-3 accuracy, per-class F1 and predictions on a fixed set."""
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(torch.from_numpy(x[i:i + 256])) for i in range(0, len(x), 256)])
    yt = torch.from_numpy(y)
    loss = nn.functional.cross_entropy(logits, yt).item()
    top3 = logits.topk(3, dim=1).indices
    pred = top3[:, 0].numpy()
    return {"loss": loss, "top1": float((pred == y).mean()),
            "top3": float((top3 == yt[:, None]).any(1).float().mean()),
            "f1": f1_score(y, pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0),
            "pred": pred}


def save_confusion(y: np.ndarray, pred: np.ndarray, names: list[str]) -> None:
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
    ax.set_title("Validation confusion (held-out augmentation; optimistic)")
    fig.tight_layout()
    fig.savefig(MODELS / "confusion.png", dpi=110)
    plt.close(fig)


def train() -> tuple[JSLTransformer, dict]:
    """Train with early stopping; save models/best.pt; return (best model, val metrics)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    MODELS.mkdir(exist_ok=True)
    names = class_names()
    signs = load_signs()
    print("Sign cores (frames at 30 fps):",
          ", ".join(f"{names[s.label]} {s.off - s.on}" for s in signs))
    xv, yv = WindowSampler(signs, per_class=40, val=True).epoch(np.random.default_rng(VAL_SEED))
    print(f"Validation set: {len(xv)} samples (held-out seed {VAL_SEED}, speed bands 0.85-0.90 / 1.20-1.25)")
    print(HONESTY)

    model = JSLTransformer()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    sampler = WindowSampler(signs, per_class=96)
    best, best_loss, bad = None, float("inf"), 0
    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        x, y = sampler.epoch(rng)
        perm = rng.permutation(len(x))
        model.train()
        tot = 0.0
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = torch.from_numpy(x[idx]), torch.from_numpy(y[idx])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        m = evaluate(model, xv, yv)
        sched.step(m["loss"])
        improved = m["loss"] < best_loss - 1e-4
        if improved:
            best_loss, bad = m["loss"], 0
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best, "epoch": epoch, "val": {k: m[k] for k in ("loss", "top1", "top3")}},
                       MODELS / "best.pt")
        else:
            bad += 1
        print(f"epoch {epoch:3d}  train_loss {tot / len(x):.4f}  val_loss {m['loss']:.4f}  "
              f"val_top1 {m['top1']:.3f}  val_top3 {m['top3']:.3f}  lr {opt.param_groups[0]['lr']:.1e}"
              f"{'  *' if improved else ''}  [{time.time() - t0:.0f}s]", flush=True)
        if bad >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no val-loss improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best)
    m = evaluate(model, xv, yv)
    save_confusion(yv, m["pred"], names)
    report = {"val_top1": m["top1"], "val_top3": m["top3"], "val_loss": m["loss"],
              "per_class_f1": {("idle" if i == NUM_CLASSES - 1 else names[i]): round(float(f), 4)
                               for i, f in enumerate(m["f1"])},
              "epochs_trained": epoch, "train_seconds": round(time.time() - t0, 1),
              "note": "Optimistic: validation is held-out augmentation of the training recordings."}
    with open(MODELS / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== Best model on validation ===")
    print(f"top-1 {m['top1']:.4f}   top-3 {m['top3']:.4f}   loss {m['loss']:.4f}")
    print("per-class F1:")
    for k, v in report["per_class_f1"].items():
        print(f"  {k:<10} {v:.3f}")
    print(HONESTY)
    return model, report
