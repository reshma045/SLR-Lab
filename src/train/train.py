"""Training loop, evaluation and reporting for the JSL classifier (GPU if available).

Recipe adapted from the 1st-place Kaggle ISLR solution: Conv1D+Transformer, label smoothing 0.1,
late dropout and adversarial weight perturbation switched on after a warm-up, one-cycle cosine LR.

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
from src.train.model import AWP, JSLNet

SEED = 7
VAL_SEED = 12345
BATCH = 64
EPOCHS = 120
WARMUP_TRICKS = 15          # epoch at which late dropout + AWP switch on
PATIENCE = 40
STATUS = MODELS / "train_status.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HONESTY = ("NOTE: 'synthetic val' re-uses the training recordings with held-out augmentation, so it is "
           "OPTIMISTIC.\n      'real val' (your newest recording of each sign, never trained on) is the "
           "honest number; record >= 3 takes per sign in the dashboard to get it.")


def write_status(**kw) -> None:
    """Atomically update models/train_status.json for the dashboard."""
    MODELS.mkdir(exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps({**kw, "updated": time.time()}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATUS)


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict:
    """Loss, top-1/top-3 accuracy, per-class F1 and predictions on a fixed set."""
    model.eval()
    logits = torch.cat([model(torch.from_numpy(x[i:i + 256]).to(DEVICE))[0].float().cpu()
                        for i in range(0, len(x), 256)])
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
    for name in ("jsl_model.onnx", "meta.json", "train_report.json", "confusion.png", "prototypes.npz"):
        if (MODELS / name).exists():
            prev.mkdir(exist_ok=True)
            shutil.copy2(MODELS / name, prev / name)


def train(profile: str | None = "default", max_epochs: int = EPOCHS) -> tuple[JSLNet, dict]:
    """Train; keep the best checkpoint (by real val if available, else synthetic); return (model, report)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    MODELS.mkdir(exist_ok=True)
    names = class_names()
    signs, real = load_signs(profile)
    n_user = sum(s.source == "user" for s in signs) + len(real)
    n_ref = sum(s.source == "ref" for s in signs)
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})")
    print(f"Training recordings: {n_ref} reference (incl. simulated webcam crops) + {n_user} of yours "
          f"(profile '{profile}'); real-val recordings held out: {len(real)}")
    xv, yv = WindowSampler(signs, per_class=40, val=True).epoch(np.random.default_rng(VAL_SEED))
    xr, yr = real_val_set(real, np.random.default_rng(VAL_SEED)) if real else (None, None)
    print(HONESTY)

    model = JSLNet().to(DEVICE)
    print(f"Model: JSLNet {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters")
    sampler = WindowSampler(signs, per_class=96)
    steps = (len(sampler.by_class) * 96 + sampler.idle_n + BATCH - 1) // BATCH
    opt = torch.optim.RAdam(model.parameters(), lr=1e-3, weight_decay=0.05, decoupled_weight_decay=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, epochs=max_epochs, steps_per_epoch=steps,
                                                pct_start=0.1, anneal_strategy="cos")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    awp = AWP(model)
    best, best_score, bad, epoch = None, float("inf"), 0, 0
    t0 = time.time()
    history = []
    for epoch in range(1, max_epochs + 1):
        tricks = epoch > WARMUP_TRICKS
        model.late_on = tricks
        x, y = sampler.epoch(rng)
        perm = rng.permutation(len(x))
        model.train()
        tot = 0.0
        for i in range(0, len(x), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            xb, yb = torch.from_numpy(x[idx]).to(DEVICE), torch.from_numpy(y[idx]).to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb)[0], yb)
            loss.backward()
            if tricks:
                awp.perturb()
                loss_fn(model(xb)[0], yb).backward()
                awp.restore()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            tot += loss.item() * len(idx)
        m = evaluate(model, xv, yv)
        r = evaluate(model, xr, yr) if xr is not None else None
        score = r["loss"] if r else m["loss"]
        improved = score < best_score - 1e-4
        if improved:
            best_score, bad = score, 0
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"state_dict": best, "epoch": epoch}, MODELS / "best.pt")
        else:
            bad += 1
        row = {"epoch": epoch, "train_loss": round(tot / len(x), 4), "val_top1": round(m["top1"], 4),
               "real_top1": round(r["top1"], 4) if r else None}
        history.append(row)
        write_status(state="training", epoch=epoch, max_epochs=max_epochs, history=history[-400:],
                     elapsed=round(time.time() - t0, 1), device=str(DEVICE))
        print(f"epoch {epoch:3d}  train_loss {row['train_loss']:.4f}  val_loss {m['loss']:.4f}  "
              f"synthetic_top1 {m['top1']:.3f}" + (f"  REAL_top1 {r['top1']:.3f}" if r else "")
              + f"  lr {opt.param_groups[0]['lr']:.1e}{'  *' if improved else ''}  [{time.time() - t0:.0f}s]",
              flush=True)
        if bad >= PATIENCE and epoch > WARMUP_TRICKS + PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best)
    model.late_on = False
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
              "per_class_f1_source": "real" if r else "synthetic", "model": "JSLNet (Conv1D+Transformer)",
              "device": str(DEVICE), "epochs_trained": epoch, "train_seconds": round(time.time() - t0, 1),
              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(MODELS / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n=== Best model ===")
    print(f"synthetic val: top-1 {m['top1']:.4f}  top-3 {m['top3']:.4f}   (optimistic)")
    if r:
        print(f"REAL val     : top-1 {r['top1']:.4f}  top-3 {r['top3']:.4f}   ({len(real)} held-out recordings)")
    print(HONESTY)
    model.cpu().eval()
    return model, report
