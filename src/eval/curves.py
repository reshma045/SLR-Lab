"""Learning curves (per-epoch training history) of the sign classifier and the gloss-free model.

Histories are stored in models/train_report.json and models/glossfree_report.json ("history").
`backfill_from_logs` recovers them from the training logs for models trained before that existed.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.common.config import MODELS, ROOT

WORD_REPORT = MODELS / "train_report.json"
GF_REPORT = MODELS / "glossfree_report.json"
_WORD = re.compile(r"epoch\s+(\d+)\s+train_loss ([\d.]+)\s+val_loss ([\d.]+)\s+synthetic_top1 ([\d.]+)"
                   r"(?:\s+REAL_top1 ([\d.]+))?")
_GF = re.compile(r"epoch\s+(\d+)\s+loss ([\d.]+)(?:\s+seen-sentence exact ([\d.]+))?")


def parse_word_log(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _WORD.search(line)
        if m:
            out.append({"epoch": int(m[1]), "train_loss": float(m[2]), "val_loss": float(m[3]),
                        "val_top1": float(m[4]), "real_top1": float(m[5]) if m[5] else None})
    return out


def parse_gf_log(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = _GF.search(line)
        if m:
            out.append({"epoch": int(m[1]), "loss": float(m[2]), "seen_exact": float(m[3]) if m[3] else None})
    return out


def _inject(report: Path, history: list[dict]) -> bool:
    if not report.exists() or not history:
        return False
    rep = json.loads(report.read_text(encoding="utf-8"))
    if rep.get("history"):
        return False
    rep["history"] = history
    report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def backfill_from_logs() -> dict:
    """Add 'history' to existing reports from artifacts_*.log / models/*.log if missing."""
    done = {}
    for rep, logs, parse in ((WORD_REPORT, ["artifacts_train.log", "models/train.log"], parse_word_log),
                             (GF_REPORT, ["artifacts_glossfree.log", "models/glossfree.log"], parse_gf_log)):
        for log in logs:
            p = ROOT / log
            if p.exists():
                h = parse(p)
                if h:
                    done[rep.name] = _inject(rep, h)
                    break
    return done


def curves() -> dict:
    """{"word": {...}, "glossfree": {...}} with histories and context for the dashboard."""
    out = {}
    for key, rep in (("word", WORD_REPORT), ("glossfree", GF_REPORT)):
        if rep.exists():
            r = json.loads(rep.read_text(encoding="utf-8"))
            out[key] = {"history": r.get("history", []), "trained": r.get("finished") or r.get("trained"),
                        "epochs": r.get("epochs_trained") or r.get("epochs"), "device": r.get("device")}
    return out


def plot(path: Path) -> Path:
    """Save a 3-panel learning-curve figure (loss, accuracy, gloss-free loss) to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = curves()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    w = c.get("word", {}).get("history", [])
    if w:
        e = [h["epoch"] for h in w]
        ax[0].plot(e, [h["train_loss"] for h in w], label="training loss")
        ax[0].plot(e, [h["val_loss"] for h in w], label="validation loss")
        ax[0].set_title("Sign classifier: loss")
        ax[1].plot(e, [h["val_top1"] for h in w], label="validation accuracy (synthetic, optimistic)")
        real = [(h["epoch"], h["real_top1"]) for h in w if h.get("real_top1") is not None]
        if real:
            ax[1].plot(*zip(*real), label="validation accuracy (your held-out takes)")
        ax[1].set_ylim(0, 1.02)
        ax[1].set_title("Sign classifier: accuracy")
        for a in ax[:2]:
            a.axvline(15, color="grey", ls=":", lw=1)
            a.text(15.5, a.get_ylim()[1] * 0.95, "late dropout + AWP on", fontsize=8, color="grey", va="top")
    g = c.get("glossfree", {}).get("history", [])
    if g:
        ax[2].plot([h["epoch"] for h in g], [h["loss"] for h in g], color="tab:purple", label="training loss")
        ax[2].set_title("Gloss-free translation: loss")
    for a in ax:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        if a.lines:
            a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
