"""Classification metrics for sign recognition: accuracy, precision, recall, F1 (per class, macro,
weighted) and a confusion table.

Predictions may be None ("nothing recognised"): that counts as a miss for the true sign's recall
and is not a false positive for any sign. Macro averages are over the signs that occur in y_true.
"""
from __future__ import annotations

from collections import Counter, defaultdict

NONE = "(none)"


def classification_report(y_true: list[str], y_pred: list[str | None]) -> dict:
    """Return {n, accuracy, macro/weighted precision/recall/f1, per_class, confusion, none_rate}."""
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    pred = [p if p is not None else NONE for p in y_pred]
    labels = sorted(set(y_true))
    tp, fp, fn = Counter(), Counter(), Counter()
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, pred):
        confusion[t][p] += 1
        if p == t:
            tp[t] += 1
        else:
            fn[t] += 1
            if p != NONE:
                fp[p] += 1
    support = Counter(y_true)
    per_class = {}
    for c in sorted(set(labels) | {p for p in pred if p != NONE}):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / support[c] if support[c] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support[c],
                        "tp": tp[c], "fp": fp[c], "fn": fn[c]}
    in_true = [per_class[c] for c in labels]
    macro = {k: (sum(r[k] for r in in_true) / len(in_true) if in_true else 0.0) for k in ("precision", "recall", "f1")}
    weighted = {k: (sum(r[k] * r["support"] for r in in_true) / n if n else 0.0) for k in ("precision", "recall", "f1")}
    return {
        "n": n,
        "accuracy": sum(tp.values()) / n if n else 0.0,
        "macro_precision": macro["precision"], "macro_recall": macro["recall"], "macro_f1": macro["f1"],
        "weighted_precision": weighted["precision"], "weighted_recall": weighted["recall"],
        "weighted_f1": weighted["f1"],
        "none_rate": pred.count(NONE) / n if n else 0.0,
        "per_class": per_class,
        "confusion": {t: dict(v) for t, v in confusion.items()},
    }


def summary(report: dict) -> dict:
    """The headline numbers only (for tables and CSV)."""
    keys = ("n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "none_rate")
    return {k: report[k] for k in keys}
