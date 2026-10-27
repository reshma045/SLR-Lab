"""Offline evaluation: accuracy, precision, recall and F1 of the full live pipeline on recorded data.

    python scripts/evaluate.py                         # all sets (reference videos x3 + your held-out takes)
    python scripts/evaluate.py --sets ref_30fps user_takes
    python scripts/evaluate.py --profile alice

Writes models/eval_report.json (shown on the dashboard's Evaluation page).
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.eval.offline import SETS, _status, evaluate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default")
    ap.add_argument("--sets", nargs="*", choices=list(SETS), default=None)
    a = ap.parse_args()
    try:
        rep = evaluate(a.profile, a.sets)
    except Exception as exc:  # noqa: BLE001 - surface any failure to the dashboard
        traceback.print_exc()
        _status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    for name, s in rep["sets"].items():
        r = s["report"]
        if r is None:
            print(f"{s['title']}: skipped ({s.get('skipped')})")
            continue
        print(f"{s['title']:<42} n={r['n']:3d}  acc {r['accuracy']:.3f}  P {r['macro_precision']:.3f}  "
              f"R {r['macro_recall']:.3f}  F1 {r['macro_f1']:.3f}  none {r['none_rate']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
