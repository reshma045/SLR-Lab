"""Train the gloss-free sign-to-Japanese translation model and report BLEU-4 / chrF / exact match.

    python scripts/train_glossfree.py [--epochs 60] [--profile default]
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

from src.glossfree.train import _status, train_glossfree  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    try:
        r = train_glossfree(a.profile or None, a.epochs)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        _status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    m = r["metrics"]
    print(f"\nUnseen test sentences ({r['test_items']} signed sentences):")
    print(f"  gloss-free : BLEU-4 {m['bleu4']:.1f}  chrF {m['chrf']:.1f}  exact {m['exact']:.1%}")
    print(f"  gloss-based: sign error rate {m['gloss_based_wer']:.1%}  exact sign sequence {m['gloss_based_exact_signs']:.1%}")
    print(f"  (seen sentences, new stitching: exact {m['seen_exact']:.1%})")
    for ex in r["examples"][:8]:
        print(f"  {' '.join(ex['signs'])}  ->  {ex['gloss_free']}   (ref {ex['reference']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
