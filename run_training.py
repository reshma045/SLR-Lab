"""One-click training: python run_training.py [--profile NAME] [--epochs N]

Writes models/jsl_model.onnx, meta.json, confusion.png, train_report.json (previous model kept in
models/previous/). Uses the 25 reference videos plus your recordings in data/user/<profile>/.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.common.config import DATA, MODELS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="default", help="user recordings to include ('' = none)")
    ap.add_argument("--epochs", type=int, default=120)
    args = ap.parse_args()
    if not (DATA / "landmarks_raw").exists():
        sys.exit("data/landmarks_full/ missing - run: python scripts/extract_landmarks.py")
    from src.train.export import export_onnx, write_meta
    from src.train.train import train, write_status

    try:
        write_status(state="starting", epoch=0, max_epochs=args.epochs, history=[])
        model, report = train(args.profile or None, args.epochs)
        write_status(state="exporting", epoch=report["epochs_trained"], max_epochs=args.epochs, history=[])
        diff = export_onnx(model)
        ok = diff < 1e-4
        print(f"\nONNX export -> {MODELS / 'jsl_model.onnx'}  parity max|diff| = {diff:.2e}  [{'PASS' if ok else 'FAIL'}]")
        write_meta(report)
        if ok and args.profile:
            import onnxruntime as ort
            from src.train import prototypes
            sess = ort.InferenceSession(str(MODELS / "jsl_model.onnx"), providers=["CPUExecutionProvider"])
            n = prototypes.rebuild(sess, args.profile)
            print(f"Personal examples (few-shot prototypes) rebuilt: {n} takes")
        write_status(state="done" if ok else "failed", report=report, parity=diff,
                     error=None if ok else "ONNX parity check failed")
        return 0 if ok else 1
    except Exception as exc:  # noqa: BLE001 - surface any failure to the dashboard
        traceback.print_exc()
        write_status(state="failed", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
