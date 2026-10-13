"""One-click training: python run_training.py -> models/best.pt, jsl_model.onnx, meta.json, confusion.png."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.common.config import DATA, MODELS  # noqa: E402


def main() -> None:
    if not (DATA / "landmarks_full").exists():
        sys.exit("data/landmarks_full/ missing - run: python scripts/extract_landmarks.py")
    from src.train.export import export_onnx, write_meta
    from src.train.train import train

    model, report = train()
    diff = export_onnx(model)
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"\nONNX export -> {MODELS / 'jsl_model.onnx'}  parity max|diff| = {diff:.2e}  [{status}]")
    write_meta(report)
    print(f"Saved {MODELS / 'meta.json'}, {MODELS / 'confusion.png'}, {MODELS / 'train_report.json'}")
    if diff >= 1e-4:
        sys.exit(1)


if __name__ == "__main__":
    main()
