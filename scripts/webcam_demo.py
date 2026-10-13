"""Standalone OpenCV webcam demo (no browser): camera -> MediaPipe -> Transformer [-> Claude].

    python scripts/webcam_demo.py                  # window; press q / Esc to quit, c = ask Claude
    python scripts/webcam_demo.py --camera 1       # another camera
    python scripts/webcam_demo.py --headless --seconds 20 --snapshot artifacts/cam.jpg   # smoke test
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.inference.claude_verifier import ClaudeVerifier  # noqa: E402
from src.inference.predictor import JSLPredictor  # noqa: E402

FONT_CANDIDATES = ["C:/Windows/Fonts/YuGothB.ttc", "C:/Windows/Fonts/meiryo.ttc",
                   "C:/Windows/Fonts/msgothic.ttc", "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]


def load_font(size: int):
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


class Overlay:
    """Draws prediction, confidence, history and Claude verdict (Japanese-capable via PIL)."""

    def __init__(self) -> None:
        self.big, self.small = load_font(44), load_font(18)

    def draw(self, img: np.ndarray, p: dict, history: list[str], claude: dict) -> np.ndarray:
        h, w = img.shape[:2]
        canvas = np.zeros((h + 130, w, 3), np.uint8)
        canvas[:h] = img
        cv2.rectangle(canvas, (0, h), (w, h + 130), (23, 17, 13), -1)
        fill = int(w * p["buffer_fill"])
        cv2.rectangle(canvas, (0, h), (fill, h + 4), (246, 130, 59), -1)
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        pred = p["prediction"]
        sign = pred["sign"] or "—"
        col = (230, 237, 243) if pred["is_confident"] else (120, 128, 138)
        d.text((14, h + 12), sign, font=self.big, fill=col)
        info = f"{pred['english'] or 'no sign'}   {pred['confidence']:.0%}" if pred["sign"] else "no sign"
        d.text((14, h + 68), info, font=self.small, fill=(139, 148, 158))
        hands = " + ".join(x for x, on in (("L", p["left_hand"]), ("R", p["right_hand"])) if on) or "no hands"
        d.text((w - 260, h + 12), f"{p['mode']} · {hands} · {p['fps_server']:.0f} fps", font=self.small,
               fill=(139, 148, 158))
        d.text((w - 260, h + 40), "History: " + " ".join(history[-4:]), font=self.small, fill=(230, 237, 243))
        if claude.get("status") in ("checking", "done", "error"):
            msg = {"checking": "Claude: checking…", "error": f"Claude: {claude.get('error')}",
                   "done": f"Claude: {claude.get('sign') or 'none'} ({claude.get('confidence')})"}[claude["status"]]
            d.text((14, h + 98), msg, font=self.small, fill=(96, 165, 250))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = until q)")
    ap.add_argument("--headless", action="store_true", help="no window; print results")
    ap.add_argument("--snapshot", type=Path, help="save the last annotated frame here")
    ap.add_argument("--no-claude", action="store_true", help="never call Claude")
    args = ap.parse_args()

    # On Windows, Media Foundation delivers ~3x the frame rate of DirectShow on typical webcams.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY)
    if not cap.isOpened() and sys.platform == "win32":
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f"Could not open camera {args.camera}")
        return 1
    pred = JSLPredictor()
    verifier = None if args.no_claude else ClaudeVerifier()
    use_claude = bool(verifier and verifier.available)
    print(f"Camera {args.camera} open · mode={pred.mode} · Claude={'on' if use_claude else 'off'}")
    overlay, history, claude = Overlay(), [], {}
    stats = {"frames": 0, "hand_frames": 0}

    def ask(trigger: str) -> None:
        if not use_claude or claude.get("status") == "checking":
            return
        frames, top3 = pred.clip_frames(), pred.current()[1]
        claude.clear()
        claude["status"] = "checking"

        def run() -> None:
            res = verifier.verify(frames, top3)
            claude.clear()
            claude.update(res)
            print(f"  Claude ({trigger}): {res.get('sign') or res.get('error') or 'none'} "
                  f"[{res.get('confidence', '')}] {res.get('latency_s', '')}s - {res.get('reason', '')}")
        threading.Thread(target=run, daemon=True).start()

    t0, last = time.time(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed")
                break
            p = pred.process_frame(frame, encode=False)
            stats["frames"] += 1
            stats["hand_frames"] += p["left_hand"] or p["right_hand"]
            if p["emitted"]:
                history.append(p["emitted"])
                print(f"[{time.time() - t0:6.1f}s] SIGN: {p['emitted']} ({p['prediction']['english']}, "
                      f"{p['prediction']['confidence']:.0%})")
                ask(p["emitted"])
            last = (frame, p)
            if not args.headless:
                shown = cv2.flip(pred.last_annotated, 1)                   # mirror for display only
                cv2.imshow("JSL recognition (q to quit, c = ask Claude)", overlay.draw(shown, p, history, claude))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("c"):
                    ask("manual")
            if args.seconds and time.time() - t0 >= args.seconds:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    dur = time.time() - t0
    print(f"\n{stats['frames']} frames in {dur:.1f}s ({stats['frames'] / max(dur, 1e-6):.1f} fps), "
          f"hands visible in {stats['hand_frames'] / max(1, stats['frames']):.0%} of frames, "
          f"emitted: {history or 'nothing'}")
    if args.snapshot and last is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.snapshot), overlay.draw(cv2.flip(pred.last_annotated, 1), last[1], history, claude))
        print(f"Snapshot: {args.snapshot}")
    pred.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
