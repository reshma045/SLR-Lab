# JSL Real-Time Recognition System — Build Spec

You are building, end to end, a real-time Japanese Sign Language (JSL) recognition demo for a
Master's thesis (University of Aizu). Work autonomously until everything below runs and is
verified. Do not stop to ask questions unless something is truly blocking; make reasonable
decisions and document them in README.md.

## Background (why the design is the way it is)

- The lab originally collected data with tactile sensor gloves, but that data was poor quality, so
  the **webcam + MediaPipe pipeline is the real classifier**.
- Training data = 25 clean bare-hand JSL reference videos (one video per sign, one professional
  signer on a blue chroma-key background, 720x480, ~30 fps, ~4-5 s each, starting with a pink title
  card). That is only **one example per class**, so augmentation and train/inference consistency are
  critical.
- The UI also shows a **glove sensor panel (heat map + IMU trace)**. Real glove integration is out of
  scope; this panel is fed by a simulator behind a clean interface so real data can be dropped in
  later. **The UI must label it honestly as "SIMULATED SENSOR FEED"** (small badge on the panel) —
  this is a thesis demo and the classifier must never use this data.

## What exists in this repo

```
data/label_map.json          "00".."24" -> Japanese sign name
data/signs_english.json      Japanese sign name -> English gloss
data/raw/                    user drops the video zip (or .mp4 files) here
scripts/extract_landmarks.py ALREADY WRITTEN AND TESTED LOGIC — run it, fix only if broken
requirements.txt             Python 3.10/3.11 (mediapipe==0.10.9 needs it)
```

Signs (25): 〜から ありがとう ある いくつか いる いろいろ おとな けれども とても どちら ほとんど
みんな もらう ゴルフ コーヒー サッカー サークル スキー スケート スポーツ テレビ ネズミ
バレーボール ビール ピアノ

## Phase 0 — Environment & data extraction

1. Create a venv, `pip install -r requirements.txt`. If Python is 3.12+, tell the user to install
   3.11 and stop. mediapipe must be 0.10.9 (the `mp.solutions.hands` API).
2. Run `python scripts/extract_landmarks.py`. It auto-finds the zip in `data/raw/`. It produces:
   - `data/landmarks_full/XX.npy` — (T,126) float32 at native 30 fps, signing frames only
   - `data/landmarks/XX.npy` — (60,126) whole sign resampled to 60 frames
   - `data/extraction_report.json`
3. Verify all 25 signs extracted, every sign has hand detection >= 80% (target >90%). If ones are
   low, inspect frames and tune (e.g. detection confidence) — do not just ignore.
4. Feature layout (must be identical in training and live inference): 126 = Left-hand slot [0:63]
   then Right-hand slot [63:126]; each hand = 21 keypoints x (x,y,z), each keypoint minus the
   wrist keypoint. Missing hand = zeros. **Frames are never mirrored before MediaPipe** (signer in
   the videos and the live user both face the camera, so raw frames match; mirror only via CSS
   for display).

## Phase 1 — Training (src/train/*, run_training.py)

Critical design point — **temporal consistency between training and live inference**:
- Live: a rolling buffer of the last **96 frames (~3.2 s at 30 fps)** is resampled (linear
  interpolation) to **60 frames** and fed to the model.
- Training must produce inputs the same way: from each sign's full native-fps sequence
  (`landmarks_full`), place it at a random offset inside a 96-frame window padded with
  *rest-pose/idle-like* frames (zeros are acceptable plus actual rest frames, see below), apply
  random speed factor 0.7–1.4, then resample to 60 frames.
- Do NOT simply squash the whole sign to 60 frames for training while live uses a different scale.

Classes: 25 signs + **1 "idle" class** (index 25, label "—"), so the live system does not hallucinate
a sign when the user is just sitting there. Build idle samples from: all-zero (no hands) sequences;
the first ~15 frames of each full sequence (signer resting, hands clasped) padded/stretched into
windows; and small-noise variants. Idle is never shown as a prediction in the UI (it just means
"no sign").

Augmentation (apply on the fly, per epoch, ≥80 effective variants per class):
mirror (swap hand slots and negate x of every keypoint — note this means a mirrored sign is a
different-handed signer, which is a valid augmentation), speed warp, Gaussian noise (σ≈0.005 on
non-zero frames), scale jitter 0.85–1.15, small random 2D rotation (±10°) applied to x,y of both hands,
random per-hand dropout of a few frames (to mimic MediaPipe misses), temporal shift.

Model (src/train/model.py) — Pre-LN Transformer encoder: Linear(126→128), sinusoidal positional
encoding (len 61 incl. CLS), learnable CLS token, 2 layers, 4 heads, FFN 128→256 (GELU), dropout 0.2,
classifier Linear(128→26), label smoothing 0.1 in the loss. Adam lr 5e-4, wd 1e-4, batch 32,
up to 200 epochs, early stopping patience 20, ReduceLROnPlateau(0.5, patience 10). Save best to
`models/best.pt`; export `models/jsl_model.onnx` (opset 14, dynamic batch, input name `input`,
shape [B,60,126], output `logits`). Verify ONNX output matches PyTorch (max abs diff < 1e-4).
Also save `models/meta.json` (class names, window=96, model_len=60, feature layout).

Validation honesty: with one real example per class, a random split of augmented data leaks.
Hold out *augmentation seeds / speed ranges* for validation and **say clearly in README and the
training printout that val accuracy is optimistic; the real test is live webcam.** Report top-1, top-3,
per-class F1, and a confusion matrix image `models/confusion.png`.

Run it on CPU; should finish in minutes. `python run_training.py` is the one-click entry point.

## Phase 2 — Inference server (src/inference/*, run_server.py)

`predictor.py` — `JSLPredictor`:
- onnxruntime CPU session; MediaPipe Hands (max 2 hands, det/track conf 0.5), same feature
  extraction code path as `scripts/extract_landmarks.py` (factor shared code into one module
  and import it from both — no duplicated logic).
- Rolling deque(maxlen=96) of per-frame 126-d vectors. Predict every 3rd frame once the buffer has
  ≥48 frames (pad the front with zeros), resample to 60, run ONNX, softmax, EMA smoothing (α=0.4).
- Decision logic: show a sign only if top class != idle, confidence ≥ 0.60, and the same sign has been
  top-1 for ≥5 consecutive predictions; after emitting, apply a ~1.5 s cooldown / require idle or
  a different sign before re-emitting the same one (so history isn't spammed).
- Draw hand skeleton on the frame (mp drawing utils), return JPEG (quality ~70, downscaled to
  ≤640 px wide) as base64.
- Payload: `{frame, prediction:{sign,english,confidence,is_confident}, top3:[...], left_hand,
  right_hand, buffer_fill, fps_server, mode: "model"|"demo"}`.
- `glove_sim.py` — `GloveSource` interface (`read() -> {left:{pressure[50],imu[6]}, right:{...}}`)
  and `SimulatedGlove` (smooth random walks; pressure correlates with detected hand openness/motion
  from the landmarks when hands are visible so it looks plausible). Documented clearly as simulated.
- If `models/jsl_model.onnx` is missing → **DEMO mode**: cycle through signs with plausible
  confidences so the UI is fully demoable before training.

`server.py` — FastAPI: `GET /` serves `frontend/index.html`; `GET /status`; `WS /ws` where the
client sends `{"type":"frame","data":"<base64 jpeg>"}` and server replies with the payload + glove
data. Drop frames if the server is behind (process only the latest) to avoid lag. Handle 0 hands
gracefully. Never crash on a bad frame. `run_server.py` starts uvicorn on 127.0.0.1:8000 and prints
the URL.

## Phase 3 — Frontend (frontend/index.html, single file, no build step)

Dark, polished, "research instrument" look: bg #0d1117, cards #161b22, borders #30363d, accent
#3b82f6, Inter font + a Japanese-capable font (Noto Sans JP) via Google Fonts. Chart.js 4.4.0 from
cdnjs. Full viewport, no page scroll on a 1080p laptop; responsive down to ~1280 px.

Layout: header (title, LIVE dot, chips: WS connected, hands detected, model loaded/demo, FPS) →
left: webcam panel (mirrored via CSS, buffer-fill bar, hands chip) → right-top: two glove heat maps
(5×10 grid each, cold #1e3a5f → warm #ef4444, avg pressure %, "SIMULATED SENSOR FEED" badge) →
right-middle: scrolling IMU line chart (3 axes, 100-sample window, no axes clutter) → bottom:
prediction panel (huge kanji ~96px, English gloss, vertical confidence bar + %, top-3 chips with
mini bars; blue glow when confident, 40% opacity when not; 200 ms cross-fade on change) → history
tape (last 8 recognised signs as pills, newest right, oldest fades). A "?" button opens a modal listing the
25 supported signs so a demo audience knows what to try.

Client logic: getUserMedia (640x480), send frames at ≤15–20 fps as JPEG via canvas (adaptive: wait
for the previous reply before sending the next), auto-reconnect with backoff (1,2,4,8 s) and a
"Reconnecting…" overlay, graceful message if camera permission is denied. Requires no external
state; opening `http://127.0.0.1:8000` must just work.

## Phase 4 — Verification (do all of these, show the evidence)

1. Extraction report: 25/25 signs, detection rates.
2. Training run completes; ONNX parity check passes; print top-1/top-3/per-class F1; confusion.png.
3. Unit-ish tests (`tests/`, pytest): feature layout/shape (126), resample length, augmentation
   shapes & mirror correctness (mirroring twice returns the original), ONNX I/O shape, predictor
   returns well-formed payload on a black frame and on a frame from one of the reference videos.
4. **Replay test**: feed each of the 25 reference videos' signing frames through `JSLPredictor`
   frame by frame (simulating the live path, including the 96-frame buffer and decision logic) and report
   how many of the 25 signs get correctly emitted. Target ≥ 22/25. If lower, fix the cause
   (usually train/inference temporal mismatch or thresholds) rather than loosening the test.
5. Start the server, hit `/status`, open a WebSocket, send a test frame, validate the payload.
   Use Playwright (chromium) to load the page with a fake video device
   (`--use-fake-device-for-media-stream --use-fake-ui-for-media-stream`), confirm no console errors,
   screenshot the UI and look at it; fix layout problems you see.
6. Write README.md: setup (Python 3.11), `python scripts/extract_landmarks.py`,
   `python run_training.py`, `python run_server.py`, open the URL; architecture diagram (ASCII is
   fine); known limitations (single signer, one example/sign, optimistic val, simulated glove panel);
   how to plug in real glove data (`GloveSource`); how to add more signs (add videos + label_map entries
   → re-extract → retrain).

## Rules

- Windows 11 is the user's machine: use `pathlib`, no bash-only scripts, no hard-coded absolute
  paths, UTF-8 everywhere (`encoding="utf-8"`, `ensure_ascii=False`). Japanese filenames can break
  OpenCV on Windows — keep using ASCII temp copies as the extraction script does.
- Type hints and short docstrings on public functions. Keep files focused (<300 lines each).
- Commit to git in logical steps (extraction, training, server, frontend, tests, docs). Don't commit
  videos, zips or `models/*.pt`; do commit `models/jsl_model.onnx` and `models/meta.json` if small.
- Prefer fixing root causes over loosening thresholds or tests. If a verification target can't be met,
  report the real number and why, rather than hiding it.

## Final message to the user

Summarise: what was built, the real verification numbers (extraction, val metrics, replay test X/25),
how to run it (3 commands), and the honest limitations.
