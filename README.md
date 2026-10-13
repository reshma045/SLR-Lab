# JSL Real-Time Recognition

Real-time Japanese Sign Language (日本手話) recognition for a Master's thesis at the University of
Aizu. A webcam feed goes through **OpenCV → MediaPipe Hands + Pose → a small Transformer**, which
classifies the last ~3.2 s of motion into one of 25 signs (or "no sign"). A dashboard lets you
record your own examples, retrain, measure your real accuracy and, optionally, ask **Claude** for a
second opinion.

![Live page](docs/ui_live_1080p.png)

> **Honesty notes.** Recognition uses the **webcam only** (no glove/IMU is connected; the optional
> glove panel is a labelled simulation). The base model is trained on **one reference video per
> sign from one signer**. For good accuracy on *your* signing, record yourself on the
> Record & Train page and retrain. Then measure it on the Self-test page.

## Quick start (Windows 11, Python 3.11)

MediaPipe 0.10.9 has no wheels for Python 3.12+, so use **Python 3.10 or 3.11**.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_server.py                  # open http://127.0.0.1:8000 in Chrome or Edge
```

`models/jsl_model.onnx` is committed, so the server works immediately. To rebuild everything from
the videos, put `jsl 25 words.zip` (or the 25 `.mp4` files) in `data/raw/`, then run:

```powershell
python scripts/extract_landmarks.py --zooms 1.3,1.6,2.1   # features + simulated close-webcam crops (~15 min)
python scripts/build_reference_sheets.py                  # reference keyframes (Record page + Claude)
python run_training.py                                    # ~5-10 min on CPU (or use the dashboard)
```

## Dashboard

| Page | What it does |
|---|---|
| **Live** | Real-time recognition: big sign display, confidence and top-3, hand/shoulder status, session log, Claude verdict (AGREES / DIFFERS), history tape |
| **Record & Train** | Guided recording of your own takes (countdown, quality check, reference keyframes, progress for all 25 signs); one-click **retrain** with live progress chart and **roll back** |
| **Self-test** | Shows a random sign, you perform it, and it records whether it was recognised. Gives real accuracy per sign, what each sign is confused with, past runs, CSV export |
| **History** | Every recognised sign with confidence and Claude's verdict and reason; search, stats, CSV export |
| **Settings** | Claude API key and model, when to ask Claude, connection test; camera picker; recognition thresholds; recording, test and display options; dark or light theme |

### Recommended workflow

1. **Settings:** pick your camera. Sit so your **head, shoulders and both hands** are visible.
2. **Record & Train:** record ≥ 5 takes per sign (guided mode walks through all 25; about 20 min).
   Press **Retrain**. Once you have ≥ 3 takes of a sign, your newest take is held out as an
   **honest test** (the "Real val" number).
3. **Self-test:** run a test to get your real accuracy and see which signs need more takes.

### Claude second opinion (optional)

The local pipeline is always the real-time classifier: offline, free, about 16 processed fps.
With a Claude API key (Settings → Claude), each recognised sign can be double-checked by Claude.
It receives 6 keyframes of your clip (with hand skeletons) plus 25 labelled reference sheets
(prompt-cached), and picks one of the 25 signs or "none" via structured output. It runs in the
background, so the live view never waits.

- Models: **Claude Haiku 5.5** (default; fastest and cheapest, runs without extended thinking for
  latency), Sonnet 5.5 or Opus 5.5 (server-side refusal fallback enabled).
- Modes: every recognised sign, only when the local model is unsure (below a set confidence), or
  only when you press **Ask Claude**.
- **Privacy:** keyframes are sent to the Anthropic API only while a key is set and Claude is
  enabled. The key is stored only in the git-ignored `.env` file. The UI shows it masked.
- Claude does not know JSL perfectly. It is a cross-check against the reference videos, not a
  replacement for recording your own data.

### OpenCV app (no browser)

```powershell
python scripts/webcam_demo.py              # OpenCV window: q / Esc quits, c asks Claude
```

Opens the camera via Media Foundation on Windows. On the test laptop, DirectShow gave only about
8 fps and MSMF about 24.

## Why the first version performed badly live, and what changed

| Problem (v1) | Fix (v2) |
|---|---|
| Hands near a laptop webcam look about 3× larger than in the full-body reference videos (palm ≈ 5.6% vs ~15–20% of frame height). v1 used raw wrist-relative coordinates. | Hand shape **divided by palm length**: scale-free |
| v1 knew nothing about *where* the hands are (face, chest…), which many signs depend on | **MediaPipe Pose**: wrist position relative to the shoulder centre (in shoulder widths), nose position, palm/shoulder ratio |
| Left/right came from MediaPipe's handedness classifier, which is unreliable on raw frames | Signer's left/right assigned by the **nearest pose wrist** |
| x and y were scaled differently (3:2 videos vs 4:3 webcams) | **Aspect-corrected** coordinates |
| The buffer counted frames: at 10 fps a sign looked 3× faster | **Time-based** buffer (any camera fps gives the same input) |
| Live hand detection (0.5) differed from extraction (0.3) | Same confidence in both paths |
| Training only saw full-body framing | Training also uses **simulated close-webcam crops** (zoom 1.3/1.6/2.1) of the reference videos |
| One signer only | **Record & Train** adds your own takes. Real accuracy is measured on held-out takes and in Self-test |

## Results

Measured on a Windows 11 laptop CPU (2026-10-13). "Replay" pushes each reference video frame by
frame through the exact live path (MediaPipe → time-based buffer → model → smoothing → decision
logic), with 1 s of rest before and 1.5 s after.

| Check | v1 (hands only) | **v2 (current)** |
|---|---|---|
| Replay, original full-body videos, 30 fps | 25/25 | **25/25**, no wrong or duplicate emissions |
| Replay at a simulated 15 fps camera | 25/25 | **25/25** |
| Replay at a simulated 10 fps camera | 11/25 (frame-count buffer) | **25/25** (time-based buffer) |
| **Replay as a close laptop webcam (zoom 1.8, never trained on)** | **10/25** | **25/25** |
| Replay, even closer (zoom 2.4, beyond training range) | – | 24/25 (スキー missed) |
| Synthetic validation (held-out augmentation; **optimistic**) | 99.9% | 99.9% top-1, 100% top-3 |
| Real validation on your held-out takes | – | appears after you record ≥ 3 takes per sign |
| Extraction | – | 25/25 signs; hands in 88–100% of frames, shoulders in 100% |
| ONNX parity | 2.9e-6 | 1.9e-6 (< 1e-4) |
| Unit tests | 25 | **32 passed** |
| UI check (headless Chromium, fake camera, 5 pages × 1920×960 and 1280×720) | – | no console errors, no page overflow, about 16 processed fps |

The zoom tests crop each reference video to a 4:3 upper-body view resized to 640×480, as if sitting
close to a laptop webcam (`python scripts/replay_test.py --zoom 1.8`). They show robustness to
framing and hand size. They do **not** show generalisation to other signers; only your own
recordings and the Self-test page measure that.

## Architecture

```
 Browser dashboard (frontend/*.html, no build)       Python server (FastAPI, run_server.py)
 ┌────────────────────────────────────┐   WS /ws     ┌──────────────────────────────────────────────────┐
 │ getUserMedia → canvas JPEG          │ ──────────► │ decode (latest frame only, drop stale)            │
 │ (≤ send_fps, waits for each reply)  │             │ MediaPipe Hands (conf 0.3) + Pose (every 2nd fr.) │
 │                                     │             │ src/common/features.py (shared with extraction):  │
 │ Live · Record & Train · Self-test   │             │  136-d = 2×63 scale-free hand shapes              │
 │ History · Settings                  │ ◄────────── │        + wrist/nose positions in shoulder widths  │
 └────────────────────────────────────┘  payload     │        + presence flags                           │
          │ REST /api/*                               │ time-based buffer: last 3.2 s → 60 instants       │
          ▼                                           │ ONNX Transformer → softmax → EMA → decision       │
 settings.json · .env (key) · app.db (SQLite:         │  (threshold / stability / cooldown from Settings) │
 history + self-tests) · data/user/<profile>/ takes   │ emit → history DB → optional Claude check         │
 · background training process → hot model reload     │ record_start → 30 fps take → data/user/…          │
                                                      └──────────────────────────────────────────────────┘
```

Model: Linear(136→128) → [CLS] + sinusoidal positions → 2 × Pre-LN Transformer encoder layers
(4 heads, FFN 256, GELU, dropout 0.2) → LayerNorm → Linear(128→26). Class 26 is **idle**, so the
system stays quiet when nobody is signing.

### Training data and train/live consistency

Each sample mimics the live buffer:

1. Take a full 30 fps sequence (rest → sign → rest): a reference video, a simulated
   close-webcam crop of it, or one of your takes (your takes make up 70% of a sign's samples when present).
2. Speed-warp it (×0.7–1.4).
3. Cut a 3.2 s window containing ≥ 80% of the active sign. Pad outside the recording with zeros or a held rest pose.
4. Augment: mirror (swap hands and negate x), body scale 0.85–1.15, position shift ±0.12, rotation ±10°,
   hand dropout, occasional pose dropout, noise.
5. Resample to 60 steps.

**Idle** samples cover empty frames, rest poses, hands entering or leaving the view, and
*partial signs* (the first ≤ 50% or last ≤ 30% of a sign), so the model waits until a sign is mostly
visible. Live, the same timeline is rebuilt from timestamps at any frame rate.

## Repository layout

```
data/label_map.json, signs_english.json   class ids -> Japanese -> English
data/raw/                                 reference videos (git-ignored)
data/user/<profile>/<sign>/               your recorded takes (git-ignored)
scripts/extract_landmarks.py              videos -> features (+ --zooms webcam crops)
scripts/build_reference_sheets.py         reference keyframe sheets (Record page + Claude)
scripts/replay_test.py                    live-path replay (--fps, --zoom)
scripts/ui_check.py                       server + WebSocket + 5-page browser check with screenshots
scripts/webcam_demo.py                    standalone OpenCV webcam app
src/common/                               config, shared features (hands + pose), simulated views
src/train/                                dataset/augmentation, model, training loop, ONNX export
src/inference/                            predictor, Claude verifier, simulated glove, FastAPI server
src/app/                                  settings, SQLite store, recordings, background trainer, REST API
frontend/                                 dashboard pages + static/{app.css, common.js, camera.js}
models/                                   jsl_model.onnx, meta.json, reports, confusion.png
hardware/glove_design/                    tactile glove embroidery files (.pes)
tests/                                    pytest suite (32 tests)
```

## Known limitations

- **Generalisation to new signers is unmeasured until you record.** Replay results use the
  reference videos (one signer). Use Record & Train and Self-test for real numbers.
- **Framing:** the head, shoulders and both hands should be visible. Without shoulders, position
  features are missing and accuracy drops (the UI warns: "Shoulders not visible").
- **Overlapping hands** (e.g. ゴルフ) lose MediaPipe detections. Very close framing (zoom ≥ 2.4)
  starts to miss signs.
- **Isolated signs only.** Lower your hands briefly between signs.
- **Glove panel** (off by default) is a simulation. The classifier never uses it.

## Plugging in a real glove

Implement `GloveSource.read()` in `src/inference/glove_sim.py` (50 pressure values and 6 IMU values per
hand), pass it to `JSLPredictor(glove=...)`, and enable the panel in Settings → Display. To use
glove data for *recognition*, add it to the feature vector and retrain.

## Adding more signs

Add the videos (file name = sign name) to `data/raw/`, add entries to `data/label_map.json` and
`data/signs_english.json`, update `NUM_SIGNS` in `src/common/config.py`, then re-run extraction,
the reference sheets and training.
