# How to use this kit (2 minutes)

1. Put the video zip (`jsl 25 words.zip`, or the 25 .mp4 files) into `data/raw/`.
2. Install Python **3.11** (not 3.12+; MediaPipe 0.10.9 needs ≤3.11) and ffmpeg is NOT required.
3. Open a terminal in this folder (or push it to your GitHub repo and open Claude Code there):
   ```
   claude
   ```
4. Send this one message:

   > Read CLAUDE.md fully and execute every phase end to end, including all verification steps.
   > Don't stop until the replay test and UI check pass; report real numbers at the end.

That's it. When it finishes:
```
python run_training.py     # only if you want to retrain
python run_server.py       # then open http://127.0.0.1:8000
```

Notes
- No git repo needed for the kit itself; if you use GitHub, commit this folder first so Claude Code can commit in steps.
- The glove panel is a labelled simulation; the classifier uses only the webcam.
- Accuracy numbers on 1 video/sign are optimistic — judge by live testing.
