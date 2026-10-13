# Hardware — tactile sensor glove

`glove_design/` holds the embroidery machine files (Brother `.pes` format) for the
tactile sensor gloves the lab originally used for data collection.

| File | Description |
|------|-------------|
| `Left/Islam_glove_Left.pes`, `Right/Islam_glove_Right.pes` | Full sensor-glove embroidery pattern (left / right hand) |
| `Left/Sample_Left.pes`, `Right/sample Right.pes` | Smaller sample / test patterns |

The glove data turned out to be too noisy to train on, so the recognition system in this
repo uses the webcam + MediaPipe only. The UI's glove panel is fed by a simulator
(`src/inference/glove_sim.py`); a real glove can be plugged in by implementing `GloveSource`.
