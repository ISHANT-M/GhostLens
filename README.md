# GhostLens

A computer vision mystery game about running models on a small device. Built with Streamlit for **UCS668 Edge AI and
Robotics: Data Center Vision** at Thapar Institute of Engineering & Technology (Prof. Jhilik Bhattacharya).

**Team:** Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma

You investigate a haunted hotel with a GhostLens Mk.II, a handheld camera that runs vision models on the device
itself. It has a 100-unit battery, 32 MB for models, and a latency limit for every mission. In each chapter you pick
the computer vision task and the model size. The rule: **use the cheapest task and the smallest model that still
answers the question.** Wrong choices really run, cost battery, and show you why they don't work.

Everything runs locally and offline after a one-time setup.

## Learning objectives

After playing, a student should be able to:

- pick the right CV task (enhancement, classification, detection, segmentation) for a question;
- explain what a classifier's softmax output does and doesn't tell you;
- explain precision, recall, F1 and IoU, and how a confidence threshold trades one for the other;
- explain why a bounding box is not enough when you need exact pixels;
- tell enhancement (inference time) from augmentation (training time);
- choose a model by memory, latency and accuracy together, not by accuracy alone;
- explain what INT8 post-training quantization does to size, speed and accuracy.

## The four chapters

| Chapter | Question | Task | What you learn |
|---|---|---|---|
| 1. The Dark Frame | Read a room number in an almost black CCTV frame | Enhance (OpenCV) | Gamma, contrast, histograms, CLAHE, denoising, clipping. Fix the input before spending compute. Enhancement vs augmentation. |
| 2. Identify the Entity | What is each of three single objects? | Classify (YOLO26n/s/m-cls) | Softmax, top-5, label sets. One object: the cheapest task and smallest model are enough. |
| 3. Find the Anomalies | What objects are in this room, and where? | Detect (YOLO26n/s/m) | Boxes, threshold, precision, recall, F1, IoU. The nano model misses the cups, the medium model doesn't fit. |
| 4. The Corrupted Region | Exactly which pixels are the stain? | Segment (our tiny U-Net) | Masks vs boxes, mask IoU. Only the INT8-quantized balanced model meets both the IoU and latency limits. |

Each chapter has the same flow: loading screen with a syllabus tip, task choice, model choice, run, result, and a
mission report with an A-D edge engineering grade and XP.

## Off the clock: Lab and Field guide

Two extra pages in the sidebar don't cost any battery.

The **GhostLens Lab** is a workbench for the building blocks behind the models. You can slide a kernel over a
photo and change stride and padding, see the feature maps of our own U-Net, work out parameters and memory for a
conv layer, plot activation functions with their derivatives, and race SGD, momentum, RMSprop and Adam on the same
loss surface.

The **Field guide** collects the loading screen tips you've found so far, has a short glossary, a 20 question quiz
(+5 XP per new right answer) and the badges you can earn while playing.

## The edge resource mechanic

- **Battery**: 100 units. Each model run costs about one unit per 5 ms of measured latency (at least 1), so the
  cost comes from how slow the model really is on this machine. Below 20 units the device goes into low power and
  heavy models are disabled. At 0 only the light models can still load.
- **Model memory**: 32 MB. A model that doesn't fit can't be loaded. The chapter 3 detector stays loaded as a watchdog,
  so chapter 4 has less room.
- **Latency limits** per mission (33 ms, 30 ms, 60 ms, 40 ms).
- **Grade**: A if every check passes (right task, right-sized model, latency, battery efficiency), one letter down per
  failed check.

## Real, measured, published, gameplay

| What | Category |
|---|---|
| Image processing, histograms, clipping, legibility | Real, computed live |
| Every model prediction (labels, boxes, masks) | Real inference |
| Precision, recall, F1, box IoU, mask IoU | Measured against our own labels |
| Model file size and CPU latency | Measured by `setup_models.py` on the local machine |
| U-Net IoU on 150 validation walls | Measured |
| YOLO26 ImageNet top-1 and COCO mAP | Published by Ultralytics, not re-measured |
| Battery units, 32 MB, latency limits, pass marks, XP, grades | Gameplay rules |

The app tags these as MEASURED, PUBLISHED or GAMEPLAY wherever they appear.

### Measured on a MacBook Air M3 (CPU, median of 12 runs)

| Model | Size | Latency | Accuracy |
|---|---|---|---|
| YOLO26n-cls / s-cls / m-cls | 5.8 / 13.6 / 23.5 MB | 7.0 / 9.5 / 16.3 ms | 71.4 / 76.0 / 78.1% top-1 (published) |
| YOLO26n / s / m | 5.5 / 20.4 / 44.3 MB | 24.9 / 49.2 / 110.5 ms | best F1 on our scene 0.60 / 0.78 / 0.82 |
| U-Net Lite FP32 / INT8 | 0.05 / 0.02 MB | 14.9 / 5.1 ms | 0.921 / 0.920 mask IoU |
| U-Net Standard FP32 / INT8 | 0.50 / 0.13 MB | 64.1 / 22.1 ms | 0.960 / 0.959 mask IoU |
| U-Net Pro FP32 / INT8 | 1.92 / 0.49 MB | 130.1 / 74.3 ms | 0.965 / 0.965 mask IoU |

Full table in `models/benchmark.json` after setup.

## Architecture

```
app.py            Streamlit entry: navigation, sidebar case file, device panel, HUD, audio
game/             rules and pages
  state.py        progress and locks (pure functions, tested with a dict)
  device.py       battery, model memory, run log
  scoring.py      XP, edge grade, efficiency check
  flow.py         loading screen, HUD, mode choice, model picker, mission report
  levels.py       chapter list, home, about, completion and case-closed screens
  level1-4.py     the four chapters
  runtime.py      cached model loading and benchmark access
  tips.py         loading screen tips
  lab.py          GhostLens Lab page
  codex.py        Field guide: tips found, glossary, quiz, badges
  achievements.py badge checks
cv/               computer vision code, no game rules
  enhancement.py  OpenCV pipeline and metrics
  classification.py, detection.py   YOLO26 wrappers, IoU matching, P/R/F1
  segmentation.py synthetic data, TinyUNet, training, INT8 quantization
  edge.py         benchmark of every model, energy units
  lab.py          maths behind the Lab (convolution, layer cost, activations, optimizers)
ui/               HTML helpers and CSS theme
scripts/          fetch_assets.py (images), make_ambience.py (synthesised audio)
setup_models.py   download weights, train U-Nets, benchmark
tests/            pytest, including headless chapter play-throughs with Streamlit AppTest
```

Progress lives in `st.session_state`. Models are cached with `st.cache_resource`, per-image results with
`st.cache_data`. All inference runs on the CPU; only U-Net training uses Apple's MPS.

## Requirements

- macOS on Apple Silicon (tested on an M3). Linux should work but is untested.
- Python 3.12
- About 1.5 GB of disk for the virtual environment and 140 MB for model weights
- Internet once, for `setup_models.py`

Python packages (`requirements.txt`): streamlit, ultralytics, torch, torchvision, opencv-python, numpy, pillow,
pytest, watchdog.

## Install and run (macOS, Apple M3)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python setup_models.py        # once: YOLO26 weights, trains 3 U-Nets (~5 min), benchmarks every model
streamlit run app.py          # http://localhost:8501
```

Optional:

```bash
python scripts/fetch_assets.py      # images are committed; this re-fetches and rewrites ATTRIBUTION.md
python scripts/make_ambience.py     # regenerates static/ambience.wav
python setup_models.py --retrain    # retrain the U-Nets and re-benchmark
```

Turn on **Demo mode** in the sidebar to unlock every chapter (useful for presentations). Sound starts after the
first click, because browsers block autoplay.

If a chapter says a model is not installed, run `python setup_models.py` and reload.

## Tests

```bash
pytest
```

56 tests: game state, XP, device rules, badges, Lab maths, OpenCV operations and metrics, IoU matching and
precision/recall, and headless play-throughs of each chapter (these need `setup_models.py` to have run).

## Course concepts covered

Task identification, classification and softmax, network building blocks and skip connections (our U-Net),
augmentation, losses and optimizers (U-Net training), precision/recall/F1, IoU, detection with YOLO26 and COCO,
segmentation, memory and latency budgets, and INT8 quantization. See [docs/SYLLABUS.md](docs/SYLLABUS.md) for the
module-by-module mapping and what is not covered.

## Limitations

- Latency, and so battery cost, is measured on one machine. Other machines give other numbers.
- Published accuracy figures for YOLO26 are quoted, not re-measured.
- Chapter 3 scores one photo with 7 hand-labelled objects. It shows how the metrics work, it isn't a benchmark.
- The U-Net is trained and evaluated on generated stains, so its IoU is in-domain only.
- Battery units are a proxy (energy ~ compute time), not measured power.
- Progress is per browser session and is not saved to disk.

## Future work

- Save progress between sessions.
- Export the chosen models to ONNX / Core ML and measure on a real edge board.
- Pruning alongside quantization in chapter 4.

## Credits

- Photos from Wikimedia Commons under CC BY, CC BY-SA and CC0. Authors and licences in
  [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md).
- YOLO26 models by Ultralytics (AGPL-3.0).
- Background audio synthesised by our own script, `scripts/make_ambience.py`.
- Course material: https://sites.google.com/thapar.edu/jhilikbhattacharya/courses/edge-ai-and-robotics-data-center-vision
  and https://github.com/jhilikb/UCS668

## License

Code is under the MIT licence (see `LICENSE`). The photos keep their own Creative Commons licences, and the YOLO26
weights are downloaded from Ultralytics at setup time, not stored in this repo.

## Team

Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma. UCS668, Thapar Institute of Engineering & Technology, 2026.
