# GhostLens

A computer vision mystery game where every model runs on a small handheld camera, inside a tight battery, memory
and latency budget.

You investigate a haunted hotel with a GhostLens Mk.II. It runs vision models on the device itself, with no server to
fall back on. It has the last 20% of its battery, 32 MB for models and a latency limit for every mission. In each of
the four chapters you pick the computer vision task and the model size. The rule is simple: **use the cheapest task
and the smallest model that still answers the question.** Wrong choices really run, cost battery and show you what
they got wrong.

Everything runs locally and offline after a one-time setup. Every model output, metric, size and latency in the game is
computed on your machine. The few numbers that aren't are labelled.

## Install and run

Tested on macOS (Apple M3) with Python 3.12. Linux should work but we haven't tried it.

```bash
git clone https://github.com/ISHANT-M/GhostLens.git && cd GhostLens
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

python setup_models.py        # once, needs internet: YOLO26 weights, trains 3 U-Nets (~5 min), benchmarks every model
streamlit run app.py          # http://localhost:8501
```

`setup_models.py` downloads about 140 MB of weights, trains our three small U-Nets and times every model on your CPU.
The results go to `models/benchmark.json`, and the game reads its costs from there. You need about 1.5 GB of disk for
the virtual environment.

Optional:

```bash
python setup_models.py --retrain    # retrain the U-Nets and re-benchmark
python scripts/fetch_assets.py      # images are committed; this re-fetches them and rewrites ATTRIBUTION.md
python scripts/make_ambience.py     # regenerates static/ambience.wav
```

If a chapter says a model is not installed, run `python setup_models.py` and reload the page.

## How to play

The top bar (HUD) shows the four chapters, the battery, model memory, chapters solved and XP. The **Menu** button on
the right of it has the page links, Sound and Demo mode toggles, the lobby charger when it's open, and "Restart the
case".

Each chapter goes the same way:

1. A title card sets the scene. The HUD never names the task before you choose it.
2. **Pick the task** on the scanner's mode dial (Enhance, Classify, Detect, Segment, or Retrain in chapter 1). A wrong
   mode really runs if you can pay for it, its output appears on the stage, and one line says why it doesn't answer
   the question.
3. **Pick the model** from the loadout: name, size in MB, measured latency, battery per run and accuracy. A model that
   doesn't fit in memory can't be loaded.
4. **Play the chapter**: the scene is on the left, the handheld scanner with the controls is on the right, and one
   feedback line sits under them.
5. Solve it and you get the **LEVEL CLEARED** screen: a grade from A to D, then three tabs. *Debrief* says what
   happened and why it worked, with the mission report, XP and battery. *Numbers* has the measured tables and charts.
   *Bonus scan* has the optional side scan. "Back to the scene" lets you look at the chapter again.

Turn on **Demo mode** in the Menu to unlock every chapter. Add `?case=1234` to the address to replay a particular case
(handy for presentations). Sound starts after the first click, because browsers block autoplay.

## The four chapters

| Chapter | Question | Task and model | CV technique you work with |
|---|---|---|---|
| 1. The Dark Frame | Read a door number in an almost black CCTV frame | Enhance, OpenCV only, no network | Gamma, brightness/contrast, global equalization, CLAHE, Gaussian/median/non-local means denoising, sharpening. A wipe slider compares the frame before and after, an OpenCV histogram strip shows clipping at 0 and 255, and a tool strip tries one tool at a time. |
| 2. Identify the Entity | What is each of three single objects, and then what is in the guest's room? | Classify, YOLO26n/s/m-cls | Top-5 softmax bars and tagging the right object. Then a sliding-window sweep: the classifier runs on a 3×3 grid of 560×374 windows of a 1280×854 room photo (or the whole photo), one label per window. |
| 3. Find the Anomalies | What objects are in the parlour, and where? | Detect, YOLO26n/s/m | Confidence threshold, precision, recall, F1, IoU matching against our own labels. A box inspector zooms on one box and shows how many pixels it has at the model's 640 px input. |
| 4. The Corrupted Region | Exactly which pixels are the stain? | Segment, our tiny U-Net (FP32 or INT8) | Mask threshold, views for mask, probability map, bounding box and FP32-vs-INT8 difference, and morphological opening and closing to clean the mask. |

What each chapter teaches, in one line:

- **Chapter 1.** Fix the input before you spend compute on it. Gamma 2.4 is the best single tool on every photo, and
  a reference pass (gamma 2.4, contrast 2.5) takes about 2 ms. Non-local means takes about 200 ms a pass (roughly 20%
  of the battery) and doesn't beat it.
- **Chapter 2.** One object per photo means one label per image is enough, and the light classifier gives the same
  top-1 as the big ones. ImageNet has no "pocket watch" class, so it says "stopwatch". A sliding window gives a rough
  idea of what is where, but no boxes and no count.
- **Chapter 3.** YOLO26n misses the cups (best recall 57%), YOLO26m doesn't fit in memory, YOLO26s meets either brief.
  The threshold doesn't change the model, it only decides which boxes you believe.
- **Chapter 4.** A box is mostly healthy wall, so you need a mask. Only U-Net Standard in INT8 meets both IoU ≥ 0.95
  and ≤ 40 ms. Pixel accuracy flatters a mask: an empty one already scores 84–91% on these walls.

When all four are solved, the Case file page shows the closed case: your grades, a table of the four decisions with
the measured number behind each, the battery timeline, side clues, badges, and a Markdown download of it all.

## The battery and the edge rules

Battery is the main resource, and it is charged from **measured** compute time.

| Rule | Value |
|---|---|
| Start | 20.0% of a full pack |
| Cost of a run | 1 ms of measured latency = 0.1% battery, at least 0.1% per run |
| Low power | below 10%: Heavy models switch off, the battery turns red, the lobby charger opens |
| Lobby charger | +10% for 40 XP, as often as you need, only in low power. Each trip fails the "Stayed in the field" check on that chapter's report, so it can't get an A |
| Emergency reserve | if a run costs more than you have left, Light models and the chapter 1 OpenCV pass still run and the battery stops at 0%. Anything bigger waits for a charge |
| Spare cell | an A grade gives +1.0%, once per chapter |
| Side scans | one optional scan per chapter, on the cleared screen. Costs its measured latency (never on the reserve), pays XP, a side clue and a Field guide tip. Not counted in the chapter's battery check |
| Model memory | 32 MB. The chapter 3 detector stays loaded as a watchdog, so chapter 4 only has 11.6 MB free |
| Latency limits | chapter 1: fits one frame at 30 fps (33 ms); chapter 2: 30 ms; chapter 3: 60 ms; chapter 4: 40 ms |

Every paid button shows its cost and what will be left ("→ 15.1% left"). The battery check on each report passes if
you used at most `max(1.25 × par, par + 0.5%)`, where par is the cheapest way through that chapter:

| Chapter | Par |
|---|---|
| 1 | one reference pass, timed live, about 0.2% |
| 2 | six light classifier scans (three photos, three room windows), 4.2% |
| 3 | one YOLO26s run, 4.9% |
| 4 | one U-Net Standard INT8 run, 2.5% |

So an ideal run spends about 11.8% and ends the case at about 12.2% after four spare cells. The grade is A if every
check on the report passes, and one letter down for each check that fails.

## Every case is different

Each case draws one random seed and builds everything from it (`game/case.py`):

- **Chapter 1**: one of four door photos (rooms 217, 209, 213 and flat 52) becomes the dark CCTV frame. A second one
  is used by the camera 04 side scan.
- **Chapter 2**: one of three anchor objects (pocket watch, padlock, teddy bear), each with its own riddle, and the
  order of the three evidence cards.
- **Chapter 3**: one of two client briefs, "Full inventory" (F1 ≥ 0.75) or "Miss nothing" (recall 100% and precision
  ≥ 60%), and which tea cup was moved.
- **Chapter 4**: two of six stained walls, one to purify and one for the side scan.

Every option in the pool is checked by the tests with the real models: the right choice always works and the wrong
ones always fail. That's also why the pool is small.

## Off the clock: HQ Workbench and Field guide

Two pages in the Menu cost no battery.

The **HQ Workbench** (the Lab) has nine benches. Each one shows the syllabus topic it covers, a "Try:" line, live
controls and results, and short "Bench notes" folded away underneath.

| Bench | What you do | What it measures |
|---|---|---|
| Convolution | Slide a kernel over a photo or your chapter 4 wall; stride, padding, ReLU, max pool | Output size formula; 8 feature maps from our trained U-Net's first block |
| Parameters | Conv layer calculator; TinyUNet layer by layer; train a linear probe | Params, MACs, FP32/FP16/INT8 weight size, activation memory. The probe freezes YOLO26n-cls and trains 2,562 parameters to tell stained from clean walls (about 93% test accuracy) |
| Activations | ReLU, Leaky ReLU, SiLU, sigmoid, tanh, GELU | Each function and its derivative from autograd |
| Architectures | Open up YOLO26n-cls, YOLO26s and our U-Net | Conv, depthwise, BatchNorm, residual, attention and SPPF counts. U-Net skip ablation: without the full-resolution skip, IoU falls from about 0.97 to about 0.2 |
| Optimizers | Race SGD, momentum, RMSprop and Adam on a bowl or Rosenbrock | Real `torch.optim` paths |
| Losses | Pick a mask (two U-Nets, empty, filled box, all stain) or grow and shrink the true mask with `cv2.dilate`/`erode` | Pixel accuracy, IoU, Dice and BCE, live |
| Normalization | Feed the U-Net ÷255, raw 0–255 and ImageNet mean/std inputs; initialise a 12-layer conv stack with and without BatchNorm | IoU about 0.97 / 0.2–0.3 / about 0, and the BatchNorm statistics behind it |
| Pruning | Run a global L1 unstructured pruning sweep of U-Net Standard at 0–90% | Sparsity, IoU, latency and file size next to INT8. 50% zeros is no faster; INT8 is more than twice as fast |
| Augmentation | Mixup and CutMix of the padlock and teddy bear | YOLO26n-cls on the real blends |

The **Field guide** has three tabs: the 54 loading screen tips you've found, a 40-term glossary, and 11 badges
(Frugal, Right tool every time, Straight A, Quantizer, Balanced eye, Clean frame, Thorough, Scholar, Clean sweep,
Inspector and the anti-badge Running on fumes). Restarting the case keeps the tips you've found.

## About the project

GhostLens is our course project for **UCS668 Edge AI and Robotics: Data Center Vision** at Thapar Institute of
Engineering & Technology, taught by Prof. Jhilik Bhattacharya.

**Team:** Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma

### Edge AI in this project

- **On-device and offline.** All inference runs in the app's own process on the CPU. Ultralytics is set to offline
  mode, so after setup nothing is downloaded or sent anywhere and no image leaves the machine.
- **Memory budget.** 32 MB for models. YOLO26m (44.3 MB) can't be loaded, and the resident watchdog leaves 11.6 MB for
  chapter 4.
- **Latency budget.** Each mission has a per-frame limit, and every latency is a median measured on this CPU, not an
  estimate from FLOPs.
- **Energy budget.** Battery is charged from measured compute time. It's a game rule: real energy is roughly power ×
  time, and we don't measure power.
- **INT8 post-training quantization.** Our U-Nets are quantized with PyTorch FX static quantization, calibrated on 64
  walls. U-Net Standard gets 3.8× smaller and 2.4× faster for −0.001 IoU, which is what puts it inside the 40 ms limit.
- **Model selection under constraints.** In every chapter the model is chosen by size, latency and accuracy together.
  The most accurate one is never the answer by default.

### Measured on a MacBook Air M3

CPU only, median of 12 runs (8 for the U-Nets), from `models/benchmark.json`. Accuracy for YOLO26 is the published
Ultralytics figure; U-Net IoU is measured on 150 validation walls; "best F1" is on our labelled parlour photo.

| Model | Size | Latency | Battery per run | Accuracy |
|---|---|---|---|---|
| YOLO26n-cls / s-cls / m-cls | 5.8 / 13.6 / 23.5 MB | 6.7 / 9.3 / 15.6 ms | 0.7 / 0.9 / 1.6% | 71.4 / 76.0 / 78.1% ImageNet top-1 (published) |
| YOLO26n / s / m | 5.5 / 20.4 / 44.3 MB | 23.9 / 48.6 / 105.0 ms | 2.4 / 4.9 / 10.5% | 40.9 / 48.6 / 53.1 COCO mAP (published); best F1 0.60 / 0.78 / 0.82 |
| YOLO26s-seg | 23.5 MB | 70.1 ms | 7.0% | 40.0 mask mAP (published) |
| U-Net Lite FP32 / INT8 | 0.05 / 0.02 MB | 14.8 / 5.1 ms | 1.5 / 0.5% | 0.921 / 0.920 mask IoU |
| U-Net Standard FP32 / INT8 | 0.50 / 0.13 MB | 60.0 / 24.5 ms | 6.0 / 2.5% | 0.960 / 0.959 mask IoU |
| U-Net Pro FP32 / INT8 | 1.92 / 0.49 MB | 125.5 / 74.5 ms | 12.6 / 7.5% | 0.965 / 0.965 mask IoU |

For the same network size, YOLO26s takes 9.3 ms to classify, 48.6 ms to detect and 70.1 ms to segment. Your numbers
will be a little different; the game uses whatever `setup_models.py` measured on your machine.

### What's measured and what's a game rule

| What | Category |
|---|---|
| Image processing, histograms, clipping, legibility | Measured live |
| Every model output (labels, boxes, masks) | Real inference |
| Precision, recall, F1, box IoU, mask IoU, Dice, pixel accuracy | Measured against our own labels |
| Model file size and CPU latency | Measured by `setup_models.py` |
| Every Workbench experiment | Measured live |
| YOLO26 ImageNet top-1 and COCO mAP | Published by Ultralytics, not re-measured |
| Battery scale, 32 MB, latency limits, pass marks, XP, grades, badges | Game rules |

The cleared screens, the model loadout and the About page all repeat this split. See
[docs/SYLLABUS.md](docs/SYLLABUS.md) for where each syllabus topic is covered.

## Project layout

```
app.py              Streamlit entry: navigation, HUD, Menu popover, lobby charger, audio
game/               rules and pages
  state.py          progress, locks, grades (pure functions, tested with a plain dict)
  device.py         battery, ledger, low power, lobby charger, spare cells, model memory
  scoring.py        XP, edge grade, battery efficiency check
  case.py           the random case: photo, anchor, brief, moved cup, walls
  flow.py           loading screen, HUD, mode dial, model loadout, paid run buttons, side scans, checklists
  levels.py         chapter list, LEVEL CLEARED screen, home, case closed, About
  level1-4.py       the four chapters
  lab.py            HQ Workbench (nine benches)
  codex.py          Field guide: tips, glossary, badges
  tips.py           loading screen tips and which chapter teaches each topic
  achievements.py   badge checks
  runtime.py        cached model loading and benchmark access
  nav.py            page registry, so pages can link to each other
cv/                 computer vision code, no game rules
  enhancement.py    OpenCV pipeline, dark frame, clipping, legibility, wipe, histogram strip, augmentation
  classification.py YOLO26-cls wrapper, sliding-window grid and coverage
  detection.py      YOLO26 wrapper, IoU matching, precision/recall/F1, false alarm reasons, zoom, input size
  segmentation.py   generated stain walls, TinyUNet, training, INT8 quantization, mask IoU, morphology cleanup
  edge.py           benchmark of every model, energy units
  lab.py            maths behind the Workbench (convolution, layer cost, activations, optimizers)
  experiments.py    Workbench experiments: mask metrics, pruning, normalization, init, architecture counts,
                    skip ablation, Mixup/CutMix, linear probe
  models.py         model paths and the "run setup" error
ui/                 HTML helpers and the dark theme CSS
assets/             Wikimedia photos, the door photo pool (level1/clues.json), chapter 3 ground truth
scripts/            fetch_assets.py (images), make_ambience.py (synthesised audio)
setup_models.py     download weights, train U-Nets, benchmark
tests/              pytest, including headless play-throughs with Streamlit AppTest
```

Progress lives in `st.session_state`. Models are cached with `st.cache_resource` and per-image results with
`st.cache_data`. All inference runs on the CPU; only U-Net training uses Apple's MPS when it's there.

## Tests

```bash
pytest -q
```

282 tests. Unit tests cover game state, XP, the battery rules and their guarantees (an ideal run leaves a margin, one
lobby trip always pays for a mandatory run), the case generator, badges, the Field guide, the cleared screen and case
summary, the Workbench maths and experiments, OpenCV operations, sliding windows, IoU matching and mask cleanup. Data
tests check every option in the case pool with the real models (each anchor is named inside its own label set, the
three ideal room windows give three different labels, YOLO26s meets each brief and YOLO26n never does, the light U-Net
leaks on every wall). Page tests play each chapter headlessly with Streamlit AppTest, including a full ideal
play-through with every side scan, recovery from an empty battery, and a check that play screens stay uncluttered. The
data and page tests need `setup_models.py` to have run.

## Limitations

- Latency, and so battery cost, is measured on one machine. Other machines give other numbers, and live timings (the
  chapter 1 pass, the Workbench) move by a few ms between runs.
- Battery is modelled as compute time, not measured energy.
- Published accuracy figures for YOLO26 are quoted, not re-measured.
- Chapter 3 scores one photo with 7 hand-labelled objects. It shows how the metrics work; it isn't a benchmark.
- The U-Net is trained and tested on generated stains, so its IoU (and the linear probe's accuracy) is in-domain only.
- The case pool is small (4 photos, 3 anchors, 2 briefs, 3 movable cups, 6 walls), because every option is verified
  with the real models.
- Progress lives in the browser session and is not saved to disk.

## Future work

- Structured (channel) pruning of the U-Net with fine-tuning, to compare a real speed-up from pruning with INT8.
- Train the U-Net with Dice loss (or BCE + Dice) and compare it with BCE on thin stains.
- Quantization-aware training for U-Net Pro.
- Real stain photos with hand-drawn masks, to measure out-of-domain IoU.
- Save progress between sessions.

## Credits

- Photos from Wikimedia Commons under CC BY, CC BY-SA, CC0 and public domain. Authors and licences are in
  [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md).
- YOLO26 models by Ultralytics (AGPL-3.0), downloaded at setup time.
- Background audio made by our own script, `scripts/make_ambience.py`.
- Course material: https://sites.google.com/thapar.edu/jhilikbhattacharya/courses/edge-ai-and-robotics-data-center-vision
  and https://github.com/jhilikb/UCS668

## Licence

Our code is under the MIT licence (see `LICENSE`). The photos keep their own licences, and the YOLO26 weights are
downloaded from Ultralytics at setup time, not stored in this repo.
