# GhostLens and the UCS668 syllabus

Where each topic of UCS668 Edge AI and Robotics: Data Center Vision is taught or practised in the game.

How to read the "Where" column:

- **Ch n** is a chapter. **Play** is what you do on the chapter's play screen. **Cleared** is the LEVEL CLEARED screen
  after it: *Debrief* (what happened and why), *Numbers* (measured tables and charts, with "The maths" folded away) and
  *Bonus scan* (the optional side scan).
- **Bench · X** is a bench of the HQ Workbench (the Lab). Each bench opens with the syllabus line it covers and has
  "Bench notes" folded under it.
- **Tip** is a loading screen tip (`game/tips.py`), collected in the Field guide. **Glossary** is the Field guide
  glossary (`game/codex.py`).

Sources: the course site and https://github.com/jhilikb/UCS668.

## Module by module

| Module | Topic | Where in GhostLens | Code | How it's practised |
|---|---|---|---|---|
| 1 | Understanding the vision problem, task identification | Every chapter starts on the mode dial (Enhance / Classify / Detect / Segment, plus Retrain in ch 1). The HUD and loading screen never name the task first. The closed case lists question → task → model → deciding number | `game/flow.py` `mode_choice`, `level1-4.py` `mode_options`, `levels.recap_rows` | A wrong task really runs if you can afford it, costs battery, and its output goes on the stage with one line: "You asked for X. This question needs Y." The cleared Debrief repeats each wrong try with its verdict. Tips: Choosing the task (3) |
| 1 | Image preprocessing (before any model) | Ch 1 | `cv/enhancement.py`, `game/level1.py` | Gamma, brightness, contrast, global equalization, CLAHE, Gaussian / median / non-local means denoising, unsharp masking. Wipe compare of the raw and enhanced frame, an OpenCV histogram strip with clipping marked at 0 and 255, a 1% clipping limit, and a tool strip (gamma 2.4, contrast ×4, brightness +100) to try one tool at a time. Cleared · Numbers: your pass against the reference, reference + non-local means (about 200 ms a pass) and brightness alone, plus each single tool's measured quality. Tips: Gamma, Histograms, CLAHE |
| 2 | First vision model: classification | Ch 2 | `game/level2.py`, `cv/classification.py` | YOLO26n/s/m-cls on three single objects, top-5 softmax bars, tag the object that matches the riddle. Cleared · Numbers: all three classifiers on all four photos, showing the bigger models only change the top-1 on the busy room photo. Tip: Classification |
| 2 | Softmax, confidence, label set | Ch 2 | `level2.softmax_maths`, `game/case.py` `ANCHORS` | Worked logit → softmax table in "The maths"; the anchor reveal ("ImageNet has no 'pocket watch' class, so the nearest one wins"). Tip: Softmax. Glossary: Logit, Softmax |
| 2 | Classification on parts of an image (sliding window) | Ch 2 room sweep | `cv/classification.py` `room_windows`, `coverage`; `level2.room_sweep` | Classify windows of a 1280×854 room photo: the whole photo or one of a 3×3 grid of 560×374 crops, chosen with pan and tilt. Each window costs a full scan and gives one label. Cleared · Numbers: the full sweep, 10 passes and still no boxes |
| 2 | Transfer learning | Bench · Parameters (linear probe) | `cv/experiments.py` `backbone_features`, `train_probe` | Freeze YOLO26n-cls and train one Linear layer (2,562 parameters) to tell stained from clean walls; train and test accuracy measured (about 93% test). Tips: Transfer learning (2), Linear probe |
| 3 | Building blocks: Conv2D, stride, padding, pooling, tensor shapes | Bench · Convolution, Bench · Parameters, ch 4 U-Net | `cv/lab.py`, `game/lab.py`, `cv/segmentation.py` `block` | Kernels on real images, output size and pooling formulas, 8 learned feature maps of our U-Net. Layer calculator: params, MACs, FP32/FP16/INT8 weights, activation memory. TinyUNet layer table. Tips: Conv2D, Pooling, Tensor shapes |
| 3 | Parameters and memory | Ch 2–4 model loadout, 32 MB model memory, Bench · Parameters | `game/device.py`, `game/flow.py` `model_picker`, `card_state` | Each loadout row shows measured size; YOLO26m doesn't fit, and the ch 3 watchdog leaves 11.6 MB for ch 4. Tip: Model size |
| 3 | Skip connections, U-Net | Ch 4, Bench · Architectures | `cv/segmentation.py` `TinyUNet`, `cv/experiments.py` `unet_without_skips` | Skip ablation on the trained U-Net: without the full-resolution skip the mask IoU falls from about 0.97 to about 0.2. Tips: Skip connections, U-Net |
| 3 | Depthwise convolution, attention, inception | Bench · Architectures | `cv/experiments.py` `inspect_model`, `params_by_category` | Counts of conv, depthwise conv, BatchNorm, residual bottlenecks, attention (C2PSA) and SPPF blocks in YOLO26n-cls, YOLO26s and our U-Net; depthwise vs standard weight count. No Inception block exists in these models, and the bench says so: SPPF is the closest idea. Tips: Depthwise convolution, Attention, Inception and attention |
| 4 | Data augmentation | Ch 1 (Retrain mode; cleared · Bonus scan augmentation batch), U-Net training | `cv/enhancement.py` `random_augment`, `level1.augmentation_panel`, `segmentation.train` | Six randomly augmented copies of the clean photo, with "digits mirrored: not label-safe" on flips. Enhancement (this frame, at inference) vs augmentation (training). The U-Net is trained with random flips. Tips: Augmentation (3) |
| 4 | Mixup / CutMix | Bench · Augmentation | `cv/experiments.py` `mixup`, `cutmix` | Real blends of the padlock and teddy bear through YOLO26n-cls; the chart shows its answer jumping instead of following the mixed label. Tip: Mixup and CutMix |
| 5 | Optimizers (SGD, momentum, RMSprop, Adam), learning rate | Bench · Optimizers, U-Net training | `cv/lab.py` `optimizer_paths`, `segmentation.train` | All four `torch.optim` optimizers on a bowl or Rosenbrock, learning rate and steps adjustable; the U-Net is trained with Adam and a cosine schedule. Tips: Optimizers, Learning rate |
| 5 | Loss functions | Bench · Losses, U-Net training | `cv/experiments.py` `mask_scores`, `grow_mask` | BCE (what our U-Net was trained with) and Dice next to IoU and pixel accuracy, for model masks, simple baselines and the true mask grown or shrunk by up to 9 px. Tips: Loss functions, Dice loss, Dice and IoU |
| 5 | Metrics: precision, recall, F1, accuracy | Ch 3, ch 4, Bench · Losses | `cv/detection.py` `scores`, `level3.py`, `level4.py` | Free threshold slider, a client brief, TP/FP/FN and P/R/F1 after each report. Cleared · Numbers: P/R/F1 against the threshold with the brief's line. Ch 4 and Bench · Losses: an empty mask scores 84–91% pixel accuracy and 0 IoU. Tips: Precision and recall, Accuracy can lie |
| 5 | Activation functions | Bench · Activations, Bench · Architectures | `cv/lab.py` `activation_curves` | ReLU, Leaky ReLU, SiLU, sigmoid, tanh, GELU with derivatives from autograd; the architecture table shows SiLU in YOLO26 and ReLU in our U-Net. Tips: ReLU, SiLU |
| 6 | Object detection, detection labels | Ch 3 | `level3.py`, `cv/detection.py`, `assets/level3/ground_truth.json` | YOLO26n/s/m on the parlour against our hand-checked boxes. Box inspector: zoom on one box and see its size at the model's 640 px input (a cup is about 18×15 px there). After a report each false alarm says why (duplicate, loose box, wrong label, nothing labelled). Cleared · Numbers: our ground truth as YOLO label lines. Tips: Detection labels, COCO |
| 6 | Input resolution and small objects | Ch 3 inspector, ch 3 Bonus scan | `detection.input_size`, `zoom_crop`; `level3.tray_scan` | The inspector's px → input-size line, and the tea tray side scan: the light detector finds no cups in the full photo and all four on a crop. Tip: Input resolution |
| 6 | IoU | Ch 3 matching (box IoU ≥ 0.5), ch 4 mask IoU | `detection.py` `iou`, `match`; `segmentation.py` `mask_iou` | Inspector shows each correct box's IoU. Cleared · Bonus scan: precision/recall/F1 as the matching IoU goes from 0.50 to 0.95. Tip: IoU |
| 6 | mAP | Ch 3 loadout (published COCO mAP50-95), ch 3 cleared | `cv/edge.py`, `level3.py` | Quoted, not computed (one scene is too small); the matching-IoU sweep shows why mAP50-95 is below mAP50. Tip: mAP |
| 6 | YOLO26, COCO, NMS | Ch 1–4 (wrong modes use YOLO26 too) | `cv/edge.py`, `level3.py` "The maths" | COCO has no padlock and no stain, shown in play; YOLO26 is NMS-free. Tips: NMS, COCO |
| 6 | Image segmentation | Ch 4 (semantic, our U-Net); YOLO26s-seg as a wrong mode in ch 2 and 3 | `level4.py`, `segmentation.py` | Mask, probability map and box views; the box view shows how much of the box is healthy wall. Mask threshold, two IoU checks. Cleared · Numbers: for the same network size, YOLO26s takes 9.3 ms to classify, 48.6 to detect, 70.1 to segment. Tip: Segmentation |
| 6 | Mask post-processing (morphology) | Ch 4 cleanup | `segmentation.clean_mask` | Opening ("remove specks") and closing ("fill gaps") with elliptical kernels of 3–9 px, applied to the mask before it is scored. Cleared · Numbers: this wall's IoU at every kernel size. On our thin-tendril stains cleanup costs IoU, and it can't fix the model's IoU on the 150 test walls |
| 7 | Efficient Edge AI: memory | 32 MB model memory, watchdog stays resident | `game/device.py`, `flow.card_state` | DOES NOT FIT badge; YOLO26m (44.3 MB) can't load |
| 7 | Compute, latency, energy | Every paid run, mission latency limits, battery | `cv/edge.py` `energy_units`, `flow.run_button` | Battery rule: 1 ms of measured compute = 0.1% battery (a game rule; real energy ≈ power × time, which the Power tip says). Ch 1 report: "Fast enough for live video" (33 ms). Tips: Edge AI, Latency budget, FLOPs vs speed, Power, The edge trade-off |
| 7 | Quantization | Ch 4 INT8 toggle, real FX post-training static quantization | `segmentation.quantize` | Only Standard INT8 meets IoU ≥ 0.95 and ≤ 40 ms. Once both precisions have run, the INT8 diff view shows the few pixels where the FP32 and INT8 masks disagree. Cleared · Numbers: latency vs IoU for all six U-Nets ("INT8 moves left, not down"). Tip: Quantization |
| 7 | Pruning | Bench · Pruning | `cv/experiments.py` `prune_copy`, `measure` | Global L1 unstructured pruning of U-Net Standard at 0–90%: sparsity, IoU, latency, file size, next to INT8 and Lite, and the pruned mask at each amount. Tips: Pruning, Unstructured pruning |
| 7 | Model selection under constraints | Ch 2–4 loadouts, ch 4 cleared | `flow.model_picker`, `level4.frontier_chart` | The same n / s / m choice decided by different limits: confidence only (ch 2), recall and memory (ch 3), IoU and latency (ch 4). Tip: The edge trade-off |
| 8 | Image normalization | Bench · Normalization; ch 1 "The maths" | `cv/experiments.py` `normalize_input`, `enc1_stats` | The U-Net on ÷255 (as trained), raw 0–255 and ImageNet mean/std inputs: IoU about 0.97 / 0.2–0.3 / about 0. Ch 1 says why enhancement is not normalization. Tips: Normalization, Image normalization |
| 8 | Feature normalization (BatchNorm) | Bench · Normalization, U-Net blocks | `segmentation.block`, `experiments.enc1_stats` | The first conv's channel means for each input scaling next to what BatchNorm stored in training. Tip: Batch norm |
| 8 | Weight initialization | Bench · Normalization | `cv/experiments.py` `init_stds` | 12 conv + ReLU layers with zeros, N(0, 1), PyTorch default or Kaiming init, with and without BatchNorm. Tip: Weight initialization |
| 8 | Regularization, overfitting, data splits | Ch 4 Bonus scan, tips | `level4.second_wall_scan` | The deployed U-Net on a second wall at the untuned threshold: the honest test number. Separate seeds for training, validation and INT8 calibration walls. Tips: Overfitting, Train, validate, test |
| 9 | GANs, diffusion | Not covered | | See below |

Enhancement (ch 1) is image **preprocessing**: it changes one frame so its content is easier to see. It is not image
normalization, which rescales every input the way the training data was scaled and makes nothing more visible.
Normalization is practised on Bench · Normalization.

## What is not covered, and why

- **Module 9 (GANs, diffusion).** Generating images doesn't fit an inference game on a small device, and training a
  generator is far beyond a laptop session. Nothing in GhostLens covers it.
- **Full fine-tuning for transfer learning.** The Workbench trains a linear probe (one layer on a frozen backbone), not
  a fine-tuned classifier on a new dataset. The professor's `Train_Models/mymodels.py` already does that.
- **Computing mAP.** One scene with 7 objects is too small for a meaningful mAP. We compute precision, recall and F1,
  sweep the matching IoU, and quote the published COCO mAP.
- **Structured pruning.** The Workbench only does unstructured pruning, which shows why zeros alone don't speed up a
  dense model. Removing whole channels and fine-tuning is left as future work.
- **Inception blocks and transformers.** None of the models in the game has an Inception block or a transformer
  backbone; Bench · Architectures says so and points at SPPF and the C2PSA attention block instead.
- **Dice loss training.** Bench · Losses scores masks with Dice, but our U-Net is trained with BCE only.
- **Real hardware and measured power.** Everything runs on a laptop CPU. Battery is modelled as compute time, not
  measured energy.

## What we took from the professor's Streamlit apps

| Professor's app | What it does | What GhostLens took |
|---|---|---|
| `Understanding_task/tasksel.py` | Pick the right CV task for a goal | The mode dial at the start of every chapter, and the closed case's question → task table |
| `Understanding choices/choice.py` | Story scene → choice → explained consequence | The overall format: story, a choice, a real consequence with a one-line explanation, then the debrief |
| `augmentation/aug.py` | Brightness / rotation / blur / noise sliders, Mixup, CutMix | Chapter 1 sliders and augmentation batch; Bench · Augmentation with Mixup and CutMix on a real classifier |
| `Train_Models/mymodels.py` | Small CNN, transfer learning | Chapter 2 pretrained classifiers; chapter 4 training our own small CNN; Bench · Parameters linear probe |
| `Detection/det.py` | Backbones, detection outputs, IoU | Chapter 3 detection, IoU matching, model size trade-off, false alarm explanations |
| `networks/*.py` | Layer builder, parameter counts, activations | Benches Convolution, Parameters, Activations and Architectures; param counts in the chapter 4 numbers |
| `optimization/optimi.py` | Optimizers | Bench · Optimizers; Adam + cosine schedule in U-Net training |

One thing we did differently: where the professor's augmentation stress test uses hand-written score formulas,
GhostLens measures every number or labels it as a published value or a game rule.
