# GhostLens and the UCS668 syllabus

Where each topic of UCS668 Edge AI and Robotics: Data Center Vision shows up in the game. "Tip" means a loading
screen tip in `game/tips.py` (they are collected in the Field guide). "Learn more" means the expander at the bottom of a chapter.

Sources: the course site and https://github.com/jhilikb/UCS668.

## Module by module

| Module | Topic | Where in GhostLens | File | Mechanic |
|---|---|---|---|---|
| 1 | Understanding the vision problem, task identification | Every chapter starts with a mode choice: Enhance / Classify / Detect / Segment | `game/flow.py` `mode_choice`, `level1-4.py` `MODE_OPTIONS` | Wrong task really runs, costs battery, shows why it can't answer. Tips: "Choosing the task" (3) |
| 2 | First vision model: classification | Chapter 2 | `game/level2.py`, `cv/classification.py` | YOLO26n/s/m-cls on three objects, top-5 softmax bars, room photo quiz ("one label for the whole image") |
| 2 | Softmax, confidence, label set | Chapter 2 | `level2.py` Learn more | Worked logit -> softmax table; "ImageNet has no pocket watch" |
| 2 | Transfer learning | Chapter 2 Learn more, tips | `level2.py`, `tips.py` | Explained, not trained in-game (see "not covered") |
| 3 | Building blocks: Conv2D, BatchNorm, ReLU, pooling | Chapter 4 U-Net, Lab convolution tab | `cv/segmentation.py` `block`, `TinyUNet`, `game/lab.py` | Our own model; param counts on the cards. Lab: kernels, stride, padding, feature maps. Tips: Conv2D, Pooling |
| 3 | Parameters, tensor shapes | Chapter 4 cards (7.7 k / 118.9 k / 472.8 k params), Lab layer table, tips | `cv/edge.py`, `cv/lab.py`, `tips.py` | Tips: Conv2D param formula, (B, C, H, W) |
| 3 | Skip connections | U-Net decoder | `cv/segmentation.py` `TinyUNet.forward` | Tips: Skip connections, U-Net |
| 3 | Depthwise conv, attention, inception, transformer | Tips only | `tips.py` | Depthwise conv tip; others not covered |
| 4 | Data augmentation | Chapter 1 augmentation panel, "Retrain" mode, U-Net training flips | `cv/enhancement.py` `random_augment`, `level1.py` `augmentation_panel`, `segmentation.py` `train` | 6 randomly augmented copies; enhancement vs augmentation |
| 4 | Mixup / CutMix | Tip only | `tips.py` | |
| 5 | Optimizers (SGD, momentum, Adam, RMSProp) | U-Net training uses Adam + cosine LR; Lab optimizer race; tips | `segmentation.py` `train`, `cv/lab.py`, `tips.py` | Lab runs all four on the same loss surface |
| 5 | Loss functions | U-Net uses BCEWithLogitsLoss; tips | `segmentation.py` `train` | |
| 5 | Metrics: precision, recall, F1, accuracy | Chapter 3 | `cv/detection.py` `scores`, `level3.py` | Threshold slider, TP/FP/FN, PR-vs-threshold chart. Tip: "Accuracy can lie" |
| 5 | Activation functions | ReLU in U-Net, sigmoid on the output; Lab activations tab; tip | `segmentation.py`, `cv/lab.py` | Lab plots each function and its derivative |
| 6 | Object detection, detection labels | Chapter 3 | `level3.py`, `cv/detection.py`, `assets/level3/ground_truth.json` | YOLO26n/s/m on the parlour, hand-checked boxes |
| 6 | IoU | Chapter 3 matching (box IoU >= 0.5), chapter 4 mask IoU | `detection.py` `iou`, `match`; `segmentation.py` `mask_iou` | |
| 6 | mAP | Published COCO mAP on model cards; Learn more; tip | `cv/edge.py`, `level3.py` | Quoted, not computed |
| 6 | YOLO26, COCO | Chapters 1-4 (wrong modes use YOLO26 too) | `cv/edge.py` | COCO has no padlock, no stain: shown in-game |
| 6 | NMS | Learn more + tip (YOLO26 is NMS-free) | `level3.py`, `tips.py` | |
| 6 | Image segmentation | Chapter 4 (semantic, our U-Net); YOLO26s-seg as a wrong mode in ch. 2 and 3 | `level4.py`, `segmentation.py` | Box vs mask, "healthy wall inside box" |
| 7 | Efficient Edge AI: memory | 32 MB model memory, cards that don't fit, watchdog stays resident | `game/device.py`, `flow.py` `model_picker` | YOLO26m doesn't fit |
| 7 | Compute / latency | Measured latency per model, mission latency limits, battery = latency / 5 ms | `cv/edge.py`, `flow.py` | Every run costs units |
| 7 | Quantization | Chapter 4 INT8 toggle, real FX post-training static quantization | `segmentation.py` `quantize` | Only Standard INT8 meets every limit |
| 7 | Pruning | Tip only | `tips.py` | |
| 8 | Image normalization | Chapter 1 (gamma, equalization, CLAHE as intensity normalization); tip on mean/std | `cv/enhancement.py`, `tips.py` | |
| 8 | Feature normalization | BatchNorm in the U-Net; tip | `segmentation.py` `block` | |
| 8 | Regularization, weight init | Tips (overfitting, train/val/test) | `tips.py` | Weight init not covered |
| 9 | GANs, diffusion | Not covered | | Out of scope |

Image processing that sits under modules 4 and 8: brightness, contrast, gamma, histogram, global equalization, CLAHE,
Gaussian / median / non-local means denoising, unsharp masking, clipping (chapter 1, `cv/enhancement.py`).

## What is not covered, and why

- **Training a classifier with transfer learning in the app.** It takes minutes per run and the professor's
  `Train_Models/mymodels.py` already does it. We explain it instead.
- **Attention, inception, transformers.** Not needed by any chapter; they would be text only.
- **Computing mAP.** One scene with 7 objects is too small for a meaningful mAP. We compute P/R/F1 and quote the
  published mAP.
- **Pruning.** We do quantization for real; pruning is a tip. Structured pruning of a 120 k param U-Net wouldn't show
  much.
- **Weight initialization.** Uses PyTorch defaults.
- **Module 9 (GANs, diffusion).** Out of scope for an edge inference game.
- **Real hardware.** Everything runs on a laptop CPU; there's no Jetson or phone NPU measurement.

## What we took from the professor's Streamlit apps

| Professor's app | What it does | What GhostLens took |
|---|---|---|
| `Understanding_task/tasksel.py` | Pick the right CV task for a goal | The mode choice at the start of every chapter |
| `Understanding choices/choice.py` | Story scene -> choice -> explained consequence | The overall format: story, a choice, a real consequence with an explanation |
| `augmentation/aug.py` | Brightness / rotation / blur / noise sliders, Mixup, CutMix | Chapter 1 lab sliders and the augmentation panel |
| `Train_Models/mymodels.py` | Small CNN, transfer learning | Chapter 2 pretrained classifiers; chapter 4 training our own small CNN |
| `Detection/det.py` | Backbones, detection outputs, IoU | Chapter 3 detection, IoU matching, model size trade-off |
| `networks/*.py` | Layer builder, parameter counts | U-Net building blocks and param counts on the chapter 4 cards |
| `networks/activations.py` | Activation functions | ReLU / sigmoid in the U-Net, ReLU tip |
| `optimization/optimi.py` | Optimizers | Adam + cosine schedule in U-Net training, optimizer tips |

One thing we did differently: where the professor's augmentation stress test uses hand-written score formulas, GhostLens
measures every number or labels it as a published value or a game rule.
