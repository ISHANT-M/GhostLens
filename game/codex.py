"""The GhostLens Field Guide: tips found, glossary and badges."""

from collections.abc import MutableMapping
from html import escape

import streamlit as st

from game import achievements
from game.tips import TIPS
from ui import components as ui

TABS = ["Tips found", "Glossary", "Badges"]

GLOSSARY = [
    ("Classification", "Gives one label to the whole image, picked from a fixed list of classes."),
    ("Detection", "Finds each object in the image and gives it a label and a bounding box."),
    ("Segmentation", "Gives a label to every pixel, so you know the exact shape of each region."),
    ("Logit", "The raw score a network outputs for a class before softmax. It can be any real number."),
    ("Softmax", "Turns logits into probabilities that are all positive and add up to 1."),
    ("IoU", "Intersection over union. The overlap area of two boxes or masks divided by their combined area."),
    ("Precision", "Of everything the model flagged, the fraction that was really correct. TP / (TP + FP)."),
    ("Recall", "Of everything that was really there, the fraction the model found. TP / (TP + FN)."),
    ("F1", "The harmonic mean of precision and recall. It is only high when both are high."),
    ("mAP", "Mean average precision. Area under the precision-recall curve, averaged over classes and often over IoU thresholds."),
    ("NMS", "Non-maximum suppression. Removes duplicate boxes for the same object by keeping the highest-scoring one."),
    ("Confidence threshold", "Detections below this score are dropped. Raising it usually raises precision and lowers recall."),
    ("Augmentation", "Random changes to training images (flips, crops, brightness) so the model learns to handle variety. Used only in training."),
    ("Enhancement", "Preprocessing one input at run time, like brightening a dark frame, so the model can see it better."),
    ("CNN", "Convolutional neural network. Stacks of small learned filters that slide over the image."),
    ("Conv2D", "A layer that slides k×k filters over the image. Parameters = k·k·C_in·C_out + C_out, independent of image size."),
    ("Pooling", "Shrinks a feature map, for example max pooling 2×2 halves width and height."),
    ("ReLU", "An activation that keeps positive values and sets negatives to zero. Cheap and trains well."),
    ("Transfer learning", "Start from a model pretrained on a big dataset and retrain only the last layers on your own classes."),
    ("Quantization", "Storing weights (and often activations) with fewer bits, which makes the model smaller and faster."),
    ("INT8", "8-bit integers. A model stored in INT8 is about 4× smaller than in FP32."),
    ("Pruning", "Removing weights or channels that matter little, to make a model smaller or faster."),
    ("Latency", "Time for one input to go through the model. At 30 fps you have about 33 ms per frame."),
    ("FLOPs / MACs", "Counts of arithmetic operations in one forward pass. One multiply-accumulate (MAC) is about 2 FLOPs."),
    ("Edge device", "A small computer near the camera, like a phone or a single-board computer, with limited memory, power and compute."),
    ("COCO", "A dataset of everyday scenes with 80 object classes, boxes and masks. Common for detection and segmentation."),
    ("ImageNet", "A large classification dataset. The usual subset has 1000 classes and about 1.2M training images."),
    ("U-Net", "A segmentation network that downsamples then upsamples, with skip connections that keep fine detail."),
    ("Skip connection", "A path that carries a tensor past some layers, either added (ResNet) or concatenated (U-Net)."),
    ("Depthwise convolution", "A conv where each channel gets its own k×k filter (groups = channels). Followed by a 1×1 conv to mix channels."),
    ("Attention", "A block that computes weights from the features themselves to decide which positions or channels matter most."),
    ("SiLU", "An activation, x·sigmoid(x). Smooth, slightly negative for small negative x. Used throughout YOLO26."),
    ("BatchNorm", "Normalizes each channel with the batch mean and std in training, and with stored running statistics at inference."),
    ("Image normalization", "Scaling inputs the same way as in training, e.g. ÷255 or subtracting the ImageNet mean and dividing by its std."),
    ("Kaiming init", "Random initial weights with std √(2 / fan_in), chosen so ReLU layers keep the signal size stable."),
    ("BCE", "Binary cross-entropy. The per-pixel loss our U-Net was trained with. Confident wrong pixels cost the most."),
    ("Dice", "2·|A∩B| / (|A| + |B|) for two masks. Like IoU it ignores the background; Dice = 2·IoU / (1 + IoU)."),
    ("Mixup", "Augmentation that blends two images with weight λ and blends their labels the same way."),
    ("CutMix", "Augmentation that pastes a patch of one image into another and mixes the labels by patch area."),
    ("Linear probe", "Train only a new last Linear layer on top of a frozen pretrained network, to test how good its features are."),
]

def tips_tab() -> None:
    seen = st.session_state.get("seen_tips", set())
    st.markdown(f'<div class="gl-codex-progress">{len(seen)} of {len(TIPS)} found</div>', unsafe_allow_html=True)
    topics = list(dict.fromkeys(topic for topic, _ in TIPS))
    hidden = []
    for topic in topics:
        found = [text for i, (t, text) in enumerate(TIPS) if t == topic and i in seen]
        missing = sum(1 for i, (t, _) in enumerate(TIPS) if t == topic and i not in seen)
        if not found:
            hidden.append(topic)
            continue
        html = f'<div class="gl-codex-topic">{escape(topic)}</div>'
        html += "".join(f'<div class="gl-codex-tip">{escape(text)}</div>' for text in found)
        if missing:
            html += f'<div class="gl-codex-tip locked">{missing} more not found yet</div>'
        st.markdown(html, unsafe_allow_html=True)
    if hidden:
        st.markdown(f'<div class="gl-codex-tip locked">Still hidden: {escape(", ".join(hidden))}. '
                    'Loading screens reveal these.</div>', unsafe_allow_html=True)


def glossary_tab() -> None:
    rows = "".join(f'<div class="gl-codex-term"><b>{escape(term)}.</b> {escape(text)}</div>'
                   for term, text in GLOSSARY)
    st.markdown(rows, unsafe_allow_html=True)


LEGACY_PREFIXES = ("quiz_", "guide_")   # the old quiz and its XP pot, from before 0.8


def forget_guide(store: MutableMapping) -> None:
    store.pop("seen_tips", None)
    for key in [k for k in store.keys() if str(k).startswith(LEGACY_PREFIXES)]:
        store.pop(key, None)


def badge_card(badge: dict, got: bool) -> str:
    cls = "gl-badge-card" + ("" if got else " locked") + (" anti" if badge["id"] == "fumes" else "")
    state = "EARNED" if got else "LOCKED"
    desc = badge["description"] if got else badge["hint"]
    return (f'<div class="{cls}"><div class="state">{state}</div><div class="name">{escape(badge["name"])}</div>'
            f'<div class="desc">{escape(desc)}</div></div>')


def badges_tab() -> None:
    got = achievements.earned_ids(st.session_state)
    st.markdown(f'<div class="gl-codex-progress">{len(got)} of {len(achievements.BADGES)} earned</div>',
                unsafe_allow_html=True)
    cards = "".join(badge_card(b, b["id"] in got) for b in achievements.all_badges())
    st.markdown(f'<div class="gl-badge-grid">{cards}</div>', unsafe_allow_html=True)


def badge_strip() -> None:
    got = achievements.earned(st.session_state)
    if not got:
        return
    chips = "".join(f'<span class="gl-badge-chip">{escape(b["name"])}</span>' for b in got)
    st.markdown(f'<div class="gl-kicker">Badges earned</div><div class="gl-badge-row">{chips}</div>',
                unsafe_allow_html=True)


def render() -> None:
    ui.scene_header("FIELD GUIDE", "The GhostLens Field Guide",
                    "Everything you picked up on the case: tips, terms and your badges.")
    tabs = st.tabs(TABS)
    for tab, fn in zip(tabs, [tips_tab, glossary_tab, badges_tab]):
        with tab:
            fn()
    st.divider()
    st.caption("The field guide stays when you restart the case. This clears it too.")
    if st.button("Forget the field guide", type="tertiary", key="forget_guide"):
        forget_guide(st.session_state)
        st.rerun()
