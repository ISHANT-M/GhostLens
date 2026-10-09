"""The GhostLens Field Guide: tips found, glossary, quiz and badges."""

import random
from collections.abc import MutableMapping
from html import escape

import streamlit as st

from game import achievements
from game.tips import TIPS
from ui import components as ui

QUIZ_XP = 5

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

# (id, question, options, answer index, explanation)
QUIZ = [
    ("q1", "You need to count people in a queue. Which task?", ["Classification", "Detection", "Segmentation"], 1,
     "Counting needs one box per person, which is detection."),
    ("q2", "You need the exact area of a leaf that is diseased. Which task?", ["Classification", "Detection", "Segmentation"], 2,
     "Area needs pixel-level masks, which is segmentation."),
    ("q3", "One object fills the frame and you only need its name. Which task?", ["Classification", "Detection", "Segmentation"], 0,
     "One label for the whole image is classification, the cheapest option."),
    ("q4", "A detector finds 8 boxes and 6 are correct. What is its precision?", ["0.60", "0.75", "0.80"], 1,
     "Precision = TP / all predictions = 6 / 8 = 0.75."),
    ("q5", "There are 10 ghosts and the detector finds 6 of them. What is recall?", ["0.6", "0.4", "1.0"], 0,
     "Recall = TP / all real objects = 6 / 10 = 0.6."),
    ("q6", "Raising the confidence threshold usually does what?", ["Raises recall", "Raises precision, lowers recall", "Nothing"], 1,
     "Fewer, more confident boxes means fewer false positives but more misses."),
    ("q7", "Two boxes have overlap 20 and union 80. What is IoU?", ["0.20", "0.25", "4.0"], 1,
     "IoU = intersection / union = 20 / 80 = 0.25."),
    ("q8", "Precision is 1.0 and recall is 0.5. What is F1?", ["0.75", "0.67", "0.50"], 1,
     "F1 = 2PR / (P + R) = 1.0 / 1.5 ≈ 0.67."),
    ("q9", "A model has 10M parameters in FP32. About how big is it?", ["10 MB", "40 MB", "80 MB"], 1,
     "FP32 is 4 bytes per parameter, so 10M × 4 = 40 MB."),
    ("q10", "The same 10M-parameter model in INT8 is about...", ["10 MB", "20 MB", "40 MB"], 0,
     "INT8 is 1 byte per parameter, so about 10 MB."),
    ("q11", "A camera runs at 30 fps. What is the latency budget per frame?", ["10 ms", "33 ms", "100 ms"], 1,
     "1000 ms / 30 frames ≈ 33 ms per frame."),
    ("q12", "When is augmentation applied?", ["Training only", "Inference only", "Both always"], 0,
     "Augmentation adds variety during training. At test time the image is used as it is."),
    ("q13", "Brightening one dark frame before the model sees it is...", ["Augmentation", "Enhancement", "Quantization"], 1,
     "Changing one input at run time is enhancement (preprocessing)."),
    ("q14", "Which augmentation is risky for digit recognition?", ["Small brightness change", "Small crop", "Rotating 180°"], 2,
     "A 6 rotated 180° looks like a 9, so the label would be wrong."),
    ("q15", "What does softmax output?", ["Raw scores", "Probabilities that add up to 1", "Bounding boxes"], 1,
     "Softmax maps logits to positive values that sum to 1."),
    ("q16", "A 3×3 Conv2D with 64 in and 128 out channels (with bias) has how many parameters?", ["73,856", "8,192", "589,824"], 0,
     "3·3·64·128 + 128 = 73,856."),
    ("q17", "What does NMS remove?", ["Low-light pixels", "Duplicate boxes on the same object", "Unused weights"], 1,
     "Non-maximum suppression keeps the best box and drops overlapping duplicates."),
    ("q18", "Main reason to use transfer learning?", ["Needs far less data", "Makes the model bigger", "Removes the need for labels"], 0,
     "A pretrained backbone already knows edges and textures, so a few hundred images can be enough."),
    ("q19", "What does quantization usually cost?", ["A lot of accuracy", "A small drop in accuracy", "More memory"], 1,
     "Going to INT8 usually loses very little accuracy while making the model about 4× smaller."),
    ("q20", "Same YOLO26s size, three tasks. Which took longest per image in our benchmark?",
     ["Classification", "Detection", "Segmentation"], 2,
     "The -seg model predicts boxes plus a mask per object. Cost depends on the network too: our tiny U-Net Lite "
     "segments the wall faster than YOLO26s detects."),
    ("q21", "A wall is 10% stain. A mask that marks no stain at all has what pixel accuracy?", ["10%", "50%", "90%"], 2,
     "Every clean pixel is right, so 90%. Its IoU is 0: pixel accuracy flatters a mask when most pixels are background."),
    ("q22", "A mask has IoU 0.5. What is its Dice score?", ["0.50", "0.67", "0.75"], 1,
     "Dice = 2·IoU / (1 + IoU) = 1.0 / 1.5 ≈ 0.67."),
    ("q23", "You set 50% of a conv's weights to zero but keep the dense tensor. CPU latency usually...",
     ["Halves", "Stays about the same", "Doubles"], 1,
     "A dense conv multiplies the zeros too. Removing whole channels or sparse kernels are what make it faster."),
    ("q24", "Which init keeps activations stable in a deep ReLU network without BatchNorm?",
     ["All zeros", "Kaiming (He)", "N(0, 1)"], 1,
     "Kaiming uses std √(2 / fan_in) for ReLU. Zeros make every neuron identical; N(0, 1) makes activations explode."),
    ("q25", "Mixup with λ = 0.7 of a cat image and 0.3 of a dog image. What is the training label?",
     ["cat", "0.7 cat + 0.3 dog", "dog"], 1, "Mixup blends the labels with the same λ as the pixels."),
    ("q26", "A depthwise 3×3 conv on 64 channels has how many weights (no bias)?", ["576", "36,864", "64"], 0,
     "One 3×3 filter per channel: 3·3·64 = 576. A normal 3×3 conv 64→64 has 36,864."),
    ("q27", "A model was trained on pixels ÷ 255. You feed it raw 0–255 values. What happens?",
     ["BatchNorm fixes it", "The output breaks", "It runs faster"], 1,
     "At inference BatchNorm uses statistics stored during training, so a 255× bigger input throws every layer off."),
    ("q28", "What does a U-Net skip connection carry?",
     ["Fine detail from the encoder to the decoder", "The loss value", "Extra training images"], 0,
     "The decoder gets the full-resolution encoder features back, so mask edges land in the right place."),
]

QUIZ_BY_ID = {q[0]: q for q in QUIZ}

# questions that revisit what each chapter just taught
CHAPTER_QUIZ = {
    1: ["q13", "q12", "q14", "q27"],
    2: ["q15", "q18", "q3", "q25"],
    3: ["q4", "q5", "q7", "q6", "q8", "q17"],
    4: ["q2", "q21", "q22", "q10", "q19", "q23", "q28"],
}


def _quiz_correct(store: MutableMapping) -> set:
    if not isinstance(store.get("quiz_correct"), set):
        store["quiz_correct"] = set(store.get("quiz_correct") or ())
    return store["quiz_correct"]


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


def _current_question(done: set):
    s = st.session_state
    left = [q for q in QUIZ if q[0] not in done]
    if not left:
        return None
    if s.get("quiz_current") not in [q[0] for q in left]:
        s["quiz_current"] = random.choice(left)[0]
    return next(q for q in left if q[0] == s["quiz_current"])


def check_answer(store: MutableMapping, qid: str, choice: int, feedback_key: str = "quiz_feedback") -> bool:
    # quiz XP is kept apart from case XP, so restarting the case doesn't take it away
    q = QUIZ_BY_ID[qid]
    done = _quiz_correct(store)
    if choice != q[3]:
        store[feedback_key] = ("bad", f"Not quite. {q[4]}")
        return False
    gained = qid not in done
    done.add(qid)
    store["guide_xp"] = store.get("guide_xp", 0) + QUIZ_XP * gained
    bonus = f" +{QUIZ_XP} guide XP." if gained else ""
    store[feedback_key] = ("ok", f"Correct.{bonus} {q[4]}")
    if feedback_key == "quiz_feedback":
        store.pop("quiz_current", None)
    return True


def inline_ids(store: MutableMapping, level: int, n: int) -> list[str]:
    """The n questions shown after a chapter, picked once: ones not yet answered right come first."""
    key = f"l{level}_iq_ids"
    if key not in store:
        pool, done = CHAPTER_QUIZ.get(level, []), _quiz_correct(store)
        store[key] = ([q for q in pool if q not in done] + [q for q in pool if q in done])[:n]
    return store[key]


def inline_question(level: int, qid: str) -> None:
    s = st.session_state
    _, text, options, _, _ = QUIZ_BY_ID[qid]
    key = f"l{level}_iq_{qid}"
    choice = st.radio(text, range(len(options)), format_func=lambda i: options[i], index=None, key=key)
    if st.button("Check", key=f"{key}_check", disabled=choice is None):
        check_answer(s, qid, choice, feedback_key=f"{key}_feedback")
    if fb := s.get(f"{key}_feedback"):
        ui.message(escape(fb[1]), fb[0])


def inline_quiz(level: int, n: int = 2) -> None:
    """A short quiz on what the chapter just taught. Right answers pay guide XP once, like the Field guide quiz."""
    ids = inline_ids(st.session_state, level, n)
    if not ids:
        return
    st.markdown('<div class="gl-kicker">Quick check · guide XP</div>', unsafe_allow_html=True)
    for qid in ids:
        inline_question(level, qid)


def forget_guide(store: MutableMapping) -> None:
    for key in ("seen_tips", "quiz_correct", "guide_xp", "quiz_feedback", "quiz_current"):
        store.pop(key, None)


def quiz_tab() -> None:
    s = st.session_state
    done = _quiz_correct(s)
    st.markdown(f'<div class="gl-codex-progress">{len(done)} of {len(QUIZ)} answered right · '
                f'guide XP {s.get("guide_xp", 0)}</div>', unsafe_allow_html=True)
    if fb := s.get("quiz_feedback"):
        ui.message(escape(fb[1]), fb[0])
    q = _current_question(done)
    if q is None:
        ui.message("You answered every question. Nice work.", "ok")
        return
    qid, text, options, _, _ = q
    choice = st.radio(text, range(len(options)), format_func=lambda i: options[i], index=None, key=f"quiz_{qid}")
    c1, c2 = st.columns(2)
    if c1.button("Check", key="quiz_check", disabled=choice is None):
        check_answer(s, qid, choice)
        st.rerun()
    if c2.button("Skip", key="quiz_skip"):
        s.pop("quiz_current", None)
        s.pop("quiz_feedback", None)
        st.rerun()


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
                    "Everything you picked up on the case: tips, terms, a quick quiz and your badges.")
    tabs = st.tabs(["Tips found", "Glossary", "Quiz", "Badges"])
    for tab, fn in zip(tabs, [tips_tab, glossary_tab, quiz_tab, badges_tab]):
        with tab:
            fn()
    st.divider()
    st.caption("The field guide stays when you restart the case. This clears it too.")
    if st.button("Forget the field guide", type="tertiary", key="forget_guide"):
        forget_guide(st.session_state)
        st.rerun()
