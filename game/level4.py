"""Chapter 4: The Corrupted Region (segmentation, model size and quantization)."""

import cv2
import numpy as np
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv import segmentation as seg
from cv.edge import energy_units
from cv.models import ModelMissing
from game import device, flow, runtime
from game.levels import case_closed, completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

LIMITS = {"latency_ms": 40, "iou": 0.95}
BOX_COLOR = (47, 134, 183)       # amber, BGR
MASK_COLOR = np.array([90, 125, 94])  # green, BGR


@st.cache_data
def scene() -> tuple[np.ndarray, np.ndarray]:
    return seg.make_scene()


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def with_box(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = seg.bounding_box(mask)
    cv2.rectangle(out, (x0, y0), (x1, y1), BOX_COLOR, 2)
    return out


def with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    out[mask] = out[mask] * 0.45 + MASK_COLOR * 0.55
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = out.astype(np.uint8)
    cv2.drawContours(out, contours, -1, (40, 60, 40), 1)
    return out


def run_classifier() -> tuple[str, float]:
    st.session_state.l4_cls = clf.classify(runtime.yolo("yolo26n-cls.pt"), scene()[0])
    return "YOLO26n-cls on the wall", runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"]


def show_classifier() -> None:
    top = st.session_state.l4_cls["top"]
    c1, c2 = st.columns([1, 1.6])
    c1.image(rgb(scene()[0]), width="stretch")
    c2.markdown("**GhostLens says:**  \n" + "  \n".join(f"`{label}` {p:.0%}" for label, p in top[:3])
                + "\n\nNo stain class in ImageNet, so it picked whatever it looks most like. Even with a 'stain' "
                  "class, one label for the whole image says nothing about which pixels to clean.")


def run_detector() -> tuple[str, float]:
    st.session_state.l4_dets = det.detect(runtime.yolo("yolo26s.pt"), scene()[0], min_conf=0.25)[0]
    return "YOLO26s on the wall", runtime.profile("detectors", "yolo26s.pt")["latency_ms"]


def show_detector() -> None:
    img, truth = scene()
    dets = st.session_state.l4_dets
    c1, c2 = st.columns([1, 1.6])
    c1.image(rgb(with_box(img, truth)), width="stretch")
    c2.markdown(
        f"**The detector found {len(dets)} COCO object(s)** and no stain, because 'stain' isn't one of its 80 classes. "
        "Suppose you trained one that knew stains. The best it could give you is a box like this one.\n\n"
        f"Purifying that box would scrub **{seg.clean_share_of_box(truth):.0%} healthy wall** along with the "
        "corruption, and destroy the evidence underneath.")


MODE_OPTIONS = {
    "Classify": {
        "blurb": "One label for the whole image. Cheapest.",
        "run": run_classifier, "show": show_classifier,
        "verdict": "Classification can say 'there's a stain'. Purification needs to know which pixels.",
    },
    "Detect": {
        "blurb": "A box around each object.",
        "run": run_detector, "show": show_detector,
        "verdict": "A box is a rough location. The stain is irregular, so most of its box is healthy wall. "
                   "You need a pixel-level boundary.",
    },
    "Segment": {
        "blurb": "A label for every pixel: stain or wall. Most expensive.",
        "verdict": "Exactly which pixels are corrupted is a segmentation question. It's the most expensive task, "
                   "so now the model has to fit the device.",
    },
}


def warm() -> None:
    scene()
    runtime.unet("standard", False)


WARMUP = [("Reading the parlour wall", scene), ("Measuring segmentation models", runtime.benchmark),
          ("Loading U-Net", warm)]


def profiles(int8: bool) -> list[dict]:
    precision = "INT8" if int8 else "FP32"
    tiers = {"lite": "Light", "standard": "Balanced", "pro": "Heavy"}
    return [{**r, "tier": tiers[r["variant"]], "name": f"{r['name']} {precision}",
             "accuracy": f"{r['iou']:.3f} IoU", "accuracy_tag": "measured"}
            for r in runtime.benchmark()["unets"] if r["precision"] == precision]


def report_checks(profile: dict) -> dict:
    s = st.session_state
    used = device.used_in_level(s, 4)
    best = runtime.profile("unets", "standard-int8")
    optimal = energy_units(best["latency_ms"])
    misses = flow.mode_misses(4)
    tried = s.get("l4_purify_tries", 1)
    return {
        "Right task": (misses == 0, "segmentation first time" if misses == 0
                       else f"tried {', '.join(m for m in s.l4_tried if m != 'Segment')} first"),
        "Met every limit first time": (tried == 1, "first purification passed" if tried == 1
                                       else f"{tried - 1} failed purification attempt(s)"),
        "Latency target": (profile["latency_ms"] <= LIMITS["latency_ms"], f"≤ {LIMITS['latency_ms']} ms"),
        "Battery": (efficient(used, optimal), f"{used} units used, {optimal} would have done it"),
    }


def render() -> None:
    ui.scene_header(
        "CHAPTER 4 · PARLOUR WALL · 04:44",
        "The Corrupted Region",
        "The watchdog flagged it at 04:40: a stain spreading over the wall between the curtains, a little bigger "
        "every minute. The hotel can purify it, but whatever you mark gets scraped off, so mark too much and you "
        "destroy healthy wall and whatever's written under the paint.",
    )
    try:
        runtime.benchmark()
        img, truth = scene()
    except ModelMissing as e:
        st.warning(str(e))
        return
    s = st.session_state
    solved = 4 in s.completed_levels

    if not solved and not flow.mode_choice(
            4, "You need the exact shape of the stain, so only the corrupted pixels get purified.",
            MODE_OPTIONS, "Segment"):
        st.image(rgb(img), width=380)
        ui.caption("EVIDENCE 05-W · PARLOUR WALL")
        return

    st.markdown(f'<div class="gl-mission"><span class="gl-kicker">Mission</span><span>purify only the stain</span>'
                f'<span>mask IoU ≥ {LIMITS["iou"]}</span><span>≤ {LIMITS["latency_ms"]} ms per frame</span></div>',
                unsafe_allow_html=True)
    int8 = st.toggle("Quantize to INT8", key="l4_int8",
                     help="Post-training quantization. The numbers on the cards are measured for both versions.")
    st.caption("Stores weights and activations as 8-bit integers instead of 32-bit floats: smaller and faster.")
    model = flow.model_picker(
        4, profiles(int8), LIMITS, slot="task",
        note="Accuracy is mask IoU measured on 150 generated walls the model never saw. The watchdog detector from "
             "chapter 3 is still loaded and using memory.")
    if model is None:
        return

    ran = s.setdefault("l4_ran", [])
    if model["id"] not in ran:
        units = energy_units(model["latency_ms"])
        if st.button(f"Run {model['name']} on the wall  ·  {units} units", type="primary"):
            flow.run_cost(4, f"{model['name']} on the wall", model["latency_ms"])
            ran.append(model["id"])
            st.rerun()
        return

    net = runtime.unet(model["variant"], model["precision"] == "INT8")
    probs = seg.predict(net, img)
    threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05, key="l4_thr",
                          help="A pixel counts as stain if the model's probability is above this.")
    mask = probs > threshold
    iou = seg.mask_iou(mask, truth)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(rgb(img), width="stretch")
        ui.caption("ORIGINAL")
    with c2:
        st.image(rgb(with_box(img, mask)), width="stretch")
        ui.caption("A BOUNDING BOX AROUND IT")
    with c3:
        st.image(rgb(with_mask(img, mask)), width="stretch")
        ui.caption(f"SEGMENTATION · {model['name'].upper()}")
    ui.readouts([
        ("Stain area", f"{mask.mean():.1%}", ""),
        ("Healthy wall inside box", f"{seg.clean_share_of_box(mask):.0%}", "bad"),
        ("Mask IoU (this wall)", f"{iou:.3f}", "ok" if iou >= LIMITS["iou"] else "warn"),
        ("Model IoU (150 walls)", f"{model['iou']:.3f}", "ok" if model["iou"] >= LIMITS["iou"] else "bad"),
    ])
    st.caption(f"{ui.tag('measured')} We painted the stain onto the wall ourselves, so the true mask is known exactly.",
               unsafe_allow_html=True)

    if solved:
        st.divider()
        completion_panel(4, [("Model deployed", model["name"]), ("Mask IoU", f"{iou:.3f}")])
        ui.lesson([
            "Segmentation answers <b>exactly which pixels?</b> A box around an irregular shape is mostly background.",
            "It's the most expensive of the three tasks: a prediction for every pixel.",
            "The light model was fast but its masks leaked. The heavy one was accurate but too slow.",
            "Quantizing the balanced model to INT8 made it 4× smaller and fast enough, for almost no accuracy.",
            "On a device, the best model is the one that meets every limit, not the most accurate one.",
        ], title="FIELD NOTES")
        st.divider()
        case_closed()
        return

    if st.button("Purify the marked pixels", type="primary"):
        s.l4_purify_tries = s.get("l4_purify_tries", 0) + 1
        problems = []
        if model["iou"] < LIMITS["iou"]:
            problems.append(f"{model['name']} only reaches {model['iou']:.3f} IoU on average. Its masks leak onto "
                            "healthy wall and miss thin tendrils, and the purification scraped the wrong pixels.")
        if model["latency_ms"] > LIMITS["latency_ms"]:
            problems.append(f"At {model['latency_ms']:.0f} ms per frame it can't keep up with the stain while it "
                            "spreads. By the time the mask is ready, it's out of date.")
        if problems:
            ui.message("<b>Purification failed.</b> " + " ".join(problems)
                       + " Try a different model, or quantize this one.", "bad")
        else:
            finish_level(4, iou, s.l4_purify_tries, report_checks(model), clue="Corruption purified")
