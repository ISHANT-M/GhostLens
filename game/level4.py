"""Chapter 4: The Corrupted Region (segmentation, model size and quantization)."""

import cv2
import numpy as np
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv import segmentation as seg
from cv.edge import energy_units
from cv.models import ModelMissing
from game import case, device, flow, runtime
from game.device import pct
from game.levels import case_closed, completion_panel, finish_level, int8_note
from game.scoring import efficient
from ui import components as ui

LIMITS = {"latency_ms": 40, "iou": 0.95}
SIDE_THRESHOLD = 0.5                  # the second wall is scored at the default threshold, untuned
BOX_COLOR = (47, 134, 183)            # amber, BGR
MASK_COLOR = np.array([90, 125, 94])  # green, BGR


@st.cache_data(show_spinner=False)
def scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    return seg.make_scene(seed)


def wall() -> tuple[np.ndarray, np.ndarray]:
    return scene(case.get_case(st.session_state).wall_seed)


@st.cache_data(show_spinner=False)
def wall_probs(variant: str, int8: bool, seed: int) -> np.ndarray:
    return seg.predict(runtime.unet(variant, int8), scene(seed)[0])


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
    st.session_state.l4_cls = clf.classify(runtime.yolo("yolo26n-cls.pt"), wall()[0])
    return "YOLO26n-cls on the wall", runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"]


def show_classifier() -> None:
    top = st.session_state.l4_cls["top"]
    c1, c2 = st.columns([1, 1.6])
    c1.image(rgb(wall()[0]), width="stretch")
    c2.markdown("**GhostLens says:**  \n" + "  \n".join(f"`{label}` {p:.0%}" for label, p in top[:3])
                + "\n\nNo stain class in ImageNet, so it picked whatever it looks most like. Even with a 'stain' "
                  "class, one label for the whole image says nothing about which pixels to clean.")


def run_detector() -> tuple[str, float]:
    st.session_state.l4_dets = det.detect(runtime.yolo("yolo26s.pt"), wall()[0], min_conf=0.25)[0]
    return "YOLO26s on the wall", runtime.profile("detectors", "yolo26s.pt")["latency_ms"]


def show_detector() -> None:
    img, truth = wall()
    dets = st.session_state.l4_dets
    c1, c2 = st.columns([1, 1.6])
    c1.image(rgb(with_box(img, truth)), width="stretch")
    c2.markdown(
        f"**The detector found {len(dets)} COCO object(s)** and no stain, because 'stain' isn't one of its 80 classes. "
        "Suppose you trained one that knew stains. The best it could give you is a box like this one.\n\n"
        f"Purifying that box would scrub **{seg.clean_share_of_box(truth):.0%} healthy wall** along with the "
        "corruption, and destroy the evidence underneath.")


def mode_options() -> dict:
    return {
        "Classify": {
            "blurb": "One label for the whole image. Cheapest.",
            "run": run_classifier, "show": show_classifier,
            "tier": "Light", "latency_ms": runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"],
            "verdict": "Classification can say 'there's a stain'. Purification needs to know which pixels.",
        },
        "Detect": {
            "blurb": "A box around each object.",
            "run": run_detector, "show": show_detector,
            "tier": "Balanced", "latency_ms": runtime.profile("detectors", "yolo26s.pt")["latency_ms"],
            "verdict": "A box is a rough location. The stain is irregular, so most of its box is healthy wall. "
                       "You need a pixel-level boundary.",
        },
        "Segment": {
            "blurb": "A label for every pixel: stain or wall. Most expensive.",
            "verdict": "Exactly which pixels are corrupted is a segmentation question. It's the most expensive "
                       "task, so now the model has to fit the device.",
        },
    }


def warm() -> None:
    wall()
    runtime.unet("standard", False)


WARMUP = [("Reading the parlour wall", wall), ("Measuring segmentation models", runtime.benchmark),
          ("Loading U-Net", warm)]


def profiles(int8: bool) -> list[dict]:
    precision = "INT8" if int8 else "FP32"
    tiers = {"lite": "Light", "standard": "Balanced", "pro": "Heavy"}
    return [{**r, "tier": tiers[r["variant"]], "name": f"{r['name']} {precision}",
             "accuracy": f"{r['iou']:.3f} IoU", "accuracy_tag": "measured"}
            for r in runtime.benchmark()["unets"] if r["precision"] == precision]


def par() -> int:
    return energy_units(runtime.profile("unets", "standard-int8")["latency_ms"])


def report_checks(profile: dict) -> dict:
    s = st.session_state
    used = device.used_in_level(s, 4)
    misses = flow.mode_misses(4)
    tried = s.get("l4_purify_tries", 1)
    return {
        "Right task": (misses == 0, "segmentation first time" if misses == 0
                       else f"tried {', '.join(m for m in s.l4_tried if m != 'Segment')} first"),
        "Met every limit first time": (tried == 1, "first purification passed" if tried == 1
                                       else f"{tried - 1} failed purification attempt(s)"),
        "Latency target": (profile["latency_ms"] <= LIMITS["latency_ms"], f"≤ {LIMITS['latency_ms']} ms"),
        "Battery": (efficient(used, par()), f"{pct(used)} used, {pct(par())} would have done it"),
    }


def purify_problems(model: dict, iou: float, threshold: float) -> list[str]:
    """Everything that stops the purification, one sentence each. Empty means it can go ahead."""
    problems = []
    if model["iou"] < LIMITS["iou"]:
        problems.append(f"{model['name']} only reaches {model['iou']:.3f} IoU on average over 150 test walls. Its "
                        "masks leak onto healthy wall and miss thin tendrils, so it can't be trusted on the next "
                        "wall either.")
    if iou < LIMITS["iou"]:
        hint = "" if problems else " Try a mask threshold nearer 0.5."
        problems.append(f"On this wall the mask at threshold {threshold:.2f} reaches {iou:.3f} IoU, below "
                        f"{LIMITS['iou']}. The purification would scrape the wrong pixels.{hint}")
    if model["latency_ms"] > LIMITS["latency_ms"]:
        problems.append(f"At {model['latency_ms']:.0f} ms per frame it can't keep up with the stain while it "
                        "spreads. By the time the mask is ready, it's out of date.")
    return problems


# side scan: the deployed model on a wall it has never seen, threshold untouched

def second_wall(deployed: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(image, predicted mask, IoU) of the deployed model on the second wall."""
    seed = case.get_case(st.session_state).wall2_seed
    img, truth = scene(seed)
    mask = wall_probs(deployed["variant"], deployed["int8"], seed) > SIDE_THRESHOLD
    return img, mask, seg.mask_iou(mask, truth)


def show_wall2() -> None:
    deployed = st.session_state.l4_deployed
    seed = case.get_case(st.session_state).wall2_seed
    img, mask, iou = second_wall(deployed)
    c1, c2, c3 = st.columns([1, 1, 1.3])
    with c1:
        st.image(rgb(img), width="stretch")
        ui.caption(f"WALL #{seed} · AS FOUND")
    with c2:
        st.image(rgb(with_mask(img, mask)), width="stretch")
        ui.caption(f"WALL #{seed} · {deployed['name'].upper()}")
    c3.markdown(f"**IoU {iou:.3f}** at threshold {SIDE_THRESHOLD}, against {deployed['iou']:.3f} on the first "
                "wall. The model never trained on this wall and you didn't tune anything on it, so this number is "
                "the honest one. A score from the image you tuned on is always a little optimistic.")


def second_wall_scan() -> flow.SideScan | None:
    s = st.session_state
    deployed = s.get("l4_deployed")
    if not deployed:
        return None
    seed = case.get_case(s).wall2_seed
    iou = second_wall(deployed)[2]
    return flow.SideScan(
        key="wall2", title="Second wall",
        blurb=f"There's a second stain on the far wall, #{seed}. Point the deployed {deployed['name']} at it, at "
              f"the default threshold of {SIDE_THRESHOLD}.",
        what=f"{deployed['name']} on wall #{seed}", latency_ms=runtime.profile("unets", deployed["id"])["latency_ms"],
        reward_xp=25, clue=f"Second wall #{seed}: IoU {iou:.3f}", run=lambda: True, show=show_wall2,
        tip_topic="Train, validate, test")


def purify(model: dict, iou: float, threshold: float) -> None:
    s = st.session_state
    s.l4_purify_tries = s.get("l4_purify_tries", 0) + 1
    problems = purify_problems(model, iou, threshold)
    if problems:
        ui.message("<b>Purification failed.</b> " + " ".join(problems)
                   + " Try a different model, or quantize this one.", "bad")
        return
    s.l4_deployed = {"id": model["id"], "name": model["name"], "variant": model["variant"],
                     "int8": model["precision"] == "INT8", "iou": iou}
    finish_level(4, iou, s.l4_purify_tries, report_checks(model), clue="Corruption purified")


def solved_view(model: dict, iou: float) -> None:
    lite, pro = runtime.profile("unets", "lite-fp32"), runtime.profile("unets", "pro-fp32")
    st.divider()
    completion_panel(4, [("Model deployed", model["name"]), ("Mask IoU", f"{iou:.3f}")], par=par())
    scan = second_wall_scan()
    if scan:
        flow.side_scan(4, scan)
    ui.lesson([
        "Segmentation answers <b>exactly which pixels?</b> A box around an irregular shape is mostly background.",
        "It's the most expensive of the three tasks: a prediction for every pixel.",
        f"The light model was fast ({lite['latency_ms']:.0f} ms) but its masks leaked ({lite['iou']:.3f} IoU). "
        f"The heavy one was accurate ({pro['iou']:.3f}) but took {pro['latency_ms']:.0f} ms a frame.",
        int8_note(),
        "On a device, the best model is the one that meets every limit, not the most accurate one.",
    ], title="FIELD NOTES")
    st.divider()
    case_closed()


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
        img, truth = wall()
    except ModelMissing as e:
        st.warning(str(e))
        return
    s = st.session_state
    seed = case.get_case(s).wall_seed
    solved = 4 in s.completed_levels

    if not solved and not flow.mode_choice(
            4, "You need the exact shape of the stain, so only the corrupted pixels get purified.",
            mode_options(), "Segment"):
        st.image(rgb(img), width=380)
        ui.caption(f"EVIDENCE 05-W · WALL #{seed}")
        return

    st.markdown(f'<div class="gl-mission"><span class="gl-kicker">Mission</span><span>purify only the stain</span>'
                f'<span>mask IoU ≥ {LIMITS["iou"]}</span><span>≤ {LIMITS["latency_ms"]} ms per frame</span></div>',
                unsafe_allow_html=True)
    s.setdefault("l4_int8", str(s.get("l4_model", "")).endswith("-int8"))   # match the loaded model on a revisit
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
        if flow.run_button(4, f"Run {model['name']} on the wall", f"{model['name']} on wall #{seed}",
                           model["latency_ms"], model["tier"], key=f"l4_run_{model['id']}"):
            ran.append(model["id"])
            st.rerun()
        return

    probs = wall_probs(model["variant"], model["precision"] == "INT8", seed)
    threshold = st.slider("Mask threshold", 0.1, 0.9, 0.5, 0.05, key="l4_thr",
                          help="A pixel counts as stain if the model's probability is above this.")
    mask = probs > threshold
    iou = seg.mask_iou(mask, truth)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(rgb(img), width="stretch")
        ui.caption(f"WALL #{seed} · ORIGINAL")
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
    st.caption(f"{ui.tag('measured')} We painted the stain onto the wall ourselves, so the true mask is known exactly. "
               "Purification needs both IoU readouts at 0.95 or more.", unsafe_allow_html=True)

    if solved:
        solved_view(model, iou)
    elif st.button("Purify the marked pixels", type="primary"):
        purify(model, iou, threshold)
