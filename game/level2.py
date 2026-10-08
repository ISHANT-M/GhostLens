"""Chapter 2: Identify the Entity (classification)."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv.edge import CLASSIFIERS, energy_units
from cv.models import ModelMissing
from game import device, flow, runtime
from game.levels import completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level2"
OBJECTS = {
    "padlock": ("padlock.jpg", "EVIDENCE 217-A · DESK"),
    "pocket watch": ("pocket_watch.jpg", "EVIDENCE 217-B · DESK DRAWER"),
    "teddy bear": ("teddy_bear.jpg", "EVIDENCE 217-C · ARMCHAIR"),
}
ANCHOR = "pocket watch"
ROOM_PHOTO = "living_room.jpg"
LIMITS = {"latency_ms": 30}
QUIZ = {
    "The room contains a window shade, and probably nothing else.":
        "No. The model always spreads 100% over its 1000 classes for the whole picture. "
        "A low top score usually means several things are competing, not that the room is empty.",
    "The window shade is in the middle of the photo, so that's where to look.":
        "Classification doesn't produce a location at all. There's no 'middle' in a list of 1000 numbers.",
    "The photo as a whole looks most like 'window shade'. It doesn't say what else is there, how many, or where.":
        None,
}


@st.cache_data(show_spinner=False)
def card_photo(filename: str) -> np.ndarray:
    """Center crop to 4:3 at 480x360, so the three evidence cards line up. The model still sees the full photo."""
    img = cv2.imread(str(ASSETS / filename))
    h, w = img.shape[:2]
    ch, cw = min(h, w * 3 // 4), min(w, h * 4 // 3)
    y, x = (h - ch) // 2, (w - cw) // 2
    img = cv2.resize(img[y:y + ch, x:x + cw], (480, 360), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def scan(filename: str, weights: str) -> dict:
    return clf.classify(runtime.yolo(weights), str(ASSETS / filename))


def wrong_mode(weights: str, group: str, label: str):
    def run() -> tuple[str, float]:
        dets, _ = det.detect(runtime.yolo(weights), str(ASSETS / "padlock.jpg"), min_conf=0.25)
        st.session_state.l2_wrong_dets = dets
        return f"{label} on the padlock", runtime.profile(group, weights)["latency_ms"]
    return run


def show_wrong() -> None:
    dets = st.session_state.get("l2_wrong_dets", [])
    c1, c2 = st.columns([1, 2])
    c1.image(str(ASSETS / "padlock.jpg"), width="stretch")
    c2.markdown(f"**{len(dets)} object(s) found.** " + (
        ", ".join(f"{d['label']} {d['conf']:.0%}" for d in dets) if dets else
        "COCO, the dataset these models learned from, has 80 classes and padlock isn't one of them. "
        "A classifier trained on ImageNet's 1000 classes knows padlocks."))


MODE_OPTIONS = {
    "Classify": {
        "blurb": "One label for the whole image. The cheapest model run.",
        "verdict": "Each photo shows one object, so 'what is it?' is the whole question. "
                   "Classification answers it for the least compute.",
    },
    "Detect": {
        "blurb": "Find every object, with a box and a label for each.",
        "run": wrong_mode("yolo26s.pt", "detectors", "YOLO26s detection"), "show": show_wrong,
        "verdict": "You paid for boxes you didn't need, and the detector doesn't even know what a padlock is. "
                   "With one object per photo there's nothing to locate. Use the cheapest task that answers the question.",
    },
    "Segment": {
        "blurb": "Label every single pixel and outline each object.",
        "run": wrong_mode("yolo26s-seg.pt", "segmenters", "YOLO26s-seg"), "show": show_wrong,
        "verdict": "Segmentation predicts something for every pixel, which is the most expensive way to ask "
                   "'what is this?'. It cost the most battery and still couldn't name a padlock.",
    },
}


def warm() -> None:
    runtime.benchmark()
    runtime.yolo("yolo26n-cls.pt")


WARMUP = [("Measuring classifiers", runtime.benchmark), ("Loading the light classifier", warm)]


def profiles() -> list[dict]:
    return [{**r, "accuracy": r["published"].replace(" top-1", ""), "accuracy_tag": "published"}
            for r in runtime.benchmark()["classifiers"]]


def prob_bars(result: dict) -> None:
    rows = "".join(
        f'<div class="gl-bar"><span class="name">{label}</span>'
        f'<span class="track"><span style="width:{p * 100:.1f}%"></span></span>'
        f'<span class="pct">{p:.1%}</span></div>'
        for label, p in result["top"]
    )
    rows += (f'<div class="gl-bar rest"><span class="name">other {result["num_classes"] - 5} classes</span>'
             f'<span class="track"></span><span class="pct">{result["rest"]:.1%}</span></div>')
    st.markdown(rows, unsafe_allow_html=True)


def scan_button(name: str, model: dict, key: str) -> None:
    units = energy_units(model["latency_ms"])
    if st.button(f"Scan  ·  {units} unit{'s' if units != 1 else ''}", key=key, width="stretch"):
        flow.run_cost(2, f"{model['name']} on {name}", model["latency_ms"])
        st.session_state.l2_scanned[name] = model["id"]
        st.rerun()


def scan_objects(model: dict) -> None:
    scanned = st.session_state.l2_scanned
    cols = st.columns(3, gap="medium")
    for col, (name, (filename, label)) in zip(cols, OBJECTS.items()):
        with col:
            st.image(card_photo(filename), width="stretch")
            ui.caption(label)
            if name in scanned:
                prob_bars(scan(filename, scanned[name]))
                st.caption(f"scanned with {runtime.profile('classifiers', scanned[name])['name']}")
            else:
                scan_button(name, model, f"scan_{name}")


def pick_anchor() -> bool:
    scanned = st.session_state.l2_scanned
    if st.session_state.get("l2_anchor_found"):
        top_label, top_p = scan(OBJECTS[ANCHOR][0], scanned[ANCHOR])["top"][0]
        ui.message(
            f"<b>Entity profile: THE TIMEKEEPER.</b> Anchored to the pocket watch. GhostLens's top class was "
            f"<span class='mono'>{top_label}</span> at {top_p:.0%}. Close, but not the real name: ImageNet has "
            "no 'pocket watch' class, so the nearest one it knows wins. A classifier can only ever answer with "
            "one of the labels it was trained on.", "ok")
        return True

    st.markdown("**The manager's note says the thing in this room *keeps hours*. Which object is it anchored to?**")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    choice = c1.radio("Anchor object", list(OBJECTS), horizontal=True, index=None, label_visibility="collapsed")
    if c2.button("Tag as anchor", disabled=choice is None):
        st.session_state.l2_attempts = st.session_state.get("l2_attempts", 0) + 1
        if choice not in scanned:
            ui.message("Scan it first. You're guessing from the photo, not from GhostLens.", "warn")
        elif choice == ANCHOR:
            st.session_state.l2_anchor_found = True
            st.rerun()
        else:
            label, p = scan(OBJECTS[choice][0], scanned[choice])["top"][0]
            ui.message(f"GhostLens says <span class='mono'>{label}</span> ({p:.0%}). "
                       "Nothing about that keeps hours.", "bad")
    return False


def report_checks(model_id: str) -> dict:
    s = st.session_state
    light = runtime.profile("classifiers", "yolo26n-cls.pt")
    model = runtime.profile("classifiers", model_id)
    used = device.used_in_level(s, 2)
    optimal = 4 * energy_units(light["latency_ms"])
    misses = flow.mode_misses(2)
    return {
        "Cheapest task that works": (misses == 0, "classification first time" if misses == 0
                                     else f"tried {', '.join(m for m in s.l2_tried if m != 'Classify')} first"),
        "Right-sized model": (model_id == light["id"], "the light model named every object correctly"
                              if model_id == light["id"] else f"{model['name']} gave the same answers for more battery"),
        "Latency target": (model["latency_ms"] <= LIMITS["latency_ms"], f"≤ {LIMITS['latency_ms']} ms per scan"),
        "Battery": (efficient(used, optimal), f"{used} units used, {optimal} would have done it"),
    }


def room_scan(model: dict) -> None:
    st.subheader("Where did it go?", anchor=False)
    st.markdown('<p class="gl-story">At 03:58 the watch was gone from the drawer. A guest downstairs sent this '
                "photo of their sitting room: \"something moved in here\". You point GhostLens at it, still in "
                "Classify mode.</p>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.image(str(ASSETS / ROOM_PHOTO), width="stretch")
        ui.caption("EVIDENCE 112-A · GUEST PHOTO · 03:58")
    with right:
        if "room" not in st.session_state.l2_scanned:
            scan_button("room", model, "scan_room")
            return
        prob_bars(scan(ROOM_PHOTO, st.session_state.l2_scanned["room"]))
        st.caption("That's the whole answer. One list of scores for the entire photo.")

    if 2 in st.session_state.completed_levels:
        return
    answer = st.radio("What does this result actually tell you?", list(QUIZ), index=None)
    if st.button("Log answer", disabled=answer is None, type="primary"):
        st.session_state.l2_attempts = st.session_state.get("l2_attempts", 0) + 1
        if QUIZ[answer]:
            ui.message(QUIZ[answer], "bad")
        else:
            wrong = max(st.session_state.l2_attempts - 2, 0)  # a clean run is one tag + one answer
            finish_level(2, max(0.0, 1 - 0.25 * wrong), wrong + 1, report_checks(model["id"]), clue="Timekeeper")


def render() -> None:
    ui.scene_header(
        "CHAPTER 2 · ROOM 217 · 03:41",
        "Identify the Entity",
        "Room 217 was unlocked when you got there. Three things look out of place: a padlock on the desk, a pocket "
        "watch in the drawer, and a teddy bear in the armchair that is older than the hotel's guest book. "
        "Each one is photographed on its own.",
    )
    try:
        runtime.benchmark()
    except ModelMissing as e:
        st.warning(str(e))
        return
    solved = 2 in st.session_state.completed_levels
    st.session_state.setdefault("l2_scanned", {})

    if not solved and not flow.mode_choice(
            2, "Each photo shows a single object. You need to know what each one is.", MODE_OPTIONS, "Classify"):
        return

    model = flow.model_picker(
        2, profiles(), LIMITS, slot="task",
        note="Accuracy is ImageNet top-1 as published by Ultralytics. Every scan costs the model's energy.")
    if model is None:
        return
    st.divider()
    scan_objects(model)
    st.divider()
    if pick_anchor() or solved:
        st.divider()
        room_scan(model)

    if solved:
        st.divider()
        completion_panel(2, [("Entity", "The Timekeeper"), ("Anchor", "pocket watch")])
        ui.lesson([
            "Classification answers one question: <b>what does this whole image show?</b> It's the cheapest task.",
            "With one object per image, a bigger model mostly buys you higher confidence on the same answer.",
            "The output is a probability for every class the model knows, and they add up to 100%.",
            "It can only answer with labels from its training set (no 'pocket watch' in ImageNet).",
            "It doesn't say <b>where</b> anything is, or <b>how many</b>. For that you need detection.",
        ], title="FIELD NOTES")

    with st.expander("Learn more: logits, softmax and confidence"):
        logits = np.array([3.1, 1.4, 0.6, -0.5])
        probs = np.exp(logits) / np.exp(logits).sum()
        st.markdown(
            "The last layer of a classifier outputs one raw number per class, called a **logit**. Logits can be any "
            "size, positive or negative. **Softmax** turns them into probabilities:"
        )
        st.latex(r"p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}")
        st.markdown(
            "| class | logit | probability |\n|---|---|---|\n"
            + "\n".join(f"| {c} | {z:+.1f} | {p:.1%} |"
                        for c, z, p in zip(["stopwatch", "compass", "barometer", "clock"], logits, probs))
        )
        st.markdown(
            "Bigger gaps between logits give a more confident softmax. The YOLO26-cls models are CNNs trained on "
            "ImageNet (1.28 M photos, 1000 classes). With transfer learning you would replace the last layer and "
            "retrain it on your own classes, as in the course's flower classifier lab. Published top-1: "
            + ", ".join(f"{v['name']} {v['published']}" for v in CLASSIFIERS.values()) + "."
        )
