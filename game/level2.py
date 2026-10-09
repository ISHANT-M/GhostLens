"""Chapter 2: Identify the Entity (classification)."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv.edge import CLASSIFIERS, energy_units
from cv.models import ModelMissing
from game import case, device, flow, runtime
from game.device import pct
from game.levels import completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level2"
OBJECTS = {"padlock": "padlock.jpg", "pocket watch": "pocket_watch.jpg", "teddy bear": "teddy_bear.jpg"}
ON = {"padlock": "on", "pocket watch": "in", "teddy bear": "in"}
ROOM_PHOTO = "living_room.jpg"
LIMITS = {"latency_ms": 30}
SCOUT = "yolo26n.pt"
LIGHT = "yolo26n-cls.pt"
PHOTOS = {**OBJECTS, "room": ROOM_PHOTO}
BIGGER_OPTIONS = ["No, every top-1 stays the same", "Yes, on at least one single-object photo",
                  "Only on the guest's room photo"]


def article(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


def quiz(label: str) -> dict[str, str | None]:
    """Three readings of the room scan, built from the label the model really gave. The last one is right."""
    return {
        f"The room contains {article(label)}, and probably nothing else.":
            "No. The model always spreads 100% over its 1000 classes for the whole picture. "
            "A low top score usually means several things are competing, not that the room is empty.",
        f"The {label} is in the middle of the photo, so that's where to look.":
            "Classification doesn't produce a location at all. There's no 'middle' in a list of 1000 numbers.",
        f"The photo as a whole looks most like '{label}'. It doesn't say what else is there, how many, or where.":
            None,
    }


def evidence(c: case.Case) -> list[tuple[str, str]]:
    """(object, caption) in this case's card order."""
    return [(name, f"EVIDENCE {c.number}-{'ABC'[i]} · {case.ANCHORS[name]['place'].removeprefix('the ').upper()}")
            for i, name in enumerate(c.evidence_order)]


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


def mode_options() -> dict:
    return {
        "Classify": {
            "blurb": "One label for the whole image. The cheapest model run.",
            "verdict": "Each photo shows one object, so 'what is it?' is the whole question. "
                       "Classification answers it for the least compute.",
        },
        "Detect": {
            "blurb": "Find every object, with a box and a label for each.",
            "run": wrong_mode("yolo26s.pt", "detectors", "YOLO26s detection"), "show": show_wrong,
            "tier": "Balanced", "latency_ms": runtime.profile("detectors", "yolo26s.pt")["latency_ms"],
            "verdict": "You paid for boxes you didn't need, and the detector doesn't even know what a padlock is. "
                       "With one object per photo there's nothing to locate. Use the cheapest task that answers the question.",
        },
        "Segment": {
            "blurb": "Label every single pixel and outline each object.",
            "run": wrong_mode("yolo26s-seg.pt", "segmenters", "YOLO26s-seg"), "show": show_wrong,
            "tier": "Balanced", "latency_ms": runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"],
            "verdict": "Segmentation predicts something for every pixel, which is the most expensive way to ask "
                       "'what is this?'. It cost the most battery and still couldn't name a padlock.",
        },
    }


def warm() -> None:
    runtime.benchmark()
    runtime.yolo("yolo26n-cls.pt")


WARMUP = [("Measuring the models", runtime.benchmark), ("Warming up GhostLens", warm)]


def profiles() -> list[dict]:
    return [{**r, "accuracy": r["published"].replace(" top-1", ""), "accuracy_tag": "published",
             "metric": "ImageNet top-1"} for r in runtime.benchmark()["classifiers"]]


# does a bigger classifier change the answer?

def changed_items(tops: dict[str, dict[str, str]], light: str = LIGHT) -> set[str]:
    """Photos where any other model's top-1 differs from the light model's."""
    return {item for model, row in tops.items() if model != light
            for item, label in row.items() if label != tops[light][item]}


def bigger_model_answer(changed: set[str]) -> int:
    """Index into BIGGER_OPTIONS."""
    if not changed:
        return 0
    return 1 if changed - {"room"} else 2


def top1s(items: list[str]) -> dict[str, dict[str, str]]:
    return {r["id"]: {item: scan(PHOTOS[item], r["id"])["top"][0][0] for item in items}
            for r in runtime.benchmark()["classifiers"]}


def resolve_bigger() -> tuple[int, str]:
    changed = changed_items(top1s(list(PHOTOS)))
    where = ", ".join(sorted(changed)) or "none"
    return bigger_model_answer(changed), (
        f"Photos where a bigger model's top-1 differs from YOLO26n-cls: {where}. On one clear object the bigger "
        "models mostly buy confidence. On a busy room there's no single right label, so they disagree.")


def top1_table() -> pd.DataFrame:
    rows = []
    for item, filename in PHOTOS.items():
        row = {"photo": "guest's room" if item == "room" else item}
        for r in runtime.benchmark()["classifiers"]:
            label, p = scan(filename, r["id"])["top"][0]
            row[f"{r['name']} · {r['latency_ms']:.1f} ms"] = f"{label} {p:.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def debrief() -> None:
    record = flow.predict("l2_pred_bigger", "Run all three classifiers on all four photos. Would a bigger model "
                          "change any top-1 answer?", BIGGER_OPTIONS, resolve=resolve_bigger,
                          reveal=lambda: st.dataframe(top1_table(), hide_index=True, width="stretch"))
    if record is None:
        return
    st.caption(f"{ui.tag('measured')} top-1 labels and latency on this machine", unsafe_allow_html=True)
    light, big = runtime.profile("classifiers", LIGHT), runtime.benchmark()["classifiers"][-1]
    ui.lesson([
        "Classification answers one question, <b>what does this whole image show?</b>, with a probability for "
        "every class the model knows.",
        f"{big['name']} costs {big['latency_ms'] / light['latency_ms']:.1f}× the battery of {light['name']} per "
        "scan. On one clear object it mostly buys confidence.",
        "It can't say <b>where</b> anything is, or <b>how many</b>, and it only knows its training labels.",
    ], title="DEBRIEF")


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


def scan_button(name: str, what: str, model: dict, key: str) -> None:
    if flow.run_button(2, "Scan", f"{model['name']} on {what}", model["latency_ms"], model["tier"], key=key,
                       primary=False):
        st.session_state.l2_scanned[name] = model["id"]
        st.rerun()


def scan_objects(c: case.Case, model: dict) -> None:
    scanned = st.session_state.l2_scanned
    cols = st.columns(3, gap="medium")
    for col, (name, caption) in zip(cols, evidence(c)):
        with col:
            st.image(card_photo(OBJECTS[name]), width="stretch")
            ui.caption(caption)
            if name in scanned:
                prob_bars(scan(OBJECTS[name], scanned[name]))
                st.caption(f"scanned with {runtime.profile('classifiers', scanned[name])['name']}")
            else:
                scan_button(name, f"the {name}", model, f"scan_{name}")


def pick_anchor(c: case.Case) -> bool:
    s = st.session_state
    scanned = s.l2_scanned
    anchor = case.ANCHORS[c.anchor]
    if s.get("l2_anchor_found"):
        top_label, top_p = scan(OBJECTS[c.anchor], scanned[c.anchor])["top"][0]
        ui.message(
            f"<b>Entity profile: {anchor['entity'].upper()}.</b> Anchored to the {c.anchor}. GhostLens's top class "
            f"was <span class='mono'>{top_label}</span> at {top_p:.0%}. {anchor['reveal']} A classifier can only "
            "ever answer with one of the labels it was trained on.", "ok")
        return True

    st.markdown(f'<div class="gl-choice"><div class="gl-kicker">The manager\'s note</div>The thing in this room '
                f'<i>{anchor["riddle"]}</i>. Which object is it anchored to?</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    choice = c1.radio("Anchor object", list(c.evidence_order), horizontal=True, index=None, key="l2_anchor_choice",
                      label_visibility="collapsed")
    if c2.button("Tag as anchor", disabled=choice is None):
        s.l2_attempts = s.get("l2_attempts", 0) + 1
        if choice not in scanned:
            ui.message("Scan it first. You're guessing from the photo, not from GhostLens.", "warn")
        elif choice == c.anchor:
            s.l2_anchor_found = True
            s.l2_anchor_label = scan(OBJECTS[choice], scanned[choice])["top"][0][0]
            st.rerun()
        else:
            label, p = scan(OBJECTS[choice], scanned[choice])["top"][0]
            ui.message(f"GhostLens says <span class='mono'>{label}</span> ({p:.0%}). {anchor['miss']}", "bad")
    return False


def par() -> int:
    """Four scans with the light classifier: three objects and the guest photo."""
    return 4 * energy_units(runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"])


def size_detail(model_id: str) -> str:
    model = runtime.profile("classifiers", model_id)
    same = not changed_items(top1s(list(OBJECTS)))
    if model_id == LIGHT:
        return ("same top-1 as the bigger models (measured)" if same
                else "the light model, though a bigger one disagrees on an object (measured)")
    return (f"{model['name']} gave the same top-1 as the light model, for more battery (measured)" if same
            else f"{model['name']} for more battery")


def report_checks(model_id: str) -> dict:
    s = st.session_state
    used = device.used_in_level(s, 2)
    misses = flow.mode_misses(2)
    return {
        "Cheapest task that works": (misses == 0, "classification first time" if misses == 0
                                     else f"tried {', '.join(m for m in s.l2_tried if m != 'Classify')} first"),
        "Right-sized model": (model_id == LIGHT, size_detail(model_id)),
        "Battery": (efficient(used, par()), f"{pct(used)} used, {pct(par())} would have done it"),
    }


def room_scan(c: case.Case, model: dict) -> None:
    s = st.session_state
    place = case.ANCHORS[c.anchor]["place"]
    st.subheader("Where did it go?", anchor=False)
    st.markdown(f'<p class="gl-story">At 03:58 the {c.anchor} was gone from {place}. A guest downstairs sent this '
                'photo of their sitting room: "something moved in here". You point GhostLens at it, still in '
                "Classify mode.</p>", unsafe_allow_html=True)
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.image(str(ASSETS / ROOM_PHOTO), width="stretch")
        ui.caption("EVIDENCE 112-A · GUEST PHOTO · 03:58")
    with right:
        if "room" not in s.l2_scanned:
            scan_button("room", "the guest photo", model, "scan_room")
            return
        result = scan(ROOM_PHOTO, s.l2_scanned["room"])
        prob_bars(result)
        st.caption("That's the whole answer. One list of scores for the entire photo.")

    if 2 in s.completed_levels:
        return
    options = quiz(result["top"][0][0])
    answer = st.radio("What does this result actually tell you?", list(options), index=None, key="l2_quiz")
    if st.button("Log answer", disabled=answer is None, type="primary"):
        s.l2_attempts = s.get("l2_attempts", 0) + 1
        if options[answer]:
            ui.message(options[answer], "bad")
        else:
            wrong = max(s.l2_attempts - 2, 0)  # a clean run is one tag + one answer
            finish_level(2, max(0.0, 1 - 0.25 * wrong), wrong + 1, report_checks(model["id"]),
                         clue=case.ANCHORS[c.anchor]["entity"])


# side scan: look into the parlour from the doorway with the light detector

def scout_matches() -> dict:
    from game import level3
    return det.evaluate(level3.scene_detections(SCOUT), level3.TRUTH["objects"], level3.SIDE_CONF)[0]


def scout_text(m: dict) -> str:
    labels = [d["label"] for d in m["tp"]]
    cups = labels.count("cup")
    return (f"YOLO26n found {labels.count('chair')} chairs and {labels.count('dining table')} table, "
            f"{'none' if cups == 0 else cups} of the 4 cups")


def show_scout() -> None:
    from game import level3
    m = scout_matches()
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.image(level3.rgb(det.draw(level3.load_scene(), m)), width="stretch")
        ui.caption("EVIDENCE 05-S · FROM THE DOORWAY · YOLO26N")
    c2.markdown(f"**{scout_text(m)}**, keeping boxes above {level3.SIDE_CONF:.2f}. From across the room the cups "
                "are only a few dozen pixels wide. That's worth knowing before you choose a detector for the "
                "inventory.")


def scout_scan() -> flow.SideScan:
    text = scout_text(scout_matches())
    return flow.SideScan(
        key="scout", title="Scout the parlour",
        blurb="The lift doors open onto the parlour. Before you go in, run the light detector from the doorway "
              "and see what it can pick out.",
        what="YOLO26n on the parlour (scout)", latency_ms=runtime.profile("detectors", SCOUT)["latency_ms"],
        reward_xp=20, clue=f"Parlour scout: {text}", run=lambda: True, show=show_scout,
        intel=("l3_light", f"Doorway scout: {text}."), tip_topic="mAP")


def render() -> None:
    s = st.session_state
    c = case.get_case(s)
    items = [f"{article(n)} {ON[n]} {case.ANCHORS[n]['place']}" for n in c.evidence_order]
    ui.scene_header(
        f"CHAPTER 2 · {c.clue['label'].upper()} · 03:41",
        "Identify the Entity",
        f"{c.clue['label']} was unlocked when you got there. Three things look out of place: {items[0]}, "
        f"{items[1]} and {items[2]}. Nobody on the staff remembers any of them. Each one is photographed on its own.",
    )
    try:
        runtime.benchmark()
    except ModelMissing as e:
        st.warning(str(e))
        return
    solved = 2 in s.completed_levels
    s.setdefault("l2_scanned", {})

    if not solved and not flow.mode_choice(
            2, "Each photo shows a single object. You need to know what each one is.", mode_options(), "Classify",
            needs="a name for each single object"):
        return

    model = flow.model_picker(
        2, profiles(), LIMITS, slot="task",
        note="ImageNet top-1 as published by Ultralytics. Every scan costs the model's measured latency. All three "
             f"are under the {LIMITS['latency_ms']} ms target, so here it's for information only.")
    if model is None:
        return
    st.divider()
    scan_objects(c, model)
    st.divider()
    if pick_anchor(c) or solved:
        st.divider()
        room_scan(c, model)

    if solved:
        st.divider()
        completion_panel(2, [("Entity", case.ANCHORS[c.anchor]["entity"]), ("Anchor", c.anchor)], par=par(),
                         debrief=debrief)
        flow.side_scan(2, scout_scan())

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
            "Bigger gaps between logits give a more confident softmax. The YOLO26-cls models are CNNs with one "
            "attention block (C2PSA) before the head, trained on ImageNet (1.28 M photos, 1000 classes). With "
            "transfer learning you would keep that backbone, replace the last layer and retrain it on your own "
            "classes. Published top-1: "
            + ", ".join(f"{v['name']} {v['published']}" for v in CLASSIFIERS.values()) + "."
        )
