"""Chapter 2: Identify the Entity (classification, then a sliding-window sweep of the guest's room)."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv.edge import CLASSIFIERS, energy_units
from cv.models import ModelMissing
from game import case, device, flow, levels, runtime
from game.device import pct
from game.levels import finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level2"
OBJECTS = {"padlock": "padlock.jpg", "pocket watch": "pocket_watch.jpg", "teddy bear": "teddy_bear.jpg"}
ON = {"padlock": "on", "pocket watch": "in", "teddy bear": "in"}
ROOM_PHOTO = "living_room.jpg"
ROOM_SIZE = (1280, 854)
LIMITS = {"latency_ms": 30}
SCOUT = "yolo26n.pt"
LIGHT = "yolo26n-cls.pt"
PHOTOS = {**OBJECTS, "room": ROOM_PHOTO}

# the room sweep
WINDOWS = clf.room_windows(*ROOM_SIZE)
WIN_SIZES = ["Whole photo", "Close-up (560×374)"]
PAN = ["left", "centre", "right"]
TILT = ["top", "middle", "bottom"]
NAMES_NEEDED = 3
# three windows that give three different top-1 labels on every classifier (checked in test_ch2)
IDEAL_WINDOWS = ("half-0-2", "half-1-2", "half-2-1")
PAR_SCANS = 6       # 3 evidence photos + 3 windows, all on the light classifier


def article(word: str) -> str:
    return f"an {word}" if word[0] in "aeiou" else f"a {word}"


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


@st.cache_data(show_spinner=False)
def room_image() -> np.ndarray:
    return cv2.imread(str(ASSETS / ROOM_PHOTO))


@st.cache_data(show_spinner=False)
def scan_window(win: str, weights: str) -> dict:
    """Classify one crop of the room photo. Cached on (window, weights), so a sweep is only ever run once."""
    x0, y0, x1, y1 = WINDOWS[win]
    return clf.classify(runtime.yolo(weights), room_image()[y0:y1, x0:x1])


def window_id(size: str, pan: str, tilt: str) -> str:
    return "full" if size == WIN_SIZES[0] else f"half-{PAN.index(pan)}-{TILT.index(tilt)}"


def window_name(win: str) -> str:
    """'Top-middle window', or 'Whole photo'."""
    if win == "full":
        return "Whole photo"
    col, row = (int(v) for v in win.split("-")[1:])
    if (col, row) == (1, 1):
        return "Centre window"
    return f"{TILT[row].capitalize()}-{'middle' if col == 1 else PAN[col]} window"


# mode dial

def wrong_mode(weights: str, group: str, label: str):
    def run() -> tuple[str, float]:
        dets, _ = det.detect(runtime.yolo(weights), str(ASSETS / "padlock.jpg"), min_conf=0.25)
        st.session_state.l2_wrong_dets = dets
        return f"{label} on the padlock", runtime.profile(group, weights)["latency_ms"]
    return run


def show_wrong() -> None:
    dets = st.session_state.get("l2_wrong_dets", [])
    img = cv2.imread(str(ASSETS / "padlock.jpg"))
    ui.evidence(cv2.cvtColor(det.draw(img, dets=dets), cv2.COLOR_BGR2RGB),
                f"EVIDENCE · THE PADLOCK · {len(dets)} OBJECTS FOUND")


def show_strip() -> None:
    c = case.get_case(st.session_state)
    for col, (name, caption) in zip(st.columns(3), evidence(c)):
        with col:
            ui.evidence(card_photo(OBJECTS[name]), caption)


def mode_options() -> dict:
    found = len(st.session_state.get("l2_wrong_dets", []))
    miss = f"Found {found} object{'s' if found != 1 else ''}; COCO has no padlock class."
    return {
        "Classify": {
            "blurb": "One label per image",
            "line": "One photo, one object, one label.",
            "verdict": "Each photo shows one object, so 'what is it?' is the whole question. "
                       "Classification answers it for the least compute.",
        },
        "Detect": {
            "blurb": "Box and label each object",
            "run": wrong_mode("yolo26s.pt", "detectors", "YOLO26s detection"), "show": show_wrong,
            "tier": "Balanced", "latency_ms": runtime.profile("detectors", "yolo26s.pt")["latency_ms"],
            "line": miss,
            "verdict": "You paid for boxes you didn't need, and the detector doesn't even know what a padlock is. "
                       "With one object per photo there's nothing to locate. Use the cheapest task that answers the question.",
        },
        "Segment": {
            "blurb": "Label every pixel",
            "run": wrong_mode("yolo26s-seg.pt", "segmenters", "YOLO26s-seg"), "show": show_wrong,
            "tier": "Balanced", "latency_ms": runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"],
            "line": f"The most expensive way to ask 'what is this?'. {miss}",
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


def changed_items(tops: dict[str, dict[str, str]], light: str = LIGHT) -> set[str]:
    """Photos where any other model's top-1 differs from the light model's."""
    return {item for model, row in tops.items() if model != light
            for item, label in row.items() if label != tops[light][item]}


def top1s(items: list[str]) -> dict[str, dict[str, str]]:
    return {r["id"]: {item: scan(PHOTOS[item], r["id"])["top"][0][0] for item in items}
            for r in runtime.benchmark()["classifiers"]}


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


def par() -> int:
    """Six scans with the light classifier: three objects and three windows of the room."""
    return PAR_SCANS * energy_units(runtime.profile("classifiers", LIGHT)["latency_ms"])


def size_detail(model_id: str) -> str:
    model = runtime.profile("classifiers", model_id)
    same = not changed_items(top1s(list(OBJECTS)))
    if model_id == LIGHT:
        return ("the smallest model, and it named every object" if same
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


def say(text: str, tone: str = "") -> None:
    """The one feedback line under the stage, kept until the next action."""
    st.session_state.l2_line = (text, tone)


# step 1: tag the anchor

def object_card(name: str, caption: str, model: dict, anchor: str) -> None:
    s = st.session_state
    scanned = s.l2_scanned
    ui.evidence(card_photo(OBJECTS[name]), caption)
    if name not in scanned:
        if flow.run_button(2, "Scan", f"{model['name']} on the {name}", model["latency_ms"], model["tier"],
                           key=f"scan_{name}", primary=False):
            scanned[name] = model["id"]
            label, p = scan(OBJECTS[name], model["id"])["top"][0]
            say(f"GhostLens says <b>{label}</b> ({p:.0%}). Tag it if it {case.ANCHORS[anchor]['riddle']}.")
            st.rerun()
        return
    prob_bars(scan(OBJECTS[name], scanned[name]))
    if st.button("Tag", key=f"l2_tag_{name}", width="stretch"):
        tag(name, anchor)


def tag(name: str, anchor: str) -> None:
    s = st.session_state
    s.l2_attempts = s.get("l2_attempts", 0) + 1
    label, p = scan(OBJECTS[name], s.l2_scanned[name])["top"][0]
    if name == anchor:
        s.l2_anchor_found = True
        s.l2_anchor_label = label
        info = case.ANCHORS[anchor]
        say(f"Anchored: <b>{info['entity']}</b>, bound to the {anchor}. The {anchor} is gone; a guest downstairs "
            "sent this photo.", "ok")
    else:
        say(f"GhostLens says {label} ({p:.0%}). {case.ANCHORS[anchor]['miss']}", "bad")
    st.rerun()


def pick_anchor(c: case.Case) -> None:
    anchor = case.ANCHORS[c.anchor]
    ui.objective(f"Find the object that {anchor['riddle']}.")
    scene, scanner = ui.stage("l2")
    with scanner:
        ui.scanner_head("GHOSTLENS MK.II · CLASSIFY", pct(st.session_state.battery))
        model = flow.model_picker(2, profiles(), LIMITS, slot="task")
    with scene:
        if model is None:
            show_strip()
        else:
            for col, (name, caption) in zip(st.columns(3, gap="medium"), evidence(c)):
                with col:
                    object_card(name, caption, model, c.anchor)
    line = st.session_state.get("l2_line")
    if model is None:
        ui.feedback("Load a classifier. Every scan costs its measured latency.")
    elif line:
        ui.feedback(*line)
    else:
        ui.feedback(f"Scan a photo, then tag the object that {anchor['riddle']}.")


# step 2: sweep the guest's room

def room_names(scans: list[dict]) -> list[str]:
    """Distinct top-1 labels, in the order they were found."""
    return list(dict.fromkeys(r["label"] for r in scans))


def dashed_rect(img: np.ndarray, box: tuple, color: tuple, dash: int = 18, thickness: int = 3) -> None:
    x0, y0, x1, y1 = box
    x1, y1 = x1 - 1, y1 - 1
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for t in range(0, length, 2 * dash):
            p = (int(a[0] + (b[0] - a[0]) * t / length), int(a[1] + (b[1] - a[1]) * t / length))
            e = min(t + dash, length)
            q = (int(a[0] + (b[0] - a[0]) * e / length), int(a[1] + (b[1] - a[1]) * e / length))
            cv2.line(img, p, q, color, thickness)
    tick = 40
    for (cx, cy), (dx, dy) in (((x0, y0), (1, 1)), ((x1, y0), (-1, 1)), ((x1, y1), (-1, -1)), ((x0, y1), (1, -1))):
        cv2.line(img, (cx, cy), (cx + dx * tick, cy), color, thickness + 3)
        cv2.line(img, (cx, cy), (cx, cy + dy * tick), color, thickness + 3)


def room_map(scans: list[dict], current: str | None) -> np.ndarray:
    """The room photo as a scan map: scanned windows tinted and labelled, the reticle dashed. Returns RGB."""
    img = room_image().copy()
    brass, ink = (100, 164, 200), (12, 22, 26)
    done = {r["win"]: r for r in scans}
    tint = img.copy()
    for win in done:
        x0, y0, x1, y1 = WINDOWS[win]
        if win != "full":
            cv2.rectangle(tint, (x0, y0), (x1, y1), brass, -1)
    img = cv2.addWeighted(tint, 0.14, img, 0.86, 0)
    for win, r in done.items():
        x0, y0, x1, y1 = WINDOWS[win]
        cv2.rectangle(img, (x0 + 1, y0 + 1), (x1 - 2, y1 - 2), brass, 2)
        text = f"{r['label']} {r['p']:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        ty = y1 - 14 if win == "full" else y0 + th + 14
        cv2.rectangle(img, (x0 + 6, ty - th - 8), (x0 + 18 + tw, ty + 8), brass, -1)
        cv2.putText(img, text, (x0 + 12, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ink, 2, cv2.LINE_AA)
    if current:
        dashed_rect(img, WINDOWS[current], (231, 228, 218))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


ROOM_DEFAULTS = {"l2_win_size": WIN_SIZES[1], "l2_pan": "centre", "l2_tilt": "middle"}


def window_controls() -> str:
    # the setdefault copy pattern from chapter 1: widget values come back after another page ran
    s = st.session_state
    for k, v in s.setdefault("l2_room_view", dict(ROOM_DEFAULTS)).items():
        s.setdefault(k, v)
    size = st.radio("Window", WIN_SIZES, key="l2_win_size")
    whole = size == WIN_SIZES[0]
    pan = st.select_slider("Pan", PAN, key="l2_pan", disabled=whole)
    tilt = st.select_slider("Tilt", TILT, key="l2_tilt", disabled=whole)
    s.l2_room_view = {k: s[k] for k in ROOM_DEFAULTS}
    return window_id(size, pan, tilt)


def room_readouts(scans: list[dict]) -> None:
    w, h = ROOM_SIZE
    named = len(room_names(scans))
    ui.readouts([("Room covered", f"{clf.coverage([WINDOWS[r['win']] for r in scans], w, h):.0%}", ""),
                 ("Windows", str(len(scans)), ""),
                 ("Things named", f"{min(named, NAMES_NEEDED)}/{NAMES_NEEDED}", "ok" if named >= NAMES_NEEDED else "")])


def scan_room_window(c: case.Case, model: dict, win: str) -> None:
    s = st.session_state
    scans = s.l2_room_scans
    if any(r["win"] == win and r["model"] == model["id"] for r in scans):
        st.button("Already scanned", key="l2_scan_window", disabled=True)
        return
    if 2 in s.completed_levels:     # reviewing a cleared chapter: look, don't spend
        st.button("Sweep closed", key="l2_scan_window", disabled=True)
        return
    if not flow.run_button(2, "Scan window", f"{model['name']} on the {window_name(win).lower()}",
                           model["latency_ms"], model["tier"], key="l2_scan_window"):
        return
    label, p = scan_window(win, model["id"])["top"][0]
    scans.append({"win": win, "model": model["id"], "label": label, "p": p})
    extra = ("One label for the entire room." if win == "full"
             else "One label per window, no position inside it.")
    say(f"{window_name(win)}: '{label}' {p:.0%}. {extra}")
    if len(room_names(scans)) >= NAMES_NEEDED and 2 not in s.completed_levels:
        tries = s.get("l2_attempts", 1)
        finish_level(2, min(1.0, NAMES_NEEDED / len(scans)), tries, report_checks(model["id"]),
                     clue=case.ANCHORS[c.anchor]["entity"])
    st.rerun()


def room_sweep(c: case.Case) -> None:
    s = st.session_state
    model = runtime.profile("classifiers", s.l2_model)
    ui.objective("Name three different things in the guest's room.")
    scene, scanner = ui.stage("l2_room")
    with scanner:
        ui.scanner_head(f"GHOSTLENS MK.II · {model['name'].upper()}", pct(s.battery))
        win = window_controls()
        scan_room_window(c, model, win)
        room_readouts(s.l2_room_scans)
    with scene:
        ui.evidence(room_map(s.l2_room_scans, win), "EVIDENCE 112-A · GUEST PHOTO · 03:58 · SCAN MAP")
    if line := s.get("l2_line"):
        ui.feedback(*line)


# cleared screen

def top1_table() -> pd.DataFrame:
    rows = []
    for item, filename in PHOTOS.items():
        row = {"photo": "guest's room" if item == "room" else item}
        for r in runtime.benchmark()["classifiers"]:
            label, p = scan(filename, r["id"])["top"][0]
            row[f"{r['name']} · {r['latency_ms']:.1f} ms"] = f"{label} {p:.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_rows(weights: str) -> list[dict]:
    """Every window of the room with one model: top-1, p and the measured model time."""
    rows = []
    for win in WINDOWS:
        r = scan_window(win, weights)
        label, p = r["top"][0]
        rows.append({"window": window_name(win), "top-1": label, "p": round(p, 2), "ms": round(r["model_ms"], 1)})
    return rows


SOFTMAX = ("The last layer of a classifier outputs one raw number per class, called a **logit**. Logits can be any "
           "size, positive or negative. **Softmax** turns them into probabilities that add up to 1:")


def softmax_maths() -> None:
    logits = np.array([3.1, 1.4, 0.6, -0.5])
    probs = np.exp(logits) / np.exp(logits).sum()
    st.markdown(SOFTMAX)
    st.latex(r"p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}")
    st.markdown("| class | logit | probability |\n|---|---|---|\n"
                + "\n".join(f"| {name} | {z:+.1f} | {p:.1%} |"
                            for name, z, p in zip(["stopwatch", "compass", "barometer", "clock"], logits, probs)))
    st.markdown(
        "Bigger gaps between logits give a more confident softmax. The YOLO26-cls models are CNNs with one "
        "attention block (C2PSA) before the head, trained on ImageNet (1.28 M photos, 1000 classes). With "
        "transfer learning you would keep that backbone, replace the last layer and retrain it on your own "
        "classes. Published top-1: "
        + ", ".join(f"{v['name']} {v['published']}" for v in CLASSIFIERS.values()) + ".")
    st.markdown("A **sliding window** runs the same classifier on crops of a big image. Each crop gets one label. "
                "It tells you roughly what is in each part of the photo, but every crop is a full forward pass, "
                "and it still never draws a box around anything.")


def numbers(model_id: str) -> None:
    st.markdown('<div class="gl-kicker">Three classifiers, four photos</div>', unsafe_allow_html=True)
    st.dataframe(top1_table(), hide_index=True, width="stretch")
    changed = changed_items(top1s(list(PHOTOS)))
    on_objects = sorted(changed - {"room"})
    st.markdown(f"Photos where a bigger model's top-1 differs from YOLO26n-cls: "
                f"**{', '.join(sorted(changed)) or 'none'}**. "
                + ("On every single object they agree, so the bigger models only buy confidence there. "
                   if not on_objects else "")
                + "On a busy room there's no single right label.")
    st.caption(f"{ui.tag('measured')} top-1 labels and latency on this machine", unsafe_allow_html=True)

    model = runtime.profile("classifiers", model_id)
    rows = sweep_rows(model_id)
    st.markdown(f'<div class="gl-kicker">The full sweep with {model["name"]}</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    total = sum(r["ms"] for r in rows)
    st.markdown(f"A full sweep = {len(rows)} passes, {total:.0f} ms, and still no boxes. "
                f"At the benchmark latency that's {pct(len(rows) * energy_units(model['latency_ms']))} of battery.")
    st.caption(f"{ui.tag('measured')} labels, p and model time on this machine "
               f"&nbsp;{ui.tag('gameplay')} battery", unsafe_allow_html=True)
    with st.expander("The maths: logits, softmax, sliding windows"):
        softmax_maths()


def cleared(c: case.Case) -> levels.Cleared:
    s = st.session_state
    info = case.ANCHORS[c.anchor]
    model_id = s.get("l2_model") or LIGHT
    model = runtime.profile("classifiers", model_id)
    scans = s.get("l2_room_scans", [])
    names = room_names(scans)
    tries = s.get("l2_attempts", 1)
    label = s.get("l2_anchor_label") or scan(OBJECTS[c.anchor], model_id)["top"][0][0]
    light, big = runtime.profile("classifiers", LIGHT), runtime.benchmark()["classifiers"][-1]

    happened = flow.wrong_mode_lines(2, mode_options()) + [
        f"You loaded {model['name']} ({model['latency_ms']:.1f} ms a scan).",
        f"You tagged the {c.anchor} in {tries} tr{'ies' if tries != 1 else 'y'}. GhostLens called it "
        f"'{label}'. {info['reveal']}",
        f"You swept {len(scans)} window{'s' if len(scans) != 1 else ''} of the guest's room and named: "
        f"{', '.join(names) or 'nothing'}.",
    ]
    why = [
        "Each evidence photo holds one object, so one label per image answers the question for the least compute.",
        "Softmax spreads 100% over ImageNet's 1000 classes. The top class wins even when the true name isn't one "
        "of them.",
        f"{big['name']} costs {big['latency_ms'] / light['latency_ms']:.1f}× the battery of {light['name']} per scan "
        "(measured). On one clear object it mostly buys confidence.",
        "A sliding window gives one label per crop: roughly what is where, never how many or exactly where.",
    ]
    return levels.Cleared(
        headline=f"Entity: {info['entity']}, anchored to the {c.anchor}.",
        happened=happened, why=why,
        concept="Image classification · softmax · ImageNet labels · sliding window",
        evidence=lambda: ui.evidence(room_map(scans, None), "EVIDENCE 112-A · YOUR SWEEP OF THE GUEST'S ROOM"),
        numbers=lambda: numbers(model_id),
        side=scout_scan(),
        stats=[("Entity", info["entity"]), ("Anchor", c.anchor), ("Classifier said", label)],
        par=par(),
    )


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
        ui.evidence(level3.rgb(det.draw(level3.load_scene(), m)), "EVIDENCE 05-S · FROM THE DOORWAY · YOLO26N")
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
    try:
        runtime.benchmark()
    except ModelMissing as e:
        st.warning(str(e))
        return
    if levels.show_cleared(2):
        levels.level_cleared(2, cleared(c))
        return
    levels.review_banner(2)
    solved = 2 in s.completed_levels
    s.setdefault("l2_scanned", {})
    s.setdefault("l2_room_scans", [])

    if not solved and s.get("l2_mode") != "Classify":
        items = [f"{article(n)} {ON[n]} {case.ANCHORS[n]['place']}" for n in c.evidence_order]
        ui.title_card(
            f"CHAPTER 2 · {c.clue['label'].upper()} · 03:41", "Identify the Entity",
            f"{c.clue['label']} was unlocked when you got there. Three things look out of place: {items[0]}, "
            f"{items[1]} and {items[2]}. Nobody on the staff remembers any of them. Each one is photographed on "
            "its own.")
    if not solved and not flow.mode_choice(
            2, "Each photo shows a single object. Find out what each one is.", mode_options(), "Classify",
            needs="a name for each single object", scene=show_strip):
        return

    if s.get("l2_anchor_found") and s.get("l2_model"):
        room_sweep(c)
    else:
        pick_anchor(c)
