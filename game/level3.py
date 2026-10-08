"""Chapter 3: Find the Anomalies (detection, threshold, precision and recall)."""

import json
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv.edge import energy_units
from cv.models import ModelMissing
from game import device, flow, runtime
from game.levels import completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level3"
TRUTH = json.loads((ASSETS / "ground_truth.json").read_text())
PASS_F1 = 0.75
LIMITS = {"latency_ms": 60}


@st.cache_data
def load_scene() -> np.ndarray:
    return cv2.imread(str(ASSETS / TRUTH["image"]))


@st.cache_data(show_spinner=False)
def scene_detections(weights: str) -> list[dict]:
    return det.detect(runtime.yolo(weights), load_scene())[0]


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def names(items: list[dict]) -> str:
    counts = pd.Series([i["label"] for i in items]).value_counts()
    return ", ".join(f"{n} × {label}" if n > 1 else label for label, n in counts.items())


def run_classifier() -> tuple[str, float]:
    st.session_state.l3_cls = clf.classify(runtime.yolo("yolo26n-cls.pt"), load_scene())
    return "YOLO26n-cls on the parlour", runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"]


def show_classifier() -> None:
    top = st.session_state.l3_cls["top"]
    c1, c2 = st.columns([1.2, 1])
    c1.image(rgb(load_scene()), width="stretch")
    c2.markdown("**GhostLens says:**  \n" + "  \n".join(f"`{label}` {p:.0%}" for label, p in top[:3])
                + "\n\nOne label for the whole room. How many cups? Which chair? Where? No idea.")


def run_segmenter() -> tuple[str, float]:
    r = runtime.yolo("yolo26s-seg.pt").predict(load_scene(), device="cpu", verbose=False)[0]
    st.session_state.l3_seg = (r.plot(boxes=False)[..., ::-1], [r.names[int(c)] for c in r.boxes.cls])
    return "YOLO26s-seg on the parlour", runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"]


def show_segmenter() -> None:
    img, labels = st.session_state.l3_seg
    c1, c2 = st.columns([1.2, 1])
    c1.image(img, width="stretch")
    seg_ms = runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"]
    det_ms = runtime.profile("detectors", "yolo26s.pt")["latency_ms"]
    c2.markdown(f"**{len(labels)} outlines:** {', '.join(labels)}.\n\n"
                f"It sort of worked. But you only needed a list and positions, and pixel outlines cost "
                f"{energy_units(seg_ms)} units where boxes from a detector of the same size cost {energy_units(det_ms)}.")


MODE_OPTIONS = {
    "Classify": {
        "blurb": "One label for the whole image. Cheapest.",
        "run": run_classifier, "show": show_classifier,
        "verdict": "Classification tells you what the room looks like overall. It can't count objects or say "
                   "where they are, and that's exactly what the report needs.",
    },
    "Detect": {
        "blurb": "A box, a label and a confidence for every object.",
        "verdict": "Several objects, and you need what and where for each. That's detection, and it's worth the "
                   "extra compute here.",
    },
    "Segment": {
        "blurb": "An exact outline for every object, pixel by pixel. Most expensive.",
        "run": run_segmenter, "show": show_segmenter,
        "verdict": "Segmentation is the most expensive way to make a list of objects. Boxes answer the question; "
                   "outlines are detail you're paying for and not using.",
    },
}


def warm() -> None:
    runtime.yolo("yolo26s.pt")
    scene_detections("yolo26s.pt")


WARMUP = [("Measuring detectors", runtime.benchmark), ("Loading the detector", warm)]


def profiles() -> list[dict]:
    return [{**r, "accuracy": r["published"], "accuracy_tag": "published"} for r in runtime.benchmark()["detectors"]]


def consequence(m: dict, s: dict) -> None:
    fp, missed = m["fp"], m["missed"]
    if s["f1"] >= PASS_F1:
        ui.message(f"<b>Report accepted.</b> {s['tp']} of {s['tp'] + s['fn']} objects found, "
                   f"{s['fp']} false alarm{'s' if s['fp'] != 1 else ''}. You picked a threshold where both kinds of "
                   "mistake stay small.", "ok")
    elif s["recall"] == 1.0:
        ui.message(f"<b>You found everything.</b> You also accused {len(fp)} innocent objects ({names(fp)}). "
                   "The night manager is not going to search the parlour for a cup that's actually a coffee pot. "
                   "Low threshold: high recall, low precision.", "bad")
    elif s["precision"] == 1.0:
        ui.message(f"<b>Everything you flagged is real.</b> But {len(missed)} objects slipped past ({names(missed)}). "
                   "One of them could be what the entity moved. High threshold: high precision, low recall.", "bad")
    else:
        ui.message(f"Some of both: {len(fp)} false alarm(s) ({names(fp)}) and {len(missed)} missed "
                   f"({names(missed)}). Move the threshold and try again.", "warn")


def pr_curve(dets: list[dict], threshold: float) -> alt.Chart:
    rows = []
    for t in np.arange(0.05, 0.96, 0.05):
        _, s = det.evaluate(dets, TRUTH["objects"], float(t))
        rows += [{"threshold": t, "metric": k.capitalize(), "value": s[k]} for k in ("precision", "recall")]
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("threshold:Q", title="confidence threshold"),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", scale=alt.Scale(domain=["Precision", "Recall"], range=["#9C4A3C", "#5E7D5A"]),
                        legend=alt.Legend(orient="top", title=None)),
    )
    rule = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color="#22211F", strokeDash=[4, 3]).encode(x="t:Q")
    return (lines + rule).properties(height=200).configure_view(stroke=None).configure(background="transparent")


def report_checks(model_id: str) -> dict:
    s = st.session_state
    model = runtime.profile("detectors", model_id)
    tried = s.get("l3_models_tried", [])
    used = device.used_in_level(s, 3)
    optimal = energy_units(runtime.profile("detectors", "yolo26s.pt")["latency_ms"])
    misses = flow.mode_misses(3)
    return {
        "Right task": (misses == 0, "detection first time" if misses == 0
                       else f"tried {', '.join(m for m in s.l3_tried if m != 'Detect')} first"),
        "Right-sized model": (len(tried) == 1, f"{model['name']} first time" if len(tried) == 1
                              else "needed a second model after the first couldn't find the cups"),
        "Latency target": (model["latency_ms"] <= LIMITS["latency_ms"], f"≤ {LIMITS['latency_ms']} ms"),
        "Battery": (efficient(used, optimal), f"{used} units used, {optimal} would have done it"),
    }


def render() -> None:
    ui.scene_header(
        "CHAPTER 3 · PARLOUR · CAM 05 · 04:12",
        "Find the Anomalies",
        "The lift went back up to the antique parlour at 04:05. The cleaner swears the tea set was rearranged "
        "overnight. The manager wants a list of every object in the room and where it is, to compare with last "
        "week's photos.",
    )
    try:
        runtime.benchmark()
    except ModelMissing as e:
        st.warning(str(e))
        return
    s = st.session_state
    solved = 3 in s.completed_levels
    device.unload(s, "task")  # the classifier from chapter 2 isn't needed any more

    if not solved and not flow.mode_choice(
            3, "Lots of objects in one frame. You need to know what each one is and where it is.",
            MODE_OPTIONS, "Detect"):
        return

    model = flow.model_picker(
        3, profiles(), LIMITS, slot="watchdog",
        note="Accuracy is COCO mAP50-95 as published by Ultralytics. The detector you load stays in memory as a "
             "watchdog for the rest of the case.")
    if model is None:
        return

    ran = s.setdefault("l3_ran", [])
    if model["id"] not in ran:
        units = energy_units(model["latency_ms"])
        if st.button(f"Run {model['name']} on the parlour  ·  {units} units", type="primary"):
            flow.run_cost(3, f"{model['name']} on the parlour", model["latency_ms"])
            ran.append(model["id"])
            st.rerun()
        return

    dets = scene_detections(model["id"])
    best_f1 = max(det.evaluate(dets, TRUTH["objects"], t / 20)[1]["f1"] for t in range(1, 19))
    reported = s.get("l3_reported") == model["id"]

    view, ctrl = st.columns([2.2, 1], gap="large")
    with ctrl:
        st.markdown(f"**Detect mode · {model['name']}**")
        threshold = st.slider("Confidence threshold", 0.05, 0.95, 0.60, 0.05, key="l3_threshold",
                              help="Only boxes the model is at least this sure about are kept. "
                                   "Moving it doesn't re-run the model.")
        shown = [d for d in dets if d["conf"] >= threshold]
        st.markdown(f'<div class="gl-kicker">{len(shown)} boxes above {threshold:.2f}</div>', unsafe_allow_html=True)
        if shown:
            st.dataframe(pd.DataFrame([{"object": d["label"], "confidence": round(d["conf"], 2)} for d in shown]),
                         hide_index=True, height=min(36 * len(shown) + 38, 290), width="stretch")
        if not solved and st.button("Submit report", type="primary", width="stretch"):
            s.l3_reported = model["id"]
            s.l3_attempts = s.get("l3_attempts", 0) + 1
            reported = True
            _, result = det.evaluate(dets, TRUTH["objects"], threshold)
            if result["f1"] >= PASS_F1:
                s.l3_pending_finish = result["f1"]

    m, sc = det.evaluate(dets, TRUTH["objects"], threshold)
    with view:
        if reported or solved:
            st.image(rgb(det.draw(load_scene(), m)), width="stretch")
            ui.caption("EVIDENCE 05-C · GREEN = CORRECT · RED = FALSE ALARM · DASHED AMBER = MISSED")
        else:
            st.image(rgb(det.draw(load_scene(), dets=shown)), width="stretch")
            ui.caption(f"EVIDENCE 05-C · GHOSTLENS DETECT MODE · {model['name'].upper()}")

    if reported or solved:
        ui.readouts([
            ("Found (TP)", str(sc["tp"]), "ok"),
            ("False alarms (FP)", str(sc["fp"]), "bad" if sc["fp"] else ""),
            ("Missed (FN)", str(sc["fn"]), "bad" if sc["fn"] else ""),
            ("Precision", f"{sc['precision']:.0%}", ""),
            ("Recall", f"{sc['recall']:.0%}", ""),
            ("F1", f"{sc['f1']:.2f}", "ok" if sc["f1"] >= PASS_F1 else "warn"),
        ])
        if not solved:
            consequence(m, sc)
            if best_f1 < PASS_F1:
                ui.message(f"<b>{model['name']} can't do this at any threshold.</b> Its best possible F1 on this room "
                           f"is {best_f1:.2f}: the tea cups are too small for it. Load a bigger model above "
                           "(it costs battery to run again).", "bad")
        c1, c2 = st.columns([1.3, 1], gap="large")
        with c1:
            st.altair_chart(pr_curve(dets, threshold), width="stretch")
        with c2:
            st.markdown(
                "**Precision**: of everything you flagged, how much was real?  \n"
                f"{sc['tp']} / ({sc['tp']} + {sc['fp']}) = **{sc['precision']:.0%}**\n\n"
                "**Recall**: of everything that was really there, how much did you find?  \n"
                f"{sc['tp']} / ({sc['tp']} + {sc['fn']}) = **{sc['recall']:.0%}**\n\n"
                f"**F1** balances the two. The report passes at F1 ≥ {PASS_F1}."
            )
        st.caption(f"{ui.tag('measured')} Scored against our hand-checked labels: 2 chairs, 1 table, 4 cups. "
                   "A box counts as correct with the right label and IoU ≥ 0.5 with a real object.",
                   unsafe_allow_html=True)
    else:
        st.info("Pick a threshold and submit your report. You'll find out what you got right afterwards.")

    if s.pop("l3_pending_finish", None) is not None:
        _, result = det.evaluate(dets, TRUTH["objects"], threshold)
        finish_level(3, result["f1"], s.l3_attempts, report_checks(model["id"]), clue="Tea set moved")

    if solved:
        st.divider()
        completion_panel(3, [("Best F1", f"{s.best_scores.get(3, 0):.2f}")])
        ui.lesson([
            "Classification: <b>what?</b> Detection: <b>what, and where?</b> Here the extra cost was worth it.",
            "The threshold doesn't change the model. It only decides which of its guesses you believe.",
            "<b>Precision</b>: of everything predicted positive, how much was actually correct?",
            "<b>Recall</b>: of everything actually positive, how much did we find?",
            "The smallest detector was cheapest but missed small objects. The biggest didn't fit. "
            "The best model is the one that meets the mission, not the extreme on either side.",
        ], title="FIELD NOTES")

    with st.expander("Learn more: IoU, NMS and mAP"):
        st.markdown(
            "**IoU (intersection over union)** measures how well two boxes overlap: the shared area divided by the "
            "total area covered by both. 1.0 is a perfect match, 0 is no overlap.\n\n"
            "**NMS.** Detectors produce many overlapping boxes for one object, and older YOLO versions remove the "
            "duplicates afterwards with non-maximum suppression. YOLO26 is trained to output one box per object, "
            "so it can skip NMS, which makes it simpler to run on small devices.\n\n"
            "**mAP** averages precision over all recall levels, all classes, and IoU thresholds from 0.5 to 0.95. "
            "It's the single number on the model cards above."
        )
