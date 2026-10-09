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
from game import case, device, flow, runtime
from game.device import pct
from game.levels import completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level3"
TRUTH = json.loads((ASSETS / "ground_truth.json").read_text())
LIMITS = {"latency_ms": 60}
THRESHOLDS = [round(t / 20, 2) for t in range(1, 20)]   # every stop on the slider
SIDE_CONF = 0.20                                          # side scans keep boxes at or above this
LIGHT = "yolo26n.pt"
AMBER = (47, 134, 183)                                    # BGR
ASKS = {
    "inventory": "The manager wants a full list of the room to compare with last week's photos. A missed cup or "
                 "a false alarm is fine, as long as both stay small.",
    "miss_nothing": "The insurer wants every object in the room on the list, no exceptions. False alarms are "
                    "tolerated, but at least 6 in 10 of your boxes must be real.",
}
FLOORS = {"inventory": 0.75, "miss_nothing": 0.6}         # the line drawn on the chart for each brief
PRED_FROM, PRED_TO = 0.60, 0.20                           # the threshold move in the prediction
DROP_OPTIONS = ["Precision", "Recall", "Both", "Neither"]
MATCH_IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95, as in mAP50-95
RED, GREEN = (60, 74, 156), (90, 125, 94)                 # BGR


@st.cache_data
def load_scene() -> np.ndarray:
    return cv2.imread(str(ASSETS / TRUTH["image"]))


@st.cache_data(show_spinner=False)
def scene_detections(weights: str) -> list[dict]:
    return det.detect(runtime.yolo(weights), load_scene())[0]


@st.cache_data(show_spinner=False)
def tray_detections(weights: str = LIGHT) -> list[dict]:
    """The model on the tea tray crop only, boxes moved back into full-frame coordinates."""
    x0, y0, x1, y1 = TRUTH["tray_box"]
    dets, _ = det.detect(runtime.yolo(weights), load_scene()[y0:y1, x0:x1])
    return [{**d, "box": [d["box"][0] + x0, d["box"][1] + y0, d["box"][2] + x0, d["box"][3] + y0]} for d in dets]


def cups() -> list[dict]:
    return [o for o in TRUTH["objects"] if o["label"] == "cup"]


def tray_matches() -> dict:
    found = [d for d in tray_detections() if d["label"] == "cup"]
    return det.evaluate(found, cups(), SIDE_CONF)[0]


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def names(items: list[dict]) -> str:
    counts = pd.Series([i["label"] for i in items]).value_counts()
    return ", ".join(f"{n} × {label}" if n > 1 else label for label, n in counts.items())


def all_scores(dets: list[dict]) -> list[tuple[float, dict]]:
    return [(t, det.evaluate(dets, TRUTH["objects"], t)[1]) for t in THRESHOLDS]


def can_meet(dets: list[dict], brief: str) -> bool:
    return any(case.brief_passes(brief, s) for _, s in all_scores(dets))


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
                f"{pct(energy_units(seg_ms))} where boxes from a detector of the same size cost "
                f"{pct(energy_units(det_ms))}.")


def mode_options() -> dict:
    return {
        "Classify": {
            "blurb": "One label for the whole image. Cheapest.",
            "run": run_classifier, "show": show_classifier,
            "tier": "Light", "latency_ms": runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"],
            "verdict": "Classification tells you what the room looks like overall. It can't count objects or say "
                       "where they are, and that's exactly what the report needs.",
        },
        "Detect": {
            "blurb": "A box, a label and a confidence for every object.",
            "verdict": "Several objects, and you need what and where for each. That's detection, and it's worth "
                       "the extra compute here.",
        },
        "Segment": {
            "blurb": "An exact outline for every object, pixel by pixel. Most expensive.",
            "run": run_segmenter, "show": show_segmenter,
            "tier": "Balanced", "latency_ms": runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"],
            "verdict": "Segmentation is the most expensive way to make a list of objects. Boxes answer the "
                       "question; outlines are detail you're paying for and not using.",
        },
    }


def warm() -> None:
    runtime.yolo("yolo26s.pt")
    scene_detections("yolo26s.pt")


WARMUP = [("Measuring the models", runtime.benchmark), ("Warming up GhostLens", warm)]


def profiles() -> list[dict]:
    intel = st.session_state.get("intel", {}).get("l3_light")
    rows = [{**r, "accuracy": r["published"].replace(" mAP", ""), "accuracy_tag": "published",
             "metric": "COCO mAP50-95"} for r in runtime.benchmark()["detectors"]]
    for r in rows:
        if r["id"] == LIGHT and intel:
            r["intel"] = intel
    return rows


def brief_card(brief: str) -> None:
    b = case.BRIEFS[brief]
    st.markdown(f'<div class="gl-brief"><div class="gl-kicker">Client brief</div><div class="title">{b["title"]}</div>'
                f'<div>{ASKS[brief]}</div><div class="rule">passes at {b["rule"]}</div></div>',
                unsafe_allow_html=True)


def accepted(brief: str, s: dict) -> None:
    if brief == "miss_nothing":
        ui.message(f"<b>Report accepted.</b> Every object found, and {s['precision']:.0%} of your boxes are real. "
                   "For this client a false alarm is cheap and a miss is not, so recall came first.", "ok")
    else:
        ui.message(f"<b>Report accepted.</b> {s['tp']} of {s['tp'] + s['fn']} objects found, "
                   f"{s['fp']} false alarm{'s' if s['fp'] != 1 else ''}. F1 {s['f1']:.2f}: you picked a threshold "
                   "where both kinds of mistake stay small.", "ok")


def consequence(brief: str, m: dict, s: dict) -> None:
    fp, missed = m["fp"], m["missed"]
    if case.brief_passes(brief, s):
        accepted(brief, s)
    elif brief == "miss_nothing" and missed:
        ui.message(f"<b>The brief says miss nothing,</b> and {len(missed)} slipped past ({names(missed)}). "
                   "Lower the threshold: recall goes up as you accept less confident boxes.", "bad")
    elif brief == "miss_nothing":
        ui.message(f"<b>Nothing missed,</b> but only {s['precision']:.0%} of your boxes are real "
                   f"({names(fp)} are false alarms). The insurer wants at least 60%. Raise the threshold a little.",
                   "bad")
    elif s["recall"] == 1.0:
        ui.message(f"<b>You found everything.</b> You also accused {len(fp)} innocent objects ({names(fp)}). "
                   "The manager is not going to search the parlour for a cup that's actually a coffee pot. "
                   "Low threshold: high recall, low precision.", "bad")
    elif s["precision"] == 1.0:
        ui.message(f"<b>Everything you flagged is real.</b> But {len(missed)} objects slipped past ({names(missed)}). "
                   "One of them could be what the entity moved. High threshold: high precision, low recall.", "bad")
    else:
        ui.message(f"Some of both: {len(fp)} false alarm(s) ({names(fp)}) and {len(missed)} missed "
                   f"({names(missed)}). Move the threshold and try again.", "warn")


def pr_curve(dets: list[dict], threshold: float, brief: str) -> alt.Chart:
    rows = [{"threshold": t, "metric": label, "value": s[k]}
            for t, s in all_scores(dets) for k, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))]
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("threshold:Q", title="confidence threshold"),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", scale=alt.Scale(domain=["Precision", "Recall", "F1"],
                                                    range=[flow.BAD, flow.OK, flow.INK]),
                        legend=alt.Legend(orient="top", title=None)),
    )
    floor = pd.DataFrame({"y": [FLOORS[brief]], "x": [0.05], "label": [f"brief: {case.BRIEFS[brief]['rule']}"]})
    rule = alt.Chart(floor).mark_rule(color=flow.WARN, strokeDash=[6, 3]).encode(y="y:Q")
    text = alt.Chart(floor).mark_text(align="left", dy=-7, color=flow.WARN, fontSize=11).encode(
        x="x:Q", y="y:Q", text="label:N")
    now = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=flow.SOFT, strokeDash=[4, 3]).encode(x="t:Q")
    return (lines + rule + text + now).properties(height=220).configure_view(stroke=None).configure(
        background="transparent")


def with_moved(img: np.ndarray, box: list[float]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(out, (x0 - 9, y0 - 9), (x1 + 9, y1 + 9), AMBER, 3)
    cv2.putText(out, "MOVED", (x0 - 9, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2, cv2.LINE_AA)
    return out


def par() -> int:
    return energy_units(runtime.profile("detectors", "yolo26s.pt")["latency_ms"])


# prediction: what can drop when the threshold goes down?

def which_drops(before: dict, after: dict) -> set[str]:
    return {k for k in ("precision", "recall", "f1") if after[k] < before[k] - 1e-9}


def drops_answer(dropped: set[str]) -> int:
    """Index into DROP_OPTIONS. Only precision and recall are asked about."""
    p, r = "precision" in dropped, "recall" in dropped
    return 2 if p and r else 0 if p else 1 if r else 3


def resolve_drops(dets: list[dict], name: str) -> tuple[int, str]:
    before = det.evaluate(dets, TRUTH["objects"], PRED_FROM)[1]
    after = det.evaluate(dets, TRUTH["objects"], PRED_TO)[1]
    return drops_answer(which_drops(before, after)), (
        f"{name}: precision {before['precision']:.0%} → {after['precision']:.0%}, recall {before['recall']:.0%} → "
        f"{after['recall']:.0%}. A lower threshold only adds boxes. Every match you had stays, so recall can't "
        "fall. The new boxes are less sure, so precision can.")


def drops_reveal(dets: list[dict]) -> None:
    rows = [{"threshold": t, **{k: round(v, 2) for k, v in det.evaluate(dets, TRUTH["objects"], t)[1].items()}}
            for t in (PRED_FROM, PRED_TO)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def threshold_prediction(dets: list[dict], model: dict) -> None:
    s = st.session_state
    if "l3_pred_drop" not in s and s.get("l3_attempts", 0):
        return
    flow.predict("l3_pred_drop", f"Suppose you moved the threshold from {PRED_FROM:.2f} down to {PRED_TO:.2f}. "
                 "Which of these can drop?", DROP_OPTIONS, resolve=lambda: resolve_drops(dets, model["name"]),
                 reveal=lambda: drops_reveal(dets), ready=s.get("l3_attempts", 0) > 0)


# after a report: why each false alarm didn't count

def alarm_crop(d: dict, nearest: dict | None) -> np.ndarray:
    img = load_scene().copy()
    boxes = [d["box"]] + ([nearest["box"]] if nearest else [])
    x0, y0 = (max(0, int(min(b[i] for b in boxes)) - 40) for i in (0, 1))
    x1, y1 = (int(max(b[i] for b in boxes)) + 40 for i in (2, 3))
    if nearest:
        a, b, c, e = map(int, nearest["box"])
        cv2.rectangle(img, (a, b), (c, e), GREEN, 2)
    a, b, c, e = map(int, d["box"])
    cv2.rectangle(img, (a, b), (c, e), RED, 2)
    return img[y0:y1, x0:x1]


def false_alarm_view(m: dict) -> None:
    alarms = det.explain_false_alarms(m, TRUTH["objects"])
    if not alarms:
        return
    names = [f"{a['det']['label']} {a['det']['conf']:.2f}" for a in alarms]
    pick = st.selectbox("Why didn't this box count?", range(len(alarms)), format_func=lambda i: names[i],
                        key="l3_alarm")
    a = alarms[pick]
    nearest = TRUTH["objects"][a["truth"]] if a["truth"] is not None else None
    c1, c2 = st.columns([1, 1.6])
    c1.image(rgb(alarm_crop(a["det"], nearest)), width="stretch")
    near = f" Nearest labelled object: {nearest['label']}, IoU {a['iou']:.2f}." if nearest else ""
    c2.markdown(f"**{det.REASONS[a['reason']]}**{near}\n\nRed: the false alarm. Green: the labelled object it "
                "was compared with.")


# after solving: how strict is "correct"?

def match_sweep(dets: list[dict], truth: list[dict], threshold: float, ious: list[float]) -> list[dict]:
    kept = [d for d in dets if d["conf"] >= threshold]
    return [{"match_iou": v, **det.scores(det.match(kept, truth, v))} for v in ious]


def sweep_chart(rows: list[dict], chosen: float) -> alt.Chart:
    df = pd.DataFrame([{"match_iou": r["match_iou"], "metric": label, "value": r[k]} for r in rows
                       for k, label in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))])
    lines = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("match_iou:Q", title="IoU needed to count as correct", scale=alt.Scale(domain=[0.5, 0.95])),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", scale=alt.Scale(domain=["Precision", "Recall", "F1"],
                                                    range=[flow.BAD, flow.OK, flow.INK]),
                        legend=alt.Legend(orient="top", title=None)))
    now = alt.Chart(pd.DataFrame({"x": [chosen]})).mark_rule(color=flow.SOFT, strokeDash=[4, 3]).encode(x="x:Q")
    return (lines + now).properties(height=200).configure_view(stroke=None).configure(background="transparent")


def debrief(dets: list[dict], threshold: float) -> None:
    chosen = st.select_slider("Matching IoU", MATCH_IOUS, value=0.5, key="l3_match_iou",
                              help="How much a box must overlap a labelled object to count as correct.")
    rows = match_sweep(dets, TRUTH["objects"], threshold, MATCH_IOUS)
    st.altair_chart(sweep_chart(rows, chosen), width="stretch")
    at = next(r for r in rows if r["match_iou"] == chosen)
    st.caption(f"{ui.tag('measured')} your boxes at threshold {threshold:.2f}, scored against our labels",
               unsafe_allow_html=True)
    light = runtime.profile("detectors", LIGHT)
    ui.lesson([
        f"At matching IoU {chosen:.2f}, {at['tp']} of your boxes still count (F1 {at['f1']:.2f}). COCO's mAP50-95 "
        "averages over 0.50 to 0.95, which is why it's always below mAP50.",
        "The threshold doesn't change the model. Lowering it keeps every match you had, so recall never falls, "
        "and precision pays for the extra boxes.",
        f"{light['name']} was cheapest but its best recall here was {light['best_recall']:.0%}: the cups are only a "
        "few dozen pixels wide. The model to ship is the one that meets the brief.",
    ], title="DEBRIEF")


def label_lines() -> str:
    names = runtime.yolo("yolo26s.pt").names
    ids = {v: k for k, v in names.items()}
    h, w = load_scene().shape[:2]
    return "\n".join(det.yolo_lines(TRUTH["objects"], w, h, ids))


def report_checks(model_id: str) -> dict:
    s = st.session_state
    model = runtime.profile("detectors", model_id)
    tried = s.get("l3_models_tried", [])
    used = device.used_in_level(s, 3)
    misses = flow.mode_misses(3)
    return {
        "Right task": (misses == 0, "detection first time" if misses == 0
                       else f"tried {', '.join(m for m in s.l3_tried if m != 'Detect')} first"),
        "Right-sized model": (len(tried) == 1, f"{model['name']} first time" if len(tried) == 1
                              else "needed a second model after the first couldn't find the cups"),
        "Latency target": (model["latency_ms"] <= LIMITS["latency_ms"], f"≤ {LIMITS['latency_ms']} ms"),
        "Battery": (efficient(used, par()), f"{pct(used)} used, {pct(par())} would have done it"),
    }


# side scan: crop to the tea tray and run the light detector again

def show_tray() -> None:
    m = tray_matches()
    x0, y0, x1, y1 = TRUTH["tray_box"]
    scene = load_scene()
    full = 640 / max(scene.shape[:2])
    crop = 640 / max(x1 - x0, y1 - y0)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.image(rgb(det.draw(scene, m)[max(0, y0 - 30):y1, x0:x1]), width="stretch")
        ui.caption("EVIDENCE 05-T · TEA TRAY · YOLO26N ON THE CROP")
    c2.markdown(f"**{len(m['tp'])} of 4 cups** above {SIDE_CONF:.2f}, from the same light model that found none of "
                f"them in the full photo. YOLO resizes every input to 640 pixels: the whole room goes in at "
                f"{full:.0%} scale, the crop at {crop:.0%}, so each cup covers about {(crop / full) ** 2:.0f}× more "
                "input pixels.")


def tray_scan() -> flow.SideScan:
    found = len(tray_matches()["tp"])
    return flow.SideScan(
        key="tray", title="Zoom on the tea tray",
        blurb="The light detector couldn't see a single cup from across the room. Crop the photo to the tea tray "
              "and run YOLO26n again, on the crop only.",
        what="YOLO26n on the tea tray crop", latency_ms=runtime.profile("detectors", LIGHT)["latency_ms"],
        reward_xp=25, clue=f"Tea tray up close: YOLO26n found {found} of the 4 cups",
        run=lambda: len(tray_matches()["tp"]) > 0, show=show_tray, tip_topic="Input resolution")


def report_view(dets: list[dict], threshold: float, brief: str, sc: dict, meetable: bool, model: dict,
                solved: bool) -> None:
    passed = case.brief_passes(brief, sc)
    ui.readouts([
        ("Found (TP)", str(sc["tp"]), "ok"),
        ("False alarms (FP)", str(sc["fp"]), "bad" if sc["fp"] else ""),
        ("Missed (FN)", str(sc["fn"]), "bad" if sc["fn"] else ""),
        ("Precision", f"{sc['precision']:.0%}", ""),
        ("Recall", f"{sc['recall']:.0%}", ""),
        ("F1", f"{sc['f1']:.2f}", "ok" if passed else "warn"),
    ])
    if not solved and not meetable:
        best = max(s["f1"] for _, s in all_scores(dets))
        top = max(s["recall"] for _, s in all_scores(dets))
        ui.message(f"<b>{model['name']} can't meet this brief at any threshold.</b> Its best F1 on this room is "
                   f"{best:.2f} and its best recall {top:.0%}: the tea cups are too small for it. Load a bigger "
                   "model above (it costs battery to run again).", "bad")
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.altair_chart(pr_curve(dets, threshold, brief), width="stretch")
    with c2:
        st.markdown(
            "**Precision**: of everything you flagged, how much was real?  \n"
            f"{sc['tp']} / ({sc['tp']} + {sc['fp']}) = **{sc['precision']:.0%}**\n\n"
            "**Recall**: of everything that was really there, how much did you find?  \n"
            f"{sc['tp']} / ({sc['tp']} + {sc['fn']}) = **{sc['recall']:.0%}**\n\n"
            f"**F1** = 2PR / (P + R) = **{sc['f1']:.2f}**. This brief passes at {case.BRIEFS[brief]['rule']}."
        )
    st.caption(f"{ui.tag('measured')} Scored against our hand-checked labels: 2 chairs, 1 table, 4 cups. "
               "A box counts as correct with the right label and IoU ≥ 0.5 with a real object.",
               unsafe_allow_html=True)


def render() -> None:
    ui.scene_header(
        "CHAPTER 3 · PARLOUR · CAM 05 · 04:12",
        "Find the Anomalies",
        "The lift went back up to the antique parlour at 04:05. The cleaner swears the tea set was rearranged "
        "overnight, but she can't say what moved. Someone needs a list of every object in the room and where it "
        "is, to compare with last week's photos.",
    )
    try:
        runtime.benchmark()
    except ModelMissing as e:
        st.warning(str(e))
        return
    s = st.session_state
    c = case.get_case(s)
    solved = 3 in s.completed_levels
    device.unload(s, "task")  # the classifier from chapter 2 isn't needed any more

    if not solved and not flow.mode_choice(
            3, "Lots of objects in one frame. You need to know what each one is and where it is.",
            mode_options(), "Detect", needs="what each object is and where it is"):
        return

    brief_card(c.brief)
    model = flow.model_picker(
        3, profiles(), LIMITS, slot="watchdog",
        note="Accuracy is COCO mAP50-95 as published by Ultralytics. The detector you load stays in memory as a "
             "watchdog for the rest of the case.")
    if model is None:
        return

    ran = s.setdefault("l3_ran", [])
    if model["id"] not in ran:
        if flow.run_button(3, f"Run {model['name']} on the parlour", f"{model['name']} on the parlour",
                           model["latency_ms"], model["tier"], key=f"l3_run_{model['id']}"):
            ran.append(model["id"])
            st.rerun()
        return

    dets = scene_detections(model["id"])
    meetable = can_meet(dets, c.brief)
    reported = s.get("l3_reported") == model["id"]
    s.setdefault("l3_threshold", case.BRIEFS[c.brief]["start"])   # set before the slider exists
    threshold_prediction(dets, model)

    view, ctrl = st.columns([2.2, 1], gap="large")
    with ctrl:
        st.markdown(f"**Detect mode · {model['name']}**")
        threshold = st.slider("Confidence threshold", 0.05, 0.95, step=0.05, key="l3_threshold",
                              help="Only boxes the model is at least this sure about are kept. "
                                   "Moving it doesn't re-run the model.")
        shown = [d for d in dets if d["conf"] >= threshold]
        st.markdown(f'<div class="gl-kicker">{len(shown)} boxes above {threshold:.2f}</div>', unsafe_allow_html=True)
        if shown:
            st.dataframe(pd.DataFrame([{"object": d["label"], "confidence": round(d["conf"], 2)} for d in shown]),
                         hide_index=True, height=min(36 * len(shown) + 38, 290), width="stretch")
        submitted = not solved and st.button("Submit report", type="primary", width="stretch")

    m, sc = det.evaluate(dets, TRUTH["objects"], threshold)
    if submitted:
        s.l3_reported = model["id"]
        s.l3_attempts = s.get("l3_attempts", 0) + 1
        reported = True
        if case.brief_passes(c.brief, sc):
            finish_level(3, sc["f1"], s.l3_attempts, report_checks(model["id"]), clue="Tea set moved")
        if (s.get("l3_pred_drop") or {}).get("right", False) is None:
            st.rerun()      # the prediction above was drawn before this submit; reveal it now

    with view:
        if reported or solved:
            img = det.draw(load_scene(), m)
            if solved:
                img = with_moved(img, TRUTH["objects"][c.moved]["box"])
            st.image(rgb(img), width="stretch")
            ui.caption("EVIDENCE 05-C · GREEN = CORRECT · RED = FALSE ALARM · DASHED AMBER = MISSED"
                       + (" · SOLID AMBER = MOVED" if solved else ""))
        else:
            st.image(rgb(det.draw(load_scene(), dets=shown)), width="stretch")
            ui.caption(f"EVIDENCE 05-C · GHOSTLENS DETECT MODE · {model['name'].upper()}")

    if reported or solved:
        if not solved:
            consequence(c.brief, m, sc)
        report_view(dets, threshold, c.brief, sc, meetable, model, solved)
        false_alarm_view(m)
    else:
        st.info("Pick a threshold and submit your report. You'll find out what you got right afterwards.")

    if solved:
        st.divider()
        completion_panel(3, [("Best F1", f"{s.best_scores.get(3, 0):.2f}"),
                             ("Brief", case.BRIEFS[c.brief]["title"]), ("Moved", case.MOVABLE[c.moved])],
                         par=par(), debrief=lambda: debrief(dets, threshold))
        flow.side_scan(3, tray_scan())

    with st.expander("Learn more: IoU, NMS, mAP and detection labels"):
        st.markdown(
            "**IoU (intersection over union)** measures how well two boxes overlap: the shared area divided by the "
            "total area covered by both. 1.0 is a perfect match, 0 is no overlap.\n\n"
            "**NMS.** Detectors produce many overlapping boxes for one object, and older YOLO versions remove the "
            "duplicates afterwards with non-maximum suppression. YOLO26 is trained to output one box per object, "
            "so it can skip NMS, which makes it simpler to run on small devices.\n\n"
            "**mAP** averages precision over all recall levels, all classes, and IoU thresholds from 0.5 to 0.95. "
            "It's the number on the model cards above.\n\n"
            "**Detection labels.** Our ground truth for this room, written the way a YOLO training set stores it: "
            "one line per object, the COCO class id, then the box centre x and y, width and height, each as a "
            "fraction of the image."
        )
        st.code(label_lines(), language=None)
