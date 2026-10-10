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
from game import case, device, flow, levels, runtime
from game.device import pct
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level3"
TRUTH = json.loads((ASSETS / "ground_truth.json").read_text())
LIMITS = {"latency_ms": 60}
THRESHOLDS = [round(t / 20, 2) for t in range(1, 20)]   # every stop on the slider
SIDE_CONF = 0.20                                          # side scans keep boxes at or above this
LIGHT = "yolo26n.pt"
INPUT = 640                                               # YOLO resizes the longest side to this
AMBER = (47, 134, 183)                                    # BGR
BRASS = (100, 164, 200)                                   # BGR of #C8A464, the inspected box
OBJECTIVES = {
    "inventory": "Full inventory · F1 ≥ 0.75",
    "miss_nothing": "Miss nothing · recall 100%, precision ≥ 60%",
}
ASKS = {
    "inventory": "The manager wanted a full list of the room to compare with last week's photos. A missed cup or "
                 "a false alarm was fine, as long as both stayed small.",
    "miss_nothing": "The insurer wanted every object in the room on the list, no exceptions. False alarms were "
                    "tolerated, but at least 6 in 10 of the boxes had to be real.",
}
FLOORS = {"inventory": 0.75, "miss_nothing": 0.6}         # the line drawn on the chart for each brief
MATCH_IOUS = [round(0.5 + 0.05 * i, 2) for i in range(10)]  # 0.50 .. 0.95, as in mAP50-95
METRICS = (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))
METRIC_COLORS = alt.Scale(domain=["Precision", "Recall", "F1"], range=[ui.CHART[3], ui.CHART[1], ui.CHART[0]])


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


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def all_scores(dets: list[dict]) -> list[tuple[float, dict]]:
    return [(t, det.evaluate(dets, TRUTH["objects"], t)[1]) for t in THRESHOLDS]


def can_meet(dets: list[dict], brief: str) -> bool:
    return any(case.brief_passes(brief, s) for _, s in all_scores(dets))


# wrong modes: they really run, and only their picture goes on the stage

def run_classifier() -> tuple[str, float]:
    st.session_state.l3_cls = clf.classify(runtime.yolo("yolo26n-cls.pt"), load_scene())
    return "YOLO26n-cls on the parlour", runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"]


def show_classifier() -> None:
    label, p = st.session_state.l3_cls["top"][0]
    ui.evidence(rgb(load_scene()), f"EVIDENCE 05-C · GHOSTLENS SAYS: {label.upper()} {p:.0%}")


def classifier_line() -> str:
    top = st.session_state.get("l3_cls")
    said = f"'{top['top'][0][0]}' for the whole room" if top else "one label for the whole room"
    return f"It said {said}. No count, no positions."


def run_segmenter() -> tuple[str, float]:
    r = runtime.yolo("yolo26s-seg.pt").predict(load_scene(), device="cpu", verbose=False)[0]
    st.session_state.l3_seg = (r.plot(boxes=False)[..., ::-1], [r.names[int(c)] for c in r.boxes.cls])
    return "YOLO26s-seg on the parlour", runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"]


def show_segmenter() -> None:
    img, labels = st.session_state.l3_seg
    ui.evidence(img, f"EVIDENCE 05-C · YOLO26S-SEG · {len(labels)} OUTLINES")


def segmenter_line() -> str:
    seg_ms = runtime.profile("segmenters", "yolo26s-seg.pt")["latency_ms"]
    det_ms = runtime.profile("detectors", "yolo26s.pt")["latency_ms"]
    return (f"Outlines cost {pct(energy_units(seg_ms))} where boxes cost {pct(energy_units(det_ms))}, "
            "and the list only needs boxes.")


def mode_options() -> dict:
    return {
        "Classify": {
            "blurb": "One label, whole image",
            "line": classifier_line(),
            "run": run_classifier, "show": show_classifier,
            "tier": "Light", "latency_ms": runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"],
            "verdict": "Classification tells you what the room looks like overall. It can't count objects or say "
                       "where they are, and that's exactly what the report needs.",
        },
        "Detect": {
            "blurb": "Box and label per object",
            "line": "",
            "verdict": "Several objects, and you need what and where for each. That's detection, and it's worth "
                       "the extra compute here.",
        },
        "Segment": {
            "blurb": "Outline every object's pixels",
            "line": segmenter_line(),
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


def consequence(brief: str, m: dict, s: dict) -> tuple[str, str]:
    """One feedback line about what the report did, and its tone."""
    fp, missed = m["fp"], m["missed"]
    if case.brief_passes(brief, s):
        return "<b>Report accepted.</b>", "ok"
    if brief == "miss_nothing" and missed:
        return (f"<b>The brief says miss nothing,</b> and {len(missed)} slipped past ({names(missed)}). "
                "Lower the threshold.", "bad")
    if brief == "miss_nothing":
        return (f"<b>Nothing missed,</b> but only {s['precision']:.0%} of your boxes are real. "
                "The insurer wants 60%. Raise the threshold a little.", "bad")
    if s["recall"] == 1.0:
        return (f"<b>You found everything</b> and accused {plural(len(fp), 'innocent object')} ({names(fp)}). "
                "Low threshold: high recall, low precision.", "bad")
    if s["precision"] == 1.0:
        return (f"<b>Everything you flagged is real,</b> but {len(missed)} slipped past ({names(missed)}). "
                "High threshold: high precision, low recall.", "bad")
    return (f"{plural(len(fp), 'false alarm')} and {len(missed)} missed. Move the threshold and try again.",
            "warn")


def cant_meet_line(dets: list[dict], model: dict) -> str:
    top = max(s["recall"] for _, s in all_scores(dets))
    return (f"<b>{model['name']} can't meet this brief at any threshold.</b> Best recall {top:.0%}: the cups are "
            "too small for it. Load a bigger model.")


def pr_curve(dets: list[dict], threshold: float, brief: str) -> alt.Chart:
    rows = [{"threshold": t, "metric": label, "value": s[k]} for t, s in all_scores(dets) for k, label in METRICS]
    lines = alt.Chart(pd.DataFrame(rows)).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("threshold:Q", title="confidence threshold"),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", scale=METRIC_COLORS, legend=alt.Legend(orient="top", title=None)),
    )
    floor = pd.DataFrame({"y": [FLOORS[brief]], "x": [0.05], "label": [f"brief: {case.BRIEFS[brief]['rule']}"]})
    rule = alt.Chart(floor).mark_rule(color=ui.CHART[0], strokeDash=[6, 3]).encode(y="y:Q")
    text = alt.Chart(floor).mark_text(align="left", dy=-7, color=ui.CHART[0], fontSize=11).encode(
        x="x:Q", y="y:Q", text="label:N")
    now = alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(color=ui.CHART[4], strokeDash=[4, 3]).encode(x="t:Q")
    return (lines + rule + text + now).properties(height=240)


def with_moved(img: np.ndarray, box: list[float]) -> np.ndarray:
    out = img.copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(out, (x0 - 9, y0 - 9), (x1 + 9, y1 + 9), AMBER, 3)
    cv2.putText(out, "MOVED", (x0 - 9, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMBER, 2, cv2.LINE_AA)
    return out


def with_inspected(img: np.ndarray, box: list[float] | None) -> np.ndarray:
    if box is None:
        return img
    out = img.copy()
    x0, y0, x1, y1 = (int(v) for v in box)
    cv2.rectangle(out, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), BRASS, 5)
    return out


def par() -> int:
    return energy_units(runtime.profile("detectors", "yolo26s.pt")["latency_ms"])


# the box inspector: zoom on one box, its size at the model's input, and after a report why it counted or not

def inspect_items(shown: list[dict], dets: list[dict], m: dict | None) -> dict[int, dict]:
    """{id: item} for the inspector radio. Detections keep their index in dets; missed objects get -(truth + 1)."""
    items = {dets.index(d): {"kind": "box", "det": d} for d in shown}
    if m is not None:
        for t in m["missed"]:
            items[-(TRUTH["objects"].index(t) + 1)] = {"kind": "missed", "truth": t}
    return items


def item_label(item: dict) -> str:
    if item["kind"] == "missed":
        return f"missed: {item['truth']['label']}"
    return f"{item['det']['label']} {item['det']['conf']:.2f}"


def item_box(item: dict) -> list[float]:
    return item["truth"]["box"] if item["kind"] == "missed" else item["det"]["box"]


def size_line(box: list[float]) -> str:
    h, w = load_scene().shape[:2]
    bw, bh = round(box[2] - box[0]), round(box[3] - box[1])
    iw, ih = det.input_size(box, (h, w), INPUT)
    return f"{bw}×{bh} px here → {iw}×{ih} px at the model's {INPUT}-px input"


def verdict_line(item: dict, m: dict) -> str:
    """'✓ correct · IoU 0.82' or '✗ false alarm · reason', once a report has been scored."""
    if item["kind"] == "missed":
        return "✗ missed · no box above the threshold overlaps it with the right label"
    d = item["det"]
    hit = next((t for t in m["tp"] if t["box"] == d["box"] and t["conf"] == d["conf"]), None)
    if hit:
        return f"✓ correct · IoU {hit['iou']:.2f}"
    alarm = next(a for a in det.explain_false_alarms(m, TRUTH["objects"])
                 if a["det"]["box"] == d["box"] and a["det"]["conf"] == d["conf"])
    return f"✗ false alarm · {det.REASONS[alarm['reason']]}"


def inspector(items: dict[int, dict], m: dict | None) -> list[float] | None:
    """Draws the inspector in the scanner. Returns the box to outline on the stage."""
    s = st.session_state
    if s.get("l3_inspect") is not None and s["l3_inspect"] not in items:
        del s["l3_inspect"]          # that box dropped below the threshold
    if not items:
        st.caption("No boxes to inspect at this threshold.")
        return None
    pick = st.radio("Box inspector", list(items), index=None, horizontal=True, key="l3_inspect",
                    format_func=lambda i: item_label(items[i]))
    if pick is None:
        return None
    s.setdefault("l3_inspected", set()).add(pick)
    item = items[pick]
    box = item_box(item)
    st.image(rgb(det.zoom_crop(load_scene(), box)), width="stretch")
    lines = [size_line(box)] + ([verdict_line(item, m)] if m is not None else [])
    st.markdown('<div class="gl-screen gl-inspect">' + "<br>".join(lines) + "</div>", unsafe_allow_html=True)
    return box


def last_report(model_id: str) -> dict | None:
    return next((r for r in reversed(st.session_state.get("l3_reports", [])) if r["model"] == model_id), None)


def submit(brief: str, model: dict, threshold: float, sc: dict) -> None:
    s = st.session_state
    s.l3_reported = model["id"]
    s.l3_attempts = s.get("l3_attempts", 0) + 1
    s.setdefault("l3_reports", []).append({"threshold": threshold, "model": model["id"], **sc})
    if case.brief_passes(brief, sc):
        s.l3_final = {"threshold": threshold, "model": model["id"]}
        levels.finish_level(3, sc["f1"], s.l3_attempts, report_checks(model["id"]), clue="Tea set moved")
    st.rerun()


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
    full = INPUT / max(scene.shape[:2])
    crop = INPUT / max(x1 - x0, y1 - y0)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        ui.evidence(rgb(det.draw(scene, m)[max(0, y0 - 30):y1, x0:x1]), "EVIDENCE 05-T · TEA TRAY · YOLO26N ON THE CROP")
    c2.markdown(f"**{len(m['tp'])} of 4 cups** above {SIDE_CONF:.2f}, from the same light model that found none of "
                f"them in the full photo. YOLO resizes every input to {INPUT} pixels: the whole room goes in at "
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


# the cleared screen

def match_sweep(dets: list[dict], truth: list[dict], threshold: float, ious: list[float]) -> list[dict]:
    kept = [d for d in dets if d["conf"] >= threshold]
    return [{"match_iou": v, **det.scores(det.match(kept, truth, v))} for v in ious]


def sweep_chart(rows: list[dict], chosen: float) -> alt.Chart:
    df = pd.DataFrame([{"match_iou": r["match_iou"], "metric": label, "value": r[k]} for r in rows
                       for k, label in METRICS])
    lines = alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=18)).encode(
        x=alt.X("match_iou:Q", title="IoU needed to count as correct", scale=alt.Scale(domain=[0.5, 0.95]),
                axis=alt.Axis(values=MATCH_IOUS, format=".2f")),
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.05]), axis=alt.Axis(format="%")),
        color=alt.Color("metric:N", scale=METRIC_COLORS, legend=alt.Legend(orient="top", title=None)))
    now = alt.Chart(pd.DataFrame({"x": [chosen]})).mark_rule(color=ui.CHART[4], strokeDash=[4, 3]).encode(x="x:Q")
    return (lines + now).properties(height=220)


def label_lines() -> str:
    ids = {v: k for k, v in runtime.yolo("yolo26s.pt").names.items()}
    h, w = load_scene().shape[:2]
    return "\n".join(det.yolo_lines(TRUTH["objects"], w, h, ids))


def alarm_rows(m: dict) -> list[dict]:
    return [{"label": a["det"]["label"], "confidence": round(a["det"]["conf"], 2), "reason": det.REASONS[a["reason"]]}
            for a in det.explain_false_alarms(m, TRUTH["objects"])]


def final_report() -> dict:
    s = st.session_state
    final = s.get("l3_final") or {"threshold": s.get("l3_threshold", 0.2), "model": s.get("l3_model", "yolo26s.pt")}
    dets = scene_detections(final["model"])
    m, sc = det.evaluate(dets, TRUTH["objects"], final["threshold"])
    return {**final, "dets": dets, "m": m, "sc": sc}


def numbers(r: dict, brief: str) -> None:
    ui.chart(pr_curve(r["dets"], r["threshold"], brief))
    st.caption(f"{ui.tag('measured')} {runtime.profile('detectors', r['model'])['name']} on the parlour, scored "
               "against our hand-checked labels (2 chairs, 1 table, 4 cups) at every threshold. Dashed: the brief "
               "and your threshold.", unsafe_allow_html=True)
    rows = alarm_rows(r["m"])
    st.markdown(f'<div class="gl-kicker">False alarms in your report · {len(rows)}</div>', unsafe_allow_html=True)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("None: every box you kept matched a labelled object.")
    st.markdown('<div class="gl-kicker">Our labels, written as a YOLO training set stores them</div>',
                unsafe_allow_html=True)
    st.code(label_lines(), language=None)
    st.caption("One line per object: COCO class id, then box centre x and y, width and height, each as a fraction "
               "of the image.")
    light = runtime.profile("detectors", LIGHT)
    st.markdown(f"{ui.tag('measured')} {light['name']} was the cheapest detector, but its best recall on this room "
                f"was {light['best_recall']:.0%} at any threshold: the cups are only a few dozen pixels wide, and "
                f"half that at the model's {INPUT}-px input.", unsafe_allow_html=True)
    sc = r["sc"]
    with st.expander("The maths"):
        st.markdown(
            "**Precision**: of everything you flagged, how much was real?  \n"
            f"TP / (TP + FP) = {sc['tp']} / ({sc['tp']} + {sc['fp']}) = **{sc['precision']:.0%}**\n\n"
            "**Recall**: of everything really there, how much did you find?  \n"
            f"TP / (TP + FN) = {sc['tp']} / ({sc['tp']} + {sc['fn']}) = **{sc['recall']:.0%}**\n\n"
            f"**F1** = 2PR / (P + R) = **{sc['f1']:.2f}**\n\n"
            "**IoU** (intersection over union): the shared area of two boxes divided by the area they cover "
            "together. A box counts as correct with the right label and IoU ≥ 0.5 with a labelled object.\n\n"
            "**NMS.** Detectors produce many overlapping boxes for one object, and older YOLO versions remove the "
            "duplicates afterwards with non-maximum suppression. YOLO26 is trained to output one box per object.\n\n"
            "**mAP** averages precision over recall levels, classes and IoU thresholds from 0.5 to 0.95. It's the "
            "published number on the detector cards.")


def explore(r: dict) -> None:
    st.markdown('<div class="gl-kicker">How strict is "correct"?</div>', unsafe_allow_html=True)
    chosen = st.select_slider("Matching IoU", MATCH_IOUS, value=0.5, key="l3_match_iou", format_func="{:.2f}".format,
                              help="How much a box must overlap a labelled object to count as correct.")
    rows = match_sweep(r["dets"], TRUTH["objects"], r["threshold"], MATCH_IOUS)
    ui.chart(sweep_chart(rows, chosen))
    at = next(x for x in rows if x["match_iou"] == chosen)
    st.caption(f"{ui.tag('measured')} At matching IoU {chosen:.2f}, {at['tp']} of your boxes still count "
               f"(F1 {at['f1']:.2f}). COCO's mAP50-95 averages over 0.50 to 0.95, which is why it's never higher "
               "than mAP50.", unsafe_allow_html=True)


def happened(r: dict) -> list[str]:
    s = st.session_state
    c = case.get_case(s)
    lines = [ASKS[c.brief]] + flow.wrong_mode_lines(3, mode_options())
    tried = [runtime.profile("detectors", i)["name"] for i in s.get("l3_models_tried", [])]
    if len(tried) > 1:
        lines.append(f"You loaded {' then '.join(tried)}: the first couldn't find the cups at any threshold.")
    for n, rep in enumerate(s.get("l3_reports", []), start=1):
        name = runtime.profile("detectors", rep["model"])["name"]
        lines.append(f"Report {n}: {name} at {rep['threshold']:.2f}, {rep['tp']} found, "
                     f"{plural(rep['fp'], 'false alarm')}, {rep['fn']} missed, F1 {rep['f1']:.2f}.")
    inspected = len(s.get("l3_inspected", ()))
    if inspected:
        lines.append(f"You inspected {inspected} box{'es' if inspected != 1 else ''} up close.")
    return lines


def cleared() -> "levels.Cleared":
    s = st.session_state
    c = case.get_case(s)
    r = final_report()
    sc = r["sc"]
    name = runtime.profile("detectors", r["model"])["name"]
    why = [
        "Several objects in one frame, and the report needed what and where for each: that's detection.",
        f"{name} was the smallest detector that could see the cups. The light one couldn't at any threshold.",
        "The threshold doesn't change the model. Lowering it keeps every match you had, so recall never falls, "
        "and precision pays for the extra boxes.",
        f"At {r['threshold']:.2f}: precision {sc['precision']:.0%}, recall {sc['recall']:.0%}, F1 {sc['f1']:.2f}, "
        f"which meets {case.BRIEFS[c.brief]['rule']}.",
    ]

    def evidence() -> None:
        img = with_moved(det.draw(load_scene(), r["m"]), TRUTH["objects"][c.moved]["box"])
        ui.evidence(rgb(img), "EVIDENCE 05-C · GREEN = CORRECT · RED = FALSE ALARM · DASHED AMBER = MISSED · "
                              "SOLID AMBER = MOVED")

    return levels.Cleared(
        headline=f"Report accepted: F1 {sc['f1']:.2f} at threshold {r['threshold']:.2f}. "
                 f"Moved: {case.MOVABLE[c.moved]}.",
        happened=happened(r), why=why,
        concept="Object detection · confidence threshold · precision/recall/F1 · IoU · mAP · input resolution",
        evidence=evidence, numbers=lambda: numbers(r, c.brief), explore=lambda: explore(r), side=tray_scan(),
        stats=[("Best F1", f"{s.best_scores.get(3, 0):.2f}"), ("Brief", case.BRIEFS[c.brief]["title"]),
               ("Moved", case.MOVABLE[c.moved])],
        par=par())


# the play screen

def draw_parlour() -> None:
    ui.evidence(rgb(load_scene()), "EVIDENCE 05-C · THE PARLOUR · CAM 05 · 04:12")


def scanner_controls(c: case.Case, model: dict, dets: list[dict], solved: bool) -> tuple:
    """Threshold, box count, inspector and the submit button. Returns (threshold, scored match, inspected box)."""
    s = st.session_state
    # set before the slider exists; the copy brings the value back after a visit to another page
    s.setdefault("l3_threshold", s.get("l3_keep_threshold", case.BRIEFS[c.brief]["start"]))
    threshold = st.slider("Confidence threshold", 0.05, 0.95, step=0.05, key="l3_threshold",
                          help="Only boxes the model is at least this sure about are kept. Moving it is free: "
                               "the model doesn't run again.")
    s.l3_keep_threshold = threshold
    shown = [d for d in dets if d["conf"] >= threshold]
    st.markdown(f'<div class="gl-kicker">{len(shown)} boxes ≥ {threshold:.2f}</div>', unsafe_allow_html=True)
    rep = last_report(model["id"])
    scored = rep is not None and rep["threshold"] == threshold
    m = det.evaluate(dets, TRUTH["objects"], threshold)[0] if scored else None
    box = inspector(inspect_items(shown, dets, m), m)
    if not solved and st.button("Submit report", type="primary", width="stretch", key="l3_submit"):
        submit(c.brief, model, threshold, det.evaluate(dets, TRUTH["objects"], threshold)[1])
    if scored:
        ui.readouts([("TP", str(rep["tp"]), "ok"), ("FP", str(rep["fp"]), "bad" if rep["fp"] else ""),
                     ("FN", str(rep["fn"]), "bad" if rep["fn"] else ""), ("P", f"{rep['precision']:.0%}", ""),
                     ("R", f"{rep['recall']:.0%}", ""), ("F1", f"{rep['f1']:.2f}", "")])
    return threshold, shown, m, box


def ready(c: case.Case) -> bool:
    """The loaded detector already ran and can meet the brief, so nothing else needs charge.
    If it can't meet the brief, the picker keeps the lobby charger in reach for a bigger one."""
    s = st.session_state
    model = s.get("l3_model")
    return model in s.get("l3_ran", []) and can_meet(scene_detections(model), c.brief)


def render() -> None:
    try:
        runtime.benchmark()
    except ModelMissing as e:
        st.warning(str(e))
        return
    if levels.show_cleared(3):
        levels.level_cleared(3, cleared())
        return
    levels.review_banner(3)
    s = st.session_state
    c = case.get_case(s)
    solved = 3 in s.completed_levels
    device.unload(s, "task")  # the classifier from chapter 2 isn't needed any more

    if not solved and s.get("l3_mode") != "Detect":
        ui.title_card("CHAPTER 3 · PARLOUR · CAM 05 · 04:12", "Find the Anomalies",
                      "The lift went back up to the antique parlour at 04:05. The cleaner swears the tea set was "
                      "rearranged overnight, but she can't say what moved. Someone needs a list of every object in "
                      "the room and where it is, to compare with last week's photos.")
    if not solved and not flow.mode_choice(
            3, "Lots of objects in one frame. Find out what each one is and where it is.",
            mode_options(), "Detect", needs="what each object is and where it is", scene=draw_parlour):
        return

    ui.objective(OBJECTIVES[c.brief])
    view, scanner = ui.stage("l3")
    with scanner:
        ui.scanner_head("GHOSTLENS MK.II · DETECT", pct(s.battery))
        model = flow.model_picker(3, profiles(), LIMITS, slot="watchdog", locked=solved, ready=ready(c))
    if model is None:
        with view:
            draw_parlour()
        return

    ran = s.setdefault("l3_ran", [])
    if model["id"] not in ran:
        with scanner:
            if flow.run_button(3, f"Run {model['name']} on the parlour", f"{model['name']} on the parlour",
                               model["latency_ms"], model["tier"], key=f"l3_run_{model['id']}"):
                ran.append(model["id"])
                st.rerun()
        with view:
            draw_parlour()
        return

    dets = scene_detections(model["id"])
    with scanner:
        threshold, shown, m, box = scanner_controls(c, model, dets, solved)
    with view:
        img = det.draw(load_scene(), m) if m is not None else det.draw(load_scene(), dets=shown)
        if solved:
            img = with_moved(img, TRUTH["objects"][c.moved]["box"])
        caption = ("EVIDENCE 05-C · GREEN = CORRECT · RED = FALSE ALARM · DASHED AMBER = MISSED" if m is not None
                   else f"EVIDENCE 05-C · GHOSTLENS DETECT MODE · {model['name'].upper()} · BOXES ≥ {threshold:.2f}")
        ui.evidence(rgb(with_inspected(img, box)), caption + (" · SOLID AMBER = MOVED" if solved else ""))

    if solved:
        return
    if m is not None and not can_meet(dets, c.brief):
        ui.feedback(cant_meet_line(dets, model), "bad")
    elif m is not None:
        ui.feedback(*consequence(c.brief, m, det.scores(m)))
    elif last_report(model["id"]):
        ui.feedback("Threshold moved since your last report. Submit again to score it.", "")
