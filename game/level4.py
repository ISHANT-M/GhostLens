"""Chapter 4: The Corrupted Region (segmentation, model size and quantization)."""

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import classification as clf
from cv import detection as det
from cv import segmentation as seg
from cv.edge import energy_units
from cv.models import ModelMissing
from game import case, device, flow, runtime
from game.device import pct
from game.levels import case_closed, completion_panel, finish_level, int8_note, int8_ratios
from game.scoring import efficient
from ui import components as ui

LIMITS = {"latency_ms": 40, "iou": 0.95}
SIDE_THRESHOLD = 0.5                  # the second wall is scored at the default threshold, untuned
BOX_COLOR = (47, 134, 183)            # amber, BGR
MASK_COLOR = np.array([90, 125, 94])  # green, BGR
INT8_OPTIONS = ["Drops by more than 0.05", "Drops by 0.01 to 0.05", "Changes by less than 0.01",
                "Improves by more than 0.01"]


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
            "blurb": "A label for every pixel: stain or wall.",
            "verdict": "Exactly which pixels are corrupted is a segmentation question. For the same network size "
                       "it's the costliest task, so now the model has to fit the device.",
        },
    }


def warm() -> None:
    wall()
    runtime.unet("standard", False)


WARMUP = [("Reading camera 05", wall), ("Measuring the models", runtime.benchmark), ("Warming up GhostLens", warm)]


def profiles(int8: bool) -> list[dict]:
    precision = "INT8" if int8 else "FP32"
    tiers = {"lite": "Light", "standard": "Balanced", "pro": "Heavy"}
    return [{**r, "tier": tiers[r["variant"]], "name": f"{r['name']} {precision}",
             "accuracy": f"{r['iou']:.3f}", "accuracy_tag": "measured", "metric": "IoU, 150 walls"}
            for r in runtime.benchmark()["unets"] if r["precision"] == precision]


# prediction: what does INT8 do to the mask?

def int8_bin(delta: float) -> int:
    """Index into INT8_OPTIONS for an IoU change (INT8 minus FP32)."""
    if delta < -0.05:
        return 0
    if delta <= -0.01:
        return 1
    return 2 if delta <= 0.01 else 3


def resolve_int8() -> tuple[int, str]:
    fp, q = runtime.profile("unets", "standard-fp32"), runtime.profile("unets", "standard-int8")
    smaller, faster, delta = int8_ratios(runtime.benchmark())
    return int8_bin(delta), (
        f"U-Net Standard went from {fp['iou']:.3f} to {q['iou']:.3f} IoU on 150 test walls ({delta:+.3f}), and got "
        f"{smaller:.1f}× smaller and {faster:.1f}× faster. Eight bits are plenty to say stain or wall.")


def int8_reveal() -> None:
    fp, q = runtime.profile("unets", "standard-fp32"), runtime.profile("unets", "standard-int8")
    ui.readouts([("Size", f"{fp['size_mb']:.2f} → {q['size_mb']:.2f} MB", "ok"),
                 ("Latency", f"{fp['latency_ms']:.0f} → {q['latency_ms']:.0f} ms", "ok"),
                 ("IoU, 150 walls", f"{fp['iou']:.3f} → {q['iou']:.3f}", "")])


def int8_prediction() -> None:
    s = st.session_state
    tried = any(str(m).endswith("-int8") for m in s.get("l4_ran", []))
    if s.get("l4_int8") or tried:
        s.l4_toggled = True
    if "l4_pred_int8" not in s and s.get("l4_toggled"):
        return
    flow.predict("l4_pred_int8", "Before you flip the switch: quantize U-Net Standard to INT8. What happens to its "
                 "mask IoU on the 150 test walls?", INT8_OPTIONS, resolve=resolve_int8, reveal=int8_reveal,
                 ready=bool(s.get("l4_toggled")))


def pixel_accuracy(mask: np.ndarray, truth: np.ndarray) -> float:
    return float((mask.astype(bool) == truth.astype(bool)).mean())


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


def purify_checks(model: dict, iou: float, threshold: float) -> dict[str, tuple[bool, str]]:
    """Every limit the purification needs, with a measured detail for each."""
    s = st.session_state
    model_ok, wall_ok = model["iou"] >= LIMITS["iou"], iou >= LIMITS["iou"]
    fast = model["latency_ms"] <= LIMITS["latency_ms"]
    used = device.memory_used(s)
    hint = " Try a mask threshold nearer 0.5." if model_ok and not wall_ok else ""
    return {
        f"Model IoU ≥ {LIMITS['iou']} (150 walls)": (model_ok, f"{model['iou']:.3f}" + (
            "" if model_ok else ". Its masks leak onto healthy wall and miss thin tendrils, so it can't be trusted "
                                "on the next wall either.")),
        f"This wall IoU ≥ {LIMITS['iou']}": (wall_ok, f"On this wall the mask at threshold {threshold:.2f} "
                                             f"reaches {iou:.3f} IoU.{hint}"),
        f"Latency ≤ {LIMITS['latency_ms']} ms": (fast, f"{model['latency_ms']:.0f} ms per frame" + (
            "" if fast else ". By the time the mask is ready, the stain has moved on.")),
        "Fits in model memory": (used <= device.MEMORY_MB, f"{model['size_mb']:.2f} MB, {used:.1f} / "
                                                          f"{device.MEMORY_MB:.0f} MB in use"),
    }


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
    checks = purify_checks(model, iou, threshold)
    if not all(ok for ok, _ in checks.values()):
        ui.message("<b>Purification failed.</b> Try a different model, or quantize this one.", "bad")
        flow.checklist("Purification checks", checks)
        return
    s.l4_deployed = {"id": model["id"], "name": model["name"], "variant": model["variant"],
                     "int8": model["precision"] == "INT8", "iou": iou}
    finish_level(4, iou, s.l4_purify_tries, report_checks(model), clue="Corruption purified")


# debrief: every model against the limits

def frontier_rows(bench: dict) -> list[dict]:
    return [{"name": f"{r['name']} {r['precision']}", "variant": r["variant"], "precision": r["precision"],
             "latency_ms": r["latency_ms"], "iou": r["iou"], "size_mb": r["size_mb"],
             "meets": r["latency_ms"] <= LIMITS["latency_ms"] and r["iou"] >= LIMITS["iou"]}
            for r in bench["unets"]]


def frontier_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    lo = min(df.iou.min(), LIMITS["iou"]) - 0.01
    box = alt.Chart(pd.DataFrame({"x": [0], "x2": [LIMITS["latency_ms"]], "y": [LIMITS["iou"]],
                                  "y2": [df.iou.max() + 0.005]})).mark_rect(color=flow.OK, opacity=0.12).encode(
        x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    x = alt.X("latency_ms:Q", title="latency, ms (measured)", scale=alt.Scale(domainMin=0))
    y = alt.Y("iou:Q", title="mask IoU, 150 walls", scale=alt.Scale(domain=[lo, df.iou.max() + 0.005]))
    arrows = alt.Chart(df).mark_line(color=flow.SOFT, strokeDash=[4, 3]).encode(x=x, y=y, detail="variant:N")
    colors = alt.Scale(domain=["FP32", "INT8"], range=[flow.SOFT, flow.INK])
    points = alt.Chart(df).mark_point(filled=True, size=80).encode(
        x=x, y=y, color=alt.Color("precision:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
        tooltip=["name:N", "latency_ms:Q", "iou:Q", "size_mb:Q"])
    labels = alt.Chart(df).mark_text(align="left", dx=7, dy=-6, fontSize=11, color=flow.INK).encode(
        x=x, y=y, text="name:N")
    chart = (box + arrows + points + labels).properties(height=260, title="INT8 moves left, not down")
    return chart.configure_view(stroke=None).configure(background="transparent")


@st.cache_data(show_spinner=False)
def wall_table(seed: int) -> pd.DataFrame:
    truth = scene(seed)[1]
    rows = [{"model": f"{r['name']} {r['precision']}", "IoU on this wall": round(seg.mask_iou(
        wall_probs(r["variant"], r["precision"] == "INT8", seed) > 0.5, truth), 3), "IoU, 150 walls": r["iou"],
        "latency ms": r["latency_ms"]} for r in runtime.benchmark()["unets"]]
    return pd.DataFrame(rows)


def same_network_line() -> str:
    cls, det_, segm = (runtime.profile(g, i)["latency_ms"] for g, i in
                       (("classifiers", "yolo26s-cls.pt"), ("detectors", "yolo26s.pt"), ("segmenters", "yolo26s-seg.pt")))
    std = runtime.profile("unets", "standard-fp32")
    return (f"Same network size, three tasks (YOLO26s): {cls:.1f} ms to classify, {det_:.1f} ms to detect, "
            f"{segm:.1f} ms to segment. The U-Net is cheap only because it is tiny ({std['params_k']:.0f}k parameters).")


def debrief() -> None:
    rows = frontier_rows(runtime.benchmark())
    st.altair_chart(frontier_chart(rows), width="stretch")
    st.caption(f"{ui.tag('measured')} latency and IoU from setup_models.py · shaded: ≤ {LIMITS['latency_ms']} ms "
               f"and IoU ≥ {LIMITS['iou']}", unsafe_allow_html=True)
    with st.expander("Every model on this wall"):
        st.dataframe(wall_table(case.get_case(st.session_state).wall_seed), hide_index=True, width="stretch")
    inside = ", ".join(r["name"] for r in rows if r["meets"]) or "none"
    ui.lesson([int8_note(), same_network_line(),
               f"Inside the box: {inside}. On a device, the best model is the one that meets every limit, "
               "not the most accurate one."], title="DEBRIEF")
    from game import lab
    if hasattr(lab, "open_lab") and st.button("Lab · Pruning: does cutting weights do the same?", type="tertiary"):
        lab.open_lab("Pruning")


def solved_view(model: dict, iou: float) -> None:
    st.divider()
    completion_panel(4, [("Model deployed", model["name"]), ("Mask IoU", f"{iou:.3f}")], par=par(),
                     debrief=debrief)
    scan = second_wall_scan()
    if scan:
        flow.side_scan(4, scan)
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
            mode_options(), "Segment", needs="the exact pixels of the stain"):
        st.image(rgb(img), width=380)
        ui.caption(f"EVIDENCE 05-W · WALL #{seed}")
        return

    st.markdown(f'<div class="gl-mission"><span class="gl-kicker">Mission</span><span>purify only the stain</span>'
                f'<span>mask IoU ≥ {LIMITS["iou"]}</span><span>≤ {LIMITS["latency_ms"]} ms per frame</span></div>',
                unsafe_allow_html=True)
    s.setdefault("l4_int8", str(s.get("l4_model", "")).endswith("-int8"))   # match the loaded model on a revisit
    int8_prediction()
    int8 = st.toggle("Quantize to INT8", key="l4_int8",
                     help="Post-training quantization. The numbers on the cards are measured for both versions.")
    st.caption("Stores weights and activations as 8-bit integers instead of 32-bit floats: smaller and faster.")
    model = flow.model_picker(
        4, profiles(int8), LIMITS, slot="task",
        note="Mask IoU measured on 150 generated walls the model never saw (same generator: in-domain). The "
             "watchdog detector from chapter 3 is still loaded and using memory.")
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
        ("Pixel accuracy", f"{pixel_accuracy(mask, truth):.1%}", ""),
        ("Mask IoU (this wall)", f"{iou:.3f}", "ok" if iou >= LIMITS["iou"] else "warn"),
        ("Model IoU (150 walls)", f"{model['iou']:.3f}", "ok" if model["iou"] >= LIMITS["iou"] else "bad"),
    ])
    st.caption(f"{ui.tag('measured')} We painted the stain onto the wall ourselves, so the true mask is known exactly. "
               f"Most pixels are wall: an empty mask would already score {1 - truth.mean():.1%} pixel accuracy, so "
               "accuracy flatters a mask. Purification needs both IoU readouts at 0.95 or more.",
               unsafe_allow_html=True)

    if solved:
        solved_view(model, iou)
    elif st.button("Purify the marked pixels", type="primary"):
        purify(model, iou, threshold)
