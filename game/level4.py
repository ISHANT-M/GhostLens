"""Chapter 4: The Corrupted Region (segmentation, morphology, model size and quantization)."""

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
from game import case, device, flow, levels, runtime
from game.device import pct
from game.scoring import efficient
from ui import components as ui

LIMITS = {"latency_ms": 40, "iou": 0.95}
SIDE_THRESHOLD = 0.5                  # the second wall is scored at the default threshold, untuned
KERNELS = [0, 3, 5, 7, 9]             # morphology kernel sizes, 0 = off
VIEWS = ["Mask", "Probability", "Box", "INT8 diff"]
BOX_COLOR = (47, 134, 183)            # amber, BGR
MASK_COLOR = np.array([90, 125, 94])  # green, BGR
DIFF_COLOR = (102, 122, 224)          # red, BGR of #E07A66
KEEP = {"l4_thr": 0.5, "l4_open": 0, "l4_close": 0, "l4_view": "Mask"}   # widget values kept across pages


def amber_lut() -> np.ndarray:
    """256-entry BGR look-up table: near black, through brass, to pale parchment. No rainbow."""
    stops = [(0, (10, 11, 10)), (128, (100, 164, 200)), (255, (188, 226, 240))]
    lut = np.zeros((256, 1, 3), np.uint8)
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        for i in range(a, b + 1):
            t = (i - a) / (b - a)
            lut[i, 0] = [round(x + (y - x) * t) for x, y in zip(ca, cb)]
    return lut


LUT = amber_lut()


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
    box = seg.bounding_box(mask)
    if box:
        cv2.rectangle(out, box[:2], box[2:], BOX_COLOR, 2)
    return out


def with_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    out[mask] = out[mask] * 0.45 + MASK_COLOR * 0.55
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = out.astype(np.uint8)
    cv2.drawContours(out, contours, -1, (40, 60, 40), 1)
    return out


def probability_view(probs: np.ndarray) -> np.ndarray:
    return cv2.LUT(cv2.cvtColor((probs * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), LUT)


def diff_view(img: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, int]:
    """The wall dimmed, with every pixel where the two masks disagree in red. Returns the image and the count."""
    xor = a.astype(bool) ^ b.astype(bool)
    out = (img * 0.45).astype(np.uint8)
    out[xor] = DIFF_COLOR
    return out, int(xor.sum())


def pixel_accuracy(mask: np.ndarray, truth: np.ndarray) -> float:
    return float((mask.astype(bool) == truth.astype(bool)).mean())


# wrong modes: they really run, and only their picture goes on the stage

def run_classifier() -> tuple[str, float]:
    st.session_state.l4_cls = clf.classify(runtime.yolo("yolo26n-cls.pt"), wall()[0])
    return "YOLO26n-cls on the wall", runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"]


def show_classifier() -> None:
    label, p = st.session_state.l4_cls["top"][0]
    ui.evidence(rgb(wall()[0]), f"EVIDENCE 05-W · GHOSTLENS SAYS: {label.upper()} {p:.0%}")


def classifier_line() -> str:
    top = st.session_state.get("l4_cls")
    said = f"'{top['top'][0][0]}'" if top else "one label"
    return f"It said {said}. No stain class, and no pixels to purify."


def run_detector() -> tuple[str, float]:
    st.session_state.l4_dets = det.detect(runtime.yolo("yolo26s.pt"), wall()[0], min_conf=0.25)[0]
    return "YOLO26s on the wall", runtime.profile("detectors", "yolo26s.pt")["latency_ms"]


def show_detector() -> None:
    img, truth = wall()
    ui.evidence(rgb(with_box(img, truth)), "EVIDENCE 05-W · THE BEST BOX A STAIN DETECTOR COULD GIVE")


def detector_line() -> str:
    found = len(st.session_state.get("l4_dets", []))
    return (f"{found} COCO objects, no stain. Even a perfect box is "
            f"{seg.clean_share_of_box(wall()[1]):.0%} healthy wall.")


def mode_options() -> dict:
    return {
        "Classify": {
            "blurb": "One label, whole image",
            "line": classifier_line(),
            "run": run_classifier, "show": show_classifier,
            "tier": "Light", "latency_ms": runtime.profile("classifiers", "yolo26n-cls.pt")["latency_ms"],
            "verdict": "Classification can say 'there's a stain'. Purification needs to know which pixels.",
        },
        "Detect": {
            "blurb": "A box around each object",
            "line": detector_line(),
            "run": run_detector, "show": show_detector,
            "tier": "Balanced", "latency_ms": runtime.profile("detectors", "yolo26s.pt")["latency_ms"],
            "verdict": "A box is a rough location. The stain is irregular, so most of its box is healthy wall. "
                       "You need a pixel-level boundary.",
        },
        "Segment": {
            "blurb": "Stain or wall, every pixel",
            "line": "",
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


def purify_checks(model: dict, iou: float, threshold: float, cleaned: bool = False) -> dict[str, tuple[bool, str]]:
    """Every limit the purification needs, with a measured detail for each."""
    s = st.session_state
    model_ok, wall_ok = model["iou"] >= LIMITS["iou"], iou >= LIMITS["iou"]
    fast = model["latency_ms"] <= LIMITS["latency_ms"]
    used = device.memory_used(s)
    hint = " Try a mask threshold nearer 0.5." if model_ok and not wall_ok else ""
    after = " after cleanup" if cleaned else ""
    return {
        f"Model IoU ≥ {LIMITS['iou']} (150 walls)": (model_ok, f"{model['iou']:.3f}" + (
            "" if model_ok else ". Its masks leak onto healthy wall and miss thin tendrils, so it can't be trusted "
                                "on the next wall either. Cleanup on one wall can't fix that.")),
        f"This wall IoU ≥ {LIMITS['iou']}": (wall_ok, f"On this wall the mask at threshold {threshold:.2f}{after} "
                                             f"reaches {iou:.3f} IoU.{hint}"),
        f"Latency ≤ {LIMITS['latency_ms']} ms": (fast, f"{model['latency_ms']:.0f} ms per frame" + (
            "" if fast else ". By the time the mask is ready, the stain has moved on.")),
        "Fits in model memory": (used <= device.MEMORY_MB, f"{model['size_mb']:.2f} MB, {used:.1f} / "
                                                          f"{device.MEMORY_MB:.0f} MB in use"),
    }


def failure_line(checks: dict) -> str:
    failed = [name for name, (ok, _) in checks.items() if not ok]
    return f"<b>Purification failed:</b> {', '.join(failed)}. Try a different model, or quantize this one."


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
        ui.evidence(rgb(img), f"WALL #{seed} · AS FOUND")
    with c2:
        ui.evidence(rgb(with_mask(img, mask)), f"WALL #{seed} · {deployed['name'].upper()}")
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


def purify(model: dict, iou: float, raw_iou: float, threshold: float, open_k: int, close_k: int) -> dict | None:
    """Returns the failed checks, or finishes the chapter."""
    s = st.session_state
    s.l4_purify_tries = s.get("l4_purify_tries", 0) + 1
    checks = purify_checks(model, iou, threshold, cleaned=bool(open_k or close_k))
    if not all(ok for ok, _ in checks.values()):
        return checks
    s.l4_deployed = {"id": model["id"], "name": model["name"], "variant": model["variant"],
                     "int8": model["precision"] == "INT8", "iou": iou, "raw_iou": raw_iou,
                     "open_k": open_k, "close_k": close_k, "threshold": threshold}
    levels.finish_level(4, iou, s.l4_purify_tries, report_checks(model), clue="Corruption purified")
    return None


# the cleared screen

def frontier_rows(bench: dict) -> list[dict]:
    return [{"name": f"{r['name']} {r['precision']}", "variant": r["variant"], "precision": r["precision"],
             "latency_ms": r["latency_ms"], "iou": r["iou"], "size_mb": r["size_mb"],
             "meets": r["latency_ms"] <= LIMITS["latency_ms"] and r["iou"] >= LIMITS["iou"]}
            for r in bench["unets"]]


def frontier_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    lo = min(df.iou.min(), LIMITS["iou"]) - 0.01
    top = df.iou.max() + 0.005
    box = alt.Chart(pd.DataFrame({"x": [0], "x2": [LIMITS["latency_ms"]], "y": [LIMITS["iou"]], "y2": [top]})).mark_rect(
        color=ui.CHART[1], opacity=0.14).encode(x="x:Q", x2="x2:Q", y="y:Q", y2="y2:Q")
    x = alt.X("latency_ms:Q", title="latency, ms (measured)", scale=alt.Scale(domainMin=0))
    y = alt.Y("iou:Q", title="mask IoU, 150 walls", scale=alt.Scale(domain=[lo, top]))
    arrows = alt.Chart(df).mark_line(color=ui.CHART[4], strokeDash=[4, 3]).encode(x=x, y=y, detail="variant:N")
    colors = alt.Scale(domain=["FP32", "INT8"], range=[ui.CHART[4], ui.CHART[0]])
    points = alt.Chart(df).mark_point(filled=True, size=80).encode(
        x=x, y=y, color=alt.Color("precision:N", scale=colors, legend=alt.Legend(orient="top", title=None)),
        tooltip=["name:N", "latency_ms:Q", "iou:Q", "size_mb:Q"])
    labels = alt.Chart(df).mark_text(align="left", dx=7, dy=-6, fontSize=11, color=ui.CHART[4]).encode(
        x=x, y=y, text="name:N")
    return (box + arrows + points + labels).properties(height=280, title="INT8 moves left, not down")


@st.cache_data(show_spinner=False)
def wall_table(seed: int) -> pd.DataFrame:
    truth = scene(seed)[1]
    rows = [{"model": f"{r['name']} {r['precision']}", "IoU on this wall": round(seg.mask_iou(
        wall_probs(r["variant"], r["precision"] == "INT8", seed) > 0.5, truth), 3), "IoU, 150 walls": r["iou"],
        "latency ms": r["latency_ms"]} for r in runtime.benchmark()["unets"]]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def cleanup_table(variant: str, int8: bool, seed: int, threshold: float) -> pd.DataFrame:
    """This wall's IoU for the deployed model at every opening and closing size, one at a time."""
    truth = scene(seed)[1]
    raw = wall_probs(variant, int8, seed) > threshold
    rows = [{"cleanup": "none", "IoU on this wall": round(seg.mask_iou(raw, truth), 3)}]
    for k in KERNELS[1:]:
        rows.append({"cleanup": f"open {k}×{k}", "IoU on this wall": round(seg.mask_iou(seg.clean_mask(raw, k, 0), truth), 3)})
        rows.append({"cleanup": f"close {k}×{k}", "IoU on this wall": round(seg.mask_iou(seg.clean_mask(raw, 0, k), truth), 3)})
    return pd.DataFrame(rows)


def same_network_line() -> str:
    cls, det_, segm = (runtime.profile(g, i)["latency_ms"] for g, i in
                       (("classifiers", "yolo26s-cls.pt"), ("detectors", "yolo26s.pt"), ("segmenters", "yolo26s-seg.pt")))
    std = runtime.profile("unets", "standard-fp32")
    return (f"Same network size, three tasks (YOLO26s): {cls:.1f} ms to classify, {det_:.1f} ms to detect, "
            f"{segm:.1f} ms to segment. The U-Net is cheap only because it is tiny ({std['params_k']:.0f}k parameters).")


def cleanup_line(d: dict) -> str:
    if not (d.get("open_k") or d.get("close_k")):
        return f"No cleanup: the raw mask at threshold {d.get('threshold', 0.5):.2f} scored {d['iou']:.3f} IoU."
    steps = [f"opening {d['open_k']}×{d['open_k']}" if d.get("open_k") else "",
             f"closing {d['close_k']}×{d['close_k']}" if d.get("close_k") else ""]
    return (f"Cleanup with {' then '.join(x for x in steps if x)} moved this wall's IoU from {d['raw_iou']:.3f} "
            f"to {d['iou']:.3f}. It can't change the model's IoU on the 150 test walls.")


def numbers() -> None:
    s = st.session_state
    d = s.l4_deployed
    seed = case.get_case(s).wall_seed
    rows = frontier_rows(runtime.benchmark())
    ui.chart(frontier_chart(rows))
    st.caption(f"{ui.tag('measured')} latency and IoU from setup_models.py · shaded: ≤ {LIMITS['latency_ms']} ms "
               f"and IoU ≥ {LIMITS['iou']} {ui.tag('gameplay')}", unsafe_allow_html=True)
    st.markdown(f'<div class="gl-kicker">Every model on wall #{seed}, threshold 0.5</div>', unsafe_allow_html=True)
    st.dataframe(wall_table(seed), hide_index=True, width="stretch")
    truth = scene(seed)[1]
    lines = [same_network_line(), levels.int8_note(), cleanup_line(d),
             f"Pixel accuracy vs IoU: an empty mask already scores {1 - truth.mean():.1%} pixel accuracy on this "
             "wall, because most pixels are healthy wall. Its IoU is 0. That's why the limit is on IoU."]
    st.markdown("\n".join(f"- {x}" for x in lines))
    st.markdown(f'<div class="gl-kicker">Cleanup effect · {d["name"]} at threshold {d.get("threshold", 0.5):.2f}</div>',
                unsafe_allow_html=True)
    st.dataframe(cleanup_table(d["variant"], d["int8"], seed, d.get("threshold", 0.5)), hide_index=True,
                 width="stretch")
    st.caption(f"{ui.tag('measured')} The painted stains have thin tendrils. Opening cuts them off and closing "
               "fills the gaps between them, so on a mask this clean, cleanup costs IoU.", unsafe_allow_html=True)
    inside = ", ".join(r["name"] for r in rows if r["meets"]) or "none"
    with st.expander("The maths"):
        st.markdown(
            "**IoU** = |predicted ∩ true| / |predicted ∪ true|, counted in pixels. It ignores the healthy wall both "
            "masks agree on.\n\n"
            "**Pixel accuracy** = correct pixels / all pixels, so the healthy wall counts too.\n\n"
            "**Opening** = erosion then dilation with the same kernel: specks smaller than the kernel vanish. "
            "**Closing** = dilation then erosion: gaps smaller than the kernel fill in.\n\n"
            "**INT8 post-training quantization** stores each weight as an 8-bit integer plus a scale, calibrated on "
            "a few walls: about 4× smaller, and integer kernels run faster.\n\n"
            f"Inside the box: {inside}. On a device, the best model is the one that meets every limit, not the most "
            "accurate one.")


def explore() -> None:
    from game import lab
    st.markdown("Does cutting weights do the same as INT8? The Lab's pruning bench zeros the smallest weights and "
                "times the result.")
    if st.button("Lab · Pruning", key="l4_lab_pruning"):
        lab.open_lab("Pruning")


def happened() -> list[str]:
    s = st.session_state
    d = s.l4_deployed
    lines = flow.wrong_mode_lines(4, mode_options())
    ran = [runtime.profile("unets", i)["name"] + (" INT8" if i.endswith("int8") else " FP32") for i in s.get("l4_ran", [])]
    if ran:
        lines.append(f"You ran {', '.join(ran)} on the wall.")
    tries = s.get("l4_purify_tries", 1)
    if tries > 1:
        lines.append(f"{tries - 1} purification attempt{'s' if tries > 2 else ''} failed a limit before this one.")
    lines.append(f"You deployed {d['name']} at threshold {d.get('threshold', 0.5):.2f}. " + cleanup_line(d))
    return lines


def cleared() -> "levels.Cleared":
    s = st.session_state
    d = s.l4_deployed
    seed = case.get_case(s).wall_seed
    prof = runtime.profile("unets", d["id"])
    smaller, faster, _ = levels.int8_ratios(runtime.benchmark())

    def evidence() -> None:
        img, _ = wall()
        mask = seg.clean_mask(wall_probs(d["variant"], d["int8"], seed) > d.get("threshold", 0.5),
                              d.get("open_k", 0), d.get("close_k", 0))
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.evidence(rgb(img), f"WALL #{seed} · ORIGINAL")
        with c2:
            ui.evidence(rgb(with_mask(img, mask)), f"MASK · {d['name'].upper()}")
        with c3:
            ui.evidence(rgb(with_box(img, mask)), f"BOX · {seg.clean_share_of_box(mask):.0%} HEALTHY WALL")

    why = [
        "Purifying needs the exact stain pixels, so this was segmentation, not a box or a label.",
        f"U-Net Standard was accurate enough on the 150 test walls, but only INT8 made it fast enough: "
        f"{smaller:.1f}× smaller and {faster:.1f}× faster for almost no IoU.",
        "U-Net Lite was fast but leaked onto healthy wall; U-Net Pro was accurate but too slow.",
        "IoU, not pixel accuracy: most of the wall is healthy, so pixel accuracy flatters any mask.",
    ]
    return levels.Cleared(
        headline=f"Corruption purified: {d['name']}, IoU {d['iou']:.3f}, {prof['latency_ms']:.1f} ms.",
        happened=happened(), why=why,
        concept="Segmentation · IoU vs pixel accuracy · morphology · INT8 PTQ · model selection under limits",
        evidence=evidence, numbers=numbers, explore=explore, side=second_wall_scan(),
        stats=[("Model deployed", d["name"]), ("Mask IoU", f"{d['iou']:.3f}")], par=par())


# the play screen

def keep_widgets() -> None:
    # widget values are dropped when another page runs, so a plain copy brings them back
    s = st.session_state
    for k, v in s.setdefault("l4_keep", dict(KEEP)).items():
        s.setdefault(k, v)


def save_widgets() -> None:
    s = st.session_state
    s.l4_keep = {k: s[k] for k in KEEP if k in s}


def draw_wall() -> None:
    img, _ = wall()
    ui.evidence(rgb(img), f"EVIDENCE 05-W · WALL #{case.get_case(st.session_state).wall_seed}")


def other_precision(model: dict) -> str:
    return f"{model['variant']}-{'fp32' if model['precision'] == 'INT8' else 'int8'}"


def stage_view(model: dict, probs: np.ndarray, mask: np.ndarray, threshold: float, open_k: int, close_k: int) -> None:
    s = st.session_state
    img, _ = wall()
    seed = case.get_case(s).wall_seed
    both = other_precision(model) in s.get("l4_ran", [])
    views = VIEWS if both else VIEWS[:3]
    if s.get("l4_view") not in views:
        s.l4_view = "Mask"
    view = st.radio("View", views, horizontal=True, key="l4_view", label_visibility="collapsed")
    if view == "Probability":
        ui.evidence(rgb(probability_view(probs)), f"PROBABILITY · {model['name'].upper()} · DARK 0 → BRASS 0.5 → PALE 1")
    elif view == "Box":
        ui.evidence(rgb(with_box(img, mask)),
                    f"BOX AROUND THE MASK · {seg.clean_share_of_box(mask):.0%} OF IT IS HEALTHY WALL")
    elif view == "INT8 diff":
        other = wall_probs(model["variant"], model["precision"] == "FP32", seed) > threshold
        other = seg.clean_mask(other, open_k, close_k)
        out, n = diff_view(img, mask, other)
        ui.evidence(rgb(out), f"FP32 VS INT8 · {n} PX DIFFER ({n / mask.size:.1%})")
    else:
        ui.evidence(rgb(with_mask(img, mask)), f"WALL #{seed} · SEGMENTATION · {model['name'].upper()}")


def render() -> None:
    try:
        runtime.benchmark()
        img, truth = wall()
    except ModelMissing as e:
        st.warning(str(e))
        return
    if levels.show_cleared(4):
        levels.level_cleared(4, cleared())
        return
    levels.review_banner(4)
    s = st.session_state
    seed = case.get_case(s).wall_seed
    solved = 4 in s.completed_levels

    if not solved and s.get("l4_mode") != "Segment":
        ui.title_card("CHAPTER 4 · PARLOUR WALL · 04:44", "The Corrupted Region",
                      "The watchdog flagged it at 04:40: a stain spreading over the wall between the curtains, a "
                      "little bigger every minute. The hotel can purify it, but whatever you mark gets scraped off, "
                      "so mark too much and you destroy healthy wall and whatever's written under the paint.")
    if not solved and not flow.mode_choice(
            4, "Find the exact shape of the stain, so only the corrupted pixels get purified.",
            mode_options(), "Segment", needs="the exact pixels of the stain", scene=draw_wall):
        return

    ui.objective(f"Mark only the stain · IoU ≥ {LIMITS['iou']} · ≤ {LIMITS['latency_ms']} ms")
    keep_widgets()
    view, scanner = ui.stage("l4")
    with scanner:
        ui.scanner_head("GHOSTLENS MK.II · SEGMENT", pct(s.battery))
        s.setdefault("l4_int8", str(s.get("l4_model", "")).endswith("-int8"))   # match the loaded model on a revisit
        if solved:   # review is read-only: keep the toggle on the deployed model
            s.l4_int8 = str(s.get("l4_model", "")).endswith("-int8")
        int8 = st.toggle("Quantize to INT8", key="l4_int8", disabled=solved,
                         help="Post-training quantization: 8-bit weights and activations instead of 32-bit floats.")
        model = flow.model_picker(4, profiles(int8), LIMITS, slot="task", locked=solved)
    if model is None:
        with view:
            draw_wall()
        return

    ran = s.setdefault("l4_ran", [])
    if model["id"] not in ran:
        with scanner:
            if flow.run_button(4, f"Run {model['name']} on the wall", f"{model['name']} on wall #{seed}",
                               model["latency_ms"], model["tier"], key=f"l4_run_{model['id']}"):
                ran.append(model["id"])
                st.rerun()
        with view:
            draw_wall()
        return

    probs = wall_probs(model["variant"], model["precision"] == "INT8", seed)
    with scanner:
        threshold = st.slider("Mask threshold", 0.1, 0.9, step=0.05, key="l4_thr",
                              help="A pixel counts as stain if the model's probability is above this.")
        st.markdown('<div class="gl-kicker">Cleanup · morphology</div>', unsafe_allow_html=True)
        fmt = {0: "off"} | {k: f"{k}×{k}" for k in KERNELS[1:]}
        open_k = st.select_slider("Open · remove specks", KERNELS, key="l4_open", format_func=fmt.get)
        close_k = st.select_slider("Close · fill gaps", KERNELS, key="l4_close", format_func=fmt.get)
        raw = probs > threshold
        mask = seg.clean_mask(raw, open_k, close_k)
        raw_iou, iou = seg.mask_iou(raw, truth), seg.mask_iou(mask, truth)
        ui.readouts([
            ("Stain area", f"{mask.mean():.1%}", ""),
            ("Wall IoU raw → clean", f"{raw_iou:.3f} → {iou:.3f}", "ok" if iou >= LIMITS["iou"] else "warn"),
            ("Model IoU (150 walls)", f"{model['iou']:.3f}", "ok" if model["iou"] >= LIMITS["iou"] else "bad"),
            ("Latency", f"{model['latency_ms']:.0f} ms", "ok" if model["latency_ms"] <= LIMITS["latency_ms"] else "bad"),
        ])
        failed = None
        if not solved and st.button("Purify the marked pixels", type="primary", width="stretch", key="l4_purify"):
            failed = purify(model, iou, raw_iou, threshold, open_k, close_k)
    save_widgets()
    with view:
        stage_view(model, probs, mask, threshold, open_k, close_k)
    if failed:
        ui.feedback(failure_line(failed), "bad")
        flow.checklist("Purification checks", failed)
