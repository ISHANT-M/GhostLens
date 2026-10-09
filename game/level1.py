"""Chapter 1: The Dark Frame (preprocessing and histograms)."""

from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import detection as det
from cv import enhancement as enh
from cv.edge import energy_units
from game import case, device, flow, runtime
from game.device import pct
from game.levels import completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level1"
CAM03 = (217, "CAM 03  2026-10-07  02:17:44")
CAM04 = (404, "CAM 04  2026-10-07  02:19:12")
COSTLY = 50     # 5% of a full pack: the cost readout turns amber
CLIP_LIMIT = 0.01   # share of pixels you may push to pure black or white
# the prediction: one tool on its own, at the strength a player would try first
SINGLE_TOOLS = {"Brightness +100": {"brightness": 100}, "Contrast ×4": {"contrast": 4.0}, "Gamma 2.4": {"gamma": 2.4}}

DEFAULTS = {
    "l1_gamma": 1.0, "l1_brightness": 0, "l1_contrast": 1.0, "l1_equalizer": "None",
    "l1_clahe_clip": 2.0, "l1_denoise": "None", "l1_denoise_strength": 1, "l1_sharpen": 0.0,
}


@st.cache_data(show_spinner=False)
def frames(source: str, crop: tuple | None, seed: int, stamp: str) -> tuple[np.ndarray, np.ndarray]:
    reference = enh.load_image(ASSETS / source, crop=list(crop) if crop else None)
    return reference, enh.make_dark_frame(reference, seed=seed, stamp=stamp)


def clue_frames(clue: dict, camera: tuple[int, str] = CAM03) -> tuple[np.ndarray, np.ndarray]:
    """(clean photo, dark CCTV frame) for one photo of the pool."""
    return frames(clue["source"], tuple(clue["crop"]) if clue["crop"] else None, *camera)


def load_frames() -> tuple[np.ndarray, np.ndarray]:
    return clue_frames(case.get_case(st.session_state).clue)


def normalise(text: str) -> str:
    return "".join(text.split()).lower()


def is_answer(text: str, clue: dict) -> bool:
    return normalise(text) in {normalise(a) for a in [clue["answer"], *clue["aliases"]]}


def where(box: list[float]) -> str:
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    row = "top" if y < 0.33 else "bottom" if y > 0.66 else "middle"
    col = "left" if x < 0.33 else "right" if x > 0.66 else "centre"
    return "the centre" if (row, col) == ("middle", "centre") else f"the {row} {col}"


def measure(out: np.ndarray, dark: np.ndarray, reference: np.ndarray, box: list[float]) -> tuple[float, float, float]:
    """(legibility, newly clipped share, evidence quality) of an enhanced frame."""
    black0, white0 = enh.clipped(dark)
    black, white = enh.clipped(out)
    new_clip = max(0.0, (black + white) - (black0 + white0))
    legib = enh.legibility(out, reference, box)
    return legib, new_clip, enh.evidence_quality(legib, new_clip, weight=3)


def reference_units(dark: np.ndarray) -> int:
    """Par for the chapter: one pass with the reference settings, timed live."""
    _, timings = enh.run_pipeline(dark, enh.Settings(**case.REFERENCE))
    return energy_units(sum(ms for _, ms in timings))


def clip_tone(new_clip: float) -> str:
    if new_clip > 3 * CLIP_LIMIT:
        return "bad"
    return "warn" if new_clip > CLIP_LIMIT else ""


def best_tools(qualities: dict[str, float]) -> set[int]:
    """Indices of the tools with the highest measured quality (ties all count)."""
    top = max(qualities.values())
    return {i for i, q in enumerate(qualities.values()) if abs(q - top) < 1e-9}


@st.cache_data(show_spinner=False)
def single_tool_quality(clue_id: str) -> dict[str, float]:
    clue = next(c for c in case.CLUES if c["id"] == clue_id)
    reference, dark = clue_frames(clue)
    return {name: measure(enh.run_pipeline(dark, enh.Settings(**kw))[0], dark, reference, clue["clue_box"])[2]
            for name, kw in SINGLE_TOOLS.items()}


def tone(v: float, good: float, ok: float) -> str:
    if v >= good:
        return "ok"
    return "warn" if v >= ok else "bad"


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def reset_lab() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v


def controls() -> enh.Settings:
    # widget values are dropped when another page runs, so a plain copy brings them back
    s = st.session_state
    for k, v in s.setdefault("l1_lab", dict(DEFAULTS)).items():
        s.setdefault(k, v)

    st.markdown('<div class="gl-kicker">Tone</div>', unsafe_allow_html=True)
    gamma = st.slider("Gamma", 0.5, 4.0, step=0.1, key="l1_gamma",
                      help="Above 1 lifts dark tones much more than bright ones.")
    brightness = st.slider("Brightness", -50, 150, step=5, key="l1_brightness",
                           help="Adds the same value to every pixel.")
    contrast = st.slider("Contrast", 0.5, 20.0, step=0.5, key="l1_contrast",
                         help="Multiplies every pixel. Spreads values apart, and pushes some past 255.")

    st.markdown('<div class="gl-kicker">Histogram tools</div>', unsafe_allow_html=True)
    equalizer = st.radio("Equalization", ["None", "Global equalization", "CLAHE"], key="l1_equalizer",
                         horizontal=True, label_visibility="collapsed")
    clahe_clip = 2.0
    if equalizer == "CLAHE":
        clahe_clip = st.slider("CLAHE clip limit", 1.0, 8.0, step=0.5, key="l1_clahe_clip")

    st.markdown('<div class="gl-kicker">Noise and detail</div>', unsafe_allow_html=True)
    method = st.selectbox("Denoise", ["None", "Gaussian blur", "Median", "Non-local means"], key="l1_denoise")
    strength = 1
    if method != "None":
        strength = st.slider("Denoise strength", 1, 6, key="l1_denoise_strength")
    sharpen = st.slider("Sharpen", 0.0, 4.0, step=0.25, key="l1_sharpen")

    st.button("Reset lab", on_click=reset_lab, type="tertiary")
    s.l1_lab = {k: s[k] for k in DEFAULTS}
    return enh.Settings(method, strength, brightness, contrast, gamma, equalizer, clahe_clip, sharpen)


def histogram_chart(before: np.ndarray, after: np.ndarray) -> alt.Chart:
    df = pd.DataFrame({
        "level": np.tile(np.arange(256), 2),
        "share": np.concatenate([enh.histogram(before), enh.histogram(after)]),
        "frame": ["Dark frame"] * 256 + ["Enhanced"] * 256,
    })
    # the clip metric counts exactly 0 and 255, so mark just those two levels
    clip_zones = pd.DataFrame({"x0": [0, 254], "x1": [1, 255]})
    zones = alt.Chart(clip_zones).mark_rect(color="#9C4A3C", opacity=0.18).encode(x="x0:Q", x2="x1:Q")
    areas = alt.Chart(df).mark_area(opacity=0.55, interpolate="step").encode(
        x=alt.X("level:Q", title="pixel brightness (0 = black, 255 = white)", scale=alt.Scale(domain=[0, 255], nice=False)),
        y=alt.Y("share:Q", title="share of pixels (sqrt scale)", scale=alt.Scale(type="sqrt"), axis=alt.Axis(format="%")),
        color=alt.Color("frame:N", scale=alt.Scale(domain=["Dark frame", "Enhanced"], range=["#9C978C", "#22211F"]),
                        legend=alt.Legend(orient="top", title=None)),
    )
    return (zones + areas).properties(height=210).configure_view(stroke=None).configure(background="transparent")


def clip_message(new_clip: float) -> None:
    ui.message(f"<b>{new_clip:.1%} of the frame is now pure black or pure white</b>, over the {CLIP_LIMIT:.1%} "
               "limit. Those pixels all have the same value, so whatever detail they had is gone for good. "
               "Ease off the contrast or brightness.", clip_tone(new_clip))


def feedback(s: enh.Settings, legib: float, new_clip: float, quality: float, units: int, box: list[float],
             solved: bool = False) -> None:
    if clip_tone(new_clip) == "bad":
        clip_message(new_clip)
    elif s.denoise_method == "Non-local means":
        ui.message(f"<b>Non-local means costs about {pct(units)} of battery per pass.</b> It compares patches across "
                   "the whole frame, so it is hundreds of times slower than a lookup table. Gamma, contrast and CLAHE "
                   f"cost about {pct(1)} and are already enough to read this plate. Expensive isn't automatically "
                   "better.", "warn")
    elif new_clip > CLIP_LIMIT:
        clip_message(new_clip)
    elif s.denoise_method == "Gaussian blur" and s.denoise_strength >= 3:
        ui.message("The blur is cleaning up the grain, but it's smearing the digits too. Noise and fine detail are "
                   "both small, fast changes, and a blur can't tell them apart.", "warn")
    elif s.sharpen >= 1.5:
        ui.message("Sharpening is making the grain louder. It boosts every small change in the image, "
                   "and in a night frame most of those changes are noise.", "warn")
    elif s.brightness > 0 and s.contrast == 1.0 and s.gamma == 1.0 and legib < 0.3:
        ui.message("The frame got lighter but the number didn't get clearer. Brightness adds the same amount to "
                   "every pixel, so the gap between the digits and the plate stays exactly as small.", "warn")
    elif quality >= case.PASS_QUALITY:
        ui.message("That's readable." if solved else "That's readable. Enter the number you can see.", "ok")
    elif legib > 0.3:
        ui.message(f"Something is showing up near {where(box)}. Keep going.", "")
    else:
        ui.message("Still too dark to make anything out. Look at the histogram: almost everything is squeezed "
                   "into the far left.", "")


def augmentation_panel(reference: np.ndarray) -> None:
    st.subheader("Same tools, different job", anchor=False)
    left, right = st.columns(2)
    with left:
        st.markdown("**Enhancement (what you just did)**  \n"
                    "Change *this one image* so that a person, or a model, can read it. "
                    "It happens at inference time, on every frame the camera sends.")
    with right:
        st.markdown("**Augmentation**  \n"
                    "Change *training images* at random, so the model has already seen dark, tilted, "
                    "blurry or noisy versions before it ever meets a real night frame.")
    st.caption("Below: one clean photo, randomly augmented the way a training loop would. "
               "A model trained on lots of these needs less enhancement later.")
    if st.button("Draw a new batch"):
        st.session_state.l1_aug_seed = int(np.random.default_rng().integers(1_000_000))
    rng = np.random.default_rng(st.session_state.get("l1_aug_seed", 7))
    small = cv2.resize(reference, (320, 213), interpolation=cv2.INTER_AREA)
    cols = st.columns(6)
    for col in cols:
        img, applied = enh.random_augment(small, rng)
        if "flip" in applied:
            applied = [*applied, "digits mirrored: not label-safe"]
        with col:
            st.image(rgb(img), width="stretch")
            ui.caption(" · ".join(applied))


DETECTOR = "yolo26s.pt"


def try_detector() -> tuple[str, float]:
    _, dark = load_frames()
    dets, _ = det.detect(runtime.yolo(DETECTOR), dark, min_conf=0.25)
    st.session_state.l1_dark_dets = dets
    return "YOLO26s on the dark frame", runtime.profile("detectors", DETECTOR)["latency_ms"]


def show_detector() -> None:
    _, dark = load_frames()
    dets = st.session_state.get("l1_dark_dets", [])
    c1, c2 = st.columns([1.2, 1])
    c1.image(rgb(det.draw(dark, dets=dets)), width="stretch")
    c2.markdown(f"**{len(dets)} object(s) found**, and no number. " + (
        ", ".join(f"{d['label']} {d['conf']:.0%}" for d in dets) if dets else
        "In the dark frame almost every pixel is between 0 and 15. There's nothing for the network to work with."))


MODE_OPTIONS = {
    "Enhance": {
        "blurb": "Adjust the frame itself: gamma, contrast, histogram tools. Plain OpenCV, no neural network.",
        "verdict": "Fix the input before spending compute on it. Gamma and contrast are lookup tables "
                   "and cost next to nothing.",
    },
    "Detect": {
        "blurb": "Run the object detector on the frame as it is and hope it spots something.",
        "run": try_detector, "show": show_detector,
        "verdict": "A neural network can't see detail that isn't in the pixels yet. You spent battery on a frame "
                   "that was never going to give an answer. Enhance first, then decide if you need a model at all.",
    },
    "Retrain": {
        "blurb": "Train a model on darkened examples so it copes with night frames by itself.",
        "verdict": "That's augmentation, and it's the right idea for the next version of GhostLens. But training "
                   "happens on a big machine before deployment, not on a handheld in a corridor at 3 am. "
                   "Right now you need this one frame readable.",
    },
}

WARMUP = [("Reading camera 03", load_frames), ("Loading forensic tools", lambda: enh.Settings())]


def report_checks(pipeline_ms: float, new_clip: float, par: int) -> dict:
    used = device.used_in_level(st.session_state, 1)
    misses = flow.mode_misses(1)
    return {
        "Cheapest tool first": (misses == 0, "enhanced before running any model" if misses == 0
                                else f"tried {', '.join(m for m in st.session_state.l1_tried if m != 'Enhance')} first"),
        f"Clipping under {CLIP_LIMIT:.0%}": (new_clip <= CLIP_LIMIT, f"{new_clip:.1%} pushed to pure black/white"),
        "Fast enough for live video": (pipeline_ms <= 1000 / 30, "your pipeline fits one frame at 30 fps"
                                       if pipeline_ms <= 1000 / 30 else "your pipeline is too slow for 30 fps"),
        "Battery": (efficient(used, par), f"{pct(used)} used, {pct(par)} would have done it"),
    }


def cam04_clue(c: case.Case) -> str:
    return f"CAM 04: figure passing {c.cam04['label']} at 02:19"


def show_cam04() -> None:
    c = case.get_case(st.session_state)
    done = st.session_state.get("l1_cam04")
    if not done:
        return
    reference, dark = clue_frames(c.cam04, CAM04)
    out, _ = enh.run_pipeline(dark, done["settings"])
    a, b = st.columns(2)
    with a:
        st.image(rgb(dark), width="stretch")
        ui.caption("EVIDENCE 04-A · AS RECORDED")
    with b:
        st.image(rgb(out), width="stretch")
        ui.caption(f"EVIDENCE 04-A · ENHANCED · QUALITY {done['quality']:.0%}")


def cam04_scan(settings: enh.Settings, pipeline_ms: float) -> flow.SideScan:
    c = case.get_case(st.session_state)

    def run() -> bool:
        reference, dark = clue_frames(c.cam04, CAM04)
        out, _ = enh.run_pipeline(dark, settings)
        quality = measure(out, dark, reference, c.cam04["clue_box"])[2]
        st.session_state.l1_cam04 = {"settings": settings, "quality": quality}
        return quality >= case.PASS_QUALITY

    return flow.SideScan(
        key="cam04", title="CAM 04 · stairwell",
        blurb=f"Camera 04 covers the stairwell by {c.cam04['place']}. It kept recording two minutes longer than "
              "camera 03, just as dark. Run your current lab settings on its last frame. It's logged if the "
              f"evidence quality reaches {case.PASS_QUALITY:.0%}.",
        what="Forensic pass on CAM 04", latency_ms=pipeline_ms, reward_xp=20, clue=cam04_clue(c),
        run=run, show=show_cam04, tip_topic="Histograms")


def submit_form(clue: dict, pipeline_ms: float, quality: float, new_clip: float, dark: np.ndarray) -> None:
    s = st.session_state
    units = energy_units(pipeline_ms)
    ui.readouts([("Forensic pass", f"≈{pct(units)}", "warn" if units >= COSTLY else ""),
                 ("Pipeline time", f"{pipeline_ms:.1f} ms", ""),
                 ("Battery left", pct(s.battery), flow.battery_tone(s.battery))])
    st.caption("Each submit runs the full pass with the tools you have on, and costs what it measured. "
               "Non-local means is the expensive one.")
    with st.form("l1_clue", border=False):
        c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
        answer = c1.text_input("What number is on the plate?", placeholder="e.g. 104")
        submitted = c2.form_submit_button(f"Analyze clue · ≈{pct(units)}", type="primary", key="l1_submit")
    st.markdown(f'<div class="gl-cost">≈{pct(units)} per pass → ≈{pct(max(0, s.battery - units))} left</div>',
                unsafe_allow_html=True)
    if not submitted:
        return
    s.l1_attempts = s.get("l1_attempts", 0) + 1
    flow.run_cost(1, "Forensic pass", pipeline_ms)
    if not is_answer(answer, clue):
        ui.message("That isn't what the plate says. Get the frame clearer and look again.", "bad")
    elif quality < case.PASS_QUALITY:
        ui.message(f"Right number, but at {quality:.0%} evidence quality nobody would accept this frame. "
                   f"Get it to {case.PASS_QUALITY:.0%} and submit again.", "warn")
    else:
        s.l1_final = {"settings": s.l1_lab, "quality": quality, "clip": new_clip, "ms": pipeline_ms}
        finish_level(1, quality, s.l1_attempts, report_checks(pipeline_ms, new_clip, reference_units(dark)),
                     clue=clue["label"])


TOOL_WHY = {
    "Gamma 2.4": "Gamma lifts dark tones far more than bright ones, and the whole plate is dark tones.",
    "Contrast ×4": "Contrast stretches the gap between the digits and the plate, but they start almost equal.",
    "Brightness +100": "Brightness adds the same to every pixel, so the gap between digits and plate stays the same.",
}


def resolve_tool(clue: dict) -> tuple[set[int], str]:
    q = single_tool_quality(clue["id"])
    best = best_tools(q)
    winner = list(q)[min(best)]
    scores = ", ".join(f"{name} {v:.2f}" for name, v in q.items())
    return best, f"Measured evidence quality: {scores}. {TOOL_WHY[winner]}"


def tool_reveal(clue: dict) -> None:
    q = single_tool_quality(clue["id"])
    ui.readouts([(name, f"{v:.0%}", "ok" if i in best_tools(q) else "") for i, (name, v) in enumerate(q.items())])


def lab_touched() -> bool:
    """True from the first time any lab control leaves its default."""
    s = st.session_state
    now = {**DEFAULTS, **s.get("l1_lab", {}), **{k: s[k] for k in DEFAULTS if k in s}}
    if now != DEFAULTS:
        s.l1_touched = True
    return bool(s.get("l1_touched"))


def tool_prediction(clue: dict) -> None:
    # only before the first slider move: once you've tried, it isn't a prediction any more
    if "l1_pred_tool" not in st.session_state and lab_touched():
        return
    flow.predict("l1_pred_tool", "Before you touch anything: which single tool, on its own, makes the plate most "
                 "readable?", list(SINGLE_TOOLS), resolve=lambda: resolve_tool(clue), reveal=lambda: tool_reveal(clue))


DEBRIEF_SETUPS = {
    "Reference (gamma 2.4, contrast 2.5)": case.REFERENCE,
    "Reference + non-local means": {**case.REFERENCE, "denoise_method": "Non-local means"},
    "Brightness +100 only": {"brightness": 100},
}


@st.cache_data(show_spinner=False)
def debrief_runs(clue_id: str) -> list[dict]:
    """The comparison rows, measured once per photo."""
    clue = next(c for c in case.CLUES if c["id"] == clue_id)
    reference, dark = clue_frames(clue)
    rows = []
    for name, kw in DEBRIEF_SETUPS.items():
        out, timings = enh.run_pipeline(dark, enh.Settings(**kw))
        _, clip, quality = measure(out, dark, reference, clue["clue_box"])
        rows.append({"setup": name, "quality": quality, "clip": clip, "ms": sum(ms for _, ms in timings)})
    return rows


def debrief_frame(final: dict | None, runs: list[dict]) -> pd.DataFrame:
    rows = ([{"setup": "Yours", **{k: final[k] for k in ("quality", "clip", "ms")}}] if final else []) + runs
    return pd.DataFrame([{"setup": r["setup"], "evidence quality": round(r["quality"], 3), "clipped": f"{r['clip']:.1%}",
                          "ms": round(r["ms"], 1), "battery": pct(energy_units(r["ms"]))} for r in rows])


def debrief(clue: dict) -> None:
    final = st.session_state.get("l1_final")
    runs = debrief_runs(clue["id"])
    ref, nlm, bright = runs
    st.dataframe(debrief_frame(final, runs), hide_index=True, width="stretch")
    st.caption(f"{ui.tag('measured')} quality, clipping and time, timed on this machine "
               f"&nbsp;{ui.tag('gameplay')} battery", unsafe_allow_html=True)
    ratio = energy_units(nlm["ms"]) / energy_units(ref["ms"])
    yours = (f"Your pass: quality {final['quality']:.2f}, {final['clip']:.1%} clipped, {final['ms']:.1f} ms."
             if final else "Lookup tables like gamma cost almost nothing.")
    ui.lesson([
        f"Non-local means changed quality by {nlm['quality'] - ref['quality']:+.2f} for {ratio:.0f}× the battery "
        "of the reference pass. Expensive isn't automatically better.",
        f"Brightness +100 on its own scores {bright['quality']:.2f}: every pixel moves by the same amount, so the "
        "digits stay as close to the plate as before.",
        f"{yours} Fix the input before you spend compute on a model.",
    ], title="DEBRIEF")


def learn_more() -> None:
    with st.expander("Learn more: histograms, gamma, CLAHE, normalization"):
        st.markdown(
            r"""
**Histogram.** For every brightness value from 0 to 255, how many pixels have it. A dark frame piles up on the left.
Good enhancement spreads the pile out without pushing pixels into the red zones at either end.

**Gamma.** $out = 255 \cdot (in/255)^{1/\gamma}$. With $\gamma > 1$ a pixel at 10 moves a lot more than a pixel at 200,
which is why gamma is the usual first step for low-light images. It's a lookup table, so it's very cheap.

**Contrast and brightness.** $out = \alpha \cdot in + \beta$, clipped to 0–255. $\alpha$ stretches the histogram,
$\beta$ only slides it.

**Global equalization vs CLAHE.** Equalization reshapes the whole histogram to be roughly flat. CLAHE does the same in
small tiles with a limit on how much it can stretch, so it doesn't blow up noise in flat regions as much.

**Why denoise first?** Every later step amplifies noise. Removing it early means less to amplify.

**Enhancement is not normalization.** Enhancement changes what one frame shows, so a person or a model can read it.
Input normalization (÷255, or subtracting the dataset mean and dividing by its std) only rescales every input into the
range the network was trained on. It makes nothing more visible, and it has to match training exactly.

**How legibility is measured.** We know where the plate is and have the clean photo. Legibility is the correlation between
your plate region and the clean one, times how much contrast the region has compared to the clean one (capped at 1).
Evidence quality = legibility − 3 × (share of pixels you pushed to pure black or white).
"""
        )


def render() -> None:
    s = st.session_state
    c = case.get_case(s)
    clue = c.clue
    ui.scene_header(
        "CHAPTER 1 · CAM 03 · 2026-10-07 02:17:44",
        "The Dark Frame",
        f"Camera 03 watches the corridor outside {clue['place']}. It stopped recording at 02:17, and this is the "
        "last frame it saved. The night staff say there's a number in it. Bring it back.",
    )
    reference, dark = clue_frames(clue)
    solved = 1 in s.completed_levels

    if not solved and not flow.mode_choice(
            1, "The frame is almost black. What's the cheapest way to get the number out of it?",
            MODE_OPTIONS, "Enhance", needs="a readable number from this one frame"):
        st.image(rgb(dark), width=480)
        ui.caption("EVIDENCE 03-A · AS RECORDED")
        return

    tool_prediction(clue)
    ctrl_col, view_col = st.columns([1, 2.4], gap="large")
    with ctrl_col:
        st.markdown("**Forensic Image Lab**")
        settings = controls()

    enhanced, timings = enh.run_pipeline(dark, settings)
    pipeline_ms = sum(ms for _, ms in timings)
    legib, new_clip, quality = measure(enhanced, dark, reference, clue["clue_box"])

    with view_col:
        a, b = st.columns(2)
        with a:
            st.image(rgb(dark), width="stretch")
            ui.caption("EVIDENCE 03-A · ORIGINAL")
        with b:
            st.image(rgb(enhanced), width="stretch")
            ui.caption("EVIDENCE 03-A · ENHANCED")
        ui.readouts([
            ("Mean brightness", f"{enh.gray(enhanced).mean():.0f} / 255", ""),
            (f"Clipped by you (limit {CLIP_LIMIT:.1%})", f"{new_clip:.1%}", clip_tone(new_clip)),
            ("Clue legibility", f"{legib:.0%}", tone(legib, 0.75, 0.4)),
            ("Evidence quality", f"{quality:.0%}", tone(quality, case.PASS_QUALITY, 0.4)),
        ])
        st.caption(f"{ui.tag('measured')} legibility, clipping, evidence quality &nbsp;{ui.tag('gameplay')} "
                   f"the {case.PASS_QUALITY:.0%} pass mark and the {CLIP_LIMIT:.0%} clipping limit",
                   unsafe_allow_html=True)
        st.altair_chart(histogram_chart(dark, enhanced), width="stretch")
        feedback(settings, legib, new_clip, quality, energy_units(pipeline_ms), clue["clue_box"], solved)

    st.divider()
    if solved:
        completion_panel(1, [("Clue recovered", clue["label"]),
                             ("Best evidence quality", f"{s.best_scores.get(1, quality):.0%}")],
                         par=reference_units(dark), debrief=lambda: debrief(clue))
        if c.cam04:
            flow.side_scan(1, cam04_scan(settings, pipeline_ms))
        augmentation_panel(reference)
    else:
        submit_form(clue, pipeline_ms, quality, new_clip, dark)
    learn_more()
