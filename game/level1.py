"""Chapter 1: The Dark Frame (preprocessing and histograms)."""

import json
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import detection as det
from cv import enhancement as enh
from cv.edge import energy_units
from game import device, flow, runtime
from game.levels import completion_panel, finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level1"
CLUE = json.loads((ASSETS / "clue.json").read_text())

DEFAULTS = {
    "l1_gamma": 1.0, "l1_brightness": 0, "l1_contrast": 1.0, "l1_equalizer": "None",
    "l1_clahe_clip": 2.0, "l1_denoise": "None", "l1_denoise_strength": 1, "l1_sharpen": 0.0,
}


@st.cache_data
def load_frames() -> tuple[np.ndarray, np.ndarray]:
    reference = enh.load_image(ASSETS / CLUE["source"])
    return reference, enh.make_dark_frame(reference)


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
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)

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


def feedback(s: enh.Settings, legib: float, new_clip: float, quality: float) -> None:
    if new_clip > 0.03:
        ui.message(f"<b>{new_clip:.0%} of the frame is now pure black or pure white.</b> Those pixels all have the same "
                   "value, so whatever detail they had is gone for good. Ease off the contrast or brightness.", "bad")
    elif s.denoise_method == "Gaussian blur" and s.denoise_strength >= 3:
        ui.message("The blur is cleaning up the grain, but it's smearing the digits too. Noise and fine detail are "
                   "both small, fast changes, and a blur can't tell them apart.", "warn")
    elif s.sharpen >= 1.5:
        ui.message("Sharpening is making the grain louder. It boosts every small change in the image, "
                   "and in a night frame most of those changes are noise.", "warn")
    elif s.brightness > 0 and s.contrast == 1.0 and s.gamma == 1.0 and legib < 0.3:
        ui.message("The frame got lighter but the number didn't get clearer. Brightness adds the same amount to "
                   "every pixel, so the gap between the digits and the plate stays exactly as small.", "warn")
    elif quality >= CLUE["pass_quality"]:
        ui.message("That's readable. Enter what you see on the plate next to the door.", "ok")
    elif legib > 0.3:
        ui.message("Something is showing up near the top right. Keep going.", "")
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
    c2.markdown(f"**{len(dets)} object(s) found**, and no room number. " + (
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


def report_checks(pipeline_ms: float, new_clip: float) -> dict:
    used = device.used_in_level(st.session_state, 1)
    misses = flow.mode_misses(1)
    return {
        "Cheapest tool first": (misses == 0, "enhanced before running any model" if misses == 0
                                else f"tried {', '.join(m for m in st.session_state.l1_tried if m != 'Enhance')} first"),
        "No clipped pixels": (new_clip <= 0.01, f"{new_clip:.1%} pushed to pure black/white"),
        "Fast enough for live video": (pipeline_ms <= 1000 / 30, "your pipeline fits one frame at 30 fps"
                                       if pipeline_ms <= 1000 / 30 else "your pipeline is too slow for 30 fps"),
        "Battery": (efficient(used, 1), f"{used} units used"),
    }


def render() -> None:
    ui.scene_header(
        "CHAPTER 1 · CAM 03 · 2026-10-07 02:17:44",
        "The Dark Frame",
        "Camera 03 watches the corridor outside the Stephen King suite. It stopped recording at 02:17, and this "
        "is the last frame it saved. The night staff say there's a room number in it. Bring it back.",
    )
    reference, dark = load_frames()
    solved = 1 in st.session_state.completed_levels

    if not solved and not flow.mode_choice(
        1, "The frame is almost black. What's the cheapest way to get the room number out of it?",
        MODE_OPTIONS, "Enhance"):
        st.image(rgb(dark), width=480)
        ui.caption("EVIDENCE 03-A · AS RECORDED")
        return

    ctrl_col, view_col = st.columns([1, 2.4], gap="large")
    with ctrl_col:
        st.markdown("**Forensic Image Lab**")
        settings = controls()

    enhanced, timings = enh.run_pipeline(dark, settings)
    pipeline_ms = sum(ms for _, ms in timings)
    black0, white0 = enh.clipped(dark)
    black, white = enh.clipped(enhanced)
    new_clip = max(0.0, (black + white) - (black0 + white0))
    legib = enh.legibility(enhanced, reference, CLUE["clue_box"])
    quality = enh.evidence_quality(legib, new_clip, weight=3)

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
            ("Pure black", f"{black:.1%}", "bad" if black > black0 + 0.02 else ""),
            ("Pure white", f"{white:.1%}", "bad" if white > 0.02 else ""),
            ("Clue legibility", f"{legib:.0%}", tone(legib, 0.75, 0.4)),
            ("Evidence quality", f"{quality:.0%}", tone(quality, CLUE["pass_quality"], 0.4)),
        ])
        st.altair_chart(histogram_chart(dark, enhanced), width="stretch")
        feedback(settings, legib, new_clip, quality)

    st.divider()
    if solved:
        completion_panel(1, [("Clue recovered", CLUE["answer"]),
                             ("Best evidence quality", f"{st.session_state.best_scores.get(1, quality):.0%}")])
    else:
        units = energy_units(pipeline_ms)
        ui.readouts([("Forensic pass", f"{units} unit{'s' if units != 1 else ''}", ""),
                     ("Pipeline time", f"{pipeline_ms:.1f} ms", ""),
                     ("Battery left", f"{st.session_state.battery}", "")])
        st.caption("Each submit runs the full pass with the tools you have on. Non-local means is the expensive one.")
        with st.form("l1_clue", border=False):
            c1, c2 = st.columns([2, 1], vertical_alignment="bottom")
            answer = c1.text_input("What number is on the plate?", placeholder="e.g. 104")
            submitted = c2.form_submit_button("Analyze clue", type="primary")
        if submitted:
            st.session_state.l1_attempts = st.session_state.get("l1_attempts", 0) + 1
            flow.run_cost(1, "Forensic pass", pipeline_ms)
            if answer.strip() != CLUE["answer"]:
                ui.message("That isn't what the plate says. Get the frame clearer and look again.", "bad")
            elif quality < CLUE["pass_quality"]:
                ui.message(f"Right number, but at {quality:.0%} evidence quality nobody would accept this frame. "
                           f"Get it to {CLUE['pass_quality']:.0%} and submit again.", "warn")
            else:
                finish_level(1, quality, st.session_state.l1_attempts, report_checks(pipeline_ms, new_clip),
                             clue=f"Room {CLUE['answer']}")

    ui.lesson([
        "Process the input before you spend compute on it. Cheap OpenCV steps can make an expensive model unnecessary.",
        "Gamma and contrast recover detail hidden in the dark end of the histogram. Brightness alone doesn't: "
        "it shifts every pixel equally.",
        "Pushing too far clips pixels to pure black or white, and that information can't come back.",
        "Enhancement fixes this input at run time. Augmentation prepares the model during training.",
    ], title="FIELD NOTES")
    augmentation_panel(reference)

    with st.expander("Learn more: histograms, gamma, CLAHE"):
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

**How legibility is measured.** We know where the plate is and have the clean photo. Legibility is the correlation between
your plate region and the clean one, times how much contrast the region has compared to the clean one (capped at 1).
Evidence quality = legibility − 3 × (share of pixels you pushed to pure black or white).
"""
        )
