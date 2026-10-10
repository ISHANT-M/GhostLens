"""Chapter 1: The Dark Frame (preprocessing and histograms)."""

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from cv import detection as det
from cv import enhancement as enh
from cv.edge import energy_units
from game import case, device, flow, levels, runtime
from game.device import pct
from game.levels import finish_level
from game.scoring import efficient
from ui import components as ui

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "level1"
CAM03 = (217, "CAM 03  2026-10-07  02:17:44")
CAM04 = (404, "CAM 04  2026-10-07  02:19:12")
COSTLY = 50     # 5% of a full pack: the cost readout turns amber
CLIP_LIMIT = 0.01   # share of pixels you may push to pure black or white
# the tool strip: one tool on its own, at the strength a player would try first
SINGLE_TOOLS = {"Brightness +100": {"brightness": 100}, "Contrast ×4": {"contrast": 4.0}, "Gamma 2.4": {"gamma": 2.4}}
PRESETS = {"l1_preset_gamma": "Gamma 2.4", "l1_preset_contrast": "Contrast ×4",
           "l1_preset_bright": "Brightness +100", "l1_preset_reset": "Reset"}
SECTIONS = ["Tone", "Histogram", "Detail"]

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


def stored_par(dark: np.ndarray) -> int:
    s = st.session_state
    if "l1_par" not in s:
        s.l1_par = reference_units(dark)
    return s.l1_par


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


# the scanner controls

def settings_from(lab: dict) -> enh.Settings:
    v = {**DEFAULTS, **lab}
    return enh.Settings(v["l1_denoise"], v["l1_denoise_strength"], v["l1_brightness"], v["l1_contrast"],
                        v["l1_gamma"], v["l1_equalizer"], v["l1_clahe_clip"], v["l1_sharpen"])


def describe(s: enh.Settings) -> str:
    """'gamma 2.4, contrast ×2.5' for the steps that are switched on."""
    parts = []
    if s.denoise_method != "None":
        parts.append(f"{s.denoise_method.lower()} {s.denoise_strength}")
    if s.gamma != 1.0:
        parts.append(f"gamma {s.gamma:g}")
    if s.contrast != 1.0:
        parts.append(f"contrast ×{s.contrast:g}")
    if s.brightness:
        parts.append(f"brightness {s.brightness:+g}")
    if s.equalizer == "CLAHE":
        parts.append(f"CLAHE clip {s.clahe_clip:g}")
    elif s.equalizer != "None":
        parts.append(s.equalizer.lower())
    if s.sharpen:
        parts.append(f"sharpen {s.sharpen:g}")
    return ", ".join(parts) or "nothing switched on"


def reset_lab() -> None:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.l1_lab = dict(DEFAULTS)


def apply_preset(name: str) -> None:
    """Tool strip: start from a clean lab and switch on one tool at the strength in SINGLE_TOOLS."""
    reset_lab()
    for k, v in SINGLE_TOOLS.get(name, {}).items():
        st.session_state[f"l1_{k}"] = v
    st.session_state.l1_lab = {k: st.session_state[k] for k in DEFAULTS}


def controls() -> enh.Settings:
    # widget values are dropped when another page runs (or a section is hidden), so a plain copy brings them back
    s = st.session_state
    for k, v in s.setdefault("l1_lab", dict(DEFAULTS)).items():
        s.setdefault(k, v)
    s.setdefault("l1_section", SECTIONS[0])

    section = st.radio("Section", SECTIONS, key="l1_section", horizontal=True, label_visibility="collapsed")
    if section == "Tone":
        st.slider("Gamma", 0.5, 4.0, step=0.1, key="l1_gamma",
                  help="Above 1 lifts dark tones much more than bright ones.")
        st.slider("Brightness", -50, 150, step=5, key="l1_brightness", help="Adds the same value to every pixel.")
        st.slider("Contrast", 0.5, 20.0, step=0.5, key="l1_contrast",
                  help="Multiplies every pixel. Spreads values apart, and pushes some past 255.")
    elif section == "Histogram":
        st.radio("Equalization", ["None", "Global equalization", "CLAHE"], key="l1_equalizer")
        if s.l1_equalizer == "CLAHE":
            st.slider("CLAHE clip limit", 1.0, 8.0, step=0.5, key="l1_clahe_clip")
    else:
        st.selectbox("Denoise", ["None", "Gaussian blur", "Median", "Non-local means"], key="l1_denoise")
        if s.l1_denoise != "None":
            st.slider("Denoise strength", 1, 6, key="l1_denoise_strength")
        st.slider("Sharpen", 0.0, 4.0, step=0.25, key="l1_sharpen")

    s.l1_lab = {k: s[k] for k in DEFAULTS}
    return settings_from(s.l1_lab)


def tool_strip() -> None:
    st.markdown('<div class="gl-kicker">Tool strip · one tool at a time</div>', unsafe_allow_html=True)
    with st.container(key="l1_tools", horizontal=True):
        for key, name in PRESETS.items():
            st.button(name, key=key, on_click=apply_preset, args=(name,), type="secondary")


# the stage

def stage_view(dark: np.ndarray, enhanced: np.ndarray, readings: list[tuple[str, str, str]]) -> None:
    s = st.session_state
    s.setdefault("l1_wipe", 0.5)
    ui.evidence(rgb(enh.wipe(dark, enhanced, s.l1_wipe)), "EVIDENCE 03-A · LEFT AS RECORDED · RIGHT ENHANCED")
    st.slider("Wipe", 0.0, 1.0, step=0.01, key="l1_wipe", label_visibility="collapsed")
    ui.evidence(rgb(enh.histogram_strip(dark, enhanced)),
                "HISTOGRAM · DIM AS RECORDED · LIGHT ENHANCED · RED AT 0 AND 255 = CLIPPED")
    ui.readouts(readings)


def clip_line(new_clip: float) -> str:
    return (f"<b>{new_clip:.1%} is now pure black or pure white</b>, over the {CLIP_LIMIT:.1%} limit. "
            "That detail is gone for good.")


def feedback(s: enh.Settings, legib: float, new_clip: float, quality: float, units: int, box: list[float],
             solved: bool = False) -> tuple[str, str]:
    """One line about what the current settings did, and its tone."""
    if clip_tone(new_clip) == "bad":
        return clip_line(new_clip), "bad"
    if s.denoise_method == "Non-local means":
        return (f"<b>Non-local means costs about {pct(units)} a pass</b>, hundreds of times a lookup table. "
                "Gamma and contrast already read this plate.", "warn")
    if new_clip > CLIP_LIMIT:
        return clip_line(new_clip), "warn"
    if s.denoise_method == "Gaussian blur" and s.denoise_strength >= 3:
        return "The blur cleans the grain but smears the digits: noise and detail are both small, fast changes.", "warn"
    if s.sharpen >= 1.5:
        return "Sharpening makes the grain louder. In a night frame most small changes are noise.", "warn"
    if s.brightness > 0 and s.contrast == 1.0 and s.gamma == 1.0 and legib < 0.3:
        return ("Lighter, not clearer. Brightness adds the same to every pixel, so digits and plate stay "
                "as close as before.", "warn")
    if quality >= case.PASS_QUALITY:
        return ("That's readable." if solved else "That's readable. Enter the number you can see."), "ok"
    if legib > 0.3:
        return f"Something is showing up near {where(box)}. Keep going.", ""
    return "Still too dark. The histogram is squeezed into the far left.", ""


# mode dial

DETECTOR = "yolo26s.pt"


def try_detector() -> tuple[str, float]:
    _, dark = load_frames()
    dets, _ = det.detect(runtime.yolo(DETECTOR), dark, min_conf=0.25)
    st.session_state.l1_dark_dets = dets
    return "YOLO26s on the dark frame", runtime.profile("detectors", DETECTOR)["latency_ms"]


def show_detector() -> None:
    _, dark = load_frames()
    dets = st.session_state.get("l1_dark_dets", [])
    ui.evidence(rgb(det.draw(dark, dets=dets)), f"EVIDENCE 03-A · YOLO26S · {len(dets)} OBJECTS")


def show_dark() -> None:
    _, dark = load_frames()
    ui.evidence(rgb(dark), "EVIDENCE 03-A · AS RECORDED")


def mode_options() -> dict:
    found = len(st.session_state.get("l1_dark_dets", []))
    return {
        "Enhance": {
            "blurb": "Fix the frame itself",
            "line": "Gamma and contrast are lookup tables.",
            "verdict": "Fix the input before spending compute on it. Gamma and contrast are lookup tables "
                       "and cost next to nothing.",
        },
        "Detect": {
            "blurb": "Run the detector as is",
            "run": try_detector, "show": show_detector,
            "line": f"Detector found {found} object{'s' if found != 1 else ''} and no number.",
            "verdict": "A neural network can't see detail that isn't in the pixels yet. You spent battery on a frame "
                       "that was never going to give an answer. Enhance first, then decide if you need a model at all.",
        },
        "Retrain": {
            "blurb": "Train on dark examples",
            "line": "Training happens on a big machine before deployment.",
            "verdict": "That's augmentation, a fix for the next version. Training happens on a big machine before "
                       "deployment. Right now you need this one frame readable.",
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


# side scan: camera 04

def cam04_clue(c: case.Case) -> str:
    return f"CAM 04: figure passing {c.cam04['label']} at 02:19"


def show_cam04() -> None:
    c = case.get_case(st.session_state)
    done = st.session_state.get("l1_cam04")
    if not done:
        return
    _, dark = clue_frames(c.cam04, CAM04)
    out, _ = enh.run_pipeline(dark, done["settings"])
    a, b = st.columns(2)
    with a:
        ui.evidence(rgb(dark), "EVIDENCE 04-A · AS RECORDED")
    with b:
        ui.evidence(rgb(out), f"EVIDENCE 04-A · ENHANCED · QUALITY {done['quality']:.0%}")


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
        blurb=f"Camera 04 covers the stairwell outside {c.cam04['place']}. It kept recording two minutes longer than "
              "camera 03, just as dark. Run your final settings on its last frame. It's logged if the "
              f"evidence quality reaches {case.PASS_QUALITY:.0%}.",
        what="Forensic pass on CAM 04", latency_ms=pipeline_ms, reward_xp=20, clue=cam04_clue(c),
        run=run, show=show_cam04, tip_topic="Histograms")


# submitting the plate

def submit_form(clue: dict, settings: enh.Settings, pipeline_ms: float, quality: float, new_clip: float,
                dark: np.ndarray) -> tuple[str, str] | None:
    """The answer box. Returns the feedback line of a submit, if there was one."""
    s = st.session_state
    units = energy_units(pipeline_ms)
    with st.form("l1_clue", border=False):
        answer = st.text_input("What number is on the plate?", placeholder="e.g. 104")
        submitted = st.form_submit_button(f"Analyze clue · ≈{pct(units)}", type="primary", key="l1_submit")
    st.markdown(f'<div class="gl-cost">≈{pct(units)} a pass → ≈{pct(max(0, s.battery - units))} left</div>',
                unsafe_allow_html=True)
    if not submitted:
        return None
    s.l1_attempts = s.get("l1_attempts", 0) + 1
    flow.run_cost(1, "Forensic pass", pipeline_ms)
    if not is_answer(answer, clue):
        return "That isn't what the plate says. Get the frame clearer and look again.", "bad"
    if quality < case.PASS_QUALITY:
        return (f"Right number, but nobody accepts a frame at {quality:.0%} evidence quality. "
                f"Get it to {case.PASS_QUALITY:.0%}.", "warn")
    # par is timed once here and stored, so the report and the cleared screen show the same number
    s.l1_par = reference_units(dark)
    s.l1_final = {"settings": dict(s.l1_lab), "pipeline": asdict(settings), "quality": quality, "clip": new_clip,
                  "ms": pipeline_ms}
    finish_level(1, quality, s.l1_attempts, report_checks(pipeline_ms, new_clip, s.l1_par),
                 clue=clue["label"])
    return None


# cleared screen

TOOL_WHY = {
    "Gamma 2.4": "Gamma lifts dark tones far more than bright ones, and the whole plate is dark tones.",
    "Contrast ×4": "Contrast stretches the gap between the digits and the plate, but they start almost equal.",
    "Brightness +100": "Brightness adds the same to every pixel, so the gap between digits and plate stays the same.",
}

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


def final_settings() -> enh.Settings:
    final = st.session_state.get("l1_final") or {}
    if final.get("pipeline"):
        return enh.Settings(**final["pipeline"])
    return settings_from(final.get("settings") or st.session_state.get("l1_lab", {}))


def augmentation_panel(reference: np.ndarray) -> None:
    st.markdown('<div class="gl-kicker">Same tools, different job</div>', unsafe_allow_html=True)
    st.markdown("**Enhancement** (what you just did) changes *this one frame* so a person or a model can read it, "
                "at inference time. **Augmentation** changes *training images* at random, so a model has already "
                "seen dark, tilted, blurry or noisy versions before it meets a real night frame.")
    if st.button("Draw a new batch", key="l1_aug_draw"):
        st.session_state.l1_aug_seed = int(np.random.default_rng().integers(1_000_000))
    rng = np.random.default_rng(st.session_state.get("l1_aug_seed", 7))
    small = cv2.resize(reference, (320, 213), interpolation=cv2.INTER_AREA)
    for col in st.columns(6):
        img, applied = enh.random_augment(small, rng)
        if "flip" in applied:
            applied = [*applied, "digits mirrored: not label-safe"]
        with col:
            ui.evidence(rgb(img), " · ".join(applied))


MATHS = r"""
**Histogram.** For every brightness value from 0 to 255, how many pixels have it. A dark frame piles up on the left.
Good enhancement spreads the pile out without pushing pixels into the red ticks at either end.

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

**How legibility is measured.** We know where the plate is and have the clean photo. Legibility is the correlation
between your plate region and the clean one, times how much contrast the region has compared to the clean one (capped
at 1). Evidence quality = legibility − 3 × (share of pixels you pushed to pure black or white).
"""


def numbers(clue: dict, dark: np.ndarray, final_out: np.ndarray) -> None:
    final = st.session_state.get("l1_final")
    runs = debrief_runs(clue["id"])
    ref, nlm, bright = runs
    st.markdown('<div class="gl-kicker">Your pass against the reference</div>', unsafe_allow_html=True)
    st.dataframe(debrief_frame(final, runs), hide_index=True, width="stretch")
    st.caption(f"{ui.tag('measured')} quality, clipping and time, timed on this machine "
               f"&nbsp;{ui.tag('gameplay')} battery", unsafe_allow_html=True)
    ratio = energy_units(nlm["ms"]) / max(energy_units(ref["ms"]), 1)
    st.markdown(f"- Non-local means changed quality by {nlm['quality'] - ref['quality']:+.2f} for about {ratio:.0f}× "
                "the battery of the reference pass.\n"
                f"- Brightness +100 on its own scores {bright['quality']:.2f}: every pixel moves by the same amount.")

    q = single_tool_quality(clue["id"])
    best = best_tools(q)
    st.markdown('<div class="gl-kicker">One tool on its own (the tool strip)</div>', unsafe_allow_html=True)
    ui.readouts([(name + (" · best" if i in best else ""), f"{v:.0%}", "ok" if i in best else "")
                 for i, (name, v) in enumerate(q.items())])
    st.caption(f"{ui.tag('measured')} evidence quality of each single tool on this photo", unsafe_allow_html=True)

    st.markdown('<div class="gl-kicker">Your final histogram</div>', unsafe_allow_html=True)
    ui.evidence(rgb(enh.histogram_strip(dark, final_out)), "DIM AS RECORDED · LIGHT YOUR FINAL FRAME")
    with st.expander("The maths"):
        st.markdown(MATHS)


def cleared(c: case.Case, reference: np.ndarray, dark: np.ndarray) -> levels.Cleared:
    s = st.session_state
    clue = c.clue
    final = s.get("l1_final") or {"quality": s.get("best_scores", {}).get(1, 0.0), "clip": 0.0, "ms": 0.0}
    settings = final_settings()
    final_out, _ = enh.run_pipeline(dark, settings)
    q = single_tool_quality(clue["id"])
    winner = list(q)[min(best_tools(q))]
    detect_cost = energy_units(runtime.profile("detectors", DETECTOR)["latency_ms"])
    attempts = s.get("l1_attempts", 1)

    happened = flow.wrong_mode_lines(1, mode_options()) + [
        f"You enhanced with {describe(settings)}: {final['ms']:.1f} ms a pass, no neural network.",
        f"You submitted {attempts} time{'s' if attempts != 1 else ''}. "
        f"{final['clip']:.1%} of the frame was newly pushed to pure black or white "
        f"(limit {CLIP_LIMIT:.0%}).",
    ]
    why = [
        f"{TOOL_WHY[winner]} On its own it scored {q[winner]:.0%} here, the best single tool (measured).",
        f"Gamma and contrast are lookup tables, so a pass costs about {pct(energy_units(final['ms']))}. "
        f"A detector on the raw frame costs {pct(detect_cost)} and has nothing to work with.",
        "Clipped pixels all end up with the same value, so whatever detail they had is gone. That's why the "
        "clip limit is part of the score.",
        "Enhancement changes this one frame for reading. Augmentation changes training images. Normalization only "
        "rescales inputs for a network.",
    ]

    def evidence() -> None:
        ui.evidence(rgb(final_out), f"EVIDENCE 03-A · ENHANCED · PLATE {c.number}")

    return levels.Cleared(
        headline=(f"Plate {c.number} recovered at {final['quality']:.0%} evidence quality in {final['ms']:.1f} ms, "
                  "no neural network."),
        happened=happened, why=why,
        concept="Image enhancement · histogram · gamma · CLAHE · denoise; enhancement vs augmentation vs "
                "normalization",
        evidence=evidence,
        numbers=lambda: numbers(clue, dark, final_out),
        explore=lambda: augmentation_panel(reference),
        side=cam04_scan(settings, final["ms"]) if c.cam04 else None,
        stats=[("Clue recovered", clue["label"]),
               ("Best evidence quality", f"{s.get('best_scores', {}).get(1, final['quality']):.0%}")],
        par=stored_par(dark),
    )


def render() -> None:
    s = st.session_state
    c = case.get_case(s)
    clue = c.clue
    reference, dark = clue_frames(clue)
    if levels.show_cleared(1):
        levels.level_cleared(1, cleared(c, reference, dark))
        return
    levels.review_banner(1)
    solved = 1 in s.completed_levels

    if not solved and s.get("l1_mode") != "Enhance":
        ui.title_card(
            "CHAPTER 1 · CAM 03 · 2026-10-07 02:17:44", "The Dark Frame",
            f"Camera 03 watches the corridor outside {clue['place']}. It stopped recording at 02:17, and this is "
            "the last frame it saved. The night staff say there's a number in it. Bring it back.")
    if not solved and not flow.mode_choice(
            1, "Get the number out of this almost black frame, as cheaply as you can.", mode_options(), "Enhance",
            needs="a readable number from this one frame", scene=show_dark):
        return

    ui.objective("Read the number on the plate.")
    scene, scanner = ui.stage("l1")
    with scanner:
        ui.scanner_head("GHOSTLENS MK.II · ENHANCE", pct(s.battery))
        settings = controls()
        tool_strip()
        enhanced, timings = enh.run_pipeline(dark, settings)
        pipeline_ms = sum(ms for _, ms in timings)
        legib, new_clip, quality = measure(enhanced, dark, reference, clue["clue_box"])
        result = None if solved else submit_form(clue, settings, pipeline_ms, quality, new_clip, dark)

    with scene:
        stage_view(dark, enhanced, [
            ("Brightness", f"{enh.gray(enhanced).mean():.0f} / 255", ""),
            (f"Clipped (limit {CLIP_LIMIT:.0%})", f"{new_clip:.1%}", clip_tone(new_clip)),
            ("Legibility", f"{legib:.0%}", tone(legib, 0.75, 0.4)),
            ("Evidence quality", f"{quality:.0%}", tone(quality, case.PASS_QUALITY, 0.4)),
        ])
    ui.feedback(*(result or feedback(settings, legib, new_clip, quality, energy_units(pipeline_ms),
                                     clue["clue_box"], solved)))
