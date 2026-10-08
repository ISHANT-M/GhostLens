"""Chapter list, chapter endings, the title screen, the lock screen and the about page."""

import streamlit as st

from game import device, state
from game.scoring import edge_grade, level_xp
from ui import components as ui

LEVELS = {
    1: {"title": "The Dark Frame", "mode": "Enhance", "concept": "Image enhancement",
        "question": "Can we get the clue out of a frame that's almost black?",
        "outro": "The plate reads 217. The Stephen King suite. The night manager hands you a key card and doesn't "
                 "offer to come with you."},
    2: {"title": "Identify the Entity", "mode": "Classify", "concept": "Image classification",
        "question": "What is this one object?",
        "outro": "The watch isn't in the guest's sitting room. The lift indicator shows someone went back up to "
                 "the second floor at 04:05. The only room open up there is the old parlour."},
    3: {"title": "Find the Anomalies", "mode": "Detect", "concept": "Object detection",
        "question": "What is in this room, and where?",
        "outro": "Your report goes to the manager. While you wait, the detector keeps running in the background "
                 "as a watchdog. Then it flags something it has no name for, on the wall between the curtains."},
    4: {"title": "The Corrupted Region", "mode": "Segment", "concept": "Segmentation + deployment",
        "question": "Exactly which pixels are affected?",
        "outro": ""},
}


def render_level(n: int) -> None:
    from game import level1, level2, level3, level4
    {1: level1, 2: level2, 3: level3, 4: level4}[n].render()


def warmup(n: int) -> list:
    from game import level1, level2, level3, level4
    return {1: level1, 2: level2, 3: level3, 4: level4}[n].WARMUP


def finish_level(n: int, quality: float, attempts: int, checks: dict[str, tuple[bool, str]],
                 clue: str | None = None) -> None:
    grade = edge_grade({k: ok for k, (ok, _) in checks.items()})
    xp = level_xp(quality, attempts, grade)
    s = st.session_state
    if state.complete_level(s, n, xp["total"], clue=clue, score=quality):
        s[f"l{n}_xp"] = xp
        s[f"l{n}_report"] = {"checks": checks, "grade": grade}
        s.setdefault("grades", {})[n] = grade
    st.rerun()


def completion_panel(n: int, rows: list[tuple[str, str]]) -> None:
    from game import flow, nav
    s = st.session_state
    report = s.get(f"l{n}_report")
    if report:
        flow.mission_report(n, report["checks"], report["grade"])
    xp = s.get(f"l{n}_xp")
    lines = [*rows]
    if xp:
        lines += [("Base", f"+{xp['base']}"), ("Result quality", f"+{xp['quality_bonus']}"),
                  ("Edge grade bonus", f"+{xp['edge_bonus']}")]
        if xp["attempt_penalty"]:
            lines.append(("Extra attempts", str(xp["attempt_penalty"])))
        lines.append(("XP earned", f"+{xp['total']}"))
    body = "".join(f"<tr><td>{a}</td><td style='text-align:right'>{b}</td></tr>" for a, b in lines)
    st.markdown(
        f'<div class="gl-lesson"><div class="title">CHAPTER {n} COMPLETE</div>'
        f'<table class="mono" style="font-size:0.85rem;min-width:18rem">{body}</table></div>',
        unsafe_allow_html=True,
    )
    outro = LEVELS[n]["outro"]
    if outro:
        st.markdown(f'<p class="gl-story gl-outro">{outro}</p>', unsafe_allow_html=True)
    if n + 1 in nav.PAGES and st.button(f"Continue to Chapter {n + 1}: {LEVELS[n + 1]['title']}", type="primary"):
        st.switch_page(nav.PAGES[n + 1])


def locked_screen(level: int) -> None:
    st.markdown(
        f'<div class="gl-locked"><div class="title">CHAPTER {level} · LOCKED</div>'
        f"<p>You haven't got this far in the case yet. Finish Chapter {level - 1}, "
        f"<i>{LEVELS[level - 1]['title']}</i>, first.</p>"
        "<p style='margin:0;color:var(--ink-soft)'>Presenting? Turn on Demo Mode in the sidebar.</p></div>",
        unsafe_allow_html=True,
    )


def home() -> None:
    from game import nav
    s = st.session_state
    st.markdown(
        '<div class="gl-title"><div class="gl-kicker" style="color:#9C978C">CASE FILE 0217 · STANLEY WING</div>'
        '<div class="word">GHOSTLENS</div>'
        '<div class="sub">a computer vision mystery in four chapters</div></div>',
        unsafe_allow_html=True,
    )
    next_level = next((n for n in LEVELS if n not in s.completed_levels), None)
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown(
            '<p class="gl-story">Camera 03 stopped recording at 02:17. Since then the night staff have reported a '
            "missing pocket watch, a rearranged tea set, and a stain on the parlour wall that wasn't there yesterday."
            "</p><p class='gl-story'>You get the case, and a GhostLens Mk.II: a handheld camera that runs computer "
            "vision models on the device itself. There's no server to fall back on. It has a battery, 32 MB for "
            "models, and that's all.</p>",
            unsafe_allow_html=True,
        )
        if next_level:
            label = "Begin the investigation" if next_level == 1 else f"Continue · Chapter {next_level}"
            if st.button(label, type="primary"):
                st.switch_page(nav.PAGES[next_level])
        else:
            case_closed()
    with right:
        st.markdown('<div class="gl-kicker">Chapters</div>', unsafe_allow_html=True)
        for n, info in LEVELS.items():
            grade = s.get("grades", {}).get(n)
            st.markdown(
                f'<div class="gl-levelrow"><span>{n}. {info["title"]}<br>'
                f'<span class="q">{info["mode"]} · {info["question"]}</span></span>'
                f'<span>{ui.stamp(state.status(s, n))}{f" <b class=mono>{grade}</b>" if grade else ""}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="gl-rule"><b>The one rule.</b> Every scan costs battery. Before you point GhostLens at '
            "something, ask: what's the cheapest thing it can run that still answers my question?</div>",
            unsafe_allow_html=True,
        )


def case_closed() -> None:
    s = st.session_state
    grades = s.get("grades", {})
    rows = "".join(f"<tr><td>Chapter {n}. {LEVELS[n]['title']}</td><td><b>{grades.get(n, '–')}</b></td></tr>"
                   for n in LEVELS)
    st.markdown(
        f'<div class="gl-report closed"><div class="grade">✓</div><div>'
        f'<div class="gl-kicker">Case 0217 · closed</div><b>The Timekeeper is gone.</b> '
        f'Battery left: {s.battery} / {device.BATTERY_START}. Total XP: {s.xp}.'
        f'<table class="mono">{rows}</table></div></div>',
        unsafe_allow_html=True,
    )
    ui.lesson([
        "Fix the input before you spend compute on it (Chapter 1).",
        "Classification answers <b>what</b>. It's the cheapest, and enough when there's one object (Chapter 2).",
        "Detection answers <b>what and where</b>. The threshold trades precision for recall (Chapter 3).",
        "Segmentation answers <b>which pixels</b>, and costs the most (Chapter 4).",
        "Augmentation helps a model cope with variety during training. Enhancement fixes one input at run time.",
        "On a device the best model is the one that meets the memory, latency and accuracy limits, not the biggest.",
        "Quantization traded almost no accuracy for a model 4× smaller and about 2× faster.",
    ], title="CASE NOTES")
    from game import codex
    codex.badge_strip()


def about() -> None:
    ui.scene_header("ABOUT", "About GhostLens",
                    "A small learning game for UCS668 Edge AI and Robotics: Data Center Vision.")
    st.markdown(
        """
**Team:** Ishant Mehndiratta, Satyam Tiwari, Ishaan Sharma
**Course:** UCS668, Thapar Institute of Engineering & Technology (Prof. Jhilik Bhattacharya)

GhostLens is an edge device: limited battery, limited memory, a latency target for every mission. The game is about
picking the cheapest computer vision task and the smallest model that still answer the question.
"""
    )
    st.markdown("#### What's real and what's a game rule")
    st.markdown(
        f"""
{ui.tag("measured")} computed on this machine: image processing, every model prediction, precision/recall and IoU
against our own labels, model file sizes, and inference latency (median of several runs, CPU, measured by `setup_models.py`).

{ui.tag("published")} accuracy figures from Ultralytics for the YOLO26 models (ImageNet top-1, COCO mAP). We didn't re-run those.

{ui.tag("gameplay")} battery units (1 unit per 5 ms of measured compute), the 32 MB memory size, and mission limits.
They are rules we chose so the measured numbers turn into decisions.
""",
        unsafe_allow_html=True,
    )
    st.markdown("#### Models")
    st.markdown(
        "- **YOLO26n / s / m-cls** (ImageNet) for classification\n"
        "- **YOLO26n / s / m** (COCO) for detection, **YOLO26s-seg** for instance segmentation\n"
        "- **Tiny U-Net Lite / Standard / Pro** (ours, plain PyTorch, trained on generated walls), "
        "plus an INT8 version of each made with post-training quantization"
    )
    st.markdown("#### Credits")
    st.markdown("Photos from Wikimedia Commons, see `assets/ATTRIBUTION.md`. "
                "Background sound is synthesised by our own script, `scripts/make_ambience.py`.")
