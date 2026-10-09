"""Chapter list, chapter endings, the title screen, the lock screen, the case summary and the about page."""

import streamlit as st

from game import case, device, state
from game.device import pct
from game.scoring import edge_grade, level_xp
from ui import components as ui

LEVELS = {
    1: {"title": "The Dark Frame", "mode": "Enhance", "concept": "Image enhancement",
        "question": "Can we get the clue out of a frame that's almost black?"},
    2: {"title": "Identify the Entity", "mode": "Classify", "concept": "Image classification",
        "question": "What is this one object?"},
    3: {"title": "Find the Anomalies", "mode": "Detect", "concept": "Object detection",
        "question": "What is in this room, and where?"},
    4: {"title": "The Corrupted Region", "mode": "Segment", "concept": "Segmentation + deployment",
        "question": "Exactly which pixels are affected?"},
}


def outro(n: int, c: case.Case) -> str:
    if n == 1:
        return (f"The plate reads {c.number}: {c.clue['place']}. The night manager hands you a key card and "
                "doesn't offer to come with you.")
    if n == 2:
        return (f"The {c.anchor} isn't in the guest's sitting room. The lift indicator shows someone went back up "
                f"to the {c.clue['floor']} floor at 04:05. The only room open up there is the old parlour.")
    if n == 3:
        return (f"Against last week's photo, one thing has moved: {case.MOVABLE[c.moved]}. Your report goes to the "
                "manager. While you wait, the detector keeps running as a watchdog. Then it flags something it "
                "has no name for, on the wall between the curtains.")
    return ""


def render_level(n: int) -> None:
    from game import level1, level2, level3, level4
    {1: level1, 2: level2, 3: level3, 4: level4}[n].render()


def warmup(n: int) -> list:
    from game import level1, level2, level3, level4
    return {1: level1, 2: level2, 3: level3, 4: level4}[n].WARMUP


def finish_level(n: int, quality: float, attempts: int, checks: dict[str, tuple[bool, str]],
                 clue: str | None = None) -> None:
    s = st.session_state
    name, result = device.lobby_check(s, n)
    checks = {**checks, name: result}
    grade = edge_grade({k: ok for k, (ok, _) in checks.items()})
    xp = level_xp(quality, attempts, grade)
    if state.complete_level(s, n, xp["total"], clue=clue, score=quality):
        xp["charge"] = device.grade_reward(s, n, grade)
        s[f"l{n}_xp"] = xp
        s[f"l{n}_report"] = {"checks": checks, "grade": grade}
        s.setdefault("grades", {})[n] = grade
    st.rerun()


def xp_rows(n: int, rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    xp = st.session_state.get(f"l{n}_xp")
    if not xp:
        return rows
    lines = [*rows, ("Base", f"+{xp['base']}"), ("Result quality", f"+{xp['quality_bonus']}"),
             ("Edge grade bonus", f"+{xp['edge_bonus']}")]
    if xp["attempt_penalty"]:
        lines.append(("Extra attempts", str(xp["attempt_penalty"])))
    return lines + [("XP earned", f"<b>+{xp['total']}</b>")]


def ledger_rows(n: int, par: int | None) -> list[tuple[str, str]]:
    s = st.session_state
    used = device.used_in_level(s, n)
    side = device.used_in_level(s, n, kind="side")
    cell = any(r["kind"] == "cell" and r["level"] == n for r in s.ledger)
    trips = device.lobby_trips(s, n)
    rows = [("Spent on the chapter", pct(used))]
    if par is not None:
        tone = "ok" if used <= par else "bad"
        rows[0] = ("Spent on the chapter", f"<span class='{tone}'>{pct(used)}</span>")
        rows.append(("Par", pct(par)))
    rows += [
        ("Side scan", pct(side) if side else "–"),
        ("Spare cell", f"<span class='ok'>+{pct(device.A_GRADE_UNITS)}</span>" if cell else "–"),
        ("Lobby trips", f"<span class='bad'>{trips}</span>" if trips else "0"),
        ("Battery now", f"<b>{pct(s.battery)}</b>"),
    ]
    return rows


def completion_panel(n: int, rows: list[tuple[str, str]], par: int | None = None) -> None:
    from game import flow, nav
    s = st.session_state
    report = s.get(f"l{n}_report")
    if report:
        flow.mission_report(n, report["checks"], report["grade"])
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(f'<div class="gl-lesson"><div class="title">CHAPTER {n} COMPLETE</div>'
                    f'{ui.kv_table(xp_rows(n, rows))}</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="gl-lesson"><div class="title">BATTERY</div>'
                    f'{ui.kv_table(ledger_rows(n, par))}</div>', unsafe_allow_html=True)
    text = outro(n, case.get_case(s))
    if text:
        st.markdown(f'<p class="gl-story gl-outro">{text}</p>', unsafe_allow_html=True)
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
    from game import flow, nav
    s = st.session_state
    st.markdown(
        f'<div class="gl-title"><div class="gl-kicker" style="color:#9C978C">{case.case_label(s)} · STANLEY WING</div>'
        '<div class="word">GHOSTLENS</div>'
        '<div class="sub">a computer vision mystery in four chapters</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown(
            '<p class="gl-story">Camera 03 stopped recording at 02:17. Since then the night staff have reported a '
            "missing heirloom, a rearranged tea set, and a stain on the parlour wall that wasn't there yesterday."
            "</p><p class='gl-story'>You get the case, and a GhostLens Mk.II: a handheld camera that runs computer "
            "vision models on the device itself. There's no server to fall back on. It has 20% battery left, "
            "32 MB for models, and that's all.</p>",
            unsafe_allow_html=True,
        )
        if state.all_solved(s):
            st.markdown("<p class='gl-story'><b>Case closed.</b> The full case file is below.</p>",
                        unsafe_allow_html=True)
        else:
            next_level = next(n for n in LEVELS if n not in s.completed_levels)
            label = "Begin the investigation" if next_level == 1 else f"Continue · Chapter {next_level}"
            if st.button(label, type="primary"):
                st.switch_page(nav.PAGES[next_level])
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
        st.markdown(flow.rules_html(), unsafe_allow_html=True)
    if state.all_solved(s):
        st.divider()
        case_closed()


# the case summary, shown on the home page and at the end of chapter 4

def case_file_rows(c: case.Case) -> list[tuple[str, str]]:
    from game import level1
    s = st.session_state
    cam = level1.cam04_clue(c) if c.cam04 and s.get("side_scans", {}).get("cam04") else "not checked"
    anchor = case.ANCHORS[c.anchor]
    brief = case.BRIEFS[c.brief]
    deployed = s.get("l4_deployed")
    wall = f"#{c.wall_seed}"
    if deployed:
        wall += f" · {deployed['name']} · IoU {deployed['iou']:.3f}"
    label = s.get("l2_anchor_label")
    return [
        ("Room", f"{c.clue['label']} · {c.clue['place']}"),
        ("Camera 04", cam),
        ("Entity", f"{anchor['entity']} · the {c.anchor}" + (f" · classifier said <i>{label}</i>" if label else "")),
        ("Client brief", f"{brief['title']} · {brief['rule']}"),
        ("Moved", case.MOVABLE[c.moved]),
        ("Wall", wall),
    ]


def battery_rows() -> list[tuple[str, str]]:
    d = device.summary(st.session_state)
    trips = d["lobby_trips"]
    lobby = (f"<span class='bad'>{trips} · +{pct(trips * device.CHARGER_UNITS)} · "
             f"−{trips * device.CHARGER_XP} XP</span>") if trips else "none"
    return [
        ("Start", pct(d["start"])),
        ("Chapter runs", f"−{pct(d['spent_main'])}"),
        ("Side scans", f"−{pct(d['spent_side'])}"),
        ("Spare cells", f"<span class='ok'>{d['spare_cells']} · +{pct(d['spare_cells'] * device.A_GRADE_UNITS)}</span>"),
        ("Lobby trips", lobby),
        ("Lowest", pct(d["lowest"])),
        ("Left", f"<b>{pct(d['left'])}</b>"),
    ]


def epilogue(c: case.Case) -> str:
    entity = case.ANCHORS[c.anchor]["entity"]
    cam = c.cam04["label"] if c.cam04 else "the stairwell"
    return (f"Four side clues, one timeline. Camera 04 saw a figure pass {cam} two minutes after camera 03 went "
            "dark. The cups were where the scout couldn't see them. The second stain is mapped to the pixel. "
            f"{entity} has nothing left to hold on to. At 05:10 the lift goes down to the lobby on its own, "
            "empty, and the doors stay open a little longer than they should.")


def int8_note() -> str:
    from game import runtime
    fp, q = runtime.profile("unets", "standard-fp32"), runtime.profile("unets", "standard-int8")
    return (f"Quantizing U-Net Standard to INT8 made it {fp['size_mb'] / q['size_mb']:.1f}× smaller and "
            f"{fp['latency_ms'] / q['latency_ms']:.1f}× faster, for {fp['iou'] - q['iou']:.3f} IoU (measured).")


def case_closed() -> None:
    from game import achievements, codex, flow
    s = st.session_state
    c = case.get_case(s)
    grades = s.get("grades", {})
    rows = "".join(f"<tr><td>Chapter {n}. {LEVELS[n]['title']}</td><td><b>{grades.get(n, '–')}</b></td></tr>"
                   for n in LEVELS)
    st.markdown(
        f'<div class="gl-report closed"><div class="grade">✓</div><div>'
        f'<div class="gl-kicker">{case.case_label(s)} · closed</div>'
        f'<b>{case.ANCHORS[c.anchor]["entity"]} is gone.</b> '
        f'Battery left: {pct(s.battery)}. Total XP: {s.xp}.'
        f'<table class="mono">{rows}</table></div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.3, 1], gap="large")
    with left:
        st.markdown('<div class="gl-kicker">Case file</div>' + ui.kv_table(case_file_rows(c), cls="gl-casesum"),
                    unsafe_allow_html=True)
    with right:
        st.markdown('<div class="gl-kicker">Battery</div>' + ui.kv_table(battery_rows()), unsafe_allow_html=True)
    chart = flow.battery_timeline_chart()
    if chart is not None:
        st.altair_chart(chart, width="stretch")

    clues = s.get("side_clues", [])
    st.markdown(f'<div class="gl-kicker">Side clues {len(clues)}/{achievements.SIDE_CLUES}</div>',
                unsafe_allow_html=True)
    if clues:
        st.markdown("".join(f"- {clue}\n" for clue in clues))
    else:
        st.caption("None logged. Each chapter has an optional side scan once it's solved.")
    if len(clues) >= achievements.SIDE_CLUES:
        st.markdown(f'<p class="gl-story gl-outro">{epilogue(c)}</p>', unsafe_allow_html=True)

    ui.lesson([
        "Fix the input before you spend compute on it (Chapter 1).",
        "Classification answers <b>what</b>. It's the cheapest, and enough when there's one object (Chapter 2).",
        "Detection answers <b>what and where</b>. The threshold trades precision for recall (Chapter 3).",
        "Segmentation answers <b>which pixels</b>, and costs the most (Chapter 4).",
        "Augmentation helps a model cope with variety during training. Enhancement fixes one input at run time.",
        "On a device the best model is the one that meets the memory, latency and accuracy limits, not the biggest.",
        int8_note(),
    ], title="CASE NOTES")
    codex.badge_strip()


def about() -> None:
    from game import flow
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

{ui.tag("gameplay")} the battery scale (1 ms of measured compute costs 0.1% of a full pack), the 32 MB memory size,
and mission limits. They are rules we chose so the measured numbers turn into decisions.
""",
        unsafe_allow_html=True,
    )
    st.markdown(flow.rules_html(), unsafe_allow_html=True)
    st.markdown("#### Every case is different")
    st.markdown(
        "Each new case draws its room photo, anchor object, client brief, moved cup and stained wall from a small "
        "pool, using one random number. Every option in the pool is checked by the tests with the real models: "
        "the right choice always works and the wrong ones always fail. Add `?case=1234` to the address to replay "
        "a particular case."
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
