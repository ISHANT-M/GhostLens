"""Chapter list, chapter endings, the title screen, the lock screen, the case summary and the about page."""

import re

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
    4: {"title": "The Corrupted Region", "mode": "Segment", "concept": "Segmentation + quantization",
        "question": "Exactly which pixels are affected?"},
}


# what each chapter taught, and the next question in the player's words. Never names the next task.
BRIDGES = {
    1: ("fix the input before you spend compute on it",
        "three objects, one photo each. What is each one?"),
    2: ("one label per image is the cheapest answer when a photo holds one thing",
        "a whole room of things. What is in it, and where is each one?"),
    3: ("the threshold doesn't change the model, it only decides which guesses you believe",
        "something is spreading on the wall. Where exactly does it end?"),
}
MODULES = {
    1: "Module 1 · task identification; image preprocessing",
    2: "Module 2 · classification",
    3: "Module 6 · detection; module 5 · metrics",
    4: "Module 6 · segmentation; module 7 · quantization",
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


def bridge_html(n: int) -> str:
    learned, question = BRIDGES[n]
    return (f'<div class="gl-bridge"><div class="gl-kicker">Next</div>You learned that {learned}. '
            f'<b>Next question:</b> {question}</div>')


def inline_quiz(n: int) -> None:
    from game import codex
    if hasattr(codex, "inline_quiz"):
        codex.inline_quiz(n, n=2)


def completion_panel(n: int, rows: list[tuple[str, str]], par: int | None = None, debrief=None) -> None:
    """Report, debrief, quiz, then the way on. Side scan and Learn more come after, from the chapter."""
    from game import flow, nav
    s = st.session_state
    report = s.get(f"l{n}_report")
    if report:
        flow.mission_report(n, report["checks"], report["grade"])
    with st.expander(f"Chapter {n} XP and battery"):
        left, right = st.columns(2, gap="large")
        left.markdown(f'<div class="gl-lesson"><div class="title">CHAPTER {n} COMPLETE</div>'
                      f'{ui.kv_table(xp_rows(n, rows))}</div>', unsafe_allow_html=True)
        right.markdown(f'<div class="gl-lesson"><div class="title">BATTERY</div>'
                       f'{ui.kv_table(ledger_rows(n, par))}</div>', unsafe_allow_html=True)
    if debrief:
        st.markdown(f'<div class="gl-kicker" style="margin-top:1rem">Chapter {n} debrief</div>',
                    unsafe_allow_html=True)
        debrief()
    inline_quiz(n)
    text = outro(n, case.get_case(s))
    if text:
        st.markdown(f'<p class="gl-story gl-outro">{text}</p>', unsafe_allow_html=True)
    if n in BRIDGES:
        st.markdown(bridge_html(n), unsafe_allow_html=True)
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
                f'<span class="q">{info["mode"] + " · " if n in s.completed_levels else ""}{info["question"]}'
                '</span></span>'
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


def bench_row(bench: dict, group: str, model_id) -> dict | None:
    return next((r for r in bench.get(group, []) if r["id"] == model_id), None)


def int8_ratios(bench: dict) -> tuple[float, float, float]:
    """(times smaller, times faster, IoU change) of U-Net Standard INT8 against FP32."""
    fp, q = bench_row(bench, "unets", "standard-fp32"), bench_row(bench, "unets", "standard-int8")
    return fp["size_mb"] / q["size_mb"], fp["latency_ms"] / q["latency_ms"], q["iou"] - fp["iou"]


def int8_note() -> str:
    from game import runtime
    smaller, faster, delta = int8_ratios(runtime.benchmark())
    return (f"Quantizing U-Net Standard to INT8 made it {smaller:.1f}× smaller and {faster:.1f}× faster, "
            f"for {delta:+.3f} IoU (measured).")


def chapter_number(n: int, store, bench: dict) -> tuple[str, str]:
    """(model, the measured number that decided the chapter)."""
    if n == 1:
        final = store.get("l1_final")
        return "OpenCV pipeline, no network", (f"evidence quality {final['quality']:.2f} in {final['ms']:.1f} ms"
                                               if final else "–")
    if n == 2:
        r = bench_row(bench, "classifiers", store.get("l2_model"))
        label = store.get("l2_anchor_label")
        said = f"top-1 '{label}', " if label else ""
        return (r["name"], f"{said}{r['latency_ms']:.1f} ms a scan") if r else ("–", "–")
    if n == 3:
        r = bench_row(bench, "detectors", store.get("l3_model"))
        f1 = store.get("best_scores", {}).get(3)
        return (r["name"], f"F1 {f1:.2f} at {r['latency_ms']:.1f} ms" if f1 is not None else "–") if r else ("–", "–")
    deployed = store.get("l4_deployed")
    r = bench_row(bench, "unets", deployed["id"]) if deployed else None
    return (deployed["name"], f"IoU {deployed['iou']:.3f} at {r['latency_ms']:.1f} ms") if r else ("–", "–")


def recap_rows(store, bench: dict) -> list[dict]:
    """One row per solved chapter: question, task, model, deciding number, syllabus module."""
    rows = []
    for n, info in LEVELS.items():
        if n not in store.get("completed_levels", ()):
            continue
        model, number = chapter_number(n, store, bench)
        rows.append({"chapter": n, "question": info["question"], "task": info["mode"], "model": model,
                     "number": number, "module": MODULES[n]})
    return rows


def chapter_forecasts(store) -> list[dict]:
    """The chapter predictions the player locked in, in chapter order."""
    keys = sorted(k for k in store.keys() if re.match(r"^l\d_pred_", str(k)))
    return [store[k] for k in keys if isinstance(store[k], dict) and "choice" in store[k]]


def viva_prompts(store, bench: dict) -> list[str]:
    final = store.get("l1_final") or {}
    cls = bench_row(bench, "classifiers", store.get("l2_model")) or {}
    f1 = store.get("best_scores", {}).get(3)
    smaller, faster, delta = int8_ratios(bench)
    return [
        f"Your frame reached evidence quality {final.get('quality', 0):.2f} in {final.get('ms', 0):.1f} ms with "
        "no neural network. Why fix the input before running a model, and why isn't that augmentation?",
        f"{cls.get('name', 'The classifier')} answered chapter 2 at {cls.get('latency_ms', 0):.1f} ms a scan. "
        "Why is classification enough for one object, and what can't it tell you?",
        f"Your best F1 was {f1 or 0:.2f}. What happens to precision and to recall when you lower the threshold, "
        "and why can recall never fall?",
        f"INT8 made U-Net Standard {smaller:.1f}× smaller and {faster:.1f}× faster for {delta:+.3f} IoU. "
        "Why did that decide chapter 4, and why didn't the bigger model win?",
    ]


def recap_markdown(label: str, rows: list[dict], forecasts: list[dict], prompts: list[str]) -> str:
    out = [f"# GhostLens · {label}", "", "| Question | Task | Model | Measured | Syllabus |", "|---|---|---|---|---|"]
    out += [f"| {r['question']} | {r['task']} | {r['model']} | {r['number']} | {r['module']} |" for r in rows]
    right = sum(1 for f in forecasts if f.get("right"))
    out += ["", f"## Forecasts: {right}/{len(forecasts)}", ""]
    out += [f"- {f['q']} You said {f['options'][f['choice']]}. {f.get('why', '')}" for f in forecasts]
    out += ["", "## Say it out loud", ""] + [f"{i}. {p}" for i, p in enumerate(prompts, start=1)]
    return "\n".join(out) + "\n"


def decisions_table(rows: list[dict]) -> str:
    body = "".join(f"<tr><td>{r['chapter']}. {r['question']}</td><td>{r['task']}</td><td>{r['model']}</td>"
                   f"<td>{r['number']}</td><td>{r['module']}</td></tr>" for r in rows)
    head = (f"<tr><th>Question</th><th>Task</th><th>Model</th><th>Deciding number {ui.tag('measured')}</th>"
            "<th>Syllabus</th></tr>")
    return f'<table class="gl-decisions">{head}{body}</table>'


def forecast_section(forecasts: list[dict]) -> None:
    right = sum(1 for f in forecasts if f.get("right"))
    st.markdown(f'<div class="gl-kicker">Forecasts · {right}/{len(forecasts)} called</div>', unsafe_allow_html=True)
    if not forecasts:
        st.caption("No predictions locked in. Each chapter asks for one before the measurement.")
        return
    for f in forecasts:
        mark = "✓" if f.get("right") else "✗" if f.get("right") is False else "·"
        st.markdown(f"- {mark} {f['q']} You said **{f['options'][f['choice']]}**. {f.get('why', '')}")


def explore_section() -> None:
    from game import lab
    tabs = getattr(lab, "TABS", [])
    left = [t for t in tabs if t not in set(st.session_state.get("lab_seen") or ())]
    if not left or not hasattr(lab, "open_lab"):
        return
    st.markdown('<div class="gl-kicker">Still to explore in the Lab</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, tab in enumerate(left):
        if cols[i % 3].button(f"Lab · {tab}", key=f"explore_{tab}", width="stretch"):
            lab.open_lab(tab)


def case_closed() -> None:
    from game import achievements, codex, flow, runtime
    s = st.session_state
    c = case.get_case(s)
    bench = runtime.benchmark()
    grades = s.get("grades", {})
    rows = "".join(f"<tr><td>Chapter {n}. {LEVELS[n]['title']}</td><td><b>{grades.get(n, '–')}</b></td></tr>"
                   for n in LEVELS)
    st.markdown(
        f'<div class="gl-report closed"><div class="grade">✓</div><div>'
        f'<div class="gl-kicker">{case.case_label(s)} · closed</div>'
        f'<b>{case.ANCHORS[c.anchor]["entity"]} is gone.</b> '
        f'Battery left: {pct(s.battery)}. Case XP: {s.xp}.'
        f'<table class="mono">{rows}</table></div></div>',
        unsafe_allow_html=True,
    )
    recap = recap_rows(s, bench)
    st.markdown('<div class="gl-kicker">Four questions, four decisions</div>' + decisions_table(recap),
                unsafe_allow_html=True)
    forecasts = chapter_forecasts(s)
    forecast_section(forecasts)
    prompts = viva_prompts(s, bench)
    st.markdown('<div class="gl-kicker">Say it out loud</div>', unsafe_allow_html=True)
    st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(prompts, start=1)))
    explore_section()

    with st.expander("Case file, battery ledger and timeline"):
        left, right = st.columns([1.3, 1], gap="large")
        left.markdown('<div class="gl-kicker">Case file</div>' + ui.kv_table(case_file_rows(c), cls="gl-casesum"),
                      unsafe_allow_html=True)
        right.markdown('<div class="gl-kicker">Battery</div>' + ui.kv_table(battery_rows()), unsafe_allow_html=True)
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
    st.download_button("Download the case file (.md)", recap_markdown(case.case_label(s), recap, forecasts, prompts),
                       file_name=f"ghostlens_case_{c.seed}.md", mime="text/markdown", type="tertiary")
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
