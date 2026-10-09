"""Chapter list, chapter endings, the title screen, the lock screen, the case summary and the about page."""

from collections.abc import Callable
from dataclasses import dataclass, field

import streamlit as st

from game import case, device, flow, state
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
    1: ("you should fix the input before you spend compute on it",
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


@dataclass
class Cleared:
    headline: str
    happened: list[str]
    why: list[str]
    concept: str
    evidence: Callable[[], None] | None = None
    numbers: Callable[[], None] | None = None
    explore: Callable[[], None] | None = None
    side: flow.SideScan | None = None
    stats: list[tuple[str, str]] = field(default_factory=list)
    par: int | None = None


LEGEND = ("<b>measured</b> on this machine: latency, size, image and mask scores · "
          "<b>published</b>: YOLO26 accuracy from Ultralytics · <b>game rule</b>: battery, memory, limits")


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
        s["just_cleared"] = n
    s[f"l{n}_review"] = False
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


def show_cleared(n: int) -> bool:
    s = st.session_state
    return n in s.get("completed_levels", ()) and not s.get(f"l{n}_review")


def _review(n: int, on: bool) -> None:
    st.session_state[f"l{n}_review"] = on


def _continue(n: int) -> None:
    from game import nav
    target = nav.PAGES.get(n + 1 if n < state.LEVEL_COUNT else "home")
    if target is not None:
        st.switch_page(target)


def continue_label(n: int) -> str:
    if n >= state.LEVEL_COUNT:
        return "Close the case"
    return f"Continue · Chapter {n + 1}: {LEVELS[n + 1]['title']}"


def cleared_head_html(n: int, headline: str, animate: bool) -> str:
    s = st.session_state
    grade = (s.get(f"l{n}_report") or {}).get("grade", "–")
    anim = " animate" if animate else ""
    charge = (s.get(f"l{n}_xp") or {}).get("charge")
    cell = f'<span class="gl-cell">SPARE CELL +{pct(device.A_GRADE_UNITS)}</span>' if charge else ""
    text = outro(n, case.get_case(s))
    story = f'<p class="gl-outro">{text}</p>' if text else ""
    bridge = ""
    if n in BRIDGES:
        bridge = f'<div class="gl-next"><span>NEXT QUESTION</span>{BRIDGES[n][1]}</div>'
    return (f'<div class="gl-cleared-head"><div class="gl-kicker">CHAPTER {n} · CLEARED</div>'
            f'<div class="gl-cleared-row"><div class="gl-stamp-big{anim}">CLEARED</div>'
            f'<div class="gl-grade g{grade}{anim}">{grade}</div>{cell}</div>'
            f'<div class="gl-headline">{headline}</div>{story}{bridge}</div>')


def bullets(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def debrief_tab(n: int, c: Cleared) -> None:
    s = st.session_state
    if c.evidence:
        c.evidence()
    st.markdown(f'<div class="gl-kicker">What happened</div>{bullets(c.happened)}'
                f'<div class="gl-kicker">Why it worked</div>{bullets(c.why)}'
                f'<div class="gl-concept">{c.concept}<span>{MODULES[n]}</span></div>', unsafe_allow_html=True)
    report = s.get(f"l{n}_report")
    if report:
        flow.checklist(f"Chapter {n} · mission report", report["checks"])
    left, right = st.columns(2, gap="large")
    left.markdown('<div class="gl-kicker">XP</div>' + ui.kv_table(xp_rows(n, c.stats)), unsafe_allow_html=True)
    right.markdown('<div class="gl-kicker">Battery</div>' + ui.kv_table(ledger_rows(n, c.par)),
                   unsafe_allow_html=True)


def numbers_tab(c: Cleared) -> None:
    st.markdown(f'<div class="gl-legend">{LEGEND}</div>', unsafe_allow_html=True)
    if c.numbers:
        c.numbers()


def bonus_tab(n: int, c: Cleared) -> None:
    if c.side:
        flow.side_scan(n, c.side)
    if c.explore:
        c.explore()
    if not c.side and not c.explore:
        st.caption("Nothing extra to scan here.")


def level_cleared(n: int, c: Cleared) -> None:
    s = st.session_state
    animate = s.get("just_cleared") == n
    if animate:
        s.pop("just_cleared")
    with st.container(key="cleared"):
        st.markdown(cleared_head_html(n, c.headline, animate), unsafe_allow_html=True)
        a, b, _ = st.columns([1.6, 1, 2])
        a.button(continue_label(n), key=f"l{n}_continue", type="primary", width="stretch",
                 on_click=_continue, args=(n,))
        b.button("Back to the scene", key=f"l{n}_back", type="tertiary", on_click=_review, args=(n, True))
        debrief, numbers, bonus = st.tabs(["Debrief", "Numbers", "Bonus scan"])
        with debrief:
            debrief_tab(n, c)
        with numbers:
            numbers_tab(c)
        with bonus:
            bonus_tab(n, c)


def review_banner(n: int) -> None:
    s = st.session_state
    if n not in s.get("completed_levels", ()) or not s.get(f"l{n}_review"):
        return
    with st.container(key=f"review_l{n}", horizontal=True, vertical_alignment="center"):
        st.markdown(f'<div class="gl-review">CHAPTER {n} · CLEARED</div>', unsafe_allow_html=True)
        st.button("Report", key=f"l{n}_report_btn", type="tertiary", on_click=_review, args=(n, False))


def locked_screen(level: int) -> None:
    st.markdown(
        f'<div class="gl-locked"><div class="title">CHAPTER {level} · LOCKED</div>'
        f"<p>You haven't got this far in the case yet. Finish Chapter {level - 1}, "
        f"<i>{LEVELS[level - 1]['title']}</i>, first.</p>"
        "<p style='margin:0;color:var(--text-dim)'>Presenting? Turn on Demo mode in the Menu.</p></div>",
        unsafe_allow_html=True,
    )


SPECS = ("20% battery", "32 MB for models", "1 ms = 0.1%")


def chapter_card_html(n: int, store) -> str:
    info = LEVELS[n]
    status = state.status(store, n)
    grade = store.get("grades", {}).get(n)
    mark = f'<span class="grade">{grade}</span>' if grade else ""
    mode = info["mode"] + " · " if n in store["completed_levels"] else ""
    return (f'<div class="gl-card {status}"><div class="top"><span class="num">CHAPTER {n}</span>'
            f'<span>{ui.stamp(status)}{mark}</span></div><div class="name">{info["title"]}</div>'
            f'<div class="q">{mode}{info["question"]}</div></div>')


def home() -> None:
    from game import nav
    s = st.session_state
    st.markdown(
        f'<div class="gl-title"><div class="gl-kicker">{case.case_label(s)} · STANLEY WING</div>'
        '<div class="word">GHOSTLENS</div>'
        '<div class="sub">a computer vision mystery in four chapters</div></div>'
        '<p class="gl-story">Camera 03 stopped recording at 02:17, and the night staff have reported a missing '
        "heirloom, a rearranged tea set and a stain that wasn't there yesterday. You get the case and a GhostLens "
        "Mk.II, a handheld camera that runs every vision model on the device itself.</p>"
        '<div class="gl-specs">' + "".join(f'<span class="gl-spec">{x}</span>' for x in SPECS) + "</div>",
        unsafe_allow_html=True,
    )
    if not state.all_solved(s):
        next_level = next(n for n in LEVELS if n not in s.completed_levels)
        label = "Begin the investigation" if next_level == 1 else f"Continue · Chapter {next_level}"
        if st.button(label, type="primary", key="home_start") and next_level in nav.PAGES:
            st.switch_page(nav.PAGES[next_level])
    st.markdown('<div class="gl-cards">' + "".join(chapter_card_html(n, s) for n in LEVELS) + "</div>",
                unsafe_allow_html=True)
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


def talking_points(store, bench: dict) -> list[str]:
    """Four statements about the player's own run, for the downloaded case file."""
    final = store.get("l1_final") or {}
    cls = bench_row(bench, "classifiers", store.get("l2_model")) or {}
    f1 = store.get("best_scores", {}).get(3)
    smaller, faster, delta = int8_ratios(bench)
    return [
        f"The frame reached evidence quality {final.get('quality', 0):.2f} in {final.get('ms', 0):.1f} ms with no "
        "neural network. Fixing the input came first, and that is enhancement, not augmentation.",
        f"{cls.get('name', 'The classifier')} answered chapter 2 at {cls.get('latency_ms', 0):.1f} ms a scan. "
        "One label per image is enough for one object, but it can't say where anything is.",
        f"The best F1 was {f1 or 0:.2f}. Lowering the threshold keeps every box that was already kept, so recall "
        "can only rise while precision can fall.",
        f"INT8 made U-Net Standard {smaller:.1f}× smaller and {faster:.1f}× faster for {delta:+.3f} IoU. That put "
        "it inside the latency limit; the bigger model was accurate but too slow.",
    ]


def recap_markdown(label: str, rows: list[dict], points: list[str]) -> str:
    out = [f"# GhostLens · {label}", "", "| Question | Task | Model | Measured | Syllabus |", "|---|---|---|---|---|"]
    out += [f"| {r['question']} | {r['task']} | {r['model']} | {r['number']} | {r['module']} |" for r in rows]
    out += ["", "## Talking points", ""] + [f"- {p}" for p in points]
    return "\n".join(out) + "\n"


def decisions_table(rows: list[dict]) -> str:
    body = "".join(f"<tr><td>{r['chapter']}. {r['question']}</td><td>{r['task']}</td><td>{r['model']}</td>"
                   f"<td>{r['number']}</td><td>{r['module']}</td></tr>" for r in rows)
    head = ("<tr><th>Question</th><th>Task</th><th>Model</th><th>Deciding number (measured)</th>"
            "<th>Syllabus</th></tr>")
    return f'<table class="gl-decisions">{head}{body}</table>'


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


def closed_head_html(store, c: case.Case, animate: bool) -> str:
    anim = " animate" if animate else ""
    grades = store.get("grades", {})
    stamps = "".join(f'<div class="gl-grade g{grades.get(n, "–")}{anim}" title="Chapter {n}">'
                     f'{grades.get(n, "–")}</div>' for n in LEVELS)
    return (f'<div class="gl-cleared-head"><div class="gl-kicker">{case.case_label(store)} · closed</div>'
            f'<div class="gl-closed-row"><div class="gl-stamp-big{anim}">CASE CLOSED</div>{stamps}</div>'
            f'<div class="gl-headline">{case.ANCHORS[c.anchor]["entity"]} is gone.</div>'
            f'<div class="gl-closed-sum">Battery left {pct(store["battery"])} · Case XP {store["xp"]}</div></div>')


def side_clues_section(c: case.Case) -> None:
    from game import achievements
    clues = st.session_state.get("side_clues", [])
    st.markdown(f'<div class="gl-kicker">Side clues {len(clues)}/{achievements.SIDE_CLUES}</div>',
                unsafe_allow_html=True)
    if clues:
        st.markdown("".join(f"- {clue}\n" for clue in clues))
    else:
        st.caption("None logged. Each chapter has an optional bonus scan once it's cleared.")
    if len(clues) >= achievements.SIDE_CLUES:
        st.markdown(f'<p class="gl-story gl-outro">{epilogue(c)}</p>', unsafe_allow_html=True)


def case_closed() -> None:
    from game import codex, flow, runtime
    s = st.session_state
    c = case.get_case(s)
    bench = runtime.benchmark()
    animate = not s.get("case_closed_seen")
    s["case_closed_seen"] = True
    with st.container(key="case_closed"):
        st.markdown(closed_head_html(s, c, animate), unsafe_allow_html=True)
        recap = recap_rows(s, bench)
        st.markdown('<div class="gl-kicker">Four questions, four decisions</div>' + decisions_table(recap),
                    unsafe_allow_html=True)
        chart = flow.battery_timeline_chart()
        if chart is not None:
            st.markdown('<div class="gl-kicker">Battery, run by run</div>', unsafe_allow_html=True)
            ui.chart(chart)
        left, right = st.columns([1.3, 1], gap="large")
        left.markdown('<div class="gl-kicker">Case file</div>' + ui.kv_table(case_file_rows(c), cls="gl-casesum"),
                      unsafe_allow_html=True)
        right.markdown('<div class="gl-kicker">Battery</div>' + ui.kv_table(battery_rows()), unsafe_allow_html=True)
        side_clues_section(c)
        codex.badge_strip()
        explore_section()
        st.download_button("Download the case file (.md)",
                           recap_markdown(case.case_label(s), recap, talking_points(s, bench)),
                           file_name=f"ghostlens_case_{c.seed}.md", mime="text/markdown", type="tertiary")


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
