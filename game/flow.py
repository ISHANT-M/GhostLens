"""Shared chapter parts: loading screen, HUD, paid buttons, mode and model choice, side scans and reports."""

import time
from collections.abc import Callable
from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st

from cv.edge import energy_units
from game import device, state
from game.device import pct
from game.tips import TIPS, random_tip
from ui import components as ui

MODES = ["Enhance", "Classify", "Detect", "Segment"]
# what each mode hands back, for the wrong-mode card
MODE_OUTPUT = {
    "Enhance": "a cleaned-up copy of the same frame",
    "Classify": "one label for the whole image",
    "Detect": "a box and a label for each object",
    "Segment": "a label for every pixel",
    "Retrain": "a new model, after hours of training somewhere else",
}
MIN_LOADING_SECONDS = 2.8
# hex copies of the CSS variables, for Altair
INK, OK, WARN, BAD, SOFT = "#22211F", "#5E7D5A", "#B7862F", "#9C4A3C", "#5B5750"


@dataclass
class SideScan:
    key: str
    title: str
    blurb: str
    what: str
    latency_ms: float
    reward_xp: int
    clue: str
    run: Callable[[], bool]
    show: Callable[[], None]
    intel: tuple[str, str] | None = None
    tip_topic: str | None = None


def loading_tip(chapter: int, seen: set) -> tuple[int, str, str]:
    from game import tips
    picked = tips.pick_tip(chapter, seen) if hasattr(tips, "pick_tip") else None
    return picked if picked and len(picked) == 3 else random_tip(seen)


def loading_html(chapter: int, title: str) -> str:
    # no mode here: the player hasn't chosen one yet
    return (f'<div class="gl-loading"><div class="chapter">CHAPTER {chapter} / {state.LEVEL_COUNT}</div>'
            f'<div class="gl-loading-title">{title}</div><div class="mode">preparing GhostLens</div></div>')


def loading_screen(chapter: int, title: str, work: list[tuple[str, Callable]]) -> None:
    seen = st.session_state.setdefault("seen_tips", set())
    i, topic, tip = loading_tip(chapter, seen)
    seen.add(i)
    box = st.empty()
    with box.container(key="loading"):
        st.markdown(loading_html(chapter, title), unsafe_allow_html=True)
        bar = st.progress(0.0)
        st.markdown(f'<div class="gl-tip"><div class="title">FIELD TIP · {topic.upper()}</div>{tip}</div>',
                    unsafe_allow_html=True)
        start = time.time()
        for k, (label, fn) in enumerate(work):
            bar.progress(k / max(len(work), 1), text=label)
            fn()
        while (elapsed := time.time() - start) < MIN_LOADING_SECONDS:
            bar.progress(min(1.0, elapsed / MIN_LOADING_SECONDS), text="Almost ready")
            time.sleep(0.08)
        bar.progress(1.0, text="Ready")
        time.sleep(0.3)
    box.empty()


# battery display

def battery_tone(units: int) -> str:
    if units >= 150:
        return "ok"
    return "warn" if units >= device.LOW_POWER_BELOW else "bad"


def chip_label(n: int, store) -> str:
    """'CH 2', or 'CH 2 · CLASSIFY' once the player has picked that chapter's mode."""
    mode = store.get(f"l{n}_mode")
    return f"CH {n} · {mode.upper()}" if mode else f"CH {n}"


def hud_html(chapter: int) -> str:
    s = st.session_state
    chips = []
    for n in range(1, state.LEVEL_COUNT + 1):
        cls = "active" if n == chapter else "done" if n in s.completed_levels else \
            "open" if state.is_unlocked(s, n) else "locked"
        chips.append(f'<span class="gl-chip {cls}">{chip_label(n, s)}</span>')
    low = "LOW POWER · " if device.low_power(s) else ""
    # the chips already show the chapter, so the left label stays short and the HUD fits on one line
    return (f'<div class="gl-hud"><span>GHOSTLENS</span>'
            f'<span class="modes">{"".join(chips)}</span>'
            f'<span><span class="{battery_tone(s.battery)}">{low}BATTERY {pct(s.battery)}</span> · '
            f'CASE XP {s.xp}</span></div>')


def battery_meter(units: int) -> str:
    width = min(units / device.CAPACITY, 1) * 100
    ticks = "".join(f'<i class="tick {name}" style="left:{p}%"></i>'
                    for name, p in (("low", 10), ("start", 20)))
    return f'<div class="meter battery {battery_tone(units)}"><span style="width:{width:.1f}%"></span>{ticks}</div>'


def last_entry_html(entry: dict) -> str:
    sign = "+" if entry["kind"] in ("lobby", "cell") else "−"
    measured = (f"{entry['ms']:.0f} ms measured" if entry["ms"]
                else {"cell": "A grade", "lobby": f"for {device.CHARGER_XP} XP"}.get(entry["kind"], entry["kind"]))
    return (f"<div class='row'><span>Last</span><span>{entry['what']}</span></div>"
            f"<div class='row sub'><span>{measured}</span><span>{sign}{pct(entry['units'])}</span></div>")


def device_panel_html() -> str:
    s = st.session_state
    low = device.low_power(s)
    used = device.memory_used(s)
    loaded = ", ".join(name for name, _ in s.resident.values()) or "nothing loaded"
    last = last_entry_html(s.ledger[-1]) if s.ledger else ""
    return (
        f'<div class="gl-device{" low" if low else ""}">'
        f'<div class="head">GHOSTLENS MK.II <span>{"LOW POWER" if low else "NORMAL"}</span></div>'
        f'<div class="gl-battery {battery_tone(s.battery)}">{pct(s.battery)}</div>'
        f'{battery_meter(s.battery)}'
        f'<div class="row sub"><span>of a full pack</span><span>low power below {pct(device.LOW_POWER_BELOW)}</span></div>'
        f'<div class="row"><span>Model memory</span><span>{used:.1f} / {device.MEMORY_MB:.0f} MB</span></div>'
        f'<div class="meter"><span style="width:{min(used / device.MEMORY_MB, 1) * 100:.0f}%"></span></div>'
        f'<div class="row sub"><span>{loaded}</span></div>{last}</div>'
    )


def low_power_toast() -> None:
    s = st.session_state
    if not device.low_power(s):
        s["low_power_warned"] = False
    elif not s.get("low_power_warned"):
        s["low_power_warned"] = True
        st.toast(f"Battery below {pct(device.LOW_POWER_BELOW)}. Heavy models are off and the lobby charger is open.")


# paying for runs

def run_cost(level: int, what: str, latency_ms: float, kind: str = "main") -> int:
    units = energy_units(latency_ms)
    device.spend(st.session_state, level, what, latency_ms, units, kind)
    return units


def cost_line(what: str, latency_ms: float, units: int) -> None:
    st.markdown(
        f'<div class="gl-cost">{ui.tag("measured")} {what}: {latency_ms:.0f} ms &nbsp;→&nbsp; '
        f'{ui.tag("gameplay")} −{pct(units)} battery</div>', unsafe_allow_html=True)


def forecast(units: int, allowed: bool) -> str:
    battery = st.session_state.battery
    if device.can_afford(st.session_state, units):
        return f"→ {pct(battery - units)} left"
    if allowed:
        return "runs on the emergency reserve"
    return f"needs {pct(units)} · {pct(battery)} left"


def charge_prompt(level: int, key: str) -> None:
    s = st.session_state
    if not device.charger_available(s):
        st.caption(f"Not enough charge. The lobby charger only takes packs below {pct(device.LOW_POWER_BELOW)}.")
        return
    ui.message(f"Walk to the lobby charger: +{pct(device.CHARGER_UNITS)} battery for {device.CHARGER_XP} XP. "
               "The trip goes on this chapter's report.", "warn")
    if st.button(f"Walk to the lobby · +{pct(device.CHARGER_UNITS)} for −{device.CHARGER_XP} XP", key=f"lobby_{key}"):
        device.lobby_charge(s, level)
        st.rerun()


def run_button(level: int, label: str, what: str, latency_ms: float, tier: str, key: str,
               kind: str = "main", primary: bool = True) -> bool:
    """A paid button. Returns True when it was clicked and the run was charged."""
    s = st.session_state
    units = energy_units(latency_ms)
    allowed = device.can_run(s, units, tier) if kind == "main" else device.can_afford(s, units)
    clicked = st.button(f"{label} · {pct(units)}", key=key, type="primary" if primary else "secondary",
                        disabled=not allowed)
    st.markdown(f'<div class="gl-cost">{forecast(units, allowed)}</div>', unsafe_allow_html=True)
    if not allowed:
        charge_prompt(level, key)
    if clicked and allowed:
        run_cost(level, what, latency_ms, kind)
        return True
    return False


# side scans

def first_unseen_tip(topic: str, seen: set) -> int | None:
    return next((i for i, (t, _) in enumerate(TIPS) if t == topic and i not in seen), None)


def reward_side_scan(scan: SideScan) -> None:
    s = st.session_state
    s.xp += scan.reward_xp
    if scan.clue not in s.side_clues:
        s.side_clues.append(scan.clue)
    if scan.intel:
        s.intel[scan.intel[0]] = scan.intel[1]
    if scan.tip_topic:
        seen = s.setdefault("seen_tips", set())
        tip = first_unseen_tip(scan.tip_topic, seen)
        if tip is not None:
            seen.add(tip)


def side_card_html(scan: SideScan, logged: bool) -> str:
    chips = [f"+{scan.reward_xp} XP", "SIDE CLUE"] + (["TIP"] if scan.tip_topic else [])
    chips_html = "".join(f'<span class="chip">{c}</span>' for c in chips)
    stamp = ' <span class="gl-stamp SOLVED">LOGGED</span>' if logged else ""
    return (f'<div class="gl-side"><div class="gl-kicker">Optional side scan{stamp}</div>'
            f'<div class="name">{scan.title}</div><div class="blurb">{scan.blurb}</div>'
            f'<div class="chips">{chips_html}</div></div>')


def side_scan(level: int, scan: SideScan) -> None:
    s = st.session_state
    done = s.side_scans.get(scan.key)
    with st.container(key=f"side_card_{scan.key}"):
        st.markdown(side_card_html(scan, done is True), unsafe_allow_html=True)
        if done is not None:
            scan.show()
        if done:
            note = f" New field guide tip: {scan.tip_topic}." if scan.tip_topic else ""
            ui.message(f"<b>Side clue logged.</b> {scan.clue}.{note}", "ok")
            return
        if done is False:
            ui.message("Nothing usable in that scan. The battery is spent all the same. You can try again.", "warn")
        if run_button(level, "Run side scan", scan.what, scan.latency_ms, "Light", key=f"side_{scan.key}",
                      kind="side", primary=False):
            ok = bool(scan.run())
            s.side_scans[scan.key] = ok
            if ok:
                reward_side_scan(scan)
            st.rerun()


# rules and the battery timeline

def rules_html() -> str:
    m, g = ui.tag("measured"), ui.tag("gameplay")
    rules = [
        f"{m} Every run's latency is measured on this machine (median of several runs).",
        f"{g} 1 ms of measured compute costs 0.1% battery. You start at {pct(device.BATTERY_START)}.",
        f"{g} Below {pct(device.LOW_POWER_BELOW)} GhostLens drops to low power. Heavy models switch off and the "
        "lobby charger opens.",
        f"{g} The lobby charger gives +{pct(device.CHARGER_UNITS)} for {device.CHARGER_XP} XP, as often as you need. "
        "Each trip shows up in that chapter's report, so it can't get an A.",
        f"{g} If a run costs more than you have left, light models still run on the emergency reserve. "
        "Anything bigger waits for a charge.",
        f"{g} An A grade earns a spare cell (+{pct(device.A_GRADE_UNITS)}) once per chapter. Optional side scans "
        "cost battery and pay XP, a side clue and a field guide tip.",
    ]
    items = "".join(f"<li>{r}</li>" for r in rules)
    return f'<div class="gl-rule gl-rules"><b>Battery rules</b><ol>{items}</ol></div>'


def timeline_frame(ledger: list[dict]) -> pd.DataFrame:
    rows = [{"step": 0, "battery": device.BATTERY_START / device.UNITS_PER_PERCENT, "what": "Start",
             "kind": "start", "chapter": 1}]
    for i, r in enumerate(ledger, start=1):
        rows.append({"step": i, "battery": r["battery"] / device.UNITS_PER_PERCENT, "what": r["what"],
                     "kind": r["kind"], "chapter": r["level"]})
    return pd.DataFrame(rows)


def battery_timeline_chart() -> alt.Chart | None:
    ledger = st.session_state.get("ledger") or []
    if not ledger:
        return None
    df = timeline_frame(ledger)
    x = alt.X("step:Q", title="run", axis=alt.Axis(tickMinStep=1))
    y = alt.Y("battery:Q", title="battery %", scale=alt.Scale(domainMin=0))
    tooltip = ["what:N", "chapter:Q", alt.Tooltip("battery:Q", format=".1f")]
    line = alt.Chart(df).mark_line(interpolate="step-after", color=INK, strokeWidth=2).encode(x=x, y=y)
    low = alt.Chart(pd.DataFrame({"y": [10]})).mark_rule(color=BAD, strokeDash=[6, 4]).encode(y="y:Q")
    start = alt.Chart(pd.DataFrame({"y": [20]})).mark_rule(color=SOFT, strokeDash=[2, 3]).encode(y="y:Q")
    events = df[df.kind.isin(["lobby", "cell", "side"])]
    colors = alt.Scale(domain=["lobby", "cell", "side"], range=[BAD, OK, WARN])
    points = alt.Chart(events).mark_point(filled=True, size=70).encode(
        x=x, y=y, tooltip=tooltip,
        color=alt.Color("kind:N", scale=colors, legend=alt.Legend(title=None, orient="bottom")))
    hover = alt.Chart(df).mark_point(opacity=0, size=60).encode(x=x, y=y, tooltip=tooltip)
    chart = (low + start + line + hover + points).properties(height=220)
    return chart.configure_view(stroke=None).configure(background="transparent")


# mode and model choice

def mode_choice(level: int, question: str, options: dict[str, dict], correct: str, needs: str = "") -> bool:
    # options: {mode: {"blurb", "verdict", "run": () -> (label, latency_ms) or None, "show": callable,
    #                  "tier": "Balanced" by default, "latency_ms": cost known before running (optional)}}
    s = st.session_state
    key = f"l{level}_mode"
    tried = s.setdefault(f"l{level}_tried", [])
    if s.get(key) == correct:
        ui.message(f"<span class='mono'>MODE · {correct.upper()}</span> &nbsp; {options[correct]['verdict']}", "ok")
        return True

    st.markdown(f'<div class="gl-choice"><div class="gl-kicker">What do you need to know?</div>{question}</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(options))
    for col, (mode, opt) in zip(cols, options.items()):
        with col:
            st.markdown(f'<div class="gl-modecard"><div class="name">{mode.upper()}</div>{opt["blurb"]}</div>',
                        unsafe_allow_html=True)
            if st.button(f"Use {mode}", key=f"{key}_{mode}", width="stretch", disabled=mode in tried):
                tried.append(mode)
                if mode == correct:
                    s[key] = mode
                    st.rerun()
                try_mode(level, opt)
                s[f"{key}_last"] = mode
                st.rerun()

    last = s.get(f"{key}_last")
    if last and last != correct:
        wrong_mode_result(level, last, options[last], needs)
    return False


def try_mode(level: int, opt: dict) -> None:
    s = st.session_state
    key = f"l{level}_mode"
    s.pop(f"{key}_cost", None)
    s.pop(f"{key}_blocked", None)
    run = opt.get("run")
    if not run:
        return
    ms = opt.get("latency_ms")
    if ms is not None and not device.can_run(s, energy_units(ms), opt.get("tier", "Balanced")):
        s[f"{key}_blocked"] = energy_units(ms)    # can't pay for it, so it isn't run
        return
    label, ms = run()
    s[f"{key}_cost"] = (label, ms, run_cost(level, label, ms))


def asked_line(mode: str, needs: str) -> str:
    """'You asked for X. This question needs Y.'"""
    asked = f"You asked for {MODE_OUTPUT.get(mode, mode)}."
    return f"{asked} This question needs {needs}." if needs else asked


def wrong_mode_result(level: int, mode: str, opt: dict, needs: str = "") -> None:
    s = st.session_state
    key = f"l{level}_mode"
    if units := s.get(f"{key}_blocked"):
        ui.message(f"{asked_line(mode, needs)} {opt['verdict']} GhostLens didn't run {mode} mode: not enough "
                   f"charge to run it (would cost {pct(units)}).", "bad")
        return
    st.markdown(f'<div class="gl-wrongmode"><div class="gl-kicker">GhostLens ran {mode} mode</div>'
                f'{asked_line(mode, needs)}</div>', unsafe_allow_html=True)
    if opt.get("show"):
        st.markdown('<div class="gl-kicker">GhostLens returned</div>', unsafe_allow_html=True)
        opt["show"]()
    if cost := s.get(f"{key}_cost"):
        cost_line(*cost)
    ui.message(opt["verdict"], "bad")


def mode_misses(level: int) -> int:
    return len([m for m in st.session_state.get(f"l{level}_tried", []) if m != st.session_state.get(f"l{level}_mode")])


def card_state(p: dict, slot: str, target_ms: float) -> tuple[bool, str, bool]:
    """(locked, badge html, locked for lack of power) for one model card."""
    s = st.session_state
    units = energy_units(p["latency_ms"])
    free = device.free_memory(s) + s.resident.get(slot, (None, 0.0))[1]
    heavy_off = p["tier"] == "Heavy" and device.low_power(s)
    short = p["tier"] != "Light" and not device.can_afford(s, units)
    if not device.fits(s, p["size_mb"], replacing=slot):
        return True, (f'<div class="gl-badge bad"><b>DOES NOT FIT</b><br>'
                      f'needs {p["size_mb"]:.1f} MB · {free:.1f} MB free</div>'), False
    if heavy_off:
        return True, '<div class="gl-badge bad">LOW POWER · heavy models disabled</div>', True
    if short:
        return True, f'<div class="gl-badge bad">NEEDS {pct(units)} · {pct(s.battery)} left</div>', True
    if not device.can_afford(s, units):
        return False, '<div class="gl-badge warn">EMERGENCY RESERVE · light models still run</div>', False
    if p["latency_ms"] > target_ms:
        return False, (f'<div class="gl-badge warn">over target · {p["latency_ms"]:.0f} ms &gt; '
                       f'{target_ms:.0f} ms</div>'), False
    return False, "", False


def model_card_html(p: dict, badge: str, selected: bool, locked: bool) -> str:
    intel = f'<div class="intel">FIELD INTEL · {p["intel"]}</div>' if p.get("intel") else ""
    return (
        f'<div class="gl-modelcard{" selected" if selected else ""}{" nofit" if locked else ""}">'
        f'<div class="gl-kicker">{p["tier"]}</div><div class="name">{p["name"]}</div>'
        f'<table class="mono">'
        f'<tr><td>size</td><td>{p["size_mb"]:.2f} MB</td></tr>'
        f'<tr><td>latency</td><td>{p["latency_ms"]:.0f} ms</td></tr>'
        f'<tr><td>{p.get("metric", "accuracy")}</td><td>{p["accuracy"]}</td></tr>'
        f'<tr><td>battery/run</td><td>{pct(energy_units(p["latency_ms"]))}</td></tr></table>'
        f'<div class="tags">{ui.tag("measured")} size, latency &nbsp;{ui.tag(p["accuracy_tag"])} '
        f'{p.get("metric", "accuracy")} '
        f'&nbsp;{ui.tag("gameplay")} battery</div>{intel}{badge}</div>'
    )


def model_picker(level: int, profiles: list[dict], limits: dict, slot: str, note: str = "") -> dict | None:
    # profile keys: id, name, tier, size_mb, latency_ms, accuracy, accuracy_tag, intel (optional).
    # Returns the profile once loaded.
    s = st.session_state
    key = f"l{level}_model"
    chosen = next((p for p in profiles if p["id"] == s.get(key)), None)
    free = device.free_memory(s) + s.resident.get(slot, (None, 0.0))[1]

    st.markdown(
        f'<div class="gl-limits"><span class="gl-kicker">Mission limits</span>'
        f'<span>latency ≤ {limits["latency_ms"]:.0f} ms</span><span>memory for this model {free:.1f} MB</span>'
        f'<span>battery {pct(s.battery)}</span></div>', unsafe_allow_html=True)
    if note:
        st.caption(note)

    power_locked = False
    cols = st.columns(len(profiles))
    for col, p in zip(cols, profiles):
        locked, badge, no_power = card_state(p, slot, limits["latency_ms"])
        power_locked |= no_power
        selected = bool(chosen and chosen["id"] == p["id"])
        with col:
            st.markdown(model_card_html(p, badge, selected, locked), unsafe_allow_html=True)
            if st.button("Loaded" if selected else f"Load {p['tier'].lower()}", key=f"{key}_{p['id']}",
                         disabled=locked or selected, width="stretch"):
                s[key] = p["id"]
                s.setdefault(f"l{level}_models_tried", []).append(p["id"])
                device.load_model(s, slot, p["name"], p["size_mb"])
                st.rerun()
    if power_locked and device.charger_available(s):
        charge_prompt(level, f"l{level}_picker")
    return chosen


def check_rows(checks: dict[str, tuple[bool, str]]) -> str:
    return "".join(
        f"<tr><td>{name}</td><td class='{'ok' if ok else 'bad'}'>{'✓' if ok else '✗'}</td><td>{detail}</td></tr>"
        for name, (ok, detail) in checks.items()
    )


def checklist(title: str, checks: dict[str, tuple[bool, str]]) -> None:
    st.markdown(f'<div class="gl-checklist"><div class="gl-kicker">{title}</div>'
                f'<table class="mono">{check_rows(checks)}</table></div>', unsafe_allow_html=True)


def mission_report(level: int, checks: dict[str, tuple[bool, str]], grade: str) -> None:
    rows = check_rows(checks)
    cell = f'<div class="gl-cell">SPARE CELL +{pct(device.A_GRADE_UNITS)}</div>' if grade == "A" else ""
    st.markdown(
        f'<div class="gl-report"><div><div class="grade g{grade}">{grade}</div>{cell}</div>'
        f'<div><div class="gl-kicker">Chapter {level} · mission report · edge engineering grade</div>'
        f'<table class="mono">{rows}</table></div></div>',
        unsafe_allow_html=True,
    )


def predict(key: str, question: str, options: list[str], resolve: Callable, reveal: Callable,
            *, ready: bool = True, xp: int = 5) -> dict | None:
    """Ask for a guess, lock it, then show the measured answer. resolve() -> (correct index or set, why)."""
    s = st.session_state
    record = s.get(key)
    if record is None:
        st.markdown(f'<div class="gl-predict"><div class="gl-kicker">Predict first</div>{question}</div>',
                    unsafe_allow_html=True)
        choice = st.radio(question, options, index=None, key=f"{key}_choice", label_visibility="collapsed")
        if st.button("Lock in", key=f"{key}_lock", disabled=choice is None):
            s[key] = {"q": question, "options": list(options), "choice": options.index(choice),
                      "correct": None, "why": "", "right": None}
            st.rerun()
        return None

    guess = record["options"][record["choice"]]
    if not ready:
        ui.message(f"Your guess: <b>{guess}</b>. You'll see if you were right after the next step.")
        return record
    if record["right"] is None:
        correct, why = resolve()
        correct = set(correct) if isinstance(correct, (set, list, tuple)) else {correct}
        record.update(correct=sorted(correct), why=why, right=record["choice"] in correct)
        if record["right"]:
            pot = "guide_xp" if key.startswith("lab_") else "xp"   # the lab is off the clock
            s[pot] = s.get(pot, 0) + xp
    answer = " / ".join(record["options"][i] for i in record["correct"])
    if record["right"]:
        ui.message(f"<b>You called it.</b> {record['why']}", "ok")
    else:
        ui.message(f"You guessed <b>{guess}</b>, the measurement says <b>{answer}</b>. {record['why']}", "warn")
    reveal()
    return record


def predictions(store) -> list[dict]:
    return [v for k, v in store.items() if "_pred_" in k and isinstance(v, dict) and "choice" in v]
