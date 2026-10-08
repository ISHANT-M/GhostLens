"""Shared chapter parts: loading screen, HUD, mode choice, model choice and mission report."""

import time
from collections.abc import Callable

import streamlit as st

from cv.edge import energy_units
from game import device, state
from game.tips import random_tip
from ui import components as ui

MODES = ["Enhance", "Classify", "Detect", "Segment"]
MIN_LOADING_SECONDS = 2.8


def loading_screen(chapter: int, title: str, mode: str, work: list[tuple[str, Callable]]) -> None:
    seen = st.session_state.setdefault("seen_tips", set())
    i, topic, tip = random_tip(seen)
    seen.add(i)
    box = st.empty()
    with box.container(key="loading"):
        st.markdown(
            f'<div class="gl-loading"><div class="chapter">CHAPTER {chapter} / {state.LEVEL_COUNT}</div>'
            f'<div class="gl-loading-title">{title}</div>'
            f'<div class="mode">GhostLens · preparing {mode} mode</div></div>',
            unsafe_allow_html=True,
        )
        bar = st.progress(0.0)
        st.markdown(f'<div class="gl-tip"><div class="title">FIELD TIP · {topic.upper()}</div>{tip}</div>',
                    unsafe_allow_html=True)
        start = time.time()
        for k, (label, fn) in enumerate(work):
            bar.progress(k / max(len(work), 1), text=label)
            fn()
        while (elapsed := time.time() - start) < MIN_LOADING_SECONDS:
            bar.progress(min(1.0, elapsed / MIN_LOADING_SECONDS), text="Ready")
            time.sleep(0.08)
        bar.progress(1.0, text="Ready")
        time.sleep(0.3)
    box.empty()


def hud_html(chapter: int) -> str:
    s = st.session_state
    chips = []
    for n, mode in enumerate(MODES, start=1):
        cls = "active" if n == chapter else "done" if n in s.completed_levels else \
            "open" if state.is_unlocked(s, n) else "locked"
        chips.append(f'<span class="gl-chip {cls}">{mode.upper()}</span>')
    tone = "bad" if device.low_power(s) else ""
    return (f'<div class="gl-hud"><span>GHOSTLENS MK.II · CH {chapter}/{state.LEVEL_COUNT}</span>'
            f'<span class="modes">{"".join(chips)}</span>'
            f'<span class="{tone}">BATTERY {s.battery}% · XP {s.xp}</span></div>')


def device_panel_html() -> str:
    s = st.session_state
    pct = s.battery / device.BATTERY_START
    color = "var(--bad)" if device.low_power(s) else "var(--warn)" if pct < 0.5 else "var(--ok)"
    used = device.memory_used(s)
    loaded = ", ".join(name for name, _ in s.resident.values()) or "nothing loaded"
    last = s.runs[-1] if s.runs else None
    last_html = (f"<div class='row'><span>Last run</span><span>{last['what']}</span></div>"
                 f"<div class='row sub'><span>{last['ms']:.0f} ms measured</span><span>−{last['units']} units</span></div>"
                 if last else "")
    mode = "LOW POWER" if device.low_power(s) else "NORMAL"
    return (
        f'<div class="gl-device"><div class="head">GHOSTLENS MK.II <span>{mode}</span></div>'
        f'<div class="row"><span>Battery</span><span>{s.battery} / {device.BATTERY_START}</span></div>'
        f'<div class="meter"><span style="width:{pct * 100:.0f}%;background:{color}"></span></div>'
        f'<div class="row"><span>Model memory</span><span>{used:.1f} / {device.MEMORY_MB:.0f} MB</span></div>'
        f'<div class="meter"><span style="width:{min(used / device.MEMORY_MB, 1) * 100:.0f}%"></span></div>'
        f'<div class="row sub"><span>{loaded}</span></div>{last_html}</div>'
    )


def run_cost(level: int, what: str, latency_ms: float) -> int:
    units = energy_units(latency_ms)
    device.spend(st.session_state, level, what, latency_ms, units)
    return units


def cost_line(what: str, latency_ms: float, units: int) -> None:
    st.markdown(
        f'<div class="gl-cost">{ui.tag("measured")} {what}: {latency_ms:.0f} ms &nbsp;→&nbsp; '
        f'{ui.tag("gameplay")} −{units} battery units</div>', unsafe_allow_html=True)


def mode_choice(level: int, question: str, options: dict[str, dict], correct: str) -> bool:
    # options: {mode: {"blurb", "verdict", "run": () -> (label, latency_ms) or None, "show": callable(result)}}
    key = f"l{level}_mode"
    tried = st.session_state.setdefault(f"l{level}_tried", [])
    if st.session_state.get(key) == correct:
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
                    st.session_state[key] = mode
                    st.rerun()
                run = opt.get("run")
                if run:
                    label, ms = run()
                    st.session_state[f"{key}_cost"] = (label, ms, run_cost(level, label, ms))
                else:
                    st.session_state.pop(f"{key}_cost", None)
                st.session_state[f"{key}_last"] = mode
                st.rerun()

    last = st.session_state.get(f"{key}_last")
    if last and last != correct:
        st.markdown(f'<div class="gl-kicker" style="margin-top:0.9rem">GhostLens ran {last} mode</div>',
                    unsafe_allow_html=True)
        if options[last].get("show"):
            options[last]["show"]()
        if cost := st.session_state.get(f"{key}_cost"):
            cost_line(*cost)
        ui.message(options[last]["verdict"], "bad")
    return False


def mode_misses(level: int) -> int:
    return len([m for m in st.session_state.get(f"l{level}_tried", []) if m != st.session_state.get(f"l{level}_mode")])


def model_picker(level: int, profiles: list[dict], limits: dict, slot: str, note: str = "") -> dict | None:
    # profile keys: id, name, tier, size_mb, latency_ms, accuracy, accuracy_tag. Returns the profile once loaded.
    s = st.session_state
    key = f"l{level}_model"
    chosen = next((p for p in profiles if p["id"] == s.get(key)), None)
    free = device.free_memory(s) + (s.resident.get(slot, (None, 0.0))[1])

    st.markdown(
        f'<div class="gl-limits"><span class="gl-kicker">Mission limits</span>'
        f'<span>latency ≤ {limits["latency_ms"]:.0f} ms</span><span>free model memory {free:.1f} MB</span>'
        f'<span>battery {s.battery} units</span></div>', unsafe_allow_html=True)
    if note:
        st.caption(note)

    cols = st.columns(len(profiles))
    for col, p in zip(cols, profiles):
        fits = p["size_mb"] <= free
        empty = s.battery == 0 and p["tier"] != "Light"
        locked = empty or device.low_power(s) and p["tier"] == "Heavy"
        target = limits["latency_ms"]
        units = energy_units(p["latency_ms"])
        badge = ""
        if not fits:
            badge = (f'<div class="gl-badge bad"><b>DOES NOT FIT</b><br>'
                     f'needs {p["size_mb"]:.1f} MB · {free:.1f} MB free</div>')
        elif empty:
            badge = '<div class="gl-badge bad">EMERGENCY RESERVE · light models only</div>'
        elif locked:
            badge = '<div class="gl-badge bad">LOW POWER · heavy models disabled</div>'
        elif p["latency_ms"] > target:
            badge = f'<div class="gl-badge warn">over target · {p["latency_ms"]:.0f} ms &gt; {target:.0f} ms</div>'
        selected = chosen and chosen["id"] == p["id"]
        with col:
            st.markdown(
                f'<div class="gl-modelcard{" selected" if selected else ""}{" nofit" if not fits or locked else ""}">'
                f'<div class="gl-kicker">{p["tier"]}</div><div class="name">{p["name"]}</div>'
                f'<table class="mono">'
                f'<tr><td>size</td><td>{p["size_mb"]:.2f} MB</td></tr>'
                f'<tr><td>latency</td><td>{p["latency_ms"]:.0f} ms</td></tr>'
                f'<tr><td>accuracy</td><td>{p["accuracy"]}</td></tr>'
                f'<tr><td>energy</td><td>{units} units/run</td></tr></table>'
                f'<div class="tags">{ui.tag("measured")} size, latency &nbsp;{ui.tag(p["accuracy_tag"])} accuracy '
                f'&nbsp;{ui.tag("gameplay")} energy</div>{badge}</div>',
                unsafe_allow_html=True,
            )
            if st.button("Loaded" if selected else f"Load {p['tier'].lower()}", key=f"{key}_{p['id']}",
                         disabled=not fits or locked or bool(selected), width="stretch"):
                s[key] = p["id"]
                s.setdefault(f"l{level}_models_tried", []).append(p["id"])
                device.load_model(s, slot, p["name"], p["size_mb"])
                st.rerun()
    return chosen


def mission_report(level: int, checks: dict[str, tuple[bool, str]], grade: str) -> None:
    rows = "".join(
        f"<tr><td>{name}</td><td class='{'ok' if ok else 'bad'}'>{'✓' if ok else '✗'}</td><td>{detail}</td></tr>"
        for name, (ok, detail) in checks.items()
    )
    st.markdown(
        f'<div class="gl-report"><div class="grade g{grade}">{grade}</div>'
        f'<div><div class="gl-kicker">Chapter {level} · mission report · edge engineering grade</div>'
        f'<table class="mono">{rows}</table></div></div>',
        unsafe_allow_html=True,
    )
