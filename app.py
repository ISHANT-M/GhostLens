import os

os.environ.setdefault("YOLO_OFFLINE", "1")  # no analytics or downloads from ultralytics while playing

import streamlit as st

from game import case, codex, device, flow, lab, levels, nav, state
from ui import components as ui

st.set_page_config(page_title="GhostLens", page_icon="◉", layout="wide")
state.init_state(st.session_state)
device.init_device(st.session_state)
ui.load_css()

# ?case=1234 pins the case for presentations, only if this session hasn't started one yet
if "case_seed" not in st.session_state and st.query_params.get("case", "").isdigit():
    st.session_state.case_seed = int(st.query_params["case"])

# hidden looping audio. Browsers only allow sound after a click, so it also starts on the next click anywhere.
AUDIO = """
<audio id="amb" src="/app/static/ambience.wav" loop preload="auto"></audio>
<script>
  const a = document.getElementById("amb");
  a.volume = 0.55;
  const start = () => a.play().catch(() => {});
  start();
  window.parent.document.addEventListener("click", start, { once: true });
</script>
"""


def restart_case() -> None:
    # a new case, not a new detective: tips, quiz answers and guide XP stay
    s = st.session_state
    state.reset_progress(s)
    device.reset_device(s)
    case.reset_case(s)
    for key in [k for k in s if k.startswith("l") and k[1:2].isdigit()]:
        del s[key]
    s.pop("loaded_chapters", None)
    s.pop("grades", None)


def lobby_popover() -> None:
    s = st.session_state
    if not device.charger_available(s):
        return
    n = device.active_level(s, s.current_level)
    with st.popover("Lobby charger", width="stretch"):
        st.markdown(f"Walk back down to the lobby and plug GhostLens in. You get "
                    f"**+{device.pct(device.CHARGER_UNITS)}** battery and lose **{device.CHARGER_XP} XP**. "
                    f"The trip goes on the Chapter {n} report, so that chapter can't get an A.")
        if st.button(f"Walk to the lobby · +{device.pct(device.CHARGER_UNITS)} for −{device.CHARGER_XP} XP",
                     key="lobby_sidebar", width="stretch"):
            device.lobby_charge(s, s.current_level)
            st.rerun()


def level_page(n: int):
    def render():
        if not state.is_unlocked(st.session_state, n):
            levels.locked_screen(n)
            return
        loaded = st.session_state.setdefault("loaded_chapters", set())
        if n not in loaded:
            info = levels.LEVELS[n]
            flow.loading_screen(n, info["title"], info["mode"], levels.warmup(n))
            loaded.add(n)
        st.session_state.current_level = n
        hud.markdown(flow.hud_html(n), unsafe_allow_html=True)
        levels.render_level(n)
    render.__name__ = f"chapter_{n}"
    return render


home = st.Page(levels.home, title="Case file", url_path="home", default=True)
level_pages = {
    n: st.Page(level_page(n), title=f"{n}. {info['title']}", url_path=f"level{n}")
    for n, info in levels.LEVELS.items()
}
lab_page = st.Page(lab.render, title="Lab", url_path="lab")
guide = st.Page(codex.render, title="Field guide", url_path="guide")
about = st.Page(levels.about, title="About", url_path="about")
nav.PAGES.update({"home": home, "lab": lab_page, "guide": guide, "about": about, **level_pages})
page = st.navigation([home, *level_pages.values(), lab_page, guide, about], position="hidden")

with st.sidebar:
    header = st.empty()
    st.page_link(home, label="Case file")
    for n, p in level_pages.items():
        status = state.status(st.session_state, n)
        c1, c2 = st.columns([4, 2], vertical_alignment="center")
        with c1:
            st.page_link(p, label=f"{n}. {levels.LEVELS[n]['title']}", disabled=status == "LOCKED")
        with c2:
            st.markdown(ui.stamp(status), unsafe_allow_html=True)
    st.markdown('<div class="gl-kicker" style="margin-top:0.8rem">Off the clock</div>', unsafe_allow_html=True)
    st.page_link(lab_page, label="GhostLens Lab")
    st.page_link(guide, label="Field guide")
    st.page_link(about, label="About")
    panel = st.empty()
    st.divider()
    st.toggle("Demo mode", key="demo_mode", help="Unlocks every chapter. Handy for presentations.")
    if st.toggle("Sound", key="sound", value=True):
        with st.container(key="audio"):
            st.iframe(AUDIO, height=1)
    if st.button("Restart the case", type="tertiary", help="New case. The field guide is kept."):
        restart_case()
        st.switch_page(home)

hud = st.empty()
page.run()

# filled after the page ran, so battery and XP include whatever just happened
s = st.session_state
header.markdown(
    f'<div class="gl-casefile"><b>GHOSTLENS</b> · {case.case_label(s)}<br>'
    f"SOLVED {len(s.completed_levels)}/{state.LEVEL_COUNT} · CLUES {len(s.clues_found)} "
    f"+ SIDE {len(s.side_clues)}/4<br>XP {s.xp} · GUIDE XP {s.get('guide_xp', 0)}</div>",
    unsafe_allow_html=True,
)
with panel.container():
    st.markdown(flow.device_panel_html(), unsafe_allow_html=True)
    lobby_popover()
flow.low_power_toast()
if s.current_level and page.url_path.startswith("level") and state.is_unlocked(s, s.current_level):
    hud.markdown(flow.hud_html(s.current_level), unsafe_allow_html=True)
