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
    # a new case, not a new detective: the field guide tips stay
    s = st.session_state
    state.reset_progress(s)
    device.reset_device(s)
    case.reset_case(s)
    for key in [k for k in s if k.startswith("l") and k[1:2].isdigit()]:
        del s[key]
    for key in ("loaded_chapters", "grades", "just_cleared", "case_closed_seen"):
        s.pop(key, None)


def level_page(n: int):
    def render():
        if not state.is_unlocked(st.session_state, n):
            levels.locked_screen(n)
            return
        loaded = st.session_state.setdefault("loaded_chapters", set())
        if n not in loaded:
            info = levels.LEVELS[n]
            flow.loading_screen(n, info["title"], levels.warmup(n))
            loaded.add(n)
        st.session_state.current_level = n
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


def lobby_charger() -> None:
    s = st.session_state
    if not device.charger_available(s):
        return
    n = device.active_level(s, s.current_level)
    st.caption(f"Lobby charger: +{device.pct(device.CHARGER_UNITS)} battery for {device.CHARGER_XP} XP. "
               f"The trip goes on the Chapter {n} report.")
    if st.button(f"Walk to the lobby · +{device.pct(device.CHARGER_UNITS)} for −{device.CHARGER_XP} XP",
                 key="lobby_menu", width="stretch"):
        device.lobby_charge(s, s.current_level)
        st.rerun()


def menu() -> None:
    s = st.session_state
    with st.popover("Menu"):
        st.markdown(f'<div class="gl-kicker">{case.case_label(s)}</div>', unsafe_allow_html=True)
        st.page_link(home, label="Case file")
        for n, p in level_pages.items():
            status = state.status(s, n)
            st.page_link(p, label=f"{n}. {levels.LEVELS[n]['title']} · {status.lower()}",
                         disabled=status == "LOCKED")
        st.page_link(lab_page, label="Lab")
        st.page_link(guide, label="Field guide")
        st.page_link(about, label="About")
        st.divider()
        st.toggle("Sound", key="sound", value=True)
        st.toggle("Demo mode", key="demo_mode", help="Unlocks every chapter. Handy for presentations.")
        lobby_charger()
        if st.button("Restart the case", type="tertiary", key="restart", help="New case. The field guide is kept."):
            restart_case()
            st.switch_page(home)


with st.container(key="hud", horizontal=True, vertical_alignment="center"):
    hud = st.empty()
    menu()
if st.session_state.get("sound", True):
    with st.container(key="audio"):
        st.iframe(AUDIO, height=1)

page.run()

# filled after the page ran, so battery and XP include whatever just happened
chapter = int(page.url_path[5:]) if page.url_path.startswith("level") else None
hud.markdown(flow.hud_html(chapter), unsafe_allow_html=True)
flow.low_power_toast()
