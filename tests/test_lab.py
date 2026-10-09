import numpy as np
import pytest
import torch
import torch.nn as nn

from cv import lab
from cv.segmentation import TinyUNet


@pytest.mark.parametrize("w,k,p,s", [(256, 3, 1, 1), (256, 3, 0, 2), (31, 5, 2, 2), (64, 1, 0, 1)])
def test_conv_output_size_matches_torch(w, k, p, s):
    out = nn.Conv2d(1, 1, k, stride=s, padding=p)(torch.zeros(1, 1, w, w))
    assert out.shape[-1] == lab.conv_output_size(w, k, p, s)


def test_conv_layer_params():
    conv = nn.Conv2d(16, 32, 3)
    assert lab.conv_layer_stats(16, 32, 3, 64, 64)["params"] == sum(p.numel() for p in conv.parameters())


def test_unet_table_total():
    model = TinyUNet(16)
    rows = lab.unet_layer_table(model)
    assert sum(r["params"] for r in rows) == sum(p.numel() for p in model.parameters())


def test_adam_reduces_bowl_loss():
    path = lab.optimizer_path("Adam", "bowl", 0.1, 100, (-4.0, 2.0))
    assert lab.loss_value("bowl", path[-1]) < lab.loss_value("bowl", path[0])


def test_relu_derivative():
    _, _, grad = lab.activation_curves("ReLU", x=[2.0, -2.0])
    assert np.allclose(grad, [1.0, 0.0])


@pytest.mark.parametrize("w", [256, 255, 128, 7])
def test_pool_output_size_matches_torch(w):
    out = torch.nn.functional.max_pool2d(torch.zeros(1, 1, w, w), 2)
    assert out.shape[-1] == lab.pool_output_size(w)


def test_silu_is_in_the_activation_list():
    x, y, grad = lab.activation_curves("SiLU", x=[0.0, 2.0])
    assert "SiLU" in lab.ACTIVATIONS
    assert np.allclose(y, [0.0, 2.0 * torch.sigmoid(torch.tensor(2.0)).item()])
    assert np.isclose(grad[0], 0.5)


# the Lab page, headless

from streamlit.testing.v1 import AppTest  # noqa: E402

from cv.edge import BENCHMARK_FILE  # noqa: E402
from game import lab as lab_page  # noqa: E402

needs_models = pytest.mark.skipif(not BENCHMARK_FILE.exists(), reason="run setup_models.py")
SCRIPT = """
import streamlit as st
from game import lab
st.session_state.setdefault("case_seed", 3)
lab.render()
"""


def open_tab(tab: str) -> AppTest:
    at = AppTest.from_string(SCRIPT, default_timeout=180)
    at.session_state["lab_open"] = tab
    return at.run()


def test_every_tab_has_a_syllabus_line_and_a_renderer():
    assert set(lab_page.SYLLABUS) == set(lab_page.TABS) == set(lab_page.RENDER)


def test_open_lab_sets_the_deep_link():
    from unittest.mock import patch
    with patch.object(lab_page.st, "session_state", {}) as store:
        lab_page.open_lab("Pruning", switch=False)
        assert store["lab_open"] == "Pruning"


@needs_models
@pytest.mark.parametrize("tab", lab_page.TABS)
def test_each_bench_renders_alone_with_no_guessing(tab):
    at = open_tab(tab)
    assert not at.exception
    assert at.session_state["lab_seen"] == {tab}
    shown = [m.value for m in at.markdown if "gl-syllabus" in m.value]
    assert shown == [f'<div class="gl-syllabus">Syllabus: {lab_page.SYLLABUS[tab]}</div>']
    assert any("gl-try" in m.value for m in at.markdown)
    assert not any(b.label == "Lock in" for b in at.button)
    assert all(r.index is not None for r in at.radio)          # no unanswered radio waiting for a guess
    assert not any("gl-lesson" in m.value for m in at.markdown)
    if tab not in ("Pruning",):                                # pruning shows its notes after the run
        assert any(e.label == "Bench notes" for e in at.expander)


def test_try_line_for_every_bench():
    assert set(lab_page.TRY) == set(lab_page.TABS)


@needs_models
def test_tab_survives_a_widget_change():
    at = open_tab("Normalization")
    at.toggle(key="lab_init_bn").set_value(True).run()
    assert at.session_state["lab_tab"] == "Normalization" and not at.exception
    at.radio(key="lab_norm_mode").set_value(lab_page.ex.INPUT_MODES[1]).run()
    assert at.session_state["lab_tab"] == "Normalization" and not at.exception


@needs_models
def test_losses_bench_grows_the_true_mask_live():
    at = open_tab("Losses")
    assert at.radio(key="lab_mask").value == lab_page.GROWN
    assert "1.000" in " ".join(m.value for m in at.markdown)   # the true mask itself: IoU 1
    at.slider(key="lab_morph").set_value(5).run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.exception and "true mask +5 px" in " ".join(str(d.value) for d in at.dataframe)
    assert ">IoU<" in text
    at.radio(key="lab_mask").set_value("empty mask").run()
    assert not at.exception and at.dataframe


@needs_models
def test_pruning_button_runs_the_sweep():
    at = open_tab("Pruning")
    assert not at.dataframe
    at.button(key="lab_prune_go").click().run()
    assert not at.exception and at.dataframe and at.session_state["lab_pruned"] is True
    assert at.select_slider(key="lab_prune_view")
    at.select_slider(key="lab_prune_view").set_value(0.9).run()
    assert not at.exception and at.session_state["lab_tab"] == "Pruning"
