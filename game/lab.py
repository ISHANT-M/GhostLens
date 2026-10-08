"""GhostLens Lab: convolution, layer cost, activations and optimizers, no battery cost."""

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from cv import lab
from cv import segmentation as seg
from cv.models import ModelMissing
from game import runtime
from ui import components as ui

ROOT = seg.ROOT
IMAGES = {"Padlock": "assets/level2/padlock.jpg", "Teddy bear": "assets/level2/teddy_bear.jpg",
          "Chapter 4 wall": None}
COLORS = ["#5E7D5A", "#B7862F", "#9C4A3C", "#22211F", "#5B5750"]
STARTS = {"bowl": {"Far left": (-4.0, 2.0), "Top right": (4.0, 2.5), "Steep wall": (-1.0, 3.0)},
          "rosenbrock": {"Classic (-1.5, 2)": (-1.5, 2.0), "Bottom left": (-1.8, -1.0), "Near the valley": (0.0, 0.0)}}
RANGES = {"bowl": ((-5, 5), (-3.5, 3.5)), "rosenbrock": ((-2, 2), (-1.5, 3))}


def chart(c: alt.Chart) -> None:
    st.altair_chart(c.configure_view(stroke=None).configure(background="transparent"), width="stretch")


def formula(text: str) -> None:
    st.markdown(f'<div class="gl-formula">{text}</div>', unsafe_allow_html=True)


def kb(n: float) -> str:
    return f"{n / 1024:,.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:,.2f} MB"


@st.cache_data(show_spinner=False)
def gray_image(name: str) -> np.ndarray:
    path = IMAGES[name]
    img = seg.make_scene()[0] if path is None else cv2.imread(str(ROOT / path))
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def to_view(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def custom_kernel() -> list[list[float]]:
    st.caption("Custom 3×3 kernel")
    rows = []
    for r in range(3):
        cols = st.columns(3)
        rows.append([cols[c].number_input(f"k{r}{c}", value=1.0 if (r, c) == (1, 1) else 0.0, step=0.5,
                                          label_visibility="collapsed", key=f"lab_k{r}{c}") for c in range(3)])
    return rows


def conv_controls() -> dict:
    c1, c2 = st.columns(2)
    image = c1.selectbox("Evidence image", list(IMAGES), key="lab_img")
    preset = c2.selectbox("Kernel", list(lab.KERNELS) + ["Custom"], key="lab_kernel")
    c3, c4, c5, c6 = st.columns(4)
    opts = {"image": image,
            "stride": c3.radio("Stride", [1, 2], horizontal=True, key="lab_stride"),
            "padding": c4.radio("Padding", ["same", "valid"], horizontal=True, key="lab_pad"),
            "relu": c5.toggle("ReLU", key="lab_relu"),
            "pool": c6.toggle("2×2 MaxPool", key="lab_pool")}
    opts["kernel"] = custom_kernel() if preset == "Custom" else lab.KERNELS[preset]
    return opts


def conv_tab() -> None:
    o = conv_controls()
    stages = lab.apply_conv(gray_image(o["image"]), o["kernel"], o["stride"], o["padding"], o["relu"], o["pool"])
    cols = st.columns(len(stages))
    for col, (name, a) in zip(cols, stages.items()):
        col.image(to_view(a), width="stretch", clamp=True)
        with col:
            ui.caption(f"{name} · 1×{a.shape[0]}×{a.shape[1]}")
    p = 1 if o["padding"] == "same" else 0
    out = lab.conv_output_size(256, 3, p, o["stride"])
    formula(f"output = (W − K + 2P) / S + 1 = (256 − 3 + 2·{p}) / {o['stride']} + 1 = {out}")
    st.markdown("Bright pixels are where the kernel pattern matches strongly. Display is rescaled per image.")
    unet_maps()


@st.cache_data(show_spinner=False)
def enc1_maps() -> np.ndarray:
    model = runtime.unet("standard", False)
    with torch.no_grad():
        maps = model.enc1(seg.to_tensor(seg.make_scene()[0])[None])[0]
    return maps.numpy()


def unet_maps() -> None:
    st.subheader("What the U-Net sees", anchor=False)
    try:
        maps = enc1_maps()
    except ModelMissing as e:
        st.warning(str(e))
        return
    st.markdown(f"The first block of the Standard U-Net has {maps.shape[0]} filters. Each one gives a feature map: "
                "a picture of where its learned pattern shows up on the chapter 4 wall. Here are 8 of them "
                + ui.tag("measured"), unsafe_allow_html=True)
    for row in range(2):
        cols = st.columns(4)
        for i, col in enumerate(cols):
            idx = row * 4 + i
            col.image(to_view(maps[idx]), width="stretch", clamp=True)
            with col:
                ui.caption(f"map {idx} · {maps.shape[1]}×{maps.shape[2]}")
    ui.lesson(["A kernel is a small grid of weights slid over the image. The U-Net learns its kernels instead of us picking them.",
               "Early maps pick up edges and brightness. Deeper layers combine them into shapes like the stain.",
               "Stride and pooling shrink the map, which cuts compute for every later layer."], title="FIELD NOTES")


def layer_calc() -> None:
    c = st.columns(5)
    c_in = c[0].number_input("In channels", 1, 1024, 3, key="lab_cin")
    c_out = c[1].number_input("Out channels", 1, 1024, 16, key="lab_cout")
    k = c[2].selectbox("Kernel size", [1, 3, 5, 7], index=1, key="lab_ks")
    hw = c[3].number_input("Input H = W", 8, 1024, 128, step=8, key="lab_hw")
    s = c[4].selectbox("Stride", [1, 2], key="lab_s")
    r = lab.conv_layer_stats(int(c_in), int(c_out), k, int(hw), int(hw), s)
    formula(f"params = K·K·Cin·Cout + Cout = {k}·{k}·{c_in}·{c_out} + {c_out} = {r['params']:,}")
    ui.readouts([("PARAMS", f"{r['params']:,}", ""), ("OUTPUT", "×".join(map(str, r["out_shape"])), ""),
                 ("MACs", f"{r['macs'] / 1e6:,.2f} M", ""), ("WEIGHTS FP32", kb(r["weights_fp32"]), ""),
                 ("FP16", kb(r["weights_fp16"]), ""), ("INT8", kb(r["weights_int8"]), "ok"),
                 ("ACTIVATIONS FP32", kb(r["activations_fp32"]), "warn")])


@st.cache_data(show_spinner=False)
def unet_table(width: int) -> pd.DataFrame:
    return pd.DataFrame(lab.unet_layer_table(seg.TinyUNet(width)))


def params_tab() -> None:
    layer_calc()
    st.subheader("TinyUNet layer by layer", anchor=False)
    variant = st.radio("Variant", list(seg.VARIANTS), index=1, horizontal=True, key="lab_variant",
                       format_func=str.title)
    width = seg.VARIANTS[variant]
    df = unet_table(width)
    total = int(df["params"].sum())
    assert total == sum(p.numel() for p in seg.TinyUNet(width).parameters())
    st.dataframe(df, hide_index=True, width="stretch")
    ui.readouts([("TOTAL PARAMS", f"{total:,}", ""), ("FP32", kb(total * 4), ""),
                 ("INT8", kb(total), "ok"), ("INPUT", "3×128×128", "")])
    ui.message("On an edge device the weights are stored once, but every layer's output activations also need RAM "
               "while it runs, and at full resolution those are often bigger than the weights.")


@st.cache_data(show_spinner=False)
def activation_frame(slope: float) -> pd.DataFrame:
    rows = []
    for name in ["ReLU", "Leaky ReLU", "Sigmoid", "Tanh", "GELU"]:
        x, y, g = lab.activation_curves(name, slope)
        rows.append(pd.DataFrame({"x": x, "value": y, "derivative": g, "function": name}))
    return pd.concat(rows)


def activation_chart(df: pd.DataFrame, field: str, title: str) -> alt.Chart:
    scale = alt.Scale(domain=["ReLU", "Leaky ReLU", "Sigmoid", "Tanh", "GELU"], range=COLORS)
    return alt.Chart(df, title=title).mark_line(strokeWidth=2).encode(
        x=alt.X("x:Q"), y=alt.Y(f"{field}:Q", title=field), color=alt.Color("function:N", scale=scale))


def activations_tab() -> None:
    slope = st.slider("Leaky ReLU slope", 0.0, 0.5, 0.1, 0.01, key="lab_slope")
    df = activation_frame(slope)
    c1, c2 = st.columns(2)
    with c1:
        chart(activation_chart(df, "value", "f(x)"))
    with c2:
        chart(activation_chart(df, "derivative", "f'(x) from autograd"))
    ui.lesson(["Sigmoid and tanh flatten out for large |x|, so their derivative goes near 0. Multiply many of those "
               "in backprop and the gradient vanishes.",
               "ReLU has derivative 1 for positive inputs, so gradients pass through. It is just max(0, x), "
               "one compare per value, which is cheap on edge chips and easy to quantize.",
               "Leaky ReLU keeps a small slope for negatives so neurons do not get stuck at 0. GELU is smooth but costs more."],
              title="FIELD NOTES")


@st.cache_data(show_spinner=False)
def run_optimizers(loss_name: str, lr: float, steps: int, start: tuple) -> pd.DataFrame:
    paths = lab.optimizer_paths(loss_name, lr, steps, start)
    return pd.concat(pd.DataFrame({"x": p[:, 0], "y": p[:, 1], "step": np.arange(len(p)), "optimizer": name})
                     for name, p in paths.items())


def optimizer_chart(loss_name: str, paths: pd.DataFrame) -> alt.Chart:
    (x0, x1), (y0, y1) = RANGES[loss_name]
    n = 40
    grid = pd.DataFrame(lab.loss_grid(loss_name, (x0, x1), (y0, y1), n))
    dx, dy = (x1 - x0) / (n - 1), (y1 - y0) / (n - 1)
    grid = grid.assign(x2=grid.x + dx, y2=grid.y + dy)
    heat = alt.Chart(grid).mark_rect().encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[x0, x1]), title="x"), x2="x2", y=alt.Y("y:Q", scale=alt.Scale(domain=[y0, y1]), title="y"),
        y2="y2", color=alt.Color("loss:Q", scale=alt.Scale(range=["#FBF9F4", "#5B5750"]), title="log loss"))
    clipped = paths[paths.x.between(x0, x1) & paths.y.between(y0, y1)]
    lines = alt.Chart(clipped).mark_line(point=alt.OverlayMarkDef(size=12), strokeWidth=2).encode(
        x="x:Q", y="y:Q", order="step:Q",
        stroke=alt.Stroke("optimizer:N", scale=alt.Scale(domain=list(lab.OPTIMIZERS), range=COLORS[:4])))
    return (heat + lines).properties(height=420).resolve_scale(color="independent")


def final_table(loss_name: str, paths: pd.DataFrame) -> pd.DataFrame:
    last = paths.groupby("optimizer").tail(1)
    return pd.DataFrame({"optimizer": last.optimizer, "steps run": last.step,
                         "final x": last.x.round(3), "final y": last.y.round(3),
                         "final loss": [lab.loss_value(loss_name, (x, y)) for x, y in zip(last.x, last.y)]})


def optimizers_tab() -> None:
    c1, c2 = st.columns(2)
    loss_name = c1.radio("Loss", ["bowl", "rosenbrock"], horizontal=True, key="lab_loss",
                         format_func={"bowl": "Elongated bowl", "rosenbrock": "Rosenbrock"}.get)
    start_name = c2.selectbox("Start point", list(STARTS[loss_name]), key=f"lab_start_{loss_name}")
    c3, c4 = st.columns(2)
    lr = c3.select_slider("Learning rate", [0.0005, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.18, 0.3],
                          value=0.03, key="lab_lr")
    steps = c4.slider("Steps", 20, 300, 100, 10, key="lab_steps")
    paths = run_optimizers(loss_name, lr, steps, STARTS[loss_name][start_name])
    chart(optimizer_chart(loss_name, paths))
    st.dataframe(final_table(loss_name, paths), hide_index=True, width="stretch")
    ui.lesson(["SGD: step straight down the gradient. Zig-zags across a narrow valley.",
               "Momentum: keeps a running velocity, so it speeds up along the valley and smooths the zig-zag.",
               "RMSprop: divides each step by a running size of that coordinate's gradient, so steep and flat directions move evenly.",
               "Adam: momentum plus RMSprop scaling. The default we used to train the U-Net.",
               "A path that stops early blew up: the learning rate was too big for that loss."], title="FIELD NOTES")


def render() -> None:
    ui.scene_header("GHOSTLENS LAB · HQ WORKBENCH", "The Lab",
                    "Back at HQ, no battery is used. Try the building blocks behind the models from the case.")
    tabs = st.tabs(["Convolution", "Parameters & memory", "Activations", "Optimizers"])
    for tab, fn in zip(tabs, [conv_tab, params_tab, activations_tab, optimizers_tab]):
        with tab:
            fn()
