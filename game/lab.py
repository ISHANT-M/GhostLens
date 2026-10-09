"""GhostLens Lab: hands-on experiments for the building blocks behind the case, no battery cost."""

import time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from cv import experiments as ex
from cv import lab
from cv import segmentation as seg
from cv.models import ModelMissing, load_yolo
from game import case, flow, nav, runtime
from ui import components as ui

ROOT = seg.ROOT
IMAGES = {"Padlock": "assets/level2/padlock.jpg", "Teddy bear": "assets/level2/teddy_bear.jpg",
          "Chapter 4 wall": None}
COLORS = ["#5E7D5A", "#B7862F", "#8A8377", "#9C4A3C", "#22211F", "#5B5750"]
STARTS = {"bowl": {"Far left": (-4.0, 2.0), "Top right": (4.0, 2.5), "Steep wall": (-1.0, 3.0)},
          "rosenbrock": {"Classic (-1.5, 2)": (-1.5, 2.0), "Bottom left": (-1.8, -1.0), "Near the valley": (0.0, 0.0)}}
RANGES = {"bowl": ((-5, 5), (-3.5, 3.5)), "rosenbrock": ((-2, 2), (-1.5, 3))}

TABS = ["Convolution", "Parameters", "Activations", "Architectures", "Optimizers", "Losses",
        "Normalization", "Pruning", "Augmentation"]
SYLLABUS = {
    "Convolution": "module 3 · building blocks: convolution, stride, padding, pooling, tensor shapes",
    "Parameters": "module 3 · parameters and memory; module 2 · transfer learning",
    "Activations": "module 5 · activation functions",
    "Architectures": "module 3 · advanced architectures: skip connections, depthwise, inception, attention",
    "Optimizers": "module 5 · optimizers",
    "Losses": "module 5 · loss functions and metrics; module 6 · segmentation",
    "Normalization": "module 8 · image and feature normalization, weight initialization",
    "Pruning": "module 7 · pruning, quantization, memory and compute",
    "Augmentation": "module 4 · augmentation: Mixup and CutMix",
}
BUDGET_MS = 40              # the chapter 4 latency limit
PRUNE_AMOUNTS = [0.0, 0.3, 0.5, 0.7, 0.9]
VAL_WALLS, VAL_SEED = 50, 999
ARCH_MODELS = {"YOLO26n-cls": ("yolo26n-cls.pt", 224), "YOLO26s": ("yolo26s.pt", 640)}
MIX_A, MIX_B = "assets/level2/padlock.jpg", "assets/level2/teddy_bear.jpg"
PADLOCK_LABELS = {"padlock", "combination_lock"}


def open_lab(tab: str, switch: bool = True) -> None:
    """Open the Lab on one tab. Call it from a button body, not an on_click callback."""
    st.session_state["lab_open"] = tab
    if switch and "lab" in nav.PAGES:
        st.switch_page(nav.PAGES["lab"])


def chart(c: alt.Chart) -> None:
    st.altair_chart(c.configure_view(stroke=None).configure(background="transparent"), width="stretch")


def formula(text: str) -> None:
    st.markdown(f'<div class="gl-formula">{text}</div>', unsafe_allow_html=True)


def syllabus(tab: str) -> None:
    st.markdown(f'<div class="gl-syllabus">Syllabus: {SYLLABUS[tab]}</div>', unsafe_allow_html=True)


def kb(n: float) -> str:
    return f"{n / 1024:,.1f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:,.2f} MB"


def wall_seed() -> int:
    return case.get_case(st.session_state).wall_seed


def image_row(images: dict[str, np.ndarray], captions: dict[str, str] | None = None) -> None:
    cols = st.columns(len(images))
    for col, (name, img) in zip(cols, images.items()):
        col.image(img, width="stretch", clamp=True, channels="BGR" if img.ndim == 3 else "RGB")
        with col:
            ui.caption(name + (f" · {captions[name]}" if captions else ""))


# convolution

@st.cache_data(show_spinner=False)
def gray_image(name: str, seed: int) -> np.ndarray:
    # seed only matters for the chapter 4 wall, but it keeps the cache apart per case
    path = IMAGES[name]
    img = seg.make_scene(seed)[0] if path is None else cv2.imread(str(ROOT / path))
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
    seed = wall_seed()
    stages = lab.apply_conv(gray_image(o["image"], seed), o["kernel"], o["stride"], o["padding"], o["relu"], o["pool"])
    cols = st.columns(len(stages))
    for col, (name, a) in zip(cols, stages.items()):
        col.image(to_view(a), width="stretch", clamp=True)
        with col:
            ui.caption(f"{name} · 1×{a.shape[0]}×{a.shape[1]}")
    p = 1 if o["padding"] == "same" else 0
    out = lab.conv_output_size(256, 3, p, o["stride"])
    formula(f"output = (W − K + 2P) / S + 1 = (256 − 3 + 2·{p}) / {o['stride']} + 1 = {out}")
    if o["pool"]:
        formula(f"after 2×2 max pool: ⌊{out} / 2⌋ = {lab.pool_output_size(out)}")
    st.markdown("Bright pixels are where the kernel pattern matches strongly. Display is rescaled per image.")
    unet_maps(seed)


@st.cache_data(show_spinner=False)
def enc1_maps(seed: int) -> np.ndarray:
    model = runtime.unet("standard", False)
    with torch.no_grad():
        maps = model.enc1(seg.to_tensor(seg.make_scene(seed)[0])[None])[0]
    return maps.numpy()


def unet_maps(seed: int) -> None:
    st.subheader("What the U-Net sees", anchor=False)
    try:
        maps = enc1_maps(seed)
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


# parameters

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


@st.cache_data(show_spinner=False)
def linear_probe(n_train: int = 60, n_test: int = 60) -> dict:
    t0 = time.perf_counter()
    net = load_yolo("yolo26n-cls.pt").model
    train_imgs, train_y = ex.probe_images(n_train, 1)
    test_imgs, test_y = ex.probe_images(n_test, 2)
    train_x, test_x = ex.backbone_features(net, train_imgs), ex.backbone_features(net, test_imgs)
    probe = ex.train_probe(train_x, train_y)
    return {"frozen": sum(p.numel() for p in net.parameters()), "trainable": sum(p.numel() for p in probe.parameters()),
            "features": train_x.shape[1], "train acc": ex.probe_accuracy(probe, train_x, train_y),
            "test acc": ex.probe_accuracy(probe, test_x, test_y), "n": (2 * n_train, 2 * n_test),
            "seconds": time.perf_counter() - t0}


def probe_section() -> None:
    st.subheader("Transfer learning: train 0.1% of a network", anchor=False)
    st.markdown("Freeze YOLO26n-cls (trained on ImageNet, it has never seen our stains) and train only a new last "
                "layer that answers one question: stained wall or clean wall? The crops come from the chapter 4 "
                "generator: 120 to train on, 120 different ones to test.")
    if not (st.session_state.get("lab_probe") or st.button("Train the linear probe", key="lab_probe_go")):
        return
    st.session_state["lab_probe"] = True
    with st.spinner("Extracting features and training..."):
        r = linear_probe()
    ui.readouts([("FROZEN PARAMS", f"{r['frozen']:,}", ""), ("TRAINED PARAMS", f"{r['trainable']:,}", "ok"),
                 ("TRAIN ACC", f"{r['train acc']:.1%}", ""), ("TEST ACC", f"{r['test acc']:.1%}", "ok"),
                 ("TIME", f"{r['seconds']:.1f} s", "")])
    formula(f"trainable = {r['features']} features × 2 classes + 2 biases = {r['trainable']:,}")
    st.markdown(ui.tag("measured") + " Same generator for training and test walls, so this is in-domain. "
                "A linear probe only works when the frozen features already separate the classes.",
                unsafe_allow_html=True)


def params_tab() -> None:
    layer_calc()
    st.subheader("TinyUNet layer by layer", anchor=False)
    variant = st.radio("Variant", list(seg.VARIANTS), index=1, horizontal=True, key="lab_variant",
                       format_func=str.title)
    df = unet_table(seg.VARIANTS[variant])
    total = int(df["params"].sum())
    st.dataframe(df, hide_index=True, width="stretch")
    ui.readouts([("TOTAL PARAMS", f"{total:,}", ""), ("FP32", kb(total * 4), ""),
                 ("INT8", kb(total), "ok"), ("INPUT", "3×128×128", "")])
    ui.message("On an edge device the weights are stored once, but every layer's output activations also need RAM "
               "while it runs, and at full resolution those are often bigger than the weights.")
    probe_section()


# activations

@st.cache_data(show_spinner=False)
def activation_frame(slope: float) -> pd.DataFrame:
    rows = []
    for name in lab.ACTIVATIONS:
        x, y, g = lab.activation_curves(name, slope)
        rows.append(pd.DataFrame({"x": x, "value": y, "derivative": g, "function": name}))
    return pd.concat(rows)


def activation_chart(df: pd.DataFrame, field: str, title: str) -> alt.Chart:
    scale = alt.Scale(domain=lab.ACTIVATIONS, range=COLORS)
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
               "one compare per value, which is cheap on edge chips and easy to quantize. Our U-Net uses it.",
               "SiLU is x·sigmoid(x): smooth, with a small dip below 0. Every YOLO26 conv block in the case uses it "
               "(see Architectures).",
               "Leaky ReLU keeps a small slope for negatives so neurons do not get stuck at 0. GELU is smooth but costs more."],
              title="FIELD NOTES")


# architectures

@st.cache_resource(show_spinner=False)
def fresh_yolo(weights: str):
    # a private copy: the chapters' predictor may fuse BatchNorm into the convs
    return load_yolo(weights).model


@st.cache_data(show_spinner=False)
def arch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    nets = {name: (fresh_yolo(w), size) for name, (w, size) in ARCH_MODELS.items()}
    nets["U-Net Std"] = (runtime.unet("standard", False), 128)
    rows, cats = {}, []
    for name, (net, size) in nets.items():
        info = ex.inspect_model(net)
        info["torch.cat calls per forward"] = ex.count_cats(net, size)
        info["activations"] = ", ".join(f"{k} ×{v}" for k, v in info["activations"].items())
        info["params"] = f"{info['params']:,}"
        rows[name] = {k: str(v) for k, v in info.items()}
        cats += [{"model": name, "category": k, "params": v} for k, v in ex.params_by_category(net).items()]
    return pd.DataFrame(rows), pd.DataFrame(cats)


def category_chart(cats: pd.DataFrame) -> alt.Chart:
    return alt.Chart(cats).mark_bar().encode(
        y=alt.Y("model:N", title=None), x=alt.X("params:Q", stack="normalize", title="share of parameters"),
        color=alt.Color("category:N", scale=alt.Scale(range=COLORS)), tooltip=["model", "category", "params"]
    ).properties(height=150)


def depthwise_readout() -> None:
    std = torch.nn.Conv2d(64, 64, 3, padding=1, bias=False)
    dw = torch.nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False)
    pw = torch.nn.Conv2d(64, 64, 1, bias=False)
    n_std = std.weight.numel()
    n_sep = dw.weight.numel() + pw.weight.numel()
    formula(f"3×3 conv 64→64: {n_std:,} weights · depthwise 3×3 + pointwise 1×1: {n_sep:,} weights "
            f"({n_std / n_sep:.1f}× fewer)")


@st.cache_data(show_spinner=False)
def skip_masks(seed: int) -> tuple[dict, dict]:
    img, truth = seg.make_scene(seed)
    model = runtime.unet("standard", False)
    variants = {"both skips (as trained)": (True, True), "no e1 skip (full res)": (False, True),
                "no e2 skip (half res)": (True, False), "no skips": (False, False)}
    masks = {k: (ex.unet_without_skips(model, img, *v) > 0.5).astype(np.float32) for k, v in variants.items()}
    return masks, {k: seg.mask_iou(m, truth) for k, m in masks.items()}


def architectures_tab() -> None:
    try:
        table, cats = arch_tables()
    except ModelMissing as e:
        st.warning(str(e))
        return
    st.markdown("Three networks from the case, opened up and counted " + ui.tag("measured"), unsafe_allow_html=True)
    st.dataframe(table, width="stretch")
    chart(category_chart(cats))
    depthwise_readout()
    ui.lesson(["Residual bottlenecks add a block's input to its output (skip connection), so gradients have a short "
               "path back. YOLO26 uses them inside its C3k2 blocks.",
               "Depthwise convs filter each channel on its own; a 1×1 conv then mixes channels. YOLO26s uses them "
               "in the detection head.",
               "Attention: one C2PSA block at the end of each YOLO26 backbone learns which positions to weight up.",
               "There is no Inception block in these models. SPPF in YOLO26s is the closest idea: it max-pools the "
               "same map at growing sizes (5×5 applied 1, 2 and 3 times) and concatenates the results.",
               "The U-Net has no attention or depthwise convs. Its two torch.cat calls are its skip connections."],
              title="FIELD NOTES")
    st.subheader("U-Net skip ablation", anchor=False)
    masks, ious = skip_masks(wall_seed())
    image_row(masks, {k: f"IoU {v:.3f}" for k, v in ious.items()})
    st.markdown(ui.tag("measured") + " The trained decoder with its skip tensors set to zero, not a retrained "
                "network. The full-resolution e1 skip carries where the stain edges are.", unsafe_allow_html=True)


# optimizers

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


# losses and metrics

@st.cache_data(show_spinner=False)
def loss_candidates(seed: int) -> tuple[dict, np.ndarray]:
    img, truth = seg.make_scene(seed)
    masks = {"Std INT8": seg.predict(runtime.unet("standard", True), img),
             "Lite FP32": seg.predict(runtime.unet("lite", False), img)}
    return {**masks, **ex.baseline_masks(truth)}, truth


@st.cache_data(show_spinner=False)
def loss_table(seed: int) -> pd.DataFrame:
    masks, truth = loss_candidates(seed)
    return pd.DataFrame([{"mask": k, **ex.mask_scores(m, truth)} for k, m in masks.items()])


def resolve_empty_accuracy(seed: int):
    acc = float(loss_table(seed).set_index("mask").loc["empty mask", "pixel accuracy"])
    return ex.accuracy_bin(acc), (f"Measured {acc:.1%}. Only {1 - acc:.1%} of this wall is stain, so a mask that "
                                  "says 'no stain anywhere' is right on every other pixel.")


def losses_reveal(seed: int) -> None:
    df = loss_table(seed)
    st.markdown("Every candidate scored against the true mask of this wall " + ui.tag("measured"),
                unsafe_allow_html=True)
    st.dataframe(df.round(3), hide_index=True, width="stretch")
    long = df.melt("mask", ["pixel accuracy", "IoU", "Dice"], var_name="metric")
    chart(alt.Chart(long).mark_bar().encode(
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("mask:N", sort=list(df["mask"]), title=None),
        yOffset="metric:N", color=alt.Color("metric:N", scale=alt.Scale(range=COLORS[:3]))).properties(height=260))
    ui.lesson(["Pixel accuracy counts the clean wall too. When most pixels are wall, an empty mask scores high and "
               "finds nothing.",
               "IoU and Dice ignore the true negatives and only look at the stain. Dice = 2·IoU / (1 + IoU), so it "
               "always reads a little higher.",
               "BCE (binary cross-entropy) is the loss our U-Net was trained with. It punishes confident wrong pixels "
               f"hardest. Hard 0/1 masks are clipped to {ex.BCE_EPS:g}–{1 - ex.BCE_EPS:g} so it stays finite.",
               "Dice loss (1 − Dice) is a common choice when the object is a small part of the image."],
              title="FIELD NOTES")


def losses_tab() -> None:
    seed = wall_seed()
    try:
        masks, truth = loss_candidates(seed)
    except ModelMissing as e:
        st.warning(str(e))
        return
    st.markdown("Five masks for your chapter 4 wall: two from real U-Nets and three made without any model.")
    image_row({"true mask": truth.astype(np.float32), **{k: (m > 0.5).astype(np.float32) for k, m in masks.items()}})
    flow.predict("lab_pred_losses", "The empty mask says 'no stain anywhere'. What is its pixel accuracy on this wall?",
                 ex.ACCURACY_BINS, lambda: resolve_empty_accuracy(seed), lambda: losses_reveal(seed))


# normalization and initialization

@st.cache_data(show_spinner=False)
def input_modes(seed: int) -> tuple[dict, dict, dict]:
    img, truth = seg.make_scene(seed)
    model = runtime.unet("standard", False)
    masks, ious, stats = {}, {}, {}
    for mode in ex.INPUT_MODES:
        x = ex.normalize_input(img, mode)
        with torch.no_grad():
            masks[mode] = (torch.sigmoid(model(x[None]))[0, 0].numpy() > 0.5).astype(np.float32)
        ious[mode] = seg.mask_iou(masks[mode], truth)
        stats[mode] = ex.enc1_stats(model, x)
    return masks, ious, stats


def bn_chart(stats: dict) -> alt.Chart:
    df = pd.DataFrame({"channel": np.arange(len(stats["input mean"])), "this input": stats["input mean"],
                       "BN running mean": stats["BN running mean"]}).melt("channel", var_name="source")
    return alt.Chart(df, title="enc1 conv output, mean per channel (before BatchNorm)").mark_point(filled=True).encode(
        x="channel:O", y=alt.Y("value:Q", title="mean"),
        color=alt.Color("source:N", scale=alt.Scale(range=COLORS[:2])), shape="source:N").properties(height=240)


def input_section(seed: int) -> None:
    st.subheader("Input scaling", anchor=False)
    masks, ious, stats = input_modes(seed)
    image_row(masks, {k: f"IoU {v:.3f}" for k, v in ious.items()})
    mode = st.radio("Compare BatchNorm statistics for", ex.INPUT_MODES, horizontal=True, key="lab_norm_mode")
    chart(bn_chart(stats[mode]))
    st.markdown(ui.tag("measured") + " The U-Net was trained on pixels ÷ 255. BatchNorm stored the mean and spread "
                "it saw in training; feed a different scale and every later layer is off. YOLO26 also only divides "
                "by 255 (Ultralytics uses mean 0, std 1). Many ImageNet classifiers expect mean/std instead.",
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def init_frame(batchnorm: bool) -> pd.DataFrame:
    rows = [{"layer": i + 1, "std": max(s, 1e-6), "init": scheme}
            for scheme in ex.INIT_SCHEMES for i, s in enumerate(ex.init_stds(scheme, batchnorm))]
    return pd.DataFrame(rows)


def resolve_init():
    std = ex.init_stds("PyTorch default", False)[-1]
    return ex.std_bin(std), (f"Measured std after layer 12: {std:.3f}. PyTorch's default conv weights have std "
                             "1/√(3·fan_in); Kaiming init for ReLU uses √(2/fan_in). With the smaller weights each "
                             "ReLU layer shrinks the signal. BatchNorm rescales it whatever the init.")


def init_reveal() -> None:
    bn = st.toggle("BatchNorm after every conv", key="lab_init_bn")
    df = init_frame(bn)
    chart(alt.Chart(df, title="activation std after each conv + ReLU (log scale)").mark_line(point=True).encode(
        x="layer:O", y=alt.Y("std:Q", scale=alt.Scale(type="log"), title="std"),
        color=alt.Color("init:N", scale=alt.Scale(domain=ex.INIT_SCHEMES, range=COLORS[:4]))).properties(height=300))
    st.markdown(ui.tag("measured") + " 12 layers, 16 channels, 3×3 kernels, input std 1. Zeros gives exactly 0 "
                "everywhere (drawn at 1e-6): every neuron computes the same thing and nothing can learn apart.",
                unsafe_allow_html=True)


def normalization_tab() -> None:
    seed = wall_seed()
    try:
        input_section(seed)
    except ModelMissing as e:
        st.warning(str(e))
    st.subheader("Weight initialization", anchor=False)
    flow.predict("lab_pred_init", "12 conv + ReLU layers with PyTorch's default init and no BatchNorm. The input "
                 "has std 1. After layer 12 the activations...", ex.STD_BINS, resolve_init, init_reveal)


# pruning

@st.cache_data(show_spinner=False)
def pruning_table(seed: int) -> pd.DataFrame:
    img, truth = seg.make_scene(seed)
    val = seg.make_dataset(VAL_WALLS, VAL_SEED)
    rows = ex.pruning_rows(runtime.unet("standard", False), PRUNE_AMOUNTS, img, truth, val)
    rows.append(ex.measure("Std INT8 (reference)", runtime.unet("standard", True), img, truth, val))
    rows.append(ex.measure("Lite FP32 (reference)", runtime.unet("lite", False), img, truth, val))
    return pd.DataFrame(rows)


def resolve_pruning(seed: int):
    df = pruning_table(seed).set_index("model")
    ms = float(df.loc["Std FP32 pruned 50%", "latency ms"])
    base = float(df.loc["Std FP32 pruned 0%", "latency ms"])
    return int(ms >= BUDGET_MS), (f"Measured {ms:.1f} ms at 50% zeros against {base:.1f} ms unpruned. "
                                  "A dense conv still multiplies every zero.")


def pruning_reveal(seed: int) -> None:
    df = pruning_table(seed)
    st.markdown("Median of 5 runs on a 384×384 wall, this machine's CPU " + ui.tag("measured"), unsafe_allow_html=True)
    st.dataframe(df.round(3), hide_index=True, width="stretch")
    bars = alt.Chart(df).mark_bar(color=COLORS[5]).encode(x=alt.X("latency ms:Q"), y=alt.Y("model:N", sort=None, title=None))
    rule = alt.Chart(pd.DataFrame({"ms": [BUDGET_MS]})).mark_rule(color=COLORS[3], strokeDash=[4, 3]).encode(x="ms:Q")
    chart((bars + rule).properties(height=230))
    ui.lesson(["Unstructured pruning sets the smallest weights to 0, but the tensor keeps its shape. A dense conv "
               "kernel multiplies the zeros anyway, so latency and the saved file stay the same.",
               "To get faster you remove whole channels (structured pruning) or use a sparse kernel the hardware "
               "supports. Then you usually fine-tune to win the accuracy back.",
               "INT8 changes the arithmetic itself: 8-bit weights, less memory traffic, faster integer kernels.",
               f"The red line is the chapter 4 limit of {BUDGET_MS} ms " + ui.tag("gameplay")], title="FIELD NOTES")


def pruning_tab() -> None:
    st.markdown("Take the Standard U-Net (FP32), zero out its smallest conv weights across the whole network "
                "(global L1 unstructured pruning) and measure again. The experiment runs after you lock in a guess.")
    seed = wall_seed()
    try:
        runtime.unet("standard", False)
    except ModelMissing as e:
        st.warning(str(e))
        return
    with st.spinner("Pruning and timing seven models..."):
        flow.predict("lab_pred_pruning", f"Prune 50% of the Standard U-Net's conv weights. Will it run under "
                     f"{BUDGET_MS} ms on this laptop?", [f"Yes, under {BUDGET_MS} ms", f"No, still {BUDGET_MS} ms or more"],
                     lambda: resolve_pruning(seed), lambda: pruning_reveal(seed))


# augmentation

@st.cache_data(show_spinner=False)
def mix_sources() -> tuple[np.ndarray, np.ndarray]:
    return ex.square(cv2.imread(str(ROOT / MIX_A))), ex.square(cv2.imread(str(ROOT / MIX_B)))


@st.cache_data(show_spinner=False)
def classify_mix(mode: str, lam: float) -> tuple[np.ndarray, float, list, float]:
    a, b = mix_sources()
    img, share = (ex.mixup(a, b, lam), lam) if mode == "Mixup" else ex.cutmix(a, b, lam)
    r = runtime.yolo("yolo26n-cls.pt").predict(img, imgsz=224, device="cpu", verbose=False)[0]
    top = [(r.names[i], float(r.probs.data[i])) for i in r.probs.top5[:3]]
    p_lock = sum(float(r.probs.data[i]) for i, name in r.names.items() if name in PADLOCK_LABELS)
    return img, share, top, p_lock


@st.cache_data(show_spinner=False)
def mix_sweep(mode: str) -> pd.DataFrame:
    rows = []
    for lam in np.round(np.linspace(0, 1, 11), 1):
        _, share, _, p_lock = classify_mix(mode, float(lam))
        rows += [{"padlock share": share, "value": share, "series": "Mixup/CutMix label"},
                 {"padlock share": share, "value": p_lock, "series": "classifier says padlock"}]
    return pd.DataFrame(rows)


def resolve_mix():
    name = classify_mix("Mixup", 0.7)[2][0][0]
    idx = 0 if name in PADLOCK_LABELS else (1 if name == "teddy" else 2)
    return idx, f"YOLO26n-cls top-1 on the 70/30 blend: {name}."


def mix_reveal() -> None:
    c1, c2 = st.columns(2)
    mode = c1.radio("Method", ["Mixup", "CutMix"], horizontal=True, key="lab_mix_mode")
    lam = c2.slider("λ (share of padlock)", 0.0, 1.0, 0.7, 0.1, key="lab_mix_lam")
    img, share, top, _ = classify_mix(mode, lam)
    a, b = mix_sources()
    image_row({"padlock": a, "teddy bear": b, mode: img})
    ui.readouts([("LABEL USED IN TRAINING", f"{share:.0%} padlock · {1 - share:.0%} teddy", "")] +
                [(f"TOP-{i + 1}", f"{name} {p:.0%}", "") for i, (name, p) in enumerate(top)])
    df = mix_sweep(mode)
    chart(alt.Chart(df, title="padlock probability vs the mixed label").mark_line(point=True).encode(
        x="padlock share:Q", y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title="padlock"),
        color=alt.Color("series:N", scale=alt.Scale(range=COLORS[:2]))).properties(height=260))
    st.markdown(ui.tag("measured") + " YOLO26n-cls on real blends. Its answer jumps from one class to the other "
                "instead of following the mixed label.", unsafe_allow_html=True)
    ui.lesson(["Mixup blends two images and uses the same blend for the labels. CutMix pastes a patch and weights "
               "the labels by patch area.",
               "Training on these soft labels teaches a model to be less sure on in-between inputs.",
               "Like every augmentation, it is used in training only. In practice λ is drawn at random each batch."],
              title="FIELD NOTES")


def augmentation_tab() -> None:
    st.markdown("Blend the padlock and the teddy bear from chapter 2, 70% padlock and 30% teddy bear, and show it "
                "to the ImageNet classifier YOLO26n-cls.")
    with st.spinner("Classifying blends..."):
        flow.predict("lab_pred_mix", "What does YOLO26n-cls name as its top-1?",
                     ["padlock (or combination lock)", "teddy bear", "something else"], resolve_mix, mix_reveal)


RENDER = {"Convolution": conv_tab, "Parameters": params_tab, "Activations": activations_tab,
          "Architectures": architectures_tab, "Optimizers": optimizers_tab, "Losses": losses_tab,
          "Normalization": normalization_tab, "Pruning": pruning_tab, "Augmentation": augmentation_tab}


def mark_seen(tab: str) -> None:
    s = st.session_state
    if not isinstance(s.get("lab_seen"), set):
        s["lab_seen"] = set(s.get("lab_seen") or ())
    s["lab_seen"].add(tab)


def remember_tab() -> None:
    st.session_state["lab_current"] = st.session_state["lab_tab"]


def render() -> None:
    ui.scene_header("GHOSTLENS LAB · HQ WORKBENCH", "The Lab",
                    "Back at HQ, no battery is used. Try the building blocks behind the models from the case.")
    s = st.session_state
    target = s.pop("lab_open", None)
    if target in TABS:
        s["lab_current"] = target
    if s.get("lab_current") in TABS:
        s["lab_tab"] = s["lab_current"]     # restored every run, so a rerun never drops back to the first tab
    tabs = st.tabs(TABS, key="lab_tab", on_change=remember_tab)
    for name, tab in zip(TABS, tabs):
        if tab.open:
            mark_seen(name)
            with tab:
                syllabus(name)
                RENDER[name]()
