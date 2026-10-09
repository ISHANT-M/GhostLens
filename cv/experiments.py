"""Small measured experiments for the Lab: losses, pruning, normalization, init, architectures, mixing."""

import copy
import statistics
import time
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.overrides import TorchFunctionMode

from cv import segmentation as seg

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_MODES = ["÷255 (as trained)", "raw 0–255", "ImageNet mean/std"]
INIT_SCHEMES = ["zeros", "N(0, 1)", "PyTorch default", "Kaiming"]
ACCURACY_BINS = ["under 50%", "50–80%", "80–95%", "over 95%"]
STD_BINS = ["explodes (std over 10)", "stays between 0.1 and 10", "shrinks (std under 0.1)"]
BCE_EPS = 1e-3


# losses and metrics

def mask_scores(prob: np.ndarray, truth: np.ndarray) -> dict:
    """Pixel accuracy, IoU, Dice and BCE of a probability map (or hard mask) against the true mask."""
    p = np.clip(prob.astype(np.float64), BCE_EPS, 1 - BCE_EPS)
    t = truth.astype(bool)
    hard = prob > 0.5
    inter, total = (hard & t).sum(), hard.sum() + t.sum()
    bce = -np.mean(np.where(t, np.log(p), np.log(1 - p)))
    return {"pixel accuracy": float((hard == t).mean()), "IoU": seg.mask_iou(hard, t),
            "Dice": float(2 * inter / total) if total else 1.0, "BCE": float(bce)}


def baseline_masks(truth: np.ndarray) -> dict[str, np.ndarray]:
    """Three masks that need no model: nothing, the stain's bounding box, everything."""
    box = np.zeros(truth.shape, np.float32)
    x0, y0, x1, y1 = seg.bounding_box(truth)
    box[y0:y1, x0:x1] = 1
    return {"empty mask": np.zeros(truth.shape, np.float32), "filled box": box,
            "all stain": np.ones(truth.shape, np.float32)}


def accuracy_bin(acc: float) -> int:
    return int(np.searchsorted([0.5, 0.8, 0.95], acc, side="right"))


# pruning

def conv_weights(model: nn.Module) -> list[nn.Conv2d]:
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def prune_copy(model: nn.Module, amount: float) -> nn.Module:
    """Global L1 unstructured pruning of every conv weight, on a copy, made permanent."""
    pruned = copy.deepcopy(model).eval()
    convs = conv_weights(pruned)
    if amount > 0:
        prune.global_unstructured([(m, "weight") for m in convs], pruning_method=prune.L1Unstructured,
                                  amount=amount)
        for m in convs:
            prune.remove(m, "weight")
    return pruned


def sparsity(model: nn.Module) -> tuple[int, int]:
    """(zero conv weights, all conv weights)."""
    w = [m.weight.detach().flatten() for m in conv_weights(model)]
    if not w:
        return 0, 0
    w = torch.cat(w)
    return int((w == 0).sum()), int(w.numel())


def median_ms(fn, runs: int = 5) -> float:
    fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@torch.no_grad()
def measure(name: str, model: nn.Module, img: np.ndarray, truth: np.ndarray, val, runs: int = 5) -> dict:
    """One row of the pruning table: sparsity, IoU on this wall and the validation walls, latency, size."""
    zeros, total = sparsity(model)
    x = seg.to_tensor(img)[None]
    return {"model": name, "sparsity": zeros / total if total else None,
            "non-zero weights": total - zeros if total else None,
            "IoU this wall": seg.mask_iou(seg.predict(model, img) > 0.5, truth),
            "IoU 50 val walls": seg.evaluate(model, val),
            "latency ms": median_ms(lambda: model(x), runs), "state_dict MB": seg.state_size_mb(model)}


def pruning_rows(model: nn.Module, amounts, img, truth, val, runs: int = 5) -> list[dict]:
    return [measure(f"Std FP32 pruned {a:.0%}", prune_copy(model, a), img, truth, val, runs) for a in amounts]


# normalization and initialization

def normalize_input(img: np.ndarray, mode: str) -> torch.Tensor:
    x = seg.to_tensor(img) * 255
    if mode == INPUT_MODES[0]:
        return x / 255
    if mode == INPUT_MODES[1]:
        return x
    mean, std = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    return (x / 255 - mean) / std


@torch.no_grad()
def enc1_stats(model: seg.TinyUNet, x: torch.Tensor) -> dict[str, np.ndarray]:
    """Per-channel mean/std of the first conv's output next to what its BatchNorm learned to expect."""
    pre = model.enc1[0](x[None])
    bn = model.enc1[1]
    return {"input mean": pre.mean((0, 2, 3)).numpy(), "input std": pre.std((0, 2, 3)).numpy(),
            "BN running mean": bn.running_mean.numpy().copy(), "BN running std": bn.running_var.sqrt().numpy()}


def init_conv(conv: nn.Conv2d, scheme: str) -> None:
    if scheme == "PyTorch default":
        return
    if scheme == "zeros":
        nn.init.zeros_(conv.weight)
    elif scheme == "N(0, 1)":
        nn.init.normal_(conv.weight, 0.0, 1.0)
    else:
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    nn.init.zeros_(conv.bias)


@torch.no_grad()
def init_stds(scheme: str, batchnorm: bool, layers: int = 12, channels: int = 16, seed: int = 0) -> list[float]:
    """Std of the activations after each of `layers` conv+ReLU layers, input std 1."""
    torch.manual_seed(seed)
    x = torch.randn(8, channels, 32, 32)
    stds = []
    for _ in range(layers):
        conv = nn.Conv2d(channels, channels, 3, padding=1)
        init_conv(conv, scheme)
        x = conv(x)
        if batchnorm:
            x = F.batch_norm(x, None, None, training=True)
        x = F.relu(x)
        stds.append(float(x.std()))
    return stds


def std_bin(std: float) -> int:
    return 0 if std > 10 else (1 if std >= 0.1 else 2)


# architectures

class CatCounter(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n += func is torch.cat
        return func(*args, **(kwargs or {}))


@torch.no_grad()
def count_cats(model: nn.Module, size: int) -> int:
    """How many torch.cat calls one forward pass makes."""
    with CatCounter() as counter:
        model.eval()(torch.zeros(1, 3, size, size))
    return counter.n


def is_depthwise(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and m.groups > 1 and m.groups == m.in_channels


def inspect_model(model: nn.Module) -> dict:
    """Count building blocks. Modules shared by several layers (YOLO's SiLU) are counted per use."""
    mods = [m for _, m in model.named_modules(remove_duplicate=False)]
    names = Counter(type(m).__name__ for m in mods)
    acts = Counter(type(m).__name__ for m in mods if isinstance(m, (nn.ReLU, nn.SiLU, nn.GELU, nn.Sigmoid,
                                                                     nn.LeakyReLU, nn.Hardswish)))
    return {"conv layers": sum(isinstance(m, nn.Conv2d) for m in mods),
            "depthwise convs": sum(is_depthwise(m) for m in mods),
            "BatchNorm layers": names["BatchNorm2d"],
            "residual bottlenecks": sum(bool(getattr(m, "add", False)) for m in mods if type(m).__name__ == "Bottleneck"),
            "attention blocks": names["Attention"], "SPPF blocks": names["SPPF"],
            "activations": dict(acts), "params": sum(p.numel() for p in model.parameters())}


def params_by_category(model: nn.Module) -> dict[str, int]:
    """Parameters split into attention / depthwise conv / conv / BatchNorm / linear. Sums to the total."""
    inside_attention = {id(p) for m in model.modules() if type(m).__name__ == "Attention" for p in m.parameters()}
    out = Counter()
    for m in model.modules():
        for p in m.parameters(recurse=False):
            if id(p) in inside_attention:
                kind = "attention"
            elif is_depthwise(m):
                kind = "depthwise conv"
            else:
                kind = {nn.Conv2d: "conv", nn.BatchNorm2d: "BatchNorm", nn.Linear: "linear"}.get(type(m), "other")
            out[kind] += p.numel()
    return dict(out)


@torch.no_grad()
def unet_without_skips(model: seg.TinyUNet, img: np.ndarray, keep_e1: bool, keep_e2: bool) -> np.ndarray:
    """The trained U-Net with its skip tensors replaced by zeros. Not retrained."""
    x = seg.to_tensor(img)[None]
    e1 = model.enc1(x)
    e2 = model.enc2(model.pool(e1))
    e3 = model.enc3(model.pool(e2))
    s1, s2 = (e1 if keep_e1 else torch.zeros_like(e1)), (e2 if keep_e2 else torch.zeros_like(e2))
    d2 = model.dec2(torch.cat([F.interpolate(e3, scale_factor=2.0, mode="nearest"), s2], 1))
    d1 = model.dec1(torch.cat([F.interpolate(d2, scale_factor=2.0, mode="nearest"), s1], 1))
    return torch.sigmoid(model.head(d1))[0, 0].numpy()


# augmentation: mixup and cutmix

def square(img: np.ndarray, size: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)


def mixup(a: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:
    return (lam * a.astype(np.float32) + (1 - lam) * b.astype(np.float32)).round().astype(np.uint8)


def cutmix(a: np.ndarray, b: np.ndarray, lam: float, seed: int = 0) -> tuple[np.ndarray, float]:
    """Paste a box of b covering about 1 - lam of a. Returns the image and the true share of a."""
    h, w = a.shape[:2]
    rng = np.random.default_rng(seed)
    bh, bw = int(round(h * np.sqrt(1 - lam))), int(round(w * np.sqrt(1 - lam)))
    y, x = int(rng.integers(0, h - bh + 1)), int(rng.integers(0, w - bw + 1))
    out = a.copy()
    out[y:y + bh, x:x + bw] = b[y:y + bh, x:x + bw]
    return out, 1 - bh * bw / (h * w)


# transfer learning: linear probe

def probe_images(n: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """n stained wall crops and n clean ones from the chapter 4 generator. Label 1 = stain."""
    rng = np.random.default_rng(seed)
    photos = seg.load_backgrounds()
    stained = [seg.make_sample(rng, photos)[0] for _ in range(n)]
    clean = [seg.random_wall(seg.TRAIN_SIZE, rng, photos) for _ in range(n)]
    return stained + clean, np.array([1] * n + [0] * n)


@torch.no_grad()
def backbone_features(net: nn.Module, images: list[np.ndarray]) -> np.ndarray:
    """The 1280 numbers a YOLO26 classifier feeds its last Linear layer, one row per image."""
    x = torch.stack([seg.to_tensor(cv2.resize(i, (224, 224))) for i in images])
    feats = []
    hook = net.model[-1].linear.register_forward_hook(lambda m, inp, out: feats.append(inp[0]))
    net.eval()(x)
    hook.remove()
    return feats[0].numpy()


def train_probe(feats: np.ndarray, labels: np.ndarray, steps: int = 300, seed: int = 0) -> nn.Linear:
    """Logistic regression on frozen features: one Linear(1280, 2) trained with Adam."""
    torch.manual_seed(seed)
    probe = nn.Linear(feats.shape[1], 2)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    x, y = torch.from_numpy(feats).float(), torch.from_numpy(labels).long()
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe(x), y).backward()
        opt.step()
    return probe


@torch.no_grad()
def probe_accuracy(probe: nn.Linear, feats: np.ndarray, labels: np.ndarray) -> float:
    return float((probe(torch.from_numpy(feats).float()).argmax(1).numpy() == labels).mean())
