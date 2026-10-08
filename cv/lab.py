"""Pure maths behind the Lab page: convolution, layer cost, activations and optimizers."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

KERNELS = {
    "Sobel x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "Sobel y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "Laplacian edges": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    "Box blur": [[1 / 9] * 3] * 3,
    "Sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "Emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
}


def conv_output_size(w: int, k: int, p: int, s: int) -> int:
    return (w - k + 2 * p) // s + 1


def apply_conv(img: np.ndarray, kernel, stride: int = 1, padding: str = "same",
               relu: bool = False, pool: bool = False) -> dict:
    """img is a 2D float array in [0, 1]. Returns every stage as a 2D array."""
    x = torch.from_numpy(np.asarray(img, np.float32))[None, None]
    k = torch.tensor(kernel, dtype=torch.float32)[None, None]
    pad = k.shape[-1] // 2 if padding == "same" else 0
    stages = {"input": x}
    out = F.conv2d(x, k, stride=stride, padding=pad)
    stages["conv"] = out
    if relu:
        out = F.relu(out)
        stages["relu"] = out
    if pool:
        out = F.max_pool2d(out, 2)
        stages["pool"] = out
    return {name: t[0, 0].numpy() for name, t in stages.items()}


def conv_layer_stats(c_in: int, c_out: int, k: int, h: int, w: int, stride: int = 1, padding: int | None = None) -> dict:
    p = k // 2 if padding is None else padding
    oh, ow = conv_output_size(h, k, p, stride), conv_output_size(w, k, p, stride)
    params = k * k * c_in * c_out + c_out
    return {
        "params": params,
        "out_shape": (c_out, oh, ow),
        "macs": k * k * c_in * c_out * oh * ow,  # one multiply-add per weight per output pixel
        "weights_fp32": params * 4, "weights_fp16": params * 2, "weights_int8": params,
        "activations_fp32": c_out * oh * ow * 4,
    }


def unet_layer_table(model: nn.Module, size: int = 128) -> list[dict]:
    """One row per learnable layer (conv and batchnorm), so the total equals sum(p.numel())."""
    rows, shapes = [], {}

    def save_shape(name):
        def hook(module, inputs, output):
            shapes[name] = tuple(output.shape[1:])
        return hook

    hooks = [m.register_forward_hook(save_shape(n)) for n, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, size, size))
    for h in hooks:
        h.remove()
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            n = sum(p.numel() for p in m.parameters(recurse=False))
            kind = f"Conv {m.kernel_size[0]}x{m.kernel_size[1]}" if isinstance(m, nn.Conv2d) else "BatchNorm"
            c_in, c_out = (m.in_channels, m.out_channels) if isinstance(m, nn.Conv2d) else (m.num_features,) * 2
            rows.append({"layer": name, "type": kind, "in": c_in, "out": c_out,
                         "output": "×".join(map(str, shapes.get(name, ()))), "params": n})
    return rows


def _act(name: str, x: torch.Tensor, slope: float) -> torch.Tensor:
    if name == "Leaky ReLU":
        return F.leaky_relu(x, slope)
    return {"ReLU": F.relu, "Sigmoid": torch.sigmoid, "Tanh": torch.tanh, "GELU": F.gelu}[name](x)


def activation_curves(name: str, slope: float = 0.1, x=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.linspace(-5, 5, 401) if x is None else torch.as_tensor(x, dtype=torch.float32)
    x = x.clone().requires_grad_(True)
    y = _act(name, x, slope)
    (grad,) = torch.autograd.grad(y.sum(), x)
    return x.detach().numpy(), y.detach().numpy(), grad.numpy()


def loss_fn(name: str, p: torch.Tensor) -> torch.Tensor:
    x, y = p[0], p[1]
    if name == "bowl":
        return 0.5 * (x ** 2 + 10 * y ** 2)
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2  # rosenbrock


OPTIMIZERS = {
    "SGD": lambda ps, lr: torch.optim.SGD(ps, lr=lr),
    "Momentum": lambda ps, lr: torch.optim.SGD(ps, lr=lr, momentum=0.9),
    "RMSprop": lambda ps, lr: torch.optim.RMSprop(ps, lr=lr),
    "Adam": lambda ps, lr: torch.optim.Adam(ps, lr=lr),
}


def optimizer_path(opt_name: str, loss_name: str, lr: float, steps: int, start) -> np.ndarray:
    p = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    opt = OPTIMIZERS[opt_name]([p], lr)
    path = [p.detach().numpy().copy()]
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(loss_name, p).backward()
        opt.step()
        if not torch.isfinite(p).all():
            break
        path.append(p.detach().numpy().copy())
    return np.array(path)


def optimizer_paths(loss_name: str, lr: float, steps: int, start) -> dict[str, np.ndarray]:
    return {name: optimizer_path(name, loss_name, lr, steps, start) for name in OPTIMIZERS}


def loss_value(loss_name: str, xy) -> float:
    return float(loss_fn(loss_name, torch.tensor(xy, dtype=torch.float32)))


def loss_grid(loss_name: str, x_range, y_range, n: int = 40) -> list[dict]:
    xs, ys = np.linspace(*x_range, n), np.linspace(*y_range, n)
    return [{"x": float(x), "y": float(y), "loss": float(np.log10(1 + loss_value(loss_name, (x, y))))}
            for x in xs for y in ys]
