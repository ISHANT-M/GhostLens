"""Level 4: a tiny U-Net trained on wall crops with painted-on stains, so we know the exact mask."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
BACKGROUNDS = [ROOT / "assets/level3/antique_room.jpg", ROOT / "assets/level1/room217.jpg",
               ROOT / "assets/level2/living_room.jpg"]
TRAIN_SIZE = 128


# synthetic data

def stain_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    """An irregular blob with tendrils: smoothed noise, thresholded, kept near a random centre."""
    noise = rng.random((size // 8, size // 8)).astype(np.float32)
    noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size // 3, size // 3)).astype(np.float32), (size, size))
    yy, xx = np.mgrid[0:size, 0:size] / size
    cx, cy = rng.uniform(0.3, 0.7, 2)
    r = rng.uniform(0.18, 0.38)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    field = 0.55 * noise + 0.25 * fine - 0.6 * dist
    return (field > np.quantile(field, 1 - rng.uniform(0.08, 0.3))).astype(np.uint8)


def paint_stain(wall: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Dark, slightly green rot with darker veins, soft at the edges."""
    h, w = mask.shape
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)
    veins = cv2.resize(rng.random((h // 4, w // 4)).astype(np.float32), (w, h))
    colour = np.array(rng.choice([[38, 52, 40], [30, 40, 48], [45, 45, 30]]), np.float32)
    stain = colour * (0.6 + 0.6 * veins[..., None])
    alpha = soft[..., None] * rng.uniform(0.75, 0.95)
    return (wall * (1 - alpha) + stain * alpha).astype(np.uint8)


def random_wall(size: int, rng: np.random.Generator, photos: list[np.ndarray]) -> np.ndarray:
    if photos and rng.random() < 0.8:
        img = photos[rng.integers(len(photos))]
        s = int(rng.integers(size, min(img.shape[:2])))
        y, x = rng.integers(0, img.shape[0] - s + 1), rng.integers(0, img.shape[1] - s + 1)
        return cv2.resize(img[y:y + s, x:x + s], (size, size), interpolation=cv2.INTER_AREA)
    base = rng.uniform(150, 230, 3).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    pattern = 12 * np.sin(xx / rng.uniform(4, 12)) * np.cos(yy / rng.uniform(4, 12))
    return np.clip(base + pattern[..., None] + rng.normal(0, 4, (size, size, 3)), 0, 255).astype(np.uint8)


def load_backgrounds() -> list[np.ndarray]:
    images = [cv2.imread(str(p)) for p in BACKGROUNDS]
    return [img for img in images if img is not None]


def make_sample(rng: np.random.Generator, photos: list[np.ndarray], size: int = TRAIN_SIZE):
    wall = random_wall(size, rng, photos)
    mask = stain_mask(size, rng)
    return paint_stain(wall, mask, rng), mask


def make_scene(seed: int = 217, size: int = 384) -> tuple[np.ndarray, np.ndarray]:
    """The fixed Level 4 image: the parlour wall between the curtains, with a stain spreading on it."""
    room = cv2.imread(str(BACKGROUNDS[0]))
    wall = cv2.resize(room[20:380, 300:660], (size, size), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    mask = stain_mask(size, rng)
    return paint_stain(wall, mask, rng), mask


# model

def block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    # nearest-neighbour upsampling + conv instead of ConvTranspose so the whole model can be quantized to INT8

    def __init__(self, width: int = 16):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.enc3 = block(3, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.pool = nn.MaxPool2d(2)
        self.dec2 = block(6 * w, 2 * w)
        self.dec1 = block(3 * w, w)
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([F.interpolate(e3, scale_factor=2.0, mode="nearest"), e2], 1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2.0, mode="nearest"), e1], 1))
        return self.head(d1)  # one logit per pixel


VARIANTS = {"lite": 4, "standard": 16, "pro": 32}


def weights_path(variant: str) -> Path:
    return ROOT / "models" / f"stain_unet_{variant}.pt"


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255


def make_dataset(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    photos = load_backgrounds()
    pairs = [make_sample(rng, photos) for _ in range(n)]
    x = torch.stack([to_tensor(i) for i, _ in pairs])
    y = torch.stack([torch.from_numpy(m).float()[None] for _, m in pairs])
    return x, y


def train(width: int, data: tuple[torch.Tensor, torch.Tensor], val: tuple[torch.Tensor, torch.Tensor],
          epochs: int = 6, batch: int = 16, log=print) -> tuple[TinyUNet, float]:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    x_all, y_all = data
    model = TinyUNet(width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = epochs * (len(x_all) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        order = torch.randperm(len(x_all))
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i:i + batch]
            x, y = x_all[idx], y_all[idx]
            if torch.rand(1) < 0.5:  # random flip, cheap augmentation
                x, y = x.flip(3), y.flip(3)
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        log(f"  width {width}  epoch {epoch + 1}/{epochs}  loss {loss.item():.4f}")
    model = model.cpu().eval()
    return model, evaluate(model, val)


@torch.no_grad()
def evaluate(model: nn.Module, val: tuple[torch.Tensor, torch.Tensor]) -> float:
    probs = torch.sigmoid(model(val[0]))
    return float(np.mean([mask_iou(p[0].numpy() > 0.5, t[0].numpy()) for p, t in zip(probs, val[1])]))


def save(model: TinyUNet, val_iou: float, variant: str) -> None:
    path = weights_path(variant)
    path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "width": VARIANTS[variant], "val_iou": val_iou}, path)


def load(variant: str = "standard") -> TinyUNet:
    from cv.models import ModelMissing
    path = weights_path(variant)
    if not path.exists():
        raise ModelMissing("The segmentation models aren't trained yet. Run `python setup_models.py` "
                           "(a few minutes) and reload the page.")
    ckpt = torch.load(path, map_location="cpu")
    model = TinyUNet(ckpt["width"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def quantize(model: TinyUNet, calibration: torch.Tensor) -> nn.Module:
    """Post-training static INT8 quantization, calibrated on a few wall images."""
    import copy
    import warnings

    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "qnnpack"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prepared = prepare_fx(copy.deepcopy(model).eval(), get_default_qconfig_mapping("qnnpack"),
                              (calibration[:1],))
        with torch.no_grad():
            for i in range(0, len(calibration), 8):
                prepared(calibration[i:i + 8])
        return convert_fx(prepared)


@torch.no_grad()
def predict(model: nn.Module, img: np.ndarray) -> np.ndarray:
    """Probability (0..1) that each pixel is corrupted. Works for any size divisible by 4."""
    return torch.sigmoid(model(to_tensor(img)[None]))[0, 0].numpy()


def state_size_mb(model: nn.Module) -> float:
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


# measurements

def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = pred.astype(bool), truth.astype(bool)
    union = (pred | truth).sum()
    return float((pred & truth).sum() / union) if union else 1.0


def bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def clean_share_of_box(mask: np.ndarray) -> float:
    """How much of the bounding box is actually clean wall."""
    box = bounding_box(mask)
    if box is None:
        return 0.0
    x0, y0, x1, y1 = box
    return 1 - float(mask[y0:y1, x0:x1].astype(bool).mean())
