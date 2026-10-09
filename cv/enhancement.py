"""Level 1 image processing. Everything works on uint8 BGR images (OpenCV's default)."""

import time
from dataclasses import dataclass

import cv2
import numpy as np


def load_image(path, width: int = 960, crop: list[float] | None = None) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if crop:
        # [x0, y0, x1, y1] as fractions, applied before resizing
        h, w = img.shape[:2]
        x0, y0, x1, y1 = crop
        img = img[round(y0 * h):round(y1 * h), round(x0 * w):round(x1 * w)]
    h, w = img.shape[:2]
    return cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def make_dark_frame(img: np.ndarray, seed: int = 217,
                    stamp: str = "CAM 03  2026-10-07  02:17:44") -> np.ndarray:
    """Turn a normal photo into a cheap night-time CCTV frame."""
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255
    x = 0.6 * x + 0.4 * x.mean(axis=2, keepdims=True)    # washed-out CCTV colour
    x = (x ** 2.4) * 0.075                      # very little light, shadows crushed
    x = cv2.GaussianBlur(x, (3, 3), 0)          # cheap lens
    x += rng.normal(0, 0.004, x.shape).astype(np.float32)   # sensor noise
    dark = np.clip(x * 255, 0, 255).astype(np.uint8)
    cv2.putText(dark, stamp, (16, dark.shape[0] - 18),
                cv2.FONT_HERSHEY_PLAIN, 1.2, (40, 40, 40), 1, cv2.LINE_AA)
    return dark


# individual operations

def brightness_contrast(img: np.ndarray, brightness: float = 0, contrast: float = 1.0) -> np.ndarray:
    # out = contrast * in + brightness, clipped to 0..255 (this is where clipping happens)
    out = img.astype(np.float32) * contrast + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma_correct(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # gamma > 1 lifts dark tones a lot and bright tones a little
    lut = np.clip(((np.arange(256) / 255.0) ** (1.0 / gamma)) * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def equalize(img: np.ndarray) -> np.ndarray:
    # global histogram equalization on brightness only, so colours don't shift
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe(img: np.ndarray, clip_limit: float = 2.0, tiles: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, method: str, strength: int) -> np.ndarray:
    if method == "Gaussian blur":
        k = 2 * strength + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if method == "Median":
        return cv2.medianBlur(img, 2 * strength + 1)
    if method == "Non-local means":
        return cv2.fastNlMeansDenoisingColored(img, None, h=strength * 3, hColor=strength * 3,
                                               templateWindowSize=7, searchWindowSize=21)
    return img


def sharpen(img: np.ndarray, amount: float) -> np.ndarray:
    # unsharp mask: add back the difference between the image and a blurred copy
    blurred = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


# pipeline

@dataclass
class Settings:
    denoise_method: str = "None"
    denoise_strength: int = 1
    brightness: float = 0
    contrast: float = 1.0
    gamma: float = 1.0
    equalizer: str = "None"          # None / Global equalization / CLAHE
    clahe_clip: float = 2.0
    sharpen: float = 0.0


def run_pipeline(img: np.ndarray, s: Settings) -> tuple[np.ndarray, list[tuple[str, float]]]:
    # denoise first, otherwise the tone steps amplify the noise too
    steps = []
    if s.denoise_method != "None":
        steps.append((s.denoise_method, lambda x: denoise(x, s.denoise_method, s.denoise_strength)))
    if s.gamma != 1.0:
        steps.append(("Gamma", lambda x: gamma_correct(x, s.gamma)))
    if s.brightness != 0 or s.contrast != 1.0:
        steps.append(("Brightness/contrast", lambda x: brightness_contrast(x, s.brightness, s.contrast)))
    if s.equalizer == "Global equalization":
        steps.append(("Equalization", equalize))
    elif s.equalizer == "CLAHE":
        steps.append(("CLAHE", lambda x: clahe(x, s.clahe_clip)))
    if s.sharpen > 0:
        steps.append(("Sharpen", lambda x: sharpen(x, s.sharpen)))

    timings = []
    out = img
    for name, fn in steps:
        t0 = time.perf_counter()
        out = fn(out)
        timings.append((name, (time.perf_counter() - t0) * 1000))
    return out, timings


# measurements

def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def histogram(img: np.ndarray) -> np.ndarray:
    """256-bin brightness histogram as fractions of all pixels."""
    h = cv2.calcHist([gray(img)], [0], None, [256], [0, 256]).ravel()
    return h / h.sum()


def clipped(img: np.ndarray) -> tuple[float, float]:
    """Fraction of pixels stuck at pure black and pure white."""
    g = gray(img)
    return float((g == 0).mean()), float((g == 255).mean())


def crop(img: np.ndarray, box: list[float]) -> np.ndarray:
    # box is [x0, y0, x1, y1] as fractions of width/height
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def legibility(enhanced: np.ndarray, reference: np.ndarray, box: list[float]) -> float:
    # 0..1: correlation with the clean photo (structure) times how much contrast the region has
    a = gray(crop(enhanced, box)).astype(np.float32).ravel()
    b = gray(crop(reference, box)).astype(np.float32).ravel()
    if a.std() < 1e-6:
        return 0.0
    correlation = float(np.corrcoef(a, b)[0, 1])
    contrast = min(a.std() / b.std(), 1.0)
    return max(correlation, 0.0) * contrast


def evidence_quality(legib: float, new_clipping: float, weight: float = 3) -> float:
    # quality = legibility - weight * share of pixels pushed to pure black/white
    return float(np.clip(legib - weight * new_clipping, 0, 1))


# augmentation (for the side panel)

def random_augment(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    """One random training-time augmentation. Returns the image and what was applied."""
    out = img.copy()
    applied = []
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
        applied.append("flip")
    if rng.random() < 0.7:
        angle = float(rng.uniform(-15, 15))
        h, w = out.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT)
        applied.append(f"rotate {angle:+.0f}°")
    if rng.random() < 0.7:
        g = float(rng.uniform(0.35, 1.4))
        out = gamma_correct(out, g)
        applied.append("darker" if g < 1 else "brighter")
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (5, 5), 0)
        applied.append("blur")
    if rng.random() < 0.4:
        noise = rng.normal(0, 12, out.shape)
        out = np.clip(out + noise, 0, 255).astype(np.uint8)
        applied.append("noise")
    return out, applied or ["unchanged"]
