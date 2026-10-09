"""Level 2: whole-image classification with YOLO26n-cls (ImageNet, 1000 classes)."""

import time

import numpy as np

INPUT_SIZE = 224


def classify(model, image, k: int = 5) -> dict:
    """Top-k labels with softmax probabilities, plus how long the forward pass took."""
    t0 = time.perf_counter()
    result = model.predict(image, imgsz=INPUT_SIZE, device="cpu", verbose=False)[0]
    total_ms = (time.perf_counter() - t0) * 1000
    probs = result.probs
    top = [(result.names[i].replace("_", " "), float(c)) for i, c in zip(probs.top5[:k], probs.top5conf[:k])]
    return {
        "top": top,
        "rest": max(0.0, 1.0 - sum(p for _, p in top)),
        "model_ms": result.speed["inference"],
        "total_ms": total_ms,
        "num_classes": len(result.names),
    }


# sliding window: classify crops of a big photo, one label per crop

WINDOW_SCALE = 0.4375     # 560x374 on the 1280x854 room photo, 3x3 grid with overlap


def room_windows(w: int, h: int) -> dict[str, tuple[int, int, int, int]]:
    """Window id -> (x0, y0, x1, y1). 'full' is the whole photo, 'half-{col}-{row}' a 3x3 grid that covers it.

    The grid windows are a bit smaller than half the photo (see the chapter 2 data test: at exactly half,
    the top-right window of the TV and bare wall came out as 'wall clock' on the light model).
    """
    ww, wh = round(w * WINDOW_SCALE), round(h * WINDOW_SCALE)
    xs = (0, (w - ww) // 2, w - ww)
    ys = (0, (h - wh) // 2, h - wh)
    wins = {"full": (0, 0, w, h)}
    for col, x in enumerate(xs):
        for row, y in enumerate(ys):
            wins[f"half-{col}-{row}"] = (x, y, x + ww, y + wh)
    return wins


def coverage(boxes, w: int, h: int) -> float:
    """Share of the photo inside at least one box (overlaps counted once)."""
    covered = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        covered[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return float(covered.mean())
