"""Level 3: object detection with YOLO26 and precision/recall against our own labelled boxes."""

import time

import cv2
import numpy as np

IOU_MATCH = 0.5


def detect(model, image, imgsz: int = 640, min_conf: float = 0.01) -> tuple[list[dict], float]:
    """Run once at a very low threshold. The slider then filters this list, no need to re-run the model."""
    t0 = time.perf_counter()
    r = model.predict(image, imgsz=imgsz, conf=min_conf, device="cpu", verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000
    dets = [
        {"label": r.names[int(c)], "conf": float(s), "box": [float(v) for v in b]}
        for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
    ]
    return sorted(dets, key=lambda d: -d["conf"]), ms


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(dets: list[dict], truth: list[dict], iou_thr: float = IOU_MATCH) -> dict:
    """Greedy matching, most confident first, same label only. Returns tp, fp and missed."""
    used = set()
    tp, fp = [], []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        best, best_iou = None, iou_thr
        for i, t in enumerate(truth):
            if i in used or t["label"] != d["label"]:
                continue
            v = iou(d["box"], t["box"])
            if v >= best_iou:
                best, best_iou = i, v
        if best is None:
            fp.append(d)
        else:
            used.add(best)
            tp.append({**d, "iou": best_iou, "truth": best})
    missed = [t for i, t in enumerate(truth) if i not in used]
    return {"tp": tp, "fp": fp, "missed": missed}


def scores(m: dict) -> dict:
    tp, fp, fn = len(m["tp"]), len(m["fp"]), len(m["missed"])
    precision = tp / (tp + fp) if tp + fp else 1.0   # nothing predicted, nothing wrong
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(dets: list[dict], truth: list[dict], threshold: float) -> tuple[dict, dict]:
    m = match([d for d in dets if d["conf"] >= threshold], truth)
    return m, scores(m)


MIN_OVERLAP = 0.1   # below this a box doesn't really touch an object

REASONS = {
    "duplicate": "Duplicate: a second box on an object that already has a correct one.",
    "loose": "Loose box: the right label, but it overlaps the object by less than IoU 0.5.",
    "other_class": "Wrong label: it sits on a labelled object of another class.",
    "unlabelled": "Nothing labelled here: no object in our ground truth at this spot.",
}


def false_alarm_reason(d: dict, truth: list[dict], matched: set, iou_thr: float = IOU_MATCH) -> tuple[str, int | None, float]:
    """(reason, truth index, IoU) for one false alarm."""
    same = [(iou(d["box"], t["box"]), i) for i, t in enumerate(truth) if t["label"] == d["label"]]
    other = [(iou(d["box"], t["box"]), i) for i, t in enumerate(truth) if t["label"] != d["label"]]
    v, i = max(same, default=(0.0, None))
    if v >= iou_thr and i in matched:
        return "duplicate", i, v
    if v >= MIN_OVERLAP:
        return "loose", i, v
    v, i = max(other, default=(0.0, None))
    if v >= MIN_OVERLAP:
        return "other_class", i, v
    return "unlabelled", None, v


def explain_false_alarms(m: dict, truth: list[dict], iou_thr: float = IOU_MATCH) -> list[dict]:
    """Every false alarm with the reason it didn't count."""
    matched = {d["truth"] for d in m["tp"]}
    out = []
    for d in m["fp"]:
        reason, i, v = false_alarm_reason(d, truth, matched, iou_thr)
        out.append({"det": d, "reason": reason, "truth": i, "iou": v})
    return out


def yolo_lines(objects: list[dict], width: int, height: int, class_ids: dict[str, int]) -> list[str]:
    """Ground truth boxes as YOLO label lines: class cx cy w h, all as fractions of the image."""
    lines = []
    for o in objects:
        x0, y0, x1, y1 = o["box"]
        cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
        lines.append(f"{class_ids[o['label']]} {cx:.4f} {cy:.4f} {(x1 - x0) / width:.4f} {(y1 - y0) / height:.4f}")
    return lines


def input_scale(shape: tuple, imgsz: int = 640) -> float:
    """YOLO resizes the longest side of the photo to imgsz before the network sees it."""
    return imgsz / max(shape[:2])


def input_size(box: list[float], shape: tuple, imgsz: int = 640) -> tuple[int, int]:
    """(width, height) in pixels of a box once the photo is resized to the model's input."""
    k = input_scale(shape, imgsz)
    return round((box[2] - box[0]) * k), round((box[3] - box[1]) * k)


def zoom_crop(img: np.ndarray, box: list[float], pad: int = 12, factor: int = 3) -> np.ndarray:
    """The box and a little context, blown up with nearest-neighbour so every real pixel stays visible."""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
    x1, y1 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
    crop = img[y0:y1, x0:x1]
    return cv2.resize(crop, (crop.shape[1] * factor, crop.shape[0] * factor), interpolation=cv2.INTER_NEAREST)


COLORS = {"tp": (90, 125, 94), "fp": (60, 74, 156), "missed": (47, 134, 183)}  # BGR: green, red, amber


def draw(img: np.ndarray, m: dict | None = None, dets: list[dict] | None = None) -> np.ndarray:
    """Draw boxes. With a match result: green = correct, red = false alarm, dashed amber = missed."""
    out = img.copy()

    def box(b, color, text, dashed=False):
        x0, y0, x1, y1 = map(int, b)
        if dashed:
            for x in range(x0, x1, 14):
                cv2.line(out, (x, y0), (min(x + 7, x1), y0), color, 2)
                cv2.line(out, (x, y1), (min(x + 7, x1), y1), color, 2)
            for y in range(y0, y1, 14):
                cv2.line(out, (x0, y), (x0, min(y + 7, y1)), color, 2)
                cv2.line(out, (x1, y), (x1, min(y + 7, y1)), color, 2)
        else:
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x0, y0 - th - 6), (x0 + tw + 6, y0), color, -1)
        cv2.putText(out, text, (x0 + 3, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if m is None:
        for d in dets or []:
            box(d["box"], (40, 40, 40), f"{d['label']} {d['conf']:.2f}")
        return out
    for t in m["missed"]:
        box(t["box"], COLORS["missed"], f"missed: {t['label']}", dashed=True)
    for d in m["fp"]:
        box(d["box"], COLORS["fp"], f"{d['label']} {d['conf']:.2f}")
    for d in m["tp"]:
        box(d["box"], COLORS["tp"], f"{d['label']} {d['conf']:.2f}")
    return out
