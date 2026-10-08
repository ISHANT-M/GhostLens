"""Measure size and speed of every model once and save it to models/benchmark.json."""

import json
import statistics
import time

import cv2
import torch

from cv import detection as det
from cv import segmentation as seg
from cv.models import MODELS_DIR, file_size_mb, load_yolo

BENCHMARK_FILE = MODELS_DIR / "benchmark.json"
ROOT = MODELS_DIR.parent

# published accuracy from the Ultralytics docs (top-1 on ImageNet, mAP50-95 on COCO)
CLASSIFIERS = {
    "yolo26n-cls.pt": {"name": "YOLO26n-cls", "tier": "Light", "published": "71.4% top-1"},
    "yolo26s-cls.pt": {"name": "YOLO26s-cls", "tier": "Balanced", "published": "76.0% top-1"},
    "yolo26m-cls.pt": {"name": "YOLO26m-cls", "tier": "Heavy", "published": "78.1% top-1"},
}
DETECTORS = {
    "yolo26n.pt": {"name": "YOLO26n", "tier": "Light", "published": "40.9 mAP"},
    "yolo26s.pt": {"name": "YOLO26s", "tier": "Balanced", "published": "48.6 mAP"},
    "yolo26m.pt": {"name": "YOLO26m", "tier": "Heavy", "published": "53.1 mAP"},
}
SEGMENTER = {"yolo26s-seg.pt": {"name": "YOLO26s-seg", "tier": "Balanced", "published": "40.0 mask mAP"}}
CALIBRATION_SEED = 555  # INT8 calibration walls, kept apart from the validation walls (seed 999)
UNETS = {"lite": "Lite", "standard": "Standard", "pro": "Pro"}


def _median_ms(fn, runs: int = 12, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(times), 1)


def _yolo_row(weights: str, info: dict, image, imgsz: int) -> tuple[dict, object]:
    model = load_yolo(weights)
    ms = _median_ms(lambda: model.predict(image, imgsz=imgsz, device="cpu", verbose=False))
    return {"id": weights, **info, "size_mb": round(file_size_mb(weights), 1), "latency_ms": ms}, model


def benchmark_all(log=print) -> dict:
    out = {"classifiers": [], "detectors": [], "segmenters": [], "unets": []}
    padlock = cv2.imread(str(ROOT / "assets/level2/padlock.jpg"))
    parlour = cv2.imread(str(ROOT / "assets/level3/antique_room.jpg"))
    truth = json.loads((ROOT / "assets/level3/ground_truth.json").read_text())["objects"]

    for w, info in CLASSIFIERS.items():
        log(f"  {w}")
        row, _ = _yolo_row(w, info, padlock, 224)
        out["classifiers"].append(row)

    for w, info in DETECTORS.items():
        log(f"  {w}")
        row, model = _yolo_row(w, info, parlour, 640)
        dets, _ = det.detect(model, parlour)
        best = max((det.evaluate(dets, truth, t / 20)[1] for t in range(1, 19)), key=lambda s: s["f1"])
        row["best_recall"] = round(max(det.evaluate(dets, truth, t / 20)[1]["recall"] for t in range(1, 19)), 2)
        row["best_f1"] = round(best["f1"], 2)
        out["detectors"].append(row)

    for w, info in SEGMENTER.items():
        log(f"  {w}")
        row, _ = _yolo_row(w, info, parlour, 640)
        out["segmenters"].append(row)

    scene, _ = seg.make_scene()
    x = seg.to_tensor(scene)[None]
    val = seg.make_dataset(150, 999)
    calib = seg.make_dataset(64, CALIBRATION_SEED)[0]
    for variant, name in UNETS.items():
        log(f"  U-Net {variant}")
        model = seg.load(variant)
        for precision, m in (("FP32", model), ("INT8", seg.quantize(model, calib))):
            with torch.no_grad():
                ms = _median_ms(lambda: m(x), runs=8, warmup=2)
            out["unets"].append({
                "id": f"{variant}-{precision.lower()}", "variant": variant, "name": f"U-Net {name}",
                "precision": precision, "params_k": round(sum(p.numel() for p in model.parameters()) / 1e3, 1),
                "size_mb": round(seg.state_size_mb(m), 2), "latency_ms": ms, "iou": round(seg.evaluate(m, val), 3),
            })

    BENCHMARK_FILE.write_text(json.dumps(out, indent=2))
    return out


def load_benchmark() -> dict | None:
    if BENCHMARK_FILE.exists():
        data = json.loads(BENCHMARK_FILE.read_text())
        if "classifiers" in data:
            return data
    return None


def energy_units(latency_ms: float) -> int:
    """Gameplay abstraction: one battery unit per 5 ms of measured compute, at least 1 per model run."""
    return max(1, int(latency_ms / 5 + 0.5))
