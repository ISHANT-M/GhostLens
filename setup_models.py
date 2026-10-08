"""One-time setup: downloads YOLO26 weights, trains the U-Nets and benchmarks everything.

Usage: python setup_models.py [--retrain]
"""

import os
import sys
from pathlib import Path

os.environ["YOLO_OFFLINE"] = "0"  # the download needs it on, even if the shell set it to 1

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
WEIGHTS = ["yolo26n-cls.pt", "yolo26s-cls.pt", "yolo26m-cls.pt", "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26s-seg.pt"]


def download_weights() -> None:
    from ultralytics import YOLO

    MODELS.mkdir(exist_ok=True)
    for name in WEIGHTS:
        if (MODELS / name).exists():
            print(f"ok          models/{name}")
            continue
        cwd = os.getcwd()
        os.chdir(MODELS)  # ultralytics saves a bare filename into the working directory
        try:
            YOLO(name)
        finally:
            os.chdir(cwd)
        print(f"downloaded  models/{name}")


def train_segmentation(force: bool) -> None:
    from cv import segmentation as seg

    todo = [v for v in seg.VARIANTS if force or not seg.weights_path(v).exists()]
    for v in seg.VARIANTS:
        if v not in todo:
            print(f"ok          models/stain_unet_{v}.pt")
    if not todo:
        return
    print("generating  3000 training walls")
    data, val = seg.make_dataset(3000, 0), seg.make_dataset(150, 999)
    for v in todo:
        print(f"training    U-Net {v} (width {seg.VARIANTS[v]})")
        model, iou = seg.train(seg.VARIANTS[v], data, val)
        seg.save(model, iou, v)
        print(f"saved       models/stain_unet_{v}.pt (validation IoU {iou:.3f})")


def benchmark() -> None:
    from cv import edge

    print("measuring   every model on this machine (CPU)")
    result = edge.benchmark_all()
    for group, rows in result.items():
        for r in rows:
            print(f"            {r['name']:16s} {r.get('precision', ''):4s} {r['size_mb']:6.2f} MB  {r['latency_ms']:6.1f} ms")


if __name__ == "__main__":
    download_weights()
    os.environ["YOLO_OFFLINE"] = "1"
    train_segmentation(force="--retrain" in sys.argv)
    benchmark()
    print("done. run: streamlit run app.py")
