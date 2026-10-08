"""Model paths and a clear error when a model file is missing."""

import os
from pathlib import Path

os.environ.setdefault("YOLO_OFFLINE", "1")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelMissing(Exception):
    pass


def model_path(name: str) -> Path:
    path = MODELS_DIR / name
    if not path.exists():
        raise ModelMissing(f"`models/{name}` is not installed yet. Run `python setup_models.py` "
                           "(see README) and reload the page.")
    return path


def load_yolo(name: str):
    from ultralytics import YOLO  # slow import, only when a level actually needs it
    return YOLO(str(model_path(name)))


def file_size_mb(name: str) -> float:
    return model_path(name).stat().st_size / 1e6


def param_count(model) -> int:
    return sum(p.numel() for p in model.model.parameters())
