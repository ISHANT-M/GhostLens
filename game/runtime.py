"""Cached access to models and benchmark numbers, shared by all chapters."""

import streamlit as st

from cv import edge
from cv import segmentation as seg
from cv.models import load_yolo


@st.cache_resource(show_spinner=False)
def yolo(weights: str):
    return load_yolo(weights)


@st.cache_resource(show_spinner=False)
def unet(variant: str, int8: bool):
    model = seg.load(variant)
    if int8:
        model = seg.quantize(model, seg.make_dataset(64, edge.CALIBRATION_SEED)[0])
    return model


@st.cache_data(show_spinner=False)
def benchmark() -> dict:
    # normally written by setup_models.py; measured here once if it's missing
    return edge.load_benchmark() or edge.benchmark_all(log=lambda *_: None)


def profile(group: str, model_id: str) -> dict:
    return next(r for r in benchmark()[group] if r["id"] == model_id)

