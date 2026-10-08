"""Level 2: whole-image classification with YOLO26n-cls (ImageNet, 1000 classes)."""

import time

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
