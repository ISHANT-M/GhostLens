"""The case file: which photo, anchor, brief and wall this run uses, all drawn from one seed."""

import json
import random
import secrets
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CLUE_FILE = Path(__file__).resolve().parent.parent / "assets" / "level1" / "clues.json"

with open(CLUE_FILE) as f:
    _clue_data = json.load(f)

PASS_QUALITY = _clue_data["pass_quality"]
REFERENCE = _clue_data["reference"]       # the settings the tests prove are enough
CLUES = _clue_data["clues"]

# chapter 2: the object the ghost is bound to. labels = ImageNet classes that count as a match
ANCHORS = {
    "pocket watch": {
        "riddle": "keeps hours", "entity": "The Timekeeper", "place": "the desk drawer",
        "labels": frozenset({"stopwatch", "analog clock", "wall clock", "digital clock"}),
        "miss": "Nothing about that keeps hours.",
        "reveal": "ImageNet has no 'pocket watch' class, so the nearest one wins.",
    },
    "padlock": {
        "riddle": "keeps things shut", "entity": "The Gatekeeper", "place": "the desk",
        "labels": frozenset({"padlock", "combination lock"}),
        "miss": "That doesn't lock anything.",
        "reveal": "'padlock' is one of ImageNet's 1000 classes, so it was named exactly.",
    },
    "teddy bear": {
        "riddle": "keeps a child company", "entity": "The Companion", "place": "the armchair",
        "labels": frozenset({"teddy"}),
        "miss": "Nobody's childhood toy.",
        "reveal": "ImageNet's class is 'teddy, teddy bear'; YOLO prints the first name.",
    },
}

# chapter 3: what the client wants from the inventory
BRIEFS = {
    "inventory": {"title": "Full inventory", "rule": "F1 ≥ 0.75", "start": 0.60},
    "miss_nothing": {"title": "Miss nothing", "rule": "recall 100% and precision ≥ 60%", "start": 0.05},
}

# chapter 3: indices into ground_truth.json objects. YOLO26s finds these at every passing threshold
MOVABLE = {
    3: "the cup on the far-left saucer",
    5: "the cup in front of the coffee pot",
    6: "the cup on the far-right saucer",
}

# chapter 4: walls where U-Net Lite stays below 0.95 IoU and Std INT8 clears 0.96
WALL_SEEDS = [28, 181, 237, 403, 457, 494]


def brief_passes(brief: str, scores: dict) -> bool:
    if brief == "miss_nothing":
        return scores["recall"] == 1.0 and scores["precision"] >= 0.6
    return scores["f1"] >= 0.75 - 1e-9     # 6 TP + 3 FP gives exactly 0.75


@dataclass(frozen=True)
class Case:
    seed: int
    clue: dict
    cam04: dict | None
    anchor: str
    evidence_order: tuple[str, ...]
    brief: str
    moved: int
    wall_seed: int
    wall2_seed: int

    @property
    def number(self) -> str:
        return self.clue["answer"]


@lru_cache(maxsize=64)
def build_case(seed: int) -> Case:
    rng = random.Random(seed)
    if len(CLUES) > 1:
        clue, cam04 = rng.sample(CLUES, 2)
    else:
        clue, cam04 = CLUES[0], None
    anchors = sorted(ANCHORS)
    wall_seed, wall2_seed = rng.sample(WALL_SEEDS, 2)
    return Case(
        seed=seed,
        clue=clue,
        cam04=cam04,
        anchor=rng.choice(anchors),
        evidence_order=tuple(rng.sample(anchors, len(anchors))),
        brief=rng.choice(sorted(BRIEFS)),
        moved=rng.choice(sorted(MOVABLE)),
        wall_seed=wall_seed,
        wall2_seed=wall2_seed,
    )


def get_case(store: MutableMapping) -> Case:
    if "case_seed" not in store:
        store["case_seed"] = secrets.randbelow(1_000_000)
    return build_case(store["case_seed"])


def reset_case(store: MutableMapping) -> None:
    store.pop("case_seed", None)


def case_label(store: MutableMapping) -> str:
    return f"CASE {get_case(store).seed % 10000:04d}"
