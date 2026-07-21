"""Expanded, deterministic query splits for ContractRAG experiments.

Design goals (finite-sample certificates + systems-paper scale):
  - Maximize calibration size (Hoeffding--Bentkus radius shrinks as 1/sqrt(n)).
  - Use every available public query where the track is small (CRAG, ASQA).
  - Keep HybridQA test = full official HybridQA-dev (3466); train/cal drawn
    from the much larger official train set.

Sizes (seed=42):
  HybridQA: train=5000, cal=3000 (from train.jsonl), test=3466 (all of
            official_dev). Previously 800/1200/2266.
  CRAG:     train=700,  cal=1000, test=1006 over the full 2706-query pool
            (previously 400/971/1335 on official split0/1 halves).
  ASQA:     train=150,  cal=350,  test=448 over the full 948-query ALCE
            release (previously 200/300/448); cal enlarged, test kept at
            the full original held-out size.
"""
from __future__ import annotations

import json
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# publicly documented targets
SPLIT_SIZES = {
    "hybridqa": {"train": 5000, "cal": 3000, "test": 3466},
    "crag":     {"train": 700,  "cal": 1000, "test": 1006},
    "asqa":     {"train": 150,  "cal": 350,  "test": 448},
    "qampari":  {"train": 150,  "cal": 350,  "test": 500},
}


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def get_splits(track_name, seed: int = 42):
    rng = random.Random(seed)
    if track_name == "hybridqa":
        train_all = load_jsonl(os.path.join(ROOT, "data", "hybridqa",
                                            "queries_train.jsonl"))
        # normalize field name used by the track
        for q in train_all:
            if "query" not in q and "question" in q:
                q["query"] = q["question"]
        dev = load_jsonl(os.path.join(ROOT, "data", "hybridqa",
                                      "queries_dev.jsonl"))
        for q in dev:
            if "query" not in q and "question" in q:
                q["query"] = q["question"]
        rng.shuffle(train_all)
        # official_dev is held out entirely as the test set (full release)
        # shuffle for reproducibility of any subsample experiments
        rng.shuffle(dev)
        n_tr = SPLIT_SIZES["hybridqa"]["train"]
        n_cal = SPLIT_SIZES["hybridqa"]["cal"]
        assert n_tr + n_cal <= len(train_all)
        return {
            "train": train_all[:n_tr],
            "cal": train_all[n_tr:n_tr + n_cal],
            "test": dev,  # all 3466
        }
    if track_name == "crag":
        qs = load_jsonl(os.path.join(ROOT, "data", "crag", "queries.jsonl"))
        rng.shuffle(qs)
        n_tr = SPLIT_SIZES["crag"]["train"]
        n_cal = SPLIT_SIZES["crag"]["cal"]
        assert n_tr + n_cal <= len(qs)
        return {
            "train": qs[:n_tr],
            "cal": qs[n_tr:n_tr + n_cal],
            "test": qs[n_tr + n_cal:],
        }
    if track_name in ("asqa", "qampari"):
        qs = load_jsonl(os.path.join(ROOT, "data", track_name, "queries.jsonl"))
        for q in qs:
            if "query" not in q and "question" in q:
                q["query"] = q["question"]
        rng.shuffle(qs)
        n_tr = SPLIT_SIZES[track_name]["train"]
        n_cal = SPLIT_SIZES[track_name]["cal"]
        assert n_tr + n_cal <= len(qs)
        return {
            "train": qs[:n_tr],
            "cal": qs[n_tr:n_tr + n_cal],
            "test": qs[n_tr + n_cal:],
        }
    raise SystemExit(f"unknown track: {track_name}")
