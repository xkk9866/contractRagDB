"""Staleness drift matrices for the CRAG freshness experiment (RQ6).

Runs the CRAG ladder under a freshness contract (Delta_f = 72h) in two index
states on the expanded cal and test splits:
  fresh: staleness shift 0h    -> experiments/cragfresh_{split}_matrix.jsonl
  stale: staleness shift +168h -> experiments/cragstale_{split}_matrix.jsonl

The stale state models a week-old index snapshot: web evidence ages past the
contract and is filtered at plan time; the KG rung remains fresh. Real
retrieval, real generation; only the index-age metadata is shifted.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.engine import run_ladder_matrix  # noqa: E402
from contractrag.tracks.crag import CragTrack  # noqa: E402
from scripts.splits import get_splits  # noqa: E402


def main():
    splits = get_splits("crag")
    configs = [("cragfresh", 0.0), ("cragstale", 168.0)]
    for tag, shift in configs:
        track = CragTrack(freshness_hours=72.0, staleness_shift_hours=shift)
        for split in ("cal", "test"):
            qs = splits[split]
            out = os.path.join(ROOT, "experiments", f"{tag}_{split}_matrix.jsonl")
            print(f"=== {tag}/{split}: {len(qs)} queries -> {out}", flush=True)
            run_ladder_matrix(track, qs, out, max_workers=28)
    print("CRAG_STALE_DONE")


if __name__ == "__main__":
    main()
