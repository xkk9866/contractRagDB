"""Run the ladder matrix for a track over its query splits.

Usage:
  python scripts/run_ladder.py hybridqa [--split train|cal|test|all] [--rungs 0,1,2,3] [--limit N]
  python scripts/run_ladder.py crag ...
  python scripts/run_ladder.py asqa ...

Splits (seed 42) — see scripts/splits.py:
  hybridqa: train=5000, cal=3000 (official train), test=3466 (full official_dev)
  crag:     train=700,  cal=1000, test=1006 (full 2706-query pool)
  asqa:     train=250,  cal=400,  test=298  (full 948-query ALCE release)
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.engine import run_ladder_matrix  # noqa: E402
from scripts.splits import get_splits  # noqa: E402


def make_track(track_name):
    if track_name == "hybridqa":
        from contractrag.tracks.hybridqa import HybridQATrack
        return HybridQATrack()
    if track_name == "crag":
        from contractrag.tracks.crag import CragTrack
        return CragTrack()
    if track_name == "asqa":
        from contractrag.tracks.asqa import AsqaTrack
        return AsqaTrack()
    if track_name == "qampari":
        from contractrag.tracks.qampari import QampariTrack
        return QampariTrack()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--split", default="all")
    ap.add_argument("--rungs", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=40)
    args = ap.parse_args()

    splits = get_splits(args.track)
    names = list(splits) if args.split == "all" else [args.split]
    track = make_track(args.track)
    for name in names:
        qs = splits[name]
        if args.limit:
            qs = qs[:args.limit]
        out = os.path.join(ROOT, "experiments",
                           f"{args.track}_{name}_matrix.jsonl")
        print(f"=== {args.track} / {name}: {len(qs)} queries -> {out}")
        rung_filter = None
        if args.rungs is not None:
            rung_filter = [int(x) for x in args.rungs.split(",")]
        run_ladder_matrix(track, qs, out, max_workers=args.workers,
                          rung_filter=rung_filter)


if __name__ == "__main__":
    main()
