"""Rebuild train/cal/test matrix JSONLs for expanded splits.

Collects every existing (qid, rung) record across related matrix files into
a cache, writes the new split files by copying cached records, and reports
which (query, rung) pairs still need execution.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.splits import get_splits, SPLIT_SIZES  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def load_matrix_cache(paths):
    cache = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cache[(r["qid"], r["rung"])] = r
    return cache


def write_split(track, split, queries, cache):
    out = os.path.join(EXP, f"{track}_{split}_matrix.jsonl")
    have = miss = 0
    with open(out, "w", encoding="utf-8") as f:
        for q in queries:
            for rung in range(4):
                rec = cache.get((q["qid"], rung))
                if rec is None:
                    miss += 1
                    continue
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                have += 1
    print(f"{track}/{split}: wrote {have} cached records; "
          f"missing {miss} (qid,rung) pairs -> {out}")
    return miss


def main():
    tracks = sys.argv[1:] or ["hybridqa", "crag", "asqa"]
    for track in tracks:
        # gather from any previous matrices for this track
        paths = [os.path.join(EXP, f) for f in os.listdir(EXP)
                 if f.startswith(track) and f.endswith("_matrix.jsonl")]
        # also harvest hybridqa from old cal/test when rebuilding hybridqa
        # (already covered by prefix)
        cache = load_matrix_cache(paths)
        print(f"== {track}: cache size {len(cache)} records from "
              f"{len(paths)} files")
        print(f"   target sizes {SPLIT_SIZES[track]}")
        splits = get_splits(track)
        total_miss = 0
        for name, qs in splits.items():
            print(f"   {name}: {len(qs)} queries")
            total_miss += write_split(track, name, qs, cache)
        print(f"== {track} total missing cells: {total_miss}")


if __name__ == "__main__":
    main()
