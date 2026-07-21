"""Rebuild score JSONLs for expanded splits from a global (qid,rung) cache."""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.splits import get_splits  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def load_score_cache(track):
    cache = {}
    for f in os.listdir(EXP):
        if not (f.startswith(track) and f.endswith("_scores.jsonl")):
            continue
        # skip stale/fresh tags for the main track rebuild
        tag = f[: -len("_scores.jsonl")]
        # e.g. hybridqa_train, crag_cal, asqa_test — not cragfresh_*
        parts = tag.split("_")
        if parts[0] != track:
            continue
        if len(parts) != 2 or parts[1] not in ("train", "cal", "test"):
            # also accept bak / anything with track_ prefix that looks like scores
            pass
        path = os.path.join(EXP, f)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cache[(r["qid"], r["rung"])] = r
    # also harvest from bak if present
    bak = os.path.join(EXP, "bak_pre_expand")
    if os.path.isdir(bak):
        for f in os.listdir(bak):
            if f.startswith(track) and f.endswith("_scores.jsonl"):
                with open(os.path.join(bak, f), encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        cache.setdefault((r["qid"], r["rung"]), r)
    return cache


def main():
    tracks = sys.argv[1:] or ["hybridqa", "crag", "asqa"]
    for track in tracks:
        # Prefer bak scores + current scores
        cache = {}
        for folder in [os.path.join(EXP, "bak_pre_expand"), EXP]:
            if not os.path.isdir(folder):
                continue
            for f in os.listdir(folder):
                if not f.endswith("_scores.jsonl"):
                    continue
                if not f.startswith(track + "_"):
                    continue
                # main track only: hybridqa_*, crag_train|cal|test, asqa_*
                rest = f[len(track) + 1: -len("_scores.jsonl")]
                if rest not in ("train", "cal", "test") and folder == EXP:
                    # still load train/cal/test-named only for current dir to
                    # avoid clobber from cragfresh; bak may have only main ones
                    if not any(rest.startswith(s) for s in
                               ("train", "cal", "test")):
                        continue
                with open(os.path.join(folder, f), encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        cache[(r["qid"], r["rung"])] = r
        print(f"== {track}: score cache {len(cache)}")
        splits = get_splits(track)
        for name, qs in splits.items():
            out = os.path.join(EXP, f"{track}_{name}_scores.jsonl")
            have = miss = 0
            with open(out, "w", encoding="utf-8") as fh:
                for q in qs:
                    for rung in range(4):
                        rec = cache.get((q["qid"], rung))
                        if rec is None:
                            miss += 1
                            continue
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        have += 1
            print(f"{track}/{name}: wrote {have}, missing {miss} -> {out}")


if __name__ == "__main__":
    main()
