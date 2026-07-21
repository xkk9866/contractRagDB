"""Rewrite-certificate experiment (HybridQA).

Base plan P*: rows=15, hybrid retrieval over FULL table pool k=16 -> rerank
-> 8 passages -> qwen-plus  (= ladder rung 2).

Single-rewrite variants (each cost-reducing, possibly loss-increasing):
  R1 prefilter   : restrict passage pool to selected rows (pre-filter pushdown)
  R2 ktrunc      : k_ret 16->8, k_ctx 8->4
  R3 norerank    : drop the cross-encoder rerank
  R4 gendown     : qwen-plus -> qwen-flash
  Rall           : all four composed

For each rewrite R_i we certify eps_i = Clopper-Pearson upper bound on
P(loss(R_i(P)) > loss(P)) at confidence 1-delta_r, plus the mean loss delta.
Composition: union bound sum(eps_i) vs. direct certificate on Rall.

Runs on the first 500 cal queries + 500 test queries of the standard split.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.engine import run_ladder_matrix  # noqa: E402
from contractrag.tracks.hybridqa import HybridQATrack  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402

BASE = dict(rows=15, k_ret=16, k_ctx=8, rerank=True, dense=True, full_pool=True,
            model="qwen-plus", nli_filter=False)

VARIANTS = [
    BASE,                                                    # 0 base
    {**BASE, "full_pool": False},                            # 1 R1 prefilter
    {**BASE, "k_ret": 8, "k_ctx": 4},                        # 2 R2 ktrunc
    {**BASE, "rerank": False},                               # 3 R3 norerank
    {**BASE, "model": "qwen-flash"},                         # 4 R4 gendown
    {**BASE, "full_pool": False, "k_ret": 8, "k_ctx": 4,
     "rerank": False, "model": "qwen-flash"},                # 5 Rall
]


def main():
    splits = get_splits("hybridqa")
    track = HybridQATrack(rungs=VARIANTS)
    for split in ["cal", "test"]:
        qs = splits[split][:500]
        out = os.path.join(ROOT, "experiments", f"hybridqa_{split}_rewrites.jsonl")
        run_ladder_matrix(track, qs, out, max_workers=28)
    print("REWRITES_DONE")


if __name__ == "__main__":
    main()
