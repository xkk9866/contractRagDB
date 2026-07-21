"""Chain-prefix variants for the telescoping composition voucher (Voucher 2.0).

run_rewrites.py produced: base P0, single rewrites P0+R_i (i=1..4), and the
full composition P4 = R4(R3(R2(R1(P0)))). The telescoping bound needs the
intermediate chain prefixes:
  P1 = R1(P0)              -- already exists as variant 1 (prefilter)
  P2 = R2(P1)              -- prefilter + k-trunc            (NEW, index 0 here)
  P3 = R3(P2)              -- prefilter + k-trunc + norerank (NEW, index 1 here)
  P4 = R4(P3)              -- already exists as variant 5 (Rall)

Runs the two missing prefixes on the same 500 cal + 500 test queries.
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

CHAIN = [
    {**BASE, "full_pool": False, "k_ret": 8, "k_ctx": 4},                  # P2
    {**BASE, "full_pool": False, "k_ret": 8, "k_ctx": 4, "rerank": False}, # P3
]


def main():
    splits = get_splits("hybridqa")
    by_qid = {q["qid"]: q for s in splits.values() for q in s}
    track = HybridQATrack(rungs=CHAIN)
    for split in ["cal", "test"]:
        # run on EXACTLY the qids of the existing base rewrites matrix (which
        # was executed on an earlier split ordering)
        base_path = os.path.join(ROOT, "experiments",
                                 f"hybridqa_{split}_rewrites.jsonl")
        qids = []
        seen = set()
        with open(base_path, encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)["qid"]
                if q not in seen:
                    seen.add(q)
                    qids.append(q)
        qs = [by_qid[q] for q in qids if q in by_qid]
        out = os.path.join(ROOT, "experiments",
                           f"hybridqa_{split}_rewrites_chain.jsonl")
        run_ladder_matrix(track, qs, out, max_workers=24)
    print("CHAIN_DONE")


if __name__ == "__main__":
    main()
