"""Attach quality losses to ladder matrices.

hybridqa: token-F1 / EM vs gold (local).
crag:     three-way judge (correct/missing/incorrect) via qwen-max (cached).

Writes experiments/{track}_{split}_scores.jsonl: {qid, rung, f1/em or label}.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.textutil import best_f1, em_score, is_abstain  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def score_hybridqa(split):
    from scripts.run_ladder import get_splits
    gold = {q["qid"]: q for q in get_splits("hybridqa")[split]}
    recs = load_jsonl(os.path.join(ROOT, "experiments", f"hybridqa_{split}_matrix.jsonl"))
    out = []
    for r in recs:
        if r["qid"] not in gold:
            continue
        g = gold[r["qid"]]["answer"]
        f1 = best_f1(r["answer"], [g])
        out.append({"qid": r["qid"], "rung": r["rung"], "f1": f1,
                    "em": em_score(r["answer"], g),
                    "abstain": float(is_abstain(r["answer"]))})
    path = os.path.join(ROOT, "experiments", f"hybridqa_{split}_scores.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"hybridqa {split}: {len(out)} scored")


def score_crag(split, tag="crag"):
    from scripts.run_ladder import get_splits
    from contractrag.judge import judge_crag_batch
    gold = {q["qid"]: q for q in get_splits("crag")[split]}
    recs = load_jsonl(os.path.join(ROOT, "experiments", f"{tag}_{split}_matrix.jsonl"))
    recs = [r for r in recs if r["qid"] in gold]
    items = []
    for r in recs:
        g = gold[r["qid"]]
        items.append({"query": g["query"], "qtime": g.get("query_time"),
                      "gold": g["answer"], "alt": g.get("alt_ans", []),
                      "pred": r["answer"]})
    print(f"judging {len(items)} answers ...")
    labels, usage = judge_crag_batch(items, model="qwen-max")
    print(f"judge cost: {usage.cost_cny:.2f} CNY over {usage.calls} calls")
    path = os.path.join(ROOT, "experiments", f"{tag}_{split}_scores.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r, lab in zip(recs, labels):
            f.write(json.dumps({"qid": r["qid"], "rung": r["rung"],
                                "label": lab}) + "\n")
    print(f"{tag} {split}: {len(recs)} scored")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--split", default="all")
    args = ap.parse_args()
    splits = ["train", "cal", "test"] if args.split == "all" else [args.split]
    for s in splits:
        if args.track == "hybridqa":
            score_hybridqa(s)
        elif args.track == "crag":
            score_crag(s)
        elif args.track in ("cragfresh", "cragstale"):
            if s == "train":
                continue
            score_crag(s, tag=args.track)
