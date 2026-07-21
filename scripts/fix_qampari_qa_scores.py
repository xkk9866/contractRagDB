"""Recompute the QAMPARI answer metrics (rec_top5/prec/f1_top5) in the
scores files with the official normalization (strip citations, truncate at
first newline), keeping the TRUE-NLI citation metrics untouched."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "external", "ALCE"))

import eval as alce_eval  # noqa: E402

from scripts.run_ladder import get_splits  # noqa: E402


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    qsplits = get_splits("qampari")
    for split in ["train", "cal", "test"]:
        gold = {q["qid"]: q for q in qsplits[split]}
        mat = {(r["qid"], r["rung"]): r for r in load_jsonl(
            os.path.join(ROOT, "experiments", f"qampari_{split}_matrix.jsonl"))}
        path = os.path.join(ROOT, "experiments", f"qampari_{split}_scores.jsonl")
        rows = load_jsonl(path)
        for r in rows:
            rec = mat[(r["qid"], r["rung"])]
            out = rec["answer"].strip().split("\n")[0]
            out = alce_eval.remove_citations(out)
            g = gold[r["qid"]]
            m = alce_eval.compute_qampari_f1(
                [{"output": out, "answers": g["answers"]}])
            r["qampari_rec_top5"] = m["qampari_rec_top5"]
            r["qampari_prec"] = m["qampari_prec"]
            r["qampari_f1_top5"] = m["qampari_f1_top5"]
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: fixed {len(rows)} rows")


if __name__ == "__main__":
    main()
