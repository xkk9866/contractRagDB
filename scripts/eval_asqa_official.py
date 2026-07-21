"""Official ALCE evaluation of ASQA ladder outputs.

Converts matrix records to ALCE format and calls the official eval functions
(compute_str_em, compute_autoais with google/t5_xxl_true_nli_mixture) from
external/ALCE/eval.py. Writes experiments/asqa_{split}_scores.jsonl with
{qid, rung, str_em, citation_rec, citation_prec}.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ALCE = os.path.join(ROOT, "external", "ALCE")
sys.path.insert(0, ALCE)

os.chdir(ALCE)  # eval.py expects its own utils on path

import eval as alce_eval  # noqa: E402  (official ALCE eval module)

# point the official evaluator at the locally downloaded TRUE-NLI weights
LOCAL_TRUE = os.path.expanduser(
    "~/.cache/modelscope/hub/models/google/t5_xxl_true_nli_mixture")
if os.path.isdir(LOCAL_TRUE):
    alce_eval.AUTOAIS_MODEL = LOCAL_TRUE


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all")
    args = ap.parse_args()
    splits = ["train", "cal", "test"] if args.split == "all" else [args.split]

    from scripts.run_ladder import get_splits
    qsplits = get_splits("asqa")

    for split in splits:
        gold = {q["qid"]: q for q in qsplits[split]}
        matrix = load_jsonl(os.path.join(ROOT, "experiments",
                                         f"asqa_{split}_matrix.jsonl"))
        out_path = os.path.join(ROOT, "experiments", f"asqa_{split}_scores.jsonl")
        done = set()
        if os.path.exists(out_path):
            done = {(r["qid"], r["rung"]) for r in load_jsonl(out_path)}
        corpus = {}
        with open(os.path.join(ROOT, "data", "asqa", "corpus.jsonl"),
                  encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                corpus[d["pid"]] = d

        items, keys = [], []
        for r in matrix:
            if r["qid"] not in gold or (r["qid"], r["rung"]) in done:
                continue
            g = gold[r["qid"]]
            docs = [{"title": corpus[pid]["title"], "text": corpus[pid]["text"]}
                    for pid in r["evidence_meta"].get("pids", []) if pid in corpus]
            items.append({"question": g["question"], "qa_pairs": g["qa_pairs"],
                          "output": r["answer"], "docs": docs})
            keys.append((r["qid"], r["rung"]))
        if not items:
            print(f"{split}: nothing to score")
            continue
        print(f"{split}: scoring {len(items)} records")

        # per-item str_em (official implementation computes mean; call per item)
        strs = []
        for it in items:
            em, _hit = alce_eval.compute_str_em([it])
            strs.append(em)

        # citation metrics need per-item values: call compute_autoais per item
        # (model is loaded once inside the module)
        recs, precs = [], []
        for i, it in enumerate(items):
            m = alce_eval.compute_autoais([it], qampari=False, at_most_citations=3)
            recs.append(m["citation_rec"])
            precs.append(m["citation_prec"])
            if (i + 1) % 100 == 0:
                print(f"  autoais {i+1}/{len(items)}", flush=True)

        with open(out_path, "a", encoding="utf-8") as f:
            for (qid, rung), em, cr, cp in zip(keys, strs, recs, precs):
                f.write(json.dumps({"qid": qid, "rung": rung, "str_em": em,
                                    "citation_rec": cr, "citation_prec": cp}) + "\n")
        print(f"{split}: wrote {len(keys)} scores")


if __name__ == "__main__":
    main()
