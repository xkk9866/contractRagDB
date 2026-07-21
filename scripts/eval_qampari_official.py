"""Official ALCE evaluation of QAMPARI ladder outputs.

Converts matrix records to ALCE format and calls the official eval functions
(compute_qampari_f1 for answer recall/precision, compute_autoais with
qampari=True and google/t5_xxl_true_nli_mixture for citations) from
external/ALCE/eval.py. Writes experiments/qampari_{split}_scores.jsonl with
{qid, rung, qampari_rec_top5, qampari_prec, qampari_f1_top5,
 citation_rec, citation_prec}.
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
    qsplits = get_splits("qampari")

    for split in splits:
        gold = {q["qid"]: q for q in qsplits[split]}
        matrix = load_jsonl(os.path.join(ROOT, "experiments",
                                         f"qampari_{split}_matrix.jsonl"))
        out_path = os.path.join(ROOT, "experiments",
                                f"qampari_{split}_scores.jsonl")
        done = set()
        if os.path.exists(out_path):
            done = {(r["qid"], r["rung"]) for r in load_jsonl(out_path)}
        corpus = {}
        with open(os.path.join(ROOT, "data", "qampari", "corpus.jsonl"),
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
                    for pid in r["evidence_meta"].get("pids", [])
                    if pid in corpus]
            items.append({"question": g["question"], "answers": g["answers"],
                          "output": r["answer"], "docs": docs})
            keys.append((r["qid"], r["rung"]))
        if not items:
            print(f"{split}: nothing to score")
            continue
        print(f"{split}: scoring {len(items)} records", flush=True)

        # per-item answer metrics (official comma-split protocol). The
        # official main() strips citations and truncates at the first
        # newline before computing QA metrics; replicate that here.
        qa = []
        for it in items:
            out = it["output"].strip().split("\n")[0]
            norm = dict(it, output=alce_eval.remove_citations(out))
            m = alce_eval.compute_qampari_f1([norm])
            qa.append(m)

        # citation metrics: official qampari mode, at most 3 citations
        recs, precs = [], []
        for i, it in enumerate(items):
            m = alce_eval.compute_autoais([it], qampari=True,
                                          at_most_citations=3)
            recs.append(m["citation_rec"])
            precs.append(m["citation_prec"])
            if (i + 1) % 100 == 0:
                print(f"  autoais {i+1}/{len(items)}", flush=True)

        with open(out_path, "a", encoding="utf-8") as f:
            for (qid, rung), m, cr, cp in zip(keys, qa, recs, precs):
                f.write(json.dumps({
                    "qid": qid, "rung": rung,
                    "qampari_rec_top5": m["qampari_rec_top5"],
                    "qampari_prec": m["qampari_prec"],
                    "qampari_f1_top5": m["qampari_f1_top5"],
                    "citation_rec": cr, "citation_prec": cp}) + "\n")
        print(f"{split}: wrote {len(keys)} scores")


if __name__ == "__main__":
    main()
