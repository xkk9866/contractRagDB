"""Build Track-C (ASQA/ALCE) data.

Creates a pooled retrieval corpus = union of all GTR top-100 passages across
the 948 ASQA queries (dedup by doc id). Retrieval plans then operate over this
~90K-passage corpus, which makes retrieval decisions non-trivial (the official
ALCE setup hands each query its own top-100; we re-index to allow plan choice).

Outputs under data/asqa/:
  corpus.jsonl   {pid, title, text}
  queries.jsonl  {qid, question, qa_pairs, answer, gold_pids(top-5 oracle ids)}
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "ALCE", "ALCE-data", "asqa_eval_gtr_top100.json")
ORACLE = os.path.join(ROOT, "external", "ALCE", "ALCE-data",
                      "asqa_eval_gtr_top100_reranked_oracle.json")
OUT = os.path.join(ROOT, "data", "asqa")
os.makedirs(OUT, exist_ok=True)


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    oracle = json.load(open(ORACLE, encoding="utf-8"))
    assert len(data) == len(oracle)
    corpus = {}
    with open(os.path.join(OUT, "queries.jsonl"), "w", encoding="utf-8") as qf:
        for ex, ox in zip(data, oracle):
            for d in ex["docs"]:
                corpus.setdefault(d["id"], {"pid": d["id"], "title": d["title"],
                                            "text": d["text"]})
            gold_pids = [d["id"] for d in ox["docs"][:5]]
            qf.write(json.dumps({
                "qid": str(ex["sample_id"]), "question": ex["question"],
                "qa_pairs": ex["qa_pairs"], "answer": ex.get("answer", ""),
                "gold_pids": gold_pids,
            }, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "corpus.jsonl"), "w", encoding="utf-8") as cf:
        for d in corpus.values():
            cf.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"queries={len(data)} corpus={len(corpus)}")


if __name__ == "__main__":
    main()
