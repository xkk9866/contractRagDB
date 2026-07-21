"""Build Track-D (QAMPARI/ALCE) data.

Creates a pooled retrieval corpus = union of all GTR top-100 passages across
the 1000 QAMPARI queries (dedup by doc id), mirroring the ASQA build. QAMPARI
answers are lists of entities; ALCE scores answer recall (rec-5) plus citation
recall/precision with the comma-split protocol.

Outputs under data/qampari/:
  corpus.jsonl   {pid, title, text}
  queries.jsonl  {qid, question, answers, answer, gold_pids(top-5 oracle ids)}
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "external", "ALCE", "ALCE-data",
                   "qampari_eval_gtr_top100.json")
ORACLE = os.path.join(ROOT, "external", "ALCE", "ALCE-data",
                      "qampari_eval_gtr_top100_reranked_oracle.json")
OUT = os.path.join(ROOT, "data", "qampari")
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
                "qid": str(ex["id"]), "question": ex["question"],
                "answers": ex["answers"], "answer": ex.get("answer", ""),
                "gold_pids": gold_pids,
            }, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "corpus.jsonl"), "w", encoding="utf-8") as cf:
        for d in corpus.values():
            cf.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"queries={len(data)} corpus={len(corpus)}")


if __name__ == "__main__":
    main()
