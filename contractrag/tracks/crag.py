"""Track B: CRAG (web pages + mock KG API, freshness-aware planning).

Ladder (cost ascending):
  rung 0: page snippets -> qwen-flash
  rung 1: BM25 top-8 chunks -> qwen-flash
  rung 2: hybrid RRF top-16 -> rerank -> 6 -> qwen-plus
  rung 3: rung-2 evidence + KG API evidence (precomputed tool calls, age 0)
          -> qwen-max

Freshness: each chunk carries age_hours (query_time - page_last_modified).
Under contract Delta_f, rungs drop web evidence older than Delta_f; the KG rung
is the fresh source. Quality loss = hallucination (answered and judged wrong);
abstention is tracked separately (missing).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from contractrag.retrieval import (PoolBM25, embed_texts, rerank_scores,
                                   rrf_fuse, top_k_margin)
from contractrag.verify import answer_support_score
from contractrag.textutil import is_abstain

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "crag")

RUNGS = [
    dict(mode="snippets", model="qwen-flash"),
    dict(mode="bm25", k_ctx=8, model="qwen-flash"),
    dict(mode="hybrid", k_ret=16, k_ctx=6, model="qwen-plus"),
    dict(mode="hybrid+kg", k_ret=16, k_ctx=6, model="qwen-max"),
]


class CragTrack:
    name = "crag"
    n_rungs = len(RUNGS)

    def __init__(self, freshness_hours: float | None = None,
                 staleness_shift_hours: float = 0.0):
        """freshness_hours: contract Delta_f (None = no constraint).
        staleness_shift_hours: simulated index lag added to all web ages
        (drift experiments)."""
        self.freshness_hours = freshness_hours
        self.staleness_shift = staleness_shift_hours
        self.queries = {}
        with open(os.path.join(DATA, "queries.jsonl"), encoding="utf-8") as f:
            for line in f:
                q = json.loads(line)
                self.queries[q["qid"]] = q
        self.kg_dir = os.path.join(DATA, "kg_evidence")
        self.emb_dir = os.path.join(DATA, "chunk_emb")

    def load_chunks(self, qid):
        return json.load(open(os.path.join(DATA, "chunks", f"{qid}.json"),
                              encoding="utf-8"))

    def _age_ok(self, age_hours):
        if self.freshness_hours is None:
            return True
        if age_hours is None:
            return True  # unknown age treated as fresh-at-crawl (crawl ~ query_time)
        return (age_hours + self.staleness_shift) <= self.freshness_hours

    # ------------------------------------------------------------------
    def retrieve(self, query, rung: int):
        cfg = RUNGS[rung]
        t0 = time.time()
        qid = query["qid"]
        chunks = self.load_chunks(qid)
        feats = {}
        passages, ages = [], []

        if cfg["mode"] == "snippets":
            for c in chunks:
                if c["cid"].endswith("snippet") and self._age_ok(c.get("age_hours")):
                    passages.append({"text": c["text"][:800], "src": "web",
                                     "age": c.get("age_hours")})
                    ages.append(c.get("age_hours"))
            passages = passages[:6]
        else:
            texts = [c["text"] for c in chunks]
            keep = [i for i, c in enumerate(chunks) if self._age_ok(c.get("age_hours"))]
            feats["frac_fresh"] = len(keep) / max(1, len(chunks))
            if keep:
                pool = [texts[i] for i in keep]
                bm = PoolBM25(pool)
                bm_scores = bm.scores(query["query"])
                feats["bm25_top"] = float(bm_scores.max()) if len(pool) else 0.0
                feats["bm25_margin"] = top_k_margin(bm_scores, 5)
                if cfg["mode"].startswith("hybrid"):
                    emb_path = os.path.join(self.emb_dir, f"{qid}.npy")
                    d_scores = None
                    if os.path.exists(emb_path):
                        emb = np.load(emb_path)
                        if emb.shape[0] == len(chunks):
                            q_emb = embed_texts([query["query"]], is_query=True)[0]
                            d_scores = (np.asarray(emb[keep], dtype=np.float32) @ q_emb)
                    if d_scores is None:
                        d_scores = bm_scores
                    feats["dense_top"] = float(d_scores.max())
                    feats["dense_margin"] = top_k_margin(d_scores, 5)
                    fused = rrf_fuse([bm_scores, d_scores])
                    order = np.argsort(-fused)[:cfg["k_ret"]]
                    cand = [(keep[i], pool[i]) for i in order]
                    rr = rerank_scores(query["query"], [c[1][:1500] for c in cand])
                    feats["rerank_top"] = float(rr.max()) if len(rr) else 0.0
                    feats["rerank_mean3"] = float(np.sort(rr)[::-1][:3].mean()) if len(rr) else 0.0
                    order2 = np.argsort(-rr)[:cfg["k_ctx"]]
                    sel = [cand[i][0] for i in order2]
                else:
                    order = np.argsort(-bm_scores)[:cfg["k_ctx"]]
                    sel = [keep[i] for i in order]
                for i in sel:
                    passages.append({"text": chunks[i]["text"][:1200], "src": "web",
                                     "age": chunks[i].get("age_hours")})
                    ages.append(chunks[i].get("age_hours"))

        if cfg["mode"] == "hybrid+kg":
            kg_path = os.path.join(self.kg_dir, f"{qid}.json")
            if os.path.exists(kg_path):
                kg = json.load(open(kg_path, encoding="utf-8"))
                for item in kg.get("evidence", [])[:4]:
                    passages.append({"text": item[:1500], "src": "kg", "age": 0.0})
                    ages.append(0.0)
                feats["kg_n"] = float(len(kg.get("evidence", [])))
            else:
                feats["kg_n"] = 0.0

        feats.setdefault("bm25_top", 0.0)
        feats.setdefault("bm25_margin", 0.0)
        feats.setdefault("dense_top", 0.0)
        feats.setdefault("dense_margin", 0.0)
        feats.setdefault("rerank_top", 0.0)
        feats.setdefault("rerank_mean3", 0.0)
        feats.setdefault("frac_fresh", 1.0)
        feats["n_evidence"] = float(len(passages))
        known_ages = [a for a in ages if a is not None]
        feats["min_age_h"] = float(min(known_ages)) if known_ages else -1.0

        evidence = {"passages": passages, "query_time": query.get("query_time")}
        return evidence, feats, time.time() - t0

    # ------------------------------------------------------------------
    def make_prompt(self, query, evidence):
        ptxt = "\n\n".join(
            f"[{'KG' if p['src']=='kg' else 'Web'} {i+1}"
            + (f", age {p['age']:.0f}h" if p.get("age") is not None else "")
            + f"] {p['text']}"
            for i, p in enumerate(evidence["passages"]))
        return [
            {"role": "system", "content":
             "You are a precise QA assistant. Use ONLY the provided evidence to "
             "answer. The current time is given; prefer fresher evidence for "
             "time-sensitive questions. Give a concise answer (a phrase or one "
             "sentence). If the evidence is insufficient or conflicting, reply "
             "exactly: I don't know."},
            {"role": "user", "content":
             f"Current time: {evidence.get('query_time')}\n\nEvidence:\n{ptxt}\n\n"
             f"Question: {query['query']}\nAnswer:"},
        ]

    def model_for(self, rung):
        return RUNGS[rung]["model"]

    def answer_features(self, query, rung, evidence, answer):
        ev_texts = [p["text"] for p in evidence["passages"]]
        support = answer_support_score(
            f"The answer to '{query['query']}' is {answer}.", ev_texts) if ev_texts else 0.0
        return {"nli_support": support, "abstain": float(is_abstain(answer)),
                "ans_len": float(len(answer.split()))}

    def evidence_meta(self, evidence):
        ages = [p.get("age") for p in evidence["passages"]]
        return {"srcs": [p["src"] for p in evidence["passages"]],
                "ages": ages}
