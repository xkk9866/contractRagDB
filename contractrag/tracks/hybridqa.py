"""Track A: HybridQA (table + text hybrid planning).

Ladder (cost ascending):
  rung 0: lexical row select (6) + BM25 passages over row-restricted pool (4)
          -> qwen-flash                                   [pre-filter, cheap]
  rung 1: rows (10) + hybrid BM25+dense RRF top-10 -> rerank -> 5 passages
          over row-restricted pool -> qwen-flash
  rung 2: rows (15) + hybrid over FULL table pool top-16 -> rerank -> 8
          -> qwen-plus                                    [post-filter, safer]
  rung 3: rows (20) + hybrid full pool top-24 -> rerank -> 12 + NLI evidence
          filter -> qwen-max

Quality: token-F1 vs gold answer; contract loss = 1{F1 < tau_q}.
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
DATA = os.path.join(ROOT, "data", "hybridqa")

RUNGS = [
    dict(rows=6, k_ret=8, k_ctx=4, rerank=False, dense=False, full_pool=False,
         model="qwen-flash", nli_filter=False),
    dict(rows=10, k_ret=10, k_ctx=5, rerank=True, dense=True, full_pool=False,
         model="qwen-flash", nli_filter=False),
    dict(rows=15, k_ret=16, k_ctx=8, rerank=True, dense=True, full_pool=True,
         model="qwen-plus", nli_filter=False),
    dict(rows=20, k_ret=24, k_ctx=12, rerank=True, dense=True, full_pool=True,
         model="qwen-max", nli_filter=True),
]


class HybridQATrack:
    name = "hybridqa"

    def __init__(self, rungs=None):
        self.rungs = rungs if rungs is not None else RUNGS
        self.n_rungs = len(self.rungs)
        self.tables = {}
        with open(os.path.join(DATA, "tables.jsonl"), encoding="utf-8") as f:
            for line in f:
                t = json.loads(line)
                self.tables[t["table_id"]] = t
        self.passages = {}
        with open(os.path.join(DATA, "passages.jsonl"), encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                self.passages[p["pid"]] = p["text"]
        ids = json.load(open(os.path.join(DATA, "passages_emb_ids.json"), encoding="utf-8"))
        self.pid2idx = {pid: i for i, pid in enumerate(ids)}
        self.emb = np.load(os.path.join(DATA, "passages_emb.npy"), mmap_mode="r")

    # ------------------------------------------------------------------
    @staticmethod
    def _tok(s):
        return set(w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()
                   if len(w) > 1)

    def select_rows(self, table, question, n):
        """Lexical overlap row scoring; returns (indices, scores)."""
        q = self._tok(question)
        scores = []
        for i, row in enumerate(table["rows"]):
            cell_toks = self._tok(" ".join(str(c) for c in row))
            inter = len(q & cell_toks)
            scores.append(inter)
        order = np.argsort(-np.array(scores), kind="stable")[:n]
        return order.tolist(), [scores[i] for i in order]

    def serialize_rows(self, table, row_idx):
        header = " | ".join(table["header"])
        lines = [f"Table: {table['title']} — {table['section_title']}", header]
        for i in row_idx:
            lines.append(" | ".join(str(c) for c in table["rows"][i]))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def retrieve(self, query, rung: int):
        """Returns (evidence dict, retrieval features, retrieval latency)."""
        cfg = self.rungs[rung]
        t0 = time.time()
        table = self.tables[query["table_id"]]
        row_idx, row_scores = self.select_rows(table, query["question"], cfg["rows"])

        if cfg["full_pool"]:
            pool_pids = sorted({p for links in table["row_links"] for p in links})
        else:
            pool_pids = sorted({p for i in row_idx for p in table["row_links"][i]})
        pool_pids = [p for p in pool_pids if p in self.passages][:800]
        pool_texts = [self.passages[p] for p in pool_pids]

        feats = {}
        chosen, ret_scores = [], np.zeros(0)
        if pool_texts:
            bm = PoolBM25(pool_texts)
            bm_scores = bm.scores(query["question"])
            feats["bm25_top"] = float(bm_scores.max())
            feats["bm25_margin"] = top_k_margin(bm_scores, 5)
            if cfg["dense"]:
                q_emb = embed_texts([query["question"]], is_query=True)[0]
                idxs = [self.pid2idx[p] for p in pool_pids]
                d_scores = np.asarray(self.emb[idxs], dtype=np.float32) @ q_emb
                feats["dense_top"] = float(d_scores.max())
                feats["dense_margin"] = top_k_margin(d_scores, 5)
                fused = rrf_fuse([bm_scores, d_scores])
            else:
                fused = bm_scores
            order = np.argsort(-fused)[:cfg["k_ret"]]
            cand = [(pool_pids[i], pool_texts[i]) for i in order]
            if cfg["rerank"] and cand:
                rr = rerank_scores(query["question"], [c[1][:1500] for c in cand])
                feats["rerank_top"] = float(rr.max())
                feats["rerank_mean3"] = float(np.sort(rr)[::-1][:3].mean())
                order2 = np.argsort(-rr)[:cfg["k_ctx"]]
                chosen = [cand[i] for i in order2]
                ret_scores = rr[order2]
            else:
                chosen = cand[:cfg["k_ctx"]]
                ret_scores = fused[order][:cfg["k_ctx"]]
        feats.setdefault("rerank_top", 0.0)
        feats.setdefault("rerank_mean3", 0.0)
        feats.setdefault("dense_top", 0.0)
        feats.setdefault("dense_margin", 0.0)
        feats["row_top_overlap"] = float(row_scores[0]) if row_scores else 0.0
        feats["n_pool"] = len(pool_pids)

        evidence = {
            "rows_text": self.serialize_rows(table, row_idx),
            "passages": [{"pid": p, "text": t[:1200]} for p, t in chosen],
        }
        return evidence, feats, time.time() - t0

    # ------------------------------------------------------------------
    def make_prompt(self, query, evidence):
        ptxt = "\n\n".join(f"[Passage {i+1}] {p['text']}"
                           for i, p in enumerate(evidence["passages"]))
        return [
            {"role": "system", "content":
             "You answer questions using the given table and passages. "
             "Reply with ONLY the short answer span (a few words), no explanation. "
             "If the evidence is insufficient, reply exactly: I don't know."},
            {"role": "user", "content":
             f"{evidence['rows_text']}\n\n{ptxt}\n\nQuestion: {query['question']}\nAnswer:"},
        ]

    def model_for(self, rung):
        return self.rungs[rung]["model"]

    def answer_features(self, query, rung, evidence, answer):
        """Post-generation sufficiency features (NLI support etc.)."""
        ev_texts = [p["text"] for p in evidence["passages"]] + [evidence["rows_text"]]
        support = answer_support_score(
            f"The answer to '{query['question']}' is {answer}.", ev_texts)
        return {"nli_support": support, "abstain": float(is_abstain(answer)),
                "ans_len": float(len(answer.split()))}
