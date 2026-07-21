"""Track C: ASQA (ALCE) — citation-completeness contracts.

Ladder (cost ascending):
  rung 0: dense top-5 -> qwen-flash, inline citations
  rung 1: hybrid top-20 -> rerank -> 5 -> qwen-flash
  rung 2: hybrid top-40 -> rerank -> 8 -> qwen-plus
  rung 3: hybrid top-60 -> rerank -> 10 -> qwen-max + targeted repair of
          unsupported sentences (one extra retrieval+rewrite round)

Contract loss (joint): 1{citation_recall < tau_c OR str_em < tau_q}, evaluated
with the official ALCE protocol (TRUE-NLI for citations) offline.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from contractrag.retrieval import (embed_texts, rerank_scores, rrf_fuse,
                                   top_k_margin, PoolBM25)
from contractrag.verify import entail_probs, split_sentences

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data", "asqa")

RUNGS = [
    dict(k_ret=5, k_ctx=5, rerank=False, model="qwen-flash", repair=False),
    dict(k_ret=20, k_ctx=5, rerank=True, model="qwen-flash", repair=False),
    dict(k_ret=40, k_ctx=8, rerank=True, model="qwen-plus", repair=False),
    dict(k_ret=60, k_ctx=10, rerank=True, model="qwen-max", repair=True),
]

CITE_INSTR = (
    "Write an accurate, engaging answer to the question using ONLY the provided "
    "search results. Every sentence MUST cite at least one source using the "
    "format [k] where k is the document number (cite at most three per "
    "sentence). Write at most five sentences. Do not add information that is "
    "not in the documents."
)


class AsqaTrack:
    name = "asqa"
    n_rungs = len(RUNGS)

    def __init__(self):
        self.corpus = {}
        self.pids = []
        with open(os.path.join(DATA, "corpus.jsonl"), encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self.corpus[d["pid"]] = d
                self.pids.append(d["pid"])
        ids = json.load(open(os.path.join(DATA, "corpus_emb_ids.json"), encoding="utf-8"))
        assert ids == self.pids or set(ids) == set(self.pids)
        self.pids = ids
        emb = np.load(os.path.join(DATA, "corpus_emb.npy"))
        self.emb = np.asarray(emb, dtype=np.float32)
        # global BM25 over the pooled corpus
        import bm25s
        self._bm25_retriever = None
        self._bm25s = bm25s

    def _bm25(self):
        if self._bm25_retriever is None:
            cache = os.path.join(DATA, "bm25s_index")
            texts = [self.corpus[p]["title"] + "\n" + self.corpus[p]["text"]
                     for p in self.pids]
            if os.path.exists(cache):
                self._bm25_retriever = self._bm25s.BM25.load(cache)
            else:
                tok = self._bm25s.tokenize(texts, show_progress=False)
                r = self._bm25s.BM25()
                r.index(tok, show_progress=False)
                r.save(cache)
                self._bm25_retriever = r
        return self._bm25_retriever

    # ------------------------------------------------------------------
    def retrieve(self, query, rung: int):
        cfg = RUNGS[rung]
        t0 = time.time()
        q = query["question"]
        feats = {}
        q_emb = embed_texts([q], is_query=True)[0]
        d_scores = self.emb @ q_emb
        d_order = np.argsort(-d_scores)[:max(cfg["k_ret"], 20)]
        feats["dense_top"] = float(d_scores[d_order[0]])
        feats["dense_margin"] = top_k_margin(d_scores[d_order], 5)

        if cfg["rerank"]:
            bm = self._bm25()
            qtok = self._bm25s.tokenize([q], show_progress=False)
            res, _sc = bm.retrieve(qtok, k=cfg["k_ret"], show_progress=False)
            bm_idx = [int(i) for i in res[0]]
            cand_idx = list(dict.fromkeys(list(d_order[:cfg["k_ret"]]) + bm_idx))
            cand_txts = [self.corpus[self.pids[i]] for i in cand_idx]
            rr = rerank_scores(q, [c["title"] + "\n" + c["text"][:1200]
                                   for c in cand_txts])
            feats["rerank_top"] = float(rr.max())
            feats["rerank_mean3"] = float(np.sort(rr)[::-1][:3].mean())
            order2 = np.argsort(-rr)[:cfg["k_ctx"]]
            sel = [cand_idx[i] for i in order2]
        else:
            sel = list(d_order[:cfg["k_ctx"]])
            feats["rerank_top"] = 0.0
            feats["rerank_mean3"] = 0.0

        passages = []
        for i in sel:
            d = self.corpus[self.pids[i]]
            passages.append({"pid": d["pid"], "title": d["title"],
                             "text": d["text"][:1000]})
        evidence = {"passages": passages}
        return evidence, feats, time.time() - t0

    # ------------------------------------------------------------------
    def make_prompt(self, query, evidence):
        docs = "\n\n".join(
            f"Document [{i+1}](Title: {p['title']}): {p['text']}"
            for i, p in enumerate(evidence["passages"]))
        return [
            {"role": "system", "content": CITE_INSTR},
            {"role": "user", "content":
             f"{docs}\n\nQuestion: {query['question']}\nAnswer:"},
        ]

    def model_for(self, rung):
        return RUNGS[rung]["model"]

    # ------------------------------------------------------------------
    @staticmethod
    def parse_citations(answer, n_docs):
        """Returns list of (sentence, [doc indices]) with 0-based indices."""
        import re
        out = []
        for sent in split_sentences(answer):
            cits = [int(m) - 1 for m in re.findall(r"\[(\d+)\]", sent)]
            cits = [c for c in cits if 0 <= c < n_docs]
            out.append((re.sub(r"\s*\[\d+\]", "", sent), cits))
        return out

    def internal_citation_score(self, answer, evidence):
        """Cheap NLI-based citation coverage (runtime feature, not the metric)."""
        parsed = self.parse_citations(answer, len(evidence["passages"]))
        if not parsed:
            return 0.0, 0.0
        covered = []
        pairs, owners = [], []
        for si, (sent, cits) in enumerate(parsed):
            if not cits:
                covered.append(0.0)
                continue
            concat = " ".join(evidence["passages"][c]["text"][:600] for c in cits[:3])
            pairs.append((concat, sent))
            owners.append(si)
            covered.append(None)
        if pairs:
            ps = entail_probs(pairs)
            for si, p in zip(owners, ps):
                covered[si] = float(p > 0.5)
        vals = [c for c in covered if c is not None]
        cov = float(np.mean([c if c is not None else 0.0 for c in covered]))
        frac_cited = float(np.mean([1.0 if c is not None else 0.0 for c in covered]))
        return cov, frac_cited

    def answer_features(self, query, rung, evidence, answer):
        cov, frac_cited = self.internal_citation_score(answer, evidence)
        return {"nli_cite_cov": cov, "frac_cited": frac_cited,
                "n_sents": float(len(split_sentences(answer))),
                "abstain": 0.0}

    # ------------------------------------------------------------------
    def plan_repair(self, query, rung, evidence, answer, afeats):
        """Targeted repair for the top rung: returns None (no repair) or a
        dict {messages, evidence} for a batched rewrite call."""
        if not RUNGS[rung]["repair"]:
            return None
        if afeats.get("nli_cite_cov", 1.0) >= 0.99:
            return None
        parsed = self.parse_citations(answer, len(evidence["passages"]))
        bad = []
        for sent, cits in parsed:
            if not cits:
                bad.append(sent)
                continue
            concat = " ".join(evidence["passages"][c]["text"][:600] for c in cits[:3])
            if float(entail_probs([(concat, sent)])[0]) < 0.5:
                bad.append(sent)
        if not bad:
            return None
        # targeted retrieval for the first two unsupported sentences
        extra = []
        for sent in bad[:2]:
            q_emb = embed_texts([sent], is_query=True)[0]
            scores = self.emb @ q_emb
            for i in np.argsort(-scores)[:2]:
                d = self.corpus[self.pids[int(i)]]
                if all(p["pid"] != d["pid"] for p in evidence["passages"]):
                    extra.append({"pid": d["pid"], "title": d["title"],
                                  "text": d["text"][:1000]})
        evidence2 = {"passages": evidence["passages"] + extra[:4]}
        docs = "\n\n".join(
            f"Document [{i+1}](Title: {p['title']}): {p['text']}"
            for i, p in enumerate(evidence2["passages"]))
        msgs = [
            {"role": "system", "content": CITE_INSTR},
            {"role": "user", "content":
             f"{docs}\n\nQuestion: {query['question']}\n\n"
             f"Draft answer (some sentences lack supporting citations):\n{answer}\n\n"
             "Rewrite the answer so EVERY sentence is fully supported by and "
             "cites the given documents. Remove claims you cannot support.\n"
             "Revised answer:"},
        ]
        return {"messages": msgs, "evidence": evidence2}

    def evidence_meta(self, evidence):
        return {"pids": [p["pid"] for p in evidence["passages"]]}
