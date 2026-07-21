"""Retrieval operators: pool BM25, dense embeddings, cross-encoder reranker.

All GPU models are process-wide singletons loaded lazily on CUDA device 0/1
(chooses the device with more free memory at load time).
"""
from __future__ import annotations

import math
import os
import threading
from collections import Counter

import numpy as np

_lock = threading.Lock()
_embedder = None
_reranker = None


def pick_device():
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}"


def get_embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=pick_device())
            _embedder.max_seq_length = 512
        return _embedder


def get_reranker():
    global _reranker
    with _lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=pick_device(),
                                     max_length=512)
        return _reranker


def embed_texts(texts, batch_size=256, is_query=False, show_progress=False):
    m = get_embedder()
    if is_query:
        texts = ["Represent this sentence for searching relevant passages: " + t
                 for t in texts]
    emb = m.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=show_progress)
    return emb.astype(np.float32)


def rerank_scores(query: str, passages: list[str], batch_size=64):
    ce = get_reranker()
    import torch
    with torch.inference_mode():
        raw = ce.predict([(query, p) for p in passages], batch_size=batch_size,
                         show_progress_bar=False)
    return np.asarray(raw, dtype=np.float32)  # already sigmoid in ST>=3


def rerank_scores_batch(pairs: list[tuple[str, str]], batch_size=128):
    ce = get_reranker()
    import torch
    with torch.inference_mode():
        raw = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return np.asarray(raw, dtype=np.float32)


class PoolBM25:
    """BM25 scoring over a small per-query candidate pool (numpy, no index)."""

    def __init__(self, docs: list[str], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs_tokens = [self._tok(d) for d in docs]
        self.doc_len = np.array([len(t) for t in self.docs_tokens], dtype=np.float32)
        self.avgdl = max(1.0, float(self.doc_len.mean())) if len(docs) else 1.0
        self.N = len(docs)
        self.df = Counter()
        for toks in self.docs_tokens:
            self.df.update(set(toks))
        self.tfs = [Counter(t) for t in self.docs_tokens]

    @staticmethod
    def _tok(s):
        return [w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()
                if len(w) > 1]

    def scores(self, query: str) -> np.ndarray:
        q = self._tok(query)
        out = np.zeros(self.N, dtype=np.float32)
        for term in q:
            df = self.df.get(term)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            for i, tf in enumerate(self.tfs):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


def rrf_fuse(rank_lists: list[np.ndarray], k=60) -> np.ndarray:
    """Reciprocal-rank fusion. rank_lists are score arrays over the same pool."""
    n = len(rank_lists[0])
    fused = np.zeros(n, dtype=np.float32)
    for scores in rank_lists:
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        fused += 1.0 / (k + ranks + 1)
    return fused


def top_k_margin(scores: np.ndarray, k: int) -> float:
    """Normalized score gap between rank-1 and rank-k (retrieval confidence)."""
    if len(scores) == 0:
        return 0.0
    s = np.sort(scores)[::-1]
    top = float(s[0])
    kth = float(s[min(k, len(s)) - 1])
    if abs(top) < 1e-9:
        return 0.0
    return (top - kth) / (abs(top) + 1e-9)
