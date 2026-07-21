"""Entailment verifier used by the Verify operator and by CiteCheck.

Wraps cross-encoder/nli-deberta-v3-base (premise=evidence, hypothesis=claim).
Returns P(entailment) via softmax over (contradiction, entailment, neutral).
"""
from __future__ import annotations

import threading

import numpy as np

_lock = threading.Lock()
_nli = None


def get_nli():
    global _nli
    with _lock:
        if _nli is None:
            from sentence_transformers import CrossEncoder
            from contractrag.retrieval import pick_device
            _nli = CrossEncoder("cross-encoder/nli-deberta-v3-base",
                                device=pick_device(), max_length=512)
        return _nli


def entail_probs(pairs: list[tuple[str, str]], batch_size=64) -> np.ndarray:
    """pairs = (premise/evidence, hypothesis/claim) -> P(entailment)."""
    if not pairs:
        return np.zeros(0, dtype=np.float32)
    ce = get_nli()
    import torch
    with torch.inference_mode():
        logits = ce.predict(pairs, batch_size=batch_size, show_progress_bar=False,
                            apply_softmax=False)
    logits = np.asarray(logits, dtype=np.float32)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs[:, 1]  # entailment index per id2label


def max_entailment(claim: str, evidences: list[str], batch_size=64) -> float:
    """Best supporting evidence probability for a claim."""
    if not evidences:
        return 0.0
    ps = entail_probs([(ev[:1800], claim) for ev in evidences], batch_size)
    return float(ps.max())


def split_sentences(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def answer_support_score(answer: str, evidences: list[str], max_sents=6) -> float:
    """Mean max-entailment over answer sentences (runtime sufficiency feature)."""
    sents = split_sentences(answer)[:max_sents]
    if not sents:
        return 0.0
    scores = [max_entailment(s, evidences[:8]) for s in sents]
    return float(np.mean(scores))
