"""Text normalization and QA metrics (SQuAD-style EM/F1)."""
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(pred: str, gold: str) -> float:
    p = normalize_answer(pred).split()
    g = normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def em_score(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def best_f1(pred: str, golds: list[str]) -> float:
    return max((f1_score(pred, g) for g in golds if g), default=0.0)


IDK_PATTERNS = [
    "i don't know", "i do not know", "cannot answer", "can't answer",
    "not sure", "unable to determine", "no information", "unknown",
    "insufficient information", "not enough information", "cannot determine",
]


def is_abstain(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(p in t[:200] for p in IDK_PATTERNS)
