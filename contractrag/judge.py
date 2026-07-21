"""Answer-quality judging.

CRAG-style three-way accuracy judge (correct / incorrect / missing) using a
strong LLM with the gold answer(s) available. Deterministic (temperature 0)
and cached, so judging is reproducible.
"""
from __future__ import annotations

from contractrag.llm import get_llm, Usage
from contractrag.textutil import is_abstain, normalize_answer

JUDGE_PROMPT = """You are grading a question-answering system.

Question: {query}
(Question was asked at: {qtime})
Gold answer: {gold}
Other acceptable answers: {alt}

System answer: {pred}

Grade the system answer:
- "correct" if it conveys the same fact(s) as the gold answer (wording may differ; extra correct detail is fine).
- "missing" if the system declined to answer or said it does not know.
- "incorrect" if it states a different or wrong fact.

Reply with exactly one word: correct, missing, or incorrect."""


def judge_crag_batch(items, model="qwen-max", max_workers=40):
    """items: list of {query, qtime, gold, alt, pred}. Returns list of labels."""
    llm = get_llm(max_workers)
    usage = Usage()
    msgs_list = []
    fast_labels = {}
    for i, it in enumerate(items):
        pred = (it["pred"] or "").strip()
        if is_abstain(pred):
            fast_labels[i] = "missing"
            msgs_list.append(None)
            continue
        if normalize_answer(pred) == normalize_answer(str(it["gold"])):
            fast_labels[i] = "correct"
            msgs_list.append(None)
            continue
        alt = [str(a) for a in (it.get("alt") or [])]
        msgs_list.append([{"role": "user", "content": JUDGE_PROMPT.format(
            query=it["query"], qtime=it.get("qtime", "n/a"), gold=str(it["gold"]),
            alt=", ".join(alt) or "none", pred=pred[:800])}])
    todo_idx = [i for i, m in enumerate(msgs_list) if m is not None]
    resps = llm.chat_batch(model, [msgs_list[i] for i in todo_idx],
                           max_tokens=8, usage=usage)
    labels = [None] * len(items)
    for i, lab in fast_labels.items():
        labels[i] = lab
    for i, r in zip(todo_idx, resps):
        t = (r["text"] or "").strip().lower()
        if "correct" in t and "incorrect" not in t:
            labels[i] = "correct"
        elif "missing" in t:
            labels[i] = "missing"
        else:
            labels[i] = "incorrect"
    return labels, usage
