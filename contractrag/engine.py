"""Ladder execution engine.

Executes every rung of a track's ladder on a set of queries, recording per
(query, rung): answer, cost (CNY), tokens, derived latency, sufficiency
features and quality losses. The full matrix enables exact offline evaluation
of any stopping policy plus calibration.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from contractrag.llm import get_llm, Usage, call_cost_cny

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments")
os.makedirs(RESULTS, exist_ok=True)


def run_ladder_matrix(track, queries, out_path, max_workers=40, gen_max_tokens=256,
                      checkpoint_every=200, rung_filter=None):
    """Execute all rungs for all queries; write JSONL records incrementally.

    Record schema:
      {qid, rung, answer, model, cost_cny, prompt_tokens, completion_tokens,
       latency_retrieval, latency_llm, features{...}}
    """
    llm = get_llm(max_workers)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["qid"], r["rung"]))
                except Exception:
                    pass
    out_f = open(out_path, "a", encoding="utf-8")

    for rung in range(track.n_rungs):
        if rung_filter is not None and rung not in rung_filter:
            continue
        todo = [q for q in queries if (q["qid"], rung) not in done]
        if not todo:
            print(f"rung {rung}: all cached")
            continue
        print(f"rung {rung}: {len(todo)} queries")

        # 1) retrieval (local, sequential; GPU batches inside)
        retrievals = []
        t0 = time.time()
        for i, q in enumerate(todo):
            ev, feats, lat = track.retrieve(q, rung)
            retrievals.append((ev, feats, lat))
            if (i + 1) % 200 == 0:
                print(f"  retrieve {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

        # 2) generation (parallel LLM)
        model = track.model_for(rung)
        prompts = [track.make_prompt(q, ev) for q, (ev, _, _) in zip(todo, retrievals)]
        usage = Usage()
        t0 = time.time()
        resps = llm.chat_batch(model, prompts, max_tokens=gen_max_tokens, usage=usage)
        print(f"  generate: {len(todo)} calls, {usage.cost_cny:.3f} CNY, "
              f"{time.time()-t0:.0f}s", flush=True)

        # 3) answer features (local GPU models; sequential loop, batched inside)
        answers = [(r["text"] or "").strip() for r in resps]
        extra_costs = [0.0] * len(todo)
        extra_lat = [0.0] * len(todo)
        afeats_list = []
        t0 = time.time()
        for i, (q, (ev, feats, lat_r)) in enumerate(zip(todo, retrievals)):
            afeats_list.append(track.answer_features(q, rung, ev, answers[i]))
            if (i + 1) % checkpoint_every == 0:
                print(f"  features {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)

        # 3b) optional repair round: plan sequentially (GPU), generate in batch
        if hasattr(track, "plan_repair"):
            plans = []
            for q, (ev, _f, _l), ans, af in zip(todo, retrievals, answers, afeats_list):
                plans.append(track.plan_repair(q, rung, ev, ans, af))
            idx = [i for i, p in enumerate(plans) if p is not None]
            if idx:
                print(f"  repair: {len(idx)} queries", flush=True)
                rresps = llm.chat_batch(model, [plans[i]["messages"] for i in idx],
                                        max_tokens=400, usage=usage)
                for i, rr in zip(idx, rresps):
                    new_ans = (rr["text"] or "").strip()
                    if new_ans:
                        answers[i] = new_ans
                        ev = plans[i].get("evidence", retrievals[i][0])
                        retrievals[i] = (ev, retrievals[i][1], retrievals[i][2])
                        afeats_list[i] = track.answer_features(todo[i], rung, ev, new_ans)
                    extra_costs[i] = call_cost_cny(model, rr["prompt_tokens"],
                                                   rr["completion_tokens"],
                                                   rr.get("gpu_s"))
                    extra_lat[i] = rr["latency"]

        for i, (q, (ev, feats, lat_r), resp) in enumerate(zip(todo, retrievals, resps)):
            feats = {**feats, **afeats_list[i]}
            rec = {
                "qid": q["qid"], "rung": rung, "answer": answers[i], "model": model,
                "cost_cny": call_cost_cny(model, resp["prompt_tokens"],
                                          resp["completion_tokens"],
                                          resp.get("gpu_s")) + extra_costs[i],
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "latency_retrieval": lat_r,
                "latency_llm": resp["latency"] + extra_lat[i],
                "features": feats,
                "evidence_meta": track.evidence_meta(ev) if hasattr(track, "evidence_meta") else {},
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print("ladder matrix complete:", out_path)


# ---------------------------------------------------------------------------
# Operator sharing / materialization: retrieval is generator-independent, so
# we run it once per (query, rung) and materialize the evidence + features +
# retrieval latency. Any number of generator families then reuse the cached
# retrieval view, paying only generation cost. This is the RAG analogue of a
# materialized common-subexpression / operator-sharing optimization.
# ---------------------------------------------------------------------------

def materialize_evidence(track, queries, out_path, rung_filter=None,
                         checkpoint_every=200):
    """Execute only the retrieval operators for all (query, rung); write
    JSONL {qid, rung, evidence, features, latency_retrieval, evidence_meta}."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["qid"], r["rung"]))
                except Exception:
                    pass
    out_f = open(out_path, "a", encoding="utf-8")
    for rung in range(track.n_rungs):
        if rung_filter is not None and rung not in rung_filter:
            continue
        todo = [q for q in queries if (q["qid"], rung) not in done]
        if not todo:
            print(f"[evidence] rung {rung}: all cached")
            continue
        print(f"[evidence] rung {rung}: {len(todo)} queries", flush=True)
        t0 = time.time()
        for i, q in enumerate(todo):
            ev, feats, lat = track.retrieve(q, rung)
            rec = {"qid": q["qid"], "rung": rung, "evidence": ev,
                   "features": feats, "latency_retrieval": lat,
                   "evidence_meta": track.evidence_meta(ev)
                   if hasattr(track, "evidence_meta") else {}}
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % checkpoint_every == 0:
                out_f.flush()
                print(f"  retrieve {i+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
        out_f.flush()
    out_f.close()
    print("evidence materialized:", out_path)


def generate_from_evidence(track, queries, evidence_path, model_map, out_path,
                           max_workers=40, gen_max_tokens=256):
    """Generate answers for one generator family from a materialized evidence
    view, reusing the shared retrieval work.

    model_map: rung -> generator model id (overrides track.model_for).
    Recorded cost is generation-only; latency_retrieval is copied from the
    shared materialized view. Supports track.plan_repair when present.
    """
    llm = get_llm(max_workers)
    qmap = {q["qid"]: q for q in queries}
    ev_by = {}
    with open(evidence_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["qid"] in qmap:
                ev_by.setdefault(r["rung"], []).append(r)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["qid"], r["rung"]))
                except Exception:
                    pass
    out_f = open(out_path, "a", encoding="utf-8")
    for rung in sorted(ev_by):
        recs = [r for r in ev_by[rung] if (r["qid"], rung) not in done]
        if not recs:
            print(f"[gen] rung {rung}: all cached")
            continue
        model = model_map[rung]
        prompts = [track.make_prompt(qmap[r["qid"]], r["evidence"]) for r in recs]
        usage = Usage()
        t0 = time.time()
        resps = llm.chat_batch(model, prompts, max_tokens=gen_max_tokens, usage=usage)
        print(f"[gen] rung {rung} ({model}): {len(recs)} calls, "
              f"{usage.cost_cny:.3f} CNY, {time.time()-t0:.0f}s", flush=True)
        answers = [(resp["text"] or "").strip() for resp in resps]
        extra_cost = [0.0] * len(recs)
        extra_lat = [0.0] * len(recs)
        evs = [r["evidence"] for r in recs]
        afeats = [track.answer_features(qmap[r["qid"]], rung, evs[i], answers[i])
                  for i, r in enumerate(recs)]
        if hasattr(track, "plan_repair"):
            plans = [track.plan_repair(qmap[recs[i]["qid"]], rung, evs[i],
                                       answers[i], afeats[i]) for i in range(len(recs))]
            idx = [i for i, p in enumerate(plans) if p is not None]
            if idx:
                rresps = llm.chat_batch(model, [plans[i]["messages"] for i in idx],
                                        max_tokens=400, usage=usage)
                for i, rr in zip(idx, rresps):
                    na = (rr["text"] or "").strip()
                    if na:
                        answers[i] = na
                        evs[i] = plans[i].get("evidence", evs[i])
                        afeats[i] = track.answer_features(qmap[recs[i]["qid"]],
                                                          rung, evs[i], na)
                    extra_cost[i] = call_cost_cny(model, rr["prompt_tokens"],
                                                  rr["completion_tokens"], rr.get("gpu_s"))
                    extra_lat[i] = rr["latency"]
        for i, (r, resp) in enumerate(zip(recs, resps)):
            rec = {
                "qid": r["qid"], "rung": rung, "answer": answers[i], "model": model,
                "cost_cny": call_cost_cny(model, resp["prompt_tokens"],
                                          resp["completion_tokens"],
                                          resp.get("gpu_s")) + extra_cost[i],
                "prompt_tokens": resp["prompt_tokens"],
                "completion_tokens": resp["completion_tokens"],
                "latency_retrieval": r["latency_retrieval"],
                "latency_llm": resp["latency"] + extra_lat[i],
                "features": {**r["features"], **afeats[i]},
                "evidence_meta": r.get("evidence_meta", {}),
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
    out_f.close()
    print("generation complete:", out_path)
