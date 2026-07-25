#!/usr/bin/env python3
"""Physical plan grid: unbundle access path from generator implementation.

The incumbent ladder is a hand-drawn diagonal through the plan space: retrieval
strength and generator tier increase together, so the optimizer never sees a
plan that retrieves deeply but generates cheaply. Measured on the cross-family
runs, that diagonal leaves the (risk, cost) region a strict contract needs
completely unpopulated.

This script executes the full Cartesian product

    access path A (4 operator configurations)  x  generator G (model tiers,
    multiple vendors, hosted and local)

reusing one materialized retrieval view per access path, so the retrieval
operators are paid once and shared by every generator -- the RAG analogue of a
materialized common subexpression. Only generation is metered per plan.

Access paths (HybridQA), each adding a physical operator:
  A0  BM25 rows=6,  k_ret=8,  ctx=4                  (shallow scan, no rerank)
  A1  + dense fusion + cross-encoder rerank, ctx=5
  A2  + full passage pool, rows=15, k_ret=16, ctx=8
  A3  + NLI evidence filter, rows=20, k_ret=24, ctx=12

Stages:
  evidence  materialize the four retrieval views (GPU, shared across models)
  generate  one batched pass per generator over all four views
  score     token-F1 / EM against gold (CPU, no LLM cost)
  analyze   build the plan grid and compare certifiers on it
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.engine import materialize_evidence, generate_from_evidence  # noqa
from contractrag.textutil import best_f1, em_score, is_abstain  # noqa
from contractrag.tracks.hybridqa import HybridQATrack  # noqa
from scripts.run_ladder import get_splits  # noqa

EXP = os.path.join(ROOT, "experiments")
TAU = 0.5
SPLITS = ["train", "cal", "test"]

# hosted generators spanning three vendors and three price tiers, plus local
# open-weight models metered by GPU seconds. Chosen to span the cost axis; the
# analyze stage reports which ones survive Pareto pruning.
HOSTED = ["qwen-flash", "qwen-plus", "qwen-max",
          "deepseek-v4-flash", "deepseek-v3.2", "glm-4.7"]
LOCAL = ["ollama/gemma3:4b", "ollama/gemma3:12b",
         "ollama/llama3.1:8b-instruct-q8_0"]


def safe(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.lower().replace("ollama/", "loc"))


def ev_path(split):
    return os.path.join(EXP, f"hqfam_{split}_evidence.jsonl")


def matrix_path(model, split):
    return os.path.join(EXP, f"hqgrid_{safe(model)}_{split}_matrix.jsonl")


def scores_path(model, split):
    return os.path.join(EXP, f"hqgrid_{safe(model)}_{split}_scores.jsonl")


def subsets(n):
    sp = get_splits("hybridqa")
    return {s: sp[s][:n[s]] for s in SPLITS}


def stage_evidence(args, n):
    track = HybridQATrack()
    subs = subsets(n)
    for split in SPLITS:
        print(f"=== materialize evidence {split}: {len(subs[split])} queries",
              flush=True)
        materialize_evidence(track, subs[split], ev_path(split))


def stage_generate(args, n):
    track = HybridQATrack()
    subs = subsets(n)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        local = m.startswith("ollama/")
        workers = args.local_workers if local else args.workers
        mm = {j: m for j in range(track.n_rungs)}
        for split in SPLITS:
            t0 = time.time()
            print(f"=== generate {m} / {split} (workers={workers})", flush=True)
            generate_from_evidence(track, subs[split], ev_path(split), mm,
                                   matrix_path(m, split), max_workers=workers)
            print(f"    {m}/{split} done in {time.time()-t0:.0f}s", flush=True)


def stage_score(args, n):
    subs = subsets(n)
    gold = {s: {q["qid"]: q for q in subs[s]} for s in SPLITS}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        for split in SPLITS:
            mp = matrix_path(m, split)
            if not os.path.exists(mp):
                print(f"skip {m}/{split}: no matrix")
                continue
            out = []
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    g = gold[split].get(r["qid"])
                    if g is None:
                        continue
                    out.append({"qid": r["qid"], "rung": r["rung"],
                                "f1": best_f1(r["answer"], [g["answer"]]),
                                "em": em_score(r["answer"], g["answer"]),
                                "abstain": float(is_abstain(r["answer"]))})
            with open(scores_path(m, split), "w", encoding="utf-8") as f:
                for o in out:
                    f.write(json.dumps(o) + "\n")
            print(f"scored {m}/{split}: {len(out)}")


# ---------------------------------------------------------------------------

# Routing features must be observable BEFORE the selected plan runs, so only
# retrieval-side signals of the cheapest access path are used. They are
# produced by local BM25/dense/rerank operators whose cost is negligible next
# to generation, which is what makes pre-execution routing honest here.
ROUTE_FEATS = ["bm25_top", "bm25_margin", "dense_top", "dense_margin",
               "rerank_top", "rerank_mean3", "row_top_overlap", "n_pool",
               "n_evidence"]


def load_plan_grid(models, split):
    """(loss, cost) matrices for every (access path, generator) plan."""
    per = {}
    for m in models:
        mp, sp = matrix_path(m, split), scores_path(m, split)
        if not (os.path.exists(mp) and os.path.exists(sp)):
            continue
        mat, sc = {}, {}
        with open(mp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                mat[(r["qid"], r["rung"])] = r
        with open(sp, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                sc[(r["qid"], r["rung"])] = r
        per[m] = (mat, sc)
    if not per:
        raise SystemExit("no plan-grid matrices found; run --stage generate")
    qids = None
    for m, (mat, sc) in per.items():
        qs = {k[0] for k in mat if k in sc}
        qids = qs if qids is None else (qids & qs)
    qids = sorted(qids)
    L, C, names = [], [], []
    for m, (mat, sc) in per.items():
        rungs = sorted({k[1] for k in mat})
        for rung in rungs:
            if not all((q, rung) in mat and (q, rung) in sc for q in qids):
                continue
            L.append([float(sc[(q, rung)]["f1"] < TAU) for q in qids])
            C.append([float(mat[(q, rung)]["cost_cny"]) for q in qids])
            names.append(f"A{rung}/{m}")
    # routing features: cheapest access path, retrieval signals only
    any_mat = next(iter(per.values()))[0]
    F = np.array([[float(any_mat[(q, 0)]["features"].get(k, 0.0))
                   for k in ROUTE_FEATS] for q in qids])
    return np.array(L), np.array(C), names, qids, F


DIAGONAL = ["A0/qwen-flash", "A1/qwen-flash", "A2/qwen-plus", "A3/qwen-max"]


def stage_analyze(args, n):
    from contractrag.certlp import (build_randomized_chain,
                                    certify_randomized_chain,
                                    pareto_frontier_idx, solve_risk_lp)
    from contractrag.plangrid import (PlanGrid, build_grid_candidates,
                                      candidate_matrices)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    Ltr, Ctr, names, _, Ftr = load_plan_grid(models, "train")
    Lca, Cca, n2, _, Fca = load_plan_grid(models, "cal")
    Lte, Cte, n3, _, Fte = load_plan_grid(models, "test")
    assert names == n2 == n3, "plan sets differ across splits"
    print(f"plans={len(names)} n_train={Ltr.shape[1]} n_cal={Lca.shape[1]} "
          f"n_test={Lte.shape[1]}")
    print(f"\n  {'plan':<40} {'cal risk':>9} {'cost mCNY':>10}")
    for i in np.argsort(Cca.mean(axis=1)):
        print(f"  {names[i]:<40} {Lca[i].mean():9.3f} "
              f"{Cca[i].mean()*1000:10.4f}")

    Lpop = np.concatenate([Lca, Lte], axis=1)
    Cpop = np.concatenate([Cca, Cte], axis=1)
    R_pop, C_pop = Lpop.mean(axis=1), Cpop.mean(axis=1)
    front = pareto_frontier_idx(Ltr.mean(axis=1), Ctr.mean(axis=1), 64)
    print(f"\ntrain Pareto frontier: {len(front)}/{len(names)}")
    for i in sorted(front, key=lambda j: Ctr[j].mean()):
        print(f"  {names[i]:<40} risk={Ltr[i].mean():.3f} "
              f"cost={Ctr[i].mean()*1000:8.4f}")

    diag = [i for i, nm in enumerate(names) if nm in DIAGONAL]
    alphas = [float(a) for a in args.alphas.split(",")]
    delta = args.delta

    def grids(sel):
        return (PlanGrid(Ltr[sel], Ctr[sel], Ftr, [names[i] for i in sel]),
                PlanGrid(Lca[sel], Cca[sel], Fca, [names[i] for i in sel]),
                PlanGrid(Lte[sel], Cte[sel], Fte, [names[i] for i in sel]))

    # ablation arms, each an (index subset, policy class, mixtures?) triple.
    # Reading down the list isolates one factor at a time: the plan space, then
    # per-query routing, then randomization.
    all_idx = list(range(len(names)))
    arms = [
        ("diag/fixed",      diag,    False, 0),
        ("diag/fixed+mix",  diag,    False, 7),
        ("diag/router",     diag,    True,  0),
        ("diag/router+mix", diag,    True,  7),
        ("grid/fixed",      all_idx, False, 0),
        ("grid/fixed+mix",  all_idx, False, 7),
        ("grid/router",     all_idx, True,  0),
        ("grid/router+mix", all_idx, True,  7),
    ]
    prepared = {}
    for tag, sel, use_router, mixg in arms:
        g_tr, g_ca, g_te = grids(sel)
        if use_router:
            cands = build_grid_candidates(g_tr, num_lam=args.num_lam)
        else:
            from contractrag.plangrid import GridCandidate
            cands = []
            for p in range(g_tr.P):
                c = GridCandidate("fixed", plan=p)
                lo, co = c.realize(g_tr)
                c.train_risk, c.train_cost = float(lo.mean()), float(co.mean())
                cands.append(c)
        Ktr = candidate_matrices(cands, g_tr)
        Kca = candidate_matrices(cands, g_ca)
        Kte = candidate_matrices(cands, g_te)
        prepared[tag] = (cands, Ktr, Kca, Kte, sel, mixg)

    print(f"\ncertified cost per query, delta={delta} (mCNY); "
          f"pop risk of the deployed policy in brackets")
    hdr = f"  {'arm':<16}" + "".join(f"{('a='+format(a,'.2f')):>18}"
                                     for a in alphas)
    print(hdr)
    rows = {}
    for tag, _, _, _ in arms:
        cands, Ktr, Kca, Kte, sel, mixg = prepared[tag]
        cells = []
        rows[tag] = {}
        for a in alphas:
            ch = build_randomized_chain(Ktr[0], Ktr[1], mix_grid=mixg)
            cp = certify_randomized_chain(Kca[0], Kca[1], ch, a, delta)
            if cp.element is None:
                cells.append(f"{'---':>18}")
                rows[tag][str(a)] = {"cost": None}
                continue
            e = cp.element
            # exact population risk/cost of the deployed (possibly mixed) policy
            pr = float(sum(w * np.concatenate([Kca[0][k], Kte[0][k]]).mean()
                           for k, w in zip(e.support, e.weights)))
            pc = float(sum(w * np.concatenate([Kca[1][k], Kte[1][k]]).mean()
                           for k, w in zip(e.support, e.weights)))
            cells.append(f"{pc*1000:12.4f} [{pr:.2f}]")
            rows[tag][str(a)] = {
                "cost": pc, "pop_risk": pr, "cal_cost": cp.cal_cost,
                "cal_risk": cp.cal_risk_hat, "violates": bool(pr > a + 1e-12),
                "plan": " + ".join(f"{w:.2f}*{cands[k].describe()}"
                                   for k, w in zip(e.support, e.weights)),
                "n_certified": cp.n_certified}
        print(f"  {tag:<16}" + "".join(cells))

    print("\nreference (evaluation only, not certifiable):")
    for a in alphas:
        lp, _ = solve_risk_lp(R_pop[front], C_pop[front], a)
        feas = np.where(R_pop <= a)[0]
        bd = float(C_pop[feas].min()) if len(feas) else float("nan")
        print(f"  alpha={a:.2f}  cheapest feasible fixed plan="
              f"{bd*1000:8.4f}  LP optimum={(lp or float('nan'))*1000:8.4f}")

    print("\ngain of the full method over the incumbent diagonal ladder:")
    for a in alphas:
        b = rows["diag/fixed"][str(a)]["cost"]
        g = rows["grid/router+mix"][str(a)]["cost"]
        if b and g:
            print(f"  alpha={a:.2f}  {b*1000:9.4f} -> {g*1000:9.4f} mCNY "
                  f"= {b/g:5.2f}x cheaper")
        else:
            print(f"  alpha={a:.2f}  {'n/a' if not b else format(b*1000,'.4f')}"
                  f" -> {'n/a' if not g else format(g*1000,'.4f')}")

    out = {"models": models, "names": names, "n_cal": int(Lca.shape[1]),
           "n_train": int(Ltr.shape[1]), "n_test": int(Lte.shape[1]),
           "delta": delta, "alphas": alphas,
           "pop_risk": R_pop.tolist(), "pop_cost": C_pop.tolist(),
           "cal_risk": Lca.mean(axis=1).tolist(),
           "cal_cost": Cca.mean(axis=1).tolist(),
           "frontier": [names[i] for i in front], "arms": rows}
    path = os.path.join(EXP, "plangrid_hybridqa.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["evidence", "generate", "score", "analyze"])
    ap.add_argument("--models", default=",".join(HOSTED))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--local_workers", type=int, default=4)
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_cal", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--alphas", default="0.20,0.25,0.30,0.34,0.40")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--num_lam", type=int, default=40)
    args = ap.parse_args()
    n = {"train": args.n_train, "cal": args.n_cal, "test": args.n_test}
    {"evidence": stage_evidence, "generate": stage_generate,
     "score": stage_score, "analyze": stage_analyze}[args.stage](args, n)


if __name__ == "__main__":
    main()
