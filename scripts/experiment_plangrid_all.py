#!/usr/bin/env python3
"""Physical plan grid on any benchmark: unbundle access path from generator.

Same construction as ``experiment_plangrid.py`` but parameterised by track, so
the claim that enlarging the search space penalises a certifier can be tested
on more than one benchmark.

The grid is the Cartesian product

    access path A (the retrieval configuration of each rung)
      x  generator G (model tier, vendor, hosted or local)

Retrieval is materialised once per access path and shared by every generator,
so retrieval operators are paid once -- the RAG analogue of a materialised
common subexpression -- and only generation is metered per plan.

Stages:
  evidence  materialise the retrieval views (GPU, shared across generators)
  generate  one batched pass per generator over every view
  score     attach the track's contract loss (CRAG: DashScope judge)
  analyze   build the grid and compare certifiers on it
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
from scripts.run_ladder import get_splits, make_track  # noqa

EXP = os.path.join(ROOT, "experiments")
SPLITS = ["train", "cal", "test"]

# short file prefix per track; "hq" is the existing HybridQA grid
PREFIX = {"hybridqa": "hq", "crag": "cg", "asqa": "aq", "qampari": "qp"}

HOSTED = ["qwen-flash", "qwen-plus", "qwen-max",
          "deepseek-v4-flash", "deepseek-v3.2", "glm-4.7"]
LOCAL = ["ollama/gemma3:4b", "ollama/gemma3:12b",
         "ollama/llama3.1:8b-instruct-q8_0"]

# the loss each track's contract is written against, and the incumbent ladder
# (the hand-drawn diagonal through the grid) it is compared to
TRACK_LOSS = {"hybridqa": ("f1", 0.5), "crag": ("label", None),
              "asqa": ("citation_rec", 50.0), "qampari": ("citation_rec", 50.0)}
DIAGONAL = ["A0/qwen-flash", "A1/qwen-flash", "A2/qwen-plus", "A3/qwen-max"]


def safe(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", model.lower().replace("ollama/", "loc"))


def ev_path(track, split):
    # reuse the HybridQA views already materialised under their old name
    stem = "hqfam" if track == "hybridqa" else f"{PREFIX[track]}fam"
    return os.path.join(EXP, f"{stem}_{split}_evidence.jsonl")


def matrix_path(track, model, split):
    return os.path.join(
        EXP, f"{PREFIX[track]}grid_{safe(model)}_{split}_matrix.jsonl")


def scores_path(track, model, split):
    return os.path.join(
        EXP, f"{PREFIX[track]}grid_{safe(model)}_{split}_scores.jsonl")


def subsets(track, n):
    sp = get_splits(track)
    return {s: sp[s][:n[s]] for s in SPLITS}


def loss_of(track, sc):
    field, tau = TRACK_LOSS[track]
    if track == "crag":
        return float(sc.get("label") != "correct")
    return float(sc.get(field, 0.0) < tau)


# ---------------------------------------------------------------------------

def stage_evidence(args, n):
    track = make_track(args.track)
    subs = subsets(args.track, n)
    for split in SPLITS:
        print(f"=== materialize evidence {args.track}/{split}: "
              f"{len(subs[split])} queries", flush=True)
        materialize_evidence(track, subs[split], ev_path(args.track, split))


def stage_generate(args, n):
    track = make_track(args.track)
    subs = subsets(args.track, n)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        local = m.startswith("ollama/")
        workers = args.local_workers if local else args.workers
        mm = {j: m for j in range(track.n_rungs)}
        for split in SPLITS:
            t0 = time.time()
            print(f"=== generate {args.track}/{m}/{split} "
                  f"(workers={workers})", flush=True)
            generate_from_evidence(track, subs[split],
                                   ev_path(args.track, split), mm,
                                   matrix_path(args.track, m, split),
                                   max_workers=workers)
            print(f"    {m}/{split} done in {time.time()-t0:.0f}s", flush=True)


def stage_score(args, n):
    """Attach the contract loss. CRAG spends a DashScope judge; the others
    are scored locally or by the official ALCE evaluator."""
    subs = subsets(args.track, n)
    gold = {s: {q["qid"]: q for q in subs[s]} for s in SPLITS}
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        for split in SPLITS:
            mp, sp = matrix_path(args.track, m, split), scores_path(
                args.track, m, split)
            if not os.path.exists(mp):
                print(f"skip {m}/{split}: no matrix")
                continue
            done = set()
            if os.path.exists(sp):
                with open(sp, encoding="utf-8") as f:
                    done = {(json.loads(l)["qid"], json.loads(l)["rung"])
                            for l in f}
            recs = []
            with open(mp, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if (r["qid"] in gold[split]
                            and (r["qid"], r["rung"]) not in done):
                        recs.append(r)
            if not recs:
                print(f"{m}/{split}: all scored")
                continue
            rows = SCORERS[args.track](args, recs, gold[split], split)
            with open(sp, "a", encoding="utf-8") as f:
                for o in rows:
                    f.write(json.dumps(o) + "\n")
            print(f"scored {m}/{split}: {len(rows)}", flush=True)


def score_local_f1(args, recs, gold, split):
    from contractrag.textutil import best_f1, em_score, is_abstain
    out = []
    for r in recs:
        g = gold[r["qid"]]["answer"]
        out.append({"qid": r["qid"], "rung": r["rung"],
                    "f1": best_f1(r["answer"], [g]),
                    "em": em_score(r["answer"], g),
                    "abstain": float(is_abstain(r["answer"]))})
    return out


def score_crag_judge(args, recs, gold, split):
    """A judge model stands in for the human annotator, at the same prices
    the deployment pays, and its bill is reported."""
    from contractrag.judge import judge_crag_batch
    items = []
    for r in recs:
        g = gold[r["qid"]]
        items.append({"query": g["query"], "qtime": g.get("query_time"),
                      "gold": g["answer"], "alt": g.get("alt_ans", []),
                      "pred": r["answer"]})
    print(f"  judging {len(items)} answers with {args.judge} ...", flush=True)
    t0 = time.time()
    labels, usage = judge_crag_batch(items, model=args.judge)
    print(f"  judge: {usage.calls} calls, {usage.cost_cny:.2f} CNY, "
          f"{time.time()-t0:.0f}s", flush=True)
    return [{"qid": r["qid"], "rung": r["rung"], "label": lab}
            for r, lab in zip(recs, labels)]


def score_alce(args, recs, gold, split):
    raise SystemExit("score ASQA/QAMPARI grids with eval_alce_grid.py "
                     "(needs the TRUE-NLI model on a free GPU)")


SCORERS = {"hybridqa": score_local_f1, "crag": score_crag_judge,
           "asqa": score_alce, "qampari": score_alce}


# ---------------------------------------------------------------------------

ROUTE_FEATS = ["bm25_top", "bm25_margin", "dense_top", "dense_margin",
               "rerank_top", "rerank_mean3", "row_top_overlap", "n_pool",
               "n_evidence"]


def load_plan_grid(track, models, split):
    """(loss, cost) matrices for every (access path, generator) plan."""
    per = {}
    for m in models:
        mp, sp = matrix_path(track, m, split), scores_path(track, m, split)
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
        raise SystemExit(f"no {track} plan-grid matrices; run --stage generate")
    qids = None
    for m, (mat, sc) in per.items():
        qs = {k[0] for k in mat if k in sc}
        qids = qs if qids is None else (qids & qs)
    qids = sorted(qids)
    L, C, names = [], [], []
    for m, (mat, sc) in per.items():
        for rung in sorted({k[1] for k in mat}):
            if not all((q, rung) in mat and (q, rung) in sc for q in qids):
                continue
            L.append([loss_of(track, sc[(q, rung)]) for q in qids])
            C.append([float(mat[(q, rung)]["cost_cny"]) for q in qids])
            names.append(f"A{rung}/{m}")
    any_mat = next(iter(per.values()))[0]
    F = np.array([[float(any_mat[(q, 0)]["features"].get(k, 0.0))
                   for k in ROUTE_FEATS] for q in qids])
    return np.array(L), np.array(C), names, qids, F


def stage_analyze(args, n):
    from contractrag.certlp import (build_randomized_chain,
                                    certify_randomized_chain,
                                    pareto_frontier_idx, solve_risk_lp)
    from contractrag.plangrid import (PlanGrid, build_grid_candidates,
                                      candidate_matrices)
    tr = args.track
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    Ltr, Ctr, names, _, Ftr = load_plan_grid(tr, models, "train")
    Lca, Cca, n2, _, Fca = load_plan_grid(tr, models, "cal")
    Lte, Cte, n3, _, Fte = load_plan_grid(tr, models, "test")
    assert names == n2 == n3, "plan sets differ across splits"
    print(f"{tr}: plans={len(names)} n_train={Ltr.shape[1]} "
          f"n_cal={Lca.shape[1]} n_test={Lte.shape[1]}")
    print(f"\n  {'plan':<40} {'cal risk':>9} {'cost mCNY':>10}")
    for i in np.argsort(Cca.mean(axis=1)):
        print(f"  {names[i]:<40} {Lca[i].mean():9.3f} "
              f"{Cca[i].mean()*1000:10.4f}")

    Lpop = np.concatenate([Lca, Lte], axis=1)
    Cpop = np.concatenate([Cca, Cte], axis=1)
    R_pop, C_pop = Lpop.mean(axis=1), Cpop.mean(axis=1)
    front = pareto_frontier_idx(Ltr.mean(axis=1), Ctr.mean(axis=1), 64)
    print(f"\ntrain Pareto frontier: {len(front)}/{len(names)}")

    diag = [i for i, nm in enumerate(names) if nm in DIAGONAL]
    alphas = [float(a) for a in args.alphas.split(",")]
    delta = args.delta

    def grids(sel):
        return (PlanGrid(Ltr[sel], Ctr[sel], Ftr, [names[i] for i in sel]),
                PlanGrid(Lca[sel], Cca[sel], Fca, [names[i] for i in sel]),
                PlanGrid(Lte[sel], Cte[sel], Fte, [names[i] for i in sel]))

    all_idx = list(range(len(names)))
    arms = [("diag/fixed", diag, False, 0),
            ("diag/fixed+mix", diag, False, 7),
            ("diag/router", diag, True, 0),
            ("diag/router+mix", diag, True, 7),
            ("grid/fixed", all_idx, False, 0),
            ("grid/fixed+mix", all_idx, False, 7),
            ("grid/router", all_idx, True, 0),
            ("grid/router+mix", all_idx, True, 7)]
    prepared = {}
    for tag, sel, use_router, mixg in arms:
        if not sel:
            continue
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
        prepared[tag] = (cands, candidate_matrices(cands, g_tr),
                         candidate_matrices(cands, g_ca),
                         candidate_matrices(cands, g_te), sel, mixg)

    print(f"\ncertified cost per query, delta={delta} (mCNY); "
          f"pop risk of the deployed policy in brackets")
    print(f"  {'arm':<16}" + "".join(f"{('a='+format(a,'.2f')):>18}"
                                     for a in alphas))
    rows = {}
    for tag in prepared:
        cands, Ktr, Kca, Kte, sel, mixg = prepared[tag]
        cells, rows[tag] = [], {}
        for a in alphas:
            ch = build_randomized_chain(Ktr[0], Ktr[1], mix_grid=mixg)
            cp = certify_randomized_chain(Kca[0], Kca[1], ch, a, delta)
            if cp.element is None:
                cells.append(f"{'---':>18}")
                rows[tag][str(a)] = {"cost": None}
                continue
            e = cp.element
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

    out = {"track": tr, "models": models, "names": names,
           "n_cal": int(Lca.shape[1]), "n_train": int(Ltr.shape[1]),
           "n_test": int(Lte.shape[1]), "delta": delta, "alphas": alphas,
           "pop_risk": R_pop.tolist(), "pop_cost": C_pop.tolist(),
           "cal_risk": Lca.mean(axis=1).tolist(),
           "cal_cost": Cca.mean(axis=1).tolist(),
           "frontier": [names[i] for i in front], "arms": rows}
    path = os.path.join(EXP, f"plangrid_{tr}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(PREFIX))
    ap.add_argument("--stage", required=True,
                    choices=["evidence", "generate", "score", "analyze"])
    ap.add_argument("--models", default=",".join(HOSTED))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--local_workers", type=int, default=8)
    ap.add_argument("--judge", default="qwen-max")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--n_cal", type=int, default=500)
    ap.add_argument("--n_test", type=int, default=600)
    ap.add_argument("--alphas", default="0.35,0.45,0.55,0.62,0.70")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--num_lam", type=int, default=40)
    args = ap.parse_args()
    n = {"train": args.n_train, "cal": args.n_cal, "test": args.n_test}
    {"evidence": stage_evidence, "generate": stage_generate,
     "score": stage_score, "analyze": stage_analyze}[args.stage](args, n)


if __name__ == "__main__":
    main()
