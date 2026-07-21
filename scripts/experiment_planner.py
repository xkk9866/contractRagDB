"""Cardinality-driven join ordering & access-path selection (reviewer point 5).

Turns the model-cascade view into a genuine query optimizer decision: for each
HybridQA query the structured predicate (row match) has an estimated
selectivity, and the table<->passage join can be executed TABLE_FIRST
(predicate pushdown, small passage pool) or TEXT_FIRST (full-table pool,
post-join). We:

  1. execute both access paths at a FIXED generator (qwen-flash) so the only
     difference is the join order, not the model tier;
  2. estimate per-query selectivity / cardinalities (contractrag.planner);
  3. fit a cost model (retrieval latency vs pool size) and a recall model
     (TABLE_FIRST success vs selectivity) on the train split;
  4. compare always-table-first, always-text-first, a cost-based
     selectivity-driven chooser, and the per-query oracle path;
  5. certify the chooser's contract with fixed-sequence testing.

Stages mirror experiment_families.py.

Usage:
  python scripts/experiment_planner.py --stage run
  python scripts/experiment_planner.py --stage score
  python scripts/experiment_planner.py --stage analyze
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.engine import run_ladder_matrix  # noqa: E402
from contractrag.tracks.hybridqa import HybridQATrack  # noqa: E402
from contractrag.planner import estimate_cardinalities  # noqa: E402
from contractrag.textutil import best_f1  # noqa: E402
from contractrag.calibrate import hb_p_value, certify_candidates  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
TAU = 0.5
N = {"train": 400, "cal": 600, "test": 800}

# Access paths at a fixed model tier (generator held at qwen-flash so the ONLY
# difference is the join order / physical retrieval plan):
#   matrix 1 (planner_*):  0 = TF-hybrid (pushdown pool, dense+rerank)
#                          1 = XF-hybrid (full pool,   dense+rerank)
#   matrix 2 (planner2_*): 0 = TF-lex    (pushdown pool, BM25 only; no GPU)
#                          1 = XF-lex    (full pool,     BM25 only; no GPU)
# TF-lex is the 'index scan' analogue: nearly free, safe only when the
# structured predicate is selective. XF-hybrid is the 'seq scan + sort'.
PATHS = [
    dict(rows=15, k_ret=20, k_ctx=8, rerank=True, dense=True, full_pool=False,
         model="qwen-flash", nli_filter=False),   # TABLE_FIRST (pushdown)
    dict(rows=15, k_ret=20, k_ctx=8, rerank=True, dense=True, full_pool=True,
         model="qwen-flash", nli_filter=False),   # TEXT_FIRST (full pool)
]
PATHS_LEX = [
    dict(rows=15, k_ret=20, k_ctx=8, rerank=False, dense=False, full_pool=False,
         model="qwen-flash", nli_filter=False),   # TF-lex (pushdown, BM25)
    dict(rows=15, k_ret=20, k_ctx=8, rerank=False, dense=False, full_pool=True,
         model="qwen-flash", nli_filter=False),   # XF-lex (full pool, BM25)
]

# GPU rental rate for local retrieval compute (RTX 4090 on-demand, CNY/s)
GPU_RATE = 2.0 / 3600.0

PATH_NAMES = {0: "TF-hybrid", 1: "XF-hybrid", 2: "TF-lex", 3: "XF-lex"}


def subsets():
    sp = get_splits("hybridqa")
    return {s: sp[s][:N[s]] for s in ["train", "cal", "test"]}


def mpath(split, which=""):
    return os.path.join(EXP, "planner%s_%s_matrix.jsonl" % (which, split))


def spath(split, which=""):
    return os.path.join(EXP, "planner%s_%s_scores.jsonl" % (which, split))


def _tok(s):
    return set(w for w in "".join(c.lower() if c.isalnum() else " " for c in s).split()
               if len(w) > 1)


def stage_run(args):
    for which, paths in [("", PATHS), ("2", PATHS_LEX)]:
        track = HybridQATrack(rungs=paths)
        subs = subsets()
        for split in ["train", "cal", "test"]:
            print("=== run access paths%s: %s (%d q)" % (which, split, len(subs[split])))
            run_ladder_matrix(track, subs[split], mpath(split, which),
                              max_workers=args.workers)


def stage_score(args):
    subs = subsets()
    gold = {s: {q["qid"]: q for q in subs[s]} for s in subs}
    for which in ["", "2"]:
        for split in ["train", "cal", "test"]:
            recs = []
            with open(mpath(split, which), encoding="utf-8") as f:
                for line in f:
                    recs.append(json.loads(line))
            out = []
            for r in recs:
                g = gold[split].get(r["qid"])
                if g is None:
                    continue
                out.append({"qid": r["qid"], "rung": r["rung"],
                            "f1": best_f1(r["answer"], [g["answer"]])})
            with open(spath(split, which), "w", encoding="utf-8") as f:
                for o in out:
                    f.write(json.dumps(o) + "\n")
            print("scored%s %s: %d" % (which, split, len(out)))


def _load(split, track):
    """Merge hybrid + lexical matrices into a 4-path view.

    Path index: 0=TF-hybrid, 1=XF-hybrid, 2=TF-lex, 3=XF-lex.
    Path cost = LLM cost + GPU-priced retrieval wall-clock (the generator is
    fixed, so path differences are retrieval-side)."""
    subs = {q["qid"]: q for q in subsets()[split]}
    mat = {}
    for which, off in [("", 0), ("2", 2)]:
        with open(mpath(split, which), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                mat.setdefault(r["qid"], {})[int(r["rung"]) + off] = r
        with open(spath(split, which), encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if s["qid"] in mat and int(s["rung"]) + off in mat[s["qid"]]:
                    mat[s["qid"]][int(s["rung"]) + off]["_f1"] = s["f1"]
    rows = {}
    for qid, d in mat.items():
        if len(d) < 4 or qid not in subs:
            continue
        card = estimate_cardinalities(track.tables[subs[qid]["table_id"]],
                                      subs[qid]["question"], _tok)
        rows[qid] = {
            "card": card,
            "paths": {j: {"f1": d[j].get("_f1", 0.0),
                          "cost": d[j]["cost_cny"]
                          + d[j]["latency_retrieval"] * GPU_RATE,
                          "ret_lat": d[j]["latency_retrieval"],
                          "n_pool": d[j]["features"].get("n_pool", 0)}
                      for j in d},
        }
    return rows


def _policy_metrics(rows, choose):
    """choose(qid, row) -> path index. Returns (violation_rate, mean_cost)."""
    viol, cost = [], []
    for qid, row in rows.items():
        j = choose(qid, row)
        p = row["paths"][j]
        viol.append(float(p["f1"] < TAU))
        cost.append(p["cost"])
    return float(np.mean(viol)), float(np.mean(cost))


def _sigma_bin(sigma, edges):
    return int(np.clip(np.digitize(sigma, edges[1:-1]), 0, len(edges) - 2))


def stage_analyze(args):
    track = HybridQATrack(rungs=PATHS)
    tr = _load("train", track)
    te = _load("test", track)
    P = 4

    # --- per-path COST estimator: retrieval latency ~ a_j + b_j * pool  ----
    # (pool sizes are known BEFORE execution from the link-table cardinality
    # estimate, so this is a true optimizer-time cost model)
    cost_fit = {}
    for j in range(P):
        pools = np.array([r["paths"][j]["n_pool"] for r in tr.values()], float)
        lats = np.array([r["paths"][j]["ret_lat"] for r in tr.values()], float)
        A = np.stack([np.ones_like(pools), pools], 1)
        coef, *_ = np.linalg.lstsq(A, lats, rcond=None)
        cost_fit[j] = coef
    llm_mean = float(np.mean([r["paths"][0]["cost"] - r["paths"][0]["ret_lat"]
                              * GPU_RATE for r in tr.values()]))

    def est_cost(row, j):
        card = row["card"]
        pool = card.n_pool_pushdown if j in (0, 2) else card.n_pool_full
        a, b = cost_fit[j]
        return llm_mean + max(0.0, a + b * pool) * GPU_RATE

    # --- per-path RECALL estimator: success rate per selectivity bin -------
    sig_tr = np.array([r["card"].pool_selectivity for r in tr.values()])
    edges = np.quantile(sig_tr, [0, .2, .4, .6, .8, 1.0])
    edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    succ = {j: np.zeros(len(edges) - 1) for j in range(P)}
    for b in range(len(edges) - 1):
        mask = [(edges[b] <= r["card"].pool_selectivity < edges[b + 1])
                for r in tr.values()]
        rows_b = [r for r, m in zip(tr.values(), mask) if m]
        for j in range(P):
            ok = [float(r["paths"][j]["f1"] >= TAU) for r in rows_b]
            succ[j][b] = float(np.mean(ok)) if ok else 0.0

    best_succ_all = max(float(np.mean([float(r["paths"][j]["f1"] >= TAU)
                                       for r in tr.values()])) for j in range(P))
    rel_target = args.recall_target  # fraction of the best path's success

    def chooser(qid, row):
        b = _sigma_bin(row["card"].pool_selectivity, edges)
        bar = rel_target * max(succ[j][b] for j in range(P))
        feas = [j for j in range(P) if succ[j][b] >= bar]
        if feas:
            return min(feas, key=lambda j: est_cost(row, j))
        return int(np.argmax([succ[j][b] for j in range(P)]))

    def oracle(qid, row):
        ok = [j for j in row["paths"] if row["paths"][j]["f1"] >= TAU]
        cand = ok if ok else list(row["paths"])
        return min(cand, key=lambda j: row["paths"][j]["cost"])

    results = {"tau": TAU, "N": N, "recall_target": rel_target,
               "gpu_rate": GPU_RATE, "path_names": PATH_NAMES,
               "cost_fit": {j: cost_fit[j].tolist() for j in cost_fit},
               "succ_by_bin": {j: succ[j].tolist() for j in succ},
               "policies": {}}
    fixed = [(f"always_{PATH_NAMES[j]}", (lambda jj: lambda q, r: jj)(j))
             for j in range(P)]
    for name, fn in fixed + [("selectivity_chooser", chooser),
                             ("oracle_path", oracle)]:
        v, c = _policy_metrics(te, fn)
        results["policies"][name] = {"violation": v, "cost_mCNY": 1000 * c}
        print("%-22s viol=%.3f cost=%.4f mCNY" % (name, v, 1000 * c))

    # stratified: per selectivity tercile, violation/cost of each fixed path
    sigmas = np.array([r["card"].pool_selectivity for r in te.values()])
    tedges = np.quantile(sigmas, [0, 0.33, 0.66, 1.0])
    strata = []
    for b in range(3):
        lo, hi = tedges[b], tedges[b + 1]
        sub = {q: r for q, r in te.items()
               if lo <= r["card"].pool_selectivity <= hi}
        rec = {"sigma_range": [float(lo), float(hi)], "n": len(sub)}
        for j in range(P):
            rec[PATH_NAMES[j] + "_viol"] = float(np.mean(
                [float(r["paths"][j]["f1"] < TAU) for r in sub.values()]))
            rec[PATH_NAMES[j] + "_mCNY"] = float(1000 * np.mean(
                [r["paths"][j]["cost"] for r in sub.values()]))
        ch = {q: chooser(q, r) for q, r in sub.items()}
        rec["chooser_share"] = {PATH_NAMES[j]:
                                float(np.mean([c == j for c in ch.values()]))
                                for j in range(P)}
        strata.append(rec)
        print("  sigma[%.2f,%.2f] n=%d: " % (lo, hi, len(sub))
              + " | ".join("%s v=%.3f c=%.4f" % (PATH_NAMES[j],
                                                 rec[PATH_NAMES[j] + "_viol"],
                                                 rec[PATH_NAMES[j] + "_mCNY"])
                           for j in range(P))
              + "  chooser->" + str({k: round(v, 2)
                                     for k, v in rec["chooser_share"].items()}))
    results["selectivity_strata"] = strata

    # ---- certified access-path selection ---------------------------------
    # Candidate family: fixed paths + selectivity choosers over a target grid
    # (train-only construction), then fixed-sequence LTT on cal at alpha.
    cal = _load("cal", track)

    def mk_chooser(rt):
        def f(qid, row):
            b = _sigma_bin(row["card"].pool_selectivity, edges)
            bar = rt * max(succ[j][b] for j in range(P))
            feas = [j for j in range(P) if succ[j][b] >= bar]
            if feas:
                return min(feas, key=lambda j: est_cost(row, j))
            return int(np.argmax([succ[j][b] for j in range(P)]))
        return f

    cand_fns = [("fixed_%s" % PATH_NAMES[j], (lambda jj: lambda q, r: jj)(j))
                for j in range(P)]
    cand_fns += [("chooser_rt%.2f" % rt, mk_chooser(rt))
                 for rt in [0.80, 0.85, 0.90, 0.95, 0.99, 1.0]]

    def losses_costs(rows, fn):
        l = np.array([float(rows[q]["paths"][fn(q, rows[q])]["f1"] < TAU)
                      for q in rows])
        c = np.array([rows[q]["paths"][fn(q, rows[q])]["cost"] for q in rows])
        return l, c

    tr_stats = [losses_costs(tr, fn) for _, fn in cand_fns]
    cal_stats = [losses_costs(cal, fn) for _, fn in cand_fns]
    order = sorted(range(len(cand_fns)),
                   key=lambda i: (tr_stats[i][0].mean(), tr_stats[i][1].mean()))
    best, diag = certify_candidates([s[0] for s in cal_stats],
                                    [s[1] for s in cal_stats],
                                    order, args.alpha, args.delta)
    if best is not None:
        name, fn = cand_fns[best]
        v, c = _policy_metrics(te, fn)
        results["certified_path_policy"] = {
            "alpha": args.alpha, "selected": name,
            "cal_risk": float(cal_stats[best][0].mean()),
            "test_violation": v, "test_cost_mCNY": 1000 * c,
            "n_candidates": len(cand_fns)}
        print("certified path policy @alpha=%.2f: %s (cal %.3f) "
              "-> test viol=%.3f cost=%.4f mCNY"
              % (args.alpha, name, cal_stats[best][0].mean(), v, 1000 * c))
    else:
        results["certified_path_policy"] = {"alpha": args.alpha, "selected": None}
        print("no path policy certifies at alpha=%.2f" % args.alpha)

    out = os.path.join(EXP, "planner_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run", "score", "analyze"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--recall_target", type=float, default=0.85)
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()
    {"run": stage_run, "score": stage_score, "analyze": stage_analyze}[args.stage](args)


if __name__ == "__main__":
    main()
