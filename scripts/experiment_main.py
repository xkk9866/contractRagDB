"""Main experiment: certification vs baselines on a track.

Pipeline:
  1. Load ladder matrices + scores for train/cal/test.
  2. Define contract loss (per track / contract type).
  3. Fit RungScorer on train; score cal/test.
  4. ContractRAG: fixed-sequence LTT certification on cal -> policy -> test.
  5. Baselines: fixed rungs, complexity router, utility router, BO thresholds,
     Abacus, oracle.
  6. Sweep alpha grid; write results JSON.

Usage: python scripts/experiment_main.py hybridqa --contract quality --tau 0.5
       python scripts/experiment_main.py crag --contract hallu
       python scripts/experiment_main.py asqa --contract citation --tau 0.5
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import (TrackData, RungScorer, eval_policy, eval_fixed,
                                oracle_policy)
from contractrag.calibrate import certify_ladder, LadderData, hb_p_value
from contractrag.baselines import (ComplexityRouter, eval_routed, utility_route,
                                   tune_utility_lambda, bo_thresholds,
                                   abacus_select)
from scripts.run_ladder import get_splits

EXP = os.path.join(ROOT, "experiments")


def make_loss_fn(track, contract, tau):
    if track == "hybridqa":
        if contract == "quality":
            return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        if contract == "hallu":
            # violation = hallucination: answered and judged incorrect
            return lambda rec, sc: float(sc.get("label") == "incorrect")
        if contract == "correct":
            return lambda rec, sc: float(sc.get("label") != "correct")
    if track == "asqa":
        if contract == "citation":
            return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)
        if contract == "joint":
            return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau
                                         or sc.get("str_em", 0.0) < 30.0)
        if contract == "citation_internal":  # provisional: internal NLI coverage
            return lambda rec, sc: float(
                rec["features"].get("nli_cite_cov", 0.0) < tau)
    if track == "qampari":
        if contract == "citation":
            return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)
        if contract == "joint":
            return lambda rec, sc: float(
                sc.get("citation_rec", 0.0) < tau
                or sc.get("qampari_rec_top5", 0.0) < 20.0)
    raise SystemExit(f"unknown contract {contract} for {track}")


def kg_costs_for(track):
    if track != "crag":
        return None
    kg_dir = os.path.join(ROOT, "data", "crag", "kg_evidence")
    out = {}
    for f in os.listdir(kg_dir):
        d = json.load(open(os.path.join(kg_dir, f), encoding="utf-8"))
        out[d["qid"]] = {"cost": d.get("plan_cost_cny", 0.0),
                         "latency": d.get("total_latency", 0.0)}
    return out


def question_of(track, q):
    return q.get("question") or q.get("query")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--alphas", default=None, help="comma list")
    args = ap.parse_args()

    loss_fn = make_loss_fn(args.track, args.contract, args.tau)
    kgc = kg_costs_for(args.track)
    splits = get_splits(args.track)
    qmap = {s: {q["qid"]: q for q in qs} for s, qs in splits.items()}

    data = {}
    for s in ["train", "cal", "test"]:
        td = TrackData(args.track, s, loss_fn, kg_costs=kgc)
        ld, qids, groups, aux = td.build()
        data[s] = {"ld": ld, "qids": qids, "aux": aux}
        print(f"{s}: {ld.n} queries x {ld.L} rungs; "
              f"per-rung risk {[round(float(x),3) for x in ld.losses.mean(axis=1)]}")

    # sufficiency scorer
    scorer = RungScorer().fit(data["train"]["ld"], data["train"]["aux"]["raw_feats"])
    for s in ["train", "cal", "test"]:
        data[s]["ld"].scores = scorer.score(data[s]["aux"]["raw_feats"])

    ld_tr, lat_tr = data["train"]["ld"], data["train"]["aux"]["latency"]
    ld_cal, lat_cal = data["cal"]["ld"], data["cal"]["aux"]["latency"]
    ld_te, lat_te = data["test"]["ld"], data["test"]["aux"]["latency"]

    if args.alphas:
        alphas = [float(x) for x in args.alphas.split(",")]
    else:
        base = float(ld_te.losses[-1].mean())  # strongest-plan risk on test
        alphas = sorted({round(a, 3) for a in
                         [base + 0.02, base + 0.05, base + 0.1, base + 0.15,
                          base + 0.2, base + 0.3]})
    print("alpha grid:", alphas)

    results = {"track": args.track, "contract": args.contract, "tau": args.tau,
               "delta": args.delta, "alphas": alphas, "methods": {}}

    def add(name, alpha, metrics, extra=None):
        results["methods"].setdefault(name, {})[str(alpha)] = {
            **metrics, **(extra or {})}
        print(f"  {name:>24s} a={alpha}: risk={metrics['risk']:.3f} "
              f"cost={metrics['cost_mean']*1000:.2f}mCNY p95={metrics['lat_p95']:.1f}s")

    # fixed rungs (alpha-independent; store under 'na')
    for j in range(ld_te.L):
        m = eval_fixed(ld_te, lat_te, j)
        results["methods"].setdefault(f"fixed_rung_{j}", {})["na"] = m
        print(f"fixed rung {j}: risk={m['risk']:.3f} cost={m['cost_mean']*1000:.2f}mCNY")
    results["methods"]["oracle"] = {"na": oracle_policy(ld_te, lat_te)}

    # complexity router (alpha-independent)
    q_tr = [question_of(args.track, qmap["train"][qid]) for qid in data["train"]["qids"]]
    q_te = [question_of(args.track, qmap["test"][qid]) for qid in data["test"]["qids"]]
    router = ComplexityRouter().fit(q_tr, ld_tr.losses)
    routed = router.route(q_te)
    results["methods"]["adaptive_rag_router"] = {
        "na": eval_routed(ld_te, lat_te, routed, progressive=False)}
    print("adaptive_rag_router:", results["methods"]["adaptive_rag_router"]["na"]["risk"])

    # build optimizer candidate space once (train data only)
    from contractrag.optimizer import build_candidates, optimize, apply_candidate
    cands = build_candidates(ld_tr)

    for alpha in alphas:
        print(f"--- alpha = {alpha}")
        # ContractRAG full optimizer (ours)
        opt = optimize(cands, ld_cal, alpha, args.delta)
        loss_t, cost_t, lat_t, stop_t = apply_candidate(opt.candidate, ld_te, lat_te)
        add("contractrag_opt", alpha, {
            "risk": float(loss_t.mean()), "cost_mean": float(cost_t.mean()),
            "cost_total": float(cost_t.sum()),
            "lat_p50": float(np.percentile(lat_t, 50)),
            "lat_p95": float(np.percentile(lat_t, 95)),
            "stop_hist": np.bincount(stop_t, minlength=ld_te.L).tolist(),
        }, {"certified": opt.certified, "selected": opt.candidate.describe(),
            "n_certified": opt.n_certified, "n_tested": opt.n_tested})

        # ContractRAG ladder-only (ablation: escalation family only)
        pol = certify_ladder(ld_cal, alpha, args.delta)
        m = eval_policy(ld_te, lat_te, pol.thresholds)
        add("contractrag", alpha, m,
            {"certified": pol.certified, "q": pol.q, "cal_risk": pol.cal_risk,
             "p_value": pol.p_value})

        # utility router (tuned on train for this alpha)
        lam = tune_utility_lambda(ld_tr, lat_tr, ld_tr.scores, alpha)
        ru = utility_route(ld_te.scores, ld_te.costs.mean(axis=1), lam)
        add("utility_router", alpha, eval_routed(ld_te, lat_te, ru), {"lambda": lam})

        # BO thresholds (train-tuned empirical constraint)
        thr = bo_thresholds(ld_tr, lat_tr, alpha, n_trials=150)
        add("bo_thresholds", alpha, eval_policy(ld_te, lat_te, thr))

        # Abacus workload selection
        j_sel, est = abacus_select(ld_cal, sample_budget=150, alpha=alpha)
        add("abacus", alpha, eval_fixed(ld_te, lat_te, j_sel),
            {"selected_rung": j_sel, "est_risk": est.tolist()})

        # naive empirical-cal (pick cheapest threshold with cal risk <= alpha,
        # no p-value correction) — ablation "no certificate"
        from contractrag.calibrate import quantile_path, policy_risk_cost
        best = None
        for qv, thr2 in quantile_path(ld_cal, 201):
            r, c = policy_risk_cost(ld_cal, thr2)
            if r <= alpha:
                best = thr2
            else:
                break
        if best is not None:
            add("empirical_no_cert", alpha, eval_policy(ld_te, lat_te, best))

    out = os.path.join(EXP, f"main_{args.track}_{args.contract}_tau{args.tau}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
