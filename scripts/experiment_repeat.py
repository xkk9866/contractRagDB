"""Repeated-split validity experiment: the paper's headline safety result.

Pools cal+test queries; repeats R times: random disjoint (cal, test) split,
run each method's selection/tuning on cal (train split stays fixed for
scorers/routers), evaluate realized risk on (a) the held-out part and (b) the
ENTIRE pooled execution matrix, treated as a fixed finite population. The
population risk is exact (no test-sampling noise), so
  P(population risk > alpha)  directly verifies the certificate;
  P(held-out risk > alpha)    is the noisy proxy reported for reference.
Also reports mean excess risk when violating and cost ratio vs strongest plan.

Usage: python scripts/experiment_repeat.py asqa --contract citation_internal \
          --tau 0.5 --alpha 0.65 --repeats 200
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import (LadderData, certify_ladder, stop_rung,
                                   quantile_path, policy_risk_cost)  # noqa: E402
from contractrag.baselines import (bo_thresholds, utility_route,
                                   tune_utility_lambda,
                                   abacus_select)  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def subset(ld, lats, idx):
    sub = LadderData(losses=ld.losses[:, idx], scores=ld.scores[:, idx],
                     costs=ld.costs[:, idx])
    return sub, lats[:, idx]


def risk_cost_of(ld, lats, thresholds):
    idx = np.arange(ld.n)
    stop = stop_rung(ld.scores, np.asarray(thresholds))
    risk = float(ld.losses[stop, idx].mean())
    cost = float(np.cumsum(ld.costs, axis=0)[stop, idx].mean())
    return risk, cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--cal_frac", type=float, default=0.4)
    args = ap.parse_args()

    loss_fn = make_loss_fn(args.track, args.contract, args.tau)
    kgc = kg_costs_for(args.track)

    # train split: fit scorer once (fixed across repeats; no leakage)
    tr = TrackData(args.track, "train", loss_fn, kg_costs=kgc)
    ld_tr, _, _, aux_tr = tr.build()
    scorer = RungScorer().fit(ld_tr, aux_tr["raw_feats"])
    ld_tr.scores = scorer.score(aux_tr["raw_feats"])
    lat_tr = aux_tr["latency"]

    # pool cal+test
    pool_parts = []
    for s in ["cal", "test"]:
        td = TrackData(args.track, s, loss_fn, kg_costs=kgc)
        ld, qids, _, aux = td.build()
        ld.scores = scorer.score(aux["raw_feats"])
        pool_parts.append((ld, aux["latency"]))
    ld_pool = LadderData(
        losses=np.concatenate([p[0].losses for p in pool_parts], axis=1),
        scores=np.concatenate([p[0].scores for p in pool_parts], axis=1),
        costs=np.concatenate([p[0].costs for p in pool_parts], axis=1))
    lat_pool = np.concatenate([p[1] for p in pool_parts], axis=1)
    n = ld_pool.n
    n_cal = int(n * args.cal_frac)
    print(f"pool {n} queries; cal {n_cal} / test {n - n_cal}; "
          f"alpha={args.alpha} delta={args.delta}")

    strongest_cost = float(np.cumsum(ld_pool.costs, axis=0)[-1].mean())

    methods = ["contractrag_opt", "contractrag", "empirical_no_cert",
               "bo_thresholds", "utility_router", "abacus"]
    stats = {m: {"risks": [], "costs": [], "pop_risks": []} for m in methods}

    idx_pool = np.arange(n)

    def pop_risk_thr(thresholds):
        """Exact risk of a threshold policy on the full finite population."""
        stop = stop_rung(ld_pool.scores, np.asarray(thresholds))
        return float(ld_pool.losses[stop, idx_pool].mean())

    lam_cache = tune_utility_lambda(ld_tr, lat_tr, ld_tr.scores, args.alpha)
    bo_thr = bo_thresholds(ld_tr, lat_tr, args.alpha, n_trials=150)
    from contractrag.optimizer import build_candidates, optimize, apply_candidate
    cands = build_candidates(ld_tr)

    rng = np.random.default_rng(0)
    for rep in range(args.repeats):
        perm = rng.permutation(n)
        ci, ti = perm[:n_cal], perm[n_cal:]
        ld_c, lat_c = subset(ld_pool, lat_pool, ci)
        ld_t, lat_t = subset(ld_pool, lat_pool, ti)

        # ContractRAG full optimizer
        opt = optimize(cands, ld_c, args.alpha, args.delta)
        loss_t, cost_t, _, _ = apply_candidate(opt.candidate, ld_t)
        loss_p, _, _, _ = apply_candidate(opt.candidate, ld_pool)
        stats["contractrag_opt"]["risks"].append(float(loss_t.mean()))
        stats["contractrag_opt"]["costs"].append(float(cost_t.mean()))
        stats["contractrag_opt"]["pop_risks"].append(float(loss_p.mean()))

        # ContractRAG ladder-only
        pol = certify_ladder(ld_c, args.alpha, args.delta)
        r, c = risk_cost_of(ld_t, lat_t, pol.thresholds)
        stats["contractrag"]["risks"].append(r)
        stats["contractrag"]["costs"].append(c)
        stats["contractrag"]["pop_risks"].append(pop_risk_thr(pol.thresholds))

        # empirical (no certificate): cheapest cal-feasible threshold
        best = np.full(ld_c.L - 1, np.inf)
        for qv, thr in quantile_path(ld_c, 201):
            rr, _ = policy_risk_cost(ld_c, thr)
            if rr <= args.alpha:
                best = thr
            else:
                break
        r, c = risk_cost_of(ld_t, lat_t, best)
        stats["empirical_no_cert"]["risks"].append(r)
        stats["empirical_no_cert"]["costs"].append(c)
        stats["empirical_no_cert"]["pop_risks"].append(pop_risk_thr(best))

        # BO thresholds (tuned on the same cal draw, empirical constraint)
        bo_thr_rep = bo_thresholds(ld_c, lat_c, args.alpha, n_trials=100,
                                   seed=rep)
        r, c = risk_cost_of(ld_t, lat_t, bo_thr_rep)
        stats["bo_thresholds"]["risks"].append(r)
        stats["bo_thresholds"]["costs"].append(c)
        stats["bo_thresholds"]["pop_risks"].append(pop_risk_thr(bo_thr_rep))

        # utility router (lambda tuned on the same cal draw)
        lam_rep = tune_utility_lambda(ld_c, lat_c, ld_c.scores, args.alpha)
        ru = utility_route(ld_t.scores, ld_t.costs.mean(axis=1), lam_rep)
        idx = np.arange(ld_t.n)
        rr = np.clip(ru, 0, ld_t.L - 1)
        r = float(ld_t.losses[rr, idx].mean())
        c = float(ld_t.costs[rr, idx].mean())
        stats["utility_router"]["risks"].append(r)
        stats["utility_router"]["costs"].append(c)
        ru_p = np.clip(utility_route(ld_pool.scores,
                                     ld_pool.costs.mean(axis=1), lam_rep),
                       0, ld_pool.L - 1)
        stats["utility_router"]["pop_risks"].append(
            float(ld_pool.losses[ru_p, idx_pool].mean()))

        # Abacus workload selection on cal (jump semantics: pays only
        # the selected plan's cost)
        j_sel, _ = abacus_select(ld_c, sample_budget=150,
                                 alpha=args.alpha, seed=rep)
        r = float(ld_t.losses[j_sel].mean())
        c = float(ld_t.costs[j_sel].mean())
        stats["abacus"]["risks"].append(r)
        stats["abacus"]["costs"].append(c)
        stats["abacus"]["pop_risks"].append(float(ld_pool.losses[j_sel].mean()))

    out = {"track": args.track, "contract": args.contract, "tau": args.tau,
           "alpha": args.alpha, "delta": args.delta, "repeats": args.repeats,
           "strongest_cost": strongest_cost, "methods": {}}
    for m in methods:
        rk = np.array(stats[m]["risks"])
        ck = np.array(stats[m]["costs"])
        pk = np.array(stats[m]["pop_risks"])
        out["methods"][m] = {
            "violation_rate": float((rk > args.alpha).mean()),
            "violation_rate_pop": float((pk > args.alpha).mean()),
            "mean_risk": float(rk.mean()),
            "mean_pop_risk": float(pk.mean()),
            "mean_excess_when_violate": float((rk[rk > args.alpha] - args.alpha).mean())
            if (rk > args.alpha).any() else 0.0,
            "mean_pop_excess_when_violate": float(
                (pk[pk > args.alpha] - args.alpha).mean())
            if (pk > args.alpha).any() else 0.0,
            "mean_cost": float(ck.mean()),
            "cost_ratio_vs_strongest": float(ck.mean() / strongest_cost),
        }
        o = out["methods"][m]
        print(f"{m:>20s}: viol={o['violation_rate']:.3f} "
              f"viol_pop={o['violation_rate_pop']:.3f} "
              f"risk={o['mean_risk']:.3f} cost_ratio={o['cost_ratio_vs_strongest']:.2f}")

    path = os.path.join(EXP, f"repeat_{args.track}_{args.contract}_a{args.alpha}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
