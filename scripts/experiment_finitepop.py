"""Finite-population validity check (reviewer fix: the repeated-draw
protocol samples calibration sets WITHOUT replacement from a fixed
execution matrix, while the HB p-value is derived for i.i.d. sampling).

Bridge, formalized in the paper's appendix:
  (a) the hypergeometric p-value P(Hyp(N, floor(N*alpha)+1, n) <= k) is
      EXACT under without-replacement sampling (MLR in the population
      violation count K puts the worst case at K0 = floor(N*alpha)+1);
  (b) the KL-Chernoff component exp(-n KL(r_hat||alpha)) remains valid
      without replacement by Hoeffding (1963, Theorem 4) convex ordering
      (the without-replacement sum's mgf is dominated by the i.i.d. one).
  finite_pop_p_value = min of the two: valid, and tighter than HB.

This script re-runs the RQ1 repeated-draw protocol at the four headline
settings, certifying each draw twice -- once with the shipped i.i.d. HB
p-value, once with the finite-population p-value -- and reports, per
setting: population-violation rate of both procedures, per-draw decision
agreement, and how often each p-value certifies a (weakly) cheaper
policy. Pure numpy/scipy over materialized matrices; no LLM calls.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import LadderData, certify_ladder  # noqa: E402
from contractrag.optimizer import (build_candidates, optimize,
                                   apply_candidate)  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")

SETTINGS = [
    ("hybridqa", "quality", 0.5, 0.35),
    ("crag", "correct", 0.5, 0.65),
    ("asqa", "citation", 50.0, 0.25),
    ("qampari", "citation", 50.0, 0.66),
]


def build_pool(track, contract, tau):
    loss_fn = make_loss_fn(track, contract, tau)
    kgc = kg_costs_for(track)
    tr = TrackData(track, "train", loss_fn, kg_costs=kgc)
    ld_tr, _, _, aux_tr = tr.build()
    scorer = RungScorer().fit(ld_tr, aux_tr["raw_feats"])
    ld_tr.scores = scorer.score(aux_tr["raw_feats"])
    parts = []
    for s in ["cal", "test"]:
        td = TrackData(track, s, loss_fn, kg_costs=kgc)
        ld, _, _, aux = td.build()
        ld.scores = scorer.score(aux["raw_feats"])
        parts.append(ld)
    ld_pool = LadderData(
        losses=np.concatenate([p.losses for p in parts], axis=1),
        scores=np.concatenate([p.scores for p in parts], axis=1),
        costs=np.concatenate([p.costs for p in parts], axis=1))
    return ld_tr, ld_pool


def dominance_check(n_pop, n, alpha):
    """Verify p_hyper(k) <= p_Bentkus(k) for every attainable count k.

    If it holds, min(p_KL, p_hyper) <= min(p_KL, p_Bentkus) pointwise, so
    the shipped i.i.d. HB p-value dominates the exact finite-population
    test and inherits its validity under without-replacement sampling.
    """
    from scipy.stats import binom, hypergeom
    k0 = int(np.floor(n_pop * alpha)) + 1
    ks = np.arange(n + 1)
    p_hyp = hypergeom.cdf(ks, n_pop, k0, n)
    p_bent = np.e * binom.cdf(ks, n, alpha)
    gap = p_hyp - np.minimum(1.0, p_bent)
    return bool(np.all(gap <= 1e-12)), float(gap.max())


def run_setting(track, contract, tau, alpha, delta, repeats, cal_frac):
    ld_tr, ld_pool = build_pool(track, contract, tau)
    n = ld_pool.n
    n_cal = int(n * cal_frac)
    cands = build_candidates(ld_tr)
    idx_pool = np.arange(n)

    dom_ok, dom_gap = dominance_check(n, n_cal, alpha)

    res = {m: {"pop_risks": [], "costs": [], "n_cert": []}
           for m in ["hb", "fp"]}
    same_policy = 0

    rng = np.random.default_rng(0)  # same seed => same draws as RQ1
    for rep in range(repeats):
        perm = rng.permutation(n)
        ci = perm[:n_cal]
        ld_c = LadderData(losses=ld_pool.losses[:, ci],
                          scores=ld_pool.scores[:, ci],
                          costs=ld_pool.costs[:, ci])
        opt_hb = optimize(cands, ld_c, alpha, delta)
        opt_fp = optimize(cands, ld_c, alpha, delta, n_pop=n)
        for tag, opt in [("hb", opt_hb), ("fp", opt_fp)]:
            loss_p, cost_p, _, _ = apply_candidate(opt.candidate, ld_pool)
            res[tag]["pop_risks"].append(float(loss_p.mean()))
            res[tag]["costs"].append(float(cost_p.mean()))
            res[tag]["n_cert"].append(opt.n_certified)
        same_policy += int(opt_hb.candidate.describe()
                           == opt_fp.candidate.describe())

    out = {"track": track, "contract": contract, "tau": tau, "alpha": alpha,
           "delta": delta, "repeats": repeats, "n_pool": n, "n_cal": n_cal,
           "hb_dominates_exact": dom_ok, "dominance_max_gap": dom_gap,
           "agreement_deployed_policy": same_policy / repeats}
    for tag in ["hb", "fp"]:
        pk = np.array(res[tag]["pop_risks"])
        ck = np.array(res[tag]["costs"])
        nc = np.array(res[tag]["n_cert"])
        out[tag] = {"violation_rate_pop": float((pk > alpha).mean()),
                    "mean_pop_risk": float(pk.mean()),
                    "mean_cost": float(ck.mean()),
                    "mean_n_certified": float(nc.mean())}
    d_cost = np.array(res["fp"]["costs"]) - np.array(res["hb"]["costs"])
    out["fp_cheaper_frac"] = float((d_cost < -1e-12).mean())
    out["fp_more_certified_frac"] = float(
        (np.array(res["fp"]["n_cert"]) > np.array(res["hb"]["n_cert"])).mean())
    print(f"{track:>9s} a={alpha}: dominance={dom_ok}"
          f" agree={out['agreement_deployed_policy']:.3f}"
          f" viol hb={out['hb']['violation_rate_pop']:.3f}"
          f" fp={out['fp']['violation_rate_pop']:.3f}"
          f" | n_cert hb={out['hb']['mean_n_certified']:.1f}"
          f" fp={out['fp']['mean_n_certified']:.1f}"
          f" | fp cheaper in {out['fp_cheaper_frac']:.3f}")
    return out


ALL_CONFIGS = [
    # every (N, n, alpha) used by a repeated-draw experiment in the paper
    ("hybridqa", 6466, 2586, [0.32, 0.35, 0.40, 0.48]),
    ("crag", 2006, 802, [0.62, 0.65, 0.70]),
    ("asqa", 798, 319, [0.15, 0.25, 0.35, 0.65]),
    ("qampari", 850, 340, [0.63, 0.66, 0.71]),
    ("families", 1100, 500, [0.35, 0.45]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=1000)
    ap.add_argument("--cal_frac", type=float, default=0.4)
    args = ap.parse_args()

    dominance = []
    for name, N, n, alphas in ALL_CONFIGS:
        for a in alphas:
            ok, gap = dominance_check(N, n, a)
            dominance.append({"config": name, "N": N, "n": n, "alpha": a,
                              "hb_dominates_exact": ok, "max_gap": gap})
            print(f"dominance {name:>9s} N={N} n={n} a={a}: {ok}")

    results = [run_setting(t, c, tau, a, args.delta, args.repeats,
                           args.cal_frac)
               for t, c, tau, a in SETTINGS]
    path = os.path.join(EXP, "finitepop_check.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dominance": dominance, "settings": results}, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
