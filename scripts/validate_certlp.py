#!/usr/bin/env python3
"""Decisive validation of certified randomized plans against the incumbent.

Protocol (identical candidate space for every method, so the comparison
isolates the certifier):
  * candidates are built on the TRAIN split only;
  * the pooled cal+test executions form a fixed finite population;
  * each of `--draws` repetitions samples a calibration subset without
    replacement, runs every certifier on it, and records the EXACT risk and
    cost of the deployed policy on the whole population;
  * violation probability = fraction of draws whose deployed population risk
    exceeds alpha. A valid certifier must keep this <= delta.

Reporting both the violation probability and the mean cost is what makes the
comparison meaningful: a cheaper policy that violates more often is not an
improvement, and the incumbent's headroom is only real if validity holds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.calibrate import hb_p_value
from contractrag.certgraph import build_element_pool, graph_certify
from contractrag.certlp import (build_randomized_chain, certified_lp,
                                certified_lp_split, certify_randomized_chain,
                                pareto_frontier_idx, screened_candidate_set,
                                solve_risk_lp, hb_upper_bound)
from contractrag.optimizer import build_candidates, apply_candidate, robust_order
from contractrag.policy import TrackData, RungScorer

TRACK_CFG = {
    "hybridqa": ("quality", 0.5, [0.30, 0.34, 0.35, 0.40]),
    "crag": ("correct", 0.5, [0.62, 0.65, 0.70]),
    "asqa": ("citation", 50.0, [0.20, 0.25, 0.30]),
    "qampari": ("citation", 50.0, [0.60, 0.66, 0.71]),
}


def make_loss_fn(track, contract, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        return lambda rec, sc: float(sc.get("label") != "correct")
    return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)


def load(track, contract, tau, split):
    td = TrackData(track, split, make_loss_fn(track, contract, tau))
    ld, qids, _, aux = td.build()
    return ld, aux["raw_feats"]


def candidate_matrices(cands, ld):
    """(K, n) loss and cost matrices of every candidate on a split."""
    K, n = len(cands), ld.n
    Lm = np.zeros((K, n))
    Cm = np.zeros((K, n))
    for k, c in enumerate(cands):
        loss, cost, _, _ = apply_candidate(c, ld)
        Lm[k] = loss
        Cm[k] = cost
    return Lm, Cm


def fixed_sequence_select(Lm_cal, Cm_cal, order, alpha, delta):
    """Incumbent: walk `order`, stop at first HB failure, keep cheapest."""
    n = Lm_cal.shape[1]
    certified = []
    for k in order:
        r = float(Lm_cal[k].mean())
        if hb_p_value(r, n, alpha) <= delta:
            certified.append((k, float(Cm_cal[k].mean())))
        else:
            break
    if not certified:
        return None
    return min(certified, key=lambda t: t[1])[0]


def bonferroni_deterministic(Lm_cal, Cm_cal, frontier, alpha, delta):
    """Ablation: simultaneous bounds over the frontier but NO randomization.

    Isolates the contribution of (F1) -- being able to test the whole
    frontier instead of a chain prefix -- from the contribution of (F2)/(F3)
    -- randomization plus the variance-reduced mixed loss.
    """
    n = Lm_cal.shape[1]
    d = delta / max(1, len(frontier))
    best, best_c = None, np.inf
    for k in frontier:
        rb = hb_upper_bound(float(Lm_cal[k].mean()), n, d)
        if rb <= alpha:
            c = float(Cm_cal[k].mean())
            if c < best_c:
                best, best_c = k, c
    return best


def run_track(track, draws, delta, cal_frac, seed, out_dir):
    contract, tau, alphas = TRACK_CFG[track]
    ld_tr, ft_tr = load(track, contract, tau, "train")
    ld_ca, ft_ca = load(track, contract, tau, "cal")
    ld_te, ft_te = load(track, contract, tau, "test")
    scorer = RungScorer().fit(ld_tr, ft_tr)
    ld_tr.scores = scorer.score(ft_tr)
    ld_ca.scores = scorer.score(ft_ca)
    ld_te.scores = scorer.score(ft_te)

    cands = build_candidates(ld_tr)
    Lm_tr, Cm_tr = candidate_matrices(cands, ld_tr)
    Lm_ca, Cm_ca = candidate_matrices(cands, ld_ca)
    Lm_te, Cm_te = candidate_matrices(cands, ld_te)
    # pooled finite population = cal + test executions
    Lm_pop = np.concatenate([Lm_ca, Lm_te], axis=1)
    Cm_pop = np.concatenate([Cm_ca, Cm_te], axis=1)
    n_pop = Lm_pop.shape[1]
    n_cal = int(round(cal_frac * n_pop))
    tr_risk = Lm_tr.mean(axis=1)
    tr_cost = Cm_tr.mean(axis=1)
    R_pop = Lm_pop.mean(axis=1)
    C_pop = Cm_pop.mean(axis=1)
    order_naive = sorted(range(len(cands)), key=lambda i: (tr_risk[i], tr_cost[i]))
    order_robust = robust_order(cands, ld_tr, folds=2, seed=0)

    print(f"\n{'='*78}\n{track}: K={len(cands)} n_train={ld_tr.n} "
          f"n_pop={n_pop} n_cal={n_cal} draws={draws}\n{'='*78}")

    results = {}
    for alpha in alphas:
        screen = screened_candidate_set(tr_risk, tr_cost, alpha, ld_tr.n,
                                        delta, 64)
        frontier = pareto_frontier_idx(tr_risk, tr_cost, 48)
        chain_det = build_randomized_chain(Lm_tr, Cm_tr, mix_grid=0)
        chain_mix = build_randomized_chain(Lm_tr, Cm_tr, mix_grid=7)
        pool = build_element_pool(Lm_tr, Cm_tr, mix_grid=7)
        # reference points on the population (evaluation only, never used to
        # pick a policy): cheapest feasible deterministic plan and LP optimum
        feas = np.where(R_pop <= alpha)[0]
        best_det_pop = float(C_pop[feas].min()) if len(feas) else None
        lp_pop, _ = solve_risk_lp(R_pop[frontier], C_pop[frontier], alpha)

        # graph arms: eta is the fraction of delta reserved for alternative
        # entry points. eta=0 is exactly the chain, so the sweep isolates the
        # value of budget recycling inside one algorithm.
        graph_arms = [("graph_eta0", 0.0, 1), ("graph_e10", 0.10, 4),
                      ("graph_e25", 0.25, 4), ("graph_e25w", 0.25, 8),
                      ("graph_e50", 0.50, 8)]
        methods = ["fs_naive", "fs_robust", "bonf_det", "clp",
                   "chain_det", "chain_mix", "chain_mix_eb"] + \
                  [g[0] for g in graph_arms]
        rng = np.random.default_rng(seed)
        acc = {m: {"viol": 0, "cost": [], "risk": [], "cert": 0, "nsup": []}
               for m in methods}
        t0 = time.time()
        for _ in range(draws):
            idx = rng.permutation(n_pop)[:n_cal]
            Lc, Cc = Lm_pop[:, idx], Cm_pop[:, idx]

            def record(name, risk, cost, cert, nsup):
                a = acc[name]
                a["risk"].append(float(risk))
                a["cost"].append(float(cost))
                a["nsup"].append(int(nsup))
                a["cert"] += int(bool(cert))
                if risk > alpha + 1e-12:
                    a["viol"] += 1

            for name, order in (("fs_naive", order_naive),
                                ("fs_robust", order_robust)):
                k = fixed_sequence_select(Lc, Cc, order, alpha, delta)
                if k is None:
                    k = int(np.argmin(tr_risk))          # documented fallback
                    record(name, R_pop[k], C_pop[k], False, 1)
                else:
                    record(name, R_pop[k], C_pop[k], True, 1)

            k = bonferroni_deterministic(Lc, Cc, screen, alpha, delta)
            if k is None:
                k = screen[int(np.argmin(tr_risk[screen]))]
                record("bonf_det", R_pop[k], C_pop[k], False, 1)
            else:
                record("bonf_det", R_pop[k], C_pop[k], True, 1)

            mx = certified_lp(Lc, Cc, tr_risk, tr_cost, alpha, delta,
                              bound="hb", screen="plausible", n_train=ld_tr.n)
            r = float(sum(w * R_pop[k]
                          for k, w in zip(mx.support, mx.support_w)))
            c = float(sum(w * C_pop[k]
                          for k, w in zip(mx.support, mx.support_w)))
            record("clp", r, c, mx.certified, len(mx.support))

            for name, ch, bd in (("chain_det", chain_det, "hb"),
                                 ("chain_mix", chain_mix, "hb"),
                                 ("chain_mix_eb", chain_mix, "auto")):
                cp = certify_randomized_chain(Lc, Cc, ch, alpha, delta,
                                              bound=bd)
                if cp.element is None:
                    k = int(np.argmin(tr_risk))
                    record(name, R_pop[k], C_pop[k], False, 1)
                else:
                    e = cp.element
                    r = float(sum(w * R_pop[k]
                                  for k, w in zip(e.support, e.weights)))
                    c = float(sum(w * C_pop[k]
                                  for k, w in zip(e.support, e.weights)))
                    record(name, r, c, True, len(e.support))

            for name, eta, ne in graph_arms:
                gc = graph_certify(Lc, Cc, pool, alpha, delta,
                                   eta=eta, n_entries=ne)
                if gc.element is None:
                    k = int(np.argmin(tr_risk))
                    record(name, R_pop[k], C_pop[k], False, 1)
                else:
                    e = gc.element
                    r = float(sum(w * R_pop[k]
                                  for k, w in zip(e.support, e.weights)))
                    c = float(sum(w * C_pop[k]
                                  for k, w in zip(e.support, e.weights)))
                    record(name, r, c, True, len(e.support))
        el = time.time() - t0

        print(f"\n  alpha={alpha:.2f}  |screen|={len(screen)} |pareto|="
              f"{len(frontier)} |chain|={len(chain_mix)}  "
              f"(population: best feasible det="
              f"{(best_det_pop or 0)*1000:.3f}  LP opt="
              f"{(lp_pop or 0)*1000:.3f} mCNY)   [{el:.1f}s]")
        print(f"  {'method':<12} {'viol.prob':>10} {'mean risk':>10} "
              f"{'mean cost':>11} {'vs incumbent':>13} {'cert%':>7} {'|sup|':>6}")
        base = float(np.mean(acc["fs_naive"]["cost"]))
        row = {}
        for m in methods:
            a = acc[m]
            vp = a["viol"] / draws
            mc = float(np.mean(a["cost"]))
            mr = float(np.mean(a["risk"]))
            sv = (base / mc) if mc > 0 else float("inf")
            print(f"  {m:<12} {vp:10.3f} {mr:10.3f} {mc*1000:11.3f} "
                  f"{sv:12.2f}x {100*a['cert']/draws:6.1f} "
                  f"{np.mean(a['nsup']):6.2f}")
            row[m] = {"viol_prob": vp, "mean_risk": mr, "mean_cost": mc,
                      "speedup_vs_incumbent": sv,
                      "cert_rate": a["cert"] / draws,
                      "mean_support": float(np.mean(a["nsup"]))}
        results[str(alpha)] = {"alpha": alpha, "methods": row,
                               "pop_best_det": best_det_pop,
                               "pop_lp_opt": lp_pop,
                               "n_screen": len(screen),
                               "n_pareto": len(frontier)}

    out = {"track": track, "contract": contract, "tau": tau, "delta": delta,
           "draws": draws, "n_pop": n_pop, "n_cal": n_cal, "n_train": ld_tr.n,
           "n_candidates": len(cands), "results": results}
    path = os.path.join(out_dir, f"certlp_validate_{track}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n  wrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=list(TRACK_CFG))
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments")
    for t in args.tracks:
        run_track(t, args.draws, args.delta, args.cal_frac, args.seed, out_dir)


if __name__ == "__main__":
    main()
