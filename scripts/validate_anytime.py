#!/usr/bin/env python3
"""Static one-shot certification vs anytime certification, on total cost.

The comparison charges every method for everything it consumes. A one-shot
certifier must first execute a calibration block to the strongest rung and
label it; that spend is real and, in the incumbent accounting, invisible. An
anytime certifier pays instead a thin ongoing audit stream. Over a deployment
horizon the question is which accumulated total is smaller, and whether the
anytime method's plan stays inside the contract at every point in time, not
merely at one nominal calibration moment.

Metrics per track/alpha:
  mean total cost per query over the horizon (deployment + calibration/audit)
  anytime violation = fraction of served queries whose deployed plan had
      population risk above alpha (a valid method keeps this at ~0 and must
      keep P(ever deploying an infeasible plan) <= delta)
  terminal plan cost, and distance to the population LP optimum.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.anytime import AnytimeCertifiedOptimizer, BettingCS
from contractrag.calibrate import hb_p_value
from contractrag.certlp import (build_randomized_chain,
                                certify_randomized_chain, pareto_frontier_idx,
                                solve_risk_lp)
from contractrag.optimizer import apply_candidate, build_candidates
from contractrag.policy import TrackData, RungScorer

TRACK_CFG = {
    "hybridqa": ("quality", 0.5, [0.30, 0.34, 0.40]),
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
    K, n = len(cands), ld.n
    Lm, Cm = np.zeros((K, n)), np.zeros((K, n))
    for k, c in enumerate(cands):
        loss, cost, _, _ = apply_candidate(c, ld)
        Lm[k], Cm[k] = loss, cost
    return Lm, Cm


def static_run(Lm, Cm, chain, alpha, delta, order, n_cal, audit_unit):
    """One-shot: pay a calibration block, certify once, then serve forever."""
    cal_idx = order[:n_cal]
    cp = certify_randomized_chain(Lm[:, cal_idx], Cm[:, cal_idx], chain,
                                  alpha, delta, bound="hb")
    if cp.element is None:
        plan_sup, plan_w = (int(np.argmin(Lm.mean(axis=1))),), (1.0,)
    else:
        plan_sup, plan_w = cp.element.support, cp.element.weights
    serve = order[n_cal:]
    cost = 0.0
    for k, w in zip(plan_sup, plan_w):
        cost += w * float(Cm[k, serve].sum())
    cal_cost = n_cal * audit_unit
    pop_risk = float(sum(w * Lm[k].mean() for k, w in zip(plan_sup, plan_w)))
    return {
        "total_cost": cost + cal_cost,
        "mean_cost": (cost + cal_cost) / len(order),
        "serve_mean_cost": cost / max(len(serve), 1),
        "cal_cost": cal_cost,
        "pop_risk": pop_risk,
        "violates": bool(pop_risk > alpha + 1e-12),
        "anytime_viol_frac": float(pop_risk > alpha + 1e-12),
        "certified": bool(cp.element is not None),
        "plan": [int(k) for k in plan_sup],
        "plan_w": [float(w) for w in plan_w],
    }


def anytime_run(Lm, Cm, alpha, delta, order, audit_rate, audit_unit,
                recert_every, seed):
    """Anytime: audit a thin stream, tighten confidence sequences, re-optimize."""
    opt = AnytimeCertifiedOptimizer(Lm, Cm, alpha, delta,
                                   audit_rate=audit_rate,
                                   audit_unit_cost=audit_unit,
                                   recert_every=recert_every)
    rng = np.random.default_rng(seed)
    st = opt.run(order, rng, log_every=max(1, len(order) // 60))
    R_pop = Lm.mean(axis=1)
    viol_steps = sum(1 for h in st.history if h["pop_risk"] > alpha + 1e-12)
    return {
        "total_cost": st.total_cost,
        "mean_cost": st.total_cost / st.t,
        "deploy_cost": st.deploy_cost,
        "audit_cost": st.audit_cost,
        "audits": st.audits,
        "switches": st.switches,
        "final_plan": int(st.plan),
        "final_pop_risk": float(R_pop[st.plan]),
        "violates": bool(R_pop[st.plan] > alpha + 1e-12),
        "anytime_viol_frac": viol_steps / max(len(st.history), 1),
        "history": st.history,
    }


def run_track(track, horizon, delta, seeds, out_dir):
    contract, tau, alphas = TRACK_CFG[track]
    ld_tr, ft_tr = load(track, contract, tau, "train")
    ld_ca, ft_ca = load(track, contract, tau, "cal")
    ld_te, ft_te = load(track, contract, tau, "test")
    scorer = RungScorer().fit(ld_tr, ft_tr)
    for ld, ft in ((ld_tr, ft_tr), (ld_ca, ft_ca), (ld_te, ft_te)):
        ld.scores = scorer.score(ft)
    cands = build_candidates(ld_tr)
    Lm_tr, Cm_tr = candidate_matrices(cands, ld_tr)
    Lm_ca, Cm_ca = candidate_matrices(cands, ld_ca)
    Lm_te, Cm_te = candidate_matrices(cands, ld_te)
    Lm = np.concatenate([Lm_ca, Lm_te], axis=1)
    Cm = np.concatenate([Cm_ca, Cm_te], axis=1)
    n_pop = Lm.shape[1]
    R_pop, C_pop = Lm.mean(axis=1), Cm.mean(axis=1)
    chain = build_randomized_chain(Lm_tr, Cm_tr, mix_grid=7)
    front = pareto_frontier_idx(Lm_tr.mean(axis=1), Cm_tr.mean(axis=1), 48)
    audit_unit = float(Cm.max(axis=0).mean())   # strongest-rung cost per query

    print(f"\n{'='*80}\n{track}: K={len(cands)} n_pop={n_pop} horizon={horizon} "
          f"audit_unit={audit_unit*1000:.3f} mCNY\n{'='*80}")
    results = {}
    for alpha in alphas:
        lp_opt, _ = solve_risk_lp(R_pop[front], C_pop[front], alpha)
        feas = np.where(R_pop <= alpha)[0]
        best_det = float(C_pop[feas].min()) if len(feas) else float("nan")
        rows = {}
        for name, cfg in [
            ("static_ncal250", {"kind": "static", "n_cal": 250}),
            ("static_ncal500", {"kind": "static", "n_cal": 500}),
            ("static_ncal1000", {"kind": "static", "n_cal": 1000}),
            ("anytime_a02", {"kind": "anytime", "rate": 0.02}),
            ("anytime_a05", {"kind": "anytime", "rate": 0.05}),
            ("anytime_a10", {"kind": "anytime", "rate": 0.10}),
        ]:
            accs = []
            for s in seeds:
                rng = np.random.default_rng(1000 + s)
                order = rng.integers(0, n_pop, size=horizon)
                if cfg["kind"] == "static":
                    accs.append(static_run(Lm, Cm, chain, alpha, delta, order,
                                           cfg["n_cal"], audit_unit))
                else:
                    accs.append(anytime_run(Lm, Cm, alpha, delta, order,
                                            cfg["rate"], audit_unit, 50, s))
            mc = float(np.mean([a["mean_cost"] for a in accs]))
            vf = float(np.mean([a["anytime_viol_frac"] for a in accs]))
            vp = float(np.mean([a["violates"] for a in accs]))
            rows[name] = {"mean_cost": mc, "anytime_viol_frac": vf,
                          "viol_prob": vp,
                          "detail": {k: v for k, v in accs[0].items()
                                     if k != "history"}}
            if cfg["kind"] == "anytime":
                rows[name]["audit_frac_of_cost"] = float(np.mean(
                    [a["audit_cost"] / max(a["total_cost"], 1e-12)
                     for a in accs]))
                rows[name]["switches"] = float(np.mean(
                    [a["switches"] for a in accs]))
        base = rows["static_ncal500"]["mean_cost"]
        print(f"\n  alpha={alpha:.2f}  (pop best det={best_det*1000:.3f}  "
              f"LP opt={(lp_opt or float('nan'))*1000:.3f} mCNY)")
        print(f"  {'method':<18} {'mean cost':>11} {'vs static500':>13} "
              f"{'viol.prob':>10} {'anytime viol':>13} {'audit%':>8}")
        for name in rows:
            r = rows[name]
            af = r.get("audit_frac_of_cost")
            print(f"  {name:<18} {r['mean_cost']*1000:11.3f} "
                  f"{base/r['mean_cost']:12.2f}x {r['viol_prob']:10.3f} "
                  f"{r['anytime_viol_frac']:13.3f} "
                  f"{(af*100 if af is not None else float('nan')):8.1f}")
        results[str(alpha)] = {"alpha": alpha, "pop_best_det": best_det,
                              "pop_lp_opt": lp_opt, "methods": rows}
    out = {"track": track, "delta": delta, "horizon": horizon,
           "n_pop": n_pop, "n_candidates": len(cands),
           "audit_unit_cost": audit_unit, "seeds": list(seeds),
           "results": results}
    path = os.path.join(out_dir, f"anytime_validate_{track}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\n  wrote {path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=list(TRACK_CFG))
    ap.add_argument("--horizon", type=int, default=20000)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments")
    for t in args.tracks:
        run_track(t, args.horizon, args.delta, range(args.seeds), out_dir)


if __name__ == "__main__":
    main()
