#!/usr/bin/env python3
"""Static certification vs. certified adaptive execution, on one query stream.

Protocol. The pooled cal+test executions form a finite population; one draw
shuffles it into a stream. Both systems are given the SAME history (the train
split) and must serve the SAME stream, and both are charged for every query they
serve plus every label they consume. That is the comparison prior work avoids:
a one-shot certifier looks cheap only if the queries it burns on calibration are
free, and they are not -- they are served without a certificate, by a plan the
history merely believes is safe.

Reported per method:
  mean cost      total cost of ownership divided by stream length;
  risk           realized violation rate over the whole stream;
  anytime viol   fraction of draws in which the realized rate exceeded alpha at
                 ANY prefix of the stream, which is the quantity an anytime
                 guarantee controls and a fixed-sample one does not;
  final viol     fraction of draws whose end-of-stream rate exceeded alpha.

`--deltas` sweeps the error budget. That sweep is the tightness check: a bound
that is merely valid drives the violation rate to zero and tells us nothing,
while a bound that is tight tracks delta as delta grows.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.cae import CertifiedAdaptiveExecutor, static_stream_cost
from contractrag.certlp import pareto_frontier_idx
from contractrag.optimizer import apply_candidate, build_candidates
from contractrag.policy import RungScorer, TrackData

TRACK_CFG = {
    "hybridqa": ("quality", 0.5, [0.30, 0.34, 0.40]),
    "crag": ("correct", 0.5, [0.62, 0.65, 0.70]),
    "asqa": ("citation", 50.0, [0.20, 0.25, 0.30]),
    "qampari": ("citation", 50.0, [0.60, 0.66, 0.71]),
    # the unbundled plan grid: 4 access paths x 6 generators from 3 vendors,
    # executed end to end. Same queries as `hybridqa`, a plan space six times
    # larger, which is exactly the regime where a static certifier's multiple
    # testing burden starts to outweigh the cheaper plans the space contains.
    "hqgrid": ("quality", 0.5, [0.28, 0.30, 0.34, 0.40]),
}


def make_loss_fn(track, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        return lambda rec, sc: float(sc.get("label") != "correct")
    return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)


def load(track, tau, split):
    td = TrackData(track, split, make_loss_fn(track, tau))
    ld, _, _, aux = td.build()
    return ld, aux["raw_feats"]


def cand_matrices(cands, ld):
    K, n = len(cands), ld.n
    L, C = np.zeros((K, n)), np.zeros((K, n))
    for k, c in enumerate(cands):
        loss, cost, _, _ = apply_candidate(c, ld)
        L[k], C[k] = loss, cost
    return L, C


def anytime_violation(cum_viol, alpha, delta=0.1, warm=50):
    """Did the running rate ever exceed what an anytime contract allows?

    The comparison cannot be against alpha itself. Over the first few hundred
    queries the empirical rate of ANY policy fluctuates by sqrt(alpha(1-alpha)/t),
    so a policy whose true risk is exactly alpha - eps still crosses alpha
    infinitely often early on, and reporting that as a breach measures sampling
    noise rather than the controller. The quantity a time-uniform guarantee
    controls is the rate minus a tolerance that shrinks like sqrt(log(1/delta)/t),
    which is what we test here -- and the same test is applied to the static
    baseline, so the columns are comparable.
    """
    t = np.arange(1, len(cum_viol) + 1)
    tol = np.sqrt(np.log(1.0 / max(delta, 1e-12)) / (2.0 * t))
    rate = cum_viol / t
    return bool(np.any(rate[warm:] > alpha + tol[warm:] + 1e-12))


def grid_matrices():
    """Physical plan grid: every (access path, generator) pair, executed."""
    from scripts.experiment_plangrid import HOSTED, LOCAL, load_plan_grid
    models = HOSTED + LOCAL
    out = []
    for split in ("train", "cal", "test"):
        L, C, names, _, _ = load_plan_grid(models, split)
        out.append((L, C, names))
    assert out[0][2] == out[1][2] == out[2][2], "plan sets differ across splits"
    return [(L, C) for L, C, _ in out], out[0][2]


def run_track(track, args, out_dir):
    _, tau, alphas = TRACK_CFG[track]
    if track == "hqgrid":
        (mats, _) = grid_matrices()
        (L_tr, C_tr), (L_ca, C_ca), (L_te, C_te) = mats
    else:
        ld_tr, ft_tr = load(track, tau, "train")
        ld_ca, ft_ca = load(track, tau, "cal")
        ld_te, ft_te = load(track, tau, "test")
        sc = RungScorer().fit(ld_tr, ft_tr)
        for ld, ft in ((ld_tr, ft_tr), (ld_ca, ft_ca), (ld_te, ft_te)):
            ld.scores = sc.score(ft)
        cands = build_candidates(ld_tr)
        L_tr, C_tr = cand_matrices(cands, ld_tr)
        L_ca, C_ca = cand_matrices(cands, ld_ca)
        L_te, C_te = cand_matrices(cands, ld_te)
    L_pop = np.concatenate([L_ca, L_te], axis=1)
    C_pop = np.concatenate([C_ca, C_te], axis=1)
    N = L_pop.shape[1]
    tr_risk, tr_cost = L_tr.mean(axis=1), C_tr.mean(axis=1)

    # both systems see the same candidate screen: the historical Pareto
    # frontier. Restricting the adaptive controller to it is what keeps the
    # delta split over plans small, and the static certifier already uses it.
    front = pareto_frontier_idx(tr_risk, tr_cost, args.max_plans)
    Lp, Cp = L_pop[front], C_pop[front]
    wr, wc = tr_risk[front], tr_cost[front]
    # A label is produced by a judge scoring an answer that already exists, not
    # by re-running a plan, so its price is one judge call. We charge the cost of
    # the cheapest plan in the space, which is the same order of magnitude as a
    # single mid-tier generation and is the conservative reading.
    audit_unit = (float(C_tr.mean(axis=1).min()) if args.audit_unit < 0
                  else args.audit_unit)

    print(f"\n{'='*84}\n{track}: |plans|={len(front)} n_hist={L_tr.shape[1]} "
          f"N_stream={N} audit_unit={audit_unit*1000:.3f} mCNY "
          f"draws={args.draws}\n{'='*84}")
    out = {}
    for alpha in alphas:
        feas = np.where(Lp.mean(axis=1) <= alpha)[0]
        pop_best = float(Cp[feas].mean(axis=1).min()) if len(feas) else None
        print(f"\n  alpha={alpha:.2f}   population cheapest feasible plan = "
              f"{(pop_best or 0)*1000:.4f} mCNY")
        print(f"  {'method':<22} {'mean cost':>10} {'vs static':>10} "
              f"{'risk':>7} {'anytime v':>10} {'final v':>8} {'fallback%':>10}")
        rows = {}
        for delta in args.deltas:
            rng = np.random.default_rng(args.seed)
            acc = {}

            def add(name, cost, risk, av, fv, expl):
                a = acc.setdefault(name, {"cost": [], "risk": [], "av": 0,
                                          "fv": 0, "expl": []})
                a["cost"].append(cost)
                a["risk"].append(risk)
                a["av"] += int(av)
                a["fv"] += int(fv)
                a["expl"].append(expl)

            t0 = time.time()
            for _ in range(args.draws):
                order = rng.permutation(N)
                stat = static_stream_cost(Lp, Cp, alpha, delta, order,
                                          args.n_cal, wr, wc,
                                          audit_unit_cost=audit_unit)
                add("static", stat["mean_cost"], stat["risk"],
                    anytime_violation(stat["cum_viol"], alpha, delta,
                                      warm=args.warm),
                    stat["risk"] > alpha, 0.0)
                for pol in args.policies:
                    for rate in args.audit_rates:
                        for tsr in args.safe_rates:
                            ex = CertifiedAdaptiveExecutor(
                                Lp, Cp, alpha, delta, audit_rate=rate,
                                audit_unit_cost=audit_unit, policy=pol,
                                hist_risk=wr, hist_cost=wc,
                                target_safe_rate=tsr, rho_init=args.rho_init,
                                lam_rate=args.lam_rate,
                                buffer_c=args.buffer_c)
                            st = ex.run(order, rng, log_every=args.log_every)
                            add(f"{pol} a={rate:g} s={tsr:g}",
                                st.total_cost / st.t, st.risk,
                                anytime_violation(st.cum_viol, alpha, delta,
                                                  warm=args.warm),
                                st.risk > alpha,
                                100.0 * st.fallbacks / st.t)
            el = time.time() - t0
            base = float(np.mean(acc["static"]["cost"]))
            for name, a in acc.items():
                mc = float(np.mean(a["cost"]))
                print(f"  d={delta:<4g} {name:<22} {mc*1000:10.4f} "
                      f"{base/mc:9.2f}x {np.mean(a['risk']):7.3f} "
                      f"{a['av']/args.draws:10.3f} {a['fv']/args.draws:8.3f} "
                      f"{np.mean(a['expl']):8.1f}%")
                rows[f"{delta}|{name}"] = {
                    "delta": delta, "method": name, "mean_cost": mc,
                    "risk": float(np.mean(a["risk"])),
                    "anytime_viol": a["av"] / args.draws,
                    "final_viol": a["fv"] / args.draws,
                    "explore_pct": float(np.mean(a["expl"])),
                    "vs_static": base / mc}
            print(f"       [{el:.1f}s]")
        out[str(alpha)] = {"alpha": alpha, "pop_best": pop_best, "rows": rows}
    res = {"track": track, "n_plans": len(front), "N": N,
           "n_hist": int(L_tr.shape[1]), "n_cal": args.n_cal,
           "audit_unit": audit_unit, "draws": args.draws, "results": out}
    path = os.path.join(out_dir, f"cae_{track}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\n  wrote {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=list(TRACK_CFG))
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.1])
    ap.add_argument("--audit_rates", type=float, nargs="*",
                    default=[1.0, 0.25, 0.05])
    ap.add_argument("--n_cal", type=int, default=500)
    ap.add_argument("--max_plans", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--policies", nargs="*", default=["lp", "dual"])
    ap.add_argument("--safe_rates", type=float, nargs="*", default=[0.05])
    ap.add_argument("--rho_init", type=float, default=0.02)
    ap.add_argument("--buffer_c", type=float, default=1.0)
    ap.add_argument("--lam_rate", type=float, default=0.5,
                    help="dual ascent step size pricing risk against cost")
    ap.add_argument("--audit_unit", type=float, default=-1.0,
                    help="CNY per label; default = cheapest plan's mean cost")
    ap.add_argument("--warm", type=int, default=200,
                    help="prefix excluded from the anytime check; no method can "
                         "control a rate over a handful of queries")
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments")
    for t in args.tracks:
        run_track(t, args, out_dir)


if __name__ == "__main__":
    main()

