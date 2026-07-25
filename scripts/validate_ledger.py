#!/usr/bin/env python3
"""Ledger-gated execution vs. one-shot certification, on identical streams.

Both systems get the same plan space, the same decline action at the same
price, the same query stream, and are charged for everything they consume --
including the queries a certifier burns on calibration, which are real queries
served without a certificate by whatever the history liked.

Columns:
  mean cost    total cost of ownership / stream length (serve + decline + audit)
  risk         realized violation rate over the whole stream
  abstain%     fraction of arrivals declined
  worst rate   max over prefixes t >= warm of V_t / t -- the quantity an anytime
               contract controls, and the one a fixed-sample guarantee says
               nothing about
  breach       fraction of draws whose worst prefix rate exceeded alpha

`--alphas` deliberately includes levels below the risk of every plan in the
space. There, one-shot certification without declines is infeasible by
construction, which is the regime a strict contract actually lives in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.certlp import pareto_frontier_idx
from contractrag.ledger import LedgerExecutor, solve_action_lp, static_stream_cost
from contractrag.optimizer import apply_candidate, build_candidates
from contractrag.policy import RungScorer, TrackData

TRACK_CFG = {
    "hybridqa": ("quality", 0.5, [0.15, 0.20, 0.25, 0.30, 0.34]),
    "crag": ("correct", 0.5, [0.35, 0.45, 0.55, 0.62, 0.70]),
    "asqa": ("citation", 50.0, [0.10, 0.15, 0.20, 0.25, 0.30]),
    "qampari": ("citation", 50.0, [0.35, 0.45, 0.55, 0.60, 0.66]),
    "hqgrid": ("quality", 0.5, [0.15, 0.20, 0.25, 0.28, 0.34]),
    "cggrid": ("correct", 0.5, [0.35, 0.45, 0.55, 0.62, 0.70]),
}

# plan-grid tracks: (benchmark, generator set) for the unbundled plan space
GRID_OF = {"hqgrid": "hybridqa", "cggrid": "crag"}


def make_loss_fn(track, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        return lambda rec, sc: float(sc.get("label") != "correct")
    return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)


def cand_matrices(cands, ld):
    L, C = np.zeros((len(cands), ld.n)), np.zeros((len(cands), ld.n))
    for k, c in enumerate(cands):
        loss, cost, _, _ = apply_candidate(c, ld)
        L[k], C[k] = loss, cost
    return L, C


def load_track(track, tau):
    if track in GRID_OF:
        from scripts.experiment_plangrid_all import (HOSTED, LOCAL,
                                                     load_plan_grid)
        bench = GRID_OF[track]
        models = HOSTED + LOCAL
        out = []
        for split in ("train", "cal", "test"):
            L, C, names, _, _ = load_plan_grid(bench, models, split)
            out.append((L, C, names))
        # a generator still being executed is present in some splits only;
        # keep the plans every split has, in the order the first one gives
        keep = set(out[0][2]) & set(out[1][2]) & set(out[2][2])
        order = [nm for nm in out[0][2] if nm in keep]
        mats = []
        for L, C, names in out:
            idx = [names.index(nm) for nm in order]
            mats.append((L[idx], C[idx]))
        return mats, order
    lds = []
    for split in ("train", "cal", "test"):
        td = TrackData(track, split, make_loss_fn(track, tau))
        ld, _, _, aux = td.build()
        lds.append((ld, aux["raw_feats"]))
    sc = RungScorer().fit(lds[0][0], lds[0][1])
    for ld, ft in lds:
        ld.scores = sc.score(ft)
    cands = build_candidates(lds[0][0])
    mats = [cand_matrices(cands, ld) for ld, _ in lds]
    return mats, [c.describe() for c in cands]


def worst_prefix_rate(cum_viol, warm=200):
    t = np.arange(1, len(cum_viol) + 1)
    return float(np.max((cum_viol / t)[warm:]))


def run_track(track, args, out_dir):
    _, tau, default_alphas = TRACK_CFG[track]
    alphas = args.alphas or default_alphas
    mats, names = load_track(track, tau)
    (L_tr, C_tr), (L_ca, C_ca), (L_te, C_te) = mats
    L_pop = np.concatenate([L_ca, L_te], axis=1)
    C_pop = np.concatenate([C_ca, C_te], axis=1)
    N = L_pop.shape[1]
    tr_risk, tr_cost = L_tr.mean(axis=1), C_tr.mean(axis=1)
    front = pareto_frontier_idx(tr_risk, tr_cost, args.max_plans)
    Lp, Cp = L_pop[front], C_pop[front]
    wr, wc = tr_risk[front], tr_cost[front]
    pop_r, pop_c = Lp.mean(axis=1), Cp.mean(axis=1)
    audit_unit = (float(tr_cost.min()) if args.audit_unit < 0
                  else args.audit_unit)
    # Declining is priced relative to the most expensive plan in the space, so
    # the sweep is comparable across tracks whose absolute prices differ by an
    # order of magnitude. x1 means a decline costs what the strongest plan
    # costs; x4 makes declining a genuinely bad outcome to be avoided.
    # When two spaces over the same task are compared, the price of declining
    # must not move with the space: it is what the operator pays a human, not a
    # property of the plan set. --abstain_abs pins it so the ladder and the grid
    # are charged the same for the same outcome.
    ref = (float(pop_c.max()) if args.abstain_abs < 0
           else args.abstain_abs / 1000.0)
    ab_costs = [m * ref for m in args.abstain_mult]

    print(f"\n{'='*96}\n{track}: |plans|={len(front)} of {L_tr.shape[0]}  "
          f"n_hist={L_tr.shape[1]}  N_stream={N}  "
          f"min plan risk={pop_r.min():.3f}  "
          f"audit={audit_unit*1000:.3f} mCNY  draws={args.draws}\n{'='*96}")
    out = {}
    for alpha in alphas:
        feas = np.where(pop_r <= alpha)[0]
        best_fixed = float(pop_c[feas].min()) if len(feas) else None
        print(f"\n  alpha={alpha:.2f}   cheapest feasible fixed plan = "
              f"{'infeasible' if best_fixed is None else format(best_fixed*1000, '.4f')}")
        rows = {}
        for ab in ab_costs:
            opt, wopt = solve_action_lp(pop_r, pop_c, alpha, ab)
            print(f"  decline price {ab*1000:8.3f} mCNY | oracle LP "
                  f"{opt*1000:8.4f} mCNY, decline {wopt[-1]*100:5.1f}%")
            print(f"  {'method':<26} {'mean cost':>10} {'vs static':>10} "
                  f"{'risk':>7} {'abstain%':>9} {'worst rate':>11} "
                  f"{'breach':>7} {'vs oracle':>10}")
            acc = {}

            def add(name, cost, risk, ab_rate, worst, breach):
                a = acc.setdefault(name, {"c": [], "r": [], "a": [], "w": [],
                                          "b": 0})
                a["c"].append(cost)
                a["r"].append(risk)
                a["a"].append(ab_rate)
                a["w"].append(worst)
                a["b"] += int(breach)

            t0 = time.time()
            rng = np.random.default_rng(args.seed)
            for _ in range(args.draws):
                order = rng.permutation(N)
                for tag, allow in (("static", True), ("static/no-decline", False)):
                    s = static_stream_cost(Lp, Cp, alpha, args.delta, order,
                                           args.n_cal, wr, wc,
                                           abstain_cost=ab,
                                           audit_unit_cost=audit_unit,
                                           allow_abstain=allow)
                    wo = worst_prefix_rate(s["cum_viol"], args.warm)
                    add(tag, s["mean_cost"], s["risk"], 100 * s["abstain_rate"],
                        wo, wo > alpha)
                for pol in args.policies:
                    for rate in args.audit_rates:
                        ex = LedgerExecutor(
                            Lp, Cp, alpha, abstain_cost=ab, hist_risk=wr,
                            hist_cost=wc, kappa=args.kappa, policy=pol,
                            audit_rate=rate, audit_unit_cost=audit_unit,
                            audit_mode=args.audit_mode, prior_n=args.prior_n)
                        st = ex.run(order, rng, log_every=10 ** 9)
                        wo = worst_prefix_rate(st.cum_viol, args.warm)
                        nm = f"ledger/{pol}" + (f" a={rate:g}" if rate < 1 else "")
                        add(nm, st.total_cost / st.t, st.risk,
                            100 * st.abstain_rate, wo, wo > alpha)
            el = time.time() - t0
            base = float(np.mean(acc["static"]["c"]))
            for name, a in acc.items():
                mc = float(np.mean(a["c"]))
                print(f"  {name:<26} {mc*1000:10.4f} {base/mc:9.2f}x "
                      f"{np.mean(a['r']):7.3f} {np.mean(a['a']):8.1f}% "
                      f"{np.mean(a['w']):11.3f} {a['b']/args.draws:7.2f} "
                      f"{mc/max(opt,1e-12):9.2f}x")
                rows[f"{ab}|{name}"] = {
                    "abstain_cost": ab, "method": name, "mean_cost": mc,
                    "risk": float(np.mean(a["r"])),
                    "abstain_rate": float(np.mean(a["a"])),
                    "worst_rate": float(np.mean(a["w"])),
                    "breach": a["b"] / args.draws, "vs_static": base / mc,
                    "vs_oracle": mc / max(opt, 1e-12), "oracle": opt}
            print(f"    [{el:.1f}s]")
        out[str(alpha)] = {"alpha": alpha, "best_fixed": best_fixed,
                           "rows": rows}
    res = {"track": track, "n_plans": len(front), "N": N,
           "audit_unit": audit_unit, "abstain_costs": ab_costs,
           "plans": [names[i] for i in front],
           "pop_risk": pop_r.tolist(), "pop_cost": pop_c.tolist(),
           "n_hist": int(L_tr.shape[1]), "n_cal": args.n_cal,
           "delta": args.delta, "draws": args.draws, "results": out}
    # a run over a few extra contract levels must not clobber the main sweep
    path = os.path.join(out_dir, f"ledger_{track}{args.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\n  wrote {path}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=["hybridqa"])
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--abstain_mult", type=float, nargs="*", default=[1.0])
    ap.add_argument("--abstain_abs", type=float, default=-1.0,
                    help="decline price in mCNY; default is the dearest plan")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file, e.g. _atcert")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--audit_rates", type=float, nargs="*", default=[1.0])
    ap.add_argument("--audit_mode", default="worst-case",
                    choices=["worst-case", "ipw"])
    ap.add_argument("--policies", nargs="*", default=["lp", "greedy"])
    ap.add_argument("--n_cal", type=int, default=500)
    ap.add_argument("--max_plans", type=int, default=16)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--prior_n", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit_unit", type=float, default=-1.0)
    ap.add_argument("--warm", type=int, default=200)
    args = ap.parse_args()
    out_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments")
    for t in args.tracks:
        run_track(t, args, out_dir)


if __name__ == "__main__":
    main()
