#!/usr/bin/env python3
"""What the deterministic guarantee buys that a probabilistic one cannot.

A calibration-based certificate says: if deployment traffic is exchangeable
with the calibration sample, the risk is at most alpha with probability
1 - delta. Every word of the antecedent is load-bearing, and in production none
of it is true -- traffic is ordered by time, hard queries cluster, corpora go
stale, a new intent appears. The ledger argument has no antecedent: V_t <=
alpha t holds pathwise, for every realisation, under every ordering, chosen
adversarially if you like. This script tries to break both and reports which
one breaks.

  order=iid          uniform shuffle -- the assumption the certifier is granted
  order=hard-first   population sorted so the hardest queries arrive first, the
                     worst case for a certificate calibrated on a random prefix
  order=drift        the two halves of the population reordered so the mixture
                     shifts partway through, i.e. covariate drift mid-stream
  order=adversarial  a greedy order that, knowing the plan the certifier will
                     deploy, front-loads the queries it fails

It also measures the two quantities a SIGMOD-style claim needs beyond validity:
how the cost gap to the offline optimum decays with stream length (regret), and
how sensitive the controller is to its only tuning constant.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.certlp import pareto_frontier_idx
from contractrag.ledger import LedgerExecutor, solve_action_lp, static_stream_cost
from scripts.validate_ledger import TRACK_CFG, load_track, worst_prefix_rate


def make_order(kind, L, C, rng, hist_risk):
    """Query orderings, from the benign one to the one built to break things."""
    N = L.shape[1]
    if kind == "iid":
        return rng.permutation(N)
    # difficulty = how many plans fail this query
    hard = L.mean(axis=0)
    if kind == "hard-first":
        return np.argsort(-hard + 1e-9 * rng.random(N))
    if kind == "easy-first":
        return np.argsort(hard + 1e-9 * rng.random(N))
    if kind == "drift":
        # first half easy, second half hard: the mixture the calibration
        # prefix sees is not the mixture that follows
        o = np.argsort(hard + 1e-9 * rng.random(N))
        a, b = o[:N // 2], o[N // 2:]
        return np.concatenate([rng.permutation(a), rng.permutation(b)])
    if kind == "adversarial":
        # the certifier will deploy something close to the historically safest
        # plan; front-load exactly the queries that plan fails, so its
        # calibration estimate is optimistic and its deployment is not
        safe = int(np.argmin(hist_risk))
        bad = np.where(L[safe] > 0.5)[0]
        good = np.where(L[safe] <= 0.5)[0]
        return np.concatenate([rng.permutation(bad), rng.permutation(good)])
    raise ValueError(kind)


def prep(track, args):
    _, tau, _ = TRACK_CFG[track]
    mats, names = load_track(track, tau)
    (L_tr, C_tr), (L_ca, C_ca), (L_te, C_te) = mats
    L = np.concatenate([L_ca, L_te], axis=1)
    C = np.concatenate([C_ca, C_te], axis=1)
    tr_risk, tr_cost = L_tr.mean(axis=1), C_tr.mean(axis=1)
    front = pareto_frontier_idx(tr_risk, tr_cost, args.max_plans)
    return (L[front], C[front], tr_risk[front], tr_cost[front],
            [names[i] for i in front], float(tr_cost.min()))


def run_orders(track, args, out):
    Lp, Cp, wr, wc, names, audit_unit = prep(track, args)
    pop_r, pop_c = Lp.mean(axis=1), Cp.mean(axis=1)
    ab = args.abstain_mult * float(pop_c.max())
    print(f"\n{'='*92}\n{track} / query order stress   |plans|={len(names)} "
          f"N={Lp.shape[1]} min risk={pop_r.min():.3f}\n{'='*92}")
    res = {}
    for alpha in (args.alphas or TRACK_CFG[track][2]):
        opt, _ = solve_action_lp(pop_r, pop_c, alpha, ab)
        print(f"\n  alpha={alpha:.2f}  oracle {opt*1000:.4f} mCNY")
        print(f"  {'order':<14} {'method':<20} {'cost':>9} {'risk':>7} "
              f"{'abst%':>7} {'worst rate':>11} {'breach':>7}")
        for kind in args.orders:
            acc = {}
            for d in range(args.draws):
                rng = np.random.default_rng(args.seed + d)
                order = make_order(kind, Lp, Cp, rng, wr)
                s = static_stream_cost(Lp, Cp, alpha, args.delta, order,
                                       args.n_cal, wr, wc, abstain_cost=ab,
                                       audit_unit_cost=audit_unit)
                wo = worst_prefix_rate(s["cum_viol"], args.warm)
                acc.setdefault("static", []).append(
                    (s["mean_cost"], s["risk"], 100 * s["abstain_rate"], wo,
                     wo > alpha))
                ex = LedgerExecutor(Lp, Cp, alpha, abstain_cost=ab,
                                    hist_risk=wr, hist_cost=wc,
                                    kappa=args.kappa,
                                    audit_unit_cost=audit_unit)
                st = ex.run(order, rng, log_every=10 ** 9)
                wo = worst_prefix_rate(st.cum_viol, args.warm)
                acc.setdefault("ledger", []).append(
                    (st.total_cost / st.t, st.risk, 100 * st.abstain_rate, wo,
                     wo > alpha))
            for m, v in acc.items():
                a = np.array(v, dtype=float)
                print(f"  {kind:<14} {m:<20} {a[:,0].mean()*1000:9.4f} "
                      f"{a[:,1].mean():7.3f} {a[:,2].mean():6.1f}% "
                      f"{a[:,3].mean():11.3f} {a[:,4].mean():7.2f}")
                res[f"{alpha}|{kind}|{m}"] = {
                    "alpha": alpha, "order": kind, "method": m,
                    "cost": float(a[:, 0].mean()), "risk": float(a[:, 1].mean()),
                    "abstain": float(a[:, 2].mean()),
                    "worst_rate": float(a[:, 3].mean()),
                    "breach": float(a[:, 4].mean()), "oracle": opt}
    out["orders"] = res


def run_regret(track, args, out):
    """Cost gap to the offline optimum as a function of stream length.

    The optimum is the LP solved on the population risks, i.e. what a system
    that already knew every plan's risk would pay. The gap is what learning
    costs. Reporting it against T is the claim that the controller is not just
    valid but efficient, and the shape of the curve is the claim's evidence.
    """
    Lp, Cp, wr, wc, names, audit_unit = prep(track, args)
    pop_r, pop_c = Lp.mean(axis=1), Cp.mean(axis=1)
    ab = args.abstain_mult * float(pop_c.max())
    N = Lp.shape[1]
    print(f"\n{'='*92}\n{track} / regret vs stream length\n{'='*92}")
    res = {}
    for alpha in (args.alphas or TRACK_CFG[track][2]):
        opt, _ = solve_action_lp(pop_r, pop_c, alpha, ab)
        print(f"\n  alpha={alpha:.2f}  oracle {opt*1000:.4f} mCNY per query")
        # The oracle knows every plan's risk, so it buys no labels. Charging the
        # controller's audit bill against it would confound two different
        # things: what learning costs, which should vanish, and what continuous
        # observation costs, which is a standing expense the audit-rate sweep
        # prices separately. So the regret column is the SERVING gap and the
        # audit bill is reported beside it.
        print(f"  {'T':>7} {'serve+decl':>11} {'audit':>8} {'regret':>9} "
              f"{'x sqrt(T)':>10} {'risk':>7} {'abst%':>7}")
        for T in args.horizons:
            if T > N * args.max_reps:
                continue
            gaps, costs, risks, abst, auds = [], [], [], [], []
            for d in range(args.draws):
                rng = np.random.default_rng(args.seed + d)
                # streams longer than the population are formed by resampling
                # it, which is what a stationary stream of the same traffic
                # would look like
                order = np.concatenate([rng.permutation(N)
                                        for _ in range(int(np.ceil(T / N)))])[:T]
                ex = LedgerExecutor(Lp, Cp, alpha, abstain_cost=ab,
                                    hist_risk=wr, hist_cost=wc,
                                    kappa=args.kappa,
                                    audit_unit_cost=audit_unit)
                st = ex.run(order, rng, log_every=10 ** 9)
                mc = (st.serve_cost + st.abstain_cost_total) / st.t
                costs.append(mc)
                auds.append(st.audit_cost / st.t)
                gaps.append(mc - opt)
                risks.append(st.risk)
                abst.append(100 * st.abstain_rate)
            g = float(np.mean(gaps))
            print(f"  {T:7d} {np.mean(costs)*1000:11.4f} "
                  f"{np.mean(auds)*1000:8.4f} {g*1000:9.4f} "
                  f"{g*np.sqrt(T)*1000:10.4f} {np.mean(risks):7.3f} "
                  f"{np.mean(abst):6.1f}%")
            res[f"{alpha}|{T}"] = {"alpha": alpha, "T": T,
                                   "cost": float(np.mean(costs)), "gap": g,
                                   "audit": float(np.mean(auds)),
                                   "oracle": opt,
                                   "risk": float(np.mean(risks)),
                                   "abstain": float(np.mean(abst))}
    out["regret"] = res


def run_kappa(track, args, out):
    """Sensitivity to the one constant the controller has.

    kappa sets how far below alpha the optimiser steers. At zero the mixture
    aims at alpha, estimation noise pushes it over half the time, and the gate
    pays for it in declines. Too large and the mixture buys safety it does not
    need. Validity is flat across the sweep by construction, which is the point
    worth showing: the constant trades cost against cost.
    """
    Lp, Cp, wr, wc, names, audit_unit = prep(track, args)
    pop_r, pop_c = Lp.mean(axis=1), Cp.mean(axis=1)
    ab = args.abstain_mult * float(pop_c.max())
    N = Lp.shape[1]
    print(f"\n{'='*92}\n{track} / sensitivity to kappa\n{'='*92}")
    res = {}
    for alpha in (args.alphas or TRACK_CFG[track][2]):
        opt, _ = solve_action_lp(pop_r, pop_c, alpha, ab)
        print(f"\n  alpha={alpha:.2f}  oracle {opt*1000:.4f} mCNY")
        print(f"  {'kappa':>7} {'mean cost':>10} {'vs oracle':>10} "
              f"{'risk':>7} {'abst%':>7} {'worst rate':>11} {'breach':>7}")
        for k in args.kappas:
            c, r, a, w, b = [], [], [], [], []
            for d in range(args.draws):
                rng = np.random.default_rng(args.seed + d)
                order = rng.permutation(N)
                ex = LedgerExecutor(Lp, Cp, alpha, abstain_cost=ab,
                                    hist_risk=wr, hist_cost=wc, kappa=k,
                                    audit_unit_cost=audit_unit)
                st = ex.run(order, rng, log_every=10 ** 9)
                wo = worst_prefix_rate(st.cum_viol, args.warm)
                c.append(st.total_cost / st.t)
                r.append(st.risk)
                a.append(100 * st.abstain_rate)
                w.append(wo)
                b.append(wo > alpha)
            mc = float(np.mean(c))
            print(f"  {k:7.2f} {mc*1000:10.4f} {mc/max(opt,1e-12):9.2f}x "
                  f"{np.mean(r):7.3f} {np.mean(a):6.1f}% {np.mean(w):11.3f} "
                  f"{np.mean(b):7.2f}")
            res[f"{alpha}|{k}"] = {"alpha": alpha, "kappa": k, "cost": mc,
                                   "vs_oracle": mc / max(opt, 1e-12),
                                   "risk": float(np.mean(r)),
                                   "abstain": float(np.mean(a)),
                                   "worst_rate": float(np.mean(w)),
                                   "breach": float(np.mean(b))}
    out["kappa"] = res


def run_delta(track, args, out):
    """Sweep the certifier's error budget; the ledger has none to sweep.

    A reader is right to ask what the breach rates mean: whether the
    certificate's failures are an artefact of an error budget set too
    generously, and whether the ledger's zeros are slack. Delta answers the
    first question quantitatively. Shrinking it widens the confidence
    correction, which is the only lever the certificate has, so if its
    breaches came from an insufficient correction they would disappear. The
    sweep is run under an exchangeable order and under the two that break
    exchangeability, and the ledger is shown on the same rows because
    Theorem 1 contains no delta to sweep.
    """
    Lp, Cp, wr, wc, names, audit_unit = prep(track, args)
    pop_r, pop_c = Lp.mean(axis=1), Cp.mean(axis=1)
    ab = args.abstain_mult * float(pop_c.max())
    N = Lp.shape[1]
    print(f"\n{'='*92}\n{track} / certifier error budget sweep   "
          f"|plans|={len(names)} N={N}\n{'='*92}")
    res = {}
    for alpha in (args.alphas or TRACK_CFG[track][2]):
        opt, _ = solve_action_lp(pop_r, pop_c, alpha, ab)
        print(f"\n  alpha={alpha:.2f}  oracle {opt*1000:.4f} mCNY")
        for kind in args.delta_orders:
            print(f"   order={kind}")
            print(f"    {'method':<14} {'delta':>7} {'mean cost':>10} "
                  f"{'vs oracle':>10} {'risk':>7} {'abst%':>7} "
                  f"{'worst rate':>11} {'breach':>7}")

            def sweep(label, delta, run):
                c, r, a, w, b = [], [], [], [], []
                for d in range(args.draws):
                    rng = np.random.default_rng(args.seed + d)
                    order = make_order(kind, Lp, Cp, rng, wr)
                    cost, risk, ab_rate, cum = run(order, rng)
                    wo = worst_prefix_rate(cum, args.warm)
                    c.append(cost)
                    r.append(risk)
                    a.append(100 * ab_rate)
                    w.append(wo)
                    b.append(wo > alpha)
                mc = float(np.mean(c))
                dtxt = f"{delta:7.4f}" if delta else f"{'n/a':>7}"
                print(f"    {label:<14} {dtxt} {mc*1000:10.4f} "
                      f"{mc/max(opt,1e-12):9.2f}x {np.mean(r):7.3f} "
                      f"{np.mean(a):6.1f}% {np.mean(w):11.3f} "
                      f"{np.mean(b):7.2f}")
                key = f"{alpha}|{kind}|{delta if delta else 'none'}|{label}"
                res[key] = {
                    "alpha": alpha, "order": kind, "delta": delta,
                    "method": label, "cost": mc,
                    "vs_oracle": mc / max(opt, 1e-12),
                    "risk": float(np.mean(r)), "abstain": float(np.mean(a)),
                    "worst_rate": float(np.mean(w)),
                    "breach": float(np.mean(b)), "oracle": opt}

            for dl in args.deltas:
                def run_static(order, rng, dl=dl):
                    s = static_stream_cost(Lp, Cp, alpha, dl, order,
                                           args.n_cal, wr, wc,
                                           abstain_cost=ab,
                                           audit_unit_cost=audit_unit)
                    return (s["mean_cost"], s["risk"], s["abstain_rate"],
                            s["cum_viol"])
                sweep("certificate", dl, run_static)

            def run_ledger(order, rng):
                ex = LedgerExecutor(Lp, Cp, alpha, abstain_cost=ab,
                                    hist_risk=wr, hist_cost=wc,
                                    kappa=args.kappa,
                                    audit_unit_cost=audit_unit)
                st = ex.run(order, rng, log_every=10 ** 9)
                return (st.total_cost / st.t, st.risk, st.abstain_rate,
                        st.cum_viol)
            sweep("ledger", None, run_ledger)
    out["delta"] = res


def run_audit(track, args, out):
    """Price of not labelling every served query.

    'worst-case' charges an unlabelled arrival a full violation and therefore
    keeps V_t <= alpha t exactly; 'ipw' charges it zero and rescales, which is
    unbiased but only bounds the ledger in probability. The comparison prices
    the difference between a guarantee that holds and one that usually holds.
    """
    Lp, Cp, wr, wc, names, audit_unit = prep(track, args)
    pop_r, pop_c = Lp.mean(axis=1), Cp.mean(axis=1)
    ab = args.abstain_mult * float(pop_c.max())
    N = Lp.shape[1]
    print(f"\n{'='*92}\n{track} / audit rate and audit mode\n{'='*92}")
    res = {}
    for alpha in (args.alphas or TRACK_CFG[track][2]):
        opt, _ = solve_action_lp(pop_r, pop_c, alpha, ab)
        print(f"\n  alpha={alpha:.2f}  oracle {opt*1000:.4f} mCNY")
        print(f"  {'mode':<11} {'rate':>6} {'mean cost':>10} {'audit$':>8} "
              f"{'risk':>7} {'abst%':>7} {'worst rate':>11} {'breach':>7}")
        for mode in ("worst-case", "ipw"):
            for rate in args.audit_rates:
                c, r, a, w, b, au = [], [], [], [], [], []
                for d in range(args.draws):
                    rng = np.random.default_rng(args.seed + d)
                    order = rng.permutation(N)
                    ex = LedgerExecutor(Lp, Cp, alpha, abstain_cost=ab,
                                        hist_risk=wr, hist_cost=wc,
                                        kappa=args.kappa, audit_rate=rate,
                                        audit_mode=mode,
                                        audit_unit_cost=audit_unit)
                    st = ex.run(order, rng, log_every=10 ** 9)
                    wo = worst_prefix_rate(st.cum_viol, args.warm)
                    c.append(st.total_cost / st.t)
                    au.append(st.audit_cost / st.t)
                    r.append(st.risk)
                    a.append(100 * st.abstain_rate)
                    w.append(wo)
                    b.append(wo > alpha)
                mc = float(np.mean(c))
                print(f"  {mode:<11} {rate:6.2f} {mc*1000:10.4f} "
                      f"{np.mean(au)*1000:8.4f} {np.mean(r):7.3f} "
                      f"{np.mean(a):6.1f}% {np.mean(w):11.3f} "
                      f"{np.mean(b):7.2f}")
                res[f"{alpha}|{mode}|{rate}"] = {
                    "alpha": alpha, "mode": mode, "rate": rate, "cost": mc,
                    "audit_cost": float(np.mean(au)),
                    "risk": float(np.mean(r)), "abstain": float(np.mean(a)),
                    "worst_rate": float(np.mean(w)),
                    "breach": float(np.mean(b)), "oracle": opt}
    out["audit"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--parts", nargs="*",
                    default=["orders", "regret", "kappa", "audit", "delta"])
    ap.add_argument("--alphas", type=float, nargs="*", default=None)
    ap.add_argument("--orders", nargs="*",
                    default=["iid", "hard-first", "easy-first", "drift",
                             "adversarial"])
    ap.add_argument("--horizons", type=int, nargs="*",
                    default=[500, 1000, 2000, 4000, 8000, 16000, 32000])
    ap.add_argument("--kappas", type=float, nargs="*",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--audit_rates", type=float, nargs="*",
                    default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--deltas", type=float, nargs="*",
                    default=[0.2, 0.1, 0.05, 0.01, 0.001])
    ap.add_argument("--delta_orders", nargs="*",
                    default=["iid", "easy-first", "drift"])
    ap.add_argument("--abstain_mult", type=float, default=1.0)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--n_cal", type=int, default=500)
    ap.add_argument("--max_plans", type=int, default=16)
    ap.add_argument("--max_reps", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=200)
    args = ap.parse_args()
    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments",
        f"ledger_stress_{args.track}.json")
    # keep the parts we are not rerunning, so --parts is additive rather than
    # silently discarding hours of earlier work
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out = json.load(f)
    out.update({"track": args.track, "abstain_mult": args.abstain_mult,
                "kappa": args.kappa, "draws": args.draws})
    fns = {"orders": run_orders, "regret": run_regret, "kappa": run_kappa,
           "audit": run_audit, "delta": run_delta}
    for p in args.parts:
        fns[p](args.track, args, out)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
