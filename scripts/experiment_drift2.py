"""Drift 2.0: mixture-restart e-detector as a closed-loop reoptimization trigger.

Beyond experiment_drift.py (single drift, fixed post-alarm policy), this adds:
  1. Delay-vs-magnitude curves: post-change violation rate is controlled by
     mixing pools; empirically validates delay ~ O(log(T|W|/delta)/g*)
     (Prop. detection-delay) where g* = post-change excess risk.
  2. Changepoint localization: |tau_hat - tau| from the O(1) CUSUM recursion.
  3. Validity: false-alarm rate over drift-free streams <= delta.
  4. Certified recalibration loop: alarm -> safe mode (final rung) for W
     labeled steps -> re-run fixed-sequence LTT on that window with a fresh
     delta budget -> resume with the newly certified policy. Compares against
     static, oracle switch, and DB-style periodic recheck.

All losses/costs are real (CRAG ladder matrices); only arrival order is
designed, as is standard in change-detection studies.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import (LadderData, certify_ladder,
                                   certify_ladder_groupwise, stop_rung,
                                   EProcessMonitor)  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def build_stream(rng, static_idx, dyn_idx, P1, P2, rho):
    """P1 pre-change draws from the static pool, then P2 post-change draws
    where each is dynamic w.p. rho (drift magnitude knob)."""
    pre = rng.choice(static_idx, P1, replace=True)
    take_dyn = rng.random(P2) < rho
    post = np.where(take_dyn, rng.choice(dyn_idx, P2, replace=True),
                    rng.choice(static_idx, P2, replace=True))
    return np.concatenate([pre, post])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="hallu")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--recal-window", type=int, default=250)
    args = ap.parse_args()

    loss_fn = make_loss_fn("crag", args.contract, 0.5)
    kgc = kg_costs_for("crag")
    splits = get_splits("crag")
    dyn = {q["qid"]: q["static_or_dynamic"] for s in splits.values() for q in s}

    data = {}
    for s in ["train", "cal", "test"]:
        td = TrackData("crag", s, loss_fn, kg_costs=kgc)
        ld, qids, groups, aux = td.build(groups_map=dyn)
        data[s] = {"ld": ld, "qids": qids, "groups": groups, "aux": aux}
    scorer = RungScorer().fit(data["train"]["ld"], data["train"]["aux"]["raw_feats"])
    for s in data:
        data[s]["ld"].scores = scorer.score(data[s]["aux"]["raw_feats"])

    ld_cal, g_cal = data["cal"]["ld"], data["cal"]["groups"]
    ld_te, g_te = data["test"]["ld"], data["test"]["groups"]
    L = ld_te.L
    cum_cost = np.cumsum(ld_te.costs, axis=0)

    pol_marg = certify_ladder(ld_cal, args.alpha, args.delta)
    pol_group = certify_ladder_groupwise(ld_cal, g_cal, args.alpha, args.delta)
    print(f"marginal q={pol_marg.q:.2f} cal_risk={pol_marg.cal_risk:.3f}")

    static_idx = np.where(np.isin(g_te, ["static", "slow-changing"]))[0]
    dyn_idx = np.where(np.isin(g_te, ["fast-changing", "real-time"]))[0]
    P1, P2 = 1500, 1500

    def step(i, thr):
        stop = stop_rung(ld_te.scores[:, i:i + 1], thr)[0]
        return float(ld_te.losses[stop, i]), float(cum_cost[stop, i])

    # ---- 1+2. delay & changepoint vs drift magnitude ----------------------
    SAFE_THR = np.full(L - 1, np.inf)          # always escalate to final rung

    def growth_rate(p_bar, alpha, bets=(0.1, 0.3, 0.5, 0.7, 0.9)):
        """g* of Prop. [detection delay]: best per-step log-growth over the
        monitor's bet grid, for Bernoulli losses with post-change mean p_bar."""
        best = 0.0
        for b in bets:
            w = b / alpha
            g = (p_bar * np.log(1 + w * (1 - alpha))
                 + (1 - p_bar) * np.log(max(1e-12, 1 - w * alpha)))
            best = max(best, g)
        return best

    delay_curve = []
    for rho in [0.25, 0.5, 0.75, 1.0]:
        delays, cps, post_risks = [], [], []
        for seed in range(30):
            rng = np.random.default_rng(1000 + seed)
            st = build_stream(rng, static_idx, dyn_idx, P1, P2, rho)
            mon = EProcessMonitor(args.alpha, args.delta)
            alarm = None
            losses = []
            for t, i in enumerate(st):
                l, _ = step(i, pol_marg.thresholds)
                losses.append(l)
                if alarm is None and mon.update(l):
                    alarm = t
                    break
            post_risks.append(float(np.mean(losses[P1:])) if len(losses) > P1
                              else None)
            if alarm is not None and alarm >= P1:
                delays.append(alarm - P1)
                cps.append(abs((mon.changepoint_hat or 0) - 1 - P1))
        pr = [r for r in post_risks if r is not None]
        # post-change mean loss from the policy's per-query losses on the
        # dynamic mixture (uncensored estimate: evaluate on the full pools)
        loss_static = np.array([step(i, pol_marg.thresholds)[0]
                                for i in static_idx])
        loss_dyn = np.array([step(i, pol_marg.thresholds)[0] for i in dyn_idx])
        p_bar = float(rho * loss_dyn.mean() + (1 - rho) * loss_static.mean())
        g_star = growth_rate(p_bar, args.alpha) if p_bar > args.alpha else None
        delay_curve.append({
            "rho": rho, "n_detected": len(delays),
            "delay_mean": float(np.mean(delays)) if delays else None,
            "delay_p90": float(np.percentile(delays, 90)) if delays else None,
            "cp_err_mean": float(np.mean(cps)) if cps else None,
            "cp_err_median": float(np.median(cps)) if cps else None,
            "post_mean_loss": p_bar, "excess_risk": p_bar - args.alpha,
            "g_star": g_star,
            "delay_x_gstar": (float(np.mean(delays)) * g_star
                              if delays and g_star else None)})
        d = delay_curve[-1]
        print(f"rho={rho:.2f} p_bar={p_bar:.3f} g*={d['g_star']} "
              f"delay={d['delay_mean']} cp_err={d['cp_err_median']} "
              f"delay*g*={d['delay_x_gstar']}")
    # theoretical alarm threshold constant: log(1/delta) + log(T|W|)
    thr_const = float(np.log(1 / args.delta) + np.log(100000 * 5))
    print(f"bound constant log(1/delta)+log(T|W|) = {thr_const:.1f}")

    # ---- 3. validity: false alarms on drift-free streams ------------------
    false_alarms, n_null = 0, 200
    for seed in range(n_null):
        rng = np.random.default_rng(5000 + seed)
        st = rng.choice(static_idx, P1 + P2, replace=True)
        mon = EProcessMonitor(args.alpha, args.delta)
        for i in st:
            l, _ = step(i, pol_marg.thresholds)
            if mon.update(l):
                false_alarms += 1
                break
    print(f"false alarms: {false_alarms}/{n_null} (delta={args.delta})")

    # ---- 4. closed-loop certified recalibration ----------------------------
    W = args.recal_window

    def run_policy(mode, seed, rho=1.0, recheck_every=250):
        rng = np.random.default_rng(9000 + seed)
        st = build_stream(rng, static_idx, dyn_idx, P1, P2, rho)
        mon = EProcessMonitor(args.alpha, args.delta / 2)
        alarm_t, recal_pol, safe_buf = None, None, []
        losses, costs = [], []
        for t, i in enumerate(st):
            if mode == "oracle":
                p = pol_group.get(g_te[i], pol_marg) if t >= P1 else pol_marg
                thr = p.thresholds if p.certified else SAFE_THR
            elif mode == "monitor_recal":
                if alarm_t is None:
                    thr = pol_marg.thresholds
                elif recal_pol is None:
                    thr = SAFE_THR          # safe mode while labeling window
                else:
                    thr = recal_pol.thresholds
            elif mode == "periodic":
                # DB-style scheduled reoptimizer: every recheck_every steps,
                # recertify on the last W observed queries (labels assumed).
                if t >= recheck_every and t % recheck_every == 0:
                    lo = max(0, t - W)
                    sub = LadderData(ld_te.losses[:, st[lo:t]],
                                     ld_te.scores[:, st[lo:t]],
                                     ld_te.costs[:, st[lo:t]])
                    recal_pol = certify_ladder(sub, args.alpha, args.delta)
                thr = (recal_pol.thresholds if recal_pol is not None
                       else pol_marg.thresholds)
            else:                            # static
                thr = pol_marg.thresholds
            l, c = step(i, thr)
            losses.append(l)
            costs.append(c)
            if mode == "monitor_recal":
                if alarm_t is None:
                    if mon.update(l):
                        alarm_t = t
                elif recal_pol is None:
                    safe_buf.append(i)
                    if len(safe_buf) >= W:
                        sub = LadderData(ld_te.losses[:, safe_buf],
                                         ld_te.scores[:, safe_buf],
                                         ld_te.costs[:, safe_buf])
                        recal_pol = certify_ladder(sub, args.alpha,
                                                   args.delta / 2)
        return {"mode": mode, "alarm_t": alarm_t,
                "recal_certified": bool(recal_pol.certified) if recal_pol else None,
                "recal_q": float(recal_pol.q) if recal_pol else None,
                "viol_phase1": float(np.mean(losses[:P1])),
                "viol_phase2": float(np.mean(losses[P1:])),
                "cost_mean": float(np.mean(costs))}

    runs = {}
    for mode in ["static", "monitor_recal", "oracle", "periodic"]:
        rs = [run_policy(mode, s) for s in range(10)]
        runs[mode] = {
            "viol_phase1": float(np.mean([r["viol_phase1"] for r in rs])),
            "viol_phase2": float(np.mean([r["viol_phase2"] for r in rs])),
            "cost_mean": float(np.mean([r["cost_mean"] for r in rs])),
            "alarm_delay_mean": float(np.mean(
                [r["alarm_t"] - P1 for r in rs if r["alarm_t"] is not None]))
            if any(r["alarm_t"] is not None for r in rs) else None,
            "recal_certified_frac": float(np.mean(
                [r["recal_certified"] for r in rs
                 if r["recal_certified"] is not None]))
            if any(r["recal_certified"] is not None for r in rs) else None,
            "runs": rs}
        r = runs[mode]
        print(f"{mode:>14s}: ph1 {r['viol_phase1']:.3f} ph2 {r['viol_phase2']:.3f} "
              f"cost {r['cost_mean']*1000:.2f}m alarm_delay {r['alarm_delay_mean']}")

    out = {"alpha": args.alpha, "delta": args.delta, "contract": args.contract,
           "recal_window": W, "delay_curve": delay_curve,
           "bound_constant": float(np.log(1 / args.delta) + np.log(100000 * 5)),
           "false_alarms": false_alarms, "n_null_streams": n_null,
           "closed_loop": runs}
    path = os.path.join(EXP, f"drift2_crag_{args.contract}_a{args.alpha}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
