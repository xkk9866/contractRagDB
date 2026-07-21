"""Online contract monitoring under workload drift (Track B, real losses).

Stream construction (real queries, real losses; only arrival order is
designed, as is standard in drift studies):
  Phase 1 (t=1..T1): queries sampled from {static, slow-changing} test pool.
  Phase 2 (t>T1):    queries sampled from {fast-changing, real-time} pool.

Policies compared on the SAME stream:
  A. static-certified ladder (calibrated marginally on cal split), never adapts;
  B. A + e-process monitor: on alarm, switch to group-conditional certified
     policies (recalibrated on cal split per dynamism group);
  C. group-conditional from the start (skyline for adaptation).

Reports violation rate per phase, alarm time, cost; plus periodic-recheck
baseline (recalibrate every K queries — the DB-style scheduled reoptimizer).
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import (certify_ladder, certify_ladder_groupwise,
                                   stop_rung, EProcessMonitor)  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="hallu")
    ap.add_argument("--alpha", type=float, default=0.15)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
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

    # policies
    pol_marg = certify_ladder(ld_cal, args.alpha, args.delta)
    pol_group = certify_ladder_groupwise(ld_cal, g_cal, args.alpha, args.delta)
    print("marginal policy q:", pol_marg.q, "cal risk:", round(pol_marg.cal_risk, 3))
    for g, p in pol_group.items():
        print(f"  group {g}: q={p.q:.2f} cal_risk={p.cal_risk:.3f} certified={p.certified}")

    # stream: phase1 sampled from static/slow pool, phase2 from fast/real-time
    # (real queries and losses; arrival order and multiplicity are designed,
    # standard in change-detection studies)
    rng = np.random.default_rng(args.seed)
    static_idx = np.where(np.isin(g_te, ["static", "slow-changing"]))[0]
    dyn_idx = np.where(np.isin(g_te, ["fast-changing", "real-time"]))[0]
    P1, P2 = 1500, 1500
    stream = np.concatenate([rng.choice(static_idx, P1, replace=True),
                             rng.choice(dyn_idx, P2, replace=True)])
    T1 = P1
    print(f"stream: {T1} static-ish then {P2} dynamic (pools "
          f"{len(static_idx)}/{len(dyn_idx)})")

    cum_cost = np.cumsum(ld_te.costs, axis=0)

    def group_thr(i):
        """Post-alarm adaptation: certified group policy where one exists,
        strongest rung (best effort + infeasibility flag) elsewhere."""
        p = pol_group.get(g_te[i])
        if p is not None and p.certified:
            return p.thresholds
        return np.zeros(ld_te.L - 1)  # threshold 0 at rung L-1... see below

    def run_policy(mode, recheck_every=None):
        mon = EProcessMonitor(args.alpha, args.delta)
        use_group = (mode == "group")
        alarm_t = None
        losses, costs = [], []
        strongest = np.full(ld_te.L - 1, np.inf)
        for t, i in enumerate(stream):
            if mode == "monitor" and alarm_t is not None:
                use = "group"
            elif mode == "periodic" and recheck_every and t >= recheck_every:
                use = "group"
            else:
                use = "group" if use_group else "marginal"
            if use == "group":
                p = pol_group.get(g_te[i], pol_marg)
                thr = p.thresholds if p.certified else strongest
            else:
                thr = pol_marg.thresholds
            stop = stop_rung(ld_te.scores[:, i:i + 1], thr)[0]
            l = float(ld_te.losses[stop, i])
            c = float(cum_cost[stop, i])
            losses.append(l)
            costs.append(c)
            if mode == "monitor" and alarm_t is None:
                if mon.update(l):
                    alarm_t = t
        return {"mode": mode, "alarm_t": alarm_t,
                "viol_phase1": float(np.mean(losses[:T1])),
                "viol_phase2": float(np.mean(losses[T1:])),
                "viol_overall": float(np.mean(losses)),
                "cost_mean": float(np.mean(costs)),
                "losses": losses, "T1": T1}

    out = {"alpha": args.alpha, "delta": args.delta, "contract": args.contract,
           "runs": {m: run_policy(m, recheck_every=200)
                    for m in ["marginal", "monitor", "group", "periodic"]}}
    # multi-seed alarm statistics + false-alarm check on drift-free streams
    alarm_delays, false_alarms = [], 0
    for seed in range(20):
        rng2 = np.random.default_rng(100 + seed)
        st = np.concatenate([rng2.choice(static_idx, P1, replace=True),
                             rng2.choice(dyn_idx, P2, replace=True)])
        mon = EProcessMonitor(args.alpha, args.delta)
        alarm = None
        for t, i in enumerate(st):
            stop = stop_rung(ld_te.scores[:, i:i + 1], pol_marg.thresholds)[0]
            if mon.update(float(ld_te.losses[stop, i])) and alarm is None:
                alarm = t
                break
        alarm_delays.append(None if alarm is None else alarm - P1)
        # drift-free control stream
        st0 = np.random.default_rng(500 + seed).choice(
            static_idx, P1 + P2, replace=True)
        mon0 = EProcessMonitor(args.alpha, args.delta)
        for i in st0:
            stop = stop_rung(ld_te.scores[:, i:i + 1], pol_marg.thresholds)[0]
            if mon0.update(float(ld_te.losses[stop, i])):
                false_alarms += 1
                break
    out["alarm_delays"] = alarm_delays
    out["false_alarms_of_20"] = false_alarms

    for m, r in out["runs"].items():
        print(f"{m:>9s}: phase1 {r['viol_phase1']:.3f} phase2 {r['viol_phase2']:.3f} "
              f"cost {r['cost_mean']*1000:.2f}mCNY alarm@{r['alarm_t']}")
    path = os.path.join(EXP, f"drift_crag_{args.contract}_a{args.alpha}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("saved", path)


if __name__ == "__main__":
    main()
