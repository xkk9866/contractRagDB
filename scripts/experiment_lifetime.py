"""Lifetime (multi-epoch) monitoring with geometric error spending
(reviewer fix: the drift closed loop covered only one alarm/recalibration).

Budget schedule: the deployment lifetime is split into epochs by alarms.
Epoch k >= 0 gets budget delta_k = delta / 2^{k+1}, split evenly between its
monitor (delta_k / 2) and its recalibration (delta_k / 2); the initial
certification uses delta_0^cert. Total error over ANY number of epochs is
sum_k delta_k <= delta. The k-th monitor's threshold grows to
log(2^{k+2}/delta), so detection delay grows only ADDITIVELY by
(k+1) log 2 / g* (Proposition lifetime-validity).

Stream (real CRAG losses, designed arrival order), four phases:
  static -> real-time (drift 1) -> static again (recovery) -> fast (drift 2).
Epoch transitions: every recalibration attempt -- alarm-driven or the
scheduled cost-recheck that fires when the deployed policy is expensive and
the monitor has been silent -- consumes the next slice of the schedule.
After an alarm: safe mode (strongest rung) for a W-query labeled window,
then fixed-sequence recertification ON that window. If the window certifies
nothing (contract infeasible on the new distribution), the system stays in
FLAGGED safe mode with the monitor re-armed -- reported, not hidden; the
safe rung carries no post-drift certificate.

Pure numpy over materialized matrices; no LLM calls.
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
                                   EProcessMonitor)  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--window", type=int, default=400)
    ap.add_argument("--quiet", type=int, default=400,
                    help="silent steps before a scheduled cost-recheck")
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    loss_fn = make_loss_fn("crag", "correct", 0.5)
    kgc = kg_costs_for("crag")
    splits = get_splits("crag")
    dyn = {q["qid"]: q["static_or_dynamic"] for s in splits.values() for q in s}

    data = {}
    for s in ["train", "cal", "test"]:
        td = TrackData("crag", s, loss_fn, kg_costs=kgc)
        ld, qids, groups, aux = td.build(groups_map=dyn)
        data[s] = {"ld": ld, "groups": groups, "aux": aux}
    scorer = RungScorer().fit(data["train"]["ld"],
                              data["train"]["aux"]["raw_feats"])
    for s in data:
        data[s]["ld"].scores = scorer.score(data[s]["aux"]["raw_feats"])

    ld_cal = data["cal"]["ld"]
    ld_te, g_te = data["test"]["ld"], data["test"]["groups"]
    L = ld_te.L
    cum_cost = np.cumsum(ld_te.costs, axis=0)
    SAFE = np.full(L - 1, np.inf)

    pools = {
        "static": np.where(np.isin(g_te, ["static", "slow-changing"]))[0],
        "fast": np.where(g_te == "fast-changing")[0],
        "rt": np.where(g_te == "real-time")[0],
    }
    for name, p in pools.items():
        per_rung = ld_te.losses[:, p].mean(axis=1)
        print(f"pool {name:>7s} (n={len(p)}): per-rung risk "
              f"{np.round(per_rung, 3)}")

    # epoch budgets: delta_k = delta / 2^{k+1}, halved between monitor/recal
    def budgets(k):
        d_k = args.delta / (2 ** (k + 1))
        return d_k / 2, d_k / 2

    d_cert0, d_mon0 = budgets(0)
    pol0 = certify_ladder(ld_cal, args.alpha, d_cert0)
    print(f"initial policy q={pol0.q:.2f} certified={pol0.certified} "
          f"cal_cost={pol0.cal_cost*1000:.2f}m (budget {d_cert0})")

    # four phases: static -> rt (drift 1) -> static (recovery) -> fast (drift 2)
    P = [1200, 1200, 1200, 1200]
    PHASE_POOLS = ["static", "rt", "static", "fast"]

    def make_stream(rng):
        return np.concatenate([rng.choice(pools[p], n, replace=True)
                               for p, n in zip(PHASE_POOLS, P)])

    def step(i, thr):
        stop = stop_rung(ld_te.scores[:, i:i + 1], thr)[0]
        return float(ld_te.losses[stop, i]), float(cum_cost[stop, i])

    def certify_window(buf, d_cert):
        sub = LadderData(ld_te.losses[:, buf], ld_te.scores[:, buf],
                         ld_te.costs[:, buf])
        return certify_ladder(sub, args.alpha, d_cert)

    def run(seed):
        rng = np.random.default_rng(4000 + seed)
        st = make_stream(rng)
        epoch = 0
        _, d_mon = budgets(0)
        mon = EProcessMonitor(args.alpha, d_mon)
        thr, flagged = pol0.thresholds, False
        in_safe, buf = False, []
        quiet, recent = 0, []
        events, losses, costs = [], [], []
        for t, i in enumerate(st):
            use = SAFE if in_safe else thr
            l, c = step(i, use)
            losses.append(l)
            costs.append(c)
            recent.append(i)
            recent = recent[-args.window:]
            if in_safe:
                buf.append(i)
                if len(buf) >= args.window:
                    d_cert, d_mon = budgets(epoch)
                    pol = certify_window(buf, d_cert)
                    events.append({"t": t, "event": "recal", "epoch": epoch,
                                   "budget": d_cert,
                                   "certified": bool(pol.certified),
                                   "q": float(pol.q)})
                    if pol.certified:
                        thr, flagged = pol.thresholds, False
                    else:
                        # honest: flagged safe mode (no post-drift certificate);
                        # monitor re-armed, scheduled recheck will retry
                        thr, flagged = SAFE, True
                    in_safe, buf, quiet = False, [], 0
                    mon = EProcessMonitor(args.alpha, d_mon)
                continue
            if mon.update(l):
                epoch += 1
                events.append({"t": t, "event": "alarm", "epoch": epoch,
                               "budget": mon.delta})
                in_safe, buf, quiet = True, [], 0
                continue
            quiet += 1
            expensive = (np.mean(costs[-args.quiet:])
                         > 1.5 * pol0.cal_cost)
            if quiet >= args.quiet and (expensive or flagged) \
                    and len(recent) >= args.window:
                # scheduled cost-recheck / flagged retry: consumes the next
                # slice of the spending schedule
                epoch += 1
                d_cert, d_mon = budgets(epoch)
                pol = certify_window(recent, d_cert)
                events.append({"t": t, "event": "recheck", "epoch": epoch,
                               "budget": d_cert,
                               "certified": bool(pol.certified),
                               "q": float(pol.q)})
                if pol.certified:
                    thr, flagged = pol.thresholds, False
                mon = EProcessMonitor(args.alpha, d_mon)
                quiet = 0
        return {"events": events,
                "viol": [float(np.mean(losses[sum(P[:k]):sum(P[:k + 1])]))
                         for k in range(len(P))],
                "cost_by_phase": [float(np.mean(costs[sum(P[:k]):
                                                      sum(P[:k + 1])]))
                                  for k in range(len(P))],
                "cost_mean": float(np.mean(costs))}

    runs = [run(s) for s in range(args.seeds)]
    # aggregate alarm delays per drift
    d1 = [e["t"] - P[0] for r in runs for e in r["events"]
          if e["event"] == "alarm" and P[0] <= e["t"] < P[0] + P[1]]
    d2 = [e["t"] - sum(P[:3]) for r in runs for e in r["events"]
          if e["event"] == "alarm" and e["t"] >= sum(P[:3])]
    recals = [e for r in runs for e in r["events"]
              if e["event"] in ("recal", "recheck")]
    cert_frac = (np.mean([e["certified"] for e in recals])
                 if recals else None)
    n_alarms = [sum(1 for e in r["events"] if e["event"] == "alarm")
                for r in runs]

    # false-alarm validity over the FULL lifetime schedule on no-drift streams
    false_alarms, n_null = 0, 100
    for seed in range(n_null):
        rng = np.random.default_rng(8000 + seed)
        st = rng.choice(pools["static"], sum(P), replace=True)
        _, d_mon = budgets(0)
        mon = EProcessMonitor(args.alpha, d_mon)
        for i in st:
            l, _ = step(i, pol0.thresholds)
            if mon.update(l):
                false_alarms += 1
                break

    out = {"alpha": args.alpha, "delta": args.delta, "window": args.window,
           "quiet": args.quiet, "phase_lengths": P, "phase_pools": PHASE_POOLS,
           "seeds": args.seeds,
           "pool_risks": {k: ld_te.losses[:, p].mean(axis=1).tolist()
                          for k, p in pools.items()},
           "initial": {"q": float(pol0.q), "certified": bool(pol0.certified),
                       "budget": d_cert0},
           "alarms_per_run_mean": float(np.mean(n_alarms)),
           "delay_drift1": {"n": len(d1),
                            "median": float(np.median(d1)) if d1 else None,
                            "mean": float(np.mean(d1)) if d1 else None},
           "delay_drift2": {"n": len(d2),
                            "median": float(np.median(d2)) if d2 else None,
                            "mean": float(np.mean(d2)) if d2 else None},
           "recal_certified_frac": cert_frac,
           "false_alarms": false_alarms, "n_null": n_null,
           "viol_by_phase": [float(np.mean([r["viol"][k] for r in runs]))
                             for k in range(len(P))],
           "cost_by_phase": [float(np.mean([r["cost_by_phase"][k]
                                            for r in runs]))
                             for k in range(len(P))],
           "static_viol_by_phase": None,
           "runs": runs}

    # static baseline (never adapts) for the same streams
    static_v = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(4000 + seed)
        st = make_stream(rng)
        ls = [step(i, pol0.thresholds)[0] for i in st]
        static_v.append([float(np.mean(ls[sum(P[:k]):sum(P[:k + 1])]))
                         for k in range(len(P))])
    out["static_viol_by_phase"] = [float(np.mean([v[k] for v in static_v]))
                                   for k in range(len(P))]

    print(f"alarms/run {out['alarms_per_run_mean']:.1f}; "
          f"delay1 med {out['delay_drift1']['median']} "
          f"delay2 med {out['delay_drift2']['median']}; "
          f"recal certified {cert_frac}; "
          f"false alarms {false_alarms}/{n_null}")
    print("viol by phase (lifetime loop):", np.round(out["viol_by_phase"], 3))
    print("viol by phase (static):      ",
          np.round(out["static_viol_by_phase"], 3))

    path = os.path.join(EXP, f"lifetime_crag_a{args.alpha}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
