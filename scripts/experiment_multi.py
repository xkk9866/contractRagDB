"""Multi-constraint contracts (quality + latency SLO) via IUT fixed-sequence
testing. Also measures the power gain of IUT over Bonferroni splitting.

Contract: P(quality violation) <= alpha_q  AND  P(latency > B) <= alpha_l.
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
                                   certify_ladder_multi, stop_rung,
                                   hb_p_value, quantile_path)  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def eval_thr(ld, lats, lat_budget, thr):
    idx = np.arange(ld.n)
    stop = stop_rung(ld.scores, np.asarray(thr))
    cum_lat = np.cumsum(lats, axis=0)[stop, idx]
    return {
        "risk_q": float(ld.losses[stop, idx].mean()),
        "risk_lat": float((cum_lat > lat_budget).mean()),
        "cost_mean": float(np.cumsum(ld.costs, axis=0)[stop, idx].mean()),
        "lat_p95": float(np.percentile(cum_lat, 95)),
    }


def certify_multi_mode(ld, aux, alphas, delta, mode="iut", num_grid=201):
    """mode: iut (delta each) or bonferroni (delta/k each)."""
    k = 1 + len(aux)
    d_each = delta if mode == "iut" else delta / k
    idx = np.arange(ld.n)
    cum_cost = np.cumsum(ld.costs, axis=0)
    best = None
    for q, thr in quantile_path(ld, num_grid):
        stop = stop_rung(ld.scores, thr)
        ok = hb_p_value(float(ld.losses[stop, idx].mean()), ld.n,
                        alphas["primary"]) <= d_each
        if ok:
            for name, lmat in aux.items():
                if hb_p_value(float(lmat[stop, idx].mean()), ld.n,
                              alphas[name]) > d_each:
                    ok = False
                    break
        if ok:
            best = thr
        else:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track", nargs="?", default="asqa")
    ap.add_argument("--contract", default="citation")
    ap.add_argument("--tau", type=float, default=50)
    ap.add_argument("--alpha_q", type=float, default=0.3)
    ap.add_argument("--alpha_l", type=float, default=0.15)
    ap.add_argument("--lat_budget", type=float, default=10.0)
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()

    loss_fn = make_loss_fn(args.track, args.contract, args.tau)
    kgc = kg_costs_for(args.track)
    data = {}
    for s in ["train", "cal", "test"]:
        td = TrackData(args.track, s, loss_fn, kg_costs=kgc)
        ld, qids, _, aux = td.build()
        data[s] = {"ld": ld, "aux": aux}
    scorer = RungScorer().fit(data["train"]["ld"], data["train"]["aux"]["raw_feats"])
    for s in data:
        data[s]["ld"].scores = scorer.score(data[s]["aux"]["raw_feats"])

    ld_cal, lat_cal = data["cal"]["ld"], data["cal"]["aux"]["latency"]
    ld_te, lat_te = data["test"]["ld"], data["test"]["aux"]["latency"]

    def lat_loss(ld, lats):
        L, n = lats.shape
        cum = np.cumsum(lats, axis=0)
        return (cum > args.lat_budget).astype(float)

    aux_cal = {"latency": lat_loss(ld_cal, lat_cal)}
    alphas = {"primary": args.alpha_q, "latency": args.alpha_l}

    out = {"track": args.track, "alpha_q": args.alpha_q,
           "alpha_l": args.alpha_l, "lat_budget": args.lat_budget,
           "delta": args.delta, "policies": {}}

    # quality-only certification (ignores latency)
    pol_q = certify_ladder(ld_cal, args.alpha_q, args.delta)
    out["policies"]["quality_only"] = eval_thr(ld_te, lat_te, args.lat_budget,
                                               pol_q.thresholds)

    # multi via IUT and via Bonferroni (T-family only)
    for mode in ["iut", "bonferroni"]:
        thr = certify_multi_mode(ld_cal, aux_cal, alphas, args.delta, mode)
        if thr is None:
            out["policies"][mode] = None
            continue
        out["policies"][mode] = eval_thr(ld_te, lat_te, args.lat_budget, thr)

    # full optimizer with heterogeneous candidates (escalate + jump) under IUT
    from contractrag.optimizer import build_candidates, optimize, apply_candidate
    ld_tr, lat_tr = data["train"]["ld"], data["train"]["aux"]["latency"]
    cands = build_candidates(ld_tr)
    opt = optimize(cands, ld_cal, args.alpha_q, args.delta,
                   lat_cal=lat_cal, lat_budget=args.lat_budget,
                   alpha_lat=args.alpha_l, ld_train=ld_tr, lat_train=lat_tr)
    if opt.certified:
        loss_t, cost_t, lat_t, _ = apply_candidate(opt.candidate, ld_te, lat_te)
        out["policies"]["opt_multi"] = {
            "risk_q": float(loss_t.mean()),
            "risk_lat": float((lat_t > args.lat_budget).mean()),
            "cost_mean": float(cost_t.mean()),
            "lat_p95": float(np.percentile(lat_t, 95)),
            "selected": opt.candidate.describe(),
        }
    else:
        out["policies"]["opt_multi"] = None

    for name, m in out["policies"].items():
        if m:
            print(f"{name:>14s}: risk_q={m['risk_q']:.3f} "
                  f"risk_lat={m['risk_lat']:.3f} cost={m['cost_mean']*1000:.2f}mCNY "
                  f"p95={m['lat_p95']:.1f}s")
        else:
            print(f"{name:>14s}: infeasible")

    path = os.path.join(EXP, f"multi_{args.track}_{args.contract}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
