"""Group-conditional contracts on CRAG (RQ5).

Certifies (a) marginal and (b) group-conditional policies on the cal split,
then reports per-dynamism-group realized violation and cost on the test
split. Also repeats over random cal/test resamples for violation rates.
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
                                   certify_ladder_groupwise, stop_rung)  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


def realized(ld, groups, thresholds_by_group, marginal_thr):
    idx = np.arange(ld.n)
    cum = np.cumsum(ld.costs, axis=0)
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        sub_scores = ld.scores[:, m]
        # marginal policy
        st_m = stop_rung(sub_scores, marginal_thr)
        # group policy
        pg = thresholds_by_group.get(g)
        st_g = stop_rung(sub_scores, pg)
        li = np.where(m)[0]
        out[g] = {
            "n": int(m.sum()),
            "marginal_risk": float(ld.losses[st_m, li].mean()),
            "marginal_cost": float(cum[st_m, li].mean()),
            "group_risk": float(ld.losses[st_g, li].mean()),
            "group_cost": float(cum[st_g, li].mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", default="correct")
    ap.add_argument("--alpha", type=float, default=0.65)
    ap.add_argument("--group_alphas", default=None,
                    help="e.g. static=0.55,slow-changing=0.65,"
                         "fast-changing=0.9,real-time=0.85")
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=200)
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

    if args.group_alphas:
        galphas = {}
        for part in args.group_alphas.split(","):
            k, v = part.split("=")
            galphas[k] = float(v)
    else:
        galphas = None

    if galphas:
        # group-wise contract; marginal baseline certifies at the mixture level
        w = {g: float((g_cal == g).mean()) for g in set(g_cal.tolist())}
        alpha_mix = sum(w[g] * galphas[g] for g in w)
        alpha_m = alpha_mix
        alpha_g = galphas
    else:
        alpha_m = args.alpha
        alpha_g = args.alpha

    def a_of(g):
        return galphas[g] if galphas else args.alpha

    pol_m = certify_ladder(ld_cal, alpha_m, args.delta)
    pols_g = certify_ladder_groupwise(ld_cal, g_cal, alpha_g, args.delta)
    thr_g = {g: p.thresholds for g, p in pols_g.items()}
    single = realized(ld_te, g_te, thr_g, pol_m.thresholds)
    for g in single:
        single[g]["alpha"] = a_of(g)
    print(json.dumps(single, indent=1))

    # repeated resampling: per-group violation probability
    pool_L = np.concatenate([ld_cal.losses, ld_te.losses], axis=1)
    pool_S = np.concatenate([ld_cal.scores, ld_te.scores], axis=1)
    pool_C = np.concatenate([ld_cal.costs, ld_te.costs], axis=1)
    pool_G = np.concatenate([g_cal, g_te])
    n = pool_L.shape[1]
    n_cal = ld_cal.n
    rng = np.random.default_rng(0)
    groups_list = sorted(set(pool_G.tolist()))
    viol = {g: {"marginal": 0, "group": 0} for g in groups_list}
    for rep in range(args.repeats):
        perm = rng.permutation(n)
        ci, ti = perm[:n_cal], perm[n_cal:]
        ldc = LadderData(pool_L[:, ci], pool_S[:, ci], pool_C[:, ci])
        ldt = LadderData(pool_L[:, ti], pool_S[:, ti], pool_C[:, ti])
        gc, gt = pool_G[ci], pool_G[ti]
        pm = certify_ladder(ldc, alpha_m, args.delta)
        pg = certify_ladder_groupwise(ldc, gc, alpha_g, args.delta)
        for g in groups_list:
            m = gt == g
            li = np.where(m)[0]
            st_m = stop_rung(ldt.scores[:, m], pm.thresholds)
            if ldt.losses[st_m, li].mean() > a_of(g):
                viol[g]["marginal"] += 1
            st_g = stop_rung(ldt.scores[:, m],
                             pg.get(g, pm).thresholds)
            if ldt.losses[st_g, li].mean() > a_of(g):
                viol[g]["group"] += 1
    for g in groups_list:
        for k in viol[g]:
            viol[g][k] /= args.repeats

    out = {"alpha": args.alpha, "group_alphas": galphas, "delta": args.delta,
           "contract": args.contract,
           "single_split": single, "violation_rates": viol,
           "group_certified": {g: bool(p.certified) for g, p in pols_g.items()}}
    tag = "ga" if galphas else f"a{args.alpha}"
    path = os.path.join(EXP, f"group_crag_{args.contract}_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(viol, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
