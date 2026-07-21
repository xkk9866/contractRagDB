"""Diagnose the 0/200 violation results: what does the certified policy
select on each repeat, how big is its safety margin, and is the zero
consistent with the HB test's conservatism?

Usage: python scripts/verify_zero_violation.py crag --contract correct --alpha 0.65
       python scripts/verify_zero_violation.py asqa --contract citation --tau 50 --alpha 0.25
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer
from contractrag.calibrate import LadderData
from contractrag.optimizer import build_candidates, optimize, apply_candidate
from scripts.experiment_main import make_loss_fn, kg_costs_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--contract", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--repeats", type=int, default=200)
    args = ap.parse_args()

    loss_fn = make_loss_fn(args.track, args.contract, args.tau)
    kgc = kg_costs_for(args.track)
    tr = TrackData(args.track, "train", loss_fn, kg_costs=kgc)
    ld_tr, _, _, aux_tr = tr.build()
    scorer = RungScorer().fit(ld_tr, aux_tr["raw_feats"])
    ld_tr.scores = scorer.score(aux_tr["raw_feats"])

    parts = []
    for s in ["cal", "test"]:
        td = TrackData(args.track, s, loss_fn, kg_costs=kgc)
        ld, _, _, aux = td.build()
        ld.scores = scorer.score(aux["raw_feats"])
        parts.append(ld)
    ld_pool = LadderData(
        losses=np.concatenate([p.losses for p in parts], axis=1),
        scores=np.concatenate([p.scores for p in parts], axis=1),
        costs=np.concatenate([p.costs for p in parts], axis=1))
    n = ld_pool.n
    n_cal = int(n * 0.4)
    cands = build_candidates(ld_tr)

    rng = np.random.default_rng(0)
    sel, cal_risks, test_risks, margins = Counter(), [], [], []
    for rep in range(args.repeats):
        perm = rng.permutation(n)
        ci, ti = perm[:n_cal], perm[n_cal:]
        ld_c = LadderData(losses=ld_pool.losses[:, ci],
                          scores=ld_pool.scores[:, ci],
                          costs=ld_pool.costs[:, ci])
        ld_t = LadderData(losses=ld_pool.losses[:, ti],
                          scores=ld_pool.scores[:, ti],
                          costs=ld_pool.costs[:, ti])
        opt = optimize(cands, ld_c, args.alpha, args.delta)
        loss_t, _, _, _ = apply_candidate(opt.candidate, ld_t)
        rt = float(loss_t.mean())
        sel[opt.candidate.describe() + ("" if opt.certified else " [UNCERT]")] += 1
        cal_risks.append(opt.cal_risk)
        test_risks.append(rt)
        margins.append(args.alpha - rt)

    cal_risks, test_risks = np.array(cal_risks), np.array(test_risks)
    viol = (test_risks > args.alpha).mean()
    print(f"{args.track} alpha={args.alpha}: pool={n}, n_cal={n_cal}, "
          f"n_test={n - n_cal}")
    print(f"violations: {int((test_risks > args.alpha).sum())}/{args.repeats} "
          f"(rate {viol:.3f})")
    # exact Clopper-Pearson upper bound for the violation probability
    from scipy.stats import beta
    k = int((test_risks > args.alpha).sum())
    ub = beta.ppf(0.95, k + 1, args.repeats - k) if k < args.repeats else 1.0
    print(f"95% CP upper bound on true violation prob: {ub:.4f}")
    print(f"cal risk:  mean {cal_risks.mean():.4f}  max {cal_risks.max():.4f}")
    print(f"test risk: mean {test_risks.mean():.4f}  max {test_risks.max():.4f} "
          f"(alpha - max = {args.alpha - test_risks.max():+.4f})")
    print(f"mean certified safety margin (alpha - cal_risk): "
          f"{(args.alpha - cal_risks).mean():.4f}")
    # HB-implied margin at this n_cal
    print(f"HB radius ~ sqrt(log(1/delta)/(2 n_cal)) = "
          f"{np.sqrt(np.log(1/args.delta)/(2*n_cal)):.4f}")
    print("selected policies across repeats:")
    for name, cnt in sel.most_common():
        print(f"  {cnt:4d}x  {name}")


if __name__ == "__main__":
    main()
