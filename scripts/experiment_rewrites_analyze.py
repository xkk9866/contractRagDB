"""Analyze rewrite-voucher experiment (RQ4).

For each rewrite R_i vs base plan P* (variant 0):
  - harm rate  rho_i = P(loss_i > loss_base)  on cal split
  - voucher    eps_hat_i = Clopper-Pearson upper bound at 1-delta_r
  - realized harm on test split (voucher validity check)
Composition:
  - union-bound voucher  sum eps_hat_i  vs realized harm of Rall (variant 5)
  - NSH violation rate: queries harmed by Rall but by no single rewrite
  - direct certificate on Rall (CP bound) --> tightness gap
Also: mean cost of each variant, mean loss.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.calibrate import binomial_upper_bound  # noqa: E402
from contractrag.textutil import best_f1  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
TAU = 0.5
DELTA_R = 0.05

NAMES = ["base", "R1 prefilter", "R2 k-trunc", "R3 no-rerank", "R4 gen-down",
         "R-all composed"]


def load(split):
    qs = {q["qid"]: q for q in get_splits("hybridqa")[split]}
    recs = {}
    with open(os.path.join(EXP, f"hybridqa_{split}_rewrites.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault(r["qid"], {})[r["rung"]] = r
    qids = [q for q, d in recs.items() if len(d) == 6]
    L = np.zeros((6, len(qids)))
    C = np.zeros((6, len(qids)))
    for i, qid in enumerate(qids):
        gold = [qs[qid]["answer"]]
        for v in range(6):
            rec = recs[qid][v]
            f1 = best_f1(rec["answer"], gold)
            L[v, i] = float(f1 < TAU)
            C[v, i] = rec["cost_cny"]
    return L, C, qids


def main():
    Lc, Cc, _ = load("cal")
    Lt, Ct, _ = load("test")
    n = Lc.shape[1]
    out = {"tau": TAU, "delta_r": DELTA_R, "n_cal": n, "n_test": Lt.shape[1],
           "variants": []}
    harms_cal = []
    for v in range(6):
        harm_cal = (Lc[v] > Lc[0]).astype(float)
        harm_test = (Lt[v] > Lt[0]).astype(float)
        eps_hat = binomial_upper_bound(int(harm_cal.sum()), n, DELTA_R)
        out["variants"].append({
            "name": NAMES[v],
            "mean_loss_cal": float(Lc[v].mean()),
            "mean_loss_test": float(Lt[v].mean()),
            "mean_cost": float(np.concatenate([Cc[v], Ct[v]]).mean()),
            "harm_rate_cal": float(harm_cal.mean()),
            "harm_rate_test": float(harm_test.mean()),
            "voucher_eps": float(eps_hat),
        })
        if 1 <= v <= 4:
            harms_cal.append(harm_cal)
        print(f"{NAMES[v]:>16s}: cost={out['variants'][-1]['mean_cost']*1000:.2f}mCNY "
              f"loss={Lc[v].mean():.3f} harm_cal={harm_cal.mean():.3f} "
              f"eps_hat={eps_hat:.3f} harm_test={harm_test.mean():.3f}")

    # composition analysis
    union_voucher = sum(v["voucher_eps"] for v in out["variants"][1:5])
    harm_all_cal = (Lc[5] > Lc[0]).astype(float)
    harm_all_test = (Lt[5] > Lt[0]).astype(float)
    any_single = np.clip(sum(harms_cal), 0, 1)
    nsh_viol = float(((harm_all_cal == 1) & (any_single == 0)).mean())
    direct = binomial_upper_bound(int(harm_all_cal.sum()), n, DELTA_R)
    out["composition"] = {
        "union_voucher": float(union_voucher),
        "direct_certificate": float(direct),
        "realized_harm_cal": float(harm_all_cal.mean()),
        "realized_harm_test": float(harm_all_test.mean()),
        "nsh_violation_rate": nsh_viol,
        "tightness_ratio": float(union_voucher / max(direct, 1e-9)),
    }
    print(f"composed: union_voucher={union_voucher:.3f} direct={direct:.3f} "
          f"realized(test)={harm_all_test.mean():.3f} NSH-viol={nsh_viol:.4f}")

    with open(os.path.join(EXP, "rewrites_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved rewrites_analysis.json")


if __name__ == "__main__":
    main()
