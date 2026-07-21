"""Voucher coverage experiment (reviewer fix for Table 3 / rewrites).

A Clopper-Pearson voucher computed on ONE calibration draw is a 1-delta_r
confidence upper bound on the harm probability; it is NOT guaranteed to
dominate the empirical harm of one finite test sample. The correct claim is
coverage over repeated draws, which this script measures directly.

Protocol: pool the cal+test rewrite matrices into a fixed finite population
of N queries. Repeat `draws` times: sample n_cal queries without replacement,
recompute every voucher on the draw, and check whether it covers the
POPULATION harm rate (exact, computed on all N queries). Report per-rewrite
coverage, telescoping path-voucher coverage, and direct-certificate coverage
against their nominal levels.

Pure numpy over materialized matrices; no LLM calls.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.calibrate import binomial_upper_bound  # noqa: E402
from scripts.experiment_voucher2 import build  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
DELTA_R = 0.05
N_CAL = 500
DRAWS = 1000

NAMES = ["R1 prefilter", "R2 k-trunc", "R3 no-rerank", "R4 gen-down"]


def main():
    # fixed finite population: cal + test pooled
    Lc, Lsc, sig_c, _ = build("cal")
    Lt, Lst, sig_t, _ = build("test")
    L = np.concatenate([Lc, Lt], axis=1)     # chain P0..P4
    Ls = np.concatenate([Lsc, Lst], axis=1)  # single rewrites vs P0
    N = L.shape[1]

    # population harm rates (exact, fixed)
    pop_single = [(Ls[v] > L[0]).astype(float) for v in range(4)]
    pop_step = [(L[i + 1] > L[i]).astype(float) for i in range(4)]
    pop_chain = (L[4] > L[0]).astype(float)
    pop_single_rate = [float(h.mean()) for h in pop_single]
    pop_chain_rate = float(pop_chain.mean())

    rng = np.random.default_rng(0)
    cov_single = np.zeros(4)
    cov_tele = 0
    cov_direct = 0
    vouchers_single = [[] for _ in range(4)]
    vouchers_tele, vouchers_direct = [], []
    for _ in range(DRAWS):
        ci = rng.choice(N, size=N_CAL, replace=False)
        # single-rewrite vouchers vs population harm
        for v in range(4):
            eps = binomial_upper_bound(int(pop_single[v][ci].sum()), N_CAL,
                                       DELTA_R)
            vouchers_single[v].append(eps)
            cov_single[v] += float(eps >= pop_single_rate[v])
        # telescoping path voucher (nominal 1 - 4*delta_r)
        tele = sum(binomial_upper_bound(int(pop_step[i][ci].sum()), N_CAL,
                                        DELTA_R) for i in range(4))
        vouchers_tele.append(tele)
        cov_tele += float(tele >= pop_chain_rate)
        # direct certificate on the chain (nominal 1 - delta_r)
        d = binomial_upper_bound(int(pop_chain[ci].sum()), N_CAL, DELTA_R)
        vouchers_direct.append(d)
        cov_direct += float(d >= pop_chain_rate)

    out = {"delta_r": DELTA_R, "n_cal": N_CAL, "draws": DRAWS,
           "population_size": N,
           "single": [], "composition": {}}
    for v in range(4):
        rec = {"name": NAMES[v],
               "pop_harm": pop_single_rate[v],
               "coverage": cov_single[v] / DRAWS,
               "nominal": 1 - DELTA_R,
               "voucher_mean": float(np.mean(vouchers_single[v])),
               "voucher_p05": float(np.percentile(vouchers_single[v], 5))}
        out["single"].append(rec)
        print(f"{NAMES[v]:>14s}: pop harm {rec['pop_harm']:.3f} "
              f"coverage {rec['coverage']:.3f} (nominal {rec['nominal']:.2f}) "
              f"voucher mean {rec['voucher_mean']:.3f}")
    out["composition"] = {
        "pop_chain_harm": pop_chain_rate,
        "telescoping": {"coverage": cov_tele / DRAWS,
                        "nominal": 1 - 4 * DELTA_R,
                        "voucher_mean": float(np.mean(vouchers_tele))},
        "direct": {"coverage": cov_direct / DRAWS,
                   "nominal": 1 - DELTA_R,
                   "voucher_mean": float(np.mean(vouchers_direct))},
    }
    c = out["composition"]
    print(f"chain pop harm {pop_chain_rate:.3f}: telescoping coverage "
          f"{c['telescoping']['coverage']:.3f} (nominal "
          f"{c['telescoping']['nominal']:.2f}), direct coverage "
          f"{c['direct']['coverage']:.3f} (nominal {c['direct']['nominal']:.2f})")

    path = os.path.join(EXP, "voucher_coverage.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
