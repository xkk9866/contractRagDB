"""Voucher 2.0 analysis: telescoping path composition + selectivity-stratified
vouchers (assumption-free replacements for the NSH union bound).

Inputs (HybridQA, 500 cal + 500 test):
  hybridqa_{split}_rewrites.jsonl        variants: 0=P0 base, 1=P1=R1(P0),
                                         2..4 = single rewrites (unused here),
                                         5=P4=R4(R3(R2(R1(P0))))
  hybridqa_{split}_rewrites_chain.jsonl  variants: 0=P2=R2(P1), 1=P3=R3(P2)

Outputs:
  1. Base-relative union voucher (old, needs NSH) vs telescoping path voucher
     (new, assumption-free) vs direct certificate; realized harm on test.
  2. NSH violation rate (queries harmed by the chain but by no single rewrite)
     -- the failure mode the telescoping bound is immune to.
  3. Selectivity-conditioned voucher for R1 (prefilter): CP bound per
     pool-selectivity stratum; realized per-stratum harm on test.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.calibrate import (binomial_upper_bound,
                                   path_composition_voucher,
                                   selectivity_conditioned_voucher)  # noqa: E402
from contractrag.textutil import best_f1  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
TAU = 0.5
DELTA_R = 0.05


def load_matrix(path, n_var):
    recs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault(r["qid"], {})[int(r["rung"])] = r
    return {q: d for q, d in recs.items() if len(d) == n_var}


def build(split):
    """Return losses[6, n] for chain P0..P4 plus sigma[n] (pool selectivity).

    Chain order: index 0..4 = P0, P1, P2, P3, P4.
    """
    # base matrices were run on an earlier split ordering; resolve gold
    # answers across all splits by qid
    qs = {q["qid"]: q for s in get_splits("hybridqa").values() for q in s}
    base = load_matrix(os.path.join(EXP, f"hybridqa_{split}_rewrites.jsonl"), 6)
    chain = load_matrix(os.path.join(EXP, f"hybridqa_{split}_rewrites_chain.jsonl"), 2)
    qids = [q for q in base if q in chain]
    L = np.zeros((5, len(qids)))
    Ls = np.zeros((4, len(qids)))       # losses of single rewrites R1..R4 vs P0
    sigma = np.zeros(len(qids))
    for i, qid in enumerate(qids):
        gold = [qs[qid]["answer"]]

        def loss(rec):
            return float(best_f1(rec["answer"], gold) < TAU)

        L[0, i] = loss(base[qid][0])           # P0
        L[1, i] = loss(base[qid][1])           # P1 = R1(P0)
        L[2, i] = loss(chain[qid][0])          # P2 = R2(P1)
        L[3, i] = loss(chain[qid][1])          # P3 = R3(P2)
        L[4, i] = loss(base[qid][5])           # P4 = R4(P3) = Rall
        for v in range(1, 5):
            Ls[v - 1, i] = loss(base[qid][v])
        n_push = base[qid][1]["features"].get("n_pool", 1)
        n_full = base[qid][0]["features"].get("n_pool", 1)
        sigma[i] = min(1.0, n_push / max(1, n_full))
    return L, Ls, sigma, qids


def main():
    Lc, Lsc, sig_c, _ = build("cal")
    Lt, Lst, sig_t, _ = build("test")
    n = Lc.shape[1]
    print(f"n_cal={n} n_test={Lt.shape[1]}")

    # ---- old voucher: base-relative union bound (needs NSH) ---------------
    base_harm_cal = [(Lsc[v] > Lc[0]).astype(float) for v in range(4)]
    union_voucher = sum(binomial_upper_bound(int(h.sum()), n, DELTA_R)
                        for h in base_harm_cal)
    # NSH violation: chain harms but no single rewrite does
    chain_harm_cal = (Lc[4] > Lc[0]).astype(float)
    any_single = np.clip(sum(base_harm_cal), 0, 1)
    nsh_viol = float(((chain_harm_cal == 1) & (any_single == 0)).mean())

    # ---- new: telescoping path voucher (assumption-free) ------------------
    step_harm = [(Lc[i + 1] > Lc[i]).astype(float) for i in range(4)]
    path = path_composition_voucher(step_harm, DELTA_R)
    direct = binomial_upper_bound(int(chain_harm_cal.sum()), n, DELTA_R)
    chain_harm_test = float((Lt[4] > Lt[0]).mean())

    comp = {
        "n_cal": n, "delta_r": DELTA_R,
        "union_voucher_NSH": float(union_voucher),
        "nsh_violation_rate": nsh_viol,
        "telescoping_voucher": path["path_voucher"],
        "step_vouchers": path["step_voucher"],
        "step_harm_rates": path["step_harm_rate"],
        "direct_certificate": float(direct),
        "realized_harm_cal": float(chain_harm_cal.mean()),
        "realized_harm_test": chain_harm_test,
    }
    print("union(NSH)      =", round(comp["union_voucher_NSH"], 3),
          " NSH-violations =", round(nsh_viol, 4))
    print("telescoping     =", round(comp["telescoping_voucher"], 3),
          " steps:", [round(x, 3) for x in comp["step_vouchers"]])
    print("direct          =", round(comp["direct_certificate"], 3))
    print("realized (test) =", round(chain_harm_test, 3))

    # ---- selectivity-conditioned voucher for R1 (prefilter) ---------------
    edges = np.quantile(sig_c, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0], edges[-1] = 0.0, 1.0 + 1e-9
    harm_r1_cal = (Lsc[0] > Lc[0]).astype(float)
    strata = selectivity_conditioned_voucher(harm_r1_cal, sig_c, edges, DELTA_R)
    # realized per-stratum harm on test
    harm_r1_test = (Lst[0] > Lt[0]).astype(float)
    idx_t = np.clip(np.digitize(sig_t, edges[1:-1]), 0, len(strata) - 1)
    for b, s in enumerate(strata):
        mask = idx_t == b
        s["test_harm"] = float(harm_r1_test[mask].mean()) if mask.any() else None
        s["n_test"] = int(mask.sum())
        print(f"sigma in [{s['bin'][0]:.3f},{s['bin'][1]:.3f}): "
              f"cal harm {s['harm_rate']:.3f} voucher {s['voucher']:.3f} "
              f"test harm {s['test_harm']}")
    flat_voucher = binomial_upper_bound(int(harm_r1_cal.sum()), n, DELTA_R)
    print(f"flat R1 voucher = {flat_voucher:.3f} (what stratification refines)")

    out = {"tau": TAU, "delta_r": DELTA_R, "composition": comp,
           "r1_flat_voucher": float(flat_voucher),
           "r1_selectivity_strata": strata,
           "sigma_edges": [float(e) for e in edges]}
    with open(os.path.join(EXP, "voucher2_analysis.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved voucher2_analysis.json")


if __name__ == "__main__":
    main()
