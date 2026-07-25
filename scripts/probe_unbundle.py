#!/usr/bin/env python3
"""Does unbundling retrieval strength from generator strength pay?

The incumbent ladder walks a hand-drawn diagonal of the physical plan space:
rung 0 pairs the weakest retrieval with the cheapest generator, rung 3 the
strongest with the most expensive. Retrieval depth and generator tier are
therefore never varied independently, which is the RAG equivalent of fixing a
join order by hand. The consequence is a hole in the achievable (risk, cost)
cloud exactly where a certifier wants to operate: plans that are cheap to
generate but retrieve well, whose risk lands just under a strict contract.

The cross-family runs already on disk let us test this at zero cost. Four
generator families were executed over the SAME materialized retrieval views,
so for every retrieval rung there are four different generators -- sixteen
genuine physical plans. We compare the (risk, cost) frontier and the certified
cost of

  diag-<fam>  : the incumbent single-family diagonal ladder (4 plans + policies)
  grid        : all 16 (retrieval rung, generator) plans jointly

If unbundling matters, `grid` should both dominate the frontier and certify
strictly cheaper plans under the same contract and the same error budget.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.calibrate import hb_p_value
from contractrag.certlp import (ChainElement, build_randomized_chain,
                                certify_randomized_chain, hb_upper_bound,
                                pareto_frontier_idx, solve_risk_lp)

EXP = os.path.join(ROOT, "experiments")
FAMS = ["qwen", "deepseek", "glm", "gemma"]
TAU = 0.5
SPLITS = ["train", "cal", "test"]


def load_family(fam, split):
    """(qid, rung) -> (loss, cost) from the family's matrix + scores."""
    mat, sc = {}, {}
    with open(os.path.join(EXP, f"hqfam_{fam}_{split}_matrix.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mat[(r["qid"], r["rung"])] = r
    p = os.path.join(EXP, f"hqfam_{fam}_{split}_scores.jsonl")
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sc[(r["qid"], r["rung"])] = r
    out = {}
    for key, r in mat.items():
        s = sc.get(key)
        if s is None:
            continue
        out[key] = (float(s.get("f1", 0.0) < TAU), float(r["cost_cny"]),
                    r["model"])
    return out


def build_grid(split):
    """Plan matrices for the 16 (rung, family) physical plans on one split."""
    per = {f: load_family(f, split) for f in FAMS}
    qids = None
    for f in FAMS:
        qs = {k[0] for k in per[f]}
        qids = qs if qids is None else (qids & qs)
    qids = sorted(qids)
    plans, names = [], []
    for rung in range(4):
        for fam in FAMS:
            col_l, col_c, ok = [], [], True
            model = None
            for q in qids:
                v = per[fam].get((q, rung))
                if v is None:
                    ok = False
                    break
                col_l.append(v[0])
                col_c.append(v[1])
                model = v[2]
            if ok:
                plans.append((np.array(col_l), np.array(col_c)))
                names.append(f"R{rung}/{fam}:{model}")
    L = np.array([p[0] for p in plans])
    C = np.array([p[1] for p in plans])
    return L, C, names, qids


def certified_cost(Lc, Cc, Ltr, Ctr, alpha, delta, mix_grid=7):
    """Cheapest certified plan cost via the randomized fixed-sequence chain."""
    chain = build_randomized_chain(Ltr, Ctr, mix_grid=mix_grid)
    cp = certify_randomized_chain(Lc, Cc, chain, alpha, delta, bound="hb")
    if cp.element is None:
        return None, None, None
    e = cp.element
    return cp.cal_cost, e.describe(), e


def main():
    delta = 0.1
    alphas = [0.20, 0.25, 0.30, 0.34, 0.40]
    Ltr, Ctr, names, q_tr = build_grid("train")
    Lca, Cca, _, q_ca = build_grid("cal")
    Lte, Cte, _, q_te = build_grid("test")
    print(f"plans={len(names)}  n_train={Ltr.shape[1]} n_cal={Lca.shape[1]} "
          f"n_test={Lte.shape[1]}")
    print("\nper-plan risk / cost (cal):")
    print(f"  {'plan':<34} {'risk':>7} {'cost mCNY':>10}")
    order = np.argsort(Cca.mean(axis=1))
    for i in order:
        print(f"  {names[i]:<34} {Lca[i].mean():7.3f} "
              f"{Cca[i].mean()*1000:10.4f}")

    # pooled population for exact evaluation
    Lpop = np.concatenate([Lca, Lte], axis=1)
    Cpop = np.concatenate([Cca, Cte], axis=1)
    R_pop, C_pop = Lpop.mean(axis=1), Cpop.mean(axis=1)

    print("\ncertified cost under the same contract and delta:")
    hdr = f"  {'alpha':>6} " + " ".join(f"{('diag-'+f):>13}" for f in FAMS) \
        + f" {'grid16':>13} {'grid/best-diag':>15}"
    print(hdr)
    rows = {}
    for a in alphas:
        cells = {}
        for fi, fam in enumerate(FAMS):
            idx = [i for i, nm in enumerate(names)
                   if nm.split("/")[1].split(":")[0] == fam]
            # the incumbent diagonal: rung r of family fam
            diag = [i for i in idx
                    if int(names[i][1]) == FAMS.index(fam) * 0 + int(names[i][1])]
            sel = idx  # family's four retrieval rungs with its own tier model
            c, desc, _ = certified_cost(Lca[sel], Cca[sel], Ltr[sel], Ctr[sel],
                                        a, delta)
            cells[fam] = c
        cg, dg, eg = certified_cost(Lca, Cca, Ltr, Ctr, a, delta)
        best_diag = min([v for v in cells.values() if v is not None],
                        default=None)
        gain = (best_diag / cg) if (best_diag and cg) else None
        line = f"  {a:6.2f} " + " ".join(
            f"{(cells[f]*1000 if cells[f] else float('nan')):13.4f}"
            for f in FAMS)
        line += f" {(cg*1000 if cg else float('nan')):13.4f}"
        line += f" {(gain if gain else float('nan')):14.2f}x"
        print(line)
        rows[str(a)] = {"alpha": a,
                        "diag": {f: cells[f] for f in FAMS},
                        "grid": cg, "grid_plan": dg,
                        "gain_vs_best_diag": gain}

    # how much of the space is on the frontier, and where the LP optimum sits
    front = pareto_frontier_idx(Ltr.mean(axis=1), Ctr.mean(axis=1), 64)
    print(f"\ntrain Pareto frontier: {len(front)}/{len(names)} plans")
    for i in front:
        print(f"  {names[i]:<34} train risk={Ltr[i].mean():.3f} "
              f"cost={Ctr[i].mean()*1000:.4f}")
    print("\npopulation LP optimum (evaluation only):")
    for a in alphas:
        lp, w = solve_risk_lp(R_pop[front], C_pop[front], a)
        feas = np.where(R_pop <= a)[0]
        bd = float(C_pop[feas].min()) if len(feas) else float("nan")
        print(f"  alpha={a:.2f}  best det={bd*1000:8.4f}  LP={(lp or float('nan'))*1000:8.4f}")

    path = os.path.join(EXP, "probe_unbundle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"names": names, "n_cal": int(Lca.shape[1]),
                   "cal_risk": R_pop.tolist(), "cal_cost": C_pop.tolist(),
                   "rows": rows}, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
