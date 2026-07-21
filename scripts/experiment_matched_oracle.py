"""Matched-feasibility oracle (reviewer fix for the RQ2 oracle comparison).

The per-query zero-loss oracle answers every query at its cheapest zero-loss
rung; its risk target (zero loss wherever attainable) differs from the
contract's (mean risk <= alpha), so cost ratios against it mix two different
problems. This script computes the *contract-matched* hindsight optimum: the
minimum-cost per-query rung assignment whose MEAN loss on the test split is
<= alpha -- the true lower envelope for any policy satisfying the contract on
that split.

With 0/1 losses and cumulative (escalation) costs the exact optimum has a
greedy form: start every query at rung 0; while mean loss > alpha, fix the
violating query whose cheapest zero-loss rung has the smallest cost increment.
Each fix lowers the violation count by exactly 1 and increments are
independent across queries, so choosing the smallest increments first is
exactly optimal (exchange argument).

Pure numpy over materialized matrices; no LLM calls.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")

SETTINGS = [
    ("hybridqa", "quality", 0.5, 0.34),
    ("crag", "correct", 0.5, 0.65),
    ("asqa", "citation", 50.0, 0.216),
    ("qampari", "citation", 50.0, 0.71),
]


def matched_oracle(losses, cum_cost, alpha):
    """Exact min-cost assignment with mean loss <= alpha (0/1 losses)."""
    L, n = losses.shape
    stop = np.zeros(n, dtype=int)
    viol = losses[0].copy()
    budget = int(np.floor(alpha * n))
    need = int(viol.sum()) - budget
    if need > 0:
        incs = []
        for i in np.where(viol == 1)[0]:
            zero = np.where(losses[:, i] == 0)[0]
            if len(zero):
                j = zero[np.argmin(cum_cost[zero, i])]
                incs.append((float(cum_cost[j, i] - cum_cost[0, i]), i, int(j)))
        incs.sort()
        if len(incs) < need:
            return None  # infeasible even in hindsight
        for _, i, j in incs[:need]:
            stop[i] = j
    idx = np.arange(n)
    return {"risk": float(losses[stop, idx].mean()),
            "cost_mean": float(cum_cost[stop, idx].mean())}


def main():
    out = {"settings": []}
    for track, contract, tau, alpha in SETTINGS:
        loss_fn = make_loss_fn(track, contract, tau)
        kgc = kg_costs_for(track)
        td = TrackData(track, "test", loss_fn, kg_costs=kgc)
        ld, _, _, aux = td.build()
        cum = np.cumsum(ld.costs, axis=0)

        # zero-loss skyline (existing oracle), for reference
        L, n = ld.L, ld.n
        stop0 = np.full(n, L - 1)
        for i in range(n):
            z = np.where(ld.losses[:, i] == 0)[0]
            if len(z):
                stop0[i] = z[0]
        idx = np.arange(n)
        sky = {"risk": float(ld.losses[stop0, idx].mean()),
               "cost_mean": float(cum[stop0, idx].mean())}

        m = matched_oracle(ld.losses, cum, alpha)
        rec = {"track": track, "contract": contract, "tau": tau,
               "alpha": alpha, "n_test": n,
               "skyline_zero_loss": sky, "matched_oracle": m}
        out["settings"].append(rec)
        print(f"{track:>9s} a={alpha}: skyline risk {sky['risk']:.3f} "
              f"cost {sky['cost_mean']*1000:.2f}m | matched risk "
              f"{m['risk']:.3f} cost {m['cost_mean']*1000:.3f}m")

    path = os.path.join(EXP, "matched_oracle.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
