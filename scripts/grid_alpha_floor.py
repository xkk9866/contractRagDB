#!/usr/bin/env python3
"""Find the strictest contract each mechanism can actually serve.

Reporting certified cost at a handful of hand-picked contract levels leaves
most of a table empty, because at strict levels the certifier returns nothing
and the entry is a dash. The dash is a real result, but it is the wrong
summary: what an operator wants to know is *how strict a contract the
mechanism can honour at all*, and how much that differs from what the plan
space physically permits.

So for each plan space we scan alpha on a fine grid and record three
thresholds:

  floor      the smallest alpha any single plan attains -- what the hardware
             and the models permit, ignoring how it is discovered
  alpha_lp   the smallest alpha the offline action LP attains once declining
             is available; declining pushes below the floor
  alpha_cert the smallest alpha the fixed-sequence certifier can certify

alpha_cert - alpha_lp is the contract range certification removes from the
menu even though the space contains a feasible policy there. It is the same
loss the paper argues about, expressed as a number rather than a dash.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.certlp import (build_randomized_chain,  # noqa: E402
                               certify_randomized_chain)
from contractrag.ledger import solve_action_lp  # noqa: E402
from contractrag.plangrid import (PlanGrid, build_grid_candidates,  # noqa
                                  candidate_matrices)
from scripts.experiment_plangrid_all import (DIAGONAL, HOSTED,  # noqa: E402
                                             LOCAL, load_plan_grid)

EXP = os.path.join(ROOT, "experiments")


def prepare(track, sel, num_lam, mix):
    """Candidate matrices for one plan subset, on all three splits."""
    mats = {}
    for split in ("train", "cal", "test"):
        L, C, names, _, F = load_plan_grid(track, HOSTED + LOCAL, split)
        mats[split] = (L, C, names, F)
    keep = set.intersection(*(set(m[2]) for m in mats.values()))
    order = [nm for nm in mats["train"][2] if nm in keep]
    if sel == "diag":
        order = [nm for nm in order if nm in DIAGONAL]
    grids = {}
    for split, (L, C, names, F) in mats.items():
        idx = [names.index(nm) for nm in order]
        grids[split] = PlanGrid(L[idx], C[idx], F, order)
    cands = build_grid_candidates(grids["train"], num_lam=num_lam)
    K = {s: candidate_matrices(cands, grids[s]) for s in grids}
    pop_L = np.concatenate([K["cal"][0], K["test"][0]], axis=1)
    pop_C = np.concatenate([K["cal"][1], K["test"][1]], axis=1)
    plan_L = np.concatenate([grids["cal"].losses, grids["test"].losses],
                            axis=1)
    plan_C = np.concatenate([grids["cal"].costs, grids["test"].costs], axis=1)
    return cands, K, pop_L, pop_C, plan_L, plan_C, order, mix


def scan(track, sel, args):
    (cands, K, pop_L, pop_C, plan_L, plan_C, names,
     mix) = prepare(track, sel, args.num_lam, args.mix)
    plan_r, plan_c = plan_L.mean(axis=1), plan_C.mean(axis=1)
    floor = float(plan_r.min())
    decline_price = float(plan_c.max())
    grid = np.arange(args.lo, args.hi + 1e-9, args.step)

    alpha_lp, lp_at = None, {}
    for a in grid:
        opt, _ = solve_action_lp(plan_r, plan_c, float(a), decline_price)
        if opt is not None and np.isfinite(opt):
            lp_at[round(float(a), 4)] = float(opt)
            if alpha_lp is None:
                alpha_lp = float(a)

    chain = build_randomized_chain(K["train"][0], K["train"][1], mix_grid=mix)
    alpha_cert, cert_at, pop_risk_at = None, {}, {}
    for a in grid:
        cp = certify_randomized_chain(K["cal"][0], K["cal"][1], chain,
                                      float(a), args.delta)
        if cp.element is None:
            continue
        e = cp.element
        pc = float(sum(w * pop_C[k].mean()
                       for k, w in zip(e.support, e.weights)))
        pr = float(sum(w * pop_L[k].mean()
                       for k, w in zip(e.support, e.weights)))
        cert_at[round(float(a), 4)] = pc
        pop_risk_at[round(float(a), 4)] = pr
        if alpha_cert is None:
            alpha_cert = float(a)

    return {"space": sel, "n_plans": len(names), "floor": floor,
            "decline_price": decline_price, "alpha_lp": alpha_lp,
            "alpha_cert": alpha_cert, "lp_at": lp_at, "cert_at": cert_at,
            "cert_pop_risk": pop_risk_at, "plans": names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=["hybridqa", "crag"])
    ap.add_argument("--lo", type=float, default=0.10)
    ap.add_argument("--hi", type=float, default=0.80)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--num_lam", type=int, default=40)
    ap.add_argument("--mix", type=int, default=7)
    args = ap.parse_args()
    out = {}
    for tr in args.tracks:
        out[tr] = {}
        for sel in ("diag", "grid"):
            r = scan(tr, sel, args)
            out[tr][sel] = r
            print(f"{tr:9s} {sel:5s} plans={r['n_plans']:3d} "
                  f"floor={r['floor']:.3f}  "
                  f"alpha_lp={r['alpha_lp']}  alpha_cert={r['alpha_cert']}  "
                  f"certifiable levels={len(r['cert_at'])}/"
                  f"{len(r['lp_at'])} feasible")
    path = os.path.join(EXP, "grid_alpha_floor.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "tracks": out}, f, indent=1,
                  default=float)
    print("wrote", path)


if __name__ == "__main__":
    main()
