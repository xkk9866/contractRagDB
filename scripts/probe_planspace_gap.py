#!/usr/bin/env python3
"""Locate the real bottleneck of the certified optimizer.

Hypothesis under test: the loss is NOT statistical (HB half-widths are
0.016-0.049 on these splits) but *combinatorial*. The candidate space is a
69-point set in the (risk, cost) plane, so the cheapest FEASIBLE candidate
sits well below the contract level alpha: the optimizer pays for risk
headroom it is allowed to spend but cannot address, because no candidate
lands near alpha.

We measure three gaps per track and alpha:
  1. feasibility slack  = alpha - risk(selected)   (wasted risk budget)
  2. deterministic gap  = cost(selected) / cost(LP optimum) - 1
     where the LP optimum allows *randomized* policies, i.e. convex
     combinations of candidates. By LP duality the optimum is a mixture of
     at most two candidates on the lower cost-risk convex hull, and its
     risk equals alpha exactly -- so randomization converts the discrete
     plan space into a continuum in the risk dimension.
  3. resolution gap     = cost of the 69-candidate space vs a densified
     per-rung threshold grid.
Everything is computed on real execution matrices; no LLM calls.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.calibrate import hb_p_value, stop_rung, quantile_path
from contractrag.optimizer import (Candidate, apply_candidate, build_candidates,
                                   build_candidates_grid, pareto_prune)
from contractrag.policy import TrackData, RungScorer

TRACKS = [
    ("hybridqa", "quality", 0.5, [0.30, 0.34, 0.35, 0.40]),
    ("crag", "correct", 0.5, [0.62, 0.65, 0.70]),
    ("asqa", "citation", 50.0, [0.20, 0.25, 0.30]),
    ("qampari", "citation", 50.0, [0.60, 0.66, 0.71]),
]


def make_loss_fn(track, contract, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        return lambda rec, sc: float(sc.get("label") != "correct")
    return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)


def load(track, contract, tau, split):
    td = TrackData(track, split, make_loss_fn(track, contract, tau))
    ld, qids, _, aux = td.build()
    return ld, aux["raw_feats"]


def hb_upper(r_hat, n, delta):
    """Certified upper bound: smallest alpha with HB p-value <= delta."""
    lo, hi = r_hat, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if hb_p_value(r_hat, n, mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def stats_of(cands, ld):
    """(risk, cost) of every candidate on a split."""
    R, C = [], []
    for c in cands:
        loss, cost, _, _ = apply_candidate(c, ld)
        R.append(float(loss.mean()))
        C.append(float(cost.mean()))
    return np.array(R), np.array(C)


def lower_convex_hull(R, C):
    """Indices on the lower-left convex hull of the (risk, cost) cloud.

    The LP  min_w  C.w  s.t.  R.w <= alpha, w >= 0, sum w = 1
    has its optimum on this hull; the randomized optimum for a given alpha
    is a mixture of two adjacent hull vertices straddling alpha.
    """
    order = np.argsort(R)
    hull = []
    for i in order:
        while hull and C[i] <= C[hull[-1]] + 1e-15:
            hull.pop()
        if not hull or C[i] < C[hull[-1]]:
            hull.append(i)
    # keep only points that are not above the segment joining neighbours
    changed = True
    while changed and len(hull) > 2:
        changed = False
        for k in range(1, len(hull) - 1):
            a, b, c = hull[k - 1], hull[k], hull[k + 1]
            if R[c] - R[a] < 1e-15:
                continue
            t = (R[b] - R[a]) / (R[c] - R[a])
            if C[b] >= C[a] + t * (C[c] - C[a]) - 1e-15:
                hull.pop(k)
                changed = True
                break
    return hull


def lp_optimum(R, C, alpha):
    """Cheapest randomized policy with risk <= alpha (exact LP solution)."""
    feas = np.where(R <= alpha + 1e-12)[0]
    best_det = float(C[feas].min()) if len(feas) else None
    best_mix = best_det
    mix_info = None
    # mixtures of any two candidates straddling alpha
    lo_idx = np.where(R <= alpha)[0]
    hi_idx = np.where(R > alpha)[0]
    for i in lo_idx:
        for j in hi_idx:
            if C[j] >= C[i]:
                continue  # only mixing toward a cheaper-but-riskier plan helps
            # w on j such that w*R_j + (1-w)*R_i = alpha
            denom = R[j] - R[i]
            if denom <= 1e-12:
                continue
            w = (alpha - R[i]) / denom
            if not (0.0 <= w <= 1.0):
                continue
            c = w * C[j] + (1 - w) * C[i]
            if best_mix is None or c < best_mix - 1e-15:
                best_mix = float(c)
                mix_info = {"safe": int(i), "risky": int(j), "w_risky": float(w),
                            "risk_safe": float(R[i]), "risk_risky": float(R[j]),
                            "cost_safe": float(C[i]), "cost_risky": float(C[j])}
    return best_det, best_mix, mix_info


def certify_fixed_sequence(cands, ld_cal, alpha, delta, order):
    """Current algorithm: walk order, stop at first failure, take cheapest."""
    n = ld_cal.n
    certified = []
    for oi in order:
        loss, cost, _, _ = apply_candidate(cands[oi], ld_cal)
        r = float(loss.mean())
        if hb_p_value(r, n, alpha) <= delta:
            certified.append((oi, r, float(cost.mean())))
        else:
            break
    if not certified:
        return None
    return min(certified, key=lambda t: t[2])


def main():
    delta = 0.1
    out = {}
    for track, contract, tau, alphas in TRACKS:
        print(f"\n{'='*72}\n{track} ({contract})\n{'='*72}")
        ld_tr, ft_tr = load(track, contract, tau, "train")
        ld_ca, ft_ca = load(track, contract, tau, "cal")
        scorer = RungScorer().fit(ld_tr, ft_tr)
        ld_tr.scores = scorer.score(ft_tr)
        ld_ca.scores = scorer.score(ft_ca)
        n = ld_ca.n
        print(f"n_train={ld_tr.n} n_cal={n} HB half-width at a=0.35: "
              f"{hb_upper(0.30, n, delta) - 0.30:.4f}")

        # default 69-candidate space
        cands = build_candidates(ld_tr)
        R_tr, C_tr = stats_of(cands, ld_tr)
        R_ca, C_ca = stats_of(cands, ld_ca)
        # densified space: independent per-rung thresholds
        cands_g = build_candidates_grid(ld_tr, grid_per_rung=24, cap=20000)
        Rg_tr, Cg_tr = stats_of(cands_g, ld_tr)
        Rg_ca, Cg_ca = stats_of(cands_g, ld_ca)
        print(f"|default space|={len(cands)}  |grid space|={len(cands_g)}")
        print(f"distinct cal risks: default={len(np.unique(np.round(R_ca,4)))} "
              f"grid={len(np.unique(np.round(Rg_ca,4)))}")

        rows = []
        for a in alphas:
            order = sorted(range(len(cands)),
                           key=lambda i: (cands[i].train_risk, cands[i].train_cost))
            sel = certify_fixed_sequence(cands, ld_ca, a, delta, order)
            det69, mix69, mi69 = lp_optimum(R_ca, C_ca, a)
            detg, mixg, mig = lp_optimum(Rg_ca, Cg_ca, a)
            # what the certifier could reach if it could test every candidate
            # (oracle over the space, no multiplicity) -- upper bound on gain
            thr_bud = a - (hb_upper(0.0, n, delta))  # slack needed by HB
            row = {
                "alpha": a,
                "sel_risk": sel[1] if sel else None,
                "sel_cost": sel[2] if sel else None,
                "sel_name": cands[sel[0]].describe() if sel else None,
                "slack": (a - sel[1]) if sel else None,
                "det69_cost": det69, "mix69_cost": mix69,
                "detgrid_cost": detg, "mixgrid_cost": mixg,
                "mix69": mi69,
            }
            rows.append(row)
            g_fs_det = (row["sel_cost"] / det69 - 1) * 100 if (sel and det69) else None
            g_det_mix = (det69 / mix69 - 1) * 100 if (det69 and mix69) else None
            g_det_grid = (det69 / detg - 1) * 100 if (det69 and detg) else None
            g_all = (row["sel_cost"] / mixg - 1) * 100 if (sel and mixg) else None
            print(f"  a={a:.2f}  sel={row['sel_name']:<14} "
                  f"risk={row['sel_risk']:.3f} slack={row['slack']:+.3f} "
                  f"cost={row['sel_cost']*1000:.3f}mCNY")
            print(f"         det69={det69*1000:.3f}  mix69={mix69*1000:.3f}  "
                  f"detgrid={detg*1000:.3f}  mixgrid={mixg*1000:.3f} (mCNY)")
            print(f"         gaps: fs->det69 {g_fs_det:+.1f}%  "
                  f"det->mix {g_det_mix:+.1f}%  det->grid {g_det_grid:+.1f}%  "
                  f"TOTAL fs->mixgrid {g_all:+.1f}%")
            if mi69:
                print(f"         LP mixture: {cands[mi69['safe']].describe()} "
                      f"(r={mi69['risk_safe']:.3f},c={mi69['cost_safe']*1000:.2f}) "
                      f"+ {mi69['w_risky']:.3f}x"
                      f"{cands[mi69['risky']].describe()} "
                      f"(r={mi69['risk_risky']:.3f},c={mi69['cost_risky']*1000:.2f})")
        out[track] = {"n_cal": n, "n_cands": len(cands),
                      "n_grid": len(cands_g), "rows": rows}

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "probe_planspace_gap.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
