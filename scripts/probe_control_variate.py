#!/usr/bin/env python3
"""Can a control variate shrink the confidence width that sets certified cost?

Diagnosis this probe tests. On every track the certified cost sits ~1.4x above
the population LP optimum, and the whole gap is the confidence width: with
n_cal calibration queries the certifier may only deploy a plan whose ESTIMATED
risk clears alpha minus that width. Widening the plan grid cannot close it, and
neither can a better search -- it is the price of turning a finite sample into a
distribution-free guarantee.

The width is driven by the variance of the per-query loss, which for a binary
contract is R(1-R), about 0.23 at R=0.35. But the loss is far from
unpredictable: a query whose retrieval already returns a decisive top passage
rarely violates the contract, whichever plan runs. If g(q) is a predictor of
that loss built from PRE-EXECUTION retrieval signals only, then

    R_hat_adj = (1/n) sum_i [ l(q_i) - g(q_i) ]  +  E_unlab[g]

is unbiased for R whenever E_unlab[g] is the exact mean of g over the query
population -- and it is computable exactly, because g needs no gold answer and
no generator call, only BM25/dense scores. So an essentially unlimited unlabeled
pool can pin it down, and the variance of the certified quantity drops from
Var(l) to Var(l - g).

This probe measures, on real execution matrices, the achievable variance ratio
Var(l-g)/Var(l) and the resulting reduction in confidence width, which converts
directly into certified cost. It commits to nothing: if the ratio is near one,
the control-variate route is dead and the gap is irreducible at this n.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.certlp import (emp_bernstein_upper, hb_upper_bound,
                                pareto_frontier_idx)
from contractrag.optimizer import apply_candidate, build_candidates
from contractrag.policy import RungScorer, TrackData

TRACK_CFG = {
    "hybridqa": ("quality", 0.5, [0.30, 0.35, 0.40]),
    "crag": ("correct", 0.5, [0.62, 0.65, 0.70]),
    "asqa": ("citation", 50.0, [0.20, 0.25, 0.30]),
    "qampari": ("citation", 50.0, [0.60, 0.66, 0.71]),
}


def make_loss_fn(track, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        return lambda rec, sc: float(sc.get("label") != "correct")
    return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)


def load(track, tau, split):
    td = TrackData(track, split, make_loss_fn(track, tau))
    ld, qids, _, aux = td.build()
    return ld, aux["raw_feats"]


def cand_matrices(cands, ld):
    K, n = len(cands), ld.n
    L = np.zeros((K, n))
    C = np.zeros((K, n))
    for k, c in enumerate(cands):
        loss, cost, _, _ = apply_candidate(c, ld)
        L[k], C[k] = loss, cost
    return L, C


def fit_predictor(X_tr, y_tr, kind="logit"):
    """Per-candidate loss predictor from pre-execution features."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y_tr)) < 2:
        return None
    if kind == "logit":
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0))
    else:
        base = HistGradientBoostingClassifier(max_depth=3, max_iter=200,
                                              learning_rate=0.08,
                                              l2_regularization=1.0)
        m = CalibratedClassifierCV(base, method="isotonic", cv=3)
    m.fit(X_tr, y_tr)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=list(TRACK_CFG))
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--kind", default="logit", choices=["logit", "gbdt"])
    args = ap.parse_args()

    for track in args.tracks:
        _, tau, alphas = TRACK_CFG[track]
        ld_tr, ft_tr = load(track, tau, "train")
        ld_ca, ft_ca = load(track, tau, "cal")
        ld_te, ft_te = load(track, tau, "test")
        sc = RungScorer().fit(ld_tr, ft_tr)
        for ld, ft in ((ld_tr, ft_tr), (ld_ca, ft_ca), (ld_te, ft_te)):
            ld.scores = sc.score(ft)
        cands = build_candidates(ld_tr)
        L_tr, C_tr = cand_matrices(cands, ld_tr)
        L_ca, C_ca = cand_matrices(cands, ld_ca)
        L_te, C_te = cand_matrices(cands, ld_te)
        # raw_feats is (rung, query, feature); the control variate may only see
        # the cheapest rung, whose signals cost a BM25/dense probe and one
        # small-model call -- negligible next to the plans being certified.
        X_tr = np.asarray(ft_tr, dtype=float)[0]
        X_ca = np.asarray(ft_ca, dtype=float)[0]
        X_te = np.asarray(ft_te, dtype=float)[0]

        tr_risk, tr_cost = L_tr.mean(axis=1), C_tr.mean(axis=1)
        front = pareto_frontier_idx(tr_risk, tr_cost, 48)
        n_ca = L_ca.shape[1]
        print(f"\n{'='*80}\n{track}: K={len(cands)} |front|={len(front)} "
              f"n_train={L_tr.shape[1]} n_cal={n_ca} n_test={L_te.shape[1]} "
              f"feat_dim={X_tr.shape[1]}\n{'='*80}")
        print(f"  {'cand':>6} {'R_cal':>7} {'corr':>6} {'var ratio':>10} "
              f"{'HB w':>7} {'CV w':>7} {'width ratio':>12}")
        ratios, wratios = [], []
        for k in front[:14]:
            m = fit_predictor(X_tr, L_tr[k], args.kind)
            if m is None:
                continue
            g_ca = m.predict_proba(X_ca)[:, 1]
            g_te = m.predict_proba(X_te)[:, 1]
            y = L_ca[k]
            # unlabeled pool = the test split's features (no gold, no LLM call
            # needed to evaluate g there); its size caps the achievable gain
            mu = float(np.concatenate([g_ca, g_te]).mean())
            d = y - g_ca
            v_y = float(np.var(y, ddof=1))
            v_d = float(np.var(d, ddof=1))
            corr = float(np.corrcoef(y, g_ca)[0, 1]) if np.std(g_ca) > 0 else 0.0
            r_hat = float(y.mean())
            hb_w = hb_upper_bound(r_hat, n_ca, args.delta) - r_hat
            # control-variate bound: shift d into [0,1], bound its mean, add mu
            db = (d + 1.0) / 2.0
            ub_d = 2.0 * emp_bernstein_upper(db, args.delta) - 1.0
            cv_w = max(0.0, (ub_d + mu) - r_hat)
            ratios.append(v_d / max(v_y, 1e-9))
            wratios.append(cv_w / max(hb_w, 1e-9))
            print(f"  {k:>6} {r_hat:7.3f} {corr:6.3f} "
                  f"{v_d/max(v_y,1e-9):10.3f} {hb_w:7.4f} {cv_w:7.4f} "
                  f"{cv_w/max(hb_w,1e-9):12.3f}")
        if ratios:
            print(f"  median var ratio={np.median(ratios):.3f}  "
                  f"median width ratio={np.median(wratios):.3f}")


if __name__ == "__main__":
    main()
