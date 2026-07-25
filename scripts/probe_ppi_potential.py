#!/usr/bin/env python3
"""Quantify the headroom for prediction-powered risk certification.

Question: the current certifier bounds R(pi)=E[loss] with a
Hoeffding-Bentkus tail bound on the calibration mean only. Its half-width
at n=350 is ~0.075, which forces the optimizer to leave a huge feasibility
margin (hence violation probabilities pinned at 0/1000).

If a train-fitted predictor g(q) of the per-query loss correlates with the
realized loss, a prediction-powered (PPI) rectifier
    R_ppi = mean_{unlabeled N}(g) + mean_{labeled n}(Y - g)
has variance ~ (1-rho^2) Var(Y)/n + Var(g)/N, i.e. effective sample size
inflated by 1/(1-rho^2). This probe measures rho and the realizable
variance reduction on the real execution matrices, per track and per
policy family, before any algorithm is written.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.calibrate import LadderData, hb_p_value, stop_rung, quantile_path
from contractrag.policy import TrackData, RungScorer

TRACKS = [
    ("hybridqa", "quality", 0.5),
    ("crag", "correct", 0.5),
    ("asqa", "citation", 50.0),
    ("qampari", "citation", 50.0),
]


def make_loss_fn(track, contract, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        if contract == "correct":
            return lambda rec, sc: float(sc.get("label") != "correct")
        return lambda rec, sc: float(sc.get("label") == "incorrect")
    if track in ("asqa", "qampari"):
        return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)
    raise ValueError(track)


def load(track, contract, tau, split):
    td = TrackData(track, split, make_loss_fn(track, contract, tau))
    ld, qids, _, aux = td.build()
    return ld, aux["raw_feats"], aux["latency"], qids


def hb_upper(r_hat, n, delta, grid=2001):
    """Smallest alpha whose HB p-value <= delta -> the certified upper bound."""
    lo, hi = r_hat, 1.0
    for a in np.linspace(lo, hi, grid):
        if hb_p_value(r_hat, n, a) <= delta:
            return float(a)
    return 1.0


def betting_upper(y, delta, grid=None):
    """Hedged-capital style upper confidence bound for E[Y], Y in [0,1].

    For a candidate mean m, wealth K_t(m) = prod_i (1 + w_i (m - Y_i)) with
    predictable bets w_i truncated to keep the factors positive is a
    nonnegative martingale under H0: E[Y]=m, so {m : K(m) < 1/delta} is a
    (1-delta) confidence set (Waudby-Smith & Ramdas, JRSSB 2024). We scan m
    and return the largest m not rejected from below, i.e. the upper bound.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return 1.0
    ms = np.linspace(0.0, 1.0, 501) if grid is None else grid
    # predictable plug-in bets (capital allocation), computed from the prefix
    running_mean = np.concatenate([[0.5], np.cumsum(y)[:-1] / np.arange(1, n)])
    running_var = np.full(n, 0.25)
    for i in range(1, n):
        running_var[i] = max(1e-4, np.var(y[:i]) + 1e-4)
    ok = []
    for m in ms:
        if m < y.mean():
            continue
        lam = np.minimum(np.sqrt(2 * np.log(1 / delta) / (n * running_var)),
                         0.5 / max(m, 1e-3))
        terms = 1.0 + lam * (m - y)
        if np.any(terms <= 0):
            continue
        logK = float(np.sum(np.log(terms)))
        if logK < np.log(1.0 / delta):
            ok.append(m)
    return float(max(ok)) if ok else 1.0


def ppi_upper(y, g_lab, g_unlab, delta):
    """PPI++ style upper bound: rectifier CI (betting) + unlabeled mean CI."""
    d = np.asarray(y, dtype=float) - np.asarray(g_lab, dtype=float)  # in [-1,1]
    n, N = len(d), len(g_unlab)
    # betting bound on the rectifier, shifted to [0,1]
    ub_d = betting_upper((d + 1.0) / 2.0, delta / 2.0) * 2.0 - 1.0
    # unlabeled mean of g: empirical Bernstein (g in [0,1], N large)
    v = float(np.var(g_unlab))
    ub_g = float(np.mean(g_unlab)) + np.sqrt(2 * v * np.log(2 / delta) / N) \
        + 7 * np.log(2 / delta) / (3 * (N - 1))
    return float(min(1.0, max(0.0, ub_g + ub_d)))


def policy_losses(ld, thr_or_rung, kind="thr"):
    idx = np.arange(ld.n)
    if kind == "thr":
        stop = stop_rung(ld.scores, np.asarray(thr_or_rung))
    else:
        stop = np.full(ld.n, int(thr_or_rung))
    return ld.losses[stop, idx], stop


def main():
    delta = 0.1
    out = {}
    for track, contract, tau in TRACKS:
        print(f"\n=== {track} ({contract}, tau={tau}) ===")
        ld_tr, ft_tr, _, _ = load(track, contract, tau, "train")
        ld_ca, ft_ca, _, _ = load(track, contract, tau, "cal")
        ld_te, ft_te, _, _ = load(track, contract, tau, "test")
        scorer = RungScorer().fit(ld_tr, ft_tr)
        ld_tr.scores = scorer.score(ft_tr)
        ld_ca.scores = scorer.score(ft_ca)
        ld_te.scores = scorer.score(ft_te)
        L = ld_tr.L
        print(f"n_train={ld_tr.n} n_cal={ld_ca.n} n_test={ld_te.n} L={L}")
        print("per-rung risk (cal):", np.round(ld_ca.losses.mean(axis=1), 3))

        # ---- cross-rung loss correlation on calibration ----
        cc = np.corrcoef(ld_ca.losses)
        print("cross-rung loss corr:\n", np.round(cc, 3))

        # ---- predictor g: train a per-rung loss model on TRAIN only ----
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        gmodels = []
        for j in range(L):
            X, y = ft_tr[j], ld_tr.losses[j]
            if len(np.unique(y)) < 2:
                gmodels.append(("const", float(y.mean())))
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X, y)
            gmodels.append(("lr", m))

        def g_of(feats, stop):
            n = feats.shape[1]
            gv = np.zeros(n)
            for j in range(L):
                mask = stop == j
                if not mask.any():
                    continue
                kind, m = gmodels[j]
                if kind == "const":
                    gv[mask] = m
                else:
                    gv[mask] = m.predict_proba(feats[j][mask])[:, 1]
            return gv

        # ---- representative policies ----
        fams = quantile_path(ld_tr, 9)
        policies = [("fixed_%d" % j, j, "fixed") for j in range(L)]
        policies += [("thr_q%.2f" % q, thr, "thr") for q, thr in fams[1:-1]]

        rows = []
        for name, param, kind in policies:
            y_ca, stop_ca = policy_losses(ld_ca, param, kind)
            g_ca = g_of(ft_ca, stop_ca)
            # unlabeled pool = train split features (labels withheld)
            _, stop_tr = policy_losses(ld_tr, param, kind)
            g_tr = g_of(ft_tr, stop_tr)
            r_hat = float(y_ca.mean())
            rho = float(np.corrcoef(y_ca, g_ca)[0, 1]) if np.std(g_ca) > 1e-9 else 0.0
            vr = float(np.var(y_ca - g_ca) / max(np.var(y_ca), 1e-12))
            ub_hb = hb_upper(r_hat, ld_ca.n, delta)
            ub_bet = betting_upper(y_ca, delta)
            ub_ppi = ppi_upper(y_ca, g_ca, g_tr, delta)
            rows.append({
                "policy": name, "r_hat": r_hat, "rho": rho,
                "var_ratio": vr, "hb": ub_hb, "bet": ub_bet, "ppi": ub_ppi,
                "hb_width": ub_hb - r_hat, "bet_width": ub_bet - r_hat,
                "ppi_width": ub_ppi - r_hat,
            })
        print(f"{'policy':<12} {'r_hat':>6} {'rho':>6} {'v(1-r2)':>8} "
              f"{'HB_ub':>7} {'BET_ub':>7} {'PPI_ub':>7} "
              f"{'HBw':>6} {'BETw':>6} {'PPIw':>6}")
        for r in rows:
            print(f"{r['policy']:<12} {r['r_hat']:6.3f} {r['rho']:6.3f} "
                  f"{r['var_ratio']:8.3f} {r['hb']:7.3f} {r['bet']:7.3f} "
                  f"{r['ppi']:7.3f} {r['hb_width']:6.3f} "
                  f"{r['bet_width']:6.3f} {r['ppi_width']:6.3f}")
        out[track] = {"n_cal": ld_ca.n, "L": L,
                      "per_rung_risk_cal": ld_ca.losses.mean(axis=1).tolist(),
                      "cross_rung_corr": cc.tolist(), "rows": rows}

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "probe_ppi_potential.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
