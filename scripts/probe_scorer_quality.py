#!/usr/bin/env python3
"""How much of the certified optimizer's cost is paid for a weak risk model?

The sufficiency score drives every stopping decision, and it is currently a
per-rung logistic regression on 18 hand-built features -- the equivalent of
running a query optimizer on a single-column histogram. This probe replaces
it with progressively stronger estimators and measures the induced shift of
the achievable (risk, cost) frontier, which upper-bounds the savings any
certifier can extract.

Estimators compared, all fit on TRAIN only and cross-fitted so the scores fed
to the certifier are out-of-fold:
  lr      : incumbent logistic regression
  gbdt    : gradient-boosted trees
  gbdt_x  : gradient-boosted trees on features augmented with cross-rung
            context (all rungs' retrieval signals visible to every rung)
  isotonic: gbdt_x followed by isotonic recalibration of P(no violation)

Reported per track: AUC/Brier of the score against the realized loss, and the
mean cost of the cheapest plan whose TEST risk stays under alpha -- the
frontier position the certifier is trying to reach.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.calibrate import LadderData, stop_rung
from contractrag.optimizer import apply_candidate, build_candidates
from contractrag.policy import TrackData, RungScorer

TRACK_CFG = {
    "hybridqa": ("quality", 0.5, [0.30, 0.34, 0.40]),
    "crag": ("correct", 0.5, [0.62, 0.65, 0.70]),
    "asqa": ("citation", 50.0, [0.20, 0.25, 0.30]),
    "qampari": ("citation", 50.0, [0.60, 0.66, 0.71]),
}


def make_loss_fn(track, contract, tau):
    if track == "hybridqa":
        return lambda rec, sc: float(sc.get("f1", 0.0) < tau)
    if track == "crag":
        return lambda rec, sc: float(sc.get("label") != "correct")
    return lambda rec, sc: float(sc.get("citation_rec", 0.0) < tau)


def load(track, contract, tau, split):
    td = TrackData(track, split, make_loss_fn(track, contract, tau))
    ld, qids, _, aux = td.build()
    return ld, aux["raw_feats"], aux["latency"]


def augment(raw):
    """Give every rung the retrieval context of all rungs.

    Escalation decisions at rung j are made after rungs 0..j have run, so the
    evidence signals of those rungs are legitimately available; exposing them
    turns the score from a per-rung snapshot into a trajectory feature. The
    later rungs' columns are masked to zero at rung j to keep the score
    causal (no lookahead).
    """
    L, n, d = raw.shape
    out = np.zeros((L, n, d * L))
    for j in range(L):
        for i in range(L):
            if i <= j:
                out[j, :, i * d:(i + 1) * d] = raw[i]
    return out


class CrossFitScorer:
    """Per-rung P(no violation) with out-of-fold predictions on train.

    Cross-fitting matters here: the same split supplies both the candidate
    parameterization and the scores, so in-fold scores would make the train
    frontier look better than anything the certifier can reproduce on fresh
    calibration data.
    """

    def __init__(self, kind="gbdt", folds=4, seed=0, calibrate=False):
        self.kind, self.folds, self.seed = kind, folds, seed
        self.calibrate = calibrate
        self.models = []
        self.cals = []

    def _new(self):
        if self.kind == "lr":
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, C=1.0))
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=1.0,
            early_stopping=False, random_state=self.seed)

    def fit(self, feats, losses):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.isotonic import IsotonicRegression
        L = feats.shape[0]
        self.models, self.cals = [], []
        self.oof = np.zeros((L, feats.shape[1]))
        for j in range(L):
            X, y = feats[j], (1.0 - losses[j])
            if len(np.unique(y)) < 2:
                self.models.append(None)
                self.cals.append(None)
                self.oof[j] = float(y.mean())
                continue
            oof = np.zeros(len(y))
            nf = min(self.folds, int(min(np.bincount(y.astype(int)))))
            if nf >= 2:
                skf = StratifiedKFold(n_splits=nf, shuffle=True,
                                      random_state=self.seed)
                for tr, va in skf.split(X, y):
                    m = self._new()
                    m.fit(X[tr], y[tr])
                    oof[va] = m.predict_proba(X[va])[:, 1]
            else:
                oof[:] = float(y.mean())
            self.oof[j] = oof
            full = self._new()
            full.fit(X, y)
            self.models.append(full)
            if self.calibrate and nf >= 2:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
                iso.fit(oof, y)
                self.cals.append(iso)
            else:
                self.cals.append(None)
        return self

    def score(self, feats, use_oof=False):
        L = feats.shape[0]
        out = np.zeros((L, feats.shape[1]))
        for j in range(L):
            if use_oof:
                out[j] = self.oof[j]
            elif self.models[j] is None:
                out[j] = float(np.mean(self.oof[j]))
            else:
                out[j] = self.models[j].predict_proba(feats[j])[:, 1]
            if self.cals[j] is not None:
                out[j] = self.cals[j].predict(out[j])
        out[L - 1] = np.inf
        return out


def quality_metrics(scores, losses):
    from sklearn.metrics import roc_auc_score
    L = losses.shape[0]
    aucs, briers = [], []
    for j in range(L - 1):          # last rung's score is +inf by construction
        y = 1.0 - losses[j]
        s = scores[j]
        if len(np.unique(y)) < 2 or not np.all(np.isfinite(s)):
            continue
        aucs.append(float(roc_auc_score(y, s)))
        briers.append(float(np.mean((s - y) ** 2)))
    return (float(np.mean(aucs)) if aucs else float("nan"),
            float(np.mean(briers)) if briers else float("nan"))


def frontier_cost_at(cands, ld, alpha):
    """Cheapest candidate whose risk on this split is <= alpha (point estimate).

    This is the frontier position, not a certified quantity: it bounds what
    any certifier could reach if confidence widths were zero, so comparing it
    across score models isolates the score's contribution.
    """
    best = None
    for c in cands:
        loss, cost, _, _ = apply_candidate(c, ld)
        if float(loss.mean()) <= alpha:
            v = float(cost.mean())
            if best is None or v < best:
                best = v
    return best


def main():
    out = {}
    for track, (contract, tau, alphas) in TRACK_CFG.items():
        print(f"\n{'='*78}\n{track} ({contract})\n{'='*78}")
        ld_tr, ft_tr, _ = load(track, contract, tau, "train")
        ld_te, ft_te, _ = load(track, contract, tau, "test")
        aug_tr, aug_te = augment(ft_tr), augment(ft_te)
        rows = []
        variants = [
            ("lr (incumbent)", "lr", ft_tr, ft_te, False),
            ("gbdt", "gbdt", ft_tr, ft_te, False),
            ("gbdt_x", "gbdt", aug_tr, aug_te, False),
            ("gbdt_x+iso", "gbdt", aug_tr, aug_te, True),
        ]
        for name, kind, Xtr, Xte, cal in variants:
            sc = CrossFitScorer(kind=kind, calibrate=cal).fit(Xtr, ld_tr.losses)
            ld_tr.scores = sc.score(Xtr, use_oof=True)   # honest train frontier
            ld_te.scores = sc.score(Xte)
            auc, brier = quality_metrics(ld_te.scores, ld_te.losses)
            cands = build_candidates(ld_tr)
            costs = {a: frontier_cost_at(cands, ld_te, a) for a in alphas}
            rows.append({"scorer": name, "auc": auc, "brier": brier,
                         "frontier_cost": costs})
            cs = "  ".join(f"a={a:.2f}:{(costs[a] or float('nan'))*1000:8.3f}"
                           for a in alphas)
            print(f"  {name:<16} AUC={auc:.4f} Brier={brier:.4f}   {cs}")
        base = rows[0]["frontier_cost"]
        print("  relative frontier cost vs incumbent (lower is better):")
        for r in rows[1:]:
            rel = "  ".join(
                f"a={a:.2f}:{(r['frontier_cost'][a]/base[a]):.3f}x"
                if base[a] and r["frontier_cost"][a] else f"a={a:.2f}:  n/a"
                for a in alphas)
            print(f"    {r['scorer']:<14} {rel}")
        out[track] = rows

    path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "probe_scorer_quality.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
