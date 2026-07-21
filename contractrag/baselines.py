"""Baselines evaluated on the same ladder matrices.

All baselines choose a stopping rung per query (or one rung for the whole
workload); none carries a distribution-free certificate. Comparison axes:
realized contract violation on test, mean cost, latency.

1. Fixed-rung j (fixed light ... fixed heavy pipelines).
2. AdaptiveRAG-style complexity router  (Jeong et al., NAACL'24, adapted):
   train a text classifier to predict the cheapest rung that answers
   correctly; route each query to the predicted rung (no escalation).
3. CA-RAG / utility router (arXiv 2606.02581, adapted): per query pick the
   rung maximizing  q_hat_j(q) - lam_c * cost_j  (workload-tuned lambda).
4. BO-tuned thresholds (Optuna): tune escalation thresholds on the train
   split to minimize cost s.t. empirical risk <= alpha (no certificate,
   direct empirical constraint -> overfits small samples).
5. Abacus (Russo et al., VLDB'26): with a small sampling budget, estimate
   each rung's quality/cost via UCB sampling on calibration data; pick the
   cheapest single rung whose quality estimate satisfies the constraint.
   Per-workload static plan, no per-query adaptivity, no certificate.
6. Conformal-Fixed (TRAQ-flavored): strongest rung for everyone (its risk
   is whatever the strongest plan achieves; cost is maximal).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 2. Adaptive-RAG style router
# ---------------------------------------------------------------------------

class ComplexityRouter:
    """TF-IDF + logistic multi-class on question text -> cheapest-correct rung."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.model = make_pipeline(
            TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
            LogisticRegression(max_iter=2000, C=2.0))

    @staticmethod
    def labels_from_losses(losses):
        """label = cheapest rung with zero loss, else final rung."""
        L, n = losses.shape
        y = np.full(n, L - 1)
        for i in range(n):
            for j in range(L):
                if losses[j, i] == 0:
                    y[i] = j
                    break
        return y

    def fit(self, questions, losses):
        y = self.labels_from_losses(losses)
        # guard: if a class is missing, clip labels
        self.model.fit(questions, y)
        return self

    def route(self, questions):
        return self.model.predict(questions).astype(int)


def eval_routed(ld, lats, rungs_per_query, progressive=False):
    """Evaluate a per-query rung assignment. If progressive, cost accumulates
    over rungs 0..j (escalation semantics); else only the chosen rung's cost
    (router jumps straight to the plan)."""
    idx = np.arange(ld.n)
    rungs = np.clip(rungs_per_query, 0, ld.L - 1)
    loss = ld.losses[rungs, idx]
    if progressive:
        cost = np.cumsum(ld.costs, axis=0)[rungs, idx]
        lat = np.cumsum(lats, axis=0)[rungs, idx]
    else:
        cost = ld.costs[rungs, idx]
        lat = lats[rungs, idx]
    return {"risk": float(loss.mean()), "cost_mean": float(cost.mean()),
            "lat_p50": float(np.percentile(lat, 50)),
            "lat_p95": float(np.percentile(lat, 95)),
            "stop_hist": np.bincount(rungs, minlength=ld.L).tolist()}


# ---------------------------------------------------------------------------
# 3. Utility router
# ---------------------------------------------------------------------------

def utility_route(scores, costs_mean, lam):
    """scores[j,i] = q_hat of stopping at rung j for query i (from RungScorer,
    finite values; final rung inf -> replace with per-rung mean estimate)."""
    L, n = scores.shape
    s = scores.copy()
    if np.isinf(s[L - 1]).any():
        s[L - 1] = np.nanmax(np.where(np.isinf(s), np.nan, s), axis=0)
        s[L - 1] = np.clip(s[L - 1] + 0.05, 0, 1)  # strongest rung prior
    util = s - lam * costs_mean[:, None]
    return util.argmax(axis=0)


def tune_utility_lambda(ld_train, lats_train, scores_train, alpha, grid=None):
    """Pick lambda minimizing cost subject to empirical risk <= alpha on train."""
    grid = grid if grid is not None else np.logspace(-3, 3, 61)
    costs_mean = ld_train.costs.mean(axis=1)
    best, best_cost = None, np.inf
    for lam in grid:
        r = eval_routed(ld_train, lats_train,
                        utility_route(scores_train, costs_mean, lam))
        if r["risk"] <= alpha and r["cost_mean"] < best_cost:
            best, best_cost = lam, r["cost_mean"]
    return best if best is not None else float(grid[0])


# ---------------------------------------------------------------------------
# 4. BO-tuned thresholds (no certificate)
# ---------------------------------------------------------------------------

def bo_thresholds(ld_train, lats_train, alpha, n_trials=200, seed=0):
    import optuna
    from contractrag.calibrate import stop_rung
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    L = ld_train.L
    lo = [float(np.min(ld_train.scores[j][np.isfinite(ld_train.scores[j])], initial=0.0))
          for j in range(L - 1)]
    hi = [float(np.max(ld_train.scores[j][np.isfinite(ld_train.scores[j])], initial=1.0))
          for j in range(L - 1)]

    def objective(trial):
        thr = np.array([trial.suggest_float(f"t{j}", lo[j], hi[j])
                        for j in range(L - 1)])
        stop = stop_rung(ld_train.scores, thr)
        idx = np.arange(ld_train.n)
        risk = ld_train.losses[stop, idx].mean()
        cost = np.cumsum(ld_train.costs, axis=0)[stop, idx].mean()
        if risk > alpha:
            return 1e6 + risk
        return cost

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    thr = np.array([study.best_params[f"t{j}"] for j in range(L - 1)])
    return thr


# ---------------------------------------------------------------------------
# 5. Abacus workload-level constrained selection
# ---------------------------------------------------------------------------

def abacus_select(ld_cal, sample_budget=120, alpha=0.3, seed=0):
    """UCB sampling over rungs on calibration data; then select the
    cheapest rung whose mean-quality estimate satisfies risk <= alpha
    (point estimate, no finite-sample correction)."""
    rng = np.random.default_rng(seed)
    L, n = ld_cal.L, ld_cal.n
    counts = np.zeros(L, dtype=int)
    loss_sums = np.zeros(L)
    # round-robin warmup then UCB on (negative) loss
    order = list(range(L)) * 3
    for j in order:
        i = rng.integers(n)
        loss_sums[j] += ld_cal.losses[j, i]
        counts[j] += 1
    for _ in range(sample_budget - len(order)):
        t = counts.sum()
        ucb = -(loss_sums / np.maximum(counts, 1)) + np.sqrt(2 * np.log(t) / np.maximum(counts, 1))
        j = int(np.argmax(ucb))
        i = rng.integers(n)
        loss_sums[j] += ld_cal.losses[j, i]
        counts[j] += 1
    est_risk = loss_sums / np.maximum(counts, 1)
    mean_cost = ld_cal.costs.mean(axis=1)
    feasible = [j for j in range(L) if est_risk[j] <= alpha]
    if not feasible:
        return L - 1, est_risk
    return int(min(feasible, key=lambda j: mean_cost[j])), est_risk


# backward-compatible alias
abacus_style_select = abacus_select