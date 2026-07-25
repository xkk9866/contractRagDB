"""Per-query plan selection over a physical plan grid, with certification.

Every policy family the incumbent optimizer can express is a stopping rule
along one hand-built ladder: it chooses *when to stop*, never *which plan to
run*. A query optimizer does the opposite -- it picks a different physical plan
per query based on estimates. Once retrieval access paths and generator
implementations are unbundled into a grid, per-query plan selection becomes the
natural policy class, and it is strictly richer than any stopping rule on a
single chain.

Formulation. With per-query cost c(q,p) and violation indicator l(q,p), the
cheapest routing that respects a contract solves

    min_{p(.)}  sum_q c(q, p(q))    s.t.  (1/n) sum_q l(q, p(q)) <= alpha,

a multiple-choice knapsack. Its LP relaxation is separable through a Lagrange
multiplier: the optimal routing has the form

    p_lambda(q) = argmin_p [ c(q,p) + lambda * lhat(q,p) ],

and sweeping lambda traces the entire risk-cost tradeoff, with at most one
fractional query at the optimum. This is why a one-dimensional family of
routers suffices to reach the LP frontier -- no search over the exponential
routing space is needed.

Estimates lhat are fit on TRAIN data from features observable *before* the
chosen plan runs (the cheapest access path's retrieval signals), so routing is
causal and the resulting policy family depends on calibration data not at all.
The family can therefore be walked by the same zero-multiplicity fixed-sequence
certifier used for mixtures, and estimation error costs optimality only: a
misestimated lhat routes badly, but the certificate still holds, exactly as a
wrong cardinality estimate costs a database optimizer performance and never
correctness.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PlanGrid:
    """Execution matrices of P physical plans on n queries.

    losses[p, i] : contract violation indicator of plan p on query i
    costs[p, i]  : execution cost of plan p on query i
    feats[i, :]  : routing features of query i, observable before any plan runs
    names[p]     : human-readable plan identifier
    """
    losses: np.ndarray
    costs: np.ndarray
    feats: np.ndarray
    names: list

    @property
    def P(self) -> int:
        return self.losses.shape[0]

    @property
    def n(self) -> int:
        return self.losses.shape[1]

    def subset(self, plan_idx):
        return PlanGrid(self.losses[plan_idx], self.costs[plan_idx], self.feats,
                        [self.names[p] for p in plan_idx])


class PlanRiskModel:
    """Per-plan violation probability from pre-execution routing features.

    One calibrated linear model per plan. Linear rather than boosted on
    purpose: with a few hundred training queries per plan, cross-fitted
    gradient boosting measurably overfits on the smaller tracks while logistic
    regression does not, and the router only needs the ranking of plans to be
    right, not the absolute probabilities.
    """

    def __init__(self, seed: int = 0):
        self.models = []
        self.base = []
        self.seed = seed

    def fit(self, grid: PlanGrid):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.models, self.base = [], []
        for p in range(grid.P):
            y = grid.losses[p]
            self.base.append(float(y.mean()))
            if len(np.unique(y)) < 2:
                self.models.append(None)
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000, C=1.0))
            m.fit(grid.feats, y)
            self.models.append(m)
        return self

    def predict(self, feats: np.ndarray) -> np.ndarray:
        """(P, n) predicted violation probabilities."""
        out = np.zeros((len(self.models), feats.shape[0]))
        for p, m in enumerate(self.models):
            out[p] = self.base[p] if m is None else m.predict_proba(feats)[:, 1]
        return out


@dataclass
class RouterPolicy:
    """Lagrangian router: pick the plan minimizing cost + lambda * risk."""
    lam: float
    plan_cost: np.ndarray          # per-plan mean cost, from TRAIN
    risk_model: PlanRiskModel

    def route(self, feats: np.ndarray) -> np.ndarray:
        lhat = self.risk_model.predict(feats)
        score = self.plan_cost[:, None] + self.lam * lhat
        return np.argmin(score, axis=0)

    def realize(self, grid: PlanGrid):
        """(loss, cost) vectors of this router on a split."""
        choice = self.route(grid.feats)
        idx = np.arange(grid.n)
        return grid.losses[choice, idx], grid.costs[choice, idx], choice

    def describe(self) -> str:
        return f"router(lam={self.lam:.4g})"


def build_router_family(grid_train: PlanGrid, num_lam: int = 40,
                        lam_lo: float = 1e-5, lam_hi: float = 1e2):
    """Lambda-indexed router family, parameterized on TRAIN data only.

    The multiplier range is set from the cost spread so that the extremes are
    the always-cheapest and always-safest routings; intermediate values trace
    the frontier. Returned in risk-ascending order, which is the fixed-sequence
    test order.
    """
    rm = PlanRiskModel().fit(grid_train)
    plan_cost = grid_train.costs.mean(axis=1)
    spread = float(plan_cost.max() - plan_cost.min()) or 1.0
    lams = np.concatenate([[0.0],
                           np.logspace(np.log10(lam_lo), np.log10(lam_hi),
                                       num_lam) * spread])
    fam = []
    for lam in lams:
        pol = RouterPolicy(float(lam), plan_cost, rm)
        loss, cost, _ = pol.realize(grid_train)
        fam.append((pol, float(loss.mean()), float(cost.mean())))
    fam.sort(key=lambda t: (t[1], t[2]))
    return fam


# ---------------------------------------------------------------------------
# unified candidate space: fixed plans + routers, with mixtures
# ---------------------------------------------------------------------------

@dataclass
class GridCandidate:
    """A deterministic policy over the grid: one fixed plan or one router."""
    kind: str                 # 'fixed' | 'router'
    plan: int = -1
    router: RouterPolicy = None
    train_risk: float = 0.0
    train_cost: float = 0.0

    def realize(self, grid: PlanGrid):
        if self.kind == "fixed":
            return grid.losses[self.plan], grid.costs[self.plan]
        loss, cost, _ = self.router.realize(grid)
        return loss, cost

    def describe(self) -> str:
        return (f"plan[{self.plan}]" if self.kind == "fixed"
                else self.router.describe())


def build_grid_candidates(grid_train: PlanGrid, num_lam: int = 40
                          ) -> list[GridCandidate]:
    """Fixed plans plus the Lagrangian router family, annotated on TRAIN."""
    cands = []
    for p in range(grid_train.P):
        c = GridCandidate("fixed", plan=p)
        loss, cost = c.realize(grid_train)
        c.train_risk, c.train_cost = float(loss.mean()), float(cost.mean())
        cands.append(c)
    for pol, r, cst in build_router_family(grid_train, num_lam=num_lam):
        cands.append(GridCandidate("router", router=pol, train_risk=r,
                                   train_cost=cst))
    return cands


def candidate_matrices(cands: list[GridCandidate], grid: PlanGrid):
    """(K, n) loss and cost matrices of every candidate on a split."""
    K = len(cands)
    L = np.zeros((K, grid.n))
    C = np.zeros((K, grid.n))
    for k, c in enumerate(cands):
        loss, cost = c.realize(grid)
        L[k], C[k] = loss, cost
    return L, C
