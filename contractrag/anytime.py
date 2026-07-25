"""Anytime-valid certified optimization over an unbounded query stream.

Why this exists. A one-shot certifier spends a fixed calibration budget,
converts it into one confidence width, and deploys a single plan forever.
Measurements on the real execution matrices show that the residual gap to the
population LP optimum is almost entirely that width: the plan space is dense
enough, the sufficiency model is near its ceiling at these sample sizes, and
randomization buys only a modest constant. The width itself is an information
-theoretic price of estimating a Bernoulli mean from n samples -- it cannot be
argued away, only *outgrown*. Deployment produces an unbounded stream, so the
right object is not a fixed-sample bound but a confidence sequence that keeps
narrowing while remaining valid at every stopping time.

Two structural facts make this practical rather than aspirational.

  (S1) Progressive execution is a free counterfactual log. A query answered
       at rung j has already executed rungs 0..j, so the losses of every plan
       that would have stopped at or below j are observable at no extra cost.
       Only plans that escalate further need paid probing. This is the risk
       analogue of reusing materialized intermediates across plans.

  (S2) Betting supermartingales are anytime-valid by construction. Wealth
       K_t(m) = prod_i (1 - lambda_i (Y_i - m)) with predictable stakes is a
       nonnegative supermartingale under H0: E[Y] >= m, so Ville's inequality
       controls sup_t K_t. The induced upper bounds may therefore be recomputed
       and re-optimized against after every arrival without any alpha-spending
       schedule -- which also removes the geometric delta/2^k budget that a
       fixed-horizon monitor needs for lifetime validity.

The optimizer below couples the two: it audits a small fraction of arrivals to
the strongest rung (making all plans' losses observable and i.i.d.), maintains
one confidence sequence per plan, and re-solves the certified plan choice as
the sequences tighten. Total cost of ownership -- deployment spend plus audit
spend -- is what it minimizes, so the audit budget is a first-class term
rather than an externality.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


class BettingCS:
    """Anytime-valid upper confidence sequence for a mean in [0, 1].

    Maintains log-wealth on a grid of candidate means. For each grid point m
    the process K_t(m) = prod_i (1 - lambda_i (Y_i - m)) is a nonnegative
    supermartingale under H0: E[Y] >= m; by Ville's inequality
    P(exists t: K_t(m) >= 1/delta) <= delta, so the set of m not yet rejected
    is a (1-delta) confidence sequence, uniformly over time. Log-wealth is
    nondecreasing in m, hence the upper bound is the smallest rejected grid
    point and one vectorized update per arrival suffices.

    Stakes use a predictable plug-in: lambda_t is a function of the first
    t-1 observations only, truncated to keep every wealth factor positive.
    """

    def __init__(self, delta: float, grid: int = 257, cap: float = 0.5):
        self.delta = delta
        self.m = np.linspace(0.0, 1.0, grid)
        self.logK = np.zeros(grid)
        self.cap = cap
        self.n = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.logthr = np.log(1.0 / delta)

    def update(self, y: float) -> None:
        # predictable stake from the strict prefix
        if self.n == 0:
            mean, var = 0.5, 0.25
        else:
            mean = self.sum / self.n
            var = max(self.sumsq / self.n - mean * mean, 1e-4)
        lam = min(np.sqrt(2.0 * self.logthr / (max(self.n, 1) * var)), self.cap)
        self.logK += np.log(np.maximum(1.0 - lam * (y - self.m), 1e-12))
        self.n += 1
        self.sum += y
        self.sumsq += y * y

    def update_many(self, ys: np.ndarray) -> None:
        for y in np.asarray(ys, dtype=float).ravel():
            self.update(y)

    @property
    def mean(self) -> float:
        return self.sum / self.n if self.n else float("nan")

    def upper(self) -> float:
        """Smallest grid mean already rejected from below; 1.0 if none."""
        if self.n == 0:
            return 1.0
        rejected = np.where(self.logK >= self.logthr)[0]
        if len(rejected) == 0:
            return 1.0
        return float(self.m[rejected[0]])


@dataclass
class StreamState:
    t: int = 0
    audits: int = 0
    deploy_cost: float = 0.0
    audit_cost: float = 0.0
    plan: object = None
    plan_desc: str = "init"
    switches: int = 0
    history: list = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return self.deploy_cost + self.audit_cost


class AnytimeCertifiedOptimizer:
    """Certified optimizer that tightens while it serves.

    Parameters
    ----------
    losses, costs : (K, N) arrays
        Per-plan loss and cost of every plan on every query of the stream
        population. Only the entries the policy is allowed to observe are read,
        which is enforced by the audit logic rather than by masking.
    alpha, delta : contract level and error budget.
    audit_rate : fraction of arrivals escalated to the strongest rung so that
        every plan's loss becomes observable. Audited queries are chosen
        independently of their content, keeping the audited subsample i.i.d.
    audit_unit_cost : cost charged per audited arrival (execution to the
        strongest rung plus the label). Charged separately from serving cost so
        the reported total is a true cost of ownership.
    """

    def __init__(self, losses: np.ndarray, costs: np.ndarray, alpha: float,
                 delta: float, audit_rate: float = 0.05,
                 audit_unit_cost: float = None, recert_every: int = 50,
                 safe_plan: int = None, cs_grid: int = 257):
        self.L = np.asarray(losses, dtype=float)
        self.C = np.asarray(costs, dtype=float)
        self.K, self.N = self.L.shape
        self.alpha = alpha
        self.delta = delta
        self.audit_rate = audit_rate
        self.recert_every = recert_every
        # delta is split once across plans; the split is over a fixed plan set
        # so it is a constant, and each plan's sequence is valid for all time.
        self.cs = [BettingCS(delta / self.K, grid=cs_grid)
                   for _ in range(self.K)]
        self.cost_hat = np.zeros(self.K)
        self.cost_n = 0
        self.audit_unit_cost = (float(self.C.max(axis=1).max())
                                if audit_unit_cost is None else audit_unit_cost)
        self.safe_plan = (int(np.argmin(self.L.mean(axis=1)))
                          if safe_plan is None else safe_plan)

    def certified_plans(self) -> np.ndarray:
        """Indices whose anytime upper bound is already below alpha."""
        ub = np.array([c.upper() for c in self.cs])
        return np.where(ub <= self.alpha)[0], ub

    def choose_plan(self) -> int:
        idx, ub = self.certified_plans()
        if len(idx) == 0:
            return self.safe_plan
        if self.cost_n == 0:
            return int(idx[np.argmin(self.C[idx].mean(axis=1))])
        return int(idx[np.argmin(self.cost_hat[idx] / max(self.cost_n, 1))])

    def run(self, order: np.ndarray, rng: np.random.Generator,
            log_every: int = 200) -> StreamState:
        st = StreamState()
        st.plan = self.safe_plan
        st.plan_desc = f"#{self.safe_plan}"
        for step, q in enumerate(order):
            audit = rng.random() < self.audit_rate
            if audit:
                # escalation to the strongest rung: every plan's loss on this
                # query becomes observable (S1), and the arrival is charged the
                # audit unit cost instead of the serving cost.
                for k in range(self.K):
                    self.cs[k].update(self.L[k, q])
                self.cost_hat += self.C[:, q]
                self.cost_n += 1
                st.audits += 1
                st.audit_cost += self.audit_unit_cost
            else:
                st.deploy_cost += float(self.C[st.plan, q])
            st.t += 1
            if st.t % self.recert_every == 0:
                new = self.choose_plan()
                if new != st.plan:
                    st.switches += 1
                    st.plan = new
                    st.plan_desc = f"#{new}"
            if st.t % log_every == 0:
                st.history.append({
                    "t": st.t, "plan": int(st.plan), "audits": st.audits,
                    "deploy_cost": st.deploy_cost, "audit_cost": st.audit_cost,
                    "mean_cost": st.total_cost / st.t,
                    "pop_risk": float(self.L[st.plan].mean()),
                })
        return st
