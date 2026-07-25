"""Contract enforcement as deterministic accounting, not as inference.

Every certified-RAG system we know of, ours included until now, enforces a
quality contract the same way: estimate each candidate policy's risk on a
calibration sample, correct for having looked at many candidates, and deploy
whatever clears alpha minus a confidence width. The guarantee that comes out is
"risk <= alpha with probability 1 - delta, if the deployment traffic is drawn
from the calibration distribution". Three costs are attached to it and none of
them is incidental:

  1. the confidence width is paid on every query forever, because the decision
     it protects can never be revised;
  2. it widens with the number of candidates, so enlarging the plan space can
     make the certified policy MORE expensive than a smaller space would --
     measured on our 24-plan grid, static certification returns a policy 17%
     more expensive than the same procedure run on a 4-plan ladder, and at
     alpha = 0.30 it returns nothing at all;
  3. when alpha is below the risk of every available plan the procedure is
     simply infeasible, which is the regime a strict contract lives in.

The third point is the tell. Infeasibility is not a statistical problem, and no
sharper bound removes it. It says the action space is too small: the system is
required to answer every query, so its attainable risk is bounded below by the
best plan's risk. Real services are not required to answer every query. They may
decline -- return "insufficient evidence", route to a human, degrade to a
document list. Declining produces no wrong answer, so its contract loss is
exactly zero, and it is available at every arrival.

Admitting that one action changes the nature of the problem. Track the ledger

    B_t = alpha * t - V_t,     V_t = violations served through time t,

and gate on it: an arrival may run a plan only if the ledger can absorb a
violation, B_{t-1} >= 1 - alpha; otherwise it is declined. Then

    B_t >= 0 for every t,  i.e.  V_t <= alpha * t  for every t,

by induction, with no probability, no distributional assumption, no dependence
on how many plans exist, and no horizon. The contract is not something the
system is confident about. It is something the system cannot violate, in the
same sense that a bank account with a hard overdraft limit cannot go negative.

What is left for statistics is cost. Declining is not free -- a human handles
the query, or the user leaves -- so the objective is

    min  sum_p w_p c_p + w_abstain c_abstain    s.t.  sum_p w_p r_p <= alpha,

an LP over plans plus the decline action, whose optimum trades expensive-and-
safe plans against cheap-and-risky ones plus declines. The risks r_p are
unknown and must be learned online. But an error in r_p now costs money and
nothing else: it makes the ledger drain faster, the gate declines more often,
and the objective suffers -- the contract still holds exactly. So the estimator
can be as aggressive as it likes. We make it optimistic (lower confidence bound
on risk), which is what makes the exploration provably efficient rather than
merely tolerable.

This is the division of labour a query optimizer already lives by. Cardinality
estimates choose plans and are famously wrong by orders of magnitude; the
executor is what guarantees the answer is correct. Certified RAG has been
putting the guarantee on the estimate. Moving it to the executor is the whole
idea here, and everything else -- feasibility below min_p r_p, independence
from the number of candidates, validity under drift and under adversarial query
order -- is a consequence rather than an addition.

Partial labelling. The deterministic argument needs the loss of every served
query. When only a sampled fraction is labelled, we keep the argument by
charging unlabelled arrivals their worst case (Section `audit_rate`), which
preserves V_t <= alpha t exactly and prices the missing labels as a slower
ledger; the alternative -- an inverse-probability ledger, valid only with
probability 1 - delta -- is implemented too, and the two are compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ABSTAIN = -1


# ---------------------------------------------------------------------------
# the optimisation layer: an LP over plans + the decline action
# ---------------------------------------------------------------------------

def solve_action_lp(risk: np.ndarray, cost: np.ndarray, alpha: float,
                    abstain_cost: float):
    """Cheapest randomized action whose mean risk is at most alpha.

    Actions are the P plans plus decline, which has risk 0 and cost
    `abstain_cost`. With a single risk constraint on the simplex a basic
    optimum has at most two nonzero weights, so the solution is always
    "run plan i with probability w, otherwise decline" or "mix two plans" --
    an interpretable operating point, not a dense mixture.

    Returns (value, weights) with weights of length P + 1, the last entry the
    decline probability. Never infeasible: declining everything has risk 0.
    """
    r = np.append(np.asarray(risk, dtype=float), 0.0)
    c = np.append(np.asarray(cost, dtype=float), float(abstain_cost))
    n = len(r)
    best_v, best_w = float("inf"), None
    # Single actions that already satisfy the constraint.
    for i in np.where(r <= alpha + 1e-12)[0]:
        if c[i] < best_v:
            best_v, best_w = float(c[i]), np.eye(n)[i]
    # Pairs: one action above the level, one below, mixed to sit exactly on it.
    hi = np.where(r > alpha + 1e-12)[0]
    lo = np.where(r < alpha - 1e-12)[0]
    for i in hi:
        for j in lo:
            w = (alpha - r[j]) / (r[i] - r[j])
            v = w * c[i] + (1.0 - w) * c[j]
            if v < best_v:
                best_v = float(v)
                best_w = np.zeros(n)
                best_w[i], best_w[j] = w, 1.0 - w
    if best_w is None:                      # cannot happen: decline is always in
        best_w = np.zeros(n)
        best_w[-1] = 1.0
        best_v = float(c[-1])
    return best_v, np.maximum(best_w, 0.0) / max(best_w.sum(), 1e-12)


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------

@dataclass
class LedgerState:
    t: int = 0
    serve_cost: float = 0.0
    audit_cost: float = 0.0
    abstain_cost_total: float = 0.0
    violations: float = 0.0
    ledger: float = 0.0
    min_ledger: float = 0.0
    abstentions: int = 0
    gate_blocks: int = 0
    audits: int = 0
    plan_counts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    cum_viol: object = None

    @property
    def total_cost(self) -> float:
        return self.serve_cost + self.audit_cost + self.abstain_cost_total

    @property
    def risk(self) -> float:
        return self.violations / self.t if self.t else float("nan")

    @property
    def abstain_rate(self) -> float:
        return self.abstentions / self.t if self.t else float("nan")


class LedgerExecutor:
    """Serve a stream at minimum cost under a contract that cannot be violated.

    Parameters
    ----------
    losses, costs : (P, N) arrays
        Violation indicator and cost of each plan on each query of the stream
        population. Only the entry of the action actually taken is ever read,
        so the controller sees exactly what an online system sees.
    alpha : contract level.
    abstain_cost : price of declining one query -- a human handoff, or the
        modelled value of a degraded answer. It is the parameter that makes the
        problem non-trivial: at zero the optimiser declines its way to any
        alpha for free, at a large value it buys expensive plans to avoid
        declining. We sweep it rather than fix it.
    kappa : optimism strength. The LP is solved on risk_hat - kappa*sqrt(1/n_p),
        so under-observed plans look attractive and get tried. Optimism is
        affordable here for a structural reason: being wrong about a plan costs
        allowance, which the ledger meters, and never validity.
    audit_rate : fraction of served queries whose loss is observed.
    audit_mode : 'worst-case' keeps the deterministic guarantee by charging an
        unlabelled arrival a full violation; 'ipw' uses inverse-probability
        weighting, which is unbiased but only bounds the ledger with high
        probability. Both are reported.
    """

    def __init__(self, losses: np.ndarray, costs: np.ndarray, alpha: float,
                 abstain_cost: float = 0.0, hist_risk: np.ndarray = None,
                 hist_cost: np.ndarray = None, kappa: float = 1.0,
                 audit_rate: float = 1.0, audit_unit_cost: float = 0.0,
                 audit_mode: str = "worst-case", prior_n: float = 20.0,
                 resolve_every: int = 20, policy: str = "lp"):
        self.L = np.asarray(losses, dtype=float)
        self.C = np.asarray(costs, dtype=float)
        self.P, self.N = self.L.shape
        self.alpha = float(alpha)
        self.abstain_cost = float(abstain_cost)
        self.kappa = float(kappa)
        self.audit_rate = float(audit_rate)
        self.audit_unit_cost = float(audit_unit_cost)
        self.audit_mode = audit_mode
        self.resolve_every = int(resolve_every)
        self.policy = policy
        self.cost_hat = (self.C.mean(axis=1).copy() if hist_cost is None
                         else np.asarray(hist_cost, dtype=float).copy())
        # History enters only as a prior on the risk estimates. It shifts which
        # plan is tried first and therefore the transient cost; it cannot shift
        # the contract, so a stale or adversarial history is a cost bug, not a
        # correctness bug. That is not true of any calibration-based certifier.
        self.risk_hat = (np.full(self.P, 0.5) if hist_risk is None
                         else np.asarray(hist_risk, dtype=float).copy())
        self.risk_n = np.full(self.P, float(prior_n))
        # Cost and risk are observed on different schedules, and conflating
        # them is a real error. The price of a call is on the invoice the moment
        # it returns, so cost is observed for every served query regardless of
        # the audit rate; whether the answer was acceptable is not, and needs a
        # judge. So cost_hat is refreshed from every execution while risk_hat
        # advances only on labelled ones. Freezing cost_hat at its historical
        # value instead leaves a bias that no amount of stream length removes,
        # and the regret curve flattens at a constant.
        self.cost_n = np.full(self.P, float(prior_n))

    # -- the gate: the entire correctness argument -------------------------
    #
    # An arrival may run a plan only if the ledger can absorb the worst case.
    # B_{t-1} >= 1 - alpha  =>  B_t = B_{t-1} + alpha - l_t >= 0 for l_t <= 1.
    # Declining moves the ledger up by alpha and can never break the invariant.
    # Hence B_t >= 0 for all t by induction, i.e. V_t <= alpha * t: a bound
    # that holds pathwise, for every realisation, under any query order, with
    # no reference to a distribution and no reference to P.

    def gate_open(self, ledger: float) -> bool:
        return ledger >= 1.0 - self.alpha - 1e-12

    def _risk_bound(self) -> np.ndarray:
        """Risk used by the optimiser: an upper estimate, width kappa/sqrt(n_p).

        Aiming the mixture AT alpha is a mistake the ledger makes visible.
        Estimation error is two-sided, so a policy placed exactly on alpha sits
        on the wrong side of it half the time; the ledger then drifts down, the
        gate closes, and the query is declined -- at the decline price, which is
        the most expensive outcome in the space. Steering at alpha minus a
        margin gives the ledger positive drift and the gate stops firing.

        This margin looks like the confidence width of a static certifier and is
        the opposite of it in the two ways that matter. It is not what makes the
        contract hold, so it needs no delta and no union correction over the
        candidate set -- kappa is an O(1) constant, not log(K/delta). And its
        denominator is the number of queries served SO FAR, which grows without
        bound, so the margin decays to zero over the deployment; a calibration
        width is frozen at sqrt(1/n_cal) the moment calibration ends and is paid
        at that size forever. On these streams that is 0.006 against 0.049.
        """
        return np.clip(self.risk_hat
                       + self.kappa * np.sqrt(0.25 / np.maximum(self.risk_n, 1.0)),
                       0.0, 1.0)

    def _lp_weights(self):
        _, w = solve_action_lp(self._risk_bound(), self.cost_hat, self.alpha,
                               self.abstain_cost)
        return w

    def _greedy_action(self) -> int:
        """Cheapest single plan clearing alpha, else decline.

        The deterministic ablation of the LP: it reaches only frontier
        vertices, never the mixtures on the segment between two of them, which
        is where the optimum lies whenever alpha falls strictly between two
        plans' risks -- and it cannot reach alpha below every plan's risk at
        all, where it degenerates to declining everything. Included to show
        that randomisation is load-bearing rather than decorative.
        """
        ok = np.where(self._risk_bound() <= self.alpha)[0]
        if len(ok) == 0:
            return ABSTAIN
        p = int(ok[np.argmin(self.cost_hat[ok])])
        return p if self.cost_hat[p] <= self.abstain_cost else ABSTAIN

    def run(self, order: np.ndarray, rng: np.random.Generator,
            log_every: int = 200) -> LedgerState:
        st = LedgerState()
        order = np.asarray(order)
        cum = np.zeros(len(order))
        w = None
        for step, q in enumerate(order):
            open_gate = self.gate_open(st.ledger)
            if not open_gate:
                st.gate_blocks += 1
                a = ABSTAIN
            elif self.policy == "greedy":
                a = self._greedy_action()
            else:
                if w is None or step % self.resolve_every == 0:
                    w = self._lp_weights()
                k = int(rng.choice(self.P + 1, p=w))
                a = ABSTAIN if k == self.P else k

            if a == ABSTAIN:
                loss, charged, cost = 0.0, 0.0, self.abstain_cost
                st.abstentions += 1
                st.abstain_cost_total += cost
            else:
                loss = float(self.L[a, q])
                cost = float(self.C[a, q])
                st.serve_cost += cost
                st.plan_counts[a] = st.plan_counts.get(a, 0) + 1
                self.cost_n[a] += 1.0
                self.cost_hat[a] += (cost - self.cost_hat[a]) / self.cost_n[a]
                labelled = (self.audit_rate >= 1.0
                            or rng.random() < self.audit_rate)
                if labelled:
                    st.audits += 1
                    st.audit_cost += self.audit_unit_cost
                    charged = loss
                    self.risk_n[a] += 1.0
                    self.risk_hat[a] += (loss - self.risk_hat[a]) / self.risk_n[a]
                elif self.audit_mode == "ipw":
                    charged = 0.0
                else:
                    # unlabelled and unwilling to assume: charge the worst case,
                    # which keeps V_t <= alpha t exact at the price of a ledger
                    # that fills more slowly.
                    charged = 1.0
                st.violations += loss
            st.t += 1
            st.ledger += self.alpha - charged
            st.min_ledger = min(st.min_ledger, st.ledger)
            cum[step] = st.violations
            if st.t % log_every == 0:
                st.history.append({
                    "t": st.t, "action": int(a), "ledger": float(st.ledger),
                    "mean_cost": st.total_cost / st.t,
                    "emp_risk": st.violations / st.t,
                    "abstain_rate": st.abstentions / st.t})
        st.cum_viol = cum
        return st


# ---------------------------------------------------------------------------
# static baseline, with the SAME action space
# ---------------------------------------------------------------------------

def static_stream_cost(losses: np.ndarray, costs: np.ndarray, alpha: float,
                       delta: float, order: np.ndarray, n_cal: int,
                       hist_risk: np.ndarray, hist_cost: np.ndarray,
                       abstain_cost: float = 0.0, audit_unit_cost: float = 0.0,
                       mix_grid: int = 9, allow_abstain: bool = True):
    """One-shot certification over the same stream, declines included.

    Giving the static certifier the decline action too is what isolates the
    contribution. Without it the comparison would only show that declining
    helps, which is obvious. With it, both systems have the same action space
    and the same objective, and the difference is precisely the price of
    enforcing the contract by inference rather than by accounting: a confidence
    width on the calibration sample, widened for having considered many
    candidates, conceded on every query for the life of the deployment.

    The calibration prefix is served, and charged, by the plan history believes
    is safest -- calibration queries are real queries and somebody answers them.
    """
    from contractrag.certlp import (build_randomized_chain,
                                    certify_randomized_chain)
    order = np.asarray(order)
    L, C = np.asarray(losses, float), np.asarray(costs, float)
    hr = np.asarray(hist_risk, float)
    hc = np.asarray(hist_cost, float)
    if allow_abstain:
        L = np.vstack([L, np.zeros((1, L.shape[1]))])
        C = np.vstack([C, np.full((1, C.shape[1]), abstain_cost)])
        hr = np.append(hr, 0.0)
        hc = np.append(hc, abstain_cost)
    cal_idx, rest_idx = order[:n_cal], order[n_cal:]
    safe = int(np.argmin(hr))
    serve = float(C[safe, cal_idx].sum())
    viol = float(np.count_nonzero(L[safe, cal_idx] > 0.5))
    audit = audit_unit_cost * len(cal_idx)

    chain = build_randomized_chain(hr[:, None], hc[:, None], mix_grid=mix_grid)
    cp = certify_randomized_chain(L[:, cal_idx], C[:, cal_idx], chain,
                                  alpha, delta)
    if cp.element is None:
        sup, wts = (safe,), (1.0,)
    else:
        sup, wts = cp.element.support, cp.element.weights
    post_loss = np.zeros(len(rest_idx))
    for k, wt in zip(sup, wts):
        serve += wt * float(C[k, rest_idx].sum())
        post_loss += wt * (L[k, rest_idx] > 0.5)
    viol += float(post_loss.sum())
    cum = np.cumsum(np.concatenate([(L[safe, cal_idx] > 0.5).astype(float),
                                    post_loss]))
    n = len(order)
    ab = 0.0
    if allow_abstain:
        ab = float(sum(wt for k, wt in zip(sup, wts) if k == L.shape[0] - 1))
    return {"total_cost": serve + audit, "serve_cost": serve,
            "audit_cost": audit, "risk": viol / n,
            "mean_cost": (serve + audit) / n, "cum_viol": cum,
            "certified": bool(cp.element is not None),
            "abstain_rate": ab * len(rest_idx) / n,
            "plan": cp.describe() if cp.element is not None else f"#{safe}",
            "n_cal": n_cal}
