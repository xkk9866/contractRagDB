"""Certified adaptive execution: the contract as a ledger, not a hypothesis test.

Where the static formulation runs out. A one-shot certifier splits its data,
turns a calibration sample into one confidence width w = Theta(sqrt(log(1/delta)
/ n_cal)), and deploys a plan whose estimated risk clears alpha - w. Three
probes on the real execution matrices show the residual gap to the population
optimum is that width and nothing else:

  * tighter bounds cannot help -- Hoeffding-Bentkus half-widths are already
    0.015-0.049 and betting bounds match them;
  * a control variate cannot help -- fitted on retrieval features, a predictor
    of per-query violation reaches correlation 0.12-0.24 on the plans a
    certifier actually deploys, a variance ratio of 0.97-1.02;
  * a richer search cannot help -- the fixed-sequence walk is never truncated on
    these candidate sets, and both Bonferroni and graph-structured budget
    recycling come out strictly worse.

So the width is irreducible at fixed n_cal. What is reducible is *paying it at
all*. The width is the price of a decision that cannot be revised: having chosen
a plan from a calibration sample, the certifier must be right the first time, so
it must be conservative by the full estimation error. But deployment is a
stream. A wrong choice is observable within a few hundred queries and can be
corrected, and the only thing that cannot be undone is a violation already
served.

That changes what has to be estimated -- namely nothing. Track the ledger

    B_t = alpha * t - V_t,      V_t = number of violations served so far,

the contract allowance still unspent. Optimise cost freely while the ledger holds
a buffer h; steer toward safer plans as the buffer is consumed. Bounding the
overdraft is the only place probability enters, and the size of the buffer is
where the argument becomes interesting.

A martingale bound such as Freedman's would give an overdraft of order sqrt(t),
which is far too pessimistic: while the controller is steering safe, the ledger
has POSITIVE drift eps = alpha - r_f, where r_f is the fallback plan's risk. A
random walk with positive drift has a maximum drawdown that does not grow with
time -- geometric, exactly as the queue length of a stable server does. Applying
Ville's inequality to the exponential supermartingale exp(-theta * B_t) with
theta matched to the drift gives, uniformly over an unbounded horizon,

    P( exists t : B_t < 0 )  <=  delta      whenever   h >= log(1/delta) / theta,
    theta  =  2 * eps / (v + eps),   v an upper bound on the loss variance.

So the buffer is a CONSTANT, not a function of t. Its cost is h/t, which decays
like 1/t. Static certification instead concedes a confidence width
Theta(sqrt(log(1/delta)/n_cal)) fixed the moment the calibration set is fixed and
paid on every query forever. That is the whole comparison: a concession that
vanishes versus one that does not.

eps is unknown, and assuming it would smuggle the guarantee back into an
estimate. Instead the fallback plan's own anytime confidence sequence supplies an
upper bound on r_f, so eps_t = alpha - hi_f(t) is a valid lower bound on the
drift at every time. Before anything is observed hi_f = 1, eps_t <= 0, the buffer
is infinite and the controller stays safe -- the correct behaviour on a cold
start, obtained without a special case. As observations accumulate hi_f tightens,
the buffer shrinks, and the optimiser is progressively released.

What the estimates are for. Risk estimates rhat still exist -- they decide which
plan to try when the ledger is in credit, and a Lagrangian multiplier prices risk
against cost so the greedy choice traces the LP frontier. But they carry no part
of the guarantee. A bad estimate spends the allowance on a plan that does not
deserve it and costs money; it can never breach the contract. This is the same
division of labour a database optimizer lives by: cardinality estimates decide
plans and are routinely wrong, while correctness rests on the executor. Prior
certified-RAG work puts the guarantee on the estimate and therefore has to pay
for its error forever.

Consequences the static formulation cannot have:
  * no calibration split is set aside, so no query is served by a needlessly
    expensive plan merely to buy statistics;
  * the per-query cost converges to the population LP optimum -- the mix of
    greedy and fallback service that keeps B_t near zero IS the optimal
    randomized plan, discovered by feedback rather than by solving for it;
  * validity is anytime and horizon-free, so the geometric alpha-spending
    schedule a fixed-horizon monitor needs disappears, and drift is handled by
    the mechanism that handles everything else -- the ledger notices.

The one assumption. The fallback plan must itself satisfy the contract, since it
is what the ledger falls back on; if no plan does, the contract is infeasible and
no method can honour it. `feasibility_cs` audits that assumption online with a
betting confidence sequence, so the assumption is testable rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# anytime-valid confidence sequence, used to AUDIT the fallback assumption
# ---------------------------------------------------------------------------

class TwoSidedCS:
    """Betting confidence sequence for a mean in [0, 1], valid at all times.

    For each grid point m, K_t(m) = prod_i (1 - lam_i (Y_i - m)) is a
    nonnegative supermartingale under H0: E[Y] >= m, and its mirror handles
    H0: E[Y] <= m; Ville's inequality bounds sup_t K_t, so spending delta/2 per
    side gives a two-sided sequence uniformly over time. Stakes are predictable
    plug-ins from the strict prefix, truncated to keep the wealth positive.

    This is not on the critical path of the guarantee -- the ledger is. It is
    used to report whether the fallback plan is provably feasible, which is the
    single assumption the ledger rests on.
    """

    def __init__(self, delta: float, grid: int = 129, cap: float = 0.5):
        d = max(delta / 2.0, 1e-12)
        self.m = np.linspace(0.0, 1.0, grid)
        self.logKu = np.zeros(grid)
        self.logKl = np.zeros(grid)
        self.logthr = float(np.log(1.0 / d))
        self.cap = cap
        self.n = 0
        self.s = 0.0
        self.ss = 0.0

    def update(self, y: float) -> None:
        if self.n == 0:
            mean, var = 0.5, 0.25
        else:
            mean = self.s / self.n
            var = max(self.ss / self.n - mean * mean, 1e-3)
        lam = min(float(np.sqrt(2.0 * self.logthr / (max(self.n, 1) * var))),
                  self.cap)
        self.logKu += np.log(np.maximum(1.0 - lam * (y - self.m), 1e-12))
        self.logKl += np.log(np.maximum(1.0 + lam * (y - self.m), 1e-12))
        self.n += 1
        self.s += y
        self.ss += y * y

    def update_many(self, ys) -> None:
        for y in np.asarray(ys, dtype=float).ravel():
            self.update(float(y))

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n else float("nan")

    def upper(self) -> float:
        if self.n == 0:
            return 1.0
        rej = np.where(self.logKu >= self.logthr)[0]
        return float(self.m[rej[0]]) if len(rej) else 1.0

    def lower(self) -> float:
        if self.n == 0:
            return 0.0
        rej = np.where(self.logKl >= self.logthr)[0]
        return float(self.m[rej[-1]]) if len(rej) else 0.0


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------

@dataclass
class CAEState:
    t: int = 0
    serve_cost: float = 0.0
    audit_cost: float = 0.0
    violations: float = 0.0
    ledger: float = 0.0
    min_ledger: float = 0.0
    fallbacks: int = 0
    explores: int = 0
    audits: int = 0
    lam: float = 0.0
    plan_counts: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    cum_viol: object = None
    fallback_hi: float = 1.0
    fallback_certified_at: int = -1

    @property
    def total_cost(self) -> float:
        return self.serve_cost + self.audit_cost

    @property
    def risk(self) -> float:
        return self.violations / self.t if self.t else float("nan")


class CertifiedAdaptiveExecutor:
    """Serve a stream at minimum cost under a deterministic anytime contract.

    Parameters
    ----------
    losses, costs : (P, N)
        Violation indicator and cost of every plan on every query of the stream
        population. Only the column of the plan actually run is read.
    alpha : contract level.
    delta : error budget, used solely for the confidence sequence that audits
        the fallback assumption. The contract itself needs no budget.
    hist_risk, hist_cost : per-plan historical means. Design inputs: they order
        candidates and initialise the risk estimates. They carry no guarantee,
        so historical/deployment distribution shift costs money, never validity.
    policy : how the optimisation layer picks a plan while the ledger is in
        credit. 'lp' re-solves min c.w s.t. rhat.w <= alpha on the current risk
        estimates and samples from the optimal mixture; 'dual' prices risk with a
        multiplier updated by dual ascent. Both are pure optimisation -- the
        ledger is what makes either safe -- and they are compared in the
        ablation because the LP mixes two ADJACENT frontier plans while the dual
        recursion tends to oscillate between the extremes, which costs money at
        no benefit.
    lam_rate : dual ascent step size, used only by policy='dual'.
    audit_rate : fraction of served queries whose loss is labelled. Below 1.0 the
        ledger is driven by an inverse-probability estimate, still unbiased but
        with variance 1/rate times larger, so the buffer is inflated by the same
        factor -- the cost of not labelling everything, priced rather than hidden.
    buffer_c : multiplier on the buffer level; 1.0 is the theory's own constant.
    target_safe_rate : fraction of arrivals the controller is willing to serve
        conservatively. The optimiser's margin below alpha is tuned online to hit
        it, which is a cost-versus-cost decision and cannot affect validity.
    """

    def __init__(self, losses: np.ndarray, costs: np.ndarray, alpha: float,
                 delta: float = 0.1, hist_risk: np.ndarray = None,
                 hist_cost: np.ndarray = None, lam_rate: float = 0.5,
                 audit_rate: float = 1.0, audit_unit_cost: float = 0.0,
                 fallback: int = None, cs_grid: int = 129,
                 buffer_c: float = 1.0, target_safe_rate: float = 0.05,
                 rho_eta: float = 2e-4, rho_init: float = 0.02,
                 kappa: float = 1.0, explore_max: float = 0.10,
                 explore_scale: float = 50.0, policy: str = "lp",
                 resolve_every: int = 25, prior_n: float = 50.0):
        self.L = np.asarray(losses, dtype=float)
        self.C = np.asarray(costs, dtype=float)
        self.P, self.N = self.L.shape
        self.alpha = float(alpha)
        self.delta = float(delta)
        self.lam_rate = float(lam_rate)
        self.audit_rate = float(audit_rate)
        self.audit_unit_cost = float(audit_unit_cost)
        self.cost_hat = (self.C.mean(axis=1).copy() if hist_cost is None
                         else np.asarray(hist_cost, dtype=float).copy())
        self.risk_hat = (np.full(self.P, 0.5) if hist_risk is None
                         else np.asarray(hist_risk, dtype=float).copy())
        # the historical mean enters as a prior of strength prior_n, so the first
        # observation refines the estimate instead of replacing it by 0 or 1.
        # Strength matters only for cost: a strong prior converges slowly to a
        # shifted deployment distribution, a weak one is noisy early. Neither can
        # affect the contract, which is the point of the design.
        self.risk_n = np.full(self.P, float(prior_n))
        # the fallback is the plan history believes is safest; the ledger leans
        # on it, and `fallback_cs` checks online whether that belief holds.
        self.fallback = (int(np.argmin(self.risk_hat)) if fallback is None
                         else int(fallback))
        # Safe mode has to run a plan whose risk is bounded, and the cheapest
        # such plan is what it should run -- falling back to the SAFEST plan
        # makes every intervention maximally expensive and forces the optimiser
        # to keep a large margin just to avoid triggering it. So every plan gets
        # its own anytime sequence, at delta/P each. The union correction is
        # affordable precisely because these sequences do not carry the contract:
        # they only certify that recovery is possible, and a 1.5x wider bound on
        # that question buys an order of magnitude cheaper recovery.
        self.cs = [TwoSidedCS(self.delta / self.P, grid=cs_grid)
                   for _ in range(self.P)]
        # cost scale so that the multiplier is dimensionless
        self.cscale = float(self.cost_hat.max() - self.cost_hat.min()) or 1.0
        self.buffer_c = float(buffer_c)
        self.kappa = float(kappa)
        self.explore_max = float(explore_max)
        self.explore_scale = float(explore_scale)
        self._w = None
        self.target_safe_rate = float(target_safe_rate)
        self.rho_eta = float(rho_eta)
        self._rho = float(rho_init)
        self.policy = policy
        self.resolve_every = int(resolve_every)
        self._log1d = float(np.log(1.0 / max(self.delta, 1e-12)))

    # The margin below alpha that the optimiser aims at is NOT a statistical
    # quantity. Validity comes from the ledger and its buffer whatever the margin
    # is, including zero. What the margin buys is quiet: at margin zero the
    # ledger has no drift, so it hovers at the buffer and every downward
    # fluctuation triggers safe service, which is expensive because safe plans
    # are expensive. Raising the margin gives the ledger positive drift and makes
    # those interventions rare.
    #
    # So the margin trades cost against cost, not risk against cost, and it can
    # be chosen by economics instead of by a confidence level -- something a
    # static certifier structurally cannot do, since its margin IS its confidence
    # width. We set it by driving the observed rate of safe-mode arrivals to a
    # target with a stochastic-approximation update. The target has a direct
    # operational reading (how often the system is allowed to be conservative)
    # and is swept in the experiments.

    def _update_rho(self, safe_mode: bool) -> None:
        self._rho = float(np.clip(
            self._rho + self.rho_eta * ((1.0 if safe_mode else 0.0)
                                       - self.target_safe_rate),
            0.0, 0.5 * self.alpha))

    def rho(self, t: int = 0) -> float:
        return self._rho

    def recovery_plan(self):
        """Cheapest plan with a certified risk below alpha, and its drift.

        Returns (plan, eps). eps = alpha - hi_p is a valid lower bound on the
        ledger's drift while that plan is running, because hi_p is an anytime
        upper bound on its risk. If no plan is certified yet, eps <= 0 and the
        caller treats the buffer as infinite, pinning the controller to the
        historically safest plan -- the cold start.
        """
        hi = np.array([c.upper() for c in self.cs])
        ok = np.where(hi < self.alpha - 1e-9)[0]
        if len(ok) == 0:
            return int(np.argmin(hi)), self.alpha - float(hi.min())
        p = int(ok[np.argmin(self.cost_hat[ok])])
        return p, self.alpha - float(hi[p])

    def buffer(self, t: int = 0) -> float:
        """Ledger level below which the controller steers safe.

        Only the recovery behaviour matters. Whatever the optimiser does above
        the buffer, a downward move is detected at the next arrival and the
        controller switches to safe service, where the ledger is a random walk
        with positive drift eps = alpha - r_f. Applying Ville's inequality to
        exp(-theta * B) with theta = 2 eps / (v + eps) bounds the drawdown of
        that walk by log(1/delta)/theta uniformly over an unbounded horizon; one
        more arrival can be in flight when the switch happens, worth (1 - alpha).
        Nothing here grows with t: the buffer is a constant, and the risk it
        withholds is buffer/t, which vanishes.

        eps comes from the fallback plan's anytime upper bound, never a point
        estimate, so before the fallback is certified eps <= 0, the buffer is
        infinite, and the controller stays safe. That is the cold start, produced
        by the formula rather than by a special case.
        """
        _, eps = self.recovery_plan()
        if eps <= 1e-9:
            return float("inf")
        hi = self.alpha - eps
        v = max(hi * (1.0 - hi), 1e-3)
        theta = 2.0 * eps / (v + eps)
        r = max(self.audit_rate, 1e-9)
        return self.buffer_c * (self._log1d / theta + (1.0 - self.alpha)) / r

    def _greedy(self, lam: float) -> int:
        """Cheapest plan once risk is priced at lam: argmin c/scale + lam*rhat.

        Sweeping lam from 0 to infinity walks the lower convex hull of the
        (risk, cost) cloud from the cheapest plan to the safest, so dual ascent
        is a search along the LP frontier rather than a heuristic -- but a search
        that only visits hull VERTICES, never the interior points a mixture
        reaches, which is why the LP variant below dominates it.
        """
        score = self.cost_hat / self.cscale + lam * self.risk_hat
        return int(np.argmin(score))

    def effective_alpha(self, ledger: float, t: int) -> float:
        """Risk target: alpha - rho_t while the ledger is above its buffer.

        Below the buffer the target drops to the safest plan's estimated risk,
        which is what guarantees the ledger recovers. The switch is checked at
        every arrival, so a negative drift is corrected within one query and the
        drawdown stays in the geometric regime the buffer is sized for.
        """
        if ledger >= self.buffer(t):
            return max(self.alpha - self.rho(t), 0.0)
        return float(self.risk_hat.min())

    def _lp_mixture(self, alpha_eff: float):
        """Optimal randomized plan on the CURRENT risk estimates.

        With one risk constraint on the simplex, a basic optimum has at most two
        nonzero weights, so the mixture is two adjacent frontier plans
        straddling the target. Estimates only ever move which two, never whether
        the contract holds. Plans the mixture selects get executed and therefore
        estimated better, so exploration falls out of optimisation and needs no
        separate budget -- the cheap plans are the ones worth learning about and
        also the ones worth running.
        """
        from contractrag.certlp import solve_risk_lp
        # The constraint uses an UPPER estimate, so the mixture the LP returns
        # aims below the target rather than at it and the ledger drifts up. The
        # margin kappa * sqrt(v / n_p) is deliberately NOT a simultaneous
        # confidence bound: it needs no union correction over plans and no
        # delta, because it is not what makes the contract hold. That is the
        # concrete payoff of putting the guarantee in the ledger -- the estimator
        # is free to be as tight as it is accurate, where a static certifier is
        # forced to a delta/K Bonferroni bound over its whole candidate set.
        half = self.kappa * np.sqrt(0.25 / np.maximum(self.risk_n, 1.0))
        r_ucb = np.minimum(self.risk_hat + half, 1.0)
        _, w = solve_risk_lp(r_ucb, self.cost_hat, alpha_eff)
        if w is None:
            w = np.zeros(self.P)
            w[int(np.argmin(r_ucb))] = 1.0
        return np.maximum(w, 0.0) / max(w.sum(), 1e-12)

    def _explore_plan(self, alpha_eff: float) -> int:
        """The cheap, under-observed plan most worth learning about.

        Restricted to plans cheaper than what the current mixture costs, since a
        plan that is not cheaper cannot improve the objective however good its
        risk turns out to be. Among those, the least-observed one, because that
        is where the estimate is worst.
        """
        budget = float(np.dot(self._w, self.cost_hat)) if self._w is not None \
            else float(self.cost_hat.max())
        cand = np.where(self.cost_hat < budget - 1e-15)[0]
        if len(cand) == 0:
            return int(np.argmin(self.cost_hat))
        return int(cand[np.argmin(self.risk_n[cand])])

    def explore_rate(self, ledger: float, t: int) -> float:
        """Exploration funded by ledger surplus, and by nothing else.

        This is the structural peculiarity of certified optimization: the plans
        worth exploring are the cheap ones, which are also the plans worth
        deploying, so exploration is not a tax on the objective. What it does
        consume is contract allowance, and the ledger measures exactly how much
        allowance has been earned. Tying the exploration rate to the surplus
        above the buffer therefore prices exploration correctly with no separate
        budget, no schedule, and no tuning of a decay rate.
        """
        b = self.buffer(t)
        if not np.isfinite(b):
            return 0.0
        surplus = ledger - b
        if surplus <= 0.0:
            return 0.0
        return float(min(self.explore_max, surplus / max(self.explore_scale, 1e-9)))

    def run(self, order: np.ndarray, rng: np.random.Generator,
            log_every: int = 200) -> CAEState:
        st = CAEState()
        order = np.asarray(order)
        cum = np.zeros(len(order))
        lam = 0.0
        w = None
        for step, q in enumerate(order):
            t = step + 1
            a_eff = self.effective_alpha(st.ledger, t)
            safe_mode = a_eff < self.alpha - self.rho(t) - 1e-9
            if safe_mode:
                st.fallbacks += 1
            self._update_rho(safe_mode)
            # the ledger sets the risk TARGET; the optimiser only picks the
            # cheapest way to hit it. The guarantee lives entirely in the
            # target, which is why the optimiser is free to be wrong.
            if safe_mode:
                # Recovery must be guaranteed, not merely likely, so safe mode
                # runs a plan carrying an anytime upper bound on its risk -- the
                # cheapest one that has it. Before anything is certified this
                # falls back to the historically safest plan, which is the only
                # point in the design where history is trusted, and it is
                # trusted only for as long as the sequences take to speak.
                rp, eps_rp = self.recovery_plan()
                p = rp if eps_rp > 1e-9 else self.fallback
            elif self.policy == "lp":
                if w is None or step % self.resolve_every == 0:
                    w = self._lp_mixture(a_eff)
                    self._w = w
                if rng.random() < self.explore_rate(st.ledger, t):
                    p = self._explore_plan(a_eff)
                    st.explores += 1
                else:
                    p = int(rng.choice(self.P, p=w))
            else:
                p = self._greedy(lam)
            l = float(self.L[p, q])
            st.serve_cost += float(self.C[p, q])
            st.t += 1
            st.plan_counts[p] = st.plan_counts.get(p, 0) + 1

            labelled = self.audit_rate >= 1.0 or rng.random() < self.audit_rate
            if labelled:
                st.audits += 1
                st.audit_cost += self.audit_unit_cost
                # inverse-probability weighting keeps the ledger unbiased when
                # only a sample of traffic is labelled
                l_obs = l / max(self.audit_rate, 1e-9)
                self.risk_n[p] += 1.0
                self.risk_hat[p] += (l - self.risk_hat[p]) / self.risk_n[p]
                self.cs[p].update(l)
                lam = max(0.0, lam + self.lam_rate * (l - self.alpha))
            else:
                l_obs = 0.0
            st.violations += l
            st.ledger += self.alpha - l_obs
            st.min_ledger = min(st.min_ledger, st.ledger)
            cum[step] = st.violations
            if st.fallback_certified_at < 0:
                _, e_rp = self.recovery_plan()
                st.fallback_hi = self.alpha - e_rp
                if e_rp > 1e-9:
                    st.fallback_certified_at = st.t
            if st.t % log_every == 0:
                st.history.append({
                    "t": st.t, "plan": int(p), "lam": float(lam),
                    "ledger": float(st.ledger),
                    "mean_cost": st.total_cost / st.t,
                    "emp_risk": st.violations / st.t,
                    "fallback_share": st.fallbacks / st.t,
                })
        st.cum_viol = cum
        st.lam = lam
        return st


# ---------------------------------------------------------------------------
# static baseline on the same stream, for an apples-to-apples cost of ownership
# ---------------------------------------------------------------------------

def static_stream_cost(losses: np.ndarray, costs: np.ndarray, alpha: float,
                       delta: float, order: np.ndarray, n_cal: int,
                       hist_risk: np.ndarray, hist_cost: np.ndarray,
                       audit_unit_cost: float = 0.0, mix_grid: int = 7):
    """Cost of ownership of the one-shot certifier over the same stream.

    The first n_cal arrivals are the calibration set. They still have to be
    served, and served without a certificate: the honest choice, and the one the
    incumbent pipeline makes, is the plan history believes is safest. Their
    labels are paid for. After calibration the fixed-sequence certifier runs
    once and its output serves the remainder unchanged. Charging the calibration
    phase is what makes the comparison fair, and it is exactly the term omitted
    when a certified system reports only its steady-state spend.
    """
    from contractrag.certlp import (build_randomized_chain,
                                    certify_randomized_chain)
    order = np.asarray(order)
    cal_idx, rest_idx = order[:n_cal], order[n_cal:]
    safe = int(np.argmin(hist_risk))
    serve = float(costs[safe, cal_idx].sum())
    viol = float(np.count_nonzero(losses[safe, cal_idx] > 0.5))
    audit = audit_unit_cost * len(cal_idx)

    chain = build_randomized_chain(np.asarray(hist_risk, dtype=float)[:, None],
                                   np.asarray(hist_cost, dtype=float)[:, None],
                                   mix_grid=mix_grid)
    cp = certify_randomized_chain(losses[:, cal_idx], costs[:, cal_idx],
                                  chain, alpha, delta)
    if cp.element is None:
        sup, wts = (safe,), (1.0,)
    else:
        sup, wts = cp.element.support, cp.element.weights
    # a deployed mixture draws a plan per query, so per-query loss and cost are
    # the weighted ones
    post_loss = np.zeros(len(rest_idx))
    for k, w in zip(sup, wts):
        serve += w * float(costs[k, rest_idx].sum())
        post_loss += w * (losses[k, rest_idx] > 0.5)
    viol += float(post_loss.sum())
    cum = np.cumsum(np.concatenate([(losses[safe, cal_idx] > 0.5).astype(float),
                                    post_loss]))
    return {"total_cost": serve + audit, "serve_cost": serve,
            "audit_cost": audit, "risk": viol / len(order),
            "mean_cost": (serve + audit) / len(order), "cum_viol": cum,
            "certified": bool(cp.element is not None),
            "plan": cp.describe() if cp.element is not None else f"#{safe}",
            "n_cal": n_cal}
