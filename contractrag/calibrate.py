"""Distribution-free risk certification for plan lattices.

Core statistical machinery of ContractRAG:

1. Hoeffding-Bentkus p-values for H0: R(policy) > alpha given bounded losses.
2. Fixed-sequence Learn-Then-Test over a cost-ordered family of policies:
   walk policies from most conservative (highest cost) to cheapest, keep the
   cheapest policy whose violation risk is certified <= alpha at FWER delta.
3. Escalation-ladder policies pi_lambda: answer at the first rung whose
   runtime sufficiency score clears its threshold; thresholds are swept along
   a one-dimensional monotone path (per-rung score quantiles), which makes
   fixed-sequence testing exact (no multiplicity correction needed).
4. Anytime-valid contract monitor (test supermartingale / e-process) for
   online violation detection under workload drift.

All guarantees are finite-sample and distribution-free (iid calibration draw).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.stats import binom


# ---------------------------------------------------------------------------
# p-values
# ---------------------------------------------------------------------------

def hb_p_value(r_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: E[loss] > alpha, losses in [0,1].

    p = min( exp(-n*KL(min(r_hat,alpha) || alpha)), e * P(Bin(n,alpha) <= ceil(n*r_hat)) )
    (Bates et al. 2021; Angelopoulos et al. 2021, Learn-Then-Test.)
    """
    if n == 0:
        return 1.0
    r = min(r_hat, alpha)

    def kl(a, b):
        a = min(max(a, 1e-12), 1 - 1e-12)
        b = min(max(b, 1e-12), 1 - 1e-12)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    p_hoeff = float(np.exp(-n * kl(r, alpha)))
    p_bent = float(np.e * binom.cdf(int(np.ceil(n * r_hat)), n, alpha))
    return min(1.0, p_hoeff, p_bent)


def finite_pop_p_value(r_hat: float, n: int, alpha: float,
                       n_pop: int) -> float:
    """Finite-population p-value for H0: population risk > alpha when the
    n calibration queries are drawn WITHOUT replacement from a fixed
    population of n_pop binary losses.

    p = min( exp(-n*KL(r_hat || alpha)),  P(Hyp(n_pop, K0, n) <= k) ),
    K0 = floor(n_pop*alpha) + 1 (smallest violation count breaking the
    contract). The hypergeometric term is EXACT under without-replacement
    sampling (MLR in K puts the worst case at K0); the KL-Chernoff term
    stays valid without replacement by Hoeffding (1963, Thm 4) convex
    ordering. Together: a valid, finite-population-tight p-value.
    """
    if n == 0:
        return 1.0
    from scipy.stats import hypergeom
    k = int(round(r_hat * n))
    k0 = int(np.floor(n_pop * alpha)) + 1
    if k0 > n_pop:  # alpha so lax no count can break it: never reject H1
        return 0.0
    p_hyper = float(hypergeom.cdf(k, n_pop, k0, n))
    r = min(r_hat, alpha)

    def kl(a, b):
        a = min(max(a, 1e-12), 1 - 1e-12)
        b = min(max(b, 1e-12), 1 - 1e-12)
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))

    p_hoeff = float(np.exp(-n * kl(r, alpha)))
    return min(1.0, p_hoeff, p_hyper)


def binomial_upper_bound(k: int, n: int, delta: float) -> float:
    """Exact Clopper-Pearson upper bound on a proportion at confidence 1-delta.

    Used to attach (eps, delta) certificates to approximate rewrite rules:
    P(rewrite increases loss) <= ub with prob >= 1-delta over calibration draw.
    """
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - delta, k + 1, n - k))


def path_composition_voucher(step_harm: list[np.ndarray], delta_r: float):
    """Assumption-free composition bound for a rewrite CHAIN.

    Let P_0 -> P_1 -> ... -> P_m be the plans produced by applying rewrites
    R_1,...,R_m in sequence (P_i = R_i(P_{i-1})). step_harm[i] is the per-query
    indicator 1{loss(P_{i+1}) > loss(P_i)} on the calibration set -- the
    *stepwise* harm of R_{i+1} against its actual predecessor, not against the
    base plan.

    Telescoping identity (no assumption): if loss(P_m,q) > loss(P_0,q) then some
    consecutive step strictly increases loss, hence
        {q: loss(P_m) > loss(P_0)}  subset of  U_i {q: loss(P_{i+1}) > loss(P_i)}.
    Union bound over Clopper-Pearson step vouchers gives, w.p. >= 1 - m*delta_r,
        rho(chain; P_0) <= sum_i eps_hat_i^step.
    Unlike the base-relative union bound this needs NO no-synergistic-harm
    assumption: the inclusion above is a deterministic tautology.

    Returns dict with per-step CP bounds, their sum, and realized step harms.
    """
    m = len(step_harm)
    n = len(step_harm[0]) if m else 0
    eps = [binomial_upper_bound(int(h.sum()), n, delta_r) for h in step_harm]
    return {
        "n": n,
        "step_harm_rate": [float(h.mean()) for h in step_harm],
        "step_voucher": [float(e) for e in eps],
        "path_voucher": float(sum(eps)),
        "delta_total": float(m * delta_r),
    }


def selectivity_conditioned_voucher(harm: np.ndarray, sigma: np.ndarray,
                                    edges: np.ndarray, delta_r: float):
    """Selectivity-stratified voucher for a single rewrite.

    Stratify calibration queries by an estimated selectivity sigma into the
    given bin edges; return a Clopper-Pearson harm bound *per stratum* (each at
    confidence 1 - delta_r / #bins, Bonferroni). A per-query voucher then reads
    off the bound for the query's stratum -- a risk analogue of a
    selectivity-indexed cost model. Returns list of dicts per bin.
    """
    nb = len(edges) - 1
    d = delta_r / max(1, nb)
    out = []
    idx = np.clip(np.digitize(sigma, edges[1:-1]), 0, nb - 1)
    for b in range(nb):
        mask = idx == b
        n_b = int(mask.sum())
        k_b = int(harm[mask].sum()) if n_b else 0
        out.append({
            "bin": [float(edges[b]), float(edges[b + 1])],
            "n": n_b, "harm_rate": (k_b / n_b) if n_b else 0.0,
            "voucher": binomial_upper_bound(k_b, n_b, d) if n_b else 1.0,
        })
    return out


# ---------------------------------------------------------------------------
# Escalation ladder policies
# ---------------------------------------------------------------------------

@dataclass
class LadderData:
    """Calibration records for an L-rung ladder executed on n queries.

    losses[j, i]  = 1 if answering query i at rung j violates the quality part
                    of the contract (loss can be any [0,1] value).
    scores[j, i]  = runtime sufficiency score of rung j on query i
                    (higher = more confident; np.inf forces stop, -np.inf forces escalate).
    costs[j, i]   = execution cost (CNY or tokens) paid *at* rung j for query i
                    (cumulative cost of a query that stops at rung j is
                    sum of costs[0..j, i] because progressive execution runs
                    all rungs up to the stopping rung).
    """
    losses: np.ndarray
    scores: np.ndarray
    costs: np.ndarray

    @property
    def L(self):
        return self.losses.shape[0]

    @property
    def n(self):
        return self.losses.shape[1]


def stop_rung(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """First rung j whose score >= threshold; final rung is forced stop."""
    L, n = scores.shape
    stop = np.full(n, L - 1, dtype=int)
    undecided = np.ones(n, dtype=bool)
    for j in range(L - 1):
        hit = undecided & (scores[j] >= thresholds[j])
        stop[hit] = j
        undecided &= ~hit
    return stop


def policy_risk_cost(data: LadderData, thresholds: np.ndarray):
    """Empirical risk and mean cumulative cost of pi_lambda on calibration data."""
    stop = stop_rung(data.scores, thresholds)
    idx = np.arange(data.n)
    risk = float(data.losses[stop, idx].mean())
    cum = np.cumsum(data.costs, axis=0)
    cost = float(cum[stop, idx].mean())
    return risk, cost


def quantile_path(data: LadderData, num_grid: int = 201):
    """One-dimensional monotone threshold family.

    Parameter q in [0,1]: threshold_j = quantile_q of rung-j scores.
    q=0  -> always stop at rung 0 (cheapest, riskiest).
    q=1  -> always escalate to final rung (most expensive, safest).
    Returns list of (q, thresholds[L-1]) from most conservative to cheapest,
    i.e. ordered by decreasing q, which is the fixed-sequence test order.
    """
    qs = np.linspace(0.0, 1.0, num_grid)
    fams = []
    for q in qs:
        thr = np.array([np.quantile(data.scores[j], q) for j in range(data.L - 1)])
        # strictly-greater semantics via tiny epsilon keeps q=1 fully conservative
        if q >= 1.0:
            thr = thr + 1e9
        fams.append((float(q), thr))
    fams.sort(key=lambda t: -t[0])  # conservative first
    return fams


@dataclass
class CertifiedPolicy:
    thresholds: np.ndarray | None
    q: float
    alpha: float
    delta: float
    cal_risk: float
    cal_cost: float
    p_value: float
    certified: bool
    fallback: str | None = None  # 'final-rung' when nothing certifies


def certify_ladder(data: LadderData, alpha: float, delta: float,
                   num_grid: int = 201, n_pop: int = None) -> CertifiedPolicy:
    """Fixed-sequence LTT over the monotone quantile path.

    Walks from the most conservative policy to the cheapest; keeps the last
    (cheapest) policy whose HB p-value for H0: risk > alpha is <= delta.
    Fixed-sequence testing controls FWER at delta with NO multiplicity
    correction because testing stops at the first failure. If n_pop is
    given, uses the finite-population p-value (calibration drawn without
    replacement from a population of that size).
    """
    if n_pop is not None:
        pv = lambda r, n, a: finite_pop_p_value(r, n, a, n_pop)
    else:
        pv = hb_p_value
    fams = quantile_path(data, num_grid)
    best = None
    for q, thr in fams:
        risk, cost = policy_risk_cost(data, thr)
        p = pv(risk, data.n, alpha)
        if p <= delta:
            best = CertifiedPolicy(thr, q, alpha, delta, risk, cost, p, True)
        else:
            break  # fixed sequence: stop at first non-rejection
    if best is None:
        # nothing certifies (alpha too strict for even the safest policy):
        # fall back to always running the final rung; report honestly.
        thr = np.full(data.L - 1, np.inf)
        risk, cost = policy_risk_cost(data, thr)
        return CertifiedPolicy(thr, 1.0, alpha, delta, risk, cost,
                               pv(risk, data.n, alpha), False, "final-rung")
    return best


def certify_ladder_groupwise(data: LadderData, groups: np.ndarray, alpha,
                             delta: float, num_grid: int = 201):
    """Group-conditional certification: one certified threshold vector per group.

    alpha may be a float (same level for all groups) or a dict group->level.
    Bonferroni split of delta across groups keeps the simultaneous guarantee.
    """
    out = {}
    uniq = sorted(set(groups.tolist()))
    d_g = delta / max(1, len(uniq))
    for g in uniq:
        a_g = alpha[g] if isinstance(alpha, dict) else alpha
        mask = groups == g
        sub = LadderData(data.losses[:, mask], data.scores[:, mask], data.costs[:, mask])
        out[g] = certify_ladder(sub, a_g, d_g, num_grid)
    return out


def certify_ladder_multi(data: LadderData, aux_losses: dict[str, np.ndarray],
                         alphas: dict[str, float], delta: float,
                         num_grid: int = 201) -> CertifiedPolicy:
    """Multi-constraint fixed-sequence LTT.

    data.losses is the primary loss; aux_losses maps constraint name ->
    (L, n) violation indicators (e.g. latency SLO exceedance per rung).
    alphas maps 'primary' and each aux name to its risk budget.

    Composite null per candidate is a UNION of per-constraint nulls, so the
    intersection-union test (Berger 1982) applies: reject iff EVERY
    constraint's HB p-value <= delta -- no Bonferroni split needed, the
    composite test has size <= delta. Fixed-sequence walk keeps FWER <= delta.
    """
    d_each = delta
    fams = quantile_path(data, num_grid)
    idx = np.arange(data.n)
    cum_cost = np.cumsum(data.costs, axis=0)
    best = None
    for q, thr in fams:
        stop = stop_rung(data.scores, thr)
        ok = True
        r_primary = float(data.losses[stop, idx].mean())
        if hb_p_value(r_primary, data.n, alphas["primary"]) > d_each:
            ok = False
        if ok:
            for name, lmat in aux_losses.items():
                r = float(lmat[stop, idx].mean())
                if hb_p_value(r, data.n, alphas[name]) > d_each:
                    ok = False
                    break
        cost = float(cum_cost[stop, idx].mean())
        if ok:
            best = CertifiedPolicy(thr, q, alphas["primary"], delta,
                                   r_primary, cost, 0.0, True)
        else:
            break
    if best is None:
        thr = np.full(data.L - 1, np.inf)
        r, c = policy_risk_cost(data, thr)
        return CertifiedPolicy(thr, 1.0, alphas["primary"], delta, r, c, 1.0,
                               False, "final-rung")
    return best


# ---------------------------------------------------------------------------
# Fixed-sequence certification over a discrete plan lattice (no ladder)
# ---------------------------------------------------------------------------

def certify_plan_sequence(losses_per_plan: list[np.ndarray], costs: list[float],
                          alpha: float, delta: float):
    """Certify the cheapest plan in a cost-DESCENDING walk.

    losses_per_plan[i]: per-query violation indicators of plan i on calibration set.
    costs[i]: mean cost of plan i. Plans are sorted by cost descending internally.
    Returns (index_of_selected_plan, diagnostics list).
    """
    order = np.argsort([-c for c in costs])
    selected = None
    diag = []
    for i in order:
        r = float(np.mean(losses_per_plan[i]))
        p = hb_p_value(r, len(losses_per_plan[i]), alpha)
        diag.append({"plan": int(i), "risk": r, "p": p, "cost": costs[i]})
        if p <= delta:
            selected = int(i)
        else:
            break
    return selected, diag


def certify_candidates(cal_losses: list[np.ndarray], cal_costs: list[np.ndarray],
                       order: list[int], alpha: float, delta: float):
    """General fixed-sequence LTT over an arbitrary prespecified candidate order.

    order: candidate indices in TEST ORDER (chosen on train data only —
    typically ascending train risk). Walks the sequence; certifies each
    candidate whose HB p-value <= delta; stops at the first failure (this is
    what makes it FWER-delta without multiplicity correction). Among certified
    candidates returns the one with the lowest mean calibration cost.

    Returns (best_index or None, diagnostics).
    """
    n = len(cal_losses[order[0]])
    certified = []
    diag = []
    for ci in order:
        r = float(np.mean(cal_losses[ci]))
        p = hb_p_value(r, n, alpha)
        diag.append({"cand": int(ci), "risk": r, "p": p,
                     "cost": float(np.mean(cal_costs[ci]))})
        if p <= delta:
            certified.append(ci)
        else:
            break
    if not certified:
        return None, diag
    best = min(certified, key=lambda ci: float(np.mean(cal_costs[ci])))
    return best, diag


# ---------------------------------------------------------------------------
# Anytime-valid contract monitor (e-process)
# ---------------------------------------------------------------------------

class EProcessMonitor:
    """Finite-horizon mixture-restart e-detector for
    H0: conditional violation rate <= alpha at every time.

    Per-step e-value e_t(w) = 1 + w (l_t - alpha) satisfies
    E[e_t(w) | F_{t-1}] <= 1 under H0 for bets w in (0, 1/alpha).
    We mix restarted e-processes over start times j in {1..T} (uniform 1/T)
    and over a Kelly-style bet grid W (uniform 1/|W|):

        M_t = sum_{j<=t} sum_{w in W} (1/(T|W|)) prod_{i=j..t} e_i(w)
              + (1 - t/T)

    Each (j, w) product is a nonnegative supermartingale equal to 1 before j,
    the weights sum to 1, so M is a nonnegative supermartingale with M_0 = 1
    and Ville's inequality gives P(sup_{t<=T} M_t >= 1/delta) <= delta.
    Restarts prevent the statistic from sinking during long compliant
    stretches; the bet mixture adapts to the (unknown) post-change risk.
    O(|W|) per update via A_w <- (A_w + 1/(T|W|)) * e_t(w).
    """

    def __init__(self, alpha: float, delta: float, horizon: int = 100000,
                 bets=(0.1, 0.3, 0.5, 0.7, 0.9)):
        self.alpha = alpha
        self.delta = delta
        self.T = horizon
        self.ws = [b / max(alpha, 1e-6) for b in bets]
        self.A = [0.0] * len(self.ws)
        self.history = []
        self.alarm_at = None
        self.t = 0
        # O(1) changepoint localization (diagnostic only; validity is from the
        # mixture M_t). CUSUM/GLR recursion on the median bet: restart the
        # single-restart wealth whenever it drops to <= 1, and remember when
        # the currently-growing run began -- the standard changepoint estimate.
        self._w_cp = self.ws[len(self.ws) // 2]
        self._logW = 0.0          # log wealth of the current run
        self._run_start = 0       # step at which the current run began
        self.changepoint_hat = None

    def update(self, loss: float) -> bool:
        """Feed one violation indicator; returns True if alarm fires now."""
        self.t += 1
        base = 1.0 / (self.T * len(self.ws))
        m_t = max(0.0, 1.0 - self.t / self.T)
        for k, w in enumerate(self.ws):
            e = max(1e-12, 1.0 + w * (loss - self.alpha))
            self.A[k] = (self.A[k] + base) * e
            m_t += self.A[k]
        self.history.append(m_t)
        # changepoint recursion (max-over-restart / CUSUM): if the run's log
        # wealth falls to <= 0, restart the run at the current step.
        inc = np.log(max(1e-12, 1.0 + self._w_cp * (loss - self.alpha)))
        if self._logW <= 0.0:
            self._logW, self._run_start = max(0.0, inc), self.t
        else:
            self._logW += inc
        if self.alarm_at is None and m_t >= 1.0 / self.delta:
            self.alarm_at = self.t
            self.changepoint_hat = self._run_start
            return True
        return False

    def reset(self):
        self.A = [0.0] * len(self.ws)
        self.t = 0
        self.alarm_at = None
        self.history = []
        self._logW = 0.0
        self._run_start = 0
        self.changepoint_hat = None
