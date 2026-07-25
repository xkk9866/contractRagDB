"""Certified linear programming over randomized execution plans.

Motivation. The fixed-sequence certifier of `optimizer.optimize` walks a
train-risk-ordered chain of candidate policies and stops at the first
Hoeffding-Bentkus non-rejection. Two structural losses follow, both measured
on the real execution matrices (see scripts/probe_planspace_gap.py):

  (L1) the walk aborts before reaching cheap router candidates that are
       demonstrably feasible on the calibration split, overpaying by
       1.2x-16x depending on the track and contract level;
  (L2) every deterministic candidate has risk strictly below alpha, so the
       admissible risk budget alpha - R(pi) is never converted into savings
       (observed slack 0.02-0.11).

This module removes both by certifying *randomized* plans. A randomized
plan is a distribution w over candidates; executing it draws k ~ w per
query. Its risk is linear in w, so the cheapest feasible randomized plan
solves a linear program, and the LP optimum sits exactly on the contract
boundary instead of below it.

Three facts make this statistically sound and computationally cheap.

  (F1) Validity for data-dependent w. Bound each candidate's risk from
       above simultaneously: on the event E = {R_k <= Rbar_k for all k in H}
       (probability >= 1 - delta by a Bonferroni split over the *train*-side
       frontier H, which is independent of the calibration data), ANY w
       supported on H satisfies R(w) = sum_k w_k R_k <= sum_k w_k Rbar_k.
       Feasibility of the LP therefore transfers to the deployed plan
       without a second test, no matter how w was chosen.

  (F2) Sparsity. With one risk constraint plus the simplex constraint, a
       basic optimal solution of the LP has at most two nonzero weights:
       the certified optimum is a mixture of two deterministic plans.

  (F3) Rao-Blackwell variance reduction. The per-query mixed loss
       Ytilde_i(w) = sum_k w_k loss_k(q_i) is the conditional expectation of
       the realized loss given the query, so it is unbiased for R(w), still
       lies in [0,1], and has variance no larger than the realized loss.
       Certifying Ytilde with an empirical-Bernstein or betting bound is
       therefore strictly tighter than certifying binary losses -- the same
       randomization that buys (F2) also buys a smaller confidence width.

Both interfaces are provided: `certified_lp` (F1 route, always feasible)
and `certified_lp_split` (train-selected w certified by a single
variance-adaptive test, tighter when the train frontier is reliable).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contractrag.calibrate import hb_p_value


# ---------------------------------------------------------------------------
# one-sided upper confidence bounds on a mean in [0, 1]
# ---------------------------------------------------------------------------

def hb_upper_bound(r_hat: float, n: int, delta: float, tol: float = 1e-6
                   ) -> float:
    """Smallest alpha whose HB p-value for H0: R > alpha is <= delta.

    Inverting the Hoeffding-Bentkus test gives a (1-delta) upper confidence
    bound valid for bounded losses with no distributional assumption.
    """
    if n == 0:
        return 1.0
    if r_hat >= 1.0:
        return 1.0
    lo, hi = float(r_hat), 1.0
    if hb_p_value(r_hat, n, hi) > delta:
        return 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if hb_p_value(r_hat, n, mid) <= delta:
            hi = mid
        else:
            lo = mid
    return hi


def emp_bernstein_upper(y: np.ndarray, delta: float) -> float:
    """Maurer-Pontil empirical-Bernstein upper bound for E[Y], Y in [0,1].

    Width scales with the sample standard deviation rather than the range,
    so it is the natural bound for the low-variance mixed loss of (F3).
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return 1.0
    v = float(np.var(y, ddof=1))
    L = np.log(2.0 / delta)
    ub = float(y.mean()) + np.sqrt(2.0 * v * L / n) + 7.0 * L / (3.0 * (n - 1))
    return float(min(1.0, max(0.0, ub)))


def betting_upper_bound(y: np.ndarray, delta: float, grid: int = 4096,
                        c: float = 0.5) -> float:
    """Hedged-capital upper confidence bound for E[Y], Y in [0,1].

    For the one-sided null H0: E[Y] >= m we bet against m with predictable
    stakes lambda_i: K_t(m) = prod_i (1 - lambda_i (Y_i - m)) is a
    nonnegative supermartingale under H0, so K(m) >= 1/delta rejects. K is
    nondecreasing in m, hence the confidence set is an interval [0, m*) and
    the returned bound is the smallest rejected m (Waudby-Smith & Ramdas,
    JRSS-B 2024). Variance-adaptive: tightens automatically when Y is
    concentrated, which is exactly the mixed-loss regime.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return 1.0
    mean = float(y.mean())
    if mean >= 1.0:
        return 1.0
    # predictable variance estimate from the strict prefix
    csum = np.concatenate([[0.0], np.cumsum(y)])
    csq = np.concatenate([[0.0], np.cumsum(y * y)])
    idx = np.arange(n)
    pm = np.where(idx > 0, csum[idx] / np.maximum(idx, 1), 0.5)
    pv = np.where(idx > 0, csq[idx] / np.maximum(idx, 1) - pm ** 2, 0.25)
    pv = np.maximum(pv, 1e-4)
    logthr = np.log(1.0 / delta)

    def rejects(m: float) -> bool:
        if m <= mean:
            return False
        lam = np.minimum(np.sqrt(2.0 * logthr / (n * pv)), c / max(1.0 - m, 1e-6))
        terms = 1.0 - lam * (y - m)
        if np.any(terms <= 1e-12):
            return False
        return float(np.sum(np.log(terms))) >= logthr

    lo, hi = mean, 1.0
    if not rejects(hi):
        return 1.0
    for _ in range(int(np.log2(grid)) + 12):
        mid = 0.5 * (lo + hi)
        if rejects(mid):
            hi = mid
        else:
            lo = mid
    return float(min(1.0, hi))


def tightest_upper_bound(y: np.ndarray, delta: float) -> tuple[float, str]:
    """Min over HB / empirical-Bernstein / betting, each valid at 1-delta.

    Taking a minimum of bounds derived from the SAME data is not generally
    valid, so we spend delta/3 on each and return the minimum: a union bound
    over three simultaneously valid bounds. The union cost is a log-factor
    and is dominated by the variance adaptivity gain on mixed losses.
    """
    d = delta / 3.0
    n = len(y)
    cands = [
        (hb_upper_bound(float(np.mean(y)), n, d), "hb"),
        (emp_bernstein_upper(y, d), "eb"),
        (betting_upper_bound(y, d), "bet"),
    ]
    return min(cands, key=lambda t: t[0])


# ---------------------------------------------------------------------------
# train-side frontier (independent of calibration data)
# ---------------------------------------------------------------------------

def pareto_frontier_idx(risk: np.ndarray, cost: np.ndarray,
                        max_keep: int = 48) -> list[int]:
    """Non-dominated indices of the (risk, cost) cloud, cost-ascending.

    Computed on TRAIN statistics only, so the resulting index set is
    independent of the calibration draw -- the property that lets a single
    Bonferroni split over this set validate every downstream LP solution.
    Thinned uniformly along the frontier to `max_keep` so the Bonferroni
    penalty stays a small constant.
    """
    order = sorted(range(len(risk)), key=lambda i: (cost[i], risk[i]))
    front, best = [], np.inf
    for i in order:
        if risk[i] < best - 1e-12:
            front.append(i)
            best = risk[i]
    if len(front) > max_keep:
        sel = np.linspace(0, len(front) - 1, max_keep).round().astype(int)
        front = [front[i] for i in sorted(set(sel.tolist()))]
    return front


def screened_candidate_set(risk: np.ndarray, cost: np.ndarray, alpha: float,
                           n_train: int, delta: float, max_keep: int = 64
                           ) -> list[int]:
    """Candidates worth certifying, screened on TRAIN statistics only.

    A pure Pareto frontier is the wrong screen when the train split is small:
    a genuinely cheap and feasible plan whose train risk is inflated by noise
    gets dominated and never reaches the certifier (observed on ASQA/QAMPARI,
    n_train = 150). Instead keep every candidate that a train-side confidence
    interval cannot rule out as feasible, i.e. whose train risk lies within
    its own two-sided width of alpha, plus the Pareto frontier itself so the
    safe end of the space is never lost. The screen uses train data only, so
    simultaneous calibration bounds over the screened set validate any LP
    solution supported on it.
    """
    m = len(risk)
    width = np.sqrt(np.log(2.0 / max(delta, 1e-6)) / (2.0 * max(n_train, 1)))
    keep = set(pareto_frontier_idx(risk, cost, max_keep // 2))
    plausible = [i for i in range(m) if risk[i] <= alpha + 2.0 * width]
    # prefer the cheapest plausible candidates: those are what the LP wants
    plausible.sort(key=lambda i: cost[i])
    for i in plausible[:max_keep]:
        keep.add(i)
    return sorted(keep)


def solve_risk_lp(risk: np.ndarray, cost: np.ndarray, alpha: float):
    """min_w cost.w  s.t.  risk.w <= alpha, w in simplex.

    Exact solution by (F2): the optimum is either a single candidate or a
    two-point mixture straddling alpha. Enumerating pairs is O(m^2) with m =
    |frontier| <= 48, i.e. microseconds, and avoids an LP dependency.
    Returns (cost, weights) or (None, None) when infeasible.
    """
    m = len(risk)
    feas = np.where(risk <= alpha + 1e-12)[0]
    if len(feas) == 0:
        return None, None
    j = int(feas[np.argmin(cost[feas])])
    best_c = float(cost[j])
    best_w = np.zeros(m)
    best_w[j] = 1.0
    risky = np.where(risk > alpha)[0]
    for i in feas:
        for k in risky:
            if cost[k] >= cost[i]:
                continue
            denom = risk[k] - risk[i]
            if denom <= 1e-12:
                continue
            t = (alpha - risk[i]) / denom          # weight on the risky plan
            if not (0.0 <= t <= 1.0):
                continue
            c = (1.0 - t) * cost[i] + t * cost[k]
            if c < best_c - 1e-15:
                best_c = float(c)
                best_w = np.zeros(m)
                best_w[i] = 1.0 - t
                best_w[k] = t
    return best_c, best_w


# ---------------------------------------------------------------------------
# certified randomized plans
# ---------------------------------------------------------------------------

@dataclass
class CertifiedMixture:
    weights: np.ndarray                # over the frontier subset
    support: list[int]                 # candidate indices with w > 0
    support_w: list[float]
    cal_risk_bound: float              # certified upper bound on R(w)
    cal_risk_hat: float                # plug-in mixed-loss mean
    cal_cost: float
    alpha: float
    delta: float
    certified: bool
    n_frontier: int
    bound_kind: str = "hb"
    fallback: str | None = None
    diagnostics: dict = field(default_factory=dict)

    def describe(self) -> str:
        if not self.support:
            return "none"
        parts = [f"{w:.3f}*#{i}" for i, w in zip(self.support, self.support_w)]
        return "mix(" + " + ".join(parts) + ")"


def certified_lp(cal_losses: np.ndarray, cal_costs: np.ndarray,
                 train_risk: np.ndarray, train_cost: np.ndarray,
                 alpha: float, delta: float, max_frontier: int = 48,
                 bound: str = "hb", screen: str = "plausible",
                 n_train: int = 0) -> CertifiedMixture:
    """Certified cheapest randomized plan (route F1).

    cal_losses[k, i] : loss of candidate k on calibration query i, in [0,1].
    cal_costs[k, i]  : cost of candidate k on calibration query i.
    train_risk/cost  : per-candidate TRAIN statistics used only to choose the
                       candidate screen (hence independent of the calibration
                       draw).
    screen           : 'plausible' keeps every candidate a train-side interval
                       cannot rule out (default, robust to small train
                       splits), 'pareto' keeps only the train frontier,
                       'all' certifies the whole space.

    Guarantee. With probability >= 1 - delta over the calibration draw, the
    deployed mixture satisfies R(w) <= alpha. Proof: Bonferroni over the
    screened set gives simultaneous validity of the per-candidate upper
    bounds; on that event the LP constraint sum_k w_k Rbar_k <= alpha implies
    R(w) = sum_k w_k R_k <= alpha for every w in the simplex, including the
    data-dependent optimizer output.
    """
    K, n = cal_losses.shape
    if screen == "all":
        H = list(range(K))
    elif screen == "pareto":
        H = pareto_frontier_idx(train_risk, train_cost, max_frontier)
    else:
        H = screened_candidate_set(train_risk, train_cost, alpha,
                                   n_train or n, delta, max_frontier)
    m = len(H)
    d_each = delta / max(1, m)
    rbar = np.ones(m)
    rhat = np.zeros(m)
    kinds = []
    for t, k in enumerate(H):
        y = cal_losses[k]
        rhat[t] = float(y.mean())
        if bound == "hb":
            rbar[t] = hb_upper_bound(rhat[t], n, d_each)
            kinds.append("hb")
        else:
            b, kind = tightest_upper_bound(y, d_each)
            rbar[t] = b
            kinds.append(kind)
    chat = np.array([float(cal_costs[k].mean()) for k in H])

    cost_lp, w = solve_risk_lp(rbar, chat, alpha)
    if cost_lp is None:
        # no frontier candidate is certifiable at this level: report honestly
        safest = int(np.argmin(rbar))
        w0 = np.zeros(m)
        w0[safest] = 1.0
        mixed = cal_losses[H[safest]]
        return CertifiedMixture(w0, [H[safest]], [1.0], float(rbar[safest]),
                                float(mixed.mean()), float(chat[safest]),
                                alpha, delta, False, m,
                                fallback="safest-frontier",
                                diagnostics={"rbar": rbar.tolist(),
                                             "rhat": rhat.tolist(),
                                             "cost": chat.tolist(),
                                             "frontier": H})
    sup = [int(H[t]) for t in np.where(w > 1e-12)[0]]
    supw = [float(w[t]) for t in np.where(w > 1e-12)[0]]
    mixed = np.zeros(n)
    for t, k in enumerate(H):
        if w[t] > 1e-12:
            mixed += w[t] * cal_losses[k]
    bound_val = float(np.dot(w, rbar))
    return CertifiedMixture(w, sup, supw, bound_val, float(mixed.mean()),
                            float(cost_lp), alpha, delta, True, m,
                            bound_kind="+".join(sorted(set(kinds))),
                            diagnostics={"rbar": rbar.tolist(),
                                         "rhat": rhat.tolist(),
                                         "cost": chat.tolist(),
                                         "frontier": H})


def certified_lp_split(cal_losses: np.ndarray, cal_costs: np.ndarray,
                       train_losses: np.ndarray, train_costs: np.ndarray,
                       alpha: float, delta: float, margin_delta: float = None,
                       max_frontier: int = 48) -> CertifiedMixture:
    """Certified randomized plan via train-side LP + one variance-adaptive test.

    Solves the LP on TRAIN statistics with a conservative margin, then spends
    the whole delta on a single upper bound for the mixed loss
    Ytilde(w) = sum_k w_k loss_k on the calibration split. Because w depends
    only on train data this is one hypothesis, so no multiplicity correction
    is needed and the bound may use the full delta; by (F3) the mixed loss is
    concentrated, so the variance-adaptive bound is markedly tighter than a
    binary-loss HB bound. Falls back to the safest train plan if the single
    test fails.
    """
    K, n_tr = train_losses.shape
    n = cal_losses.shape[1]
    tr_risk = train_losses.mean(axis=1)
    tr_cost = train_costs.mean(axis=1)
    H = pareto_frontier_idx(tr_risk, tr_cost, max_frontier)
    md = margin_delta if margin_delta is not None else delta
    # shrink the train risks upward by their own confidence width so the LP
    # target is feasible with high probability on fresh data
    tr_bar = np.array([hb_upper_bound(float(tr_risk[k]), n_tr, md / max(1, len(H)))
                       for k in H])
    tr_c = np.array([float(tr_cost[k]) for k in H])
    _, w = solve_risk_lp(tr_bar, tr_c, alpha)
    if w is None:
        safest = int(np.argmin(tr_bar))
        w = np.zeros(len(H))
        w[safest] = 1.0
    mixed = np.zeros(n)
    cost_mix = 0.0
    for t, k in enumerate(H):
        if w[t] > 1e-12:
            mixed += w[t] * cal_losses[k]
            cost_mix += w[t] * float(cal_costs[k].mean())
    ub, kind = tightest_upper_bound(mixed, delta)
    sup = [int(H[t]) for t in np.where(w > 1e-12)[0]]
    supw = [float(w[t]) for t in np.where(w > 1e-12)[0]]
    ok = ub <= alpha
    return CertifiedMixture(w, sup, supw, float(ub), float(mixed.mean()),
                            float(cost_mix), alpha, delta, ok, len(H),
                            bound_kind=kind,
                            fallback=None if ok else "train-lp-rejected",
                            diagnostics={"train_bar": tr_bar.tolist(),
                                         "frontier": H})


def mixture_population_risk(pop_losses: np.ndarray, weights: np.ndarray,
                            frontier: list[int]) -> float:
    """Exact risk of a mixture on a finite population (for evaluation only)."""
    r = 0.0
    for t, k in enumerate(frontier):
        if weights[t] > 1e-12:
            r += weights[t] * float(pop_losses[k].mean())
    return float(r)


# ---------------------------------------------------------------------------
# randomized plans as first-class citizens of a fixed-sequence chain
# ---------------------------------------------------------------------------

@dataclass
class ChainElement:
    """One testable plan: a deterministic candidate or a two-plan mixture."""
    support: tuple[int, ...]
    weights: tuple[float, ...]
    train_risk: float
    train_cost: float

    def is_mixture(self) -> bool:
        return len(self.support) > 1

    def describe(self) -> str:
        if not self.is_mixture():
            return f"#{self.support[0]}"
        return " + ".join(f"{w:.2f}*#{k}"
                          for k, w in zip(self.support, self.weights))

    def realize(self, losses: np.ndarray) -> np.ndarray:
        """Rao-Blackwellized per-query loss of this plan on a split."""
        if not self.is_mixture():
            return losses[self.support[0]]
        out = np.zeros(losses.shape[1])
        for k, w in zip(self.support, self.weights):
            out += w * losses[k]
        return out


def build_randomized_chain(train_losses: np.ndarray, train_costs: np.ndarray,
                           mix_grid: int = 7, max_frontier: int = 64
                           ) -> list[ChainElement]:
    """Fixed-sequence chain over the Pareto frontier, refined by mixtures.

    The incumbent chain has two defects. It is ordered by train risk over the
    *whole* candidate space, so plans that are expensive AND risky sit in the
    middle and truncate the walk early; and it contains only deterministic
    plans, whose achievable risks are a sparse discrete set, so the walk stops
    at a plan whose risk lies well below alpha and the remaining risk budget
    is never converted into savings.

    Both are fixed here without paying any multiplicity price. Pruning to the
    train Pareto frontier makes risk-ascending order coincide with
    cost-descending order, so no dominated plan can truncate the walk. Then
    every adjacent pair on the frontier is interpolated by a fixed grid of
    mixture weights, which densifies the reachable risks into a near-continuum
    while keeping every chain element a function of TRAIN data only -- exactly
    the condition under which a fixed-sequence walk controls FWER at delta
    with no correction.
    """
    tr_risk = train_losses.mean(axis=1)
    tr_cost = train_costs.mean(axis=1)
    front = pareto_frontier_idx(tr_risk, tr_cost, max_frontier)
    # frontier is cost-ascending / risk-descending; walk it risk-ascending
    front = sorted(front, key=lambda k: tr_risk[k])
    chain: list[ChainElement] = []
    for pos, k in enumerate(front):
        chain.append(ChainElement((int(k),), (1.0,), float(tr_risk[k]),
                                  float(tr_cost[k])))
        if pos + 1 < len(front):
            j = front[pos + 1]
            for t in np.linspace(0.0, 1.0, mix_grid + 2)[1:-1]:
                r = (1 - t) * tr_risk[k] + t * tr_risk[j]
                c = (1 - t) * tr_cost[k] + t * tr_cost[j]
                chain.append(ChainElement((int(k), int(j)),
                                          (float(1 - t), float(t)),
                                          float(r), float(c)))
    chain.sort(key=lambda e: (e.train_risk, e.train_cost))
    return chain


@dataclass
class CertifiedPlan:
    element: ChainElement | None
    certified: bool
    cal_risk_hat: float
    cal_risk_bound: float
    cal_cost: float
    alpha: float
    delta: float
    n_tested: int
    n_certified: int
    bound_kind: str = "hb"
    fallback: str | None = None
    diagnostics: list = field(default_factory=list)

    def describe(self) -> str:
        return self.element.describe() if self.element else "none"


def certify_randomized_chain(cal_losses: np.ndarray, cal_costs: np.ndarray,
                             chain: list[ChainElement], alpha: float,
                             delta: float, bound: str = "auto",
                             keep_diagnostics: bool = False) -> CertifiedPlan:
    """Fixed-sequence certification over a chain containing mixtures.

    Walks the chain from safest to riskiest, stopping at the first plan whose
    upper confidence bound on risk exceeds alpha, and deploys the cheapest
    plan among those certified. FWER is delta with no correction: the first
    plan in the chain whose true risk exceeds alpha is the only place an error
    can originate, and testing it is a single hypothesis.

    `bound='auto'` spends delta on a Hoeffding-Bentkus bound for deterministic
    (binary-loss) elements and on min(HB, empirical-Bernstein) at delta/2 each
    for mixtures, whose Rao-Blackwellized losses are concentrated enough that
    the variance-adaptive bound usually wins.
    """
    n = cal_losses.shape[1]
    cost_mean = cal_costs.mean(axis=1)
    certified = []
    diag = []
    for e in chain:
        y = e.realize(cal_losses)
        r_hat = float(y.mean())
        # a certified plan only needs the TEST decision, not the inverted
        # bound: reject H0: R > alpha iff the p-value is at most delta.
        if bound == "hb" or not e.is_mixture():
            ok = hb_p_value(r_hat, n, alpha) <= delta
            kind = "hb"
        elif bound == "eb":
            ok = emp_bernstein_upper(y, delta) <= alpha
            kind = "eb"
        else:
            ok_hb = hb_p_value(r_hat, n, alpha) <= delta / 2.0
            ok_eb = emp_bernstein_upper(y, delta / 2.0) <= alpha
            ok = ok_hb or ok_eb
            kind = "hb" if ok_hb else ("eb" if ok_eb else "none")
        cost = float(sum(w * cost_mean[k]
                         for k, w in zip(e.support, e.weights)))
        if keep_diagnostics:
            diag.append({"plan": e.describe(), "train_risk": e.train_risk,
                         "cal_risk": r_hat, "ok": bool(ok), "kind": kind,
                         "cal_cost": cost})
        if ok:
            certified.append((e, r_hat, cost, kind))
        else:
            break
    n_tested = len(diag) if keep_diagnostics else len(certified) + 1
    if not certified:
        return CertifiedPlan(None, False, float("nan"), float("nan"),
                             float("nan"), alpha, delta, n_tested, 0,
                             fallback="nothing-certified", diagnostics=diag)
    best = min(certified, key=lambda t: t[2])
    ub = hb_upper_bound(best[1], n, delta)
    return CertifiedPlan(best[0], True, best[1], ub, best[2], alpha,
                         delta, n_tested, len(certified), best[3],
                         diagnostics=diag)
