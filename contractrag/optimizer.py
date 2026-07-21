"""Contract-Cascades: certified cost minimization over a rich policy space.

A *policy* is any deterministic mapping q -> execution schedule; here we
materialize each candidate as (stop_rungs, mode) where mode 'escalate' pays
cumulative cost 0..j and mode 'jump' pays only rung j (router semantics:
commit to a plan upfront, no fallback signal).

Candidate families (all parameterized on TRAIN data only; calibration data
is touched exactly once by the fixed-sequence test):
  T-family: escalation thresholds along the 1-D quantile path;
  U-family: utility-router jump policies over a lambda grid;
  G-family: group-conditional thresholds (optional);
  F-family: fixed rungs (jump), the classical plans.

Certification: candidates are ordered by train-set risk (safest first);
fixed-sequence Hoeffding-Bentkus tests on the calibration set walk this
order and stop at the first non-rejection; among certified candidates the
cheapest (calibration mean cost) is deployed.  FWER <= delta, hence the
deployed policy satisfies R(pi) <= alpha with prob >= 1-delta.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contractrag.calibrate import hb_p_value, stop_rung
from contractrag.baselines import utility_route


@dataclass
class Candidate:
    kind: str            # 'thr' | 'util' | 'fixed' | 'group'
    param: object        # thresholds array | lambda | rung index
    mode: str            # 'escalate' | 'jump'
    train_risk: float = 0.0
    train_cost: float = 0.0

    def describe(self):
        if self.kind == "thr":
            return f"thr(q={self.param[0]:.2f})"
        if self.kind == "util":
            return f"util(lam={self.param:.3g})"
        if self.kind == "fixed":
            return f"fixed({self.param})"
        return self.kind


def apply_candidate(cand: Candidate, ld, lats=None):
    """Realized (loss, cost, lat, stop) vectors of a candidate on a dataset."""
    idx = np.arange(ld.n)
    if cand.kind == "thr":
        stop = stop_rung(ld.scores, np.asarray(cand.param[1]))
    elif cand.kind == "util":
        stop = utility_route(ld.scores, ld.costs.mean(axis=1), cand.param)
    elif cand.kind == "fixed":
        stop = np.full(ld.n, int(cand.param))
    else:
        raise ValueError(cand.kind)
    stop = np.clip(stop, 0, ld.L - 1)
    loss = ld.losses[stop, idx]
    if cand.mode == "escalate":
        cost = np.cumsum(ld.costs, axis=0)[stop, idx]
        lat = np.cumsum(lats, axis=0)[stop, idx] if lats is not None else None
    else:
        cost = ld.costs[stop, idx]
        lat = lats[stop, idx] if lats is not None else None
    return loss, cost, lat, stop


def build_candidates(ld_train, num_thr=40, num_lam=25) -> list[Candidate]:
    from contractrag.calibrate import quantile_path
    cands = []
    for q, thr in quantile_path(ld_train, num_thr):
        cands.append(Candidate("thr", (q, thr), "escalate"))
    for lam in np.logspace(-3, 3, num_lam):
        cands.append(Candidate("util", float(lam), "jump"))
    for j in range(ld_train.L):
        cands.append(Candidate("fixed", j, "jump"))
    # annotate with train stats
    for c in cands:
        loss, cost, _, _ = apply_candidate(c, ld_train)
        c.train_risk = float(loss.mean())
        c.train_cost = float(cost.mean())
    return cands


def build_candidates_grid(ld_train, grid_per_rung: int = 20, cap: int = 200000
                          ) -> list[Candidate]:
    """Full Cartesian threshold plan space (for plan-space scalability study).

    Instead of the 1-D quantile path, enumerate independent per-rung
    thresholds: each of the L-1 escalation thresholds ranges over a
    per-rung score-quantile grid of size `grid_per_rung`, giving up to
    grid_per_rung^(L-1) escalation policies -- a combinatorial plan space
    analogous to join-order enumeration. Capped at `cap` candidates (uniform
    subsample of the product) so the certifier's O(K n) cost is the only thing
    that grows. Fixed and utility families are appended.
    """
    import itertools
    L = ld_train.L
    grids = [np.quantile(ld_train.scores[j], np.linspace(0, 1, grid_per_rung))
             for j in range(L - 1)]
    combos = list(itertools.product(*[range(grid_per_rung) for _ in range(L - 1)]))
    if len(combos) > cap:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(combos), size=cap, replace=False)
        combos = [combos[i] for i in sel]
    cands = []
    for combo in combos:
        thr = np.array([grids[j][combo[j]] for j in range(L - 1)])
        cands.append(Candidate("thr", (0.0, thr), "escalate"))
    for lam in np.logspace(-3, 3, 25):
        cands.append(Candidate("util", float(lam), "jump"))
    for j in range(L):
        cands.append(Candidate("fixed", j, "jump"))
    for c in cands:
        loss, cost, _, _ = apply_candidate(c, ld_train)
        c.train_risk = float(loss.mean())
        c.train_cost = float(cost.mean())
    return cands


def pareto_prune(cands: list[Candidate]) -> list[Candidate]:
    """Dominance-prune a plan space on TRAIN statistics.

    Keep candidates on the (train_risk, train_cost) Pareto frontier: a
    candidate is dropped if another has <= risk and <= cost with one strict.
    This is the optimizer analogue of dominance pruning in join enumeration;
    it uses train data only, so certification validity is untouched, and it
    turns a combinatorial space into a short monotone frontier on which the
    fixed-sequence walk is well-ordered."""
    order = sorted(range(len(cands)), key=lambda i: (cands[i].train_cost,
                                                     cands[i].train_risk))
    front, best_risk = [], np.inf
    # ascending cost: keep only strictly improving risk
    for i in order:
        if cands[i].train_risk < best_risk - 1e-12:
            front.append(cands[i])
            best_risk = cands[i].train_risk
    return front


@dataclass
class OptimizedPolicy:
    candidate: Candidate | None
    certified: bool
    cal_risk: float
    cal_cost: float
    n_certified: int
    n_tested: int
    diagnostics: list = field(default_factory=list)


def robust_order(cands: list[Candidate], ld_train, folds: int = 2,
                 seed: int = 0) -> list[int]:
    """Selection-robust walk order for combinatorial plan spaces.

    Naive train-risk ordering breaks on large enumerated spaces: with a small
    train split, the head of the order is dominated by combos whose LOW train
    risk is selection noise; their calibration risk fails the very first test
    and the fixed-sequence walk certifies nothing. Ordering by the WORST risk
    across train folds (a cross-fit upper proxy) demotes overfit combos while
    leaving genuinely safe candidates in place. The order still depends on
    train data only, so certification validity is untouched (any
    data-independent-of-cal order is admissible for fixed-sequence LTT).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ld_train.n)
    parts = np.array_split(perm, folds)
    keys = []
    for c in cands:
        loss, cost, _, _ = apply_candidate(c, ld_train)
        worst = max(float(loss[p].mean()) for p in parts if len(p))
        keys.append((worst, float(cost.mean())))
    return sorted(range(len(cands)), key=lambda i: keys[i])


def optimize(cands: list[Candidate], ld_cal, alpha: float, delta: float,
             lat_cal=None, lat_budget: float = None, alpha_lat: float = None,
             ld_train=None, lat_train=None, order=None,
             n_pop: int = None) -> OptimizedPolicy:
    """Fixed-sequence LTT over candidates in a train-data order.

    Single constraint: order by train risk ascending (or a caller-provided
    train-only order such as robust_order). With a latency SLO
    (lat_budget/alpha_lat given), order by the composite feasibility margin
    max_j(train_risk_j / alpha_j) ascending, computed on the train split
    (ld_train/lat_train) -- the order stays independent of calibration data.
    Each candidate's composite null is tested by intersection-union (every
    constraint's HB p-value <= delta). If n_pop is given, tests use the
    finite-population p-value (hypergeometric-exact + KL-Chernoff) for
    calibration sets drawn without replacement from a population of that
    size.
    """
    if n_pop is not None:
        from contractrag.calibrate import finite_pop_p_value
        pv = lambda r, n, a: finite_pop_p_value(r, n, a, n_pop)
    else:
        pv = hb_p_value
    if order is not None:
        pass
    elif lat_budget is not None and ld_train is not None:
        keys = []
        for c in cands:
            loss, cost, lat, _ = apply_candidate(c, ld_train, lat_train)
            m = max(float(loss.mean()) / alpha,
                    float((lat > lat_budget).mean()) / alpha_lat)
            keys.append((m, float(cost.mean())))
        order = sorted(range(len(cands)), key=lambda i: keys[i])
    else:
        order = sorted(range(len(cands)), key=lambda i: (cands[i].train_risk,
                                                         cands[i].train_cost))
    certified = []
    diag = []
    n = ld_cal.n
    for oi in order:
        c = cands[oi]
        loss, cost, lat, _ = apply_candidate(c, ld_cal, lat_cal)
        r = float(loss.mean())
        p = pv(r, n, alpha)
        ok = p <= delta
        if ok and lat_budget is not None:
            r_lat = float((lat > lat_budget).mean())
            ok = pv(r_lat, n, alpha_lat) <= delta
        diag.append({"cand": c.describe(), "train_risk": c.train_risk,
                     "cal_risk": r, "p": p, "cal_cost": float(cost.mean())})
        if ok:
            certified.append((oi, r, float(cost.mean())))
        else:
            break
    if not certified:
        # fall back to strongest fixed plan (always answers): report uncertified
        strongest = max(range(len(cands)),
                        key=lambda i: (cands[i].kind == "fixed", cands[i].param
                                       if cands[i].kind == "fixed" else -1))
        c = cands[strongest]
        loss, cost, _, _ = apply_candidate(c, ld_cal)
        return OptimizedPolicy(c, False, float(loss.mean()),
                               float(cost.mean()), 0, len(diag), diag)
    best = min(certified, key=lambda t: t[2])
    return OptimizedPolicy(cands[best[0]], True, best[1], best[2],
                           len(certified), len(diag), diag)
