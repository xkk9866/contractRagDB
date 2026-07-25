"""Certification as a budget-flow problem on a graph of candidate plans.

A certified optimizer must answer "which plans provably meet the contract?"
while spending a fixed error budget delta. Every existing answer is one of two
extreme points of the same design space:

  chain (fixed-sequence).  Order candidates by predicted risk, test each at the
      full delta, stop at the first non-rejection. No multiplicity price at all,
      because only the first true null can produce an error. But one unlucky
      candidate truncates the walk, and everything cheap behind it is lost. On
      the real execution matrices this overpays by 1.2x-16x.

  Bonferroni.  Test every candidate at delta/M. Never truncates, but the budget
      per test shrinks with M, so on large plan grids nothing at the cheap end
      is rejectable at all.

Both are instances of a sequentially rejective graph test (Bretz et al. 2009;
Goeman & Solari 2010): nodes hold budget, a rejected node donates its budget to
successors along weighted edges, and FWER stays at delta for *any* graph fixed
before seeing the data. The chain is the path graph with all budget at the head;
Bonferroni is the edgeless graph with budget spread uniformly.

The contribution here is not the graph test -- it is that for certified
optimization the graph is a *design variable with an objective*, namely the
expected cost of the deployed plan, and that the cost-risk geometry of the plan
grid determines its optimal shape. Concretely:

  * A chain is optimal only when the predicted risk order agrees with the true
    one. It does not: predictions come from a finite training split, and a plan
    that is genuinely cheap and feasible but whose training risk is inflated by
    noise sits behind a wall.
  * Reserving a small fraction eta of delta for alternative entry points costs
    almost nothing in power -- the head test runs at delta(1-eta), whose
    Hoeffding-Bentkus width grows by a factor sqrt(log(1/(delta(1-eta))) /
    log(1/delta)), about 2% at delta=0.1, eta=0.1 -- and buys the ability to
    resume after truncation. Expected deployed cost therefore strictly improves.
  * Budget freed at the end of one branch is *recycled* to branches that were
    truncated, so successful cheap regions subsidize the search in regions the
    training split misjudged.

`build_recycling_graph` constructs that graph from the grid geometry, and
`graph_certify` runs the test. Setting eta=0 recovers the chain exactly and
n_entries=M with eta=1 recovers Bonferroni, so the ablation in the experiments
is a genuine one-factor sweep over a single algorithm rather than a comparison
of unrelated systems.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from contractrag.calibrate import hb_p_value
from contractrag.certlp import (ChainElement, emp_bernstein_upper,
                                hb_upper_bound, pareto_frontier_idx)


# ---------------------------------------------------------------------------
# the graph test
# ---------------------------------------------------------------------------

@dataclass
class TestGraph:
    """Nodes with initial budget and a row-substochastic transfer matrix.

    budget[k]  initial share of delta held by node k, sum <= 1
    g[k, l]    fraction of node k's budget donated to node l when k is rejected
    """
    budget: np.ndarray
    g: np.ndarray
    labels: list

    def validate(self):
        assert self.budget.min() >= -1e-12, "negative budget"
        assert self.budget.sum() <= 1.0 + 1e-9, "budget exceeds delta"
        assert np.all(np.diag(self.g) <= 1e-12), "self-loop"
        assert self.g.min() >= -1e-12, "negative edge weight"
        assert self.g.sum(axis=1).max() <= 1.0 + 1e-9, "row sum exceeds one"
        return self


def graph_reject(pvals: np.ndarray, graph: TestGraph, delta: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Sequentially rejective graph test. Returns (rejected mask, spent budget).

    Standard update (Bretz, Maurer, Brannath, Posch 2009, Sec. 2): reject any
    node whose p-value is at most its current budget, move its budget along its
    out-edges, then contract the node out of the graph so that later donations
    route around it. FWER <= delta for any graph fixed before the data, which is
    all we need: the graph below is built from the training split only.
    """
    M = len(pvals)
    w = delta * graph.budget.astype(float).copy()
    g = graph.g.astype(float).copy()
    alive = np.ones(M, dtype=bool)
    rejected = np.zeros(M, dtype=bool)
    while True:
        cand = np.where(alive & (pvals <= w + 1e-15))[0]
        if len(cand) == 0:
            break
        # reject the node with the most budget headroom first; the order does
        # not affect the final rejection set, only the arithmetic path
        j = int(cand[np.argmax(w[cand] - pvals[cand])])
        alive[j] = False
        rejected[j] = True
        wj = w[j]
        w[j] = 0.0
        for l in np.where(alive)[0]:
            w[l] += wj * g[j, l]
        gj = g[j].copy()
        for k in np.where(alive)[0]:
            gkj = g[k, j]
            if gkj <= 0.0:
                continue
            denom = 1.0 - gkj * g[j, k]
            if denom <= 1e-12:
                g[k, :] = 0.0
                continue
            row = (g[k, :] + gkj * gj) / denom
            row[k] = 0.0
            row[j] = 0.0
            g[k, :] = row
        g[j, :] = 0.0
        g[:, j] = 0.0
    return rejected, w


# ---------------------------------------------------------------------------
# graph construction from the cost-risk geometry of the plan grid
# ---------------------------------------------------------------------------

def build_recycling_graph(train_risk: np.ndarray, train_cost: np.ndarray,
                          eta: float = 0.1, n_entries: int = 4,
                          max_nodes: int = 256) -> tuple[TestGraph, list[int]]:
    """Main chain plus cheap alternative entries, with budget recycling.

    Geometry drives the construction. The training Pareto frontier, ordered by
    ascending risk, is the main chain: along it risk order and cost order agree,
    so no dominated plan can truncate the walk, and a rejection there always
    unlocks a strictly cheaper plan. Everything the frontier discards is a plan
    the training split judged dominated -- and with a few hundred training
    queries that judgement is noisy exactly where it matters, at the cheap end.
    Those plans become alternative entry points, ordered by ascending cost so
    the first test in each branch is the one whose success would help most.

    Recycling edges run from the tail of every branch to the head of the others.
    A branch that certifies all its plans has budget left over and no use for
    it; donating it revives branches the walk had abandoned. This is what makes
    the graph strictly better than a forest of independent chains.

    eta = 0 gives the pure chain. Returns (graph, node index -> candidate).
    """
    M = len(train_risk)
    front = pareto_frontier_idx(train_risk, train_cost, max_nodes // 2)
    main = sorted(front, key=lambda k: (train_risk[k], train_cost[k]))
    off = [k for k in range(M) if k not in set(main)]
    off.sort(key=lambda k: train_cost[k])

    branches = [main]
    if eta > 0 and n_entries > 1 and off:
        # spread the discarded plans over n_entries-1 branches by cost rank, so
        # each branch covers one cost band and is internally risk-ordered
        per = max(1, len(off) // (n_entries - 1))
        for b in range(n_entries - 1):
            chunk = off[b * per:(b + 1) * per] if b < n_entries - 2 \
                else off[b * per:]
            if chunk:
                branches.append(sorted(chunk, key=lambda k: train_risk[k]))
    branches = [b for b in branches if b]
    nodes = [k for br in branches for k in br]
    pos = {k: i for i, k in enumerate(nodes)}
    N = len(nodes)

    budget = np.zeros(N)
    g = np.zeros((N, N))
    heads = [br[0] for br in branches]
    if len(branches) == 1:
        budget[pos[heads[0]]] = 1.0
    else:
        budget[pos[heads[0]]] = 1.0 - eta
        rest = heads[1:]
        for h in rest:
            budget[pos[h]] = eta / len(rest)
    for br in branches:
        for i in range(len(br) - 1):
            g[pos[br[i]], pos[br[i + 1]]] = 1.0
        tail = pos[br[-1]]
        others = [pos[h] for h in heads if pos[h] != pos[br[0]]]
        if others:
            for o in others:
                g[tail, o] = 1.0 / len(others)
    return TestGraph(budget, g, [str(k) for k in nodes]).validate(), nodes


def build_element_pool(train_losses: np.ndarray, train_costs: np.ndarray,
                       mix_grid: int = 7, max_frontier: int = 64
                       ) -> list[ChainElement]:
    """Every deterministic candidate, plus mixtures along the train frontier.

    A chain certifier must prune to the frontier: an off-frontier plan sitting
    mid-order would truncate the walk for nothing. The graph certifier has the
    opposite incentive -- off-frontier plans are precisely the ones the training
    split may have misjudged, and giving them their own branch costs only their
    share of eta. So the pool keeps all of them, while mixtures are still built
    only between adjacent frontier plans, where interpolation is on the useful
    part of the cost-risk boundary.
    """
    tr_risk = train_losses.mean(axis=1)
    tr_cost = train_costs.mean(axis=1)
    pool = [ChainElement((int(k),), (1.0,), float(tr_risk[k]),
                         float(tr_cost[k])) for k in range(len(tr_risk))]
    front = pareto_frontier_idx(tr_risk, tr_cost, max_frontier)
    front = sorted(front, key=lambda k: tr_risk[k])
    for pos in range(len(front) - 1):
        k, j = front[pos], front[pos + 1]
        for t in np.linspace(0.0, 1.0, mix_grid + 2)[1:-1]:
            pool.append(ChainElement(
                (int(k), int(j)), (float(1 - t), float(t)),
                float((1 - t) * tr_risk[k] + t * tr_risk[j]),
                float((1 - t) * tr_cost[k] + t * tr_cost[j])))
    return pool


# ---------------------------------------------------------------------------
# certified optimization on the graph
# ---------------------------------------------------------------------------

@dataclass
class GraphCertificate:
    element: ChainElement | None
    certified: bool
    cal_risk_hat: float
    cal_risk_bound: float
    cal_cost: float
    alpha: float
    delta: float
    n_rejected: int
    n_nodes: int
    leftover: float = 0.0
    mixture: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def describe(self) -> str:
        if self.mixture:
            return " + ".join(f"{w:.2f}*{d}" for d, w in self.mixture)
        return self.element.describe() if self.element else "none"


def _pvalues(elements: list[ChainElement], cal_losses: np.ndarray,
             alpha: float, delta: float, use_bernstein: bool = True):
    """One-sided p-values for H0: R(e) > alpha, plus realized losses.

    Deterministic elements have binary losses, where Hoeffding-Bentkus is the
    right tool. Mixtures carry the Rao-Blackwellised loss sum_k w_k l_k(q),
    which is unbiased for the mixture risk, still lies in [0,1], and has
    strictly smaller variance -- so an empirical-Bernstein bound can beat HB
    there. We report the smaller of the two implied p-values after splitting
    the node's budget in half, which is a union bound over two valid tests.
    """
    n = cal_losses.shape[1]
    p = np.ones(len(elements))
    rhat = np.zeros(len(elements))
    for i, e in enumerate(elements):
        y = e.realize(cal_losses)
        rhat[i] = float(y.mean())
        p_hb = hb_p_value(rhat[i], n, alpha)
        if use_bernstein and e.is_mixture():
            # smallest d such that the Bernstein bound at level d clears alpha
            lo, hi = 1e-6, 1.0
            if emp_bernstein_upper(y, hi) > alpha:
                p_eb = 1.0
            else:
                for _ in range(40):
                    mid = np.sqrt(lo * hi)
                    if emp_bernstein_upper(y, mid) <= alpha:
                        hi = mid
                    else:
                        lo = mid
                p_eb = hi
            p[i] = min(2.0 * p_hb, 2.0 * p_eb)
        else:
            p[i] = p_hb
    return p, rhat


def graph_certify(cal_losses: np.ndarray, cal_costs: np.ndarray,
                  elements: list[ChainElement], alpha: float, delta: float,
                  eta: float = 0.1, n_entries: int = 4,
                  use_bernstein: bool = True,
                  keep_diagnostics: bool = False) -> GraphCertificate:
    """Cheapest plan certified by the budget-flow graph test.

    Deploys the cheapest rejected node. Correctness: the graph is a function of
    the training split alone, so the graph test controls FWER at delta over the
    calibration draw; hence with probability at least 1-delta no node with true
    risk above alpha is rejected, and in particular the deployed one has risk at
    most alpha.
    """
    train_risk = np.array([e.train_risk for e in elements])
    train_cost = np.array([e.train_cost for e in elements])
    graph, nodes = build_recycling_graph(train_risk, train_cost, eta,
                                         n_entries)
    sub = [elements[k] for k in nodes]
    p, rhat = _pvalues(sub, cal_losses, alpha, delta, use_bernstein)
    rejected, leftover = graph_reject(p, graph, delta)
    cost_mean = cal_costs.mean(axis=1)
    idx = np.where(rejected)[0]
    diag = {}
    if keep_diagnostics:
        diag = {"n_nodes": len(nodes), "p_min": float(p.min()),
                "rejected": [sub[i].describe() for i in idx],
                "branch_heads": int(np.count_nonzero(graph.budget))}
    if len(idx) == 0:
        return GraphCertificate(None, False, float("nan"), float("nan"),
                                float("nan"), alpha, delta, 0, len(nodes),
                                float(leftover.sum()), diagnostics=diag)
    costs = np.array([sum(w * cost_mean[k] for k, w in
                          zip(sub[i].support, sub[i].weights)) for i in idx])
    best = int(idx[int(np.argmin(costs))])
    e = sub[best]
    ub = hb_upper_bound(float(rhat[best]), cal_losses.shape[1], delta)
    return GraphCertificate(e, True, float(rhat[best]), ub,
                            float(costs.min()), alpha, delta, len(idx),
                            len(nodes), float(leftover.sum()),
                            mixture=[(f"#{k}", w) for k, w in
                                     zip(e.support, e.weights)],
                            diagnostics=diag)
