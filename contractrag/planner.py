"""Cardinality-driven physical planning for hybrid (table + text) retrieval.

This module gives ContractRAG the classical database-optimizer machinery the
model-cascade view lacks: selectivity/cardinality estimation, access-path
selection, and cost-based join ordering between a structured source (table
rows) and an unstructured source (linked passages). The chosen access path is
still handed to the *certification* layer, so plan choice is cost-based while
soundness stays statistical.

Join semantics (HybridQA-style). A query touches two sources:
  - S (structured): rows of a table matching a lexical/structured predicate;
  - T (text): passages linked from rows (row -> passages foreign key).
The answer needs evidence joined across S and T. Two join orders exist:
  * TABLE_FIRST  (predicate pushdown): select rows by the predicate, then
    retrieve passages only from the selected rows' link set (small pool).
  * TEXT_FIRST   : retrieve passages from the whole table's pool, then join
    back to their rows (large pool, higher recall on vague predicates).
  * HYBRID_UNION : run both and fuse (highest recall, highest cost).

The right order depends on the predicate's selectivity: a highly selective
predicate makes TABLE_FIRST cheap and safe; a vague predicate makes it drop
relevant passages, so TEXT_FIRST wins on recall. This is exactly the
selectivity-vs-access-path tradeoff of a relational optimizer, transplanted
to retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AccessPath(str, Enum):
    TABLE_FIRST = "table_first"     # predicate pushdown, small pool
    TEXT_FIRST = "text_first"       # full pool, post-join
    HYBRID_UNION = "hybrid_union"   # both, fused


@dataclass
class Cardinalities:
    """Estimated intermediate sizes for a query's join."""
    n_rows_total: int
    n_rows_sel: int          # rows surviving the structured predicate
    n_pool_pushdown: int     # passages reachable from selected rows
    n_pool_full: int         # passages reachable from the whole table

    @property
    def row_selectivity(self) -> float:
        return self.n_rows_sel / max(1, self.n_rows_total)

    @property
    def pool_selectivity(self) -> float:
        return self.n_pool_pushdown / max(1, self.n_pool_full)


def estimate_cardinalities(table, question, tokenizer, match_threshold: int = 1
                           ) -> Cardinalities:
    """Selectivity/cardinality estimator from the structured predicate.

    A row 'matches' the predicate if it shares >= match_threshold content
    tokens with the question (a cheap proxy for a WHERE clause). Pool sizes
    come from the row->passage link table (the foreign key).
    """
    q = tokenizer(question)
    row_links = table["row_links"]
    n_rows = len(table["rows"])
    sel_rows = []
    for i, row in enumerate(table["rows"]):
        cells = tokenizer(" ".join(str(c) for c in row))
        if len(q & cells) >= match_threshold:
            sel_rows.append(i)
    push_pool = {p for i in sel_rows for p in row_links[i]}
    full_pool = {p for links in row_links for p in links}
    return Cardinalities(
        n_rows_total=n_rows,
        n_rows_sel=max(1, len(sel_rows)),
        n_pool_pushdown=max(1, len(push_pool)),
        n_pool_full=max(1, len(full_pool)),
    )


@dataclass
class CostModel:
    """Per-operator cost constants (seconds), fit from measured executions.

    total cost(access path) ~= c_embed * (#passages embedded)
                             + c_rerank * (#passages reranked)
                             + c_gen (fixed per query, model-dependent).
    Only retrieval-side terms differ across access paths; the generator cost
    is added later by the ladder. Defaults are order-of-magnitude seeds and
    are overwritten by fit().
    """
    c_rerank: float = 2.0e-3
    c_embed: float = 3.0e-4
    c_fixed: float = 5.0e-3

    def path_cost(self, card: Cardinalities, k_ret: int) -> dict:
        """Estimated retrieval cost per access path (arbitrary time units)."""
        def one(pool):
            n_re = min(k_ret, pool)
            return self.c_fixed + self.c_embed * pool + self.c_rerank * n_re
        return {
            AccessPath.TABLE_FIRST: one(card.n_pool_pushdown),
            AccessPath.TEXT_FIRST: one(card.n_pool_full),
            AccessPath.HYBRID_UNION: one(card.n_pool_pushdown)
            + one(card.n_pool_full),
        }

    def fit(self, pools: np.ndarray, k_rets: np.ndarray, latencies: np.ndarray):
        """Least-squares fit of (c_fixed, c_embed, c_rerank) to measured
        retrieval latencies. pools[i], k_rets[i], latencies[i] per execution."""
        n_re = np.minimum(k_rets, pools)
        A = np.stack([np.ones_like(pools, dtype=float),
                      pools.astype(float), n_re.astype(float)], axis=1)
        coef, *_ = np.linalg.lstsq(A, latencies.astype(float), rcond=None)
        self.c_fixed, self.c_embed, self.c_rerank = (float(max(0.0, c)) for c in coef)
        return self


@dataclass
class RecallModel:
    """Estimated evidence-recall of each access path as a function of the
    predicate's pool selectivity sigma = pool_pushdown / pool_full.

    TABLE_FIRST recall degrades when sigma is small (pushdown drops relevant
    passages linked from unselected rows). We model it as recall_tf(sigma) =
    r_lo + (r_hi - r_lo) * sigma^gamma, calibrated from data; TEXT_FIRST and
    HYBRID_UNION recall are ~constant (full pool). Coefficients are fit on the
    training split by matching realized loss.
    """
    r_lo: float = 0.55
    r_hi: float = 0.98
    gamma: float = 0.5
    r_text: float = 0.95

    def recall(self, path: AccessPath, sigma: float) -> float:
        if path == AccessPath.TABLE_FIRST:
            return self.r_lo + (self.r_hi - self.r_lo) * (sigma ** self.gamma)
        if path == AccessPath.TEXT_FIRST:
            return self.r_text
        return max(self.r_text, self.r_lo + (self.r_hi - self.r_lo) * (sigma ** self.gamma))

    def fit(self, sigmas: np.ndarray, tf_ok: np.ndarray, xf_ok: np.ndarray):
        """Fit the TABLE_FIRST recall curve r_lo/r_hi/gamma to per-query
        success indicators (1 = access path produced a non-violating answer).
        Simple grid search over gamma; endpoints from selectivity extremes."""
        self.r_text = float(np.mean(xf_ok)) if len(xf_ok) else self.r_text
        if len(sigmas) < 8:
            return self
        lo_mask = sigmas <= np.quantile(sigmas, 0.25)
        hi_mask = sigmas >= np.quantile(sigmas, 0.75)
        self.r_lo = float(np.clip(np.mean(tf_ok[lo_mask]) if lo_mask.any() else 0.55,
                                  0.01, 0.99))
        self.r_hi = float(np.clip(np.mean(tf_ok[hi_mask]) if hi_mask.any() else 0.98,
                                  self.r_lo + 1e-3, 0.999))
        best_g, best_err = self.gamma, np.inf
        for g in np.linspace(0.2, 3.0, 29):
            pred = self.r_lo + (self.r_hi - self.r_lo) * (sigmas ** g)
            err = float(np.mean((pred - tf_ok) ** 2))
            if err < best_err:
                best_g, best_err = g, err
        self.gamma = float(best_g)
        return self


def select_access_path(card: Cardinalities, cost: CostModel, recall: RecallModel,
                       recall_target: float = 0.9) -> AccessPath:
    """Cost-based access-path / join-order choice: cheapest path whose
    estimated recall meets the target; fall back to the highest-recall path.
    This is the plan-choice analogue of an optimizer picking an index scan vs
    a full scan from estimated selectivity."""
    sigma = card.pool_selectivity
    costs = cost.path_cost(card, k_ret=24)
    feasible = [p for p in AccessPath if recall.recall(p, sigma) >= recall_target]
    if feasible:
        return min(feasible, key=lambda p: costs[p])
    return max(AccessPath, key=lambda p: recall.recall(p, sigma))
