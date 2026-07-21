"""Assemble ladder matrices + scores into LadderData; train sufficiency
scorers; evaluate stopping policies and baselines.

Policy semantics (progressive execution): a query stops at the first rung j
whose sufficiency score s_j >= lambda_j; cumulative cost/latency include all
rungs up to j (retrieval reuse means rung j's retrieval work subsumes earlier
rungs' in practice; we charge the conservative cumulative sum).
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")

FEATURES = ["bm25_top", "bm25_margin", "dense_top", "dense_margin", "rerank_top",
            "rerank_mean3", "nli_support", "abstain", "ans_len", "frac_fresh",
            "min_age_h", "kg_n", "n_evidence", "nli_cite_cov", "frac_cited",
            "n_sents", "row_top_overlap", "n_pool"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def feature_vector(feats: dict) -> list[float]:
    return [float(feats.get(k, 0.0)) for k in FEATURES]


class TrackData:
    """(query, rung) records with losses, costs, features for one split."""

    def __init__(self, track: str, split: str, loss_fn, kg_costs=None):
        matrix = load_jsonl(os.path.join(EXP, f"{track}_{split}_matrix.jsonl"))
        score_path = os.path.join(EXP, f"{track}_{split}_scores.jsonl")
        scores = load_jsonl(score_path) if os.path.exists(score_path) else []
        smap = {(s["qid"], s["rung"]): s for s in scores}
        by_q = {}
        for r in matrix:
            key = (r["qid"], r["rung"])
            sc = smap.get(key, {})
            by_q.setdefault(r["qid"], {})[r["rung"]] = (r, sc)
        # keep only queries with all rungs present
        self.n_rungs = max(max(d.keys()) for d in by_q.values()) + 1
        self.qids = [q for q, d in by_q.items() if len(d) == self.n_rungs]
        self.records = {q: by_q[q] for q in self.qids}
        self.loss_fn = loss_fn
        self.kg_costs = kg_costs or {}

    def build(self, groups_map=None):
        """Returns (LadderData, qids, groups, aux) with aux per (rung, qid)."""
        from contractrag.calibrate import LadderData
        L, n = self.n_rungs, len(self.qids)
        losses = np.zeros((L, n))
        raw_feats = np.zeros((L, n, len(FEATURES)))
        costs = np.zeros((L, n))
        lats = np.zeros((L, n))
        aux = {}
        for i, qid in enumerate(self.qids):
            for j in range(L):
                rec, sc = self.records[qid][j]
                losses[j, i] = self.loss_fn(rec, sc)
                raw_feats[j, i] = feature_vector(rec["features"])
                c = rec["cost_cny"]
                lat = rec["latency_retrieval"] + rec["latency_llm"]
                if j == L - 1 and qid in self.kg_costs:
                    c += self.kg_costs[qid]["cost"]
                    lat += self.kg_costs[qid]["latency"]
                costs[j, i] = c
                lats[j, i] = lat
                aux[(j, qid)] = {"answer": rec["answer"], "score": sc}
        groups = None
        if groups_map:
            groups = np.array([groups_map.get(q, "na") for q in self.qids])
        # scores filled later by a scorer
        ld = LadderData(losses=losses, scores=np.zeros((L, n)), costs=costs)
        return ld, self.qids, groups, {"latency": lats, "raw_feats": raw_feats}


class RungScorer:
    """Per-rung logistic model: P(no contract violation | stop at rung j).

    Trained on the train split; used as the sufficiency score everywhere.
    Falls back to a single informative feature if sklearn fails.
    """

    def __init__(self):
        self.models = []

    def fit(self, ld, raw_feats):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        L = ld.L
        self.models = []
        for j in range(L):
            X, y = raw_feats[j], 1.0 - ld.losses[j]
            if len(np.unique(y)) < 2:
                self.models.append(("const", float(y.mean())))
                continue
            m = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=1000, C=1.0))
            m.fit(X, y)
            self.models.append(("lr", m))
        return self

    def score(self, raw_feats):
        L = raw_feats.shape[0]
        out = np.zeros((L, raw_feats.shape[1]))
        for j in range(L):
            kind, m = self.models[j]
            if kind == "const":
                out[j] = m
            else:
                out[j] = m.predict_proba(raw_feats[j])[:, 1]
        # final rung always stops
        out[L - 1] = np.inf
        return out


# ---------------------------------------------------------------------------
# policy evaluation
# ---------------------------------------------------------------------------

def eval_policy(ld, lats, thresholds):
    """Returns dict of realized metrics for stopping policy `thresholds`."""
    from contractrag.calibrate import stop_rung
    stop = stop_rung(ld.scores, np.asarray(thresholds))
    idx = np.arange(ld.n)
    loss = ld.losses[stop, idx]
    cum_cost = np.cumsum(ld.costs, axis=0)[stop, idx]
    cum_lat = np.cumsum(lats, axis=0)[stop, idx]
    return {
        "risk": float(loss.mean()),
        "cost_mean": float(cum_cost.mean()),
        "cost_total": float(cum_cost.sum()),
        "lat_p50": float(np.percentile(cum_lat, 50)),
        "lat_p95": float(np.percentile(cum_lat, 95)),
        "stop_hist": np.bincount(stop, minlength=ld.L).tolist(),
    }


def eval_fixed(ld, lats, rung):
    idx = np.arange(ld.n)
    loss = ld.losses[rung, idx]
    return {
        "risk": float(loss.mean()),
        "cost_mean": float(ld.costs[rung].mean()),
        "lat_p50": float(np.percentile(lats[rung], 50)),
        "lat_p95": float(np.percentile(lats[rung], 95)),
        "stop_hist": [int(rung == j) * ld.n for j in range(ld.L)],
    }


def oracle_policy(ld, lats):
    """Cheapest rung with zero loss per query (skyline, unachievable)."""
    L, n = ld.L, ld.n
    cum_cost = np.cumsum(ld.costs, axis=0)
    cum_lat = np.cumsum(lats, axis=0)
    stop = np.full(n, L - 1)
    for i in range(n):
        for j in range(L):
            if ld.losses[j, i] == 0:
                stop[i] = j
                break
    idx = np.arange(n)
    return {
        "risk": float(ld.losses[stop, idx].mean()),
        "cost_mean": float(cum_cost[stop, idx].mean()),
        "lat_p50": float(np.percentile(cum_lat[stop, idx], 50)),
        "lat_p95": float(np.percentile(cum_lat[stop, idx], 95)),
        "stop_hist": np.bincount(stop, minlength=L).tolist(),
    }
