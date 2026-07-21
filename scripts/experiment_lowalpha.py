"""Low-risk contracts with certified abstention (reviewer fix: contract
levels were only shown in the loose regime).

Strict contracts (alpha = 0.05/0.10/0.20) are infeasible for every
always-answer plan on these workloads (even the strongest plan's risk is
0.12-0.61). The deployable semantics at such levels is SELECTIVE: the ladder
may escalate a query to a human instead of answering. Contract:

    P(answered AND violating) <= alpha   AND   P(deferred) <= beta.

Both are bounded losses, so the intersection-union fixed-sequence test
applies unchanged. The policy family adds one knob: after the final rung,
defer if the final-rung sufficiency score (a train-fit logistic model, like
every other rung's) is below kappa.

For each track and each alpha we report the smallest deferral budget beta in
{0.25, 0.5, 0.75} whose contract certifies, the selected (q, kappa), and
realized test risk / deferral / cost. Pure numpy; no LLM calls.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import (hb_p_value, quantile_path,
                                   stop_rung)  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
DELTA = 0.1
ALPHAS = [0.05, 0.10, 0.20]
BETAS = [0.25, 0.50, 0.75, 0.90]

TRACKS = [
    ("hybridqa", "quality", 0.5),
    ("crag", "correct", 0.5),
    ("asqa", "citation", 50.0),
    ("qampari", "citation", 50.0),
]


def fit_abstain_scorer(ld_tr, feats_tr):
    """Final-rung sufficiency model (RungScorer skips it: score = inf)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    j = ld_tr.L - 1
    X, y = feats_tr[j], 1.0 - ld_tr.losses[j]
    if len(np.unique(y)) < 2:
        return None
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=1000, C=1.0))
    m.fit(X, y)
    return m


def eval_cand(ld, feats, abst_model, thr, kappa):
    """Per-query (answered-violation, deferred, cumulative cost)."""
    idx = np.arange(ld.n)
    stop = stop_rung(ld.scores, np.asarray(thr))
    s_abst = abst_model.predict_proba(feats[ld.L - 1])[:, 1]
    defer = (stop == ld.L - 1) & (s_abst < kappa)
    loss_ans = ld.losses[stop, idx] * (~defer)
    cost = np.cumsum(ld.costs, axis=0)[stop, idx]
    return loss_ans, defer.astype(float), cost


def main():
    out = {"delta": DELTA, "alphas": ALPHAS, "betas": BETAS, "tracks": {}}
    for track, contract, tau in TRACKS:
        loss_fn = make_loss_fn(track, contract, tau)
        kgc = kg_costs_for(track)
        data = {}
        for s in ["train", "cal", "test"]:
            td = TrackData(track, s, loss_fn, kg_costs=kgc)
            ld, _, _, aux = td.build()
            data[s] = {"ld": ld, "feats": aux["raw_feats"]}
        scorer = RungScorer().fit(data["train"]["ld"], data["train"]["feats"])
        for s in data:
            data[s]["ld"].scores = scorer.score(data[s]["feats"])
        abst = fit_abstain_scorer(data["train"]["ld"], data["train"]["feats"])
        if abst is None:
            continue

        ld_tr, f_tr = data["train"]["ld"], data["train"]["feats"]
        ld_cal, f_cal = data["cal"]["ld"], data["cal"]["feats"]
        ld_te, f_te = data["test"]["ld"], data["test"]["feats"]

        # candidate grid: quantile-path q x abstention-quantile kappa
        s_tr = abst.predict_proba(f_tr[ld_tr.L - 1])[:, 1]
        kappas = [0.0] + [float(np.quantile(s_tr, q))
                          for q in np.linspace(0.05, 0.95, 19)]
        qpath = quantile_path(ld_tr, 41)
        cands = [(thr, k) for _, thr in qpath for k in kappas]

        tr_stats = [eval_cand(ld_tr, f_tr, abst, thr, k) for thr, k in cands]
        cal_stats = [eval_cand(ld_cal, f_cal, abst, thr, k)
                     for thr, k in cands]

        strongest = float(np.cumsum(ld_te.costs, axis=0)[-1].mean())
        trec = {"strongest_cost_mCNY": 1000 * strongest, "settings": []}
        for alpha in ALPHAS:
            best = None
            for beta in BETAS:
                # train-only walk order: composite feasibility margin
                keys = [(max(l.mean() / alpha, d.mean() / beta),
                         float(c.mean()))
                        for l, d, c in tr_stats]
                order = sorted(range(len(cands)), key=lambda i: keys[i])
                certified = []
                for oi in order:
                    l, d, c = cal_stats[oi]
                    ok = (hb_p_value(float(l.mean()), ld_cal.n, alpha) <= DELTA
                          and hb_p_value(float(d.mean()), ld_cal.n, beta)
                          <= DELTA)
                    if ok:
                        certified.append((oi, float(c.mean())))
                    else:
                        break
                if certified:
                    oi, _ = min(certified, key=lambda t: t[1])
                    thr, k = cands[oi]
                    l_t, d_t, c_t = eval_cand(ld_te, f_te, abst, thr, k)
                    ans = d_t == 0
                    best = {
                        "alpha": alpha, "beta": beta,
                        "n_certified": len(certified),
                        "kappa": float(k),
                        "test_risk_answered_marginal": float(l_t.mean()),
                        "test_defer_rate": float(d_t.mean()),
                        "test_risk_conditional": float(
                            l_t[ans].mean()) if ans.any() else None,
                        "test_cost_mCNY": 1000 * float(c_t.mean()),
                    }
                    break
            if best is None:
                best = {"alpha": alpha, "beta": None, "certified": False}
            trec["settings"].append(best)
            print(f"{track:>9s} a={alpha:.2f}: " + (
                f"beta={best['beta']} defer={best['test_defer_rate']:.3f} "
                f"risk={best['test_risk_answered_marginal']:.3f} "
                f"cost={best['test_cost_mCNY']:.2f}m "
                f"(K_cert={best['n_certified']})"
                if best.get("beta") is not None else "infeasible"))
        out["tracks"][track] = trec

    path = os.path.join(EXP, "lowalpha_abstain.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
