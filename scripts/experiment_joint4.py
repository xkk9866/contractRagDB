"""Joint multi-constraint evidence contracts (reviewer fix: RQ on
multi-dimensional contracts was under-validated).

Part 1 (CRAG): four-constraint joint contract
    P(not correct) <= a_q  AND  P(hallucination) <= a_h  AND
    P(stale evidence used) <= a_f  AND  P(latency > B) <= a_l
  certified jointly by intersection-union fixed-sequence testing; per-
  constraint empirical risks and HB p-values of the deployed policy are
  reported on cal and test, plus the IUT-vs-Bonferroni power comparison
  (number of candidates certified, cheapest certified cost).

Part 2 (ASQA): four-constraint joint contract
    P(strEM < 30) <= a_q AND P(cit-recall < 50) <= a_r AND
    P(cit-precision < 50) <= a_p AND P(latency > B) <= a_l.

Part 3 (multi-constraint drift monitoring, CRAG fresh->stale): one
  e-detector per constraint, each at delta_mon/4 (union bound = global
  false-alarm budget delta_mon). The index silently falls one week behind:
  only the freshness monitor should alarm (drift attribution), with the
  global budget verified on in-sync controls.

Pure numpy over materialized matrices; no LLM calls.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import (hb_p_value, quantile_path, stop_rung,
                                   EProcessMonitor)  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
DELTA = 0.1


def build_multi(track, split, loss_fn, kgc, constraints):
    """LadderData + dict of extra (L, n) violation matrices + latency."""
    td = TrackData(track, split, loss_fn, kg_costs=kgc)
    ld, qids, _, aux = td.build()
    L, n = ld.L, ld.n
    mats = {name: np.zeros((L, n)) for name in constraints}
    for i, qid in enumerate(td.qids):
        for j in range(L):
            rec, sc = td.records[qid][j]
            for name, fn in constraints.items():
                mats[name][j, i] = fn(rec, sc)
    return ld, mats, aux, td


def crag_fresh_loss(rec, sc):
    meta = rec.get("evidence_meta", {})
    for s, a in zip(meta.get("srcs", []), meta.get("ages", [])):
        if s == "web" and a is not None and a > 72.0:
            return 1.0
    return 0.0


def walk_multi(ld, aux_losses, lat_loss, alphas, delta, mode):
    """Fixed-sequence walk over the quantile path under IUT or Bonferroni.

    Returns (selected thr or None, n_certified, cheapest_cost, diagnostics).
    """
    k = 2 + len(aux_losses)  # primary + aux + latency
    d_each = delta if mode == "iut" else delta / k
    idx = np.arange(ld.n)
    cum_cost = np.cumsum(ld.costs, axis=0)
    best_thr, best_cost, n_cert = None, np.inf, 0
    for q, thr in quantile_path(ld, 201):
        stop = stop_rung(ld.scores, thr)
        risks = {"primary": float(ld.losses[stop, idx].mean())}
        for name, m in aux_losses.items():
            risks[name] = float(m[stop, idx].mean())
        risks["latency"] = float(lat_loss[stop, idx].mean())
        ok = all(hb_p_value(risks[name], ld.n, alphas[name]) <= d_each
                 for name in risks)
        if not ok:
            break
        n_cert += 1
        cost = float(cum_cost[stop, idx].mean())
        if cost < best_cost:
            best_thr, best_cost = thr, cost
    return best_thr, n_cert, best_cost


def eval_multi(ld, aux_losses, lat_loss, lats, thr, alphas):
    idx = np.arange(ld.n)
    stop = stop_rung(ld.scores, np.asarray(thr))
    risks = {"primary": float(ld.losses[stop, idx].mean())}
    for name, m in aux_losses.items():
        risks[name] = float(m[stop, idx].mean())
    risks["latency"] = float(lat_loss[stop, idx].mean())
    out = {"risks": risks,
           "p_values": {name: hb_p_value(risks[name], ld.n, alphas[name])
                        for name in risks},
           "cost_mean": float(np.cumsum(ld.costs, axis=0)[stop, idx].mean()),
           "lat_p95": float(np.percentile(
               np.cumsum(lats, axis=0)[stop, idx], 95))}
    return out


def run_joint(track, contract, tau, constraints, alphas, lat_budget):
    loss_fn = make_loss_fn(track, contract, tau)
    kgc = kg_costs_for(track)
    data = {}
    for s in ["train", "cal", "test"]:
        ld, mats, aux, _ = build_multi(track, s, loss_fn, kgc, constraints)
        data[s] = {"ld": ld, "mats": mats, "aux": aux}
    scorer = RungScorer().fit(data["train"]["ld"],
                              data["train"]["aux"]["raw_feats"])
    for s in data:
        data[s]["ld"].scores = scorer.score(data[s]["aux"]["raw_feats"])

    def lat_mat(s):
        return (np.cumsum(data[s]["aux"]["latency"], axis=0)
                > lat_budget).astype(float)

    rec = {"track": track, "contract": contract, "alphas": alphas,
           "lat_budget": lat_budget, "delta": DELTA,
           "n_cal": data["cal"]["ld"].n, "n_test": data["test"]["ld"].n}
    for mode in ["iut", "bonferroni"]:
        thr, n_cert, cost = walk_multi(
            data["cal"]["ld"], data["cal"]["mats"], lat_mat("cal"),
            alphas, DELTA, mode)
        if thr is None:
            rec[mode] = {"certified": False}
            print(f"{track} {mode}: nothing certifies")
            continue
        m_cal = eval_multi(data["cal"]["ld"], data["cal"]["mats"],
                           lat_mat("cal"), data["cal"]["aux"]["latency"],
                           thr, alphas)
        m_te = eval_multi(data["test"]["ld"], data["test"]["mats"],
                          lat_mat("test"), data["test"]["aux"]["latency"],
                          thr, alphas)
        rec[mode] = {"certified": True, "n_certified": n_cert,
                     "cal": m_cal, "test": m_te}
        print(f"{track} {mode}: {n_cert} certified, cost "
              f"{m_te['cost_mean']*1000:.2f}mCNY, test risks "
              + " ".join(f"{k}={v:.3f}" for k, v in m_te["risks"].items()))
    return rec


def drift_multi():
    """Part 3: per-constraint monitors on the fresh->stale-unaware stream."""
    from scripts.experiment_stale import (TaggedTrackData, loss_correct,
                                          uses_dated_web)
    kgc = kg_costs_for("crag")

    def build(tag, split, order=None):
        td = TaggedTrackData(tag, split, loss_correct, kg_costs=kgc)
        if order is not None:
            keep = set(td.qids)
            td.qids = [q for q in order if q in keep]
            td.records = {q: td.records[q] for q in td.qids}
        ld, qids, _, aux = td.build()
        L, n = ld.L, ld.n
        mats = {"hallu": np.zeros((L, n)), "fresh": np.zeros((L, n))}
        for i, qid in enumerate(td.qids):
            for j in range(L):
                rec, sc = td.records[qid][j]
                mats["hallu"][j, i] = float(sc.get("label") == "incorrect")
                mats["fresh"][j, i] = crag_fresh_loss(rec, sc)  # in-sync audit
        return ld, mats, aux, qids

    ld_fc, m_fc, aux_fc, q_fc = build("cragfresh", "cal")
    ld_ft, m_ft, aux_ft, q_ft = build("cragfresh", "test")
    scorer = RungScorer().fit(ld_fc, aux_fc["raw_feats"])
    ld_fc.scores = scorer.score(aux_fc["raw_feats"])
    ld_ft.scores = scorer.score(aux_ft["raw_feats"])

    from contractrag.calibrate import certify_ladder
    alpha_q = 0.75
    pol = certify_ladder(ld_fc, alpha_q, DELTA / 2)
    lat_ft = np.cumsum(aux_ft["latency"], axis=0)
    idx = np.arange(ld_ft.n)
    stop_all = stop_rung(ld_ft.scores, pol.thresholds)

    # per-constraint loss streams under the deployed policy; the "unaware"
    # phase re-audits the SAME executions with ages shifted +168h, so every
    # dated web item violates the 72h contract
    l_q = ld_ft.losses[stop_all, idx]
    l_h = m_ft["hallu"][stop_all, idx]
    l_f_fresh = m_ft["fresh"][stop_all, idx]           # in-sync audit
    # unaware audit (+168h): any dated web usage now violates the 72h contract
    td_ft = TaggedTrackData("cragfresh", "test", loss_correct, kg_costs=kgc)
    td_ft.qids = q_ft
    dated = np.zeros((ld_ft.L, ld_ft.n))
    for i, qid in enumerate(td_ft.qids):
        for j in range(ld_ft.L):
            rec, _ = td_ft.records[qid][j]
            dated[j, i] = float(uses_dated_web(rec))
    l_f_unaware = dated[stop_all, idx]
    l_l = (lat_ft[stop_all, idx] > 8.0).astype(float)

    alphas = {"quality": alpha_q, "hallu": 0.30, "fresh": 0.05,
              "latency": 0.10}
    streams = {"quality": (l_q, l_q), "hallu": (l_h, l_h),
               "fresh": (l_f_fresh, l_f_unaware), "latency": (l_l, l_l)}
    print("fresh-state rates:", {k: round(float(v[0].mean()), 3)
                                 for k, v in streams.items()})
    print("unaware rates:", {k: round(float(v[1].mean()), 3)
                             for k, v in streams.items()})

    P1, P2, d_mon = 1000, 1000, DELTA / 2
    n = ld_ft.n
    alarms = {k: [] for k in alphas}
    for seed in range(20):
        rng = np.random.default_rng(700 + seed)
        i1 = rng.choice(n, P1, replace=True)
        i2 = rng.choice(n, P2, replace=True)
        mons = {k: EProcessMonitor(alphas[k], d_mon / 4) for k in alphas}
        alarm = {k: None for k in alphas}
        for t, i in enumerate(np.concatenate([i1, i2])):
            phase2 = t >= P1
            for k in alphas:
                l = streams[k][1][i] if phase2 else streams[k][0][i]
                if alarm[k] is None and mons[k].update(float(l)):
                    alarm[k] = t
        for k in alphas:
            alarms[k].append(alarm[k])
    false_alarms = 0
    for seed in range(40):
        rng = np.random.default_rng(900 + seed)
        ii = rng.choice(n, P1 + P2, replace=True)
        mons = {k: EProcessMonitor(alphas[k], d_mon / 4) for k in alphas}
        fired = False
        for i in ii:
            for k in alphas:
                if mons[k].update(float(streams[k][0][i])):
                    fired = True
                    break
            if fired:
                break
        false_alarms += int(fired)

    out = {"alphas": alphas, "delta_mon": d_mon, "P1": P1,
           "policy_q": pol.q,
           "fresh_rates": {k: float(v[0].mean()) for k, v in streams.items()},
           "unaware_rates": {k: float(v[1].mean()) for k, v in streams.items()},
           "alarm_delays": {k: [a - P1 if a is not None and a >= P1 else a
                                for a in alarms[k]] for k in alphas},
           "false_alarms_of_40": false_alarms}
    for k in alphas:
        det = [a - P1 for a in alarms[k] if a is not None and a >= P1]
        early = [a for a in alarms[k] if a is not None and a < P1]
        print(f"monitor[{k}]: detected {len(det)}/20 "
              f"median delay {np.median(det) if det else None} "
              f"early alarms {len(early)}")
    print(f"false alarms on in-sync controls: {false_alarms}/40 "
          f"(global budget {d_mon})")
    return out


def main():
    out = {}
    # CRAG: quality / hallucination / freshness / latency
    out["crag"] = run_joint(
        "crag", "correct", 0.5,
        {"hallu": lambda rec, sc: float(sc.get("label") == "incorrect"),
         "fresh": crag_fresh_loss},
        {"primary": 0.65, "hallu": 0.30, "fresh": 0.05, "latency": 0.10},
        lat_budget=8.0)
    # ASQA: quality / citation recall / citation precision / latency
    out["asqa"] = run_joint(
        "asqa", "citation", 50.0,
        {"cit_prec": lambda rec, sc: float(sc.get("citation_prec", 0.0) < 50.0),
         "quality": lambda rec, sc: float(sc.get("str_em", 0.0) < 30.0)},
        {"primary": 0.25, "cit_prec": 0.35, "quality": 0.40, "latency": 0.15},
        lat_budget=15.0)
    # multi-constraint drift monitoring with a global error budget
    out["drift_multi"] = drift_multi()

    path = os.path.join(EXP, "joint4_contracts.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
