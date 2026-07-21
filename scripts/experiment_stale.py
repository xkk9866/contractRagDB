"""Index-staleness drift experiment (RQ6b) on real re-executed matrices.

Three real execution states over the SAME queries:
  fresh   cragfresh_*: index in sync, freshness filter Delta_f=72h.
  unaware cragfresh_* re-audited: the index silently falls one week behind;
          the executor still serves the same plans/answers (it believes its
          ages), but every dated web evidence item is actually 168h older
          and violates Delta_f. Joint loss = 1[not correct OR used dated
          web evidence].
  aware   cragstale_*: executor knows the lag (ages shifted +168h), its
          freshness filter drops out-of-contract evidence; freshness holds
          by construction, correctness degrades. Joint loss = 1[not correct].

Protocol: certify ladder on fresh cal; stream phase 1 fresh / phase 2
unaware; e-process monitor on the joint loss; on alarm switch to the
policy certified on aware (stale-state) calibration data.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer, load_jsonl  # noqa: E402
from contractrag.calibrate import certify_ladder, stop_rung, EProcessMonitor  # noqa: E402
from scripts.experiment_main import kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")


class TaggedTrackData(TrackData):
    def __init__(self, tag, split, loss_fn, kg_costs=None):
        matrix = load_jsonl(os.path.join(EXP, f"{tag}_{split}_matrix.jsonl"))
        scores = load_jsonl(os.path.join(EXP, f"{tag}_{split}_scores.jsonl"))
        smap = {(s["qid"], s["rung"]): s for s in scores}
        by_q = {}
        for r in matrix:
            key = (r["qid"], r["rung"])
            by_q.setdefault(r["qid"], {})[r["rung"]] = (r, smap.get(key, {}))
        self.n_rungs = max(max(d.keys()) for d in by_q.values()) + 1
        self.qids = [q for q, d in by_q.items() if len(d) == self.n_rungs]
        self.records = {q: by_q[q] for q in self.qids}
        self.loss_fn = loss_fn
        self.kg_costs = kg_costs or {}


def loss_correct(rec, sc):
    return float(sc.get("label") != "correct")


def uses_dated_web(rec):
    meta = rec.get("evidence_meta", {})
    for src, age in zip(meta.get("srcs", []), meta.get("ages", [])):
        if src == "web" and age is not None:
            return True
    return False


def loss_unaware(rec, sc):
    # index one week behind: every dated web item is 168h older than the
    # executor believes and violates Delta_f = 72h
    return float(sc.get("label") != "correct" or uses_dated_web(rec))


def build(tag, split, kgc, loss_fn, order=None):
    td = TaggedTrackData(tag, split, loss_fn, kg_costs=kgc)
    if order is not None:
        keep = set(td.qids)
        td.qids = [q for q in order if q in keep]
        td.records = {q: td.records[q] for q in td.qids}
    ld, qids, groups, aux = td.build()
    return ld, qids, aux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.70)
    ap.add_argument("--delta", type=float, default=0.1)
    args = ap.parse_args()
    kgc = kg_costs_for("crag")

    ld_fc, q_fc, aux_fc = build("cragfresh", "cal", kgc, loss_correct)
    ld_ft, q_ft, aux_ft = build("cragfresh", "test", kgc, loss_correct)
    ld_ut, q_ut, _ = build("cragfresh", "test", kgc, loss_unaware, order=q_ft)
    ld_sc, q_sc, aux_sc = build("cragstale", "cal", kgc, loss_correct, order=q_fc)
    ld_st, q_st, aux_st = build("cragstale", "test", kgc, loss_correct, order=q_ft)
    assert q_sc == q_fc and q_st == q_ft and q_ut == q_ft, "query alignment"

    scorer = RungScorer().fit(ld_fc, aux_fc["raw_feats"])
    ld_fc.scores = scorer.score(aux_fc["raw_feats"])
    ld_ft.scores = scorer.score(aux_ft["raw_feats"])
    ld_ut.scores = ld_ft.scores  # same executions, only the audit differs
    ld_sc.scores = scorer.score(aux_sc["raw_feats"])
    ld_st.scores = scorer.score(aux_st["raw_feats"])

    print("fresh   cal per-rung risk:", np.round(ld_fc.losses.mean(axis=1), 3))
    print("unaware test per-rung risk:", np.round(ld_ut.losses.mean(axis=1), 3))
    print("aware   cal per-rung risk:", np.round(ld_sc.losses.mean(axis=1), 3))

    pol_fresh = certify_ladder(ld_fc, args.alpha, args.delta)
    pol_aware = certify_ladder(ld_sc, args.alpha, args.delta)
    print("fresh policy q:", pol_fresh.q, "certified:", pol_fresh.certified)
    print("aware policy q:", pol_aware.q, "certified:", pol_aware.certified)

    n = ld_ft.n
    rng = np.random.default_rng(0)
    P1, P2 = 2000, 2500
    idx1 = rng.choice(n, P1, replace=True)
    idx2 = rng.choice(n, P2, replace=True)
    cum_f = np.cumsum(ld_ft.costs, axis=0)
    cum_s = np.cumsum(ld_st.costs, axis=0)

    def run(mode):
        mon = EProcessMonitor(args.alpha, args.delta)
        alarm_t = None
        losses, costs = [], []
        for t, i in enumerate(np.concatenate([idx1, idx2])):
            stale = t >= P1
            adapted = (mode == "oracle" and stale) or \
                      (mode == "monitor" and alarm_t is not None)
            if adapted:
                ld, cum, thr = ld_st, cum_s, pol_aware.thresholds
            elif stale:
                ld, cum, thr = ld_ut, cum_f, pol_fresh.thresholds
            else:
                ld, cum, thr = ld_ft, cum_f, pol_fresh.thresholds
            stop = stop_rung(ld.scores[:, i:i + 1], thr)[0]
            l = float(ld.losses[stop, i])
            losses.append(l)
            costs.append(float(cum[stop, i]))
            if mode == "monitor" and alarm_t is None and mon.update(l):
                alarm_t = t
        return {"alarm_t": alarm_t,
                "viol_phase1": float(np.mean(losses[:P1])),
                "viol_phase2": float(np.mean(losses[P1:])),
                "cost_mean": float(np.mean(costs)), "losses": losses, "T1": P1}

    out = {"alpha": args.alpha, "delta": args.delta,
           "fresh_rung_risk": ld_fc.losses.mean(axis=1).tolist(),
           "unaware_rung_risk": ld_ut.losses.mean(axis=1).tolist(),
           "aware_rung_risk": ld_sc.losses.mean(axis=1).tolist(),
           "runs": {m: run(m) for m in ["static", "monitor", "oracle"]}}

    delays = []
    for seed in range(20):
        r2 = np.random.default_rng(300 + seed)
        i1 = r2.choice(n, P1, replace=True)
        i2 = r2.choice(n, P2, replace=True)
        mon = EProcessMonitor(args.alpha, args.delta)
        alarm = None
        for t, i in enumerate(np.concatenate([i1, i2])):
            ld = ld_ut if t >= P1 else ld_ft
            stop = stop_rung(ld.scores[:, i:i + 1], pol_fresh.thresholds)[0]
            if mon.update(float(ld.losses[stop, i])) and alarm is None:
                alarm = t
                break
        delays.append(None if alarm is None else alarm - P1)
    out["alarm_delays"] = delays

    false_alarms = 0
    for seed in range(20):
        r2 = np.random.default_rng(600 + seed)
        ii = r2.choice(n, P1 + P2, replace=True)
        mon = EProcessMonitor(args.alpha, args.delta)
        for i in ii:
            stop = stop_rung(ld_ft.scores[:, i:i + 1], pol_fresh.thresholds)[0]
            if mon.update(float(ld_ft.losses[stop, i])):
                false_alarms += 1
                break
    out["false_alarms_of_20"] = false_alarms

    for m, r in out["runs"].items():
        print(f"{m:>8s}: phase1 {r['viol_phase1']:.3f} phase2 {r['viol_phase2']:.3f} "
              f"cost {r['cost_mean']*1000:.2f}mCNY alarm@{r['alarm_t']}")
    print("delays:", delays)
    print("false alarms:", false_alarms, "/20")
    path = os.path.join(EXP, f"stale_crag_a{args.alpha}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("saved", path)


if __name__ == "__main__":
    main()
