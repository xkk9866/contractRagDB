"""Online audit accounting + subsampled-audit monitoring (reviewer fix:
auditing cost was not accounted and full-traffic auditing was implicit).

Part 1 (audit cost, measured): the CRAG contract auditor is the qwen-max
three-way judge; every judge call is cached with real token counts, so its
per-call cost is measured, not estimated. Abstentions and exact matches are
resolved by a zero-cost fast path (contractrag/judge.py), so the effective
per-AUDITED-query cost is (LLM-judged fraction) x (mean judge call cost).

Part 2 (TCO): amortized total cost of ownership per deployed query =
deployment LLM spend + audit-rate x audit cost + calibration amortization
(n_cal audited queries per recertification cycle). Reports how the headline
savings change when auditing is billed.

Part 3 (validity under audit subsampling): if each query is audited
independently with probability r, the audited subsequence is itself i.i.d.
from the deployment distribution, so the e-detector's guarantee is unchanged
and only the clock dilates: delays in wall-clock queries scale as 1/r.
Measured on the CRAG workload-drift stream at r in {1, 0.25, 0.1}.

Pure numpy + one SQLite scan; no LLM calls.
"""
import json
import os
import sqlite3
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.llm import CACHE_PATH, PRICES  # noqa: E402
from contractrag.policy import TrackData, RungScorer  # noqa: E402
from contractrag.calibrate import (certify_ladder, stop_rung,
                                   EProcessMonitor)  # noqa: E402
from contractrag.textutil import is_abstain, normalize_answer  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402
from scripts.experiment_main import make_loss_fn, kg_costs_for  # noqa: E402

EXP = os.path.join(ROOT, "experiments")
LABELS = ("correct", "missing", "incorrect")


def judge_call_cost():
    """Measured mean cost of one qwen-max judge call from the LLM cache."""
    con = sqlite3.connect(CACHE_PATH)
    rows = con.execute(
        "SELECT prompt_tokens, completion_tokens, response FROM cache "
        "WHERE model='qwen-max' AND completion_tokens <= 8").fetchall()
    pin, pout = PRICES["qwen-max"]
    costs = [(pt * pin + ct * pout) / 1e6 for pt, ct, resp in rows
             if (resp or "").strip().lower().split()
             and (resp or "").strip().lower().split()[0].strip(".") in LABELS]
    con.close()
    return float(np.mean(costs)), len(costs)


def judged_fraction():
    """Fraction of CRAG answers that need an LLM judge call (the rest hit
    the exact-match / abstention fast path)."""
    splits = get_splits("crag")
    gold = {q["qid"]: q for s in splits.values() for q in s}
    n, judged = 0, 0
    for split in ["cal", "test"]:
        with open(os.path.join(EXP, f"crag_{split}_matrix.jsonl"),
                  encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                pred = (r["answer"] or "").strip()
                n += 1
                if is_abstain(pred):
                    continue
                g = gold.get(r["qid"])
                if g and normalize_answer(pred) == normalize_answer(
                        str(g.get("answer", ""))):
                    continue
                judged += 1
    return judged / n, n


def main():
    out = {}

    # ---- Part 1: measured audit cost --------------------------------------
    c_call, n_calls = judge_call_cost()
    frac_judged, n_answers = judged_fraction()
    c_audit = frac_judged * c_call
    out["audit_cost"] = {
        "judge_model": "qwen-max",
        "mean_judge_call_cost_mCNY": 1000 * c_call,
        "n_cached_judge_calls": n_calls,
        "llm_judged_fraction": frac_judged,
        "effective_audit_cost_mCNY": 1000 * c_audit,
    }
    print(f"judge call cost {1000*c_call:.2f} mCNY over {n_calls} calls; "
          f"{frac_judged:.2%} of answers need the judge -> effective "
          f"{1000*c_audit:.2f} mCNY per audited query")

    # ---- Part 2: TCO with audits billed ------------------------------------
    # deployment costs from the RQ2 table (CRAG alpha=0.65): certified 0.30,
    # strongest 6.00 mCNY; recompute here from matrices to avoid hardcoding
    loss_fn = make_loss_fn("crag", "correct", 0.5)
    kgc = kg_costs_for("crag")
    data = {}
    for s in ["train", "cal", "test"]:
        td = TrackData("crag", s, loss_fn, kg_costs=kgc)
        ld, _, _, aux = td.build()
        data[s] = {"ld": ld, "aux": aux}
    scorer = RungScorer().fit(data["train"]["ld"],
                              data["train"]["aux"]["raw_feats"])
    for s in data:
        data[s]["ld"].scores = scorer.score(data[s]["aux"]["raw_feats"])
    ld_cal, ld_te = data["cal"]["ld"], data["test"]["ld"]

    from contractrag.optimizer import build_candidates, optimize, \
        apply_candidate
    cands = build_candidates(data["train"]["ld"])
    opt = optimize(cands, ld_cal, 0.65, 0.1)
    loss_t, cost_t, _, _ = apply_candidate(opt.candidate, ld_te)
    c_dep = float(cost_t.mean())
    c_strong = float(ld_te.costs[-1].mean())  # jump semantics, as in RQ2
    n_cal = ld_cal.n
    cycle = 100000  # queries between recertifications
    tco = []
    for r in [1.0, 0.25, 0.10, 0.05, 0.0]:
        c_mon = r * c_audit
        c_calib = n_cal * c_audit / cycle
        total = c_dep + c_mon + c_calib
        tco.append({"audit_rate": r,
                    "deploy_mCNY": 1000 * c_dep,
                    "monitor_audit_mCNY": 1000 * c_mon,
                    "calib_amortized_mCNY": 1000 * c_calib,
                    "total_mCNY": 1000 * total,
                    "savings_vs_strongest": c_strong / total})
        print(f"audit rate {r:>4}: total {1000*total:.2f} mCNY/query, "
              f"savings vs strongest {c_strong/total:.1f}x")
    out["tco_crag_a0.65"] = {"deploy_mCNY": 1000 * c_dep,
                             "strongest_mCNY": 1000 * c_strong,
                             "n_cal": n_cal, "recert_cycle": cycle,
                             "rows": tco}

    # ---- Part 3: subsampled-audit monitoring -------------------------------
    splits = get_splits("crag")
    dyn = {q["qid"]: q["static_or_dynamic"] for s in splits.values()
           for q in s}
    td = TrackData("crag", "test", loss_fn, kg_costs=kgc)
    ld, qids, groups, aux = td.build(groups_map=dyn)
    ld.scores = scorer.score(aux["raw_feats"])
    pol = certify_ladder(ld_cal, 0.66, 0.1)
    static_idx = np.where(np.isin(groups, ["static", "slow-changing"]))[0]
    dyn_idx = np.where(np.isin(groups,
                               ["fast-changing", "real-time"]))[0]
    P1, P2 = 1500, 6000

    def stream_loss(i):
        stop = stop_rung(ld.scores[:, i:i + 1], pol.thresholds)[0]
        return float(ld.losses[stop, i])

    rates = [1.0, 0.25, 0.10]
    res = {}
    for r in rates:
        delays, false_alarms = [], 0
        for seed in range(20):
            rng = np.random.default_rng(1234 + seed)
            st = np.concatenate([rng.choice(static_idx, P1, replace=True),
                                 rng.choice(dyn_idx, P2, replace=True)])
            audit = rng.random(P1 + P2) < r
            mon = EProcessMonitor(0.66, 0.1)
            alarm = None
            for t, i in enumerate(st):
                if audit[t] and mon.update(stream_loss(i)):
                    alarm = t
                    break
            if alarm is not None and alarm >= P1:
                delays.append(alarm - P1)
            # drift-free control
            st0 = rng.choice(static_idx, P1 + P2, replace=True)
            audit0 = rng.random(P1 + P2) < r
            mon0 = EProcessMonitor(0.66, 0.1)
            for t, i in enumerate(st0):
                if audit0[t] and mon0.update(stream_loss(i)):
                    false_alarms += 1
                    break
        res[str(r)] = {"n_detected": len(delays),
                       "delay_median": float(np.median(delays))
                       if delays else None,
                       "delay_mean": float(np.mean(delays))
                       if delays else None,
                       "false_alarms_of_20": false_alarms}
        print(f"audit rate {r}: detected {len(delays)}/20, median delay "
              f"{res[str(r)]['delay_median']}, false alarms "
              f"{false_alarms}/20")
    out["subsampled_monitoring"] = {"alpha": 0.66, "delta": 0.1,
                                    "P1": P1, "P2": P2, "rates": res}

    path = os.path.join(EXP, "audit_accounting.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
