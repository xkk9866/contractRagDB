#!/usr/bin/env python3
"""What the bookkeeping costs at run time.

A guarantee enforced per query is only useful if enforcing it is cheap
next to the query itself. This times the executor's per-arrival work
against the retrieval and generation latencies actually recorded in the
execution matrices, and writes the result for the paper to quote.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.ledger import LedgerExecutor, solve_action_lp  # noqa
from scripts.validate_ledger import TRACK_CFG, load_track  # noqa

EXP = os.path.join(ROOT, "experiments")


def measured_latency(track):
    """Mean retrieval and generation seconds per query, from the matrices."""
    ret, gen = [], []
    for f in glob.glob(os.path.join(EXP, f"{track}_*_matrix.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("latency_retrieval") is not None:
                    ret.append(float(r["latency_retrieval"]))
                if r.get("latency_llm") is not None:
                    gen.append(float(r["latency_llm"]))
    return (float(np.mean(ret)) if ret else float("nan"),
            float(np.mean(gen)) if gen else float("nan"),
            len(gen))


def time_it(fn, reps):
    # warm up so the first call's import/allocation cost is not counted
    for _ in range(min(reps, 50)):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", default=["hybridqa", "crag"])
    ap.add_argument("--reps", type=int, default=20000)
    args = ap.parse_args()

    out = {}
    for track in args.tracks:
        loss_field, tau, alphas = TRACK_CFG[track]
        mats, names = load_track(track, tau)
        (Ltr, Ctr), (Lca, Cca), (Lte, Cte) = mats
        L = np.concatenate([Lca, Lte], axis=1)
        C = np.concatenate([Cca, Cte], axis=1)
        alpha = alphas[len(alphas) // 2]
        ab = float(C.mean(axis=1).max())
        ex = LedgerExecutor(L, C, alpha, ab,
                            hist_risk=Ltr.mean(axis=1),
                            hist_cost=Ctr.mean(axis=1))
        rng = np.random.default_rng(0)
        P = ex.P

        # the per-arrival pieces, timed separately
        risk = ex._risk_bound()
        w = ex._lp_weights()
        led = 1.0

        t_gate = time_it(lambda: ex.gate_open(led), args.reps)
        t_lp = time_it(lambda: solve_action_lp(risk, ex.cost_hat, alpha, ab),
                       max(args.reps // 20, 200))
        t_draw = time_it(lambda: int(rng.choice(P + 1, p=w)), args.reps)

        def update():
            ex.risk_n[0] += 1.0
            ex.risk_hat[0] += (0.0 - ex.risk_hat[0]) / ex.risk_n[0]
        t_upd = time_it(update, args.reps)

        # the LP is re-solved every resolve_every arrivals, so amortise it
        every = ex.resolve_every
        per_query = t_gate + t_lp / every + t_draw + t_upd
        ret_s, gen_s, n_obs = measured_latency(track)

        out[track] = {
            "plans": int(P), "alpha": alpha, "resolve_every": int(every),
            "gate_s": t_gate, "lp_s": t_lp, "lp_amortised_s": t_lp / every,
            "draw_s": t_draw, "update_s": t_upd, "per_query_s": per_query,
            "retrieval_s": ret_s, "generation_s": gen_s,
            "n_latency_obs": n_obs,
            "pct_of_generation": 100.0 * per_query / gen_s,
            "pct_of_pipeline": 100.0 * per_query / (gen_s + ret_s),
        }
        r = out[track]
        print(f"\n=== {track}: P={P} plans, alpha={alpha:g}, "
              f"LP re-solved every {every} arrivals")
        print(f"  gate check      {t_gate*1e6:9.3f} us")
        print(f"  action LP       {t_lp*1e6:9.3f} us  "
              f"({t_lp/every*1e6:.3f} us amortised)")
        print(f"  sample action   {t_draw*1e6:9.3f} us")
        print(f"  ledger update   {t_upd*1e6:9.3f} us")
        print(f"  TOTAL per query {per_query*1e6:9.3f} us")
        print(f"  retrieval       {ret_s*1e3:9.1f} ms   (measured)")
        print(f"  generation      {gen_s*1e3:9.1f} ms   (measured, "
              f"n={n_obs})")
        print(f"  overhead        {r['pct_of_generation']:9.5f}% of "
              f"generation, {r['pct_of_pipeline']:.5f}% of the pipeline")

    path = os.path.join(EXP, "overhead.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", path)


if __name__ == "__main__":
    main()
