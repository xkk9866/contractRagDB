"""System microbenchmarks for the paper's engineering section.

Measures, from the real execution matrices:
  1. Optimizer runtime: candidate generation (train) + fixed-sequence
     certification (cal) wall-clock, and scaling with candidate count.
  2. Total-cost-of-ownership per query: LLM token spend + local GPU compute
     (recorded wall-clock of the retrieval/rerank/NLI stage, priced at an
     RTX-4090 cloud rental rate) + KG-API surcharge (CRAG rung 3).

Usage: python scripts/bench_system.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.policy import TrackData, RungScorer
from contractrag.optimizer import build_candidates, build_candidates_grid, optimize
from scripts.experiment_main import make_loss_fn, kg_costs_for

EXP = os.path.join(ROOT, "experiments")

# RTX 4090 on-demand cloud rental, CNY per GPU-second (~2.0 CNY/hour).
GPU_RATE_CNY_S = 2.0 / 3600.0

TRACKS = [
    ("hybridqa", "quality", 0.5, 0.35),
    ("crag", "correct", 0.5, 0.65),
    ("asqa", "citation", 50.0, 0.25),
]


def load_track(track, contract, tau):
    loss_fn = make_loss_fn(track, contract, tau)
    kgc = kg_costs_for(track)
    data = {}
    for s in ["train", "cal"]:
        td = TrackData(track, s, loss_fn, kg_costs=kgc)
        ld, qids, groups, aux = td.build()
        data[s] = (ld, aux)
    scorer = RungScorer().fit(data["train"][0], data["train"][1]["raw_feats"])
    for s in data:
        data[s][0].scores = scorer.score(data[s][1]["raw_feats"])
    return data


def bench_optimizer(out):
    for track, contract, tau, alpha in TRACKS:
        data = load_track(track, contract, tau)
        ld_tr, ld_cal = data["train"][0], data["cal"][0]
        t0 = time.perf_counter()
        cands = build_candidates(ld_tr)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        pol = optimize(cands, ld_cal, alpha, 0.1)
        t_cert = time.perf_counter() - t0
        # scaling: denser parameter grids
        scaling = []
        for num_thr, num_lam in [(40, 25), (200, 100), (1000, 500)]:
            t0 = time.perf_counter()
            cs = build_candidates(ld_tr, num_thr=num_thr, num_lam=num_lam)
            tb = time.perf_counter() - t0
            t0 = time.perf_counter()
            optimize(cs, ld_cal, alpha, 0.1)
            tc = time.perf_counter() - t0
            scaling.append({"n_candidates": len(cs), "build_s": tb,
                            "certify_s": tc})
        # large Cartesian plan space (plan-space scalability, up to ~1e5)
        plan_space = []
        for g in [8, 12, 20, 32, 46]:
            t0 = time.perf_counter()
            cs = build_candidates_grid(ld_tr, grid_per_rung=g)
            tb = time.perf_counter() - t0
            t0 = time.perf_counter()
            pl = optimize(cs, ld_cal, alpha, 0.1)
            tc = time.perf_counter() - t0
            plan_space.append({"grid_per_rung": g, "n_candidates": len(cs),
                               "enumerate_s": tb, "certify_s": tc,
                               "n_tested": pl.n_tested})
        out[track] = {
            "n_train": ld_tr.n, "n_cal": ld_cal.n,
            "n_candidates": len(cands),
            "build_s": t_build, "certify_s": t_cert,
            "n_tested": pol.n_tested, "n_certified": pol.n_certified,
            "scaling": scaling,
            "plan_space": plan_space,
        }
        print(f"{track}: {len(cands)} candidates, build {t_build:.2f}s, "
              f"certify {t_cert:.3f}s ({pol.n_tested} tested); scaling "
              + ", ".join(f"{s['n_candidates']}c/{s['certify_s']:.2f}s"
                          for s in scaling))
        print(f"  plan-space: "
              + ", ".join(f"{s['n_candidates']}c/{s['certify_s']:.2f}s"
                          for s in plan_space))


def bench_cost(out):
    for track, contract, tau, alpha in TRACKS:
        loss_fn = make_loss_fn(track, contract, tau)
        kgc = kg_costs_for(track)
        td = TrackData(track, "test", loss_fn, kg_costs=kgc)
        L = td.n_rungs
        llm, gpu, kg, lat_llm = ([[] for _ in range(L)] for _ in range(4))
        for qid in td.qids:
            for j in range(L):
                rec, sc = td.records[qid][j]
                llm[j].append(rec["cost_cny"])
                gpu[j].append(rec["latency_retrieval"] * GPU_RATE_CNY_S)
                lat_llm[j].append(rec["latency_llm"])
                k = 0.0
                if j == L - 1 and qid in (kgc or {}):
                    k = kgc[qid]["cost"]
                kg[j].append(k)
        rows = []
        for j in range(L):
            rows.append({
                "rung": j,
                "llm_mCNY": 1000 * float(np.mean(llm[j])),
                "gpu_mCNY": 1000 * float(np.mean(gpu[j])),
                "kg_mCNY": 1000 * float(np.mean(kg[j])),
                "retr_s": float(np.mean(gpu[j])) / GPU_RATE_CNY_S,
                "llm_s": float(np.mean(lat_llm[j])),
            })
            r = rows[-1]
            print(f"{track} rung{j}: llm={r['llm_mCNY']:.3f} "
                  f"gpu={r['gpu_mCNY']:.3f} kg={r['kg_mCNY']:.3f} mCNY "
                  f"(retr {r['retr_s']:.2f}s, llm {r['llm_s']:.2f}s)")
        out.setdefault("cost", {})[track] = rows


def bench_sharing(out):
    """Operator sharing / materialization: retrieval is generator-independent,
    so evaluating F generator families over the same query set pays retrieval
    ONCE if materialized, vs F times if each family re-runs its pipeline.
    We report the retrieval-time fraction and the resulting speedup for F=4."""
    F = 4
    rows = {}
    for track, contract, tau, _alpha in TRACKS:
        loss_fn = make_loss_fn(track, contract, tau)
        td = TrackData(track, "test", loss_fn, kg_costs=kg_costs_for(track))
        ret, gen = 0.0, 0.0
        for qid in td.qids:
            for j in range(td.n_rungs):
                rec, _ = td.records[qid][j]
                ret += rec["latency_retrieval"]
                gen += rec["latency_llm"]
        shared = ret + F * gen        # retrieval once, generation per family
        unshared = F * (ret + gen)    # everything per family
        rows[track] = {"retrieval_s": ret, "generation_s": gen,
                       "retrieval_frac": ret / max(1e-9, ret + gen),
                       "families": F,
                       "wall_shared_s": shared, "wall_unshared_s": unshared,
                       "speedup": unshared / max(1e-9, shared)}
        print(f"{track} sharing (F={F}): retrieval {100*rows[track]['retrieval_frac']:.0f}% "
              f"of a run, speedup {rows[track]['speedup']:.2f}x")
    out["operator_sharing"] = rows


def main():
    res = {"gpu_rate_cny_per_s": GPU_RATE_CNY_S, "optimizer": {}}
    bench_optimizer(res["optimizer"])
    bench_cost(res)
    bench_sharing(res)
    path = os.path.join(EXP, "bench_system.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("saved", path)


if __name__ == "__main__":
    main()
