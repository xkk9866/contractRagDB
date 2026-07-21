"""Cross-family / cross-provider generalization (reviewer point 4).

We hold the retrieval ladder FIXED and swap only the generator family, so the
same certified optimizer is exercised across:
  - qwen     : Alibaba Qwen tiers            (hosted, token-metered)
  - deepseek : DeepSeek V4/V3.2 MoE tiers    (hosted, different vendor)
  - glm      : Zhipu GLM tiers               (hosted, different vendor)
  - gemma    : Google Gemma-3 1b/4b/12b/27b  (LOCAL ollama, GPU-second-metered)

Operator sharing: retrieval is generator-independent, so it is materialized
ONCE (engine.materialize_evidence) and reused by every family
(engine.generate_from_evidence). Recorded family cost is generation-only.

Usage:
  python scripts/experiment_families.py --stage evidence
  python scripts/experiment_families.py --stage generate --families qwen,deepseek,glm
  python scripts/experiment_families.py --stage generate --families gemma
  python scripts/experiment_families.py --stage score --families qwen,deepseek,glm,gemma
  python scripts/experiment_families.py --stage certify --families qwen,deepseek,glm,gemma
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contractrag.engine import materialize_evidence, generate_from_evidence  # noqa: E402
from contractrag.tracks.hybridqa import HybridQATrack  # noqa: E402
from contractrag.textutil import best_f1, em_score, is_abstain  # noqa: E402
from scripts.run_ladder import get_splits  # noqa: E402

EXP = os.path.join(ROOT, "experiments")

FAMILIES = {
    "qwen":     ["qwen-flash", "qwen-flash", "qwen-plus", "qwen-max"],
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-flash", "deepseek-v3.2",
                 "deepseek-v4-pro"],
    "glm":      ["glm-4.7", "glm-4.7", "glm-5", "glm-5.2"],
    "gemma":    ["ollama/gemma3:1b", "ollama/gemma3:4b", "ollama/gemma3:12b",
                 "ollama/gemma3:27b"],
}

N = {"train": 300, "cal": 500, "test": 600}
TAU = 0.5


def subsets():
    sp = get_splits("hybridqa")
    return {s: sp[s][:N[s]] for s in ["train", "cal", "test"]}


def ev_path(split):
    return os.path.join(EXP, "hqfam_%s_evidence.jsonl" % split)


def matrix_path(fam, split):
    return os.path.join(EXP, "hqfam_%s_%s_matrix.jsonl" % (fam, split))


def scores_path(fam, split):
    return os.path.join(EXP, "hqfam_%s_%s_scores.jsonl" % (fam, split))


def stage_evidence(args):
    track = HybridQATrack()
    subs = subsets()
    for split in ["train", "cal", "test"]:
        print("=== materialize evidence: %s (%d q)" % (split, len(subs[split])))
        materialize_evidence(track, subs[split], ev_path(split))


def stage_generate(args):
    track = HybridQATrack()
    subs = subsets()
    for fam in [x.strip() for x in args.families.split(",")]:
        mm = {j: m for j, m in enumerate(FAMILIES[fam])}
        workers = 6 if fam == "gemma" else args.workers
        for split in ["train", "cal", "test"]:
            print("=== generate %s / %s" % (fam, split))
            generate_from_evidence(track, subs[split], ev_path(split), mm,
                                   matrix_path(fam, split), max_workers=workers)


def stage_score(args):
    subs = subsets()
    gold = {s: {q["qid"]: q for q in subs[s]} for s in subs}
    for fam in [x.strip() for x in args.families.split(",")]:
        for split in ["train", "cal", "test"]:
            recs = []
            with open(matrix_path(fam, split), encoding="utf-8") as f:
                for line in f:
                    recs.append(json.loads(line))
            out = []
            for r in recs:
                g = gold[split].get(r["qid"])
                if g is None:
                    continue
                f1 = best_f1(r["answer"], [g["answer"]])
                out.append({"qid": r["qid"], "rung": r["rung"], "f1": f1,
                            "em": em_score(r["answer"], g["answer"]),
                            "abstain": float(is_abstain(r["answer"]))})
            with open(scores_path(fam, split), "w", encoding="utf-8") as f:
                for o in out:
                    f.write(json.dumps(o) + "\n")
            print("scored %s/%s: %d" % (fam, split, len(out)))


def _load_ladder(fam, split, loss_fn):
    from contractrag.policy import feature_vector
    from contractrag.calibrate import LadderData
    width = len(feature_vector({}))
    mat = {}
    with open(matrix_path(fam, split), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mat.setdefault(r["qid"], {})[r["rung"]] = [r, {}]
    with open(scores_path(fam, split), encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["qid"] in mat and s["rung"] in mat[s["qid"]]:
                mat[s["qid"]][s["rung"]][1] = s
    L = max(max(d) for d in mat.values()) + 1
    qids = [q for q, d in mat.items() if len(d) == L]
    n = len(qids)
    losses = np.zeros((L, n)); costs = np.zeros((L, n)); lats = np.zeros((L, n))
    raw = np.zeros((L, n, width))
    for i, qid in enumerate(qids):
        for j in range(L):
            rec, sc = mat[qid][j]
            losses[j, i] = loss_fn(rec, sc)
            costs[j, i] = rec["cost_cny"]
            lats[j, i] = rec["latency_retrieval"] + rec["latency_llm"]
            raw[j, i] = feature_vector(rec["features"])
    ld = LadderData(losses=losses, scores=np.zeros((L, n)), costs=costs)
    return ld, lats, raw, qids


# contract level per family: hosted families at the main HybridQA level; the
# local Gemma-3 family's plan space is infeasible at 0.35 (strongest rung risk
# 0.395), so its contract is set one notch looser -- and the 0.35
# infeasibility is itself reported (honest non-certifiability).
ALPHA_FAM = {"gemma": 0.45}


def _family_data(fam, loss_fn):
    from contractrag.policy import RungScorer
    ld_tr, lat_tr, raw_tr, _ = _load_ladder(fam, "train", loss_fn)
    ld_cal, lat_cal, raw_cal, _ = _load_ladder(fam, "cal", loss_fn)
    ld_te, lat_te, raw_te, _ = _load_ladder(fam, "test", loss_fn)
    scorer = RungScorer().fit(ld_tr, raw_tr)
    ld_tr.scores = scorer.score(raw_tr)
    ld_cal.scores = scorer.score(raw_cal)
    ld_te.scores = scorer.score(raw_te)
    return ld_tr, lat_tr, ld_cal, lat_cal, ld_te, lat_te


def _loss_fn(rec, sc):
    return float(sc.get("f1", 0.0) < TAU)


def stage_certify(args):
    from contractrag.policy import eval_fixed, oracle_policy
    from contractrag.optimizer import (build_candidates, build_candidates_grid,
                                       robust_order, optimize, apply_candidate)
    from contractrag.baselines import utility_route, tune_utility_lambda, eval_routed

    delta = args.delta
    table = {"track": "hybridqa", "delta": delta, "tau": TAU,
             "N": N, "grid_per_rung": args.grid, "families": {}}
    for fam in [x.strip() for x in args.families.split(",")]:
        alpha = ALPHA_FAM.get(fam, args.alpha)
        ld_tr, lat_tr, ld_cal, lat_cal, ld_te, lat_te = _family_data(fam, _loss_fn)

        per_rung_risk = [float(ld_te.losses[j].mean()) for j in range(ld_te.L)]
        per_rung_cost = [float(ld_te.costs[j].mean()) for j in range(ld_te.L)]
        strongest = eval_fixed(ld_te, lat_te, ld_te.L - 1)
        orac = oracle_policy(ld_te, lat_te)

        # headline: the default 69-candidate policy space (as in RQ1/RQ2);
        # the Cartesian grid with a robust cross-fit order is reported as the
        # plan-space-scale diagnostic
        cands = build_candidates(ld_tr)
        opt = optimize(cands, ld_cal, alpha, delta)
        cands_all = build_candidates_grid(ld_tr, grid_per_rung=args.grid)
        optg = optimize(cands_all, ld_cal, alpha, delta,
                        order=robust_order(cands_all, ld_tr))
        loss_t, cost_t, lat_t, _ = apply_candidate(opt.candidate, ld_te, lat_te)
        lg, cg, _, _ = apply_candidate(optg.candidate, ld_te, lat_te)

        lam = tune_utility_lambda(ld_tr, lat_tr, ld_tr.scores, alpha)
        ru = utility_route(ld_te.scores, ld_te.costs.mean(axis=1), lam)
        util = eval_routed(ld_te, lat_te, ru)

        rec = {
            "alpha": alpha,
            "per_rung_risk": per_rung_risk,
            "per_rung_cost_mCNY": [1000 * c for c in per_rung_cost],
            "strongest_risk": strongest["risk"],
            "strongest_cost_mCNY": 1000 * strongest["cost_mean"],
            "oracle_risk": orac["risk"], "oracle_cost_mCNY": 1000 * orac["cost_mean"],
            "contractrag": {
                "certified": opt.certified, "selected": opt.candidate.describe(),
                "test_risk": float(loss_t.mean()),
                "test_cost_mCNY": 1000 * float(cost_t.mean()),
                "n_certified": opt.n_certified, "n_tested": opt.n_tested,
                "n_candidates": len(cands),
                "lat_p95": float(np.percentile(lat_t, 95)),
            },
            "contractrag_grid": {
                "certified": optg.certified, "selected": optg.candidate.describe(),
                "test_risk": float(lg.mean()),
                "test_cost_mCNY": 1000 * float(cg.mean()),
                "n_candidates": len(cands_all),
            },
            "utility_router": {"lambda": lam, "test_risk": util["risk"],
                               "test_cost_mCNY": 1000 * util["cost_mean"]},
            "cost_saving_vs_strongest": (strongest["cost_mean"] /
                                         max(1e-12, float(cost_t.mean()))),
        }
        if fam in ALPHA_FAM:  # also probe feasibility at the hosted level
            probe = optimize(cands, ld_cal, args.alpha, delta)
            rec["certified_at_%.2f" % args.alpha] = bool(probe.certified)
        table["families"][fam] = rec
        c = rec["contractrag"]
        g = rec["contractrag_grid"]
        print("\n### %s (alpha=%.2f): per-rung risk %s"
              % (fam, alpha, [round(x, 3) for x in per_rung_risk]))
        print("  strongest: risk=%.3f cost=%.2fmCNY"
              % (rec["strongest_risk"], rec["strongest_cost_mCNY"]))
        print("  ContractRAG[%dc]: certified=%s risk=%.3f cost=%.3fmCNY "
              "saving=%.1fx (sel %s)"
              % (c["n_candidates"], c["certified"], c["test_risk"],
                 c["test_cost_mCNY"], rec["cost_saving_vs_strongest"],
                 c["selected"]))
        print("  ContractRAG[grid %dc]: certified=%s risk=%.3f cost=%.3fmCNY (sel %s)"
              % (g["n_candidates"], g["certified"], g["test_risk"],
                 g["test_cost_mCNY"], g["selected"]))
        print("  utility_router: risk=%.3f cost=%.3fmCNY"
              % (rec["utility_router"]["test_risk"], rec["utility_router"]["test_cost_mCNY"]))
        print("  oracle: risk=%.3f cost=%.2fmCNY" % (rec["oracle_risk"], rec["oracle_cost_mCNY"]))

    out = os.path.join(EXP, "families_hybridqa.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=1)
    print("\nsaved", out)


def stage_safety(args):
    """RQ1-style repeated-draw safety per family: pool cal+test, resample the
    calibration/test partition `draws` times, re-certify each draw over the
    grid plan space, count draws whose realized test risk exceeds alpha.
    Pure numpy over the materialized matrices (no LLM calls)."""
    from contractrag.optimizer import build_candidates, apply_candidate
    from contractrag.calibrate import hb_p_value

    delta, draws = args.delta, args.draws
    out = {"delta": delta, "draws": draws, "tau": TAU, "families": {}}
    for fam in [x.strip() for x in args.families.split(",")]:
        alpha = ALPHA_FAM.get(fam, args.alpha)
        ld_tr, _, ld_cal, _, ld_te, _ = _family_data(fam, _loss_fn)
        cands = build_candidates(ld_tr)
        order = sorted(range(len(cands)), key=lambda i: (cands[i].train_risk,
                                                         cands[i].train_cost))

        # pooled per-candidate loss/cost matrices (evaluate each candidate once)
        pool_losses = np.concatenate([ld_cal.losses, ld_te.losses], axis=1)
        pool_scores = np.concatenate([ld_cal.scores, ld_te.scores], axis=1)
        pool_costs = np.concatenate([ld_cal.costs, ld_te.costs], axis=1)
        from contractrag.calibrate import LadderData
        pool = LadderData(losses=pool_losses, scores=pool_scores, costs=pool_costs)
        K, n_pool = len(cands), pool.n
        Lmat = np.zeros((K, n_pool), dtype=np.float32)
        Cmat = np.zeros((K, n_pool), dtype=np.float32)
        for k, cand in enumerate(cands):
            l, cst, _, _ = apply_candidate(cand, pool)
            Lmat[k], Cmat[k] = l, cst
        n_cal = ld_cal.n

        # HB acceptance threshold: p(r) monotone in r, so 'p <= delta' is
        # 'k_viol <= k_star' for a fixed n_cal/alpha/delta
        k_star = -1
        for k in range(n_cal + 1):
            if hb_p_value(k / n_cal, n_cal, alpha) <= delta:
                k_star = k
            else:
                break

        # exact finite-population risk of each candidate (whole pooled matrix)
        pop_risk = Lmat.mean(axis=1)

        rng = np.random.default_rng(0)
        viol_cert = viol_emp = viol_cert_pop = viol_emp_pop = 0
        costs_cert, costs_emp, risks_cert = [], [], []
        for _ in range(draws):
            perm = rng.permutation(n_pool)
            ci, ti = perm[:n_cal], perm[n_cal:]
            kv = (Lmat[:, ci] > 0).sum(axis=1)          # violations on cal draw
            cal_cost = Cmat[:, ci].mean(axis=1)
            # fixed-sequence walk in the train order, stop at first failure
            sel_cert, best_cost = None, np.inf
            for oi in order:
                if kv[oi] > k_star:
                    break
                if cal_cost[oi] < best_cost:
                    sel_cert, best_cost = oi, cal_cost[oi]
            # empirical-no-cert: cheapest with empirical cal risk <= alpha
            ok = np.where(kv / n_cal <= alpha)[0]
            sel_emp = ok[np.argmin(cal_cost[ok])] if len(ok) else order[0]
            for sel, is_cert in [(sel_cert, True), (sel_emp, False)]:
                if sel is None:
                    continue
                r_te = float(Lmat[sel, ti].mean())
                c_te = float(Cmat[sel, ti].mean())
                if is_cert:
                    viol_cert += int(r_te > alpha)
                    viol_cert_pop += int(pop_risk[sel] > alpha)
                    costs_cert.append(c_te)
                    risks_cert.append(r_te)
                else:
                    viol_emp += int(r_te > alpha)
                    viol_emp_pop += int(pop_risk[sel] > alpha)
                    costs_emp.append(c_te)
        rec = {"alpha": alpha, "n_candidates": K,
               "viol_prob_certified": viol_cert / draws,
               "viol_prob_empirical": viol_emp / draws,
               "viol_prob_certified_pop": viol_cert_pop / draws,
               "viol_prob_empirical_pop": viol_emp_pop / draws,
               "mean_cost_certified_mCNY": 1000 * float(np.mean(costs_cert)),
               "mean_cost_empirical_mCNY": 1000 * float(np.mean(costs_emp)),
               "mean_test_risk_certified": float(np.mean(risks_cert))}
        out["families"][fam] = rec
        print("%-9s alpha=%.2f: viol_prob cert=%.3f emp=%.3f | cost cert=%.2f "
              "emp=%.2f mCNY (K=%d)"
              % (fam, alpha, rec["viol_prob_certified"],
                 rec["viol_prob_empirical"], rec["mean_cost_certified_mCNY"],
                 rec["mean_cost_empirical_mCNY"], K))
    path = os.path.join(EXP, "families_safety.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["evidence", "generate", "score", "certify",
                             "safety"])
    ap.add_argument("--families", default="qwen,deepseek,glm,gemma")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    {"evidence": stage_evidence, "generate": stage_generate,
     "score": stage_score, "certify": stage_certify,
     "safety": stage_safety}[args.stage](args)


if __name__ == "__main__":
    main()
