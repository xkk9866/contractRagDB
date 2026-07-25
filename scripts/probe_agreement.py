#!/usr/bin/env python3
"""Is contract violation predictable from cross-plan answer agreement?

The control-variate probe established that violation is essentially
unpredictable from retrieval-side features: on the plans a certifier actually
deploys, the correlation between a fitted predictor and the realized loss is
0.12-0.24 and the variance ratio is 0.97-1.02. That kills every route which
needs a cheap predictor of per-query risk -- prediction-powered inference,
per-query routing, variance-reduced certification.

But those features are all measured BEFORE any answer exists. A different signal
is available for the price of a second cheap execution: run two inexpensive plans
that disagree in their implementation -- a different retrieval path, a different
generator family -- and compare their answers. Agreement is a statement about the
answer, not about the question, and it is exactly the quantity a redundant
execution buys.

This probe measures, on real matrices with no new LLM calls, how well agreement
among cheap plans predicts violation, both of the cheap plans themselves and of
the expensive ones. If the correlation is materially higher than 0.2, then a
redundancy operator earns its place in the physical algebra: it converts an
unpredictable risk into an observable one, which is what a certified optimizer
needs in order to spend its risk budget query by query instead of globally.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contractrag.textutil import best_f1  # noqa

EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments")
FAMS = ["qwen", "deepseek", "glm", "gemma"]
TAU = 0.5


def load(fam, split):
    mp = os.path.join(EXP, f"hqfam_{fam}_{split}_matrix.jsonl")
    sp = os.path.join(EXP, f"hqfam_{fam}_{split}_scores.jsonl")
    if not (os.path.exists(mp) and os.path.exists(sp)):
        return None, None
    mat, sc = {}, {}
    with open(mp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            mat[(r["qid"], r["rung"])] = r
    with open(sp, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sc[(r["qid"], r["rung"])] = r
    return mat, sc


def auc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="cal")
    args = ap.parse_args()

    data = {}
    for fam in FAMS:
        m, s = load(fam, args.split)
        if m is not None:
            data[fam] = (m, s)
    if len(data) < 2:
        raise SystemExit("need at least two families")
    fams = list(data)
    rungs = sorted({k[1] for k in next(iter(data.values()))[0]})
    qids = None
    for fam in fams:
        q = {k[0] for k in data[fam][0]}
        qids = q if qids is None else (qids & q)
    qids = sorted(qids)
    print(f"families={fams} rungs={rungs} n={len(qids)} split={args.split}")

    # per-rung cost, to see what the redundancy actually costs
    print(f"\n  {'plan':<22} {'risk':>7} {'cost mCNY':>10}")
    cost = {}
    for fam in fams:
        for r in rungs:
            mat, sc = data[fam]
            if not all((q, r) in mat and (q, r) in sc for q in qids):
                continue
            c = float(np.mean([mat[(q, r)]["cost_cny"] for q in qids]))
            v = float(np.mean([sc[(q, r)]["f1"] < TAU for q in qids]))
            cost[(fam, r)] = c
            print(f"  A{r}/{fam:<18} {v:7.3f} {c*1000:10.4f}")

    def answers(fam, r):
        return [data[fam][0][(q, r)]["answer"] for q in qids]

    def losses(fam, r):
        return np.array([float(data[fam][1][(q, r)]["f1"] < TAU) for q in qids])

    # agreement among the cheapest rung across families: pairwise answer F1
    print(f"\nagreement signals built from rung 0 answers only "
          f"(cost = sum of the cheap plans involved)")
    header = (f"  {'signal':<34} {'target':<16} {'corr':>7} {'AUC':>7} "
              f"{'var ratio':>10}")
    print(header)
    rows = []
    for k in range(2, len(fams) + 1):
        for combo in combinations(fams, k):
            ans = {f: answers(f, 0) for f in combo}
            agree = np.zeros(len(qids))
            npair = 0
            for a, b in combinations(combo, 2):
                agree += np.array([best_f1(ans[a][i], [ans[b][i]])
                                   for i in range(len(qids))])
                npair += 1
            agree /= max(npair, 1)
            sig = f"agree({'+'.join(c[:4] for c in combo)})"
            for tf in fams:
                for tr in rungs:
                    if (tf, tr) not in cost:
                        continue
                    if tr == 0 and tf in combo:
                        tag = f"A0/{tf}*"      # target is inside the signal
                    else:
                        tag = f"A{tr}/{tf}"
                    y = losses(tf, tr)
                    c = float(np.corrcoef(y, agree)[0, 1])
                    a_ = auc(y, -agree)
                    # variance of the residual after regressing y on agree
                    z = np.polyval(np.polyfit(agree, y, 1), agree)
                    vr = float(np.var(y - z, ddof=1) / max(np.var(y, ddof=1),
                                                           1e-9))
                    rows.append((sig, tag, c, a_, vr, len(combo)))
    # print the most informative pairing per target
    best = {}
    for sig, tag, c, a_, vr, nk in rows:
        if tag not in best or abs(c) > abs(best[tag][2]):
            best[tag] = (sig, tag, c, a_, vr, nk)
    for tag in sorted(best, key=lambda t: -abs(best[t][2])):
        sig, _, c, a_, vr, nk = best[tag]
        print(f"  {sig:<34} {tag:<16} {c:7.3f} {a_:7.3f} {vr:10.3f}")

    print(f"\nfull table (all signal/target pairs):")
    print(header)
    for sig, tag, c, a_, vr, nk in sorted(rows, key=lambda t: -abs(t[2]))[:40]:
        print(f"  {sig:<34} {tag:<16} {c:7.3f} {a_:7.3f} {vr:10.3f}")


if __name__ == "__main__":
    main()
