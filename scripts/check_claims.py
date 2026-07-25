#!/usr/bin/env python3
"""Recompute every quantitative claim the paper makes, from the raw results.

Run before every build. A claim that cannot be regenerated here does not go
into the paper.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
TRACKS = ["hybridqa", "crag", "asqa", "qampari"]


def load(n):
    p = os.path.join(EXP, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


print("=" * 72)
print("1. risk floor of the plan space (min_p r_p) -- how many exceed 0.5?")
floors = {}
for t in TRACKS + ["hqgrid"]:
    d = load(f"ledger_{t}.json")
    if d:
        floors[t] = min(d["pop_risk"])
        print(f"   {t:<10} min risk = {floors[t]:.3f}  "
              f"n_plans={d['n_plans']}  N={d['N']}")
print(f"   -> above 0.5: {sum(v > 0.5 for v in floors.values())} of "
      f"{len(floors)}")

print("=" * 72)
print("2. breaches: any stream where the worst prefix rate exceeded alpha?")
tot = {"static/no-decline": [0, 0], "static": [0, 0], "ledger/lp": [0, 0],
       "ledger/greedy": [0, 0]}
for t in TRACKS + ["hqgrid"]:
    d = load(f"ledger_{t}.json")
    if not d:
        continue
    for a, blk in d["results"].items():
        for r in blk["rows"].values():
            m = r["method"]
            key = m.split(" a=")[0]
            if key in tot:
                tot[key][0] += r["breach"] * d["draws"]
                tot[key][1] += d["draws"]
for m, (b, n) in tot.items():
    print(f"   {m:<22} {int(round(b)):>6} breaches / {n:>6} streams "
          f"= {100*b/max(n,1):.2f}%")

print("=" * 72)
print("3. cost relative to the offline action LP (mean over alpha levels,")
print("   decline priced at the most expensive plan)")
tab = {}
for t in TRACKS:
    d = load(f"ledger_{t}.json")
    if not d:
        continue
    ab_ref = max(d["pop_cost"])
    for m in ["static/no-decline", "static", "ledger/greedy", "ledger/lp"]:
        v = [r["vs_oracle"] for blk in d["results"].values()
             for r in blk["rows"].values()
             if r["method"] == m and abs(r["abstain_cost"] - ab_ref) < 1e-12]
        if v:
            tab.setdefault(m, {})[t] = float(np.mean(v))
for m, row in tab.items():
    print(f"   {m:<22} " + "  ".join(f"{t}={row[t]:.2f}x" for t in row))
if "static" in tab and "ledger/lp" in tab:
    imp = [tab["static"][t] / tab["ledger/lp"][t] for t in tab["static"]]
    print(f"   -> ledger over static+decline: "
          f"{min(imp):.2f}x to {max(imp):.2f}x")
    nd = [tab["static/no-decline"][t] / tab["static"][t]
          for t in tab["static"]]
    print(f"   -> what declining alone buys the certifier: "
          f"{min(nd):.3f}x to {max(nd):.3f}x")

print("=" * 72)
print("4. non-exchangeable orders: breach rate by method and order")
for t in TRACKS:
    d = load(f"ledger_stress_{t}.json")
    if not d or "orders" not in d:
        continue
    per = {}
    for v in d["orders"].values():
        per.setdefault((v["order"], v["method"]), []).append(v["breach"])
    line = []
    for o in ["easy-first", "drift", "adversarial", "hard-first", "iid"]:
        s = np.mean(per.get((o, "static"), [np.nan]))
        l = np.mean(per.get((o, "ledger"), [np.nan]))
        line.append(f"{o}: {s:.2f}/{l:.2f}")
    print(f"   {t:<10} (static/ledger) " + "  ".join(line))

print("=" * 72)
print("5. worst prefix rate of the ledger vs alpha -- is it ever above?")
worst = []
for t in TRACKS + ["hqgrid"]:
    for f in (f"ledger_{t}.json",):
        d = load(f)
        if not d:
            continue
        for a, blk in d["results"].items():
            for r in blk["rows"].values():
                if r["method"].startswith("ledger/lp"):
                    worst.append((r["worst_rate"] - float(a), t, a,
                                  r["worst_rate"]))
    d = load(f"ledger_stress_{t}.json")
    if d:
        for part in ("orders", "kappa", "audit"):
            for v in (d.get(part) or {}).values():
                if v.get("method", "ledger") == "ledger" and \
                        v.get("mode", "worst-case") == "worst-case":
                    worst.append((v["worst_rate"] - v["alpha"], t,
                                  v["alpha"], v["worst_rate"]))
worst.sort(reverse=True)
print(f"   {len(worst)} configurations checked; largest (rate - alpha):")
for g, t, a, w in worst[:4]:
    print(f"     {t:<10} alpha={float(a):.2f}  worst={w:.4f}  gap={g:+.4f}")

print("=" * 72)
print("6. regret decay: regret * sqrt(T), first vs last horizon")
for t in TRACKS:
    d = load(f"ledger_stress_{t}.json")
    if not d or "regret" not in d:
        continue
    by_a = {}
    for v in d["regret"].values():
        by_a.setdefault(v["alpha"], []).append(v)
    for a, vs in sorted(by_a.items()):
        vs.sort(key=lambda v: v["T"])
        f_, l_ = vs[0], vs[-1]
        print(f"   {t:<10} a={a:.2f}  T {f_['T']}->{l_['T']}  "
              f"regret {f_['gap']*1000:.4f}->{l_['gap']*1000:.4f} "
              f"({f_['gap']/max(l_['gap'],1e-12):.2f}x down, "
              f"sqrt(T) up {np.sqrt(l_['T']/f_['T']):.2f}x)  "
              f"rel {l_['gap']/l_['oracle']*100:.1f}%")
        break

print("=" * 72)
print("7. kappa sweep: cost spread and breaches")
for t in TRACKS:
    d = load(f"ledger_stress_{t}.json")
    if not d or "kappa" not in d:
        continue
    by_a = {}
    for v in d["kappa"].values():
        by_a.setdefault(v["alpha"], []).append(v)
    a = sorted(by_a)[len(by_a) // 2]
    vs = sorted(by_a[a], key=lambda v: v["kappa"])
    r = [v["vs_oracle"] for v in vs]
    b = sum(v["breach"] for v in vs)
    print(f"   {t:<10} a={a:.2f}  kappa {vs[0]['kappa']}..{vs[-1]['kappa']}: "
          f"cost {min(r):.2f}x..{max(r):.2f}x "
          f"(spread {100*(max(r)/min(r)-1):.0f}%)  breaches={b:.2f}")

print("=" * 72)
print("8. audit accounting: worst-case vs ipw at low label rates")
for t in TRACKS:
    d = load(f"ledger_stress_{t}.json")
    if not d or "audit" not in d:
        continue
    for rate in (0.25, 0.1):
        wc = [v for v in d["audit"].values()
              if v["mode"] == "worst-case" and abs(v["rate"] - rate) < 1e-9]
        ip = [v for v in d["audit"].values()
              if v["mode"] == "ipw" and abs(v["rate"] - rate) < 1e-9]
        if wc and ip:
            print(f"   {t:<10} rate={rate:.2f}  worst-case breach="
                  f"{np.mean([v['breach'] for v in wc]):.2f} "
                  f"decline={np.mean([v['abstain'] for v in wc]):.0f}%  |  "
                  f"ipw breach={np.mean([v['breach'] for v in ip]):.2f} "
                  f"decline={np.mean([v['abstain'] for v in ip]):.0f}%")

print("=" * 72)
print("8c. does a stricter error budget rescue the certificate?")
print("    breach rate at delta=0.2 vs delta=0.001 (a 200x tightening),")
print("    under the orderings that break exchangeability")
n_moved, n_cells = 0, 0
for t in TRACKS + ["hqgrid", "cggrid"]:
    d = load(f"ledger_stress_{t}.json")
    if not d or "delta" not in d:
        continue
    for order in ("easy-first", "drift"):
        rows = [v for v in d["delta"].values() if v["order"] == order]
        def pick(meth, dl):
            s = [v for v in rows if v["method"] == meth and v["delta"] == dl]
            return (np.mean([v["breach"] for v in s]),
                    np.mean([v["risk"] / v["alpha"] for v in s])) if s \
                else (float("nan"),) * 2
        hi, hr = pick("certificate", 0.2)
        lo, lr = pick("certificate", 0.001)
        lb, lrk = pick("ledger", None)
        n_cells += 1
        moved = abs(hi - lo) > 0.005
        n_moved += moved
        print(f"   {t:9s} {order:11s} cert breach {hi:.2f} -> {lo:.2f}"
              f"{'  MOVED' if moved else '':8s}  risk/alpha {hr:.2f} -> "
              f"{lr:.2f}   ledger breach {lb:.2f} risk/alpha {lrk:.2f}")
print(f"   cells whose breach rate responded to delta: {n_moved}/{n_cells}")

print("=" * 72)
print("8b. is the zero breach rate bought with slack?")
print("    worst prefix rate and mean risk as fractions of alpha; 1.00 means")
print("    the allowance is spent exactly")
for t in TRACKS + ["hqgrid", "cggrid"]:
    d = load(f"ledger_{t}.json")
    if not d:
        continue
    ref = min(d["abstain_costs"], key=lambda x: abs(x - max(d["pop_cost"])))
    w, m = [], []
    for a, blk in d["results"].items():
        for r in blk["rows"].values():
            if (r["method"] == "ledger/lp"
                    and abs(r["abstain_cost"] - ref) < 1e-12):
                w.append(r["worst_rate"] / float(a))
                m.append(r["risk"] / float(a))
    print(f"   {t:9s} worst/alpha {min(w):.2f}-{max(w):.2f}   "
          f"risk/alpha {min(m):.2f}-{max(m):.2f}")

print("=" * 72)
print("9. plan grid vs ladder under static certification")
print("   same candidate construction on both spaces, so only the plan space")
print("   differs; 'ratio' is certified cost over that space's own offline")
print("   optimum, which is what RQ6 claims moves in a task-dependent way")
DIAG = ("A0/qwen-flash", "A1/qwen-flash", "A2/qwen-plus", "A3/qwen-max")


def two_point_lp(R, C, alpha):
    from itertools import combinations
    best = min((float(C[i]) for i in range(len(R)) if R[i] <= alpha),
               default=None)
    for i, j in combinations(range(len(R)), 2):
        lo, hi = (i, j) if R[i] < R[j] else (j, i)
        if R[lo] <= alpha <= R[hi]:
            w = (R[hi] - alpha) / (R[hi] - R[lo])
            c = w * C[lo] + (1 - w) * C[hi]
            best = c if best is None else min(best, c)
    return best


for bench, grid in (("hybridqa", "hqgrid"), ("crag", "cggrid")):
    d = load(f"plangrid_{bench}.json")
    if not d:
        continue
    R, C = np.array(d["pop_risk"]), np.array(d["pop_cost"])
    diag = [i for i, nm in enumerate(d["names"]) if nm in DIAG]
    print(f"   {bench}: grid plans={len(d['names'])} "
          f"frontier={len(d['frontier'])} risk floor "
          f"ladder={R[diag].min():.3f} grid={R.min():.3f}")
    for a in d["alphas"]:
        cells = []
        for sel, arm in ((diag, "diag/router+mix"),
                         (list(range(len(R))), "grid/router+mix")):
            off = two_point_lp(R[sel], C[sel], a)
            cert = d["arms"].get(arm, {}).get(str(a), {}).get("cost")
            cells.append((off, cert))
        (ol, cl), (og, cg) = cells
        f = (lambda x: f"{x*1000:8.3f}" if x else "    none")
        rat = (lambda c, o: f"{c/o:5.2f}x" if (c and o) else "    --")
        print(f"     alpha={a:.2f}  ladder off={f(ol)} cert={f(cl)} "
              f"{rat(cl, ol)}   grid off={f(og)} cert={f(cg)} {rat(cg, og)}"
              + (f"   cert grid/ladder={cg/cl:.3f}" if (cl and cg) else ""))
    # the grid run was priced at one fixed decline cost; charge the ladder the
    # same, or the two spaces are not being compared at the same price
    g = load(f"ledger_{grid}.json")
    price = max(g["abstain_costs"]) if g else None
    for tag in (bench, grid):
        e = load(f"ledger_{tag}.json")
        if not e:
            continue
        ref = min(e["abstain_costs"], key=lambda x: abs(x - price))
        v = [r["vs_oracle"] for blk in e["results"].values()
             for r in blk["rows"].values()
             if r["method"] == "ledger/lp"
             and abs(r["abstain_cost"] - ref) < 1e-12]
        br = [r["breach"] for blk in e["results"].values()
              for r in blk["rows"].values()
              if r["method"] == "ledger/lp"
              and abs(r["abstain_cost"] - ref) < 1e-12]
        print(f"     ledger on {tag:9s} decline price={ref*1000:.3f} mCNY  "
              f"{min(v):.2f}-{max(v):.2f}x offline (mean {np.mean(v):.2f}) "
              f"max breach={max(br):.2f}")

print("=" * 72)
print("10. static confidence width vs ledger margin")
for t in TRACKS:
    d = load(f"ledger_{t}.json")
    if not d:
        continue
    K = 69
    ncal, delta = d["n_cal"], d["delta"]
    w = np.sqrt(np.log(2 * K / delta) / (2 * ncal))
    # ledger margin once the stream has concentrated on ~3 plans
    n_eff = d["N"] / 3.0
    m1 = np.sqrt(0.25 / n_eff)
    print(f"   {t:<10} K={K} n_cal={ncal}: w_stat={w:.4f}   "
          f"ledger margin at kappa=1 after N={d['N']}: {m1:.4f}  "
          f"({w/m1:.1f}x tighter)")

print("=" * 72)
print("11. Table 7 (plan grid): alpha floors, levels served, cost at a_cert")
af = load("grid_alpha_floor.json")
if af:
    for bench, grid in (("hybridqa", "hqgrid"), ("crag", "cggrid")):
        for sel, tag in (("diag", bench), ("grid", grid)):
            r = af["tracks"][bench][sel]
            ac = r["alpha_cert"]
            cert = next((v for k, v in r["cert_at"].items()
                         if abs(float(k) - ac) < 1e-6), None)
            d = load(f"ledger_{tag}_atcert.json")
            lp = None
            if d:
                k = min(d["results"], key=lambda x: abs(float(x) - ac))
                lp = next((v for v in d["results"][k]["rows"].values()
                           if v["method"] == "ledger/lp"), None)
            print(f"   {bench:<9} {sel:<5} P={r['n_plans']:2d} "
                  f"floor={r['floor']:.3f} a_cert={ac:.2f} "
                  f"gap={ac - r['floor']:.3f} "
                  f"levels={len(r['cert_at'])}/{len(r['lp_at'])} "
                  f"cert={cert * 1000:.2f} "
                  f"ledger={lp['mean_cost'] * 1000:.2f}")

print("=" * 72)
print("12. does the ledger really serve every contract level? (alpha sweep)")
cells = brs = 0
lo, hi = 9.0, 0.0
for tag in ("hybridqa", "hqgrid", "crag", "cggrid"):
    d = load(f"ledger_{tag}_sweep.json")
    if not d:
        continue
    ref = max(d.get("abstain_costs") or [0])
    rows = []
    for a in sorted(float(x) for x in d["results"]):
        v = next((v for v in d["results"][str(a)]["rows"].values()
                  if v["method"] == "ledger/lp"
                  and abs(v["abstain_cost"] - ref) < 1e-9), None)
        if v is None:
            continue
        w = v["worst_rate"] / a
        lo, hi = min(lo, w), max(hi, w)
        cells += 1
        brs += int(v["breach"])
        rows.append((a, w, v["abstain_rate"], v["vs_oracle"]))
    q = {a: (w, ab, vo) for a, w, ab, vo in rows}
    dec = " ".join(f"a={a:.2f}:{q[a][1]:.1f}%/{q[a][2]:.2f}x"
                   for a in (0.10, 0.30, 0.55) if a in q)
    print(f"   {tag:<9} levels={len(rows)} {dec}")
print(f"   TOTAL cells={cells} breaches={brs} "
      f"worst/alpha in [{lo:.3f}, {hi:.3f}]")

print("=" * 72)
print("13. Table 2 (orderings): adherence and price, both directions")
for t in TRACKS:
    d = load(f"ledger_stress_{t}.json")
    if not d or "orders" not in d:
        continue
    rows = d["orders"]
    al = sorted({v["alpha"] for v in rows.values()})
    a = al[len(al) // 2]
    sw, sc, lw, lc = [], [], [], []
    for kind in ("iid", "hard-first", "easy-first", "drift", "adversarial"):
        s, l = rows.get(f"{a}|{kind}|static"), rows.get(f"{a}|{kind}|ledger")
        if s is None or l is None:
            continue
        o = max(s["oracle"], 1e-12)
        sw.append(s["worst_rate"] / a)
        sc.append(s["cost"] / o)
        lw.append(l["worst_rate"] / a)
        lc.append(l["cost"] / o)
    print(f"   {t:<9} a={a:g}  LTT w/a {min(sw):.2f}-{max(sw):.2f} "
          f"cost {min(sc):.2f}-{max(sc):.2f}x | "
          f"ledger w/a {min(lw):.2f}-{max(lw):.2f} "
          f"cost {min(lc):.2f}-{max(lc):.2f}x")
print("=" * 72)
print("14. Table 1 (main): pivotal alpha*, PathV counts, Cost/OPT")
lo_r, hi_r, lo_c, hi_c, cl, ch = 9, 0, 9, 0, 9, 0
for t in TRACKS:
    d = load(f"ledger_{t}.json")
    if not d:
        continue
    floor = float(np.min(d["pop_risk"]))
    al = sorted(float(x) for x in d["results"])
    a = next((x for x in al if x >= floor), al[-1])
    dr = int(d.get("draws", 30))
    ref = max(d["pop_cost"])
    blk = d["results"][str(a)]["rows"]
    got = {}
    for m in ("static/no-decline", "static", "ledger/lp"):
        got[m] = next((v for v in blk.values() if v["method"] == m
                       and abs(v["abstain_cost"] - ref) < 1e-12), None)
    g, s = got["ledger/lp"], got["static/no-decline"]
    sd = got["static"]
    frac = g["worst_rate"] / a
    lo_r, hi_r = min(lo_r, frac), max(hi_r, frac)
    lo_c, hi_c = min(lo_c, g["vs_oracle"]), max(hi_c, g["vs_oracle"])
    cl = min(cl, sd["vs_oracle"])
    ch = max(ch, sd["vs_oracle"])
    print(f"   {t:<9} a*={a:.2f} floor={floor:.3f} | ledger sup={g['worst_rate']:.3f}"
          f" ({100*frac:.0f}% of a*) PathV={round(g['breach']*dr)}/{dr}"
          f" cost/OPT={g['vs_oracle']:.2f}"
          f" | LTT PathV={round(s['breach']*dr)}/{dr} cost/OPT={s['vs_oracle']:.2f}"
          f" | LTT+dec risk={sd['risk']:.3f} cost/OPT={sd['vs_oracle']:.2f}")
print(f"   CLAIM ledger spends {100*lo_r:.0f}-{100*hi_r:.0f}% of allowance, "
      f"costs {lo_c:.2f}-{hi_c:.2f}x OPT; LTT+dec (same action space) "
      f"{cl:.2f}-{ch:.2f}x")

print("=" * 72)
print("15. Table 3 (sweep): breached streams and mean cost over 15 levels")
tot = {}
for t in TRACKS:
    d = load(f"ledger_{t}_sweep.json")
    if not d:
        continue
    ref = max(d.get("abstain_costs") or [0])
    dr = int(d.get("draws", 20))
    for m in ("static/no-decline", "static", "ledger/lp"):
        nb, n, vs = 0, 0, []
        for a in d["results"]:
            v = next((x for x in d["results"][a]["rows"].values()
                      if x["method"] == m
                      and abs(x["abstain_cost"] - ref) < 1e-9), None)
            if v:
                nb += round(v["breach"] * dr)
                n += dr
                vs.append(v["vs_oracle"])
        e = tot.setdefault(m, [0, 0, []])
        e[0] += nb
        e[1] += n
        e[2] += vs
for m, (nb, n, vs) in tot.items():
    print(f"   {m:<20} breached {nb}/{n} streams | cost/OPT "
          f"{min(vs):.2f}-{max(vs):.2f} mean {np.mean(vs):.2f}")

print("=" * 72)
print("16. Table 4 (overhead): enforcement vs measured pipeline latency")
d = load("overhead.json")
if d:
    per = [v["per_query_s"] * 1e6 for v in d.values()]
    pct = [v["pct_of_pipeline"] for v in d.values()]
    gen = [v["generation_s"] for v in d.values()]
    calls = sum(v["n_latency_obs"] for v in d.values())
    for t, v in d.items():
        print(f"   {t:<9} {v['per_query_s']*1e6:6.1f} us enforcement | "
              f"retr {v['retrieval_s']*1e3:5.0f} ms gen {v['generation_s']*1e3:6.0f} ms"
              f" -> {v['pct_of_pipeline']:.4f}% of pipeline")
    print(f"   CLAIM {min(per):.0f}-{max(per):.0f} us, "
          f"{min(pct):.4f}-{max(pct):.4f}% of pipeline, "
          f"gen {min(gen):.1f}-{max(gen):.1f} s, {calls} metered calls")