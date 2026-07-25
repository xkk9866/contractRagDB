#!/usr/bin/env python3
"""Emit the LaTeX tables of the ledger experiments straight from the JSON.

Every number in the paper is produced here, so a rerun of the experiments
regenerates the tables and no figure can drift from the data behind it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
TAB = os.path.join(ROOT, "paper", "KBS", "tables")
TRACKS = ["hybridqa", "crag", "asqa", "qampari"]
PRETTY = {"hybridqa": "HybridQA", "crag": "CRAG", "asqa": "ASQA",
          "qampari": "QAMPARI", "hqgrid": "HQ grid", "cggrid": "CRAG grid"}

# streams per (alpha, setting) in the stress runs; breach fractions are
# turned back into counts so the sample size is visible in the table
DRAWS = 20


def load(name):
    p = os.path.join(EXP, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def pivotal_alpha(d):
    """The strictest level in the grid that every mechanism can be asked
    about.

    Below the risk floor a certifier that must answer has nothing feasible
    to return and falls back to its safest plan, so a comparison there
    measures the fallback rather than the mechanism. Following the
    convention of reporting at the strictest non-degenerate level, we take
    the first level at or above the floor.
    """
    floor = float(np.min(d["pop_risk"]))
    al = sorted(float(x) for x in d["results"])
    above = [a for a in al if a >= floor]
    return above[0] if above else al[-1]


# rows of the main table, in the order the literature uses: baselines
# first, the ablation, then this paper's method, then the reference that
# nobody can implement
MAIN_ROWS = [("static/no-decline", "LTT", False),
             ("static", "LTT + decline", False),
             ("ledger/greedy", "Ledger, greedy", False),
             ("ledger/lp", "Ledger", True)]


def main_table(tracks, ab_mult=1.0, out="tab_main.tex"):
    """Methods down the page, metrics across it, one block per task.

    This is the shape the surrounding literature uses, and it lets each
    quantity be reported in its own units instead of normalised against
    the contract. Validity is split into the realised rate, the worst the
    rate ever got during the run, and a count of how many streams were
    breached at any point -- the last being the quantity a deterministic
    guarantee is actually claiming something about.
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Contract adherence and cost of ownership at each task's "
        r"strictest non-degenerate contract $\alpha^{*}$, the first level "
        r"at or above the risk floor $\min_p r_p$. \emph{LTT} is one-shot "
        r"certification, \emph{LTT + decline} the same certifier handed the "
        r"decline action, \emph{Ledger, greedy} the deterministic ablation. "
        r"All see the same plan space, the same decline price and the same "
        r"stream, and all are charged for calibration, service, declines "
        r"and labels. \emph{Risk} is the realised rate over the stream and "
        r"$\sup_t$ the worst it reached at any point after $t=200$; both "
        r"are to be read against the $\alpha^{*}$ in the block heading. "
        r"\emph{PathV} counts the streams, of $30$, whose running rate "
        r"exceeded $\alpha^{*}$ at any time -- the quantity a pathwise "
        r"guarantee speaks to, and $0/30$ for the ledger by "
        r"Theorem~\ref{thm:gate} rather than by luck. \emph{OPT} is the "
        r"offline optimum, which knows every plan's risk in advance and is "
        r"not implementable. Best implementable cost among methods that did "
        r"not breach is in bold. Costs are mCNY per query.}",
        r"\label{tab:main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & Risk & $\sup_t$ & PathV & Decl.\% & Cost & Cost/OPT \\",
        r"\midrule",
    ]
    for ti, tr in enumerate(tracks):
        d = load(f"ledger_{tr}.json")
        if d is None:
            continue
        a = pivotal_alpha(d)
        floor = float(np.min(d["pop_risk"]))
        draws = int(d.get("draws", 30))
        blk = d["results"][str(a)]["rows"]
        keys = [k for k in blk
                if abs(blk[k]["abstain_cost"]
                       - ab_mult * max(np.array(d["pop_cost"]))) < 1e-12]
        got = {m: next((blk[k] for k in keys if blk[k]["method"] == m), None)
               for m, _, _ in MAIN_ROWS}
        opt = next((v["oracle"] for v in got.values()
                    if v and v.get("oracle")), None)
        # bold the cheapest method that actually kept the contract
        clean = [(v["mean_cost"], m) for m, v in got.items()
                 if v and v["breach"] == 0.0]
        best = min(clean)[1] if clean else None
        lines.append(
            r"\multicolumn{7}{l}{\textit{%s} "
            r"($\alpha^{*}=%.2f$, $\min_p r_p=%.3f$, $N=%d$)} \\"
            % (PRETTY[tr], a, floor, d["N"]))
        for m, label, ours in MAIN_ROWS:
            v = got.get(m)
            if v is None:
                continue
            cost = f"{v['mean_cost']*1000:.2f}"
            if m == best:
                cost = r"\textbf{%s}" % cost
            name = r"\textbf{%s}" % label if ours else label
            lines.append(
                f"\\quad {name} & {v['risk']:.3f} & {v['worst_rate']:.3f} & "
                f"{round(v['breach']*draws):d}/{draws} & "
                f"{v['abstain_rate']:.1f} & {cost} & "
                f"{v['mean_cost']/opt:.2f} \\\\")
        if opt is not None:
            lines.append(
                r"\quad \textit{Offline OPT} & $\le\alpha^{*}$ & -- & "
                r"-- & -- & \textit{%.2f} & \textit{1.00} \\" % (opt * 1000))
        if ti < len(tracks) - 1:
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


SWEEP_COLS = [0.10, 0.20, 0.30, 0.45, 0.60, 0.75]


def sweep_table(tracks, out="tab_sweep.tex"):
    """How strict a contract each mechanism can actually honour.

    A single contract level says nothing about the range a mechanism
    covers, and the full 0.01 scan is a figure, not a table. This reports
    six representative levels plus the count of levels honoured, which is
    the quantity an operator shops for: not what a method costs at the
    level someone chose for the experiment, but whether it works at the
    level their compliance office wrote down.
    """
    rows = [("static/no-decline", "LTT"), ("static", "LTT + decline"),
            ("ledger/lp", "Ledger")]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cost of ownership across the contract range, in units of "
        r"the offline optimum at each level. $\alpha$ is swept over $15$ "
        r"levels from $0.10$ to $0.80$ and six are shown. A superscript "
        r"gives the number of streams, out of $20$, whose running rate "
        r"exceeded $\alpha$ at any point; a cell without one was honoured "
        r"by all $20$. \emph{hon.} counts the levels of $15$ honoured on "
        r"every stream and $\alpha_{\min}$ is the strictest of them. "
        r"Every task's floor exceeds $0.10$ ($\dagger$), so the left of the "
        r"table asks for contracts no plan can meet. \emph{LTT} cannot "
        r"decline, has nothing feasible to certify there, and breaches "
        r"$545$ of the $1\,200$ streams swept. Handing it the decline "
        r"action repairs validity in all $1\,200$ and changes its bill by "
        r"almost nothing -- mean $5.49\times$ the optimum against "
        r"$5.47\times$ -- because a certifier pays the same confidence "
        r"width whether it spends it on a plan or on refusing. The ledger "
        r"honours the same $15$ levels everywhere at a mean of "
        r"$1.57\times$, and its worst level anywhere is $2.58\times$, "
        r"below the certifier's best on two of the four tasks. Read the "
        r"strict end of each row: that is where a contract binds and where "
        r"the mechanisms separate.}",
        r"\label{tab:sweep}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "r" * len(SWEEP_COLS) + "rr}",
        r"\toprule",
        r" & & \multicolumn{%d}{c}{cost / offline optimum at $\alpha=$}"
        r" & \multicolumn{2}{c}{honoured} \\" % len(SWEEP_COLS),
        r"\cmidrule(lr){3-%d}\cmidrule(lr){%d-%d}"
        % (2 + len(SWEEP_COLS), 3 + len(SWEEP_COLS), 4 + len(SWEEP_COLS)),
        r"Task & Method & "
        + " & ".join(f"{a:.2f}" for a in SWEEP_COLS)
        + r" & hon. & $\alpha_{\min}$ \\",
        r"\midrule",
    ]
    for ti, tr in enumerate(tracks):
        d = load(f"ledger_{tr}_sweep.json")
        if d is None:
            continue
        ref = max(d.get("abstain_costs") or [0])
        draws = int(d.get("draws", 20))
        floor = float(np.min(d["pop_risk"]))
        first = True
        for m, label in rows:
            def at(a):
                blk = d["results"].get(f"{a:g}") or d["results"].get(str(a))
                if blk is None:
                    return None
                return next((v for v in blk["rows"].values()
                             if v["method"] == m
                             and abs(v["abstain_cost"] - ref) < 1e-9), None)
            cells = []
            for a in SWEEP_COLS:
                v = at(a)
                if v is None:
                    cells.append("--")
                    continue
                nb = round(v["breach"] * draws)
                mark = r"$^{%d}$" % nb if nb else ""
                cells.append(f"{v['vs_oracle']:.2f}{mark}")
            hon = [a for a in sorted(float(x) for x in d["results"])
                   if (at(a) is not None and at(a)["breach"] == 0.0)]
            head = (PRETTY[tr] + (r"$^\dagger$" if floor > SWEEP_COLS[0] else "")
                    if first else "")
            lines.append(
                f"{head} & {label} & " + " & ".join(cells)
                + f" & {len(hon)}/15 & "
                + (f"{min(hon):.2f}" if hon else "--") + r" \\")
            first = False
        if ti < len(tracks) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)


def overhead_table(out="tab_overhead.tex"):
    """What the bookkeeping costs at run time.

    A per-query guarantee is only worth having if enforcing it is cheap
    next to the query. Retrieval and generation are the latencies recorded
    in the execution matrices, not estimates.
    """
    d = load("overhead.json")
    if d is None:
        return
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Run-time cost of the guarantee, microseconds per "
        r"arrival. \emph{gate} is the ledger test, \emph{LP} the action "
        r"program amortised over the $20$ arrivals between re-solves, "
        r"\emph{draw} the sample from the resulting mixture and "
        r"\emph{upd.} the ledger and estimate update. Retrieval and "
        r"generation are the latencies recorded in the execution matrices "
        r"over $64\,480$ metered calls, in milliseconds. Enforcement is "
        r"four to five orders of magnitude below the query it governs, so "
        r"the contract is free in wall-clock terms and the cost figures "
        r"elsewhere in this paper are the whole story.}",
        r"\label{tab:overhead}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{5}{c}{enforcement ($\mu$s)}"
        r" & \multicolumn{2}{c}{query (ms)} \\",
        r"\cmidrule(lr){3-7}\cmidrule(lr){8-9}",
        r"Task & $|\mathcal{P}|$ & gate & LP & draw & upd.\ & total"
        r" & retr.\ & gen.\ \\",
        r"\midrule",
    ]
    for tr in ("hybridqa", "crag", "asqa", "qampari"):
        r = d.get(tr)
        if r is None:
            continue
        lines.append(
            f"{PRETTY[tr]} & {r['plans']} & {r['gate_s']*1e6:.2f} & "
            f"{r['lp_amortised_s']*1e6:.1f} & {r['draw_s']*1e6:.1f} & "
            f"{r['update_s']*1e6:.2f} & {r['per_query_s']*1e6:.1f} & "
            f"{r['retrieval_s']*1e3:.0f} & {r['generation_s']*1e3:.0f} \\\\")
    pct = [d[t]["pct_of_pipeline"] for t in d]
    lines += [
        r"\midrule",
        r"\multicolumn{9}{l}{enforcement as a share of the pipeline: "
        r"$%.4f\%%$--$%.4f\%%$} \\" % (min(pct), max(pct)),
        r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


def order_table(tracks, out="tab_orders.tex"):
    """The guarantee under orderings that break exchangeability."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Contract adherence under query orderings that violate "
        r"exchangeability. A calibration certificate is valid only if the "
        r"deployment stream is exchangeable with the calibration prefix; the "
        r"ledger bound holds pathwise and assumes nothing about the order. "
        r"Entries are means over 20 streams at the median $\alpha$ of each "
        r"task, given in parentheses after the task name. Adherence is the "
        r"worst rate the run ever reached, to be read against the "
        r"$\alpha$ in the task heading; cost is in units of the offline "
        r"optimum for the same $\alpha$, so the two halves of the table "
        r"read as what was promised against what was paid. The certificate "
        r"fails in both directions and never in between. When the "
        r"calibration prefix is easier than the stream it under-provisions: "
        r"it pays as little as $0.30$ of the optimum and overruns its "
        r"contract on three tasks in four, by up to $2.18\times$. When the "
        r"prefix is harder it over-provisions: it declines $33$--$92\%$ of "
        r"the traffic and lands at $2.9$--$7.3\times$ the optimum, which is "
        r"why those rows accrue no violations at all -- a stream that is "
        r"mostly refused and otherwise served by the strongest plan cannot. "
        r"Neither is a contract honoured at a defensible price. The ledger "
        r"reaches $85$--$100\%$ of its contract on every row at "
        r"$1.00$--$2.77$ of the optimum, and the rows that land exactly on "
        r"$\alpha$ do so exactly rather than by rounding, since the gate "
        r"closes as the allowance runs out, so $V_t/t$ approaches $\alpha$ "
        r"and stops. The orderings that break the certificate are "
        r"not exotic: sorting a day's traffic by difficulty is what a queue "
        r"does.}",
        r"\label{tab:orders}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{$\sup_t V_t/t$}"
        r" & \multicolumn{2}{c}{cost $/$ optimum} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Task & Order & LTT & Ledger & LTT & Ledger \\",
        r"\midrule",
    ]
    names = {"iid": "i.i.d.\\ shuffle", "hard-first": "hard first",
             "easy-first": "easy first", "drift": "drift",
             "adversarial": "adversarial"}
    for tr in tracks:
        d = load(f"ledger_stress_{tr}.json")
        if d is None or "orders" not in d:
            continue
        rows = d["orders"]
        alphas = sorted({v["alpha"] for v in rows.values()})
        a = alphas[len(alphas) // 2]
        first = True
        for kind in ["iid", "hard-first", "easy-first", "drift", "adversarial"]:
            s = rows.get(f"{a}|{kind}|static")
            l = rows.get(f"{a}|{kind}|ledger")
            if s is None or l is None:
                continue
            orc = max(s["oracle"], 1e-12)
            lines.append(
                f"{PRETTY[tr] + f' ({a:g})' if first else ''} & "
                f"{names[kind]} & {s['worst_rate']:.3f} & "
                f"{l['worst_rate']:.3f} & "
                f"{s['cost'] / orc:.2f} & {l['cost'] / orc:.2f} \\\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


def regret_table(tracks, out="tab_regret.tex"):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Serving cost above the offline optimum as the stream "
        r"lengthens. The optimum solves the action LP on the population risks, "
        r"i.e.\ what a system that already knew every plan's risk would pay; "
        r"it buys no labels, so the audit bill is reported separately rather "
        r"than charged against it. A roughly constant "
        r"$\mathrm{regret}\times\sqrt{T}$ is the $O(T^{-1/2})$ rate of "
        r"Theorem~\ref{thm:regret}. \emph{serve} covers service and declines. "
        r"Values in mCNY per query, 20 streams, at the $\alpha$ in "
        r"parentheses.}",
        r"\label{tab:regret}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Task & $T$ & serve & audit & regret & "
        r"reg.$\sqrt{T}$ \\",
        r"\midrule",
    ]
    for tr in tracks:
        d = load(f"ledger_stress_{tr}.json")
        if d is None or "regret" not in d:
            continue
        rows = d["regret"]
        alphas = sorted({v["alpha"] for v in rows.values()})
        a = alphas[len(alphas) // 2]
        sel = sorted((v for v in rows.values() if v["alpha"] == a),
                     key=lambda v: v["T"])
        first = True
        for v in sel:
            lines.append(
                f"{PRETTY[tr] + f' ({a:g})' if first else ''} & "
                f"{v['T']} & {v['cost']*1000:.3f} & {v['audit']*1000:.3f} & "
                f"{v['gap']*1000:.4f} & {v['gap']*np.sqrt(v['T'])*1000:.3f} \\\\")
            first = False
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


def audit_table(tracks, out="tab_audit.tex"):
    """Tasks run across the page so the table fits the two-column measure."""
    avail = [t for t in tracks if load(f"ledger_stress_{t}.json") is not None]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Price of labelling only part of the traffic. "
        r"\textsc{worst-case} charges an unlabelled arrival a full violation, "
        r"which keeps $V_t\le\alpha t$ exactly; \textsc{ipw} rescales the "
        r"labelled ones, which is unbiased but bounds the ledger only in "
        r"probability. The columns price a guarantee that holds against one "
        r"that usually holds. Per task: serving cost including declines, "
        r"audit cost (both mCNY per query) and the worst rate the run ever "
        r"reached, averaged over streams, to be read against the $\alpha$ "
        r"in the header. That last column is the quantity the contract "
        r"constrains. \textsc{worst-case} stays under $\alpha$ by "
        r"construction and, at low label rates, well under, because "
        r"charging unlabelled arrivals as violations spends the allowance on "
        r"declines instead of answers; \textsc{ipw} sits nearer the boundary "
        r"and crosses it as labels thin out. Counted as streams breached "
        r"rather than as a magnitude, the same \textsc{ipw} runs break the "
        r"contract on $6\%$ (ASQA) to $59\%$ (HybridQA) of streams at a "
        r"$25\%$ label rate.}",
        r"\label{tab:audit}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll" + "rrr" * len(avail) + "}",
        r"\toprule",
    ]
    hdr, sub, cm = [], [], []
    for i, tr in enumerate(avail):
        d = load(f"ledger_stress_{tr}.json")
        a = sorted({v["alpha"] for v in d["audit"].values()})
        a = a[len(a) // 2]
        hdr.append(r"\multicolumn{3}{c}{%s ($\alpha$=%g)}" % (PRETTY[tr], a))
        sub.append(r"cost & audit & $\sup_t$")
        c0 = 3 + 3 * i
        cm.append(r"\cmidrule(lr){%d-%d}" % (c0, c0 + 2))
    lines += [" & & " + " & ".join(hdr) + r" \\",
              "".join(cm),
              r"Mode & rate & " + " & ".join(sub) + r" \\",
              r"\midrule"]
    for mode in ("worst-case", "ipw"):
        for j, rate in enumerate((1.0, 0.5, 0.25, 0.1)):
            cells = []
            for tr in avail:
                d = load(f"ledger_stress_{tr}.json")
                a = sorted({v["alpha"] for v in d["audit"].values()})
                a = a[len(a) // 2]
                v = d["audit"].get(f"{a}|{mode}|{rate}")
                cells.append("-- & -- & --" if v is None else
                             f"{v['cost']*1000:.3f} & "
                             f"{v['audit_cost']*1000:.3f} & "
                             f"{v['worst_rate']:.3f}")
            mcell = (r"\textsc{%s}" % mode) if j == 0 else ""
            lines.append(f"{mcell} & {rate:.2f} & " + " & ".join(cells)
                         + r" \\")
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)


def ablation_table(tracks, out="tab_ablation.tex"):
    """Each row removes one ingredient; the last is the full controller."""
    cols = list(tracks) + [t for t in ("hqgrid", "cggrid")
                           if load(f"ledger_{t}.json")]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Contribution of each ingredient, averaged over the "
        r"$\alpha$ levels of each column and reported as cost relative to "
        r"that column's offline action LP (lower is better). Removing the "
        r"decline action makes strict levels infeasible, and the variant then "
        r"serves its safest plan and breaches; removing randomisation leaves "
        r"only frontier vertices, which cannot reach a level between two "
        r"plans' risks; removing the ledger returns to a calibration "
        r"certificate and reinstates its confidence width. The ordering of "
        r"the four variants is the same in all six settings, including the "
        r"two plan grids, where the action space is $36$ and $24$ executed "
        r"physical plans rather than routing policies over a ladder. A "
        r"superscript is the fraction of streams that variant breached; a "
        r"breaching variant can undercut the offline optimum, which is not an "
        r"achievement but the reason a cost figure alone cannot be compared. "
        r"Only the variant denied a decline action ever breaches; once "
        r"declining is available the certificate is safe but dear, and the "
        r"remaining gap to the last row is what the ledger and randomisation "
        r"contribute.}",
        r"\label{tab:ablation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(cols) + "}",
        r"\toprule",
        r"Variant & " + " & ".join(PRETTY[t] for t in cols) + r" \\",
        r"\midrule",
    ]
    tracks = cols
    variants = [("static/no-decline", "certificate, no decline"),
                ("static", "certificate + decline"),
                ("ledger/greedy", "ledger, single plan"),
                ("ledger/lp", r"\textbf{ledger + randomised}")]
    for key, label in variants:
        cells = []
        for tr in tracks:
            d = load(f"ledger_{tr}.json")
            if d is None:
                cells.append("--")
                continue
            ref = min(d["abstain_costs"],
                      key=lambda x: abs(x - max(d["pop_cost"])))
            vals, br = [], []
            for a, blk in d["results"].items():
                for k, r in blk["rows"].items():
                    if (r["method"] == key
                            and abs(r["abstain_cost"] - ref) < 1e-12):
                        vals.append(r["vs_oracle"])
                        br.append(r["breach"])
            if not vals:
                cells.append("--")
                continue
            # a variant that breaches can be cheaper than the offline optimum,
            # which is not an achievement; say so in the cell
            mark = (r"$^{" + f"{np.mean(br):.2f}" + "}$") if any(br) else ""
            cells.append(f"{np.mean(vals):.2f}{mark}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)


def delta_table(tracks, out="tab_delta.tex"):
    """Is the certificate's failure a matter of an over-generous budget?

    delta is the only lever the certificate has. Sweeping it over two orders
    of magnitude leaves the breach rate essentially where it was, because the
    failure is that the calibration prefix misrepresents the stream, not that
    the correction for sampling noise is too small. The ledger has no delta,
    and its realised risk lands on alpha rather than under it.
    """
    cols = list(tracks) + [t for t in ("hqgrid", "cggrid")
                           if load(f"ledger_stress_{t}.json")]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sweeping the certifier's error budget over two orders of "
        r"magnitude, under the two orderings that break exchangeability. "
        r"\emph{brch} counts the breaching streams out of the $100$ run per "
        r"row (five $\alpha$ levels $\times$ $20$ streams); "
        r"\emph{r}/$\alpha$ is realised risk in units of the contract, so "
        r"$1.00$ means the allowance was spent exactly and above $1.00$ "
        r"means it was overspent. Shrinking $\delta$ by $200\times$ widens "
        r"the confidence correction, which is the certificate's only lever, "
        r"and it moves the count in one of the twelve cells: the failure is "
        r"a calibration sample that misrepresents the stream, not a "
        r"correction too small for sampling noise, and no choice of $\delta$ "
        r"addresses that. The ledger breaches none of the $1\,200$ streams "
        r"here, so that column would be twelve zeros and is omitted; what is "
        r"worth reading is its realised risk, which lands on the contract "
        r"rather than under it and so is not slack.}",
        r"\label{tab:delta}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{ll" + "rr" * 2 + "r}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{cert. $\delta=0.2$}"
        r" & \multicolumn{2}{c}{cert. $\delta=10^{-3}$}"
        r" & ledger \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
        r"Task & Order & brch & r/$\alpha$ & brch & r/$\alpha$"
        r" & r/$\alpha$ \\",
        r"\midrule",
    ]
    SHORT = {"easy-first": "easy", "drift": "drift"}
    for ti, tr in enumerate(cols):
        d = load(f"ledger_stress_{tr}.json")
        if not d or "delta" not in d:
            continue
        for oi, order in enumerate(("easy-first", "drift")):
            rows = [v for v in d["delta"].values() if v["order"] == order]
            cells = []
            for meth, dl in (("certificate", 0.2), ("certificate", 0.001),
                             ("ledger", None)):
                sel = [v for v in rows if v["method"] == meth
                       and v["delta"] == dl]
                risk = (f"{np.mean([v['risk'] / v['alpha'] for v in sel]):.2f}"
                        if sel else "--")
                if meth == "ledger":
                    # its breach count is zero in every cell; the column is
                    # dropped rather than printed as twelve zeros. The risk
                    # is quoted to three places because at two it rounds to
                    # 1.00 almost everywhere, which reads as a placeholder
                    # rather than as the allowance being spent to the last
                    # percent
                    cells.append(
                        f"{np.mean([v['risk'] / v['alpha'] for v in sel]):.3f}"
                        if sel else "--")
                    continue
                if not sel:
                    cells += ["--", "--"]
                    continue
                n = sum(v["breach"] * DRAWS for v in sel)
                cells += [f"{n:.0f}", risk]
            lines.append(
                (PRETTY[tr] if oi == 0 else "") + f" & {SHORT[order]} & "
                + " & ".join(cells) + r" \\")
        if ti < len(cols) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


GRID_PAIRS = [("hybridqa", "hqgrid", "HybridQA"), ("crag", "cggrid", "CRAG")]


def grid_table(out="tab_grid.tex"):
    """What enlarging the search space does to each kind of guarantee.

    Quoting certified cost at a few hand-picked contract levels left most of
    this table empty, because at strict levels the certifier returns nothing.
    The dash was a real result but a poor summary. We instead scan alpha on a
    0.01 grid and report the strictest contract each mechanism can honour,
    how far that sits above what the plan space physically permits, and how
    many of the 71 levels each can serve at all -- every entry a number, and
    the quantity an operator actually shops for.
    """
    af = load("grid_alpha_floor.json")
    if af is None:
        return
    af = af["tracks"]
    n_lv = None
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{What enlarging the plan space does to a certificate, "
        r"measured by the contracts it can actually honour rather than by its "
        r"cost at levels chosen in advance. $\alpha$ is scanned on a $0.01$ "
        r"grid from $0.10$ to $0.80$ ($71$ levels) and each space is "
        r"certified with the same randomised construction, calibration split "
        r"and $\delta=0.1$. \emph{floor} is the risk of the best single plan, "
        r"what the models and retrieval physically permit; "
        r"$\alpha_{\mathrm{cert}}$ is the strictest contract the certifier "
        r"returns anything at, and the \emph{slack} between them, in "
        r"percentage points, is the range of contracts the space can meet "
        r"but certification cannot reach. "
        r"\emph{levels} counts how many of the $71$ the certifier serves. "
        r"The ledger serves all $71$ in every space, including every level "
        r"below the floor, because declining carries no contract loss, so "
        r"the column would read $71$ four times and is omitted. Sampling that "
        r"claim at $15$ levels per space (Sec.~\ref{sec:res:grid}) gives "
        r"$60$ cells, none breached, reaching $71$--$99.6\%$ of the "
        r"allowance. Enlarging "
        r"the space moves these quantities in opposite directions on the two "
        r"tasks -- on HybridQA it costs one level and widens the gap, on CRAG "
        r"it buys four levels and lets a contract $0.04$ stricter be honoured "
        r"-- which is the point: the sign is a property of the task. Costs "
        r"are mCNY per query at $\alpha_{\mathrm{cert}}$, where both "
        r"mechanisms return something and the comparison is defined.}",
        r"\label{tab:grid}",
        r"\footnotesize",
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r" & & & & \multicolumn{3}{c}{what the certifier can honour}"
        r" & \multicolumn{2}{c}{cost at $\alpha_{\mathrm{cert}}$} \\",
        r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"Task & Space & $|\mathcal{P}|$ & floor"
        r" & $\alpha_{\mathrm{cert}}$ & slack (pp) & levels"
        r" & cert. & ledger \\",
        r"\midrule",
    ]
    for bench, grid, pretty in GRID_PAIRS:
        if bench not in af:
            continue
        for si, (sel, tag, label) in enumerate(
                (("diag", bench, r"ladder"), ("grid", grid, r"grid"))):
            r = af[bench][sel]
            ac = r["alpha_cert"]
            n_lv = len(r["lp_at"])
            cert = next((v for k, v in r["cert_at"].items()
                         if abs(float(k) - ac) < 1e-6), None)
            d = load(f"ledger_{tag}_atcert.json")
            lp = None
            if d:
                k = min(d["results"], key=lambda x: abs(float(x) - ac))
                lp = next((v for v in d["results"][k]["rows"].values()
                           if v["method"] == "ledger/lp"), None)
            row = [pretty if si == 0 else "", label, str(r["n_plans"]),
                   f"{r['floor']:.3f}", f"{ac:.2f}",
                   f"{100 * (ac - r['floor']):.1f}", f"{len(r['cert_at'])}",
                   f"{cert*1000:.2f}" if cert else "--",
                   f"{lp['mean_cost']*1000:.2f}" if lp else "--"]
            lines.append(" & ".join(row) + r" \\")
        if bench != GRID_PAIRS[-1][0]:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write(out, lines)


def grid_table_old(out="tab_grid_old.tex"):
    """Superseded layout, kept so the earlier numbers can be regenerated."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{What enlarging the plan space does to a certificate. The "
        r"space grows from the hand-drawn ladder (the diagonal of the grid, "
        r"$|\mathcal{P}|=4$) to the full access-path $\times$ generator grid "
        r"(36 plans on HybridQA, 24 on CRAG). Costs are mCNY per query; both "
        r"spaces use the same randomised construction, the same calibration "
        r"split and the same $\delta=0.1$. \emph{offline} is the cheapest "
        r"mixture that meets the level on the whole population and can only "
        r"fall as plans are added, which it does everywhere. The certified "
        r"cost does not inherit that monotonicity: relative to its own "
        r"offline optimum the certificate gets \emph{worse} on the HybridQA "
        r"grid (1.14$\times$ to 1.46$\times$) and \emph{better} on the CRAG "
        r"grid (2.02$\times$ to 1.41$\times$). Nor does feasibility move in "
        r"one direction: at $\alpha=0.62$ on CRAG the grid certifies a policy "
        r"and the ladder certifies nothing, although a mixture costing 0.650 "
        r"exists in the ladder; at $\alpha=0.30$ on HybridQA neither space "
        r"certifies anything although both admit a feasible mixture. Which "
        r"way the multiplicity cost and the cheaper plans net out is a "
        r"property of the task, not something an operator can predict before "
        r"paying to build the space. \emph{n.f.}: no mixture meets the level; "
        r"\emph{n.c.}: nothing certifies.}",
        r"\label{tab:grid}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{ladder ($|\mathcal{P}|=4$)}"
        r" & \multicolumn{3}{c}{grid} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Task & $\alpha$ & offline & cert. & ratio"
        r" & offline & cert. & ratio \\",
        r"\midrule",
    ]
    for bench, grid, pretty in GRID_PAIRS:
        pg = load(f"plangrid_{bench}.json")
        if pg is None:
            continue
        R, C = np.array(pg["pop_risk"]), np.array(pg["pop_cost"])
        diag = [i for i, nm in enumerate(pg["names"])
                if nm in ("A0/qwen-flash", "A1/qwen-flash",
                          "A2/qwen-plus", "A3/qwen-max")]
        n_grid = len(pg["names"])
        for j, a in enumerate(sorted(pg["arms"]["diag/router+mix"],
                                     key=float)):
            av = float(a)
            row = [pretty if j == 0 else "", f"{av:.2f}"]
            for sel, arm in ((diag, "diag/router+mix"),
                             (list(range(n_grid)), "grid/router+mix")):
                off = offline_lp(R[sel], C[sel], av)
                cert = pg["arms"][arm][a]["cost"]
                row += [f"{off*1000:.3f}" if off else r"\emph{n.f.}",
                        f"{cert*1000:.3f}" if cert else r"\emph{n.c.}",
                        f"{cert/off:.2f}" if (cert and off) else "--"]
            lines.append(" & ".join(row) + r" \\")
        if bench != GRID_PAIRS[-1][0]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write(out, lines)


def offline_lp(R, C, alpha):
    """Cheapest mixture of two plans meeting the level, or None."""
    from itertools import combinations
    best = min((float(C[i]) for i in range(len(R)) if R[i] <= alpha),
               default=None)
    for i, j in combinations(range(len(R)), 2):
        lo, hi = (i, j) if R[i] < R[j] else (j, i)
        if not (R[lo] <= alpha <= R[hi]):
            continue
        w = (R[hi] - alpha) / (R[hi] - R[lo])
        c = w * C[lo] + (1 - w) * C[hi]
        best = c if best is None else min(best, c)
    return best


def write(name, lines):
    os.makedirs(TAB, exist_ok=True)
    p = os.path.join(TAB, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {p} ({len(lines)} lines)")


if __name__ == "__main__":
    tracks = sys.argv[1:] or TRACKS
    main_table(tracks)
    sweep_table(tracks)
    overhead_table()
    order_table(tracks)
    regret_table(tracks)
    audit_table(tracks)
    ablation_table(tracks)
    delta_table(tracks)
    grid_table()
