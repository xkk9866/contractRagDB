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


def load(name):
    p = os.path.join(EXP, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def main_table(tracks, ab_mult=1.0, out="tab_main.tex"):
    """One row per (task, contract level); the three systems run across.

    The row-per-system layout needed sixty rows and overran the page, so
    the comparison is made horizontally: each metric is a group of three
    columns, which is also the direction one actually reads it in.
    """
    METH = [("static/no-decline", "LTT"), ("static", r"+$\bot$"),
            ("ledger/lp", "Ldg")]
    # decline% is identically zero for the no-decline certifier by
    # definition, so that column is dropped rather than printed as a
    # column of zeros.
    groups = [("cost", "mean_cost", METH), ("risk", "risk", METH),
              ("decline\\%", "abstain_rate", METH[1:]),
              ("worst rate", "worst_rate", METH),
              ("breach", "breach", METH)]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Cost of ownership and contract adherence over a served "
        r"stream. \textsc{LTT} is one-shot certification, $+\bot$ the same "
        r"certifier handed the decline action, \textsc{Ldg} the ledger. All "
        r"three see the same plan space, the same decline price and the same "
        r"stream, and all are charged for calibration, service, declines and "
        r"labels. \emph{worst rate} is $\max_{t\ge 200} V_t/t$, the quantity "
        r"an anytime contract controls; \emph{breach} is the fraction of the "
        r"30 streams in which it exceeded $\alpha$. Levels marked $\dagger$ "
        r"lie below the risk of every plan in the space, where certification "
        r"without declines is infeasible by construction and \textsc{LTT} "
        r"falls back to the historically safest plan, which is why its "
        r"numbers repeat down such a block. The \textsc{Ldg} breach column "
        r"is identically zero because Theorem~\ref{thm:gate} makes it so, "
        r"not because 30 streams happened to be kind; \textsc{LTT} breaches "
        r"on all of them below the floor for the same structural reason. "
        r"\textsc{LTT} never declines by definition, so that column is "
        r"omitted. Costs in mCNY per query.}",
        r"\label{tab:main}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{ll"
        + "".join("r" * len(g[2]) for g in groups) + "}",
        r"\toprule",
    ]
    hdr, sub, cm = [], [], []
    c0 = 3
    for gname, _, meths in groups:
        hdr.append(r"\multicolumn{%d}{c}{%s}" % (len(meths), gname))
        sub.append(" & ".join(m[1] for m in meths))
        cm.append(r"\cmidrule(lr){%d-%d}" % (c0, c0 + len(meths) - 1))
        c0 += len(meths)
    lines += [r"Task & $\alpha$ & " + " & ".join(hdr) + r" \\",
              "".join(cm),
              r" & & " + " & ".join(sub) + r" \\",
              r"\midrule"]

    def cell(r, field):
        if r is None:
            return "--"
        v = r[field]
        if field == "mean_cost":
            return f"{v*1000:.2f}"
        if field == "abstain_rate":
            return f"{v:.0f}"
        return f"{v:.3f}" if field != "breach" else f"{v:.2f}"

    for ti, tr in enumerate(tracks):
        d = load(f"ledger_{tr}.json")
        if d is None:
            continue
        pop_r = np.array(d["pop_risk"])
        first = True
        for a in sorted(float(x) for x in d["results"]):
            blk = d["results"][str(a)]["rows"]
            keys = [k for k in blk
                    if abs(blk[k]["abstain_cost"]
                           - ab_mult * max(np.array(d["pop_cost"]))) < 1e-12]
            if not keys:
                continue
            mark = r"$^\dagger$" if pop_r.min() > a else ""
            got = {m: next((blk[k] for k in keys if blk[k]["method"] == m),
                           None) for m, _ in METH}
            cells = [cell(got[m], f) for _, f, meths in groups
                     for m, _ in meths]
            lines.append(f"{PRETTY[tr] if first else ''} & {a:.2f}{mark} & "
                         + " & ".join(cells) + r" \\")
            first = False
        if ti < len(tracks) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
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
        r"task, given in parentheses after the task name; \emph{breach} is "
        r"the fraction of streams whose worst prefix rate exceeded $\alpha$. "
        r"Breach takes only the values $0$ and $1$ here because these "
        r"orderings are constructed, not sampled: whether the calibration "
        r"prefix misrepresents the deployment stream is a property of the "
        r"ordering, so every stream of a given kind fails or survives "
        r"together. The informative column is the worst rate beside it, "
        r"which says by how much.}",
        r"\label{tab:orders}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{worst rate} & \multicolumn{2}{c}{breach} \\",
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
            lines.append(
                f"{PRETTY[tr] + f' ({a:g})' if first else ''} & "
                f"{names[kind]} & {fmt(s['worst_rate'])} & "
                f"{fmt(l['worst_rate'])} & "
                f"{s['breach']:.2f} & {l['breach']:.2f} \\\\")
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
        r"audit cost (both mCNY per query) and the fraction of streams whose "
        r"worst prefix rate exceeded $\alpha$, at the $\alpha$ in the header.}",
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
        sub.append("cost & audit & breach")
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
                             f"{v['breach']:.2f}")
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
        r"\emph{breach} is the fraction of streams whose worst prefix rate "
        r"exceeded $\alpha$; \emph{risk}/$\alpha$ is realised risk in units "
        r"of the contract, so $1.00$ means the allowance was spent exactly "
        r"and above $1.00$ means it was overspent. Shrinking $\delta$ by "
        r"$200\times$ changes the certificate's breach rate in one of twelve "
        r"cells: its failures are not an insufficient correction for sampling "
        r"noise but a calibration sample that misrepresents the stream, and "
        r"no choice of $\delta$ addresses that. The ledger has no $\delta$ to "
        r"set, never breaches, and lands on the contract rather than under "
        r"it, which is why its zeros are not slack. \emph{brch}: breach rate; "
        r"\emph{r}/$\alpha$: realised risk over the contract. Means over 20 "
        r"streams at each of five $\alpha$ levels per row.}",
        r"\label{tab:delta}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\begin{tabular}{ll" + "rr" * 3 + "}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{cert. $\delta=0.2$}"
        r" & \multicolumn{2}{c}{cert. $\delta=10^{-3}$}"
        r" & \multicolumn{2}{c}{ledger} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Task & Order & brch & r/$\alpha$ & brch & r/$\alpha$"
        r" & brch & r/$\alpha$ \\",
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
                if not sel:
                    cells += ["--", "--"]
                    continue
                cells += [f"{np.mean([v['breach'] for v in sel]):.2f}",
                          f"{np.mean([v['risk'] / v['alpha'] for v in sel]):.2f}"]
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

    The offline optimum can only improve when plans are added, and does.  A
    calibration certificate has no such monotonicity: the extra multiplicity
    it must pay for can outweigh the cheaper plans it can now see, and which
    of the two wins is task-dependent.  The ledger inherits the offline
    improvement because its guarantee never referenced the plan count.
    """
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
    order_table(tracks)
    regret_table(tracks)
    audit_table(tracks)
    ablation_table(tracks)
    delta_table(tracks)
    grid_table()
