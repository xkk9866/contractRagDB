#!/usr/bin/env python3
"""Figures for the ledger paper, drawn from the experiment JSON.

Three figures, each carrying one argument that a table carries worse:

  fig_frontier  why declining changes the problem -- the cheapest attainable
                cost as a function of the contract level, with and without the
                decline action, and where the two methods actually operate
  fig_ledger    the gate at work over a single stream, under three orderings:
                the running violation rate against the contract (top) and the
                ledger that produces it (bottom)
  fig_regret    the cost gap to the offline optimum against stream length, and
                the composition of the bill as the contract tightens
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EXP = os.path.join(ROOT, "experiments")
FIG = os.path.join(ROOT, "paper", "KBS", "figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_LEDGER = "#1b5e9c"
C_STATIC = "#c0392b"
C_ORACLE = "#555555"
C_SOFT = "#a8c8e4"
C_ACCENT = "#d98218"
C_DEAD = "#ececec"
TRACKS = ["hybridqa", "crag", "asqa", "qampari"]
PRETTY = {"hybridqa": "HybridQA", "crag": "CRAG", "asqa": "ASQA",
          "qampari": "QAMPARI"}


def load(n):
    p = os.path.join(EXP, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=400)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png")


def median_alpha(d):
    a = sorted(float(x) for x in d["results"])
    return a[len(a) // 2]


def row_at(d, alpha, method):
    ab_ref = max(d["pop_cost"])
    return next((v for v in d["results"][str(alpha)]["rows"].values()
                 if v["method"] == method
                 and abs(v["abstain_cost"] - ab_ref) < 1e-12), None)


# ---------------------------------------------------------------------------

def fig_frontier():
    """Cheapest attainable cost against the contract level."""
    from contractrag.ledger import solve_action_lp
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15))
    axes = axes.reshape(1, 4)
    for ax, tr in zip(axes.ravel(), TRACKS):
        d = load(f"ledger_{tr}.json")
        if d is None:
            continue
        r = np.array(d["pop_risk"])
        c = np.array(d["pop_cost"]) * 1000
        cb, floor = c.max(), r.min()
        top = c.max() * 1.6
        grid = np.linspace(0.02, min(r.max() * 1.02, 0.78), 300)
        # r, c and cb are all in mCNY here; the LP is scale-free in cost
        with_ab = np.array([solve_action_lp(r, c, a, cb)[0] for a in grid])
        plans_only = np.array([c[r <= a].min() if (r <= a).any() else np.nan
                               for a in grid])

        # the region no pipeline can reach without declining
        ax.axvspan(0, floor, color=C_DEAD, lw=0, zorder=0)
        if floor / grid[-1] > 0.25:  # else the band is too narrow to label
            ax.text(floor * 0.5, c.min() * 1.35,
                    "no plan\nfeasible here", ha="center", va="bottom",
                    fontsize=6.3, color="#8a8a8a", linespacing=1.2, zorder=1)
        ax.scatter(r, c, s=9, color=C_ORACLE, alpha=0.5, lw=0, zorder=3)
        ax.plot(grid, plans_only, color=C_STATIC, lw=1.2, ls=(0, (4, 2)),
                zorder=4)
        ax.plot(grid, with_ab, color=C_LEDGER, lw=1.6, zorder=5)

        a0 = median_alpha(d)
        for m, col, mk in (("static", C_STATIC, "s"),
                           ("ledger/lp", C_LEDGER, "o")):
            v = row_at(d, a0, m)
            if v:
                ax.scatter([v["risk"]], [v["mean_cost"] * 1000], s=44,
                           marker=mk, color=col, edgecolor="white", lw=0.9,
                           zorder=7)
        ax.axvline(a0, color="#333333", lw=0.7, ls=":", zorder=2)
        ax.annotate(rf"$\alpha={a0:g}$", xy=(a0, top * 0.93),
                    xytext=(2.5, 0), textcoords="offset points", fontsize=6.8,
                    color="#333333", va="top")
        ax.set_yscale("log")
        ax.set_xlim(0, grid[-1])
        ax.set_ylim(c.min() * 0.55, top)
        ax.set_title(f"{PRETTY[tr]}   " r"$\min_p r_p=$" f"{floor:.2f}",
                     pad=3, fontsize=8.5)
        ax.grid(alpha=0.22, ls=":", which="major")
    axes[0, 0].set_ylabel("cost per query (mCNY)")
    for ax in axes[0, :]:
        ax.set_xlabel(r"contract level $\alpha$", labelpad=1.5)
    handles = [
        Line2D([], [], color=C_LEDGER, lw=1.6,
               label=r"attainable with $\bot$ (action LP)"),
        Line2D([], [], color=C_STATIC, lw=1.2, ls=(0, (4, 2)),
               label="attainable with plans only"),
        Line2D([], [], marker="o", color=C_LEDGER, lw=0, ms=6,
               markeredgecolor="white", label="ledger, realised"),
        Line2D([], [], marker="s", color=C_STATIC, lw=0, ms=6,
               markeredgecolor="white", label="one-shot cert., realised"),
        Line2D([], [], marker="o", color=C_ORACLE, lw=0, ms=3.6, alpha=0.55,
               label="candidate plan"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.055), columnspacing=1.3,
               handletextpad=0.5, fontsize=7.4)
    fig.tight_layout(w_pad=1.1, rect=(0, 0.085, 1, 1))
    save(fig, "fig_frontier")


# ---------------------------------------------------------------------------

def fig_ledger(track="hybridqa", alpha=None):
    """One stream, three orderings: the rate on top, the ledger below."""
    from contractrag.ledger import LedgerExecutor, static_stream_cost
    from scripts.ledger_stress import make_order, prep

    class A:
        max_plans = 16
    Lp, Cp, wr, wc, names, audit_unit = prep(track, A)
    ab = float(Cp.mean(axis=1).max())
    N = Lp.shape[1]
    if alpha is None:
        alpha = median_alpha(load(f"ledger_{track}.json"))

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.25), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1.0]})
    titles = ("i.i.d. shuffle", "easy queries first", "mid-stream drift")
    for j, (kind, ttl) in enumerate(zip(("iid", "easy-first", "drift"),
                                        titles)):
        rng = np.random.default_rng(3)
        order = make_order(kind, Lp, Cp, rng, wr)
        ex = LedgerExecutor(Lp, Cp, alpha, abstain_cost=ab, hist_risk=wr,
                            hist_cost=wc, kappa=1.0,
                            audit_unit_cost=audit_unit)
        st = ex.run(order, rng, log_every=10 ** 9)
        s = static_stream_cost(Lp, Cp, alpha, 0.1, order, 500, wr, wc,
                               abstain_cost=ab, audit_unit_cost=audit_unit)
        t = np.arange(1, N + 1)
        led, stat = st.cum_viol / t, s["cum_viol"] / t
        B = alpha * t - st.cum_viol

        ax = axes[0, j]
        ax.axhline(alpha, color="k", lw=1.0, ls="--", zorder=5)
        bad = stat > alpha
        if bad.any():
            ax.fill_between(t, alpha, stat, where=bad, color=C_STATIC,
                            alpha=0.16, lw=0, zorder=2)
        ax.plot(t, stat, color=C_STATIC, lw=1.15, ls=(0, (4, 2)), zorder=4)
        ax.plot(t, led, color=C_LEDGER, lw=1.5, zorder=6)
        ax.axvline(500, color=C_ACCENT, lw=0.8, ls=":", zorder=3)
        hi = max(alpha * 1.55, np.nanmax(stat[100:]) * 1.10)
        ax.set_ylim(0, hi)
        ax.set_title(ttl, pad=3)
        ax.grid(alpha=0.22, ls=":")
        if j == 0:
            ax.set_ylabel(r"violation rate $V_t/t$")
            ax.annotate(r"contract $\alpha$", xy=(N * 0.99, alpha),
                        xytext=(0, 3), textcoords="offset points",
                        ha="right", fontsize=7)
            ax.annotate("calibration\nends", xy=(500, hi * 0.97),
                        xytext=(5, 0), textcoords="offset points",
                        fontsize=6.4, color=C_ACCENT, va="top",
                        linespacing=1.2)
        if bad.any():
            k = int(np.argmax(stat[200:] - alpha)) + 200
            ax.annotate("contract\nbreached", xy=(t[k], stat[k]),
                        xytext=(-6, -14), textcoords="offset points",
                        fontsize=6.4, color=C_STATIC, ha="right",
                        linespacing=1.2)

        ax = axes[1, j]
        ax.axhline(0, color="k", lw=0.6, zorder=4)
        ax.axhline(1 - alpha, color=C_ACCENT, lw=0.9, ls=":", zorder=4)
        ax.fill_between(t, 0, B, where=B >= 0, color=C_SOFT, alpha=0.55,
                        lw=0, zorder=2)
        ax.plot(t, B, color=C_LEDGER, lw=1.0, zorder=5)
        closed = B < 1 - alpha
        if closed.any():
            ax.fill_between(t, 0, 1, where=closed, transform=ax.get_xaxis_transform(),
                            color=C_ACCENT, alpha=0.13, lw=0, zorder=1)
        ax.set_xlabel(r"queries served $t$")
        ax.set_xlim(0, N)
        ax.set_ylim(-max(2.0, B.max() * 0.04), max(B.max() * 1.12, 4))
        ax.grid(alpha=0.22, ls=":")
        if j == 0:
            ax.set_ylabel(r"ledger $B_t$")
            ax.annotate(r"reserve $1-\alpha$", xy=(N * 0.36, 1 - alpha),
                        xytext=(0, 5), textcoords="offset points",
                        fontsize=6.4, color=C_ACCENT)
    handles = [
        Line2D([], [], color=C_LEDGER, lw=1.5, label="ledger-gated execution"),
        Line2D([], [], color=C_STATIC, lw=1.15, ls=(0, (4, 2)),
               label="one-shot certificate"),
        Line2D([], [], color="k", lw=1.0, ls="--", label=r"contract $\alpha$"),
        Patch(facecolor=C_ACCENT, alpha=0.16,
              label="gate closed (declining)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.055), columnspacing=1.5,
               handletextpad=0.6)
    fig.tight_layout(h_pad=0.6, w_pad=1.1)
    save(fig, "fig_ledger")


# ---------------------------------------------------------------------------

def fig_regret():
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.4))
    ax = axes[0]
    cols = {"hybridqa": C_LEDGER, "crag": C_ACCENT, "asqa": "#2e7d5b",
            "qampari": C_STATIC}
    anchor = None
    for tr, mk in zip(TRACKS, ("o", "s", "^", "D")):
        d = load(f"ledger_stress_{tr}.json")
        if d is None or "regret" not in d:
            continue
        by_a = {}
        for v in d["regret"].values():
            by_a.setdefault(v["alpha"], []).append(v)
        a = sorted(by_a)[len(by_a) // 2]
        vs = sorted(by_a[a], key=lambda v: v["T"])
        T = np.array([v["T"] for v in vs], float)
        g = np.array([v["gap"] for v in vs]) * 1000
        ax.plot(T, g, marker=mk, ms=3.6, lw=1.15, color=cols[tr],
                markeredgecolor="white", markeredgewidth=0.5,
                label=f"{PRETTY[tr]} ($\\alpha={a:g}$)")
        if tr == "hybridqa":
            anchor = (T, g)
    if anchor is not None:
        T, g = anchor
        ref = g[0] * np.sqrt(T[0] / T)
        ax.plot(T, ref, color=C_ORACLE, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.annotate(r"$\propto T^{-1/2}$", xy=(T[-1], ref[-1]),
                    xytext=(-2, -11), textcoords="offset points",
                    ha="right", fontsize=7.2, color=C_ORACLE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 4.5)
    ax.set_xlabel("stream length $T$")
    ax.set_ylabel("serving cost above\noffline optimum (mCNY)")
    ax.set_title("learning cost vanishes", pad=3)
    ax.grid(alpha=0.22, ls=":", which="both")
    ax.legend(frameon=False, loc="upper right", handlelength=1.6,
              borderaxespad=0.2, labelspacing=0.28)

    ax = axes[1]
    d = load("ledger_hybridqa.json")
    if d:
        alphas = sorted(float(a) for a in d["results"])
        ab_ref = max(d["pop_cost"])
        au_unit = d["audit_unit"] * 1000
        ser, aud, dec, base = [], [], [], []
        for a in alphas:
            v = row_at(d, a, "ledger/lp")
            tot = v["mean_cost"] * 1000
            dr = v["abstain_rate"] / 100.0
            dc = dr * ab_ref * 1000
            ac = au_unit * (1 - dr)
            dec.append(dc)
            aud.append(ac)
            ser.append(max(tot - dc - ac, 0.0))
            base.append(row_at(d, a, "static")["mean_cost"] * 1000)
        x = np.arange(len(alphas))
        b0 = np.array(ser)
        b1 = b0 + np.array(dec)
        ax.bar(x, ser, 0.6, color=C_LEDGER, label="serving (LLM calls)")
        ax.bar(x, dec, 0.6, bottom=b0, color=C_SOFT, label="declines")
        ax.bar(x, aud, 0.6, bottom=b1, color=C_ACCENT, label="audit (labels)")
        ax.plot(x, base, marker="s", ms=4.4, lw=1.2, color=C_STATIC,
                markeredgecolor="white", markeredgewidth=0.6,
                label="one-shot certificate, total")
        for xi, (t_, b_) in enumerate(zip(b1 + np.array(aud), base)):
            ax.annotate(f"{b_/t_:.1f}$\\times$", xy=(xi, t_),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.6, color=C_LEDGER)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:g}" for a in alphas])
        ax.set_xlabel(r"contract level $\alpha$ (HybridQA)")
        ax.set_ylabel("cost per query (mCNY)")
        ax.set_title("where the bill goes, and the gap it closes", pad=3)
        ax.set_ylim(0, max(base) * 1.52)
        ax.grid(alpha=0.22, ls=":", axis="y")
        ax.legend(frameon=False, loc="upper center", ncol=2,
                  handlelength=1.3, columnspacing=1.0, labelspacing=0.28,
                  borderaxespad=0.1, fontsize=6.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_regret")


if __name__ == "__main__":
    which = sys.argv[1:] or ["frontier", "ledger", "regret"]
    if "frontier" in which:
        fig_frontier()
    if "ledger" in which:
        fig_ledger()
    if "regret" in which:
        fig_regret()
