"""Generate paper figures and LaTeX tables from experiment JSON artifacts.

Figures:
  F1 cost-risk scatter per track (methods at one alpha)  [main]
  F2 calibration: target alpha vs realized violation (sweep)
  F3 violation-rate bars across repeated splits (safety headline)
  F4 drift: violation time series + e-process alarm
  F5 rewrite vouchers: eps-hat vs realized harm; composed vs direct
Tables:
  T1 main comparison per track (risk / cost / p95 / stop histogram)
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, "experiments")
FIG_DIRS = [
    os.path.join(ROOT, "paper", "figures"),
    os.path.join(ROOT, "paper", "KBS", "figures"),
]
for _d in FIG_DIRS:
    os.makedirs(_d, exist_ok=True)
# Back-compat alias used by older call sites in this file.
FIG = FIG_DIRS[0]

# Opaque label background so annotations stay readable over curves/shading.
_LABEL_BBOX = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=1.0)
_LEGEND_KW = dict(frameon=True, fancybox=False, framealpha=1.0,
                  edgecolor="#d0d0d0", borderpad=0.35)

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    # Match the journal body font (STIX) so figures typeset seamlessly.
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "axes.linewidth": 0.7, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "savefig.transparent": False,
})


def savefig(fig, out):
    """Write the same figure into every paper figure directory."""
    for d in FIG_DIRS:
        path = os.path.join(d, out)
        fig.savefig(path, bbox_inches="tight")
        print("wrote", path)

TRACK_NAME = {"hybridqa": "HybridQA", "crag": "CRAG", "asqa": "ASQA",
              "qampari": "QAMPARI"}
CONTRACT_NAME = {"quality": "quality", "correct": "correctness",
                 "citation": "citation", "hallu": "hallucination"}

METHOD_STYLE = {
    "contractrag_opt": dict(label="ContractRAG", color="#d62728", marker="*", s=140, zorder=5),
    "contractrag": dict(label="ContractRAG (ladder)", color="#ff7f0e", marker="^", s=60, zorder=4),
    "empirical_no_cert": dict(label="Empirical (no cert.)", color="#8c564b", marker="v", s=50),
    "bo_thresholds": dict(label="BO-tuned", color="#9467bd", marker="D", s=40),
    "utility_router": dict(label="Utility router", color="#2ca02c", marker="s", s=40),
    "abacus": dict(label="Abacus", color="#1f77b4", marker="P", s=50),
    "abacus_style": dict(label="Abacus", color="#1f77b4", marker="P", s=50),  # legacy key
    "adaptive_rag_router": dict(label="Complexity router", color="#e377c2", marker="X", s=50),
    "oracle": dict(label="Oracle", color="#7f7f7f", marker="o", s=40),
}


def load(name):
    path = os.path.join(EXP, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def method_entry(methods_dict, key):
    """Resolve a method key; accept legacy abacus_style alias."""
    if key in methods_dict:
        return methods_dict[key]
    if key == "abacus" and "abacus_style" in methods_dict:
        return methods_dict["abacus_style"]
    return None


def fig_cost_risk(track, contract, tau, alpha_key, out):
    res = load(f"main_{track}_{contract}_tau{tau}.json")
    if res is None:
        print("missing main results for", track)
        return
    fig, ax = plt.subplots(figsize=(4.3, 2.6))
    alpha = float(alpha_key)
    for m, per_alpha in res["methods"].items():
        entry = per_alpha.get(alpha_key) or per_alpha.get("na")
        if entry is None or m.startswith("fixed_rung"):
            continue
        # unify legacy key for styling
        style_key = "abacus" if m == "abacus_style" else m
        st = METHOD_STYLE.get(style_key) or METHOD_STYLE.get(m)
        if st is None:
            continue
        if m == "abacus_style":
            continue  # drawn via abacus alias if present; else fall through below
        ax.scatter(entry["cost_mean"] * 1000, entry["risk"],
                   c=st["color"], marker=st["marker"], s=st["s"],
                   label=st["label"], zorder=st.get("zorder", 3),
                   edgecolors="black", linewidths=0.4)
    # legacy-only abacus_style (no abacus key)
    if "abacus" not in res["methods"] and "abacus_style" in res["methods"]:
        per = res["methods"]["abacus_style"]
        entry = per.get(alpha_key) or per.get("na")
        if entry:
            st = METHOD_STYLE["abacus"]
            ax.scatter(entry["cost_mean"] * 1000, entry["risk"],
                       c=st["color"], marker=st["marker"], s=st["s"],
                       label=st["label"], zorder=st.get("zorder", 3),
                       edgecolors="black", linewidths=0.4)
    # fixed rungs as a curve
    xs, ys = [], []
    for j in range(8):
        e = res["methods"].get(f"fixed_rung_{j}", {}).get("na")
        if e:
            xs.append(e["cost_mean"] * 1000)
            ys.append(e["risk"])
    ax.plot(xs, ys, "k--o", lw=0.8, ms=3, label="Fixed plans", zorder=2)
    ax.axhline(alpha, color="red", lw=0.9, ls=":", zorder=1)
    ylo, yhi = ax.get_ylim()
    if yhi > alpha:
        ax.axhspan(alpha, yhi, color="red", alpha=0.06, zorder=0)
        ax.set_ylim(ylo, yhi)
    ax.annotate("contract violated", xy=(0.99, 0.975),
                xycoords="axes fraction", ha="right", va="top",
                fontsize=7, color="red", alpha=0.9, zorder=10,
                bbox=_LABEL_BBOX)
    ax.set_xlabel("mean cost (mCNY / query)")
    ax.set_ylabel("realized violation rate")
    ax.set_title(f"{TRACK_NAME.get(track, track)} "
                 f"({CONTRACT_NAME.get(contract, contract)}), "
                 + rf"$\alpha={alpha}$")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              ncol=1, fontsize=7, handletextpad=0.4)
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


def fig_calibration(tracks_contracts, out):
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    markers = ["o", "s", "^", "D"]
    for (track, contract, tau), mk in zip(tracks_contracts, markers):
        res = load(f"main_{track}_{contract}_tau{tau}.json")
        if res is None:
            continue
        alphas, risks = [], []
        for a in res["alphas"]:
            e = res["methods"].get("contractrag_opt", {}).get(str(a))
            if e:
                alphas.append(a)
                risks.append(e["risk"])
        ax.plot(alphas, risks, marker=mk, ms=4, lw=1,
                label=TRACK_NAME.get(track, track))
    lim = ax.get_xlim()
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label=r"$y = x$ (contract level)")
    ax.fill_between([0, 1], [0, 1], [1, 1], color="red", alpha=0.05,
                    zorder=0)
    ax.set_xlim(lim)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"contract level $\alpha$")
    ax.set_ylabel("realized violation rate")
    ax.legend(fontsize=7.5, loc="upper left", **_LEGEND_KW)
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


def fig_violation_bars(specs, out):
    """specs: list of (track, contract, alpha) repeat_ files."""
    entries = []
    for track, contract, alpha in specs:
        r = load(f"repeat_{track}_{contract}_a{alpha}.json")
        if r:
            entries.append((track, r))
    if not entries:
        print("no repeat files yet")
        return
    methods = ["contractrag_opt", "contractrag", "empirical_no_cert",
               "bo_thresholds", "utility_router", "abacus"]
    fig, ax = plt.subplots(figsize=(4.6, 2.4))
    width = 0.8 / len(methods)
    xs = np.arange(len(entries))
    for k, m in enumerate(methods):
        vals = []
        for e in entries:
            entry = method_entry(e[1]["methods"], m)
            vals.append(entry.get("violation_rate", np.nan) if entry else np.nan)
        st = METHOD_STYLE[m]
        ax.bar(xs + k * width, vals, width, label=st["label"], color=st["color"])
    delta = entries[0][1]["delta"]
    ax.axhline(delta, color="red", ls=":", lw=1, label=rf"$\delta={delta}$")
    ax.set_xticks(xs + 0.4 - width / 2)
    ax.set_xticklabels([e[0] for e in entries])
    ax.set_ylabel(r"$\Pr(\mathrm{test\ risk} > \alpha)$")
    ax.legend(ncol=2, fontsize=7, **_LEGEND_KW)
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


def fig_drift(alpha, out):
    r = load(f"drift_crag_a{alpha}.json")
    if r is None:
        print("no drift results yet")
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    runs = r["runs"]
    T1 = runs["marginal"]["T1"]
    for mode, color in [("marginal", "#8c564b"), ("monitor", "#d62728"),
                        ("group", "#2ca02c")]:
        losses = np.array(runs[mode]["losses"], dtype=float)
        w = 100
        roll = np.convolve(losses, np.ones(w) / w, mode="valid")
        axes[0].plot(np.arange(len(roll)) + w, roll, lw=1, color=color,
                     label=mode)
    axes[0].axvline(T1, color="k", ls="--", lw=0.8)
    axes[0].axhline(r["alpha"], color="red", ls=":", lw=0.8)
    at = runs["monitor"]["alarm_t"]
    if at is not None:
        axes[0].axvline(at, color="#d62728", ls="-.", lw=0.8)
        axes[0].annotate("alarm", (at, r["alpha"] + 0.02), color="#d62728",
                         fontsize=7)
    axes[0].set_xlabel("query index t")
    axes[0].set_ylabel("violation rate (window 100)")
    axes[0].legend(fontsize=7, **_LEGEND_KW)

    delays = [d for d in r["alarm_delays"] if d is not None]
    axes[1].hist(delays, bins=15, color="#d62728", alpha=0.8)
    axes[1].set_xlabel("detection delay (queries after shift)")
    axes[1].set_ylabel("runs / 20 seeds")
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


def table_main(track, contract, tau, alpha_key, fname):
    res = load(f"main_{track}_{contract}_tau{tau}.json")
    if res is None:
        return
    rows = []
    order = ["fixed_rung_0", "fixed_rung_1", "fixed_rung_2", "fixed_rung_3",
             "adaptive_rag_router", "utility_router", "bo_thresholds",
             "abacus", "empirical_no_cert", "contractrag",
             "contractrag_opt", "oracle"]
    for m in order:
        per = method_entry(res["methods"], m)
        if not per:
            continue
        e = per.get(alpha_key) or per.get("na")
        if not e:
            continue
        st = METHOD_STYLE.get(m, {"label": m})
        rows.append((st["label"], e["risk"], e["cost_mean"] * 1000,
                     e.get("lat_p95", float("nan"))))
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & Violation & Cost (mCNY) & p95 lat.\ (s) \\",
             r"\midrule"]
    for label, risk, cost, lat in rows:
        lines.append(f"{label} & {risk:.3f} & {cost:.2f} & {lat:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(ROOT, "paper", fname), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("wrote", fname)


def fig_violation_sweep(specs, out):
    """Violation rate vs alpha for each method, one panel per track."""
    fig, axes = plt.subplots(1, len(specs), figsize=(2.4 * len(specs), 2.3),
                             sharey=True)
    if len(specs) == 1:
        axes = [axes]
    methods = ["contractrag_opt", "contractrag", "empirical_no_cert",
               "bo_thresholds", "utility_router", "abacus"]
    for ax, (track, contract, alphas) in zip(axes, specs):
        for m in methods:
            xs, ys = [], []
            for a in alphas:
                r = load(f"repeat_{track}_{contract}_a{a}.json")
                entry = method_entry(r["methods"], m) if r else None
                if entry:
                    xs.append(a)
                    # exact finite-population risk when available (direct
                    # verification of the certificate); held-out otherwise
                    ys.append(entry.get("violation_rate_pop",
                                        entry["violation_rate"]))
            if xs:
                st = METHOD_STYLE[m]
                ax.plot(xs, ys, marker=st["marker"], ms=4, lw=1.1,
                        color=st["color"], label=st["label"],
                        zorder=st.get("zorder", 3))
        ax.axhline(0.1, color="red", ls=":", lw=1, zorder=2)
        ax.set_ylim(-0.03, 0.72)
        ax.set_title(TRACK_NAME.get(track, track), fontsize=9)
        ax.set_xlabel(r"contract level $\alpha$")
    axes[0].set_ylabel(r"$\Pr(\mathrm{population\ risk} > \alpha)$")
    # Below the delta line on HybridQA left: only near-zero curves live there.
    axes[0].annotate(
        r"budget $\delta$", xy=(0.02, 0.1),
        xycoords=("axes fraction", "data"), fontsize=7,
        color="red", ha="left", va="top",
        xytext=(0, -2), textcoords="offset points",
        bbox=_LABEL_BBOX, zorder=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7.5, ncol=6,
               loc="upper center", bbox_to_anchor=(0.5, 1.13),
               columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


def fig_drift_file(name, out):
    r = load(name)
    if r is None:
        print("no drift results yet:", name)
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    runs = r["runs"]
    T1 = runs["marginal"]["T1"]
    for mode, color, label in [("marginal", "#8c564b", "static-certified"),
                               ("monitor", "#d62728", "+ e-process monitor"),
                               ("group", "#2ca02c", "group-conditional")]:
        losses = np.array(runs[mode]["losses"], dtype=float)
        w = 150
        roll = np.convolve(losses, np.ones(w) / w, mode="valid")
        n_all = len(losses)
        axes[0].plot(np.arange(len(roll)) + w, roll, lw=1.1, color=color,
                     label=label, zorder=3)
    axes[0].axvspan(T1, n_all, color="k", alpha=0.05, zorder=0)
    axes[0].axvline(T1, color="k", ls="--", lw=0.8, zorder=4)
    # Keep labels outside the dense post-drift curves.
    axes[0].annotate(
        "drift onset", xy=(T1, 1.02), xycoords=("data", "axes fraction"),
        xytext=(-4, 0), textcoords="offset points",
        fontsize=7, color="k", ha="right", va="bottom",
        bbox=_LABEL_BBOX, zorder=10, annotation_clip=False,
    )
    axes[0].axhline(r["alpha"], color="red", ls=":", lw=0.8, zorder=2)
    # Post-drift curves sit well above alpha on the right — label stays readable.
    axes[0].annotate(
        r"contract $\alpha$", xy=(0.98, r["alpha"]),
        xycoords=("axes fraction", "data"), fontsize=7, color="red",
        xytext=(0, -8), textcoords="offset points", ha="right", va="top",
        bbox=_LABEL_BBOX, zorder=10,
    )
    at = runs["monitor"]["alarm_t"]
    if at is not None:
        axes[0].axvline(at, color="#d62728", ls="-.", lw=0.8, zorder=4)
        axes[0].annotate(
            "alarm", xy=(at, 1.02), xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points",
            color="#d62728", fontsize=7, ha="left", va="bottom",
            bbox=_LABEL_BBOX, zorder=10, annotation_clip=False,
        )
    axes[0].set_xlabel("query index $t$")
    axes[0].set_ylabel("violation rate (rolling 150)")
    axes[0].legend(
        fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.08),
        ncol=3, borderaxespad=0.0, **_LEGEND_KW,
    )
    delays = [d for d in r["alarm_delays"] if d is not None]
    axes[1].hist(delays, bins=12, color="#d62728", alpha=0.85)
    if delays:
        med = float(np.median(delays))
        axes[1].axvline(med, color="k", ls="--", lw=0.8)
        axes[1].annotate(
            f"median {med:.0f}", (med, 0.95), fontsize=7,
            xycoords=("data", "axes fraction"), va="top",
            xytext=(3, 0), textcoords="offset points",
            bbox=_LABEL_BBOX, zorder=10,
        )
    axes[1].set_xlabel("detection delay after drift onset")
    axes[1].set_ylabel("runs (of 20)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    savefig(fig, out)
    plt.close(fig)


def fig_stale(name, out):
    r = load(name)
    if r is None:
        print("no stale results yet:", name)
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    runs = r["runs"]
    T1 = runs["static"]["T1"]
    for mode, color, label in [("static", "#8c564b", "static-certified"),
                               ("oracle", "#7f7f7f", "oracle recalibration"),
                               ("monitor", "#d62728", "+ e-process monitor")]:
        losses = np.array(runs[mode]["losses"], dtype=float)
        w = 150
        roll = np.convolve(losses, np.ones(w) / w, mode="valid")
        n_all = len(losses)
        axes[0].plot(np.arange(len(roll)) + w, roll, lw=1.1, color=color,
                     label=label, zorder=3)
    axes[0].axvspan(T1, n_all, color="k", alpha=0.05, zorder=0)
    axes[0].axvline(T1, color="k", ls="--", lw=0.8, zorder=4)
    # Gap between the high static curve (~0.9+) and the recovered band (~0.75).
    axes[0].annotate(
        "index falls\n1 week behind", xy=(0.78, 0.825),
        xycoords=("axes fraction", "data"),
        fontsize=7, color="k", ha="center", va="center",
        bbox=_LABEL_BBOX, zorder=10,
    )
    axes[0].axhline(r["alpha"], color="red", ls=":", lw=0.8, zorder=2)
    axes[0].annotate(
        r"contract $\alpha$", xy=(0.02, r["alpha"]),
        xycoords=("axes fraction", "data"), fontsize=7, color="red",
        xytext=(0, -10), textcoords="offset points", va="top",
        bbox=_LABEL_BBOX, zorder=10,
    )
    at = runs["monitor"]["alarm_t"]
    if at is not None:
        axes[0].axvline(at, color="#d62728", ls="-.", lw=0.8, zorder=4)
        axes[0].annotate(
            "alarm", xy=(at, 1.02), xycoords=("data", "axes fraction"),
            xytext=(4, 0), textcoords="offset points",
            color="#d62728", fontsize=7, ha="left", va="bottom",
            bbox=_LABEL_BBOX, zorder=10, annotation_clip=False,
        )
    axes[0].set_xlabel("query index $t$")
    axes[0].set_ylabel("joint violation (rolling 150)")
    axes[0].set_ylim(top=1.0)
    axes[0].legend(
        fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.08),
        ncol=3, borderaxespad=0.0, **_LEGEND_KW,
    )
    delays = [d for d in r["alarm_delays"] if d is not None]
    axes[1].hist(delays, bins=12, color="#d62728", alpha=0.85)
    if delays:
        med = float(np.median(delays))
        axes[1].axvline(med, color="k", ls="--", lw=0.8)
        axes[1].annotate(
            f"median {med:.0f}", (med, 0.95), fontsize=7,
            xycoords=("data", "axes fraction"), va="top",
            xytext=(3, 0), textcoords="offset points",
            bbox=_LABEL_BBOX, zorder=10,
        )
    axes[1].set_xlabel("detection delay after index lag onset")
    axes[1].set_ylabel("runs (of 20)")
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    savefig(fig, out)
    plt.close(fig)


def fig_drift2(name, out):
    """Drift 2.0: delay scaling with magnitude + changepoint localization."""
    r = load(name)
    if r is None:
        print("no drift2 results yet:", name)
        return
    curve = [d for d in r["delay_curve"] if d["delay_mean"] is not None
             and d.get("g_star") and d["g_star"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    g = [d["g_star"] for d in curve]
    dl = [d["delay_mean"] for d in curve]
    p90 = [d["delay_p90"] for d in curve]
    axes[0].plot(g, dl, "o-", color="#d62728", lw=1.5, label="measured delay (mean)")
    axes[0].plot(g, p90, "s--", color="#ff9896", lw=1, ms=4,
                 label="measured delay (p90)")
    gg = np.linspace(min(g) * 0.9, max(g) * 1.1, 100)
    const = r.get("bound_constant", np.log(1 / r["delta"]) + np.log(5e5))
    axes[0].plot(gg, const / gg, ":", color="#333333", lw=1.4,
                 label=r"bound $\frac{\log(1/\delta)+\log(T|W|)}{g^\ast}$")
    axes[0].set_xlabel(r"post-change optimal log-growth $g^\ast$")
    axes[0].set_ylabel("detection delay (queries)")
    axes[0].set_yscale("log")
    axes[0].set_title(r"delay $\leq$ bound, scales as $1/g^\ast$ (Prop. 4)")
    axes[0].legend(fontsize=6.5, loc="upper right", **_LEGEND_KW)
    rho = [d["rho"] for d in curve]
    cp = [d["cp_err_median"] for d in curve]
    axes[1].bar([str(x) for x in rho], cp, color="#1f77b4", width=0.55)
    axes[1].set_xlabel(r"drift magnitude $\rho$ (post-change dynamic fraction)")
    axes[1].set_ylabel(r"median $|\hat\tau - \tau|$")
    axes[1].set_title("changepoint localization")
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


def fig_families(out):
    """RQ8: cross-family certified cost-risk (per family: rungs + policies)."""
    r = load("families_hybridqa.json")
    if r is None:
        print("no families results yet")
        return
    fams = r["families"]
    order = [f for f in ["qwen", "deepseek", "glm", "gemma"] if f in fams]
    label = {"qwen": "Qwen (hosted)", "deepseek": "DeepSeek (hosted)",
             "glm": "GLM (hosted)", "gemma": "Gemma-3 (local GPU)"}
    fig, axes = plt.subplots(1, len(order), figsize=(7.0, 2.2), sharey=True)
    for ax, fam in zip(np.atleast_1d(axes), order):
        d = fams[fam]
        rungs_c = d["per_rung_cost_mCNY"]
        rungs_r = d["per_rung_risk"]
        ax.plot(rungs_c, rungs_r, "o-", color="#bbbbbb", ms=4, lw=1,
                label="fixed rungs", zorder=1)
        ax.scatter([d["strongest_cost_mCNY"]], [d["strongest_risk"]],
                   marker="s", color="#1f77b4", s=45, label="strongest",
                   zorder=3)
        c = d["contractrag"]
        ax.scatter([c["test_cost_mCNY"]], [c["test_risk"]], marker="*",
                   color="#d62728", s=140, label="ContractRAG", zorder=4)
        u = d["utility_router"]
        ax.scatter([u["test_cost_mCNY"]], [u["test_risk"]], marker="^",
                   color="#2ca02c", s=45, label="utility router", zorder=3)
        ax.axhline(d.get("alpha", r.get("alpha", 0.35)), color="#7f7f7f",
                   lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(label[fam], fontsize=8.5)
        ax.set_xlabel("cost (mCNY/query, log)")
    np.atleast_1d(axes)[0].set_ylabel("violation rate")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=4,
               loc="upper center", bbox_to_anchor=(0.5, 1.14),
               columnspacing=1.2, handletextpad=0.4)
    fig.tight_layout()
    savefig(fig, out)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "new":
        fig_drift2("drift2_crag_correct_a0.66.json", "drift2.pdf")
        fig_families("families.pdf")
    if which == "smoke":
        fig_cost_risk("asqa", "citation_internal", 0.5, "0.647",
                      "cost_risk_asqa_internal.pdf")
        fig_violation_bars([("asqa", "citation_internal", 0.65)],
                           "violation_bars_smoke.pdf")
    if which == "all":
        fig_cost_risk("hybridqa", "quality", 0.5, "0.34",
                      "cost_risk_hybridqa.pdf")
        fig_cost_risk("crag", "correct", 0.5, "0.65", "cost_risk_crag.pdf")
        fig_cost_risk("asqa", "citation", 50.0, "0.216", "cost_risk_asqa.pdf")
        fig_cost_risk("qampari", "citation", 50.0, "0.66",
                      "cost_risk_qampari.pdf")
        fig_calibration([("hybridqa", "quality", 0.5),
                         ("crag", "correct", 0.5),
                         ("asqa", "citation", 50.0),
                         ("qampari", "citation", 50.0)], "calibration.pdf")
        fig_violation_bars([("hybridqa", "quality", 0.35),
                            ("crag", "correct", 0.65),
                            ("asqa", "citation", 0.25)], "violation_bars.pdf")
        fig_violation_sweep([
            ("hybridqa", "quality", [0.32, 0.35, 0.4, 0.48]),
            ("crag", "correct", [0.62, 0.65, 0.7]),
            ("asqa", "citation", [0.15, 0.25, 0.35]),
            ("qampari", "citation", [0.63, 0.66, 0.71]),
        ], "violation_sweep.pdf")
        fig_drift_file("drift_crag_correct_a0.66.json", "drift.pdf")
        fig_stale("stale_crag_a0.75.json", "stale.pdf")
        table_main("hybridqa", "quality", 0.5, "0.34", "table_hybridqa.tex")
        table_main("crag", "correct", 0.5, "0.65", "table_crag.tex")
        table_main("asqa", "citation", 50.0, "0.216", "table_asqa.tex")
        table_main("qampari", "citation", 50.0, "0.66", "table_qampari.tex")
