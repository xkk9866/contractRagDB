#!/usr/bin/env python3
"""Publication-quality concept figures 3--5 for the KBS paper (double-column)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "KBS" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C = {
    "ink": "#1f2a37",
    "muted": "#5b6775",
    "line": "#c5ced8",
    "blue": "#2f6fed",
    "blue_lt": "#e8f0fe",
    "teal": "#0f7a6c",
    "teal_lt": "#e4f4f1",
    "orange": "#c45c26",
    "orange_lt": "#fceee6",
    "red": "#c0392b",
    "red_lt": "#fdecea",
    "green": "#1f7a4c",
    "green_lt": "#e8f6ee",
    "green_md": "#b7dfc8",
    "purple": "#5b4db8",
    "purple_lt": "#efeafc",
    "sand": "#f7f4ef",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def rounded(ax, xy, w, h, fc, ec, lw=1.4, r=0.06, z=2):
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.01,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
    )
    ax.add_patch(box)
    return box


def arrow(ax, p1, p2, color, lw=1.6, rad=0.0, ms=12):
    a = FancyArrowPatch(
        p1, p2, arrowstyle="-|>",
        mutation_scale=ms, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=3,
        shrinkA=1, shrinkB=1,
    )
    ax.add_patch(a)
    return a


def save(fig, name: str):
    for ext in ("pdf", "png"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(
            path, dpi=320, bbox_inches="tight", pad_inches=0.06,
            facecolor="white", edgecolor="none",
        )
        print("wrote", path)


def fig_rewrites():
    # Extra bottom margin so the compose/footer strip never collides with R4.
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Base plan
    rounded(ax, (0.15, 1.15), 2.75, 4.2, C["blue_lt"], C["blue"], lw=1.8, r=0.1)
    ax.text(1.52, 5.05, r"Base plan $P^{*}$", fontsize=12.5,
            fontweight="bold", color=C["blue"], ha="center")
    ax.text(1.52, 4.72, "strongest / highest cost", fontsize=8.5,
            color=C["muted"], ha="center", style="italic")

    stages = [
        (r"Retrieve  $k{=}16$", "hybrid BM25 + dense"),
        ("Rerank", "cross-encoder"),
        ("Generate", "qwen-plus"),
        ("Repair / NLI", "entailment check"),
    ]
    y0 = 4.10
    for i, (title, sub) in enumerate(stages):
        y = y0 - i * 0.70
        rounded(ax, (0.4, y - 0.28), 2.25, 0.56, "white", C["blue"], lw=1.25, r=0.06)
        ax.text(1.52, y + 0.06, title, fontsize=10, fontweight="bold",
                color=C["ink"], ha="center", va="center")
        ax.text(1.52, y - 0.16, sub, fontsize=8, color=C["muted"], ha="center", va="center")
        if i < len(stages) - 1:
            arrow(ax, (1.52, y - 0.32), (1.52, y - 0.40), C["blue"], lw=1.3, ms=11)

    # Rewrites
    ax.text(5.7, 5.20, "Rewrite lattice (cost " + "\u2193" + ")", fontsize=12,
            fontweight="bold", color=C["ink"], ha="center")
    ax.text(5.7, 4.88, "HybridQA calibration vouchers",
            fontsize=8.5, color=C["muted"], ha="center", style="italic")

    rewrites = [
        ("R1  prefilter pushdown", "\u03b5\u0302 = 0.026",
         "retrieve \u2192 filter \u2192 rerank \u2192 gen \u2192 NLI", 4.15, C["teal"], C["teal_lt"]),
        ("R2  truncate k", "\u03b5\u0302 = 0.081",
         "retrieve k=8 \u2192 rerank \u2192 gen \u2192 NLI", 3.20, C["orange"], C["orange_lt"]),
        ("R3  elide rerank", "\u03b5\u0302 = 0.078",
         "retrieve \u2192 gen \u2192 NLI", 2.25, C["purple"], C["purple_lt"]),
        ("R4  downgrade generator", "\u03b5\u0302 = 0.157",
         "retrieve \u2192 rerank \u2192 qwen-flash \u2192 NLI", 1.30, C["red"], C["red_lt"]),
    ]
    for name, eps, plan, y, col, bg in rewrites:
        arrow(ax, (2.95, 3.10), (3.55, y), col, lw=1.5, ms=11)
        rounded(ax, (3.55, y - 0.38), 4.2, 0.78, bg, col, lw=1.4, r=0.07)
        ax.text(3.75, y + 0.14, name, fontsize=9.5, fontweight="bold",
                color=C["ink"], ha="left", va="center")
        rounded(ax, (6.55, y + 0.0), 1.05, 0.32, "white", col, lw=1.15, r=0.04)
        ax.text(7.075, y + 0.16, eps, fontsize=8.5, color=col,
                ha="center", va="center", fontweight="bold")
        ax.text(3.75, y - 0.18, plan, fontsize=8, color=C["muted"], ha="left", va="center")

    # Dedicated footer band well below R4 (R4 bottom ≈ 0.92).
    rounded(ax, (3.55, 0.08), 4.2, 0.68, C["sand"], C["orange"], lw=1.15, r=0.04)
    ax.text(5.65, 0.42,
            "compose along a path: telescoping voucher (Prop. 1)\n"
            "union bound = search heuristic only",
            fontsize=8, color=C["orange"], ha="center", va="center",
            fontweight="bold", linespacing=1.35)

    # Division of labor
    rounded(ax, (8.05, 1.15), 4.3, 4.2, "#f4f6f9", "#8a96a3", lw=1.4, r=0.1)
    ax.text(10.2, 5.05, "Division of labor", fontsize=12,
            fontweight="bold", color=C["ink"], ha="center")
    ax.text(10.2, 4.72, "mirrors classical cost-based optimizers",
            fontsize=8.5, color=C["muted"], ha="center", style="italic")

    roles = [
        (C["orange_lt"], C["orange"], "Vouchers",
         "ORDER the rewrite lattice\n(may be loose / optimistic)"),
        (C["red_lt"], C["red"], "Certification",
         "alone GUARANTEES the\ndeployed policy (FWER \u2264 \u03b4)"),
        (C["green_lt"], C["green"], "Assumption NSH",
         "measured on real workloads;\nnot required for telescoping"),
    ]
    for i, (fc, ec, title, body) in enumerate(roles):
        y = 4.00 - i * 0.95
        rounded(ax, (8.3, y - 0.35), 3.8, 0.88, fc, ec, lw=1.4, r=0.07)
        ax.text(10.2, y + 0.24, title, fontsize=10.5, fontweight="bold",
                color=ec, ha="center", va="center")
        ax.text(10.2, y - 0.12, body, fontsize=8.5, color=C["ink"],
                ha="center", va="center", linespacing=1.3)

    ax.text(10.2, 0.38,
            "Like a cost model: wrong estimates hurt optimality, never soundness.",
            fontsize=7.8, color=C["muted"], ha="center", va="center", style="italic")

    save(fig, "fig_rewrites")
    plt.close(fig)


def fig_runtime():
    fig = plt.figure(figsize=(12.8, 4.6))
    fig.patch.set_facecolor("white")

    # (a) ladder
    ax = fig.add_axes([0.02, 0.06, 0.48, 0.88])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.0)
    ax.axis("off")
    rounded(ax, (0.08, 0.12), 9.84, 5.7, C["green_lt"], C["green"], lw=1.6, r=0.12)
    ax.text(0.4, 5.4, "(a)  Certified progressive ladder", fontsize=12,
            fontweight="bold", color=C["green"], ha="left")
    ax.text(0.4, 4.95, "stop when  s\u2097 \u2265 \u03bb\u0302\u2097   \u00b7   thresholds certified on calibration",
            fontsize=8.5, color=C["muted"], ha="left", style="italic")

    plans = [
        (r"$P_0$", "cheap", C["green_md"]),
        (r"$P_1$", "mid", "#7ec89a"),
        (r"$P_2$", "strong", "#4aa873"),
        (r"$P_3$", "always stop", C["green"]),
    ]
    xs = [1.25, 3.55, 5.85, 8.15]
    cy = 2.85
    for (lab, sub, col), x in zip(plans, xs):
        rounded(ax, (x - 0.78, cy - 0.58), 1.56, 1.2, "white", col, lw=1.7, r=0.08)
        ax.text(x, cy + 0.22, lab, fontsize=13.5, fontweight="bold",
                color=col, ha="center", va="center")
        ax.text(x, cy - 0.22, sub, fontsize=8.5, color=C["muted"], ha="center", va="center")

    for i, x in enumerate(xs[:-1]):
        xm = (x + xs[i + 1]) / 2
        arrow(ax, (x + 0.8, cy), (xm - 0.38, cy), C["green"], lw=1.5, ms=10)
        d = 0.36
        diamond = Polygon(
            [(xm, cy + d), (xm + d, cy), (xm, cy - d), (xm - d, cy)],
            closed=True, facecolor="white", edgecolor=C["green"],
            linewidth=1.5, zorder=4,
        )
        ax.add_patch(diamond)
        ax.text(xm, cy, "s\u2265\u03bb\u0302?", fontsize=7.5,
                color=C["ink"], ha="center", va="center", zorder=5, fontweight="bold")
        arrow(ax, (xm + 0.38, cy), (xs[i + 1] - 0.8, cy), C["green"], lw=1.5, ms=10)
        arrow(ax, (xm, cy + d + 0.02), (xm, cy + 1.45), C["red"], lw=1.4, ms=11)
        rounded(ax, (xm - 0.95, cy + 1.45), 1.9, 0.5, C["red_lt"], C["red"], lw=1.2, r=0.05)
        ax.text(xm, cy + 1.7, "STOP (certified)", fontsize=8.5,
                fontweight="bold", color=C["red"], ha="center", va="center")

    rounded(ax, (0.4, 0.35), 9.2, 0.72, "white", C["green"], lw=1.15, r=0.05)
    ax.text(5.0, 0.71,
            "\u03bb\u0302 certified on calibration  \u00b7  score models only order stops  \u00b7  "
            "final rung always available",
            fontsize=8.2, color=C["green"], ha="center", va="center", fontweight="bold")

    # (b) e-process
    ax = fig.add_axes([0.55, 0.16, 0.42, 0.72])
    rng = np.random.default_rng(7)
    t = np.arange(0, 81)
    tau = 40
    mt = np.ones(len(t), dtype=float)
    for i in range(1, len(t)):
        if t[i] < tau:
            mt[i] = max(0.6, mt[i - 1] * np.exp(rng.normal(-0.01, 0.08)))
        else:
            mt[i] = mt[i - 1] * np.exp(0.085 + rng.normal(0, 0.04))
    thr = 10.0
    alarm_idx = int(np.argmax(mt >= thr))

    ax.set_yscale("log")
    ax.plot(t, mt, color=C["purple"], lw=2.4, label=r"$M_t$ (restart mixture)", zorder=3)
    ax.axhline(thr, color=C["red"], ls="--", lw=1.7, label=r"threshold $1/\delta$")
    ax.axvline(tau, color="#9aa5b1", ls=":", lw=1.5)
    # Keep "drift onset" left of the vline so Mt never crosses it.
    ax.text(tau - 1.2, 0.55, "drift onset", fontsize=9, color=C["muted"],
            rotation=90, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.12", fc="white",
                      ec="none", alpha=0.92), zorder=6)
    ax.scatter([t[alarm_idx]], [mt[alarm_idx]], s=140, c=C["red"],
               marker="*", zorder=5, edgecolors="white", linewidths=0.7)
    ax.annotate(
        "ALARM", xy=(t[alarm_idx], mt[alarm_idx]),
        xytext=(t[alarm_idx] + 6, 55),
        fontsize=10, fontweight="bold", color=C["red"],
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.92),
        arrowprops=dict(arrowstyle="-|>", color=C["red"], lw=1.3),
        zorder=6,
    )
    # Caption sits in the clear lower-right corner; short leader avoids Mt.
    ax.annotate(
        "safe mode \u2192 recalibrate \u2192 new \u03c0\u0302",
        xy=(t[alarm_idx], thr),
        xytext=(55, 0.52),
        fontsize=8.2, color=C["purple"], ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C["purple"],
                  lw=0.8, alpha=0.95),
        arrowprops=dict(
            arrowstyle="-|>", color=C["purple"], lw=1.1,
            connectionstyle="arc3,rad=0.15",
        ),
        zorder=6,
    )

    ax.set_xlim(0, 80)
    ax.set_ylim(0.4, 80)
    ax.set_xlabel("time (queries)", fontsize=10, color=C["ink"])
    ax.set_ylabel(r"$M_t$  (test supermartingale)", fontsize=10, color=C["ink"])
    ax.set_title("(b)  Anytime-valid e-process monitor", fontsize=12,
                 fontweight="bold", color=C["purple"], loc="left", pad=10)
    ax.tick_params(labelsize=9, colors=C["muted"])
    for spine in ax.spines.values():
        spine.set_color(C["line"])
    ax.grid(True, which="both", ls=":", lw=0.6, color="#e2e8f0", alpha=0.9)
    ax.legend(
        loc="upper left", fontsize=8.5, frameon=True, fancybox=False,
        edgecolor=C["line"], framealpha=1.0,
    )
    ax.text(
        0.5, -0.20,
        "Ville bound: P(\u2203t: Mt \u2265 1/\u03b4) \u2264 \u03b4 under the contract"
        "  \u00b7  O(1) update per query",
        transform=ax.transAxes, fontsize=8, color=C["muted"], ha="center",
    )

    save(fig, "fig_runtime")
    plt.close(fig)


def fig_lifetime():
    fig, ax = plt.subplots(figsize=(12.6, 4.3))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 12.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    # legend only (caption carries the title)
    ax.plot(9.4, 4.05, marker="*", markersize=13, color=C["red"], linestyle="None")
    ax.text(9.65, 4.05, "alarm", fontsize=9.5, color=C["ink"], va="center")
    ax.plot(10.7, 4.05, marker="o", markersize=9, markerfacecolor="white",
            markeredgecolor=C["blue"], markeredgewidth=2.0, linestyle="None")
    ax.text(10.95, 4.05, "scheduled restart", fontsize=9.5, color=C["ink"], va="center")

    epochs = [
        (0.3, 4.05, "epoch 0", r"$\delta_0=\delta/2$", C["green_lt"], "alarm"),
        (4.05, 7.7, "epoch 1", r"$\delta_1=\delta/4$", C["orange_lt"], "horizon"),
        (7.7, 10.2, "epoch 2", r"$\delta_2=\delta/8$", C["purple_lt"], "cont"),
    ]
    baseline = 1.45
    top = 3.35

    for x0, x1, lab, bud, bg, kind in epochs:
        rounded(ax, (x0, baseline - 0.12), x1 - x0 - 0.06, top - baseline + 0.55,
                bg, "#d0d7de", lw=1.1, r=0.07, z=1)
        ax.text((x0 + x1) / 2, top + 0.2, lab, fontsize=11.5, fontweight="bold",
                color=C["ink"], ha="center")
        ax.text((x0 + x1) / 2, top - 0.18, bud, fontsize=10, color=C["muted"], ha="center")

        xs = np.linspace(x0 + 0.2, x1 - 0.28, 80)
        if kind == "alarm":
            ys = baseline + 0.25 + 1.35 * ((xs - x0) / (x1 - x0)) ** 1.6
            ys = ys + 0.08 * np.sin(np.linspace(0, 8, len(xs)))
            ax.plot(xs, ys, color=C["green"], lw=2.4, zorder=3)
            ax.text(x0 + 0.35, ys[10] + 0.28, r"$M_t$", fontsize=11,
                    color=C["green"], fontweight="bold")
            ax.plot([x0 + 0.15, x1 - 0.1], [ys[-1], ys[-1]],
                    ls="--", lw=1.1, color=C["red"], alpha=0.5, zorder=2)
            ax.text(x0 + 0.22, ys[-1] + 0.12, r"$1/\delta_0$", fontsize=8.5,
                    color=C["red"], alpha=0.85, va="bottom")
            # Star first, then label offset so the marker never covers "alarm".
            ax.plot(xs[-1], ys[-1], marker="*", markersize=18, color=C["red"], zorder=5)
            ax.annotate(
                "alarm", xy=(xs[-1], ys[-1]),
                xytext=(-28, 22), textcoords="offset points",
                fontsize=9.5, color=C["red"], fontweight="bold",
                ha="right", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.8,
                                shrinkA=0, shrinkB=6),
            )
        elif kind == "horizon":
            ys = baseline + 0.35 + 0.12 * np.sin(np.linspace(0, 10, len(xs)))
            ax.plot(xs, ys, color=C["orange"], lw=2.4, zorder=3)
            ax.text(x0 + 0.35, ys[12] + 0.3, r"$M_t$", fontsize=11,
                    color=C["orange"], fontweight="bold")
            ax.plot(xs[-1], baseline + 0.35, marker="o", markersize=11,
                    markerfacecolor="white", markeredgecolor=C["blue"],
                    markeredgewidth=2.2, zorder=5)
            ax.annotate(
                r"horizon $T$", xy=(xs[-1], baseline + 0.35),
                xytext=(0, 28), textcoords="offset points",
                fontsize=9.5, color=C["blue"], fontweight="bold",
                ha="center", va="bottom", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=C["blue"], lw=0.8,
                                shrinkA=0, shrinkB=5),
            )
        else:
            xs2 = np.linspace(x0 + 0.2, x1 - 0.55, 40)
            ys = baseline + 0.4 + 0.55 * ((xs2 - x0) / (x1 - x0))
            ax.plot(xs2, ys, color=C["purple"], lw=2.4, zorder=3)
            ax.text(x0 + 0.3, ys[6] + 0.28, r"$M_t$", fontsize=11,
                    color=C["purple"], fontweight="bold")
            ax.text(x1 - 0.35, baseline + 1.05, "...", fontsize=20,
                    color=C["muted"], ha="center", fontweight="bold")

    ax.annotate(
        "", xy=(11.75, baseline - 0.32), xytext=(0.3, baseline - 0.32),
        arrowprops=dict(arrowstyle="-|>", color=C["ink"], lw=1.4),
    )
    ax.text(11.95, baseline - 0.32, "queries", fontsize=10,
            color=C["ink"], va="center")

    rounded(ax, (2.5, 0.25), 2.7, 0.72, C["red_lt"], C["red"], lw=1.2, r=0.05)
    ax.text(3.85, 0.61, "safe mode \u2192 recertify", fontsize=9,
            color=C["red"], fontweight="bold", ha="center", va="center")
    arrow(ax, (3.85, 0.98), (4.0, baseline - 0.18), C["red"], lw=1.2, ms=10)

    rounded(ax, (5.35, 0.25), 3.05, 0.72, C["blue_lt"], C["blue"], lw=1.2, r=0.05)
    ax.text(6.875, 0.61, "scheduled restart \u2192 recertify", fontsize=9,
            color=C["blue"], fontweight="bold", ha="center", va="center")
    arrow(ax, (6.875, 0.98), (7.6, baseline - 0.18), C["blue"], lw=1.2, ms=10)

    rounded(ax, (8.7, 0.18), 3.6, 0.88, "#f4f6f9", "#8a96a3", lw=1.2, r=0.05)
    ax.text(10.5, 0.78, r"$\sum_{k=0}^{\infty}\delta/2^{k+1}=\delta$",
            fontsize=11.5, color=C["ink"], ha="center", va="center", fontweight="bold")
    ax.text(10.5, 0.4, "validity over arbitrarily many epochs",
            fontsize=8.5, color=C["muted"], ha="center", va="center", style="italic")

    save(fig, "fig_lifetime")
    plt.close(fig)


if __name__ == "__main__":
    fig_rewrites()
    fig_runtime()
    fig_lifetime()
    print("done")
