#!/usr/bin/env python3
"""Publication figures for the ledger paper: method, contrast, controller.

Drawn at 4K with the soft-pastel modular style of contemporary AI-system
figures, but with formulas and claims that match the manuscript exactly.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "paper", "KBS", "figures")
os.makedirs(FIG, exist_ok=True)

# pastel palette from the reference boards
C = {
    "blue": "#c0daf4", "blue_d": "#6496d5", "blue_t": "#1a4a7a",
    "coral": "#e88d8a", "coral_d": "#c45c58", "coral_t": "#7a2e2b",
    "sand": "#e4cf92", "sand_d": "#c9a84a", "sand_t": "#6b5420",
    "mint": "#b4daae", "mint_d": "#5a9e5a", "mint_t": "#2d5a2d",
    "yellow": "#f8d666", "yellow_d": "#d4a017", "yellow_t": "#6b5010",
    "lav": "#E2D5E7", "lav_d": "#8B7B9E", "lav_t": "#4a3d5c",
    "pink": "#F7CECC", "cream": "#FFF5C4", "sky": "#9FCEFC",
    "softb": "#D9E8FB", "softg": "#F3FEF6", "white": "#ffffff",
    "ink": "#1f2a37", "mute": "#5a6570", "line": "#8a95a1",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "bold",
    "mathtext.fontset": "dejavusans",
    "text.color": C["ink"],
    "axes.linewidth": 0,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
})


def rounded(ax, xy, w, h, fc, ec=None, lw=1.6, r=0.03, z=1, alpha=1.0):
    box = FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor=fc, edgecolor=ec or _darken(fc), linewidth=lw,
        alpha=alpha, zorder=z, mutation_aspect=0.4,
    )
    ax.add_patch(box)
    # soft shadow
    sh = FancyBboxPatch(
        (xy[0] + 0.006, xy[1] - 0.008), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={r}",
        facecolor="#00000018", edgecolor="none", zorder=z - 1,
        mutation_aspect=0.4,
    )
    ax.add_patch(sh)
    return box


def _darken(hex_c, f=0.72):
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r*f):02x}{int(g*f):02x}{int(b*f):02x}"


def arrow(ax, p1, p2, color=None, lw=2.2):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color=color or C["blue_d"],
                        lw=lw, mutation_scale=14),
        zorder=5,
    )


def txt(ax, x, y, s, size=9, color=None, ha="left", va="center",
        weight="bold", wrap=None):
    ax.text(x, y, s, fontsize=size, color=color or C["ink"],
            ha=ha, va=va, fontweight=weight, zorder=6,
            linespacing=1.25)


def badge(ax, x, y, w, h, label, fc, size=7.5):
    rounded(ax, (x, y), w, h, fc, r=0.02, lw=1.2, z=3)
    txt(ax, x + w / 2, y + h / 2, label, size=size, ha="center",
        color=_darken(fc, 0.45))


def save(fig, name):
    # 14.2 in * 400 dpi ≈ 5680 px → above 4K UHD width
    for ext in ("png", "pdf"):
        path = os.path.join(FIG, f"{name}.{ext}")
        fig.savefig(path, dpi=400 if ext == "png" else 300,
                    bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Figure 1 — method framework (3 columns)
# ---------------------------------------------------------------------------

def fig_method():
    fig, ax = plt.subplots(figsize=(14.2, 7.6), dpi=200)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")

    # column panels
    rounded(ax, (0.25, 1.55), 4.15, 5.7, C["softb"], C["blue_d"], lw=2.0, r=0.04)
    rounded(ax, (4.75, 1.55), 4.35, 5.7, C["softg"], C["mint_d"], lw=2.0, r=0.04)
    rounded(ax, (9.45, 1.55), 4.5, 5.7, "#fff8e6", C["sand_d"], lw=2.0, r=0.04)

    # headers
    for x, lab, col in (
        (0.45, "1   Query & Contract", C["blue_t"]),
        (4.95, "2   Ledger Gate", C["mint_t"]),
        (9.65, "3   Cost Optimisation", C["sand_t"]),
    ):
        txt(ax, x, 6.9, lab, size=13, color=col)

    # --- col 1 ---
    rounded(ax, (0.5, 5.55), 3.65, 1.05, C["white"], C["blue_d"], lw=1.2)
    txt(ax, 0.7, 6.3, "User query  $q_t$", size=9.5, color=C["blue_t"])
    txt(ax, 0.7, 5.9, "RAG service under a rate contract", size=8,
        color=C["mute"], weight="normal")

    rounded(ax, (0.5, 4.15), 3.65, 1.2, C["white"], C["blue_d"], lw=1.2)
    txt(ax, 0.7, 5.05, "Evidence contract", size=9.5, color=C["blue_t"])
    badge(ax, 0.7, 4.35, 1.45, 0.45, r"Risk  $V_t/t \leq \alpha$", C["blue"], 7)
    badge(ax, 2.35, 4.35, 1.55, 0.45, r"Cost  $\downarrow$", C["sky"], 7.5)

    rounded(ax, (0.5, 2.0), 3.65, 1.95, C["white"], C["blue_d"], lw=1.2)
    txt(ax, 0.7, 3.65, r"Plan space  $P$", size=9.5, color=C["blue_t"])
    # mini pipeline chips
    for i, (lab, xc) in enumerate([("Retrieve", 0.75), ("Rerank", 1.85),
                                   ("Generate", 2.95)]):
        badge(ax, xc, 3.05, 0.95, 0.38, lab, C["blue"], 7)
        if i < 2:
            arrow(ax, (xc + 0.97, 3.24), (xc + 1.08, 3.24), C["blue_d"], 1.4)
    txt(ax, 0.7, 2.65, "Model families: Qwen · DeepSeek · GLM · Gemma · Llama",
        size=7.2, color=C["mute"], weight="normal")
    txt(ax, 0.7, 2.25, r"$A = P \cup \{\bot\}$   decline is an action",
        size=8, color=C["blue_t"])

    # --- col 2 ---
    rounded(ax, (5.0, 5.7), 3.85, 0.9, C["white"], C["mint_d"], lw=1.2)
    txt(ax, 6.92, 6.15, r"$B_{t-1}=\alpha(t-1)-V_{t-1}$",
        size=12, ha="center", color=C["mint_t"])

    rounded(ax, (5.0, 4.15), 3.85, 1.35, "#e8f6e8", C["mint_d"], lw=1.4)
    txt(ax, 5.2, 5.2, r"IF  $B_{t-1}\geq 1-\alpha$", size=9, color=C["mint_t"])
    txt(ax, 5.2, 4.75, "Gate OPEN", size=11, color=C["mint_d"])
    txt(ax, 5.2, 4.4, r"$\rightarrow$  solve Action LP, sample $a\sim w$",
        size=8, color=C["mute"], weight="normal")

    rounded(ax, (5.0, 2.55), 3.85, 1.4, "#fdeceb", C["coral_d"], lw=1.4)
    txt(ax, 5.2, 3.65, r"ELSE  $B_{t-1}<1-\alpha$", size=9, color=C["coral_t"])
    txt(ax, 5.2, 3.2, "Gate CLOSED", size=11, color=C["coral_d"])
    txt(ax, 5.2, 2.8, r"$\rightarrow$  Decline $\bot$  with  $\ell_t=0$",
        size=8.5, color=C["mute"], weight="normal")

    rounded(ax, (5.0, 1.75), 3.85, 0.65, C["mint"], C["mint_d"], lw=1.2)
    txt(ax, 6.92, 2.08,
        r"Pathwise:  $V_t\leq\alpha t$  ·  no $\delta$  ·  any order",
        size=7.8, ha="center", color=C["mint_t"])

    # --- col 3 ---
    steps = [
        (6.35, "Action LP",
         r"$\min_w\sum w_a c_a$  s.t.  $\sum w_a\tilde r_a\leq\alpha$"),
        (5.55, "Optimistic risk",
         r"$\tilde r_a=\hat r_a+\kappa/\sqrt{4n_a}$  (cost device, not certificate)"),
        (4.75, "Sample & execute",
         r"draw $a\sim w$; serve plan or decline $\bot$"),
        (3.95, "Audit & update",
         r"pay $c_a$; if labelled observe $\ell_t$;  $B\leftarrow B+\alpha-\ell_t$"),
        (3.15, "Learn",
         r"update $\hat c_a$ always, $\hat r_a$ when audited"),
    ]
    for y, title, body in steps:
        rounded(ax, (9.7, y - 0.15), 4.0, 0.7, C["white"], C["sand_d"], lw=1.1)
        txt(ax, 9.9, y + 0.32, title, size=8.5, color=C["sand_t"])
        txt(ax, 9.9, y + 0.02, body, size=7.2, color=C["mute"], weight="normal")

    rounded(ax, (9.7, 1.75), 4.0, 0.9, C["yellow"], C["yellow_d"], lw=1.3)
    txt(ax, 11.7, 2.35, r"Regret  $O\!\left(c_{\max}\sqrt{P\log T/T}\right)$",
        size=9.5, ha="center", color=C["yellow_t"])
    txt(ax, 11.7, 1.95, "validity removed from the estimate → aggressive steering",
        size=7, ha="center", color=C["mute"], weight="normal")

    # arrows between columns
    arrow(ax, (4.4, 4.4), (4.75, 4.4), C["blue_d"], 2.6)
    arrow(ax, (9.1, 4.4), (9.45, 4.4), C["mint_d"], 2.6)

    # bottom principles strip
    rounded(ax, (0.25, 0.25), 13.7, 1.1, "#fff0ef", C["coral"], lw=1.5, r=0.03)
    items = [
        (0.55, "Validity by accounting",
         r"Identity $V_t\leq\alpha t$ on every path"),
        (5.0, "Cost by learning",
         "Statistics serves cost alone"),
        (9.3, "Decline as first-class action",
         r"Every $\alpha>0$ feasible; floor unlocked"),
    ]
    for x, t1, t2 in items:
        txt(ax, x, 0.95, t1, size=9.5, color=C["coral_t"])
        txt(ax, x, 0.55, t2, size=7.5, color=C["mute"], weight="normal")

    save(fig, "fig_method_framework")


# ---------------------------------------------------------------------------
# Figure 2 — paradigm contrast (a)(b)(c) / (d)
# ---------------------------------------------------------------------------

def fig_contrast():
    fig, ax = plt.subplots(figsize=(14.2, 8.4), dpi=200)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")

    # top three
    panels = [
        (0.25, C["softb"], C["blue_d"], "(a)  Standard RAG",
         ["$Q$ → Retrieve → Generate → $A$",
          "No contract · no cost control",
          "Always answers; risk unbounded"]),
        (4.85, C["pink"], C["coral_d"], "(b)  Learn-then-Test",
         [r"Calibrate → $\hat r_p+w_{\mathrm{stat}}\leq\alpha$",
          "Pays confidence width forever",
          r"Width grows with $|P|$",
          r"Fails under drift · $\alpha<$floor infeasible"]),
        (9.45, C["lav"], C["lav_d"], "(c)  Online e-process",
         [r"Monitor $E_t$; alarm at $1/\delta$",
          "Recertify with spent budget",
          r"Guarantee is $1-\delta$, not pathwise",
          "Still needs statistical assumptions"]),
    ]
    for x, fc, ec, title, lines in panels:
        rounded(ax, (x, 5.35), 4.35, 2.8, fc, ec, lw=1.8, r=0.04)
        txt(ax, x + 0.2, 7.85, title, size=11, color=_darken(fc, 0.4))
        for i, line in enumerate(lines):
            txt(ax, x + 0.25, 7.35 - i * 0.45, "•  " + line, size=8.2,
                color=C["ink"] if i == 0 else C["mute"],
                weight="bold" if i == 0 else "normal")

    # bottom ours
    rounded(ax, (0.25, 0.35), 13.7, 4.75, C["softg"], C["mint_d"], lw=2.2, r=0.04)
    txt(ax, 0.5, 4.75, "(d)  Ours: Ledger-gated execution",
        size=13, color=C["mint_t"])

    # three substeps
    subs = [
        (0.5, "1  Query & Gate",
         [r"$B_{t-1}=\alpha(t-1)-V_{t-1}$",
          r"Open iff $B_{t-1}\geq 1-\alpha$",
          r"Else force Decline $\bot$ ($\ell=0$)"]),
        (4.9, "2  Optimistic Action LP",
         [r"$\min_w\sum w_a\hat c_a$ s.t. $\sum w_a\tilde r_a\leq\alpha$",
          r"Actions: plans $P\cup\{\bot\}$",
          r"Basic optimum: $\leq 2$ nonzero weights"]),
        (9.3, "3  Execute · Audit · Update",
         [r"Serve $a$ or decline; pay $c_a$",
          r"If audited: observe $\ell_t\in\{0,1\}$",
          r"$B\leftarrow B+\alpha-\ell_t$; update $\hat r,\hat c$"]),
    ]
    for x, title, lines in subs:
        rounded(ax, (x, 2.15), 4.1, 2.3, C["white"], C["mint_d"], lw=1.3, r=0.03)
        txt(ax, x + 0.15, 4.15, title, size=10, color=C["mint_t"])
        for i, line in enumerate(lines):
            txt(ax, x + 0.2, 3.65 - i * 0.4, line, size=8, color=C["ink"],
                weight="normal")

    # right callouts
    callouts = [
        (0.5, r"Pathwise $V_t\leq\alpha t$"),
        (3.7, r"No $\delta$ · no union bound"),
        (7.1, r"Immune to $|P|$"),
        (10.4, r"Every $\alpha>0$ feasible"),
    ]
    for x, lab in callouts:
        badge(ax, x, 1.35, 2.9, 0.55, lab, C["mint"], 7.8)

    txt(ax, 7.1, 0.7,
        "Validity by accounting  ·  Cost by learning  ·  Decline unlocks the risk floor",
        size=9, ha="center", color=C["mint_t"])

    # arrows top → bottom
    for x in (2.4, 7.0, 11.6):
        arrow(ax, (x, 5.35), (x, 5.1), C["mint_d"], 1.6)

    save(fig, "fig_paradigm_contrast")


# ---------------------------------------------------------------------------
# Figure 3 — controller loop
# ---------------------------------------------------------------------------

def fig_controller():
    fig, ax = plt.subplots(figsize=(14.2, 7.8), dpi=200)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")

    # left stream
    rounded(ax, (0.2, 1.7), 2.7, 5.7, "#fff0ef", C["coral"], lw=1.8, r=0.04)
    txt(ax, 0.4, 7.05, "Stream", size=12, color=C["coral_t"])
    txt(ax, 0.4, 6.5, r"Queries $q_t$", size=9, color=C["ink"])
    txt(ax, 0.4, 6.15, r"$t=1,2,\ldots$", size=8, color=C["mute"], weight="normal")

    rounded(ax, (0.4, 4.85), 2.3, 1.0, C["white"], C["coral_d"], lw=1.1)
    txt(ax, 0.55, 5.55, r"Contract $\alpha$", size=9, color=C["coral_t"])
    txt(ax, 0.55, 5.15, r"rate: $V_t\leq\alpha t$", size=7.5,
        color=C["mute"], weight="normal")

    rounded(ax, (0.4, 2.1), 2.3, 2.5, C["white"], C["coral_d"], lw=1.1)
    txt(ax, 0.55, 4.3, r"Plan catalog $P$", size=9, color=C["coral_t"])
    txt(ax, 0.55, 3.85, "plan   cost   risk", size=7.5, color=C["mute"],
        weight="normal")
    for i, (p, c, r) in enumerate([("p1", "c1", "r1"), ("p2", "c2", "r2"),
                                   ("…", "…", "…"), (r"⊥", r"c_⊥", "0")]):
        txt(ax, 0.55, 3.45 - i * 0.32, f"{p}      {c}      {r}", size=7.5,
            color=C["ink"], weight="normal")

    # center controller
    rounded(ax, (3.15, 1.7), 7.7, 5.7, C["softb"], C["blue_d"], lw=2.0, r=0.04)
    txt(ax, 7.0, 7.05, "Ledger-gated Controller", size=13, ha="center",
        color=C["blue_t"])

    # three modules
    mods = [
        (3.4, C["blue"], C["blue_d"], "1  Gate",
         [r"$B_{t-1}\;\geq\;1-\alpha$ ?",
          "Yes → open", "No → closed",
          r"Closed ⇒ force $\bot$"]),
        (5.9, C["cream"], C["sand_d"], "2  Optimiser",
         ["Action LP",
          r"$\min$ cost s.t. risk $\leq\alpha$",
          r"$\tilde r_a=\hat r_a+\kappa/\sqrt{4n_a}$",
          r"$w$ over $P\cup\{\bot\}$"]),
        (8.4, C["mint"], C["mint_d"], "3  Accounting",
         [r"Observe $c_a$ always",
          r"Observe $\ell_t$ if audited",
          r"$B\leftarrow B+\alpha-\ell_t$",
          r"Update $\hat r_a,\hat c_a$"]),
    ]
    for x, fc, ec, title, lines in mods:
        rounded(ax, (x, 3.35), 2.25, 3.3, fc, ec, lw=1.5, r=0.03)
        txt(ax, x + 0.12, 6.35, title, size=10, color=_darken(fc, 0.4))
        for i, line in enumerate(lines):
            txt(ax, x + 0.12, 5.85 - i * 0.45, line, size=7.8,
                color=C["ink"], weight="normal")

    arrow(ax, (5.65, 5.0), (5.9, 5.0), C["blue_d"], 2.0)
    arrow(ax, (8.15, 5.0), (8.4, 5.0), C["sand_d"], 2.0)

    # feedback
    rounded(ax, (3.4, 2.0), 7.2, 1.05, C["white"], C["mint_d"], lw=1.3)
    txt(ax, 7.0, 2.7, r"Feedback: ledger state $B_t$  →  Gate",
        size=9.5, ha="center", color=C["mint_t"])
    txt(ax, 7.0, 2.25,
        r"Theorem: $B_t\geq 0$  $\Rightarrow$  $V_t\leq\alpha t$  on every path",
        size=8, ha="center", color=C["mute"], weight="normal")
    # curved-ish return arrow marker
    arrow(ax, (9.5, 3.35), (4.5, 3.35), C["mint_d"], 1.5)

    # right service
    rounded(ax, (11.1, 1.7), 2.9, 5.7, C["lav"], C["lav_d"], lw=1.8, r=0.04)
    txt(ax, 11.3, 7.05, "Service", size=12, color=C["lav_t"])

    rounded(ax, (11.3, 5.5), 2.5, 1.2, C["white"], C["mint_d"], lw=1.2)
    txt(ax, 12.55, 6.4, "Answer", size=10, ha="center", color=C["mint_d"])
    txt(ax, 12.55, 5.9, "(serve plan)", size=7.5, ha="center",
        color=C["mute"], weight="normal")

    rounded(ax, (11.3, 4.0), 2.5, 1.2, C["white"], C["coral_d"], lw=1.2)
    txt(ax, 12.55, 4.9, r"Decline $\bot$", size=10, ha="center",
        color=C["coral_d"])
    txt(ax, 12.55, 4.4, "(to human / queue)", size=7.5, ha="center",
        color=C["mute"], weight="normal")

    rounded(ax, (11.3, 2.0), 2.5, 1.7, C["white"], C["lav_d"], lw=1.2)
    txt(ax, 11.45, 3.4, "Guarantees", size=9, color=C["lav_t"])
    txt(ax, 11.45, 2.95, r"Pathwise $V_t\leq\alpha t$", size=7.5,
        color=C["ink"], weight="normal")
    txt(ax, 11.45, 2.55, r"No $\delta$ · no $|P|$ price", size=7.5,
        color=C["ink"], weight="normal")
    txt(ax, 11.45, 2.15, r"Cost $\rightarrow$ OPT", size=7.5,
        color=C["ink"], weight="normal")

    arrow(ax, (2.9, 4.6), (3.15, 4.6), C["coral_d"], 2.2)
    arrow(ax, (10.85, 5.0), (11.1, 5.0), C["blue_d"], 2.2)

    # bottom three ingredients
    rounded(ax, (0.2, 0.2), 13.8, 1.25, C["cream"], C["sand_d"], lw=1.5, r=0.03)
    bits = [
        (0.4, "1  Ledger", "removes validity from estimates"),
        (5.0, "2  Decline", r"unlocks every $\alpha>0$ below the floor"),
        (9.5, "3  Randomised LP", "spends the allowance, not a width"),
    ]
    for x, t1, t2 in bits:
        txt(ax, x, 1.05, t1, size=10, color=C["sand_t"])
        txt(ax, x, 0.55, t2, size=8, color=C["mute"], weight="normal")

    save(fig, "fig_controller_loop")


if __name__ == "__main__":
    fig_method()
    fig_contrast()
    fig_controller()
