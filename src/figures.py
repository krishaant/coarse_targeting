import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from pathlib import Path
from targeting import *
from networks import two_camps

BLUE, ORANGE, TEAL = "#1f6feb", "#d9730d", "#0f9b8e"
INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e3e3e0"
SURF = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 150,
})

def tidy(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

# ---------------------------------------------------- Fig 1: opinion paths
OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

A3, x3, lab3 = line3()
T = 20
policies = {"All exposure on Ana (the extremist)": np.array([0.3, 0., 0.]),
            "All exposure on Ben (the bridge)":    np.array([0., 0.3, 0.])}
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
for ax, (name, b) in zip(axes, policies.items()):
    path = simulate(A3, x3, b, T=T)
    for i, (lab, c) in enumerate(zip(lab3, [ORANGE, TEAL, BLUE])):
        ax.plot(path[:, i], color=c, lw=2, label=lab if ax is axes[0] else None)
    ax.axhline(1.0, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("round"); ax.set_xlim(0, T); ax.set_ylim(-0.03, 1.08)
    tot = loss(A3, x3, b, T=T, delta=1.0)
    ax.annotate(f"total distance from the message\nover {T} rounds: {tot:.2f}",
                (0.97, 0.06), xycoords="axes fraction", ha="right", fontsize=9,
                color=INK)
    tidy(ax)
axes[0].set_ylabel("opinion (1 = agrees with the message)")
axes[0].legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 0.22),
               fontsize=9, ncol=3, handlelength=1.2, columnspacing=1.0)
fig.suptitle("Same budget, two targets. The bridge wins on the long run.",
             fontsize=12, fontweight="bold", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "fig1_paths.png", bbox_inches="tight")

# ---------------------------------------------------- Fig 2: horizon crossover
Ts = np.arange(1, 41)
L_ext = [loss(A3, x3, policies["All exposure on Ana (the extremist)"], T=t, delta=1.0) for t in Ts]
L_brd = [loss(A3, x3, policies["All exposure on Ben (the bridge)"], T=t, delta=1.0) for t in Ts]
cross = Ts[np.argmax(np.array(L_brd) < np.array(L_ext))]
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(Ts, L_ext, color=ORANGE, lw=2)
ax.plot(Ts, L_brd, color=TEAL, lw=2)
ax.annotate("target the extremist", (Ts[-1], L_ext[-1]), xytext=(-6, 8),
            textcoords="offset points", color=ORANGE, fontweight="bold", ha="right")
ax.annotate("target the bridge", (Ts[-1], L_brd[-1]), xytext=(-6, -14),
            textcoords="offset points", color=TEAL, fontweight="bold", ha="right")
ax.axvline(cross, color=MUTED, lw=1, ls=(0, (4, 3)))
ax.annotate(f"the ranking flips\nat round {cross}", (cross, max(L_ext) * 0.42),
            xytext=(10, 0), textcoords="offset points", color=INK, fontsize=9)
ax.set_xlabel("rounds the planner cares about")
ax.set_ylabel("total distance from the message\n(lower is better)")
ax.set_title("Short horizon: hit the extremist. Long horizon: hit the bridge.")
tidy(ax); fig.tight_layout(); fig.savefig(OUT / "fig2_crossover.png", bbox_inches="tight")

# ---------------------------------------------------- bigger network
A, x0, lab = two_camps()
n = len(x0); budget, Tb, delta = 1.0, 25, 0.95
b_opt = optimal_weights(A, x0, budget, T=Tb, delta=delta)
b_stat = static_rule(x0, budget)
b_cent = centrality_rule(A, budget)
b_unif = np.full(n, budget / n)
losses = {"optimal": loss(A, x0, b_opt, T=Tb, delta=delta),
          "distance rule": loss(A, x0, b_stat, T=Tb, delta=delta),
          "centrality rule": loss(A, x0, b_cent, T=Tb, delta=delta),
          "spread evenly": loss(A, x0, b_unif, T=Tb, delta=delta)}

# Fig 3: network drawn with size = optimal exposure, fill = initial opinion
G = nx.from_numpy_array((A + A.T > 0).astype(int) - np.eye(n, dtype=int))
pos = nx.spring_layout(G, seed=7, k=0.9)
fig, ax = plt.subplots(figsize=(6.6, 5.0))
nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GRID, width=1.2)
nodes = nx.draw_networkx_nodes(
    G, pos, ax=ax, node_color=x0, cmap="Blues", vmin=0, vmax=1,
    node_size=120 + 2400 * b_opt, edgecolors="#9a9a9a", linewidths=1.0)
for i, l in enumerate(lab):
    ax.annotate(l, pos[i], xytext=(0, -17), textcoords="offset points",
                ha="center", fontsize=8, color=INK,
                fontweight="bold" if b_opt[i] > 0.05 else "normal")
cb = fig.colorbar(nodes, ax=ax, shrink=0.72)
cb.set_label("starting opinion (dark = already agrees)", fontsize=9)
cb.outline.set_visible(False)
ax.set_title("Optimal exposure (circle size) on a network with two camps")
ax.axis("off"); fig.tight_layout(); fig.savefig(OUT / "fig3_network.png", bbox_inches="tight")

# Fig 4: per-node comparison + total loss
fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.4, 4.2),
                               gridspec_kw={"width_ratios": [1.5, 1]})
y = np.arange(n)[::-1]
axL.hlines(y, np.minimum(b_stat, b_opt), np.maximum(b_stat, b_opt), color=GRID, lw=2)
axL.scatter(b_stat, y, color=ORANGE, s=52, zorder=3, edgecolors=SURF, linewidths=2)
axL.scatter(b_opt, y, color=BLUE, s=52, zorder=3, edgecolors=SURF, linewidths=2)
axL.set_yticks(y); axL.set_yticklabels(lab, fontsize=9)
axL.set_xlabel("exposure weight")
axL.set_title("Where the budget actually goes")
axL.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="optimal"),
                    Line2D([], [], marker="o", ls="", color=ORANGE, label="distance rule")],
           frameon=False, loc="lower right", fontsize=9)
axL.grid(axis="x", color=GRID, lw=0.8); axL.set_axisbelow(True)
names = list(losses); vals = [losses[k] for k in names]
bars = axR.bar(names, vals, width=0.62,
               color=[BLUE] + ["#c9c9c4"] * 3)
for bar, v in zip(bars, vals):
    axR.annotate(f"{v:.1f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                 textcoords="offset points", ha="center", fontsize=9, color=INK)
axR.set_ylabel("total distance from the message\n(lower is better)")
axR.set_title("How much the naive rules cost")
axR.tick_params(axis="x", labelrotation=20)
tidy(axR); fig.tight_layout(); fig.savefig(OUT / "fig4_policies.png", bbox_inches="tight")

print("crossover round:", cross)
print({k: round(v, 2) for k, v in losses.items()})
print("bridge weight: optimal %.3f  distance rule %.3f  centrality %.3f"
      % (b_opt[-1], b_stat[-1], b_cent[-1]))
print("budget spent on the friendly camp: optimal %.1f%%  distance rule %.1f%%"
      % (100 * b_opt[5:10].sum() / budget, 100 * b_stat[5:10].sum() / budget))
