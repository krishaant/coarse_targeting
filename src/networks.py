"""Test networks for the coarse targeting simulation."""
import numpy as np

from targeting import row_normalise


def two_camps(n_each=5, bridge_links=3, loud_hub=True, seed=1):
    """Two dense camps joined by a single bridge node.

    Camp L starts opposed to the message and contains LOUD, who is both the
    most extreme member and the best connected one inside that camp.
    Camp R already agrees. BRIDGE sits between them with a middling opinion
    and only a handful of connections.

    Returns (A, x0, labels) where A is a row-stochastic listening matrix.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_each + 1
    bridge = 2 * n_each
    W = np.eye(n)

    for lo, hi in [(0, n_each), (n_each, 2 * n_each)]:
        for i in range(lo, hi):
            for j in range(lo, hi):
                if i != j:
                    W[i, j] = 1.0

    if loud_hub:                              # everyone in L over-weights LOUD
        W[1:n_each, 0] = 3.0

    for a in range(1, 1 + bridge_links):                  # BRIDGE <-> camp L
        W[bridge, a] = 1.0
        W[a, bridge] = 1.0
    for a in range(n_each, n_each + bridge_links):        # BRIDGE <-> camp R
        W[bridge, a] = 1.0
        W[a, bridge] = 1.0

    A = row_normalise(W)
    x0 = np.concatenate([[0.0],
                         rng.uniform(0.05, 0.20, n_each - 1),
                         rng.uniform(0.80, 0.95, n_each),
                         [0.5]])
    labels = (["LOUD"]
              + [f"L{i + 2}" for i in range(n_each - 1)]
              + [f"R{i + 1}" for i in range(n_each)]
              + ["BRIDGE"])
    return A, x0, labels


if __name__ == "__main__":
    from targeting import (loss, optimal_weights, static_rule,
                           centrality_rule, eig_centrality)

    A, x0, labels = two_camps()
    n, budget, T, delta = len(x0), 1.0, 25, 0.95
    b_opt = optimal_weights(A, x0, budget, T=T, delta=delta)
    b_stat = static_rule(x0, budget)
    b_cent = centrality_rule(A, budget)
    cent = eig_centrality(A)

    print("total distance from the message (lower is better)")
    for name, b in [("optimal", b_opt), ("distance rule", b_stat),
                    ("centrality rule", b_cent),
                    ("spread evenly", np.full(n, budget / n))]:
        print(f"  {name:<16}{loss(A, x0, b, T=T, delta=delta):7.3f}")

    print(f"\n{'node':8}{'x0':>6}{'cent':>7}{'b_opt':>7}{'b_dist':>8}")
    for i, lab in enumerate(labels):
        print(f"{lab:8}{x0[i]:6.2f}{cent[i]:7.3f}{b_opt[i]:7.3f}{b_stat[i]:8.3f}")
