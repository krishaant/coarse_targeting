"""Coarse targeting in social networks: a small simulation.

Opinion dynamics (DeGroot) with a single broadcast message and
person-specific exposure weights.

    x(t+1) = (I - B) A x(t) + B m

    x : vector of opinions in [0, 1]
    A : row-stochastic listening matrix (who listens to whom)
    B : diag(b), b_i = how much of i's update comes from the broadcast
    m : the planner's message (fixed, same for everyone)

The planner picks b once, subject to a budget, to minimise

    J = sum_t delta^t * || x(t) - m ||^2
"""
import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------- dynamics
def simulate(A, x0, b, m=1.0, T=20):
    """Run the dynamics forward. Returns the (T+1, n) path of opinions."""
    b = np.asarray(b, dtype=float)
    x = np.asarray(x0, dtype=float).copy()
    path = [x.copy()]
    for _ in range(T):
        x = (1.0 - b) * (A @ x) + b * m
        path.append(x.copy())
    return np.array(path)


def loss(A, x0, b, m=1.0, T=20, delta=0.95):
    """Discounted sum of squared distance from the agenda."""
    path = simulate(A, x0, b, m=m, T=T)[1:]          # skip t=0
    w = delta ** np.arange(1, T + 1)
    return float((w * ((path - m) ** 2).sum(axis=1)).sum())


# ---------------------------------------------------------------- networks
def row_normalise(W):
    return W / W.sum(axis=1, keepdims=True)


def line3():
    """Ana - Ben - Cal. Ana is the extremist, Ben the moderate bridge."""
    W = np.array([[1., 1., 0.],
                  [1., 1., 1.],
                  [0., 1., 1.]])
    return row_normalise(W), np.array([0.0, 0.5, 1.0]), ["Ana", "Ben", "Cal"]


def two_clusters(n_each=5, bridge_strength=1.0, seed=0):
    """Two dense clusters joined by one bridge node.

    Cluster L starts opposed to the message, cluster R starts close to it,
    and the bridge sits in the middle with a moderate opinion.
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
    # bridge connects to one node in each cluster
    for anchor in (0, n_each):
        W[bridge, anchor] = bridge_strength
        W[anchor, bridge] = bridge_strength
    A = row_normalise(W)
    x0 = np.concatenate([rng.uniform(0.00, 0.15, n_each),    # opposed cluster
                         rng.uniform(0.75, 0.95, n_each),    # friendly cluster
                         [0.5]])                             # the bridge
    labels = [f"L{i+1}" for i in range(n_each)] + \
             [f"R{i+1}" for i in range(n_each)] + ["BRIDGE"]
    return A, x0, labels


# ---------------------------------------------------------------- targeting
def optimal_weights(A, x0, budget, m=1.0, T=20, delta=0.95, b_max=0.5):
    """Numerically minimise the discounted loss subject to sum(b) = budget."""
    n = len(x0)
    b_init = np.full(n, budget / n)
    cons = [{"type": "eq", "fun": lambda b: b.sum() - budget}]
    res = minimize(lambda b: loss(A, x0, b, m=m, T=T, delta=delta),
                   b_init, method="SLSQP",
                   bounds=[(0.0, b_max)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    return res.x


def static_rule(x0, budget, m=1.0, b_max=0.5):
    """The one-shot benchmark: exposure proportional to distance from m."""
    gap = np.maximum(m - np.asarray(x0), 0)
    b = budget * gap / gap.sum()
    return np.clip(b, 0, b_max)


def centrality_rule(A, budget, b_max=0.5):
    """Exposure proportional to eigenvector centrality of the listening matrix."""
    vals, vecs = np.linalg.eig(A.T)
    v = np.abs(np.real(vecs[:, np.argmax(np.real(vals))]))
    b = budget * v / v.sum()
    return np.clip(b, 0, b_max)


def eig_centrality(A):
    vals, vecs = np.linalg.eig(A.T)
    v = np.abs(np.real(vecs[:, np.argmax(np.real(vals))]))
    return v / v.sum()
