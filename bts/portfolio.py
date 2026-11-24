"""Constant-weight portfolios rebalanced every month, and weights optimised on past data.

The post (2025-05-29) fixes target weights, rebalances back to them, and searches for the weights that
maximise end-of-period wealth on past data with gradient ascent. With monthly rebalancing, end wealth is
prod(1 + w . r_t), so maximising it is maximising the mean of log(1 + w . r_t), which is concave in w:
on the simplex (weights >= 0, summing to 1) it has a single peak.

Optimisers:
- ``wealth``: exponentiated gradient ascent on mean log(1 + w . r) (stays on the simplex by construction);
- ``sharpe`` and ``minvar``: projected gradient steps onto the simplex. For ``sharpe`` pass returns in excess of
  the risk-free rate (``X - rf[:, None]``): since the weights sum to 1, the portfolio's excess return is then
  ``w . X``, and the objective is the Sharpe ratio of the window.
"""
from __future__ import annotations

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w : w >= 0, sum w = 1} (Duchi et al. 2008)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1), 0.0)


def log_growth(X: np.ndarray, w: np.ndarray) -> float:
    """Mean log growth per period of the rebalanced portfolio with weights w."""
    return float(np.log1p(X @ w).mean())


def optimise(X: np.ndarray, objective: str, w0: np.ndarray | None = None, steps: int = 3000) -> np.ndarray:
    n = X.shape[1]
    w = np.full(n, 1.0 / n) if w0 is None else np.asarray(w0, dtype=float).copy()
    Xc = X - X.mean(0)
    for t in range(steps):
        p = X @ w
        if objective == "wealth":
            g = (X / (1 + p)[:, None]).mean(0)
            w = w * np.exp(0.1 * g / (np.abs(g).max() + 1e-12))
            w /= w.sum()
            continue
        step = 0.05 * 0.995 ** t
        if objective == "sharpe":
            m, s = p.mean(), p.std()
            g = X.mean(0) / s - m * ((p - m)[:, None] * Xc).mean(0) / s ** 3
            w = project_simplex(w + step * g / (np.abs(g).max() + 1e-12))
        elif objective == "minvar":
            g = 2 * ((p - p.mean())[:, None] * Xc).mean(0)
            w = project_simplex(w - step * g / (np.abs(g).max() + 1e-12))
        else:
            raise ValueError(objective)
    return w
