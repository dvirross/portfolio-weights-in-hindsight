"""A faithful port of the search in Dvir's Colab notebook "portfolio weights gradient decent.ipynb"
(created 2025-05-27; the parametrised version ran on 2025-05-29/30, the date of the LinkedIn post).

Kept as in the notebook, so the article can show what was actually done:
- prices are sampled every ``cycle`` trading days starting at row 1; returns between samples are combined
  with fixed weights (equivalent to rebalancing every ``cycle`` trading days, no costs);
- objectives: ``"cummulative"`` (the notebook's default: the time average of the wealth curve),
  ``"simple"`` (end wealth), ``"minimize_sum_of_losses"``, ``"minimize_count_of_losses"``;
- one step: every neighbour that adds ``delta`` to one or two weights and renormalises; with probability
  1/``stochastic_choices`` a random neighbour is taken (even if worse), otherwise the best neighbour; there is
  no "stay" option, the search runs a fixed number of steps from equal weights and returns the whole path.

Not kept: the notebook's price field (mid of adjusted High and Low) is a data choice made by the caller, and
the deleted cell that built its 2010-2025 data frame. ``random`` is replaced by a seeded numpy generator.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def period_returns(prices: pd.DataFrame, cycle: int = 30) -> np.ndarray:
    idx = list(range(1, len(prices), cycle))
    return prices.iloc[idx].pct_change().iloc[1:].to_numpy()


def profit(R: np.ndarray, w: np.ndarray, kind: str = "cummulative") -> float:
    p = R @ w
    if kind == "cummulative":
        return float(np.cumprod(1 + p).mean())
    if kind == "simple":
        return float(np.prod(1 + p))
    if kind == "minimize_sum_of_losses":
        return float(p[p < 0].sum())
    if kind == "minimize_count_of_losses":
        return float(-(p < 0).sum())
    raise ValueError(kind)


def neighbours(w: np.ndarray, delta: float) -> list[np.ndarray]:
    out = []
    for k in (1, 2):
        for comb in combinations(range(len(w)), k):
            v = w.copy()
            v[list(comb)] += delta
            out.append(v / v.sum())
    return out


def search(R: np.ndarray, kind: str = "cummulative", steps: int = 200, delta: float = 0.01,
           stochastic_choices: int = 10, seed: int = 0, w0: np.ndarray | None = None) -> pd.DataFrame:
    """Run the notebook's search and return the path: one row per step, weights then ``profit``."""
    rng = np.random.default_rng(seed)
    n = R.shape[1]
    w = np.full(n, 1 / n) if w0 is None else np.asarray(w0, dtype=float)
    path = [(w, profit(R, w, kind))]
    for _ in range(steps):
        cands = neighbours(path[-1][0], delta)
        if rng.integers(stochastic_choices) == 1:
            c = cands[rng.integers(len(cands))]
            path.append((c, profit(R, c, kind)))
        else:
            path.append(max(((c, profit(R, c, kind)) for c in cands), key=lambda x: x[1]))
    W = pd.DataFrame([p[0] for p in path])
    W["profit"] = [p[1] for p in path]
    return W
