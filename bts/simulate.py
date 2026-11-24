"""Leveraged series built from real daily returns, and portfolios rebalanced on a schedule.

Daily-reset leverage (the construction of a leveraged ETF, as in the risk-and-ruin article):
    r_L(t) = L * r(t) - (L - 1) * rf(t) - fee / 252,
floored at -100% (a fund cannot lose more than everything; after a wipe-out it stays at zero).

Portfolio of several funds with target weights w, rebalanced back to w at the end of selected days
(``schedule``: "daily", "monthly", "quarterly", "annual", "never", or a relative band such as 0.25 meaning
"rebalance at a month end when some weight is more than 25% away from its target, relative").
A rebalance costs ``cost`` times the traded value (sum of absolute weight changes times portfolio value).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEE = 0.0095      # annual management fee typical of 3x funds


def leveraged(r: pd.DataFrame, rf: pd.Series, L: float = 3.0, fee: float = FEE) -> pd.DataFrame:
    rf = rf.reindex(r.index).fillna(0.0)
    out = r.mul(L).sub(rf * (L - 1), axis=0) - fee / 252
    out = out.clip(lower=-1.0)
    dead = (out <= -1.0).cumsum().shift(1, fill_value=0) > 0
    return out.mask(dead, 0.0)


def _is_rebalance_day(index: pd.DatetimeIndex, schedule) -> np.ndarray:
    if schedule == "daily":
        return np.ones(len(index), bool)
    if schedule == "never":
        return np.zeros(len(index), bool)
    p = index.to_period("M")
    month_end = np.r_[p[1:] != p[:-1], True]
    if schedule in ("monthly",) or isinstance(schedule, float):
        return month_end
    if schedule == "quarterly":
        return month_end & np.isin(index.month, [3, 6, 9, 12])
    if schedule == "annual":
        return month_end & (index.month == 12)
    raise ValueError(schedule)


def rebalanced_path(r: pd.DataFrame, w: np.ndarray, schedule="monthly", cost: float = 0.0) -> pd.DataFrame:
    """Daily portfolio value (starting at 1) and turnover, holding weights that drift between rebalances."""
    R = r.to_numpy()
    w = np.asarray(w, float)
    flag = _is_rebalance_day(r.index, schedule)
    band = schedule if isinstance(schedule, float) else None
    hold = w.copy()          # current weights (fractions of portfolio value)
    v = 1.0
    val = np.empty(len(R))
    turn = np.zeros(len(R))
    for t in range(len(R)):
        g = hold * (1 + R[t])
        growth = g.sum()
        v *= growth
        hold = g / growth if growth > 0 else w.copy()
        if flag[t] and (band is None or np.any(np.abs(hold - w) > band * w)):
            traded = np.abs(hold - w).sum()
            turn[t] = traded
            v *= 1 - cost * traded
            hold = w.copy()
        val[t] = v
    return pd.DataFrame({"value": val, "turnover": turn}, index=r.index)
