import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bts.data import parse_csv_text  # noqa: E402


def test_parse_keeps_first_dated_block_only():
    text = "Some header text\n\n,Mkt-RF,RF\n19260701,  0.10,  0.01\n19260702, -0.20,  0.01\n\n Annual\n,Mkt-RF,RF\n1927, 5, 1\n"
    df = parse_csv_text(text)
    assert list(df.columns) == ["Mkt-RF", "RF"]
    assert len(df) == 2
    assert df.iloc[1]["Mkt-RF"] == -0.002


def test_project_simplex():
    from bts.portfolio import project_simplex
    w = project_simplex(np.array([0.5, 2.0, -1.0]))
    assert np.isclose(w.sum(), 1) and (w >= 0).all()
    assert np.allclose(project_simplex(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5])


def test_wealth_optimum_is_unique_and_beats_corners():
    from bts.portfolio import log_growth, optimise
    rng = np.random.default_rng(1)
    X = rng.normal(0.008, 0.05, size=(240, 4)) + np.array([0.0, 0.002, 0.004, 0.0])
    sols = [optimise(X, "wealth", w0=rng.dirichlet(np.ones(4))) for _ in range(5)]
    assert np.ptp(np.array(sols), 0).max() < 1e-3
    best = log_growth(X, sols[0])
    assert all(best >= log_growth(X, e) - 1e-9 for e in np.eye(4))


def test_original_search_port():
    from bts.original import neighbours, profit, search
    w = np.full(3, 1 / 3)
    nb = neighbours(w, 0.01)
    assert len(nb) == 6 and all(np.isclose(v.sum(), 1) and (v > 0).all() for v in nb)
    R = np.array([[0.10, 0.00, -0.05], [0.05, 0.01, 0.02], [-0.02, 0.01, 0.03]])
    assert np.isclose(profit(R, np.array([1.0, 0, 0]), "simple"), 1.10 * 1.05 * 0.98)
    assert np.isclose(profit(R, np.array([1.0, 0, 0]), "cummulative"), np.mean([1.10, 1.10 * 1.05, 1.10 * 1.05 * 0.98]))
    assert profit(R, np.array([0, 0, 1.0]), "minimize_count_of_losses") == -1
    path = search(R, "simple", steps=50, stochastic_choices=10**9)  # never a random move
    assert path["profit"].iloc[-1] >= path["profit"].iloc[0]
    assert len(path) == 51


def test_leveraged_construction_and_floor():
    from bts.simulate import leveraged
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    r = pd.DataFrame({"a": [0.01, -0.40, 0.10, 0.05]}, index=idx)
    rf = pd.Series(0.0001, index=idx)
    out = leveraged(r, rf, 3.0, fee=0.0)
    assert np.isclose(out.a.iloc[0], 0.03 - 0.0002)
    assert out.a.iloc[1] == -1.0                      # 3 x -40% is floored at -100%
    assert (out.a.iloc[2:] == 0).all()                # and the fund stays at zero


def test_rebalanced_path_matches_monthly_compounding():
    from bts.simulate import rebalanced_path
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0.0005, 0.01, (len(idx), 2)), index=idx, columns=["a", "b"])
    w = np.array([0.3, 0.7])
    v = rebalanced_path(r, w, "monthly")["value"].iloc[-1]
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    assert np.isclose(v, np.prod(1 + m.to_numpy() @ w))
    never = rebalanced_path(r, w, "never")["value"].iloc[-1]
    assert np.isclose(never, (1 + r).prod().to_numpy() @ w)
    daily = rebalanced_path(r, w, "daily")["value"].iloc[-1]
    assert np.isclose(daily, np.prod(1 + r.to_numpy() @ w))


def test_rebalancing_cost():
    from bts.simulate import rebalanced_path
    idx = pd.date_range("2020-01-30", periods=3, freq="B")   # 30 Jan, 31 Jan (month end), 3 Feb
    r = pd.DataFrame({"a": [0.0, 0.10, 0.0], "b": [0.0, 0.0, 0.0]}, index=idx)
    p = rebalanced_path(r, np.array([0.5, 0.5]), "monthly", cost=0.01)
    hold = np.array([0.55, 0.5]) / 1.05
    traded = np.abs(hold - 0.5).sum()
    assert np.isclose(p.turnover.iloc[1], traded)
    assert np.isclose(p.value.iloc[1], 1.05 * (1 - 0.01 * traded))
