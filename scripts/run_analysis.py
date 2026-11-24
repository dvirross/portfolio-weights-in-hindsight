"""All results for the article "The Best Portfolio, Had We Known the Future".

Assets: the 10 Kenneth R. French value-weighted industry portfolios (1x) and daily-reset 3x versions built from
their real daily returns (bts.simulate.leveraged). Writes CSV files to results/ and prints the numbers quoted in
the article. Runs in a few minutes.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bts.data import load, load_industries, monthly  # noqa: E402
from bts.original import profit as nb_profit, search  # noqa: E402
from bts.portfolio import optimise  # noqa: E402
from bts.simulate import leveraged, rebalanced_path  # noqa: E402

RES = ROOT / "results"
WINDOW, HOLD = 120, 12
SEP = "-" * 72


def ann_growth(p) -> float:
    p = np.asarray(p, float)
    w = np.prod(1 + p)
    return float(w ** (12 / len(p)) - 1) if w > 0 else -1.0


def summary(p, rf) -> dict:
    p = np.asarray(p, float)
    wealth = np.cumprod(1 + p)
    e = p - rf
    return dict(growth=ann_growth(p), sharpe=float(e.mean() / e.std(ddof=1) * np.sqrt(12)),
                vol=float(p.std(ddof=1) * np.sqrt(12)),
                max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()), end_wealth=float(wealth[-1]))


def main() -> None:
    RES.mkdir(exist_ok=True)
    f = load()
    ind = load_industries()
    rf_d = f["RF"].reindex(ind.index).fillna(0.0)
    last = f.index[-1]
    ind = ind.loc[:last]
    rf_d = rf_d.loc[:last]
    lev = leveraged(ind, rf_d, 3.0)
    full_months = ind.index.to_period("M").unique()
    if not ind.index[-1].is_month_end and (ind.index[-1] + pd.offsets.BDay(1)).month == ind.index[-1].month:
        full_months = full_months[:-1]
    M1 = monthly(ind).loc["1927-01":full_months[-1]]
    M3 = monthly(lev).loc[M1.index]
    rf_m = monthly(f["RF"]).loc[M1.index].to_numpy()
    mkt_m = monthly(f["MKT"]).loc[M1.index].to_numpy()
    cols = list(M1.columns)
    print(f"data: {ind.index[0].date()} to {ind.index[-1].date()}; monthly sample {M1.index[0]} to {M1.index[-1]}")
    print("3x worst day per industry:", {c: round(float(lev[c].min()), 3) for c in cols})
    print("3x wiped out:", [c for c in cols if (lev[c] <= -1).any()])

    # The tracking check of the 3x construction against TQQQ is scripts/check_tracking.py.

    # ---- 1. two-asset hills (1x, monthly rebalancing, full sample) -----------------------------------------------
    rows = []
    for a, b in [("Hlth", "Enrgy"), ("HiTec", "Utils")]:
        X = M1[[a, b]].to_numpy()
        for w in np.round(np.linspace(0, 1, 101), 2):
            wv = np.array([w, 1 - w])
            bh = (np.cumprod(1 + X, 0)[-1] @ wv) ** (12 / len(X)) - 1
            rows.append(dict(pair=f"{a}-{b}", w_first=w, growth=ann_growth(X @ wv), growth_never_rebalanced=bh))
    hill = pd.DataFrame(rows)
    hill.to_csv(RES / "hill_pairs.csv", index=False)
    print(SEP)
    for k, g in hill.groupby("pair", sort=False):
        top = g.loc[g.growth.idxmax()]
        print(f"{k}: ends {g.growth.iloc[0]:.4f} / {g.growth.iloc[-1]:.4f}; 50/50 {g.growth.iloc[50]:.4f}; "
              f"peak w={top.w_first:.2f} growth {top.growth:.4f}; 50/50 never rebalanced {g.growth_never_rebalanced.iloc[50]:.4f}")

    # ---- 2. three-asset simplex (1x, full sample), and search paths -----------------------------------------------
    tri = ["Hlth", "Enrgy", "Utils"]
    X = M1[tri].to_numpy()
    grid = []
    step = 0.02
    for i in np.arange(0, 1 + 1e-9, step):
        for j in np.arange(0, 1 - i + 1e-9, step):
            w = np.array([i, j, max(0.0, 1 - i - j)])
            grid.append(dict(w_Hlth=round(i, 2), w_Enrgy=round(j, 2), w_Utils=round(w[2], 2),
                             growth=ann_growth(X @ w), time_average=nb_profit(X, w, "cummulative"),
                             losing_months=-nb_profit(X, w, "minimize_count_of_losses")))
    G = pd.DataFrame(grid)
    G.to_csv(RES / "simplex_grid.csv", index=False)
    paths = []
    for kind in ("simple", "minimize_count_of_losses"):
        for sc, lab in ((10 ** 9, "greedy"), (10, "random 1 in 10")):
            for s, w0 in enumerate([None, np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.1, 0.8])]):
                P = search(X, kind, steps=400, delta=0.01, stochastic_choices=sc, seed=s, w0=w0)
                P.columns = [f"w_{c}" for c in tri] + ["objective"]
                P["kind"], P["mode"], P["start"], P["step"] = kind, lab, s, range(len(P))
                paths.append(P)
    paths = pd.concat(paths)
    paths.to_csv(RES / "simplex_paths.csv", index=False)
    top = G.loc[G.growth.idxmax()]
    print(SEP)
    print(f"simplex {tri}: peak growth {top.growth:.4f} at {top.w_Hlth:.2f}/{top.w_Enrgy:.2f}/{top.w_Utils:.2f}; "
          f"equal-ish growth {ann_growth(X @ np.full(3, 1 / 3)):.4f}")
    for (kind, mode), g in paths.groupby(["kind", "mode"]):
        ends = g[g.step == g.step.max()]
        best = g.loc[g.groupby("start").objective.idxmax()]
        print(f"  {kind:26s} {mode:15s} best objective by start: {np.round(best.objective.to_numpy(), 4).tolist()}")

    # ---- 2b. local-peak experiment: random starts, greedy vs random moves, four objectives (4 industries) --------
    sub = ["HiTec", "Hlth", "Enrgy", "Utils"]
    rng = np.random.default_rng(7)
    lp = []
    for period in [("1995", "2004"), ("2010", "2019")]:
        Xd = ind.loc[period[0]:period[1], sub]
        Xp = (1 + Xd).cumprod().iloc[list(range(1, len(Xd), 30))].pct_change().iloc[1:].to_numpy()
        starts = [rng.dirichlet(np.ones(4)) for _ in range(8)]
        for kind in ("simple", "cummulative", "minimize_sum_of_losses", "minimize_count_of_losses"):
            for sc, lab in ((10 ** 9, "greedy"), (10, "random 1 in 10")):
                for s, w0 in enumerate(starts):
                    P = search(Xp, kind, steps=600, delta=0.01, stochastic_choices=sc, seed=s, w0=w0)
                    k = int(P.profit.to_numpy().argmax())
                    lp.append(dict(period=f"{period[0]}-{period[1]}", kind=kind, mode=lab, start=s, best=P.profit.iloc[k],
                                   final=P.profit.iloc[-1], **{f"w_{c}": P.iloc[k, i] for i, c in enumerate(sub)}))
    lp = pd.DataFrame(lp)
    lp.to_csv(RES / "local_peaks.csv", index=False)
    print(SEP)
    for (per, kind, mode), g in lp.groupby(["period", "kind", "mode"], sort=False):
        W = g[[f"w_{c}" for c in sub]].to_numpy()
        print(f"{per} {kind:26s} {mode:15s} best {g.best.min():.3f}..{g.best.max():.3f}  final {g.final.min():.3f}..{g.final.max():.3f}  weight spread {np.ptp(W, 0).max():.2f}")

    # ---- 3. rebalanced vs never rebalanced (daily simulation, equal weights) -------------------------------------
    rv = {}
    for lab, R in (("1x", ind), ("3x", lev)):
        R = R.loc["1927-01-01":]
        w = np.full(10, 0.1)
        rv[f"{lab}_monthly"] = rebalanced_path(R, w, "monthly")["value"]
        rv[f"{lab}_never"] = rebalanced_path(R, w, "never")["value"]
    rv = pd.DataFrame(rv)
    rv.resample("ME").last().to_csv(RES / "rebalanced_vs_never.csv")
    print(SEP)
    for lab in ("1x", "3x"):
        a, b = rv[f"{lab}_monthly"].iloc[-1], rv[f"{lab}_never"].iloc[-1]
        yrs = (rv.index[-1] - rv.index[0]).days / 365.25
        print(f"{lab}: monthly x{a:,.0f} ({a ** (1 / yrs) - 1:.4f}/yr), never x{b:,.0f} ({b ** (1 / yrs) - 1:.4f}/yr), ratio {a / b:.2f}")
    nev = rebalanced_path(lev.loc["1927-01-01":], np.full(10, 0.1), "never")
    print("3x never rebalanced: weights at end",
          {c: round(float(x), 3) for c, x in zip(cols, (1 + lev.loc["1927-01-01":]).prod().to_numpy() / (1 + lev.loc["1927-01-01":]).prod().sum())})

    # ---- 4. walk-forward: weights chosen on 10 years, held 12 months -----------------------------------------------
    wf_rows, wf_w = [], []
    for lab, M in (("1x", M1), ("3x", M3)):
        R = M.to_numpy()
        n = R.shape[1]
        for s in range(WINDOW, len(R) - HOLD + 1, HOLD):
            X, Y = R[s - WINDOW:s], R[s:s + HOLD]
            w_end = optimise(X, "wealth")
            pt = search(X, "cummulative", steps=300, delta=0.01, stochastic_choices=10, seed=s)
            w_avg = pt.iloc[int(pt.profit.to_numpy().argmax()), :-1].to_numpy()
            # Sharpe ratio on returns above the risk-free rate of the same months (weights sum to 1, so the
            # portfolio's excess return is w . (r - rf)).
            X_ex = X - rf_m[s - WINDOW:s, None]
            W = {"end wealth": w_end, "time-average wealth (notebook default)": w_avg,
                 "Sharpe ratio": optimise(X_ex, "sharpe", steps=1500), "minimum variance": optimise(X, "minvar", steps=1500),
                 "equal weights": np.full(n, 1 / n), "half end wealth, half equal": 0.5 * w_end + 0.5 / n}
            for k, w in W.items():
                wf_w.append(dict(assets=lab, start=str(M.index[s]), objective=k, in_sample_growth=ann_growth(X @ w),
                                 next_year=ann_growth(Y @ w), **{c: w[i] for i, c in enumerate(cols)}))
                for j in range(HOLD):
                    wf_rows.append(dict(assets=lab, month=str(M.index[s + j]), objective=k, ret=float(Y[j] @ w)))
    wf = pd.DataFrame(wf_rows)
    ww = pd.DataFrame(wf_w)
    wf.to_csv(RES / "walkforward_returns.csv", index=False)
    ww.to_csv(RES / "walkforward_weights.csv", index=False)
    months = pd.PeriodIndex(wf.month.unique(), freq="M")
    pos = M1.index.get_indexer(months)
    rf_o, mkt_o = rf_m[pos], mkt_m[pos]
    table = []
    for (lab, k), g in wf.groupby(["assets", "objective"], sort=False):
        ins = ww[(ww.assets == lab) & (ww.objective == k)]
        big = ins[cols].max(axis=1)
        table.append(dict(assets=lab, objective=k, in_sample_growth=ins.in_sample_growth.mean(),
                          mean_largest_weight=big.mean(), share_largest_over_0_9=(big > 0.9).mean(),
                          **summary(g.ret.to_numpy(), rf_o)))
    table.append(dict(assets="1x", objective="market", **summary(mkt_o, rf_o)))
    T = pd.DataFrame(table)
    T.to_csv(RES / "walkforward_summary.csv", index=False)
    print(SEP)
    print(f"walk-forward {months[0]} to {months[-1]} ({len(months)} months, {len(months) // 12} windows)")
    with pd.option_context("display.width", 200):
        print(T.round(3).to_string(index=False))
    # rank correlation between in-sample and next-year growth across windows for end wealth
    for lab in ("1x", "3x"):
        e = ww[(ww.assets == lab) & (ww.objective == "end wealth")]
        q = ww[(ww.assets == lab) & (ww.objective == "equal weights")]
        print(f"{lab}: end-wealth beats equal weights next year in {(e.next_year.to_numpy() > q.next_year.to_numpy()).mean():.2f} of windows; "
              f"in-sample beats in {(e.in_sample_growth.to_numpy() > q.in_sample_growth.to_numpy()).mean():.2f}; "
              f"corr(in-sample, next year) {np.corrcoef(e.in_sample_growth, e.next_year)[0, 1]:.2f}")
    e3 = ww[(ww.assets == "3x") & (ww.objective == "end wealth")]
    top = e3[cols].idxmax(axis=1)
    print("3x end-wealth: most frequent top industry:", top.value_counts().head(5).to_dict())

    # ---- 5. rebalancing frequency and costs (equal weights) ---------------------------------------------------------
    fr = []
    for lab, R in (("1x", ind), ("3x", lev)):
        R = R.loc["1927-01-01":]
        for sched in ("daily", "monthly", "quarterly", "annual", 0.25, "never"):
            for cost in (0.0, 0.001, 0.005):
                P = rebalanced_path(R, np.full(10, 0.1), sched, cost)
                yrs = (P.index[-1] - P.index[0]).days / 365.25
                fr.append(dict(assets=lab, schedule=str(sched) if not isinstance(sched, float) else f"band {sched:.0%}",
                               cost=cost, growth=P.value.iloc[-1] ** (1 / yrs) - 1,
                               turnover_per_year=P.turnover.sum() / yrs, rebalances_per_year=(P.turnover > 0).sum() / yrs))
    fr = pd.DataFrame(fr)
    fr.to_csv(RES / "frequency_costs.csv", index=False)
    print(SEP)
    print(fr.pivot_table(index=["assets", "schedule"], columns="cost", values="growth", sort=False).round(4).to_string())
    print(fr[fr.cost == 0][["assets", "schedule", "turnover_per_year", "rebalances_per_year"]].round(2).to_string(index=False))

    # ---- 6. sudden events: equal-weight 3x portfolio, monthly rebalancing vs never, around two crashes -------------
    ev = []
    for name, a, b in (("1987", "1987-08-01", "1988-12-31"), ("2020", "2020-01-01", "2020-12-31")):
        R = lev.loc[a:b]
        m = rebalanced_path(R, np.full(10, 0.1), "monthly")
        n_ = rebalanced_path(R, np.full(10, 0.1), "never")
        one = rebalanced_path(ind.loc[a:b], np.full(10, 0.1), "monthly")
        d = pd.DataFrame({"event": name, "date": R.index, "3x_monthly": m.value.to_numpy(), "3x_never": n_.value.to_numpy(),
                          "1x_monthly": one.value.to_numpy()})
        ev.append(d)
        print(f"{name}: 3x monthly end {m.value.iloc[-1]:.3f} min {m.value.min():.3f}; never end {n_.value.iloc[-1]:.3f} min {n_.value.min():.3f}; "
              f"1x end {one.value.iloc[-1]:.3f} min {one.value.min():.3f}")
    pd.concat(ev).to_csv(RES / "events.csv", index=False)

    # ---- 7. sensitivity: real 3x funds cost more than the construction (TQQQ tracked about 1.2 points/yr below it)
    sens = []
    for extra in (0.0, 0.015):
        L3 = leveraged(ind, rf_d, 3.0, fee=0.0095 + extra)
        Ms = monthly(L3).loc[M1.index].to_numpy()
        for k in ("end wealth", "equal weights", "minimum variance"):
            sub_w = ww[(ww.assets == "3x") & (ww.objective == k)]
            rets = []
            for _, row in sub_w.iterrows():
                s0 = M1.index.get_loc(pd.Period(row.start, "M"))
                rets += list(Ms[s0:s0 + HOLD] @ row[cols].to_numpy(float))
            sens.append(dict(extra_cost=extra, objective=k, growth=ann_growth(rets)))
    sens = pd.DataFrame(sens)
    sens.to_csv(RES / "sensitivity_cost.csv", index=False)
    print(SEP)
    print(sens.round(4).to_string(index=False))
    # the worst single days
    for c, R in (("1x", ind.mean(axis=1)), ("3x", lev.mean(axis=1))):
        print(f"equal-weight {c}: worst day {R.min():.3f} on {R.idxmin().date()}")


if __name__ == "__main__":
    main()
