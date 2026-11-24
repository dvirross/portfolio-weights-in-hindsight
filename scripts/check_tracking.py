"""How close is the daily-reset 3x construction to a real 3x fund?

Compares TQQQ with 3 x QQQ - 2 x rf - 0.95%/yr over five years of IBKR daily closes, published in the public
repository dvirross/leveraged-etf-compounding (data/ibkr-qqq-daily-5y.csv, data/ibkr-tqqq-daily-5y.csv;
regular-hours closes, not adjusted for distributions). Usage:
    python scripts/check_tracking.py /path/to/leveraged-etf-compounding
Writes results/tracking_check.csv.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bts.data import load  # noqa: E402
from bts.simulate import FEE  # noqa: E402


def main(repo: Path) -> None:
    q = pd.read_csv(repo / "data/ibkr-qqq-daily-5y.csv", parse_dates=["date"]).set_index("date").close
    t = pd.read_csv(repo / "data/ibkr-tqqq-daily-5y.csv", parse_dates=["date"]).set_index("date").close
    rf = load()["RF"]
    d = pd.DataFrame({"qqq": q.pct_change(), "tqqq": t.pct_change()}).dropna()
    d = d.loc[: rf.index[-1]]
    d["rf"] = rf.reindex(d.index).fillna(0.0)
    d["built"] = 3 * d.qqq - 2 * d.rf - FEE / 252
    yrs = len(d) / 252
    g = lambda r: (1 + r).prod() ** (1 / yrs) - 1
    diff = d.tqqq - d.built
    out = pd.DataFrame([dict(start=d.index[0].date(), end=d.index[-1].date(), days=len(d),
                             growth_tqqq=g(d.tqqq), growth_built=g(d.built), growth_qqq_price=g(d.qqq),
                             corr_daily=d.tqqq.corr(d.built), mean_daily_diff_bp=diff.mean() * 1e4,
                             tracking_error_annual=diff.std() * np.sqrt(252))])
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results/tracking_check.csv", index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
