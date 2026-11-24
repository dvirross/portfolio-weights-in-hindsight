# The Best Portfolio, Had We Known the Future

Reproducible analysis for the article [**The Best Portfolio, Had We Known the Future**](https://dvirross.com/quant-research/notes/the-best-portfolio-had-we-known-the-future) on dvirross.com ([Hebrew version](https://dvirross.com/he/quant-research/notes/the-best-portfolio-had-we-known-the-future)).

The article starts from a simple strategy: fix a weight for each part of a portfolio and rebalance back to those weights from time to time. It then searches past data for the best weights, and asks what those weights do in the years the search did not see. Every number and figure in the article comes from this repository.

## What is here

| Path | Content |
|---|---|
| `bts/data.py` | Download and parse the Kenneth R. French daily files (10 industry portfolios, market, risk-free rate). |
| `bts/simulate.py` | Daily-reset 3× series built from real daily returns (3 × return − 2 × risk-free rate − 0.95% a year, floored at −100%), and a daily simulator of a portfolio rebalanced daily, monthly, quarterly, yearly, on a relative band, or never, with costs on the value traded. |
| `bts/original.py` | The weight search from the author's 2025 notebook: from equal weights, each step tries adding 1% to one or two weights (then renormalising) and moves to the best neighbour; one step in ten goes to a random neighbour. Four objectives: end wealth, time-average wealth, sum of losses, count of losses. |
| `bts/portfolio.py` | Exponentiated-gradient ascent on mean log growth (maximise end wealth), and projected gradient for the Sharpe ratio (on returns above the risk-free rate) and for minimum variance. |
| `scripts/download_data.py` | Downloads the raw files into `data/raw/` and records their SHA-256. The raw files are not stored here; the recorded hashes are in `data/raw/*.sha256`. |
| `scripts/run_analysis.py` | Produces every table in `results/` (about a minute). |
| `scripts/check_tracking.py` | Compares the 3× construction with the real TQQQ, using the IBKR QQQ and TQQQ daily closes of [dvirross/leveraged-etf-compounding](https://github.com/dvirross/leveraged-etf-compounding). |
| `scripts/make_figures.py` | The article's Plotly figures (JSON specs) and static PNG/WebP versions; `figures/CAPTIONS.md` holds captions and alt text. |
| `tests/` | Unit tests of the parser, the simplex projection, the single peak of end wealth, the port of the 2025 search, the 3× construction and the rebalancing simulator. |

## Run

```
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_analysis.py
pytest
python scripts/check_tracking.py /path/to/leveraged-etf-compounding
BROWSER_PATH=/path/to/chrome python scripts/make_figures.py   # static export needs a Chromium for kaleido
```

`make_figures.py` writes the Plotly specs and WebP files to `figures/site/` when run outside the website repository.

## Main settings

- Assets: the 10 value-weighted US industry portfolios of the Kenneth R. French Data Library, daily from July 1926 (monthly returns from 1927 for the walk-forward test), and their 3× versions.
- Walk-forward test: at the start of every year from 1937 to 2025, weights are chosen from the previous 120 months, held for 12 months with monthly rebalancing, and chosen again. These settings were fixed before any out-of-sample result was looked at.
- Local-peak experiment: four industries (Technology, Health, Energy, Utilities), 1995–2004 and 2010–2019, rebalancing every 30 trading days, steps of 1%, 600 steps, eight random starts, with and without random steps.
- No taxes. Trading costs only where stated (0.1% and 0.5% of the amount traded).

## Data

The data are from the [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), which builds them from CRSP. They are downloaded by the script and not redistributed here. The 3× series are constructions, not funds that could be bought in 1930; over the five years where both exist the construction grew about 1.2 percentage points a year faster than the real TQQQ (`results/tracking_check.csv`).

## Licence

Code: MIT (see `LICENSE`). Result tables and figures: free to reuse with attribution to the article.

Nothing here is investment advice.
