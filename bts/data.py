"""Download and parse Kenneth R. French Data Library daily files.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
Values in the files are percent returns; they are converted to decimals here.
The raw files are not redistributed; run ``scripts/download_data.py``.
"""
from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "factors": "F-F_Research_Data_Factors_daily_CSV.zip",
    "momentum": "F-F_Momentum_Factor_daily_CSV.zip",
    "industries10": "10_Industry_Portfolios_daily_CSV.zip",
}
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(name: str, dest: Path = RAW) -> Path:
    url = BASE + FILES[name]
    data = urlopen(Request(url, headers={"User-Agent": "research-download"}), timeout=120).read()
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / FILES[name]
    out.write_bytes(data)
    (dest / (FILES[name] + ".sha256")).write_text(hashlib.sha256(data).hexdigest() + "\n")
    return out


def parse_csv_text(text: str) -> pd.DataFrame:
    """Parse the first daily table of a French CSV file.

    Free-text lines come first, then a header row that starts with a comma, then ``YYYYMMDD,...`` rows,
    then possibly a footer or further tables. Only the first block of dated rows is kept.
    """
    lines = text.splitlines()
    header_idx = next(i for i, l in enumerate(lines) if l.startswith(",") and len(l.split(",")) > 1)
    cols = ["date"] + [c.strip() for c in lines[header_idx].split(",")[1:]]
    rows = []
    for l in lines[header_idx + 1:]:
        parts = [x.strip() for x in l.split(",")]
        if len(parts) == len(cols) and len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows:
            break
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date").astype(float) / 100.0


def parse_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        return parse_csv_text(z.read(z.namelist()[0]).decode("latin-1"))


def load(raw: Path = RAW) -> pd.DataFrame:
    """Daily market, risk-free, SMB, HML and momentum returns on the common dates.

    ``MKT`` is Mkt-RF + RF: the value-weighted CRSP market return including dividends.
    """
    f = parse_zip(raw / FILES["factors"])
    m = parse_zip(raw / FILES["momentum"])
    m.columns = [c.strip().upper() for c in m.columns]
    df = f.join(m, how="inner")
    df["MKT"] = df["Mkt-RF"] + df["RF"]
    return df


def load_industries(raw: Path = RAW) -> pd.DataFrame:
    """Daily value-weighted returns of the 10 industry portfolios (first table of the file)."""
    df = parse_zip(raw / FILES["industries10"])
    if (df <= -0.9999).any().any():
        raise ValueError("missing-data codes (-99.99) found in the industry file")
    return df


def monthly(daily: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compound daily returns into calendar-month returns."""
    return (1 + daily).groupby(daily.index.to_period("M")).prod() - 1
