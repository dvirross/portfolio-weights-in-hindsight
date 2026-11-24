"""Download the Kenneth R. French daily files into data/raw/ and record their SHA-256.

Run from the package root:  python scripts/download_data.py
The raw files are not committed (see .gitignore).
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bts.data import FILES, download, load  # noqa: E402

if __name__ == "__main__":
    for name in FILES:
        print("downloaded", download(name).name)
    df = load()
    print(f"{len(df)} common days, {df.index[0].date()} to {df.index[-1].date()}")
