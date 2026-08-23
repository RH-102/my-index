from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


BASE_DATE = "2026-08-12"
BENCHMARKS = {
    "Nasdaq100": "^NDX",
    "SP500": "^GSPC",
}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "benchmark_history.csv"


def extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series:
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(0):
            close = raw[ticker]["Close"]
        elif "Close" in raw.columns.get_level_values(0):
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            raise RuntimeError(f"Close column not found for {ticker}")
    else:
        if "Close" not in raw.columns:
            raise RuntimeError(f"Close column not found for {ticker}")
        close = raw["Close"]

    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index, errors="coerce").tz_localize(None)
    close = close[~close.index.isna()].sort_index()

    if close.empty:
        raise RuntimeError(f"No valid closing prices for {ticker}")

    return close


def download_close(ticker: str) -> pd.Series:
    end_date = (datetime.now().date() + timedelta(days=1)).isoformat()
    raw = yf.download(
        ticker,
        start=BASE_DATE,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
        multi_level_index=True,
    )
    return extract_close(raw, ticker)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    closes = {
        name: download_close(ticker)
        for name, ticker in BENCHMARKS.items()
    }

    base_ts = pd.Timestamp(BASE_DATE)
    for name, series in closes.items():
        if base_ts not in series.index:
            raise RuntimeError(
                f"{BASE_DATE} closing price is missing for {name}; "
                "benchmark normalization was not saved."
            )

    common_dates = set(closes["Nasdaq100"].index).intersection(
        set(closes["SP500"].index)
    )
    common_dates = sorted(date for date in common_dates if date >= base_ts)

    if not common_dates:
        raise RuntimeError("No common benchmark trading dates found")

    ndx_base = float(closes["Nasdaq100"].loc[base_ts])
    spx_base = float(closes["SP500"].loc[base_ts])
    updated_at = datetime.now().isoformat(timespec="seconds")

    rows = []
    for date in common_dates:
        ndx_close = float(closes["Nasdaq100"].loc[date])
        spx_close = float(closes["SP500"].loc[date])
        ndx_level = ndx_close / ndx_base * 100.0
        spx_level = spx_close / spx_base * 100.0

        rows.append({
            "Date": date.date().isoformat(),
            "Nasdaq100Close": ndx_close,
            "Nasdaq100Level": ndx_level,
            "Nasdaq100ReturnPct": ndx_level - 100.0,
            "SP500Close": spx_close,
            "SP500Level": spx_level,
            "SP500ReturnPct": spx_level - 100.0,
            "BaseDate": BASE_DATE,
            "UpdatedAt": updated_at,
        })

    out = pd.DataFrame(rows).sort_values("Date")
    out.to_csv(OUTPUT_FILE, index=False)

    latest = out.iloc[-1]
    print("SUCCESS")
    print(f"Benchmark base date: {BASE_DATE} = 100")
    print(f"Latest benchmark date: {latest['Date']}")
    print(f"Nasdaq-100 return: {latest['Nasdaq100ReturnPct']:+.2f}%")
    print(f"S&P 500 return: {latest['SP500ReturnPct']:+.2f}%")


if __name__ == "__main__":
    main()
