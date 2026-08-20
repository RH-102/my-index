from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo
import re

import pandas as pd
import pandas_market_calendars as mcal
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "top100_history.csv"

START_YEAR = 2026
QUARTER_MONTHS = [2, 5, 8, 11]
TIMEZONE = ZoneInfo("America/New_York")

SOURCES = {
    "NASDAQ": "https://stockanalysis.com/list/nasdaq-stocks/",
    "NYSE": "https://stockanalysis.com/list/nyse-stocks/",
}

COLUMNS = [
    "SnapshotDate",
    "DataDate",
    "Status",
    "Rank",
    "Company",
    "Symbol",
    "Exchange",
    "MarketCap",
    "MarketCapDisplay",
    "Source",
]


def parse_market_cap(value):
    if pd.isna(value):
        return None

    text = str(value).strip().replace("$", "").replace(",", "")
    if not text or text in {"-", "N/A", "nan"}:
        return None

    multiplier = 1.0
    suffix = text[-1].upper()

    if suffix in {"T", "B", "M", "K"}:
        text = text[:-1]
        multiplier = {
            "T": 1e12,
            "B": 1e9,
            "M": 1e6,
            "K": 1e3,
        }[suffix]

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def format_market_cap(value):
    value = float(value)

    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def normalize_company_name(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def fetch_exchange(exchange, url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=45)
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    table = None
    for candidate in tables:
        required = {"Symbol", "Company Name", "Market Cap"}
        if required.issubset(set(candidate.columns)):
            table = candidate.copy()
            break

    if table is None:
        raise RuntimeError(
            f"Could not find the stock table for {exchange}."
        )

    table = table[["Symbol", "Company Name", "Market Cap"]].copy()
    table.columns = ["Symbol", "Company", "MarketCapRaw"]
    table["Exchange"] = exchange
    table["MarketCap"] = table["MarketCapRaw"].map(parse_market_cap)
    table = table.dropna(subset=["Symbol", "Company", "MarketCap"])
    table = table[table["MarketCap"] > 0]

    return table[["Symbol", "Company", "Exchange", "MarketCap"]]


def fetch_current_top100():
    frames = []

    for exchange, url in SOURCES.items():
        print(f"Downloading {exchange} market-cap list...")
        frames.append(fetch_exchange(exchange, url))

    combined = pd.concat(frames, ignore_index=True)
    combined["CompanyKey"] = combined["Company"].map(normalize_company_name)

    # A company can have multiple share classes (for example GOOG/GOOGL).
    # Keep only the largest market-cap listing so the ranking is by company.
    combined = combined.sort_values("MarketCap", ascending=False)
    combined = combined.drop_duplicates(subset=["CompanyKey"], keep="first")

    top100 = combined.head(100).copy().reset_index(drop=True)

    if len(top100) < 100:
        raise RuntimeError(
            f"Only {len(top100)} companies were available after filtering."
        )

    top100["Rank"] = range(1, 101)
    top100["MarketCapDisplay"] = top100["MarketCap"].map(format_market_cap)

    return top100[
        [
            "Rank",
            "Company",
            "Symbol",
            "Exchange",
            "MarketCap",
            "MarketCapDisplay",
        ]
    ]


def quarter_end_dates(year):
    nyse = mcal.get_calendar("NYSE")
    dates = []

    for month in QUARTER_MONTHS:
        start = pd.Timestamp(year=year, month=month, day=1)
        end = start + pd.offsets.MonthEnd(1)

        schedule = nyse.schedule(
            start_date=start.date().isoformat(),
            end_date=end.date().isoformat(),
        )

        if schedule.empty:
            raise RuntimeError(
                f"No NYSE trading dates found for {year}-{month:02d}."
            )

        dates.append(schedule.index[-1].date())

    return dates


def make_snapshot(top100, snapshot_date, data_date, status):
    snapshot = top100.copy()
    snapshot.insert(0, "SnapshotDate", snapshot_date.isoformat())
    snapshot.insert(1, "DataDate", data_date.isoformat())
    snapshot.insert(2, "Status", status)
    snapshot["Source"] = "StockAnalysis"
    return snapshot[COLUMNS]


def load_history():
    if not OUTPUT_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)

    history = pd.read_csv(OUTPUT_FILE)

    for column in COLUMNS:
        if column not in history.columns:
            history[column] = None

    return history[COLUMNS]


def main():
    today = datetime.now(TIMEZONE).date()
    history = load_history()

    # Include all four quarterly targets for the current year, even if a
    # future target has not arrived yet. Future targets get a clearly marked
    # Temporary snapshot using today's data, then are replaced by Official
    # data when the actual quarter-end trading day arrives.
    snapshot_targets = []

    for year in range(START_YEAR, today.year + 1):
        for target in quarter_end_dates(year):
            if year < today.year or target <= today:
                snapshot_targets.append(target)
            elif year == today.year:
                snapshot_targets.append(target)

    if not snapshot_targets:
        print("No quarterly snapshot dates are available.")
        return

    existing_dates = (
        set(history["SnapshotDate"].astype(str))
        if not history.empty
        else set()
    )

    needs_current_data = False

    for target in snapshot_targets:
        target_text = target.isoformat()

        # Exact quarter-end day must replace any Temporary snapshot.
        if target == today:
            needs_current_data = True

        # Missing past or future snapshot gets today's data as Temporary.
        elif target_text not in existing_dates:
            needs_current_data = True

    if not needs_current_data:
        print("Quarterly Top 100 history is already up to date.")
        return

    top100 = fetch_current_top100()
    new_snapshots = []

    for target in snapshot_targets:
        target_text = target.isoformat()
        existing = history[
            history["SnapshotDate"].astype(str) == target_text
        ]

        if target == today:
            # Exact quarter-end trading day: replace Temporary with Official.
            history = history[
                history["SnapshotDate"].astype(str) != target_text
            ]

            new_snapshots.append(
                make_snapshot(top100, target, today, "Official")
            )

            print(f"Saved official snapshot for {target_text}.")

        elif existing.empty:
            # No reliable historical snapshot is available yet. Use today's
            # ranking temporarily, while keeping the true DataDate visible.
            new_snapshots.append(
                make_snapshot(top100, target, today, "Temporary")
            )

            if target > today:
                print(
                    f"Saved future placeholder for {target_text} "
                    f"using data from {today.isoformat()}."
                )
            else:
                print(
                    f"Saved historical placeholder for {target_text} "
                    f"using data from {today.isoformat()}."
                )

    if new_snapshots:
        history = pd.concat(
            [history] + new_snapshots,
            ignore_index=True,
        )

    history["Rank"] = pd.to_numeric(
        history["Rank"],
        errors="coerce",
    )

    history["MarketCap"] = pd.to_numeric(
        history["MarketCap"],
        errors="coerce",
    )

    history = history.sort_values(
        ["SnapshotDate", "Rank"],
        ascending=[True, True],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUTPUT_FILE, index=False)

    print("\nSUCCESS")
    print(f"Snapshots stored: {history['SnapshotDate'].nunique()}")
    print(f"Rows stored: {len(history)}")


if __name__ == "__main__":
    main()
