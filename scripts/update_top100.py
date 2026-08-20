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
LIST_SIZE = 125

SOURCES = {
    "NASDAQ": "https://stockanalysis.com/list/nasdaq-stocks/",
    "NYSE": "https://stockanalysis.com/list/nyse-stocks/",
}

FINANCIALS_URL = "https://stockanalysis.com/stocks/sector/financials/"

COLUMNS = [
    "SnapshotDate",
    "DataDate",
    "Status",
    "ListType",
    "Rank",
    "Company",
    "Symbol",
    "Exchange",
    "MarketCap",
    "MarketCapDisplay",
    "Source",
]

LIST_TYPES = [
    "All",
    "ExFinancials",
]


def headers():
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0 Safari/537.36"
        )
    }


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


def read_stock_table(url, required_columns):
    response = requests.get(
        url,
        headers=headers(),
        timeout=45,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    for candidate in tables:
        if required_columns.issubset(set(candidate.columns)):
            return candidate.copy()

    raise RuntimeError(
        f"Could not find the expected stock table at {url}."
    )


def fetch_exchange(exchange, url):
    table = read_stock_table(
        url,
        {"Symbol", "Company Name", "Market Cap"},
    )

    table = table[["Symbol", "Company Name", "Market Cap"]].copy()
    table.columns = ["Symbol", "Company", "MarketCapRaw"]
    table["Exchange"] = exchange
    table["MarketCap"] = table["MarketCapRaw"].map(parse_market_cap)

    table = table.dropna(
        subset=["Symbol", "Company", "MarketCap"]
    )
    table = table[table["MarketCap"] > 0]

    return table[
        ["Symbol", "Company", "Exchange", "MarketCap"]
    ]


def fetch_financial_symbols():
    financial_symbols = set()

    # StockAnalysis currently paginates the Financials sector list.
    # Read several pages so the exclusion list is not limited to only
    # the first page of financial companies.
    for page in range(1, 5):
        url = FINANCIALS_URL
        if page > 1:
            url = f"{FINANCIALS_URL}?page={page}"

        try:
            table = read_stock_table(
                url,
                {"Symbol", "Company Name"},
            )
        except Exception:
            # A page beyond the end may not exist. Once we already have
            # symbols, it is safe to stop pagination.
            if financial_symbols:
                break
            raise

        symbols = (
            table["Symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

        before = len(financial_symbols)
        financial_symbols.update(symbols.tolist())

        # No new symbols means pagination has reached the end.
        if len(financial_symbols) == before:
            break

    if len(financial_symbols) < 100:
        raise RuntimeError(
            "Financial-sector symbol list looks incomplete."
        )

    print(
        f"Financial-sector symbols loaded: "
        f"{len(financial_symbols)}"
    )

    return financial_symbols


def rank_list(frame):
    ranked = frame.head(LIST_SIZE).copy().reset_index(drop=True)

    if len(ranked) < LIST_SIZE:
        raise RuntimeError(
            f"Only {len(ranked)} companies were available; "
            f"{LIST_SIZE} are required."
        )

    ranked["Rank"] = range(1, LIST_SIZE + 1)
    ranked["MarketCapDisplay"] = ranked["MarketCap"].map(
        format_market_cap
    )

    return ranked[
        [
            "Rank",
            "Company",
            "Symbol",
            "Exchange",
            "MarketCap",
            "MarketCapDisplay",
        ]
    ]


def fetch_current_lists():
    frames = []

    for exchange, url in SOURCES.items():
        print(f"Downloading {exchange} market-cap list...")
        frames.append(fetch_exchange(exchange, url))

    combined = pd.concat(frames, ignore_index=True)

    combined["Symbol"] = (
        combined["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    combined["CompanyKey"] = combined["Company"].map(
        normalize_company_name
    )

    # A company can have multiple share classes (for example GOOG/GOOGL).
    # Keep only the largest market-cap listing so the ranking is by company.
    combined = combined.sort_values("MarketCap", ascending=False)
    combined = combined.drop_duplicates(
        subset=["CompanyKey"],
        keep="first",
    )

    financial_symbols = fetch_financial_symbols()

    all_top125 = rank_list(combined)

    nonfinancial = combined[
        ~combined["Symbol"].isin(financial_symbols)
    ].copy()

    ex_financials_top125 = rank_list(nonfinancial)

    return {
        "All": all_top125,
        "ExFinancials": ex_financials_top125,
    }


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


def make_snapshot(
    ranked,
    list_type,
    snapshot_date,
    data_date,
    status,
):
    snapshot = ranked.copy()
    snapshot.insert(0, "SnapshotDate", snapshot_date.isoformat())
    snapshot.insert(1, "DataDate", data_date.isoformat())
    snapshot.insert(2, "Status", status)
    snapshot.insert(3, "ListType", list_type)
    snapshot["Source"] = "StockAnalysis"

    return snapshot[COLUMNS]


def empty_history():
    return pd.DataFrame(columns=COLUMNS)


def load_history():
    if not OUTPUT_FILE.exists():
        return empty_history()

    history = pd.read_csv(OUTPUT_FILE)

    # Migrate the old Top-100 format by rebuilding the temporary snapshots.
    # At the time of this migration there are no legacy official snapshots.
    if "ListType" not in history.columns:
        print(
            "Legacy Top-100 file detected. "
            "Rebuilding it as Top-125 with two list types."
        )
        return empty_history()

    for column in COLUMNS:
        if column not in history.columns:
            history[column] = None

    return history[COLUMNS]


def snapshot_is_complete(history, target_text):
    subset = history[
        history["SnapshotDate"].astype(str) == target_text
    ]

    if subset.empty:
        return False

    for list_type in LIST_TYPES:
        count = len(
            subset[subset["ListType"] == list_type]
        )
        if count != LIST_SIZE:
            return False

    return True


def main():
    today = datetime.now(TIMEZONE).date()
    history = load_history()

    # Show all four quarterly targets for the current year. Future targets
    # use a clearly marked Temporary snapshot until their actual trading
    # date arrives, at which point they are replaced by Official data.
    snapshot_targets = []

    for year in range(START_YEAR, today.year + 1):
        for target in quarter_end_dates(year):
            if year <= today.year:
                snapshot_targets.append(target)

    if not snapshot_targets:
        print("No quarterly snapshot dates are available.")
        return

    needs_current_data = False

    for target in snapshot_targets:
        target_text = target.isoformat()

        if target == today:
            needs_current_data = True
        elif not snapshot_is_complete(history, target_text):
            needs_current_data = True

    if not needs_current_data:
        print("Quarterly Top 125 history is already up to date.")
        return

    current_lists = fetch_current_lists()
    new_snapshots = []

    for target in snapshot_targets:
        target_text = target.isoformat()
        complete = snapshot_is_complete(history, target_text)

        if target == today:
            history = history[
                history["SnapshotDate"].astype(str) != target_text
            ]

            for list_type, ranked in current_lists.items():
                new_snapshots.append(
                    make_snapshot(
                        ranked,
                        list_type,
                        target,
                        today,
                        "Official",
                    )
                )

            print(f"Saved official Top 125 snapshots for {target_text}.")

        elif not complete:
            history = history[
                history["SnapshotDate"].astype(str) != target_text
            ]

            for list_type, ranked in current_lists.items():
                new_snapshots.append(
                    make_snapshot(
                        ranked,
                        list_type,
                        target,
                        today,
                        "Temporary",
                    )
                )

            print(
                f"Saved temporary Top 125 snapshots for {target_text} "
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
        ["SnapshotDate", "ListType", "Rank"],
        ascending=[True, True, True],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUTPUT_FILE, index=False)

    print("\nSUCCESS")
    print(f"Snapshot dates stored: {history['SnapshotDate'].nunique()}")
    print(f"List types stored: {history['ListType'].nunique()}")
    print(f"Rows stored: {len(history)}")


if __name__ == "__main__":
    main()
