from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


# ============================================================
# Configuration
# ============================================================

BACKFILL_START = "2026-08-12"


# ============================================================
# File locations
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

HOLDINGS_FILE = DATA_DIR / "holdings.csv"
LATEST_FILE = DATA_DIR / "latest.csv"
INDEX_HISTORY_FILE = DATA_DIR / "index_history.csv"
HOLDINGS_HISTORY_FILE = DATA_DIR / "holdings_history.csv"


# ============================================================
# Read holdings
# ============================================================

holdings = pd.read_csv(HOLDINGS_FILE)

holdings["Symbol"] = (
    holdings["Symbol"]
    .astype(str)
    .str.strip()
    .str.upper()
)

holdings["Quantity"] = pd.to_numeric(
    holdings["Quantity"]
)

symbols = holdings["Symbol"].tolist()

print(
    f"Downloading history for {len(symbols)} securities "
    f"from {BACKFILL_START}..."
)


# ============================================================
# Download daily price history
# yfinance end date is exclusive, so add one day.
# ============================================================

end_date = (
    datetime.now().date()
    + timedelta(days=1)
).isoformat()

prices = yf.download(
    tickers=symbols,
    start=BACKFILL_START,
    end=end_date,
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    threads=True,
    progress=False,
    multi_level_index=True,
)


# ============================================================
# Build a price map for every symbol and trading date
# ============================================================

price_history = {}
missing_symbols = []

for symbol in symbols:

    try:
        close = prices[symbol]["Close"].dropna()

        if close.empty:
            missing_symbols.append(symbol)
            continue

        price_history[symbol] = {
            idx.date().isoformat(): float(value)
            for idx, value in close.items()
        }

    except Exception:
        missing_symbols.append(symbol)


if missing_symbols:

    print("\nMissing price history:")

    for symbol in missing_symbols:
        print(f" - {symbol}")

    raise RuntimeError(
        "Some securities have no price history. "
        "No index data was saved."
    )


# ============================================================
# Use only trading dates available for every holding
# ============================================================

common_dates = set.intersection(
    *[
        set(price_history[symbol].keys())
        for symbol in symbols
    ]
)

common_dates = sorted(common_dates)

if not common_dates:
    raise RuntimeError(
        "No common trading dates were found for all holdings."
    )

if BACKFILL_START not in common_dates:
    raise RuntimeError(
        f"{BACKFILL_START} is not available for every holding. "
        "Backfill was stopped to avoid an incorrect index history."
    )

print(
    f"Common trading dates: {common_dates[0]} "
    f"through {common_dates[-1]}"
)


# ============================================================
# Build complete historical snapshots
# ============================================================

holdings_history_rows = []
index_history_rows = []

daily_totals = {}

for market_date in common_dates:

    snapshot = holdings[
        ["Symbol", "Name", "Quantity"]
    ].copy()

    snapshot["Price"] = snapshot["Symbol"].map(
        lambda symbol: price_history[symbol][market_date]
    )

    snapshot["MarketValue"] = (
        snapshot["Quantity"]
        * snapshot["Price"]
    )

    total_value = float(
        snapshot["MarketValue"].sum()
    )

    snapshot["Weight"] = (
        snapshot["MarketValue"]
        / total_value
    )

    daily_totals[market_date] = total_value

    for _, row in snapshot.iterrows():
        holdings_history_rows.append(
            {
                "Date": market_date,
                "Symbol": row["Symbol"],
                "Name": row["Name"],
                "Quantity": row["Quantity"],
                "Price": row["Price"],
                "MarketValue": row["MarketValue"],
                "Weight": row["Weight"],
            }
        )


# ============================================================
# Index level: first day = 100
# ============================================================

base_market_value = daily_totals[BACKFILL_START]
previous_level = None
updated_at = datetime.now().isoformat(timespec="seconds")

for market_date in common_dates:

    total_value = daily_totals[market_date]

    index_level = (
        total_value
        / base_market_value
        * 100
    )

    if previous_level is None:
        daily_return = 0.0
    else:
        daily_return = (
            index_level
            / previous_level
            - 1
        )

    index_history_rows.append(
        {
            "Date": market_date,
            "TotalMarketValue": total_value,
            "IndexLevel": index_level,
            "DailyReturn": daily_return,
            "NumberOfHoldings": len(holdings),
            "UpdatedAt": updated_at,
        }
    )

    previous_level = index_level


# ============================================================
# Save full holdings history
# ============================================================

holdings_history = pd.DataFrame(
    holdings_history_rows
)

holdings_history = holdings_history.sort_values(
    ["Date", "Weight"],
    ascending=[True, False]
)

holdings_history.to_csv(
    HOLDINGS_HISTORY_FILE,
    index=False
)


# ============================================================
# Save full index history
# ============================================================

index_history = pd.DataFrame(
    index_history_rows
)

index_history = index_history.sort_values("Date")

index_history.to_csv(
    INDEX_HISTORY_FILE,
    index=False
)


# ============================================================
# Save latest snapshot
# ============================================================

latest_market_date = common_dates[-1]

latest = holdings[
    ["Symbol", "Name", "Quantity"]
].copy()

latest["Price"] = latest["Symbol"].map(
    lambda symbol: price_history[symbol][latest_market_date]
)

latest["PriceDate"] = latest_market_date

latest["MarketValue"] = (
    latest["Quantity"]
    * latest["Price"]
)

latest_total_value = float(
    latest["MarketValue"].sum()
)

latest["Weight"] = (
    latest["MarketValue"]
    / latest_total_value
)

latest = (
    latest
    .sort_values(
        by="Weight",
        ascending=False
    )
    .reset_index(drop=True)
)

latest.to_csv(
    LATEST_FILE,
    index=False
)


# ============================================================
# Display results
# ============================================================

latest_index = index_history.iloc[-1]

print("\nSUCCESS")
print(f"Backfill start: {BACKFILL_START}")
print(f"Latest market date: {latest_market_date}")
print(f"Trading days saved: {len(common_dates)}")
print(f"Total market value: ${latest_total_value:,.2f}")
print(f"Index level: {latest_index['IndexLevel']:.2f}")
print(f"Daily return: {latest_index['DailyReturn'] * 100:.2f}%")
print(f"Holdings saved per day: {len(holdings)}")

print("\nTop 10 holdings:")

for _, row in latest.head(10).iterrows():
    print(
        f"{row['Symbol']:6s} "
        f"${row['Price']:10.2f} "
        f"{row['Weight'] * 100:7.2f}%"
    )
