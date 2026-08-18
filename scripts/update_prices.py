from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf


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
    f"Downloading prices for "
    f"{len(symbols)} securities..."
)


# ============================================================
# Download latest daily prices
# ============================================================

prices = yf.download(
    tickers=symbols,
    period="5d",
    interval="1d",
    auto_adjust=False,
    group_by="ticker",
    threads=True,
    progress=False,
    multi_level_index=True,
)


price_map = {}
date_map = {}
missing = []


for symbol in symbols:

    try:

        ticker_data = prices[symbol]

        close = (
            ticker_data["Close"]
            .dropna()
        )

        if close.empty:

            missing.append(symbol)
            continue

        latest_price = float(
            close.iloc[-1]
        )

        latest_date = (
            close.index[-1]
            .date()
            .isoformat()
        )

        price_map[symbol] = latest_price
        date_map[symbol] = latest_date

    except Exception:

        missing.append(symbol)


# ============================================================
# Safety check
# ============================================================

if missing:

    print("\nMissing prices:")

    for symbol in missing:
        print(f" - {symbol}")

    raise RuntimeError(
        "Some securities have no price data. "
        "No index data was saved."
    )


# ============================================================
# Calculate market values and weights
# ============================================================

holdings["Price"] = (
    holdings["Symbol"]
    .map(price_map)
)

holdings["PriceDate"] = (
    holdings["Symbol"]
    .map(date_map)
)

holdings["MarketValue"] = (
    holdings["Quantity"]
    * holdings["Price"]
)

total_value = (
    holdings["MarketValue"]
    .sum()
)

holdings["Weight"] = (
    holdings["MarketValue"]
    / total_value
)


# Sort by largest weight
holdings = (
    holdings
    .sort_values(
        by="Weight",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# Save latest snapshot
# ============================================================

holdings.to_csv(
    LATEST_FILE,
    index=False
)


# ============================================================
# Determine market date
# ============================================================

latest_market_date = max(
    date_map.values()
)


# ============================================================
# Save DAILY HOLDINGS HISTORY
# ============================================================

daily_history = holdings[
    [
        "Symbol",
        "Name",
        "Quantity",
        "Price",
        "MarketValue",
        "Weight",
    ]
].copy()


daily_history.insert(
    0,
    "Date",
    latest_market_date
)


if HOLDINGS_HISTORY_FILE.exists():

    holdings_history = pd.read_csv(
        HOLDINGS_HISTORY_FILE
    )

    # Remove existing records for the same
    # trading day, so rerunning the workflow
    # does not create duplicates.
    holdings_history = (
        holdings_history[
            holdings_history["Date"]
            != latest_market_date
        ]
    )

    holdings_history = pd.concat(
        [
            holdings_history,
            daily_history
        ],
        ignore_index=True
    )

else:

    holdings_history = daily_history


holdings_history = (
    holdings_history
    .sort_values(
        ["Date", "Weight"],
        ascending=[True, False]
    )
)

holdings_history.to_csv(
    HOLDINGS_HISTORY_FILE,
    index=False
)


# ============================================================
# Save INDEX HISTORY
# ============================================================

today_row = pd.DataFrame(
    [
        {
            "Date": latest_market_date,
            "TotalMarketValue": total_value,
            "NumberOfHoldings": len(holdings),
            "UpdatedAt": datetime.now().isoformat(
                timespec="seconds"
            ),
        }
    ]
)


if INDEX_HISTORY_FILE.exists():

    index_history = pd.read_csv(
        INDEX_HISTORY_FILE
    )

    # Prevent duplicate dates
    index_history = (
        index_history[
            index_history["Date"]
            != latest_market_date
        ]
    )

    index_history = pd.concat(
        [
            index_history,
            today_row
        ],
        ignore_index=True
    )

else:

    index_history = today_row


index_history = (
    index_history
    .sort_values("Date")
)

index_history.to_csv(
    INDEX_HISTORY_FILE,
    index=False
)


# ============================================================
# Display results
# ============================================================

print("\nSUCCESS")

print(
    f"Market date: "
    f"{latest_market_date}"
)

print(
    f"Total market value: "
    f"${total_value:,.2f}"
)

print(
    f"Holdings saved: "
    f"{len(holdings)}"
)

print("\nTop 10 holdings:")


for _, row in holdings.head(10).iterrows():

    print(
        f"{row['Symbol']:6s} "
        f"${row['Price']:10.2f} "
        f"{row['Weight'] * 100:7.2f}%"
    )
