from pathlib import Path
from datetime import datetime
import pandas as pd
import yfinance as yf


# -----------------------------
# File locations
# -----------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

HOLDINGS_FILE = DATA_DIR / "holdings.csv"
LATEST_FILE = DATA_DIR / "latest.csv"
HISTORY_FILE = DATA_DIR / "index_history.csv"


# -----------------------------
# Read holdings
# -----------------------------
holdings = pd.read_csv(HOLDINGS_FILE)

holdings["Symbol"] = holdings["Symbol"].astype(str).str.strip().str.upper()
holdings["Quantity"] = pd.to_numeric(holdings["Quantity"])

symbols = holdings["Symbol"].tolist()

print(f"Downloading prices for {len(symbols)} securities...")


# -----------------------------
# Download latest prices
# -----------------------------
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

        close = ticker_data["Close"].dropna()

        if close.empty:
            missing.append(symbol)
            continue

        price_map[symbol] = float(close.iloc[-1])
        date_map[symbol] = close.index[-1].date().isoformat()

    except Exception:
        missing.append(symbol)


# -----------------------------
# Safety check
# -----------------------------
if missing:
    print("\nMissing prices:")
    for symbol in missing:
        print(f" - {symbol}")

    raise RuntimeError(
        "Some securities have no price data. "
        "No index data was saved, to prevent incorrect index calculations."
    )


# -----------------------------
# Calculate market values
# -----------------------------
holdings["Price"] = holdings["Symbol"].map(price_map)
holdings["PriceDate"] = holdings["Symbol"].map(date_map)

holdings["MarketValue"] = holdings["Quantity"] * holdings["Price"]

total_value = holdings["MarketValue"].sum()

holdings["Weight"] = holdings["MarketValue"] / total_value

holdings = holdings.sort_values(
    by="Weight",
    ascending=False
).reset_index(drop=True)


# -----------------------------
# Save latest holdings
# -----------------------------
holdings.to_csv(LATEST_FILE, index=False)


# -----------------------------
# Save index history
# -----------------------------
latest_market_date = max(date_map.values())

today_row = pd.DataFrame(
    [
        {
            "Date": latest_market_date,
            "TotalMarketValue": total_value,
            "NumberOfHoldings": len(holdings),
            "UpdatedAt": datetime.now().isoformat(timespec="seconds"),
        }
    ]
)


if HISTORY_FILE.exists():
    history = pd.read_csv(HISTORY_FILE)

    # Do not duplicate the same trading day
    history = history[history["Date"] != latest_market_date]

    history = pd.concat(
        [history, today_row],
        ignore_index=True
    )

else:
    history = today_row


history = history.sort_values("Date")
history.to_csv(HISTORY_FILE, index=False)


# -----------------------------
# Display results
# -----------------------------
print("\nSUCCESS")
print(f"Market date: {latest_market_date}")
print(f"Total market value: ${total_value:,.2f}")

print("\nTop 10 holdings:")

for _, row in holdings.head(10).iterrows():
    print(
        f"{row['Symbol']:6s} "
        f"${row['Price']:10.2f} "
        f"{row['Weight'] * 100:7.2f}%"
    )
