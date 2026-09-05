from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


# ============================================================
# Configuration
# ============================================================

BACKFILL_START = "2026-08-12"
BASE_INDEX_LEVEL = 100.0
ANCHOR_SYMBOL = "NVDA"


# ============================================================
# File locations
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

REBALANCES_FILE = DATA_DIR / "rebalances.csv"
HOLDINGS_FILE = DATA_DIR / "holdings.csv"
LATEST_FILE = DATA_DIR / "latest.csv"
INDEX_HISTORY_FILE = DATA_DIR / "index_history.csv"
HOLDINGS_HISTORY_FILE = DATA_DIR / "holdings_history.csv"
DIVISOR_HISTORY_FILE = DATA_DIR / "divisor_history.csv"


# ============================================================
# Read immutable rebalance snapshots
#
# Semantics:
# - EffectiveDate is the trading-day CLOSE at which the rebalance occurs.
# - That day's return is still earned by the old basket.
# - At that close, the new basket is installed and the divisor is reset so
#   the Index Level is exactly unchanged.
# - The new basket drives returns from the next trading day onward.
# ============================================================

rebalances = pd.read_csv(REBALANCES_FILE)

required_columns = {"EffectiveDate", "Symbol", "Name", "Quantity"}
missing_columns = required_columns - set(rebalances.columns)
if missing_columns:
    raise RuntimeError(
        "rebalances.csv is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

rebalances["EffectiveDate"] = pd.to_datetime(
    rebalances["EffectiveDate"], errors="raise"
).dt.date.astype(str)

rebalances["Symbol"] = (
    rebalances["Symbol"].astype(str).str.strip().str.upper()
)
rebalances["Name"] = rebalances["Name"].astype(str).str.strip()
rebalances["Quantity"] = pd.to_numeric(
    rebalances["Quantity"], errors="raise"
)

if (rebalances["Quantity"] <= 0).any():
    raise RuntimeError("All rebalance quantities must be positive.")

if rebalances.duplicated(["EffectiveDate", "Symbol"]).any():
    raise RuntimeError(
        "Each EffectiveDate may contain each symbol only once."
    )

effective_dates = sorted(rebalances["EffectiveDate"].unique())

if not effective_dates:
    raise RuntimeError("rebalances.csv contains no snapshots.")

if effective_dates[0] != BACKFILL_START:
    raise RuntimeError(
        f"The first rebalance snapshot must be effective on {BACKFILL_START}."
    )

snapshots = {}

for effective_date in effective_dates:
    snapshot = (
        rebalances.loc[
            rebalances["EffectiveDate"] == effective_date,
            ["Symbol", "Name", "Quantity"],
        ]
        .copy()
        .sort_values("Symbol")
        .reset_index(drop=True)
    )

    anchor = snapshot.loc[snapshot["Symbol"] == ANCHOR_SYMBOL]

    if len(anchor) != 1:
        raise RuntimeError(
            f"{effective_date}: every snapshot must contain "
            f"{ANCHOR_SYMBOL} exactly once."
        )

    anchor_quantity = float(anchor.iloc[0]["Quantity"])
    if abs(anchor_quantity - 1.0) > 1e-9:
        raise RuntimeError(
            f"{effective_date}: {ANCHOR_SYMBOL} Quantity must equal 1."
        )

    snapshots[effective_date] = snapshot

symbols = sorted(rebalances["Symbol"].unique())

print(
    f"Downloading history for {len(symbols)} securities "
    f"from {BACKFILL_START}..."
)


# ============================================================
# Download daily price history
# ============================================================

end_date = (datetime.now().date() + timedelta(days=1)).isoformat()

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
# Build price map
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
        "Some securities have no price history. No index data was saved."
    )

calendar_dates = sorted(
    date
    for date in price_history[ANCHOR_SYMBOL]
    if date >= BACKFILL_START
)

if not calendar_dates:
    raise RuntimeError("No trading dates were found.")

if BACKFILL_START not in calendar_dates:
    raise RuntimeError(
        f"{BACKFILL_START} is not available for {ANCHOR_SYMBOL}."
    )

# Rebalances are close-of-day events, so every stored effective date must be
# an actual trading date once that date is in the historical window.
for effective_date in effective_dates:
    if effective_date <= calendar_dates[-1] and effective_date not in calendar_dates:
        raise RuntimeError(
            f"{effective_date}: rebalance EffectiveDate is not a trading day."
        )


# ============================================================
# Helpers
# ============================================================

def aggregate_value(snapshot, market_date):
    missing = [
        symbol
        for symbol in snapshot["Symbol"]
        if market_date not in price_history[symbol]
    ]
    if missing:
        raise RuntimeError(
            f"Missing prices on {market_date} for active holdings: "
            + ", ".join(missing)
        )

    return float(
        sum(
            float(row.Quantity) * price_history[row.Symbol][market_date]
            for row in snapshot.itertuples(index=False)
        )
    )


def build_market_snapshot(snapshot, market_date):
    result = snapshot.copy()
    result["Price"] = result["Symbol"].map(
        lambda symbol: price_history[symbol][market_date]
    )
    result["MarketValue"] = result["Quantity"] * result["Price"]
    total = float(result["MarketValue"].sum())
    result["Weight"] = result["MarketValue"] / total
    return result, total


# ============================================================
# Build index history
#
# Rebalance-day continuity:
# 1) Calculate that day's close using the OLD basket/divisor.
# 2) Install the NEW basket at the same close.
# 3) NewDivisor = NewBasketValueAtClose / ExistingIndexLevel.
#
# Therefore a rebalance changes holdings/weights but cannot create an
# artificial index gain or loss.
# ============================================================

first_snapshot = snapshots[BACKFILL_START]
first_value = aggregate_value(first_snapshot, BACKFILL_START)
current_divisor = first_value / BASE_INDEX_LEVEL
current_snapshot = first_snapshot
current_effective_date = BACKFILL_START

divisor_history_rows = [
    {
        "EffectiveDate": BACKFILL_START,
        "Divisor": current_divisor,
        "NumberOfHoldings": len(current_snapshot),
    }
]

holdings_history_rows = []
index_history_rows = []

previous_level = None
updated_at = datetime.now().isoformat(timespec="seconds")

for market_date in calendar_dates:
    # First calculate today's close with the basket that was active
    # throughout the trading session.
    pre_rebalance_value = aggregate_value(
        current_snapshot,
        market_date,
    )
    index_level = pre_rebalance_value / current_divisor

    if previous_level is None:
        daily_return = 0.0
    else:
        daily_return = index_level / previous_level - 1.0

    # A snapshot dated today is installed AFTER today's close.
    if (
        market_date in snapshots
        and market_date != current_effective_date
    ):
        new_snapshot = snapshots[market_date]
        new_basket_value = aggregate_value(
            new_snapshot,
            market_date,
        )

        current_divisor = new_basket_value / index_level
        current_snapshot = new_snapshot
        current_effective_date = market_date

        divisor_history_rows.append(
            {
                "EffectiveDate": current_effective_date,
                "Divisor": current_divisor,
                "NumberOfHoldings": len(current_snapshot),
            }
        )

        # Numerical safety check: the new basket must reproduce the same level.
        continuity_level = new_basket_value / current_divisor
        if abs(continuity_level - index_level) > 1e-10:
            raise RuntimeError(
                f"{market_date}: divisor reset failed continuity check."
            )

    # Holdings shown for a rebalance date are the NEW after-close holdings.
    market_snapshot, basket_value = build_market_snapshot(
        current_snapshot,
        market_date,
    )

    displayed_level = basket_value / current_divisor
    if abs(displayed_level - index_level) > 1e-8:
        raise RuntimeError(
            f"{market_date}: displayed basket does not match Index Level."
        )

    for _, row in market_snapshot.iterrows():
        holdings_history_rows.append(
            {
                "Date": market_date,
                "EffectiveDate": current_effective_date,
                "Symbol": row["Symbol"],
                "Name": row["Name"],
                "Quantity": row["Quantity"],
                "Price": row["Price"],
                "MarketValue": row["MarketValue"],
                "Weight": row["Weight"],
            }
        )

    index_history_rows.append(
        {
            "Date": market_date,
            "IndexLevel": index_level,
            "DailyReturn": daily_return,
            "NumberOfHoldings": len(current_snapshot),
            "Divisor": current_divisor,
            "EffectiveDate": current_effective_date,
            "UpdatedAt": updated_at,
        }
    )

    previous_level = index_level


# ============================================================
# Save histories
# ============================================================

holdings_history = pd.DataFrame(holdings_history_rows).sort_values(
    ["Date", "Weight"],
    ascending=[True, False],
)
holdings_history.to_csv(HOLDINGS_HISTORY_FILE, index=False)

index_history = pd.DataFrame(index_history_rows).sort_values("Date")
index_history.to_csv(INDEX_HISTORY_FILE, index=False)

divisor_history = pd.DataFrame(divisor_history_rows)
divisor_history.to_csv(DIVISOR_HISTORY_FILE, index=False)


# ============================================================
# Save latest current-view files
# ============================================================

latest_market_date = str(index_history.iloc[-1]["Date"])

latest_snapshot = current_snapshot.copy()
latest_snapshot[
    ["Symbol", "Name", "Quantity"]
].to_csv(HOLDINGS_FILE, index=False)

latest, latest_basket_value = build_market_snapshot(
    latest_snapshot,
    latest_market_date,
)

latest["PriceDate"] = latest_market_date
latest = latest[
    [
        "Symbol",
        "Name",
        "Quantity",
        "Price",
        "PriceDate",
        "MarketValue",
        "Weight",
    ]
]

latest = (
    latest.sort_values("Weight", ascending=False)
    .reset_index(drop=True)
)

latest.to_csv(LATEST_FILE, index=False)


# ============================================================
# Display results
# ============================================================

latest_index = index_history.iloc[-1]

print("\nSUCCESS")
print(f"Backfill start: {BACKFILL_START}")
print(f"Latest market date: {latest_market_date}")
print(f"Trading days saved: {len(index_history)}")
print(f"Active rebalance: {current_effective_date}")
print(f"Divisor: {current_divisor:.12f}")
print(f"Index level: {latest_index['IndexLevel']:.6f}")
print(
    f"Daily return: "
    f"{latest_index['DailyReturn'] * 100:.4f}%"
)
print(f"Holdings: {len(latest_snapshot)}")
