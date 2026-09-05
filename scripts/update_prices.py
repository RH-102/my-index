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
# Each EffectiveDate must contain a complete portfolio snapshot.
# The first snapshot starts the index. Future snapshots are appended;
# old snapshots are never edited.
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
    rebalances["Symbol"]
    .astype(str)
    .str.strip()
    .str.upper()
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
        f"The first rebalance snapshot must be effective on "
        f"{BACKFILL_START}."
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
        "Some securities have no price history. "
        "No index data was saved."
    )

if ANCHOR_SYMBOL not in price_history:
    raise RuntimeError(
        f"{ANCHOR_SYMBOL} price history is unavailable."
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


# ============================================================
# Helpers
# ============================================================

def effective_date_for(market_date):
    eligible = [
        effective_date
        for effective_date in effective_dates
        if effective_date <= market_date
    ]

    if not eligible:
        return None

    return eligible[-1]


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
            float(row.Quantity)
            * price_history[row.Symbol][market_date]
            for row in snapshot.itertuples(index=False)
        )
    )


def build_market_snapshot(snapshot, market_date):
    result = snapshot.copy()

    result["Price"] = result["Symbol"].map(
        lambda symbol: price_history[symbol][market_date]
    )

    result["MarketValue"] = (
        result["Quantity"]
        * result["Price"]
    )

    total = float(result["MarketValue"].sum())

    result["Weight"] = result["MarketValue"] / total

    return result, total


# ============================================================
# Build index history with regime-specific quantities + divisor
#
# First regime:
#   Divisor = basket value / 100
#
# Later rebalances:
#   NewDivisor = value of NEW basket at the prior close
#                / prior Index Level
#
# This keeps the Index Level continuous across rebalances even
# though NVDA is reset to Quantity = 1 and all other quantities
# are recomputed from verified market-cap weights.
# ============================================================

holdings_history_rows = []
index_history_rows = []
divisor_history_rows = []

current_effective_date = None
current_divisor = None
previous_market_date = None
previous_level = None
updated_at = datetime.now().isoformat(timespec="seconds")

for market_date in calendar_dates:
    active_effective_date = effective_date_for(market_date)

    if active_effective_date is None:
        continue

    active_snapshot = snapshots[active_effective_date]

    # A rebalance regime change becomes effective on this trading day.
    if active_effective_date != current_effective_date:
        if current_effective_date is None:
            if market_date != BACKFILL_START:
                raise RuntimeError(
                    "The first active trading date does not match "
                    f"{BACKFILL_START}."
                )

            initial_value = aggregate_value(
                active_snapshot,
                market_date,
            )

            current_divisor = (
                initial_value / BASE_INDEX_LEVEL
            )

        else:
            if previous_market_date is None or previous_level is None:
                raise RuntimeError(
                    "Cannot calculate a rebalance divisor without "
                    "a prior trading day."
                )

            new_basket_at_prior_close = aggregate_value(
                active_snapshot,
                previous_market_date,
            )

            current_divisor = (
                new_basket_at_prior_close
                / previous_level
            )

        current_effective_date = active_effective_date

        divisor_history_rows.append(
            {
                "EffectiveDate": current_effective_date,
                "Divisor": current_divisor,
                "NumberOfHoldings": len(active_snapshot),
            }
        )

    market_snapshot, basket_value = build_market_snapshot(
        active_snapshot,
        market_date,
    )

    index_level = basket_value / current_divisor

    if previous_level is None:
        daily_return = 0.0
    else:
        daily_return = index_level / previous_level - 1.0

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
            "NumberOfHoldings": len(active_snapshot),
            "Divisor": current_divisor,
            "EffectiveDate": current_effective_date,
            "UpdatedAt": updated_at,
        }
    )

    previous_market_date = market_date
    previous_level = index_level


# ============================================================
# Save histories
# ============================================================

holdings_history = pd.DataFrame(holdings_history_rows)
holdings_history = holdings_history.sort_values(
    ["Date", "Weight"],
    ascending=[True, False],
)

holdings_history.to_csv(
    HOLDINGS_HISTORY_FILE,
    index=False,
)

index_history = pd.DataFrame(index_history_rows)
index_history = index_history.sort_values("Date")

index_history.to_csv(
    INDEX_HISTORY_FILE,
    index=False,
)

divisor_history = pd.DataFrame(divisor_history_rows)
divisor_history.to_csv(
    DIVISOR_HISTORY_FILE,
    index=False,
)


# ============================================================
# Save latest portfolio snapshot
# holdings.csv remains a convenient current-view file, but
# rebalances.csv is the canonical history/source of truth.
# ============================================================

latest_market_date = index_history.iloc[-1]["Date"]
latest_effective_date = index_history.iloc[-1]["EffectiveDate"]
latest_snapshot = snapshots[latest_effective_date].copy()

latest_snapshot[
    ["Symbol", "Name", "Quantity"]
].to_csv(
    HOLDINGS_FILE,
    index=False,
)

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
    latest
    .sort_values(
        by="Weight",
        ascending=False,
    )
    .reset_index(drop=True)
)

latest.to_csv(
    LATEST_FILE,
    index=False,
)


# ============================================================
# Display results
# ============================================================

latest_index = index_history.iloc[-1]

print("\nSUCCESS")
print(f"Backfill start: {BACKFILL_START}")
print(f"Latest market date: {latest_market_date}")
print(f"Trading days saved: {len(index_history)}")
print(f"Active rebalance: {latest_effective_date}")
print(f"Divisor: {latest_index['Divisor']:.12f}")
print(f"Index level: {latest_index['IndexLevel']:.2f}")
print(
    f"Daily return: "
    f"{latest_index['DailyReturn'] * 100:.2f}%"
)
print(f"Holdings: {len(latest_snapshot)}")

print("\nTop 10 holdings:")

for _, row in latest.head(10).iterrows():
    print(
        f"{row['Symbol']:6s} "
        f"${row['Price']:10.2f} "
        f"{row['Weight'] * 100:7.2f}%"
    )
