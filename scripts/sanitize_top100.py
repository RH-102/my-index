from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "top100_history.csv"

# Only compare against a very recent prior snapshot. This catches provider
# glitches without freezing genuine long-term changes in shares outstanding.
MAX_REFERENCE_AGE_DAYS = 7

# A move larger than 50% in market cap over a few days is suspicious enough
# to require a price cross-check.
SUSPICIOUS_UP_RATIO = 1.50
SUSPICIOUS_DOWN_RATIO = 2.0 / 3.0

# If market cap still differs from the price-implied value by more than 30%,
# treat the source market cap as bad and replace it with the price-implied cap.
MAX_PRICE_IMPLIED_DEVIATION = 0.30

# If price validation is unavailable, only auto-correct truly extreme moves.
EXTREME_UP_RATIO = 1.80
EXTREME_DOWN_RATIO = 0.55


def format_market_cap(value: float) -> str:
    value = float(value)
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


def yf_symbol(symbol: str) -> str:
    # yfinance uses BRK-B rather than BRK.B, etc.
    return str(symbol).strip().upper().replace(".", "-")


def price_ratio(symbol: str, reference_date: date, current_date: date) -> float | None:
    try:
        ticker = yf.Ticker(yf_symbol(symbol))
        start = pd.Timestamp(reference_date) - pd.Timedelta(days=5)
        end = pd.Timestamp(current_date) + pd.Timedelta(days=2)
        prices = ticker.history(
            start=start.date().isoformat(),
            end=end.date().isoformat(),
            auto_adjust=False,
            actions=False,
        )
        if prices.empty or "Close" not in prices.columns:
            return None

        closes = pd.to_numeric(prices["Close"], errors="coerce").dropna()
        if closes.empty:
            return None

        idx = pd.to_datetime(closes.index).tz_localize(None)
        closes.index = idx

        ref_cutoff = pd.Timestamp(reference_date)
        cur_cutoff = pd.Timestamp(current_date)
        ref_candidates = closes[closes.index <= ref_cutoff]
        cur_candidates = closes[closes.index <= cur_cutoff]
        if ref_candidates.empty or cur_candidates.empty:
            return None

        ref_price = float(ref_candidates.iloc[-1])
        cur_price = float(cur_candidates.iloc[-1])
        if ref_price <= 0 or cur_price <= 0:
            return None
        return cur_price / ref_price
    except Exception as exc:
        print(f"Price sanity check failed for {symbol}: {exc}")
        return None


def latest_reference_map(history: pd.DataFrame, current_date: date) -> dict[str, tuple[float, date]]:
    rows = history.copy()
    rows = rows[rows["ListType"].astype(str) == "All"]
    rows["DataDateParsed"] = pd.to_datetime(rows["DataDate"], errors="coerce")
    rows["MarketCapNum"] = pd.to_numeric(rows["MarketCap"], errors="coerce")
    rows["SymbolNorm"] = rows["Symbol"].astype(str).str.strip().str.upper()
    rows = rows.dropna(subset=["DataDateParsed", "MarketCapNum", "SymbolNorm"])
    rows = rows[rows["DataDateParsed"].dt.date < current_date]
    if rows.empty:
        return {}

    # One row per company per data date is enough. Prefer Official if duplicate
    # dates exist, then choose the latest prior observation.
    rows["StatusPriority"] = rows["Status"].astype(str).map({"Official": 1}).fillna(0)
    rows = rows.sort_values(["DataDateParsed", "StatusPriority"])
    rows = rows.drop_duplicates(subset=["SymbolNorm"], keep="last")

    result: dict[str, tuple[float, date]] = {}
    for _, row in rows.iterrows():
        result[str(row["SymbolNorm"])] = (
            float(row["MarketCapNum"]),
            row["DataDateParsed"].date(),
        )
    return result


def corrected_cap(
    symbol: str,
    current_cap: float,
    current_date: date,
    reference_cap: float,
    reference_date: date,
) -> tuple[float, str | None]:
    age_days = (current_date - reference_date).days
    if age_days <= 0 or age_days > MAX_REFERENCE_AGE_DAYS or reference_cap <= 0:
        return current_cap, None

    ratio = current_cap / reference_cap
    if SUSPICIOUS_DOWN_RATIO <= ratio <= SUSPICIOUS_UP_RATIO:
        return current_cap, None

    px_ratio = price_ratio(symbol, reference_date, current_date)
    if px_ratio is not None:
        expected_cap = reference_cap * px_ratio
        if expected_cap > 0:
            deviation = abs(current_cap / expected_cap - 1.0)
            if deviation > MAX_PRICE_IMPLIED_DEVIATION:
                note = (
                    f"{symbol}: source cap {format_market_cap(current_cap)} vs "
                    f"price-implied {format_market_cap(expected_cap)} "
                    f"from {format_market_cap(reference_cap)} on {reference_date} "
                    f"(price ratio {px_ratio:.3f}); corrected."
                )
                return expected_cap, note
            return current_cap, None

    # Fallback only for very large moves when the price cross-check is missing.
    if ratio >= EXTREME_UP_RATIO or ratio <= EXTREME_DOWN_RATIO:
        note = (
            f"{symbol}: extreme source cap jump from {format_market_cap(reference_cap)} "
            f"on {reference_date} to {format_market_cap(current_cap)}; "
            "price validation unavailable, reverted to prior reliable cap."
        )
        return reference_cap, note

    print(
        f"SANITY WARNING {symbol}: market cap moved {ratio:.2f}x in {age_days} days, "
        "but price validation was unavailable; value left unchanged."
    )
    return current_cap, None


def rerank_current_rows(history: pd.DataFrame, current_date: date) -> pd.DataFrame:
    current_text = current_date.isoformat()
    mask_current = history["DataDate"].astype(str) == current_text
    if not mask_current.any():
        print(f"No Top 125 rows use DataDate {current_text}; sanity check skipped.")
        return history

    references = latest_reference_map(history, current_date)
    if not references:
        print("No recent prior Top 125 reference data; sanity check skipped.")
        return history

    # Compute corrections once from the All list, then apply the same corrected
    # company market cap to every list type for today's data.
    all_current = history[
        mask_current & (history["ListType"].astype(str) == "All")
    ].copy()
    all_current["MarketCapNum"] = pd.to_numeric(all_current["MarketCap"], errors="coerce")

    corrections: dict[str, float] = {}
    for _, row in all_current.dropna(subset=["MarketCapNum"]).iterrows():
        symbol = str(row["Symbol"]).strip().upper()
        reference = references.get(symbol)
        if reference is None:
            continue
        new_cap, note = corrected_cap(
            symbol=symbol,
            current_cap=float(row["MarketCapNum"]),
            current_date=current_date,
            reference_cap=reference[0],
            reference_date=reference[1],
        )
        if note:
            corrections[symbol] = new_cap
            print("SANITY CORRECTION", note)

    if not corrections:
        print("Top 125 sanity check: no corrections needed.")
        return history

    symbol_norm = history["Symbol"].astype(str).str.strip().str.upper()
    for symbol, cap in corrections.items():
        mask = mask_current & (symbol_norm == symbol)
        history.loc[mask, "MarketCap"] = cap
        history.loc[mask, "MarketCapDisplay"] = format_market_cap(cap)

    # Re-rank each current snapshot/list independently after corrections.
    current_snapshot_dates = history.loc[mask_current, "SnapshotDate"].astype(str).unique()
    for snapshot_date in current_snapshot_dates:
        for list_type in history.loc[
            mask_current & (history["SnapshotDate"].astype(str) == snapshot_date),
            "ListType",
        ].astype(str).unique():
            group_mask = (
                mask_current
                & (history["SnapshotDate"].astype(str) == snapshot_date)
                & (history["ListType"].astype(str) == list_type)
            )
            group = history.loc[group_mask].copy()
            group["MarketCapNum"] = pd.to_numeric(group["MarketCap"], errors="coerce")
            group = group.sort_values("MarketCapNum", ascending=False).reset_index()
            group["Rank"] = range(1, len(group) + 1)
            for _, ranked_row in group.iterrows():
                history.loc[int(ranked_row["index"]), "Rank"] = int(ranked_row["Rank"])

    return history


def main() -> None:
    if not DATA_FILE.exists():
        print("Top 125 history file does not exist; sanity check skipped.")
        return

    history = pd.read_csv(DATA_FILE)
    if history.empty:
        print("Top 125 history is empty; sanity check skipped.")
        return

    data_dates = pd.to_datetime(history["DataDate"], errors="coerce").dropna()
    if data_dates.empty:
        print("Top 125 history has no valid DataDate; sanity check skipped.")
        return

    current_date = data_dates.max().date()
    history = rerank_current_rows(history, current_date)
    history["Rank"] = pd.to_numeric(history["Rank"], errors="coerce")
    history["MarketCap"] = pd.to_numeric(history["MarketCap"], errors="coerce")
    history = history.sort_values(
        ["SnapshotDate", "ListType", "Rank"],
        ascending=[True, True, True],
    )
    history.to_csv(DATA_FILE, index=False)
    print(f"Top 125 sanity check complete for DataDate {current_date}.")


if __name__ == "__main__":
    main()
