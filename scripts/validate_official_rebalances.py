from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

REBALANCES_FILE = DATA_DIR / "rebalances.csv"
OFFICIAL_CHECKS_FILE = DATA_DIR / "official_share_checks.csv"

BACKFILL_START = "2026-08-12"
ANCHOR_SYMBOL = "NVDA"
MAX_RELATIVE_QTY_ERROR = 0.001  # 0.10%


rebalances = pd.read_csv(REBALANCES_FILE, dtype=str).fillna("")
official = pd.read_csv(OFFICIAL_CHECKS_FILE, dtype=str).fillna("")

required_rebalance_columns = {
    "EffectiveDate",
    "Symbol",
    "Name",
    "Quantity",
    "SourceType",
    "SourceSnapshotDate",
}
missing = required_rebalance_columns - set(rebalances.columns)
if missing:
    raise RuntimeError(
        "rebalances.csv is missing required columns: "
        + ", ".join(sorted(missing))
    )

required_official_columns = {
    "SnapshotDate",
    "Symbol",
    "OfficialSharesOutstanding",
    "ADRRatio",
    "ListedShareEquivalent",
    "SharesSourceURL",
    "SharesSourceDate",
}
missing = required_official_columns - set(official.columns)
if missing:
    raise RuntimeError(
        "official_share_checks.csv is missing required columns: "
        + ", ".join(sorted(missing))
    )

rebalances["Symbol"] = rebalances["Symbol"].str.strip().str.upper()
rebalances["SourceType"] = rebalances["SourceType"].str.strip()
rebalances["SourceSnapshotDate"] = rebalances["SourceSnapshotDate"].str.strip()
rebalances["Quantity"] = pd.to_numeric(rebalances["Quantity"], errors="raise")

official["Symbol"] = official["Symbol"].str.strip().str.upper()
official["SnapshotDate"] = official["SnapshotDate"].str.strip()


def positive_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def effective_listed_shares(row):
    explicit = positive_number(row["ListedShareEquivalent"])
    if explicit is not None:
        return explicit

    shares = positive_number(row["OfficialSharesOutstanding"])
    if shares is None:
        raise RuntimeError(
            f"{row['SnapshotDate']} {row['Symbol']}: missing positive "
            "OfficialSharesOutstanding or ListedShareEquivalent."
        )

    ratio = positive_number(row["ADRRatio"])
    if ratio is None:
        raise RuntimeError(
            f"{row['SnapshotDate']} {row['Symbol']}: missing positive ADRRatio."
        )

    return shares / ratio


effective_dates = sorted(rebalances["EffectiveDate"].unique())
if not effective_dates:
    raise RuntimeError("rebalances.csv contains no snapshots.")

if effective_dates[0] != BACKFILL_START:
    raise RuntimeError(
        f"The first rebalance snapshot must be {BACKFILL_START}."
    )

for effective_date in effective_dates:
    snapshot = rebalances.loc[
        rebalances["EffectiveDate"] == effective_date
    ].copy()

    source_types = set(snapshot["SourceType"])
    source_dates = set(snapshot["SourceSnapshotDate"])

    if effective_date == BACKFILL_START:
        if source_types != {"Initial"}:
            raise RuntimeError(
                f"{effective_date}: inception snapshot must use SourceType=Initial."
            )
        continue

    if source_types != {"Official"}:
        raise RuntimeError(
            f"{effective_date}: every post-inception rebalance must use "
            "SourceType=Official. Temporary/provider market-cap data cannot "
            "drive the actual index."
        )

    if len(source_dates) != 1 or "" in source_dates:
        raise RuntimeError(
            f"{effective_date}: all rows must share one nonblank "
            "SourceSnapshotDate."
        )

    source_snapshot_date = next(iter(source_dates))
    ledger = official.loc[
        official["SnapshotDate"] == source_snapshot_date
    ].copy()

    if ledger.empty:
        raise RuntimeError(
            f"{effective_date}: no official share ledger exists for "
            f"{source_snapshot_date}."
        )

    if ledger.duplicated("Symbol").any():
        dupes = sorted(
            ledger.loc[ledger.duplicated("Symbol", keep=False), "Symbol"].unique()
        )
        raise RuntimeError(
            f"{source_snapshot_date}: duplicate official share rows: "
            + ", ".join(dupes)
        )

    required_symbols = set(snapshot["Symbol"])
    available_symbols = set(ledger["Symbol"])
    missing_symbols = sorted(required_symbols - available_symbols)
    if missing_symbols:
        raise RuntimeError(
            f"{effective_date}: missing official share checks for: "
            + ", ".join(missing_symbols)
        )

    ledger = ledger.set_index("Symbol")

    for symbol in sorted(required_symbols):
        row = ledger.loc[symbol]
        if not str(row["SharesSourceURL"]).strip():
            raise RuntimeError(
                f"{source_snapshot_date} {symbol}: SharesSourceURL is required."
            )
        if not str(row["SharesSourceDate"]).strip():
            raise RuntimeError(
                f"{source_snapshot_date} {symbol}: SharesSourceDate is required."
            )

    if ANCHOR_SYMBOL not in required_symbols:
        raise RuntimeError(
            f"{effective_date}: {ANCHOR_SYMBOL} must be in every actual snapshot."
        )

    anchor_shares = effective_listed_shares(ledger.loc[ANCHOR_SYMBOL])

    for record in snapshot.itertuples(index=False):
        expected_quantity = (
            effective_listed_shares(ledger.loc[record.Symbol]) / anchor_shares
        )
        actual_quantity = float(record.Quantity)

        relative_error = abs(actual_quantity / expected_quantity - 1.0)
        if relative_error > MAX_RELATIVE_QTY_ERROR:
            raise RuntimeError(
                f"{effective_date} {record.Symbol}: Quantity {actual_quantity:.10f} "
                f"does not match official-share target {expected_quantity:.10f}; "
                f"relative error={relative_error:.3%}."
            )

    print(
        f"Validated official rebalance {effective_date} from "
        f"snapshot {source_snapshot_date}: {len(snapshot)} holdings."
    )

print("Official rebalance validation passed.")
