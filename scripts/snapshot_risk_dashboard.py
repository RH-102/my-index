"""Persist one daily snapshot of every row in risk_dashboard.csv.

The history is intentionally long-format so new indicators can be added later
without changing the file schema. Re-running on the same Eastern Time date
replaces that day's snapshot instead of creating duplicates.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DASHBOARD_FILE = DATA_DIR / "risk_dashboard.csv"
HISTORY_FILE = DATA_DIR / "risk_dashboard_history.csv"
TIMEZONE = ZoneInfo("America/New_York")

OUTPUT_COLUMNS = [
    "SnapshotDate",
    "SnapshotTime",
    "RowType",
    "Indicator",
    "NumericValue",
    "Unit",
    "CurrentValue",
    "HistoryPercentile",
    "RecentChange",
    "Rating",
    "RatingLevel",
    "DataDate",
    "Source",
    "Note",
    "DashboardUpdatedAt",
]


def parse_numeric_value(value) -> tuple[float | None, str]:
    """Extract a machine-readable number/unit while preserving the source text."""
    text = "" if pd.isna(value) else str(value).strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    numeric = float(match.group(0)) if match else None

    unit = ""
    lowered = text.lower()
    if "bp" in lowered:
        unit = "bp"
    elif "pp" in lowered:
        unit = "pp"
    elif "%" in text:
        unit = "%"
    elif re.search(r"\d\s*x\b", lowered) or lowered.endswith("x"):
        unit = "x"

    return numeric, unit


def main() -> None:
    if not DASHBOARD_FILE.exists():
        raise RuntimeError("risk_dashboard.csv does not exist")

    dashboard = pd.read_csv(DASHBOARD_FILE)
    if dashboard.empty or "Indicator" not in dashboard.columns:
        raise RuntimeError("risk_dashboard.csv is empty or malformed")

    now = datetime.now(TIMEZONE)
    snapshot_date = now.date().isoformat()
    snapshot_time = now.isoformat(timespec="seconds")

    rows = []
    for _, row in dashboard.iterrows():
        indicator = row.get("Indicator", "")
        if pd.isna(indicator) or str(indicator).strip() == "":
            continue

        numeric, unit = parse_numeric_value(row.get("CurrentValue", ""))
        rows.append(
            {
                "SnapshotDate": snapshot_date,
                "SnapshotTime": snapshot_time,
                "RowType": row.get("RowType", ""),
                "Indicator": indicator,
                "NumericValue": numeric,
                "Unit": unit,
                "CurrentValue": row.get("CurrentValue", ""),
                "HistoryPercentile": row.get("HistoryPercentile", ""),
                "RecentChange": row.get("RecentChange", ""),
                "Rating": row.get("Rating", ""),
                "RatingLevel": row.get("RatingLevel", ""),
                "DataDate": row.get("DataDate", ""),
                "Source": row.get("Source", ""),
                "Note": row.get("Note", ""),
                "DashboardUpdatedAt": row.get("UpdatedAt", ""),
            }
        )

    new = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)

    if HISTORY_FILE.exists():
        old = pd.read_csv(HISTORY_FILE)
        old = old.reindex(columns=OUTPUT_COLUMNS)
        # One canonical snapshot per Eastern Time calendar day.
        old = old[old["SnapshotDate"].astype(str) != snapshot_date]
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new

    combined = combined.sort_values(["SnapshotDate", "RowType", "Indicator"])
    combined.to_csv(HISTORY_FILE, index=False)

    print(
        f"Saved {len(new)} dashboard rows for {snapshot_date}; "
        f"history now has {len(combined)} rows."
    )


if __name__ == "__main__":
    main()
