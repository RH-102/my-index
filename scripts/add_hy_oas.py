"""Add ICE BofA US High Yield OAS 3-month change to the risk dashboard.

FRED series: BAMLH0A0HYM2 (percent, daily).
The repository CSV is the durable local cache. FRED is used only to refresh it;
if FRED is temporarily unavailable, the last good local history is reused.
"""

from __future__ import annotations

from datetime import datetime
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DASHBOARD_FILE = DATA_DIR / "risk_dashboard.csv"
HISTORY_FILE = DATA_DIR / "risk_history.csv"
CACHE_FILE = DATA_DIR / "risk_source_hy_oas.csv"

SERIES_ID = "BAMLH0A0HYM2"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TIMEZONE = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
REQUEST_TIMEOUT = (8, 20)

RISK_LABELS = {
    0: "🟢安全",
    1: "🟡注意",
    2: "🟠高风险",
    3: "🔴极高风险",
}


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    out = frame.iloc[:, :2].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna().sort_values("Date")
    out = out.drop_duplicates("Date", keep="last").reset_index(drop=True)
    return out


def load_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return pd.DataFrame(columns=["Date", "Value"])
    try:
        return clean_frame(pd.read_csv(CACHE_FILE))
    except Exception as exc:
        print(f"Could not read local HY OAS cache: {exc}")
        return pd.DataFrame(columns=["Date", "Value"])


def save_cache(frame: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(frame)
    if out.empty:
        raise RuntimeError("Refusing to save an empty HY OAS cache")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    disk = out.copy()
    disk["Date"] = disk["Date"].dt.strftime("%Y-%m-%d")
    disk.to_csv(CACHE_FILE, index=False)
    print(
        f"Saved {CACHE_FILE.name}: {len(out)} observations through "
        f"{out.iloc[-1]['Date'].date()}"
    )
    return out


def fetch_hy_oas() -> pd.DataFrame:
    cached = load_cache()
    params = {"id": SERIES_ID}

    # Once the full local history exists, only request recent observations.
    if len(cached) >= 250:
        start = (cached.iloc[-1]["Date"] - pd.Timedelta(days=14)).date().isoformat()
        params["cosd"] = start

    try:
        response = requests.get(
            FRED_URL,
            params=params,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        fresh = clean_frame(pd.read_csv(StringIO(response.text)))
        if fresh.empty:
            raise RuntimeError("FRED returned no HY OAS observations")

        combined = pd.concat([cached, fresh], ignore_index=True)
        return save_cache(combined)

    except Exception as exc:
        if len(cached) >= 250:
            print(
                f"HY OAS remote refresh failed ({exc}); using local cache "
                f"through {cached.iloc[-1]['Date'].date()}."
            )
            return cached
        raise RuntimeError(
            f"HY OAS unavailable and no usable local cache exists: {exc}"
        ) from exc


def rolling_3m_change(frame: pd.DataFrame) -> pd.DataFrame:
    current = frame[["Date", "Value"]].copy().sort_values("Date")
    current["LookupDate"] = current["Date"] - pd.DateOffset(months=3)

    past = frame[["Date", "Value"]].copy().sort_values("Date")
    past.columns = ["PastDate", "PastValue"]

    merged = pd.merge_asof(
        current.sort_values("LookupDate"),
        past,
        left_on="LookupDate",
        right_on="PastDate",
        direction="backward",
        tolerance=pd.Timedelta(days=10),
    )
    merged["ChangeBp"] = (merged["Value"] - merged["PastValue"]) * 100.0
    return merged.dropna(subset=["PastValue", "ChangeBp"]).sort_values("Date")


def risk_level(change_bp: float) -> int:
    # User-defined absolute thresholds, in basis points.
    # <50 / 50-100 / 100-150 / >=150
    if change_bp >= 150:
        return 3
    if change_bp >= 100:
        return 2
    if change_bp >= 50:
        return 1
    return 0


def percentile_rank(values: pd.Series, current: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= current).mean() * 100.0)


def main() -> None:
    if not DASHBOARD_FILE.exists():
        print("risk_dashboard.csv does not exist yet; HY OAS enrichment skipped.")
        return

    try:
        hy = fetch_hy_oas()
    except Exception as exc:
        print(f"HY OAS enrichment skipped: {exc}")
        return

    changes = rolling_3m_change(hy)
    if changes.empty:
        print("HY OAS history is too short to calculate a 3-month change.")
        return

    latest = changes.iloc[-1]
    change_bp = float(latest["ChangeBp"])
    current_oas = float(latest["Value"])
    past_oas = float(latest["PastValue"])
    pct = percentile_rank(changes["ChangeBp"], change_bp)
    level = risk_level(change_bp)
    updated_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")

    dashboard = pd.read_csv(DASHBOARD_FILE)

    # Keep the prior user preference: do not publish Mortgage Delinquency.
    dashboard = dashboard[
        ~dashboard["Indicator"].isin(["HY OAS 3M Change", "Mortgage Delinquency"])
    ].copy()

    row = {
        "RowType": "Metric",
        "Indicator": "HY OAS 3M Change",
        "CurrentValue": f"{change_bp:+.0f}bp",
        "HistoryPercentile": f"{pct:.1f}%",
        "RecentChange": f"当前 OAS {current_oas:.2f}% | 3M前 {past_oas:.2f}%",
        "Rating": RISK_LABELS[level],
        "RatingLevel": level,
        "DataDate": pd.Timestamp(latest["Date"]).date().isoformat(),
        "Source": "Local cache: FRED BAMLH0A0HYM2",
        "Note": (
            "ICE BofA US High Yield Index Option-Adjusted Spread 3-month change. "
            "3M Change = current HY OAS minus approximately 3 months earlier. "
            "Rating uses absolute change thresholds: <50bp safe; 50-<100bp watch; "
            "100-<150bp high risk; >=150bp extreme risk."
        ),
        "UpdatedAt": updated_at,
    }

    dashboard = pd.concat([dashboard, pd.DataFrame([row])], ignore_index=True)
    dashboard.to_csv(DASHBOARD_FILE, index=False)

    # Also keep the full rolling 3M-change history locally for later analysis.
    hy_history = changes[["Date", "ChangeBp"]].copy()
    hy_history["Date"] = pd.to_datetime(hy_history["Date"]).dt.date.astype(str)
    hy_history = hy_history.rename(columns={"ChangeBp": "HY_OAS_3M_Change_bp"})

    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
        if "HY_OAS_3M_Change_bp" in history.columns:
            history = history.drop(columns=["HY_OAS_3M_Change_bp"])
        history = history.merge(hy_history, on="Date", how="outer")
        history = history.sort_values("Date")
    else:
        history = hy_history

    history.to_csv(HISTORY_FILE, index=False)

    print("SUCCESS")
    print(f"HY OAS current: {current_oas:.2f}%")
    print(f"HY OAS 3M ago: {past_oas:.2f}%")
    print(f"HY OAS 3M change: {change_bp:+.0f}bp")
    print(f"HY OAS 3M-change percentile: {pct:.1f}%")
    print(f"HY OAS rating: {RISK_LABELS[level]}")


if __name__ == "__main__":
    main()
