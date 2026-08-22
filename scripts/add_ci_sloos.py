"""Add C&I SLOOS tightening-standards indicator to the risk dashboard.

FRED series: DRTSCILM
Net Percentage of Domestic Banks Tightening Standards for Commercial and
Industrial Loans to Large and Middle-Market Firms (percent, quarterly).

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
CACHE_FILE = DATA_DIR / "risk_source_ci_sloos.csv"

SERIES_ID = "DRTSCILM"
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
        print(f"Could not read local C&I SLOOS cache: {exc}")
        return pd.DataFrame(columns=["Date", "Value"])


def save_cache(frame: pd.DataFrame) -> pd.DataFrame:
    out = clean_frame(frame)
    if out.empty:
        raise RuntimeError("Refusing to save an empty C&I SLOOS cache")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    disk = out.copy()
    disk["Date"] = disk["Date"].dt.strftime("%Y-%m-%d")
    disk.to_csv(CACHE_FILE, index=False)
    print(
        f"Saved {CACHE_FILE.name}: {len(out)} observations through "
        f"{out.iloc[-1]['Date'].date()}"
    )
    return out


def fetch_ci_sloos() -> pd.DataFrame:
    cached = load_cache()

    try:
        response = requests.get(
            FRED_URL,
            params={"id": SERIES_ID},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        fresh = clean_frame(pd.read_csv(StringIO(response.text)))
        if len(fresh) < 100:
            raise RuntimeError(
                f"FRED returned only {len(fresh)} C&I SLOOS observations"
            )

        combined = pd.concat([cached, fresh], ignore_index=True)
        return save_cache(combined)

    except Exception as exc:
        if len(cached) >= 100:
            print(
                f"C&I SLOOS remote refresh failed ({exc}); using local cache "
                f"through {cached.iloc[-1]['Date'].date()}."
            )
            return cached
        raise RuntimeError(
            f"C&I SLOOS unavailable and no usable local cache exists: {exc}"
        ) from exc


def risk_level(value: float) -> int:
    # User-defined thresholds. To remove boundary overlap:
    # <=5 safe; >5 to <=10 watch; >10 to <=20 high; >20 extreme.
    if value > 20:
        return 3
    if value > 10:
        return 2
    if value > 5:
        return 1
    return 0


def percentile_rank(values: pd.Series, current: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan")
    return float((clean <= current).mean() * 100.0)


def main() -> None:
    if not DASHBOARD_FILE.exists():
        print("risk_dashboard.csv does not exist yet; C&I SLOOS enrichment skipped.")
        return

    try:
        sloos = fetch_ci_sloos()
    except Exception as exc:
        print(f"C&I SLOOS enrichment skipped: {exc}")
        return

    current = float(sloos.iloc[-1]["Value"])
    current_date = pd.Timestamp(sloos.iloc[-1]["Date"])
    previous = float(sloos.iloc[-2]["Value"]) if len(sloos) >= 2 else float("nan")
    year_ago = float(sloos.iloc[-5]["Value"]) if len(sloos) >= 5 else float("nan")
    qoq = current - previous if pd.notna(previous) else float("nan")
    pct = percentile_rank(sloos["Value"], current)
    level = risk_level(current)
    updated_at = datetime.now(TIMEZONE).isoformat(timespec="seconds")

    dashboard = pd.read_csv(DASHBOARD_FILE)
    dashboard = dashboard[
        ~dashboard["Indicator"].isin(["C&I SLOOS", "Mortgage Delinquency"])
    ].copy()

    previous_text = "--" if pd.isna(previous) else f"{previous:.1f}%"
    year_text = "--" if pd.isna(year_ago) else f"{year_ago:.1f}%"
    qoq_text = "--" if pd.isna(qoq) else f"{qoq:+.1f}pp"

    row = {
        "RowType": "Metric",
        "Indicator": "C&I SLOOS",
        "CurrentValue": f"{current:.1f}%",
        "HistoryPercentile": f"{pct:.1f}%",
        "RecentChange": f"上季 {previous_text} | QoQ {qoq_text} | 1年前 {year_text}",
        "Rating": RISK_LABELS[level],
        "RatingLevel": level,
        "DataDate": current_date.date().isoformat(),
        "Source": "Local cache: FRED DRTSCILM",
        "Note": (
            "Net percentage of domestic banks tightening C&I lending standards "
            "for large and middle-market firms. Rating thresholds: <=5% safe; "
            ">5% to <=10% watch; >10% to <=20% high risk; >20% extreme risk."
        ),
        "UpdatedAt": updated_at,
    }

    dashboard = pd.concat([dashboard, pd.DataFrame([row])], ignore_index=True)
    dashboard.to_csv(DASHBOARD_FILE, index=False)

    sloos_history = sloos[["Date", "Value"]].copy()
    sloos_history["Date"] = pd.to_datetime(sloos_history["Date"]).dt.date.astype(str)
    sloos_history = sloos_history.rename(columns={"Value": "CI_SLOOS"})

    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
        if "CI_SLOOS" in history.columns:
            history = history.drop(columns=["CI_SLOOS"])
        history = history.merge(sloos_history, on="Date", how="outer")
        history = history.sort_values("Date")
    else:
        history = sloos_history

    history.to_csv(HISTORY_FILE, index=False)

    print("SUCCESS")
    print(f"C&I SLOOS current: {current:.1f}%")
    print(f"C&I SLOOS percentile: {pct:.1f}%")
    print(f"C&I SLOOS rating: {RISK_LABELS[level]}")


if __name__ == "__main__":
    main()
