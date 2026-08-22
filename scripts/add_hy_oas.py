"""Add ICE BofA US High Yield OAS 3-month change to the risk dashboard.

FRED series: BAMLH0A0HYM2 (percent, daily).
The repository CSV is the durable local cache. Remote sources are only used
for refreshes; if they fail, the last good local cache is reused.
"""

from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
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
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_TABLE_URL = f"https://fred.stlouisfed.org/data/{SERIES_ID}"
TIMEZONE = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
REQUEST_TIMEOUT = (6, 15)

RISK_LABELS = {
    0: "🟢安全",
    1: "🟡注意",
    2: "🟠高风险",
    3: "🔴极高风险",
}


class SimpleTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = []
        self.in_cell = False

    def handle_starttag(self, tag, attrs):
        if tag in {"td", "th"}:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.in_cell:
            self.current_row.append("".join(self.current_cell).strip())
            self.in_cell = False
        elif tag == "tr":
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Date", "Value"])
    out = frame.iloc[:, :2].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return (
        out.dropna()
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


def parse_fred_table_html(text: str) -> pd.DataFrame:
    parser = SimpleTableParser()
    parser.feed(text)
    rows = []
    for row in parser.rows:
        if len(row) < 2:
            continue
        date = pd.to_datetime(row[0], errors="coerce")
        value = pd.to_numeric(row[1], errors="coerce")
        if pd.notna(date) and pd.notna(value):
            rows.append({"Date": date, "Value": float(value)})
    return clean_frame(pd.DataFrame(rows))


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


def fetch_remote_hy_oas() -> pd.DataFrame:
    errors = []

    # FRED's static table page is often more reliable than fredgraph.csv.
    try:
        response = requests.get(
            FRED_TABLE_URL,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        frame = parse_fred_table_html(response.text)
        if len(frame) >= 60:
            print(f"HY OAS loaded from FRED table page: {len(frame)} observations")
            return frame
        errors.append(f"table page returned only {len(frame)} observations")
    except Exception as exc:
        errors.append(f"table page failed: {exc}")

    try:
        response = requests.get(
            FRED_CSV_URL,
            params={"id": SERIES_ID},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        frame = clean_frame(pd.read_csv(StringIO(response.text)))
        if len(frame) >= 60:
            print(f"HY OAS loaded from FRED CSV: {len(frame)} observations")
            return frame
        errors.append(f"CSV returned only {len(frame)} observations")
    except Exception as exc:
        errors.append(f"CSV failed: {exc}")

    raise RuntimeError("; ".join(errors))


def fetch_hy_oas() -> pd.DataFrame:
    cached = load_cache()
    try:
        fresh = fetch_remote_hy_oas()
        return save_cache(pd.concat([cached, fresh], ignore_index=True))
    except Exception as exc:
        if len(cached) >= 60:
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
            "Rating thresholds: <50bp safe; 50-<100bp watch; "
            "100-<150bp high risk; >=150bp extreme risk."
        ),
        "UpdatedAt": updated_at,
    }

    dashboard = pd.concat([dashboard, pd.DataFrame([row])], ignore_index=True)
    dashboard.to_csv(DASHBOARD_FILE, index=False)

    hy_history = changes[["Date", "ChangeBp"]].copy()
    hy_history["Date"] = pd.to_datetime(hy_history["Date"]).dt.date.astype(str)
    hy_history = hy_history.rename(columns={"ChangeBp": "HY_OAS_3M_Change_bp"})
    if HISTORY_FILE.exists():
        history = pd.read_csv(HISTORY_FILE)
        if "HY_OAS_3M_Change_bp" in history.columns:
            history = history.drop(columns=["HY_OAS_3M_Change_bp"])
        history = history.merge(hy_history, on="Date", how="outer").sort_values("Date")
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
