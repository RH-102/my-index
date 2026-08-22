"""Add ICE BofA US High Yield OAS 3-month change to the risk dashboard.

Primary series: FRED BAMLH0A0HYM2 (percent, daily).
The script keeps a local repository cache and uses multiple public endpoints so
one temporary FRED timeout does not prevent the dashboard from updating.
"""

from __future__ import annotations

import csv
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
MIRROR_CSV_URL = f"https://govspending.org/api/export/fred/{SERIES_ID}.csv"
MIRROR_JSON_URL = f"https://govspending.org/api/export/fred/{SERIES_ID}.json"
TIMEZONE = ZoneInfo("America/New_York")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
REQUEST_TIMEOUT = (6, 12)

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

    cols = list(frame.columns)
    if len(cols) < 2:
        return pd.DataFrame(columns=["Date", "Value"])

    date_col = next(
        (c for c in cols if str(c).lower() in {"date", "observation_date", "period"}),
        cols[0],
    )
    value_col = next(
        (c for c in cols if str(c).lower() in {"value", SERIES_ID.lower()}),
        cols[1],
    )

    out = frame[[date_col, value_col]].copy()
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


def parse_metadata_csv(text: str) -> pd.DataFrame:
    """Parse exports that include metadata lines before the date/value rows."""
    observations = []
    for row in csv.reader(StringIO(text)):
        if len(row) < 2:
            continue
        date = pd.to_datetime(str(row[0]).strip(), errors="coerce")
        value = pd.to_numeric(str(row[1]).strip(), errors="coerce")
        if pd.notna(date) and pd.notna(value):
            observations.append({"Date": date, "Value": float(value)})
    return clean_frame(pd.DataFrame(observations))


def extract_json_series(payload) -> pd.DataFrame:
    """Recursively locate date/value observations in a JSON export."""
    observations = []

    def visit(node):
        if isinstance(node, dict):
            lower = {str(k).lower(): k for k in node.keys()}
            date_key = next(
                (lower[k] for k in ("date", "observation_date", "period") if k in lower),
                None,
            )
            value_key = next(
                (lower[k] for k in ("value", SERIES_ID.lower()) if k in lower),
                None,
            )
            if date_key is not None and value_key is not None:
                date = pd.to_datetime(node.get(date_key), errors="coerce")
                value = pd.to_numeric(node.get(value_key), errors="coerce")
                if pd.notna(date) and pd.notna(value):
                    observations.append({"Date": date, "Value": float(value)})

            # Some exports use a date:value dictionary.
            if node and all(not isinstance(v, (dict, list)) for v in node.values()):
                for key, value in node.items():
                    date = pd.to_datetime(key, errors="coerce")
                    num = pd.to_numeric(value, errors="coerce")
                    if pd.notna(date) and pd.notna(num):
                        observations.append({"Date": date, "Value": float(num)})

            for value in node.values():
                visit(value)

        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)
    return clean_frame(pd.DataFrame(observations))


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

    # 1) JSON mirror. This avoids CSV metadata parsing problems and is usually
    # more reliable from GitHub Actions than direct FRED requests.
    try:
        response = requests.get(MIRROR_JSON_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = extract_json_series(response.json())
        if len(frame) >= 60:
            print(f"HY OAS loaded from JSON mirror: {len(frame)} observations")
            return frame
        errors.append(f"JSON mirror returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"JSON mirror failed: {exc}")

    # 2) CSV mirror; parse row-by-row because its export contains metadata.
    try:
        response = requests.get(MIRROR_CSV_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = parse_metadata_csv(response.text)
        if len(frame) >= 60:
            print(f"HY OAS loaded from CSV mirror: {len(frame)} observations")
            return frame
        errors.append(f"CSV mirror returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"CSV mirror failed: {exc}")

    # 3) FRED table page.
    try:
        response = requests.get(FRED_TABLE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = parse_fred_table_html(response.text)
        if len(frame) >= 60:
            print(f"HY OAS loaded from FRED table page: {len(frame)} observations")
            return frame
        errors.append(f"FRED table returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"FRED table failed: {exc}")

    # 4) FRED graph CSV.
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
        errors.append(f"FRED CSV returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"FRED CSV failed: {exc}")

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
        "Source": "FRED BAMLH0A0HYM2 (via local cache / public mirror)",
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
    print(f"HY OAS percentile: {pct:.1f}%")
    print(f"HY OAS rating: {RISK_LABELS[level]}")


if __name__ == "__main__":
    main()
