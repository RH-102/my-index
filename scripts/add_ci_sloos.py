"""Add C&I SLOOS tightening-standards indicator to the risk dashboard.

Primary concept: net percentage of domestic banks tightening standards for
C&I loans to large and middle-market firms. The repository CSV is the durable
local cache. The script prefers the Federal Reserve Board source, with FRED and
a public FRED mirror as fallbacks.
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
CACHE_FILE = DATA_DIR / "risk_source_ci_sloos.csv"

SERIES_ID = "DRTSCILM"
FED_SERIES_CODE = "SUBLPDCILS_N.Q"
FED_OUTPUT_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "filetype=csv&from=&label=include&lastobs=&layout=seriescolumn&"
    "rel=SLOOS&series=2f17df6d07977715676ad71c7a655bbd&to=&type=package"
)
FRED_TABLE_URL = f"https://fred.stlouisfed.org/data/{SERIES_ID}"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MIRROR_CSV_URL = f"https://govspending.org/api/export/fred/{SERIES_ID}.csv"
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
    date_col = next((c for c in cols if str(c).lower() in {"date", "observation_date"}), cols[0])
    value_col = next(
        (c for c in cols if str(c).lower() in {"value", SERIES_ID.lower()}),
        cols[1] if len(cols) > 1 else cols[0],
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


def parse_fed_csv(text: str) -> pd.DataFrame:
    rows = list(csv.reader(StringIO(text)))
    target_col = None

    for row in rows:
        for idx, cell in enumerate(row):
            if FED_SERIES_CODE in str(cell):
                target_col = idx
                break
        if target_col is not None:
            break

    if target_col is None:
        raise RuntimeError("Could not locate C&I SLOOS series in Federal Reserve CSV")

    observations = []
    for row in rows:
        if not row or target_col >= len(row):
            continue
        period = str(row[0]).strip()
        if not period or "Q" not in period:
            continue
        try:
            date = pd.Period(period, freq="Q").start_time.normalize()
            value = float(str(row[target_col]).strip())
            observations.append({"Date": date, "Value": value})
        except Exception:
            continue

    frame = clean_frame(pd.DataFrame(observations))
    if len(frame) < 100:
        raise RuntimeError(f"Federal Reserve SLOOS history too short: {len(frame)} rows")
    return frame


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


def fetch_remote_ci_sloos() -> pd.DataFrame:
    errors = []

    # 1) Federal Reserve Board official data download.
    try:
        response = requests.get(FED_OUTPUT_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = parse_fed_csv(response.text)
        print(f"C&I SLOOS loaded from Federal Reserve Board: {len(frame)} observations")
        return frame
    except Exception as exc:
        errors.append(f"Federal Reserve failed: {exc}")

    # 2) FRED table page.
    try:
        response = requests.get(FRED_TABLE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = parse_fred_table_html(response.text)
        if len(frame) >= 100:
            print(f"C&I SLOOS loaded from FRED table page: {len(frame)} observations")
            return frame
        errors.append(f"FRED table returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"FRED table failed: {exc}")

    # 3) Public CSV mirror of the FRED series.
    try:
        response = requests.get(MIRROR_CSV_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        frame = clean_frame(pd.read_csv(StringIO(response.text)))
        if len(frame) >= 100:
            print(f"C&I SLOOS loaded from FRED mirror: {len(frame)} observations")
            return frame
        errors.append(f"mirror returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"mirror failed: {exc}")

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
        if len(frame) >= 100:
            print(f"C&I SLOOS loaded from FRED CSV: {len(frame)} observations")
            return frame
        errors.append(f"FRED CSV returned {len(frame)} rows")
    except Exception as exc:
        errors.append(f"FRED CSV failed: {exc}")

    raise RuntimeError("; ".join(errors))


def fetch_ci_sloos() -> pd.DataFrame:
    cached = load_cache()
    try:
        fresh = fetch_remote_ci_sloos()
        return save_cache(pd.concat([cached, fresh], ignore_index=True))
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
        "Source": "Federal Reserve Board SLOOS / FRED DRTSCILM",
        "Note": (
            "Net percentage of domestic banks tightening C&I lending standards "
            "for large and middle-market firms. Thresholds: <=5% safe; "
            ">5%-10% watch; >10%-20% high risk; >20% extreme risk."
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
        history = history.merge(sloos_history, on="Date", how="outer").sort_values("Date")
    else:
        history = sloos_history
    history.to_csv(HISTORY_FILE, index=False)

    print("SUCCESS")
    print(f"C&I SLOOS current: {current:.1f}%")
    print(f"C&I SLOOS percentile: {pct:.1f}%")
    print(f"C&I SLOOS rating: {RISK_LABELS[level]}")


if __name__ == "__main__":
    main()
