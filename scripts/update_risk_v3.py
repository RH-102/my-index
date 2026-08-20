"""Market-risk updater with persistent local source caches.

The repository CSV files are the durable data store. Remote sources are used
only to refresh them. If a public source is temporarily unavailable, the last
successful local history is used so the dashboard can still be generated.

Local source files:
- data/risk_source_nasdaq_forward_pe.csv
- data/risk_source_sp500_forward_pe.csv
- data/risk_source_dfii10.csv
- data/risk_source_mortgage.csv
- data/risk_source_credit_gap.csv

Sources:
- Nasdaq-100 / S&P 500 forward P/E: History of Market
- 10Y real yield (DFII10-equivalent underlying series): U.S. Treasury
- Mortgage delinquency (DRSFRMACBS underlying series): Federal Reserve Board
- Credit-to-GDP gap: BIS
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

import update_risk as base
import update_risk_v2 as v2


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
REQUEST_TIMEOUT = (8, 20)

DATA_DIR = base.DATA_DIR
CACHE_NDX = DATA_DIR / "risk_source_nasdaq_forward_pe.csv"
CACHE_SPX = DATA_DIR / "risk_source_sp500_forward_pe.csv"
CACHE_DFII10 = DATA_DIR / "risk_source_dfii10.csv"
CACHE_MORTGAGE = DATA_DIR / "risk_source_mortgage.csv"
CACHE_CREDIT_GAP = DATA_DIR / "risk_source_credit_gap.csv"

TREASURY_ARCHIVE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rate-archives/par-real-yield-curve-rates-2003-2023.csv"
)
TREASURY_YEAR_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)

FED_MORTGAGE_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "filetype=csv&from=&label=include&lastobs=&layout=seriescolumn&"
    "rel=CHGDEL&series=e77af32404312ad2d45dd60dfa36477a&to=&type=package"
)
FED_MORTGAGE_SERIES = "STFBQDSS%STFBAILSS_XEOP_XDO_MA.Q"


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the standard local database format: Date, Value."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Date", "Value"])

    out = frame[["Date", "Value"]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    out = out.dropna().sort_values("Date")
    out = out.drop_duplicates("Date", keep="last").reset_index(drop=True)
    return out


def _load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Value"])
    try:
        return _clean_frame(pd.read_csv(path))
    except Exception as exc:
        print(f"Local cache {path.name} could not be read: {exc}")
        return pd.DataFrame(columns=["Date", "Value"])


def _save_cache(path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    out = _clean_frame(frame)
    if out.empty:
        raise RuntimeError(f"Refusing to save empty cache {path.name}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    disk = out.copy()
    disk["Date"] = disk["Date"].dt.strftime("%Y-%m-%d")
    disk.to_csv(path, index=False)
    print(
        f"Saved local cache {path.name}: {len(out)} observations through "
        f"{out.iloc[-1]['Date'].date()}"
    )
    return out


def _merge_cache(path: Path, fresh: pd.DataFrame) -> pd.DataFrame:
    cached = _load_cache(path)
    combined = pd.concat([cached, _clean_frame(fresh)], ignore_index=True)
    return _save_cache(path, combined)


def _cached_refresh(path: Path, label: str, fetcher, min_rows: int) -> pd.DataFrame:
    """Refresh a source, merge to disk, or use the last good local copy."""
    cached = _load_cache(path)

    try:
        fresh = _clean_frame(fetcher())
        if len(fresh) < min_rows:
            raise RuntimeError(
                f"{label} refresh returned only {len(fresh)} observations"
            )
        return _merge_cache(path, fresh)
    except Exception as exc:
        if len(cached) >= min_rows:
            print(
                f"{label} remote refresh failed ({exc}); using local cache "
                f"through {cached.iloc[-1]['Date'].date()}."
            )
            return cached
        raise RuntimeError(
            f"{label} unavailable and no usable local cache exists: {exc}"
        ) from exc


def fetch_forward_pe_cached(url: str, label: str) -> pd.DataFrame:
    path = CACHE_NDX if "Nasdaq" in label else CACHE_SPX
    min_rows = 500 if "Nasdaq" in label else 800
    return _cached_refresh(
        path,
        label,
        lambda: v2.fetch_forward_pe(url, label),
        min_rows,
    )


def _normalise_treasury_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "date" not in frame.columns or "10 yr" not in frame.columns:
        raise RuntimeError(
            "Unexpected Treasury columns: " + ", ".join(map(str, frame.columns))
        )
    out = frame[["date", "10 yr"]].copy()
    out.columns = ["Date", "Value"]
    return _clean_frame(out)


def _fetch_treasury_year(year: int) -> pd.DataFrame:
    response = requests.get(
        TREASURY_YEAR_URL.format(year=year),
        params={
            "_format": "csv",
            "field_tdr_date_value": str(year),
            "page": "",
            "type": "daily_treasury_real_yield_curve",
        },
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return _normalise_treasury_frame(pd.read_csv(StringIO(response.text)))


def fetch_treasury_full_history() -> pd.DataFrame:
    """Bootstrap the local DFII10-equivalent database from official Treasury data."""
    response = requests.get(
        TREASURY_ARCHIVE_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    frames = [_normalise_treasury_frame(pd.read_csv(StringIO(response.text)))]

    current_year = datetime.now().year
    for year in range(2024, current_year + 1):
        try:
            frames.append(_fetch_treasury_year(year))
        except Exception as exc:
            if year == current_year:
                raise RuntimeError(
                    f"Treasury current-year real yield failed: {exc}"
                ) from exc
            print(f"Treasury real-yield year {year} skipped: {exc}")

    frame = _clean_frame(pd.concat(frames, ignore_index=True))
    if len(frame) < 500:
        raise RuntimeError(
            f"Treasury real-yield history too short: {len(frame)} observations"
        )
    return frame


def fetch_dfii10_cached() -> pd.DataFrame:
    """Use local history; once seeded, refresh only the current Treasury year."""
    cached = _load_cache(CACHE_DFII10)
    current_year = datetime.now().year

    try:
        if len(cached) >= 500:
            fresh = _fetch_treasury_year(current_year)
        else:
            fresh = fetch_treasury_full_history()

        merged = _merge_cache(CACHE_DFII10, fresh)
        print(
            f"Local DFII10 history: {len(merged)} observations, "
            f"{merged.iloc[0]['Date'].date()} to {merged.iloc[-1]['Date'].date()}"
        )
        return merged
    except Exception as exc:
        if len(cached) >= 500:
            print(
                f"DFII10 remote refresh failed ({exc}); using local cache "
                f"through {cached.iloc[-1]['Date'].date()}."
            )
            return cached
        raise


def fetch_fed_mortgage() -> pd.DataFrame:
    response = requests.get(
        FED_MORTGAGE_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    rows = list(csv.reader(StringIO(response.text)))
    target_col = None

    for row in rows:
        for index, cell in enumerate(row):
            if FED_MORTGAGE_SERIES in str(cell):
                target_col = index
                break
        if target_col is not None:
            break

    if target_col is None:
        for row in rows:
            for index, cell in enumerate(row):
                text = str(cell).lower()
                if "delinquency" in text and "residential" in text and "all banks" in text:
                    target_col = index
                    break
            if target_col is not None:
                break

    if target_col is None:
        raise RuntimeError("Could not locate mortgage delinquency series in Fed CSV")

    observations = []
    for row in rows:
        if not row:
            continue
        period = str(row[0]).strip()
        if not re.fullmatch(r"\d{4}Q[1-4]", period) or target_col >= len(row):
            continue
        try:
            observations.append(
                {
                    "Date": pd.Period(period, freq="Q").end_time.normalize(),
                    "Value": float(str(row[target_col]).strip()),
                }
            )
        except Exception:
            continue

    frame = _clean_frame(pd.DataFrame(observations))
    if len(frame) < 40:
        raise RuntimeError(
            f"Mortgage delinquency history too short: {len(frame)} observations"
        )
    return frame


def fetch_mortgage_cached() -> pd.DataFrame:
    return _cached_refresh(
        CACHE_MORTGAGE,
        "Mortgage delinquency",
        fetch_fed_mortgage,
        40,
    )


def fetch_credit_gap_cached() -> pd.DataFrame:
    return _cached_refresh(
        CACHE_CREDIT_GAP,
        "BIS credit-to-GDP gap",
        v2.fetch_bis_credit_gap,
        100,
    )


def fetch_official_series(series_id: str) -> pd.DataFrame:
    if series_id == "DFII10":
        return fetch_dfii10_cached()
    if series_id == "DRSFRMACBS":
        return fetch_mortgage_cached()
    return v2.fetch_fred(series_id)


def relabel_sources() -> None:
    if not base.DASHBOARD_FILE.exists():
        return
    frame = pd.read_csv(base.DASHBOARD_FILE)
    if frame.empty:
        return

    frame["Source"] = frame["Source"].astype(str)
    frame["Source"] = frame["Source"].str.replace(
        "History of Market + FRED DFII10",
        "Local cache: History of Market + U.S. Treasury 10Y real yield",
        regex=False,
    )
    frame["Source"] = frame["Source"].str.replace(
        "FRED DFII10", "Local cache: U.S. Treasury 10Y real yield", regex=False
    )
    frame["Source"] = frame["Source"].str.replace(
        "FRED DRSFRMACBS", "Local cache: Federal Reserve Board CHGDEL", regex=False
    )
    frame.loc[
        frame["Source"].eq("History of Market"), "Source"
    ] = "Local cache: History of Market"
    frame.loc[
        frame["Source"].str.contains("BIS", na=False), "Source"
    ] = "Local cache: BIS Q.US.P.A.C"
    frame.to_csv(base.DASHBOARD_FILE, index=False)


# Patch the base calculator so every source is backed by the repository cache.
base.fetch_forward_pe = fetch_forward_pe_cached
base.fetch_bis_credit_gap = fetch_credit_gap_cached
base.fetch_fred = fetch_official_series


if __name__ == "__main__":
    try:
        base.main()
        relabel_sources()
    except Exception as exc:
        print(f"Risk refresh skipped: {exc}")
        print(
            "Any source caches successfully refreshed before the error were kept; "
            "the existing dashboard files were retained."
        )
