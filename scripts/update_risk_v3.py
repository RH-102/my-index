"""Market-risk updater with official-source fallbacks.

This wrapper keeps the calculation/rating logic in update_risk.py, but avoids
blocking on FRED when GitHub Actions cannot reach fred.stlouisfed.org.

Sources:
- Nasdaq-100 / S&P 500 forward P/E: History of Market (via update_risk_v2)
- 10Y real yield: U.S. Treasury Daily Treasury Par Real Yield Curve Rates
- Mortgage delinquency: Federal Reserve Board CHGDEL Data Download Program
- Credit-to-GDP gap: BIS (via update_risk_v2)
"""

from __future__ import annotations

import csv
import re
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

import update_risk as base
import update_risk_v2 as v2


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
REQUEST_TIMEOUT = (8, 20)

TREASURY_ARCHIVE_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rate-archives/par-real-yield-curve-rates-2003-2023.csv"
)
TREASURY_YEAR_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
    "daily-treasury-rates.csv/{year}/all"
)

# Official Federal Reserve Board preformatted package: Delinquency rates / All banks.
FED_MORTGAGE_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "filetype=csv&from=&label=include&lastobs=&layout=seriescolumn&"
    "rel=CHGDEL&series=e77af32404312ad2d45dd60dfa36477a&to=&type=package"
)
FED_MORTGAGE_SERIES = "STFBQDSS%STFBAILSS_XEOP_XDO_MA.Q"


def _normalise_treasury_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if "date" not in frame.columns or "10 yr" not in frame.columns:
        raise RuntimeError(
            "Unexpected Treasury columns: " + ", ".join(map(str, frame.columns))
        )
    out = frame[["date", "10 yr"]].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return out.dropna()


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


def fetch_treasury_dfii10() -> pd.DataFrame:
    """Fetch long history with one archive request plus recent yearly CSVs."""

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
                raise RuntimeError(f"Treasury current-year real yield failed: {exc}") from exc
            print(f"Treasury real-yield year {year} skipped: {exc}")

    frame = pd.concat(frames, ignore_index=True)
    frame = (
        frame.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )

    if len(frame) < 500:
        raise RuntimeError(f"Treasury real-yield history too short: {len(frame)} observations")
    if int(frame.iloc[-1]["Date"].year) < current_year:
        raise RuntimeError("Treasury current-year real-yield data is unavailable")

    print(
        f"Treasury 10Y real yield: {len(frame)} observations, "
        f"{frame.iloc[0]['Date'].date()} to {frame.iloc[-1]['Date'].date()}"
    )
    return frame


def fetch_fed_mortgage() -> pd.DataFrame:
    """Fetch the source series behind FRED DRSFRMACBS from the Fed Board."""

    response = requests.get(FED_MORTGAGE_URL, headers=HEADERS, timeout=REQUEST_TIMEOUT)
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
            observations.append({
                "Date": pd.Period(period, freq="Q").end_time.normalize(),
                "Value": float(str(row[target_col]).strip()),
            })
        except Exception:
            continue

    frame = pd.DataFrame(observations)
    if frame.empty:
        raise RuntimeError("Fed mortgage CSV contained no quarterly observations")
    frame = (
        frame.sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < 40:
        raise RuntimeError(f"Mortgage delinquency history too short: {len(frame)} observations")

    print(
        f"Fed mortgage delinquency: {len(frame)} observations, "
        f"{frame.iloc[0]['Date'].date()} to {frame.iloc[-1]['Date'].date()}"
    )
    return frame


def fetch_official_series(series_id: str) -> pd.DataFrame:
    if series_id == "DFII10":
        return fetch_treasury_dfii10()
    if series_id == "DRSFRMACBS":
        return fetch_fed_mortgage()
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
        "History of Market + U.S. Treasury 10Y real yield",
        regex=False,
    )
    frame["Source"] = frame["Source"].str.replace(
        "FRED DFII10", "U.S. Treasury 10Y real yield", regex=False
    )
    frame["Source"] = frame["Source"].str.replace(
        "FRED DRSFRMACBS", "Federal Reserve Board CHGDEL", regex=False
    )
    frame.to_csv(base.DASHBOARD_FILE, index=False)


base.fetch_forward_pe = v2.fetch_forward_pe
base.fetch_bis_credit_gap = v2.fetch_bis_credit_gap
base.fetch_fred = fetch_official_series


if __name__ == "__main__":
    try:
        base.main()
        relabel_sources()
    except Exception as exc:
        print(f"Risk refresh skipped: {exc}")
        print("Existing risk_dashboard.csv and risk_history.csv were retained unchanged.")
