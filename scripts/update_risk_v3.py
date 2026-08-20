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
import xml.etree.ElementTree as ET
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
TREASURY_XML_URL = (
    "https://home.treasury.gov/resource-center/data-chart-center/"
    "interest-rates/pages/xml"
)

FED_MORTGAGE_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "filetype=csv&from=&label=include&lastobs=&layout=seriescolumn&"
    "rel=CHGDEL&series=9c215a53d083d9416a3f3625bcb1223d&to=&type=package"
)
FED_MORTGAGE_SERIES = "STFBQDSS%STFBAILSS_XEOP_XDO_MA.Q"


def _local_name(tag: str) -> str:
    return tag.split("}")[-1].upper()


def _parse_treasury_xml(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    rows = []

    for entry in root.iter():
        if _local_name(entry.tag) != "ENTRY":
            continue

        values = {}
        for element in entry.iter():
            name = _local_name(element.tag)
            text = (element.text or "").strip()
            if text:
                values[name] = text

        date_text = values.get("NEW_DATE") or values.get("QUOTE_DATE") or values.get("DATE")
        yield_text = values.get("BC_10YEAR") or values.get("BC_10_YEAR") or values.get("10_YEAR")

        if date_text is None or yield_text is None:
            continue

        try:
            rows.append({
                "Date": pd.to_datetime(date_text, errors="raise"),
                "Value": float(yield_text),
            })
        except Exception:
            continue

    return rows


def _fetch_treasury_year(year: int) -> pd.DataFrame:
    response = requests.get(
        TREASURY_XML_URL,
        params={
            "data": "daily_treasury_real_yield_curve",
            "field_tdr_date_value": str(year),
        },
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    rows = _parse_treasury_xml(response.text)
    if not rows:
        raise RuntimeError(f"Treasury returned no real-yield data for {year}")
    return pd.DataFrame(rows)


def fetch_treasury_dfii10() -> pd.DataFrame:
    """Fetch long history quickly: one archive file + recent yearly feeds."""

    frames = []

    response = requests.get(
        TREASURY_ARCHIVE_URL,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    archive = pd.read_csv(StringIO(response.text))
    archive.columns = [str(c).strip().lower() for c in archive.columns]
    if "date" not in archive.columns or "10 yr" not in archive.columns:
        raise RuntimeError(
            "Unexpected Treasury archive columns: " + ", ".join(map(str, archive.columns))
        )
    archive = archive[["date", "10 yr"]].copy()
    archive.columns = ["Date", "Value"]
    frames.append(archive)

    current_year = datetime.now().year
    for year in range(2024, current_year + 1):
        try:
            frames.append(_fetch_treasury_year(year))
        except Exception as exc:
            # Historical 2024/2025 gaps should not block today's dashboard.
            # Current-year data is required so the displayed real yield is fresh.
            if year == current_year:
                raise
            print(f"Treasury real-yield year {year} skipped: {exc}")

    frame = pd.concat(frames, ignore_index=True)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
    frame = (
        frame.dropna()
        .sort_values("Date")
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
