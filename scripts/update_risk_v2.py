"""Reliability wrapper for the market-risk updater.

Keep the calculation/rating logic in update_risk.py, but make public-source
requests fail fast. FRED is attempted first; if its CSV endpoint is slow or
unavailable, the exact underlying Federal Reserve Board series is loaded from
the Board's Data Download Program instead.
"""

import csv
from io import StringIO
import re
import time

import pandas as pd
import requests

import update_risk as base


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}
MAX_ATTEMPTS = 2
REQUEST_TIMEOUT = (10, 30)
FRED_TIMEOUT = (5, 12)
RETRY_SLEEP_SECONDS = 2

# Official Federal Reserve Board fallbacks for the two FRED series used here.
# These are the underlying Board data that FRED republishes.
FED_H15_TIPS_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=H15&series=0b98a66d3ff5e1ea0fbf88adc59b387f&lastobs=&from=&to=&"
    "filetype=csv&label=include&layout=seriescolumn&type=package"
)
FED_H15_TIPS_ID = "H15/H15/RIFLGFCY10_XII_N.B"

FED_CHGDEL_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=CHGDEL&series=e77af32404312ad2d45dd60dfa36477a&lastobs=&from=&to=&"
    "filetype=csv&label=include&layout=seriescolumn&type=package"
)
FED_CHGDEL_ID = "CHGDEL/CHGDEL/STFBQDSS%STFBAILSS_XEOP_XDO_MA.Q"


def _series_frame(dates, values):
    try:
        if len(dates) != len(values) or len(dates) < 10:
            return None
        frame = base.valid_pe_frame(dates, values)
        if len(frame) >= 10:
            return frame
    except Exception:
        pass
    return None


def _extract_from_forward_node(node):
    """Extract the public site's `forward` series regardless of JSON layout."""
    candidates = []

    def add(dates, values):
        frame = _series_frame(dates, values)
        if frame is not None:
            candidates.append(frame)

    def walk(value):
        if isinstance(value, dict):
            if len(value) >= 10 and all(
                not isinstance(v, (dict, list)) for v in value.values()
            ):
                add(list(value.keys()), list(value.values()))

            arrays = {
                base.key_norm(k): v
                for k, v in value.items()
                if isinstance(v, list)
            }
            date_arrays = [
                v for k, v in arrays.items()
                if "date" in k or k in {"x", "time", "period", "dates"}
            ]
            numeric_arrays = [
                v for k, v in arrays.items()
                if k in {
                    "value", "values", "y", "pe", "forwardpe", "fwdpe",
                    "forward", "series", "data"
                }
                or "forwardpe" in k
                or "fwdpe" in k
            ]
            for dates in date_arrays:
                for vals in numeric_arrays:
                    add(dates, vals)

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            pairs = [
                row for row in value
                if isinstance(row, (list, tuple)) and len(row) >= 2
            ]
            if len(pairs) >= 10:
                add([row[0] for row in pairs], [row[1] for row in pairs])

            rows = [row for row in value if isinstance(row, dict)]
            if len(rows) >= 10:
                dates = []
                vals = []
                for row in rows:
                    date_value = None
                    pe_value = None

                    for key, item in row.items():
                        norm = base.key_norm(key)
                        if date_value is None and (
                            "date" in norm or norm in {"x", "time", "period"}
                        ):
                            date_value = item

                    preferred = [
                        "forwardpe", "fwdpe", "pe", "value", "y",
                        "forward", "estimate", "multiple"
                    ]
                    for wanted in preferred:
                        for key, item in row.items():
                            norm = base.key_norm(key)
                            if norm == wanted or wanted in norm:
                                try:
                                    float(item)
                                    pe_value = item
                                    break
                                except Exception:
                                    pass
                        if pe_value is not None:
                            break

                    if pe_value is None:
                        numerics = []
                        for key, item in row.items():
                            if item == date_value:
                                continue
                            try:
                                number = float(item)
                                if 1 < number < 100:
                                    numerics.append(item)
                            except Exception:
                                pass
                        if len(numerics) == 1:
                            pe_value = numerics[0]

                    if date_value is not None and pe_value is not None:
                        dates.append(date_value)
                        vals.append(pe_value)

                add(dates, vals)

            for child in value:
                walk(child)

    walk(node)

    if not candidates:
        raise RuntimeError(
            "Could not parse the public forward-P/E series from the `forward` field."
        )

    candidates.sort(key=len, reverse=True)
    return candidates[0]


def fetch_forward_pe(url: str, label: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()

            if not isinstance(payload, dict) or "forward" not in payload:
                raise RuntimeError(
                    f"Unexpected {label} payload: missing `forward` field"
                )

            frame = _extract_from_forward_node(payload["forward"])
            if len(frame) < 100:
                raise RuntimeError(
                    f"{label}: only {len(frame)} forward P/E observations found"
                )

            print(
                f"{label} forward P/E: {len(frame)} observations, "
                f"{frame.iloc[0]['Date'].date()} to "
                f"{frame.iloc[-1]['Date'].date()}"
            )
            return frame
        except Exception as exc:
            last_error = exc
            print(f"{label} forward P/E attempt {attempt + 1} failed: {exc}")
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"{label} forward P/E failed: {last_error}")


def _parse_ddp_date(text: str):
    value = str(text).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return pd.to_datetime(value, errors="coerce")

    match = re.fullmatch(r"(\d{4})-?Q([1-4])", value, re.IGNORECASE)
    if match:
        return pd.Period(f"{match.group(1)}Q{match.group(2)}", freq="Q").end_time.normalize()

    return pd.NaT


def _parse_fed_ddp(text: str, unique_id: str) -> pd.DataFrame:
    """Parse a Federal Reserve DDP series-column CSV robustly."""
    rows = list(csv.reader(StringIO(text)))
    if not rows:
        raise RuntimeError("Federal Reserve DDP returned an empty file")

    target_col = None
    short_id = unique_id.split("/")[-1]

    for row in rows:
        for idx, cell in enumerate(row):
            cell_text = str(cell).strip()
            if unique_id in cell_text or short_id in cell_text:
                target_col = idx
                break
        if target_col is not None:
            break

    if target_col is None:
        raise RuntimeError(f"Could not find DDP series column {short_id}")

    dates = []
    values = []
    for row in rows:
        if len(row) <= target_col:
            continue
        dt = _parse_ddp_date(row[0])
        if pd.isna(dt):
            continue
        number = pd.to_numeric(pd.Series([row[target_col]]), errors="coerce").iloc[0]
        if pd.isna(number):
            continue
        dates.append(dt)
        values.append(float(number))

    frame = pd.DataFrame({"Date": dates, "Value": values})
    frame = frame.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    frame = frame.reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"No usable DDP observations for {short_id}")
    return frame


def _fetch_fed_ddp(url: str, unique_id: str, label: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            frame = _parse_fed_ddp(response.text, unique_id)
            print(
                f"Federal Reserve fallback {label}: {len(frame)} observations through "
                f"{frame.iloc[-1]['Date'].date()}"
            )
            return frame
        except Exception as exc:
            last_error = exc
            print(f"Federal Reserve fallback {label} attempt {attempt + 1} failed: {exc}")
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"Federal Reserve fallback {label} failed: {last_error}")


def fetch_fred(series_id: str) -> pd.DataFrame:
    """Fetch from FRED first, then fall back to the exact Board source series."""
    starts = {
        "DFII10": "2003-01-01",
        "DRSFRMACBS": "1991-01-01",
    }
    url = base.FRED_BASE.format(series_id)
    params = {"cosd": starts.get(series_id, "1950-01-01")}

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=FRED_TIMEOUT,
        )
        response.raise_for_status()
        frame = pd.read_csv(StringIO(response.text)).iloc[:, :2].copy()
        frame.columns = ["Date", "Value"]
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        frame = frame.dropna().sort_values("Date").reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"No observations returned for {series_id}")
        print(
            f"FRED {series_id}: {len(frame)} observations through "
            f"{frame.iloc[-1]['Date'].date()}"
        )
        return frame
    except Exception as exc:
        print(f"FRED {series_id} unavailable ({exc}); using Federal Reserve fallback.")

    if series_id == "DFII10":
        return _fetch_fed_ddp(
            FED_H15_TIPS_URL,
            FED_H15_TIPS_ID,
            "DFII10 / 10Y inflation-indexed Treasury",
        )

    if series_id == "DRSFRMACBS":
        return _fetch_fed_ddp(
            FED_CHGDEL_URL,
            FED_CHGDEL_ID,
            "DRSFRMACBS / single-family mortgage delinquency",
        )

    raise RuntimeError(f"No fallback configured for FRED series {series_id}")


def fetch_bis_credit_gap() -> pd.DataFrame:
    last_error = None

    for attempt in range(MAX_ATTEMPTS):
        try:
            response = requests.get(
                base.BIS_GAP_URL,
                headers={
                    "Accept": "application/vnd.sdmx.data+csv;version=1.0.0;labels=id",
                    **HEADERS,
                },
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            frame = pd.read_csv(StringIO(response.text))
            upper = {str(c).upper(): c for c in frame.columns}
            time_col = upper.get("TIME_PERIOD") or upper.get("TIME_PERIOD_START")
            value_col = upper.get("OBS_VALUE")
            if time_col is None or value_col is None:
                raise RuntimeError(
                    "Unexpected BIS columns: " + ", ".join(map(str, frame.columns))
                )

            def parse_quarter(value):
                text = str(value).strip()
                try:
                    return pd.Period(text, freq="Q").end_time.normalize()
                except Exception:
                    return pd.to_datetime(text, errors="coerce")

            out = pd.DataFrame(
                {
                    "Date": frame[time_col].map(parse_quarter),
                    "Value": pd.to_numeric(frame[value_col], errors="coerce"),
                }
            ).dropna()

            out = (
                out.sort_values("Date")
                .drop_duplicates("Date", keep="last")
                .reset_index(drop=True)
            )
            if out.empty:
                raise RuntimeError("No BIS credit-gap observations returned")

            print(
                f"BIS credit-to-GDP gap: {len(out)} observations through "
                f"{out.iloc[-1]['Date'].date()}"
            )
            return out
        except Exception as exc:
            last_error = exc
            print(f"BIS credit gap attempt {attempt + 1} failed: {exc}")
            if attempt + 1 < MAX_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"BIS credit gap failed after retries: {last_error}")


base.fetch_forward_pe = fetch_forward_pe
base.fetch_fred = fetch_fred
base.fetch_bis_credit_gap = fetch_bis_credit_gap


if __name__ == "__main__":
    try:
        base.main()
    except Exception as exc:
        print(f"Risk refresh skipped: {exc}")
        print("Existing risk_dashboard.csv and risk_history.csv were retained unchanged.")
