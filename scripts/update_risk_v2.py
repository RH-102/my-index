"""Reliability wrapper for the market-risk updater.

It keeps the calculation/rating logic in update_risk.py, but makes the two
network-sensitive inputs more tolerant of public-source schema changes and
transient FRED timeouts.
"""

from io import StringIO
import time

import pandas as pd
import requests

import update_risk as base


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; my-index/1.0)"}


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
            # {"2020-01-03": 18.2, ...}
            if len(value) >= 10 and all(
                not isinstance(v, (dict, list)) for v in value.values()
            ):
                add(list(value.keys()), list(value.values()))

            # {date:[...], value:[...]} / {x:[...], y:[...]}
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
            # [[date, pe], ...]
            pairs = [
                row for row in value
                if isinstance(row, (list, tuple)) and len(row) >= 2
            ]
            if len(pairs) >= 10:
                add([row[0] for row in pairs], [row[1] for row in pairs])

            # [{date: ..., pe/value: ...}, ...]
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

                    # Last resort: use the only numeric scalar other than date.
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
    for attempt in range(4):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=(15, 90),
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
            if attempt < 3:
                time.sleep(3 * (attempt + 1))

    raise RuntimeError(f"{label} forward P/E failed: {last_error}")


def fetch_fred(series_id: str) -> pd.DataFrame:
    starts = {
        "DFII10": "2003-01-01",
        "DRSFRMACBS": "1991-01-01",
    }
    url = base.FRED_BASE.format(series_id)
    params = {"cosd": starts.get(series_id, "1950-01-01")}
    last_error = None

    for attempt in range(5):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HEADERS,
                timeout=(15, 120),
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
            last_error = exc
            print(f"FRED {series_id} attempt {attempt + 1} failed: {exc}")
            if attempt < 4:
                time.sleep(5 * (attempt + 1))

    raise RuntimeError(f"FRED {series_id} failed after retries: {last_error}")


# Patch only the unreliable input functions; retain all calculations/ratings.
base.fetch_forward_pe = fetch_forward_pe
base.fetch_fred = fetch_fred


if __name__ == "__main__":
    base.main()
